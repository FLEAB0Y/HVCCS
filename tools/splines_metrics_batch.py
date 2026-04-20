import argparse
import csv
import json
import os

import numpy as np


def safe_div(numerator, denominator, eps=1e-12):
	if denominator is None:
		return float("nan")
	if not np.isfinite(denominator):
		return float("nan")
	if abs(denominator) <= eps:
		return float("nan")
	return float(numerator / denominator)


def collect_npz_files_by_suffix(input_dir, suffix):
	if not os.path.isdir(input_dir):
		raise FileNotFoundError(f"Directory not found: {input_dir}")

	files = []
	for name in os.listdir(input_dir):
		full_path = os.path.join(input_dir, name)
		if not os.path.isfile(full_path):
			continue
		if not name.endswith(suffix):
			continue
		files.append(name)
	files.sort()
	return files


def build_id_to_path_map(input_dir, suffix):
	mapping = {}
	for name in collect_npz_files_by_suffix(input_dir, suffix):
		sample_id = name[: -len(suffix)]
		mapping[sample_id] = os.path.join(input_dir, name)
	return mapping


def load_spline_npz(npz_path):
	data = np.load(npz_path, allow_pickle=False)
	if "time_sec" not in data or "coeffs" not in data:
		raise KeyError(f"Missing required keys in {npz_path}. Need time_sec and coeffs")

	time_sec = data["time_sec"].astype(np.float64)
	coeffs = data["coeffs"].astype(np.float64)

	if coeffs.ndim != 4:
		raise ValueError(f"Unexpected coeffs shape {coeffs.shape}, expected (K,3,4,S)")
	if coeffs.shape[1] != 3 or coeffs.shape[2] != 4:
		raise ValueError(f"Unexpected coeffs shape {coeffs.shape}, expected axis=3 and cubic=4")
	if len(time_sec) != coeffs.shape[3] + 1:
		raise ValueError(
			f"time_sec length {len(time_sec)} mismatches segments {coeffs.shape[3]} in {npz_path}"
		)
	if np.any(np.diff(time_sec) <= 0):
		raise ValueError(f"time_sec must be strictly increasing in {npz_path}")

	return time_sec, coeffs


def load_pose_array(npy_path):
	if not os.path.isfile(npy_path):
		raise FileNotFoundError(f"Pose file not found: {npy_path}")

	data = np.load(npy_path)
	if data.ndim == 4 and data.shape[0] == 1:
		data = data[0]

	if data.ndim != 3 or data.shape[-1] != 3:
		raise ValueError(
			f"Unexpected pose shape {data.shape}. Expected (frames, keypoints, 3) or (1, frames, keypoints, 3)."
		)

	return data.astype(np.float64, copy=False)


def build_uniform_sample_times(start_sec, end_sec, fps, tol=1e-12):
	if fps <= 0:
		raise ValueError(f"upsample fps must be > 0, got {fps}")
	if end_sec <= start_sec + tol:
		return np.asarray([], dtype=np.float64)

	dt = 1.0 / float(fps)
	count = int(np.floor((end_sec - start_sec) / dt + tol)) + 1
	ts = start_sec + np.arange(count, dtype=np.float64) * dt
	if ts.size == 0:
		return np.asarray([start_sec, end_sec], dtype=np.float64)
	if ts[-1] < end_sec - tol:
		ts = np.concatenate([ts, np.asarray([end_sec], dtype=np.float64)])
	return ts


def eval_spline_pose_at_times(time_sec, coeffs, ts):
	if coeffs.ndim != 4 or coeffs.shape[2] != 4:
		raise ValueError(f"Unexpected coeffs shape {coeffs.shape}, expected (K, D, 4, S)")
	if len(time_sec) != coeffs.shape[3] + 1:
		raise ValueError("time_sec and coeffs segment count mismatch")
	if np.any(np.diff(time_sec) <= 0):
		raise ValueError("time_sec must be strictly increasing")

	seg_count = coeffs.shape[3]
	seg_idx = np.searchsorted(time_sec, ts, side="right") - 1
	seg_idx = np.clip(seg_idx, 0, seg_count - 1)

	out = np.empty((len(ts), coeffs.shape[0], coeffs.shape[1]), dtype=np.float64)
	for n, sidx in enumerate(seg_idx):
		tau = float(ts[n] - time_sec[sidx])
		a = coeffs[:, :, 0, sidx]
		b = coeffs[:, :, 1, sidx]
		c = coeffs[:, :, 2, sidx]
		d = coeffs[:, :, 3, sidx]
		out[n] = ((a * tau + b) * tau + c) * tau + d
	return out


def build_pose_time_sec(num_frames, pose_fps):
	if pose_fps <= 0:
		raise ValueError(f"pose_fps must be > 0, got {pose_fps}")
	return np.arange(num_frames, dtype=np.float64) / float(pose_fps)


def resample_pose_at_times(pose_time_sec, pose_data, ts):
	out = np.full((len(ts), pose_data.shape[1], pose_data.shape[2]), np.nan, dtype=np.float64)
	for j in range(pose_data.shape[1]):
		for d in range(pose_data.shape[2]):
			out[:, j, d] = np.interp(
				ts,
				pose_time_sec,
				pose_data[:, j, d],
				left=np.nan,
				right=np.nan,
			)
	valid = np.isfinite(out).all(axis=(1, 2))
	return out, valid


def downsample_pose_linear(pose_time_sec, pose_data, target_fps):
	if target_fps <= 0:
		raise ValueError(f"target_fps must be > 0, got {target_fps}")
	ts_ds = build_uniform_sample_times(float(pose_time_sec[0]), float(pose_time_sec[-1]), target_fps)
	pose_ds, valid = resample_pose_at_times(pose_time_sec, pose_data, ts_ds)
	if not np.any(valid):
		raise ValueError("No valid samples produced while downsampling pose")
	return ts_ds[valid], pose_ds[valid]


def eval_pose_linear_from_controls(control_ts, control_pose, eval_ts):
	out = np.full((len(eval_ts), control_pose.shape[1], control_pose.shape[2]), np.nan, dtype=np.float64)
	for j in range(control_pose.shape[1]):
		for d in range(control_pose.shape[2]):
			out[:, j, d] = np.interp(
				eval_ts,
				control_ts,
				control_pose[:, j, d],
				left=np.nan,
				right=np.nan,
			)
	valid = np.isfinite(out).all(axis=(1, 2))
	return out, valid


def rigid_align_points_3d(src_points, dst_points):
	if src_points.ndim != 2 or dst_points.ndim != 2:
		raise ValueError("src_points and dst_points must be 2D arrays")
	if src_points.shape != dst_points.shape:
		raise ValueError(f"shape mismatch: src={src_points.shape}, dst={dst_points.shape}")
	if src_points.shape[1] != 3:
		raise ValueError("rigid alignment requires 3D points")

	mu_src = np.mean(src_points, axis=0)
	mu_dst = np.mean(dst_points, axis=0)
	src_c = src_points - mu_src
	dst_c = dst_points - mu_dst

	H = src_c.T @ dst_c
	U, _, Vt = np.linalg.svd(H)
	R = Vt.T @ U.T
	if np.linalg.det(R) < 0:
		Vt[-1, :] *= -1.0
		R = Vt.T @ U.T
	t = mu_dst - (R @ mu_src)
	aligned = src_points @ R.T + t
	return aligned, R, t


def compute_wham_rte_jitter(gt_pose_mm, pred_pose_mm, fps, root_kpt_idx=0):
	if gt_pose_mm.shape != pred_pose_mm.shape:
		raise ValueError(f"pose shape mismatch: gt={gt_pose_mm.shape}, pred={pred_pose_mm.shape}")
	if gt_pose_mm.ndim != 3 or gt_pose_mm.shape[2] != 3:
		raise ValueError(f"unexpected pose shape: {gt_pose_mm.shape}")
	if fps <= 0:
		raise ValueError(f"fps must be > 0, got {fps}")
	if root_kpt_idx < 0 or root_kpt_idx >= gt_pose_mm.shape[1]:
		raise ValueError(f"root_kpt_idx={root_kpt_idx} out of range")

	gt_root = gt_pose_mm[:, root_kpt_idx, :]
	pred_root = pred_pose_mm[:, root_kpt_idx, :]
	pred_root_aligned, _, _ = rigid_align_points_3d(pred_root, gt_root)

	if gt_root.shape[0] >= 2:
		disp = float(np.sum(np.linalg.norm(np.diff(gt_root, axis=0), axis=1)))
	else:
		disp = 0.0
	if disp > 1e-9:
		rte = np.linalg.norm(gt_root - pred_root_aligned, axis=1)
		rte_percent = float(np.mean(rte / disp) * 100.0)
	else:
		rte_percent = float("nan")

	if pred_pose_mm.shape[0] >= 4:
		jerk_mmps3 = (
			pred_pose_mm[3:]
			- 3.0 * pred_pose_mm[2:-1]
			+ 3.0 * pred_pose_mm[1:-2]
			- pred_pose_mm[:-3]
		) * (float(fps) ** 3)
		jitter_frame_mmps3 = np.linalg.norm(jerk_mmps3, axis=2).mean(axis=1)
		jitter_mps3 = float(np.mean(jitter_frame_mmps3) / 1000.0)
		jitter_10mps3 = float(jitter_mps3 / 10.0)
	else:
		jitter_mps3 = float("nan")
		jitter_10mps3 = float("nan")

	return {
		"rte_percent_pose_upsampled": rte_percent,
		"jitter_10mps3_pose_upsampled": jitter_10mps3,
		"jitter_mps3_pose_upsampled": jitter_mps3,
	}


def build_overlap_intervals(gt_time_sec, pred_time_sec, eps=1e-12):
	gt_seg = len(gt_time_sec) - 1
	pred_seg = len(pred_time_sec) - 1

	i = 0
	j = 0
	intervals = []
	total_duration = 0.0

	while i < gt_seg and j < pred_seg:
		gt_end = gt_time_sec[i + 1]
		pred_end = pred_time_sec[j + 1]

		left = max(gt_time_sec[i], pred_time_sec[j])
		right = min(gt_end, pred_end)

		if right - left > eps:
			intervals.append((i, j, float(left), float(right)))
			total_duration += float(right - left)

		if gt_end <= pred_end + eps:
			i += 1
		if pred_end <= gt_end + eps:
			j += 1

	return intervals, total_duration


def local_to_global_coeff(local_coeff, start_t):
	# local_coeff: [a, b, c, d], y(t)=a*(t-s)^3 + b*(t-s)^2 + c*(t-s) + d
	a, b, c, d = local_coeff
	s = start_t

	g3 = a
	g2 = b - 3.0 * a * s
	g1 = c - 2.0 * b * s + 3.0 * a * (s ** 2)
	g0 = d - c * s + b * (s ** 2) - a * (s ** 3)
	return np.array([g3, g2, g1, g0], dtype=np.float64)


def integrate_square_of_poly(poly_desc, left, right):
	sq = np.polymul(poly_desc, poly_desc)
	sq_int = np.polyint(sq)
	return float(np.polyval(sq_int, right) - np.polyval(sq_int, left))


def shift_local_cubic_to_new_origin(local_coeff, shift):
	"""Shift y(t)=a*t^3+b*t^2+c*t+d to y(u)=y(u+shift) around a new local origin."""
	a, b, c, d = local_coeff
	s = float(shift)
	return np.array(
		[
			a,
			b + 3.0 * a * s,
			c + 2.0 * b * s + 3.0 * a * (s ** 2),
			d + c * s + b * (s ** 2) + a * (s ** 3),
		],
		dtype=np.float64,
	)


def clamp_small_negative(value, tol):
	if value < 0.0 and value > -tol:
		return 0.0
	return value


def evaluate_segment_curve(coeff_k3_4, seg_start_t, ts):
	# coeff_k3_4 shape: (K, 3, 4), ts shape: (M,)
	tau = (ts - seg_start_t)[:, None, None]
	a = coeff_k3_4[:, :, 0][None, :, :]
	b = coeff_k3_4[:, :, 1][None, :, :]
	c = coeff_k3_4[:, :, 2][None, :, :]
	d = coeff_k3_4[:, :, 3][None, :, :]
	return ((a * tau + b) * tau + c) * tau + d


def evaluate_segment_derivatives(coeff_k3_4, seg_start_t, ts):
	# coeff_k3_4 shape: (K, 3, 4), ts shape: (M,)
	tau = (ts - seg_start_t)[:, None, None]
	a = coeff_k3_4[:, :, 0][None, :, :]
	b = coeff_k3_4[:, :, 1][None, :, :]
	c = coeff_k3_4[:, :, 2][None, :, :]

	vel = (3.0 * a * tau + 2.0 * b) * tau + c
	acc = 6.0 * a * tau + 2.0 * b
	return vel, acc


def compute_analytic_metrics(gt_time_sec, gt_coeffs, pred_time_sec, pred_coeffs, intervals, total_duration):
	num_kpts, num_dims, _, _ = gt_coeffs.shape
	channels = num_kpts * num_dims

	pos_sq_integral = 0.0
	vel_sq_integral = 0.0
	acc_sq_integral = 0.0

	for kpt in range(num_kpts):
		for dim in range(num_dims):
			for gt_idx, pred_idx, left, right in intervals:
				gt_local = gt_coeffs[kpt, dim, :, gt_idx]
				pred_local = pred_coeffs[kpt, dim, :, pred_idx]

				# Integrate in overlap-local time u=t-left to reduce cancellation error.
				gt_u = shift_local_cubic_to_new_origin(gt_local, left - gt_time_sec[gt_idx])
				pred_u = shift_local_cubic_to_new_origin(pred_local, left - pred_time_sec[pred_idx])
				diff_u = pred_u - gt_u
				seg_len = float(right - left)

				pos_contrib = integrate_square_of_poly(diff_u, 0.0, seg_len)
				d1 = np.array([3.0 * diff_u[0], 2.0 * diff_u[1], diff_u[2]], dtype=np.float64)
				vel_contrib = integrate_square_of_poly(d1, 0.0, seg_len)
				d2 = np.array([6.0 * diff_u[0], 2.0 * diff_u[1]], dtype=np.float64)
				acc_contrib = integrate_square_of_poly(d2, 0.0, seg_len)

				pos_sq_integral += clamp_small_negative(pos_contrib, tol=1e-12)
				vel_sq_integral += clamp_small_negative(vel_contrib, tol=1e-12)
				acc_sq_integral += clamp_small_negative(acc_contrib, tol=1e-9)

	duration_channels = total_duration * channels
	if duration_channels <= 0:
		raise ValueError("Non-positive overlap duration")

	pos_mse = pos_sq_integral / duration_channels
	vel_mse = vel_sq_integral / duration_channels
	acc_mse = acc_sq_integral / duration_channels

	pos_mse = clamp_small_negative(pos_mse, tol=1e-15)
	vel_mse = clamp_small_negative(vel_mse, tol=1e-15)
	acc_mse = clamp_small_negative(acc_mse, tol=1e-12)

	return {
		"pos_sq_integral": pos_sq_integral,
		"vel_sq_integral": vel_sq_integral,
		"acc_sq_integral": acc_sq_integral,
		"channels": channels,
		"duration_sec": total_duration,
		"crmse_mm": float(np.sqrt(pos_mse)),
		"vel_rmse_mmps": float(np.sqrt(vel_mse)),
		"acc_rmse_mmps2": float(np.sqrt(acc_mse)),
	}


def compute_sampled_metrics(
	gt_time_sec,
	gt_coeffs,
	pred_time_sec,
	pred_coeffs,
	intervals,
	total_duration,
	samples_per_interval,
):
	if samples_per_interval < 2:
		raise ValueError(f"samples_per_interval must be >= 2, got {samples_per_interval}")

	if total_duration <= 0:
		raise ValueError("Non-positive overlap duration")

	mpjpe_integral = 0.0
	all_joint_errors = []
	all_abs_xyz_errors = []
	total_dense_time_samples = 0

	gt_pos_min = np.inf
	gt_pos_max = -np.inf
	gt_vel_min = np.inf
	gt_vel_max = -np.inf
	gt_acc_min = np.inf
	gt_acc_max = -np.inf

	for idx, (gt_idx, pred_idx, left, right) in enumerate(intervals):
		endpoint = idx == len(intervals) - 1
		ts = np.linspace(left, right, samples_per_interval, endpoint=endpoint, dtype=np.float64)
		if ts.size < 2:
			continue

		gt_values = evaluate_segment_curve(gt_coeffs[:, :, :, gt_idx], gt_time_sec[gt_idx], ts)
		pred_values = evaluate_segment_curve(pred_coeffs[:, :, :, pred_idx], pred_time_sec[pred_idx], ts)
		gt_vel, gt_acc = evaluate_segment_derivatives(gt_coeffs[:, :, :, gt_idx], gt_time_sec[gt_idx], ts)

		gt_pos_min = min(gt_pos_min, float(np.min(gt_values)))
		gt_pos_max = max(gt_pos_max, float(np.max(gt_values)))
		gt_vel_min = min(gt_vel_min, float(np.min(gt_vel)))
		gt_vel_max = max(gt_vel_max, float(np.max(gt_vel)))
		gt_acc_min = min(gt_acc_min, float(np.min(gt_acc)))
		gt_acc_max = max(gt_acc_max, float(np.max(gt_acc)))

		diff = pred_values - gt_values
		joint_err = np.linalg.norm(diff, axis=2)  # (M, K)
		mpjpe_t = joint_err.mean(axis=1)
		total_dense_time_samples += int(ts.size)

		mpjpe_integral += float(np.trapz(mpjpe_t, ts))
		all_joint_errors.append(joint_err.reshape(-1))
		all_abs_xyz_errors.append(np.abs(diff).reshape(-1))

	if len(all_joint_errors) == 0:
		raise ValueError("No sampled points generated for sampled metrics")

	joint_errors = np.concatenate(all_joint_errors, axis=0)
	abs_xyz_errors = np.concatenate(all_abs_xyz_errors, axis=0)
	mpjpe_cont = mpjpe_integral / total_duration

	gt_pos_range = float(gt_pos_max - gt_pos_min)
	gt_vel_range = float(gt_vel_max - gt_vel_min)
	gt_acc_range = float(gt_acc_max - gt_acc_min)

	return {
		"mpjpe_cont_mm": float(mpjpe_cont),
		"mpjpe_integral_mm_sec": float(mpjpe_integral),
		"mae_dense_xyz_mm": float(np.mean(abs_xyz_errors)),
		"p95_joint_err_mm": float(np.percentile(joint_errors, 95.0)),
		"max_joint_err_mm": float(np.max(joint_errors)),
		"num_dense_time_samples": int(total_dense_time_samples),
		"num_dense_joint_samples": int(joint_errors.size),
		"gt_pos_range_mm": gt_pos_range,
		"gt_vel_range_mmps": gt_vel_range,
		"gt_acc_range_mmps2": gt_acc_range,
	}


def compute_pose_upsampled_metrics(
	pred_time_sec,
	pred_coeffs,
	pose_time_sec,
	pose_data,
	intervals,
	upsample_fps,
):
	if upsample_fps is None or upsample_fps <= 0:
		raise ValueError(f"upsample_fps must be > 0, got {upsample_fps}")
	if len(intervals) == 0:
		raise ValueError("No overlap intervals for upsampled pose metrics")

	overlap_start = float(intervals[0][2])
	overlap_end = float(intervals[-1][3])
	target_ts = build_uniform_sample_times(overlap_start, overlap_end, upsample_fps)
	if target_ts.size == 0:
		raise ValueError("No upsample timestamps generated")

	pred_pose_up = eval_spline_pose_at_times(pred_time_sec, pred_coeffs, target_ts)
	gt_pose_up, valid_mask = resample_pose_at_times(pose_time_sec, pose_data, target_ts)
	if not np.any(valid_mask):
		raise ValueError("No valid overlap timestamps with gt pose for upsampled MPJPE")

	target_ts_valid = target_ts[valid_mask]
	pred_valid = pred_pose_up[valid_mask]
	gt_valid = gt_pose_up[valid_mask]
	diff = pred_valid - gt_valid
	abs_err = np.abs(diff)
	joint_err = np.linalg.norm(pred_valid - gt_valid, axis=2)
	mpjpe_series = joint_err.mean(axis=1)

	pose_crmse = float(np.sqrt(np.mean(diff ** 2)))
	gt_range = float(np.max(gt_valid) - np.min(gt_valid)) if gt_valid.size > 0 else float("nan")
	if np.isfinite(gt_range) and gt_range > 1e-12:
		pose_nrmse = float(pose_crmse / gt_range)
	else:
		pose_nrmse = float("nan")

	wham = compute_wham_rte_jitter(gt_valid, pred_valid, float(upsample_fps), root_kpt_idx=0)

	return {
		"upsample_fps": float(upsample_fps),
		"num_upsample_points": int(pred_valid.shape[0]),
		"target_ts_valid": target_ts_valid,
		"pose_crmse_upsampled_mm": pose_crmse,
		"pose_nrmse_range_upsampled": pose_nrmse,
		"pose_mae_upsampled_mm": float(np.mean(abs_err)),
		"pose_p95ae_upsampled_mm": float(np.percentile(abs_err, 95.0)),
		"pose_maxae_upsampled_mm": float(np.max(abs_err)),
		"pose_abs_err_sum_mm": float(np.sum(abs_err)),
		"pose_joint_err_sum_mm": float(np.sum(joint_err)),
		"mpjpe_pose_upsampled_mm": float(np.mean(mpjpe_series)),
		"p95_joint_err_pose_upsampled_mm": float(np.percentile(joint_err, 95.0)),
		"max_joint_err_pose_upsampled_mm": float(np.max(joint_err)),
		"rte_percent_pose_upsampled": wham["rte_percent_pose_upsampled"],
		"jitter_10mps3_pose_upsampled": wham["jitter_10mps3_pose_upsampled"],
		"jitter_mps3_pose_upsampled": wham["jitter_mps3_pose_upsampled"],
	}


def compute_linear_interp_reference_metrics(pose_time_sec, pose_data, eval_ts, downsample_fps):
	ctrl_ts, ctrl_pose = downsample_pose_linear(pose_time_sec, pose_data, downsample_fps)
	lin_pred_all, lin_valid = eval_pose_linear_from_controls(ctrl_ts, ctrl_pose, eval_ts)
	gt_eval, gt_valid = resample_pose_at_times(pose_time_sec, pose_data, eval_ts)
	valid = lin_valid & gt_valid
	if not np.any(valid):
		raise ValueError("No valid samples for linear interpolation reference metrics")

	pred_valid = lin_pred_all[valid]
	gt_valid_pose = gt_eval[valid]
	diff = pred_valid - gt_valid_pose
	abs_err = np.abs(diff)
	joint_err = np.linalg.norm(diff, axis=2)
	mpjpe_series = joint_err.mean(axis=1)

	return {
		"linear_downsample_fps": float(downsample_fps),
		"linear_num_upsample_points": int(pred_valid.shape[0]),
		"linear_mpjpe_pose_upsampled_mm": float(np.mean(mpjpe_series)),
		"linear_p95_joint_err_pose_upsampled_mm": float(np.percentile(joint_err, 95.0)),
		"linear_max_joint_err_pose_upsampled_mm": float(np.max(joint_err)),
		"linear_pose_mae_upsampled_mm": float(np.mean(abs_err)),
		"linear_pose_abs_err_sum_mm": float(np.sum(abs_err)),
	}


def build_metric_definitions():
	return {
		"curve_error_metrics": [
			"crmse_mm",
			"vel_rmse_mmps",
			"acc_rmse_mmps2",
			"ncrmse_by_pos_range",
			"nvel_rmse_by_vel_range",
			"nacc_rmse_by_acc_range",
		],
		"sampled_curve_point_metrics": [
			"mae_dense_xyz_mm",
			"mpjpe_cont_mm",
			"p95_joint_err_mm",
			"max_joint_err_mm",
		],
		"sampled_pose_point_metrics": [
			"pose_crmse_upsampled_mm",
			"pose_nrmse_range_upsampled",
			"pose_mae_upsampled_mm",
			"pose_p95ae_upsampled_mm",
			"pose_maxae_upsampled_mm",
			"mpjpe_pose_upsampled_mm",
			"p95_joint_err_pose_upsampled_mm",
			"max_joint_err_pose_upsampled_mm",
			"rte_percent_pose_upsampled",
			"jitter_10mps3_pose_upsampled",
		],
		"linear_interpolation_reference_metrics": [
			"linear_mpjpe_pose_upsampled_mm",
			"linear_p95_joint_err_pose_upsampled_mm",
			"linear_max_joint_err_pose_upsampled_mm",
			"linear_pose_mae_upsampled_mm",
		],
		"per_point_average_metrics": [
			"crmse_mm",
			"vel_rmse_mmps",
			"acc_rmse_mmps2",
			"mae_dense_xyz_mm",
			"mpjpe_cont_mm",
			"pose_crmse_upsampled_mm",
			"pose_mae_upsampled_mm",
			"mpjpe_pose_upsampled_mm",
			"linear_mpjpe_pose_upsampled_mm",
		],
		"accumulative_metrics": [
			"pos_sq_integral",
			"vel_sq_integral",
			"acc_sq_integral",
			"mpjpe_integral_mm_sec",
			"pose_abs_err_sum_mm",
			"pose_joint_err_sum_mm",
			"linear_pose_abs_err_sum_mm",
		],
	}


def summarize_results(rows, analytic_global):
	summary = {
		"num_files": len(rows),
		"analytic_global": analytic_global,
		"metric_definitions": build_metric_definitions(),
	}
	if len(rows) == 0:
		return summary

	numeric_keys = [
		"crmse_mm",
		"ncrmse_by_pos_range",
		"vel_rmse_mmps",
		"nvel_rmse_by_vel_range",
		"acc_rmse_mmps2",
		"nacc_rmse_by_acc_range",
		"mae_dense_xyz_mm",
		"mpjpe_cont_mm",
		"mpjpe_integral_mm_sec",
		"nmpjpe_by_pos_range",
		"pose_crmse_upsampled_mm",
		"pose_nrmse_range_upsampled",
		"pose_mae_upsampled_mm",
		"pose_p95ae_upsampled_mm",
		"pose_maxae_upsampled_mm",
		"pose_abs_err_sum_mm",
		"pose_joint_err_sum_mm",
		"rte_percent_pose_upsampled",
		"jitter_10mps3_pose_upsampled",
		"jitter_mps3_pose_upsampled",
		"mpjpe_pose_upsampled_mm",
		"nmpjpe_pose_upsampled_by_pos_range",
		"linear_mpjpe_pose_upsampled_mm",
		"linear_p95_joint_err_pose_upsampled_mm",
		"linear_max_joint_err_pose_upsampled_mm",
		"linear_pose_mae_upsampled_mm",
		"linear_pose_abs_err_sum_mm",
		"mpjpe_pose_minus_linear_mm",
		"p95_joint_err_mm",
		"np95_joint_err_by_pos_range",
		"p95_joint_err_pose_upsampled_mm",
		"np95_joint_err_pose_upsampled_by_pos_range",
		"max_joint_err_mm",
		"nmax_joint_err_by_pos_range",
		"max_joint_err_pose_upsampled_mm",
		"nmax_joint_err_pose_upsampled_by_pos_range",
		"num_dense_time_samples",
		"num_dense_joint_samples",
		"num_upsample_points",
		"linear_num_upsample_points",
		"gt_pos_range_mm",
		"gt_vel_range_mmps",
		"gt_acc_range_mmps2",
	]
	per_file_stats = {}
	for key in numeric_keys:
		values = [r[key] for r in rows if key in r and r[key] is not None]
		if len(values) == 0:
			continue
		arr = np.asarray(values, dtype=np.float64)
		per_file_stats[key] = {
			"mean": float(arr.mean()),
			"median": float(np.median(arr)),
			"p95": float(np.percentile(arr, 95.0)),
			"max": float(arr.max()),
		}

	summary["per_file_stats"] = per_file_stats
	return summary


def write_rows_csv(rows, csv_path):
	os.makedirs(os.path.dirname(csv_path), exist_ok=True)
	if len(rows) == 0:
		with open(csv_path, "w", newline="") as fp:
			fp.write("sample_id\n")
		return

	header = []
	for row in rows:
		for key in row.keys():
			if key not in header:
				header.append(key)

	with open(csv_path, "w", newline="") as fp:
		writer = csv.DictWriter(fp, fieldnames=header)
		writer.writeheader()
		writer.writerows(rows)


def run_metrics(
	gt_dir,
	pred_dir,
	gt_pose_dir,
	output_dir,
	gt_suffix,
	pred_suffix,
	gt_pose_suffix,
	gt_pose_fps,
	samples_per_interval,
	upsample_fps,
	linear_downsample_fps,
	max_files,
):
	gt_map = build_id_to_path_map(gt_dir, gt_suffix)
	pred_map = build_id_to_path_map(pred_dir, pred_suffix)
	pose_map = build_id_to_path_map(gt_pose_dir, gt_pose_suffix)

	gt_ids = set(gt_map.keys())
	pred_ids = set(pred_map.keys())
	pose_ids = set(pose_map.keys())
	matched_ids = sorted(gt_ids & pred_ids & pose_ids)

	if max_files is not None and max_files > 0:
		matched_ids = matched_ids[:max_files]

	print(f"Ground truth files: {len(gt_map)}")
	print(f"Prediction files: {len(pred_map)}")
	print(f"Ground truth pose files: {len(pose_map)}")
	print(f"Matched files: {len(matched_ids)}")

	rows = []

	total_pos_sq = 0.0
	total_vel_sq = 0.0
	total_acc_sq = 0.0
	total_duration_channels = 0.0

	failed = 0
	for idx, sample_id in enumerate(matched_ids, start=1):
		gt_path = gt_map[sample_id]
		pred_path = pred_map[sample_id]
		pose_path = pose_map[sample_id]

		try:
			gt_t, gt_c = load_spline_npz(gt_path)
			pr_t, pr_c = load_spline_npz(pred_path)

			if gt_c.shape[0] != pr_c.shape[0] or gt_c.shape[1] != pr_c.shape[1]:
				raise ValueError(
					f"Keypoint/dim mismatch: gt={gt_c.shape[:2]} pred={pr_c.shape[:2]}"
				)

			intervals, duration = build_overlap_intervals(gt_t, pr_t)
			if len(intervals) == 0 or duration <= 0:
				raise ValueError("No overlap intervals between ground truth and prediction")

			analytic = compute_analytic_metrics(gt_t, gt_c, pr_t, pr_c, intervals, duration)
			sampled = compute_sampled_metrics(
				gt_t,
				gt_c,
				pr_t,
				pr_c,
				intervals,
				duration,
				samples_per_interval=samples_per_interval,
			)

			pose_data = load_pose_array(pose_path)
			pose_time_sec = build_pose_time_sec(pose_data.shape[0], gt_pose_fps)
			pose_up = compute_pose_upsampled_metrics(
				pred_time_sec=pr_t,
				pred_coeffs=pr_c,
				pose_time_sec=pose_time_sec,
				pose_data=pose_data,
				intervals=intervals,
				upsample_fps=upsample_fps,
			)
			linear_ref = compute_linear_interp_reference_metrics(
				pose_time_sec=pose_time_sec,
				pose_data=pose_data,
				eval_ts=pose_up["target_ts_valid"],
				downsample_fps=linear_downsample_fps,
			)

			row = {
				"sample_id": sample_id,
				"duration_sec": analytic["duration_sec"],
				"duration_channels": analytic["duration_sec"] * analytic["channels"],
				"pos_sq_integral": analytic["pos_sq_integral"],
				"vel_sq_integral": analytic["vel_sq_integral"],
				"acc_sq_integral": analytic["acc_sq_integral"],
				"crmse_mm": analytic["crmse_mm"],
				"ncrmse_by_pos_range": safe_div(analytic["crmse_mm"], sampled["gt_pos_range_mm"]),
				"vel_rmse_mmps": analytic["vel_rmse_mmps"],
				"nvel_rmse_by_vel_range": safe_div(analytic["vel_rmse_mmps"], sampled["gt_vel_range_mmps"]),
				"acc_rmse_mmps2": analytic["acc_rmse_mmps2"],
				"nacc_rmse_by_acc_range": safe_div(analytic["acc_rmse_mmps2"], sampled["gt_acc_range_mmps2"]),
				"mae_dense_xyz_mm": sampled["mae_dense_xyz_mm"],
				"mpjpe_cont_mm": sampled["mpjpe_cont_mm"],
				"mpjpe_integral_mm_sec": sampled["mpjpe_integral_mm_sec"],
				"nmpjpe_by_pos_range": safe_div(sampled["mpjpe_cont_mm"], sampled["gt_pos_range_mm"]),
				"pose_crmse_upsampled_mm": pose_up["pose_crmse_upsampled_mm"],
				"pose_nrmse_range_upsampled": pose_up["pose_nrmse_range_upsampled"],
				"pose_mae_upsampled_mm": pose_up["pose_mae_upsampled_mm"],
				"pose_p95ae_upsampled_mm": pose_up["pose_p95ae_upsampled_mm"],
				"pose_maxae_upsampled_mm": pose_up["pose_maxae_upsampled_mm"],
				"pose_abs_err_sum_mm": pose_up["pose_abs_err_sum_mm"],
				"pose_joint_err_sum_mm": pose_up["pose_joint_err_sum_mm"],
				"rte_percent_pose_upsampled": pose_up["rte_percent_pose_upsampled"],
				"jitter_10mps3_pose_upsampled": pose_up["jitter_10mps3_pose_upsampled"],
				"jitter_mps3_pose_upsampled": pose_up["jitter_mps3_pose_upsampled"],
				"mpjpe_pose_upsampled_mm": pose_up["mpjpe_pose_upsampled_mm"],
				"nmpjpe_pose_upsampled_by_pos_range": safe_div(
					pose_up["mpjpe_pose_upsampled_mm"], sampled["gt_pos_range_mm"]
				),
				"linear_downsample_fps": linear_ref["linear_downsample_fps"],
				"linear_num_upsample_points": linear_ref["linear_num_upsample_points"],
				"linear_mpjpe_pose_upsampled_mm": linear_ref["linear_mpjpe_pose_upsampled_mm"],
				"linear_p95_joint_err_pose_upsampled_mm": linear_ref["linear_p95_joint_err_pose_upsampled_mm"],
				"linear_max_joint_err_pose_upsampled_mm": linear_ref["linear_max_joint_err_pose_upsampled_mm"],
				"linear_pose_mae_upsampled_mm": linear_ref["linear_pose_mae_upsampled_mm"],
				"linear_pose_abs_err_sum_mm": linear_ref["linear_pose_abs_err_sum_mm"],
				"mpjpe_pose_minus_linear_mm": (
					pose_up["mpjpe_pose_upsampled_mm"] - linear_ref["linear_mpjpe_pose_upsampled_mm"]
				),
				"p95_joint_err_mm": sampled["p95_joint_err_mm"],
				"np95_joint_err_by_pos_range": safe_div(sampled["p95_joint_err_mm"], sampled["gt_pos_range_mm"]),
				"p95_joint_err_pose_upsampled_mm": pose_up["p95_joint_err_pose_upsampled_mm"],
				"np95_joint_err_pose_upsampled_by_pos_range": safe_div(
					pose_up["p95_joint_err_pose_upsampled_mm"], sampled["gt_pos_range_mm"]
				),
				"max_joint_err_mm": sampled["max_joint_err_mm"],
				"nmax_joint_err_by_pos_range": safe_div(sampled["max_joint_err_mm"], sampled["gt_pos_range_mm"]),
				"max_joint_err_pose_upsampled_mm": pose_up["max_joint_err_pose_upsampled_mm"],
				"nmax_joint_err_pose_upsampled_by_pos_range": safe_div(
					pose_up["max_joint_err_pose_upsampled_mm"], sampled["gt_pos_range_mm"]
				),
				"upsample_fps": pose_up["upsample_fps"],
				"num_upsample_points": pose_up["num_upsample_points"],
				"num_dense_time_samples": sampled["num_dense_time_samples"],
				"num_dense_joint_samples": sampled["num_dense_joint_samples"],
				"gt_pos_range_mm": sampled["gt_pos_range_mm"],
				"gt_vel_range_mmps": sampled["gt_vel_range_mmps"],
				"gt_acc_range_mmps2": sampled["gt_acc_range_mmps2"],
				"gt_path": gt_path,
				"pred_path": pred_path,
				"gt_pose_path": pose_path,
			}
			rows.append(row)

			total_pos_sq += analytic["pos_sq_integral"]
			total_vel_sq += analytic["vel_sq_integral"]
			total_acc_sq += analytic["acc_sq_integral"]
			total_duration_channels += analytic["duration_sec"] * analytic["channels"]

			print(
				f"[{idx}/{len(matched_ids)}] {sample_id}: "
				f"cRMSE={analytic['crmse_mm']:.4f} mm, "
				f"ncRMSE={safe_div(analytic['crmse_mm'], sampled['gt_pos_range_mm']):.6f}, "
				f"MAE_pose={pose_up['pose_mae_upsampled_mm']:.4f} mm, "
				f"RTE={pose_up['rte_percent_pose_upsampled']:.4f}%, "
				f"MPJPE_cont={sampled['mpjpe_cont_mm']:.4f} mm, "
				f"MPJPE_pose_up={pose_up['mpjpe_pose_upsampled_mm']:.4f} mm, "
				f"LinearMPJPE={linear_ref['linear_mpjpe_pose_upsampled_mm']:.4f} mm"
			)
		except Exception as exc:
			failed += 1
			print(f"[{idx}/{len(matched_ids)}] Failed {sample_id}: {exc}")

	analytic_global = None
	if total_duration_channels > 0:
		global_pos_mse = clamp_small_negative(total_pos_sq / total_duration_channels, tol=1e-15)
		global_vel_mse = clamp_small_negative(total_vel_sq / total_duration_channels, tol=1e-15)
		global_acc_mse = clamp_small_negative(total_acc_sq / total_duration_channels, tol=1e-12)
		analytic_global = {
			"crmse_mm": float(np.sqrt(global_pos_mse)),
			"vel_rmse_mmps": float(np.sqrt(global_vel_mse)),
			"acc_rmse_mmps2": float(np.sqrt(global_acc_mse)),
			"duration_channels": float(total_duration_channels),
		}

	summary = summarize_results(rows, analytic_global)
	summary["failed_files"] = failed
	summary["matched_files"] = len(matched_ids)
	summary["gt_suffix"] = gt_suffix
	summary["pred_suffix"] = pred_suffix
	summary["gt_pose_suffix"] = gt_pose_suffix
	summary["gt_pose_fps"] = float(gt_pose_fps)
	summary["upsample_fps"] = float(upsample_fps)
	summary["linear_downsample_fps"] = float(linear_downsample_fps)
	summary["samples_per_interval"] = int(samples_per_interval)

	os.makedirs(output_dir, exist_ok=True)
	csv_path = os.path.join(output_dir, "metrics_per_file.csv")
	summary_path = os.path.join(output_dir, "metrics_summary.json")

	write_rows_csv(rows, csv_path)
	with open(summary_path, "w") as fp:
		json.dump(summary, fp, indent=2)

	print("=" * 60)
	print(f"Saved per-file metrics: {csv_path}")
	print(f"Saved summary: {summary_path}")
	if analytic_global is not None:
		print(
			"Global analytic metrics: "
			f"cRMSE={analytic_global['crmse_mm']:.4f} mm, "
			f"vRMSE={analytic_global['vel_rmse_mmps']:.4f} mm/s, "
			f"aRMSE={analytic_global['acc_rmse_mmps2']:.4f} mm/s^2"
		)


def build_argparser():
	parser = argparse.ArgumentParser(
		description=(
			"Compare predicted realtime spline curves against offline splines_fit ground truth. "
			"Use analytic integration whenever possible; use dense sampling for non-analytic metrics."
		)
	)
	parser.add_argument(
		"--gt-dir",
		type=str,
		default="/home/data/ztw/AtheletePose3D/h36m_pose_cam_1/test/S2_cam_1_120fps_notaknot_splines",
		help="Ground truth spline directory (from splines_fit).",
	)
	parser.add_argument(
		"--pred-dir",
		type=str,
		default="/home/ztw/HVCCS/res/splines_fit_baseline",
		help="Prediction spline directory (e.g., Kalman, ABG or Baseline realtime output).",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="/home/ztw/HVCCS/res/splines_metrics_batch",
		help="Output directory for CSV/JSON metrics.",
	)
	parser.add_argument(
		"--gt-pose-dir",
		type=str,
		default="/home/data/ztw/AtheletePose3D/h36m_pose_cam_1/test/S2_cam_1_120fps",
		help="Ground truth pose directory (npy), used by upsampled pose MPJPE.",
	)
	parser.add_argument(
		"--gt-suffix",
		type=str,
		default="_notaknot_spline.npz",
		help="Ground truth filename suffix for matching sample IDs.",
	)
	parser.add_argument(
		"--pred-suffix",
		type=str,
		default="_baseline_realtime_spline.npz",
		help="Prediction filename suffix for matching sample IDs.",
	)
	parser.add_argument(
		"--gt-pose-suffix",
		type=str,
		default=".npy",
		help="Ground truth pose filename suffix for matching sample IDs.",
	)
	parser.add_argument(
		"--gt-pose-fps",
		type=float,
		default=120.0,
		help="FPS for ground truth pose timestamps.",
	)
	parser.add_argument(
		"--samples-per-interval",
		type=int,
		default=40,
		help="Dense sampling points per overlap interval for non-analytic metrics.",
	)
	parser.add_argument(
		"--upsample-fps",
		type=float,
		default=120.0,
		help="Uniform resampling FPS for pred_spline-vs-gt_pose MPJPE metrics.",
	)
	parser.add_argument(
		"--linear-downsample-fps",
		type=float,
		default=30.0,
		help="Downsample FPS for linear interpolation MPJPE reference from GT pose.",
	)
	parser.add_argument(
		"--max-files",
		type=int,
		default=None,
		help="Optional cap for quick test runs.",
	)
	return parser


if __name__ == "__main__":
	# -----------------------------------------------------------------
	# Direct config mode: edit values below and run this script directly.
	# Set USE_CLI=True if you prefer passing parameters via command line.
	# -----------------------------------------------------------------
	USE_CLI = False

	if USE_CLI:
		args = build_argparser().parse_args()
		run_metrics(
			gt_dir=args.gt_dir,
			pred_dir=args.pred_dir,
			gt_pose_dir=args.gt_pose_dir,
			output_dir=args.output_dir,
			gt_suffix=args.gt_suffix,
			pred_suffix=args.pred_suffix,
			gt_pose_suffix=args.gt_pose_suffix,
			gt_pose_fps=args.gt_pose_fps,
			samples_per_interval=args.samples_per_interval,
			upsample_fps=args.upsample_fps,
			linear_downsample_fps=args.linear_downsample_fps,
			max_files=args.max_files,
		)
	else:
		gt_splines_dir = "/home/data/ztw/AtheletePose3D/h36m_pose_cam_1/test/S2_cam_1_120fps_notaknot_splines"
		pred_splines_dir = "/home/ztw/HVCCS/res/splines_fit_baseline"
		gt_pose_dir = "/home/data/ztw/AtheletePose3D/h36m_pose_cam_1/test/S2_cam_1_120fps"
		output_dir = "/home/ztw/HVCCS/res/splines_metrics_batch"
		gt_suffix = "_notaknot_spline.npz"
		pred_suffix = "_30fps_baseline_realtime_spline.npz"
		gt_pose_suffix = ".npy"
		gt_pose_fps = 120.0
		samples_per_interval = 40
		upsample_fps = 120.0
		linear_downsample_fps = 30.0
		max_files = None

		run_metrics(
			gt_dir=gt_splines_dir,
			pred_dir=pred_splines_dir,
			gt_pose_dir=gt_pose_dir,
			output_dir=output_dir,
			gt_suffix=gt_suffix,
			pred_suffix=pred_suffix,
			gt_pose_suffix=gt_pose_suffix,
			gt_pose_fps=gt_pose_fps,
			samples_per_interval=samples_per_interval,
			upsample_fps=upsample_fps,
			linear_downsample_fps=linear_downsample_fps,
			max_files=max_files,
		)
