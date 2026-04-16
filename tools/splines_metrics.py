import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.artist import Artist
import numpy as np


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


def extract_pose_channel_points(pose_data, keypoint_idx, axis_idx, pose_fps):
	if pose_fps <= 0:
		raise ValueError(f"pose_fps must be > 0, got {pose_fps}")
	if keypoint_idx < 0 or keypoint_idx >= pose_data.shape[1]:
		raise ValueError(
			f"keypoint_idx={keypoint_idx} out of range [0, {pose_data.shape[1] - 1}] for pose file"
		)
	if axis_idx < 0 or axis_idx >= pose_data.shape[2]:
		raise ValueError(
			f"axis_idx={axis_idx} out of range [0, {pose_data.shape[2] - 1}] for pose file"
		)

	num_frames = pose_data.shape[0]
	pose_time_sec = np.arange(num_frames, dtype=np.float64) / float(pose_fps)
	pose_values = pose_data[:, keypoint_idx, axis_idx].astype(np.float64, copy=False)
	return pose_time_sec, pose_values


def load_spline_npz(npz_path):
	if not os.path.isfile(npz_path):
		raise FileNotFoundError(f"Spline file not found: {npz_path}")

	data = np.load(npz_path, allow_pickle=False)
	if "time_sec" not in data or "coeffs" not in data:
		raise KeyError(f"Missing keys in {npz_path}. Required: time_sec, coeffs")

	time_sec = data["time_sec"].astype(np.float64)
	coeffs = data["coeffs"].astype(np.float64)

	if coeffs.ndim != 4:
		raise ValueError(f"Unexpected coeffs shape {coeffs.shape}, expected (K, D, 4, S)")
	if coeffs.shape[2] != 4:
		raise ValueError(f"Unexpected coeffs shape {coeffs.shape}, cubic coeff dimension must be 4")
	if len(time_sec) != coeffs.shape[3] + 1:
		raise ValueError(
			f"time_sec length {len(time_sec)} does not match segments {coeffs.shape[3]}"
		)
	if np.any(np.diff(time_sec) <= 0):
		raise ValueError(f"time_sec must be strictly increasing in {npz_path}")

	return time_sec, coeffs


def resolve_channel(coeffs, spline_id):
	num_kpt = coeffs.shape[0]
	num_dim = coeffs.shape[1]
	total = num_kpt * num_dim

	if spline_id < 0 or spline_id >= total:
		raise ValueError(f"spline_id={spline_id} out of range [0, {total - 1}]")

	kpt = spline_id // num_dim
	dim = spline_id % num_dim
	return kpt, dim, total


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


def collect_overlap_frame_markers(gt_time_sec, pred_time_sec, intervals, tol=1e-12):
	if len(intervals) == 0:
		return np.asarray([], dtype=np.float64)

	overlap_start = float(intervals[0][2])
	overlap_end = float(intervals[-1][3])

	gt_markers = gt_time_sec[(gt_time_sec >= overlap_start - tol) & (gt_time_sec <= overlap_end + tol)]
	pred_markers = pred_time_sec[(pred_time_sec >= overlap_start - tol) & (pred_time_sec <= overlap_end + tol)]
	all_markers = np.sort(np.concatenate([gt_markers, pred_markers]).astype(np.float64, copy=False))

	if all_markers.size == 0:
		return all_markers

	dedup = [all_markers[0]]
	for t in all_markers[1:]:
		if abs(t - dedup[-1]) > tol:
			dedup.append(t)
	return np.asarray(dedup, dtype=np.float64)


def local_to_global_coeff(local_coeff, seg_start_t):
	# y(t) = a*(t-s)^3 + b*(t-s)^2 + c*(t-s) + d
	a, b, c, d = local_coeff
	s = seg_start_t

	g3 = a
	g2 = b - 3.0 * a * s
	g1 = c - 2.0 * b * s + 3.0 * a * (s ** 2)
	g0 = d - c * s + b * (s ** 2) - a * (s ** 3)
	return np.array([g3, g2, g1, g0], dtype=np.float64)


def integrate_square_poly(poly_desc, left, right):
	sq = np.polymul(poly_desc, poly_desc)
	sq_int = np.polyint(sq)
	return float(np.polyval(sq_int, right) - np.polyval(sq_int, left))


def shift_local_cubic_to_new_origin(local_coeff, shift):
	"""Shift y(t)=a*t^3+b*t^2+c*t+d to y(u)=y(u+shift) with u's origin at new point.

	This keeps integration in a local time window to avoid catastrophic cancellation
	from converting to global-time coefficients.
	"""
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


def eval_local_cubic(local_coeff, seg_start_t, ts):
	dt = ts - seg_start_t
	a, b, c, d = local_coeff
	return ((a * dt + b) * dt + c) * dt + d


def eval_local_cubic_derivative(local_coeff, seg_start_t, ts):
	dt = ts - seg_start_t
	a, b, c, _ = local_coeff
	return (3.0 * a * dt + 2.0 * b) * dt + c


def rigid_align_points_3d(src_points, dst_points):
	"""Rigidly align src to dst with Kabsch (rotation + translation, no scale)."""
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


def compute_pose_all_channel_metrics(gt_pose_mm, pred_pose_mm, fps):
	"""Compute all-channel (17x3) metrics from pose samples at the same timestamps."""
	if gt_pose_mm.shape != pred_pose_mm.shape:
		raise ValueError(f"pose shape mismatch: gt={gt_pose_mm.shape}, pred={pred_pose_mm.shape}")
	if gt_pose_mm.ndim != 3 or gt_pose_mm.shape[2] != 3:
		raise ValueError(f"unexpected pose shape: {gt_pose_mm.shape}")
	if fps <= 0:
		raise ValueError(f"fps must be > 0, got {fps}")

	diff = pred_pose_mm - gt_pose_mm
	abs_err = np.abs(diff)

	crmse = float(np.sqrt(np.mean(diff ** 2)))
	gt_range = float(np.max(gt_pose_mm) - np.min(gt_pose_mm)) if gt_pose_mm.size > 0 else float("nan")
	if np.isfinite(gt_range) and gt_range > 1e-12:
		nrmse = float(crmse / gt_range)
	else:
		nrmse = float("nan")

	mae = float(np.mean(abs_err)) if abs_err.size > 0 else float("nan")
	p95 = float(np.percentile(abs_err, 95.0)) if abs_err.size > 0 else float("nan")
	maxae = float(np.max(abs_err)) if abs_err.size > 0 else float("nan")

	if gt_pose_mm.shape[0] >= 2:
		vel_gt = np.diff(gt_pose_mm, axis=0) * float(fps)
		vel_pred = np.diff(pred_pose_mm, axis=0) * float(fps)
		vel_rmse = float(np.sqrt(np.mean((vel_pred - vel_gt) ** 2)))
	else:
		vel_rmse = float("nan")

	if gt_pose_mm.shape[0] >= 3:
		acc_gt = np.diff(gt_pose_mm, n=2, axis=0) * float(fps ** 2)
		acc_pred = np.diff(pred_pose_mm, n=2, axis=0) * float(fps ** 2)
		acc_rmse = float(np.sqrt(np.mean((acc_pred - acc_gt) ** 2)))
	else:
		acc_rmse = float("nan")

	joint_err = np.linalg.norm(diff, axis=2)
	mpjpe = float(np.mean(joint_err)) if joint_err.size > 0 else float("nan")

	return {
		"cRMSE_mm": crmse,
		"NRMSE_range": nrmse,
		"GT_range_mm": gt_range,
		"MAE_mm": mae,
		"P95AE_mm": p95,
		"MaxAE_mm": maxae,
		"VelRMSE_mmps": vel_rmse,
		"AccRMSE_mmps2": acc_rmse,
		"MPJPE_upsampled_mm": mpjpe,
		"channel_count": int(gt_pose_mm.shape[1] * gt_pose_mm.shape[2]),
	}


def compute_wham_rte_jitter(gt_pose_mm, pred_pose_mm, fps, root_kpt_idx=0):
	"""Compute WHAM-style RTE/Jitter from aligned root trajectory and all-joint jerk.

	- RTE: rigid alignment on root translation (fixed scale), then normalize by GT path length.
	- Jitter: mean over time of mean over joints of jerk norm, divided by 10 (m/s^3 scale).
	"""
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
		"RTE_percent": rte_percent,
		"Jitter_10mps3": jitter_10mps3,
		"Jitter_mps3": jitter_mps3,
		"root_sample_count": int(gt_pose_mm.shape[0]),
	}


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
	"""Evaluate spline pose values at arbitrary global timestamps.

	time_sec: shape (S+1,)
	coeffs: shape (K, D, 4, S)
	ts: shape (N,)
	returns: shape (N, K, D)
	"""
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


def resample_pose_at_times(pose_time_sec, pose_data, ts):
	"""Resample pose points at target timestamps with linear interpolation.

	returns:
		pose_resampled: shape (N, K, D)
		valid_mask: shape (N,), True when all joints/dims are finite at that time
	"""
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
	"""Downsample pose sequence to target fps by timestamp resampling."""
	if target_fps <= 0:
		raise ValueError(f"target_fps must be > 0, got {target_fps}")
	ts_ds = build_uniform_sample_times(float(pose_time_sec[0]), float(pose_time_sec[-1]), target_fps)
	pose_ds, valid = resample_pose_at_times(pose_time_sec, pose_data, ts_ds)
	if not np.any(valid):
		raise ValueError("No valid samples produced while downsampling pose")
	return ts_ds[valid], pose_ds[valid]


def eval_pose_linear_from_controls(control_ts, control_pose, eval_ts):
	"""Evaluate piecewise-linear pose defined by control points at eval timestamps."""
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


def build_linear_channel_coeffs(control_ts, control_values):
	"""Build piecewise-linear coefficients as cubic form [a,b,c,d] with a=b=0 per segment."""
	if control_ts.ndim != 1 or control_values.ndim != 1:
		raise ValueError("control_ts and control_values must be 1D arrays")
	if len(control_ts) != len(control_values):
		raise ValueError("control_ts and control_values length mismatch")
	if len(control_ts) < 2:
		raise ValueError("At least 2 control points are required for linear interpolation")
	if np.any(np.diff(control_ts) <= 0):
		raise ValueError("control_ts must be strictly increasing")

	seg = len(control_ts) - 1
	coeffs = np.empty((4, seg), dtype=np.float64)
	for i in range(seg):
		dt = float(control_ts[i + 1] - control_ts[i])
		slope = float((control_values[i + 1] - control_values[i]) / dt)
		coeffs[:, i] = np.array([0.0, 0.0, slope, float(control_values[i])], dtype=np.float64)
	return control_ts.astype(np.float64, copy=False), coeffs


def compute_channel_metrics(
	gt_time_sec,
	gt_channel_coeffs,
	pred_time_sec,
	pred_channel_coeffs,
	intervals,
	total_duration,
	samples_per_interval,
):
	if total_duration <= 0:
		raise ValueError("No overlap duration between the two curves")
	if samples_per_interval < 2:
		raise ValueError(f"samples_per_interval must be >= 2, got {samples_per_interval}")

	pos_sq_integral = 0.0
	vel_sq_integral = 0.0
	acc_sq_integral = 0.0

	abs_err_samples = []
	t_dense_all = []
	gt_dense_all = []
	pred_dense_all = []
	gt_vel_dense_all = []
	pred_vel_dense_all = []

	for idx, (gt_idx, pred_idx, left, right) in enumerate(intervals):
		gt_local = gt_channel_coeffs[:, gt_idx]
		pred_local = pred_channel_coeffs[:, pred_idx]

		# Integrate in overlap-local time u = t - left for better numerical stability.
		gt_u = shift_local_cubic_to_new_origin(gt_local, left - gt_time_sec[gt_idx])
		pred_u = shift_local_cubic_to_new_origin(pred_local, left - pred_time_sec[pred_idx])
		diff_u = pred_u - gt_u
		seg_len = float(right - left)

		pos_contrib = integrate_square_poly(diff_u, 0.0, seg_len)
		d1 = np.array([3.0 * diff_u[0], 2.0 * diff_u[1], diff_u[2]], dtype=np.float64)
		vel_contrib = integrate_square_poly(d1, 0.0, seg_len)
		d2 = np.array([6.0 * diff_u[0], 2.0 * diff_u[1]], dtype=np.float64)
		acc_contrib = integrate_square_poly(d2, 0.0, seg_len)

		# Guard against tiny negative values from floating-point round-off.
		if pos_contrib < 0.0 and pos_contrib > -1e-12:
			pos_contrib = 0.0
		if vel_contrib < 0.0 and vel_contrib > -1e-12:
			vel_contrib = 0.0
		if acc_contrib < 0.0 and acc_contrib > -1e-9:
			acc_contrib = 0.0

		pos_sq_integral += pos_contrib
		vel_sq_integral += vel_contrib
		acc_sq_integral += acc_contrib

		endpoint = idx == len(intervals) - 1
		ts = np.linspace(left, right, samples_per_interval, endpoint=endpoint, dtype=np.float64)
		if len(ts) == 0:
			continue

		gt_vals = eval_local_cubic(gt_local, gt_time_sec[gt_idx], ts)
		pred_vals = eval_local_cubic(pred_local, pred_time_sec[pred_idx], ts)
		gt_vel_vals = eval_local_cubic_derivative(gt_local, gt_time_sec[gt_idx], ts)
		pred_vel_vals = eval_local_cubic_derivative(pred_local, pred_time_sec[pred_idx], ts)
		abs_err_samples.append(np.abs(pred_vals - gt_vals))

		t_dense_all.append(ts)
		gt_dense_all.append(gt_vals)
		pred_dense_all.append(pred_vals)
		gt_vel_dense_all.append(gt_vel_vals)
		pred_vel_dense_all.append(pred_vel_vals)

	abs_err = np.concatenate(abs_err_samples, axis=0) if abs_err_samples else np.asarray([], dtype=np.float64)
	t_dense = np.concatenate(t_dense_all, axis=0) if t_dense_all else np.asarray([], dtype=np.float64)
	gt_dense = np.concatenate(gt_dense_all, axis=0) if gt_dense_all else np.asarray([], dtype=np.float64)
	pred_dense = np.concatenate(pred_dense_all, axis=0) if pred_dense_all else np.asarray([], dtype=np.float64)
	gt_vel_dense = np.concatenate(gt_vel_dense_all, axis=0) if gt_vel_dense_all else np.asarray([], dtype=np.float64)
	pred_vel_dense = np.concatenate(pred_vel_dense_all, axis=0) if pred_vel_dense_all else np.asarray([], dtype=np.float64)

	crmse = float(np.sqrt(pos_sq_integral / total_duration))
	vel_rmse = float(np.sqrt(vel_sq_integral / total_duration))
	acc_rmse = float(np.sqrt(acc_sq_integral / total_duration))

	gt_range = float(np.max(gt_dense) - np.min(gt_dense)) if gt_dense.size > 0 else float("nan")
	if np.isfinite(gt_range) and gt_range > 1e-12:
		nrmse = float(crmse / gt_range)
	else:
		nrmse = float("nan")

	mae = float(abs_err.mean()) if abs_err.size > 0 else float("nan")
	p95 = float(np.percentile(abs_err, 95.0)) if abs_err.size > 0 else float("nan")
	maxae = float(abs_err.max()) if abs_err.size > 0 else float("nan")

	return {
		"cRMSE_mm": crmse,
		"NRMSE_range": nrmse,
		"GT_range_mm": gt_range,
		"MAE_mm": mae,
		"P95AE_mm": p95,
		"MaxAE_mm": maxae,
		"VelRMSE_mmps": vel_rmse,
		"AccRMSE_mmps2": acc_rmse,
		"duration_sec": float(total_duration),
		"t_dense": t_dense,
		"gt_dense": gt_dense,
		"pred_dense": pred_dense,
		"gt_vel_dense": gt_vel_dense,
		"pred_vel_dense": pred_vel_dense,
	}


def axis_name(dim_idx):
	names = ["X", "Y", "Z"]
	if dim_idx < len(names):
		return names[dim_idx]
	return f"D{dim_idx}"


def plot_two_splines(
	output_path,
	title,
	metrics,
	t_dense,
	gt_dense,
	pred_dense,
	gt_vel_dense,
	pred_vel_dense,
	upsample_time_sec=None,
	upsample_abs_err=None,
	upsample_mpjpe_series=None,
	pose_time_sec=None,
	pose_values=None,
	frame_marker_times=None,
	plot_dpi=320,
):
	if t_dense.size == 0:
		raise ValueError("No dense curve samples to plot")

	fig, axes = plt.subplots(
		2,
		1,
		figsize=(12, 8.2),
		sharex=True,
		gridspec_kw={"height_ratios": [3.2, 1.3]},
	)
	ax = axes[0]
	ax_err = axes[1]

	pos_gt_line, = ax.plot(
		t_dense * 1000.0,
		gt_dense,
		color="tab:blue",
		linewidth=1.8,
		label="Position GT",
	)
	pos_pred_line, = ax.plot(
		t_dense * 1000.0,
		pred_dense,
		color="tab:red",
		linewidth=1.4,
		label="Position Pred",
	)

	pose_scatter = None
	if pose_time_sec is not None and pose_values is not None and len(pose_time_sec) > 0:
		left = float(t_dense[0])
		right = float(t_dense[-1])
		mask = (pose_time_sec >= left - 1e-12) & (pose_time_sec <= right + 1e-12)
		if np.any(mask):
			pose_scatter = ax.scatter(
				pose_time_sec[mask] * 1000.0,
				pose_values[mask],
				s=12,
				alpha=0.85,
				color="tab:purple",
				label="Pose Points",
				zorder=3,
			)

	ax_vel = ax.twinx()
	vel_gt_line, = ax_vel.plot(
		t_dense * 1000.0,
		gt_vel_dense,
		color="tab:green",
		linewidth=1.35,
		alpha=0.9,
		label="Velocity GT",
	)
	vel_pred_line, = ax_vel.plot(
		t_dense * 1000.0,
		pred_vel_dense,
		color="tab:orange",
		linewidth=1.2,
		alpha=0.9,
		label="Velocity Pred",
	)

	ax.set_title(title, fontsize=12)
	ax.set_ylabel("Position (mm)")
	ax_vel.set_ylabel("Velocity (mm/s)")

	# Denser reference grid: major + minor
	ax.minorticks_on()
	ax.grid(which="major", linestyle="-", linewidth=0.55, alpha=0.50)
	ax.grid(which="minor", linestyle=":", linewidth=0.45, alpha=0.45)

	# Draw one vertical reference line per frame boundary in overlap range.
	if frame_marker_times is not None and len(frame_marker_times) > 0:
		for idx, t_sec in enumerate(frame_marker_times):
			if idx % 5 == 0:
				ax.axvline(t_sec * 1000.0, color="0.60", linewidth=0.60, alpha=0.24, zorder=0)
			else:
				ax.axvline(t_sec * 1000.0, color="0.75", linewidth=0.45, alpha=0.18, zorder=0)

	lines: list[Artist] = [pos_gt_line, pos_pred_line, vel_gt_line, vel_pred_line]
	labels = ["Position GT", "Position Pred", "Velocity GT", "Velocity Pred"]
	if pose_scatter is not None:
		lines.append(pose_scatter)
		labels.append("Pose Points")
	ax.legend(lines, labels, loc="upper right", fontsize=9)

	# Upsampled absolute error and MPJPE series.
	if (
		upsample_time_sec is not None
		and upsample_abs_err is not None
		and upsample_mpjpe_series is not None
		and len(upsample_time_sec) > 0
	):
		err_line, = ax_err.plot(
			upsample_time_sec * 1000.0,
			upsample_abs_err,
			color="tab:brown",
			linewidth=0.9,
			marker="o",
			markersize=2.8,
			alpha=0.85,
			label="Abs Error (resampled)",
		)
		mpjpe_line, = ax_err.plot(
			upsample_time_sec * 1000.0,
			upsample_mpjpe_series,
			color="tab:cyan",
			linewidth=1.1,
			alpha=0.95,
			label="MPJPE (resampled)",
		)

		imax = int(np.argmax(upsample_abs_err))
		t_max = float(upsample_time_sec[imax] * 1000.0)
		e_max = float(upsample_abs_err[imax])
		ax_err.scatter([t_max], [e_max], color="red", s=20, zorder=4, label="Max Abs Error")
		ax_err.annotate(
			f"max={e_max:.3f} mm",
			xy=(t_max, e_max),
			xytext=(8, 6),
			textcoords="offset points",
			fontsize=8,
			color="red",
		)

		ax_err.legend([err_line, mpjpe_line], ["Abs Error (resampled)", "MPJPE (resampled)"], loc="upper right", fontsize=9)

	ax_err.minorticks_on()
	ax_err.grid(which="major", linestyle="-", linewidth=0.45, alpha=0.45)
	ax_err.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.35)
	ax_err.set_ylabel("Error (mm)")
	ax_err.set_xlabel("Time (ms)")

	metric_text = (
		f"cRMSE: {metrics['cRMSE_mm']:.4f} mm\n"
		f"NRMSE(range): {metrics['NRMSE_range']:.4f}\n"
		f"MAE: {metrics['MAE_mm']:.4f} mm\n"
		f"P95AE: {metrics['P95AE_mm']:.4f} mm\n"
		f"MaxAE: {metrics['MaxAE_mm']:.4f} mm\n"
		f"RTE: {metrics.get('RTE_percent', float('nan')):.4f} %\n"
		f"Jitter: {metrics.get('Jitter_10mps3', float('nan')):.4f} (10 m/s^3)\n"
		f"MPJPE@{metrics.get('upsample_fps', float('nan')):.1f}Hz: {metrics.get('MPJPE_upsampled_mm', float('nan')):.4f} mm\n"
		f"Overlap: {metrics['duration_sec']:.4f} s"
	)

	ax.text(
		0.015,
		0.985,
		metric_text,
		transform=ax.transAxes,
		fontsize=9,
		va="top",
		ha="left",
		bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
	)

	fig.tight_layout()
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	fig.savefig(output_path, dpi=int(plot_dpi))
	plt.close(fig)


def run_compare(
	gt_file,
	pred_file,
	spline_id,
	samples_per_interval,
	output_path,
	plot_dpi=320,
	upsample_fps=90.0,
	pose_file=None,
	pose_fps=30.0,
	linear_downsample_fps=30.0,
	linear_output_path=None,
):
	gt_t, gt_coeffs = load_spline_npz(gt_file)
	pred_t, pred_coeffs = load_spline_npz(pred_file)

	if gt_coeffs.shape[0] != pred_coeffs.shape[0] or gt_coeffs.shape[1] != pred_coeffs.shape[1]:
		raise ValueError(
			f"Shape mismatch: gt={gt_coeffs.shape[:2]} vs pred={pred_coeffs.shape[:2]}"
		)

	kpt, dim, total = resolve_channel(gt_coeffs, spline_id)
	intervals, duration = build_overlap_intervals(gt_t, pred_t)
	if len(intervals) == 0 or duration <= 0:
		raise ValueError("No overlap between ground-truth and prediction spline time ranges")

	pose_time_sec = None
	pose_values = None
	pose_data = None
	if not pose_file:
		raise ValueError("pose_file is required: all metrics are computed from GT pose vs upsampled prediction")

	pose_data = load_pose_array(pose_file)
	if pose_data.shape[1] != pred_coeffs.shape[0] or pose_data.shape[2] != pred_coeffs.shape[1]:
		raise ValueError(
			f"pose shape mismatch with spline channels: pose={pose_data.shape[1:]} vs spline={pred_coeffs.shape[:2]}"
		)
	pose_time_sec, pose_values = extract_pose_channel_points(
		pose_data=pose_data,
		keypoint_idx=kpt,
		axis_idx=dim,
		pose_fps=pose_fps,
	)

	metrics = compute_channel_metrics(
		gt_time_sec=gt_t,
		gt_channel_coeffs=gt_coeffs[kpt, dim],
		pred_time_sec=pred_t,
		pred_channel_coeffs=pred_coeffs[kpt, dim],
		intervals=intervals,
		total_duration=duration,
		samples_per_interval=samples_per_interval,
	)

	title = (
		f"Spline Compare | id={spline_id} / total={total} "
		f"(keypoint={kpt}, axis={axis_name(dim)})"
	)
	frame_marker_times = collect_overlap_frame_markers(gt_t, pred_t, intervals)

	upsample_time_sec = None
	upsample_abs_err = None
	upsample_mpjpe_series = None
	if upsample_fps is None or upsample_fps <= 0:
		raise ValueError(f"upsample_fps must be > 0 for all-channel metrics, got {upsample_fps}")

	overlap_start = max(float(pred_t[0]), float(pose_time_sec[0]))
	overlap_end = min(float(pred_t[-1]), float(pose_time_sec[-1]))
	if overlap_end <= overlap_start:
		raise ValueError("No overlap between prediction spline timeline and GT pose timeline")

	upsample_time_all = build_uniform_sample_times(overlap_start, overlap_end, upsample_fps)
	upsample_time_sec = upsample_time_all
	if upsample_time_sec.size > 0:
		pred_pose_up = eval_spline_pose_at_times(pred_t, pred_coeffs, upsample_time_sec)
		gt_pose_up, valid_mask = resample_pose_at_times(pose_time_sec, pose_data, upsample_time_sec)
		if np.any(valid_mask):
			valid_times = upsample_time_sec[valid_mask]
			pred_valid = pred_pose_up[valid_mask]
			gt_valid = gt_pose_up[valid_mask]

			joint_err = np.linalg.norm(pred_valid - gt_valid, axis=2)
			upsample_mpjpe_series = joint_err.mean(axis=1)
			upsample_abs_err = np.abs(
				pred_valid[:, kpt, dim] - gt_valid[:, kpt, dim]
			).astype(np.float64, copy=False)
			upsample_time_sec = valid_times

			metrics.update(
				compute_pose_all_channel_metrics(
					gt_pose_mm=gt_valid,
					pred_pose_mm=pred_valid,
					fps=float(upsample_fps),
				)
			)
			metrics.update(
				compute_wham_rte_jitter(
					gt_pose_mm=gt_valid,
					pred_pose_mm=pred_valid,
					fps=float(upsample_fps),
					root_kpt_idx=0,
				)
			)
			metrics["upsample_fps"] = float(upsample_fps)
		else:
			raise ValueError("No valid overlapping pose samples after resampling")
	else:
		raise ValueError("No upsample timestamps generated; check overlap range and upsample_fps")

	plot_two_splines(
		output_path=output_path,
		title=title,
		metrics=metrics,
		t_dense=metrics["t_dense"],
		gt_dense=metrics["gt_dense"],
		pred_dense=metrics["pred_dense"],
		gt_vel_dense=metrics["gt_vel_dense"],
		pred_vel_dense=metrics["pred_vel_dense"],
		upsample_time_sec=upsample_time_sec,
		upsample_abs_err=upsample_abs_err,
		upsample_mpjpe_series=upsample_mpjpe_series,
		pose_time_sec=pose_time_sec,
		pose_values=pose_values,
		frame_marker_times=frame_marker_times,
		plot_dpi=plot_dpi,
	)

	if linear_output_path is None:
		linear_output_path = os.path.join(os.path.dirname(output_path), "downsample_linear.png")

	# Linear interpolation baseline:
	# 1) downsample GT pose to target FPS
	# 2) connect points with piecewise linear interpolation
	# 3) compare this linear curve to GT pose curve on all 17x3 channels
	ctrl_ts, ctrl_pose = downsample_pose_linear(pose_time_sec, pose_data, linear_downsample_fps)
	lin_pred_all, lin_valid = eval_pose_linear_from_controls(ctrl_ts, ctrl_pose, upsample_time_all)
	gt_all, gt_valid = resample_pose_at_times(pose_time_sec, pose_data, upsample_time_all)
	valid_lin = lin_valid & gt_valid
	if not np.any(valid_lin):
		raise ValueError("No valid overlapping samples for linear interpolation baseline")

	ts_lin = upsample_time_all[valid_lin]
	gt_lin = gt_all[valid_lin]
	pred_lin = lin_pred_all[valid_lin]

	lin_scatter_metrics = compute_pose_all_channel_metrics(
		gt_pose_mm=gt_lin,
		pred_pose_mm=pred_lin,
		fps=float(upsample_fps),
	)
	lin_scatter_metrics.update(
		compute_wham_rte_jitter(
			gt_pose_mm=gt_lin,
			pred_pose_mm=pred_lin,
			fps=float(upsample_fps),
			root_kpt_idx=0,
		)
	)
	lin_scatter_metrics["upsample_fps"] = float(upsample_fps)

	# Curve metrics for linear baseline are computed against gt_spline (not gt_pose points).
	lin_curve_t, lin_curve_coeff = build_linear_channel_coeffs(ctrl_ts, ctrl_pose[:, kpt, dim])
	lin_intervals, lin_duration = build_overlap_intervals(gt_t, lin_curve_t)
	if len(lin_intervals) == 0 or lin_duration <= 0:
		raise ValueError("No overlap between gt_spline and linear baseline timeline")
	lin_curve_metrics = compute_channel_metrics(
		gt_time_sec=gt_t,
		gt_channel_coeffs=gt_coeffs[kpt, dim],
		pred_time_sec=lin_curve_t,
		pred_channel_coeffs=lin_curve_coeff,
		intervals=lin_intervals,
		total_duration=lin_duration,
		samples_per_interval=samples_per_interval,
	)

	lin_metrics = {
		"cRMSE_mm": lin_curve_metrics["cRMSE_mm"],
		"NRMSE_range": lin_curve_metrics["NRMSE_range"],
		"MAE_mm": lin_scatter_metrics["MAE_mm"],
		"P95AE_mm": lin_scatter_metrics["P95AE_mm"],
		"MaxAE_mm": lin_scatter_metrics["MaxAE_mm"],
		"RTE_percent": lin_scatter_metrics["RTE_percent"],
		"Jitter_10mps3": lin_scatter_metrics["Jitter_10mps3"],
		"Jitter_mps3": lin_scatter_metrics["Jitter_mps3"],
		"MPJPE_upsampled_mm": lin_scatter_metrics["MPJPE_upsampled_mm"],
		"upsample_fps": float(upsample_fps),
		"duration_sec": lin_curve_metrics["duration_sec"],
		"channel_count": lin_scatter_metrics.get("channel_count", float("nan")),
	}

	t_dense_lin = lin_curve_metrics["t_dense"]
	gt_dense_lin = lin_curve_metrics["gt_dense"]
	pred_dense_lin = lin_curve_metrics["pred_dense"]
	gt_vel_lin = lin_curve_metrics["gt_vel_dense"]
	pred_vel_lin = lin_curve_metrics["pred_vel_dense"]

	lin_abs_err = np.abs(pred_lin[:, kpt, dim] - gt_lin[:, kpt, dim])
	lin_mpjpe = np.linalg.norm(pred_lin - gt_lin, axis=2).mean(axis=1)

	lin_title = (
		f"Linear Interp Baseline | id={spline_id} / total={total} "
		f"(keypoint={kpt}, axis={axis_name(dim)})"
	)
	plot_two_splines(
		output_path=linear_output_path,
		title=lin_title,
		metrics=lin_metrics,
		t_dense=t_dense_lin,
		gt_dense=gt_dense_lin,
		pred_dense=pred_dense_lin,
		gt_vel_dense=gt_vel_lin,
		pred_vel_dense=pred_vel_lin,
		upsample_time_sec=ts_lin,
		upsample_abs_err=lin_abs_err,
		upsample_mpjpe_series=lin_mpjpe,
		pose_time_sec=ts_lin,
		pose_values=gt_lin[:, kpt, dim],
		frame_marker_times=collect_overlap_frame_markers(gt_t, lin_curve_t, lin_intervals),
		plot_dpi=plot_dpi,
	)

	print(f"Total spline dimensions: {total} ({gt_coeffs.shape[0]} * {gt_coeffs.shape[1]})")
	print(f"Selected spline id: {spline_id} -> keypoint={kpt}, axis={axis_name(dim)}")
	print(f"Saved plot: {output_path}")
	if pose_file:
		print(f"Pose overlay: {pose_file} | pose_fps={pose_fps}")
	print(f"Linear interpolation plot: {linear_output_path} | downsample_fps={linear_downsample_fps}")
	print(
		"Metrics: "
		f"cRMSE={metrics['cRMSE_mm']:.6f} mm, "
		f"NRMSE(range)={metrics['NRMSE_range']:.6f}, "
		f"MAE={metrics['MAE_mm']:.6f} mm, "
		f"P95AE={metrics['P95AE_mm']:.6f} mm, "
		f"MaxAE={metrics['MaxAE_mm']:.6f} mm, "
		f"Channels={metrics.get('channel_count', float('nan'))}, "
		f"RTE={metrics.get('RTE_percent', float('nan')):.6f} %, "
		f"Jitter={metrics.get('Jitter_10mps3', float('nan')):.6f} (10 m/s^3), "
		f"MPJPE@{metrics.get('upsample_fps', float('nan')):.1f}Hz={metrics.get('MPJPE_upsampled_mm', float('nan')):.6f} mm"
	)
	print(
		"LinearMetrics: "
		f"cRMSE={lin_metrics['cRMSE_mm']:.6f} mm, "
		f"NRMSE(range)={lin_metrics['NRMSE_range']:.6f}, "
		f"MAE={lin_metrics['MAE_mm']:.6f} mm, "
		f"P95AE={lin_metrics['P95AE_mm']:.6f} mm, "
		f"MaxAE={lin_metrics['MaxAE_mm']:.6f} mm, "
		f"Channels={lin_metrics.get('channel_count', float('nan'))}, "
		f"RTE={lin_metrics.get('RTE_percent', float('nan')):.6f} %, "
		f"Jitter={lin_metrics.get('Jitter_10mps3', float('nan')):.6f} (10 m/s^3), "
		f"MPJPE@{lin_metrics.get('upsample_fps', float('nan')):.1f}Hz={lin_metrics.get('MPJPE_upsampled_mm', float('nan')):.6f} mm"
	)
	print("LinearMetricsNote: cRMSE/NRMSE use gt_spline curves; MAE/P95/MaxAE/RTE/Jitter/MPJPE use 120fps gt_pose sample points.")


def build_argparser():
	parser = argparse.ArgumentParser(
		description=(
			"Plot one specified spline dimension from two spline npz files on the same figure "
			"and annotate continuous-curve metrics."
		)
	)
	parser.add_argument("--gt-file", type=str, required=True, help="Ground-truth spline npz path")
	parser.add_argument("--pred-file", type=str, required=True, help="Predicted spline npz path")
	parser.add_argument(
		"--spline-id",
		type=int,
		default=0,
		help="Spline dimension index in [0, K*D-1], typically [0, 50] for 17*3.",
	)
	parser.add_argument(
		"--samples-per-interval",
		type=int,
		default=40,
		help="Dense samples per overlap interval for MAE/P95/Max approximation.",
	)
	parser.add_argument(
		"--output-path",
		type=str,
		default="/home/ztw/HVCCS/res/splines_metrics/splines_compare_plot.png",
		help="Output plot path",
	)
	parser.add_argument(
		"--plot-dpi",
		type=int,
		default=320,
		help="Output plot DPI for saved figure.",
	)
	parser.add_argument(
		"--upsample-fps",
		type=float,
		default=120.0,
		help="Uniform resampling FPS for spline MPJPE metric and abs-error plotting.",
	)
	parser.add_argument(
		"--pose-file",
		type=str,
		default=None,
		help="Ground-truth pose npy path for overlay and pred_spline-vs-gt_pose upsampled MPJPE.",
	)
	parser.add_argument(
		"--pose-fps",
		type=float,
		default=30.0,
		help="FPS for pose-file timestamps alignment. Used by all-channel metrics and pose overlay.",
	)
	parser.add_argument(
		"--linear-downsample-fps",
		type=float,
		default=30.0,
		help="Downsample FPS for linear interpolation baseline built from GT pose points.",
	)
	parser.add_argument(
		"--linear-output-path",
		type=str,
		default=None,
		help="Output path for linear interpolation baseline plot. Default: <output_dir>/downsample_linear.png",
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
		run_compare(
			gt_file=args.gt_file,
			pred_file=args.pred_file,
			spline_id=args.spline_id,
			samples_per_interval=args.samples_per_interval,
			output_path=args.output_path,
			plot_dpi=args.plot_dpi,
			upsample_fps=args.upsample_fps,
			pose_file=args.pose_file,
			pose_fps=args.pose_fps,
			linear_downsample_fps=args.linear_downsample_fps,
			linear_output_path=args.linear_output_path,
		)
	else:
		gt_file = "/home/data/ztw/AtheletePose3D/h36m_pose_cam_1/test/S2_cam_1_120fps_notaknot_splines/Running_37_cam_1_h36m_notaknot_spline.npz"
		pred_file = "/home/ztw/HVCCS/res/splines_fit_abg/Running_37_cam_1_h36m_30fps_abg_realtime_spline.npz"
		gt_pose_file = "/home/data/ztw/AtheletePose3D/h36m_pose_cam_1/test/S2_cam_1_120fps/Running_37_cam_1_h36m.npy"
		gt_pose_fps = 120.0 # 120.0. or 60.0 depending on the source of the pose file and its timestamp alignment with the splines.
		spline_id = 0               # valid range: 0..50 for 17x3
		samples_per_interval = 40   # dense sampling for MAE/P95/Max approximation
		upsample_fps = 120.0        # uniform spline resampling FPS for MPJPE/abs-error points
		output_path = "/home/ztw/HVCCS/res/splines_metrics/downsample_abg.png"
		linear_downsample_fps = 30.0
		linear_output_path = "/home/ztw/HVCCS/res/splines_metrics/downsample_linear.png"
		plot_dpi = 640              # increase saved plot resolution
		run_compare(
			gt_file=gt_file,
			pred_file=pred_file,
			spline_id=spline_id,
			samples_per_interval=samples_per_interval,
			output_path=output_path,
			plot_dpi=plot_dpi,
			upsample_fps=upsample_fps,
			pose_file=gt_pose_file,
			pose_fps=gt_pose_fps,
			linear_downsample_fps=linear_downsample_fps,
			linear_output_path=linear_output_path,
		)
