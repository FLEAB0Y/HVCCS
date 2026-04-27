import argparse
import csv
import json
import os

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _is_within_root(path_value, root_dir):
	path_abs = os.path.abspath(path_value)
	root_abs = os.path.abspath(root_dir)
	try:
		return os.path.commonpath([path_abs, root_abs]) == root_abs
	except ValueError:
		return False


def resolve_runtime_path(path_value, project_root=PROJECT_ROOT):
	path_str = str(path_value)
	if os.path.isabs(path_str):
		return path_str

	resolved = os.path.abspath(os.path.join(project_root, path_str))
	if not _is_within_root(resolved, project_root):
		raise ValueError(f"Relative path escapes project root: {path_value}")
	return resolved


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


def extract_match_key(name, suffix):
	if suffix and name.endswith(suffix):
		stem = name[: -len(suffix)]
	else:
		stem = os.path.splitext(name)[0]

	tokens = [tok for tok in stem.split("_") if tok]
	if len(tokens) == 0:
		return stem

	running_idx = -1
	for idx, tok in enumerate(tokens):
		if tok.lower() == "running":
			running_idx = idx
			break

	if running_idx < 0:
		return stem

	h36m_idx = -1
	for idx in range(running_idx + 1, len(tokens)):
		if tokens[idx].lower().startswith("h36m"):
			h36m_idx = idx
			break

	if h36m_idx < 0:
		return "_".join(tokens[running_idx:])

	return "_".join(tokens[running_idx : h36m_idx + 1])


def build_id_to_path_map(input_dir, suffix):
	mapping = {}
	for name in collect_npz_files_by_suffix(input_dir, suffix):
		sample_id = extract_match_key(name, suffix)
		file_path = os.path.join(input_dir, name)
		if sample_id in mapping:
			raise ValueError(
				f"Duplicate match key '{sample_id}' in {input_dir}:\n"
				f"- {mapping[sample_id]}\n"
				f"- {file_path}"
			)
		mapping[sample_id] = file_path
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


def compute_root_trajectory_error_and_jitter(gt_pose_mm, pred_pose_mm, fps, root_kpt_idx=0):
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


def similarity_align_points_3d(src_points, dst_points):
	if src_points.ndim != 2 or dst_points.ndim != 2:
		raise ValueError("src_points and dst_points must be 2D arrays")
	if src_points.shape != dst_points.shape:
		raise ValueError(f"shape mismatch: src={src_points.shape}, dst={dst_points.shape}")
	if src_points.shape[1] != 3:
		raise ValueError("similarity alignment requires 3D points")

	mu_src = np.mean(src_points, axis=0)
	mu_dst = np.mean(dst_points, axis=0)
	src_c = src_points - mu_src
	dst_c = dst_points - mu_dst

	H = src_c.T @ dst_c
	U, S, Vt = np.linalg.svd(H)
	R = Vt.T @ U.T
	if np.linalg.det(R) < 0:
		Vt[-1, :] *= -1.0
		R = Vt.T @ U.T

	var_src = float(np.sum(src_c ** 2))
	if var_src > 1e-12:
		scale = float(np.sum(S) / var_src)
	else:
		scale = 1.0

	t = mu_dst - scale * (R @ mu_src)
	aligned = scale * (src_points @ R.T) + t
	return aligned, scale, R, t


def compute_pose_alignment_metrics(gt_pose_mm, pred_pose_mm, fps, time_sec=None, eps=1e-12):
	if gt_pose_mm.shape != pred_pose_mm.shape:
		raise ValueError(f"pose shape mismatch: gt={gt_pose_mm.shape}, pred={pred_pose_mm.shape}")
	if gt_pose_mm.ndim != 3 or gt_pose_mm.shape[2] != 3:
		raise ValueError(f"unexpected pose shape: {gt_pose_mm.shape}")
	if fps <= 0:
		raise ValueError(f"fps must be > 0, got {fps}")

	parent = get_default_parent_indices(gt_pose_mm.shape[1])
	nm_list = []
	for fidx in range(gt_pose_mm.shape[0]):
		gt_frame = gt_pose_mm[fidx]
		pred_frame = pred_pose_mm[fidx]

		sum_gt_bone = 0.0
		sum_pred_bone = 0.0
		for j in range(gt_frame.shape[0]):
			p = int(parent[j])
			if p < 0 or p >= gt_frame.shape[0]:
				continue
			sum_gt_bone += float(np.linalg.norm(gt_frame[j] - gt_frame[p]))
			sum_pred_bone += float(np.linalg.norm(pred_frame[j] - pred_frame[p]))
		if sum_pred_bone > 1e-12:
			s = float(sum_gt_bone / sum_pred_bone)
		else:
			s = 1.0
		pred_nm = s * pred_frame
		nm_list.append(float(np.linalg.norm(pred_nm - gt_frame, axis=1).mean()))

	if gt_pose_mm.shape[0] >= 2:
		if time_sec is not None and len(time_sec) == gt_pose_mm.shape[0]:
			dt = np.diff(np.asarray(time_sec, dtype=np.float64))
			valid = dt > float(eps)
			if np.any(valid):
				vel_gt = np.diff(gt_pose_mm, axis=0)[valid] / dt[valid, None, None]
				vel_pred = np.diff(pred_pose_mm, axis=0)[valid] / dt[valid, None, None]
				vel_diff = vel_pred - vel_gt
				mpjve_series = np.linalg.norm(vel_diff, axis=2).mean(axis=1)
				mpjve = float(np.mean(mpjve_series))
			else:
				mpjve_series = np.asarray([], dtype=np.float64)
				mpjve = float("nan")
		else:
			vel_gt = np.diff(gt_pose_mm, axis=0) * float(fps)
			vel_pred = np.diff(pred_pose_mm, axis=0) * float(fps)
			vel_diff = vel_pred - vel_gt
			mpjve_series = np.linalg.norm(vel_diff, axis=2).mean(axis=1)
			mpjve = float(np.mean(mpjve_series))
	else:
		mpjve_series = np.asarray([], dtype=np.float64)
		mpjve = float("nan")

	nm_arr = np.asarray(nm_list, dtype=np.float64)
	nm_p95 = float(np.percentile(nm_arr, 95.0)) if nm_arr.size > 0 else float("nan")
	nm_max = float(np.max(nm_arr)) if nm_arr.size > 0 else float("nan")
	mpjve_p95 = float(np.percentile(mpjve_series, 95.0)) if mpjve_series.size > 0 else float("nan")
	mpjve_max = float(np.max(mpjve_series)) if mpjve_series.size > 0 else float("nan")

	return {
		"nmpjpe_pose_upsampled_mm": float(np.mean(nm_list)) if nm_list else float("nan"),
		"nmpjpe_pose_upsampled_p95_mm": nm_p95,
		"nmpjpe_pose_upsampled_max_mm": nm_max,
		"mpjve_pose_upsampled_mmps": mpjve,
		"mpjve_pose_upsampled_p95_mmps": mpjve_p95,
		"mpjve_pose_upsampled_max_mmps": mpjve_max,
	}


H36M17_PARENT_INDICES = np.array(
	[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
	dtype=np.int64,
)

# H36M-17 joint index convention used by this project:
# 0 hip, 1 rhip, 2 rknee, 3 rfoot, 4 lhip, 5 lknee, 6 lfoot,
# 7 spine, 8 thorax, 9 neck, 10 head,
# 11 lshoulder, 12 lelbow, 13 lwrist,
# 14 rshoulder, 15 relbow, 16 rwrist.
H36M17_JOINT_NAMES = [
	"hip",
	"rhip",
	"rknee",
	"rfoot",
	"lhip",
	"lknee",
	"lfoot",
	"spine",
	"thorax",
	"neck",
	"head",
	"lshoulder",
	"lelbow",
	"lwrist",
	"rshoulder",
	"relbow",
	"rwrist",
]

# Bone list as (child_joint_idx, parent_joint_idx). This matches the figure.
H36M17_BONE_EDGES = [
	(1, 0),   # rhip -> hip
	(2, 1),   # rknee -> rhip
	(3, 2),   # rfoot -> rknee
	(4, 0),   # lhip -> hip
	(5, 4),   # lknee -> lhip
	(6, 5),   # lfoot -> lknee
	(7, 0),   # spine -> hip
	(8, 7),   # thorax -> spine
	(9, 8),   # neck -> thorax
	(10, 9),  # head -> neck
	(11, 8),  # lshoulder -> thorax
	(12, 11), # lelbow -> lshoulder
	(13, 12), # lwrist -> lelbow
	(14, 8),  # rshoulder -> thorax
	(15, 14), # relbow -> rshoulder
	(16, 15), # rwrist -> relbow
]


def get_default_parent_indices(num_joints):
	if num_joints == 17:
		return H36M17_PARENT_INDICES.copy()

	parent = np.full(num_joints, -1, dtype=np.int64)
	if num_joints >= 2:
		parent[1:] = np.arange(0, num_joints - 1, dtype=np.int64)
	return parent


def get_joint_edge_pairs(parent_indices, joint_idx):
	"""Get available edges around joint_idx for robust bone normalization."""
	parent = np.asarray(parent_indices, dtype=np.int64)
	num_joints = int(parent.shape[0])
	edges = []
	p = int(parent[joint_idx])
	if p >= 0 and p < num_joints:
		edges.append((joint_idx, p))
	else:
		children = np.where(parent == int(joint_idx))[0]
		for c in children:
			edges.append((int(c), int(joint_idx)))
	return edges


def compute_bone_length_normalized_mpjpe_percent(gt_pose_mm, pred_pose_mm, parent_indices=None, eps=1e-12):
	"""Compute relative MPJPE (%) normalized by bone length and displacement range.

	For each joint j with edges from get_joint_edge_pairs(parent, j):
	1) Per-frame proxy bone length:
	   L_j(t) = mean_{(a,b) in edges(j)} ||GT_a(t)-GT_b(t)||_2
	2) Mean proxy length: Lbar_j = mean_t L_j(t), for L_j(t) > eps
	3) Bone-normalized error: E_bone_j(t) = ||Pred_j(t)-GT_j(t)||_2 / Lbar_j
	4) Disp-range normalized error: E_disp_j(t) = ||Pred_j(t)-GT_j(t)||_2 / R_j
	   where R_j = ||max_t GT_j(t) - min_t GT_j(t)||_2

	Special case - ROOT JOINT (hip, j=0):
	- Hip has no parent (only child edges), so no bone-length normalization
	- Only raw MPJPE is computed for hip (added to per_joint but excluded from BL aggregation)

	Units: Input poses in millimeters (mm); output percentages (%).
	NMPJPE (Normalized MPJPE): Pose error normalized by per-joint displacement range.
	RTE (Root Trajectory Error): Rigid-aligned root trajectory error.

	Connectivity follows H36M17_PARENT_INDICES / H36M17_BONE_EDGES.
	"""
	if gt_pose_mm.shape != pred_pose_mm.shape:
		raise ValueError(f"pose shape mismatch: gt={gt_pose_mm.shape}, pred={pred_pose_mm.shape}")
	if gt_pose_mm.ndim != 3 or gt_pose_mm.shape[2] != 3:
		raise ValueError(f"unexpected pose shape: {gt_pose_mm.shape}")

	num_joints = gt_pose_mm.shape[1]
	if parent_indices is None:
		parent = get_default_parent_indices(num_joints)
	else:
		parent = np.asarray(parent_indices, dtype=np.int64)
		if parent.shape[0] != num_joints:
			raise ValueError(f"parent_indices length {parent.shape[0]} mismatch num_joints {num_joints}")

	joint_err = np.linalg.norm(pred_pose_mm - gt_pose_mm, axis=2)
	rel_err_bone = np.full_like(joint_err, np.nan, dtype=np.float64)
	rel_err_disp = np.full_like(joint_err, np.nan, dtype=np.float64)
	bone_len_mean = np.full(num_joints, np.nan, dtype=np.float64)
	disp_range = np.full(num_joints, np.nan, dtype=np.float64)

	for j in range(num_joints):
		# Special case: root joint (hip) has parent=-1, no bone-length normalization.
		# Only store raw error, will skip BL aggregation for hip.
		if j == 0:  # hip (root)
			rel_err_bone[:, j] = np.full(gt_pose_mm.shape[0], np.nan, dtype=np.float64)
			continue
		
		edges = get_joint_edge_pairs(parent, j)
		if len(edges) == 0:
			continue

		bone_series = []
		for a, b in edges:
			bone_series.append(np.linalg.norm(gt_pose_mm[:, a, :] - gt_pose_mm[:, b, :], axis=1))
		bone_len_t = np.mean(np.stack(bone_series, axis=0), axis=0)
		valid_bone = bone_len_t > float(eps)
		if not np.any(valid_bone):
			continue

		bone_len_mean[j] = float(np.mean(bone_len_t[valid_bone]))
		rel_err_bone[:, j] = joint_err[:, j] / bone_len_mean[j]

		joint_min = np.min(gt_pose_mm[:, j, :], axis=0)
		joint_max = np.max(gt_pose_mm[:, j, :], axis=0)
		rng = float(np.linalg.norm(joint_max - joint_min))
		if rng > float(eps):
			disp_range[j] = rng
			rel_err_disp[:, j] = joint_err[:, j] / rng

	per_joint_percent = np.full(num_joints, np.nan, dtype=np.float64)
	per_joint_disp_percent = np.full(num_joints, np.nan, dtype=np.float64)
	for j in range(num_joints):
		v = np.isfinite(rel_err_bone[:, j])
		if np.any(v):
			per_joint_percent[j] = float(np.mean(rel_err_bone[v, j]) * 100.0)

		vd = np.isfinite(rel_err_disp[:, j])
		if np.any(vd):
			per_joint_disp_percent[j] = float(np.mean(rel_err_disp[vd, j]) * 100.0)

	v_global = np.isfinite(rel_err_bone)
	if np.any(v_global):
		global_percent = float(np.mean(rel_err_bone[v_global]) * 100.0)
		global_p95 = float(np.percentile(rel_err_bone[v_global] * 100.0, 95.0))
		global_max = float(np.max(rel_err_bone[v_global] * 100.0))
	else:
		global_percent = float("nan")
		global_p95 = float("nan")
		global_max = float("nan")

	v_global_disp = np.isfinite(rel_err_disp)
	if np.any(v_global_disp):
		global_disp_percent = float(np.mean(rel_err_disp[v_global_disp]) * 100.0)
		global_disp_p95 = float(np.percentile(rel_err_disp[v_global_disp] * 100.0, 95.0))
		global_disp_max = float(np.max(rel_err_disp[v_global_disp] * 100.0))
	else:
		global_disp_percent = float("nan")
		global_disp_p95 = float("nan")
		global_disp_max = float("nan")

	return {
		"bl_mpjpe_percent_pose_upsampled": global_percent,
		"bl_mpjpe_p95_percent_pose_upsampled": global_p95,
		"bl_mpjpe_max_percent_pose_upsampled": global_max,
		"bl_per_joint_percent_pose_upsampled": per_joint_percent,
		"bone_length_per_joint_mm": bone_len_mean,
		"bl_valid_joint_count_pose_upsampled": int(np.sum(np.isfinite(per_joint_percent))),
		"disp_range_mpjpe_percent_pose_upsampled": global_disp_percent,
		"disp_range_mpjpe_p95_percent_pose_upsampled": global_disp_p95,
		"disp_range_mpjpe_max_percent_pose_upsampled": global_disp_max,
		"disp_range_per_joint_percent_pose_upsampled": per_joint_disp_percent,
		"disp_range_per_joint_mm": disp_range,
		"disp_range_valid_joint_count_pose_upsampled": int(np.sum(np.isfinite(per_joint_disp_percent))),
	}


def compute_bone_length_normalized_mpjve_percent(gt_pose_mm, pred_pose_mm, fps, parent_indices=None, eps=1e-12):
	"""Compute velocity error (MPJVE) normalized by bone length - BL-MPJVE.
	
	For each joint j with edges from get_joint_edge_pairs(parent, j):
	1) Velocity error: vel_err_j(t) = ||vel_pred_j(t) - vel_gt_j(t)||_2 where vel = dp/dt
	2) Per-joint mean bone length: Lbar_j = mean_t L_j(t), for L_j(t) > eps
	3) Bone-normalized velocity error: E_vel_bone_j(t) = vel_err_j(t) / Lbar_j
	4) Global BL-MPJVE = mean(E_vel_bone) * 100 (%)
	
	Special case - ROOT JOINT (hip, j=0):
	- Hip has no parent, so no bone-length normalization for velocity
	- Raw MPJVE is computed but excluded from BL-MPJVE aggregation
	
	Units: Input poses in mm, fps in frames/second; output in %/100 and mm/s units.
	
	Args:
		gt_pose_mm: Ground-truth pose (T, K, 3) in mm
		pred_pose_mm: Predicted pose (T, K, 3) in mm
		fps: Frames per second for velocity calculation (dt = 1/fps)
		parent_indices: Parent indices (default: H36M17_PARENT_INDICES)
		eps: Threshold for valid bone length
		
	Returns:
		Dict with BL-MPJVE metrics (mean/p95/max), raw MPJVE, per-joint values.
	"""
	if gt_pose_mm.shape != pred_pose_mm.shape:
		raise ValueError(f"pose shape mismatch: gt={gt_pose_mm.shape}, pred={pred_pose_mm.shape}")
	if gt_pose_mm.ndim != 3 or gt_pose_mm.shape[2] != 3:
		raise ValueError(f"unexpected pose shape: {gt_pose_mm.shape}")
	if fps <= 0:
		raise ValueError(f"fps must be > 0, got {fps}")
	if gt_pose_mm.shape[0] < 2:
		num_joints = gt_pose_mm.shape[1]
		return {
			"bl_mpjve_percent_pose_upsampled": float("nan"),
			"bl_mpjve_p95_percent_pose_upsampled": float("nan"),
			"bl_mpjve_max_percent_pose_upsampled": float("nan"),
			"bl_per_joint_percent_pose_upsampled": [float("nan")] * num_joints,
			"bl_valid_joint_count_pose_upsampled": 0,
			"mpjve_mmps": float("nan"),
			"mpjve_per_joint_mmps": [float("nan")] * num_joints,
		}
	
	num_joints = gt_pose_mm.shape[1]
	if parent_indices is None:
		parent = get_default_parent_indices(num_joints)
	else:
		parent = np.asarray(parent_indices, dtype=np.int64)
		if parent.shape[0] != num_joints:
			raise ValueError(f"parent_indices length {parent.shape[0]} mismatch num_joints {num_joints}")
	
	# Compute velocities: (T-1, K, 3)
	vel_gt = np.diff(gt_pose_mm, axis=0) * float(fps)
	vel_pred = np.diff(pred_pose_mm, axis=0) * float(fps)
	
	# Velocity errors: (T-1, K)
	vel_err = np.linalg.norm(vel_pred - vel_gt, axis=2)
	
	# Bone-length normalized velocity error: (T-1, K)
	rel_vel_err_bone = np.full_like(vel_err, np.nan, dtype=np.float64)
	bone_len_mean = np.full(num_joints, np.nan, dtype=np.float64)
	
	for j in range(num_joints):
		# Hip (j=0) has no parent, skip BL normalization
		if j == 0:
			continue
		
		edges = get_joint_edge_pairs(parent, j)
		if len(edges) == 0:
			continue

		bone_series = []
		for a, b in edges:
			bone_series.append(np.linalg.norm(gt_pose_mm[:, a, :] - gt_pose_mm[:, b, :], axis=1))
		bone_len_t = np.mean(np.stack(bone_series, axis=0), axis=0)
		valid_bone = bone_len_t > float(eps)
		if not np.any(valid_bone):
			continue

		bone_len_mean[j] = float(np.mean(bone_len_t[valid_bone]))
		rel_vel_err_bone[:, j] = vel_err[:, j] / bone_len_mean[j]
	
	# Per-joint metrics
	per_joint_percent = np.full(num_joints, np.nan, dtype=np.float64)
	per_joint_raw_mmps = np.full(num_joints, np.nan, dtype=np.float64)
	for j in range(num_joints):
		v = np.isfinite(rel_vel_err_bone[:, j])
		if np.any(v):
			per_joint_percent[j] = float(np.mean(rel_vel_err_bone[v, j]) * 100.0)
		
		raw_v = np.isfinite(vel_err[:, j])
		if np.any(raw_v):
			per_joint_raw_mmps[j] = float(np.mean(vel_err[raw_v, j]))
	
	# Global metrics (exclude hip j=0)
	v_global = np.isfinite(rel_vel_err_bone)
	if np.any(v_global):
		global_percent = float(np.mean(rel_vel_err_bone[v_global]) * 100.0)
		global_p95 = float(np.percentile(rel_vel_err_bone[v_global] * 100.0, 95.0))
		global_max = float(np.max(rel_vel_err_bone[v_global] * 100.0))
	else:
		global_percent = float("nan")
		global_p95 = float("nan")
		global_max = float("nan")
	
	raw_global = np.isfinite(vel_err)
	if np.any(raw_global):
		raw_global_mmps = float(np.mean(vel_err[raw_global]))
	else:
		raw_global_mmps = float("nan")
	
	return {
		"bl_mpjve_percent_pose_upsampled": global_percent,
		"bl_mpjve_p95_percent_pose_upsampled": global_p95,
		"bl_mpjve_max_percent_pose_upsampled": global_max,
		"bl_per_joint_percent_pose_upsampled": per_joint_percent,
		"bone_length_per_joint_mm": bone_len_mean,
		"bl_valid_joint_count_pose_upsampled": int(np.sum(np.isfinite(per_joint_percent))),
		"mpjve_mmps": raw_global_mmps,
		"mpjve_per_joint_mmps": per_joint_raw_mmps,
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
		"vel_sq_integral": vel_sq_integral,
		"acc_sq_integral": acc_sq_integral,
		"channels": channels,
		"duration_sec": total_duration,
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

	if len(all_joint_errors) == 0:
		raise ValueError("No sampled points generated for sampled metrics")

	joint_errors = np.concatenate(all_joint_errors, axis=0)
	mpjpe_cont = mpjpe_integral / total_duration

	gt_pos_range = float(gt_pos_max - gt_pos_min)
	gt_vel_range = float(gt_vel_max - gt_vel_min)
	gt_acc_range = float(gt_acc_max - gt_acc_min)

	return {
		"mpjpe_cont_mm": float(mpjpe_cont),
		"mpjpe_integral_mm_sec": float(mpjpe_integral),
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
	joint_err = np.linalg.norm(pred_valid - gt_valid, axis=2)
	mpjpe_series = joint_err.mean(axis=1)

	rte_jitter = compute_root_trajectory_error_and_jitter(gt_valid, pred_valid, float(upsample_fps), root_kpt_idx=0)
	align = compute_pose_alignment_metrics(
		gt_valid,
		pred_valid,
		float(upsample_fps),
		time_sec=target_ts_valid,
	)
	bone_norm = compute_bone_length_normalized_mpjpe_percent(gt_valid, pred_valid)

	ret = {
		"upsample_fps": float(upsample_fps),
		"num_upsample_points": int(pred_valid.shape[0]),
		"target_ts_valid": target_ts_valid,
		"pose_joint_err_sum_mm": float(np.sum(joint_err)),
		"mpjpe_pose_upsampled_mm": float(np.mean(mpjpe_series)),
		"mpjpe_pose_upsampled_p95_mm": float(np.percentile(mpjpe_series, 95.0)),
		"mpjpe_pose_upsampled_max_mm": float(np.max(mpjpe_series)),
		"p95_joint_err_pose_upsampled_mm": float(np.percentile(joint_err, 95.0)),
		"max_joint_err_pose_upsampled_mm": float(np.max(joint_err)),
		"rte_percent_pose_upsampled": rte_jitter["rte_percent_pose_upsampled"],
		"jitter_10mps3_pose_upsampled": rte_jitter["jitter_10mps3_pose_upsampled"],
		"jitter_mps3_pose_upsampled": rte_jitter["jitter_mps3_pose_upsampled"],
	}
	ret.update(align)
	ret.update(bone_norm)
	ret.update(
		compute_channelwise_pose_metrics(
			gt_pose_mm=gt_valid,
			pred_pose_mm=pred_valid,
			time_sec=target_ts_valid,
			fps=float(upsample_fps),
		)
	)
	return ret


def compute_linear_interp_reference_metrics(pose_time_sec, pose_data, eval_ts, downsample_fps, eval_fps):
	ctrl_ts, ctrl_pose = downsample_pose_linear(pose_time_sec, pose_data, downsample_fps)
	lin_pred_all, lin_valid = eval_pose_linear_from_controls(ctrl_ts, ctrl_pose, eval_ts)
	gt_eval, gt_valid = resample_pose_at_times(pose_time_sec, pose_data, eval_ts)
	valid = lin_valid & gt_valid
	if not np.any(valid):
		raise ValueError("No valid samples for linear interpolation reference metrics")

	pred_valid = lin_pred_all[valid]
	gt_valid_pose = gt_eval[valid]
	diff = pred_valid - gt_valid_pose
	joint_err = np.linalg.norm(diff, axis=2)
	mpjpe_series = joint_err.mean(axis=1)
	eval_ts_valid = np.asarray(eval_ts, dtype=np.float64)[valid]
	align = compute_pose_alignment_metrics(
		gt_valid_pose,
		pred_valid,
		fps=float(eval_fps),
		time_sec=eval_ts_valid,
	)
	bone_norm = compute_bone_length_normalized_mpjpe_percent(gt_valid_pose, pred_valid)
	rte_jitter = compute_root_trajectory_error_and_jitter(gt_valid_pose, pred_valid, float(eval_fps), root_kpt_idx=0)

	ret = {
		"linear_downsample_fps": float(downsample_fps),
		"linear_num_upsample_points": int(pred_valid.shape[0]),
		"linear_mpjpe_pose_upsampled_mm": float(np.mean(mpjpe_series)),
		"linear_mpjpe_pose_upsampled_p95_mm": float(np.percentile(mpjpe_series, 95.0)),
		"linear_mpjpe_pose_upsampled_max_mm": float(np.max(mpjpe_series)),
		"linear_p95_joint_err_pose_upsampled_mm": float(np.percentile(joint_err, 95.0)),
		"linear_max_joint_err_pose_upsampled_mm": float(np.max(joint_err)),
	}
	ret["linear_nmpjpe_pose_upsampled_mm"] = align["nmpjpe_pose_upsampled_mm"]
	ret["linear_nmpjpe_pose_upsampled_p95_mm"] = align["nmpjpe_pose_upsampled_p95_mm"]
	ret["linear_nmpjpe_pose_upsampled_max_mm"] = align["nmpjpe_pose_upsampled_max_mm"]
	ret["linear_mpjve_pose_upsampled_mmps"] = align["mpjve_pose_upsampled_mmps"]
	ret["linear_mpjve_pose_upsampled_p95_mmps"] = align["mpjve_pose_upsampled_p95_mmps"]
	ret["linear_mpjve_pose_upsampled_max_mmps"] = align["mpjve_pose_upsampled_max_mmps"]
	ret["linear_bl_mpjpe_percent_pose_upsampled"] = bone_norm["bl_mpjpe_percent_pose_upsampled"]
	ret["linear_bl_mpjpe_p95_percent_pose_upsampled"] = bone_norm["bl_mpjpe_p95_percent_pose_upsampled"]
	ret["linear_bl_mpjpe_max_percent_pose_upsampled"] = bone_norm["bl_mpjpe_max_percent_pose_upsampled"]
	ret["linear_disp_range_mpjpe_percent_pose_upsampled"] = bone_norm["disp_range_mpjpe_percent_pose_upsampled"]
	ret["linear_disp_range_mpjpe_p95_percent_pose_upsampled"] = bone_norm["disp_range_mpjpe_p95_percent_pose_upsampled"]
	ret["linear_disp_range_mpjpe_max_percent_pose_upsampled"] = bone_norm["disp_range_mpjpe_max_percent_pose_upsampled"]
	ret["linear_bl_per_joint_percent_pose_upsampled"] = bone_norm["bl_per_joint_percent_pose_upsampled"]
	ret["linear_bone_length_per_joint_mm"] = bone_norm["bone_length_per_joint_mm"]
	ret["linear_disp_range_per_joint_percent_pose_upsampled"] = bone_norm["disp_range_per_joint_percent_pose_upsampled"]
	ret["linear_disp_range_per_joint_mm"] = bone_norm["disp_range_per_joint_mm"]
	ret["linear_rte_percent_pose_upsampled"] = rte_jitter["rte_percent_pose_upsampled"]
	ret["linear_jitter_10mps3_pose_upsampled"] = rte_jitter["jitter_10mps3_pose_upsampled"]
	ch_stats = compute_channelwise_pose_metrics(
		gt_pose_mm=gt_valid_pose,
		pred_pose_mm=pred_valid,
		time_sec=eval_ts_valid,
		fps=float(eval_fps),
	)
	for k, v in ch_stats.items():
		ret[f"linear_{k}"] = v
	return ret


def compute_channelwise_pose_metrics(gt_pose_mm, pred_pose_mm, time_sec=None, fps=120.0, eps=1e-12):
	"""Compute per-channel metrics for all 51 channels and aggregate mean/max/p95."""
	if gt_pose_mm.shape != pred_pose_mm.shape:
		raise ValueError(f"pose shape mismatch: gt={gt_pose_mm.shape}, pred={pred_pose_mm.shape}")
	if gt_pose_mm.ndim != 3 or gt_pose_mm.shape[2] != 3:
		raise ValueError(f"unexpected pose shape: {gt_pose_mm.shape}")

	parent = get_default_parent_indices(gt_pose_mm.shape[1])
	metric_lists = {
		"mpjpe_mm": [],
		"nmpjpe_mm": [],
		"mpjve_mmps": [],
		"bl_mpjpe_percent": [],
		"disp_range_mpjpe_percent": [],
	}

	for j in range(gt_pose_mm.shape[1]):
		p = int(parent[j])
		if p >= 0 and p < gt_pose_mm.shape[1]:
			bone_gt_t = np.linalg.norm(gt_pose_mm[:, j, :] - gt_pose_mm[:, p, :], axis=1)
			bone_pred_t = np.linalg.norm(pred_pose_mm[:, j, :] - pred_pose_mm[:, p, :], axis=1)
			bone_mean = float(np.mean(bone_gt_t)) if bone_gt_t.size > 0 else float("nan")
			sum_pred_bone = float(np.sum(bone_pred_t)) if bone_pred_t.size > 0 else 0.0
			if sum_pred_bone > eps:
				s_nm = float(np.sum(bone_gt_t) / sum_pred_bone)
			else:
				s_nm = 1.0
		else:
			bone_mean = float("nan")
			s_nm = 1.0

		joint_min = np.min(gt_pose_mm[:, j, :], axis=0)
		joint_max = np.max(gt_pose_mm[:, j, :], axis=0)
		disp_range = float(np.linalg.norm(joint_max - joint_min))

		for d in range(gt_pose_mm.shape[2]):
			gt = gt_pose_mm[:, j, d]
			pred = pred_pose_mm[:, j, d]
			diff = pred - gt
			abs_err = np.abs(diff)

			mpjpe = float(np.mean(abs_err)) if abs_err.size > 0 else float("nan")

			nmp = float(np.mean(np.abs(s_nm * pred - gt)))

			if gt.shape[0] >= 2:
				if time_sec is not None and len(time_sec) == len(gt):
					dt = np.diff(np.asarray(time_sec, dtype=np.float64))
					valid_dt = dt > eps
					if np.any(valid_dt):
						vel_gt = np.diff(gt)[valid_dt] / dt[valid_dt]
						vel_pred = np.diff(pred)[valid_dt] / dt[valid_dt]
						mpjve = float(np.mean(np.abs(vel_pred - vel_gt)))
					else:
						mpjve = float("nan")
				else:
					vel_gt = np.diff(gt) * float(fps)
					vel_pred = np.diff(pred) * float(fps)
					mpjve = float(np.mean(np.abs(vel_pred - vel_gt)))
			else:
				mpjve = float("nan")

			if np.isfinite(bone_mean) and bone_mean > eps:
				bl = float(np.mean(abs_err / bone_mean) * 100.0)
			else:
				bl = float("nan")

			if np.isfinite(disp_range) and disp_range > eps:
				drp = float(np.mean(abs_err / disp_range) * 100.0)
			else:
				drp = float("nan")

			metric_lists["mpjpe_mm"].append(mpjpe)
			metric_lists["nmpjpe_mm"].append(nmp)
			metric_lists["mpjve_mmps"].append(mpjve)
			metric_lists["bl_mpjpe_percent"].append(bl)
			metric_lists["disp_range_mpjpe_percent"].append(drp)

	out = {}
	for name, vals in metric_lists.items():
		arr = np.asarray(vals, dtype=np.float64)
		valid = arr[np.isfinite(arr)]
		if valid.size > 0:
			out[f"ch_mean_{name}"] = float(np.mean(valid))
			out[f"ch_max_{name}"] = float(np.max(valid))
			out[f"ch_p95_{name}"] = float(np.percentile(valid, 95.0))
			out[f"ch_valid_count_{name}"] = int(valid.size)
		else:
			out[f"ch_mean_{name}"] = float("nan")
			out[f"ch_max_{name}"] = float("nan")
			out[f"ch_p95_{name}"] = float("nan")
			out[f"ch_valid_count_{name}"] = 0

	return out


def summarize_results(rows, analytic_global):
	summary = {
		"num_files": len(rows),
	}
	if len(rows) == 0:
		return summary

	# Only keep the required metrics in summary
	numeric_keys = [
		# MPJPE
		"mpjpe_pose_upsampled_mm",
		"mpjpe_pose_upsampled_p95_mm",
		"mpjpe_pose_upsampled_max_mm",
		# MPJVE
		"mpjve_pose_upsampled_mmps",
		"mpjve_pose_upsampled_p95_mmps",
		"mpjve_pose_upsampled_max_mmps",
		# MPJPE_BL
		"bl_mpjpe_percent_pose_upsampled",
		"bl_mpjpe_p95_percent_pose_upsampled",
		"bl_mpjpe_max_percent_pose_upsampled",
		# RTE and Jitter
		"rte_percent_pose_upsampled",
		"jitter_10mps3_pose_upsampled",
		# Linear MPJPE
		"linear_mpjpe_pose_upsampled_mm",
		"linear_mpjpe_pose_upsampled_p95_mm",
		"linear_mpjpe_pose_upsampled_max_mm",
		# Linear MPJVE
		"linear_mpjve_pose_upsampled_mmps",
		"linear_mpjve_pose_upsampled_p95_mmps",
		"linear_mpjve_pose_upsampled_max_mmps",
		# Linear MPJPE_BL
		"linear_bl_mpjpe_percent_pose_upsampled",
		"linear_bl_mpjpe_p95_percent_pose_upsampled",
		"linear_bl_mpjpe_max_percent_pose_upsampled",
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


def write_rows_csv_compact(rows, csv_path):
	"""Write compact metrics CSV with each file taking 2 rows (pred and linear)."""
	os.makedirs(os.path.dirname(csv_path), exist_ok=True)
	if len(rows) == 0:
		with open(csv_path, "w", newline="") as fp:
			fp.write("sample_id\n")
		return

	# Define metric columns to keep (in order)
	metric_cols = [
		"V-MPJPE(mm)", "p95-MPJPE(mm)", "Max-MPJPE(mm)",
		"V-MPJVE(mm/s)", "p95-MPJVE(mm/s)", "Max-MPJVE(mm/s)",
		"V-MPJPE_BL(%)", "p95-MPJPE_BL(%)", "Max-MPJPE_BL(%)",
		"RTE(%)", "Jitter(10mps3)"
	]

	# Build output rows with sample_id and method columns
	output_rows = []
	for row in rows:
		# Row 1: pred values
		pred_row = {"sample_id": row["sample_id"], "method": "pred"}
		pred_row["V-MPJPE(mm)"] = row.get("mpjpe_pose_upsampled_mm", float("nan"))
		pred_row["p95-MPJPE(mm)"] = row.get("mpjpe_pose_upsampled_p95_mm", float("nan"))
		pred_row["Max-MPJPE(mm)"] = row.get("mpjpe_pose_upsampled_max_mm", float("nan"))
		pred_row["V-MPJVE(mm/s)"] = row.get("mpjve_pose_upsampled_mmps", float("nan"))
		pred_row["p95-MPJVE(mm/s)"] = row.get("mpjve_pose_upsampled_p95_mmps", float("nan"))
		pred_row["Max-MPJVE(mm/s)"] = row.get("mpjve_pose_upsampled_max_mmps", float("nan"))
		pred_row["V-MPJPE_BL(%)"] = row.get("bl_mpjpe_percent_pose_upsampled", float("nan"))
		pred_row["p95-MPJPE_BL(%)"] = row.get("bl_mpjpe_p95_percent_pose_upsampled", float("nan"))
		pred_row["Max-MPJPE_BL(%)"] = row.get("bl_mpjpe_max_percent_pose_upsampled", float("nan"))
		pred_row["RTE(%)"] = row.get("rte_percent_pose_upsampled", float("nan"))
		pred_row["Jitter(10mps3)"] = row.get("jitter_10mps3_pose_upsampled", float("nan"))
		output_rows.append(pred_row)

		# Row 2: linear values
		linear_row = {"sample_id": row["sample_id"], "method": "linear"}
		linear_row["V-MPJPE(mm)"] = row.get("linear_mpjpe_pose_upsampled_mm", float("nan"))
		linear_row["p95-MPJPE(mm)"] = row.get("linear_mpjpe_pose_upsampled_p95_mm", float("nan"))
		linear_row["Max-MPJPE(mm)"] = row.get("linear_mpjpe_pose_upsampled_max_mm", float("nan"))
		linear_row["V-MPJVE(mm/s)"] = row.get("linear_mpjve_pose_upsampled_mmps", float("nan"))
		linear_row["p95-MPJVE(mm/s)"] = row.get("linear_mpjve_pose_upsampled_p95_mmps", float("nan"))
		linear_row["Max-MPJVE(mm/s)"] = row.get("linear_mpjve_pose_upsampled_max_mmps", float("nan"))
		linear_row["V-MPJPE_BL(%)"] = row.get("linear_bl_mpjpe_percent_pose_upsampled", float("nan"))
		linear_row["p95-MPJPE_BL(%)"] = row.get("linear_bl_mpjpe_p95_percent_pose_upsampled", float("nan"))
		linear_row["Max-MPJPE_BL(%)"] = row.get("linear_bl_mpjpe_max_percent_pose_upsampled", float("nan"))
		linear_row["RTE(%)"] = row.get("linear_rte_percent_pose_upsampled", float("nan"))
		linear_row["Jitter(10mps3)"] = row.get("linear_jitter_10mps3_pose_upsampled", float("nan"))
		output_rows.append(linear_row)

	# Write CSV with sample_id, method, and metrics
	fieldnames = ["sample_id", "method"] + metric_cols
	with open(csv_path, "w", newline="") as fp:
		writer = csv.DictWriter(fp, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(output_rows)


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
	gt_dir = resolve_runtime_path(gt_dir)
	pred_dir = resolve_runtime_path(pred_dir)
	gt_pose_dir = resolve_runtime_path(gt_pose_dir)
	output_dir = resolve_runtime_path(output_dir)

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
				eval_fps=upsample_fps,
			)

			row = {
				"sample_id": sample_id,
				# MPJPE metrics (pred)
				"mpjpe_pose_upsampled_mm": pose_up["mpjpe_pose_upsampled_mm"],
				"mpjpe_pose_upsampled_p95_mm": pose_up["mpjpe_pose_upsampled_p95_mm"],
				"mpjpe_pose_upsampled_max_mm": pose_up["mpjpe_pose_upsampled_max_mm"],
				# MPJVE metrics (pred)
				"mpjve_pose_upsampled_mmps": pose_up["mpjve_pose_upsampled_mmps"],
				"mpjve_pose_upsampled_p95_mmps": pose_up["mpjve_pose_upsampled_p95_mmps"],
				"mpjve_pose_upsampled_max_mmps": pose_up["mpjve_pose_upsampled_max_mmps"],
				# MPJPE_BL metrics (pred)
				"bl_mpjpe_percent_pose_upsampled": pose_up["bl_mpjpe_percent_pose_upsampled"],
				"bl_mpjpe_p95_percent_pose_upsampled": pose_up["bl_mpjpe_p95_percent_pose_upsampled"],
				"bl_mpjpe_max_percent_pose_upsampled": pose_up["bl_mpjpe_max_percent_pose_upsampled"],
				# RTE and Jitter (pred)
				"rte_percent_pose_upsampled": pose_up["rte_percent_pose_upsampled"],
				"jitter_10mps3_pose_upsampled": pose_up["jitter_10mps3_pose_upsampled"],
				# MPJPE metrics (linear)
				"linear_mpjpe_pose_upsampled_mm": linear_ref["linear_mpjpe_pose_upsampled_mm"],
				"linear_mpjpe_pose_upsampled_p95_mm": linear_ref["linear_mpjpe_pose_upsampled_p95_mm"],
				"linear_mpjpe_pose_upsampled_max_mm": linear_ref["linear_mpjpe_pose_upsampled_max_mm"],
				# MPJVE metrics (linear)
				"linear_mpjve_pose_upsampled_mmps": linear_ref["linear_mpjve_pose_upsampled_mmps"],
				"linear_mpjve_pose_upsampled_p95_mmps": linear_ref["linear_mpjve_pose_upsampled_p95_mmps"],
				"linear_mpjve_pose_upsampled_max_mmps": linear_ref["linear_mpjve_pose_upsampled_max_mmps"],
				# MPJPE_BL metrics (linear)
				"linear_bl_mpjpe_percent_pose_upsampled": linear_ref["linear_bl_mpjpe_percent_pose_upsampled"],
				"linear_bl_mpjpe_p95_percent_pose_upsampled": linear_ref["linear_bl_mpjpe_p95_percent_pose_upsampled"],
				"linear_bl_mpjpe_max_percent_pose_upsampled": linear_ref["linear_bl_mpjpe_max_percent_pose_upsampled"],
				# RTE and Jitter (linear)
				"linear_rte_percent_pose_upsampled": linear_ref["linear_rte_percent_pose_upsampled"],
				"linear_jitter_10mps3_pose_upsampled": linear_ref["linear_jitter_10mps3_pose_upsampled"],
			}
			rows.append(row)

			total_vel_sq += analytic["vel_sq_integral"]
			total_acc_sq += analytic["acc_sq_integral"]
			total_duration_channels += analytic["duration_sec"] * analytic["channels"]

			print(
				f"[{idx}/{len(matched_ids)}] {sample_id}: "
				f"MPJPE(V/p95/Max)={pose_up['mpjpe_pose_upsampled_mm']:.2f}/"
				f"{pose_up['mpjpe_pose_upsampled_p95_mm']:.2f}/"
				f"{pose_up['mpjpe_pose_upsampled_max_mm']:.2f} mm, "
				f"MPJVE(V/p95/Max)={pose_up['mpjve_pose_upsampled_mmps']:.2f}/"
				f"{pose_up['mpjve_pose_upsampled_p95_mmps']:.2f}/"
				f"{pose_up['mpjve_pose_upsampled_max_mmps']:.2f} mm/s, "
				f"MPJPE_BL(V/p95/Max)={pose_up['bl_mpjpe_percent_pose_upsampled']:.2f}/"
				f"{pose_up['bl_mpjpe_p95_percent_pose_upsampled']:.2f}/"
				f"{pose_up['bl_mpjpe_max_percent_pose_upsampled']:.2f}%, "
				f"RTE={pose_up['rte_percent_pose_upsampled']:.2f}%, "
				f"Jitter={pose_up['jitter_10mps3_pose_upsampled']:.4f}"
			)
		except Exception as exc:
			failed += 1
			print(f"[{idx}/{len(matched_ids)}] Failed {sample_id}: {exc}")

	analytic_global = None
	if total_duration_channels > 0:
		global_vel_mse = clamp_small_negative(total_vel_sq / total_duration_channels, tol=1e-15)
		global_acc_mse = clamp_small_negative(total_acc_sq / total_duration_channels, tol=1e-12)
		analytic_global = {
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

	write_rows_csv_compact(rows, csv_path)
	with open(summary_path, "w") as fp:
		json.dump(summary, fp, indent=2)

	print("=" * 60)
	print(f"Saved per-file metrics: {csv_path}")
	print(f"Saved summary: {summary_path}")
	if analytic_global is not None:
		print(
			"Global analytic metrics: "
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
		default="res/splines_fit_baseline",
		help="Prediction spline directory (e.g., Kalman, ABG or Baseline realtime output).",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="res/splines_metrics_batch",
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
		gt_splines_dir = "/Users/twz/demo_sys_user/h36m_pose_cam_1/test/S2_cam_1_120fps_notaknot_splines/"
		pred_splines_dir = "res/splines_fit_baseline"
		gt_pose_dir = "/Users/twz/demo_sys_user/h36m_pose_cam_1/test/S2_cam_1_120fps"
		output_dir = "res/splines_metrics_batch"
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
