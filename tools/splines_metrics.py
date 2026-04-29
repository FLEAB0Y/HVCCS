import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.artist import Artist
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


def similarity_align_points_3d(src_points, dst_points):
	"""Similarity-align src to dst with Umeyama (scale + rotation + translation)."""
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
	"""Compute N-MPJPE/MPJVE on synchronized pose samples."""
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
			valid_dt = dt > eps
			if np.any(valid_dt):
				vel_gt = np.diff(gt_pose_mm, axis=0)[valid_dt] / dt[valid_dt, None, None]
				vel_pred = np.diff(pred_pose_mm, axis=0)[valid_dt] / dt[valid_dt, None, None]
				vel_diff = vel_pred - vel_gt
				mpjve = float(np.linalg.norm(vel_diff, axis=2).mean())
			else:
				mpjve = float("nan")
		else:
			vel_gt = np.diff(gt_pose_mm, axis=0) * float(fps)
			vel_pred = np.diff(pred_pose_mm, axis=0) * float(fps)
			vel_diff = vel_pred - vel_gt
			mpjve = float(np.linalg.norm(vel_diff, axis=2).mean())
	else:
		mpjve = float("nan")

	return {
		"NMPJPE_upsampled_mm": float(np.mean(nm_list)) if nm_list else float("nan"),
		"MPJVE_upsampled_mmps": mpjve,
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

	# Fallback: simple chain for unknown layouts.
	parent = np.full(num_joints, -1, dtype=np.int64)
	if num_joints >= 2:
		parent[1:] = np.arange(0, num_joints - 1, dtype=np.int64)
	return parent


def get_joint_edge_pairs(parent_indices, joint_idx):
	"""Get available bone edges touching joint_idx.

	Preferred edge is (joint_idx -> parent). If parent is absent (e.g., root),
	fall back to child edges to avoid NaN bone-normalized metrics.
	"""
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


def compute_joint_bone_proxy_series(pose_mm, parent_indices, joint_idx, eps=1e-12):
	"""Return per-frame bone-length proxy for one joint using available adjacent bones."""
	edges = get_joint_edge_pairs(parent_indices, joint_idx)
	if len(edges) == 0:
		return np.asarray([], dtype=np.float64)

	series = []
	for a, b in edges:
		edge_len = np.linalg.norm(pose_mm[:, a, :] - pose_mm[:, b, :], axis=1)
		series.append(edge_len)
	stacked = np.stack(series, axis=0)
	bone_t = np.mean(stacked, axis=0)
	valid = bone_t > float(eps)
	return bone_t[valid]


def compute_bone_length_normalized_mpjpe_percent(gt_pose_mm, pred_pose_mm, parent_indices=None, eps=1e-12):
	"""Compute relative MPJPE (%) normalized by bone length and displacement range.

	For each joint j with valid parent p (excluding root hip):
	1) Per-frame bone length: L_j(t) = ||GT_j(t) - GT_p(t)||_2
	2) Per-joint mean bone length: Lbar_j = mean_t L_j(t), for L_j(t) > eps
	3) Bone-normalized error: E_bone_j(t) = ||Pred_j(t)-GT_j(t)||_2 / Lbar_j
	4) Disp-range normalized error: E_disp_j(t) = ||Pred_j(t)-GT_j(t)||_2 / R_j
	   where R_j = ||max_t GT_j(t) - min_t GT_j(t)||_2

	Special case - ROOT JOINT (hip, j=0):
	- Hip has no parent (parent=-1), so no bone-length normalization
	- Only raw MPJPE is computed for hip (added to per_joint but skipped in BL calculation)

	Note: joint-parent connectivity follows H36M17_PARENT_INDICES / H36M17_BONE_EDGES.
	
	Units: Input poses in millimeters (mm); output percentages (%).
	NMPJPE (Normalized MPJPE): Pose error normalized by per-joint displacement range.
	RTE (Root Trajectory Error): Rigid-aligned root trajectory error (hip-only tracking).
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
		p = int(parent[j])
		
		# Special case: root joint (hip) has parent=-1, no bone-length normalization.
		# Only store raw error, will skip BL aggregation for hip.
		if j == 0:  # hip (root)
			rel_err_bone[:, j] = np.full(gt_pose_mm.shape[0], np.nan, dtype=np.float64)
			continue
		
		if p < 0 or p >= num_joints:
			continue

		bone_len_t = np.linalg.norm(gt_pose_mm[:, j, :] - gt_pose_mm[:, p, :], axis=1)
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

	# For global BL-MPJPE, exclude hip (j=0) which has no bone-length reference.
	# Only compute BL-MPJPE for joints 1-16 which all have parent bones.
	v_global = np.isfinite(rel_err_bone)
	if np.any(v_global):
		global_percent = float(np.mean(rel_err_bone[v_global]) * 100.0)
	else:
		global_percent = float("nan")

	v_global_disp = np.isfinite(rel_err_disp)
	if np.any(v_global_disp):
		global_disp_percent = float(np.mean(rel_err_disp[v_global_disp]) * 100.0)
	else:
		global_disp_percent = float("nan")

	return {
		"BL_MPJPE_percent": global_percent,
		"BL_per_joint_percent": per_joint_percent.tolist(),
		"BL_bone_length_per_joint_mm": bone_len_mean.tolist(),
		"BL_valid_joint_count": int(np.sum(np.isfinite(per_joint_percent))),
		"DispRange_MPJPE_percent": global_disp_percent,
		"DispRange_per_joint_percent": per_joint_disp_percent.tolist(),
		"DispRange_per_joint_mm": disp_range.tolist(),
		"DispRange_valid_joint_count": int(np.sum(np.isfinite(per_joint_disp_percent))),
	}


def compute_bone_length_normalized_mpjve_percent(gt_pose_mm, pred_pose_mm, fps, parent_indices=None, eps=1e-12):
	"""Compute velocity error (MPJVE) normalized by bone length - BL-MPJVE.
	
	For each joint j with valid parent p (excluding root hip):
	1) Velocity error: vel_err_j(t) = ||vel_pred_j(t) - vel_gt_j(t)||_2 where vel = dp/dt
	2) Per-joint mean bone length: Lbar_j = mean_t ||GT_j(t) - GT_p(t)||_2
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
		Dict with:
			- BL_MPJVE_percent: Global bone-length normalized velocity error (%)
			- BL_per_joint_percent: Per-joint BL-MPJVE (%)
			- BL_bone_length_per_joint_mm: Mean bone length per joint (mm)
			- BL_valid_joint_count: Count of joints with valid BL-MPJVE
			- MPJVE_mmps: Global raw MPJVE (mm/s)
			- MPJVE_per_joint_mmps: Per-joint raw MPJVE (mm/s)
	"""
	if gt_pose_mm.shape != pred_pose_mm.shape:
		raise ValueError(f"pose shape mismatch: gt={gt_pose_mm.shape}, pred={pred_pose_mm.shape}")
	if gt_pose_mm.ndim != 3 or gt_pose_mm.shape[2] != 3:
		raise ValueError(f"unexpected pose shape: {gt_pose_mm.shape}")
	if fps <= 0:
		raise ValueError(f"fps must be > 0, got {fps}")
	if gt_pose_mm.shape[0] < 2:
		return {
			"BL_MPJVE_percent": float("nan"),
			"BL_per_joint_percent": [float("nan")] * gt_pose_mm.shape[1],
			"BL_bone_length_per_joint_mm": [float("nan")] * gt_pose_mm.shape[1],
			"BL_valid_joint_count": 0,
			"MPJVE_mmps": float("nan"),
			"MPJVE_per_joint_mmps": [float("nan")] * gt_pose_mm.shape[1],
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
	
	# Bone lengths over full time series: (T, K)
	bone_len_per_frame = np.zeros((gt_pose_mm.shape[0], num_joints), dtype=np.float64)
	for j in range(num_joints):
		p = int(parent[j])
		if j == 0 or p < 0 or p >= num_joints:
			bone_len_per_frame[:, j] = np.nan
		else:
			bone_len_per_frame[:, j] = np.linalg.norm(
				gt_pose_mm[:, j, :] - gt_pose_mm[:, p, :], axis=1
			)
	
	# Mean bone length per joint
	bone_len_mean = np.full(num_joints, np.nan, dtype=np.float64)
	for j in range(num_joints):
		valid_bl = bone_len_per_frame[:, j] > float(eps)
		if np.any(valid_bl):
			bone_len_mean[j] = float(np.mean(bone_len_per_frame[valid_bl, j]))
	
	# Bone-length normalized velocity error: (T-1, K)
	rel_vel_err_bone = np.full_like(vel_err, np.nan, dtype=np.float64)
	for j in range(num_joints):
		if np.isfinite(bone_len_mean[j]) and bone_len_mean[j] > float(eps):
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
	else:
		global_percent = float("nan")
	
	raw_global = np.isfinite(vel_err)
	if np.any(raw_global):
		raw_global_mmps = float(np.mean(vel_err[raw_global]))
	else:
		raw_global_mmps = float("nan")
	
	return {
		"BL_MPJVE_percent": global_percent,
		"BL_per_joint_percent": per_joint_percent.tolist(),
		"BL_bone_length_per_joint_mm": bone_len_mean.tolist(),
		"BL_valid_joint_count": int(np.sum(np.isfinite(per_joint_percent))),
		"MPJVE_mmps": raw_global_mmps,
		"MPJVE_per_joint_mmps": per_joint_raw_mmps.tolist(),
	}


def compute_pose_all_channel_metrics(gt_pose_mm, pred_pose_mm, fps, time_sec=None):
	"""Compute all-channel (17x3) metrics from pose samples at the same timestamps."""
	if gt_pose_mm.shape != pred_pose_mm.shape:
		raise ValueError(f"pose shape mismatch: gt={gt_pose_mm.shape}, pred={pred_pose_mm.shape}")
	if gt_pose_mm.ndim != 3 or gt_pose_mm.shape[2] != 3:
		raise ValueError(f"unexpected pose shape: {gt_pose_mm.shape}")
	if fps <= 0:
		raise ValueError(f"fps must be > 0, got {fps}")

	diff = pred_pose_mm - gt_pose_mm

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
	align_metrics = compute_pose_alignment_metrics(gt_pose_mm, pred_pose_mm, fps, time_sec=time_sec)
	bone_norm = compute_bone_length_normalized_mpjpe_percent(gt_pose_mm, pred_pose_mm)

	ret = {
		"VelRMSE_mmps": vel_rmse,
		"AccRMSE_mmps2": acc_rmse,
		"MPJPE_upsampled_mm": mpjpe,
		"channel_count": int(gt_pose_mm.shape[1] * gt_pose_mm.shape[2]),
	}
	ret.update(align_metrics)
	ret.update(bone_norm)
	return ret


def compute_pose_selected_channel_metrics(
	gt_pose_mm,
	pred_pose_mm,
	keypoint_idx,
	axis_idx,
	fps,
	time_sec=None,
	eps=1e-12,
):
	"""Compute batch-aligned metrics for one selected channel only.

	Returned metric types match batch output categories:
	- V/p95/Max MPJPE
	- V/p95/Max MPJVE
	- V/p95/Max MPJPE_BL
	"""
	if gt_pose_mm.shape != pred_pose_mm.shape:
		raise ValueError(f"pose shape mismatch: gt={gt_pose_mm.shape}, pred={pred_pose_mm.shape}")
	if gt_pose_mm.ndim != 3 or gt_pose_mm.shape[2] != 3:
		raise ValueError(f"unexpected pose shape: {gt_pose_mm.shape}")
	if fps <= 0:
		raise ValueError(f"fps must be > 0, got {fps}")
	if keypoint_idx < 0 or keypoint_idx >= gt_pose_mm.shape[1]:
		raise ValueError(f"keypoint_idx={keypoint_idx} out of range")
	if axis_idx < 0 or axis_idx >= gt_pose_mm.shape[2]:
		raise ValueError(f"axis_idx={axis_idx} out of range")

	gt = gt_pose_mm[:, keypoint_idx, axis_idx]
	pred = pred_pose_mm[:, keypoint_idx, axis_idx]
	diff = pred - gt
	abs_err = np.abs(diff)

	# Bone-length proxy for selected keypoint (same normalization idea as batch BL metric).
	parent = get_default_parent_indices(gt_pose_mm.shape[1])
	edges = get_joint_edge_pairs(parent, keypoint_idx)
	if len(edges) > 0:
		bone_gt_list = []
		for a, b in edges:
			bone_gt_list.append(np.linalg.norm(gt_pose_mm[:, a, :] - gt_pose_mm[:, b, :], axis=1))
		bone_gt_t = np.mean(np.stack(bone_gt_list, axis=0), axis=0)
		bone_valid = bone_gt_t > eps
		bone_mean = float(np.mean(bone_gt_t[bone_valid])) if np.any(bone_valid) else float("nan")
	else:
		bone_mean = float("nan")

	# MPJVE in mm/s: use timestamp dt if provided.
	if gt.shape[0] >= 2:
		if time_sec is not None and len(time_sec) == len(gt):
			dt = np.diff(np.asarray(time_sec, dtype=np.float64))
			valid_dt = dt > eps
			if np.any(valid_dt):
				vel_gt = np.diff(gt)[valid_dt] / dt[valid_dt]
				vel_pred = np.diff(pred)[valid_dt] / dt[valid_dt]
				mpjve_series = np.abs(vel_pred - vel_gt)
				mpjve = float(np.mean(mpjve_series))
			else:
				mpjve_series = np.asarray([], dtype=np.float64)
				mpjve = float("nan")
		else:
			vel_gt = np.diff(gt) * float(fps)
			vel_pred = np.diff(pred) * float(fps)
			mpjve_series = np.abs(vel_pred - vel_gt)
	else:
		mpjve_series = np.asarray([], dtype=np.float64)

	if np.isfinite(bone_mean) and bone_mean > eps:
		bl_series = abs_err / bone_mean * 100.0
	else:
		bl_series = np.asarray([], dtype=np.float64)

	return {
		"V_MPJPE_mm": float(np.mean(abs_err)) if abs_err.size > 0 else float("nan"),
		"P95_MPJPE_mm": float(np.percentile(abs_err, 95.0)) if abs_err.size > 0 else float("nan"),
		"Max_MPJPE_mm": float(np.max(abs_err)) if abs_err.size > 0 else float("nan"),
		"V_MPJVE_mmps": float(np.mean(mpjve_series)) if mpjve_series.size > 0 else float("nan"),
		"P95_MPJVE_mmps": float(np.percentile(mpjve_series, 95.0)) if mpjve_series.size > 0 else float("nan"),
		"Max_MPJVE_mmps": float(np.max(mpjve_series)) if mpjve_series.size > 0 else float("nan"),
		"V_MPJPE_BL_percent": float(np.mean(bl_series)) if bl_series.size > 0 else float("nan"),
		"P95_MPJPE_BL_percent": float(np.percentile(bl_series, 95.0)) if bl_series.size > 0 else float("nan"),
		"Max_MPJPE_BL_percent": float(np.max(bl_series)) if bl_series.size > 0 else float("nan"),
	}


def compute_selected_channel_rte_jitter(
	gt_pose_mm,
	pred_pose_mm,
	keypoint_idx,
	axis_idx,
	fps,
	eps=1e-12,
):
	"""Compute selected-channel RTE/Jitter.

	- RTE: mean absolute error divided by GT trajectory path length on this channel.
	- Jitter: mean absolute jerk on this channel, converted to 10 m/s^3 scale.
	"""
	if gt_pose_mm.shape != pred_pose_mm.shape:
		raise ValueError(f"pose shape mismatch: gt={gt_pose_mm.shape}, pred={pred_pose_mm.shape}")
	if gt_pose_mm.ndim != 3 or gt_pose_mm.shape[2] != 3:
		raise ValueError(f"unexpected pose shape: {gt_pose_mm.shape}")
	if fps <= 0:
		raise ValueError(f"fps must be > 0, got {fps}")
	if keypoint_idx < 0 or keypoint_idx >= gt_pose_mm.shape[1]:
		raise ValueError(f"keypoint_idx={keypoint_idx} out of range")
	if axis_idx < 0 or axis_idx >= gt_pose_mm.shape[2]:
		raise ValueError(f"axis_idx={axis_idx} out of range")

	gt_ch = gt_pose_mm[:, keypoint_idx, axis_idx]
	pred_ch = pred_pose_mm[:, keypoint_idx, axis_idx]

	if gt_ch.shape[0] >= 2:
		disp = float(np.sum(np.abs(np.diff(gt_ch))))
	else:
		disp = 0.0
	if disp > eps:
		rte = np.abs(pred_ch - gt_ch)
		rte_percent = float(np.mean(rte / disp) * 100.0)
	else:
		rte_percent = float("nan")

	if pred_ch.shape[0] >= 4:
		jerk_mmps3 = (
			pred_ch[3:]
			- 3.0 * pred_ch[2:-1]
			+ 3.0 * pred_ch[1:-2]
			- pred_ch[:-3]
		) * (float(fps) ** 3)
		jitter_mps3 = float(np.mean(np.abs(jerk_mmps3)) / 1000.0)
		jitter_10mps3 = float(jitter_mps3 / 10.0)
	else:
		jitter_mps3 = float("nan")
		jitter_10mps3 = float("nan")

	return {
		"RTE_percent": rte_percent,
		"Jitter_10mps3": jitter_10mps3,
		"Jitter_mps3": jitter_mps3,
	}


def compute_root_trajectory_error_and_jitter(gt_pose_mm, pred_pose_mm, fps, root_kpt_idx=0):
	"""Compute RTE and Jitter following the standard definition.

	RTE: rigid-align predicted root trajectory to GT, compute mean 3D error,
	normalize by GT total root displacement (path length), expressed as %.
	Jitter: mean jerk norm (3rd derivative) over all joints / 10 in m/s^3.
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

	vel_rmse = float(np.sqrt(vel_sq_integral / total_duration))
	acc_rmse = float(np.sqrt(acc_sq_integral / total_duration))

	gt_range = float(np.max(gt_dense) - np.min(gt_dense)) if gt_dense.size > 0 else float("nan")

	return {
		"GT_range_mm": gt_range,
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
	upsample_channel_err_series=None,
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

	# Selected-channel error series.
	if (
		upsample_time_sec is not None
		and upsample_channel_err_series is not None
		and len(upsample_time_sec) > 0
	):
		err_line, = ax_err.plot(
			upsample_time_sec * 1000.0,
			upsample_channel_err_series,
			color="tab:cyan",
			linewidth=1.1,
			alpha=0.95,
			label="Selected-channel MPJPE",
		)

		ax_err.legend([err_line], ["Selected-channel MPJPE"], loc="upper right", fontsize=9)

	ax_err.minorticks_on()
	ax_err.grid(which="major", linestyle="-", linewidth=0.45, alpha=0.45)
	ax_err.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.35)
	ax_err.set_ylabel("Selected-channel Error (mm)")
	ax_err.set_xlabel("Time (ms)")

	metric_text = (
		f"V-MPJPE / p95 / Max: {metrics.get('V_MPJPE_mm', float('nan')):.4f} / "
		f"{metrics.get('P95_MPJPE_mm', float('nan')):.4f} / "
		f"{metrics.get('Max_MPJPE_mm', float('nan')):.4f} mm\n"
		f"V-MPJVE / p95 / Max: {metrics.get('V_MPJVE_mmps', float('nan')):.4f} / "
		f"{metrics.get('P95_MPJVE_mmps', float('nan')):.4f} / "
		f"{metrics.get('Max_MPJVE_mmps', float('nan')):.4f} mm/s\n"
		f"V-MPJPE_BL / p95 / Max: {metrics.get('V_MPJPE_BL_percent', float('nan')):.4f} / "
		f"{metrics.get('P95_MPJPE_BL_percent', float('nan')):.4f} / "
		f"{metrics.get('Max_MPJPE_BL_percent', float('nan')):.4f} %\n"
		f"RTE: {metrics.get('RTE_percent', float('nan')):.4f} %\n"
		f"Jitter: {metrics.get('Jitter_10mps3', float('nan')):.4f} (10 m/s^3)\n"
		f"Selected channel @ {metrics.get('upsample_fps', float('nan')):.1f} Hz\n"
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
	gt_file = resolve_runtime_path(gt_file)
	pred_file = resolve_runtime_path(pred_file)
	output_path = resolve_runtime_path(output_path)
	if pose_file:
		pose_file = resolve_runtime_path(pose_file)
	if linear_output_path is not None:
		linear_output_path = resolve_runtime_path(linear_output_path)

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
	upsample_channel_err_series = None
	if upsample_fps is None or upsample_fps <= 0:
		raise ValueError(f"upsample_fps must be > 0 for selected-channel metrics, got {upsample_fps}")

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

			upsample_channel_err_series = np.abs(pred_valid[:, kpt, dim] - gt_valid[:, kpt, dim])
			upsample_time_sec = valid_times

			metrics.update(
				compute_pose_selected_channel_metrics(
					gt_pose_mm=gt_valid,
					pred_pose_mm=pred_valid,
					keypoint_idx=kpt,
					axis_idx=dim,
					fps=float(upsample_fps),
					time_sec=valid_times,
				)
			)
			metrics.update(
				compute_root_trajectory_error_and_jitter(
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
		upsample_channel_err_series=upsample_channel_err_series,
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
	# 3) compare on selected spline channel (kpt, axis)
	ctrl_ts, ctrl_pose = downsample_pose_linear(pose_time_sec, pose_data, linear_downsample_fps)
	lin_pred_all, lin_valid = eval_pose_linear_from_controls(ctrl_ts, ctrl_pose, upsample_time_all)
	gt_all, gt_valid = resample_pose_at_times(pose_time_sec, pose_data, upsample_time_all)
	valid_lin = lin_valid & gt_valid
	if not np.any(valid_lin):
		raise ValueError("No valid overlapping samples for linear interpolation baseline")

	ts_lin = upsample_time_all[valid_lin]
	gt_lin = gt_all[valid_lin]
	pred_lin = lin_pred_all[valid_lin]

	lin_scatter_metrics = compute_pose_selected_channel_metrics(
		gt_pose_mm=gt_lin,
		pred_pose_mm=pred_lin,
		keypoint_idx=kpt,
		axis_idx=dim,
		fps=float(upsample_fps),
		time_sec=ts_lin,
	)
	lin_scatter_metrics.update(
		compute_root_trajectory_error_and_jitter(
			gt_pose_mm=gt_lin,
			pred_pose_mm=pred_lin,
			fps=float(upsample_fps),
			root_kpt_idx=0,
		)
	)
	lin_scatter_metrics["upsample_fps"] = float(upsample_fps)

	# Curve metrics for linear baseline are computed against gt_spline (not gt_pose points).
	# Clip to the same overlap window as pred so both plots cover an identical time range.
	lin_curve_t, lin_curve_coeff = build_linear_channel_coeffs(ctrl_ts, ctrl_pose[:, kpt, dim])
	lin_intervals_full, _ = build_overlap_intervals(gt_t, lin_curve_t)
	if len(lin_intervals_full) == 0:
		raise ValueError("No overlap between gt_spline and linear baseline timeline")
	lin_intervals = [
		(gi, pi, max(float(l), overlap_start), min(float(r), overlap_end))
		for gi, pi, l, r in lin_intervals_full
		if min(float(r), overlap_end) > max(float(l), overlap_start) + 1e-12
	]
	lin_duration = sum(float(r) - float(l) for _, _, l, r in lin_intervals)
	if len(lin_intervals) == 0 or lin_duration <= 0:
		raise ValueError("No overlap between gt_spline and linear baseline within pred time window")
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
		"V_MPJPE_mm": lin_scatter_metrics.get("V_MPJPE_mm", float("nan")),
		"P95_MPJPE_mm": lin_scatter_metrics.get("P95_MPJPE_mm", float("nan")),
		"Max_MPJPE_mm": lin_scatter_metrics.get("Max_MPJPE_mm", float("nan")),
		"V_MPJVE_mmps": lin_scatter_metrics.get("V_MPJVE_mmps", float("nan")),
		"P95_MPJVE_mmps": lin_scatter_metrics.get("P95_MPJVE_mmps", float("nan")),
		"Max_MPJVE_mmps": lin_scatter_metrics.get("Max_MPJVE_mmps", float("nan")),
		"V_MPJPE_BL_percent": lin_scatter_metrics.get("V_MPJPE_BL_percent", float("nan")),
		"P95_MPJPE_BL_percent": lin_scatter_metrics.get("P95_MPJPE_BL_percent", float("nan")),
		"Max_MPJPE_BL_percent": lin_scatter_metrics.get("Max_MPJPE_BL_percent", float("nan")),
		"RTE_percent": lin_scatter_metrics["RTE_percent"],
		"Jitter_10mps3": lin_scatter_metrics["Jitter_10mps3"],
		"Jitter_mps3": lin_scatter_metrics["Jitter_mps3"],
		"upsample_fps": float(upsample_fps),
		"duration_sec": lin_curve_metrics["duration_sec"],
	}

	t_dense_lin = lin_curve_metrics["t_dense"]
	gt_dense_lin = lin_curve_metrics["gt_dense"]
	pred_dense_lin = lin_curve_metrics["pred_dense"]
	gt_vel_lin = lin_curve_metrics["gt_vel_dense"]
	pred_vel_lin = lin_curve_metrics["pred_vel_dense"]
	lin_channel_err = np.abs(pred_lin[:, kpt, dim] - gt_lin[:, kpt, dim])

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
		upsample_channel_err_series=lin_channel_err,
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
		f"V/p95/Max MPJPE={metrics.get('V_MPJPE_mm', float('nan')):.6f}/"
		f"{metrics.get('P95_MPJPE_mm', float('nan')):.6f}/"
		f"{metrics.get('Max_MPJPE_mm', float('nan')):.6f} mm, "
		f"V/p95/Max MPJVE={metrics.get('V_MPJVE_mmps', float('nan')):.6f}/"
		f"{metrics.get('P95_MPJVE_mmps', float('nan')):.6f}/"
		f"{metrics.get('Max_MPJVE_mmps', float('nan')):.6f} mm/s, "
		f"V/p95/Max MPJPE_BL={metrics.get('V_MPJPE_BL_percent', float('nan')):.6f}/"
		f"{metrics.get('P95_MPJPE_BL_percent', float('nan')):.6f}/"
		f"{metrics.get('Max_MPJPE_BL_percent', float('nan')):.6f} %, "
		f"RTE={metrics.get('RTE_percent', float('nan')):.6f} %, "
		f"Jitter={metrics.get('Jitter_10mps3', float('nan')):.6f} (10 m/s^3), "
		f"@{metrics.get('upsample_fps', float('nan')):.1f}Hz"
	)
	print(
		"LinearMetrics: "
		f"V/p95/Max MPJPE={lin_metrics.get('V_MPJPE_mm', float('nan')):.6f}/"
		f"{lin_metrics.get('P95_MPJPE_mm', float('nan')):.6f}/"
		f"{lin_metrics.get('Max_MPJPE_mm', float('nan')):.6f} mm, "
		f"V/p95/Max MPJVE={lin_metrics.get('V_MPJVE_mmps', float('nan')):.6f}/"
		f"{lin_metrics.get('P95_MPJVE_mmps', float('nan')):.6f}/"
		f"{lin_metrics.get('Max_MPJVE_mmps', float('nan')):.6f} mm/s, "
		f"V/p95/Max MPJPE_BL={lin_metrics.get('V_MPJPE_BL_percent', float('nan')):.6f}/"
		f"{lin_metrics.get('P95_MPJPE_BL_percent', float('nan')):.6f}/"
		f"{lin_metrics.get('Max_MPJPE_BL_percent', float('nan')):.6f} %, "
		f"RTE={lin_metrics.get('RTE_percent', float('nan')):.6f} %, "
		f"Jitter={lin_metrics.get('Jitter_10mps3', float('nan')):.6f} (10 m/s^3), "
		f"@{lin_metrics.get('upsample_fps', float('nan')):.1f}Hz"
	)
	print("LinearMetricsNote: all metrics are computed on the selected channel using gt_pose sample points.")


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
		help="Dense samples per overlap interval for curve-domain metrics approximation.",
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
		help="Uniform resampling FPS for spline MPJPE and pose-derived metrics.",
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
	base_dir = PROJECT_ROOT
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
		gt_file = "/Users/twz/demo_sys_user/h36m_pose_cam_1/test/S2_cam_1_120fps_notaknot_splines/Running_37_cam_1_h36m_notaknot_spline.npz"
		pred_file = "res/test/decoded_Running_37_cam_1_h36m_30fps_baseline_realtime_spline.npz"
		gt_pose_file = "/Users/twz/demo_sys_user/h36m_pose_cam_1/test/S2_cam_1_120fps/Running_37_cam_1_h36m.npy"
		gt_pose_fps = 120.0 # 120.0. or 60.0 depending on the source of the pose file and its timestamp alignment with the splines.
		spline_id = 0               # valid range: 0..50 for 17x3
		samples_per_interval = 40   # dense sampling for MAE/P95/Max approximation
		upsample_fps = 120.0        # uniform spline resampling FPS for MPJPE/abs-error points
		output_path = "res/splines_metrics/splines_compare_plot.png"
		linear_downsample_fps = 30.0
		linear_output_path = "res/splines_metrics/downsample_linear.png"
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
