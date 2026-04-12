from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np


# Mapping copied from AthletePose3D/stats_test/get_kinematics.py.
# Key: Human3.6M joint index, Value: COCO joint index (or a list to average).
COCOIndex = Union[int, List[int]]


COCO_TO_H36M_MAP: Dict[int, COCOIndex] = {
	0: [12, 11],
	1: 12,
	2: 14,
	3: 16,
	4: 11,
	5: 13,
	6: 15,
	7: [6, 5, 12, 11],
	8: [6, 5],
	9: [6, 5, 0, 0],
	10: 0,
	11: 5,
	12: 7,
	13: 9,
	14: 6,
	15: 8,
	16: 10,
}

# COCO joints 1..4 (eyes/ears) are not present in H36M. We approximate them
# with head-related joints to keep consistent shape when conversion is needed.
H36M_TO_COCO_MAP: Dict[int, int] = {
	0: 10,
	1: 9,
	2: 9,
	3: 9,
	4: 9,
	5: 11,
	6: 14,
	7: 12,
	8: 15,
	9: 13,
	10: 16,
	11: 4,
	12: 1,
	13: 5,
	14: 1,
	15: 2,
	16: 6,
}


def normalize_pose_format(pose_format: str) -> str:
	norm = pose_format.strip().lower().replace("_", "").replace(".", "")
	if norm in {"h36m", "human36m", "human36mrm", "h36mrm"}:
		return "h36m"
	if norm in {"coco", "cocorm"}:
		return "coco"
	raise ValueError(
		f"Unsupported pose format: {pose_format}. "
		"Use one of COCO, Human3.6M, COCO_RM, Human3.6M_RM."
	)


def to_tjc(pose: np.ndarray, coord_dims: int = 3) -> np.ndarray:
	arr = np.asarray(pose)
	arr = np.squeeze(arr)

	if arr.ndim == 3:
		if arr.shape[-1] >= coord_dims:
			return arr[..., :coord_dims]
		if arr.shape[1] >= coord_dims:
			return np.transpose(arr, (0, 2, 1))[..., :coord_dims]
		raise ValueError(f"Cannot infer pose layout from shape {arr.shape}.")

	if arr.ndim == 2:
		if arr.shape[1] >= coord_dims and arr.shape[1] <= 6:
			return arr[None, :, :coord_dims]
		if arr.shape[0] >= coord_dims and arr.shape[0] <= 6:
			return arr.T[None, :, :coord_dims]
		if arr.shape[1] % coord_dims == 0:
			joints = arr.shape[1] // coord_dims
			return arr.reshape(arr.shape[0], joints, coord_dims)
		if arr.shape[0] % coord_dims == 0:
			joints = arr.shape[0] // coord_dims
			return arr.T.reshape(arr.shape[1], joints, coord_dims)
		raise ValueError(f"Cannot infer pose layout from shape {arr.shape}.")

	if arr.ndim == 1:
		if arr.shape[0] % coord_dims != 0:
			raise ValueError(f"Cannot infer pose layout from shape {arr.shape}.")
		joints = arr.shape[0] // coord_dims
		return arr.reshape(1, joints, coord_dims)

	raise ValueError(f"Unsupported pose ndarray shape: {arr.shape}")


def coco_to_h36m(pose: np.ndarray) -> np.ndarray:
	t, _, c = pose.shape
	out = np.zeros((t, 17, c), dtype=pose.dtype)
	for h36m_idx, coco_idx in COCO_TO_H36M_MAP.items():
		if isinstance(coco_idx, list):
			out[:, h36m_idx] = pose[:, coco_idx].mean(axis=1)
		else:
			out[:, h36m_idx] = pose[:, coco_idx]
	return out


def h36m_to_coco(pose: np.ndarray) -> np.ndarray:
	t, _, c = pose.shape
	out = np.zeros((t, 17, c), dtype=pose.dtype)
	for coco_idx, h36m_idx in H36M_TO_COCO_MAP.items():
		out[:, coco_idx] = pose[:, h36m_idx]
	return out


def convert_pose_format(pose: np.ndarray, src_format: str, dst_format: str) -> np.ndarray:
	src = normalize_pose_format(src_format)
	dst = normalize_pose_format(dst_format)
	if src == dst:
		return pose
	if src == "coco" and dst == "h36m":
		return coco_to_h36m(pose)
	if src == "h36m" and dst == "coco":
		return h36m_to_coco(pose)
	raise ValueError(f"Unsupported conversion: {src_format} -> {dst_format}")


def align_root(pose: np.ndarray, root_index: int) -> np.ndarray:
	if not (0 <= root_index < pose.shape[1]):
		raise ValueError(f"root_index={root_index} out of range for {pose.shape[1]} joints")
	return pose - pose[:, root_index : root_index + 1, :]


def compute_mpjpe(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
	valid = np.isfinite(pred).all(axis=-1) & np.isfinite(gt).all(axis=-1)
	if not np.any(valid):
		raise ValueError("No valid joints to evaluate after NaN/Inf masking.")

	errors = np.linalg.norm(pred - gt, axis=-1)
	masked_errors = np.where(valid, errors, np.nan)

	mpjpe = float(np.nanmean(masked_errors))
	per_joint = np.nanmean(masked_errors, axis=0)
	per_frame = np.nanmean(masked_errors, axis=1)
	return mpjpe, per_joint, per_frame


def main() -> None:
	# ---------------------------------------------------------------------
	# User configuration section: edit values below instead of CLI args.
	# ---------------------------------------------------------------------
	pred_path = "/home/ztw/HVCCS/res/decode_res/pose_recon_codec_h36m.npy"
	gt_path = "/home/data/ztw/AtheletePose3D/data/train_set/S3/Running_0_cam_1_h36m.npy"

	# Supported values: "COCO", "Human3.6M", "COCO_RM", "Human3.6M_RM".
	# For *_h36m.npy files, use "Human3.6M".
	pred_format = "Human3.6M"
	gt_format = "Human3.6M"

	# If None, auto-select:
	# - same src formats -> use that format
	# - different src formats -> use Human3.6M
	# For *_h36m.npy comparison, keep this as "Human3.6M".
	eval_format = "Human3.6M"

	# Keep first 2 or 3 coordinates from input data.
	coord_dims = 3

	# If True, convert to root-relative pose before MPJPE.
	root_align = False
	root_index = 0

	# Scale multiplier for unit conversion (e.g. meter->mm use 1000.0).
	scale = 1.0

	# Optional detailed reports.
	print_per_joint = False
	print_per_frame = False

	if coord_dims not in (2, 3):
		raise ValueError(f"coord_dims must be 2 or 3, got {coord_dims}")

	if not pred_path or not gt_path:
		raise ValueError(
			"Please set pred_path and gt_path in main() before running."
		)

	pred_file = Path(pred_path)
	gt_file = Path(gt_path)
	if not pred_file.exists():
		raise FileNotFoundError(f"Pred file not found: {pred_file}")
	if not gt_file.exists():
		raise FileNotFoundError(f"GT file not found: {gt_file}")

	pred = np.load(str(pred_file))
	gt = np.load(str(gt_file))

	pred = to_tjc(pred, coord_dims=coord_dims)
	gt = to_tjc(gt, coord_dims=coord_dims)

	final_eval_format = eval_format
	if final_eval_format is None:
		if normalize_pose_format(pred_format) == normalize_pose_format(gt_format):
			final_eval_format = pred_format
		else:
			final_eval_format = "Human3.6M"

	pred = convert_pose_format(pred, pred_format, final_eval_format)
	gt = convert_pose_format(gt, gt_format, final_eval_format)

	min_frames = min(pred.shape[0], gt.shape[0])
	min_joints = min(pred.shape[1], gt.shape[1])

	if pred.shape[0] != gt.shape[0]:
		print(
			f"[Info] Frame count mismatch: pred={pred.shape[0]}, gt={gt.shape[0]}. "
			f"Using first {min_frames} frames."
		)
	if pred.shape[1] != gt.shape[1]:
		print(
			f"[Info] Joint count mismatch: pred={pred.shape[1]}, gt={gt.shape[1]}. "
			f"Using first {min_joints} joints."
		)

	pred = pred[:min_frames, :min_joints] * scale
	gt = gt[:min_frames, :min_joints] * scale

	if root_align:
		pred = align_root(pred, root_index)
		gt = align_root(gt, root_index)

	mpjpe, per_joint, per_frame = compute_mpjpe(pred, gt)

	print("=== MPJPE Result ===")
	print(f"pred file: {pred_file}")
	print(f"gt file: {gt_file}")
	print(f"pred shape (after normalize): {pred.shape}")
	print(f"gt shape (after normalize): {gt.shape}")
	print(f"evaluation format: {normalize_pose_format(final_eval_format)}")
	print(f"root align: {root_align} (root index={root_index})")
	print(f"scale: {scale}")
	print(f"MPJPE: {mpjpe:.6f}")

	if print_per_joint:
		print("Per-joint MPJPE:")
		for j, err in enumerate(per_joint):
			print(f"  joint[{j:02d}] = {err:.6f}")

	if print_per_frame:
		print("Per-frame MPJPE:")
		for i, err in enumerate(per_frame):
			print(f"  frame[{i:05d}] = {err:.6f}")


if __name__ == "__main__":
	main()
