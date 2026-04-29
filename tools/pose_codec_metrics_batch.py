import argparse
import csv
import json
import os
import sys

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
		raise ValueError(f"sample fps must be > 0, got {fps}")
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


def compute_mpjpe_metrics(gt_pose_mm, pred_pose_mm):
	"""Compute MPJPE and related metrics.
	
	Args:
		gt_pose_mm: Ground-truth pose (T, K, 3) in mm
		pred_pose_mm: Predicted pose (T, K, 3) in mm
		
	Returns:
		Dict with V-MPJPE, p95-MPJPE, max-MPJPE.
	"""
	if gt_pose_mm.shape != pred_pose_mm.shape:
		raise ValueError(f"pose shape mismatch: gt={gt_pose_mm.shape}, pred={pred_pose_mm.shape}")
	if gt_pose_mm.ndim != 3 or gt_pose_mm.shape[2] != 3:
		raise ValueError(f"unexpected pose shape: {gt_pose_mm.shape}")

	joint_err = np.linalg.norm(pred_pose_mm - gt_pose_mm, axis=2)  # (T, K)
	mpjpe_series = joint_err.mean(axis=1)  # (T,)

	return {
		"mpjpe_mm": float(np.mean(mpjpe_series)),
		"mpjpe_p95_mm": float(np.percentile(mpjpe_series, 95.0)),
		"mpjpe_max_mm": float(np.max(mpjpe_series)),
	}


def compute_rmse_metric(gt_pose_mm, pred_pose_mm):
	"""Compute RMSE on original coordinates (no normalization).

	Args:
		gt_pose_mm: Ground-truth pose (T, K, 3) in mm
		pred_pose_mm: Predicted pose (T, K, 3) in mm

	Returns:
		Dict with rmse_mm.
	"""
	if gt_pose_mm.shape != pred_pose_mm.shape:
		raise ValueError(f"pose shape mismatch: gt={gt_pose_mm.shape}, pred={pred_pose_mm.shape}")
	if gt_pose_mm.ndim != 3 or gt_pose_mm.shape[2] != 3:
		raise ValueError(f"unexpected pose shape: {gt_pose_mm.shape}")

	diff = pred_pose_mm - gt_pose_mm
	rmse_mm = float(np.sqrt(np.mean(np.square(diff))))

	return {
		"rmse_mm": rmse_mm,
	}


def compute_kd_metrics(gt_pose_mm, pred_pose_mm, eps=1e-12):
	"""Compute normalized Keypoint Difference (KD) using per-file min-max normalization.

	Normalization (per file, based on GT):
	  normalized = 2 * (coord - gt_min) / (gt_max - gt_min) - 1
	Both GT and pred are normalized with the same GT-derived min/max.
	L2 error is computed on normalized coordinates and multiplied by 100 to yield %.

	Args:
		gt_pose_mm: Ground-truth pose (T, K, 3)
		pred_pose_mm: Predicted pose (T, K, 3)
		eps: Guard against zero range

	Returns:
		Dict with kd_mean_pct, kd_p95_pct, kd_max_pct (all in %).
	"""
	if gt_pose_mm.shape != pred_pose_mm.shape:
		raise ValueError(f"pose shape mismatch: gt={gt_pose_mm.shape}, pred={pred_pose_mm.shape}")
	if gt_pose_mm.ndim != 3 or gt_pose_mm.shape[2] != 3:
		raise ValueError(f"unexpected pose shape: {gt_pose_mm.shape}")

	# Global min/max from GT for this file
	gt_min = float(np.min(gt_pose_mm))
	gt_max = float(np.max(gt_pose_mm))
	gt_range = gt_max - gt_min
	if gt_range < eps:
		raise ValueError(f"GT pose range too small for normalization: range={gt_range}")

	# Normalize both GT and pred with the same GT min/max
	gt_norm = 2.0 * (gt_pose_mm - gt_min) / gt_range - 1.0
	pred_norm = 2.0 * (pred_pose_mm - gt_min) / gt_range - 1.0

	# L2 error per (frame, joint) on normalized coords, scaled to %
	joint_err_norm = np.linalg.norm(pred_norm - gt_norm, axis=2) * 100.0  # (T, K)
	all_err = joint_err_norm.ravel()

	return {
		"kd_mean_pct": float(np.mean(all_err)),
		"kd_p95_pct": float(np.percentile(all_err, 95.0)),
		"kd_max_pct": float(np.max(all_err)),
	}


def compute_compression_ratio(num_frames, num_joints, num_dims, i_frame_interval, quant_bits):
	"""Compute compression ratio for codec.
	
	GT: unquantized float64 (8 bytes per value)
	Pred: I-frames unquantized (8 bytes), P-frames quantized (quant_bits/8 bytes)
	
	Args:
		num_frames: Total number of frames
		num_joints: Number of joints per frame
		num_dims: Number of dimensions per joint (3 for 3D pose)
		i_frame_interval: Interval between I-frames
		quant_bits: Quantization bits for P-frames
		
	Returns:
		Compression ratio as float.
	"""
	# Ground truth size: all float64
	gt_size_bytes = num_frames * num_joints * num_dims * 8

	# Predicted size: I-frames and P-frames
	# Frame indices: 0, 1, 2, ..., num_frames-1
	# I-frames at: 0, i_frame_interval, 2*i_frame_interval, ...
	i_frame_count = (num_frames - 1) // i_frame_interval + 1
	p_frame_count = num_frames - i_frame_count
	
	# I-frame: float64 (8 bytes per value)
	i_frame_size = i_frame_count * num_joints * num_dims * 8
	
	# P-frame: quantized (quant_bits bits per value)
	quant_bytes_per_value = quant_bits / 8.0
	p_frame_size = p_frame_count * num_joints * num_dims * quant_bytes_per_value

	pred_size_bytes = i_frame_size + p_frame_size

	if pred_size_bytes <= 0:
		return float("nan")

	compression_ratio = float(gt_size_bytes / pred_size_bytes)
	return compression_ratio


def compute_bitrate(num_frames, num_joints, num_dims, i_frame_interval, quant_bits, fps):
	"""Compute bitrate in kbps.
	
	Args:
		num_frames: Total number of frames
		num_joints: Number of joints per frame
		num_dims: Number of dimensions per joint
		i_frame_interval: Interval between I-frames
		quant_bits: Quantization bits for P-frames
		fps: Frames per second
		
	Returns:
		Bitrate in kbps.
	"""
	# Predicted size in bits
	i_frame_count = (num_frames - 1) // i_frame_interval + 1
	p_frame_count = num_frames - i_frame_count
	
	i_frame_bits = i_frame_count * num_joints * num_dims * 64  # 64 bits for float64
	p_frame_bits = p_frame_count * num_joints * num_dims * quant_bits

	total_bits = i_frame_bits + p_frame_bits
	duration_sec = num_frames / float(fps)
	
	if duration_sec <= 0:
		return float("nan")

	bitrate_kbps = float(total_bits / duration_sec / 1000.0)
	return bitrate_kbps


def run_metrics(
	pred_dir,
	gt_pose_dir,
	output_dir,
	pred_suffix,
	gt_pose_suffix,
	gt_pose_fps,
	codec_fps,
	i_frame_interval,
	quant_bits,
	max_files,
):
	pred_dir = resolve_runtime_path(pred_dir)
	gt_pose_dir = resolve_runtime_path(gt_pose_dir)
	output_dir = resolve_runtime_path(output_dir)

	pred_map = build_id_to_path_map(pred_dir, pred_suffix)
	pose_map = build_id_to_path_map(gt_pose_dir, gt_pose_suffix)

	pred_ids = set(pred_map.keys())
	pose_ids = set(pose_map.keys())
	matched_ids = sorted(pred_ids & pose_ids)

	if max_files is not None and max_files > 0:
		matched_ids = matched_ids[:max_files]

	print(f"Prediction spline files: {len(pred_map)}")
	print(f"Ground truth pose files: {len(pose_map)}")
	print(f"Matched files: {len(matched_ids)}")
	print(f"Codec: i_frame_interval={i_frame_interval}, quant_bits={quant_bits}")

	rows = []
	failed = 0

	for idx, sample_id in enumerate(matched_ids, start=1):
		pred_path = pred_map[sample_id]
		pose_path = pose_map[sample_id]

		try:
			pr_t, pr_c = load_spline_npz(pred_path)
			pose_data = load_pose_array(pose_path)

			# Build time grids
			pose_time_sec = np.arange(pose_data.shape[0], dtype=np.float64) / float(gt_pose_fps)
			codec_time_start = float(pr_t[0])
			codec_time_end = float(pr_t[-1])

			# Sample pred_splines at codec_fps
			codec_ts = build_uniform_sample_times(codec_time_start, codec_time_end, codec_fps)
			if codec_ts.size == 0:
				raise ValueError("No codec timestamps generated")

			pred_pose_codec = eval_spline_pose_at_times(pr_t, pr_c, codec_ts)

			# Resample gt_pose to codec timestamps
			gt_pose_codec, gt_valid = resample_pose_at_times(pose_time_sec, pose_data, codec_ts)
			if not np.any(gt_valid):
				raise ValueError("No valid overlap between codec and gt pose")

			codec_ts_valid = codec_ts[gt_valid]
			pred_valid = pred_pose_codec[gt_valid]
			gt_valid_pose = gt_pose_codec[gt_valid]

			# Compute metrics
			mpjpe_metrics = compute_mpjpe_metrics(gt_valid_pose, pred_valid)
			rmse_metrics = compute_rmse_metric(gt_valid_pose, pred_valid)
			kd_metrics = compute_kd_metrics(gt_valid_pose, pred_valid)
			compression_ratio = compute_compression_ratio(
				num_frames=pred_valid.shape[0],
				num_joints=pred_valid.shape[1],
				num_dims=pred_valid.shape[2],
				i_frame_interval=i_frame_interval,
				quant_bits=quant_bits,
			)
			bitrate_kbps = compute_bitrate(
				num_frames=pred_valid.shape[0],
				num_joints=pred_valid.shape[1],
				num_dims=pred_valid.shape[2],
				i_frame_interval=i_frame_interval,
				quant_bits=quant_bits,
				fps=codec_fps,
			)

			row = {
				"sample_id": sample_id,
				"mpjpe_mm": mpjpe_metrics["mpjpe_mm"],
				"mpjpe_p95_mm": mpjpe_metrics["mpjpe_p95_mm"],
				"mpjpe_max_mm": mpjpe_metrics["mpjpe_max_mm"],
				"rmse_mm": rmse_metrics["rmse_mm"],
				"kd_mean_pct": kd_metrics["kd_mean_pct"],
				"kd_p95_pct": kd_metrics["kd_p95_pct"],
				"kd_max_pct": kd_metrics["kd_max_pct"],
				"compression_ratio": compression_ratio,
				"bitrate_kbps": bitrate_kbps,
				"num_frames": pred_valid.shape[0],
			}
			rows.append(row)

			print(
				f"[{idx}/{len(matched_ids)}] {sample_id}: "
				f"MPJPE(V/p95/Max)={mpjpe_metrics['mpjpe_mm']:.2f}/"
				f"{mpjpe_metrics['mpjpe_p95_mm']:.2f}/"
				f"{mpjpe_metrics['mpjpe_max_mm']:.2f} mm, "
				f"RMSE={rmse_metrics['rmse_mm']:.2f} mm, "
				f"KD(mean/p95/max)={kd_metrics['kd_mean_pct']:.2f}/"
				f"{kd_metrics['kd_p95_pct']:.2f}/"
				f"{kd_metrics['kd_max_pct']:.2f}%, "
				f"CR={compression_ratio:.2f}x, "
				f"Bitrate={bitrate_kbps:.2f} kbps"
			)
		except Exception as exc:
			failed += 1
			print(f"[{idx}/{len(matched_ids)}] Failed {sample_id}: {exc}")

	# Write results
	os.makedirs(output_dir, exist_ok=True)
	csv_path = os.path.join(output_dir, "codec_metrics_per_file.csv")
	summary_path = os.path.join(output_dir, "codec_metrics_summary.json")

	# Write CSV
	if len(rows) > 0:
		fieldnames = ["sample_id", "mpjpe_mm", "mpjpe_p95_mm", "mpjpe_max_mm", "rmse_mm", "kd_mean_pct", "kd_p95_pct", "kd_max_pct", "compression_ratio", "bitrate_kbps", "num_frames"]
		with open(csv_path, "w", newline="") as fp:
			writer = csv.DictWriter(fp, fieldnames=fieldnames)
			writer.writeheader()
			writer.writerows(rows)
	else:
		with open(csv_path, "w", newline="") as fp:
			fp.write("sample_id\n")

	# Write summary
	summary = {
		"num_files": len(rows),
		"failed_files": failed,
		"matched_files": len(matched_ids),
		"pred_suffix": pred_suffix,
		"gt_pose_suffix": gt_pose_suffix,
		"gt_pose_fps": float(gt_pose_fps),
		"codec_fps": float(codec_fps),
		"i_frame_interval": int(i_frame_interval),
		"quant_bits": int(quant_bits),
	}

	if len(rows) > 0:
		mpjpe_vals = np.asarray([r["mpjpe_mm"] for r in rows], dtype=np.float64)
		mpjpe_p95_vals = np.asarray([r["mpjpe_p95_mm"] for r in rows], dtype=np.float64)
		mpjpe_max_vals = np.asarray([r["mpjpe_max_mm"] for r in rows], dtype=np.float64)
		rmse_vals = np.asarray([r["rmse_mm"] for r in rows], dtype=np.float64)
		kd_mean_vals = np.asarray([r["kd_mean_pct"] for r in rows], dtype=np.float64)
		kd_p95_vals = np.asarray([r["kd_p95_pct"] for r in rows], dtype=np.float64)
		kd_max_vals = np.asarray([r["kd_max_pct"] for r in rows], dtype=np.float64)
		cr_vals = np.asarray([r["compression_ratio"] for r in rows], dtype=np.float64)
		br_vals = np.asarray([r["bitrate_kbps"] for r in rows], dtype=np.float64)

		summary["per_file_stats"] = {
			"mpjpe_mm": {
				"mean": float(np.mean(mpjpe_vals)),
				"median": float(np.median(mpjpe_vals)),
				"p95": float(np.percentile(mpjpe_vals, 95.0)),
				"max": float(np.max(mpjpe_vals)),
			},
			"mpjpe_p95_mm": {
				"mean": float(np.mean(mpjpe_p95_vals)),
				"median": float(np.median(mpjpe_p95_vals)),
				"p95": float(np.percentile(mpjpe_p95_vals, 95.0)),
				"max": float(np.max(mpjpe_p95_vals)),
			},
			"mpjpe_max_mm": {
				"mean": float(np.mean(mpjpe_max_vals)),
				"median": float(np.median(mpjpe_max_vals)),
				"p95": float(np.percentile(mpjpe_max_vals, 95.0)),
				"max": float(np.max(mpjpe_max_vals)),
			},
			"rmse_mm": {
				"mean": float(np.mean(rmse_vals)),
				"median": float(np.median(rmse_vals)),
				"p95": float(np.percentile(rmse_vals, 95.0)),
				"max": float(np.max(rmse_vals)),
			},
			"kd_mean_pct": {
				"mean": float(np.mean(kd_mean_vals)),
				"median": float(np.median(kd_mean_vals)),
				"p95": float(np.percentile(kd_mean_vals, 95.0)),
				"max": float(np.max(kd_mean_vals)),
			},
			"kd_p95_pct": {
				"mean": float(np.mean(kd_p95_vals)),
				"median": float(np.median(kd_p95_vals)),
				"p95": float(np.percentile(kd_p95_vals, 95.0)),
				"max": float(np.max(kd_p95_vals)),
			},
			"kd_max_pct": {
				"mean": float(np.mean(kd_max_vals)),
				"median": float(np.median(kd_max_vals)),
				"p95": float(np.percentile(kd_max_vals, 95.0)),
				"max": float(np.max(kd_max_vals)),
			},
			"compression_ratio": {
				"mean": float(np.mean(cr_vals)),
				"median": float(np.median(cr_vals)),
				"p95": float(np.percentile(cr_vals, 95.0)),
				"max": float(np.max(cr_vals)),
			},
			"bitrate_kbps": {
				"mean": float(np.mean(br_vals)),
				"median": float(np.median(br_vals)),
				"p95": float(np.percentile(br_vals, 95.0)),
				"max": float(np.max(br_vals)),
			},
		}

	with open(summary_path, "w") as fp:
		json.dump(summary, fp, indent=2)

	print("=" * 60)
	print(f"Saved per-file metrics: {csv_path}")
	print(f"Saved summary: {summary_path}")


def build_argparser():
	parser = argparse.ArgumentParser(
		description=(
			"Evaluate codec performance by comparing pred_splines sampled at codec_fps "
			"against ground truth pose. Computes MPJPE and compression ratio."
		)
	)
	parser.add_argument(
		"--pred-dir",
		type=str,
		default="res/codec_splines",
		help="Prediction spline directory (encoded/decoded splines).",
	)
	parser.add_argument(
		"--gt-pose-dir",
		type=str,
		default="/Users/twz/demo_sys_user/h36m_pose_cam_1_downsample/test/S2_cam_1_30fps",
		help="Ground truth pose directory (npy files).",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="res/codec_metrics",
		help="Output directory for CSV/JSON metrics.",
	)
	parser.add_argument(
		"--pred-suffix",
		type=str,
		default="_codec_spline.npz",
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
		default=30.0,
		help="FPS for ground truth pose timestamps.",
	)
	parser.add_argument(
		"--codec-fps",
		type=float,
		default=30.0,
		help="FPS for codec (sampling rate from pred_splines).",
	)
	parser.add_argument(
		"--i-frame-interval",
		type=int,
		default=30,
		help="Interval between I-frames for compression calculation.",
	)
	parser.add_argument(
		"--quant-bits",
		type=int,
		default=8,
		help="Quantization bits per value for P-frames.",
	)
	parser.add_argument(
		"--max-files",
		type=int,
		default=None,
		help="Optional cap for quick test runs.",
	)
	return parser


if __name__ == "__main__":
	USE_CLI = len(sys.argv) > 1

	if USE_CLI:
		args = build_argparser().parse_args()
		run_metrics(
			pred_dir=args.pred_dir,
			gt_pose_dir=args.gt_pose_dir,
			output_dir=args.output_dir,
			pred_suffix=args.pred_suffix,
			gt_pose_suffix=args.gt_pose_suffix,
			gt_pose_fps=args.gt_pose_fps,
			codec_fps=args.codec_fps,
			i_frame_interval=args.i_frame_interval,
			quant_bits=args.quant_bits,
			max_files=args.max_files,
		)
	else:
		# Direct config mode
		pred_splines_dir = "res/codec_splines"
		gt_pose_dir = "/Users/twz/demo_sys_user/h36m_pose_cam_1_downsample/test/S2_cam_1_30fps"
		output_dir = "res/codec_metrics"
		pred_suffix = "_codec_spline.npz"
		gt_pose_suffix = ".npy"
		gt_pose_fps = 30.0
		codec_fps = 30.0
		i_frame_interval = 30
		quant_bits = 8
		max_files = None

		run_metrics(
			pred_dir=pred_splines_dir,
			gt_pose_dir=gt_pose_dir,
			output_dir=output_dir,
			pred_suffix=pred_suffix,
			gt_pose_suffix=gt_pose_suffix,
			gt_pose_fps=gt_pose_fps,
			codec_fps=codec_fps,
			i_frame_interval=i_frame_interval,
			quant_bits=quant_bits,
			max_files=max_files,
		)
