import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np


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


def eval_local_cubic(local_coeff, seg_start_t, ts):
	dt = ts - seg_start_t
	a, b, c, d = local_coeff
	return ((a * dt + b) * dt + c) * dt + d


def eval_local_cubic_derivative(local_coeff, seg_start_t, ts):
	dt = ts - seg_start_t
	a, b, c, _ = local_coeff
	return (3.0 * a * dt + 2.0 * b) * dt + c


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

		gt_global = local_to_global_coeff(gt_local, gt_time_sec[gt_idx])
		pred_global = local_to_global_coeff(pred_local, pred_time_sec[pred_idx])
		diff_global = pred_global - gt_global

		pos_sq_integral += integrate_square_poly(diff_global, left, right)

		d1 = np.array([3.0 * diff_global[0], 2.0 * diff_global[1], diff_global[2]], dtype=np.float64)
		vel_sq_integral += integrate_square_poly(d1, left, right)

		d2 = np.array([6.0 * diff_global[0], 2.0 * diff_global[1]], dtype=np.float64)
		acc_sq_integral += integrate_square_poly(d2, left, right)

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
):
	if t_dense.size == 0:
		raise ValueError("No dense curve samples to plot")

	fig, ax = plt.subplots(figsize=(12, 6.5))

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
		linestyle="--",
		label="Position Pred",
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
		linestyle="--",
		alpha=0.9,
		label="Velocity Pred",
	)

	ax.set_title(title, fontsize=12)
	ax.set_xlabel("Time (ms)")
	ax.set_ylabel("Position (mm)")
	ax_vel.set_ylabel("Velocity (mm/s)")

	# Denser reference grid: major + minor
	ax.minorticks_on()
	ax.grid(which="major", linestyle="-", linewidth=0.55, alpha=0.50)
	ax.grid(which="minor", linestyle=":", linewidth=0.45, alpha=0.45)

	lines = [pos_gt_line, pos_pred_line, vel_gt_line, vel_pred_line]
	labels = ["Position GT", "Position Pred", "Velocity GT", "Velocity Pred"]
	ax.legend(lines, labels, loc="upper right", fontsize=9)

	metric_text = (
		f"cRMSE: {metrics['cRMSE_mm']:.4f} mm\n"
		f"NRMSE(range): {metrics['NRMSE_range']:.4f}\n"
		f"GT range: {metrics['GT_range_mm']:.4f} mm\n"
		f"MAE: {metrics['MAE_mm']:.4f} mm\n"
		f"P95AE: {metrics['P95AE_mm']:.4f} mm\n"
		f"MaxAE: {metrics['MaxAE_mm']:.4f} mm\n"
		f"VelRMSE: {metrics['VelRMSE_mmps']:.4f} mm/s\n"
		f"AccRMSE: {metrics['AccRMSE_mmps2']:.4f} mm/s^2\n"
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
	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def with_spline_id_in_filename(output_path, spline_id):
	dirname = os.path.dirname(output_path)
	basename = os.path.basename(output_path)
	stem, ext = os.path.splitext(basename)

	# If sid already exists, replace it; otherwise append.
	if re.search(r"_sid\d+$", stem):
		stem = re.sub(r"_sid\d+$", f"_sid{spline_id}", stem)
	else:
		stem = f"{stem}_sid{spline_id}"

	return os.path.join(dirname, f"{stem}{ext}")


def run_compare(gt_file, pred_file, spline_id, samples_per_interval, output_path):
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

	output_path = with_spline_id_in_filename(output_path, spline_id)

	plot_two_splines(
		output_path=output_path,
		title=title,
		metrics=metrics,
		t_dense=metrics["t_dense"],
		gt_dense=metrics["gt_dense"],
		pred_dense=metrics["pred_dense"],
		gt_vel_dense=metrics["gt_vel_dense"],
		pred_vel_dense=metrics["pred_vel_dense"],
	)

	print(f"Total spline dimensions: {total} ({gt_coeffs.shape[0]} * {gt_coeffs.shape[1]})")
	print(f"Selected spline id: {spline_id} -> keypoint={kpt}, axis={axis_name(dim)}")
	print(f"Saved plot: {output_path}")
	print(
		"Metrics: "
		f"cRMSE={metrics['cRMSE_mm']:.6f} mm, "
		f"NRMSE(range)={metrics['NRMSE_range']:.6f}, "
		f"GT_range={metrics['GT_range_mm']:.6f} mm, "
		f"MAE={metrics['MAE_mm']:.6f} mm, "
		f"P95AE={metrics['P95AE_mm']:.6f} mm, "
		f"MaxAE={metrics['MaxAE_mm']:.6f} mm, "
		f"VelRMSE={metrics['VelRMSE_mmps']:.6f} mm/s, "
		f"AccRMSE={metrics['AccRMSE_mmps2']:.6f} mm/s^2"
	)


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
		)
	else:
		gt_file = "/home/data/ztw/AtheletePose3D/h36m_pose_cam_1/train/S1_cam_1_60fps_notaknot_splines/Axel_1_cam_1_h36m_notaknot_spline.npz"
		pred_file = "/home/data/ztw/AtheletePose3D/h36m_pose_cam_1/train/S1_cam_1_60fps_clamped_splines/Axel_1_cam_1_h36m_clamped_spline.npz"
		spline_id = 1               # valid range: 0..50 for 17x3
		samples_per_interval = 40   # dense sampling for MAE/P95/Max approximation
		output_path = "/home/ztw/HVCCS/res/splines_metrics/Alex_1_cam_1_sid1.png"

		run_compare(
			gt_file=gt_file,
			pred_file=pred_file,
			spline_id=spline_id,
			samples_per_interval=samples_per_interval,
			output_path=output_path,
		)
