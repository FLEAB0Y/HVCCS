#!/usr/bin/env python3
"""
Visualize pose_codec_metrics_batch results.

Reads codec_metrics_summary.json from res/pose_metrics_batch_test subdirectories,
filters baseline_* folders, and generates dual-axis plots for each stat
(mean/median/p95/max):

1) MPJPE + RMSE
2) KD + Compression Ratio
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import make_interp_spline


PLOT_BOX_ASPECT = 1.0 / 1.1
MAIN_LINE_WIDTH = 2.2
DEFAULT_FONT_SIZE = 23


def parse_folder_name(folder_name):
	"""Extract predictor and quantization level from folder name like 'baseline_q8'."""
	parts = folder_name.rsplit("_", 1)
	if len(parts) != 2:
		return None, None
	return parts[0], parts[1]


def get_q_level_order():
	"""Ordered quantization levels for x-axis (uniform spacing)."""
	return ["q4", "q6", "q8", "q10", "q12", "q14", "q16", "q64"]


def get_q_label(q_level):
	"""Display label for quantization level."""
	if q_level == "q64":
		return "Original\n(64bits)"
	if q_level.startswith("q"):
		return f"{q_level[1:]}bits"
	return q_level


def load_baseline_data(base_dir):
	"""Load per_file_stats from baseline_* folders only."""
	base_path = Path(base_dir)
	data = {}

	if not base_path.exists():
		print(f"[ERROR] Directory not found: {base_dir}")
		return data

	for subfolder in sorted(base_path.iterdir()):
		if not subfolder.is_dir():
			continue

		predictor, q_level = parse_folder_name(subfolder.name)
		if predictor != "baseline":
			continue

		summary_path = subfolder / "codec_metrics_summary.json"
		if not summary_path.exists():
			print(f"[SKIP] Missing codec_metrics_summary.json in {subfolder.name}")
			continue

		try:
			with open(summary_path, "r") as f:
				summary = json.load(f)
			per_file_stats = summary.get("per_file_stats", {})
			data[q_level] = per_file_stats
			print(f"[OK] Loaded {subfolder.name}")
		except Exception as exc:
			print(f"[ERROR] Failed to load {summary_path}: {exc}")

	return data


def _set_single_y_axis_from_series(ax, series_list):
	"""Set one shared left y-axis with 0 as the baseline for all provided series."""
	valid_chunks = [arr[np.isfinite(arr)] for arr in series_list]
	valid_chunks = [arr for arr in valid_chunks if arr.size > 0]
	if not valid_chunks:
		return

	combined = np.concatenate(valid_chunks)
	y_max = float(np.max(combined))
	if y_max <= 0.0:
		y_max = 1.0
	ax.set_ylim(0.0, y_max * 1.05)


def _plot_smooth_series(ax, x_vals, y_vals, color, marker, label, line_width=3.2):
	"""Plot smooth curve and return line handle plus curve evaluator."""
	valid = np.isfinite(y_vals)
	valid_x = x_vals[valid]
	valid_y = y_vals[valid]
	if valid_y.size == 0:
		return None, np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), None

	order = np.argsort(valid_x)
	valid_x = valid_x[order]
	valid_y = valid_y[order]

	unique_x, inv = np.unique(valid_x, return_inverse=True)
	if unique_x.size != valid_x.size:
		agg_y = np.empty(unique_x.size, dtype=np.float64)
		for i in range(unique_x.size):
			agg_y[i] = np.mean(valid_y[inv == i])
		valid_x = unique_x
		valid_y = agg_y

	if valid_y.size >= 3:
		spline_degree = min(3, valid_y.size - 1)
		spline = make_interp_spline(valid_x, valid_y, k=spline_degree)
		x_smooth = np.linspace(valid_x[0], valid_x[-1], 300)
		y_smooth = spline(x_smooth)

		eval_fn = lambda x_query: spline(x_query)

		line = ax.plot(
			x_smooth,
			y_smooth,
			color=color,
			linewidth=line_width,
			label=label,
		)[0]
	else:
		eval_fn = lambda x_query: np.interp(x_query, valid_x, valid_y, left=np.nan, right=np.nan)

		line = ax.plot(
			valid_x,
			valid_y,
			color=color,
			linewidth=line_width,
			label=label,
		)[0]
	return line, valid_x, valid_y, eval_fn


def _set_uniform_compression_axis(ax, font_size, x_min=0, x_max=10, tick_step=2):
	"""Set x-axis to uniform compression ticks with reference lines."""
	ticks = np.arange(float(x_min), float(x_max) + 1e-9, float(tick_step), dtype=np.float64)
	ax.set_xlim(float(x_min), float(x_max))
	ax.set_xticks(ticks)
	ax.set_xticklabels([f"{int(t)}" for t in ticks], fontsize=font_size)
	for t in ticks:
		ax.axvline(float(t), color="gray", linestyle=":", linewidth=1.2, alpha=0.4, zorder=0)


def _apply_fixed_axes_layout(fig, ax_main, ax_secondary=None):
	"""Use the same plot box for all figures, independent of label count."""
	fig.subplots_adjust(left=0.18, right=0.86, bottom=0.15, top=0.95)
	ax_main.set_box_aspect(PLOT_BOX_ASPECT)
	if ax_secondary is not None:
		ax_secondary.set_position(ax_main.get_position())


def _style_axis_frame_and_ticks(ax, tick_label_size, spine_width=2.0, tick_width=1.8, tick_length=7):
	"""Apply thick black frame and ticks for consistent visual style."""
	for spine in ax.spines.values():
		spine.set_color("black")
		spine.set_linewidth(spine_width)
	ax.tick_params(axis="both", which="both", color="black", width=tick_width, length=tick_length, labelsize=tick_label_size)


def _hide_zero_y_tick_label(ax):
	"""Hide y=0 label to avoid duplicated '0' at the axis origin."""
	ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "" if np.isclose(y, 0.0) else f"{y:g}"))


def plot_stat_mpjpe_rmse(stat_name, baseline_data, output_dir, font_size=12, fig_size=(8, 8)):
	"""Plot MPJPE and RMSE in one figure using a single left y-axis."""
	q_order = get_q_level_order()

	mpjpe_vals = []
	rmse_vals = []
	cr_vals = []
	for q in q_order:
		stats = baseline_data.get(q, {})
		mpjpe_vals.append(stats.get("mpjpe_mm", {}).get(stat_name, np.nan))
		rmse_vals.append(stats.get("rmse_mm", {}).get(stat_name, np.nan))
		cr_vals.append(stats.get("compression_ratio", {}).get(stat_name, np.nan))

	mpjpe_arr = np.asarray(mpjpe_vals, dtype=np.float64)
	rmse_arr = np.asarray(rmse_vals, dtype=np.float64)
	x_arr = np.asarray(cr_vals, dtype=np.float64)

	fig, ax1 = plt.subplots(figsize=fig_size)

	line1, _, _, _ = _plot_smooth_series(ax1, x_arr, mpjpe_arr, "tab:blue", "o", "MPJPE", line_width=MAIN_LINE_WIDTH)
	line2, _, _, _ = _plot_smooth_series(ax1, x_arr, rmse_arr, "tab:orange", "s", "RMSE", line_width=MAIN_LINE_WIDTH)

	_set_uniform_compression_axis(ax1, font_size, tick_step=2)
	ax1.set_xlabel("Compression Ratio (x)", fontsize=font_size + 2)
	ax1.set_ylabel("MPJPE & RMSE (mm)", color="black", fontsize=font_size + 2)
	ax1.tick_params(axis="y", labelcolor="black", labelsize=font_size)
	ax1.tick_params(axis="x", labelsize=font_size)
	ax1.grid(True, alpha=0.3, linewidth=1.0)

	_set_single_y_axis_from_series(ax1, [mpjpe_arr, rmse_arr])
	_hide_zero_y_tick_label(ax1)
	_style_axis_frame_and_ticks(ax1, tick_label_size=font_size)
	_apply_fixed_axes_layout(fig, ax1)

	legend_entries = []
	if line1 is not None:
		legend_entries.append((line1, "MPJPE"))
	if line2 is not None:
		legend_entries.append((line2, "RMSE"))
	if legend_entries:
		ax1.legend(
			[line for line, _ in legend_entries],
			[label for _, label in legend_entries],
			loc="best",
			fontsize=font_size,
		)

	output_path = Path(output_dir) / f"{stat_name}_mpjpe_rmse.png"
	plt.savefig(output_path, dpi=150)
	plt.close()
	print(f"[PLOT] Saved: {output_path}")


def plot_stat_kd_cr(stat_name, baseline_data, output_dir, font_size=12, fig_size=(8, 8)):
	"""Plot KD curve with compression ratio as x-axis.

	KD curve is plotted at 10x scale visually, while KD axis labels remain
	in original values. Adds Zhen Lu's comparison points and interpolated
	Ours KD values at the same compression ratios.
	"""
	q_order = get_q_level_order()

	kd_vals = []
	cr_vals = []
	for q in q_order:
		stats = baseline_data.get(q, {})
		kd_vals.append(stats.get("kd_mean_pct", {}).get(stat_name, np.nan))
		cr_vals.append(stats.get("compression_ratio", {}).get(stat_name, np.nan))

	kd_arr = np.asarray(kd_vals, dtype=np.float64)
	cr_arr = np.asarray(cr_vals, dtype=np.float64)
	x_arr = cr_arr.copy()

	kd_scale = 10.0
	kd_scaled_arr = kd_arr * kd_scale

	fig, ax1 = plt.subplots(figsize=fig_size)

	line1, ours_x, ours_kd_scaled, ours_curve_eval = _plot_smooth_series(
		ax1,
		x_arr,
		kd_scaled_arr,
		"tab:green",
		"o",
		"Ours",
		line_width=MAIN_LINE_WIDTH,
	)

	_set_uniform_compression_axis(ax1, font_size, tick_step=2)
	ax1.set_xlabel("Compression Ratio (x)", fontsize=font_size + 2)
	ax1.set_ylabel("KD (%)", color="black", fontsize=font_size + 2)
	ax1.tick_params(axis="y", labelcolor="black", labelsize=font_size)
	ax1.tick_params(axis="x", labelsize=font_size)
	ax1.grid(True, alpha=0.3, linewidth=1.0)

	valid_kd = kd_scaled_arr[np.isfinite(kd_scaled_arr)]
	if valid_kd.size > 0:
		y_max = float(np.max(valid_kd))
		if y_max <= 0.0:
			y_max = 1.0
		ax1.set_ylim((0.0, y_max * 1.05))

	# Keep KD tick labels as original values while plotting at 10x scale.
	ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y / kd_scale:g}"))
	ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "" if np.isclose(y, 0.0) else f"{(y / kd_scale):g}"))
	_style_axis_frame_and_ticks(ax1, tick_label_size=font_size)

	# Add Zhen Lu's point and interpolate Ours at the same compression ratio.
	zhen_x = np.asarray([7.1], dtype=np.float64)
	zhen_kd_pct = np.asarray([0.45], dtype=np.float64)
	zhen_kd_scaled = zhen_kd_pct * kd_scale
	zhen_handle = ax1.plot(
		zhen_x,
		zhen_kd_scaled,
		linestyle="none",
		marker="D",
		markersize=8,
		color="black",
		label="Zhen Lu's",
	)[0]

	# Label Zhen Lu's KD value on the left of the point.
	for xq, yq_scaled in zip(zhen_x, zhen_kd_scaled):
		ax1.annotate(
			f"{(yq_scaled / kd_scale):.2f}%",
			xy=(xq, yq_scaled),
			xytext=(-8, 0),
			textcoords="offset points",
			ha="right",
			va="center",
			fontsize=max(10, font_size - 1),
			color="black",
		)

	if ours_curve_eval is not None and ours_x is not None and ours_x.size >= 2:
		ours_at_zhen_scaled = np.asarray(ours_curve_eval(zhen_x), dtype=np.float64)
		for xq, y_zhen_scaled, y_ours_scaled in zip(zhen_x, zhen_kd_scaled, ours_at_zhen_scaled):
			if not np.isfinite(y_ours_scaled):
				continue

			# Reference line for alignment at the same compression ratio.
			ax1.axvline(xq, color="gray", linestyle="--", linewidth=1.6, alpha=0.65)
			ax1.plot([xq, xq], [y_zhen_scaled, y_ours_scaled], color="gray", linestyle=":", linewidth=1.6, alpha=0.9)

			y_ours_pct = y_ours_scaled / kd_scale
			ax1.plot(
				xq,
				y_ours_scaled,
				linestyle="none",
				marker="o",
				markersize=8,
				color="tab:green",
			)
			ax1.annotate(
				f"{y_ours_pct:.2f}%",
				xy=(xq, y_ours_scaled),
				xytext=(-8, 0),
				textcoords="offset points",
				ha="right",
				va="center",
				fontsize=max(10, font_size - 1),
				color="tab:green",
			)

	_apply_fixed_axes_layout(fig, ax1)

	legend_entries = []
	if line1 is not None:
		legend_entries.append((line1, "Ours"))
	legend_entries.append((zhen_handle, "Zhen Lu's"))
	if legend_entries:
		ax1.legend(
			[line for line, _ in legend_entries],
			[label for _, label in legend_entries],
			loc="best",
			fontsize=font_size,
		)

	output_path = Path(output_dir) / f"{stat_name}_kd_cr.png"
	plt.savefig(output_path, dpi=150)
	plt.close()
	print(f"[PLOT] Saved: {output_path}")


def main():
	base_dir = "res/pose_metrics_batch_test"
	output_dir = "res/pose_metrics_batch_plots"
	font_size = DEFAULT_FONT_SIZE
	fig_size = (8, 8)

	print("Loading baseline codec metrics...")
	baseline_data = load_baseline_data(base_dir)

	if not baseline_data:
		print("[ERROR] No baseline data found. Exiting.")
		return

	Path(output_dir).mkdir(parents=True, exist_ok=True)

	for stat_name in ["mean", "median", "p95", "max"]:
		try:
			plot_stat_mpjpe_rmse(
				stat_name,
				baseline_data,
				output_dir,
				font_size=font_size,
				fig_size=fig_size,
			)
		except Exception as exc:
			print(f"[ERROR] Failed to plot {stat_name} MPJPE/RMSE: {exc}")

		try:
			plot_stat_kd_cr(
				stat_name,
				baseline_data,
				output_dir,
				font_size=font_size,
				fig_size=fig_size,
			)
		except Exception as exc:
			print(f"[ERROR] Failed to plot {stat_name} KD/CR: {exc}")

	print(f"[DONE] All plots saved to {output_dir}")


if __name__ == "__main__":
	main()
