import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np


def build_camera_fx(image_width: int, hfov_deg: float) -> float:
	"""Compute camera focal length in pixel from horizontal FOV."""
	hfov_rad = np.deg2rad(hfov_deg)
	return (image_width * 0.5) / np.tan(hfov_rad * 0.5)


def rasterize_capsule_mask(
	image_height: int,
	image_width: int,
	rod_length_px: float,
	rod_diameter_px: float,
) -> np.ndarray:
	"""Render a horizontal capsule (cylinder side silhouette) in image center."""
	cy = (image_height - 1) * 0.5
	cx = (image_width - 1) * 0.5

	radius = max(rod_diameter_px * 0.5, 0.5)
	half_body = max(rod_length_px * 0.5 - radius, 0.0)

	y, x = np.ogrid[:image_height, :image_width]
	dx = np.abs(x - cx)
	dy = np.abs(y - cy)

	# Signed distance of a 2D capsule aligned with x-axis.
	qx = dx - half_body
	qx_clip = np.maximum(qx, 0.0)
	dist = np.sqrt(qx_clip * qx_clip + dy * dy)

	return dist <= radius


def estimate_rod_length_px_from_rgb(rgb: np.ndarray, threshold: int = 200) -> int:
	"""Estimate rod horizontal length (px) from an RGB image by white-region extraction."""
	if rgb.ndim != 3 or rgb.shape[2] != 3:
		return 0

	gray = rgb.mean(axis=2)
	white_mask = gray >= float(threshold)
	ys, xs = np.where(white_mask)
	if ys.size == 0 or xs.size == 0:
		return 0
	return int(xs.max() - xs.min() + 1)


def run_experiment(
	output_dir: Path,
	image_width: int,
	image_height: int,
	hfov_deg: float,
	rod_length_m: float,
	rod_diameter_m: float,
	start_distance_m: float,
	end_distance_m: float,
	step_distance_m: float,
	capture_interval_s: float,
	save_images: bool,
) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	if save_images:
		(output_dir / "rgb_frames").mkdir(parents=True, exist_ok=True)

	fx = build_camera_fx(image_width=image_width, hfov_deg=hfov_deg)

	distances = np.arange(start_distance_m, end_distance_m + 1e-9, step_distance_m)
	pixel_counts = []
	projected_lengths = []
	projected_diameters = []
	region_lengths_px = []
	region_widths_px = []
	rgb_estimated_lengths_px = []

	for idx, distance_m in enumerate(distances):
		rod_length_px = fx * rod_length_m / distance_m
		rod_diameter_px = fx * rod_diameter_m / distance_m

		mask = rasterize_capsule_mask(
			image_height=image_height,
			image_width=image_width,
			rod_length_px=rod_length_px,
			rod_diameter_px=rod_diameter_px,
		)

		rgb = np.zeros((image_height, image_width, 3), dtype=np.uint8)
		rgb[mask] = 255
		white_pixels = int(mask.sum())

		# Measure rod bounding box from the rendered mask.
		ys, xs = np.where(mask)
		if ys.size > 0 and xs.size > 0:
			region_length_px = int(xs.max() - xs.min() + 1)
			region_width_px = int(ys.max() - ys.min() + 1)
		else:
			region_length_px = 0
			region_width_px = 0

		pixel_counts.append(white_pixels)
		projected_lengths.append(float(rod_length_px))
		projected_diameters.append(float(rod_diameter_px))
		region_lengths_px.append(region_length_px)
		region_widths_px.append(region_width_px)
		rgb_estimated_lengths_px.append(estimate_rod_length_px_from_rgb(rgb))

		if save_images:
			frame_name = f"frame_{idx:03d}_d_{distance_m:.2f}m.png"
			plt.imsave(output_dir / "rgb_frames" / frame_name, rgb)

	result = np.column_stack(
		[
			distances,
			np.array(projected_lengths),
			np.array(projected_diameters),
			np.array(pixel_counts),
			np.array(region_lengths_px),
			np.array(region_widths_px),
		]
	)
	header = (
		"distance_m,projected_length_px,projected_diameter_px,white_pixel_count,"
		"rod_region_length_px,rod_region_width_px"
	)
	np.savetxt(output_dir / "pixel_occupancy.csv", result, delimiter=",", header=header, comments="")

	measured_lengths = np.array(region_lengths_px, dtype=np.float64)
	theoretical_lengths = np.array(projected_lengths, dtype=np.float64)
	length_errors = measured_lengths - theoretical_lengths
	length_errors_cm = length_errors / fx * distances * 100.0

	fig, ax_left = plt.subplots(figsize=(8, 5))
	ax_left.scatter(
		distances,
		measured_lengths,
		s=18,
		color="tab:orange",
		edgecolors="none",
		label="Spatially quantized rod pixel length in image",
	)
	ax_left.plot(
		distances,
		theoretical_lengths,
		color="red",
		linewidth=1.4,
		label="Theoretical length",
	)
	ax_left.fill_between(
		distances,
		measured_lengths,
		theoretical_lengths,
		color="tab:blue",
		alpha=0.22,
		label="Error region",
	)

	y_min = min(np.min(measured_lengths), np.min(theoretical_lengths))
	y_max = max(np.max(measured_lengths), np.max(theoretical_lengths))
	y_pad = max((y_max - y_min) * 0.08, 1.0)
	ax_left.set_ylim(float(y_min - y_pad), float(y_max + y_pad))

	for x in distances:
		ax_left.axvline(x=x, color="gray", linewidth=0.4, alpha=0.18, zorder=0)

	ax_right = ax_left.twinx()
	ax_right.vlines(
		distances,
		0.0,
		length_errors_cm,
		color="tab:blue",
		linewidth=0.9,
		alpha=0.7,
		label="Spatial quantization error",
	)
	ax_right.scatter(
		distances,
		length_errors_cm,
		s=14,
		color="tab:blue",
		edgecolors="none",
		label="_nolegend_",
	)
	ax_right.axhline(0.0, color="tab:blue", linewidth=0.9, linestyle="--", alpha=0.7)
	ax_right.set_ylim(-1.0, 1.0)
	ax_right.set_yticks(np.linspace(-1.0, 1.0, 11))
	ax_right.yaxis.set_minor_locator(AutoMinorLocator(2))
	ax_right.grid(axis="y", which="major", alpha=0.25, linestyle=":")
	ax_right.grid(axis="y", which="minor", alpha=0.15, linestyle=":")

	ax_left.set_xlabel("Camera Distance (m)")
	ax_left.set_ylabel("Horizontal Pixels (px)")
	ax_right.set_ylabel("Spatial Quantization Error (cm)")
	ax_left.set_title("Rod Pixel Length vs Camera Distance")
	ax_left.grid(axis="y", alpha=0.3)

	handles_left, labels_left = ax_left.get_legend_handles_labels()
	handles_right, labels_right = ax_right.get_legend_handles_labels()
	ax_left.legend(handles_left + handles_right, labels_left + labels_right, loc="best")

	fig.tight_layout()
	fig.savefig(output_dir / "pixel_occupancy_curve.png", dpi=160)
	plt.close(fig)

	# Estimate physical rod length from RGB-image measurement result.
	est_projected_lengths = np.array(rgb_estimated_lengths_px, dtype=np.float64)
	est_rod_lengths_m = est_projected_lengths * distances / fx
	true_rod_lengths_m = np.full_like(est_rod_lengths_m, rod_length_m)
	rod_length_errors_m = est_rod_lengths_m - true_rod_lengths_m

	fig_len, ax_len = plt.subplots(figsize=(8, 5))
	ax_len.scatter(
		distances,
		est_rod_lengths_m,
		s=16,
		color="tab:green",
		label="Estimated rod length from RGB image",
	)
	ax_len.vlines(
		distances,
		true_rod_lengths_m,
		est_rod_lengths_m,
		color="tab:green",
		linewidth=0.9,
		alpha=0.8,
		label="Error length",
	)
	ax_len.plot(
		distances,
		true_rod_lengths_m,
		color="black",
		linewidth=1.4,
		linestyle="--",
		label="Ground truth rod length",
	)
	for x, y, err in zip(distances, est_rod_lengths_m, rod_length_errors_m):
		ax_len.text(
			x,
			y,
			f"{err * 100.0:+.2f}cm",
			fontsize=6,
			color="tab:green",
			alpha=0.9,
			ha="left",
			va="bottom",
		)
	ax_len.set_xlabel("Camera Distance (m)")
	ax_len.set_ylabel("Estimated Rod Length (m)")
	ax_len.set_title("Rod Length Estimation from RGB Image vs Ground Truth")
	ax_len.grid(alpha=0.3)
	ax_len.legend(loc="best")
	fig_len.tight_layout()
	fig_len.savefig(output_dir / "rod_length_estimation_curve.png", dpi=160)
	plt.close(fig_len)

	# Speed estimation from measured quantized pixel length (rod_region_length_px).
	est_lengths_px = np.array(region_lengths_px, dtype=np.float64)
	est_lengths_px = np.maximum(est_lengths_px, 1.0)
	est_distances_m = fx * rod_length_m / est_lengths_px
	times_s = np.arange(len(est_distances_m), dtype=np.float64) * capture_interval_s
	inst_speeds_mps = np.diff(est_distances_m) / capture_interval_s
	times_mid_s = times_s[:-1] + capture_interval_s * 0.5

	# Smoothed speed (moving average) for clearer trend visualization.
	if inst_speeds_mps.size >= 5:
		smooth_window = 5
		kernel = np.ones(smooth_window, dtype=np.float64) / float(smooth_window)
		smoothed_speeds_mps = np.convolve(inst_speeds_mps, kernel, mode="same")
	else:
		smoothed_speeds_mps = inst_speeds_mps.copy()

	# Theoretical constant speed implied by distance schedule and frame interval.
	theoretical_speed_mps = (end_distance_m - start_distance_m) / max((len(distances) - 1) * capture_interval_s, 1e-12)
	theoretical_speeds_mps = np.full_like(times_mid_s, theoretical_speed_mps)

	speed_result = np.column_stack([times_mid_s, inst_speeds_mps, est_distances_m[:-1], est_distances_m[1:]])
	np.savetxt(
		output_dir / "speed_estimation.csv",
		speed_result,
		delimiter=",",
		header="time_mid_s,instant_speed_mps,distance_prev_m,distance_next_m",
		comments="",
	)

	fig_spd, ax_spd = plt.subplots(figsize=(8, 5))
	for t in times_mid_s:
		ax_spd.axvline(x=t, color="gray", linewidth=0.4, alpha=0.18, zorder=0)

	ax_spd.scatter(
		times_mid_s,
		inst_speeds_mps,
		s=18,
		color="tab:purple",
		edgecolors="none",
		label="Instantaneous speed",
	)
	ax_spd.plot(
		times_mid_s,
		inst_speeds_mps,
		color="tab:purple",
		linewidth=1.0,
		alpha=0.6,
		label="_nolegend_",
	)
	ax_spd.plot(
		times_mid_s,
		smoothed_speeds_mps,
		color="tab:orange",
		linewidth=2.0,
		label="Smoothed speed (5-point moving average)",
	)
	ax_spd.plot(
		times_mid_s,
		theoretical_speeds_mps,
		color="black",
		linewidth=1.4,
		linestyle="--",
		label="Ground truth speed",
	)
	ax_spd.set_xlabel("Time (s)")
	ax_spd.set_ylabel("Estimated Speed (m/s)")
	ax_spd.set_title("Rod Speed vs Time (from Quantized Pixel Length)")
	ax_spd.grid(alpha=0.3)
	ax_spd.legend(loc="best")
	fig_spd.tight_layout()
	fig_spd.savefig(output_dir / "speed_estimation_curve.png", dpi=160)
	plt.close(fig_spd)

	print("Experiment finished.")
	print(f"Output directory: {output_dir}")
	print(f"Distance samples: {len(distances)} ({start_distance_m:.2f}m -> {end_distance_m:.2f}m, step {step_distance_m:.2f}m)")
	print(f"Min white pixels: {int(np.min(pixel_counts))}")
	print(f"Max white pixels: {int(np.max(pixel_counts))}")
	print(f"Min rod region length(px): {int(np.min(region_lengths_px))}")
	print(f"Max rod region length(px): {int(np.max(region_lengths_px))}")
	print(f"Min rod region width(px): {int(np.min(region_widths_px))}")
	print(f"Max rod region width(px): {int(np.max(region_widths_px))}")
	print(f"Min length error(px): {float(np.min(length_errors)):.4f}")
	print(f"Max length error(px): {float(np.max(length_errors)):.4f}")
	print(f"Min length error(cm): {float(np.min(length_errors_cm)):.4f}")
	print(f"Max length error(cm): {float(np.max(length_errors_cm)):.4f}")
	print(f"Capture interval(s): {capture_interval_s:.4f}")
	print(f"Speed min(m/s): {float(np.min(inst_speeds_mps)):.6f}")
	print(f"Speed max(m/s): {float(np.max(inst_speeds_mps)):.6f}")
	print(f"Speed mean(m/s): {float(np.mean(inst_speeds_mps)):.6f}")
	print(f"Theoretical speed(m/s): {theoretical_speed_mps:.6f}")
	print(f"CSV: {output_dir / 'pixel_occupancy.csv'}")
	print(f"Curve: {output_dir / 'pixel_occupancy_curve.png'}")
	print(f"Length estimation curve: {output_dir / 'rod_length_estimation_curve.png'}")
	print(f"Speed CSV: {output_dir / 'speed_estimation.csv'}")
	print(f"Speed curve: {output_dir / 'speed_estimation_curve.png'}")
	if save_images:
		print(f"RGB frames: {output_dir / 'rgb_frames'}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Simulate monocular RGB imaging of a 1m-long, 1cm-diameter white rod in black 3D space "
			"and compare occupied pixel counts from 5m to 6m with 1cm intervals."
		)
	)
	parser.add_argument("--output-dir", type=Path, default=Path("res/spatial_error_res"))
	parser.add_argument("--image-width", type=int, default=1920)
	parser.add_argument("--image-height", type=int, default=1080)
	parser.add_argument("--hfov-deg", type=float, default=60.0)
	parser.add_argument("--rod-length-m", type=float, default=1.0)
	parser.add_argument("--rod-diameter-m", type=float, default=0.01)
	parser.add_argument("--start-distance-m", type=float, default=5.5)
	parser.add_argument("--end-distance-m", type=float, default=6.0)
	parser.add_argument("--step-distance-m", type=float, default=0.01)
	parser.add_argument("--capture-interval-s", type=float, default=0.01)
	parser.add_argument(
		"--no-save-images",
		action="store_true",
		help="If set, skip saving per-distance RGB frames and only export CSV + curve.",
	)
	return parser.parse_args()


def main() -> None:
	# Set True to use command-line arguments; False to use main-config values below.
	use_cli_args = False

	if use_cli_args:
		args = parse_args()
		run_experiment(
			output_dir=args.output_dir,
			image_width=args.image_width,
			image_height=args.image_height,
			hfov_deg=args.hfov_deg,
			rod_length_m=args.rod_length_m,
			rod_diameter_m=args.rod_diameter_m,
			start_distance_m=args.start_distance_m,
			end_distance_m=args.end_distance_m,
			step_distance_m=args.step_distance_m,
			capture_interval_s=args.capture_interval_s,
			save_images=not args.no_save_images,
		)
		return

	# Main-config experiment parameters (edit these directly).
	output_dir = Path("res/spatial_error_res")
	image_width = 1920
	image_height = 1080
	hfov_deg = 60.0
	rod_length_m = 1.0
	rod_diameter_m = 0.01
	start_distance_m = 2.5
	end_distance_m = 3.1
	num_samples = 61
	capture_interval_s = 1./30.0
	save_images = True

	if num_samples < 2:
		raise ValueError("num_samples must be >= 2")
	step_distance_m = (end_distance_m - start_distance_m) / float(num_samples - 1)

	run_experiment(
		output_dir=output_dir,
		image_width=image_width,
		image_height=image_height,
		hfov_deg=hfov_deg,
		rod_length_m=rod_length_m,
		rod_diameter_m=rod_diameter_m,
		start_distance_m=start_distance_m,
		end_distance_m=end_distance_m,
		step_distance_m=step_distance_m,
		capture_interval_s=capture_interval_s,
		save_images=save_images,
	)


if __name__ == "__main__":
	main()
