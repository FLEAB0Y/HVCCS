import os

import numpy as np

try:
	from scipy.interpolate import CubicSpline
except ImportError as exc:
	raise ImportError(
		"scipy is required for spline fitting. Install it with: pip install scipy"
	) from exc


def cubic_hermite_coefficients(x0, v0, x1, v1, dt):
	# y(t) = a * tau^3 + b * tau^2 + c * tau + d, tau = t - t0
	d = x0
	c = v0
	a = (2.0 * (x0 - x1) + dt * (v0 + v1)) / (dt ** 3)
	b = (3.0 * (x1 - x0) - dt * (2.0 * v0 + v1)) / (dt ** 2)
	return np.stack([a, b, c, d], axis=0)


def collect_h36m_npy_files(input_dir):
	if not os.path.isdir(input_dir):
		raise FileNotFoundError(f"Input directory not found: {input_dir}")

	files = []
	for name in os.listdir(input_dir):
		full_path = os.path.join(input_dir, name)
		if not os.path.isfile(full_path):
			continue
		files.append(name)

	files.sort()
	return files


def load_pose_array(npy_path):
	data = np.load(npy_path)

	# Common shape variants: (1, F, K, 3) or (F, K, 3)
	if data.ndim == 4 and data.shape[0] == 1:
		data = data[0]

	if data.ndim != 3 or data.shape[-1] != 3:
		raise ValueError(
			f"Unexpected npy shape {data.shape}. Expected (frames, keypoints, 3) or (1, frames, keypoints, 3)."
		)

	if data.shape[1] != 17:
		raise ValueError(
			f"Unexpected keypoint count {data.shape[1]}. This script expects 17 keypoints for h36m data."
		)

	if data.shape[0] < 2:
		raise ValueError(f"At least 2 frames are required, got {data.shape[0]}")

	return data.astype(np.float64, copy=False)


def fit_not_a_knot_splines(pose_data, fps=120.0):
	if fps <= 0:
		raise ValueError(f"fps must be > 0, got {fps}")

	num_frames, num_keypoints, num_dims = pose_data.shape
	if num_frames < 4:
		raise ValueError(
			f"Not-a-knot cubic spline fitting needs at least 4 frames, got {num_frames}"
		)
	time_sec = np.arange(num_frames, dtype=np.float64) / float(fps)

	# scipy CubicSpline coeff layout: c[k, i] for (x - x_i) ** (3-k)
	coeffs = np.empty((num_keypoints, num_dims, 4, num_frames - 1), dtype=np.float64)
	rmse = np.empty((num_keypoints, num_dims), dtype=np.float64)
	max_abs = np.empty((num_keypoints, num_dims), dtype=np.float64)

	for kp_idx in range(num_keypoints):
		for axis_idx in range(num_dims):
			y = pose_data[:, kp_idx, axis_idx]
			spline = CubicSpline(time_sec, y, bc_type="not-a-knot")
			coeffs[kp_idx, axis_idx] = spline.c

			recon = spline(time_sec)
			diff = recon - y
			rmse[kp_idx, axis_idx] = np.sqrt(np.mean(diff * diff))
			max_abs[kp_idx, axis_idx] = np.max(np.abs(diff))

	return time_sec, coeffs, rmse, max_abs


def estimate_frame_velocity(pose_data, dt):
	# Use central difference for interior frames and one-sided difference at boundaries.
	vel = np.empty_like(pose_data, dtype=np.float64)
	vel[0] = (pose_data[1] - pose_data[0]) / dt
	vel[-1] = (pose_data[-1] - pose_data[-2]) / dt
	if pose_data.shape[0] > 2:
		vel[1:-1] = (pose_data[2:] - pose_data[:-2]) / (2.0 * dt)
	return vel


def fit_clamped_truth_splines(pose_data, fps=120.0):
	if fps <= 0:
		raise ValueError(f"fps must be > 0, got {fps}")

	num_frames, num_keypoints, num_dims = pose_data.shape
	if num_frames < 2:
		raise ValueError(f"Clamped Hermite fitting needs at least 2 frames, got {num_frames}")

	dt = 1.0 / float(fps)
	time_sec = np.arange(num_frames, dtype=np.float64) * dt
	vel = estimate_frame_velocity(pose_data, dt)

	coeffs = np.empty((num_keypoints, num_dims, 4, num_frames - 1), dtype=np.float64)

	for seg_idx in range(num_frames - 1):
		x0 = pose_data[seg_idx].reshape(-1)
		v0 = vel[seg_idx].reshape(-1)
		x1 = pose_data[seg_idx + 1].reshape(-1)
		v1 = vel[seg_idx + 1].reshape(-1)

		seg_coeff = cubic_hermite_coefficients(x0, v0, x1, v1, dt).T
		coeffs[:, :, :, seg_idx] = seg_coeff.reshape(num_keypoints, num_dims, 4)

	# Reconstruct values at frame timestamps to report fitting error statistics.
	recon = np.empty_like(pose_data, dtype=np.float64)
	recon[0] = coeffs[:, :, 3, 0]
	for seg_idx in range(num_frames - 1):
		tau = dt
		a = coeffs[:, :, 0, seg_idx]
		b = coeffs[:, :, 1, seg_idx]
		c = coeffs[:, :, 2, seg_idx]
		d = coeffs[:, :, 3, seg_idx]
		recon[seg_idx + 1] = ((a * tau + b) * tau + c) * tau + d

	diff = recon - pose_data
	rmse = np.sqrt(np.mean(diff * diff, axis=0))
	max_abs = np.max(np.abs(diff), axis=0)

	return time_sec, coeffs, rmse, max_abs


def save_spline_result(save_path, source_shape, fps, bc_type, time_sec, coeffs, rmse, max_abs):
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	np.savez_compressed(
		save_path,
		source_shape=np.asarray(source_shape, dtype=np.int32),
		fps=np.asarray([fps], dtype=np.float64),
		bc_type=np.asarray([bc_type]),
		time_sec=time_sec,
		coeffs=coeffs,
		rmse=rmse,
		max_abs=max_abs,
	)


def process_folder(input_dir, output_dir, fps=120.0, fit_mode="notaknot"):
	fit_mode = fit_mode.lower().strip()
	if fit_mode not in {"notaknot", "clamped"}:
		raise ValueError(f"fit_mode must be one of ['notaknot', 'clamped'], got: {fit_mode}")
	# Allow output_dir templates like ".../{fit_mode}_splines".
	output_dir = output_dir.replace("{fit_mode}", fit_mode)

	files = collect_h36m_npy_files(input_dir)
	print(f"[{fit_mode}] Found {len(files)} file(s) in: {input_dir}")

	processed = 0
	failed = 0

	for idx, name in enumerate(files, start=1):
		input_path = os.path.join(input_dir, name)
		stem = os.path.splitext(name)[0]
		suffix = "notaknot_spline" if fit_mode == "notaknot" else "clamped_spline"
		output_path = os.path.join(output_dir, f"{stem}_{suffix}.npz")

		try:
			pose_data = load_pose_array(input_path)
			if fit_mode == "notaknot":
				time_sec, coeffs, rmse, max_abs = fit_not_a_knot_splines(pose_data, fps=fps)
				bc_type = "not-a-knot"
			else:
				time_sec, coeffs, rmse, max_abs = fit_clamped_truth_splines(pose_data, fps=fps)
				bc_type = "clamped_truth_hermite"
			save_spline_result(
				output_path,
				source_shape=pose_data.shape,
				fps=fps,
				bc_type=bc_type,
				time_sec=time_sec,
				coeffs=coeffs,
				rmse=rmse,
				max_abs=max_abs,
			)
			print(
				f"[{idx}/{len(files)}] Saved: {output_path} | "
				f"mean_rmse={rmse.mean():.6e}, max_abs={max_abs.max():.6e}"
			)
			processed += 1
		except Exception as exc:
			print(f"[{idx}/{len(files)}] Failed: {input_path} -> {exc}")
			failed += 1

	print("=" * 60)
	print(f"[{fit_mode}] Done. processed={processed}, failed={failed}, total={len(files)}")


if __name__ == "__main__":
	Label = "test/S2_cam_1"
	# Fitting mode: "notaknot" uses global cubic spline; "clamped" uses segment Hermite with truth endpoints.
	fit_mode = "clamped"
	# Source frame rate in Hz used to compute timestamps and dt.
	fps = 30.0
	# Input folder containing files that end with "h36m.npy".
	input_dir = f"/home/data/ztw/AtheletePose3D/h36m_pose_cam_1_downsample/{Label}_30fps"
	# Output folder template for fitted spline npz files. {fit_mode} will be replaced at runtime.
	output_dir = f"/home/data/ztw/AtheletePose3D/h36m_pose_cam_1_downsample/{Label}_30fps_{{fit_mode}}_splines"
	

	process_folder(
		input_dir=input_dir,
		output_dir=output_dir,
		fps=fps,
		fit_mode=fit_mode,
	)
