import os

import numpy as np

try:
	from scipy.interpolate import CubicSpline
except ImportError as exc:
	raise ImportError(
		"scipy is required for spline fitting. Install it with: pip install scipy"
	) from exc


def collect_h36m_npy_files(input_dir):
	if not os.path.isdir(input_dir):
		raise FileNotFoundError(f"Input directory not found: {input_dir}")

	files = []
	for name in os.listdir(input_dir):
		full_path = os.path.join(input_dir, name)
		if not os.path.isfile(full_path):
			continue
		if not name.lower().endswith("h36m.npy"):
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

	if data.shape[0] < 4:
		raise ValueError(
			f"Not-a-knot cubic spline fitting needs at least 4 frames, got {data.shape[0]}"
		)

	return data.astype(np.float64, copy=False)


def fit_not_a_knot_splines(pose_data, fps=120.0):
	if fps <= 0:
		raise ValueError(f"fps must be > 0, got {fps}")

	num_frames, num_keypoints, num_dims = pose_data.shape
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


def save_spline_result(save_path, source_shape, fps, time_sec, coeffs, rmse, max_abs):
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	np.savez_compressed(
		save_path,
		source_shape=np.asarray(source_shape, dtype=np.int32),
		fps=np.asarray([fps], dtype=np.float64),
		bc_type=np.asarray(["not-a-knot"]),
		time_sec=time_sec,
		coeffs=coeffs,
		rmse=rmse,
		max_abs=max_abs,
	)


def process_folder(input_dir, output_dir, fps=120.0):
	files = collect_h36m_npy_files(input_dir)
	print(f"Found {len(files)} file(s) in: {input_dir}")

	processed = 0
	failed = 0

	for idx, name in enumerate(files, start=1):
		input_path = os.path.join(input_dir, name)
		stem = os.path.splitext(name)[0]
		output_path = os.path.join(output_dir, f"{stem}_notaknot_spline.npz")

		try:
			pose_data = load_pose_array(input_path)
			time_sec, coeffs, rmse, max_abs = fit_not_a_knot_splines(pose_data, fps=fps)
			save_spline_result(
				output_path,
				source_shape=pose_data.shape,
				fps=fps,
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
	print(f"Done. processed={processed}, failed={failed}, total={len(files)}")


if __name__ == "__main__":
	input_dir = "/home/data/ztw/AtheletePose3D/data/valid_set/S2"
	output_dir = "/home/data/ztw/AtheletePose3D/data/valid_set/S2_splines"
	fps = 120.0

	process_folder(input_dir=input_dir, output_dir=output_dir, fps=fps)
