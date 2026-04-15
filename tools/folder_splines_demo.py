import os
import numpy as np
import matplotlib.pyplot as plt


def collect_spline_files(input_dir, suffix="_notaknot_spline.npz"):
	if not os.path.isdir(input_dir):
		raise FileNotFoundError(f"Input directory not found: {input_dir}")

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


def validate_keypoint(keypoint_idx, num_keypoints):
	if not isinstance(keypoint_idx, int):
		raise TypeError(f"keypoint_idx must be int, got {type(keypoint_idx)}")
	if keypoint_idx < 0 or keypoint_idx >= num_keypoints:
		raise ValueError(
			f"keypoint_idx out of range: {keypoint_idx}. Valid range: 0 to {num_keypoints - 1}"
		)


def load_pose_array(npy_path):
	if not os.path.isfile(npy_path):
		raise FileNotFoundError(f"Pose npy not found: {npy_path}")

	data = np.load(npy_path)
	if data.ndim == 4 and data.shape[0] == 1:
		data = data[0]

	if data.ndim != 3 or data.shape[-1] != 3:
		raise ValueError(
			f"Unexpected npy shape {data.shape}. Expected (frames, keypoints, 3) or (1, frames, keypoints, 3)."
		)

	return data.astype(np.float64, copy=False)


def spline_npz_to_pose_npy_name(npz_name, suffix="_notaknot_spline"):
	stem = os.path.splitext(npz_name)[0]
	if not stem.endswith(suffix):
		raise ValueError(f"Unexpected spline filename format: {npz_name}")
	return f"{stem[: -len(suffix)]}.npy"


def load_spline_npz(npz_path):
	data = np.load(npz_path, allow_pickle=False)

	required_keys = ["time_sec", "coeffs", "bc_type"]
	missing = [k for k in required_keys if k not in data]
	if missing:
		raise KeyError(f"Missing keys in {npz_path}: {missing}")

	time_sec = data["time_sec"].astype(np.float64)
	coeffs = data["coeffs"].astype(np.float64)
	bc_type = str(data["bc_type"].reshape(-1)[0])
	fps = float(data["fps"].reshape(-1)[0]) if "fps" in data else None

	if coeffs.ndim != 4:
		raise ValueError(
			f"Unexpected coeffs shape {coeffs.shape}, expected (K, 3, 4, num_segments)"
		)
	if coeffs.shape[1] != 3 or coeffs.shape[2] != 4:
		raise ValueError(
			f"Unexpected coeffs shape {coeffs.shape}, expected axis=3 and cubic-order=4"
		)
	if len(time_sec) != coeffs.shape[3] + 1:
		raise ValueError(
			f"time_sec length {len(time_sec)} does not match coeff segments {coeffs.shape[3]}"
		)

	return time_sec, coeffs, bc_type, fps


def downsample_discrete_points(pose_data, keypoint_idx, time_sec, fps_fallback, downsample_factor):
	if downsample_factor <= 0:
		raise ValueError(
			f"downsample_factor must be > 0, got {downsample_factor}"
		)

	validate_keypoint(keypoint_idx, pose_data.shape[1])
	num_frames = pose_data.shape[0]

	if len(time_sec) == num_frames:
		base_time_sec = time_sec
	else:
		if fps_fallback is None or fps_fallback <= 0:
			raise ValueError(
				"fps is missing in spline file and fps_fallback is invalid; cannot align discrete point timestamps"
			)
		base_time_sec = np.arange(num_frames, dtype=np.float64) / float(fps_fallback)

	selected = np.arange(0, num_frames, downsample_factor, dtype=np.int32)
	return base_time_sec[selected], pose_data[selected, keypoint_idx, :]


def evaluate_single_channel(time_sec, coeff_channel, samples_per_segment=20):
	if samples_per_segment < 2:
		raise ValueError(f"samples_per_segment must be >= 2, got {samples_per_segment}")

	num_segments = coeff_channel.shape[1]
	t_dense_all = []
	y_dense_all = []

	for seg_idx in range(num_segments):
		t0 = time_sec[seg_idx]
		t1 = time_sec[seg_idx + 1]
		if seg_idx == num_segments - 1:
			t_seg = np.linspace(t0, t1, samples_per_segment, endpoint=True)
		else:
			t_seg = np.linspace(t0, t1, samples_per_segment, endpoint=False)

		dt = t_seg - t0
		c0 = coeff_channel[0, seg_idx]
		c1 = coeff_channel[1, seg_idx]
		c2 = coeff_channel[2, seg_idx]
		c3 = coeff_channel[3, seg_idx]

		y_seg = ((c0 * dt + c1) * dt + c2) * dt + c3

		t_dense_all.append(t_seg)
		y_dense_all.append(y_seg)

	t_dense = np.concatenate(t_dense_all, axis=0)
	y_dense = np.concatenate(y_dense_all, axis=0)
	return t_dense, y_dense


def plot_keypoint_xyz_splines(
	time_sec,
	coeffs,
	point_time_sec,
	point_xyz,
	keypoint_idx,
	save_path,
	downsample_factor,
	title_prefix="",
):
	num_keypoints = coeffs.shape[0]
	validate_keypoint(keypoint_idx, num_keypoints)

	axis_names = ["X", "Y", "Z"]
	fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

	for axis_idx, axis_name in enumerate(axis_names):
		coeff_channel = coeffs[keypoint_idx, axis_idx]  # (4, num_segments)
		t_dense, y_dense = evaluate_single_channel(time_sec, coeff_channel, samples_per_segment=20)

		axes[axis_idx].plot(
			t_dense * 1000.0,
			y_dense,
			linewidth=1.2,
			color="tab:blue",
			label="spline curve",
		)
		axes[axis_idx].scatter(
			point_time_sec * 1000.0,
			point_xyz[:, axis_idx],
			s=14,
			alpha=0.85,
			color="tab:orange",
			label=f"discrete points (x{downsample_factor})",
		)
		axes[axis_idx].set_ylabel(f"{axis_name} (mm)")
		axes[axis_idx].grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
		axes[axis_idx].legend(loc="upper right", fontsize=8)

	fig.suptitle(
		f"{title_prefix} | keypoint={keypoint_idx} | Spline(not-a-knot) + Discrete Points",
		fontsize=13,
	)
	axes[-1].set_xlabel("Time (ms)")
	fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.97))

	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	fig.savefig(save_path, dpi=180)
	plt.close(fig)


def process_folder(
	input_dir,
	source_pose_dir,
	output_dir,
	keypoint_idx=0,
	point_downsample_factor=1,
	fps_fallback=120.0,
):
	files = collect_spline_files(input_dir)
	print(f"Found {len(files)} spline file(s) in: {input_dir}")
	print(f"Using keypoint_idx={keypoint_idx} (valid range 0-16 for h36m)")
	print(
		f"Discrete point source: {source_pose_dir} | downsample_factor={point_downsample_factor}"
	)

	processed = 0
	failed = 0

	for idx, name in enumerate(files, start=1):
		npz_path = os.path.join(input_dir, name)
		stem = os.path.splitext(name)[0]
		save_path = os.path.join(
			output_dir,
			f"{stem}_kp{keypoint_idx}_xyz_ds{point_downsample_factor}.png",
		)

		try:
			time_sec, coeffs, bc_type, fps = load_spline_npz(npz_path)
			if bc_type != "not-a-knot":
				print(f"[{idx}/{len(files)}] Warning: bc_type={bc_type}, file={name}")

			source_npy_name = spline_npz_to_pose_npy_name(name)
			source_npy_path = os.path.join(source_pose_dir, source_npy_name)
			pose_data = load_pose_array(source_npy_path)

			point_time_sec, point_xyz = downsample_discrete_points(
				pose_data=pose_data,
				keypoint_idx=keypoint_idx,
				time_sec=time_sec,
				fps_fallback=fps if fps is not None else fps_fallback,
				downsample_factor=point_downsample_factor,
			)

			plot_keypoint_xyz_splines(
				time_sec=time_sec,
				coeffs=coeffs,
				point_time_sec=point_time_sec,
				point_xyz=point_xyz,
				keypoint_idx=keypoint_idx,
				save_path=save_path,
				downsample_factor=point_downsample_factor,
				title_prefix=stem,
			)
			print(f"[{idx}/{len(files)}] Saved: {save_path}")
			processed += 1
		except Exception as exc:
			print(f"[{idx}/{len(files)}] Failed: {npz_path} -> {exc}")
			failed += 1

	print("=" * 60)
	print(f"Done. processed={processed}, failed={failed}, total={len(files)}")


if __name__ == "__main__":
	input_dir = "/home/data/ztw/AtheletePose3D/h36m_pose_cam_1_downsample/test/S2_cam_1_30fps_notaknot_splines"
	source_pose_dir = "/home/data/ztw/AtheletePose3D/h36m_pose_cam_1_downsample/test/S2_cam_1_30fps"
	output_dir = "/home/ztw/HVCCS/res/S3_splines_plots"
	keypoint_idx = 0  # set in [0, 16]
	point_downsample_factor = 1  # set >= 1

	process_folder(
		input_dir=input_dir,
		source_pose_dir=source_pose_dir,
		output_dir=output_dir,
		keypoint_idx=keypoint_idx,
		point_downsample_factor=point_downsample_factor,
	)
