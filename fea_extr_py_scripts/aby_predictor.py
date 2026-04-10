import argparse
import os

import numpy as np


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

	if data.ndim == 4 and data.shape[0] == 1:
		data = data[0]

	if data.ndim != 3 or data.shape[-1] != 3:
		raise ValueError(
			f"Unexpected npy shape {data.shape}. Expected (frames, keypoints, 3) or (1, frames, keypoints, 3)."
		)
	if data.shape[1] != 17:
		raise ValueError(f"Expected 17 keypoints for h36m data, got {data.shape[1]}")
	if data.shape[0] < 2:
		raise ValueError(f"At least 2 frames are required, got {data.shape[0]}")

	return data.astype(np.float64, copy=False)


def cubic_hermite_coefficients(x0, v0, x1, v1, dt):
	d = x0
	c = v0
	a = (2.0 * (x0 - x1) + dt * (v0 + v1)) / (dt ** 3)
	b = (3.0 * (x1 - x0) - dt * (2.0 * v0 + v1)) / (dt ** 2)
	return np.stack([a, b, c, d], axis=0)


def fit_realtime_segments_abg(
	pose_data,
	fps=120.0,
	alpha=0.65,
	beta=0.08,
	gamma=0.005,
):
	if fps <= 0:
		raise ValueError(f"fps must be > 0, got {fps}")
	if alpha < 0 or beta < 0 or gamma < 0:
		raise ValueError("alpha/beta/gamma must be non-negative")

	num_frames, num_keypoints, num_dims = pose_data.shape
	channels = num_keypoints * num_dims
	dt = 1.0 / float(fps)
	time_sec = np.arange(num_frames, dtype=np.float64) * dt

	z = pose_data.reshape(num_frames, channels)

	x = z[0].copy()
	v = np.zeros(channels, dtype=np.float64)
	a = np.zeros(channels, dtype=np.float64)

	x_est = np.empty((num_frames, channels), dtype=np.float64)
	v_est = np.empty((num_frames, channels), dtype=np.float64)
	a_est = np.empty((num_frames, channels), dtype=np.float64)
	x_est[0] = x
	v_est[0] = v
	a_est[0] = a

	for k in range(1, num_frames):
		# Predict to current frame k
		x_pred = x + v * dt + 0.5 * a * (dt ** 2)
		v_pred = v + a * dt
		a_pred = a

		residual = z[k] - x_pred

		x = x_pred + alpha * residual
		v = v_pred + (beta / dt) * residual
		a = a_pred + (2.0 * gamma / (dt ** 2)) * residual

		x_est[k] = x
		v_est[k] = v
		a_est[k] = a

	coeffs = np.empty((num_frames - 1, channels, 4), dtype=np.float64)
	pred_x_next = np.empty((num_frames - 1, channels), dtype=np.float64)
	pred_v_next = np.empty((num_frames - 1, channels), dtype=np.float64)

	for k in range(num_frames - 1):
		x0 = x_est[k]
		v0 = v_est[k]
		a0 = a_est[k]

		x1_pred = x0 + v0 * dt + 0.5 * a0 * (dt ** 2)
		v1_pred = v0 + a0 * dt

		pred_x_next[k] = x1_pred
		pred_v_next[k] = v1_pred
		coeffs[k] = cubic_hermite_coefficients(x0, v0, x1_pred, v1_pred, dt).T

	coeffs = coeffs.transpose(1, 2, 0).reshape(num_keypoints, num_dims, 4, num_frames - 1)
	x_est = x_est.reshape(num_frames, num_keypoints, num_dims)
	v_est = v_est.reshape(num_frames, num_keypoints, num_dims)
	a_est = a_est.reshape(num_frames, num_keypoints, num_dims)
	pred_x_next = pred_x_next.reshape(num_frames - 1, num_keypoints, num_dims)
	pred_v_next = pred_v_next.reshape(num_frames - 1, num_keypoints, num_dims)

	return {
		"time_sec": time_sec,
		"coeffs": coeffs,
		"x_est": x_est,
		"v_est": v_est,
		"a_est": a_est,
		"pred_x_next": pred_x_next,
		"pred_v_next": pred_v_next,
		"fps": float(fps),
		"dt": float(dt),
		"alpha": float(alpha),
		"beta": float(beta),
		"gamma": float(gamma),
	}


def save_result(save_path, source_shape, result):
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	np.savez_compressed(
		save_path,
		source_shape=np.asarray(source_shape, dtype=np.int32),
		predictor=np.asarray(["alpha_beta_gamma"]),
		bc_type=np.asarray(["hermite_endpoint_prediction"]),
		fps=np.asarray([result["fps"]], dtype=np.float64),
		dt=np.asarray([result["dt"]], dtype=np.float64),
		alpha=np.asarray([result["alpha"]], dtype=np.float64),
		beta=np.asarray([result["beta"]], dtype=np.float64),
		gamma=np.asarray([result["gamma"]], dtype=np.float64),
		time_sec=result["time_sec"],
		coeffs=result["coeffs"],
		x_est=result["x_est"],
		v_est=result["v_est"],
		a_est=result["a_est"],
		pred_x_next=result["pred_x_next"],
		pred_v_next=result["pred_v_next"],
	)


def process_folder(
	input_dir,
	output_dir,
	fps=120.0,
	alpha=0.65,
	beta=0.08,
	gamma=0.005,
):
	files = collect_h36m_npy_files(input_dir)
	print(f"[ABG] Found {len(files)} file(s) in: {input_dir}")

	processed = 0
	failed = 0

	for idx, name in enumerate(files, start=1):
		input_path = os.path.join(input_dir, name)
		stem = os.path.splitext(name)[0]
		output_path = os.path.join(output_dir, f"{stem}_abg_realtime_spline.npz")
		try:
			pose_data = load_pose_array(input_path)
			result = fit_realtime_segments_abg(
				pose_data=pose_data,
				fps=fps,
				alpha=alpha,
				beta=beta,
				gamma=gamma,
			)
			save_result(output_path, source_shape=pose_data.shape, result=result)
			print(f"[{idx}/{len(files)}] Saved: {output_path}")
			processed += 1
		except Exception as exc:
			print(f"[{idx}/{len(files)}] Failed: {input_path} -> {exc}")
			failed += 1

	print("=" * 60)
	print(f"[ABG] Done. processed={processed}, failed={failed}, total={len(files)}")


def build_argparser():
	parser = argparse.ArgumentParser(
		description="Realtime segment fitting with alpha-beta-gamma prediction for x_{k+1}, v_{k+1}."
	)
	parser.add_argument(
		"--input-dir",
		type=str,
		default="/home/data/ztw/AtheletePose3D/data/train_set/S3",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="/home/ztw/HVCCS/res/splines_fit_abg",
	)
	parser.add_argument("--fps", type=float, default=120.0)
	parser.add_argument("--alpha", type=float, default=0.65)
	parser.add_argument("--beta", type=float, default=0.08)
	parser.add_argument("--gamma", type=float, default=0.005)
	return parser


if __name__ == "__main__":
	args = build_argparser().parse_args()
	process_folder(
		input_dir=args.input_dir,
		output_dir=args.output_dir,
		fps=args.fps,
		alpha=args.alpha,
		beta=args.beta,
		gamma=args.gamma,
	)
