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
	# y(t) = a * tau^3 + b * tau^2 + c * tau + d, tau = t - t0
	d = x0
	c = v0
	a = (2.0 * (x0 - x1) + dt * (v0 + v1)) / (dt ** 3)
	b = (3.0 * (x1 - x0) - dt * (2.0 * v0 + v1)) / (dt ** 2)
	return np.stack([a, b, c, d], axis=0)


class KalmanCVPredictor:
	has_acceleration = False

	def __init__(
		self,
		channels,
		process_acc_var=3e5,
		measurement_var=9.0,
		init_pos_var=1.0,
		init_vel_var=1e4,
	):
		self.channels = int(channels)
		self.process_acc_var = float(process_acc_var)
		self.measurement_var = float(measurement_var)
		self.init_pos_var = float(init_pos_var)
		self.init_vel_var = float(init_vel_var)
		self._initialized = False

	def initialize(self, measurement0):
		self.x = measurement0.copy()
		self.v = np.zeros(self.channels, dtype=np.float64)

		self.p00 = np.full(self.channels, self.init_pos_var, dtype=np.float64)
		self.p01 = np.zeros(self.channels, dtype=np.float64)
		self.p11 = np.full(self.channels, self.init_vel_var, dtype=np.float64)
		self._initialized = True

	def update(self, measurement_k, dt):
		if not self._initialized:
			raise RuntimeError("Kalman predictor is not initialized")

		x_pred = self.x + dt * self.v
		v_pred = self.v

		q11 = self.process_acc_var * (dt ** 4) / 4.0
		q12 = self.process_acc_var * (dt ** 3) / 2.0
		q22 = self.process_acc_var * (dt ** 2)

		p00_pred = self.p00 + 2.0 * dt * self.p01 + (dt ** 2) * self.p11 + q11
		p01_pred = self.p01 + dt * self.p11 + q12
		p11_pred = self.p11 + q22

		residual = measurement_k - x_pred
		s = p00_pred + self.measurement_var
		k0 = p00_pred / s
		k1 = p01_pred / s

		self.x = x_pred + k0 * residual
		self.v = v_pred + k1 * residual

		self.p00 = (1.0 - k0) * p00_pred
		self.p01 = (1.0 - k0) * p01_pred
		self.p11 = p11_pred - k1 * p01_pred

	def get_state(self):
		return self.x, self.v, None


class ABGPredictor:
	has_acceleration = True

	def __init__(self, channels, alpha=0.65, beta=0.08, gamma=0.005):
		self.channels = int(channels)
		self.alpha = float(alpha)
		self.beta = float(beta)
		self.gamma = float(gamma)
		if self.alpha < 0 or self.beta < 0 or self.gamma < 0:
			raise ValueError("alpha/beta/gamma must be non-negative")
		self._initialized = False

	def initialize(self, measurement0):
		self.x = measurement0.copy()
		self.v = np.zeros(self.channels, dtype=np.float64)
		self.a = np.zeros(self.channels, dtype=np.float64)
		self._initialized = True

	def update(self, measurement_k, dt):
		if not self._initialized:
			raise RuntimeError("ABG predictor is not initialized")

		x_pred = self.x + self.v * dt + 0.5 * self.a * (dt ** 2)
		v_pred = self.v + self.a * dt
		a_pred = self.a

		residual = measurement_k - x_pred

		self.x = x_pred + self.alpha * residual
		self.v = v_pred + (self.beta / dt) * residual
		self.a = a_pred + (2.0 * self.gamma / (dt ** 2)) * residual

	def get_state(self):
		return self.x, self.v, self.a


def create_predictor(
	predictor_type,
	channels,
	process_acc_var=3e5,
	measurement_var=9.0,
	init_pos_var=1.0,
	init_vel_var=1e4,
	alpha=0.65,
	beta=0.08,
	gamma=0.005,
):
	if predictor_type == "kalman":
		return KalmanCVPredictor(
			channels=channels,
			process_acc_var=process_acc_var,
			measurement_var=measurement_var,
			init_pos_var=init_pos_var,
			init_vel_var=init_vel_var,
		)
	if predictor_type == "abg":
		return ABGPredictor(
			channels=channels,
			alpha=alpha,
			beta=beta,
			gamma=gamma,
		)
	raise ValueError(f"Unsupported predictor_type: {predictor_type}")


def run_predictor_states(pose_data, fps, predictor):
	if fps <= 0:
		raise ValueError(f"fps must be > 0, got {fps}")

	num_frames, num_keypoints, num_dims = pose_data.shape
	channels = num_keypoints * num_dims
	dt = 1.0 / float(fps)
	time_sec = np.arange(num_frames, dtype=np.float64) * dt
	z = pose_data.reshape(num_frames, channels)

	predictor.initialize(z[0])
	x0, v0, a0 = predictor.get_state()

	x_est = np.empty((num_frames, channels), dtype=np.float64)
	v_est = np.empty((num_frames, channels), dtype=np.float64)
	a_est = np.empty((num_frames, channels), dtype=np.float64) if predictor.has_acceleration else None

	x_est[0] = x0
	v_est[0] = v0
	if a_est is not None:
		a_est[0] = a0

	for k in range(1, num_frames):
		predictor.update(z[k], dt)
		xk, vk, ak = predictor.get_state()
		x_est[k] = xk
		v_est[k] = vk
		if a_est is not None:
			a_est[k] = ak

	return time_sec, dt, x_est, v_est, a_est


def build_realtime_hermite_segments(x_est, v_est, a_est, dt):
	num_frames, channels = x_est.shape
	coeffs = np.empty((num_frames - 1, channels, 4), dtype=np.float64)
	pred_x_next = np.empty((num_frames - 1, channels), dtype=np.float64)
	pred_v_next = np.empty((num_frames - 1, channels), dtype=np.float64)

	for k in range(num_frames - 1):
		xk = x_est[k]
		vk = v_est[k]

		if a_est is None:
			x1_pred = xk + dt * vk
			v1_pred = vk
		else:
			ak = a_est[k]
			x1_pred = xk + vk * dt + 0.5 * ak * (dt ** 2)
			v1_pred = vk + ak * dt

		pred_x_next[k] = x1_pred
		pred_v_next[k] = v1_pred
		coeffs[k] = cubic_hermite_coefficients(xk, vk, x1_pred, v1_pred, dt).T

	return coeffs, pred_x_next, pred_v_next


def fit_realtime_segments_with_predictor(pose_data, fps, predictor):
	num_frames, num_keypoints, num_dims = pose_data.shape
	time_sec, dt, x_est, v_est, a_est = run_predictor_states(pose_data, fps, predictor)
	coeffs, pred_x_next, pred_v_next = build_realtime_hermite_segments(x_est, v_est, a_est, dt)

	coeffs = coeffs.transpose(1, 2, 0).reshape(num_keypoints, num_dims, 4, num_frames - 1)
	x_est = x_est.reshape(num_frames, num_keypoints, num_dims)
	v_est = v_est.reshape(num_frames, num_keypoints, num_dims)
	pred_x_next = pred_x_next.reshape(num_frames - 1, num_keypoints, num_dims)
	pred_v_next = pred_v_next.reshape(num_frames - 1, num_keypoints, num_dims)

	result = {
		"time_sec": time_sec,
		"coeffs": coeffs,
		"x_est": x_est,
		"v_est": v_est,
		"pred_x_next": pred_x_next,
		"pred_v_next": pred_v_next,
		"fps": float(fps),
		"dt": float(dt),
	}
	if a_est is not None:
		result["a_est"] = a_est.reshape(num_frames, num_keypoints, num_dims)

	return result


def save_result(save_path, source_shape, result, predictor_name):
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	predictor_label = "kalman_cv" if predictor_name == "kalman" else "alpha_beta_gamma"
	np.savez_compressed(
		save_path,
		source_shape=np.asarray(source_shape, dtype=np.int32),
		predictor=np.asarray([predictor_label]),
		bc_type=np.asarray(["hermite_endpoint_prediction"]),
		fps=np.asarray([result["fps"]], dtype=np.float64),
		dt=np.asarray([result["dt"]], dtype=np.float64),
		time_sec=result["time_sec"],
		coeffs=result["coeffs"],
		x_est=result["x_est"],
		v_est=result["v_est"],
		pred_x_next=result["pred_x_next"],
		pred_v_next=result["pred_v_next"],
		a_est=result["a_est"] if "a_est" in result else np.asarray([], dtype=np.float64),
		alpha=np.asarray([result["alpha"]], dtype=np.float64) if "alpha" in result else np.asarray([], dtype=np.float64),
		beta=np.asarray([result["beta"]], dtype=np.float64) if "beta" in result else np.asarray([], dtype=np.float64),
		gamma=np.asarray([result["gamma"]], dtype=np.float64) if "gamma" in result else np.asarray([], dtype=np.float64),
	)


def process_folder(
	input_dir,
	output_dir,
	predictor_type="kalman",
	fps=120.0,
	process_acc_var=3e5,
	measurement_var=9.0,
	init_pos_var=1.0,
	init_vel_var=1e4,
	alpha=0.65,
	beta=0.08,
	gamma=0.005,
):
	predictor_type = predictor_type.lower().strip()
	if predictor_type not in {"kalman", "abg"}:
		raise ValueError(f"predictor_type must be one of ['kalman', 'abg'], got: {predictor_type}")

	files = collect_h36m_npy_files(input_dir)
	print(f"[{predictor_type}] Found {len(files)} file(s) in: {input_dir}")

	processed = 0
	failed = 0

	for idx, name in enumerate(files, start=1):
		input_path = os.path.join(input_dir, name)
		stem = os.path.splitext(name)[0]
		suffix = "kalman_realtime_spline" if predictor_type == "kalman" else "abg_realtime_spline"
		output_path = os.path.join(output_dir, f"{stem}_{suffix}.npz")
		try:
			pose_data = load_pose_array(input_path)
			channels = pose_data.shape[1] * pose_data.shape[2]
			predictor = create_predictor(
				predictor_type=predictor_type,
				channels=channels,
				process_acc_var=process_acc_var,
				measurement_var=measurement_var,
				init_pos_var=init_pos_var,
				init_vel_var=init_vel_var,
				alpha=alpha,
				beta=beta,
				gamma=gamma,
			)
			result = fit_realtime_segments_with_predictor(
				pose_data=pose_data,
				fps=fps,
				predictor=predictor,
			)
			if predictor_type == "abg":
				result["alpha"] = float(alpha)
				result["beta"] = float(beta)
				result["gamma"] = float(gamma)
			save_result(output_path, source_shape=pose_data.shape, result=result, predictor_name=predictor_type)
			print(f"[{idx}/{len(files)}] Saved: {output_path}")
			processed += 1
		except Exception as exc:
			print(f"[{idx}/{len(files)}] Failed: {input_path} -> {exc}")
			failed += 1

	print("=" * 60)
	print(f"[{predictor_type}] Done. processed={processed}, failed={failed}, total={len(files)}")


if __name__ == "__main__":
	# Predictor type: "kalman" (constant-velocity Kalman) or "abg" (alpha-beta-gamma).
	predictor_type = "kalman"
	# Input folder containing files that end with "h36m.npy".
	input_dir = "/home/data/ztw/AtheletePose3D/data/train_set/S3"
	# Output folder for saved spline files. Set to None to use auto default by predictor type.
	output_dir = None
	# Source frame rate in Hz used to build the time axis and dt.
	fps = 120.0

	# Kalman process noise variance (acceleration model). Larger value tracks motion changes faster.
	process_acc_var = 3e5
	# Kalman measurement noise variance. Larger value trusts observations less.
	measurement_var = 9.0
	# Initial variance of position state in Kalman filter.
	init_pos_var = 1.0
	# Initial variance of velocity state in Kalman filter.
	init_vel_var = 1e4

	# ABG gain for position correction.
	alpha = 0.65
	# ABG gain for velocity correction.
	beta = 0.08
	# ABG gain for acceleration correction.
	gamma = 0.005

	if output_dir is None:
		output_dir = "/home/ztw/HVCCS/res/splines_fit_kalman" if predictor_type == "kalman" else "/home/ztw/HVCCS/res/splines_fit_abg"

	process_folder(
		input_dir=input_dir,
		output_dir=output_dir,
		predictor_type=predictor_type,
		fps=fps,
		process_acc_var=process_acc_var,
		measurement_var=measurement_var,
		init_pos_var=init_pos_var,
		init_vel_var=init_vel_var,
		alpha=alpha,
		beta=beta,
		gamma=gamma,
	)