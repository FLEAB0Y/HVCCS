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


def fit_hermite_segment_from_endpoint_states(x_prev, x_curr, v_prev, v_curr, dt):
	"""Fit one segment [k, k+1] from endpoint states only."""
	return cubic_hermite_coefficients(x_prev, v_prev, x_curr, v_curr, dt).T


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


class BaselineTruthHistoryPredictor:
	has_acceleration = True
	has_jerk = True

	def __init__(self, channels):
		self.channels = int(channels)
		self._initialized = False

	def initialize(self, measurement0):
		self.x = measurement0.copy()
		self.v = np.zeros(self.channels, dtype=np.float64)
		self.a = np.zeros(self.channels, dtype=np.float64)
		self.j = np.zeros(self.channels, dtype=np.float64)
		self._prev_x = self.x.copy()
		self._prev_v_half = self.v.copy()
		self._prev_a = self.a.copy()
		self._initialized = True

	def update(self, measurement_k, dt):
		if not self._initialized:
			raise RuntimeError("Baseline predictor is not initialized")

		x_new = measurement_k.copy()
		# Instantaneous finite-difference states from truth coordinates:
		# v_{k-0.5} = (x_k - x_{k-1}) / dt
		# a_{k-1}   = (v_{k-0.5} - v_{k-1.5}) / dt
		# j_{k-1.5} = (a_{k-1} - a_{k-2}) / dt
		v_half = (x_new - self._prev_x) / dt
		a_new = (v_half - self._prev_v_half) / dt
		j_new = (a_new - self._prev_a) / dt

		self.x = x_new
		self.v = v_half
		self.a = a_new
		self.j = j_new
		self._prev_x = x_new
		self._prev_v_half = v_half
		self._prev_a = a_new

	def get_state(self):
		return self.x, self.v, self.a


def select_torch_device(cuda_device):
	import torch

	if cuda_device < 0:
		return torch.device("cpu")
	if not torch.cuda.is_available():
		return torch.device("cpu")
	if cuda_device >= torch.cuda.device_count():
		return torch.device("cpu")
	return torch.device(f"cuda:{cuda_device}")


class MambaPredictor:
	has_acceleration = False

	def __init__(
		self,
		channels,
		num_keypoints,
		num_dims,
		checkpoint_path,
		history_len=8,
		cuda_device=-1,
	):
		if not checkpoint_path:
			raise ValueError("mamba checkpoint_path is required when predictor_type='mamba'")
		self.channels = int(channels)
		self.num_keypoints = int(num_keypoints)
		self.num_dims = int(num_dims)
		if self.channels != self.num_keypoints * self.num_dims:
			raise ValueError(
				f"channels mismatch: channels={self.channels}, num_keypoints={self.num_keypoints}, num_dims={self.num_dims}"
			)

		self.history_len = int(history_len)
		self.model, self.device = self._load_model(checkpoint_path, cuda_device)
		if self.history_len <= 0:
			self.history_len = 8

		self._history = []
		self._prev_x = None
		self._initialized = False

	def _resolve_predictor_class(self):
		"""Resolve predictor class only from spline-training program modules."""
		candidates = [
			("splines_fit_train", "SpatioTemporalPredictor"),
			("fea_extr_py_scripts.splines_fit_train", "SpatioTemporalPredictor"),
		]
		loaded = []
		for module_name, class_name in candidates:
			try:
				module = __import__(module_name, fromlist=[class_name])
				cls = getattr(module, class_name)
				loaded.append((module_name, cls))
			except Exception:
				continue

		if not loaded:
			raise ImportError(
				"Failed to import SpatioTemporalPredictor from splines_fit_train.py"
			)

		return loaded

	def _infer_model_hparams_from_state_dict(self, state_dict, meta, meta_in_dim, meta_num_keypoints):
		gnn_hidden = int(meta.get("gnn_hidden", 0))
		if gnn_hidden <= 0 and "gnn.gcn1.fc.weight" in state_dict:
			gnn_hidden = int(state_dict["gnn.gcn1.fc.weight"].shape[0])
		if gnn_hidden <= 0:
			gnn_hidden = 128

		mamba_n_layer = int(meta.get("mamba_n_layer", 0))
		if mamba_n_layer <= 0:
			block_ids = set()
			for name in state_dict.keys():
				if name.startswith("mamba_blocks."):
					parts = name.split(".")
					if len(parts) > 1 and parts[1].isdigit():
						block_ids.add(int(parts[1]))
			if block_ids:
				mamba_n_layer = max(block_ids) + 1
		if mamba_n_layer <= 0:
			mamba_n_layer = 4

		mamba_d_conv = int(meta.get("mamba_d_conv", 0))
		if mamba_d_conv <= 0 and "mamba_blocks.0.mamba.conv1d.weight" in state_dict:
			mamba_d_conv = int(state_dict["mamba_blocks.0.mamba.conv1d.weight"].shape[-1])
		if mamba_d_conv <= 0:
			mamba_d_conv = 4

		mamba_d_state = int(meta.get("mamba_d_state", 0))
		if mamba_d_state <= 0 and "mamba_blocks.0.mamba.A_log" in state_dict:
			mamba_d_state = int(state_dict["mamba_blocks.0.mamba.A_log"].shape[-1])
		if mamba_d_state <= 0:
			mamba_d_state = 64

		mamba_expand = int(meta.get("mamba_expand", 0))
		if mamba_expand <= 0 and "mamba_blocks.0.mamba.in_proj.weight" in state_dict and gnn_hidden > 0:
			in_proj_out = int(state_dict["mamba_blocks.0.mamba.in_proj.weight"].shape[0])
			d_inner = in_proj_out // 2
			if d_inner > 0:
				mamba_expand = max(1, int(round(d_inner / float(gnn_hidden))))
		if mamba_expand <= 0:
			mamba_expand = 4

		return {
			"in_dim": meta_in_dim,
			"gnn_hidden": gnn_hidden,
			"mamba_d_state": mamba_d_state,
			"mamba_d_conv": mamba_d_conv,
			"mamba_expand": mamba_expand,
			"mamba_n_layer": mamba_n_layer,
			"num_nodes": meta_num_keypoints,
		}

	def _load_model(self, checkpoint_path, cuda_device):
		import inspect

		try:
			import torch
		except ImportError as exc:
			raise ImportError("torch is required for mamba predictor") from exc

		if not os.path.exists(checkpoint_path):
			raise FileNotFoundError(f"mamba checkpoint not found: {checkpoint_path}")

		ckpt = torch.load(checkpoint_path, map_location="cpu")
		if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
			state_dict = ckpt["model_state_dict"]
			meta = ckpt.get("meta", {})
		else:
			state_dict = ckpt
			meta = {}

		meta_num_keypoints = int(meta.get("num_keypoints", self.num_keypoints))
		meta_in_dim = int(meta.get("model_in_dim", self.num_dims))
		meta_out_dim = int(meta.get("model_out_dim", meta_in_dim))
		if meta_num_keypoints != self.num_keypoints:
			raise ValueError(
				f"mamba checkpoint num_keypoints mismatch: ckpt={meta_num_keypoints}, data={self.num_keypoints}"
			)
		if meta_in_dim != self.num_dims:
			raise ValueError(
				f"mamba checkpoint model_in_dim mismatch: ckpt={meta_in_dim}, data={self.num_dims}"
			)

		if self.history_len <= 0:
			self.history_len = int(meta.get("history_len", 8))

		classes = self._resolve_predictor_class()
		base_kwargs = self._infer_model_hparams_from_state_dict(
			state_dict=state_dict,
			meta=meta,
			meta_in_dim=meta_in_dim,
			meta_num_keypoints=meta_num_keypoints,
		)

		model = None
		last_error = None
		for module_name, predictor_cls in classes:
			try:
				init_sig = inspect.signature(predictor_cls.__init__)
				model_kwargs = dict(base_kwargs)
				if "out_dim" in init_sig.parameters:
					model_kwargs["out_dim"] = meta_out_dim
				elif meta_out_dim != meta_in_dim:
					# This class cannot represent xyz+vxyz outputs.
					continue
				model = predictor_cls(**model_kwargs)
				print(f"Loaded SpatioTemporalPredictor from: {module_name}")
				break
			except Exception as exc:
				last_error = exc
				continue

		if model is None:
			if meta_out_dim != meta_in_dim:
				raise ValueError(
					"Loaded checkpoint expects model_out_dim != model_in_dim, but no available "
					"SpatioTemporalPredictor implementation with out_dim support could be used."
				) from last_error
			raise RuntimeError("Failed to initialize SpatioTemporalPredictor") from last_error

		model.load_state_dict(state_dict, strict=True)
		device = select_torch_device(cuda_device)
		model = model.to(device)
		model.eval()
		return model, device

	def initialize(self, measurement0):
		self.x = measurement0.copy()
		self.v = np.zeros(self.channels, dtype=np.float64)
		self._history = [measurement0.reshape(self.num_keypoints, self.num_dims).copy()]
		self._prev_x = self.x.copy()
		self._initialized = True

	def update(self, measurement_k, dt):
		if not self._initialized:
			raise RuntimeError("Mamba predictor is not initialized")

		self.x = measurement_k.copy()
		if self._prev_x is None:
			self.v = np.zeros(self.channels, dtype=np.float64)
		else:
			self.v = (self.x - self._prev_x) / dt

		self._history.append(self.x.reshape(self.num_keypoints, self.num_dims).copy())
		if len(self._history) > self.history_len:
			self._history.pop(0)

		if len(self._history) >= self.history_len:
			import torch

			seq = np.stack(self._history[-self.history_len :], axis=0).astype(np.float32, copy=False)
			seq_tensor = torch.from_numpy(seq).unsqueeze(0).to(self.device)
			with torch.no_grad():
				pred_next = self.model(seq_tensor).detach().cpu().numpy()[0]

			if pred_next.shape[0] != self.num_keypoints or pred_next.shape[1] < self.num_dims:
				raise ValueError(
					f"Unexpected mamba prediction shape {pred_next.shape}, expected ({self.num_keypoints}, >= {self.num_dims})"
				)

			next_x = pred_next[:, : self.num_dims].reshape(-1).astype(np.float64, copy=False)
			if pred_next.shape[1] >= 2 * self.num_dims:
				pred_v = pred_next[:, self.num_dims : 2 * self.num_dims].reshape(-1)
				self.v = pred_v.astype(np.float64, copy=False)
			else:
				# If model does not expose velocity head, derive slope from predicted next position.
				self.v = (next_x - self.x) / dt

		self._prev_x = self.x.copy()

	def get_state(self):
		return self.x, self.v, None


def create_predictor(
	predictor_type,
	channels,
	num_keypoints=17,
	num_dims=3,
	process_acc_var=3e5,
	measurement_var=9.0,
	init_pos_var=1.0,
	init_vel_var=1e4,
	alpha=0.65,
	beta=0.08,
	gamma=0.005,
	mamba_checkpoint_path="",
	mamba_history_len=8,
	mamba_cuda_device=-1,
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
	if predictor_type == "mamba":
		return MambaPredictor(
			channels=channels,
			num_keypoints=num_keypoints,
			num_dims=num_dims,
			checkpoint_path=mamba_checkpoint_path,
			history_len=mamba_history_len,
			cuda_device=mamba_cuda_device,
		)
	if predictor_type == "baseline":
		return BaselineTruthHistoryPredictor(channels=channels)
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


def truth_acc_k_minus_1_from_positions(x_est, dt, k):
	"""Compute a_{k-1} from truth positions via finite differences."""
	channels = x_est.shape[1]
	if k < 2:
		return np.zeros(channels, dtype=np.float64)
	v_k_m05 = (x_est[k] - x_est[k - 1]) / dt
	v_k_m15 = (x_est[k - 1] - x_est[k - 2]) / dt
	return (v_k_m05 - v_k_m15) / dt


def truth_jerk_k_minus_1p5_from_positions(x_est, dt, k):
	"""Compute j_{k-1.5} from truth positions via finite differences."""
	channels = x_est.shape[1]
	if k < 3:
		return np.zeros(channels, dtype=np.float64)
	a_k_m1 = truth_acc_k_minus_1_from_positions(x_est, dt, k)
	a_k_m2 = truth_acc_k_minus_1_from_positions(x_est, dt, k - 1)
	return (a_k_m1 - a_k_m2) / dt


def build_realtime_hermite_segments(
	x_est,
	v_est,
	a_est,
	dt,
	velocity_mode="endpoint",
	enforce_position_continuity=True,
	baseline_anchor_with_truth=True,
):
	num_frames, channels = x_est.shape
	coeffs = np.empty((num_frames - 1, channels, 4), dtype=np.float64)
	pred_x_next = np.empty((num_frames - 1, channels), dtype=np.float64)
	pred_v_next = np.empty((num_frames - 1, channels), dtype=np.float64)
	prev_seg_right_x = None
	prev_seg_right_v = None
	prev_seg_right_a = None

	for k in range(num_frames - 1):
		# Strict 0-latency: when frame k is available, fit segment [k, k+1]
		# using current state at k and one-step predictor extrapolation.
		xk = x_est[k]
		vk = v_est[k]

		if velocity_mode == "endpoint":
			if a_est is None:
				x1_pred = xk + dt * vk
				v1_pred = vk
			else:
				ak = a_est[k]
				x1_pred = xk + vk * dt + 0.5 * ak * (dt ** 2)
				v1_pred = vk + ak * dt
		elif velocity_mode == "history_accel_extrapolation":
			if a_est is None:
				raise ValueError("a_est is required for history_accel_extrapolation mode")
			j_truth = truth_jerk_k_minus_1p5_from_positions(x_est, dt, k)
			a_truth = truth_acc_k_minus_1_from_positions(x_est, dt, k)

			if baseline_anchor_with_truth:
				# Re-anchor left endpoint to current truth-derived state to prevent drift accumulation.
				x_left = xk
				v_left = vk
				a_left = a_truth
			else:
				if (
					enforce_position_continuity
					and prev_seg_right_x is not None
					and prev_seg_right_v is not None
					and prev_seg_right_a is not None
				):
					x_left = prev_seg_right_x
					v_left = prev_seg_right_v
					a_left = prev_seg_right_a
				else:
					x_left = xk
					v_left = vk
					a_left = a_truth

			# Integrate truth-derived jerk over one frame to obtain predicted right endpoint.
			x_right = (
				x_left
				+ v_left * dt
				+ 0.5 * a_left * (dt ** 2)
				+ (1.0 / 6.0) * j_truth * (dt ** 3)
			)
			v_right = v_left + a_left * dt + 0.5 * j_truth * (dt ** 2)
			a_right = a_left + j_truth * dt
		else:
			raise ValueError(f"Unsupported velocity_mode: {velocity_mode}")

		if velocity_mode == "history_accel_extrapolation":
			pred_x_next[k] = x_right
			pred_v_next[k] = v_right
			coeffs[k] = fit_hermite_segment_from_endpoint_states(
				x_prev=x_left,
				x_curr=x_right,
				v_prev=v_left,
				v_curr=v_right,
				dt=dt,
			)
			if enforce_position_continuity:
				prev_seg_right_x = x_right.copy()
				prev_seg_right_v = v_right.copy()
				prev_seg_right_a = a_right.copy()
		else:
			if enforce_position_continuity:
				if prev_seg_right_x is None:
					x_left = xk
				else:
					x_left = prev_seg_right_x
				x_right = x1_pred
			else:
				x_left = xk
				x_right = x1_pred

			pred_x_next[k] = x_right
			pred_v_next[k] = v1_pred
			coeffs[k] = fit_hermite_segment_from_endpoint_states(
				x_prev=x_left,
				x_curr=x_right,
				v_prev=vk,
				v_curr=v1_pred,
				dt=dt,
			)

			if enforce_position_continuity:
				prev_seg_right_x = x_right.copy()

	return coeffs, pred_x_next, pred_v_next


def fit_realtime_segments_with_predictor(
	pose_data,
	fps,
	predictor,
	velocity_mode="endpoint",
	enforce_position_continuity=True,
	baseline_anchor_with_truth=True,
):
	if fps <= 0:
		raise ValueError(f"fps must be > 0, got {fps}")

	num_frames, num_keypoints, num_dims = pose_data.shape
	time_sec, dt, x_est, v_est, a_est = run_predictor_states(pose_data, fps, predictor)

	coeffs, pred_x_next, pred_v_next = build_realtime_hermite_segments(
		x_est,
		v_est,
		a_est,
		dt,
		velocity_mode=velocity_mode,
		enforce_position_continuity=enforce_position_continuity,
		baseline_anchor_with_truth=baseline_anchor_with_truth,
	)

	coeffs = coeffs.transpose(1, 2, 0).reshape(num_keypoints, num_dims, 4, num_frames - 1)
	x_est = x_est.reshape(num_frames, num_keypoints, num_dims)
	v_est = v_est.reshape(num_frames, num_keypoints, num_dims)
	pred_x_next = pred_x_next.reshape(num_frames - 1, num_keypoints, num_dims)
	pred_v_next = pred_v_next.reshape(num_frames - 1, num_keypoints, num_dims)

	bc_type = "hermite_zero_latency_predictive_c0" if enforce_position_continuity else "hermite_zero_latency_predictive"
	if velocity_mode == "history_accel_extrapolation" and baseline_anchor_with_truth:
		bc_type = "hermite_zero_latency_predictive_truth_anchor"

	result = {
		"time_sec": time_sec,
		"coeffs": coeffs,
		"x_est": x_est,
		"v_est": v_est,
		"pred_x_next": pred_x_next,
		"pred_v_next": pred_v_next,
		"fps": float(fps),
		"dt": float(dt),
		"bc_type": bc_type,
	}
	if velocity_mode == "history_accel_extrapolation":
		result["velocity_mode"] = "history_accel_extrapolation"
		result["baseline_anchor_with_truth"] = bool(baseline_anchor_with_truth)
	if a_est is not None:
		result["a_est"] = a_est.reshape(num_frames, num_keypoints, num_dims)

	return result


def save_result(save_path, source_shape, result, predictor_name):
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	predictor_label_map = {
		"kalman": "kalman_cv",
		"abg": "alpha_beta_gamma",
		"mamba": "mamba",
		"baseline": "truth_history_baseline",
	}
	predictor_label = predictor_label_map.get(predictor_name, predictor_name)
	np.savez_compressed(
		save_path,
		source_shape=np.asarray(source_shape, dtype=np.int32),
		predictor=np.asarray([predictor_label]),
		bc_type=np.asarray([result.get("bc_type", "hermite_endpoint_prediction")]),
		fps=np.asarray([result["fps"]], dtype=np.float64),
		dt=np.asarray([result["dt"]], dtype=np.float64),
		time_sec=result["time_sec"],
		coeffs=result["coeffs"],
		x_est=result["x_est"],
		v_est=result["v_est"],
		pred_x_next=result["pred_x_next"],
		pred_v_next=result["pred_v_next"],
		a_est=result["a_est"] if "a_est" in result else np.asarray([], dtype=np.float64),
		velocity_mode=np.asarray([result["velocity_mode"]], dtype="U64")
		if "velocity_mode" in result
		else np.asarray([], dtype="U64"),
		baseline_anchor_with_truth=np.asarray([result["baseline_anchor_with_truth"]], dtype=np.bool_)
		if "baseline_anchor_with_truth" in result
		else np.asarray([], dtype=np.bool_),
		alpha=np.asarray([result["alpha"]], dtype=np.float64) if "alpha" in result else np.asarray([], dtype=np.float64),
		beta=np.asarray([result["beta"]], dtype=np.float64) if "beta" in result else np.asarray([], dtype=np.float64),
		gamma=np.asarray([result["gamma"]], dtype=np.float64) if "gamma" in result else np.asarray([], dtype=np.float64),
		mamba_history_len=np.asarray([result["mamba_history_len"]], dtype=np.float64)
		if "mamba_history_len" in result
		else np.asarray([], dtype=np.float64),
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
	mamba_checkpoint_path="",
	mamba_history_len=8,
	mamba_cuda_device=-1,
):
	predictor_type = predictor_type.lower().strip()
	if predictor_type == "aby":
		predictor_type = "abg"
	if predictor_type not in {"kalman", "abg", "mamba", "baseline"}:
		raise ValueError(
			f"predictor_type must be one of ['kalman', 'abg', 'aby', 'mamba', 'baseline'], got: {predictor_type}"
		)

	files = collect_h36m_npy_files(input_dir)
	print(f"[{predictor_type}] Found {len(files)} file(s) in: {input_dir}")

	processed = 0
	failed = 0
	mamba_predictor = None

	for idx, name in enumerate(files, start=1):
		input_path = os.path.join(input_dir, name)
		stem = os.path.splitext(name)[0]
		suffix_map = {
			"kalman": "kalman_realtime_spline",
			"abg": "abg_realtime_spline",
			"mamba": "mamba_realtime_spline",
			"baseline": "baseline_realtime_spline",
		}
		suffix = suffix_map[predictor_type]
		output_path = os.path.join(output_dir, f"{stem}_{suffix}.npz")
		try:
			pose_data = load_pose_array(input_path)
			channels = pose_data.shape[1] * pose_data.shape[2]
			if predictor_type == "mamba":
				if mamba_predictor is None:
					mamba_predictor = create_predictor(
						predictor_type=predictor_type,
						channels=channels,
						num_keypoints=pose_data.shape[1],
						num_dims=pose_data.shape[2],
						mamba_checkpoint_path=mamba_checkpoint_path,
						mamba_history_len=mamba_history_len,
						mamba_cuda_device=mamba_cuda_device,
					)
				predictor = mamba_predictor
			else:
				predictor = create_predictor(
					predictor_type=predictor_type,
					channels=channels,
					num_keypoints=pose_data.shape[1],
					num_dims=pose_data.shape[2],
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
				velocity_mode="history_accel_extrapolation" if predictor_type == "baseline" else "endpoint",
				baseline_anchor_with_truth=(predictor_type == "baseline"),
			)
			if predictor_type == "abg":
				result["alpha"] = float(alpha)
				result["beta"] = float(beta)
				result["gamma"] = float(gamma)
			if predictor_type == "mamba":
				result["mamba_history_len"] = float(mamba_history_len)
			save_result(output_path, source_shape=pose_data.shape, result=result, predictor_name=predictor_type)
			print(f"[{idx}/{len(files)}] Saved: {output_path}")
			processed += 1
		except Exception as exc:
			print(f"[{idx}/{len(files)}] Failed: {input_path} -> {exc}")
			failed += 1

	print("=" * 60)
	print(f"[{predictor_type}] Done. processed={processed}, failed={failed}, total={len(files)}")


if __name__ == "__main__":
	# Predictor type: "kalman", "abg" (or alias "aby"), "mamba", "baseline".
	predictor_type = "baseline"
	# Input folder containing source pose files.
	input_dir = "/home/data/ztw/AtheletePose3D/h36m_pose_cam_1_downsample/test/S2_cam_1_30fps"
	# Output folder for saved spline files. Set to None to use auto default by predictor type.
	output_dir = None
	# Source frame rate in Hz used to build the time axis and dt.
	fps = 30.0

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

	# Mamba checkpoint for sequence prediction (required when predictor_type == "mamba").
	mamba_checkpoint_path = "/home/ztw/HVCCS/checkpoints/splines_mamba_runs/train_gpu0_e1000/ckpt/best.pt"
	# History window length used by mamba predictor.
	mamba_history_len = 8
	# CUDA device for mamba predictor. Set to -1 for CPU.
	mamba_cuda_device = 0

	if output_dir is None:
		if predictor_type == "kalman":
			output_dir = "/home/ztw/HVCCS/res/splines_fit_kalman_0latency3"
		elif predictor_type in {"abg", "aby"}:
			output_dir = "/home/ztw/HVCCS/res/splines_fit_abg_0latency3"
		elif predictor_type == "baseline":
			output_dir = "/home/ztw/HVCCS/res/splines_fit_baseline_0latency4"
		else:
			output_dir = "/home/ztw/HVCCS/res/splines_fit_mamba_0latency"

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
		mamba_checkpoint_path=mamba_checkpoint_path,
		mamba_history_len=mamba_history_len,
		mamba_cuda_device=mamba_cuda_device,
	)