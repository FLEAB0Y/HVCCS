import json
import os
import re
import threading
import time
from collections import deque

import numpy as np

from server import THStreamServiceServicer, serve
from realtime_offline_splines_fit import create_predictor, fit_realtime_segments_with_predictor


DEFAULT_CODEC_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "checkpoints", "grpc_offline_splines_codec_config.json")
)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_runtime_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"codec config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError("config root must be JSON object")

    for key in ("common", "sender", "receiver"):
        if key not in cfg or not isinstance(cfg[key], dict):
            raise ValueError(f"missing required config section: {key}")

    return cfg


def _is_within_root(path_value: str, root_dir: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(path_value), os.path.abspath(root_dir)]) == os.path.abspath(root_dir)
    except ValueError:
        return False


def resolve_runtime_path(path_value: str, project_root: str = PROJECT_ROOT) -> str:
    if os.path.isabs(path_value):
        return os.path.abspath(path_value)

    resolved = os.path.abspath(os.path.join(project_root, path_value))
    if not _is_within_root(resolved, project_root):
        raise ValueError(
            f"relative path escapes HVCCS root: {path_value}. "
            "Use an absolute path for resources outside HVCCS."
        )
    return resolved


def load_huffman_codebook(codebook_path: str):
    codebook_path = resolve_runtime_path(codebook_path)

    if not os.path.exists(codebook_path):
        raise FileNotFoundError(f"entropy codebook not found: {codebook_path}")

    with open(codebook_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    symbols = obj.get("symbols", [])
    if not isinstance(symbols, list) or not symbols:
        raise ValueError("invalid codebook: symbols must be a non-empty list")

    encode_map = {}
    for item in symbols:
        sym = int(item.get("symbol"))
        code = str(item.get("code", ""))
        if not code:
            continue
        encode_map[sym] = code

    if not encode_map:
        raise ValueError("invalid codebook: no valid symbol->code entries")

    meta = obj.get("meta", {})
    quant_bits = int(meta.get("quant_bits", 8))
    clip_abs = float(meta.get("clip_abs", 1.0))
    levels = 1 << quant_bits
    for sym in encode_map.keys():
        if sym < 0 or sym >= levels:
            raise ValueError(f"symbol out of quant_bits range: sym={sym}, quant_bits={quant_bits}")

    return {
        "path": codebook_path,
        "encode_map": encode_map,
        "quant_bits": quant_bits,
        "clip_abs": clip_abs,
    }


def resolve_codebook_path_by_bits(codebook_path: str, quant_bits: int) -> str:
    codebook_path = resolve_runtime_path(codebook_path)

    root, ext = os.path.splitext(codebook_path)
    ext = ext if ext else ".json"

    candidates = []
    if "{quant_bits}" in codebook_path:
        candidates.append(codebook_path.format(quant_bits=int(quant_bits)))

    candidates.append(codebook_path)

    replaced_root = re.sub(r"_q\d+$", f"_q{int(quant_bits)}", root)
    candidates.append(replaced_root + ext)

    if not root.endswith(f"_q{int(quant_bits)}"):
        candidates.append(f"{root}_q{int(quant_bits)}{ext}")

    uniq_candidates = []
    for p in candidates:
        if p not in uniq_candidates:
            uniq_candidates.append(p)

    for p in uniq_candidates:
        if os.path.exists(p):
            return p

    tried = "\n".join(uniq_candidates)
    raise FileNotFoundError(
        f"entropy codebook not found for quant_bits={quant_bits}. tried:\n{tried}"
    )


def build_huffman_decode_tree(encode_map: dict) -> dict:
    root = {}
    for sym, code in encode_map.items():
        node = root
        for bit in code:
            node = node.setdefault(bit, {})
        if "sym" in node:
            raise ValueError("invalid codebook: duplicated huffman code")
        node["sym"] = int(sym)
    return root


def dequantize_uniform(q_values: np.ndarray, quant_bits: int, clip_abs: float) -> np.ndarray:
    levels = 1 << quant_bits
    scale = (2.0 * clip_abs) / (levels - 1)
    return q_values.astype(np.float32) * scale - clip_abs


class ReceiveStats:
    def __init__(self, max_points: int = 300):
        self.max_points = max_points
        self.latency_data = []
        self.bandwidth_data = []
        self.total_packets = 0
        self.total_bytes = 0
        self.bytes_buffer = deque(maxlen=10)
        self.last_update = time.time()

    def add_latency(self, latency_ms: float):
        self.latency_data.append(latency_ms)
        if len(self.latency_data) > self.max_points:
            self.latency_data.pop(0)

    def add_packet(self, data_size: int):
        self.total_packets += 1
        self.total_bytes += data_size

        current_time = time.time()
        self.bytes_buffer.append(data_size)

        # 参考 grpc2socket.py：每100ms更新一次带宽，窗口为最近10个100ms数据
        if current_time - self.last_update >= 0.1:
            current_bandwidth = sum(self.bytes_buffer)
            self.bandwidth_data.append(current_bandwidth)
            if len(self.bandwidth_data) > self.max_points:
                self.bandwidth_data.pop(0)
            self.last_update = current_time

    def summary(self) -> str:
        current_latency = self.latency_data[-1] if self.latency_data else 0.0
        avg_latency = sum(self.latency_data) / len(self.latency_data) if self.latency_data else 0.0
        current_bw = self.bandwidth_data[-1] if self.bandwidth_data else 0.0
        avg_bw = sum(self.bandwidth_data) / len(self.bandwidth_data) if self.bandwidth_data else 0.0

        return (
            f"延迟: 当前 {current_latency:.2f} ms | 平均 {avg_latency:.2f} ms | "
            f"带宽: 当前 {current_bw / 1024:.2f} KB/s | 平均 {avg_bw / 1024:.2f} KB/s | "
            f"总包数: {self.total_packets} | 总流量: {self.total_bytes / 1024:.2f} KB"
        )


def extract_timestamp_ms(ext_desc: str):
    """优先解析纯时间戳；兼容 JSON 元数据中的 timestamp_ms/t1_ms。"""
    if not ext_desc:
        return None

    desc = ext_desc.strip()
    if not desc:
        return None

    if desc.isdigit():
        return int(desc)

    try:
        meta = json.loads(desc)
        if isinstance(meta, dict):
            if "timestamp_ms" in meta:
                return int(meta["timestamp_ms"])
            if "t1_ms" in meta:
                return int(meta["t1_ms"])
    except Exception:
        return None

    return None


def decode_payload(
    payload_bytes: bytes,
    payload_dtype: str,
    entropy_codec: str,
    quant_scale: float,
    quant_bits: int,
    clip_abs: float,
    entropy_bit_length: int,
    huffman_decode_tree: dict | None,
    expected_dims: int,
):
    if payload_dtype == "qint16":
        if quant_scale <= 0:
            raise ValueError(f"quant_scale must be > 0, got {quant_scale}")
        values = np.frombuffer(payload_bytes, dtype=np.int16).astype(np.float32) / quant_scale
    elif payload_dtype == "qidx_huff":
        if entropy_codec != "huffman":
            raise ValueError(f"qidx_huff requires entropy_codec='huffman', got {entropy_codec}")
        if huffman_decode_tree is None:
            raise ValueError("huffman decode tree is required for qidx_huff")
        if entropy_bit_length < 0:
            raise ValueError(f"invalid entropy_bit_length: {entropy_bit_length}")

        bit_str = "".join(format(b, "08b") for b in payload_bytes)
        if entropy_bit_length > len(bit_str):
            raise ValueError("entropy_bit_length exceeds payload bits")
        bit_str = bit_str[:entropy_bit_length]

        decoded_symbols = []
        node = huffman_decode_tree
        for bit in bit_str:
            if bit not in node:
                raise ValueError("invalid huffman stream")
            node = node[bit]
            if "sym" in node:
                decoded_symbols.append(int(node["sym"]))
                node = huffman_decode_tree

        if len(decoded_symbols) != expected_dims:
            raise ValueError(
                f"decoded symbol dims mismatch: got {len(decoded_symbols)}, expected {expected_dims}"
            )

        values = dequantize_uniform(
            np.asarray(decoded_symbols, dtype=np.int32),
            quant_bits=quant_bits,
            clip_abs=clip_abs,
        )
    elif payload_dtype == "float32":
        values = np.frombuffer(payload_bytes, dtype=np.float32)
    else:
        raise ValueError(f"unsupported payload dtype: {payload_dtype}")

    return values.astype(np.float32, copy=False)


def parse_meta(ext_desc: str):
    if not ext_desc:
        return {}
    text = ext_desc.strip()
    if not text:
        return {}
    if text.startswith("{") and text.endswith("}"):
        try:
            meta = json.loads(text)
            if isinstance(meta, dict):
                return meta
        except Exception:
            return {}
    return {}


def get_payload_bytes(payload, camel_attr: str, snake_attr: str) -> bytes:
    value = getattr(payload, camel_attr, None)
    if value is None:
        value = getattr(payload, snake_attr, b"")
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    return b""


def get_payload_str(payload, camel_attr: str, snake_attr: str) -> str:
    value = getattr(payload, camel_attr, None)
    if value is None:
        value = getattr(payload, snake_attr, "")
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return ""
    return str(value)


def get_payload_seq(payload) -> str:
    seq = getattr(payload, "seqNo", None)
    if seq is None:
        seq = getattr(payload, "seq_no", "")
    return str(seq)


def _normalize_predictor_type(predictor_type: str) -> str:
    name = str(predictor_type).lower().strip()
    if name == "aby":
        name = "abg"
    if name not in {"kalman", "abg", "mamba", "baseline"}:
        raise ValueError(
            f"receiver.spline_predictor_type must be one of ['kalman', 'abg', 'aby', 'mamba', 'baseline'], got: {predictor_type}"
        )
    return name


def _build_channel_spline_path(save_dir: str, save_file: str, channel: int, multi_channel: bool) -> str:
    root, ext = os.path.splitext(save_file)
    if not ext:
        ext = ".npz"
    if multi_channel:
        return os.path.join(save_dir, f"{root}_ch{channel}{ext}")
    return os.path.join(save_dir, f"{root}{ext}")


def save_spline_result(save_path: str, result: dict, predictor_name: str, source_shape, channel: int):
    predictor_label_map = {
        "kalman": "kalman_cv",
        "abg": "alpha_beta_gamma",
        "mamba": "mamba",
        "baseline": "truth_history_baseline",
    }
    predictor_label = predictor_label_map.get(predictor_name, predictor_name)

    np.savez_compressed(
        save_path,
        channel=np.asarray([channel], dtype=np.int32),
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
        alpha=np.asarray([result["alpha"]], dtype=np.float64) if "alpha" in result else np.asarray([], dtype=np.float64),
        beta=np.asarray([result["beta"]], dtype=np.float64) if "beta" in result else np.asarray([], dtype=np.float64),
        gamma=np.asarray([result["gamma"]], dtype=np.float64) if "gamma" in result else np.asarray([], dtype=np.float64),
        mamba_history_len=np.asarray([result["mamba_history_len"]], dtype=np.float64)
        if "mamba_history_len" in result
        else np.asarray([], dtype=np.float64),
    )


def _get_receiver_param(receiver_cfg: dict, spline_key: str, base_key: str, default_value):
    # Prefer receiver.spline_xxx, fallback to receiver.xxx for compatibility with splines_fitter main names.
    if spline_key in receiver_cfg:
        return receiver_cfg.get(spline_key)
    if base_key in receiver_cfg:
        return receiver_cfg.get(base_key)
    return default_value


def main():
    cfg = load_runtime_config(DEFAULT_CODEC_CONFIG_PATH)
    common_cfg = cfg["common"]
    receiver_cfg = cfg["receiver"]

    grpc_port = int(receiver_cfg["grpc_port"])
    report_interval = float(receiver_cfg.get("report_interval", 1.0))
    poll_interval = float(receiver_cfg.get("poll_interval", 0.01))
    idle_timeout_sec = float(receiver_cfg.get("idle_timeout_sec", 5.0))
    debug = bool(receiver_cfg.get("debug", False))
    save_max_frames = int(receiver_cfg.get("save_max_frames", 0))
    spline_fit_enabled = bool(receiver_cfg.get("spline_fit_enabled", True))

    if report_interval <= 0:
        raise ValueError("receiver.report_interval must be > 0")
    if poll_interval <= 0:
        raise ValueError("receiver.poll_interval must be > 0")
    if idle_timeout_sec < 0:
        raise ValueError("receiver.idle_timeout_sec must be >= 0")

    save_dir = str(receiver_cfg.get("save_dir", "res/decode_res"))
    save_dir = resolve_runtime_path(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    spline_save_file = str(
        _get_receiver_param(receiver_cfg, "spline_save_file", "output_file", "decoded_all_channels_spline_fit.npz")
    )

    spline_predictor_type = _normalize_predictor_type(
        _get_receiver_param(receiver_cfg, "spline_predictor_type", "predictor_type", "baseline")
    )
    spline_fps = float(
        _get_receiver_param(receiver_cfg, "spline_fps", "fps", cfg.get("sender", {}).get("fps", 30.0))
    )
    spline_process_acc_var = float(
        _get_receiver_param(receiver_cfg, "spline_process_acc_var", "process_acc_var", 3e5)
    )
    spline_measurement_var = float(
        _get_receiver_param(receiver_cfg, "spline_measurement_var", "measurement_var", 9.0)
    )
    spline_init_pos_var = float(
        _get_receiver_param(receiver_cfg, "spline_init_pos_var", "init_pos_var", 1.0)
    )
    spline_init_vel_var = float(
        _get_receiver_param(receiver_cfg, "spline_init_vel_var", "init_vel_var", 1e4)
    )
    spline_alpha = float(_get_receiver_param(receiver_cfg, "spline_alpha", "alpha", 0.65))
    spline_beta = float(_get_receiver_param(receiver_cfg, "spline_beta", "beta", 0.08))
    spline_gamma = float(_get_receiver_param(receiver_cfg, "spline_gamma", "gamma", 0.005))
    spline_mamba_checkpoint_path = str(
        _get_receiver_param(receiver_cfg, "spline_mamba_checkpoint_path", "mamba_checkpoint_path", "")
    ).strip()
    spline_mamba_history_len = int(
        _get_receiver_param(receiver_cfg, "spline_mamba_history_len", "mamba_history_len", 8)
    )
    spline_mamba_cuda_device = int(
        _get_receiver_param(receiver_cfg, "spline_mamba_cuda_device", "mamba_cuda_device", -1)
    )

    if spline_mamba_checkpoint_path:
        spline_mamba_checkpoint_path = resolve_runtime_path(spline_mamba_checkpoint_path)

    if spline_fps <= 0:
        raise ValueError("receiver.spline_fps must be > 0")

    default_num_keypoints = int(common_cfg.get("num_keypoints", 17))
    default_coord_dims = int(common_cfg.get("coord_dims", 3))
    default_quant_scale = float(common_cfg.get("quant_scale", 1000.0))
    default_quant_bits = int(common_cfg.get("quant_bits", 8))
    default_clip_abs = float(common_cfg.get("clip_abs", 0.0))
    entropy_enabled = bool(common_cfg.get("entropy_enabled", True))
    entropy_codec = str(common_cfg.get("entropy_codec", "huffman" if entropy_enabled else "none"))
    packet_tag = str(common_cfg.get("packet_tag", "POSE_RES_V1"))

    huffman_decode_tree = None
    if entropy_enabled and entropy_codec == "huffman":
        codebook_path_base = str(
            common_cfg.get(
                "entropy_codebook_path",
                "checkpoints/grpc_online_splines_entropy_codebook.json",
            )
        )
        codebook_path = resolve_codebook_path_by_bits(codebook_path_base, quant_bits=default_quant_bits)
        codebook = load_huffman_codebook(codebook_path)
        if "quant_bits" in common_cfg and int(common_cfg["quant_bits"]) != int(codebook["quant_bits"]):
            raise ValueError(
                f"quant_bits mismatch: config={int(common_cfg['quant_bits'])}, "
                f"codebook={int(codebook['quant_bits'])}, path={codebook['path']}"
            )
        if default_clip_abs <= 0:
            default_clip_abs = float(codebook["clip_abs"])
        if "quant_bits" not in common_cfg:
            default_quant_bits = int(codebook["quant_bits"])
        huffman_decode_tree = build_huffman_decode_tree(codebook["encode_map"])

    if entropy_enabled and entropy_codec not in ("huffman",):
        raise ValueError(f"unsupported entropy_codec: {entropy_codec}; zlib has been removed")

    servicer = THStreamServiceServicer()
    server_thread = threading.Thread(target=serve, args=(servicer, grpc_port), daemon=True)
    server_thread.start()

    stats = ReceiveStats()
    last_report_time = time.time()
    prev_recon_by_channel = {}
    saved_frames_by_channel = {}
    received_pose_frame_count = 0
    has_received_payload = False
    last_payload_time = time.time()

    print(
        f"接收端已启动，监听 gRPC 端口: {grpc_port}, "
        "仅保存样条拟合结果"
    )
    if idle_timeout_sec > 0:
        print(f"空闲自动停止已启用: idle_timeout_sec={idle_timeout_sec:.1f}s（首次收到数据后生效）")
    else:
        print("空闲自动停止已关闭: idle_timeout_sec<=0")
    if spline_fit_enabled:
        print(
            f"样条拟合已启用: predictor={spline_predictor_type}, fps={spline_fps}, "
            f"save_file={spline_save_file}"
        )

    try:
        while True:
            if servicer.receive_data_buffer.get_size() < 1:
                time.sleep(poll_interval)
                now = time.time()
                if (
                    idle_timeout_sec > 0
                    and has_received_payload
                    and (now - last_payload_time) >= idle_timeout_sec
                ):
                    print(
                        f"\n超过 {idle_timeout_sec:.1f}s 未接收到新数据，自动停止接收并保存结果"
                    )
                    break
            else:
                payload = servicer.receive_data_buffer.get_items()
                has_received_payload = True
                last_payload_time = time.time()

                rgb_data = get_payload_bytes(payload, "rgbData", "rgb_data")
                point_data = get_payload_bytes(payload, "pointData", "point_data")
                face_data = get_payload_bytes(payload, "faceData", "face_data")
                limb_data = get_payload_bytes(payload, "limbData", "limb_data")
                ext_data = get_payload_bytes(payload, "extData", "ext_data")
                ext_desc = get_payload_str(payload, "extDesc", "ext_desc")
                seq_no = get_payload_seq(payload)

                data_size = (
                    len(rgb_data)
                    + len(point_data)
                    + len(face_data)
                    + len(limb_data)
                    + len(ext_data)
                )
                stats.add_packet(data_size)

                meta = parse_meta(ext_desc)
                channel = int(meta.get("channel", 0))
                payload_dtype = str(meta.get("payload_dtype", "float32"))
                entropy_codec = str(meta.get("entropy_codec", "none"))
                quant_scale = float(meta.get("quant_scale", default_quant_scale))
                quant_bits = int(meta.get("quant_bits", default_quant_bits))
                clip_abs = float(meta.get("clip_abs", default_clip_abs))
                entropy_bit_length = int(meta.get("entropy_bit_length", 0))
                num_keypoints = int(meta.get("num_keypoints", default_num_keypoints))
                coord_dims = int(meta.get("coord_dims", default_coord_dims))
                packet_kind = str(meta.get("kind", "I"))
                tag = str(meta.get("tag", ""))
                t0_ms = int(meta.get("t0_ms", 0)) if str(meta.get("t0_ms", "")).isdigit() else 0

                try:
                    decoded_values = decode_payload(
                        limb_data,
                        payload_dtype=payload_dtype,
                        entropy_codec=entropy_codec,
                        quant_scale=quant_scale,
                        quant_bits=quant_bits,
                        clip_abs=clip_abs,
                        entropy_bit_length=entropy_bit_length,
                        huffman_decode_tree=huffman_decode_tree,
                        expected_dims=num_keypoints * coord_dims,
                    )

                    expected_dims = num_keypoints * coord_dims
                    if decoded_values.size != expected_dims:
                        raise ValueError(
                            f"decoded dims mismatch: got {decoded_values.size}, expected {expected_dims}"
                        )

                    prev_recon = prev_recon_by_channel.get(channel)
                    if packet_kind == "P" and prev_recon is not None:
                        recon_flat = (prev_recon + decoded_values).astype(np.float32, copy=False)
                    else:
                        recon_flat = decoded_values.astype(np.float32, copy=False)
                    prev_recon_by_channel[channel] = recon_flat

                    if spline_fit_enabled:
                        pose_frame = recon_flat.reshape(num_keypoints, coord_dims).copy()
                        received_pose_frame_count += 1
                        if channel not in saved_frames_by_channel:
                            saved_frames_by_channel[channel] = []
                        saved_frames_by_channel[channel].append(pose_frame)
                        if save_max_frames > 0 and received_pose_frame_count >= save_max_frames:
                            break

                    # 解码结束后记录 t1，统计 t1-t0，覆盖编码/量化/解码路径时延。
                    t1_ms = int(time.time() * 1000)
                    if t0_ms > 0:
                        stats.add_latency(float(t1_ms - t0_ms))
                    else:
                        # 兼容旧格式：当没有 t0_ms 时回退到已有时间戳字段。
                        sent_ts = extract_timestamp_ms(ext_desc)
                        if sent_ts is not None:
                            stats.add_latency(float(t1_ms - sent_ts))
                except Exception as decode_exc:
                    if debug:
                        print(f"decode error: {decode_exc}")

                if debug:
                    print(
                        f"recv seq={seq_no}, bytes={data_size}, "
                        f"tag={tag}, kind={packet_kind}, channel={channel}, extDesc={ext_desc[:80]}"
                    )

            now = time.time()
            if now - last_report_time >= report_interval:
                print(stats.summary())
                last_report_time = now

    except KeyboardInterrupt:
        print("\n接收被用户中断")

    if spline_fit_enabled:
        if not saved_frames_by_channel:
            print("未执行样条拟合：没有可用的解码帧")
        else:
            channels = sorted(saved_frames_by_channel.keys())
            multi_channel = len(channels) > 1
            for channel in channels:
                pose_frames = saved_frames_by_channel[channel]
                if len(pose_frames) < 2:
                    print(f"跳过 channel={channel} 的样条拟合：帧数不足 2，实际={len(pose_frames)}")
                    continue

                try:
                    pose_data = np.asarray(pose_frames, dtype=np.float64)
                    channels_count = pose_data.shape[1] * pose_data.shape[2]
                    predictor = create_predictor(
                        predictor_type=spline_predictor_type,
                        channels=channels_count,
                        num_keypoints=pose_data.shape[1],
                        num_dims=pose_data.shape[2],
                        process_acc_var=spline_process_acc_var,
                        measurement_var=spline_measurement_var,
                        init_pos_var=spline_init_pos_var,
                        init_vel_var=spline_init_vel_var,
                        alpha=spline_alpha,
                        beta=spline_beta,
                        gamma=spline_gamma,
                        mamba_checkpoint_path=spline_mamba_checkpoint_path,
                        mamba_history_len=spline_mamba_history_len,
                        mamba_cuda_device=spline_mamba_cuda_device,
                    )

                    result = fit_realtime_segments_with_predictor(
                        pose_data=pose_data,
                        fps=spline_fps,
                        predictor=predictor,
                        velocity_mode="history_accel_extrapolation" if spline_predictor_type == "baseline" else "endpoint",
                    )

                    if spline_predictor_type == "abg":
                        result["alpha"] = float(spline_alpha)
                        result["beta"] = float(spline_beta)
                        result["gamma"] = float(spline_gamma)
                    if spline_predictor_type == "mamba":
                        result["mamba_history_len"] = float(spline_mamba_history_len)

                    spline_save_path = _build_channel_spline_path(
                        save_dir=save_dir,
                        save_file=spline_save_file,
                        channel=channel,
                        multi_channel=multi_channel,
                    )
                    save_spline_result(
                        save_path=spline_save_path,
                        result=result,
                        predictor_name=spline_predictor_type,
                        source_shape=pose_data.shape,
                        channel=channel,
                    )
                    print(
                        f"样条拟合结果已保存: {spline_save_path}, "
                        f"channel={channel}, coeffs_shape={result['coeffs'].shape}"
                    )
                except Exception as fit_exc:
                    print(f"channel={channel} 样条拟合失败: {fit_exc}")
    else:
        print("未执行样条拟合：receiver.spline_fit_enabled=false")

    print("最终统计:")
    print(stats.summary())


if __name__ == "__main__":
    main()

