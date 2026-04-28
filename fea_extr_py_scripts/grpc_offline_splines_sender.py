import json
import os
import re
import threading
import time

import numpy as np

from client import THStreamClient
from THStreamData import THStreamDataPayload


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


def quantize_uniform(values: np.ndarray, quant_bits: int, clip_abs: float) -> np.ndarray:
    levels = 1 << quant_bits
    clipped = np.clip(values, -clip_abs, clip_abs)
    scale = (levels - 1) / (2.0 * clip_abs)
    q = np.rint((clipped + clip_abs) * scale)
    return q.astype(np.int32)


def dequantize_uniform(q_values: np.ndarray, quant_bits: int, clip_abs: float) -> np.ndarray:
    levels = 1 << quant_bits
    scale = (2.0 * clip_abs) / (levels - 1)
    return q_values.astype(np.float32) * scale - clip_abs


def huffman_encode_symbols(symbols: np.ndarray, encode_map: dict) -> tuple[bytes, int]:
    bits = []
    for value in symbols.tolist():
        sym = int(value)
        code = encode_map.get(sym)
        if code is None:
            raise ValueError(f"symbol {sym} missing from entropy codebook")
        bits.append(code)

    bitstream = "".join(bits)
    bit_length = len(bitstream)
    if bit_length == 0:
        return b"", 0

    pad_bits = (8 - (bit_length % 8)) % 8
    if pad_bits > 0:
        bitstream += "0" * pad_bits

    out = bytearray()
    for i in range(0, len(bitstream), 8):
        out.append(int(bitstream[i:i + 8], 2))

    return bytes(out), bit_length


def to_tjc_from_npy(pose: np.ndarray, coord_dims: int = 3) -> np.ndarray:
    """Normalize common npy layouts into shape (T, J, C)."""
    arr = np.asarray(pose)
    arr = np.squeeze(arr)

    if arr.ndim == 3:
        if arr.shape[-1] >= coord_dims:
            return arr[..., :coord_dims]
        if arr.shape[1] >= coord_dims:
            return np.transpose(arr, (0, 2, 1))[..., :coord_dims]
        raise ValueError(f"cannot infer npy pose layout from shape {arr.shape}")

    if arr.ndim == 2:
        if arr.shape[1] % coord_dims == 0:
            joints = arr.shape[1] // coord_dims
            return arr.reshape(arr.shape[0], joints, coord_dims)
        if arr.shape[0] % coord_dims == 0:
            joints = arr.shape[0] // coord_dims
            return arr.T.reshape(arr.shape[1], joints, coord_dims)
        raise ValueError(f"cannot infer npy pose layout from shape {arr.shape}")

    if arr.ndim == 1:
        if arr.shape[0] % coord_dims != 0:
            raise ValueError(f"cannot infer npy pose layout from shape {arr.shape}")
        joints = arr.shape[0] // coord_dims
        return arr.reshape(1, joints, coord_dims)

    raise ValueError(f"unsupported npy pose shape: {arr.shape}")


def parse_pose_from_line(
    line: str,
    num_keypoints: int = 17,
    coord_dims: int = 3,
    start_col: int = 0,
):
    """Parse one text line into one pose frame with shape (17, 3)."""
    tokens = [token.strip() for token in line.replace(",", " ").split()]
    expected_values = num_keypoints * coord_dims
    end_col = start_col + expected_values
    if len(tokens) < end_col:
        return None

    try:
        flat_values = np.array(
            list(map(float, tokens[start_col:end_col])), dtype=np.float32
        )
    except ValueError:
        return None

    if flat_values.shape[0] != expected_values:
        return None

    return flat_values.reshape(num_keypoints, coord_dims)


def iter_h36m_frames(
    feature_file: str,
    num_keypoints: int = 17,
    coord_dims: int = 3,
    start_col: int = 0,
):
    """Iterate pose frames from .npy or text file."""
    suffix = os.path.splitext(feature_file)[1].lower()
    if suffix == ".npy":
        pose_array = np.load(feature_file)
        pose_tjc = to_tjc_from_npy(pose_array, coord_dims=coord_dims)
        if pose_tjc.shape[1] != num_keypoints:
            raise ValueError(
                f"npy joint count mismatch: file has {pose_tjc.shape[1]} joints, "
                f"expected {num_keypoints}"
            )
        for index in range(pose_tjc.shape[0]):
            yield pose_tjc[index].astype(np.float32, copy=False)
        return

    with open(feature_file, "r", encoding="utf-8") as feature_fp:
        for line in feature_fp:
            pose_frame = parse_pose_from_line(
                line,
                num_keypoints=num_keypoints,
                coord_dims=coord_dims,
                start_col=start_col,
            )
            if pose_frame is not None:
                yield pose_frame


class GRPCPoseSender:
    def __init__(self, host: str, port: int, buffer_limit: int = 80):
        self.client = THStreamClient(host=host, port=port)
        self._stop_event = threading.Event()
        self._thread = None
        self.buffer_limit = max(int(buffer_limit), 1)

    def _send_loop(self):
        while not self._stop_event.is_set():
            if self.client.send_data_buffer.get_size() > 0:
                self.client.send_data()
            else:
                time.sleep(0.001)

    def start(self):
        self._thread = threading.Thread(target=self._send_loop, daemon=True)
        self._thread.start()

    def send_encoded_packet(self, encoded_payload: bytes, meta: dict):
        ext_desc = json.dumps(meta, separators=(",", ":"), ensure_ascii=False)

        payload = THStreamDataPayload(
            rgb_data=b"\x00",
            point_data=b"\x00",
            face_data=b"\x00",
            limb_data=encoded_payload,
            ext_data=b"\x00",
            ext_desc=ext_desc,
        )

        while self.client.send_data_buffer.get_size() >= self.buffer_limit:
            time.sleep(0.002)
        self.client.send_data_buffer.add_item(payload)

    def shutdown(self, drain_timeout: float = 2.0):
        deadline = time.time() + max(drain_timeout, 0.0)
        while self.client.send_data_buffer.get_size() > 0 and time.time() < deadline:
            time.sleep(0.005)

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        try:
            self.client.channel.close()
        except Exception:
            pass


def encode_payload(
    values: np.ndarray,
    quantize: bool,
    quant_scale: float,
    entropy_enabled: bool,
    entropy_codec: str,
    quant_bits: int,
    clip_abs: float,
    huffman_encode_map: dict | None,
):
    bit_length = 0
    if quantize:
        if entropy_enabled and entropy_codec == "huffman":
            if huffman_encode_map is None:
                raise ValueError("huffman codebook is required when entropy_codec='huffman'")
            q_values = quantize_uniform(values, quant_bits=quant_bits, clip_abs=clip_abs)
            payload, bit_length = huffman_encode_symbols(q_values, huffman_encode_map)
            payload_dtype = "qidx_huff"
        else:
            q_values = np.round(values * quant_scale)
            q_values = np.clip(q_values, -32768, 32767).astype(np.int16)
            payload = q_values.tobytes()
            payload_dtype = "qint16"
    else:
        payload = values.astype(np.float32).tobytes()
        payload_dtype = "float32"

    if not entropy_enabled:
        entropy_codec = "none"

    return payload, payload_dtype, entropy_codec, bit_length


def decode_payload_local(
    payload: bytes,
    payload_dtype: str,
    entropy_codec: str,
    quant_scale: float,
    quant_bits: int,
    clip_abs: float,
    bit_length: int,
    huffman_decode_tree: dict | None,
    expected_dims: int,
) -> np.ndarray:
    if payload_dtype == "qint16":
        if quant_scale <= 0:
            raise ValueError(f"quant_scale must be > 0, got {quant_scale}")
        arr = np.frombuffer(payload, dtype=np.int16).astype(np.float32) / quant_scale
    elif payload_dtype == "qidx_huff":
        if entropy_codec != "huffman":
            raise ValueError(f"qidx_huff requires entropy_codec='huffman', got {entropy_codec}")
        if huffman_decode_tree is None:
            raise ValueError("huffman decode tree is required for qidx_huff")
        if bit_length < 0:
            raise ValueError(f"invalid bit_length: {bit_length}")

        bit_str = "".join(format(b, "08b") for b in payload)
        if bit_length > len(bit_str):
            raise ValueError("bit_length exceeds payload bits")
        bit_str = bit_str[:bit_length]

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
        arr = dequantize_uniform(
            np.asarray(decoded_symbols, dtype=np.int32),
            quant_bits=quant_bits,
            clip_abs=clip_abs,
        )
    elif payload_dtype == "float32":
        arr = np.frombuffer(payload, dtype=np.float32)
    else:
        raise ValueError(f"unsupported payload dtype: {payload_dtype}")
    return arr.astype(np.float32, copy=False)


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


def stream_pose_frames(
    feature_file: str,
    sender: GRPCPoseSender,
    fps: float,
    max_frames: int,
    start_col: int,
    i_frame_interval: int,
    quantize_i_frame: bool,
    quantize_p_frame: bool,
    quant_scale: float,
    entropy_enabled: bool,
    entropy_codec: str,
    quant_bits: int,
    clip_abs: float,
    huffman_encode_map: dict | None,
    huffman_decode_tree: dict | None,
    packet_tag: str,
    channel: int,
    num_keypoints: int,
    coord_dims: int,
    debug: bool,
):
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    interval = 1.0 / fps
    sent_count = 0
    prev_recon: np.ndarray | None = None
    next_input_time = time.perf_counter()

    for frame_idx, pose_frame in enumerate(
        iter_h36m_frames(
            feature_file,
            num_keypoints=num_keypoints,
            coord_dims=coord_dims,
            start_col=start_col,
        )
    ):
        # 在进入残差编码前进行节拍控制，确保数据输入残差编码模块的速率受 fps 控制。
        now_perf = time.perf_counter()
        if now_perf < next_input_time:
            time.sleep(next_input_time - now_perf)
        next_input_time += interval

        t0_ms = int(time.time() * 1000)

        current_flat = pose_frame.reshape(-1).astype(np.float32, copy=False)
        is_i_frame = prev_recon is None or (i_frame_interval > 0 and sent_count % i_frame_interval == 0)

        if is_i_frame:
            packet_kind = "I"
            payload_values = current_flat
            use_quantize = quantize_i_frame
        else:
            packet_kind = "P"
            if prev_recon is None:
                payload_values = current_flat
                packet_kind = "I"
                use_quantize = quantize_i_frame
            else:
                payload_values = (current_flat - prev_recon).astype(np.float32, copy=False)
                use_quantize = quantize_p_frame

        expected_dims = num_keypoints * coord_dims
        encoded_payload, payload_dtype, entropy_codec, bit_length = encode_payload(
            payload_values,
            quantize=use_quantize,
            quant_scale=quant_scale,
            entropy_enabled=entropy_enabled,
            entropy_codec=entropy_codec,
            quant_bits=quant_bits,
            clip_abs=clip_abs,
            huffman_encode_map=huffman_encode_map,
        )

        # 使用本地解码结果更新历史重建帧，保证与接收端重建基准一致。
        decoded_values = decode_payload_local(
            encoded_payload,
            payload_dtype=payload_dtype,
            entropy_codec=entropy_codec,
            quant_scale=quant_scale,
            quant_bits=quant_bits,
            clip_abs=clip_abs,
            bit_length=bit_length,
            huffman_decode_tree=huffman_decode_tree,
            expected_dims=expected_dims,
        )
        if packet_kind == "P" and prev_recon is not None:
            recon_flat = (prev_recon + decoded_values).astype(np.float32, copy=False)
        else:
            recon_flat = decoded_values.astype(np.float32, copy=False)

        timestamp_ms = int(time.time() * 1000)

        meta = {
            "tag": packet_tag,
            "kind": packet_kind,
            "frame_idx": int(frame_idx),
            "timestamp_ms": int(timestamp_ms),
            "t0_ms": int(t0_ms),
            "channel": int(channel),
            "num_keypoints": int(num_keypoints),
            "coord_dims": int(coord_dims),
            "pose_dims": int(num_keypoints * coord_dims),
            "payload_dtype": payload_dtype,
            "entropy_codec": entropy_codec,
            "entropy_bit_length": int(bit_length),
            "quantize": int(use_quantize),
            "quantize_i_frame": int(quantize_i_frame),
            "quantize_p_frame": int(quantize_p_frame),
            "quant_scale": float(quant_scale),
            "quant_bits": int(quant_bits),
            "clip_abs": float(clip_abs),
        }
        sender.send_encoded_packet(encoded_payload, meta)
        sent_count += 1
        prev_recon = recon_flat

        if debug:
            print(
                f"sent frame={frame_idx}, kind={packet_kind}, channel={channel}, "
                f"t0_ms={t0_ms}, timestamp_ms={timestamp_ms}, bytes={len(encoded_payload)}"
            )

        if max_frames > 0 and sent_count >= max_frames:
            break

    return sent_count


def main():
    cfg = load_runtime_config(DEFAULT_CODEC_CONFIG_PATH)
    common_cfg = cfg["common"]
    sender_cfg = cfg["sender"]

    feature_file = str(sender_cfg["feature_file"])
    feature_file = resolve_runtime_path(feature_file)

    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"feature_file not found: {feature_file}")

    channel = int(sender_cfg.get("channel", 0))
    if channel < 0 or channel > 50:
        raise ValueError(f"sender.channel must be in [0, 50], got {channel}")

    sender = GRPCPoseSender(
        host=str(sender_cfg["server_addr"]),
        port=int(sender_cfg["port_num"]),
        buffer_limit=int(sender_cfg.get("buffer_limit", 80)),
    )
    sender.start()

    entropy_enabled = bool(common_cfg.get("entropy_enabled", True))
    entropy_codec = str(common_cfg.get("entropy_codec", "huffman" if entropy_enabled else "none"))
    quant_bits = int(common_cfg.get("quant_bits", 8))
    clip_abs = float(common_cfg.get("clip_abs", 0.0))

    huffman_encode_map = None
    huffman_decode_tree = None
    if entropy_enabled and entropy_codec == "huffman":
        codebook_path_base = str(
            common_cfg.get(
                "entropy_codebook_path",
                "checkpoints/grpc_online_splines_entropy_codebook.json",
            )
        )
        codebook_path = resolve_codebook_path_by_bits(codebook_path_base, quant_bits=quant_bits)
        codebook = load_huffman_codebook(codebook_path)
        huffman_encode_map = codebook["encode_map"]
        if "quant_bits" in common_cfg and int(common_cfg["quant_bits"]) != int(codebook["quant_bits"]):
            raise ValueError(
                f"quant_bits mismatch: config={int(common_cfg['quant_bits'])}, "
                f"codebook={int(codebook['quant_bits'])}, path={codebook['path']}"
            )
        if clip_abs <= 0:
            clip_abs = float(codebook["clip_abs"])
        if "quant_bits" not in common_cfg:
            quant_bits = int(codebook["quant_bits"])
        huffman_decode_tree = build_huffman_decode_tree(huffman_encode_map)

    if entropy_enabled and entropy_codec not in ("huffman",):
        raise ValueError(f"unsupported entropy_codec: {entropy_codec}; zlib has been removed")

    try:
        sent = stream_pose_frames(
            feature_file=feature_file,
            sender=sender,
            fps=float(sender_cfg["fps"]),
            max_frames=int(sender_cfg.get("max_frames", 0)),
            start_col=int(sender_cfg.get("start_col", 0)),
            i_frame_interval=int(common_cfg.get("i_frame_interval", 30)),
            quantize_i_frame=bool(common_cfg.get("quantize_i_frame", common_cfg.get("quantize", True))),
            quantize_p_frame=bool(common_cfg.get("quantize_p_frame", common_cfg.get("quantize", True))),
            quant_scale=float(common_cfg.get("quant_scale", 1000.0)),
            entropy_enabled=entropy_enabled,
            entropy_codec=entropy_codec,
            quant_bits=quant_bits,
            clip_abs=clip_abs,
            huffman_encode_map=huffman_encode_map,
            huffman_decode_tree=huffman_decode_tree,
            packet_tag=str(common_cfg.get("packet_tag", "POSE_RES_V1")),
            channel=channel,
            num_keypoints=int(common_cfg.get("num_keypoints", 17)),
            coord_dims=int(common_cfg.get("coord_dims", 3)),
            debug=bool(sender_cfg.get("debug", False)),
        )
        print(
            "sender_done: "
            f"sent_frames={sent}, "
            f"fps={float(sender_cfg['fps'])}, "
            f"channel={channel}, "
            f"file={feature_file}"
        )
    finally:
        sender.shutdown(drain_timeout=2.0)


if __name__ == "__main__":
    main()
