import argparse
import json
import os
import re
import signal
import threading
import time
import traceback

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from client import THStreamClient
from THStreamData import THStreamDataPayload


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CODEC_CONFIG_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "grpc_online_avatar_fea_codec_config.json",
)


def load_runtime_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"codec config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError("config root must be JSON object")

    for key in ("common", "sender"):
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


def build_pose_codec_context(config_path: str) -> dict:
    cfg = load_runtime_config(config_path)
    common_cfg = cfg["common"]
    sender_cfg = cfg["sender"]

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

    face_dims = int(common_cfg.get("face_dims", 52))
    face_i_frame_interval = int(common_cfg.get("face_i_frame_interval", common_cfg.get("i_frame_interval", 30)))
    face_quantize_i_frame = bool(common_cfg.get("face_quantize_i_frame", common_cfg.get("quantize_i_frame", common_cfg.get("quantize", True))))
    face_quantize_p_frame = bool(common_cfg.get("face_quantize_p_frame", common_cfg.get("quantize_p_frame", common_cfg.get("quantize", True))))
    face_quant_scale = float(common_cfg.get("face_quant_scale", common_cfg.get("quant_scale", 1000.0)))
    face_entropy_enabled = bool(common_cfg.get("face_entropy_enabled", entropy_enabled))
    face_entropy_codec = str(common_cfg.get("face_entropy_codec", entropy_codec if face_entropy_enabled else "none"))
    face_quant_bits = int(common_cfg.get("face_quant_bits", quant_bits))
    face_clip_abs = float(common_cfg.get("face_clip_abs", clip_abs))

    face_huffman_encode_map = None
    face_huffman_decode_tree = None
    if face_entropy_enabled and face_entropy_codec == "huffman":
        face_codebook_path_base = str(
            common_cfg.get(
                "face_entropy_codebook_path",
                common_cfg.get(
                    "entropy_codebook_path",
                    "checkpoints/grpc_online_splines_entropy_codebook.json",
                ),
            )
        )
        face_codebook_path = resolve_codebook_path_by_bits(face_codebook_path_base, quant_bits=face_quant_bits)
        face_codebook = load_huffman_codebook(face_codebook_path)
        face_huffman_encode_map = face_codebook["encode_map"]
        if "face_quant_bits" in common_cfg and int(common_cfg["face_quant_bits"]) != int(face_codebook["quant_bits"]):
            raise ValueError(
                f"face_quant_bits mismatch: config={int(common_cfg['face_quant_bits'])}, "
                f"codebook={int(face_codebook['quant_bits'])}, path={face_codebook['path']}"
            )
        if face_clip_abs <= 0:
            face_clip_abs = float(face_codebook["clip_abs"])
        if "face_quant_bits" not in common_cfg:
            face_quant_bits = int(face_codebook["quant_bits"])
        face_huffman_decode_tree = build_huffman_decode_tree(face_huffman_encode_map)

    if face_entropy_enabled and face_entropy_codec not in ("huffman",):
        raise ValueError(f"unsupported face_entropy_codec: {face_entropy_codec}; zlib has been removed")

    return {
        "config": cfg,
        "packet_tag": str(common_cfg.get("packet_tag", "POSE_RES_V1")),
        "i_frame_interval": int(common_cfg.get("i_frame_interval", 30)),
        "coord_dims": int(common_cfg.get("coord_dims", 3)),
        "quantize": bool(common_cfg.get("quantize", True)),
        "quantize_i_frame": bool(common_cfg.get("quantize_i_frame", common_cfg.get("quantize", True))),
        "quantize_p_frame": bool(common_cfg.get("quantize_p_frame", common_cfg.get("quantize", True))),
        "quant_scale": float(common_cfg.get("quant_scale", 1000.0)),
        "entropy_enabled": entropy_enabled,
        "entropy_codec": entropy_codec,
        "quant_bits": quant_bits,
        "clip_abs": clip_abs,
        "huffman_encode_map": huffman_encode_map,
        "huffman_decode_tree": huffman_decode_tree,
        "channel": int(sender_cfg.get("channel", 0)),
        "buffer_limit": int(sender_cfg.get("buffer_limit", 5)),
        "sender_debug": bool(sender_cfg.get("debug", False)),
        "sender_server_addr": str(sender_cfg.get("server_addr", "127.0.0.1")),
        "sender_port": int(sender_cfg.get("port_num", 50051)),
        "face_dims": face_dims,
        "face_i_frame_interval": face_i_frame_interval,
        "face_quantize_i_frame": face_quantize_i_frame,
        "face_quantize_p_frame": face_quantize_p_frame,
        "face_quant_scale": face_quant_scale,
        "face_entropy_enabled": face_entropy_enabled,
        "face_entropy_codec": face_entropy_codec,
        "face_quant_bits": face_quant_bits,
        "face_clip_abs": face_clip_abs,
        "face_huffman_encode_map": face_huffman_encode_map,
        "face_huffman_decode_tree": face_huffman_decode_tree,
    }


class FrameDataManager:
    """管理帧数据，整合同一帧的姿势和面部特征。"""

    def __init__(self, codec_ctx: dict, debug: bool = False):
        self.lock = threading.Lock()
        self.frame_data = {}
        self.max_frames = 10
        self.last_valid_face_data = None
        self.last_two_pose_frames = []
        self.frame_count = 0
        self.last_processed_landmarks = None
        self.codec_ctx = codec_ctx
        self.debug = debug

        self.prev_recon_by_channel = {}
        self.sent_count_by_channel = {}
        self.prev_face_recon_by_channel = {}
        self.sent_face_count_by_channel = {}

    def update_pose_data(self, timestamp_ms, pose_data):
        with self.lock:
            if timestamp_ms not in self.frame_data:
                self.frame_data[timestamp_ms] = {"pose": None, "face": None}
            self.frame_data[timestamp_ms]["pose"] = pose_data
            self._try_send_complete_frame(timestamp_ms)
            self._cleanup_old_frames()

    def update_face_data(self, timestamp_ms, face_data):
        with self.lock:
            if timestamp_ms not in self.frame_data:
                self.frame_data[timestamp_ms] = {"pose": None, "face": None}
            self.frame_data[timestamp_ms]["face"] = face_data
            self._try_send_complete_frame(timestamp_ms)
            self._cleanup_old_frames()

    def _encode_pose_with_codec(self, pose_bytes: bytes, timestamp_ms: int):
        pose_values = np.fromstring(pose_bytes.decode("utf-8"), sep=",", dtype=np.float32)
        if pose_values.size == 0:
            raise ValueError("empty pose frame")

        coord_dims = int(self.codec_ctx["coord_dims"])
        if coord_dims <= 0:
            raise ValueError(f"invalid coord_dims: {coord_dims}")
        if pose_values.size % coord_dims != 0:
            raise ValueError(
                f"pose dims mismatch: dims={pose_values.size}, coord_dims={coord_dims}"
            )

        channel = int(self.codec_ctx["channel"])
        sent_count = int(self.sent_count_by_channel.get(channel, 0))
        prev_recon = self.prev_recon_by_channel.get(channel)

        is_i_frame = prev_recon is None or (
            int(self.codec_ctx["i_frame_interval"]) > 0 and sent_count % int(self.codec_ctx["i_frame_interval"]) == 0
        )
        if is_i_frame:
            packet_kind = "I"
            payload_values = pose_values
            use_quantize = bool(self.codec_ctx["quantize_i_frame"])
        else:
            packet_kind = "P"
            payload_values = (pose_values - prev_recon).astype(np.float32, copy=False)
            use_quantize = bool(self.codec_ctx["quantize_p_frame"])

        encoded_payload, payload_dtype, entropy_codec, bit_length = encode_payload(
            payload_values,
            quantize=use_quantize,
            quant_scale=float(self.codec_ctx["quant_scale"]),
            entropy_enabled=bool(self.codec_ctx["entropy_enabled"]),
            entropy_codec=str(self.codec_ctx["entropy_codec"]),
            quant_bits=int(self.codec_ctx["quant_bits"]),
            clip_abs=float(self.codec_ctx["clip_abs"]),
            huffman_encode_map=self.codec_ctx["huffman_encode_map"],
        )

        decoded_values = decode_payload_local(
            encoded_payload,
            payload_dtype=payload_dtype,
            entropy_codec=entropy_codec,
            quant_scale=float(self.codec_ctx["quant_scale"]),
            quant_bits=int(self.codec_ctx["quant_bits"]),
            clip_abs=float(self.codec_ctx["clip_abs"]),
            bit_length=int(bit_length),
            huffman_decode_tree=self.codec_ctx["huffman_decode_tree"],
            expected_dims=int(pose_values.size),
        )

        if packet_kind == "P" and prev_recon is not None:
            recon_flat = (prev_recon + decoded_values).astype(np.float32, copy=False)
        else:
            recon_flat = decoded_values.astype(np.float32, copy=False)

        self.prev_recon_by_channel[channel] = recon_flat
        self.sent_count_by_channel[channel] = sent_count + 1

        num_keypoints = int(pose_values.size // coord_dims)
        meta = {
            "tag": str(self.codec_ctx["packet_tag"]),
            "kind": packet_kind,
            "frame_idx": int(sent_count),
            "timestamp_ms": int(timestamp_ms),
            "channel": channel,
            "num_keypoints": num_keypoints,
            "coord_dims": coord_dims,
            "pose_dims": int(pose_values.size),
            "payload_dtype": payload_dtype,
            "entropy_codec": entropy_codec,
            "entropy_bit_length": int(bit_length),
            "quantize": int(use_quantize),
            "quantize_i_frame": int(bool(self.codec_ctx["quantize_i_frame"])),
            "quantize_p_frame": int(bool(self.codec_ctx["quantize_p_frame"])),
            "quant_scale": float(self.codec_ctx["quant_scale"]),
            "quant_bits": int(self.codec_ctx["quant_bits"]),
            "clip_abs": float(self.codec_ctx["clip_abs"]),
        }
        return encoded_payload, meta

    def _encode_face_with_codec(self, face_bytes: bytes, timestamp_ms: int):
        face_values = np.fromstring(face_bytes.decode("utf-8"), sep=",", dtype=np.float32)
        if face_values.size == 0:
            raise ValueError("empty face frame")

        expected_face_dims = int(self.codec_ctx.get("face_dims", 52))
        if expected_face_dims > 0 and face_values.size != expected_face_dims:
            raise ValueError(
                f"face dims mismatch: dims={face_values.size}, expected={expected_face_dims}"
            )

        channel = int(self.codec_ctx["channel"])
        sent_count = int(self.sent_face_count_by_channel.get(channel, 0))
        prev_recon = self.prev_face_recon_by_channel.get(channel)

        is_i_frame = prev_recon is None or (
            int(self.codec_ctx["face_i_frame_interval"]) > 0 and sent_count % int(self.codec_ctx["face_i_frame_interval"]) == 0
        )
        if is_i_frame:
            packet_kind = "I"
            payload_values = face_values
            use_quantize = bool(self.codec_ctx["face_quantize_i_frame"])
        else:
            packet_kind = "P"
            payload_values = (face_values - prev_recon).astype(np.float32, copy=False)
            use_quantize = bool(self.codec_ctx["face_quantize_p_frame"])

        encoded_payload, payload_dtype, entropy_codec, bit_length = encode_payload(
            payload_values,
            quantize=use_quantize,
            quant_scale=float(self.codec_ctx["face_quant_scale"]),
            entropy_enabled=bool(self.codec_ctx["face_entropy_enabled"]),
            entropy_codec=str(self.codec_ctx["face_entropy_codec"]),
            quant_bits=int(self.codec_ctx["face_quant_bits"]),
            clip_abs=float(self.codec_ctx["face_clip_abs"]),
            huffman_encode_map=self.codec_ctx["face_huffman_encode_map"],
        )

        decoded_values = decode_payload_local(
            encoded_payload,
            payload_dtype=payload_dtype,
            entropy_codec=entropy_codec,
            quant_scale=float(self.codec_ctx["face_quant_scale"]),
            quant_bits=int(self.codec_ctx["face_quant_bits"]),
            clip_abs=float(self.codec_ctx["face_clip_abs"]),
            bit_length=int(bit_length),
            huffman_decode_tree=self.codec_ctx["face_huffman_decode_tree"],
            expected_dims=int(face_values.size),
        )

        if packet_kind == "P" and prev_recon is not None:
            recon_flat = (prev_recon + decoded_values).astype(np.float32, copy=False)
        else:
            recon_flat = decoded_values.astype(np.float32, copy=False)

        self.prev_face_recon_by_channel[channel] = recon_flat
        self.sent_face_count_by_channel[channel] = sent_count + 1

        meta = {
            "kind": packet_kind,
            "frame_idx": int(sent_count),
            "face_dims": int(face_values.size),
            "payload_dtype": payload_dtype,
            "entropy_codec": entropy_codec,
            "entropy_bit_length": int(bit_length),
            "quantize": int(use_quantize),
            "quantize_i_frame": int(bool(self.codec_ctx["face_quantize_i_frame"])),
            "quantize_p_frame": int(bool(self.codec_ctx["face_quantize_p_frame"])),
            "quant_scale": float(self.codec_ctx["face_quant_scale"]),
            "quant_bits": int(self.codec_ctx["face_quant_bits"]),
            "clip_abs": float(self.codec_ctx["face_clip_abs"]),
        }
        return encoded_payload, meta

    def _try_send_complete_frame(self, timestamp_ms):
        frame = self.frame_data.get(timestamp_ms)
        if frame and frame["pose"] is not None and frame["face"] is not None:
            if hasattr(self, "client"):
                smoothed_pose = self._smooth_pose_data(frame["pose"])
                t_encode_ms = int(time.time() * 1000)
                timing_meta = {
                    "t_begin": int(timestamp_ms),
                    "t_encode": t_encode_ms,
                }

                try:
                    limb_payload, limb_meta = self._encode_pose_with_codec(smoothed_pose, int(timestamp_ms))
                except Exception as codec_exc:
                    if self.debug:
                        print(f"编码失败，回退原始csv: {codec_exc}")
                    limb_payload = smoothed_pose
                    limb_meta = {
                        "kind": "I",
                        "frame_idx": 0,
                        "num_keypoints": int(self.codec_ctx.get("coord_dims", 3)),
                        "coord_dims": int(self.codec_ctx.get("coord_dims", 3)),
                        "pose_dims": 0,
                        "payload_dtype": "csv",
                    }

                try:
                    face_payload, face_meta = self._encode_face_with_codec(frame["face"], int(timestamp_ms))
                except Exception as codec_exc:
                    if self.debug:
                        print(f"face 编码失败，回退原始csv: {codec_exc}")
                    face_payload = frame["face"]
                    face_meta = {
                        "kind": "I",
                        "frame_idx": 0,
                        "face_dims": int(self.codec_ctx.get("face_dims", 52)),
                        "payload_dtype": "csv",
                    }

                codec_meta = {
                    "tag": str(self.codec_ctx["packet_tag"]),
                    "timestamp_ms": int(timestamp_ms),
                    "channel": int(self.codec_ctx.get("channel", 0)),
                    "limb": limb_meta,
                    "face": face_meta,
                }

                payload_send = THStreamDataPayload(
                    rgb_data=b"\x00",
                    point_data=b"\x00",
                    face_data=face_payload,
                    limb_data=limb_payload,
                    ext_data=json.dumps(timing_meta, separators=(",", ":")).encode("utf-8"),
                    ext_desc=json.dumps(codec_meta, separators=(",", ":"), ensure_ascii=False),
                )
                self.client.send_data_buffer.add_item(payload_send)
                del self.frame_data[timestamp_ms]

    def _smooth_pose_data(self, current_pose_data):
        self.frame_count += 1

        if self.frame_count <= 2:
            self.last_two_pose_frames.append(current_pose_data)
            if len(self.last_two_pose_frames) > 2:
                self.last_two_pose_frames.pop(0)
            return current_pose_data

        try:
            current_pose_values = [float(x) for x in current_pose_data.decode("utf-8").split(",")]

            prev_frames_values = []
            for frame_data in self.last_two_pose_frames:
                prev_frames_values.append([float(x) for x in frame_data.decode("utf-8").split(",")])

            if all(len(prev_values) == len(current_pose_values) for prev_values in prev_frames_values):
                smoothed_values = []
                for i in range(len(current_pose_values)):
                    avg_value = (
                        current_pose_values[i]
                        + prev_frames_values[0][i]
                        + prev_frames_values[1][i]
                    ) / 3.0
                    smoothed_values.append(avg_value)

                self.last_two_pose_frames.pop(0)
                self.last_two_pose_frames.append(current_pose_data)

                smoothed_str = ",".join(map(str, smoothed_values))
                return smoothed_str.encode("utf-8")

        except Exception as e:
            print(f"平滑处理时出错: {e}")

        self.last_two_pose_frames.append(current_pose_data)
        if len(self.last_two_pose_frames) > 2:
            self.last_two_pose_frames.pop(0)
        return current_pose_data

    def _cleanup_old_frames(self):
        if len(self.frame_data) > self.max_frames:
            oldest_timestamp = min(self.frame_data.keys())
            del self.frame_data[oldest_timestamp]

    def set_client(self, client):
        self.client = client


def run_client(client):
    client.run()


def process_pose_result(result, timestamp_ms, frame_data_manager, frame_width, frame_height, debug=False):
    try:
        if result.pose_landmarks:
            for pose_landmark in result.pose_landmarks:
                landmarks_data = []
                prev_landmarks = frame_data_manager.last_processed_landmarks

                for i, landmark in enumerate(pose_landmark):
                    if landmark.visibility < 0.5 and prev_landmarks and i < len(prev_landmarks):
                        landmarks_data.append(prev_landmarks[i])
                        if debug:
                            print(f"关键点 {i} 可见性低 ({landmark.visibility:.2f})，使用上一帧数据")
                    else:
                        landmarks_data.append(
                            (
                                landmark.x * frame_width,
                                frame_height - landmark.y * frame_height,
                                landmark.z * frame_width,
                            )
                        )

                frame_data_manager.last_processed_landmarks = landmarks_data.copy()
                flat_landmarks = [coord for landmark in landmarks_data for coord in landmark]
                landmarks_str = ",".join(map(str, flat_landmarks))
                landmarks_data_bytes = landmarks_str.encode("utf-8")
                frame_data_manager.update_pose_data(timestamp_ms, landmarks_data_bytes)

            if debug:
                print(f"姿势数据大小: {len(landmarks_data_bytes)}")
                current_timestamp_ms = int(time.time() * 1000)
                print(f"姿势处理延迟: {current_timestamp_ms - timestamp_ms} ms")
    except Exception as exc:
        print(f"[pose_callback] 处理异常: {exc}")
        traceback.print_exc()


def process_face_result(result, timestamp_ms, frame_data_manager, debug=False):
    try:
        if result.face_blendshapes:
            for blendshape in result.face_blendshapes:
                blendshape_data = [category.score for category in blendshape]
                blendshape_str = ",".join(map(str, blendshape_data))
                blendshape_data_bytes = blendshape_str.encode("utf-8")

                frame_data_manager.last_valid_face_data = blendshape_data_bytes
                frame_data_manager.update_face_data(timestamp_ms, blendshape_data_bytes)

                if debug:
                    print(f"面部表情数据大小: {len(blendshape_data_bytes)}")
                    current_timestamp_ms = int(time.time() * 1000)
                    print(f"面部处理延迟: {current_timestamp_ms - timestamp_ms} ms")
        else:
            if frame_data_manager.last_valid_face_data is not None:
                if debug:
                    print("未检测到面部，使用上一次有效的面部数据")
                frame_data_manager.update_face_data(timestamp_ms, frame_data_manager.last_valid_face_data)
            else:
                if debug:
                    print("未检测到面部，且无历史数据，使用默认值")
                default_face_dims = int(frame_data_manager.codec_ctx.get("face_dims", 52))
                default_face_data = ",".join(["0.0"] * default_face_dims)
                default_face_bytes = default_face_data.encode("utf-8")
                frame_data_manager.update_face_data(timestamp_ms, default_face_bytes)
    except Exception as exc:
        print(f"[face_callback] 处理异常: {exc}")
        traceback.print_exc()


def main(
    server_addr=None,
    port_num=None,
    pose_model_path=None,
    face_model_path=None,
    debug=False,
    codec_config_path=DEFAULT_CODEC_CONFIG_PATH,
):
    vision_running_mode = mp.tasks.vision.RunningMode

    codec_ctx = build_pose_codec_context(codec_config_path)
    cfg = codec_ctx["config"]
    sender_cfg = cfg["sender"]

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_pose_model = os.path.join(project_root, "data", "pose_landmarker_full.task")
    default_face_model = os.path.join(project_root, "data", "face_landmarker_v2_with_blendshapes.task")

    pose_model_path = pose_model_path or default_pose_model
    face_model_path = face_model_path or default_face_model
    server_addr = server_addr or str(sender_cfg.get("server_addr", codec_ctx["sender_server_addr"]))
    port_num = int(port_num or int(sender_cfg.get("port_num", codec_ctx["sender_port"])))
    debug = bool(debug or codec_ctx["sender_debug"])
    buffer_limit = max(int(codec_ctx.get("buffer_limit", 5)), 1)

    if not os.path.exists(pose_model_path):
        raise FileNotFoundError(f"姿势模型文件不存在: {pose_model_path}")
    if not os.path.exists(face_model_path):
        raise FileNotFoundError(f"面部模型文件不存在: {face_model_path}")

    client = THStreamClient(host=server_addr, port=port_num)
    client_thread = threading.Thread(target=run_client, args=(client,))
    client_thread.start()

    frame_data_manager = FrameDataManager(codec_ctx=codec_ctx, debug=debug)
    frame_data_manager.set_client(client)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose_base_options = python.BaseOptions(model_asset_path=pose_model_path)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base_options,
        running_mode=vision_running_mode.LIVE_STREAM,
        output_segmentation_masks=False,
        result_callback=lambda result, _, timestamp_ms: process_pose_result(
            result,
            timestamp_ms,
            frame_data_manager,
            frame_width,
            frame_height,
            debug=debug,
        ),
    )
    pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

    face_base_options = python.BaseOptions(model_asset_path=face_model_path)
    face_options = vision.FaceLandmarkerOptions(
        base_options=face_base_options,
        running_mode=vision_running_mode.LIVE_STREAM,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        result_callback=lambda result, _, timestamp_ms: process_face_result(
            result,
            timestamp_ms,
            frame_data_manager,
            debug=debug,
        ),
    )
    face_detector = vision.FaceLandmarker.create_from_options(face_options)

    interrupted = {"sigint": False}

    def _on_sigint(signum, _frame):
        interrupted["sigint"] = True
        print(f"收到 SIGINT(signum={signum})，准备退出...")

    try:
        signal.signal(signal.SIGINT, _on_sigint)
    except Exception:
        pass

    base_wall_ms = int(time.time() * 1000)
    base_mono_ns = time.monotonic_ns()
    last_stream_timestamp_ms = 0

    def _next_stream_timestamp_ms() -> int:
        nonlocal last_stream_timestamp_ms
        candidate = base_wall_ms + (time.monotonic_ns() - base_mono_ns) // 1_000_000
        if candidate <= last_stream_timestamp_ms:
            candidate = last_stream_timestamp_ms + 1
        last_stream_timestamp_ms = int(candidate)
        return int(candidate)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("摄像头读帧失败，结束发送循环")
                break

            frame_timestamp_ms = _next_stream_timestamp_ms()

            buffer_size = client.send_data_buffer.get_size()
            wait_begin = time.time()
            while buffer_size >= buffer_limit:
                if (time.time() - wait_begin) > 2.0:
                    print(
                        f"发送缓冲区持续满载超过2s(size={buffer_size}, limit={buffer_limit})，"
                        "等待接收端消费..."
                    )
                    wait_begin = time.time()
                time.sleep(0.01)
                buffer_size = client.send_data_buffer.get_size()

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            pose_detector.detect_async(mp_image, int(frame_timestamp_ms))
            face_detector.detect_async(mp_image, int(frame_timestamp_ms))

            if debug:
                print(f"帧时间戳: {frame_timestamp_ms}")

    except KeyboardInterrupt:
        if interrupted["sigint"]:
            print("程序被 SIGINT 中断")
        else:
            print("程序捕获到 KeyboardInterrupt（非显式 SIGINT）")
    except Exception as exc:
        print(f"主循环异常退出: {exc}")
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("资源已释放")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行姿势和面部表情检测发送器")
    parser.add_argument("--server_addr", type=str, help="服务器地址（默认取配置文件 sender.server_addr）")
    parser.add_argument("--port_num", type=int, help="端口号（默认取配置文件 sender.port_num）")
    parser.add_argument("--pose_model_path", type=str, help="姿势模型路径")
    parser.add_argument("--face_model_path", type=str, help="面部模型路径")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument(
        "--codec_config_path",
        type=str,
        default=DEFAULT_CODEC_CONFIG_PATH,
        help="编解码配置路径",
    )

    args = parser.parse_args()

    main(
        server_addr=args.server_addr,
        port_num=args.port_num,
        pose_model_path=args.pose_model_path,
        face_model_path=args.face_model_path,
        debug=args.debug,
        codec_config_path=args.codec_config_path,
    )
