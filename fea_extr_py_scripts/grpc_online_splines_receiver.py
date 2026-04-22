import json
import os
import threading
import time
import zlib
from collections import deque

import numpy as np

from server import THStreamServiceServicer, serve


DEFAULT_CODEC_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "checkpoints", "grpc_online_splines_codec_config.json")
)


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


def decode_payload(payload_bytes: bytes, payload_dtype: str, entropy_codec: str, quant_scale: float):
    raw = payload_bytes
    if entropy_codec == "zlib":
        raw = zlib.decompress(raw)
    elif entropy_codec not in ("", "none"):
        raise ValueError(f"unsupported entropy codec: {entropy_codec}")

    if payload_dtype == "qint16":
        if quant_scale <= 0:
            raise ValueError(f"quant_scale must be > 0, got {quant_scale}")
        values = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / quant_scale
    elif payload_dtype == "float32":
        values = np.frombuffer(raw, dtype=np.float32)
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


def main():
    cfg = load_runtime_config(DEFAULT_CODEC_CONFIG_PATH)
    common_cfg = cfg["common"]
    receiver_cfg = cfg["receiver"]

    grpc_port = int(receiver_cfg["grpc_port"])
    report_interval = float(receiver_cfg.get("report_interval", 1.0))
    poll_interval = float(receiver_cfg.get("poll_interval", 0.01))
    debug = bool(receiver_cfg.get("debug", False))
    save_enabled = bool(receiver_cfg.get("save_enabled", True))
    save_max_frames = int(receiver_cfg.get("save_max_frames", 0))

    if report_interval <= 0:
        raise ValueError("receiver.report_interval must be > 0")
    if poll_interval <= 0:
        raise ValueError("receiver.poll_interval must be > 0")

    save_dir = str(receiver_cfg.get("save_dir", "res/decode_res"))
    if not os.path.isabs(save_dir):
        save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", save_dir))
    os.makedirs(save_dir, exist_ok=True)

    save_file = str(receiver_cfg.get("save_file", "decoded_all_channels.npy"))
    save_path = os.path.join(save_dir, save_file)

    default_num_keypoints = int(common_cfg.get("num_keypoints", 17))
    default_coord_dims = int(common_cfg.get("coord_dims", 3))
    default_quant_scale = float(common_cfg.get("quant_scale", 1000.0))
    packet_tag = str(common_cfg.get("packet_tag", "POSE_RES_V1"))

    servicer = THStreamServiceServicer()
    server_thread = threading.Thread(target=serve, args=(servicer, grpc_port), daemon=True)
    server_thread.start()

    stats = ReceiveStats()
    last_report_time = time.time()
    prev_recon_by_channel = {}
    saved_frames = []

    print(
        f"接收端已启动，监听 gRPC 端口: {grpc_port}, "
        f"save_all_channels=true, save_enabled={save_enabled}, save_path={save_path}"
    )

    try:
        while True:
            if servicer.receive_data_buffer.get_size() < 1:
                time.sleep(poll_interval)
            else:
                payload = servicer.receive_data_buffer.get_items()

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

                    if save_enabled:
                        saved_frames.append(recon_flat.reshape(num_keypoints, coord_dims).copy())
                        if save_max_frames > 0 and len(saved_frames) >= save_max_frames:
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

    if save_enabled:
        if saved_frames:
            decoded_arr = np.asarray(saved_frames, dtype=np.float32)
            np.save(save_path, decoded_arr)
            print(f"解码结果已保存: {save_path}, shape={decoded_arr.shape}")
        else:
            print("未保存任何帧：可能未接收到可解码数据")

    print("最终统计:")
    print(stats.summary())


if __name__ == "__main__":
    main()

