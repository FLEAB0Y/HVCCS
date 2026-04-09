import argparse
import json
import os
import time
import zlib
from concurrent import futures
from collections import deque
from typing import Dict, Optional

import grpc
import numpy as np
import torch

from poselandmark_online_encoder_v2 import SpatioTemporalPredictor
import data_stream_pb2_grpc
from server import THStreamServiceServicer


PACKET_TAG = b"POSE99V1"
DEFAULT_CHECKPOINT_PATH = "/home/ztw/HVCCS/checkpoints/pose_mamba_init_k17_d3_w32_h128_ds64_dc4_ex4_nl4_s2026.pt"


def _is_cuda_arch_supported(cuda_device: int) -> bool:
    compiled_arches = set(torch.cuda.get_arch_list())
    if not compiled_arches:
        return True
    major, minor = torch.cuda.get_device_capability(cuda_device)
    return f"sm_{major}{minor}" in compiled_arches


def select_torch_device(cuda_device: int) -> torch.device:
    if cuda_device < 0:
        print("cuda_device < 0，使用 CPU")
        return torch.device("cpu")

    if not torch.cuda.is_available():
        print("CUDA 不可用，自动回退到 CPU")
        return torch.device("cpu")

    if cuda_device >= torch.cuda.device_count():
        print(f"cuda_device={cuda_device} 超出可用设备数量，自动回退到 CPU")
        return torch.device("cpu")

    try:
        if not _is_cuda_arch_supported(cuda_device):
            major, minor = torch.cuda.get_device_capability(cuda_device)
            print(
                f"检测到当前 GPU 架构 sm_{major}{minor} 不受当前 PyTorch 支持，自动回退到 CPU"
            )
            return torch.device("cpu")
    except Exception as exc:
        print(f"CUDA 架构检查失败({exc})，自动回退到 CPU")
        return torch.device("cpu")

    return torch.device(f"cuda:{cuda_device}")


class MetaCheckError(RuntimeError):
    pass


class PosePacketCodec:
    @staticmethod
    def from_bytes(
        payload: bytes,
        expected_dims: int,
        payload_dtype: str,
        entropy_codec: str,
        quant_scale: float,
    ) -> np.ndarray:
        raw = payload
        if entropy_codec == "zlib":
            raw = zlib.decompress(raw)
        elif entropy_codec not in ("", "none"):
            raise ValueError(f"unsupported entropy codec: {entropy_codec}")

        if payload_dtype == "qint16":
            if quant_scale <= 0:
                raise ValueError(f"quant_scale must be positive, got {quant_scale}")
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / quant_scale
        elif payload_dtype == "float32":
            arr = np.frombuffer(raw, dtype=np.float32)
        else:
            raise ValueError(f"unsupported payload dtype: {payload_dtype}")

        if arr.size != expected_dims:
            raise ValueError(f"invalid pose dims: got {arr.size}, expected {expected_dims}")
        return arr.copy()


class MambaPredictivePoseDecoder:
    def __init__(self, model: SpatioTemporalPredictor, num_keypoints: int, coord_dims: int, window_size: int = 10):
        self.model = model
        self.num_keypoints = num_keypoints
        self.coord_dims = coord_dims
        self.pose_dims = num_keypoints * coord_dims
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.device = next(model.parameters()).device
        self.decoded_count = 0

    def _predict_next_pose(self) -> np.ndarray:
        if len(self.history) < self.window_size:
            raise RuntimeError("history is not ready for P-frame prediction")

        seq = np.stack(list(self.history), axis=0).astype(np.float32).reshape(
            self.window_size,
            self.num_keypoints,
            self.coord_dims,
        )
        seq_tensor = torch.tensor(seq, device=self.device).unsqueeze(0)  # (1, L, 33, 3)
        with torch.no_grad():
            pred = self.model(seq_tensor).view(-1).detach().cpu().numpy()
        return pred

    def decode_one(self, point_data: bytes, meta: Dict) -> np.ndarray:
        data = PosePacketCodec.from_bytes(
            point_data,
            expected_dims=self.pose_dims,
            payload_dtype=str(meta.get("payload_dtype", "float32")),
            entropy_codec=str(meta.get("entropy_codec", "none")),
            quant_scale=float(meta.get("quant_scale", 1000.0)),
        )
        kind = meta.get("kind", "I")

        if kind == "I" or len(self.history) < self.window_size:
            frame = data
        elif kind == "P":
            pred = self._predict_next_pose()
            frame = (pred + data).astype(np.float32)
        else:
            raise ValueError(f"unknown packet kind: {kind}")

        self.history.append(frame.copy())
        self.decoded_count += 1
        return frame


def load_model_checkpoint(model, checkpoint_path):
    if not checkpoint_path:
        raise ValueError("checkpoint path is required")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        meta = ckpt.get("meta", {})
    else:
        state_dict = ckpt
        meta = {}

    model.load_state_dict(state_dict, strict=True)
    if meta:
        print(f"loaded_checkpoint_meta: {meta}")
    print(f"loaded_checkpoint: {checkpoint_path}")
    return True


def build_decoder_model(
    num_keypoints: int,
    model_in_dim: int,
    gnn_hidden: int,
    mamba_d_state: int,
    mamba_d_conv: int,
    mamba_expand: int,
    mamba_n_layer: int,
    model_seed: int,
    device: torch.device,
    checkpoint_path: str,
) -> SpatioTemporalPredictor:
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    model = SpatioTemporalPredictor(
        in_dim=model_in_dim,
        gnn_hidden=gnn_hidden,
        mamba_d_state=mamba_d_state,
        mamba_d_conv=mamba_d_conv,
        mamba_expand=mamba_expand,
        mamba_n_layer=mamba_n_layer,
        num_nodes=num_keypoints,
    ).to(device)
    load_model_checkpoint(model, checkpoint_path)
    model.eval()
    return model


def validate_stream_meta_once(meta: Dict, expected: Dict) -> None:
    """只在首个有效包进行一次超参数一致性校验。"""
    missing = []
    mismatches = []
    for key, expected_value in expected.items():
        if key not in meta:
            missing.append(key)
            continue

        got_value = meta[key]
        if isinstance(expected_value, int):
            try:
                got_value = int(got_value)
            except (TypeError, ValueError):
                mismatches.append(f"{key}: got={meta[key]} expected={expected_value}")
                continue

        if got_value != expected_value:
            mismatches.append(f"{key}: got={got_value} expected={expected_value}")

    if missing or mismatches:
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if mismatches:
            parts.append(f"mismatch={mismatches}")
        raise MetaCheckError("meta_check_failed: " + "; ".join(parts))


def start_local_grpc_server(port: int):
    servicer = THStreamServiceServicer()
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    data_stream_pb2_grpc.add_THStreamServiceServicer_to_server(servicer, grpc_server)
    grpc_server.add_insecure_port(f"[::]:{port}")
    grpc_server.start()
    print(f"Server started, listening on port {port}")
    return servicer, grpc_server


def run_receiver_service(
    host: str,
    port: int,
    window_size: int,
    num_keypoints: int,
    model_in_dim: int,
    history_len: int,
    gnn_hidden: int,
    mamba_d_state: int,
    mamba_d_conv: int,
    mamba_expand: int,
    mamba_n_layer: int,
    i_frame_interval: int,
    quantize_i_frame: int,
    quantize_p_frame: int,
    quant_scale: float,
    entropy_enabled: int,
    entropy_level: int,
    model_seed: int,
    cuda_device: int,
    checkpoint_path: str,
    max_frames: Optional[int],
    debug: bool,
    output_path: Optional[str],
) -> None:
    device = select_torch_device(cuda_device)
    model = build_decoder_model(
        num_keypoints=num_keypoints,
        model_in_dim=model_in_dim,
        gnn_hidden=gnn_hidden,
        mamba_d_state=mamba_d_state,
        mamba_d_conv=mamba_d_conv,
        mamba_expand=mamba_expand,
        mamba_n_layer=mamba_n_layer,
        model_seed=model_seed,
        device=device,
        checkpoint_path=checkpoint_path,
    )
    decoder = MambaPredictivePoseDecoder(
        model=model,
        num_keypoints=num_keypoints,
        coord_dims=model_in_dim,
        window_size=window_size,
    )

    servicer, grpc_server = start_local_grpc_server(port=port)
    print(
        f"decoder_listening: host={host}, port={port}, device={device}, "
        f"window_size={window_size}, predictor=mamba"
    )

    reconstructed = []
    errors = 0
    start_ts = time.time()
    meta_checked = False
    expected_meta = {
        "num_keypoints": int(num_keypoints),
        "model_in_dim": int(model_in_dim),
        "pose_dims": int(num_keypoints * model_in_dim),
        "history_len": int(history_len),
        "window_size": int(window_size),
        "gnn_hidden": int(gnn_hidden),
        "mamba_d_state": int(mamba_d_state),
        "mamba_d_conv": int(mamba_d_conv),
        "mamba_expand": int(mamba_expand),
        "mamba_n_layer": int(mamba_n_layer),
        "i_frame_interval": int(i_frame_interval),
        "quantize_i_frame": int(quantize_i_frame),
        "quantize_p_frame": int(quantize_p_frame),
        "quant_scale": float(quant_scale),
        "entropy_enabled": int(entropy_enabled),
        "entropy_level": int(entropy_level),
    }

    try:
        while True:
            if servicer.receive_data_buffer.get_size() == 0:
                time.sleep(0.001)
                continue

            req = servicer.receive_data_buffer.get_items()
            req_ext_data = bytes(getattr(req, "extData", b""))
            if req_ext_data != PACKET_TAG:
                continue

            try:
                req_ext_desc = getattr(req, "extDesc", "")
                req_point_data = bytes(getattr(req, "pointData", b""))
                meta = json.loads(req_ext_desc) if req_ext_desc else {}

                if not meta_checked:
                    validate_stream_meta_once(meta, expected_meta)
                    meta_checked = True
                    print(f"decoder_meta_check: passed {expected_meta}")

                frame = decoder.decode_one(req_point_data, meta)
                reconstructed.append(frame)

                if debug and decoder.decoded_count % 30 == 0:
                    print(
                        f"[decoder] decoded={decoder.decoded_count}, "
                        f"frame_idx={meta.get('frame_idx', -1)}, "
                        f"mean_abs={float(np.mean(np.abs(frame))):.6f}"
                    )

                if max_frames is not None and decoder.decoded_count >= max_frames:
                    break
            except Exception as exc:
                if isinstance(exc, MetaCheckError):
                    raise
                errors += 1
                if debug:
                    print(f"[decoder] decode error: {exc}")
    except KeyboardInterrupt:
        print("decoder_stopped: keyboard interrupt")
    finally:
        # 显式关闭 gRPC 服务，避免解释器退出时后台线程报错
        stop_event = grpc_server.stop(grace=0.5)
        stop_event.wait(timeout=2.0)

    elapsed = time.time() - start_ts

    if output_path and reconstructed:
        save_path = output_path
        if output_path.endswith("/") or os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)
            save_path = os.path.join(output_path, "pose_recon_codec_h36m.npy")
        else:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

        arr = np.stack(reconstructed, axis=0)
        if save_path.endswith(".npy"):
            np.save(save_path, arr.reshape(arr.shape[0], num_keypoints, model_in_dim))
        else:
            np.savetxt(save_path, arr, fmt="%.6f", delimiter=",")
        print(f"decoder_saved: path={save_path}, shape={arr.shape}")

    print(
        f"decoder_done: decoded={decoder.decoded_count}, errors={errors}, "
        f"elapsed_s={elapsed:.3f}, fps={(decoder.decoded_count / max(elapsed, 1e-6)):.2f}"
    )


def main() -> None:
    # ===== 模型超参数（程序内统一定义，不通过命令行传递） =====
    MODEL_HPARAMS = {
        # 每帧关键点数量（Human3.6M: 17）
        "num_keypoints": 17,
        # 输入关键点维度，当前姿态是xyz三维
        "model_in_dim": 3,
        # 时序历史长度
        "history_len": 8,
        # GNN隐藏维度，同时作为Mamba的d_model（质量优先：更大容量）
        "gnn_hidden": 128,
        # Mamba状态维度（质量优先：更强记忆）
        "mamba_d_state": 64,
        # Mamba局部卷积宽度（当前 causal_conv1d 约束为 2~4）
        "mamba_d_conv": 4,
        # Mamba通道扩展倍率
        "mamba_expand": 4,
        # Mamba堆叠层数
        "mamba_n_layer": 4,
        # I帧间隔（0表示仅冷启动I帧；>0表示按间隔插入I帧）
        "i_frame_interval": 8,
        # 是否量化I帧
        "quantize_i_frame": 0,
        # 是否量化P帧（残差）
        "quantize_p_frame": 0,
        # 量化比例（q = round(x * quant_scale)）
        "quant_scale": 1000.0,
        # 是否开启熵编码
        "entropy_enabled": 0,
        # zlib压缩级别（0~9）
        "entropy_level": 6,
        # 模型初始化种子，编解码两端需一致
        "model_seed": 2026,
    }

    parser = argparse.ArgumentParser(description="Mamba pose decoder gRPC receiver")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/ztw/HVCCS/res/decode_res/pose_recon_codec_h36m.npy",
    )
    args = parser.parse_args()

    if args.window_size != MODEL_HPARAMS["history_len"]:
        raise ValueError(
            f"window_size({args.window_size}) 必须与 history_len({MODEL_HPARAMS['history_len']}) 一致"
        )

    if not os.path.exists(DEFAULT_CHECKPOINT_PATH):
        raise FileNotFoundError(f"checkpoint not found: {DEFAULT_CHECKPOINT_PATH}")

    run_receiver_service(
        host=args.host,
        port=args.port,
        window_size=args.window_size,
        num_keypoints=MODEL_HPARAMS["num_keypoints"],
        model_in_dim=MODEL_HPARAMS["model_in_dim"],
        history_len=MODEL_HPARAMS["history_len"],
        gnn_hidden=MODEL_HPARAMS["gnn_hidden"],
        mamba_d_state=MODEL_HPARAMS["mamba_d_state"],
        mamba_d_conv=MODEL_HPARAMS["mamba_d_conv"],
        mamba_expand=MODEL_HPARAMS["mamba_expand"],
        mamba_n_layer=MODEL_HPARAMS["mamba_n_layer"],
        i_frame_interval=MODEL_HPARAMS["i_frame_interval"],
        quantize_i_frame=MODEL_HPARAMS["quantize_i_frame"],
        quantize_p_frame=MODEL_HPARAMS["quantize_p_frame"],
        quant_scale=MODEL_HPARAMS["quant_scale"],
        entropy_enabled=MODEL_HPARAMS["entropy_enabled"],
        entropy_level=MODEL_HPARAMS["entropy_level"],
        model_seed=MODEL_HPARAMS["model_seed"],
        cuda_device=args.cuda_device,
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
        max_frames=args.max_frames,
        debug=args.debug,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
