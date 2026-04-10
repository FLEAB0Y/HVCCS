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
DEFAULT_CODEC_CONFIG_PATH = "/home/ztw/HVCCS/checkpoints/pose_codec_config.json"

MODEL_HPARAM_KEYS = (
    "num_keypoints",
    "model_in_dim",
    "history_len",
    "gnn_hidden",
    "mamba_d_state",
    "mamba_d_conv",
    "mamba_expand",
    "mamba_n_layer",
    "i_frame_interval",
    "quantize_i_frame",
    "quantize_p_frame",
    "quant_scale",
    "entropy_enabled",
    "entropy_level",
    "model_seed",
)

DECODER_RUNTIME_KEYS = (
    "host",
    "port",
    "window_size",
    "cuda_device",
    "max_frames",
    "debug",
    "output_path",
)


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "on"}:
            return True
        if v in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"invalid bool value: {value}")


def load_runtime_codec_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"codec_config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    print(f"loaded_codec_config: {config_path}")

    if not isinstance(config, dict):
        raise ValueError("codec config root must be a JSON object")

    missing_top_keys = [
        key for key in ("checkpoint_path", "model_hparams", "decoder_runtime") if key not in config
    ]
    if missing_top_keys:
        raise ValueError(f"codec config missing required keys: {missing_top_keys}")

    checkpoint_path = str(config["checkpoint_path"]).strip()
    if not checkpoint_path:
        raise ValueError("checkpoint_path in codec config is empty")

    config_hparams = config["model_hparams"]
    if not isinstance(config_hparams, dict):
        raise ValueError("model_hparams in codec config must be a JSON object")
    missing_hparam_keys = [key for key in MODEL_HPARAM_KEYS if key not in config_hparams]
    if missing_hparam_keys:
        raise ValueError(f"model_hparams missing required keys: {missing_hparam_keys}")

    model_hparams = {
        "num_keypoints": int(config_hparams["num_keypoints"]),
        "model_in_dim": int(config_hparams["model_in_dim"]),
        "history_len": int(config_hparams["history_len"]),
        "gnn_hidden": int(config_hparams["gnn_hidden"]),
        "mamba_d_state": int(config_hparams["mamba_d_state"]),
        "mamba_d_conv": int(config_hparams["mamba_d_conv"]),
        "mamba_expand": int(config_hparams["mamba_expand"]),
        "mamba_n_layer": int(config_hparams["mamba_n_layer"]),
        "i_frame_interval": int(config_hparams["i_frame_interval"]),
        "quantize_i_frame": int(_to_bool(config_hparams["quantize_i_frame"])),
        "quantize_p_frame": int(_to_bool(config_hparams["quantize_p_frame"])),
        "quant_scale": float(config_hparams["quant_scale"]),
        "entropy_enabled": int(_to_bool(config_hparams["entropy_enabled"])),
        "entropy_level": int(config_hparams["entropy_level"]),
        "model_seed": int(config_hparams["model_seed"]),
    }

    config_decoder_runtime = config["decoder_runtime"]
    if not isinstance(config_decoder_runtime, dict):
        raise ValueError("decoder_runtime in codec config must be a JSON object")
    missing_decoder_runtime_keys = [
        key for key in DECODER_RUNTIME_KEYS if key not in config_decoder_runtime
    ]
    if missing_decoder_runtime_keys:
        raise ValueError(
            f"decoder_runtime missing required keys: {missing_decoder_runtime_keys}"
        )

    decoder_runtime = {
        "host": str(config_decoder_runtime["host"]),
        "port": int(config_decoder_runtime["port"]),
        "window_size": int(config_decoder_runtime["window_size"]),
        "cuda_device": int(config_decoder_runtime["cuda_device"]),
        "max_frames": int(config_decoder_runtime["max_frames"]),
        "debug": _to_bool(config_decoder_runtime["debug"]),
        "output_path": str(config_decoder_runtime["output_path"]),
    }

    if decoder_runtime["window_size"] < 1:
        raise ValueError(
            f"decoder_runtime.window_size must be >= 1, got {decoder_runtime['window_size']}"
        )
    if decoder_runtime["max_frames"] < 1:
        raise ValueError(
            f"decoder_runtime.max_frames must be >= 1, got {decoder_runtime['max_frames']}"
        )
    if not decoder_runtime["output_path"].strip():
        raise ValueError("decoder_runtime.output_path is empty")

    return {
        "checkpoint_path": checkpoint_path,
        "model_hparams": model_hparams,
        "decoder_runtime": decoder_runtime,
    }


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
    runtime_cfg = load_runtime_codec_config(DEFAULT_CODEC_CONFIG_PATH)
    model_hparams = runtime_cfg["model_hparams"]
    decoder_runtime = runtime_cfg["decoder_runtime"]
    checkpoint_path = runtime_cfg["checkpoint_path"]

    if decoder_runtime["window_size"] != model_hparams["history_len"]:
        raise ValueError(
            "decoder_runtime.window_size("
            f"{decoder_runtime['window_size']}) 必须与 history_len({model_hparams['history_len']}) 一致"
        )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    run_receiver_service(
        host=decoder_runtime["host"],
        port=decoder_runtime["port"],
        window_size=decoder_runtime["window_size"],
        num_keypoints=model_hparams["num_keypoints"],
        model_in_dim=model_hparams["model_in_dim"],
        history_len=model_hparams["history_len"],
        gnn_hidden=model_hparams["gnn_hidden"],
        mamba_d_state=model_hparams["mamba_d_state"],
        mamba_d_conv=model_hparams["mamba_d_conv"],
        mamba_expand=model_hparams["mamba_expand"],
        mamba_n_layer=model_hparams["mamba_n_layer"],
        i_frame_interval=model_hparams["i_frame_interval"],
        quantize_i_frame=model_hparams["quantize_i_frame"],
        quantize_p_frame=model_hparams["quantize_p_frame"],
        quant_scale=model_hparams["quant_scale"],
        entropy_enabled=model_hparams["entropy_enabled"],
        entropy_level=model_hparams["entropy_level"],
        model_seed=model_hparams["model_seed"],
        cuda_device=decoder_runtime["cuda_device"],
        checkpoint_path=checkpoint_path,
        max_frames=decoder_runtime["max_frames"],
        debug=decoder_runtime["debug"],
        output_path=decoder_runtime["output_path"],
    )


if __name__ == "__main__":
    main()
