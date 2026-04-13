import argparse
import csv
import json
import os
import random
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from mamba_ssm import Mamba
except Exception:
    Mamba = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


H36M_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]


def to_tjc_from_npy(pose: np.ndarray, coord_dims: int = 3) -> np.ndarray:
    arr = np.asarray(pose)
    arr = np.squeeze(arr)

    if arr.ndim == 3:
        if arr.shape[-1] >= coord_dims:
            return arr[..., :coord_dims]
        if arr.shape[1] >= coord_dims:
            return np.transpose(arr, (0, 2, 1))[..., :coord_dims]
        raise ValueError(f"Cannot infer npy pose layout from shape {arr.shape}")

    if arr.ndim == 2:
        if arr.shape[1] % coord_dims == 0:
            joints = arr.shape[1] // coord_dims
            return arr.reshape(arr.shape[0], joints, coord_dims)
        if arr.shape[0] % coord_dims == 0:
            joints = arr.shape[0] // coord_dims
            return arr.T.reshape(arr.shape[1], joints, coord_dims)
        raise ValueError(f"Cannot infer npy pose layout from shape {arr.shape}")

    if arr.ndim == 1:
        if arr.shape[0] % coord_dims != 0:
            raise ValueError(f"Cannot infer npy pose layout from shape {arr.shape}")
        joints = arr.shape[0] // coord_dims
        return arr.reshape(1, joints, coord_dims)

    raise ValueError(f"Unsupported npy pose shape: {arr.shape}")


def build_adjacency_matrix(num_nodes: int = 17) -> torch.Tensor:
    adj = np.eye(num_nodes, dtype=np.float32)
    if num_nodes == 17:
        for i, j in H36M_CONNECTIONS:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    else:
        for i in range(num_nodes - 1):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0

    rowsum = adj.sum(axis=1, keepdims=True)
    adj = adj / np.clip(rowsum, 1e-8, None)
    return torch.tensor(adj, dtype=torch.float32)


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return torch.matmul(adj, x)


class GNNIntraFrameEncoder(nn.Module):
    def __init__(self, in_dim: int = 3, hidden_dim: int = 16, num_nodes: int = 17):
        super().__init__()
        self.gcn1 = GraphConvolution(in_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.register_buffer("adj", build_adjacency_matrix(num_nodes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.gcn1(x, self.adj))
        h = self.relu(self.gcn2(h, self.adj))
        return h


class MambaResidualBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba_ssm is required. Install it with: pip install mamba-ssm")
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.mamba(x)
        return self.norm(x + residual)


class SpatioTemporalPredictor(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: Optional[int] = None,
        gnn_hidden: int = 16,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_n_layer: int = 1,
        num_nodes: int = 17,
    ):
        super().__init__()
        self.gnn = GNNIntraFrameEncoder(in_dim, gnn_hidden, num_nodes=num_nodes)
        if mamba_n_layer < 1:
            raise ValueError("mamba_n_layer must be >= 1")

        self.mamba_blocks = nn.ModuleList(
            [
                MambaResidualBlock(
                    d_model=gnn_hidden,
                    d_state=mamba_d_state,
                    d_conv=mamba_d_conv,
                    expand=mamba_expand,
                )
                for _ in range(mamba_n_layer)
            ]
        )
        self.mamba = self.mamba_blocks

        if out_dim is None:
            out_dim = in_dim
        self.predictor = nn.Linear(gnn_hidden, out_dim)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, num_nodes, coord_dims = x_seq.shape
        x_flat = x_seq.view(bsz * seq_len, num_nodes, coord_dims)

        gnn_out = self.gnn(x_flat)
        gnn_out = gnn_out.view(bsz, seq_len, num_nodes, -1).permute(0, 2, 1, 3).reshape(bsz * num_nodes, seq_len, -1)

        mamba_out = gnn_out
        for block in self.mamba_blocks:
            mamba_out = block(mamba_out)

        last_step_features = mamba_out[:, -1, :]
        pred_out = self.predictor(last_step_features)
        return pred_out.view(bsz, num_nodes, self.predictor.out_features)


DEFAULT_CHECKPOINT_PATH = "/home/ztw/HVCCS/checkpoints/pose_mamba_init_k17_d3_o6_w8_f1_h256_ds128_dc4_ex4_nl8_s2026.pt"
DEFAULT_SPLINE_DATA_ROOT = "/home/data/ztw/AtheletePose3D/h36m_pose_cam_1"
DEFAULT_TRAIN_GT_ROOTS = [
    f"{DEFAULT_SPLINE_DATA_ROOT}/train/S1_cam_1_60fps_notaknot_splines",
    f"{DEFAULT_SPLINE_DATA_ROOT}/train/S2_cam_1_60fps_notaknot_splines",
    f"{DEFAULT_SPLINE_DATA_ROOT}/train/S5_cam_1_60fps_notaknot_splines",
    f"{DEFAULT_SPLINE_DATA_ROOT}/train/S3_cam_1_120fps_notaknot_splines",
    f"{DEFAULT_SPLINE_DATA_ROOT}/train/S4_cam_1_120fps_notaknot_splines",
]
DEFAULT_VALID_GT_ROOTS = [
    f"{DEFAULT_SPLINE_DATA_ROOT}/val/S1_cam_1_60fps_notaknot_splines",
    f"{DEFAULT_SPLINE_DATA_ROOT}/val/S3_cam_1_60fps_notaknot_splines",
    f"{DEFAULT_SPLINE_DATA_ROOT}/valid/S2_cam_1_120fps_notaknot_splines",
]
DEFAULT_TEST_GT_ROOTS = [
    f"{DEFAULT_SPLINE_DATA_ROOT}/test/S1_cam_1_60fps_notaknot_splines",
    f"{DEFAULT_SPLINE_DATA_ROOT}/test/S3_cam_1_60fps_notaknot_splines",
    f"{DEFAULT_SPLINE_DATA_ROOT}/test/S2_cam_1_120fps_notaknot_splines",
]
DEFAULT_OUTPUT_ROOT = "/home/ztw/HVCCS/checkpoints/splines_mamba_runs"


TRAIN_HPARAMS = {
    "train_gt_roots": DEFAULT_TRAIN_GT_ROOTS,
    "valid_gt_roots": DEFAULT_VALID_GT_ROOTS,
    "test_gt_roots": DEFAULT_TEST_GT_ROOTS,
    "checkpoint_path": DEFAULT_CHECKPOINT_PATH,
    "output_root": DEFAULT_OUTPUT_ROOT,
    "history_len": 8,
    "loss_crmse_weight": 1.0,
    "loss_vel_rmse_weight": 0.3,
    "loss_acc_rmse_weight": 0.05,
    "epochs": 20,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "num_workers": 0,
    "grad_clip": 1.0,
    "seed": 2026,
    "cuda_device": 0,
    "max_gt_files": 0,
    "cache_max_files": 6,
    "log_interval": 100,
    "save_every": 5,
    "run_name": "",
    "enable_tensorboard": True,
    "tb_logdir": "",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_cuda_arch_supported(cuda_device: int) -> bool:
    compiled_arches = set(torch.cuda.get_arch_list())
    if not compiled_arches:
        return True
    major, minor = torch.cuda.get_device_capability(cuda_device)
    return f"sm_{major}{minor}" in compiled_arches


def select_torch_device(cuda_device: int) -> torch.device:
    if cuda_device < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    if cuda_device >= torch.cuda.device_count():
        return torch.device("cpu")
    try:
        if not _is_cuda_arch_supported(cuda_device):
            major, minor = torch.cuda.get_device_capability(cuda_device)
            print(f"检测到当前 GPU 架构 sm_{major}{minor} 不受当前 PyTorch 支持，自动回退到 CPU")
            return torch.device("cpu")
    except Exception as exc:
        print(f"CUDA 架构检查失败({exc})，自动回退到 CPU")
        return torch.device("cpu")
    return torch.device(f"cuda:{cuda_device}")


def derive_sample_root_from_gt_root(gt_root: str) -> str:
    suffix = "_notaknot_splines"
    gt_root = str(gt_root)
    if not gt_root.endswith(suffix):
        raise ValueError(
            f"gt root must end with '{suffix}', got: {gt_root}. "
            "样本路径推导规则是删去该后缀。"
        )
    return gt_root[: -len(suffix)]


def infer_dt_from_time_axis(time_sec: np.ndarray) -> float:
    if time_sec.ndim != 1 or len(time_sec) < 2:
        raise ValueError(f"invalid time_sec shape: {time_sec.shape}")
    dts = np.diff(time_sec.astype(np.float64))
    if np.any(dts <= 0):
        raise ValueError("time_sec must be strictly increasing")
    return float(np.median(dts))


def extract_pair_metadata(
    gt_path: str,
    sample_path: str,
    history_len: int,
    num_keypoints: int,
    coord_dims: int,
) -> Optional[Dict]:
    pose_arr = np.load(sample_path)
    pose_seq = to_tjc_from_npy(pose_arr, coord_dims=coord_dims)
    if pose_seq.shape[1] != num_keypoints:
        raise ValueError(
            f"joint mismatch in {sample_path}: got {pose_seq.shape[1]}, expected {num_keypoints}"
        )

    with np.load(gt_path, allow_pickle=False) as gt_data:
        if "coeffs" not in gt_data.files or "time_sec" not in gt_data.files:
            raise KeyError(f"missing coeffs/time_sec in gt file: {gt_path}")
        coeffs_shape = gt_data["coeffs"].shape
        time_sec = gt_data["time_sec"].astype(np.float64)
        fps = float(gt_data["fps"][0]) if "fps" in gt_data.files else float("nan")

    if len(coeffs_shape) != 4 or coeffs_shape[2] != 4:
        raise ValueError(f"invalid coeffs shape in {gt_path}: {coeffs_shape}")
    if coeffs_shape[0] != num_keypoints or coeffs_shape[1] != coord_dims:
        raise ValueError(
            f"coeff shape mismatch in {gt_path}: got {coeffs_shape[:2]}, "
            f"expected {(num_keypoints, coord_dims)}"
        )

    dt = infer_dt_from_time_axis(time_sec)
    num_frames = int(pose_seq.shape[0])
    num_segments = int(coeffs_shape[3])

    min_k = int(history_len - 1)
    max_k = int(min(num_frames - 2, num_segments - 1))
    if max_k < min_k:
        return None

    return {
        "gt_path": str(gt_path),
        "sample_path": str(sample_path),
        "num_frames": num_frames,
        "num_segments": num_segments,
        "num_samples": int(max_k - min_k + 1),
        "min_k": min_k,
        "max_k": max_k,
        "dt": float(dt),
        "fps": fps,
    }


def discover_spline_pairs(
    gt_roots: Sequence[str],
    history_len: int,
    num_keypoints: int,
    coord_dims: int,
    max_gt_files: int = 0,
    allow_empty: bool = False,
) -> Tuple[List[Dict], Dict[str, int]]:
    pair_meta: List[Dict] = []
    total_gt_files = 0
    missing_sample = 0
    invalid_pair = 0

    for gt_root_str in gt_roots:
        gt_root = Path(gt_root_str)
        if not gt_root.exists():
            raise FileNotFoundError(f"gt root not found: {gt_root_str}")

        sample_root = Path(derive_sample_root_from_gt_root(str(gt_root)))
        if not sample_root.exists():
            raise FileNotFoundError(f"sample root not found (derived from gt root): {sample_root}")

        gt_files = sorted(gt_root.rglob("*_notaknot_spline.npz"))
        for gt_file in gt_files:
            total_gt_files += 1
            rel = gt_file.relative_to(gt_root)
            gt_name = gt_file.name
            suffix = "_notaknot_spline.npz"
            if not gt_name.endswith(suffix):
                continue
            sample_name = gt_name[: -len(suffix)] + ".npy"
            sample_file = sample_root / rel.parent / sample_name
            if not sample_file.exists():
                missing_sample += 1
                continue

            meta = extract_pair_metadata(
                gt_path=str(gt_file),
                sample_path=str(sample_file),
                history_len=history_len,
                num_keypoints=num_keypoints,
                coord_dims=coord_dims,
            )
            if meta is None:
                invalid_pair += 1
                continue

            meta["gt_root"] = str(gt_root)
            meta["sample_root"] = str(sample_root)
            pair_meta.append(meta)

            if max_gt_files > 0 and len(pair_meta) >= max_gt_files:
                break

        if max_gt_files > 0 and len(pair_meta) >= max_gt_files:
            break

    if not pair_meta and not allow_empty:
        raise RuntimeError(f"No valid spline pairs found from roots: {list(gt_roots)}")

    stats = {
        "total_gt_files": total_gt_files,
        "paired_files": len(pair_meta),
        "missing_sample": missing_sample,
        "invalid_pair": invalid_pair,
    }
    return pair_meta, stats


def load_pose_sequence(sample_path: str, coord_dims: int, num_keypoints: int) -> np.ndarray:
    arr = np.load(sample_path)
    seq = to_tjc_from_npy(arr, coord_dims=coord_dims)
    seq = seq[:, :num_keypoints, :coord_dims].astype(np.float32, copy=False)
    return seq


def load_spline_coeffs(gt_path: str, num_keypoints: int, coord_dims: int) -> np.ndarray:
    with np.load(gt_path, allow_pickle=False) as data:
        coeffs = data["coeffs"].astype(np.float32)
    if coeffs.shape[0] != num_keypoints or coeffs.shape[1] != coord_dims or coeffs.shape[2] != 4:
        raise ValueError(
            f"coeff shape mismatch in {gt_path}: {coeffs.shape}, expected ({num_keypoints}, {coord_dims}, 4, S)"
        )
    return coeffs


class SplineSegmentDataset(Dataset):
    def __init__(
        self,
        pair_meta: Sequence[Dict],
        history_len: int,
        num_keypoints: int,
        coord_dims: int,
        cache_max_files: int = 6,
    ) -> None:
        self.pair_meta = list(pair_meta)
        self.history_len = int(history_len)
        self.num_keypoints = int(num_keypoints)
        self.coord_dims = int(coord_dims)
        self.cache_max_files = int(max(cache_max_files, 1))
        self._cache: OrderedDict[str, Tuple[np.ndarray, np.ndarray]] = OrderedDict()

        self.samples: List[Tuple[int, int]] = []
        for pair_idx, meta in enumerate(self.pair_meta):
            self.samples.extend((pair_idx, k) for k in range(meta["min_k"], meta["max_k"] + 1))

        if not self.samples:
            raise RuntimeError("No valid segment samples were constructed")

    def __len__(self) -> int:
        return len(self.samples)

    def _get_pair_arrays(self, pair_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        meta = self.pair_meta[pair_idx]
        key = meta["gt_path"]
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        pose_seq = load_pose_sequence(meta["sample_path"], self.coord_dims, self.num_keypoints)
        coeffs = load_spline_coeffs(meta["gt_path"], self.num_keypoints, self.coord_dims)

        self._cache[key] = (pose_seq, coeffs)
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_max_files:
            self._cache.popitem(last=False)
        return pose_seq, coeffs

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pair_idx, k = self.samples[index]
        meta = self.pair_meta[pair_idx]
        pose_seq, coeffs = self._get_pair_arrays(pair_idx)

        start = k - self.history_len + 1
        history = pose_seq[start : k + 1]
        target_coeff = coeffs[:, :, :, k]
        dt = np.float32(meta["dt"])

        return (
            torch.from_numpy(history.copy()),
            torch.from_numpy(target_coeff.copy()),
            torch.tensor(dt, dtype=torch.float32),
        )


def estimate_velocity_from_history(history: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    # history: (B, H, J, C), dt: (B,)
    _, history_len, _, _ = history.shape
    t = torch.arange(history_len, device=history.device, dtype=history.dtype)
    t = t - t.mean()
    denom = torch.sum(t * t).clamp_min(1e-8)
    slope_per_frame = (history * t.view(1, history_len, 1, 1)).sum(dim=1) / denom
    return slope_per_frame / dt.view(-1, 1, 1).clamp_min(1e-8)


def cubic_hermite_coefficients_torch(
    x0: torch.Tensor,
    v0: torch.Tensor,
    x1: torch.Tensor,
    v1: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    dtv = dt.view(-1, 1, 1).clamp_min(1e-8)
    a = (2.0 * (x0 - x1) + dtv * (v0 + v1)) / (dtv ** 3)
    b = (3.0 * (x1 - x0) - dtv * (2.0 * v0 + v1)) / (dtv ** 2)
    c = v0
    d = x0
    return torch.stack([a, b, c, d], dim=-1)


def spline_rmse_metrics(
    pred_coeff: torch.Tensor,
    target_coeff: torch.Tensor,
    dt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 对齐 tools/splines_metrics.py 的 cRMSE / VelRMSE / AccRMSE 定义。
    diff = pred_coeff - target_coeff
    a = diff[..., 0]
    b = diff[..., 1]
    c = diff[..., 2]
    d = diff[..., 3]
    dtv = dt.view(-1, 1, 1).clamp_min(1e-8)

    int_pos = (
        (a * a) * (dtv ** 7) / 7.0
        + (a * b) * (dtv ** 6) / 3.0
        + (2.0 * a * c + b * b) * (dtv ** 5) / 5.0
        + (a * d + b * c) * (dtv ** 4) / 2.0
        + (2.0 * b * d + c * c) * (dtv ** 3) / 3.0
        + (c * d) * (dtv ** 2)
        + (d * d) * dtv
    )

    int_vel = (
        9.0 * (a * a) * (dtv ** 5) / 5.0
        + 3.0 * (a * b) * (dtv ** 4)
        + (4.0 * b * b + 6.0 * a * c) * (dtv ** 3) / 3.0
        + 2.0 * (b * c) * (dtv ** 2)
        + (c * c) * dtv
    )

    int_acc = 12.0 * (a * a) * (dtv ** 3) + 12.0 * (a * b) * (dtv ** 2) + 4.0 * (b * b) * dtv

    duration = dt.clamp_min(1e-8)
    pos_mse = int_pos.mean(dim=(1, 2)) / duration
    vel_mse = int_vel.mean(dim=(1, 2)) / duration
    acc_mse = int_acc.mean(dim=(1, 2)) / duration

    crmse = torch.sqrt(pos_mse.clamp_min(1e-12))
    vel_rmse = torch.sqrt(vel_mse.clamp_min(1e-12))
    acc_rmse = torch.sqrt(acc_mse.clamp_min(1e-12))
    return crmse, vel_rmse, acc_rmse


def compute_weighted_spline_loss(
    pred_coeff: torch.Tensor,
    target_coeff: torch.Tensor,
    dt: torch.Tensor,
    w_crmse: float,
    w_vel: float,
    w_acc: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    crmse, vel_rmse, acc_rmse = spline_rmse_metrics(pred_coeff, target_coeff, dt)
    per_sample = w_crmse * crmse + w_vel * vel_rmse + w_acc * acc_rmse
    loss = per_sample.mean()
    metrics = {
        "loss": float(loss.detach().item()),
        "cRMSE_mm": float(crmse.mean().detach().item()),
        "VelRMSE_mmps": float(vel_rmse.mean().detach().item()),
        "AccRMSE_mmps2": float(acc_rmse.mean().detach().item()),
    }
    return loss, metrics


def run_epoch(
    model: SpatioTemporalPredictor,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    coord_dims: int,
    w_crmse: float,
    w_vel: float,
    w_acc: float,
    train: bool,
    log_interval: int,
) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    sums = {
        "loss": 0.0,
        "cRMSE_mm": 0.0,
        "VelRMSE_mmps": 0.0,
        "AccRMSE_mmps2": 0.0,
    }
    total_count = 0

    for step, (history, target_coeff, dt) in enumerate(loader, start=1):
        history = history.to(device, non_blocking=True)
        target_coeff = target_coeff.to(device, non_blocking=True)
        dt = dt.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            pred = model(history)
            expected_out_dim = 2 * coord_dims
            if pred.shape[-1] < expected_out_dim:
                raise ValueError(
                    f"model output dim is {pred.shape[-1]}, but spline training needs at least {expected_out_dim} (xyz+vxyz)"
                )

            pred_next_pos = pred[..., :coord_dims]
            pred_next_vel = pred[..., coord_dims:expected_out_dim]
            xk = history[:, -1]
            vk = estimate_velocity_from_history(history, dt)
            pred_coeff = cubic_hermite_coefficients_torch(xk, vk, pred_next_pos, pred_next_vel, dt)

            loss, batch_metrics = compute_weighted_spline_loss(
                pred_coeff=pred_coeff,
                target_coeff=target_coeff,
                dt=dt,
                w_crmse=w_crmse,
                w_vel=w_vel,
                w_acc=w_acc,
            )

            if train:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

        batch_size = int(history.size(0))
        total_count += batch_size
        for key in sums:
            sums[key] += batch_metrics[key] * batch_size

        if log_interval > 0 and step % log_interval == 0:
            phase = "train" if train else "val"
            print(
                f"[{phase}] step={step}, loss={batch_metrics['loss']:.6f}, "
                f"cRMSE={batch_metrics['cRMSE_mm']:.6f}, "
                f"VelRMSE={batch_metrics['VelRMSE_mmps']:.6f}, "
                f"AccRMSE={batch_metrics['AccRMSE_mmps2']:.6f}"
            )

    if total_count <= 0:
        return {key: float("nan") for key in sums}
    return {key: value / total_count for key, value in sums.items()}


def save_epoch_metrics_csv(csv_path: str, records: List[Dict]) -> None:
    if not records:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = list(records[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def save_loss_curve_png(png_path: str, records: List[Dict]) -> None:
    if plt is None:
        return
    if not records:
        return
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    epochs = [int(r["epoch"]) for r in records]
    train_loss = [float(r["train_loss"]) for r in records]
    val_loss = [float(r["val_loss"]) if r["val_loss"] != "" else np.nan for r in records]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="train_weighted_spline_loss", linewidth=2.0)
    ax.plot(epochs, val_loss, label="val_weighted_spline_loss", linewidth=2.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted Spline Loss")
    ax.set_title("Spline Training Convergence")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def write_manifest_csv(csv_path: str, pair_meta: Sequence[Dict]) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = [
        "gt_root",
        "sample_root",
        "gt_path",
        "sample_path",
        "fps",
        "dt",
        "num_frames",
        "num_segments",
        "num_samples",
        "min_k",
        "max_k",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for meta in pair_meta:
            writer.writerow({k: meta.get(k, "") for k in fieldnames})


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[SpatioTemporalPredictor, Dict]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        meta = dict(ckpt.get("meta", {}))
    else:
        state_dict = ckpt
        meta = {}

    num_keypoints = int(meta.get("num_keypoints", 17))
    model_in_dim = int(meta.get("model_in_dim", 3))
    model_out_dim = int(meta.get("model_out_dim", model_in_dim))
    gnn_hidden = int(meta.get("gnn_hidden", 128))
    mamba_d_state = int(meta.get("mamba_d_state", 64))
    mamba_d_conv = int(meta.get("mamba_d_conv", 4))
    mamba_expand = int(meta.get("mamba_expand", 4))
    mamba_n_layer = int(meta.get("mamba_n_layer", 4))

    model = SpatioTemporalPredictor(
        in_dim=model_in_dim,
        out_dim=model_out_dim,
        gnn_hidden=gnn_hidden,
        mamba_d_state=mamba_d_state,
        mamba_d_conv=mamba_d_conv,
        mamba_expand=mamba_expand,
        mamba_n_layer=mamba_n_layer,
        num_nodes=num_keypoints,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)

    return model, {
        "num_keypoints": num_keypoints,
        "model_in_dim": model_in_dim,
        "model_out_dim": model_out_dim,
        "gnn_hidden": gnn_hidden,
        "mamba_d_state": mamba_d_state,
        "mamba_d_conv": mamba_d_conv,
        "mamba_expand": mamba_expand,
        "mamba_n_layer": mamba_n_layer,
    }


def check_runtime_backend(model: SpatioTemporalPredictor, device: torch.device, history_len: int, num_keypoints: int, coord_dims: int) -> None:
    if device.type != "cpu":
        return
    # Some mamba/causal-conv builds are CUDA-only. Probe once to fail fast with a clear message.
    try:
        with torch.no_grad():
            dummy = torch.zeros((1, history_len, num_keypoints, coord_dims), dtype=torch.float32, device=device)
            _ = model(dummy)
    except RuntimeError as exc:
        msg = str(exc)
        if "Expected x.is_cuda() to be true" in msg or "causal_conv1d" in msg:
            raise RuntimeError(
                "当前 mamba_ssm/causal_conv1d 后端为 CUDA-only，CPU 无法前向。"
                "请使用 --cuda_device >= 0 并确保 CUDA 可用，或安装支持 CPU 的后端版本。"
            ) from exc
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Mamba predictor with spline-curve supervision")
    parser.add_argument("--train_gt_roots", nargs="+", default=TRAIN_HPARAMS["train_gt_roots"])
    parser.add_argument("--valid_gt_roots", nargs="*", default=TRAIN_HPARAMS["valid_gt_roots"])
    parser.add_argument("--test_gt_roots", nargs="*", default=TRAIN_HPARAMS["test_gt_roots"])
    parser.add_argument("--checkpoint_path", type=str, default=TRAIN_HPARAMS["checkpoint_path"])
    parser.add_argument("--output_root", type=str, default=TRAIN_HPARAMS["output_root"])
    parser.add_argument("--history_len", type=int, default=TRAIN_HPARAMS["history_len"])
    parser.add_argument("--loss_crmse_weight", type=float, default=TRAIN_HPARAMS["loss_crmse_weight"])
    parser.add_argument("--loss_vel_rmse_weight", type=float, default=TRAIN_HPARAMS["loss_vel_rmse_weight"])
    parser.add_argument("--loss_acc_rmse_weight", type=float, default=TRAIN_HPARAMS["loss_acc_rmse_weight"])
    parser.add_argument("--epochs", type=int, default=TRAIN_HPARAMS["epochs"])
    parser.add_argument("--batch_size", type=int, default=TRAIN_HPARAMS["batch_size"])
    parser.add_argument("--lr", type=float, default=TRAIN_HPARAMS["lr"])
    parser.add_argument("--weight_decay", type=float, default=TRAIN_HPARAMS["weight_decay"])
    parser.add_argument("--num_workers", type=int, default=TRAIN_HPARAMS["num_workers"])
    parser.add_argument("--grad_clip", type=float, default=TRAIN_HPARAMS["grad_clip"])
    parser.add_argument("--seed", type=int, default=TRAIN_HPARAMS["seed"])
    parser.add_argument("--cuda_device", type=int, default=TRAIN_HPARAMS["cuda_device"])
    parser.add_argument("--max_gt_files", type=int, default=TRAIN_HPARAMS["max_gt_files"])
    parser.add_argument("--cache_max_files", type=int, default=TRAIN_HPARAMS["cache_max_files"])
    parser.add_argument("--log_interval", type=int, default=TRAIN_HPARAMS["log_interval"])
    parser.add_argument("--save_every", type=int, default=TRAIN_HPARAMS["save_every"])
    parser.add_argument("--run_name", type=str, default=TRAIN_HPARAMS["run_name"])
    parser.add_argument("--enable_tensorboard", type=int, default=int(TRAIN_HPARAMS["enable_tensorboard"]))
    parser.add_argument("--tb_logdir", type=str, default=TRAIN_HPARAMS["tb_logdir"])
    args = parser.parse_args()

    if args.save_every < 1:
        raise ValueError(f"save_every must be >= 1, got {args.save_every}")
    if args.history_len < 2:
        raise ValueError(f"history_len must be >= 2, got {args.history_len}")

    set_seed(args.seed)
    device = select_torch_device(args.cuda_device)
    print(f"train_device: {device}")

    model, model_hparams = load_model_from_checkpoint(args.checkpoint_path, device=device)
    print(f"loaded_checkpoint: {args.checkpoint_path}")
    print(f"model_hparams: {model_hparams}")

    num_keypoints = int(model_hparams["num_keypoints"])
    coord_dims = int(model_hparams["model_in_dim"])
    model_out_dim = int(model_hparams["model_out_dim"])
    expected_out_dim = 2 * coord_dims
    if model_out_dim < expected_out_dim:
        raise ValueError(
            f"model_out_dim={model_out_dim} is incompatible with direct velocity prediction; need at least {expected_out_dim}"
        )

    train_pairs, train_stats = discover_spline_pairs(
        gt_roots=args.train_gt_roots,
        history_len=args.history_len,
        num_keypoints=num_keypoints,
        coord_dims=coord_dims,
        max_gt_files=args.max_gt_files,
        allow_empty=False,
    )
    valid_pairs, valid_stats = discover_spline_pairs(
        gt_roots=args.valid_gt_roots,
        history_len=args.history_len,
        num_keypoints=num_keypoints,
        coord_dims=coord_dims,
        max_gt_files=args.max_gt_files,
        allow_empty=True,
    )
    test_pairs, test_stats = discover_spline_pairs(
        gt_roots=args.test_gt_roots,
        history_len=args.history_len,
        num_keypoints=num_keypoints,
        coord_dims=coord_dims,
        max_gt_files=args.max_gt_files,
        allow_empty=True,
    )

    if args.valid_gt_roots and not valid_pairs:
        raise RuntimeError(f"No validation pairs found under roots: {args.valid_gt_roots}")

    print(f"train_pair_stats: {train_stats}")
    print(f"valid_pair_stats: {valid_stats}")
    print(f"test_pair_stats: {test_stats}")

    train_set = SplineSegmentDataset(
        pair_meta=train_pairs,
        history_len=args.history_len,
        num_keypoints=num_keypoints,
        coord_dims=coord_dims,
        cache_max_files=args.cache_max_files,
    )
    val_set = (
        SplineSegmentDataset(
            pair_meta=valid_pairs,
            history_len=args.history_len,
            num_keypoints=num_keypoints,
            coord_dims=coord_dims,
            cache_max_files=args.cache_max_files,
        )
        if valid_pairs
        else None
    )
    test_set = (
        SplineSegmentDataset(
            pair_meta=test_pairs,
            history_len=args.history_len,
            num_keypoints=num_keypoints,
            coord_dims=coord_dims,
            cache_max_files=args.cache_max_files,
        )
        if test_pairs
        else None
    )

    print(f"train_samples: {len(train_set)}")
    print(f"valid_samples: {0 if val_set is None else len(val_set)}")
    print(f"test_samples: {0 if test_set is None else len(test_set)}")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        if val_set is not None
        else None
    )
    test_loader = (
        DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        if test_set is not None
        else None
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = args.run_name.strip() or f"spline_mamba_h{args.history_len}_{timestamp}"
    run_dir = os.path.join(args.output_root, run_prefix)
    ckpt_dir = os.path.join(run_dir, "ckpt")
    logs_dir = os.path.join(run_dir, "logs")
    manifest_dir = os.path.join(run_dir, "manifest")
    config_dir = os.path.join(run_dir, "config")
    for path in [ckpt_dir, logs_dir, manifest_dir, config_dir]:
        os.makedirs(path, exist_ok=True)

    write_manifest_csv(os.path.join(manifest_dir, "train_pairs.csv"), train_pairs)
    write_manifest_csv(os.path.join(manifest_dir, "val_pairs.csv"), valid_pairs)
    write_manifest_csv(os.path.join(manifest_dir, "test_pairs.csv"), test_pairs)

    run_config = {
        "args": vars(args),
        "model_hparams": model_hparams,
        "train_pair_stats": train_stats,
        "valid_pair_stats": valid_stats,
        "test_pair_stats": test_stats,
    }
    with open(os.path.join(config_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    metrics_csv_path = os.path.join(logs_dir, "epoch_metrics.csv")
    loss_png_path = os.path.join(logs_dir, "loss_curve.png")

    writer = None
    if bool(args.enable_tensorboard):
        if SummaryWriter is None:
            print("tensorboard 未安装，跳过在线标量写入。请先安装: pip install tensorboard")
        else:
            tb_root = args.tb_logdir.strip() or os.path.join(run_dir, "tb")
            os.makedirs(tb_root, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_root)
            print(f"tensorboard_logdir: {tb_root}")

    print(f"run_dir: {run_dir}")
    print(f"metrics_csv: {metrics_csv_path}")
    print(f"loss_png: {loss_png_path}")

    if args.epochs <= 0:
        print("epochs <= 0: 仅完成数据配对/样本构建检查，不执行训练与测试前向。")
        if writer is not None:
            writer.close()
        return

    check_runtime_backend(
        model=model,
        device=device,
        history_len=args.history_len,
        num_keypoints=num_keypoints,
        coord_dims=coord_dims,
    )

    best_metric = float("inf")
    best_path = os.path.join(ckpt_dir, "best.pt")
    last_path = os.path.join(ckpt_dir, "last.pt")
    epoch_records: List[Dict] = []
    last_ckpt = None

    for epoch in range(1, args.epochs + 1):
        train_stats_epoch = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
            coord_dims=coord_dims,
            w_crmse=args.loss_crmse_weight,
            w_vel=args.loss_vel_rmse_weight,
            w_acc=args.loss_acc_rmse_weight,
            train=True,
            log_interval=args.log_interval,
        )

        val_stats_epoch = None
        if val_loader is not None:
            val_stats_epoch = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=optimizer,
                grad_clip=args.grad_clip,
                coord_dims=coord_dims,
                w_crmse=args.loss_crmse_weight,
                w_vel=args.loss_vel_rmse_weight,
                w_acc=args.loss_acc_rmse_weight,
                train=False,
                log_interval=0,
            )

        msg = (
            f"epoch={epoch}/{args.epochs}, "
            f"train_loss={train_stats_epoch['loss']:.6f}, "
            f"train_cRMSE={train_stats_epoch['cRMSE_mm']:.6f}, "
            f"train_VelRMSE={train_stats_epoch['VelRMSE_mmps']:.6f}, "
            f"train_AccRMSE={train_stats_epoch['AccRMSE_mmps2']:.6f}"
        )
        if val_stats_epoch is not None:
            msg += (
                f", val_loss={val_stats_epoch['loss']:.6f}, "
                f"val_cRMSE={val_stats_epoch['cRMSE_mm']:.6f}, "
                f"val_VelRMSE={val_stats_epoch['VelRMSE_mmps']:.6f}, "
                f"val_AccRMSE={val_stats_epoch['AccRMSE_mmps2']:.6f}"
            )
        print(msg)

        row = {
            "epoch": epoch,
            "train_loss": f"{train_stats_epoch['loss']:.8f}",
            "train_cRMSE_mm": f"{train_stats_epoch['cRMSE_mm']:.8f}",
            "train_VelRMSE_mmps": f"{train_stats_epoch['VelRMSE_mmps']:.8f}",
            "train_AccRMSE_mmps2": f"{train_stats_epoch['AccRMSE_mmps2']:.8f}",
            "val_loss": "" if val_stats_epoch is None else f"{val_stats_epoch['loss']:.8f}",
            "val_cRMSE_mm": "" if val_stats_epoch is None else f"{val_stats_epoch['cRMSE_mm']:.8f}",
            "val_VelRMSE_mmps": "" if val_stats_epoch is None else f"{val_stats_epoch['VelRMSE_mmps']:.8f}",
            "val_AccRMSE_mmps2": "" if val_stats_epoch is None else f"{val_stats_epoch['AccRMSE_mmps2']:.8f}",
        }
        epoch_records.append(row)
        save_epoch_metrics_csv(metrics_csv_path, epoch_records)
        save_loss_curve_png(loss_png_path, epoch_records)

        if writer is not None:
            writer.add_scalar("loss/train", train_stats_epoch["loss"], epoch)
            writer.add_scalar("metric/train_cRMSE_mm", train_stats_epoch["cRMSE_mm"], epoch)
            writer.add_scalar("metric/train_VelRMSE_mmps", train_stats_epoch["VelRMSE_mmps"], epoch)
            writer.add_scalar("metric/train_AccRMSE_mmps2", train_stats_epoch["AccRMSE_mmps2"], epoch)
            if val_stats_epoch is not None:
                writer.add_scalar("loss/val", val_stats_epoch["loss"], epoch)
                writer.add_scalar("metric/val_cRMSE_mm", val_stats_epoch["cRMSE_mm"], epoch)
                writer.add_scalar("metric/val_VelRMSE_mmps", val_stats_epoch["VelRMSE_mmps"], epoch)
                writer.add_scalar("metric/val_AccRMSE_mmps2", val_stats_epoch["AccRMSE_mmps2"], epoch)
            writer.flush()

        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "meta": {
                "task": "spline_fit_mamba",
                "arch": "SpatioTemporalPredictor",
                "predictor": "mamba",
                "num_keypoints": num_keypoints,
                "model_in_dim": coord_dims,
                "model_out_dim": model_out_dim,
                "history_len": int(args.history_len),
                "loss_crmse_weight": float(args.loss_crmse_weight),
                "loss_vel_rmse_weight": float(args.loss_vel_rmse_weight),
                "loss_acc_rmse_weight": float(args.loss_acc_rmse_weight),
                "train_loss": float(train_stats_epoch["loss"]),
                "train_cRMSE_mm": float(train_stats_epoch["cRMSE_mm"]),
                "train_VelRMSE_mmps": float(train_stats_epoch["VelRMSE_mmps"]),
                "train_AccRMSE_mmps2": float(train_stats_epoch["AccRMSE_mmps2"]),
                "val_loss": None if val_stats_epoch is None else float(val_stats_epoch["loss"]),
                "val_cRMSE_mm": None if val_stats_epoch is None else float(val_stats_epoch["cRMSE_mm"]),
                "val_VelRMSE_mmps": None if val_stats_epoch is None else float(val_stats_epoch["VelRMSE_mmps"]),
                "val_AccRMSE_mmps2": None if val_stats_epoch is None else float(val_stats_epoch["AccRMSE_mmps2"]),
                "base_checkpoint": args.checkpoint_path,
                "train_gt_roots": list(args.train_gt_roots),
                "valid_gt_roots": list(args.valid_gt_roots),
                "test_gt_roots": list(args.test_gt_roots),
                "epoch": int(epoch),
                "seed": int(args.seed),
            },
        }
        last_ckpt = ckpt

        if epoch % args.save_every == 0:
            periodic_path = os.path.join(ckpt_dir, f"ep{epoch:03d}.pt")
            torch.save(ckpt, periodic_path)
            print(f"checkpoint_saved: {periodic_path}")

        monitor_metric = train_stats_epoch["loss"] if val_stats_epoch is None else val_stats_epoch["loss"]
        if monitor_metric < best_metric:
            best_metric = monitor_metric
            torch.save(ckpt, best_path)

    if last_ckpt is not None:
        torch.save(last_ckpt, last_path)
        print(f"last_checkpoint: {last_path}")

    if test_loader is not None:
        test_stats_epoch = run_epoch(
            model=model,
            loader=test_loader,
            device=device,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
            coord_dims=coord_dims,
            w_crmse=args.loss_crmse_weight,
            w_vel=args.loss_vel_rmse_weight,
            w_acc=args.loss_acc_rmse_weight,
            train=False,
            log_interval=0,
        )
        print(
            "test_metrics: "
            f"loss={test_stats_epoch['loss']:.6f}, "
            f"cRMSE={test_stats_epoch['cRMSE_mm']:.6f}, "
            f"VelRMSE={test_stats_epoch['VelRMSE_mmps']:.6f}, "
            f"AccRMSE={test_stats_epoch['AccRMSE_mmps2']:.6f}"
        )

    if writer is not None:
        writer.close()

    print(f"train_done: best_monitor_metric={best_metric:.6f}")
    print(f"best_checkpoint: {best_path}")


if __name__ == "__main__":
    main()
