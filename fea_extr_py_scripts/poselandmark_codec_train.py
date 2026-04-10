import argparse
import csv
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from poselandmark_online_encoder_v2 import SpatioTemporalPredictor, to_tjc_from_npy

try:
	from torch.utils.tensorboard import SummaryWriter
except Exception:
	SummaryWriter = None

try:
	import matplotlib.pyplot as plt
except Exception:
	plt = None


DEFAULT_CHECKPOINT_PATH = "/home/ztw/HVCCS/checkpoints/pose_mamba_init_k17_d3_w32_h128_ds64_dc4_ex4_nl4_s2026.pt"
DEFAULT_TRAIN_ROOTS = [
	"/home/data/ztw/AtheletePose3D/data/train_set/S3",
	"/home/data/ztw/AtheletePose3D/data/train_set/S4",
]
DEFAULT_VALID_ROOTS = [
	"/home/data/ztw/AtheletePose3D/data/valid_set/S2",
]
DEFAULT_OUTPUT_DIR = "/home/ztw/HVCCS/checkpoints"

# ===== 训练超参数（程序内统一定义，可按需直接修改） =====
TRAIN_HPARAMS = {
	"train_roots": DEFAULT_TRAIN_ROOTS,
	"valid_roots": DEFAULT_VALID_ROOTS,
	"checkpoint_path": DEFAULT_CHECKPOINT_PATH,
	"output_dir": DEFAULT_OUTPUT_DIR,
	"history_len": 8,
	"epochs": 10,
	"batch_size": 64,
	"lr": 1e-4,
	"weight_decay": 1e-5,
	"num_workers": 0,
	"grad_clip": 1.0,
	"seed": 2026,
	"cuda_device": 0,
	"max_files": 0,
	"log_interval": 100,
	"save_every": 10,
	"run_name": "",
	"enable_tensorboard": True,
	"tb_logdir": "/home/ztw/HVCCS/checkpoints/tb_runs",
	"curve_png": "",
	"curve_csv": "",
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
			print(
				f"检测到当前 GPU 架构 sm_{major}{minor} 不受当前 PyTorch 支持，自动回退到 CPU"
			)
			return torch.device("cpu")
	except Exception as exc:
		print(f"CUDA 架构检查失败({exc})，自动回退到 CPU")
		return torch.device("cpu")
	return torch.device(f"cuda:{cuda_device}")


def collect_cam1_h36m_files(data_roots: Sequence[str], max_files: int = 0, allow_empty: bool = False) -> List[str]:
	all_files: List[str] = []
	for data_root in data_roots:
		root = Path(data_root)
		if not root.exists():
			raise FileNotFoundError(f"data_root not found: {data_root}")
		all_files.extend(str(p) for p in root.rglob("*_cam_1_h36m.npy"))

	files = sorted(set(all_files))
	if max_files > 0:
		files = files[:max_files]
	if not files and not allow_empty:
		raise RuntimeError(f"No *_cam_1_h36m.npy files found under roots: {list(data_roots)}")
	return files


class PoseNextFrameDataset(Dataset):
	def __init__(
		self,
		file_list: Sequence[str],
		history_len: int,
		num_keypoints: int,
		coord_dims: int,
	) -> None:
		if history_len < 1:
			raise ValueError(f"history_len must be >= 1, got {history_len}")

		self.history_len = int(history_len)
		self.arrays: List[torch.Tensor] = []
		self.samples: List[Tuple[int, int]] = []

		for file_path in file_list:
			arr = np.load(file_path)
			tjc = to_tjc_from_npy(arr, coord_dims=coord_dims)

			if tjc.shape[1] != num_keypoints:
				raise ValueError(
					f"joint mismatch in {file_path}: got {tjc.shape[1]}, expected {num_keypoints}"
				)

			if tjc.shape[0] <= self.history_len:
				continue

			seq = tjc[:, :num_keypoints, :coord_dims].astype(np.float32, copy=False)
			seq_tensor = torch.from_numpy(seq)
			seq_idx = len(self.arrays)
			self.arrays.append(seq_tensor)

			num_samples = seq.shape[0] - self.history_len
			self.samples.extend((seq_idx, s) for s in range(num_samples))

		if not self.samples:
			raise RuntimeError("No valid samples constructed; check history_len and data files")

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
		seq_idx, start = self.samples[index]
		seq = self.arrays[seq_idx]
		x = seq[start : start + self.history_len]
		y = seq[start + self.history_len]
		return x, y


def mpjpe_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
	return torch.linalg.norm(pred - target, dim=-1).mean()


def save_curve_csv(curve_csv_path: str, train_curve: List[float], val_curve: List[Optional[float]]) -> None:
	with open(curve_csv_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["epoch", "train_mpjpe", "val_mpjpe"])
		for i, train_v in enumerate(train_curve, start=1):
			val_v = val_curve[i - 1]
			writer.writerow([i, f"{train_v:.8f}", "" if val_v is None else f"{val_v:.8f}"])


def save_curve_png(curve_png_path: str, train_curve: List[float], val_curve: List[Optional[float]]) -> bool:
	if plt is None:
		return False

	epochs = np.arange(1, len(train_curve) + 1)
	val_plot = np.array([np.nan if v is None else v for v in val_curve], dtype=np.float32)

	fig, ax = plt.subplots(figsize=(8, 5))
	ax.plot(epochs, np.array(train_curve, dtype=np.float32), label="train_mpjpe", linewidth=2.0)
	ax.plot(epochs, val_plot, label="val_mpjpe", linewidth=2.0)
	ax.set_xlabel("Epoch")
	ax.set_ylabel("MPJPE")
	ax.set_title("Training Convergence")
	ax.grid(True, alpha=0.3)
	ax.legend()
	fig.tight_layout()
	fig.savefig(curve_png_path, dpi=150)
	plt.close(fig)
	return True


def load_model_from_checkpoint(
	checkpoint_path: str,
	device: torch.device,
) -> Tuple[SpatioTemporalPredictor, Dict]:
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
	gnn_hidden = int(meta.get("gnn_hidden", 128))
	mamba_d_state = int(meta.get("mamba_d_state", 64))
	mamba_d_conv = int(meta.get("mamba_d_conv", 4))
	mamba_expand = int(meta.get("mamba_expand", 4))
	mamba_n_layer = int(meta.get("mamba_n_layer", 4))

	model = SpatioTemporalPredictor(
		in_dim=model_in_dim,
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
		"gnn_hidden": gnn_hidden,
		"mamba_d_state": mamba_d_state,
		"mamba_d_conv": mamba_d_conv,
		"mamba_expand": mamba_expand,
		"mamba_n_layer": mamba_n_layer,
	}


def run_epoch(
	model: SpatioTemporalPredictor,
	loader: DataLoader,
	device: torch.device,
	optimizer: torch.optim.Optimizer,
	grad_clip: float,
	train: bool,
	log_interval: int,
) -> float:
	if train:
		model.train()
	else:
		model.eval()

	total_loss = 0.0
	total_count = 0

	for step, (x, y) in enumerate(loader, start=1):
		x = x.to(device, non_blocking=True)
		y = y.to(device, non_blocking=True)

		if train:
			optimizer.zero_grad(set_to_none=True)

		with torch.set_grad_enabled(train):
			pred = model(x)
			loss = mpjpe_loss(pred, y)

			if train:
				loss.backward()
				if grad_clip > 0:
					torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
				optimizer.step()

		batch_size = x.size(0)
		total_loss += float(loss.item()) * batch_size
		total_count += batch_size

		if log_interval > 0 and step % log_interval == 0:
			phase = "train" if train else "val"
			print(f"[{phase}] step={step}, loss={loss.item():.6f}")

	return total_loss / max(total_count, 1)


def main() -> None:
	parser = argparse.ArgumentParser(description="Train pose mamba model with MPJPE loss")
	parser.add_argument("--train_roots", nargs="+", default=TRAIN_HPARAMS["train_roots"])
	parser.add_argument("--valid_roots", nargs="*", default=TRAIN_HPARAMS["valid_roots"])
	parser.add_argument("--checkpoint_path", type=str, default=TRAIN_HPARAMS["checkpoint_path"])
	parser.add_argument("--output_dir", type=str, default=TRAIN_HPARAMS["output_dir"])
	parser.add_argument("--history_len", type=int, default=TRAIN_HPARAMS["history_len"])
	parser.add_argument("--epochs", type=int, default=TRAIN_HPARAMS["epochs"])
	parser.add_argument("--batch_size", type=int, default=TRAIN_HPARAMS["batch_size"])
	parser.add_argument("--lr", type=float, default=TRAIN_HPARAMS["lr"])
	parser.add_argument("--weight_decay", type=float, default=TRAIN_HPARAMS["weight_decay"])
	parser.add_argument("--num_workers", type=int, default=TRAIN_HPARAMS["num_workers"])
	parser.add_argument("--grad_clip", type=float, default=TRAIN_HPARAMS["grad_clip"])
	parser.add_argument("--seed", type=int, default=TRAIN_HPARAMS["seed"])
	parser.add_argument("--cuda_device", type=int, default=TRAIN_HPARAMS["cuda_device"])
	parser.add_argument("--max_files", type=int, default=TRAIN_HPARAMS["max_files"])
	parser.add_argument("--log_interval", type=int, default=TRAIN_HPARAMS["log_interval"])
	parser.add_argument("--save_every", type=int, default=TRAIN_HPARAMS["save_every"])
	parser.add_argument("--run_name", type=str, default=TRAIN_HPARAMS["run_name"])
	parser.add_argument("--enable_tensorboard", action="store_true", default=bool(TRAIN_HPARAMS["enable_tensorboard"]))
	parser.add_argument("--tb_logdir", type=str, default=TRAIN_HPARAMS["tb_logdir"])
	parser.add_argument("--curve_png", type=str, default=TRAIN_HPARAMS["curve_png"])
	parser.add_argument("--curve_csv", type=str, default=TRAIN_HPARAMS["curve_csv"])
	args = parser.parse_args()

	if args.save_every < 1:
		raise ValueError(f"save_every must be >= 1, got {args.save_every}")

	set_seed(args.seed)
	device = select_torch_device(args.cuda_device)
	print(f"train_device: {device}")

	train_files = collect_cam1_h36m_files(args.train_roots, max_files=args.max_files)
	valid_files = collect_cam1_h36m_files(args.valid_roots, allow_empty=True)
	if args.valid_roots and not valid_files:
		raise RuntimeError(f"No validation files found under roots: {args.valid_roots}")

	print(f"train_roots: {args.train_roots}")
	print(f"valid_roots: {args.valid_roots}")
	print(f"train_files_count: {len(train_files)}")
	print(f"valid_files_count: {len(valid_files)}")
	print(f"train_files_example: {train_files[:3]}")
	if valid_files:
		print(f"valid_files_example: {valid_files[:3]}")

	model, model_hparams = load_model_from_checkpoint(args.checkpoint_path, device=device)
	print(f"loaded_checkpoint: {args.checkpoint_path}")
	print(f"model_hparams: {model_hparams}")

	train_set = PoseNextFrameDataset(
		file_list=train_files,
		history_len=args.history_len,
		num_keypoints=int(model_hparams["num_keypoints"]),
		coord_dims=int(model_hparams["model_in_dim"]),
	)
	val_set = None
	if valid_files:
		val_set = PoseNextFrameDataset(
			file_list=valid_files,
			history_len=args.history_len,
			num_keypoints=int(model_hparams["num_keypoints"]),
			coord_dims=int(model_hparams["model_in_dim"]),
		)

	print(f"train_samples: {len(train_set)}")
	if val_set is not None:
		print(f"valid_samples: {len(val_set)}")
	else:
		print("valid_samples: 0 (no validation set)")

	pin_memory = device.type == "cuda"
	train_loader = DataLoader(
		train_set,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=pin_memory,
		drop_last=False,
	)
	val_loader = None
	if val_set is not None:
		val_loader = DataLoader(
			val_set,
			batch_size=args.batch_size,
			shuffle=False,
			num_workers=args.num_workers,
			pin_memory=pin_memory,
			drop_last=False,
		)

	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=args.lr,
		weight_decay=args.weight_decay,
	)

	os.makedirs(args.output_dir, exist_ok=True)
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_prefix = args.run_name.strip() or f"pose_mamba_train_cam1_mpjpe_h{args.history_len}_{timestamp}"
	curve_csv_path = args.curve_csv.strip() or os.path.join(args.output_dir, f"{run_prefix}_curve.csv")
	curve_png_path = args.curve_png.strip() or os.path.join(args.output_dir, f"{run_prefix}_curve.png")

	writer = None
	if args.enable_tensorboard:
		if SummaryWriter is None:
			print("tensorboard 未安装，跳过在线标量写入。请先安装: pip install tensorboard")
		else:
			tb_root = args.tb_logdir.strip() or os.path.join(args.output_dir, "tb_runs")
			tb_logdir = os.path.join(tb_root, run_prefix)
			os.makedirs(tb_logdir, exist_ok=True)
			writer = SummaryWriter(log_dir=tb_logdir)
			print(f"tensorboard_logdir: {tb_logdir}")

	print(f"curve_csv: {curve_csv_path}")
	if plt is not None:
		print(f"curve_png: {curve_png_path}")
	else:
		print("matplotlib 未安装，跳过 PNG 曲线导出")

	best_val = float("inf")
	best_path = os.path.join(args.output_dir, f"{run_prefix}_best.pt")
	final_path = os.path.join(args.output_dir, f"{run_prefix}_last.pt")

	train_curve: List[float] = []
	val_curve: List[Optional[float]] = []
	last_ckpt = None

	for epoch in range(1, args.epochs + 1):
		train_loss = run_epoch(
			model=model,
			loader=train_loader,
			device=device,
			optimizer=optimizer,
			grad_clip=args.grad_clip,
			train=True,
			log_interval=args.log_interval,
		)

		msg = f"epoch={epoch}/{args.epochs}, train_mpjpe={train_loss:.6f}"

		val_loss = None
		if val_loader is not None:
			val_loss = run_epoch(
				model=model,
				loader=val_loader,
				device=device,
				optimizer=optimizer,
				grad_clip=args.grad_clip,
				train=False,
				log_interval=0,
			)
			msg += f", val_mpjpe={val_loss:.6f}"
		print(msg)

		train_curve.append(float(train_loss))
		val_curve.append(None if val_loss is None else float(val_loss))
		save_curve_csv(curve_csv_path, train_curve, val_curve)
		_ = save_curve_png(curve_png_path, train_curve, val_curve)

		if writer is not None:
			writer.add_scalar("mpjpe/train", float(train_loss), epoch)
			if val_loss is not None:
				writer.add_scalar("mpjpe/val", float(val_loss), epoch)
			writer.flush()

		ckpt = {
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"meta": {
				"arch": "SpatioTemporalPredictor",
				"predictor": "mamba",
				"num_keypoints": int(model_hparams["num_keypoints"]),
				"model_in_dim": int(model_hparams["model_in_dim"]),
				"history_len": int(args.history_len),
				"gnn_hidden": int(model_hparams["gnn_hidden"]),
				"mamba_d_state": int(model_hparams["mamba_d_state"]),
				"mamba_d_conv": int(model_hparams["mamba_d_conv"]),
				"mamba_expand": int(model_hparams["mamba_expand"]),
				"mamba_n_layer": int(model_hparams["mamba_n_layer"]),
				"train_loss_mpjpe": float(train_loss),
				"val_loss_mpjpe": None if val_loss is None else float(val_loss),
				"base_checkpoint": args.checkpoint_path,
				"train_roots": list(args.train_roots),
				"valid_roots": list(args.valid_roots),
				"cam_filter": "cam_1_h36m",
				"epoch": int(epoch),
				"seed": int(args.seed),
			},
		}
		last_ckpt = ckpt

		if epoch % args.save_every == 0:
			periodic_path = os.path.join(args.output_dir, f"{run_prefix}_ep{epoch:03d}.pt")
			torch.save(ckpt, periodic_path)
			print(f"checkpoint_saved: {periodic_path}")

		metric = train_loss if val_loss is None else val_loss
		if metric < best_val:
			best_val = metric
			torch.save(ckpt, best_path)

	if last_ckpt is not None:
		torch.save(last_ckpt, final_path)
		print(f"last_checkpoint: {final_path}")

	if writer is not None:
		writer.close()

	print(f"train_done: best_metric={best_val:.6f}")
	print(f"best_checkpoint: {best_path}")


if __name__ == "__main__":
	main()
