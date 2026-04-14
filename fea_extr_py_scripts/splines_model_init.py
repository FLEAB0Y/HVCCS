import argparse
import os
from datetime import datetime

import numpy as np
import torch

try:
	from splines_fit_train import SpatioTemporalPredictor
except Exception:
	from fea_extr_py_scripts.splines_fit_train import SpatioTemporalPredictor


def count_params(module: torch.nn.Module):
	total = sum(p.numel() for p in module.parameters())
	trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
	return total, trainable


def build_ckpt_name(hparams: dict) -> str:
	"""将关键超参数编码到 checkpoint 文件名，便于后续追踪与对齐。"""
	return (
		"pose_mamba_init"
		f"_k{hparams['num_keypoints']}"
		f"_d{hparams['model_in_dim']}"
		f"_o{hparams['model_out_dim']}"
		f"_w{hparams['history_len']}"
		f"_f{hparams['future_len']}"
		f"_h{hparams['gnn_hidden']}"
		f"_ds{hparams['mamba_d_state']}"
		f"_dc{hparams['mamba_d_conv']}"
		f"_ex{hparams['mamba_expand']}"
		f"_nl{hparams['mamba_n_layer']}"
		f"_s{hparams['model_seed']}.pt"
	)


def main() -> None:
	parser = argparse.ArgumentParser(description="Initialize and save Mamba pose checkpoint")

	# ===== 模型超参数（程序内统一定义，不通过命令行传递） =====
	MODEL_HPARAMS = {
		# 每帧关键点数量（Human3.6M: 17）
		"num_keypoints": 17,
		# 输入关键点维度，当前姿态是xyz三维
		"model_in_dim": 3,
		# 输出维度：xyz + vxyz
		"model_out_dim": 6,
		# 训练目标：未来1帧
		"future_len": 1,
		# 时序历史长度：输入8帧
		"history_len": 8,
		# GNN隐藏维度，同时作为Mamba的d_model（质量优先：更大容量）
		"gnn_hidden": 256,
		# Mamba状态维度（质量优先：更强记忆）
		"mamba_d_state": 128,
		# Mamba局部卷积宽度（当前 causal_conv1d 约束为 2~4）
		"mamba_d_conv": 4,
		# Mamba通道扩展倍率
		"mamba_expand": 4,
		# Mamba堆叠层数
		"mamba_n_layer": 8,
		# 输出目标模式：预测坐标+速度
		"target_mode": "coord_and_velocity",
		# 模型初始化随机种子
		"model_seed": 2026,
	}

	parser.add_argument("--cuda_device", type=int, default=-1, help="Use CPU when set to -1")
	parser.add_argument(
		"--out_dir",
		type=str,
		default="/home/ztw/HVCCS/checkpoints",
		help="checkpoint 输出目录；文件名会自动附带超参数",
	)
	args = parser.parse_args()

	expected_stream_dims = MODEL_HPARAMS["num_keypoints"] * MODEL_HPARAMS["model_in_dim"]
	expected_target_dims = MODEL_HPARAMS["num_keypoints"] * MODEL_HPARAMS["model_out_dim"]
	if expected_stream_dims <= 0:
		raise ValueError(f"invalid stream dims: {expected_stream_dims}")
	if expected_target_dims <= 0:
		raise ValueError(f"invalid target dims: {expected_target_dims}")

	torch.manual_seed(MODEL_HPARAMS["model_seed"])
	np.random.seed(MODEL_HPARAMS["model_seed"])

	if args.cuda_device >= 0 and torch.cuda.is_available():
		device = torch.device(f"cuda:{args.cuda_device}")
	else:
		device = torch.device("cpu")

	model = SpatioTemporalPredictor(
		in_dim=MODEL_HPARAMS["model_in_dim"],
		out_dim=MODEL_HPARAMS["model_out_dim"],
		gnn_hidden=MODEL_HPARAMS["gnn_hidden"],
		mamba_d_state=MODEL_HPARAMS["mamba_d_state"],
		mamba_d_conv=MODEL_HPARAMS["mamba_d_conv"],
		mamba_expand=MODEL_HPARAMS["mamba_expand"],
		mamba_n_layer=MODEL_HPARAMS["mamba_n_layer"],
		num_nodes=MODEL_HPARAMS["num_keypoints"],
	).to(device)
	model.eval()

	total_params, trainable_params = count_params(model)
	mamba_params, _ = count_params(model.mamba)

	os.makedirs(args.out_dir, exist_ok=True)
	ckpt_name = build_ckpt_name(MODEL_HPARAMS)
	out_path = os.path.join(args.out_dir, ckpt_name)
	ckpt = {
		"model_state_dict": model.state_dict(),
		"meta": {
			"arch": "SpatioTemporalPredictor",
			"predictor": "mamba",
			"num_keypoints": MODEL_HPARAMS["num_keypoints"],
			"model_in_dim": MODEL_HPARAMS["model_in_dim"],
			"model_out_dim": MODEL_HPARAMS["model_out_dim"],
			"future_len": MODEL_HPARAMS["future_len"],
			"history_len": MODEL_HPARAMS["history_len"],
			"target_mode": MODEL_HPARAMS["target_mode"],
			"gnn_hidden": MODEL_HPARAMS["gnn_hidden"],
			"mamba_d_state": MODEL_HPARAMS["mamba_d_state"],
			"mamba_d_conv": MODEL_HPARAMS["mamba_d_conv"],
			"mamba_expand": MODEL_HPARAMS["mamba_expand"],
			"mamba_n_layer": MODEL_HPARAMS["mamba_n_layer"],
			"model_seed": MODEL_HPARAMS["model_seed"],
			"device_init": str(device),
			"created_at": datetime.now().isoformat(timespec="seconds"),
		},
	}
	torch.save(ckpt, out_path)

	print(f"checkpoint_saved: {out_path}")
	print(f"stream_dims_per_frame: {expected_stream_dims} ({MODEL_HPARAMS['num_keypoints']}x{MODEL_HPARAMS['model_in_dim']})")
	print(f"target_dims_per_frame: {expected_target_dims} ({MODEL_HPARAMS['num_keypoints']}x{MODEL_HPARAMS['model_out_dim']})")
	print(f"history_len: {MODEL_HPARAMS['history_len']}")
	print(f"future_len: {MODEL_HPARAMS['future_len']}")
	print(f"target_mode: {MODEL_HPARAMS['target_mode']}")
	print(f"model_params_total: {total_params}")
	print(f"model_params_trainable: {trainable_params}")
	print(f"mamba_params: {mamba_params}")


if __name__ == "__main__":
    main()
