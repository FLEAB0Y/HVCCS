import argparse
import json
import os
import threading
import time
import zlib
import torch
import torch.nn as nn
import numpy as np

from THStreamData import THStreamDataPayload
from client import THStreamClient

try:
    from mamba_ssm import Mamba
except ImportError:
    print("Mamba 未安装，请使用 pip install mamba-ssm 安装，或替换为其他序列模型")


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


def pose99_to_bytes(values):
    return np.asarray(values, dtype=np.float32).reshape(-1).tobytes()


def encode_pose_payload(
    values,
    packet_kind,
    quant_scale,
    quantize_i_frame,
    quantize_p_frame,
    entropy_enabled,
    entropy_level,
):
    arr = np.asarray(values, dtype=np.float32).reshape(-1)

    use_quant = (packet_kind == "I" and quantize_i_frame) or (packet_kind == "P" and quantize_p_frame)
    if use_quant:
        q = np.round(arr * quant_scale)
        q = np.clip(q, -32768, 32767).astype(np.int16)
        payload = q.tobytes()
        payload_dtype = "qint16"
    else:
        payload = arr.astype(np.float32).tobytes()
        payload_dtype = "float32"

    entropy_codec = "none"
    if entropy_enabled:
        payload = zlib.compress(payload, level=entropy_level)
        entropy_codec = "zlib"

    raw_bytes = int(arr.size * 4)
    encoded_bytes = int(len(payload))

    return payload, {
        "payload_dtype": payload_dtype,
        "entropy_codec": entropy_codec,
    }, raw_bytes, encoded_bytes


def to_tjc_from_npy(pose: np.ndarray, coord_dims: int = 3) -> np.ndarray:
    """将常见 npy 姿态布局归一化为 (T, J, C)。"""
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


def parse_pose_from_line(line, num_keypoints=33, coord_dims=3, start_col=52):
    """兼容逗号/空格分隔，按给定关键点数提取姿态特征。"""
    tokens = [t.strip() for t in line.replace(',', ' ').split()]
    num_values = num_keypoints * coord_dims
    end_col = start_col + num_values
    if len(tokens) < end_col:
        return None
    try:
        pose_values = np.array(list(map(float, tokens[start_col:end_col])), dtype=np.float32)
    except ValueError:
        return None
    if pose_values.shape[0] != num_values:
        return None
    return pose_values.reshape(num_keypoints, coord_dims)


def iter_pose_frames(filepath, num_keypoints=33, coord_dims=3):
    suffix = os.path.splitext(filepath)[1].lower()
    if suffix == ".npy":
        arr = np.load(filepath)
        tjc = to_tjc_from_npy(arr, coord_dims=coord_dims)
        if tjc.shape[1] != num_keypoints:
            raise ValueError(
                f"npy joint count mismatch: file has {tjc.shape[1]} joints, "
                f"but model expects {num_keypoints}"
            )
        for i in range(tjc.shape[0]):
            yield tjc[i].astype(np.float32, copy=False)
        return

    if num_keypoints != 33 or coord_dims != 3:
        raise ValueError(
            "Text feature parsing currently supports 33x3 layout only; "
            "for other layouts use .npy input"
        )

    with open(filepath, 'r') as f:
        for line in f:
            pose = parse_pose_from_line(line, num_keypoints=num_keypoints, coord_dims=coord_dims)
            if pose is not None:
                yield pose


class GRPCPoseSender:
    def __init__(self, host="127.0.0.1", port=50051, interval=1.0 / 120.0):
        self.client = THStreamClient(host=host, port=port)
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None

    def _send_loop(self):
        while not self._stop_event.is_set():
            if self.client.send_data_buffer.get_size() > 0:
                self.client.send_data()
            else:
                time.sleep(0.001)

    def start(self):
        self._thread = threading.Thread(target=self._send_loop, daemon=True)
        self._thread.start()

    def send_pose_packet(self, point_data_bytes, meta):
        payload = THStreamDataPayload(
            rgb_data=b"",
            point_data=point_data_bytes,
            face_data=b"",
            limb_data=b"",
            ext_data=PACKET_TAG,
            ext_desc=json.dumps(meta, separators=(",", ":")),
        )
        while self.client.send_data_buffer.get_size() >= 80:
            time.sleep(0.002)
        self.client.send_data_buffer.add_item(payload)

    def shutdown(self, drain_timeout=2.0):
        deadline = time.time() + max(drain_timeout, 0.0)
        while self.client.send_data_buffer.get_size() > 0 and time.time() < deadline:
            time.sleep(0.005)

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        # 主动关闭 channel，减少进程退出阶段的 gRPC 噪声日志
        try:
            self.client.channel.close()
        except Exception:
            pass


# MediaPipe Pose 33个关键点的连接对
MEDIAPIPE_POSE_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8), (9,10), # 头部
    (11,12), (11,13), (13,15), (15,17), (15,19), (15,21), # 左臂
    (12,14), (14,16), (16,18), (16,20), (16,22), # 右臂
    (11,23), (12,24), (23,24), # 躯干
    (23,25), (25,27), (27,29), (27,31), (29,31), # 左腿
    (24,26), (26,28), (28,30), (28,32), (30,32)  # 右腿
]

# Human3.6M 17个关键点拓扑连接
H36M_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

def build_adjacency_matrix(num_nodes=33):
    adj = np.eye(num_nodes)
    if num_nodes == 33:
        for (i, j) in MEDIAPIPE_POSE_CONNECTIONS:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    elif num_nodes == 17:
        for (i, j) in H36M_CONNECTIONS:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    else:
        # 非 33 关键点时缺少固定拓扑，退化为相邻链式连接。
        for i in range(num_nodes - 1):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
    rowsum = adj.sum(axis=1)
    adj = adj / rowsum[:, np.newaxis]
    return torch.tensor(adj, dtype=torch.float32)

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        x = self.fc(x)
        output = torch.matmul(adj, x)
        return output

class GNN_IntraFrame_Encoder(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=16, num_nodes=33):
        super().__init__()
        self.gcn1 = GraphConvolution(in_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.register_buffer('adj', build_adjacency_matrix(num_nodes))
        
    def forward(self, x):
        h = self.relu(self.gcn1(x, self.adj))
        h = self.relu(self.gcn2(h, self.adj))
        return h


class MambaResidualBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.mamba(x)
        return self.norm(x + residual)

class SpatioTemporalPredictor(nn.Module):
    def __init__(
        self,
        in_dim=3,
        gnn_hidden=16,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_n_layer=1,
        num_nodes=33,
    ):
        super().__init__()
        self.gnn = GNN_IntraFrame_Encoder(in_dim, gnn_hidden, num_nodes=num_nodes)
        if mamba_n_layer < 1:
            raise ValueError("mamba_n_layer must be >= 1")

        # 多层 Mamba 堆叠：提升时序表达能力（质量优先场景）
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
        # 兼容已有参数统计逻辑
        self.mamba = self.mamba_blocks
        self.predictor = nn.Linear(gnn_hidden, in_dim)

    def forward(self, x_seq):
        B, L, N, C = x_seq.shape
        x_flat = x_seq.view(B * L, N, C)
        
        # 帧内空间聚合
        gnn_out = self.gnn(x_flat) # (B*L, 33, hidden)
        
        # 帧间时序建模 (视为 33 个独立序列)
        gnn_out = gnn_out.view(B, L, N, -1).permute(0, 2, 1, 3).reshape(B * N, L, -1)
        mamba_out = gnn_out
        for block in self.mamba_blocks:
            mamba_out = block(mamba_out)
        
        last_step_features = mamba_out[:, -1, :] # (B*N, hidden)
        pred_xyz = self.predictor(last_step_features)
        return pred_xyz.view(B, N, self.predictor.out_features)


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

def stream_and_encode(
    filepath,
    model,
    sender=None,
    window_size=10,
    fps=30,
    debug=False,
    max_frames=None,
    model_hparams=None,
):
    interval = 1.0 / fps
    history_buffer = []
    device = next(model.parameters()).device
    sent_count = 0
    total_raw_bytes = 0
    total_encoded_bytes = 0
    i_raw_bytes = 0
    i_encoded_bytes = 0
    p_raw_bytes = 0
    p_encoded_bytes = 0
    
    model_hparams = model_hparams or {}
    num_keypoints = int(model_hparams.get("num_keypoints", 33))
    coord_dims = int(model_hparams.get("model_in_dim", 3))
    i_frame_interval = int(model_hparams.get("i_frame_interval", 0))
    quantize_i_frame = bool(model_hparams.get("quantize_i_frame", False))
    quantize_p_frame = bool(model_hparams.get("quantize_p_frame", True))
    quant_scale = float(model_hparams.get("quant_scale", 1000.0))
    entropy_enabled = bool(model_hparams.get("entropy_enabled", True))
    entropy_level = int(model_hparams.get("entropy_level", 6))

    if quant_scale <= 0:
        raise ValueError(f"quant_scale must be positive, got {quant_scale}")
    if entropy_level < 0 or entropy_level > 9:
        raise ValueError(f"entropy_level must be in [0, 9], got {entropy_level}")

    print(f"开始流式处理文件... 目标帧率: {fps} fps")
    for line_idx, pose_data in enumerate(
        iter_pose_frames(filepath, num_keypoints=num_keypoints, coord_dims=coord_dims)
    ):
            start_time = time.time()

            current_tensor = torch.tensor(pose_data, device=device).unsqueeze(0)
            force_i_frame = i_frame_interval > 0 and sent_count > 0 and (sent_count % i_frame_interval == 0)
            
            if len(history_buffer) < window_size or force_i_frame:
                packet_kind = "I"
                packet_values = current_tensor.view(-1).detach().cpu().numpy()
                if len(history_buffer) >= window_size:
                    history_buffer.pop(0)
                history_buffer.append(current_tensor)
            else:
                seq_input = torch.cat(history_buffer, dim=0).unsqueeze(0) # (1, window, 33, 3)
                with torch.no_grad():
                    predicted_tensor = model(seq_input)
                
                # 计算预测残差作为流式发送的源数据
                packet_kind = "P"
                residuals = (current_tensor - predicted_tensor).view(-1)
                packet_values = residuals.detach().cpu().numpy()
                history_buffer.pop(0)
                history_buffer.append(current_tensor)

                if debug:
                    print(
                        f"Frame {line_idx}: residual_dim={residuals.numel()}, "
                        f"mean_abs={residuals.abs().mean().item():.6f}, "
                        f"max_abs={residuals.abs().max().item():.6f}"
                    )

            if sender is not None:
                payload_bytes, payload_meta, raw_bytes, encoded_bytes = encode_pose_payload(
                    packet_values,
                    packet_kind=packet_kind,
                    quant_scale=quant_scale,
                    quantize_i_frame=quantize_i_frame,
                    quantize_p_frame=quantize_p_frame,
                    entropy_enabled=entropy_enabled,
                    entropy_level=entropy_level,
                )
                total_raw_bytes += raw_bytes
                total_encoded_bytes += encoded_bytes
                if packet_kind == "I":
                    i_raw_bytes += raw_bytes
                    i_encoded_bytes += encoded_bytes
                else:
                    p_raw_bytes += raw_bytes
                    p_encoded_bytes += encoded_bytes

                meta = {
                    "tag": "POSE99V1",
                    "kind": packet_kind,
                    "frame_idx": int(sent_count),
                    "window_size": int(window_size),
                    "i_frame_interval": int(i_frame_interval),
                    "history_len": int(model_hparams.get("history_len", window_size)),
                    "num_keypoints": num_keypoints,
                    "predictor": "mamba",
                    "model_in_dim": coord_dims,
                    "pose_dims": int(num_keypoints * coord_dims),
                    "quantize_i_frame": int(quantize_i_frame),
                    "quantize_p_frame": int(quantize_p_frame),
                    "quant_scale": float(quant_scale),
                    "entropy_enabled": int(entropy_enabled),
                    "entropy_level": int(entropy_level),
                    "gnn_hidden": int(model_hparams.get("gnn_hidden", model.predictor.in_features)),
                    "mamba_d_state": int(model_hparams.get("mamba_d_state", 16)),
                    "mamba_d_conv": int(model_hparams.get("mamba_d_conv", 4)),
                    "mamba_expand": int(model_hparams.get("mamba_expand", 2)),
                    "mamba_n_layer": int(model_hparams.get("mamba_n_layer", 1)),
                    "timestamp_ms": int(time.time() * 1000),
                }
                meta.update(payload_meta)
                sender.send_pose_packet(payload_bytes, meta)

            sent_count += 1
            if max_frames is not None and sent_count >= max_frames:
                break
                
            # 帧率控制
            elapsed = time.time() - start_time
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    if total_raw_bytes > 0:
        overall_ratio = total_raw_bytes / max(total_encoded_bytes, 1)
        overall_saving = 1.0 - (total_encoded_bytes / total_raw_bytes)
        print(
            "codec_stats: "
            f"raw_bytes={total_raw_bytes}, "
            f"encoded_bytes={total_encoded_bytes}, "
            f"ratio={overall_ratio:.4f}x, "
            f"saving={overall_saving * 100.0:.2f}%"
        )
        if i_raw_bytes > 0:
            i_ratio = i_raw_bytes / max(i_encoded_bytes, 1)
            i_saving = 1.0 - (i_encoded_bytes / i_raw_bytes)
            print(
                "codec_stats_I: "
                f"raw={i_raw_bytes}, encoded={i_encoded_bytes}, "
                f"ratio={i_ratio:.4f}x, saving={i_saving * 100.0:.2f}%"
            )
        if p_raw_bytes > 0:
            p_ratio = p_raw_bytes / max(p_encoded_bytes, 1)
            p_saving = 1.0 - (p_encoded_bytes / p_raw_bytes)
            print(
                "codec_stats_P: "
                f"raw={p_raw_bytes}, encoded={p_encoded_bytes}, "
                f"ratio={p_ratio:.4f}x, saving={p_saving * 100.0:.2f}%"
            )

    return sent_count

if __name__ == "__main__":
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
        "quantize_i_frame": False,
        # 是否量化P帧（残差）
        "quantize_p_frame": False,
        # 量化比例（q = round(x * quant_scale)）
        "quant_scale": 1000.0,
        # 是否开启熵编码
        "entropy_enabled": False,
        # zlib压缩级别（0~9）
        "entropy_level": 6,
        # 模型初始化种子，编解码两端需一致
        "model_seed": 2026,
    }

    parser = argparse.ArgumentParser(description="Mamba pose encoder with gRPC sender")
    parser.add_argument("--feature_file", type=str, default="/home/data/ztw/AtheletePose3D/data/train_set/S3/Running_0_cam_1_h36m.npy")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--window_size", type=int, default=8)

    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    try:
        if args.window_size != MODEL_HPARAMS["history_len"]:
            raise ValueError(
                f"window_size({args.window_size}) 必须与 history_len({MODEL_HPARAMS['history_len']}) 一致"
            )

        if not os.path.exists(DEFAULT_CHECKPOINT_PATH):
            raise FileNotFoundError(f"checkpoint not found: {DEFAULT_CHECKPOINT_PATH}")

        torch.manual_seed(MODEL_HPARAMS["model_seed"])
        np.random.seed(MODEL_HPARAMS["model_seed"])

        device = select_torch_device(args.cuda_device)
        model = SpatioTemporalPredictor(
            in_dim=MODEL_HPARAMS["model_in_dim"],
            gnn_hidden=MODEL_HPARAMS["gnn_hidden"],
            mamba_d_state=MODEL_HPARAMS["mamba_d_state"],
            mamba_d_conv=MODEL_HPARAMS["mamba_d_conv"],
            mamba_expand=MODEL_HPARAMS["mamba_expand"],
            mamba_n_layer=MODEL_HPARAMS["mamba_n_layer"],
            num_nodes=MODEL_HPARAMS["num_keypoints"],
        ).to(device)
        load_model_checkpoint(model, DEFAULT_CHECKPOINT_PATH)
        model.eval()
        print(f"使用设备: {device}, host={args.host}, port={args.port}, checkpoint={DEFAULT_CHECKPOINT_PATH}")

        sender = GRPCPoseSender(host=args.host, port=args.port, interval=1.0 / max(args.fps, 1.0))
        sender.start()

        sent = stream_and_encode(
            args.feature_file,
            model,
            sender=sender,
            window_size=args.window_size,
            fps=args.fps,
            debug=args.debug,
            max_frames=args.max_frames,
            model_hparams={
                "num_keypoints": MODEL_HPARAMS["num_keypoints"],
                "model_in_dim": MODEL_HPARAMS["model_in_dim"],
                "history_len": MODEL_HPARAMS["history_len"],
                "gnn_hidden": MODEL_HPARAMS["gnn_hidden"],
                "mamba_d_state": MODEL_HPARAMS["mamba_d_state"],
                "mamba_d_conv": MODEL_HPARAMS["mamba_d_conv"],
                "mamba_expand": MODEL_HPARAMS["mamba_expand"],
                "mamba_n_layer": MODEL_HPARAMS["mamba_n_layer"],
                "i_frame_interval": MODEL_HPARAMS["i_frame_interval"],
                "quantize_i_frame": MODEL_HPARAMS["quantize_i_frame"],
                "quantize_p_frame": MODEL_HPARAMS["quantize_p_frame"],
                "quant_scale": MODEL_HPARAMS["quant_scale"],
                "entropy_enabled": MODEL_HPARAMS["entropy_enabled"],
                "entropy_level": MODEL_HPARAMS["entropy_level"],
            },
        )

        sender.shutdown(drain_timeout=2.0)
        print(f"encoder_done: sent={sent}, window_size={args.window_size}, predictor=mamba")
    except Exception as e:
        print(f"执行出错: {e}")
