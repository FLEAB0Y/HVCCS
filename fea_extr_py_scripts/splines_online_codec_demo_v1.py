from dataclasses import dataclass
import json
import os
import time
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline


@dataclass
class RealtimeSplineConfig:
    feature_file_path: str
    window_size: int = 5
    pred_len: int = 3
    spline_degree: int = 3
    quant_scale: float = 1000.0
    total_features: int = 151
    face_dims: int = 52
    velocity_decay: float = 1.0
    eval_frames_per_packet: int = 2
    vis_joint: int = 0
    vis_axis: int = 0
    debug: bool = False
    timer: bool = False
    encoder_log_dir: Optional[str] = None
    encoder_vis_dir: Optional[str] = None
    decoder_vis_dir: Optional[str] = None


def load_pose_features(
    file_path: str,
    total_features: int = 151,
    face_dims: int = 52,
) -> np.ndarray:
    """
    读取每行 151 维特征，只保留后 99 维 pose landmarks xyz。
    返回 shape: (num_frames, 99)。
    """
    pose_rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            parts = [x.strip() for x in line.split(",") if x.strip() != ""]
            if len(parts) < total_features:
                continue
            values = [float(v) for v in parts[:total_features]]
            pose_rows.append(values[face_dims:total_features])

    if not pose_rows:
        raise ValueError(f"文件 {file_path} 无可用姿态特征数据。")

    pose = np.asarray(pose_rows, dtype=np.float64)
    expected_pose_dims = total_features - face_dims
    if pose.shape[1] != expected_pose_dims:
        raise ValueError(
            f"姿态维度异常: got {pose.shape[1]}, expected {expected_pose_dims}"
        )
    return pose


class FuturePredictor:
    """低延迟实时未来帧预测器（默认速度外推，无需训练模型）。"""

    def __init__(self, pred_len: int, velocity_decay: float = 1.0):
        self.pred_len = pred_len
        self.velocity_decay = velocity_decay

    def predict(self, history: np.ndarray) -> np.ndarray:
        """
        history: (window_size, dims)
        return: (pred_len, dims)
        """
        if history.shape[0] < 2:
            return np.repeat(history[-1:, :], self.pred_len, axis=0)

        last = history[-1]
        prev = history[-2]
        velocity = last - prev

        preds = []
        for step in range(1, self.pred_len + 1):
            gain = self.velocity_decay ** (step - 1)
            preds.append(last + step * gain * velocity)
        return np.asarray(preds, dtype=np.float64)

class SplineProcessor:
    """实时滑动窗口样条拟合。"""

    def __init__(self, window_size: int = 50, pred_len: int = 10, degree: int = 3):
        self.window_size = window_size
        self.pred_len = pred_len
        self.degree = degree

    def fit(self, history: np.ndarray, predicted: np.ndarray) -> BSpline:
        """
        history: (window_size, dims)
        predicted: (pred_len, dims)
        返回带多维控制点的 BSpline。
        """
        points = np.vstack([history, predicted])
        x = np.arange(points.shape[0], dtype=np.float64)
        degree = min(self.degree, points.shape[0] - 1)
        if degree < 1:
            raise ValueError("样条拟合至少需要2个点。")
        spl = make_interp_spline(x, points, k=degree)
        return spl


class SplineCodec:
    """样条参数量化编码/解码。"""

    def __init__(self, quant_scale: float = 1000.0):
        if quant_scale <= 0:
            raise ValueError("quant_scale 必须 > 0")
        self.quant_scale = quant_scale

    def encode(self, spline: BSpline, t0_frame: int, frame_start: int) -> Dict:
        t = np.asarray(spline.t, dtype=np.float64)
        c = np.asarray(spline.c, dtype=np.float64)
        return {
            "t0_frame": int(t0_frame),
            "frame_start": int(frame_start),
            "k": int(spline.k),
            "quant_scale": float(self.quant_scale),
            "t_shape": list(t.shape),
            "c_shape": list(c.shape),
            "t_q": np.round(t * self.quant_scale).astype(np.int32).tolist(),
            "c_q": np.round(c * self.quant_scale).astype(np.int32).tolist(),
        }

    def decode(self, packet: Dict) -> BSpline:
        scale = float(packet["quant_scale"])
        t = np.asarray(packet["t_q"], dtype=np.float64).reshape(packet["t_shape"]) / scale
        c = np.asarray(packet["c_q"], dtype=np.float64).reshape(packet["c_shape"]) / scale
        k = int(packet["k"])
        return BSpline(t, c, k)


class InMemoryReceiver:
    """模拟接收端：解码并重建样条。"""

    def __init__(self, codec: SplineCodec):
        self.codec = codec
        self.reconstructed_packets: List[Dict] = []
        self.reconstructed_points: List[np.ndarray] = []
        self.reconstructed_frame_idx: List[int] = []

    def on_packet(self, packet: Dict, eval_points: np.ndarray, current_u: float) -> np.ndarray:
        spline = self.codec.decode(packet)
        y = spline(eval_points)
        y_current = spline(current_u)
        self.reconstructed_packets.append(packet)
        self.reconstructed_points.append(np.asarray(y_current, dtype=np.float64))
        self.reconstructed_frame_idx.append(int(packet["t0_frame"]))
        return np.asarray(y, dtype=np.float64)


def _ensure_output_dirs(config: RealtimeSplineConfig) -> Dict[str, str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    res_dir = os.path.join(project_root, "res")
    encoder_dir = config.encoder_vis_dir or os.path.join(res_dir, "splines_encoder")
    decoder_dir = config.decoder_vis_dir or os.path.join(res_dir, "splines_decoder")
    encoder_log_dir = config.encoder_log_dir or os.path.join(res_dir, "encoder_log")
    os.makedirs(encoder_dir, exist_ok=True)
    os.makedirs(decoder_dir, exist_ok=True)
    os.makedirs(encoder_log_dir, exist_ok=True)
    return {"encoder": encoder_dir, "decoder": decoder_dir, "encoder_log": encoder_log_dir}


def _plot_encoder_frame(
    save_path: str,
    frame_idx: int,
    full_x: np.ndarray,
    full_coord: np.ndarray,
    pred_x: np.ndarray,
    pred_coord: np.ndarray,
    spline_x: np.ndarray,
    spline_coord: np.ndarray,
    joint_idx: int,
    axis_name: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(11, 4.5), sharex=True)
    ax.plot(full_x, full_coord, color="tab:blue", linewidth=1.8, label=f"observed_{axis_name}")
    ax.scatter(pred_x, pred_coord, color="tab:red", s=20, label="predicted")
    ax.plot(spline_x, spline_coord, color="tab:purple", linewidth=1.6, alpha=0.9, label="spline")
    ax.set_ylabel(axis_name)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax.set_title(f"Encoder frame {frame_idx:05d} | joint={joint_idx} | axis={axis_name}")
    ax.set_xlabel("global frame index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()


def _plot_decoder_frame(
    save_path: str,
    frame_idx: int,
    frame_indices: List[int],
    values_coord: np.ndarray,
    joint_idx: int,
    axis_name: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(11, 4.5), sharex=True)
    ax.plot(frame_indices, values_coord, color="tab:blue", linewidth=2.0, label=f"reconstructed_{axis_name}")
    ax.scatter(frame_indices, values_coord, color="tab:blue", s=16)
    ax.set_ylabel(axis_name)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax.set_title(f"Decoder frame {frame_idx:05d} | joint={joint_idx} | axis={axis_name}")
    ax.set_xlabel("global frame index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()


def run_realtime_spline_pipeline(
    config: RealtimeSplineConfig,
    sender: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    """
    实时闭环：读取特征 -> 历史窗口预测 -> 样条拟合 -> 编码发送 -> 接收解码。
    sender 不传时默认本地回环。
    """
    pose = load_pose_features(
        file_path=config.feature_file_path,
        total_features=config.total_features,
        face_dims=config.face_dims,
    )

    predictor = FuturePredictor(config.pred_len, velocity_decay=config.velocity_decay)
    processor = SplineProcessor(
        window_size=config.window_size,
        pred_len=config.pred_len,
        degree=config.spline_degree,
    )
    codec = SplineCodec(config.quant_scale)
    receiver = InMemoryReceiver(codec)
    vis_dirs: Optional[Dict[str, str]] = _ensure_output_dirs(config) if config.debug else None

    num_frames, dims = pose.shape
    if num_frames < config.window_size:
        raise ValueError(
            f"帧数不足: num_frames={num_frames}, window_size={config.window_size}"
        )

    total_packets = 0
    errors = []
    frame_times = []

    for t0 in range(config.window_size - 1, num_frames):
        if config.timer:
            frame_tic = time.perf_counter()

        start = t0 - config.window_size + 1
        history = pose[start:t0 + 1]
        predicted = predictor.predict(history)

        spline = processor.fit(history, predicted)
        packet = codec.encode(spline, t0_frame=t0, frame_start=start)

        if config.debug and vis_dirs is not None:
            packet_id = total_packets + 1
            log_path = os.path.join(vis_dirs["encoder_log"], f"packet{packet_id:05d}.json")
            with open(log_path, "w", encoding="utf-8") as log_file:
                json.dump(packet, log_file, ensure_ascii=False)

        if sender is not None:
            sender(packet)

        # 在接收端重建当前与未来若干点，评估量化误差。
        local_eval = np.arange(
            config.window_size - 1,
            config.window_size - 1 + config.eval_frames_per_packet,
            dtype=np.float64,
        )
        reconstructed = receiver.on_packet(
            packet,
            local_eval,
            current_u=float(config.window_size - 1),
        )
        original_ref = spline(local_eval)
        mse = float(np.mean((reconstructed - original_ref) ** 2))
        errors.append(mse)

        if config.debug and vis_dirs is not None:
            vis_frame = total_packets + 1
            max_joint = max(0, dims // 3 - 1)
            joint_idx = max(0, min(config.vis_joint, max_joint))
            axis_idx = max(0, min(config.vis_axis, 2))
            axis_name = ["X", "Y", "Z"][axis_idx]
            coord_idx = joint_idx * 3 + axis_idx

            full_x = np.arange(0, t0 + 1, dtype=np.float64)
            full_coord = pose[:t0 + 1, coord_idx]
            pred_x = np.arange(t0 + 1, t0 + 1 + config.pred_len, dtype=np.float64)
            pred_coord = predicted[:, coord_idx]

            fine_u = np.linspace(0.0, config.window_size + config.pred_len - 1, 200)
            fine_global_x = start + fine_u
            fine_coord = spline(fine_u)[:, coord_idx]

            encoder_img = os.path.join(vis_dirs["encoder"], f"encoder_frame{vis_frame:05d}.png")
            _plot_encoder_frame(
                save_path=encoder_img,
                frame_idx=vis_frame,
                full_x=full_x,
                full_coord=full_coord,
                pred_x=pred_x,
                pred_coord=pred_coord,
                spline_x=fine_global_x,
                spline_coord=fine_coord,
                joint_idx=joint_idx,
                axis_name=axis_name,
            )

            decoder_values_coord = np.asarray(receiver.reconstructed_points, dtype=np.float64)[:, coord_idx]
            decoder_img = os.path.join(vis_dirs["decoder"], f"decoder_frame{vis_frame:05d}.png")
            _plot_decoder_frame(
                save_path=decoder_img,
                frame_idx=vis_frame,
                frame_indices=receiver.reconstructed_frame_idx,
                values_coord=decoder_values_coord,
                joint_idx=joint_idx,
                axis_name=axis_name,
            )

        total_packets += 1

        if config.timer:
            frame_time_ms = (time.perf_counter() - frame_tic) * 1000.0
            frame_times.append(frame_time_ms)
            print(f"[timer] frame {total_packets:05d} took {frame_time_ms:.3f} ms")

    avg_frame_time_ms = float(np.mean(frame_times)) if frame_times else 0.0
    if config.timer:
        print(f"[timer] average per-frame time: {avg_frame_time_ms:.3f} ms")

    return {
        "num_frames": int(num_frames),
        "pose_dims": int(dims),
        "packets_sent": int(total_packets),
        "avg_quant_mse": float(np.mean(errors)) if errors else 0.0,
        "avg_frame_time_ms": avg_frame_time_ms,
        "quant_scale": float(config.quant_scale),
        "encoder_log_dir": vis_dirs["encoder_log"] if vis_dirs is not None else None,
        "encoder_vis_dir": vis_dirs["encoder"] if vis_dirs is not None else None,
        "decoder_vis_dir": vis_dirs["decoder"] if vis_dirs is not None else None,
    }

def run_realtime_demo():
    config = RealtimeSplineConfig(
        feature_file_path="/home/ztw/HVCCS/quantitative_error_compensation/features/id01.txt",
        window_size=5,
        pred_len=3,
        spline_degree=3,
        quant_scale=2000.0,
        velocity_decay=0.97,
        eval_frames_per_packet=2,
        vis_joint=0,
        vis_axis=0,
        debug=True,
        timer=True,
        encoder_log_dir="/home/ztw/HVCCS/res/encoder_log",
        encoder_vis_dir="/home/ztw/HVCCS/res/splines_encoder",
        decoder_vis_dir="/home/ztw/HVCCS/res/splines_decoder",
    )
    summary = run_realtime_spline_pipeline(config)
    print("实时样条编码闭环完成:", summary)

if __name__ == "__main__":
    run_realtime_demo()