import os
import numpy as np
import matplotlib.pyplot as plt


# ----------------- Config -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input: ..\features\id08.txt
INPUT_FILE = os.path.join(BASE_DIR, "..", "features", "id08.txt")

# Output features: ..\res\upsample\features\id08_120fps.txt
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "res", "upsample", "features", "id08_120fps.txt")

# Visualization output dir: ..\res\upsample\visualization
VIS_OUTPUT_DIR = os.path.join(BASE_DIR, "..", "res", "upsample", "visualization")

DIM = 151           # number of channels
SOURCE_FPS = 30.0   # original fps
TARGET_FPS = 120.0  # target fps
SEG_POINTS = 5      # points per quadratic segment
VIS_CHANNEL = 64    # channel index to visualize (0 ~ DIM-1)
# --------------------------------------------------


def load_data(path, dim):
    """Load id08.txt, return numpy array (T, D)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"输入文件不存在: {path}")

    data_list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p for p in line.split(",") if p.strip() != ""]
            if len(parts) < dim:
                continue
            vals = []
            for i in range(dim):
                try:
                    vals.append(float(parts[i]))
                except ValueError:
                    vals.append(0.0)
            data_list.append(vals)

    data = np.array(data_list, dtype=np.float64)
    if data.shape[0] < 3:
        raise ValueError(f"有效帧数太少: {data.shape[0]}")
    print(f"读取到 {data.shape[0]} 帧，每帧 {data.shape[1]} 维")
    return data


def build_time_axis(num_frames, fps):
    dt = 1.0 / fps
    return np.arange(num_frames, dtype=np.float64) * dt


def fit_segment_quadratic(t_seg, y_seg):
    A = np.vstack([t_seg**2, t_seg, np.ones_like(t_seg)]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, y_seg, rcond=None)
    a, b, c = coeffs
    return a, b, c


def evaluate_quadratic(a, b, c, t):
    return a * t**2 + b * t + c


def piecewise_quadratic_fit(t, y, seg_points=5):
    n = len(t)
    if n <= seg_points:
        a, b, c = fit_segment_quadratic(t, y)
        return evaluate_quadratic(a, b, c, t)

    segments = []
    start = 0
    while start < n - 1:
        end = min(start + seg_points, n)
        if end - start < 3:
            if segments:
                prev_start, _ = segments[-1]
                segments[-1] = (prev_start, end)
            else:
                segments.append((start, end))
            break
        segments.append((start, end))
        start += seg_points - 1

    seg_coeffs = []
    for (s, e) in segments:
        a, b, c = fit_segment_quadratic(t[s:e], y[s:e])
        seg_coeffs.append((s, e, (a, b, c)))

    y_fit = np.zeros_like(y)
    counts = np.zeros_like(y, dtype=np.float64)

    for (s, e, (a, b, c)) in seg_coeffs:
        idx = np.arange(s, e)
        yi = evaluate_quadratic(a, b, c, t[idx])

        rel = (idx - s) / max(1, e - s - 1)
        w = 1.0 - np.abs(rel - 0.5) * 2.0
        w = np.clip(w, 0.1, 1.0)

        y_fit[idx] += yi * w
        counts[idx] += w

    counts[counts == 0] = 1.0
    y_fit /= counts
    return y_fit


def resample_piecewise_quadratic(t, y_fit, target_fps):
    duration = t[-1]
    target_dt = 1.0 / target_fps
    tout = np.arange(0.0, duration + 1e-8, target_dt, dtype=np.float64)
    yout = np.interp(tout, t, y_fit)
    return tout, yout


def process_all_channels(data, source_fps, target_fps, seg_points=5):
    T, D = data.shape
    t = build_time_axis(T, source_fps)

    y0 = data[:, 0]
    y0_fit = piecewise_quadratic_fit(t, y0, seg_points=seg_points)
    tout, y0_out = resample_piecewise_quadratic(t, y0_fit, target_fps)

    T_out = len(tout)
    out = np.zeros((T_out, D), dtype=np.float64)
    out[:, 0] = y0_out

    for d in range(1, D):
        yd = data[:, d]
        yd_fit = piecewise_quadratic_fit(t, yd, seg_points=seg_points)
        _, yd_out = resample_piecewise_quadratic(t, yd_fit, target_fps)
        out[:, d] = yd_out

    return tout, out


def save_data(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            line = ",".join(f"{v:.6f}" for v in row)
            f.write(line + ",\n")
    print(f"已保存到: {os.path.abspath(path)}")


def plot_channel(t_src, y_src, tout, y_out, channel_index, out_dir):
    """Save a plot of one channel: original vs fitted & upsampled."""
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(t_src, y_src, "o-", label="Original", markersize=3, alpha=0.7)
    plt.plot(tout, y_out, "-", label="Fitted + upsampled", linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel(f"Value")
    plt.title(f"Channel {channel_index}: piecewise quadratic fit & upsample")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"channel_{channel_index:03d}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"图像已保存到: {os.path.abspath(out_path)}")


def main():
    input_path = os.path.abspath(INPUT_FILE)
    data = load_data(input_path, DIM)
    T, D = data.shape

    tout, data_out = process_all_channels(
        data, SOURCE_FPS, TARGET_FPS, seg_points=SEG_POINTS
    )

    print(f"原始时长: {T / SOURCE_FPS:.3f}s, 新时长: {tout[-1]:.3f}s, 新帧数: {len(tout)}")

    output_path = os.path.abspath(OUTPUT_FILE)
    save_data(output_path, data_out)

    if 0 <= VIS_CHANNEL < D:
        t_src = build_time_axis(T, SOURCE_FPS)
        y_src = data[:, VIS_CHANNEL]
        _, y_fit_out = resample_piecewise_quadratic(
            t_src,
            piecewise_quadratic_fit(t_src, y_src, seg_points=SEG_POINTS),
            TARGET_FPS,
        )
        plot_channel(t_src, y_src, tout, y_fit_out, VIS_CHANNEL, VIS_OUTPUT_DIR)
    else:
        print(f"VIS_CHANNEL={VIS_CHANNEL} 超出范围 (0~{D-1})，跳过绘图。")


if __name__ == "__main__":
    main()