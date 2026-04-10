import os
import re
import numpy as np
import matplotlib.pyplot as plt


def parse_motion_id_cam(filename_stem):
    # Expected stem format: <motion>_<id>_cam_<cam_num>[_h36m|_coco]
    match = re.match(
        r"^(?P<motion>.+)_(?P<id>\d+)_cam_(?P<cam>\d+)(?:_(?P<fmt>h36m|coco))?$",
        filename_stem,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return match.group("motion"), int(match.group("id")), int(match.group("cam"))


def npy_file_sort_key(filename):
    stem = os.path.splitext(filename)[0]
    parsed = parse_motion_id_cam(stem)

    if parsed is None:
        # Put unmatched names at the end and keep deterministic lexical order.
        return (1, stem.lower())

    motion_name, motion_id, cam_num = parsed
    # Primary sort by id (numeric), then motion and cam for stable ordering.
    return (0, motion_id, motion_name.lower(), cam_num, stem.lower())


def collect_npy_files(input_dir, sort_by_id=True, limit=None):
    npy_files = []

    for f in os.listdir(input_dir):
        full_path = os.path.join(input_dir, f)
        if not os.path.isfile(full_path):
            continue
        if not f.lower().endswith("h36m.npy"):
            continue

        stem = os.path.splitext(f)[0]
        parsed = parse_motion_id_cam(stem)
        # Only keep cam_1 files because cam_2/3/4 are duplicated content.
        if parsed is not None and parsed[2] != 1:
            continue

        npy_files.append(f)

    if sort_by_id:
        npy_files.sort(key=npy_file_sort_key)

    if limit is not None and limit > 0:
        npy_files = npy_files[:limit]

    return npy_files


def load_pose_array(npy_path):
    data = np.load(npy_path)

    # Common shape variants: (1, F, K, 3) or (F, K, 3)
    if data.ndim == 4 and data.shape[0] == 1:
        data = data[0]

    if data.ndim != 3 or data.shape[-1] != 3:
        raise ValueError(
            f"Unexpected npy shape {data.shape}. Expected (frames, keypoints, 3) or (1, frames, keypoints, 3)."
        )

    return data


def downsample_pose_array(pose_data, downsample_factor=4):
    if downsample_factor <= 0:
        raise ValueError(f"downsample_factor must be > 0, got {downsample_factor}")
    return pose_data[::downsample_factor]


def plot_xyz_features(pose_data, save_path, effective_fps, title_prefix=""):
    # pose_data: (num_frames, num_keypoints, 3)
    num_frames, num_keypoints, _ = pose_data.shape
    if effective_fps <= 0:
        raise ValueError(f"effective_fps must be > 0, got {effective_fps}")

    time_ms = np.arange(num_frames) * (1000.0 / effective_fps)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axis_names = ["X", "Y", "Z"]

    for axis_idx, axis_name in enumerate(axis_names):
        ax = axes[axis_idx]
        values = pose_data[:, :, axis_idx]  # (num_frames, num_keypoints)

        for kp_idx in range(num_keypoints):
            label = f"kp_{kp_idx}" if num_keypoints <= 20 else None
            ax.plot(time_ms, values[:, kp_idx], linewidth=0.8, alpha=0.8, label=label)

        ax.set_ylabel(f"{axis_name} (mm)")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        if num_keypoints <= 20:
            ax.legend(loc="upper right", fontsize=8, ncol=2)

    fig.suptitle(f"{title_prefix} Pose Features (mm) vs Time", fontsize=14)
    axes[-1].set_xlabel("Time (ms)")
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.97))
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def process_folder(input_dir, output_dir, sort_by_id=True, limit=None,
                   original_fps=120, downsample_factor=4):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if original_fps <= 0:
        raise ValueError(f"original_fps must be > 0, got {original_fps}")
    if downsample_factor <= 0:
        raise ValueError(f"downsample_factor must be > 0, got {downsample_factor}")

    effective_fps = original_fps / downsample_factor
    os.makedirs(output_dir, exist_ok=True)
    npy_files = collect_npy_files(input_dir, sort_by_id=sort_by_id, limit=limit)
    print(f"Found {len(npy_files)} npy file(s) in: {input_dir}")
    print(f"Downsample config: original_fps={original_fps}, downsample_factor={downsample_factor}, effective_fps={effective_fps}")

    processed = 0
    failed = 0

    for idx, npy_name in enumerate(npy_files, start=1):
        npy_path = os.path.join(input_dir, npy_name)
        base_name = os.path.splitext(npy_name)[0]
        save_path = os.path.join(output_dir, f"{base_name}_xyz_features.png")

        try:
            pose_data = load_pose_array(npy_path)
            pose_data = downsample_pose_array(pose_data, downsample_factor=downsample_factor)
            plot_xyz_features(pose_data, save_path, effective_fps=effective_fps, title_prefix=base_name)
            print(f"[{idx}/{len(npy_files)}] Saved: {save_path}")
            processed += 1
        except Exception as e:
            print(f"[{idx}/{len(npy_files)}] Failed: {npy_name} -> {e}")
            failed += 1

    print("=" * 60)
    print(f"Done. processed={processed}, failed={failed}, total_selected={len(npy_files)}")


if __name__ == "__main__":
    # -----------------------------
    # Direct config (edit here)
    # -----------------------------
    input_dir = "/home/data/ztw/AtheletePose3D/data/train_set/S3"
    # input_dir = "/home/ztw/HVCCS/res/decode_res"
    output_dir = "/home/ztw/HVCCS/res/Athelete3D120fpsFeatures"
    limit = 3               # e.g. 10, None means all npy files
    sort_by_id = True        # True: sort by motion_id_cam_num id, False: filesystem order
    original_fps = 120       # Raw data fps
    downsample_factor = 1    # Keep 1 frame, drop next 3 frames

    process_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        sort_by_id=sort_by_id,
        limit=limit,
        original_fps=original_fps,
        downsample_factor=downsample_factor,
    )
