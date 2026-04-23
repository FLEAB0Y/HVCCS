import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from pathlib import Path

# -------------------------
# Config
# -------------------------
D_TOTAL = 151
BLEND_D = 52
POSE_START = 52
POSE_END = 151  # exclusive
N_LANDMARKS = 33
POSE_D = 99  # 33 * 3

EPS = 1e-6


# -------------------------
# IO: load txt (each line = a frame, comma separated, trailing comma allowed)
# -------------------------
def load_feature_txt(path: str, expected_dim: int = D_TOTAL) -> np.ndarray:
    """
    Load a per-frame txt:
      - each line is a frame
      - comma-separated floats
      - trailing comma may exist
      - expected 151 dims per line (per README-2)

    Returns
      X: (T, expected_dim) float32
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if line.endswith(","):
                line = line[:-1]

            parts = [p for p in line.split(",") if p != ""]
            try:
                vec = np.asarray([float(x) for x in parts], dtype=np.float32)
            except ValueError as e:
                raise ValueError(f"Line {ln}: cannot parse floats. Example: {line[:120]}...") from e

            if vec.size != expected_dim:
                raise ValueError(f"Line {ln}: expected {expected_dim} dims, got {vec.size}")
            rows.append(vec)

    if not rows:
        raise ValueError("No valid rows loaded from file.")
    return np.stack(rows, axis=0)


# -------------------------
# Feature slicing
# -------------------------
def split_blend_pose(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    X: (T,151)
    Returns:
      blend: (T,52)
      pose:  (T,33,3)
    """
    if X.ndim != 2 or X.shape[1] != D_TOTAL:
        raise ValueError(f"Expected X shape (T,{D_TOTAL}), got {X.shape}")

    blend = X[:, :BLEND_D]  # (T,52)
    pose_flat = X[:, POSE_START:POSE_END]  # (T,99)
    if pose_flat.shape[1] != POSE_D:
        raise ValueError(f"Pose slice should be {POSE_D} dims, got {pose_flat.shape[1]}")
    pose = pose_flat.reshape(X.shape[0], N_LANDMARKS, 3)  # (T,33,3)
    return blend, pose


# -------------------------
# Valid frame detection for pose
# -------------------------
def pose_valid_mask(
    pose: np.ndarray,
    min_valid_points: int = 20,
    zero_is_missing: bool = True
) -> np.ndarray:
    """
    Heuristic validity:
      - frame is valid if >= min_valid_points landmarks are non-zero (any of x,y,z non-zero)
      - also requires finite numbers

    pose: (T,33,3)
    returns: (T,) bool
    """
    finite = np.isfinite(pose).all(axis=(1, 2))
    if not zero_is_missing:
        return finite

    nonzero_points = np.sum(np.any(np.abs(pose) > EPS, axis=2), axis=1)  # (T,)
    return finite & (nonzero_points >= min_valid_points)


# -------------------------
# No-GT "FDD-like" proxy metrics
# -------------------------
def no_gt_fdd_blendshape(blend: np.ndarray) -> float:
    """
    No-GT FDD-like for blendshape:
      mean over 52 dims of std over time.

    blend: (T,52)
    """
    if blend.ndim != 2 or blend.shape[1] != BLEND_D:
        raise ValueError(f"Expected blend shape (T,{BLEND_D}), got {blend.shape}")
    per_dim_dyn = np.std(blend, axis=0)  # (52,)
    return float(np.mean(per_dim_dyn))


def no_gt_fdd_pose(pose: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> float:
    """
    No-GT FDD-like for pose landmarks:
      for each landmark i:
        r_t = ||p_t,i||_2
        dyn_i = std_t(r_t)
      score = mean_i dyn_i

    pose: (T,33,3)
    valid_mask: (T,) bool. If provided, compute only on valid frames.
    """
    if pose.ndim != 3 or pose.shape[1:] != (N_LANDMARKS, 3):
        raise ValueError(f"Expected pose shape (T,{N_LANDMARKS},3), got {pose.shape}")

    if valid_mask is None:
        valid_mask = np.ones((pose.shape[0],), dtype=bool)

    P = pose[valid_mask]  # (Tv,33,3)
    if P.shape[0] < 2:
        # Not enough frames to compute std meaningfully
        return float("nan")

    r = np.linalg.norm(P, axis=2)  # (Tv,33)
    dyn = np.std(r, axis=0)        # (33,)
    return float(np.mean(dyn))


# -------------------------
# Report
# -------------------------
@dataclass
class NoGtFddReport:
    n_frames: int
    blend_all_zero: bool
    pose_valid_ratio: float
    fdd_like_blend: float
    fdd_like_pose: float


def compute_no_gt_fdd_scores(txt_path: str, min_valid_points: int = 20) -> NoGtFddReport:
    X = load_feature_txt(txt_path, expected_dim=D_TOTAL)
    blend, pose = split_blend_pose(X)

    # blend diagnostics
    blend_all_zero = bool(np.all(np.abs(blend) <= EPS))

    # pose validity
    valid = pose_valid_mask(pose, min_valid_points=min_valid_points, zero_is_missing=True)
    valid_ratio = float(np.mean(valid))

    # scores
    f_blend = no_gt_fdd_blendshape(blend)
    f_pose = no_gt_fdd_pose(pose, valid_mask=valid)

    return NoGtFddReport(
        n_frames=int(X.shape[0]),
        blend_all_zero=blend_all_zero,
        pose_valid_ratio=valid_ratio,
        fdd_like_blend=f_blend,
        fdd_like_pose=f_pose
    )


def compute_no_gt_fdd_scores_in_dir(
    dir_path: str,
    min_valid_points: int = 20,
    pattern: str = "*.txt"
) -> Dict[str, NoGtFddReport]:
    reports: Dict[str, NoGtFddReport] = {}
    p = Path(dir_path)
    if not p.is_dir():
        raise ValueError(f"Not a directory: {dir_path}")

    for fp in sorted(p.glob(pattern)):
        if not fp.is_file():
            continue
        try:
            rep = compute_no_gt_fdd_scores(str(fp), min_valid_points=min_valid_points)
            reports[fp.name] = rep
        except Exception as e:
            print(f"[WARN] Skip {fp.name}: {e}")
    return reports


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # 直接设置目标文件夹路径
    DIR_PATH = "quantitative_error_compensation/upsample_features"  # TODO: 修改为实际目录
    reports = compute_no_gt_fdd_scores_in_dir(DIR_PATH, min_valid_points=20)

    print("==== No-GT FDD-like Scores (Directory) ====")
    for name, report in reports.items():
        print(f"\nFile: {name}")
        print(f"Frames: {report.n_frames}")
        print(f"Blend all-zero: {report.blend_all_zero}")
        print(f"Pose valid frame ratio: {report.pose_valid_ratio:.4f}")
        print(f"Blendshape no-GT FDD-like: {report.fdd_like_blend:.6f}")
        print(f"PoseLandmark no-GT FDD-like: {report.fdd_like_pose:.6f}")
