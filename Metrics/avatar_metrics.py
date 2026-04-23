import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, List

import numpy as np


# =========================
# Config / constants
# =========================
D_TOTAL = 151
BLEND_D = 52
POSE_START = 52
POSE_END = 151  # exclusive
N_LANDMARKS = 33
POSE_D = 99

EPS = 1e-6  # near-zero threshold (missing detection)


# =========================
# IO
# =========================
def load_feature_txt(path: str, expected_dim: int = D_TOTAL) -> np.ndarray:
    """
    Each line is a frame, comma-separated floats, trailing comma allowed.
    Returns: X (T, expected_dim) float32
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
                raise ValueError(f"{path} line {ln}: cannot parse floats.") from e

            if vec.size != expected_dim:
                raise ValueError(f"{path} line {ln}: expected {expected_dim}, got {vec.size}")
            rows.append(vec)

    if not rows:
        raise ValueError(f"{path}: no valid rows loaded.")
    return np.stack(rows, axis=0)


def split_blend_pose(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    X: (T,151)
    Returns:
      blend: (T,52)
      pose:  (T,33,3)
    """
    if X.ndim != 2 or X.shape[1] != D_TOTAL:
        raise ValueError(f"Expected X shape (T,{D_TOTAL}), got {X.shape}")

    blend = X[:, :BLEND_D]
    pose_flat = X[:, POSE_START:POSE_END]
    if pose_flat.shape[1] != POSE_D:
        raise ValueError(f"Pose slice should be {POSE_D} dims, got {pose_flat.shape[1]}")
    pose = pose_flat.reshape(X.shape[0], N_LANDMARKS, 3)
    return blend, pose


# =========================
# Valid frame mask
# =========================
def pose_valid_mask(
    pose: np.ndarray,
    min_valid_points: int = 20,
    zero_is_missing: bool = True,
) -> np.ndarray:
    """
    Frame is valid if:
      - finite
      - and >= min_valid_points landmarks have (x,y,z) not all near 0.
    """
    finite = np.isfinite(pose).all(axis=(1, 2))
    if not zero_is_missing:
        return finite

    nonzero_points = np.sum(np.any(np.abs(pose) > EPS, axis=2), axis=1)
    return finite & (nonzero_points >= min_valid_points)


# =========================
# No-GT FDD-like (unchanged)
# =========================
def no_gt_fdd_blendshape(blend: np.ndarray) -> float:
    per_dim_dyn = np.std(blend, axis=0, ddof=0)
    return float(np.mean(per_dim_dyn))


def no_gt_fdd_pose(pose: np.ndarray, valid_mask_arr: Optional[np.ndarray] = None) -> float:
    """
    mean_i std_t(||p_{t,i}||_2)
    """
    if valid_mask_arr is None:
        valid_mask_arr = np.ones((pose.shape[0],), dtype=bool)

    P = pose[valid_mask_arr]
    if P.shape[0] < 2:
        return float("nan")

    r = np.linalg.norm(P, axis=2)       # (Tv,33)
    dyn = np.std(r, axis=0, ddof=0)     # (33,)
    return float(np.mean(dyn))


# =========================
# Helpers for stats
# =========================
def _nan_percentile(x: np.ndarray, q: float) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def _nan_mean(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def _nan_median(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.median(x))


# =========================
# Jitter / spike metrics (speed & jerk)
# =========================
def compute_speed_jerk_stats(pose: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    """
    speed_t = mean_i ||p_t,i - p_{t-1,i}||_2
    jerk_t  = mean_i ||(p_t - p_{t-1}) - (p_{t-1} - p_{t-2})||_2
    """
    T = pose.shape[0]
    speed = np.full((T,), np.nan, dtype=np.float32)
    jerk = np.full((T,), np.nan, dtype=np.float32)

    dp = np.linalg.norm(pose[1:] - pose[:-1], axis=2)  # (T-1,33)
    speed[1:] = np.mean(dp, axis=1)

    ddp = (pose[2:] - pose[1:-1]) - (pose[1:-1] - pose[:-2])  # (T-2,33,3)
    ddp_norm = np.linalg.norm(ddp, axis=2)                    # (T-2,33)
    jerk[2:] = np.mean(ddp_norm, axis=1)

    speed[~valid] = np.nan
    jerk[~valid] = np.nan

    speed_p99 = _nan_percentile(speed, 99)
    jerk_p99 = _nan_percentile(jerk, 99)

    speed_spike_ratio = float(np.mean(speed[np.isfinite(speed)] > speed_p99)) if np.isfinite(speed_p99) else float("nan")
    jerk_spike_ratio = float(np.mean(jerk[np.isfinite(jerk)] > jerk_p99)) if np.isfinite(jerk_p99) else float("nan")

    return {
        "speed_mean": _nan_mean(speed),
        "speed_p95": _nan_percentile(speed, 95),
        "speed_p99": speed_p99,
        "speed_spike_ratio_gt_p99": speed_spike_ratio,
        "jerk_mean": _nan_mean(jerk),
        "jerk_p95": _nan_percentile(jerk, 95),
        "jerk_p99": jerk_p99,
        "jerk_spike_ratio_gt_p99": jerk_spike_ratio,
    }


# =========================
# Bone-length CV -> single score
# =========================
def _cv(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 5:
        return float("nan")
    mu = float(np.mean(x))
    if abs(mu) < EPS:
        return float("nan")
    return float(np.std(x, ddof=0) / abs(mu))


def bone_edges_mediapipe_like() -> Dict[str, Tuple[int, int]]:
    """
    MediaPipe Pose landmark indices (33):
      11/12 shoulders, 13/14 elbows, 15/16 wrists,
      23/24 hips, 25/26 knees, 27/28 ankles.
    """
    return {
        "shoulder_width": (11, 12),
        "hip_width": (23, 24),

        "left_upper_arm": (11, 13),
        "left_lower_arm": (13, 15),
        "right_upper_arm": (12, 14),
        "right_lower_arm": (14, 16),

        "left_thigh": (23, 25),
        "left_shin": (25, 27),
        "right_thigh": (24, 26),
        "right_shin": (26, 28),
    }


def compute_bone_cv_score(
    pose: np.ndarray,
    valid: np.ndarray,
    edges: Optional[Dict[str, Tuple[int, int]]] = None,
    agg: str = "mean",  # "mean" or "median"
) -> float:
    """
    Returns a SINGLE scalar:
      bone_cv_score = mean/median over selected bones of CV(length_t)
    Smaller is better (more physically consistent).
    """
    if edges is None:
        edges = bone_edges_mediapipe_like()

    P = pose[valid]
    if P.shape[0] < 5:
        return float("nan")

    cvs = []
    for _, (a, b) in edges.items():
        L = np.linalg.norm(P[:, a, :] - P[:, b, :], axis=1)
        cvs.append(_cv(L))

    cvs = np.asarray(cvs, dtype=np.float32)
    return _nan_mean(cvs) if agg == "mean" else _nan_median(cvs)


# =========================
# Symmetry -> single score
# =========================
def compute_symmetry_score(
    pose: np.ndarray,
    valid: np.ndarray,
    agg: str = "mean",  # "mean" or "median"
) -> float:
    """
    SINGLE scalar symmetry score, combining:
      (A) bone length symmetry (left vs right bone length)
      (B) dynamic symmetry (left vs right landmark dynamics)

    Smaller is better (more symmetric / structurally consistent).

    score = agg( [rel_diff_len_pairs..., rel_diff_dyn_pairs...] )
    """
    P = pose[valid]
    if P.shape[0] < 5:
        return float("nan")

    eps = 1e-6
    vals = []

    # (A) Length symmetry on left/right bone pairs
    bone_pairs = [
        ((11, 13), (12, 14)),  # upper arms
        ((13, 15), (14, 16)),  # lower arms
        ((23, 25), (24, 26)),  # thighs
        ((25, 27), (26, 28)),  # shins
    ]
    for (la, lb), (ra, rb) in bone_pairs:
        L = np.linalg.norm(P[:, la, :] - P[:, lb, :], axis=1)
        R = np.linalg.norm(P[:, ra, :] - P[:, rb, :], axis=1)
        denom = 0.5 * (L + R) + eps
        vals.append(float(np.mean(np.abs(L - R) / denom)))

    # (B) Dynamic symmetry on left/right landmark pairs
    r = np.linalg.norm(P, axis=2)           # (Tv,33)
    dyn = np.std(r, axis=0, ddof=0)         # (33,)

    landmark_pairs = [
        (11, 12),  # shoulders
        (13, 14),  # elbows
        (15, 16),  # wrists
        (23, 24),  # hips
        (25, 26),  # knees
        (27, 28),  # ankles
    ]
    for l, rr in landmark_pairs:
        denom = 0.5 * (dyn[l] + dyn[rr]) + eps
        vals.append(float(abs(dyn[l] - dyn[rr]) / denom))

    v = np.asarray(vals, dtype=np.float32)
    return _nan_mean(v) if agg == "mean" else _nan_median(v)


# =========================
# Report row
# =========================
@dataclass
class RowResult:
    file: str
    n_frames: int

    pose_valid_ratio: float

    # FDD-like (unchanged)
    fdd_like_blend: float
    fdd_like_pose: float

    # jitter/spike
    speed_mean: float
    speed_p95: float
    speed_p99: float
    speed_spike_ratio_gt_p99: float

    jerk_mean: float
    jerk_p95: float
    jerk_p99: float
    jerk_spike_ratio_gt_p99: float

    # single-score outputs
    bone_cv_score: float
    symmetry_score: float


def evaluate_one_file(
    path: str,
    min_valid_points: int = 20,
    bone_agg: str = "mean",
    sym_agg: str = "mean",
) -> Dict[str, object]:
    X = load_feature_txt(path, expected_dim=D_TOTAL)
    blend, pose = split_blend_pose(X)

    valid = pose_valid_mask(pose, min_valid_points=min_valid_points, zero_is_missing=True)
    valid_ratio = float(np.mean(valid))

    f_blend = no_gt_fdd_blendshape(blend)
    f_pose = no_gt_fdd_pose(pose, valid_mask_arr=valid)

    sj = compute_speed_jerk_stats(pose, valid)

    bone_score = compute_bone_cv_score(pose, valid, agg=bone_agg)
    sym_score = compute_symmetry_score(pose, valid, agg=sym_agg)

    row = RowResult(
        file=path,
        n_frames=int(X.shape[0]),
        pose_valid_ratio=valid_ratio,
        fdd_like_blend=float(f_blend),
        fdd_like_pose=float(f_pose),

        speed_mean=float(sj["speed_mean"]),
        speed_p95=float(sj["speed_p95"]),
        speed_p99=float(sj["speed_p99"]),
        speed_spike_ratio_gt_p99=float(sj["speed_spike_ratio_gt_p99"]),

        jerk_mean=float(sj["jerk_mean"]),
        jerk_p95=float(sj["jerk_p95"]),
        jerk_p99=float(sj["jerk_p99"]),
        jerk_spike_ratio_gt_p99=float(sj["jerk_spike_ratio_gt_p99"]),

        bone_cv_score=float(bone_score),
        symmetry_score=float(sym_score),
    )
    return asdict(row)


# =========================
# Batch / output
# =========================
def find_txt_files(root: str, recursive: bool = True) -> List[str]:
    out = []
    if recursive:
        for r, _, files in os.walk(root):
            for fn in files:
                if fn.lower().endswith(".txt"):
                    out.append(os.path.join(r, fn))
    else:
        for fn in os.listdir(root):
            if fn.lower().endswith(".txt"):
                out.append(os.path.join(root, fn))
    out.sort()
    return out


def write_csv(rows: List[Dict[str, object]], out_path: str) -> None:
    # Fixed header order
    ordered = [
        "file", "n_frames",
        "pose_valid_ratio",
        "fdd_like_blend", "fdd_like_pose",
        "speed_mean", "speed_p95", "speed_p99", "speed_spike_ratio_gt_p99",
        "jerk_mean", "jerk_p95", "jerk_p99", "jerk_spike_ratio_gt_p99",
        "bone_cv_score",
        "symmetry_score",
    ]
    import csv
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in ordered})


def write_jsonl(rows: List[Dict[str, object]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Batch no-GT pose/blend quality metrics for 151-dim per-frame txt files (single bone/symmetry scores)."
    )
    ap.add_argument("--input_dir", type=str, required=True, help="Directory containing .txt files")
    ap.add_argument("--out_dir", type=str, default=None, help="Output directory (default: input_dir)")
    ap.add_argument("--recursive", action="store_true", help="Recursively search subfolders")
    ap.add_argument("--min_valid_points", type=int, default=20, help="Min non-zero landmarks for a valid frame")

    ap.add_argument("--bone_agg", type=str, default="mean", choices=["mean", "median"],
                    help="Aggregation over bone CVs to a single score")
    ap.add_argument("--sym_agg", type=str, default="mean", choices=["mean", "median"],
                    help="Aggregation over symmetry components to a single score")

    args = ap.parse_args()

    in_dir = args.input_dir
    out_dir = args.out_dir or in_dir
    os.makedirs(out_dir, exist_ok=True)

    files = find_txt_files(in_dir, recursive=args.recursive)
    if not files:
        raise SystemExit(f"No .txt files found in {in_dir} (recursive={args.recursive})")

    rows = []
    errors = []
    for fp in files:
        try:
            rows.append(evaluate_one_file(
                fp,
                min_valid_points=args.min_valid_points,
                bone_agg=args.bone_agg,
                sym_agg=args.sym_agg,
            ))
        except Exception as e:
            errors.append({"file": fp, "error": repr(e)})

    csv_path = os.path.join(out_dir, "pose_quality_report_single_scores.csv")
    jsonl_path = os.path.join(out_dir, "pose_quality_report_single_scores.jsonl")
    write_csv(rows, csv_path)
    write_jsonl(rows, jsonl_path)

    if errors:
        err_path = os.path.join(out_dir, "pose_quality_errors.jsonl")
        with open(err_path, "w", encoding="utf-8") as f:
            for er in errors:
                f.write(json.dumps(er, ensure_ascii=False) + "\n")
        print(f"[WARN] {len(errors)} files failed. See: {err_path}")

    print(f"[OK] Processed: {len(rows)} files")
    print(f"[OK] CSV : {csv_path}")
    print(f"[OK] JSONL: {jsonl_path}")


if __name__ == "__main__":
    main()
