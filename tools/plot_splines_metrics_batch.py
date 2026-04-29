#!/usr/bin/env python3
"""
Visualize splines_metrics_batch results.

Reads metrics_summary.json from res/metrics_batch_test subdirectories,
organizes by metric, and generates plots for mean/median/p95/max.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


def parse_folder_name(folder_name):
    """Extract predictor and quantization level from folder name like 'abg_q4'."""
    parts = folder_name.rsplit("_", 1)
    if len(parts) == 2:
        predictor, q_level = parts
        return predictor, q_level
    return None, None


def load_metrics_data(base_dir):
    """
    Load all metrics from subdirectories in base_dir.
    
    Returns:
        dict: {(predictor, q_level): metrics_dict, ...}
    """
    data = {}
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"[ERROR] Directory not found: {base_dir}")
        return data
    
    for subfolder in sorted(base_path.iterdir()):
        if not subfolder.is_dir():
            continue
        
        folder_name = subfolder.name
        predictor, q_level = parse_folder_name(folder_name)
        
        if predictor is None or q_level is None:
            print(f"[SKIP] Skipping folder with unexpected name: {folder_name}")
            continue
        
        summary_path = subfolder / "metrics_summary.json"
        if not summary_path.exists():
            print(f"[SKIP] No metrics_summary.json in {folder_name}")
            continue
        
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
            data[(predictor, q_level)] = summary
            print(f"[OK] Loaded {folder_name}")
        except Exception as e:
            print(f"[ERROR] Failed to load {summary_path}: {e}")
    
    return data


def load_pose_bitrate_map(base_dir):
    """Load bitrate_kbps stats from baseline_* folders in pose metrics outputs."""
    bitrate_map = {}
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"[WARN] Pose metrics directory not found: {base_dir}")
        return bitrate_map

    for subfolder in sorted(base_path.iterdir()):
        if not subfolder.is_dir():
            continue

        predictor, q_level = parse_folder_name(subfolder.name)
        if predictor != "baseline" or q_level is None:
            continue

        summary_path = subfolder / "codec_metrics_summary.json"
        if not summary_path.exists():
            continue

        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
            per_file_stats = summary.get("per_file_stats", {})
            bitrate_stats = per_file_stats.get("bitrate_kbps", {})
            if isinstance(bitrate_stats, dict) and bitrate_stats:
                bitrate_map[q_level] = bitrate_stats
        except Exception as e:
            print(f"[WARN] Failed to load bitrate_kbps from {summary_path}: {e}")

    return bitrate_map


def extract_metric_keys(data):
    """Extract unique metric keys from loaded data."""
    metric_keys = set()
    for summary in data.values():
        if "per_file_stats" in summary:
            metric_keys.update(summary["per_file_stats"].keys())
    return sorted(metric_keys)


def organize_by_metric(data, metric_keys):
    """
    Organize data by metric.
    
    Returns:
        dict: {metric_key: {(predictor, q_level): {stat: value, ...}, ...}, ...}
    """
    by_metric = defaultdict(dict)
    
    for (predictor, q_level), summary in data.items():
        if "per_file_stats" not in summary:
            continue
        
        per_file_stats = summary["per_file_stats"]
        for metric in metric_keys:
            if metric in per_file_stats:
                by_metric[metric][(predictor, q_level)] = per_file_stats[metric]
    
    return by_metric


def get_q_level_order():
    """Return ordered list of quantization levels."""
    return ["q4", "q6", "q8", "q10", "q12", "q14", "q16", "q64"]


def get_q_label(q_level):
    """Get display label for quantization level."""
    if q_level == "q64":
        return "Original\n(64bits)"
    if q_level.startswith("q"):
        return f"{q_level[1:]}bits"
    return q_level


def _set_uniform_compression_axis(ax, font_size, x_min=0, x_max=10):
    """Set x-axis to uniform integer compression ticks with guide lines."""
    ticks = np.arange(x_min, x_max + 1, dtype=np.int64)
    ax.set_xlim(float(x_min), float(x_max))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=font_size)
    for t in ticks:
        ax.axvline(float(t), color="gray", linestyle=":", linewidth=0.8, alpha=0.35, zorder=0)


def _apply_bitrate_axis_break(raw_x_vals, q_order, compress_scale=0.15):
    """Compress the x-axis segment after q16 (towards q64) for display only."""
    disp_x_vals = np.array(raw_x_vals, dtype=np.float64, copy=True)

    try:
        q16_idx = q_order.index("q16")
        q64_idx = q_order.index("q64")
    except ValueError:
        return disp_x_vals, None

    q16_rate = raw_x_vals[q16_idx] if q16_idx < len(raw_x_vals) else np.nan
    q64_rate = raw_x_vals[q64_idx] if q64_idx < len(raw_x_vals) else np.nan
    if not (np.isfinite(q16_rate) and np.isfinite(q64_rate) and q64_rate > q16_rate):
        return disp_x_vals, None

    mask = np.isfinite(disp_x_vals) & (disp_x_vals > q16_rate)
    disp_x_vals[mask] = q16_rate + (disp_x_vals[mask] - q16_rate) * float(compress_scale)

    break_center = q16_rate + (q64_rate - q16_rate) * float(compress_scale) * 0.5
    return disp_x_vals, float(break_center)


def _configure_bitrate_axis(ax, raw_x_vals, disp_x_vals, q_order, font_size, break_center=None):
    """Set bitrate ticks from transformed x while showing original bitrate labels."""
    tick_pos = []
    tick_lbl = []
    for idx, q in enumerate(q_order):
        x_raw = raw_x_vals[idx]
        x_disp = disp_x_vals[idx]
        if np.isfinite(x_raw) and np.isfinite(x_disp):
            tick_pos.append(float(x_disp))
            tick_lbl.append(f"{x_raw:.1f}")

    if tick_pos:
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl, fontsize=font_size)

        x_min = min(tick_pos)
        x_max = max(tick_pos)
        if x_max <= x_min:
            x_max = x_min + 1.0
        pad = (x_max - x_min) * 0.05
        ax.set_xlim(x_min - pad, x_max + pad)

        for x in tick_pos:
            ax.axvline(x, color="gray", linestyle=":", linewidth=0.8, alpha=0.35, zorder=0)

    if break_center is not None:
        d = 0.012
        kwargs = dict(transform=ax.get_xaxis_transform(), color="black", clip_on=False, linewidth=1.2)
        ax.plot([break_center - 0.04, break_center - 0.01], [-d, +d], **kwargs)
        ax.plot([break_center + 0.01, break_center + 0.04], [-d, +d], **kwargs)


def _plot_smooth_series(ax, x_vals, y_vals, color, label, linestyle="-"):
    """Plot smoothed curve on arbitrary x-values and overlay scatter points."""
    valid = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_valid = x_vals[valid]
    y_valid = y_vals[valid]
    if y_valid.size == 0:
        return

    order = np.argsort(x_valid)
    x_valid = x_valid[order]
    y_valid = y_valid[order]

    unique_x, inv = np.unique(x_valid, return_inverse=True)
    if unique_x.size != x_valid.size:
        agg_y = np.empty(unique_x.size, dtype=np.float64)
        for i in range(unique_x.size):
            agg_y[i] = np.mean(y_valid[inv == i])
        x_valid = unique_x
        y_valid = agg_y

    if y_valid.size > 1:
        k = min(3, y_valid.size - 1)
        spline = make_interp_spline(x_valid, y_valid, k=k)
        x_smooth = np.linspace(x_valid[0], x_valid[-1], 300)
        y_smooth = spline(x_smooth)
        ax.plot(x_smooth, y_smooth, label=label, color=color, linewidth=2, linestyle=linestyle)
    else:
        ax.plot(x_valid, y_valid, label=label, color=color, linewidth=2, linestyle=linestyle)

    ax.scatter(x_valid, y_valid, color=color, s=50, zorder=5)


def get_stat_title(stat_name):
    """Get display title for aggregation stat."""
    stat_map = {
        "mean": "mean",
        "median": "media",
        "p95": "p95",
        "max": "max",
    }
    return stat_map.get(stat_name, stat_name)


def get_metric_ylabel(metric_name):
    """Infer y-axis label with units from metric name."""
    base_name = metric_name
    if base_name.startswith("linear_"):
        base_name = base_name[len("linear_") :]

    if "bl_mpjpe" in base_name:
        return r"MPJPE$_{BL}$ (%)"
    if "mpjpe" in base_name:
        return "MPJPE (mm)"
    if "mpjve" in base_name:
        return "MPJVE (mm/s)"
    if "rte" in base_name:
        return "RTE (%)"
    if "jitter" in base_name:
        return r"Jitter (mm/s$^3$)"
    return metric_name


def plot_metric_stat(
    metric_name,
    stat_name,
    data_by_predictor,
    by_metric,
    bitrate_map,
    output_path,
    font_size=12,
    fig_size=(8, 8),
):
    """
    Plot a single metric+stat combination with smooth curves.
    
    Args:
        metric_name: e.g., 'mpjpe_pose_upsampled_mm' (non-linear metric)
        stat_name: 'mean', 'median', 'p95', or 'max'
        data_by_predictor: dict for this metric: {predictor: {q_level: {stat: value, ...}, ...}, ...}
        by_metric: full by_metric dict for accessing linear variants
        output_path: where to save the plot
    """
    q_order = get_q_level_order()
    keypoint_count = 17.0
    x_values_raw = np.asarray(
        [bitrate_map.get(q, {}).get(stat_name, np.nan) / keypoint_count for q in q_order],
        dtype=np.float64,
    )
    x_values_disp, break_center = _apply_bitrate_axis_break(
        x_values_raw,
        q_order,
        compress_scale=0.08,
    )
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot each predictor
    predictors_to_plot = ["baseline", "abg", "kalman"]
    legend_labels = {
        "baseline": "Ours",
        "abg": r"$\alpha$-$\beta$-$\gamma$ Filter",
        "kalman": "Kalman Filter",
        "baseline_linear": "Linear Interpolation",
    }
    colors = {"baseline": "blue", "abg": "orange", "kalman": "green"}
    
    for predictor in predictors_to_plot:
        values = []
        for q in q_order:
            if predictor in data_by_predictor and q in data_by_predictor[predictor]:
                value = data_by_predictor[predictor][q].get(stat_name, np.nan)
                values.append(value)
            else:
                values.append(np.nan)

        _plot_smooth_series(
            ax,
            x_values_disp,
            np.asarray(values, dtype=np.float64),
            color=colors.get(predictor, "gray"),
            label=legend_labels[predictor],
        )
    
    # Plot baseline_linear separately
    linear_metric = f"linear_{metric_name}"
    if linear_metric in by_metric:
        values_linear = []
        for q in q_order:
            if ("baseline", q) in by_metric[linear_metric]:
                value = by_metric[linear_metric][("baseline", q)].get(stat_name, np.nan)
                values_linear.append(value)
            else:
                values_linear.append(np.nan)

        _plot_smooth_series(
            ax,
            x_values_disp,
            np.asarray(values_linear, dtype=np.float64),
            color="red",
            label=legend_labels["baseline_linear"],
            linestyle="--",
        )
    
    # Customize plot
    _configure_bitrate_axis(
        ax,
        raw_x_vals=x_values_raw,
        disp_x_vals=x_values_disp,
        q_order=q_order,
        font_size=font_size,
        break_center=break_center,
    )
    ax.set_xlabel("Bitrate per Keypoint (kbps)", fontsize=font_size)
    ax.set_ylabel(get_metric_ylabel(metric_name), fontsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    ax.tick_params(axis="x", labelsize=font_size)
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = {label: handle for handle, label in zip(handles, labels)}
    preferred_order = [
        "Linear Interpolation",
        r"$\alpha$-$\beta$-$\gamma$ Filter",
        "Kalman Filter",
        "Ours",
    ]
    ordered_labels = [label for label in preferred_order if label in label_to_handle]
    if ordered_labels:
        ordered_handles = [label_to_handle[label] for label in ordered_labels]
        ax.legend(ordered_handles, ordered_labels, fontsize=font_size)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"[PLOT] Saved: {output_path}")


def main():
    base_dir = "res/metrics_batch_test"
    pose_base_dir = "res/pose_metrics_batch_test"
    output_base_dir = "res/metrics_batch_plots"
    font_size = 18
    fig_size = (8, 8)
    
    # Load all data
    print("Loading metrics data...")
    data = load_metrics_data(base_dir)
    bitrate_map = load_pose_bitrate_map(pose_base_dir)
    
    if not data:
        print("[ERROR] No data loaded. Exiting.")
        return
    
    print(f"[INFO] Loaded data for {len(data)} configurations")
    print(f"[INFO] Loaded bitrate map for {len(bitrate_map)} q-levels")
    
    # Extract and organize metrics
    metric_keys = extract_metric_keys(data)
    print(f"[INFO] Found {len(metric_keys)} metrics")
    
    by_metric = organize_by_metric(data, metric_keys)
    
    # Filter to non-linear metrics only (linear variants will be plotted alongside them)
    non_linear_metrics = [m for m in metric_keys if not m.startswith("linear_")]
    
    # For each non-linear metric, organize by predictor and generate plots
    for metric_name in sorted(non_linear_metrics):
        if metric_name not in by_metric:
            continue
            
        metric_data = by_metric[metric_name]
        
        # Organize by predictor
        data_by_predictor = defaultdict(dict)
        for (predictor, q_level), stats in metric_data.items():
            data_by_predictor[predictor][q_level] = stats
        
        # Create output directory for this metric
        metric_output_dir = Path(output_base_dir) / metric_name
        metric_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[METRIC] Processing {metric_name}")
        
        # Generate plots for each stat
        for stat in ["mean", "median", "p95", "max"]:
            output_path = metric_output_dir / f"{stat}.png"
            try:
                plot_metric_stat(
                    metric_name,
                    stat,
                    data_by_predictor,
                    by_metric,
                    bitrate_map,
                    output_path,
                    font_size=font_size,
                    fig_size=fig_size,
                )
            except Exception as e:
                print(f"[ERROR] Failed to plot {metric_name}/{stat}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n[DONE] All plots saved to {output_base_dir}")


if __name__ == "__main__":
    main()
