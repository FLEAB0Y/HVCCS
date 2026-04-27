import argparse
import json
import os
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from typing import Dict, Iterable, List, Tuple

import numpy as np


def to_tjc_from_npy(pose: np.ndarray, coord_dims: int = 3) -> np.ndarray:
    """Normalize common npy layouts into shape (T, J, C)."""
    arr = np.asarray(pose)
    arr = np.squeeze(arr)

    if arr.ndim == 3:
        if arr.shape[-1] >= coord_dims:
            return arr[..., :coord_dims]
        if arr.shape[1] >= coord_dims:
            return np.transpose(arr, (0, 2, 1))[..., :coord_dims]
        raise ValueError(f"cannot infer npy pose layout from shape {arr.shape}")

    if arr.ndim == 2:
        if arr.shape[1] % coord_dims == 0:
            joints = arr.shape[1] // coord_dims
            return arr.reshape(arr.shape[0], joints, coord_dims)
        if arr.shape[0] % coord_dims == 0:
            joints = arr.shape[0] // coord_dims
            return arr.T.reshape(arr.shape[1], joints, coord_dims)
        raise ValueError(f"cannot infer npy pose layout from shape {arr.shape}")

    if arr.ndim == 1:
        if arr.shape[0] % coord_dims != 0:
            raise ValueError(f"cannot infer npy pose layout from shape {arr.shape}")
        joints = arr.shape[0] // coord_dims
        return arr.reshape(1, joints, coord_dims)

    raise ValueError(f"unsupported npy pose shape: {arr.shape}")


def list_npy_files(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        if input_path.lower().endswith(".npy"):
            return [os.path.abspath(input_path)]
        raise ValueError(f"input file is not .npy: {input_path}")

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"input path does not exist: {input_path}")

    files = []
    for name in sorted(os.listdir(input_path)):
        if name.lower().endswith(".npy"):
            files.append(os.path.abspath(os.path.join(input_path, name)))

    if not files:
        raise ValueError(f"no .npy files found in: {input_path}")
    return files


def load_runtime_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"codec config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError("config root must be JSON object")

    for key in ("common", "sender", "receiver"):
        if key not in cfg or not isinstance(cfg[key], dict):
            raise ValueError(f"missing required config section: {key}")

    return cfg


def resolve_path(base_dir: str, value: str) -> str:
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(base_dir, value))


def _is_within_root(path_value: str, root_dir: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(path_value), os.path.abspath(root_dir)]) == os.path.abspath(root_dir)
    except ValueError:
        return False


def resolve_runtime_path(path_value: str, project_root: str) -> str:
    """
    Path policy:
    - absolute path: use as-is (for files outside HVCCS)
    - relative path: resolve under HVCCS root and forbid escaping the root
    """
    if os.path.isabs(path_value):
        return os.path.abspath(path_value)

    resolved = os.path.abspath(os.path.join(project_root, path_value))
    if not _is_within_root(resolved, project_root):
        raise ValueError(
            f"relative path escapes HVCCS root: {path_value}. "
            "Use an absolute path for resources outside HVCCS."
        )
    return resolved


def resolve_existing_path(path_value: str) -> str:
    if os.path.isabs(path_value):
        if os.path.exists(path_value):
            return path_value
        raise FileNotFoundError(f"path does not exist: {path_value}")

    candidates = [
        os.path.abspath(path_value),
        os.path.abspath(os.path.join(os.path.dirname(__file__), path_value)),
    ]

    uniq_candidates = []
    for candidate in candidates:
        if candidate not in uniq_candidates:
            uniq_candidates.append(candidate)

    for candidate in uniq_candidates:
        if os.path.exists(candidate):
            return candidate

    tried = "\n".join(uniq_candidates)
    raise FileNotFoundError(f"path not found. tried:\n{tried}")


def parse_quant_bits_list(value) -> List[int]:
    if isinstance(value, list):
        bits_list = [int(x) for x in value]
    else:
        bits_list = [int(value)]

    if not bits_list:
        raise ValueError("quant_bits_list is empty")

    uniq_sorted = sorted(set(bits_list))
    for bits in uniq_sorted:
        if bits < 1 or bits > 16:
            raise ValueError(f"quant_bits must be in [1,16], got {bits}")
    return uniq_sorted


def make_output_path_for_bits(base_output_path: str, quant_bits: int, use_suffix: bool) -> str:
    if not use_suffix:
        return base_output_path
    root, ext = os.path.splitext(base_output_path)
    ext = ext if ext else ".json"
    return f"{root}_q{quant_bits}{ext}"


def iter_frames_from_npy(files: Iterable[str], coord_dims: int, num_keypoints: int):
    for file_path in files:
        pose_array = np.load(file_path)
        pose_tjc = to_tjc_from_npy(pose_array, coord_dims=coord_dims)
        if pose_tjc.shape[1] != num_keypoints:
            raise ValueError(
                f"npy joint count mismatch: {file_path} has {pose_tjc.shape[1]} joints, "
                f"expected {num_keypoints}"
            )
        yield file_path, pose_tjc.astype(np.float32, copy=False)


def quantize_uniform(values: np.ndarray, quant_bits: int, clip_abs: float) -> np.ndarray:
    levels = 1 << quant_bits
    clipped = np.clip(values, -clip_abs, clip_abs)
    scale = (levels - 1) / (2.0 * clip_abs)
    q = np.rint((clipped + clip_abs) * scale)
    return q.astype(np.int32)


def dequantize_uniform(q_values: np.ndarray, quant_bits: int, clip_abs: float) -> np.ndarray:
    levels = 1 << quant_bits
    scale = (2.0 * clip_abs) / (levels - 1)
    return q_values.astype(np.float32) * scale - clip_abs


def collect_payload_abs_values(
    files: List[str],
    num_keypoints: int,
    coord_dims: int,
    i_frame_interval: int,
    include_i_frames: bool,
) -> np.ndarray:
    abs_values = []

    for _, sequence in iter_frames_from_npy(files, coord_dims=coord_dims, num_keypoints=num_keypoints):
        prev_recon = None
        sent_count = 0

        for frame in sequence:
            current_flat = frame.reshape(-1).astype(np.float32, copy=False)
            is_i_frame = prev_recon is None or (
                i_frame_interval > 0 and sent_count % i_frame_interval == 0
            )

            if is_i_frame:
                payload_values = current_flat
                prev_recon = current_flat
            else:
                payload_values = (current_flat - prev_recon).astype(np.float32, copy=False)
                prev_recon = current_flat

            if include_i_frames or (not is_i_frame):
                abs_values.append(np.abs(payload_values))

            sent_count += 1

    if not abs_values:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(abs_values).astype(np.float32, copy=False)


@dataclass
class SymbolStats:
    counts: np.ndarray
    i_frames: int
    p_frames: int
    total_frames: int


def collect_symbol_distribution(
    files: List[str],
    num_keypoints: int,
    coord_dims: int,
    i_frame_interval: int,
    quant_bits: int,
    clip_abs: float,
    include_i_frames: bool,
) -> SymbolStats:
    levels = 1 << quant_bits
    counts = np.zeros((levels,), dtype=np.int64)

    i_frames = 0
    p_frames = 0
    total_frames = 0

    for _, sequence in iter_frames_from_npy(files, coord_dims=coord_dims, num_keypoints=num_keypoints):
        prev_recon = None
        sent_count = 0

        for frame in sequence:
            total_frames += 1
            current_flat = frame.reshape(-1).astype(np.float32, copy=False)
            is_i_frame = prev_recon is None or (
                i_frame_interval > 0 and sent_count % i_frame_interval == 0
            )

            if is_i_frame:
                payload_values = current_flat
                i_frames += 1
            else:
                payload_values = (current_flat - prev_recon).astype(np.float32, copy=False)
                p_frames += 1

            q_values = quantize_uniform(payload_values, quant_bits=quant_bits, clip_abs=clip_abs)
            recon_values = dequantize_uniform(q_values, quant_bits=quant_bits, clip_abs=clip_abs)

            if is_i_frame:
                prev_recon = recon_values
            else:
                prev_recon = (prev_recon + recon_values).astype(np.float32, copy=False)

            if include_i_frames or (not is_i_frame):
                counts += np.bincount(q_values, minlength=levels)

            sent_count += 1

    return SymbolStats(
        counts=counts,
        i_frames=i_frames,
        p_frames=p_frames,
        total_frames=total_frames,
    )


def build_huffman_code_lengths(counts_arr: np.ndarray) -> Dict[int, int]:
    nonzero = [(int(sym), int(freq)) for sym, freq in enumerate(counts_arr.tolist()) if freq > 0]
    if not nonzero:
        return {}

    if len(nonzero) == 1:
        only_sym = nonzero[0][0]
        return {only_sym: 1}

    serial = count()
    heap: List[Tuple[int, int, dict]] = []
    for sym, freq in nonzero:
        node = {"sym": sym, "left": None, "right": None}
        heappush(heap, (freq, next(serial), node))

    while len(heap) > 1:
        freq_a, _, node_a = heappop(heap)
        freq_b, _, node_b = heappop(heap)
        parent = {"sym": None, "left": node_a, "right": node_b}
        heappush(heap, (freq_a + freq_b, next(serial), parent))

    root = heappop(heap)[2]
    lengths: Dict[int, int] = {}

    def dfs(node: dict, depth: int):
        if node["sym"] is not None:
            lengths[int(node["sym"])] = max(depth, 1)
            return
        dfs(node["left"], depth + 1)
        dfs(node["right"], depth + 1)

    dfs(root, 0)
    return lengths


def build_canonical_codes(code_lengths: Dict[int, int]) -> Dict[int, str]:
    if not code_lengths:
        return {}

    ordered = sorted(code_lengths.items(), key=lambda x: (x[1], x[0]))
    result: Dict[int, str] = {}

    code = 0
    prev_len = ordered[0][1]

    for sym, curr_len in ordered:
        if curr_len > prev_len:
            code <<= (curr_len - prev_len)
        result[sym] = format(code, f"0{curr_len}b")
        code += 1
        prev_len = curr_len

    return result


def summarize_and_export(
    output_path: str,
    input_path: str,
    files: List[str],
    quant_bits: int,
    clip_abs: float,
    clip_percentile: float,
    i_frame_interval: int,
    include_i_frames: bool,
    stats: SymbolStats,
    code_lengths: Dict[int, int],
    codes: Dict[int, str],
):
    counts = stats.counts
    total_symbols = int(np.sum(counts))
    if total_symbols <= 0:
        raise ValueError("no symbols collected, cannot export codebook")

    probs = counts.astype(np.float64) / float(total_symbols)
    nonzero_mask = counts > 0

    entropy_bits = float(-np.sum(probs[nonzero_mask] * np.log2(probs[nonzero_mask])))

    avg_code_len = 0.0
    huff_total_bits = 0
    for sym, freq in enumerate(counts.tolist()):
        if freq <= 0:
            continue
        length = int(code_lengths.get(sym, 0))
        huff_total_bits += int(freq) * length
        avg_code_len += float(freq) * length
    avg_code_len /= float(total_symbols)

    fixed_total_bits = total_symbols * quant_bits
    compression_ratio = float(huff_total_bits) / float(fixed_total_bits) if fixed_total_bits > 0 else 1.0

    top_symbols = sorted(
        [(int(sym), int(freq)) for sym, freq in enumerate(counts.tolist()) if freq > 0],
        key=lambda x: x[1],
        reverse=True,
    )

    symbols = []
    for sym, freq in top_symbols:
        prob = float(freq) / float(total_symbols)
        symbols.append(
            {
                "symbol": sym,
                "count": freq,
                "prob": prob,
                "code_len": int(code_lengths.get(sym, 0)),
                "code": codes.get(sym, ""),
            }
        )

    report = {
        "meta": {
            "input_path": os.path.abspath(input_path),
            "num_files": len(files),
            "files": files,
            "quant_bits": int(quant_bits),
            "levels": int(1 << quant_bits),
            "clip_abs": float(clip_abs),
            "clip_percentile": float(clip_percentile),
            "i_frame_interval": int(i_frame_interval),
            "include_i_frames": bool(include_i_frames),
            "total_frames": int(stats.total_frames),
            "i_frames": int(stats.i_frames),
            "p_frames": int(stats.p_frames),
            "total_symbols": int(total_symbols),
        },
        "analysis": {
            "entropy_bits_per_symbol": entropy_bits,
            "avg_huffman_bits_per_symbol": float(avg_code_len),
            "fixed_bits_per_symbol": int(quant_bits),
            "estimated_fixed_total_bits": int(fixed_total_bits),
            "estimated_huffman_total_bits": int(huff_total_bits),
            "estimated_compression_ratio_vs_fixed": compression_ratio,
            "estimated_saving_percent_vs_fixed": float((1.0 - compression_ratio) * 100.0),
        },
        "symbols": symbols,
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("entropy training done")
    print(f"output: {output_path}")
    print(f"files: {len(files)}")
    print(f"frames: total={stats.total_frames}, I={stats.i_frames}, P={stats.p_frames}")
    print(f"symbols: {total_symbols}")
    print(f"quant_bits: {quant_bits}, levels: {1 << quant_bits}")
    print(f"clip_abs: {clip_abs:.6f}")
    print(f"entropy H: {entropy_bits:.4f} bits/symbol")
    print(f"huffman avg: {avg_code_len:.4f} bits/symbol")
    print(f"fixed: {quant_bits:.4f} bits/symbol")
    print(f"ratio(huffman/fixed): {compression_ratio:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train entropy codebook(s) from config JSON."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="../checkpoints/grpc_online_splines_codec_config.json",
        help="Path to codec runtime config JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg_path = resolve_existing_path(args.config_path)

    cfg = load_runtime_config(cfg_path)
    common_cfg = cfg["common"]
    sender_cfg = cfg["sender"]
    train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train", {}), dict) else {}

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_dir = os.path.dirname(cfg_path)

    num_keypoints = int(common_cfg.get("num_keypoints", 17))
    coord_dims = int(common_cfg.get("coord_dims", 3))
    i_frame_interval = int(common_cfg.get("i_frame_interval", 30))
    include_i_frames = bool(
        train_cfg.get("include_i_frames", common_cfg.get("entropy_include_i_frames", False))
    )
    clip_percentile = float(
        train_cfg.get("clip_percentile", common_cfg.get("entropy_clip_percentile", 99.5))
    )
    configured_clip_abs = float(common_cfg.get("clip_abs", 0.0))

    quant_bits_source = train_cfg.get(
        "quant_bits_list",
        common_cfg.get("quant_bits_list", common_cfg.get("quant_bits", 8)),
    )
    quant_bits_list = parse_quant_bits_list(quant_bits_source)

    input_path_raw = str(
        train_cfg.get(
            "input_path",
            common_cfg.get("entropy_train_input_path", sender_cfg.get("feature_file", "")),
        )
    )
    if not input_path_raw:
        raise ValueError("missing training input path in config")
    input_path = resolve_runtime_path(input_path_raw, project_root=project_root)
    if os.path.isfile(input_path) and input_path.lower().endswith(".npy"):
        # If sender.feature_file is used directly, train on the whole directory by default.
        input_path = os.path.dirname(input_path)

    output_json_raw = str(
        train_cfg.get(
            "output_json",
            common_cfg.get(
                "entropy_codebook_path",
                "checkpoints/grpc_online_splines_entropy_codebook.json",
            ),
        )
    )
    base_output_json = resolve_runtime_path(output_json_raw, project_root=project_root)

    if num_keypoints <= 0 or coord_dims <= 0:
        raise ValueError("num_keypoints and coord_dims must be > 0")
    if i_frame_interval < 1:
        raise ValueError("i_frame_interval must be >= 1")
    if clip_percentile <= 0 or clip_percentile > 100:
        raise ValueError("clip_percentile must be in (0, 100]")

    files = list_npy_files(input_path)

    inferred_clip_abs = configured_clip_abs
    if inferred_clip_abs <= 0:
        abs_values = collect_payload_abs_values(
            files=files,
            num_keypoints=num_keypoints,
            coord_dims=coord_dims,
            i_frame_interval=i_frame_interval,
            include_i_frames=include_i_frames,
        )
        if abs_values.size <= 0:
            raise ValueError("failed to infer clip_abs: no payload values collected")
        inferred_clip_abs = float(np.percentile(abs_values, clip_percentile))
        inferred_clip_abs = max(inferred_clip_abs, 1e-6)

    multi_bits = len(quant_bits_list) > 1
    for quant_bits in quant_bits_list:
        output_json = make_output_path_for_bits(base_output_json, quant_bits=quant_bits, use_suffix=multi_bits)

        stats = collect_symbol_distribution(
            files=files,
            num_keypoints=num_keypoints,
            coord_dims=coord_dims,
            i_frame_interval=i_frame_interval,
            quant_bits=quant_bits,
            clip_abs=float(inferred_clip_abs),
            include_i_frames=include_i_frames,
        )

        code_lengths = build_huffman_code_lengths(stats.counts)
        codes = build_canonical_codes(code_lengths)

        summarize_and_export(
            output_path=output_json,
            input_path=input_path,
            files=files,
            quant_bits=quant_bits,
            clip_abs=float(inferred_clip_abs),
            clip_percentile=clip_percentile,
            i_frame_interval=i_frame_interval,
            include_i_frames=include_i_frames,
            stats=stats,
            code_lengths=code_lengths,
            codes=codes,
        )


if __name__ == "__main__":
    main()
