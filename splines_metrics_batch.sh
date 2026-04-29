#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

GT_SPLINES_DIR="/Users/twz/demo_sys_user/h36m_pose_cam_1/test/S2_cam_1_120fps_notaknot_splines"
GT_POSE_DIR="/Users/twz/demo_sys_user/h36m_pose_cam_1/test/S2_cam_1_120fps"
GT_SUFFIX="_notaknot_spline.npz"
GT_POSE_SUFFIX=".npy"
BASE_INPUT_DIR="res/sender_receiver_test"
BASE_OUTPUT_DIR="res/metrics_batch_test"

PREDICTORS=("abg" "kalman" "baseline")
Q_LEVELS=("q4" "q6" "q8" "q10" "q12" "q14" "q16" "q64")

total=0
success=0
fail=0

for predictor in "${PREDICTORS[@]}"; do
    for q in "${Q_LEVELS[@]}"; do
        folder="${predictor}_${q}"
        pred_dir="${BASE_INPUT_DIR}/${folder}"

        if [[ ! -d "$pred_dir" ]]; then
            echo "[SKIP] ${pred_dir} does not exist, skipping."
            continue
        fi

        pred_suffix="_${predictor}_${q}_realtime_spline.npz"
        output_dir="${BASE_OUTPUT_DIR}/${folder}"

        total=$((total + 1))
        echo ""
        echo "========== [${total}] ${folder} =========="
        echo "  pred-dir:    ${pred_dir}"
        echo "  pred-suffix: ${pred_suffix}"
        echo "  output-dir:  ${output_dir}"

        python tools/splines_metrics_batch.py \
            --gt-dir "${GT_SPLINES_DIR}" \
            --pred-dir "${pred_dir}" \
            --gt-pose-dir "${GT_POSE_DIR}" \
            --output-dir "${output_dir}" \
            --gt-suffix "${GT_SUFFIX}" \
            --pred-suffix "${pred_suffix}" \
            --gt-pose-suffix "${GT_POSE_SUFFIX}"

        code=$?
        if [[ $code -ne 0 ]]; then
            echo "[ERROR] ${folder} failed with exit code ${code}"
            fail=$((fail + 1))
        else
            echo "[OK] ${folder} -> ${output_dir}"
            success=$((success + 1))
        fi
    done
done

echo ""
echo "========== Done: ${success}/${total} succeeded, ${fail} failed =========="
