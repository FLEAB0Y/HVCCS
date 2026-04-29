#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

GT_POSE_DIR="/Users/twz/demo_sys_user/h36m_pose_cam_1_downsample/test/S2_cam_1_30fps"
GT_POSE_SUFFIX=".npy"
BASE_INPUT_DIR="res/sender_receiver_test"
BASE_OUTPUT_DIR="res/pose_metrics_batch_test"

total=0
success=0
fail=0

# Iterate over all subfolders in res/sender_receiver_test
# Subfolder names follow pattern: {predictor}_{q_level}, e.g., abg_q4, kalman_q8, baseline_q16
for folder in "${BASE_INPUT_DIR}"/*; do
	if [[ ! -d "$folder" ]]; then
		continue
	fi

	folder_name=$(basename "$folder")

	# Extract predictor and quantization level from folder name
	# Pattern: (abg|kalman|baseline)_q{bits}
	if [[ ! "$folder_name" =~ ^(abg|kalman|baseline)_q([0-9]+)$ ]]; then
		echo "[SKIP] ${folder_name} does not match pattern, skipping."
		continue
	fi

	predictor="${BASH_REMATCH[1]}"
	q_str="${BASH_REMATCH[2]}"
	quant_bits="$q_str"

	# Build pred_suffix and output directory
	pred_suffix="_${predictor}_q${q_str}_realtime_spline.npz"
	output_dir="${BASE_OUTPUT_DIR}/${folder_name}"

	total=$((total + 1))
	echo ""
	echo "========== [${total}] ${folder_name} =========="
	echo "  pred-dir:         ${folder}"
	echo "  pred-suffix:      ${pred_suffix}"
	echo "  quant-bits:       ${quant_bits}"
	echo "  i-frame-interval: 30"
	echo "  output-dir:       ${output_dir}"

	python tools/pose_codec_metrics_batch.py \
		--pred-dir "${folder}" \
		--gt-pose-dir "${GT_POSE_DIR}" \
		--output-dir "${output_dir}" \
		--pred-suffix "${pred_suffix}" \
		--gt-pose-suffix "${GT_POSE_SUFFIX}" \
		--gt-pose-fps 30.0 \
		--codec-fps 30.0 \
		--i-frame-interval 30 \
		--quant-bits "${quant_bits}"

	code=$?
	if [[ $code -ne 0 ]]; then
		echo "[ERROR] ${folder_name} failed with exit code ${code}"
		fail=$((fail + 1))
	else
		echo "[OK] ${folder_name} -> ${output_dir}"
		success=$((success + 1))
	fi
done

echo ""
echo "========== Done: ${success}/${total} succeeded, ${fail} failed =========="
