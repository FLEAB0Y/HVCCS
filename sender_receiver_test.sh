#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_PATH="checkpoints/grpc_offline_splines_codec_config.json"
Npy_ROOT="/Users/twz/demo_sys_user/h36m_pose_cam_1_downsample/test/S2_cam_1_30fps"

predictors=(abg kalman baseline)
quant_bits_list=(4 6 8 10 12 14 16)

selected_predictors=()
selected_quant_bits=()

usage() {
	echo "Usage: bash sender_receiver_test.sh [--predictor NAME]... [--q BITS]..."
	echo ""
	echo "Options:"
	echo "  --predictor NAME   Run only selected predictor(s): abg | kalman | baseline"
	echo "  --q BITS           Run only selected quant bits: 4 | 6 | 8 | 10 | 12 | 14 | 16 | 64"
	echo "                     (64 means disable quantization and entropy coding)"
	echo "  -h, --help         Show this help"
}

contains_item() {
	local needle="$1"
	shift
	local item
	for item in "$@"; do
		if [[ "$item" == "$needle" ]]; then
			return 0
		fi
	done
	return 1
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--predictor)
			if [[ $# -lt 2 ]]; then
				echo "[ERROR] --predictor requires a value"
				usage
				exit 1
			fi
			if ! contains_item "$2" "${predictors[@]}"; then
				echo "[ERROR] invalid predictor: $2"
				usage
				exit 1
			fi
			selected_predictors+=("$2")
			shift 2
			;;
		--q)
			if [[ $# -lt 2 ]]; then
				echo "[ERROR] --q requires a value"
				usage
				exit 1
			fi
			if [[ "$2" != "64" ]] && ! contains_item "$2" "${quant_bits_list[@]}"; then
				echo "[ERROR] invalid q value: $2"
				usage
				exit 1
			fi
			selected_quant_bits+=("$2")
			shift 2
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			echo "[ERROR] unknown argument: $1"
			usage
			exit 1
			;;
	esac
done

if [[ ${#selected_predictors[@]} -eq 0 ]]; then
	selected_predictors=("${predictors[@]}")
fi

if [[ ${#selected_quant_bits[@]} -eq 0 ]]; then
	selected_quant_bits=("${quant_bits_list[@]}")
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
	echo "[ERROR] config not found: $CONFIG_PATH"
	exit 1
fi

if [[ ! -d "$Npy_ROOT" ]]; then
	echo "[ERROR] npy root not found: $Npy_ROOT"
	exit 1
fi

npy_files=()
if builtin command -v mapfile >/dev/null 2>&1; then
	mapfile -t npy_files < <(find "$Npy_ROOT" -maxdepth 1 -type f -name "*.npy" | sort)
else
	while IFS= read -r file_path; do
		npy_files+=("$file_path")
	done < <(find "$Npy_ROOT" -maxdepth 1 -type f -name "*.npy" | sort)
fi
if [[ ${#npy_files[@]} -eq 0 ]]; then
	echo "[ERROR] no .npy files found under: $Npy_ROOT"
	exit 1
fi

config_backup="${CONFIG_PATH}.bak_sender_receiver_test"
cp "$CONFIG_PATH" "$config_backup"
restore_config() {
	if [[ -f "$config_backup" ]]; then
		mv "$config_backup" "$CONFIG_PATH"
	fi
}
trap restore_config EXIT

update_config_for_case() {
	local predictor="$1"
	local quant_bits="$2"
	local codebook_path="$3"
	local feature_file="$4"
	local save_dir="$5"
	local spline_save_file="$6"

	python - "$CONFIG_PATH" "$predictor" "$quant_bits" "$codebook_path" "$feature_file" "$save_dir" "$spline_save_file" <<'PY'
import json
import sys

config_path, predictor, quant_bits, codebook_path, feature_file, save_dir, spline_save_file = sys.argv[1:]
quant_bits = int(quant_bits)

with open(config_path, "r", encoding="utf-8") as f:
	cfg = json.load(f)

cfg.setdefault("common", {})
cfg.setdefault("sender", {})
cfg.setdefault("receiver", {})

if quant_bits == 64:
	cfg["common"]["quantize"] = False
	cfg["common"]["quantize_i_frame"] = False
	cfg["common"]["quantize_p_frame"] = False
	cfg["common"]["entropy_enabled"] = False
	cfg["common"]["entropy_codec"] = "none"
else:
	with open(codebook_path, "r", encoding="utf-8") as f:
		codebook_obj = json.load(f)

	codebook_meta = codebook_obj.get("meta", {}) if isinstance(codebook_obj, dict) else {}
	codebook_clip_abs = codebook_meta.get("clip_abs", None)

	cfg["common"]["quantize"] = True
	cfg["common"]["quantize_i_frame"] = bool(cfg["common"].get("quantize_i_frame", cfg["common"].get("quantize", True)))
	cfg["common"]["quantize_p_frame"] = bool(cfg["common"].get("quantize_p_frame", cfg["common"].get("quantize", True)))
	cfg["common"]["entropy_enabled"] = True
	cfg["common"]["entropy_codec"] = "huffman"
	cfg["common"]["quant_bits"] = quant_bits
	cfg["common"]["entropy_codebook_path"] = codebook_path
	if codebook_clip_abs is not None:
		cfg["common"]["clip_abs"] = float(codebook_clip_abs)

cfg["sender"]["feature_file"] = feature_file

cfg["receiver"]["predictor_type"] = predictor
cfg["receiver"]["idle_timeout_sec"] = 2.0
cfg["receiver"]["spline_fit_enabled"] = True
cfg["receiver"]["save_dir"] = save_dir
cfg["receiver"]["spline_save_file"] = spline_save_file

with open(config_path, "w", encoding="utf-8") as f:
	json.dump(cfg, f, ensure_ascii=False, indent=4)
	f.write("\n")
PY
}

run_one_case() {
	local predictor="$1"
	local quant_bits="$2"
	local codebook_path="$3"
	local npy_file="$4"

	local stem
	stem="$(basename "$npy_file" .npy)"
	local result_dir="res/sender_receiver_test/${predictor}_q${quant_bits}"
	local spline_save_file="${stem}_${predictor}_q${quant_bits}_realtime_spline.npz"

	mkdir -p "$result_dir"

	update_config_for_case \
		"$predictor" \
		"$quant_bits" \
		"$codebook_path" \
		"$npy_file" \
		"$result_dir" \
		"$spline_save_file"

	echo ""
	echo "========== predictor=${predictor} | q=${quant_bits} | file=${stem} =========="
	echo "[INFO] receiver first, then sender"

	python fea_extr_py_scripts/grpc_offline_splines_receiver.py &
	local receiver_pid=$!

	sleep 1

	python fea_extr_py_scripts/grpc_offline_splines_sender.py
	local sender_exit=$?
	if [[ $sender_exit -ne 0 ]]; then
		echo "[ERROR] sender failed with exit code ${sender_exit}"
		kill "$receiver_pid" >/dev/null 2>&1 || true
		wait "$receiver_pid" >/dev/null 2>&1 || true
		return "$sender_exit"
	fi

	# receiver should exit automatically after idle_timeout_sec (2s).
	local wait_sec=0
	while kill -0 "$receiver_pid" >/dev/null 2>&1; do
		if [[ $wait_sec -ge 15 ]]; then
			echo "[ERROR] receiver did not exit in expected time"
			kill "$receiver_pid" >/dev/null 2>&1 || true
			wait "$receiver_pid" >/dev/null 2>&1 || true
			return 1
		fi
		sleep 1
		wait_sec=$((wait_sec + 1))
	done

	wait "$receiver_pid"
	local receiver_exit=$?
	if [[ $receiver_exit -ne 0 && $receiver_exit -ne 130 ]]; then
		echo "[ERROR] receiver exited unexpectedly with code ${receiver_exit}"
		return "$receiver_exit"
	fi

	echo "[INFO] done -> ${result_dir}/${spline_save_file}"
	return 0
}

echo "Total npy files: ${#npy_files[@]}"
echo "Predictors: ${selected_predictors[*]}"
echo "Quant bits: ${selected_quant_bits[*]}"

for predictor in "${selected_predictors[@]}"; do
	for quant_bits in "${selected_quant_bits[@]}"; do
		codebook_path=""
		if [[ "$quant_bits" == "64" ]]; then
			echo "[INFO] q=64 selected: disable quantization and entropy coding"
		else
			codebook_path="checkpoints/grpc_online_splines_entropy_codebook_q${quant_bits}.json"
			if [[ ! -f "$codebook_path" ]]; then
				echo "[ERROR] codebook not found: $codebook_path"
				exit 1
			fi
		fi

		for npy_file in "${npy_files[@]}"; do
			run_one_case "$predictor" "$quant_bits" "$codebook_path" "$npy_file"
			code=$?
			if [[ $code -ne 0 ]]; then
				echo "[ERROR] case failed: predictor=${predictor}, q=${quant_bits}, file=${npy_file}"
				exit "$code"
			fi
		done
	done
done

echo ""
echo "All sender/receiver cases completed."
