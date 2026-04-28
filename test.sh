#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

run_step() {
	local title="$1"
	shift

	echo ""
	echo "========== ${title} =========="
	echo "CMD: $*"
	"$@"
	local code=$?
	if [[ $code -ne 0 ]]; then
		echo "[ERROR] ${title} failed with exit code ${code}"
		exit "$code"
	fi
}

run_step \
	"1/6 entropy codec train" \
	python fea_extr_py_scripts/splines_entropy_codec_train.py --config-path checkpoints/grpc_offline_splines_codec_config.json

echo ""
echo "========== 2/6 receiver + 3/6 sender =========="
echo "Receiver will start first. Sender starts after 1 second."
echo "After sender finishes, press Ctrl+C to stop receiver, then workflow will continue."

(
	sleep 1
	echo "[INFO] launching sender..."
	python fea_extr_py_scripts/grpc_offline_splines_sender.py
) &
sender_pid=$!

echo "[INFO] launching receiver..."
python fea_extr_py_scripts/grpc_offline_splines_receiver.py
receiver_exit=$?

wait "$sender_pid"
sender_exit=$?

if [[ $sender_exit -ne 0 ]]; then
	echo "[ERROR] sender failed with exit code ${sender_exit}"
	exit "$sender_exit"
fi

if [[ $receiver_exit -ne 0 && $receiver_exit -ne 130 ]]; then
	echo "[ERROR] receiver exited unexpectedly with code ${receiver_exit}"
	exit "$receiver_exit"
fi

echo "[INFO] receiver stopped, continue next steps."

run_step \
	"4/6 realtime offline splines fit" \
	python fea_extr_py_scripts/realtime_offline_splines_fit.py

run_step \
	"5/6 splines metrics" \
	python tools/splines_metrics.py

run_step \
	"6/6 splines metrics batch" \
	python tools/splines_metrics_batch.py

echo ""
echo "All steps completed."
