#!/bin/bash
# Launch Stage 2 Kokoro v4 fine-tune on LLM-T1-NC80.
# Run: ssh LLM-T1-NC80 "bash /mnt/kikiri-tts/run_stage2.sh"
#
# NOT launched via `accelerate launch`: train_second.py wraps every module in
# torch.nn.DataParallel itself, so it already spans both H100s. Adding accelerate's
# --num_processes 2 spawns two processes that each DataParallel over both GPUs
# (~4x the memory) and race on the same epoch_2nd_*.pth files — that combination
# produced both the epoch-5 OOM and the corrupted checkpoints in the earlier run.

set -euo pipefail

KIKIRI=/mnt/kikiri-tts
LOG=$KIKIRI/logs/stage2_v4.log

if pgrep -f train_second > /dev/null 2>&1; then
    echo "Already running (PID $(pgrep -f train_second | tr '\n' ' ')). Exiting."
    exit 1
fi

mkdir -p "$KIKIRI/logs"
cd "$KIKIRI/StyleTTS2"
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1
# image default /mnt/hf_home_cache is root-owned; WavLMLoss is built even at lambda_slm=0
export HF_HOME=$KIKIRI/hf_cache
mkdir -p "$HF_HOME"

echo "Launching Stage 2 (DataParallel over 2x H100)..."
nohup python -u train_second.py \
    --config_path ../configs/config_david_v4.yml \
    > "$LOG" 2>&1 &

PID=$!
echo "Started PID: $PID"
echo "Log: $LOG"

sleep 20
if ! kill -0 $PID 2>/dev/null; then
    echo "ERROR: process died. Last 30 log lines:"
    tail -30 "$LOG"
    exit 1
fi
echo "Process alive after 20s. Recent log:"
tail -15 "$LOG"
