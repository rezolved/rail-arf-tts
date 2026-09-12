#!/bin/bash
# Streams every training step line live from the VM (no polling delay, no aggregation).
# Run from local machine: bash code/tail_steps.sh
# Ctrl+C to stop (does not kill the remote training process).

REMOTE_HOST="LLM-T1-NC80"
REMOTE_LOG="/mnt/kikiri-tts/logs/stage2_v5.log"

ssh "$REMOTE_HOST" "tail -n 20 -f $REMOTE_LOG" | grep --line-buffered -E "Epoch \[|Traceback|Error"
