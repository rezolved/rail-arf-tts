#!/bin/bash
# Live progress display for Stage 2 training — run in a separate terminal.
# Shows an epoch bar, a step bar, and the key losses (Dur, LM) updating in place.
# Run from local machine: bash code/live_progress.sh
# Ctrl+C to stop (does not kill the remote training process).

REMOTE_HOST="LLM-T1-NC80"
REMOTE_LOG="/mnt/kikiri-tts/logs/stage2_v5.log"
TOTAL_EPOCHS=10
STEPS_PER_EPOCH=194
BAR_WIDTH=30

draw_bar() {
    local current="$1" total="$2" width="$3"
    local filled=$((current * width / total))
    ((filled > width)) && filled=$width
    local bar
    bar=$(printf "%${filled}s" | tr ' ' '#')
    bar="${bar}$(printf "%$((width - filled))s" | tr ' ' '-')"
    echo -n "$bar"
}

echo "[live] Watching ${REMOTE_HOST}:${REMOTE_LOG}. Ctrl+C to stop."
echo

ssh "$REMOTE_HOST" "tail -n 0 -F $REMOTE_LOG" | while IFS= read -r line; do
    if [[ "$line" == *"Traceback"* ]]; then
        echo
        echo "[live] !!! CRASH — traceback in log !!!"
        continue
    fi

    if [[ "$line" =~ ^Epochs:\ ([0-9]+)$ ]]; then
        echo
        echo "[live] Epoch ${BASH_REMATCH[1]} complete, saving checkpoint..."
        continue
    fi

    if [[ "$line" =~ Epoch\ \[([0-9]+)/([0-9]+)\],\ Step\ \[([0-9]+)/([0-9]+)\] ]]; then
        epoch="${BASH_REMATCH[1]}"
        epoch_total="${BASH_REMATCH[2]}"
        step="${BASH_REMATCH[3]}"
        step_total="${BASH_REMATCH[4]}"

        dur=$(grep -oE "Dur Loss: [0-9.]+" <<<"$line" | grep -oE "[0-9.]+")
        lm=$(grep -oE "LM Loss: [0-9.]+" <<<"$line" | grep -oE "[0-9.]+")

        epoch_bar=$(draw_bar "$epoch" "$epoch_total" "$BAR_WIDTH")
        step_bar=$(draw_bar "$step" "$step_total" "$BAR_WIDTH")

        printf "\r\033[KEpoch [%s] %d/%d  |  Step [%s] %d/%d  |  Dur: %-8s LM: %-8s" \
            "$epoch_bar" "$epoch" "$epoch_total" \
            "$step_bar" "$step" "$step_total" \
            "$dur" "$lm"
    fi
done
