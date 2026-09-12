#!/bin/bash
# Watches VM for new Stage 2 checkpoints, rsync-s them locally, keeps top-3 by val_loss.
# Run from local machine: bash code/sync_checkpoints.sh
# Exits when training log shows "Epoch [20/20]" or on Ctrl+C.

set -euo pipefail

REMOTE_DIR="LLM-T1-NC80:/mnt/kikiri-tts/StyleTTS2/logs/kokoro-david-v4"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)/results/checkpoints"
REMOTE_LOG="LLM-T1-NC80:/mnt/kikiri-tts/logs/stage2_v4.log"
LOCAL_LOG_DIR="$(cd "$(dirname "$0")/.." && pwd)/logs"
POLL_INTERVAL=60
TOP_K=3  # keep only top-K checkpoints by val_loss

mkdir -p "$LOCAL_DIR" "$LOCAL_LOG_DIR"
echo "[sync] Watching $REMOTE_DIR → $LOCAL_DIR (keeping top-${TOP_K} by val_loss)"
echo "[sync] Polling every ${POLL_INTERVAL}s. Ctrl+C to stop."

prune_checkpoints() {
    # Extract val_loss from each .pth filename via python, sort, delete worst
    local count
    count=$(ls "$LOCAL_DIR"/epoch_2nd_*.pth 2>/dev/null | wc -l)
    if [ "$count" -le "$TOP_K" ]; then return; fi

    python3 - "$LOCAL_DIR" "$TOP_K" <<'EOF'
import sys, os, torch, glob

local_dir, top_k = sys.argv[1], int(sys.argv[2])
files = glob.glob(os.path.join(local_dir, "epoch_2nd_*.pth"))

losses = []
for f in files:
    try:
        ckpt = torch.load(f, map_location="cpu", weights_only=False)
        losses.append((ckpt.get("val_loss", float("inf")), f))
    except Exception:
        losses.append((float("inf"), f))

losses.sort(key=lambda x: x[0])
to_delete = losses[top_k:]
for loss, f in to_delete:
    print(f"[sync] Removing epoch {os.path.basename(f)} (val_loss={loss:.4f})")
    os.remove(f)

print(f"[sync] Kept top-{top_k}: " + ", ".join(
    f"{os.path.basename(f)} ({l:.4f})" for l, f in losses[:top_k]
))
EOF
}

while true; do
    rsync -az --include="epoch_2nd_*.pth" --exclude="*" \
        "${REMOTE_DIR}/" "${LOCAL_DIR}/" 2>/dev/null
    rsync -az "${REMOTE_LOG}" "${LOCAL_LOG_DIR}/stage2_v4.log" 2>/dev/null

    count=$(ls "$LOCAL_DIR"/epoch_2nd_*.pth 2>/dev/null | wc -l | tr -d ' ')
    echo "[sync $(date '+%H:%M:%S')] ${count} checkpoints locally"

    prune_checkpoints

    if ssh LLM-T1-NC80 "grep -q 'Epoch \[20/20\]' /mnt/kikiri-tts/logs/stage2_v4.log 2>/dev/null"; then
        echo "[sync] Training complete. Final sync..."
        rsync -az --include="epoch_2nd_*.pth" --exclude="*" \
            "${REMOTE_DIR}/" "${LOCAL_DIR}/"
        rsync -az "${REMOTE_LOG}" "${LOCAL_LOG_DIR}/stage2_v4.log" 2>/dev/null
        prune_checkpoints
        echo "[sync] Done. Best checkpoints in: $LOCAL_DIR"
        break
    fi

    sleep "$POLL_INTERVAL"
done
