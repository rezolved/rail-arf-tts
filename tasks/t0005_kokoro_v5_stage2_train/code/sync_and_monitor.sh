#!/bin/bash
# Watches VM for Stage 2 progress: rsyncs the log (fast, checked every cycle) and any new
# epoch_2nd_*.pth checkpoints (slow, ~1.6GB each), keeps only the top-K by val_loss on both
# the local copy and the VM's ephemeral disk, and prints an epoch progress bar.
#
# Also evaluates t0003/t0004's stop-early criterion the moment it's knowable: t0003 found v3's
# successful Stage 2 opened at Dur loss ~0.8 and every failed run (including t0001's v4) opened
# at 8-17. That's readable from the very first logged step, well before any checkpoint exists,
# so it's checked from the log alone on every poll and printed loudly once available.
#
# Run from local machine: bash code/sync_and_monitor.sh
# Exits when the log shows the final configured epoch, on a crash, or on Ctrl+C.

set -uo pipefail

REMOTE_HOST="LLM-T1-NC80"
REMOTE_CKPT_DIR="/mnt/kikiri-tts/StyleTTS2/logs/kokoro-david-v5"
REMOTE_LOG="/mnt/kikiri-tts/logs/stage2_v5.log"
CKPT_GLOB="epoch_2nd_*.pth"
TOTAL_EPOCHS=10
DUR_LOSS_HEALTHY_MAX=2.0 # v3's successful run opened at 0.81; every failure opened at 8-17

LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)/results/checkpoints"
LOCAL_LOG_DIR="$(cd "$(dirname "$0")/.." && pwd)/logs"
LOCAL_LOG="${LOCAL_LOG_DIR}/stage2_v5.log"
POLL_INTERVAL=30
TOP_K=2 # keep only top-K checkpoints by val_loss, locally and on the VM

mkdir -p "$LOCAL_DIR" "$LOCAL_LOG_DIR"
echo "[sync] Watching ${REMOTE_HOST}:${REMOTE_CKPT_DIR} (keeping top-${TOP_K} by val_loss)"
echo "[sync] Polling every ${POLL_INTERVAL}s. Ctrl+C to stop."

# Print the top-K survivors and delete the rest from a checkpoint directory (local path, or
# "host:path" for a remote one reachable over the same ssh alias used elsewhere in this repo).
prune_checkpoints() {
    local where="$1" # "local" or "remote"
    local dir="$2"

    python3 - "$where" "$dir" "$TOP_K" "$REMOTE_HOST" "$CKPT_GLOB" <<'EOF'
import glob, os, subprocess, sys, tempfile

where, dir_, top_k, host, pattern = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]


def load_val_loss(load_bytes_fn, files):
    import torch

    losses = []
    for f in files:
        try:
            data = load_bytes_fn(f)
            ckpt = torch.load(data, map_location="cpu", weights_only=False)
            losses.append((ckpt.get("val_loss", float("inf")), f))
        except Exception:
            losses.append((float("inf"), f))
    return losses


if where == "local":
    files = glob.glob(os.path.join(dir_, pattern))
    if len(files) <= top_k:
        sys.exit(0)
    losses = load_val_loss(lambda f: f, files)
    losses.sort(key=lambda x: x[0])
    for loss, f in losses[top_k:]:
        print(f"[sync] local: removing {os.path.basename(f)} (val_loss={loss:.4f})")
        os.remove(f)
    print(
        "[sync] local top-%d: " % top_k
        + ", ".join(f"{os.path.basename(f)} ({l:.4f})" for l, f in losses[:top_k])
    )
else:
    ls = subprocess.run(
        ["ssh", host, f"ls {dir_}/{pattern} 2>/dev/null"], capture_output=True, text=True
    )
    remote_files = [line for line in ls.stdout.splitlines() if line.strip()]
    if len(remote_files) <= top_k:
        sys.exit(0)

    def fetch(f):
        with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
            subprocess.run(["scp", "-q", f"{host}:{f}", tmp.name], check=True)
            return tmp.name

    losses = load_val_loss(fetch, remote_files)
    losses.sort(key=lambda x: x[0])
    to_delete = [f for _, f in losses[top_k:]]
    if to_delete:
        subprocess.run(["ssh", host, "rm -f " + " ".join(to_delete)], check=False)
        for loss, f in losses[top_k:]:
            print(f"[sync] remote: removing {os.path.basename(f)} (val_loss={loss:.4f})")
    print(
        "[sync] remote top-%d: " % top_k
        + ", ".join(f"{os.path.basename(f)} ({l:.4f})" for l, f in losses[:top_k])
    )
EOF
}

print_progress_bar() {
    local epoch="$1" total="$2"
    local width=30
    local filled=$((epoch * width / total))
    local bar
    bar=$(printf "%${filled}s" | tr ' ' '#')
    bar="${bar}$(printf "%$((width - filled))s" | tr ' ' '-')"
    printf "[sync] progress [%s] epoch %d/%d\n" "$bar" "$epoch" "$total"
}

# The t0003 "Dur loss on an early step" stop criterion was retired: it was calibrated against v3's
# high-LR profile (lr/ft_lr 1e-4), and this run uses v4-safe 3e-5, which descends ~3x slower and
# would trip the threshold as a false alarm. The real failure mode turned out to be a hard crash at
# the joint_epoch boundary (see logs/README.md), which the crash check below already covers.

while true; do
    rsync -az "${REMOTE_HOST}:${REMOTE_LOG}" "$LOCAL_LOG" 2>/dev/null

    current_epoch=0
    if [ -f "$LOCAL_LOG" ]; then
        current_epoch=$(grep -oE "Epoch \[[0-9]+/${TOTAL_EPOCHS}\]" "$LOCAL_LOG" | tail -1 | grep -oE "[0-9]+" | head -1)
        current_epoch=${current_epoch:-0}
    fi
    print_progress_bar "$current_epoch" "$TOTAL_EPOCHS"

    rsync -az --include="${CKPT_GLOB}" --exclude="*" \
        "${REMOTE_HOST}:${REMOTE_CKPT_DIR}/" "${LOCAL_DIR}/" 2>/dev/null
    prune_checkpoints "local" "$LOCAL_DIR"
    prune_checkpoints "remote" "$REMOTE_CKPT_DIR"

    # Crashes here are not always tracebacks: train_second.py's own NaN guards used to call
    # sys.exit(1) after a bare print, so the process vanishes with no traceback at all.
    if grep -qE "Traceback \(most recent call last\)|NaN detected" "$LOCAL_LOG" 2>/dev/null; then
        echo "[sync] Remote log shows a crash — check ${LOCAL_LOG}. Stopping sync."
        break
    fi

    # Non-fatal now, but worth surfacing: a skipped step means a batch produced non-finite grads.
    skips=$(grep -c "\[skip\]" "$LOCAL_LOG" 2>/dev/null || echo 0)
    if [ "${skips:-0}" -gt 0 ]; then
        echo "[sync] ${skips} step(s) skipped so far on non-finite grads (training continued)."
    fi

    if [ "$current_epoch" -ge "$TOTAL_EPOCHS" ] 2>/dev/null; then
        echo "[sync] Training complete. Final sync..."
        rsync -az "${REMOTE_HOST}:${REMOTE_LOG}" "$LOCAL_LOG" 2>/dev/null
        rsync -az --include="${CKPT_GLOB}" --exclude="*" \
            "${REMOTE_HOST}:${REMOTE_CKPT_DIR}/" "${LOCAL_DIR}/" 2>/dev/null
        prune_checkpoints "local" "$LOCAL_DIR"
        prune_checkpoints "remote" "$REMOTE_CKPT_DIR"
        echo "[sync] Done. Best checkpoints in: $LOCAL_DIR"
        break
    fi

    sleep "$POLL_INTERVAL"
done
