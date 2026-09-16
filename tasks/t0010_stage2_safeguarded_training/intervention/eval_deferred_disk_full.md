---
spec_version: "1"
task_id: "t0010_stage2_safeguarded_training"
intervention_type: "eval_deferred"
created_at: "2026-09-16T06:30:00Z"
---
# Intervention: Harness Eval Deferred — Ephemeral Disk Full

## Problem

LLM-T1-NC80 ephemeral disk (`/dev/root`) is 100% full (119 GB used / 119 GB total). The
`tts_eval_harness` requires writing synthesized audio clips to the local filesystem during
evaluation. With zero free space, synthesis will fail immediately.

## Evidence

```
Filesystem      Size  Used Avail Use% Mounted on
/dev/root       119G  119G     0 100% /
```

Observed at: 2026-09-16T06:27:00Z

## Impact

- `eval_all_checkpoints.py` cannot run on VM
- `speaker_sim`, `ttfb_ms`, `rtf` metrics are null for all epochs
- REQ-6 and REQ-7 are partially satisfied (framework exists, execution deferred)

## Checkpoints Secured

Both checkpoints are downloaded locally and DVC-tracked:

- `epoch_2nd_00016.pth` (primary, epoch 17, val_loss 0.853) — persistent share + DVC
- `epoch_2nd_00014.pth` (backup, epoch 15, val_loss 0.818) — persistent share + DVC

## Resolution Path

To run eval manually after freeing VM disk:

```bash
# Free disk: remove training data and optimizer states no longer needed
ssh LLM-T1-NC80
du -sh ~/kokoro-finetune/Data/ ~/kokoro-finetune/logs/ /mnt/tmp/ 2>/dev/null
# Remove cached audio, optimizer, or ephemeral training intermediates

# Then run:
cd ~/kokoro-finetune
python tasks/t0010_stage2_safeguarded_training/code/eval_all_checkpoints.py \
  --checkpoint-dir /mnt/cache/persist/t0010/checkpoints \
  --output-dir /mnt/cache/persist/t0010/eval_results \
  --config /mnt/cache/persist/t0010/logs/config_david_v10.yml \
  --voicepack data/david_v3_best_voicepack.pt \
  --centroid data/reference_centroid.npy \
  --resemblyzer-venv ~/resemblyzer-venv \
  --repo-root /path/to/rail-arf-tts
```

Alternatively run eval locally once the local Kokoro environment is set up.

## Teardown Attempt (2026-09-16)

Eval was re-attempted during teardown after context indicated 24GB freed on `/mnt` (ephemeral disk).
Actual VM state at teardown time:

- Root disk `/dev/root` still at 100% full (119G used / 119G total)
- Python torch import failed with `ncclCommResume` undefined symbol in default env
- No working Python environment with torch was found on root or `/mnt`
- Resemblyzer venv at `~/resemblyzer-venv` was not present (only in miniconda3/envs/stt which lacks
  torch)

Cleared `~/.cache/whisper` (2.9 GB) to allow Azure ML stop (API blocked stop when disk 100% full).
VM stopped at 2026-09-16T07:10:15Z. Eval metrics remain null.

## Status

harness eval not completed — root disk full, no torch environment available; VM stopped
2026-09-16T07:10:15Z
