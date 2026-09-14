---
spec_version: "1"
task_id: "t0009_stage2_training_failure_forensics"
date_completed: "2026-09-14"
---
## Summary

This forensic task audited 13 Kokoro StyleTTS2 Stage 2 training launches across tasks t0001, t0004,
t0005, and t0006 (~\$440 GPU spend, 19+ launches) to identify why Stage 2 consistently diverges. The
audit produced a ranked root-cause answer, a 6-module safeguard library
(`t0009_training_safeguards`), and seven result tables. The only successful run (v6c,
`val_loss=0.849`) differed from failed runs on two variables: a correctly-keyed Stage 1 checkpoint
(`multispeaker=true`) and `joint_epoch=6` instead of 3.

## Metrics

- **Runs audited**: 13 training launches across 4 tasks
- **Logs surviving**: 1 of 13 (7.7%) — only `t0006_run03_v6c` committed its log
- **Root causes identified**: 2 primary (DP checkpoint mismatch; `joint_epoch` too early) + 1
  supporting (train_LM=true causes LM Loss explosion; patch 5 masks it)
- **Patches classified**: 7 total — 2 fix-cause (patches 5, 7), 4 hide-symptom (patches 2, 3, 4, and
  partial 1), 1 neutral (patch 6)
- **Val loss — only successful run (v6c)**: 0.849 at epoch 6 (pre-divergence, pre-GAN activation)
- **Val loss — v3 reference run**: 0.506 (different loss configuration; not directly comparable)
- **Data audit**: 1,557 train clips, 96 val clips, 0 train/val overlap; v5 val == val_96 (confirmed)
- **Safeguard library**: 4 core modules + 2 test scripts; health-gate thresholds calibrated on v6c
  log

## Verification

- `verify_task_metrics.py`: PASS — `metrics.json` is `{}` (no registered project metrics apply to
  forensic analysis)
- `verify_task_results.py`: PASS — all required files present and non-empty
- Answer asset `t0009-stage2-forensics-answer`: present at
  `tasks/t0009_stage2_training_failure_forensics/assets/answer/t0009-stage2-forensics-answer/`
- Library asset `t0009_training_safeguards`: present at
  `tasks/t0009_stage2_training_failure_forensics/assets/library/t0009_training_safeguards/`
- Offline replay test (`test_replay.py`): all gates fire on failure traces; pass on v6c log
