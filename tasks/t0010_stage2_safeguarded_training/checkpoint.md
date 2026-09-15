---
spec_version: "1"
task_id: "t0010_stage2_safeguarded_training"
updated_at: "2026-09-15T11:10:00Z"
completed_steps: 7
next_step_number: 7
next_step_id: "planning"
---
# Task Objective

First controlled Stage 2 run implementing the two t0009 root-cause fixes: DP-aware checkpoint loader
and joint_epoch=8. Uses the t0009 safeguard library (JSONL logger, health gates, per-epoch
checkpoints). Evaluates each epoch checkpoint with the t0008 harness. One variable changed from v6c.

* * *

## Step History

### Step 1 — create-branch

Branch `task/t0010_stage2_safeguarded_training` created. Worktree initialized at
`/home/azureuser/rail-metarepo/real-repos/rail-arf-tts-worktrees/t0010_stage2_safeguarded_training`.
Step 1 is a mechanical setup step with no research output.

### Step 2 — check-deps

Both dependencies verified as completed: `t0008_tts_eval_harness_baselines` and
`t0009_stage2_training_failure_forensics`. Output: `logs/steps/002_check-deps/deps_report.json`.

### Step 4 — research-papers

Skipped: no new literature needed; approach is fully specified from t0009 forensics findings.

### Step 5 — research-internet

Skipped: all required knowledge comes from prior tasks t0008 and t0009; no external research needed.

### Step 3 — init-folders

Created all mandatory task folder subdirectories and placed `.gitkeep` files; output in
`logs/steps/003_init-folders/folders_created.txt`. Aggregator cache populated under
`tasks/t0010_stage2_safeguarded_training/ctx/` (gitignored, not committed).

### Step 11 — creative-thinking

Skipped: single-variable controlled experiment; approach is fully defined by t0009 root-cause
findings.

### Step 6 — research-code

Reviewed t0009 safeguard library (`train_second_safeguarded.py` 1027 lines) and t0008
`tts_eval_harness` library in full. Key output: `research/research_code.md` (verificator PASSED, 0
errors) and `research/research_summary.md` (126 lines). Critical bug found: `CheckpointManager` at
line 403 of training script is missing required `joint_epoch` argument — must fix when copying
script into task `code/`.

* * *

## Cross-Step Decisions

* **`CheckpointManager` bug**: line 403 of `train_second_safeguarded.py` omits required
  `joint_epoch` — implementation must apply one-line fix:
  `CheckpointManager(log_dir=log_dir, run_id=_run_id, joint_epoch=joint_epoch)`.
* **Training script is copy-not-import**: `train_second_safeguarded.py` is not a registered library;
  must be copied into `tasks/t0010_stage2_safeguarded_training/code/`.
* **Safeguard library components are import-not-copy**: `StepLogger`, `CheckpointManager`,
  `HealthGate`, `capture_run_config` are all registered under `t0009_training_safeguards` — import
  directly.

* * *

## Next Step Notes

Step 7 (planning) is next. The planning agent should read `research/research_summary.md` for a
compact overview. Key inputs: (1) base config at
`tasks/t0009_stage2_training_failure_forensics/data/configs/t0006_run03_v6c.yml` — change only
`joint_epoch: 6 → 8` and `epochs_2nd: 10 → 20`; (2) training script to copy is
`tasks/t0009_stage2_training_failure_forensics/code/train_second_safeguarded.py` with one-line
CheckpointManager fix; (3) evaluation requires batch approach post-training: `extract_decoder` →
`run_eval.py` → `score_speaker_sim.py` per epoch checkpoint; (4) expected compute ~$35 for 20 epochs
× 6 min + eval on LLM-T1-NC80.
