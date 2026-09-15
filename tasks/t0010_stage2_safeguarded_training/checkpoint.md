---
spec_version: "1"
task_id: "t0010_stage2_safeguarded_training"
updated_at: "2026-09-15T11:05:00Z"
completed_steps: 8
next_step_number: 8
next_step_id: "setup-machines"
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

### Step 7 — planning

Produced `plan/plan.md` (spec_version "2", 12 REQ items, 12 steps across 5 milestones). Key
decisions: copy `train_second_safeguarded.py` → `code/train_second_v10.py` with one-line
CheckpointManager fix (REQ-3); batch evaluation after training completes; explicit variant format
for `results/metrics.json` (one variant per epoch + `best`); rejection criteria pre-registered.
Verificator PASSED with 0 errors.

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

Step 8 (setup-machines) is next. The setup-machines agent should start LLM-T1-NC80 via the
`setup-remote-machine` skill, verify 2 × H100 NVL GPUs via `nvidia-smi`, confirm
`~/kokoro-finetune/` repo is present, and deploy the idle watchdog
(`arf/scripts/utils/idle_watchdog.sh` with 60-min idle threshold and `TERMINATE_CMD` pointing to
Azure ML compute stop). Key plan references: `plan/plan.md` Steps 4–6 (Milestone 2: VM Setup). The
full config for `config_david_v10.yml` is specified in Step 6 of the plan. Expected cost: ~$35–42
total for training + eval on H100. Budget remaining: $5,000 project budget, ~$45 task cap.
