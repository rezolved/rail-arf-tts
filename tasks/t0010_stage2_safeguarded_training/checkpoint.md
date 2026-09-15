---
spec_version: "1"
task_id: "t0010_stage2_safeguarded_training"
updated_at: "2026-09-15T12:15:00Z"
completed_steps: 8
next_step_number: 9
next_step_id: "implementation"
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

### Step 8 — setup-machines

LLM-T1-NC80 acquired (2×H100 NVL, CUDA 12.2, driver 535.161.08). Idle watchdog deployed (PID
confirmed, 60-min idle threshold). Full environment rebuilt from scratch — ephemeral disk was wiped
after prior VM stop and root disk is 100% full; `~/kokoro-finetune` symlinked to
`/mnt/tmp/kikiri-tts/StyleTTS2`.

Key outputs on VM:

* `~/kokoro-finetune/first_stage_v3.pth` (1.7 GB Stage 1 checkpoint)
* `~/kokoro-finetune/data/v4/train/wavs/` — 1557 training wav files
* `~/kokoro-finetune/data/v4/val/wavs/` — 96 val wav files
* `~/kokoro-finetune/data/data_list_v5_train_250.txt` and `data/val_list.txt`
* `~/kokoro-finetune/configs/config_david_v10.yml` (joint_epoch=8, epochs=20)
* `~/kokoro-finetune/train_second_safeguarded.py` + 4 support modules from t0009

Machine log: `logs/steps/008_setup-machines/machine_log.json`. Notable issue: `az ml compute show`
API consistently exceeds the 60s hardcoded timeout; acquire was performed manually via ARM REST +
SSH preflight + on-VM lock placement.

**Caution**: `~/kokoro-finetune` is on ephemeral disk (`/mnt/tmp/`). All training outputs
(checkpoints, JSONL log) must be downloaded to local worktree before VM teardown.

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

Step 9 (implementation) is next. The implementation agent must:

1. Copy `tasks/t0009_stage2_training_failure_forensics/code/train_second_safeguarded.py` to
   `tasks/t0010_stage2_safeguarded_training/code/train_second_v10.py` and apply the one-line
   `CheckpointManager` fix at line 403 (add `joint_epoch=joint_epoch`). Run ruff + mypy; confirm 0
   errors.

2. Write `code/eval_all_checkpoints.py` and `code/aggregate_results.py` (plan Steps 2–3).

3. SSH to LLM-T1-NC80 and launch training from `~/kokoro-finetune/`:
   `python train_second_v10.py -p configs/config_david_v10.yml --run-id v10` (wrapped in
   `run_with_logs.py`). Monitor the first 50 lines for the parameter-count assertion.

4. After training (≤20 epochs or health-gate fire): download `logs/v10/` to
   `tasks/t0010_stage2_safeguarded_training/data/run_v10/`.

5. Run batch checkpoint evaluation and results aggregation per plan Steps 8–10.

**Critical VM caution**: `~/kokoro-finetune` is on ephemeral disk — download ALL outputs before
teardown. The watchdog will shut down the VM after 60 min idle; ensure training completes or the VM
is kept busy.

**VM state at step 8 completion**: LLM-T1-NC80 is RUNNING with watchdog active. All data is staged.
No lock file currently held (manual acquire; implementation step must place a fresh lock or use
`azure_ml_vm run` to hold it).
