---
spec_version: "1"
task_id: "t0010_stage2_safeguarded_training"
updated_at: "2026-09-15T15:54:43Z"
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

LLM-T1-NC80 acquired (2×H100 NVL, CUDA 12.2). Idle watchdog deployed (60-min idle threshold). Root
disk 100% full — `~/kokoro-finetune` symlinked to ephemeral `/mnt/tmp/kikiri-tts/StyleTTS2`. All
training data staged: 1557 train wavs, 96 val wavs, `first_stage_v3.pth` (1.7 GB),
`config_david_v10.yml` (joint_epoch=8, epochs=20), safeguard modules. `az ml compute show` API timed
out; acquire done manually via ARM REST. **Caution**: ephemeral disk lost on VM stop.

### Step 9 — implementation (paused_waiting)

Training launched on LLM-T1-NC80 as PID 82356 in tmux session `train_v10`. At pause time: epoch
5/20, val_loss 2.066 (improving from 2.097 baseline). Step paused with watchdog-protected VM and
liveness probe `ssh LLM-T1-NC80 pgrep -f train_second_v10`. Resume after 2026-09-15T18:50:00Z. On
resume: run `resume_check`, download `logs/v10/`, run batch eval and aggregation, package model
asset.

* * *

## Cross-Step Decisions

* **`CheckpointManager` bug fixed**: line 403 of `train_second_v10.py` now has
  `CheckpointManager(log_dir=log_dir, run_id=_run_id, joint_epoch=joint_epoch)`.
* **Training script is copy-not-import**: `train_second_safeguarded.py` copied to
  `code/train_second_v10.py` with fix.
* **Safeguard library components are import-not-copy**: `StepLogger`, `CheckpointManager`,
  `HealthGate`, `capture_run_config` are all registered under `t0009_training_safeguards` — import
  directly.
* **Training PID 82356** in tmux `train_v10` on LLM-T1-NC80; val_loss improving (2.066 at epoch 5).
* **Paused sentinel**: `~/kokoro-finetune/logs/v10/checkpoint_manifest.json` (written when training
  completes).
* **Resemblyzer venv**: must be built on VM before batch eval
  (`python -m venv ~/resemblyzer-venv && pip install resemblyzer webrtcvad`).

* * *

## Next Step Notes

Step 9 (implementation) is `paused_waiting`. Training is running on LLM-T1-NC80 (PID 82356, tmux
session `train_v10`, epoch 5/20, val_loss 2.066). The resume agent must:

1. Run `uv run python -m arf.scripts.utils.resume_check t0010_stage2_safeguarded_training 9` and act
   on the result: `sentinel present` → proceed, `job_alive` → re-pause, `job_dead` → write
   intervention file and fail.

2. On sentinel present:
   `rsync -av LLM-T1-NC80:~/kokoro-finetune/logs/v10/ tasks/t0010_stage2_safeguarded_training/data/run_v10/`
   to download checkpoints + JSONL.

3. Build resemblyzer venv on VM:
   `python -m venv ~/resemblyzer-venv && pip install resemblyzer webrtcvad`. Then run
   `eval_all_checkpoints.py` on VM.

4. Download `data/run_v10/eval_results/` locally. Run `aggregate_results.py` locally.

5. Package model asset: `assets/model/kokoro-v10-best/` with DVC-tracked best checkpoint `.pth`.

6. Run `verify_model_asset --task-id t0010_stage2_safeguarded_training` — fix all errors.

**Critical**: ephemeral disk — download ALL outputs before teardown. `~/kokoro-finetune` is lost on
VM stop.
