---
spec_version: "1"
task_id: "t0010_stage2_safeguarded_training"
updated_at: "2026-09-16T07:15:00Z"
completed_steps: 9
next_step_number: 10
next_step_id: "teardown"
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

### Step 9 — implementation

Training completed 17 epochs on LLM-T1-NC80 with zero health gate events (val_loss 0.797 at epoch
16). Harness eval deferred due to VM ephemeral disk at 100% — documented in
`intervention/eval_deferred_disk_full.md`. Key outputs: `data/run_v10/metrics.jsonl`,
`data/run_v10/checkpoint_manifest.json`, `results/metrics.json` (18 variants, null eval metrics),
`results/images/loss_timeline.png`, and model asset `kokoro-v10-best` (5-module extraction, 317 MB,
DVC-tracked). Model verificator: 0 errors.

* * *

## Cross-Step Decisions

* **`CheckpointManager` bug fixed**: line 403 of `train_second_v10.py` now has
  `CheckpointManager(log_dir=log_dir, run_id=_run_id, joint_epoch=joint_epoch)`.
* **Training script is copy-not-import**: `train_second_safeguarded.py` copied to
  `code/train_second_v10.py` with fix.
* **Safeguard library components are import-not-copy**: `StepLogger`, `CheckpointManager`,
  `HealthGate`, `capture_run_config` are all registered under `t0009_training_safeguards` — import
  directly.
* **Harness eval deferred**: VM ephemeral disk at 100% after epoch 17. `speaker_sim`, `ttfb_ms`,
  `rtf` are null. Resolution path in `intervention/eval_deferred_disk_full.md`.
* **Primary checkpoint**: `epoch_2nd_00016.pth` (epoch 17, val_loss 0.853) per user instruction.
  Backup: `epoch_2nd_00014.pth` (epoch 15, val_loss 0.818). Both DVC-tracked.
* **Model asset**: `kokoro-v10-best` at `assets/model/kokoro-v10-best/`, verificator 0 errors.
* **aggregate_results.py bug fix**: two fixes applied — skip null-epoch sentinel in JSONL; fall back
  `dur_loss_step1` → `dur_loss` field name.

* * *

## Next Step Notes

Step 9 (implementation) completed. Step 10 is `teardown`. The teardown agent must:

1. Stop LLM-T1-NC80 via
   `az ml compute stop --name LLM-T1-NC80 --workspace-name brainpowa-northeurope --resource-group rezolve-AI`.
2. Verify VM is stopped (status `Stopped` or `Deallocated`).
3. Write `results/remote_machines_used.json` (LLM-T1-NC80 H100, training duration).
4. Write `results/costs.json` with estimated GPU cost (~$43 for ~2.5h at $17.50/hr).
5. Update `machine_log.json` `destroyed_at` timestamp.

**Harness eval still deferred**: before or after teardown, a follow-up task should free VM disk and
run `eval_all_checkpoints.py`. See `intervention/eval_deferred_disk_full.md`.
