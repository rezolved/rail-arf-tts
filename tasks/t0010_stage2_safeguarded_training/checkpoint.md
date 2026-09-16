---
spec_version: "1"
task_id: "t0010_stage2_safeguarded_training"
updated_at: "2026-09-16T07:36:00Z"
completed_steps: 14
next_step_number: 15
next_step_id: "reporting"
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

### Step 10 — teardown

LLM-T1-NC80 stopped at 2026-09-16T07:10:15Z after 19.54 h ($272.78). Harness eval re-attempted but
blocked by root disk at 100% full and no working torch env on VM (ncclCommResume error in default
Python, no conda/venv with torch). Azure ML refused stop with full disk — cleared `~/.cache/whisper`
(2.9 GB) to unblock. Key outputs: `results/costs.json`, `results/remote_machines_used.json`,
`machine_log.json` updated, `verify_machines_destroyed` 0 errors.

### Step 12 — results

Wrote `results/results_summary.md` and `results/results_detailed.md` covering all 17 training
epochs. Generated `results/images/speaker_sim_curve.png` placeholder using val_loss as proxy
(speaker_sim eval deferred). `verify_task_metrics.py` and `verify_task_results.py` both passed with
0 errors. REQ-6 and REQ-7 marked Partial (eval framework exists, metrics null).

### Step 13 — compare-literature

Produced `results/compare_literature.md` comparing v10 against the v6c prior run and the t0008
baselines. Key finding: val_loss improved −6% (0.849 → 0.797) and 0 health gate events confirm the
t0009 fixes worked; speaker_sim comparison deferred pending harness eval. Verificator PASSED, 0
errors, 0 warnings.

### Step 14 — suggestions

Generated 3 suggestions in `results/suggestions.json` (IDs S-0010-01 to S-0010-03): (1) run deferred
v10 harness eval locally to obtain null speaker\_sim/TTFB/RTF metrics (high priority); (2) fix
watchdog TERMINATE\_CMD to use Python Azure ML SDK with retry plus disk-fill guard at 90% (high
priority); (3) extend v10 training to epoch 20 using persistent share for checkpoints, contingent on
speaker\_sim still below 0.85 (medium priority). Verificator PASSED, 0 errors.

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

Step 14 (suggestions) completed. Step 15 is `reporting`. The reporting agent should run all
verificators (`verify_task_file`, `verify_task_dependencies`, `verify_suggestions`,
`verify_task_metrics`, `verify_task_results`, `verify_task_folder`, `verify_logs`,
`verify_model_asset`, `verify_machines_destroyed`), capture session transcripts, set `task.json`
`status` to "completed" with `end_time`, then commit and run poststep. The model asset
`kokoro-v10-best` at `assets/model/kokoro-v10-best/` is DVC-tracked and was verified clean in step
9\. The three suggestions in `results/suggestions.json` (S-0010-01 to S-0010-03) are ready.
