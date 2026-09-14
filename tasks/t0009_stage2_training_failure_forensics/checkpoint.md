---
spec_version: "1"
task_id: "t0009_stage2_training_failure_forensics"
updated_at: "2026-09-14T15:49:30Z"
completed_steps: 10
next_step_number: 8
next_step_id: "implementation"
---
# Task Objective

Audit every Kokoro training run's logs, data, pipeline and checkpoints to explain why Stage 2
breaks, and ship checkpoint and log safeguards.

* * *

## Step History

### Step 1 — create-branch

Branch `task/t0009_stage2_training_failure_forensics` created. Initial folder structure initialized
in `tasks/t0009_stage2_training_failure_forensics/`. Step 1 is a mechanical setup step with no
research output.

### Step 2 — check-deps

Both dependencies verified as completed: `t0005_kokoro_v5_stage2_train` and
`t0006_kokoro_v5_stage2_subset`. Verification passed with zero errors and zero warnings. Output:
`logs/steps/002_check-deps/deps_report.json`.

### Step 4 — research-papers

Skipped: no papers in the corpus are directly relevant to forensic analysis of local Kokoro training
logs.

### Step 5 — research-internet

Skipped: all inputs (logs, checkpoints, code) are local; no external data or new documentation
needed.

### Step 10 — setup-machines

Skipped: analysis runs locally on stored logs and checkpoints; no GPU required.

### Step 11 — teardown

Skipped: no remote machines provisioned.

### Step 3 — init-folders

Mandatory task folder structure created by `init_task_folders`; 13 directories with `.gitkeep` files
plus `__init__.py` files for the code package. Aggregator cache written to
`tasks/t0009_stage2_training_failure_forensics/ctx/` (gitignored).

### Step 13 — compare-literature

Skipped: this is a forensic audit task; results are not comparable to published quantitative
baselines.

### Step 6 — research-code

Reviewed 7 completed tasks; cited 6. No registered libraries found. Key output:
`tasks/t0009_stage2_training_failure_forensics/research/research_code.md` documenting the 7-patch
stack in `train_second_patched.py`, the checkpoint loading silent-failure bug in t0005/t0006, the
top-2 val_loss pruning risk, and all config-code inconsistencies. Research summary also produced at
`tasks/t0009_stage2_training_failure_forensics/research/research_summary.md`.

### Step 7 — planning

Full forensic plan written at `tasks/t0009_stage2_training_failure_forensics/plan/plan.md` — 15
steps across 8 milestones covering all 8 REQ items. `verify_plan` passed with 0 errors. Budget
confirmed at ≤ \$25 (local/CPU + ≤ 30 min VM for log retrieval). The safeguard library base is
`train_second_patched.py` from t0005; health gate thresholds derived from research (Dur Loss step-1
< 2.0, `acoustic_norm` < 20, val spike ≤ 0.05, consecutive skips ≤ 50).

* * *

## Cross-Step Decisions

* All reusable code must be copied into `tasks/t0009_stage2_training_failure_forensics/code/`; no
  libraries exist to import.
* v3's `train_second_patch.diff` is the forensic baseline — use it as the starting point for the
  pipeline audit in planning.

* * *

## Next Step Notes

Step 7 (`planning`) completed. The implementation plan is at `plan/plan.md`. Proceed to step 8
(`implementation`): read `plan/plan.md` and execute all 15 steps across the 8 milestones in order.
Start with Milestone 1 (log collection from git — no VM needed first). Milestone 2 requires VM SSH
only if surviving logs are found. The safeguard library (Milestone 7) should be implemented after
log parsing (Milestones 2-3) so that health gate thresholds can be cross-checked against actual log
data. Key files to produce: `code/parse_logs.py`, `code/jsonl_logger.py`, `code/health_gates.py`,
`code/checkpoint_manager.py`, `code/test_replay.py`, and the answer asset.
