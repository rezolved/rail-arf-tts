---
spec_version: "1"
task_id: "t0009_stage2_training_failure_forensics"
updated_at: "2026-09-14T15:50:00Z"
completed_steps: 9
next_step_number: 7
next_step_id: "planning"
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

* * *

## Cross-Step Decisions

* All reusable code must be copied into `tasks/t0009_stage2_training_failure_forensics/code/`; no
  libraries exist to import.
* v3's `train_second_patch.diff` is the forensic baseline — use it as the starting point for the
  pipeline audit in planning.

* * *

## Next Step Notes

Step 6 (`research-code`) completed. The primary forensic inputs are documented in
`research/research_code.md` and compressed in `research/research_summary.md`. Proceed to step 7
(`planning`): load `research/research_summary.md` and `research/research_code.md` to design the full
forensic approach. Key decisions for planning: (1) which logs to parse and in what order, (2) health
gate thresholds derived from the log data, (3) safeguard library structure (JSONL logger,
health-gate monitor, per-epoch checkpoint retention with SHA-256), (4) offline replay test scope.
Budget is local/CPU only except for optional VM access to retrieve surviving logs (~30 min, ~$7).
