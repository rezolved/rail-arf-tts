---
spec_version: "1"
task_id: "t0016_v3_recipe_recovery"
updated_at: "2026-09-17T13:52:30Z"
completed_steps: 7
next_step_number: 6
next_step_id: "research-code"
---
# Task Objective

Reconstruct the exact Kokoro v3 Stage 2 recipe (config, data list, patches, epochs, environment)
from VM files, DVC artifacts and checkpoint forensics, without access to the original author.

* * *

## Step History

### Step 1 — create-branch

Branch `task/t0016_v3_recipe_recovery` created. Initial folder structure initialized in
`tasks/t0016_v3_recipe_recovery/`. Step 1 is a mechanical setup step with no research output.

### Step 2 — check-deps

Ran `verify_task_dependencies.py` (via `prestep` and again via `run_with_logs` for the log record):
both declared dependencies, `t0006_kokoro_v5_stage2_subset` and
`t0009_stage2_training_failure_forensics`, have `status: "completed"` in their `task.json`, and the
verificator reported PASSED with 0 errors and 0 warnings. Result recorded in
`logs/steps/002_check-deps/deps_report.json`. No caveats.

### Step 3 — init-folders

Created the mandatory task folder structure via `init_task_folders` (12 directories with `.gitkeep`,
plus `__init__.py` and `code/__init__.py`), recorded in
`logs/steps/003_init-folders/folders_created.txt`. Populated the local, gitignored aggregator
context cache at `tasks/t0016_v3_recipe_recovery/ctx/` (task_types, costs, tasks, metrics,
suggestions) for reuse by downstream step-executors. No caveats.

### Step 4 — research-papers

Skipped, per `step_tracker.json`: this task is checkpoint/VM forensics, not literature-driven — no
papers in the corpus bear on Kokoro/StyleTTS2 recipe recovery from local artifacts.

### Step 5 — research-internet

Skipped, per `step_tracker.json`: `task_description.md`'s evidence sources are entirely internal (VM
home dir, DVC artifacts, prior task checkpoints/code); no external internet research is required or
listed as an evidence source.

### Step 11 — creative-thinking

Skipped, per `step_tracker.json`: `task_description.md` fully specifies the evidence sources,
priority order, and reconstruction methodology; no exploratory alternative-approach analysis is
called for.

### Step 13 — compare-literature

Skipped, per `step_tracker.json`: this task reconstructs an internal training recipe from artifacts;
it does not produce results comparable to published external baselines.

* * *

## Cross-Step Decisions

* * *

## Next Step Notes

Folder structure and aggregator context cache are ready. `tasks/t0016_v3_recipe_recovery/ctx/` holds
cached `task_types.json`, `costs.json`, `tasks.json`, `metrics.json`, and `suggestions.json` — read
these instead of re-running aggregators. Proceed to step 6, `research-code`: review
`t0006_kokoro_v5_stage2_subset`, `t0009_stage2_training_failure_forensics`, `t0015`, and `t0002`
code/results (checkpoint forensics scripts, audio quality gate, packaging recipe) per
`step_tracker.json`'s description, and write `research/research_code.md`.
