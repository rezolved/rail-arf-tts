---
spec_version: "1"
task_id: "t0016_v3_recipe_recovery"
updated_at: "2026-09-17T13:49:16Z"
completed_steps: 2
next_step_number: 3
next_step_id: "init-folders"
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

* * *

## Cross-Step Decisions

* * *

## Next Step Notes

Both dependencies (t0006, t0009) are confirmed completed — no blockers for this task. Proceed to
step 3, `init-folders`: create the mandatory task folder structure via `init_task_folders`, then
populate the aggregator context cache (`tasks/t0016_v3_recipe_recovery/ctx/`) before committing.
