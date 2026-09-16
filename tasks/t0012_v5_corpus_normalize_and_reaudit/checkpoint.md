---
spec_version: "1"
task_id: "t0012_v5_corpus_normalize_and_reaudit"
updated_at: "2026-09-16T06:35:35Z"
completed_steps: 2
next_step_number: 3
next_step_id: "init-folders"
---
# Task Objective

Fix the clipping heuristic that over-flags peak-normalized clips, LUFS-normalize the full v5 corpus
to -14 LUFS, and produce a near-full-corpus clean train manifest.

* * *

## Step History

### Step 1 — create-branch

Branch `task/t0012_v5_corpus_normalize_and_reaudit` created. Initial folder structure initialized in
`tasks/t0012_v5_corpus_normalize_and_reaudit/`. Step 1 is a mechanical setup step with no research
output.

### Step 2 — check-deps

Verified dependency `t0011_v5_data_quality_audit` has `status: completed` in its `task.json`, so the
dependency check passed with 0 errors and 0 warnings. Result recorded in
`logs/steps/002_check-deps/deps_report.json`. No caveats for downstream steps.

* * *

## Cross-Step Decisions

* * *

## Next Step Notes

Step 2 completed successfully; the only declared dependency (`t0011_v5_data_quality_audit`) is
satisfied. Proceed to step 3 (`init-folders`) per `step_tracker.json`: create the mandatory task
folder structure via `init_task_folders` and populate the aggregator cache under `ctx/`.
