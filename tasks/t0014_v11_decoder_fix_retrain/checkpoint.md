---
spec_version: "1"
task_id: "t0014_v11_decoder_fix_retrain"
updated_at: "2026-09-16T14:59:30Z"
completed_steps: 2
next_step_number: 3
next_step_id: "init-folders"
---
# Task Objective

Fix the ignore_modules bug that left v10's HiFi-GAN decoder worse than random, retrain on t0012's
1531-clip normalized corpus, and gate completion on an actual audible-speech check.

* * *

## Step History

### Step 1 — create-branch

Branch `task/t0014_v11_decoder_fix_retrain` created. Initial folder structure initialized in
`tasks/t0014_v11_decoder_fix_retrain/`. Step 1 is a mechanical setup step with no research output.

### Step 2 — check-deps

Ran `verify_task_dependencies.py` (via prestep, then again wrapped in `run_with_logs` for the audit
trail) — PASSED with no errors or warnings. All three declared dependencies
(`t0010_stage2_safeguarded_training`, `t0012_v5_corpus_normalize_and_reaudit`,
`t0013_v10_synthesis_quality_forensics`) are `completed`. Wrote
`logs/steps/002_check-deps/deps_report.json`.

* * *

## Cross-Step Decisions

* * *

## Next Step Notes

Step 2 completed successfully — all dependencies satisfied. Proceed to step 3 (`init-folders`) per
step_tracker.json.
