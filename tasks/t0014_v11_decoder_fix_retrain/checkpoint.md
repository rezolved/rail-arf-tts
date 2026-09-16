---
spec_version: "1"
task_id: "t0014_v11_decoder_fix_retrain"
updated_at: "2026-09-16T14:58:16Z"
completed_steps: 1
next_step_number: 2
next_step_id: "check-deps"
---
# Task Objective

Fix the ignore_modules bug that left v10's HiFi-GAN decoder worse than random, retrain on t0012's
1531-clip normalized corpus, and gate completion on an actual audible-speech check.

* * *

## Step History

### Step 1 — create-branch

Branch `task/t0014_v11_decoder_fix_retrain` created. Initial folder structure initialized in
`tasks/t0014_v11_decoder_fix_retrain/`. Step 1 is a mechanical setup step with no research output.

* * *

## Cross-Step Decisions

* * *

## Next Step Notes

Step 1 completed successfully. The task branch and folder are ready. Proceed to step 2 per
step_tracker.json.
