---
spec_version: "1"
task_id: "t0008_tts_eval_harness_baselines"
updated_at: "2026-09-14T15:31:00Z"
completed_steps: 2
next_step_number: 3
next_step_id: "init-folders"
---
# Task Objective

Build a reusable speaker_sim/TTFB/RTF harness and score ElevenLabs David, base Kokoro, v3 and the
best t0005/t0006 checkpoints.

* * *

## Step History

### Step 1 — create-branch

Branch `task/t0008_tts_eval_harness_baselines` created. Initial folder structure initialized in
`tasks/t0008_tts_eval_harness_baselines/`. Step 1 is a mechanical setup step with no research
output.

### Step 2 — check-deps

Dependency `t0006_kokoro_v5_stage2_subset` verified as completed. Output written to
`logs/steps/002_check-deps/deps_report.json`.

* * *

## Cross-Step Decisions

* * *

## Next Step Notes

Step 2 check-deps passed: `t0006_kokoro_v5_stage2_subset` is completed and its checkpoints and
reference bundles are available. Proceed to step 3 init-folders per step_tracker.json. The 15-step
plan continues with research-papers, research-internet, research-code, planning, setup-machines,
implementation, teardown, results, suggestions, and reporting; creative-thinking (step 11) and
compare-literature (step 13) are pre-skipped.
