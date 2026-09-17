---
spec_version: "1"
task_id: "t0018_zero_shot_cloning_calibration"
updated_at: "2026-09-17T13:49:21Z"
completed_steps: 2
next_step_number: 3
next_step_id: "init-folders"
---
# Task Objective

Benchmark three open zero-shot voice-cloning TTS models (F5-TTS, CosyVoice 2, Chatterbox) on David
reference audio with the t0008 harness to calibrate the reachable speaker_sim/TTFB envelope.

* * *

## Step History

### Step 1 — create-branch

Branch `task/t0018_zero_shot_cloning_calibration` created. Initial folder structure initialized in
`tasks/t0018_zero_shot_cloning_calibration/`. Step 1 is a mechanical setup step with no research
output.

### Step 2 — check-deps

Ran `verify_task_dependencies.py` for `t0018_zero_shot_cloning_calibration`: the single declared
dependency, `t0008_tts_eval_harness_baselines`, has `status: "completed"` in its `task.json`, so the
check passed with 0 errors and 0 warnings. Result written to
`logs/steps/002_check-deps/deps_report.json`.

* * *

## Cross-Step Decisions

* * *

## Next Step Notes

Step 2 confirmed the `t0008_tts_eval_harness_baselines` dependency is satisfied (harness, prompt
set, and val_96 split outputs from t0008 are available for reuse). Proceed to step 3
(`init-folders`) per `step_tracker.json`: create the mandatory task folder structure and populate
the aggregator cache.
