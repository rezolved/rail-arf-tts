---
spec_version: "1"
task_id: "t0018_zero_shot_cloning_calibration"
updated_at: "2026-09-17T13:50:41Z"
completed_steps: 3
next_step_number: 4
next_step_id: "research-papers"
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

### Step 3 — init-folders

Ran `init_task_folders` to create the mandatory task folder structure (`plan/`, `research/`,
`results/`, `results/images/`, `corrections/`, `intervention/`, `code/`, `logs/commands/`,
`logs/searches/`, `logs/sessions/`, `logs/steps/`, `assets/answer/`) with `.gitkeep` files; recorded
in `logs/steps/003_init-folders/folders_created.txt`. Populated the local aggregator cache at
`tasks/t0018_zero_shot_cloning_calibration/ctx/` (task_types, costs, tasks, metrics, suggestions) —
gitignored and not committed.

* * *

## Cross-Step Decisions

* * *

## Next Step Notes

Task folder structure and aggregator cache are in place. Proceed to step 4 (`research-papers`):
review any corpus papers on F5-TTS, CosyVoice 2, Chatterbox, and GE2E speaker-similarity evaluation
methodology, per `step_tracker.json`. The `ctx/` cache in the task folder holds pre-fetched
aggregator output (task types, costs, tasks, metrics, suggestions) for reuse instead of re-running
aggregators.
