---
spec_version: "1"
task_id: "t0021_zero_shot_latency_reduction"
updated_at: "2026-09-18T10:20:00Z"
completed_steps: 3
next_step_number: 4
next_step_id: "research-papers"
---
# Task Objective

Profile where CosyVoice2 and Chatterbox spend their 1.3-2.9 s TTFB and test streaming, chunking,
vLLM/TensorRT backends and precision to find the best reachable TTFB at unchanged speaker_sim.

* * *

## Step History

### Step 1 — create-branch

Branch `task/t0021_zero_shot_latency_reduction` created. Initial folder structure initialized in
`tasks/t0021_zero_shot_latency_reduction/`. Step 1 is a mechanical setup step with no research
output.

### Step 2 — check-deps

`verify_task_dependencies.py` passed with no errors or warnings. The sole dependency,
`t0018_zero_shot_cloning_calibration`, is `completed` and provides the F5-TTS/CosyVoice2/Chatterbox
benchmark, adapters, reference clips, and prompt sets this task will reuse. Result recorded in
`logs/steps/002_check-deps/deps_report.json`.

### Step 3 — init-folders

Ran `init_task_folders` to create the mandatory task folder structure (`plan/`, `research/`,
`results/`, `results/images/`, `corrections/`, `intervention/`, `code/`,
`logs/{commands,searches,sessions,steps}/`, `assets/answer/`), recording
`logs/steps/003_init-folders/folders_created.txt`. Populated the local `ctx/` aggregator cache
(`task_types.json`, `costs.json`, `tasks.json`, `metrics.json`, `suggestions.json`) for downstream
subagents to reuse instead of re-running aggregators; `ctx/` is gitignored and not committed.

* * *

## Cross-Step Decisions

* * *

## Next Step Notes

Step 3 created the folder skeleton and seeded `tasks/t0021_zero_shot_latency_reduction/ctx/` with
`task_types.json`, `costs.json`, `tasks.json`, `metrics.json`, and `suggestions.json` — read these
instead of re-running aggregators. Proceed to step 4 (`research-papers`): review corpus papers on
TTS latency optimization (streaming decoding, TensorRT/vLLM inference acceleration, flow-matching
vocoders) relevant to CosyVoice2 and Chatterbox, per the step description in `step_tracker.json`.
