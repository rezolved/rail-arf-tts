---
spec_version: "1"
task_id: "t0015_v11_duration_blowup_forensics"
updated_at: "2026-09-17T07:56:30Z"
completed_steps: 8
next_step_number: 6
next_step_id: "research-code"
---
# Task Objective

Root-cause why kokoro-v11-best's synthesis runs 15-20x too long and sounds like a droning babble to
a human, despite passing the clipping/flatness noise gate, then close that gate's blind spot.

* * *

## Step History

### Step 1 — create-branch

Branch `task/t0015_v11_duration_blowup_forensics` created. Initial folder structure initialized in
`tasks/t0015_v11_duration_blowup_forensics/`. Step 1 is a mechanical setup step with no research
output.

### Step 2 — check-deps

Ran `verify_task_dependencies.py` (via prestep and again via `run_with_logs`) against both
dependencies: `t0013_v10_synthesis_quality_forensics` and `t0014_v11_decoder_fix_retrain`. Both have
`status: "completed"` in their `task.json`, so the check passed with 0 errors and 0 warnings. Result
recorded in `logs/steps/002_check-deps/deps_report.json`.

### Step 3 — init-folders

Ran `init_task_folders` to create the mandatory folder structure (`plan/`, `research/`, `results/`,
`results/images/`, `corrections/`, `intervention/`, `code/`,
`logs/commands|searches|sessions|steps/`, `assets/`), each with `.gitkeep`; `expected_assets` is
`{}` so no asset-type subdirectory was added. Populated the aggregator cache at
`tasks/t0015_v11_duration_blowup_forensics/ctx/` (task_types.json, costs.json, tasks.json,
metrics.json, suggestions.json) for downstream subagents to reuse; `ctx/` is gitignored and not
committed.

### Step 4 — research-papers

Skipped: this is an empirical debugging/forensics task about this project's own checkpoints and
code, and no published-paper evidence bears on the specific duration-blowup root cause. StyleTTS2
background is already present in the corpus from prior dependency tasks.

### Step 5 — research-internet

Skipped: `task_description.md` already specifies the exact reproduction recipe, the parameters to
sweep, and the ASR library (faster-whisper) already vendored in t0008's environment, so no new
external research is needed.

### Step 8 — setup-machines

Skipped: `task_description.md` states no GPU is required for the diagnostic, localization, and
inference-parameter-sweep work; GPU-based predictor retraining is explicitly scoped out as a
follow-up task.

### Step 10 — teardown

Skipped: no remote machine was provisioned (`setup-machines` was skipped), so there is nothing to
tear down.

### Step 13 — compare-literature

Skipped: this task produces internal forensic evidence about this project's own checkpoint and code,
not quantitative results comparable to a published baseline.

* * *

## Cross-Step Decisions

* * *

## Next Step Notes

Step 3 (`init-folders`) completed: task folder structure and aggregator cache are ready. Steps 4-5
(research-papers, research-internet) and step 8 (setup-machines) plus step 10 (teardown) and step 13
(compare-literature) are already marked skipped in `step_tracker.json`. Proceed to step 6
(`research-code`) — review t0013's `infer_styletts2.py` and `audio_quality_check.py` and t0014's
training config/logs (`ignore_modules`, Stage 2 epoch logs) to understand the reusable inference
recipe and whether `predictor`/`predictor_encoder` received gradient updates during t0014's
finetune.
