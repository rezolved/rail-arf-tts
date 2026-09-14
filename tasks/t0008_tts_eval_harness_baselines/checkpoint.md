---
spec_version: "1"
task_id: "t0008_tts_eval_harness_baselines"
updated_at: "2026-09-14T15:32:30Z"
completed_steps: 5
next_step_number: 4
next_step_id: "research-papers"
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

### Step 11 — creative-thinking

Skipped — structured evaluation against fixed success criteria; no open-ended analysis needed at
this stage.

### Step 13 — compare-literature

Skipped — no published speaker_sim/TTFB baselines for Kokoro-82M David voice fine-tuning exist in
the corpus to compare against.

### Step 3 — init-folders

Created all mandatory task subdirectories including `assets/library` for the expected library asset;
aggregator cache populated in `tasks/t0008_tts_eval_harness_baselines/ctx/` (5 files: task_types,
costs, tasks, metrics, suggestions). Key output: `logs/steps/003_init-folders/folders_created.txt`.

* * *

## Cross-Step Decisions

* * *

## Next Step Notes

Step 3 init-folders complete: all 12 directories created with `.gitkeep` files, `__init__.py` and
`code/__init__.py` written, and aggregator cache ready in `ctx/`. Proceed to step 4 research-papers
per step_tracker.json. The research-papers step should focus on GE2E speaker embeddings,
resemblyzer, and TTS latency benchmarking papers in the corpus. The ctx/tasks.json,
ctx/metrics.json, and ctx/suggestions.json cache files are available for downstream steps.
