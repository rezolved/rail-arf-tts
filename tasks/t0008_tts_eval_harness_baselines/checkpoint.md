---
spec_version: "1"
task_id: "t0008_tts_eval_harness_baselines"
updated_at: "2026-09-14T15:36:30Z"
completed_steps: 6
next_step_number: 5
next_step_id: "research-internet"
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

### Step 4 — research-papers

Paper corpus is empty (zero papers, zero categories) — all prior tasks were implementation/training
tasks without paper downloads. Wrote `research/research_papers.md` with `status: "partial"`, all 7
mandatory sections present, and an empty Paper Index; verificator passes with zero errors.

* * *

## Cross-Step Decisions

* * *

## Next Step Notes

Step 4 research-papers complete: corpus was empty so output is `status: "partial"` with no cited
papers. Step 5 research-internet is the primary literature step for this task — it should search for
GE2E speaker embeddings (Wan et al. 2018), resemblyzer, StyleTTS2/Kokoro-82M architecture, streaming
TTS TTFB measurement methodology, and WER thresholds for TTS quality. The research-internet output
will be the main literature foundation for planning (step 7). The ctx/ aggregator cache is still
available from step 3.
