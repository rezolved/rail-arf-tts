---
spec_version: "1"
task_id: "t0008_tts_eval_harness_baselines"
updated_at: "2026-09-14T15:54:00Z"
completed_steps: 8
next_step_number: 7
next_step_id: "planning"
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

### Step 5 — research-internet

Conducted 11 searches, found 23 sources, discovered 4 papers (GE2E/Wan2018, StyleTTS2/Li2023,
VERSA2024, AnalyzeSim2025). Key findings: resemblyzer same-speaker cosine ≈ 0.85+ validates the
success criterion; Kokoro GPU RTF ≈ 0.03 on A100, first-chunk 28–97 ms; ElevenLabs p50 TTFB 264–335
ms; WER thresholds < 5% clean, > 20% hard failure. Output: `research/research_internet.md`.

### Step 6 — research-code

Reviewed 7 completed tasks, 0 registered libraries. Key outputs: five-module checkpoint packaging
via `tasks/t0002.../code/extract_decoder_generic.py` (38 lines, copy into task); mandatory synthesis
entry point `tasks/t0003.../code/build_pipeline.py` (import directly; `lang_code="b"` + brand
lexicon); t0005 best = `epoch_2nd_00003.pth` (val 0.848); t0006 v6d best = `epoch_2nd_00006.pth`
(val 0.846); all checkpoints DVC-only. Research summary written to `research/research_summary.md`.

* * *

## Cross-Step Decisions

* * *

## Next Step Notes

Step 7 is planning. The planner should read `research/research_summary.md` for a compact overview,
then dive into `research/research_internet.md` for detailed TTFB measurement methodology and
`research/research_code.md` for checkpoint paths and the five-module extraction requirement. The
plan must specify: (1) checkpoint packaging steps for t0005/t0006 raw `.pth` files via a copy of
`extract_decoder_generic.py`; (2) the eight synthesis adapters (each calling `build_pipeline`); (3)
the centroid-half split design (679/679 seed=42) for resemblyzer; (4) the VM setup for TTFB
measurement (filler corpus confirmation/copy + `dvc push`); and (5) the reporting format (variant
metrics.json, tables, two charts). Budget ≤ $30 total for this task.
