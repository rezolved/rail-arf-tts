---
spec_version: "1"
task_id: "t0011_v5_data_quality_audit"
updated_at: "2026-09-15T10:55:00Z"
completed_steps: 9
next_step_number: 7
next_step_id: "planning"
---
# Task Objective

Audit all 1557 v5 training clips for clipping, silence, LUFS, duration outliers, and OOV tokens.
Produce a cleaned train manifest for use in the full-corpus Stage 2 run after t0010 confirms the
safeguards work.

* * *

## Step History

### Step 1 — create-branch

Branch `task/t0011_v5_data_quality_audit` created. Initial folder structure initialized in
`tasks/t0011_v5_data_quality_audit/`. Step 1 is a mechanical setup step with no research output.

### Step 2 — check-deps

Dependency `t0009_stage2_training_failure_forensics` verified as completed. Output written to
`logs/steps/002_check-deps/deps_report.json`. No errors or warnings.

### Step 3 — init-folders

Task folder structure created via `init_task_folders` (12 directories with `.gitkeep` files,
`__init__.py` stubs). Aggregator cache populated in `tasks/t0011_v5_data_quality_audit/ctx/`
(task_types, costs, tasks, metrics, suggestions). All ctx files are gitignored and local-only.

### Step 4 — research-papers

Paper corpus has zero entries; domain knowledge synthesized about ITU-R BS.1770/EBU R128 loudness
standards, pyloudnorm, silence detection thresholds, and TTS corpus curation practices. Output at
`research/research_papers.md` (status: partial). Verificator passed with zero errors.

### Step 5 — research-internet

Skipped: all audit work is local; no new external sources needed.

### Step 8 — setup-machines

Skipped: CPU-only data audit, no GPU required.

### Step 10 — teardown

Skipped: no remote machines provisioned.

### Step 13 — compare-literature

Skipped: data audit produces no quantitative results comparable to published baselines.

### Step 6 — research-code

Reviewed 9 completed tasks; found 2 libraries (`tts_eval_harness`, `t0009_training_safeguards`);
identified 6 reusable code items. Key output: `research/research_code.md` (verified, zero errors)
and `research/research_summary.md` (101-line compressed summary for downstream agents). Critical
finding: t0009 explicitly deferred all audio-level metrics because DVC data was not pulled — t0011
must `dvc pull` first, then reuse the manifest-parsing, histogram, and OOV-detection patterns from
t0003/t0009.

* * *

## Cross-Step Decisions

* * *

## Next Step Notes

All research steps are now complete (research-papers, research-code; research-internet skipped).
`research/research_summary.md` is written and ready. Proceed to step 7 (planning): design the audit
pipeline using `soundfile`/`librosa`/`pyloudnorm` for audio metrics, the `_parse_list`
manifest-parser pattern from t0009, OOV constants from t0003, and the histogram helper pattern from
t0009. The plan must specify which scripts will be created in `code/` (e.g. `paths.py`,
`audit_audio.py`, `audit_transcripts.py`, `plot_histograms.py`, `build_manifest.py`), inputs,
outputs, and algorithm for each. DVC pull of `data/v5/` is the first implementation step.
