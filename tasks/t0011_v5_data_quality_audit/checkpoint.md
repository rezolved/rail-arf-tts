---
spec_version: "1"
task_id: "t0011_v5_data_quality_audit"
updated_at: "2026-09-15T10:46:00Z"
completed_steps: 8
next_step_number: 6
next_step_id: "research-code"
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

* * *

## Cross-Step Decisions

* * *

## Next Step Notes

Step 4 (research-papers) completed. Key finding: use ITU-R BS.1770/EBU R128 integrated loudness via
pyloudnorm; flag clips with peak > -1 dBFS, silence > 30%, duration < 1.5s or > 15s. Proceed to step
6 (research-code) to review t0009 code for reusable audio loading and metrics utilities. The
research_summary.md is produced after all research steps complete.
