---
spec_version: "1"
task_id: "t0012_v5_corpus_normalize_and_reaudit"
updated_at: "2026-09-16T06:42:01Z"
completed_steps: 9
next_step_number: 7
next_step_id: "planning"
---
# Task Objective

Fix the clipping heuristic that over-flags peak-normalized clips, LUFS-normalize the full v5 corpus
to -14 LUFS, and produce a near-full-corpus clean train manifest.

* * *

## Step History

### Step 1 — create-branch

Branch `task/t0012_v5_corpus_normalize_and_reaudit` created. Initial folder structure initialized in
`tasks/t0012_v5_corpus_normalize_and_reaudit/`. Step 1 is a mechanical setup step with no research
output.

### Step 2 — check-deps

Verified dependency `t0011_v5_data_quality_audit` has `status: completed` in its `task.json`, so the
dependency check passed with 0 errors and 0 warnings. Result recorded in
`logs/steps/002_check-deps/deps_report.json`. No caveats for downstream steps.

### Step 3 — init-folders

Created the mandatory task folder structure (`plan/`, `research/`, `results/`, `code/`,
`corrections/`, `intervention/`, `assets/`, `logs/{commands,searches,sessions,steps}/`) via
`init_task_folders`, recorded in `logs/steps/003_init-folders/folders_created.txt`. Populated the
gitignored `ctx/` aggregator cache (`task_types.json`, `costs.json`, `tasks.json`, `metrics.json`,
`suggestions.json`) for downstream subagents to reuse.

### Step 4 — research-papers

Skipped: LUFS normalization (EBU R128) is a standard, well-documented technique; no literature
validation is needed beyond t0011's prior research and its own creative-thinking findings.

### Step 5 — research-internet

Skipped: not in data-analysis `optional_steps`; task operates entirely on local audio data and prior
task code, no new external information needed.

### Step 8 — setup-machines

Skipped: task is CPU-only (pyloudnorm + soundfile on 1557 clips), no remote compute needed.

### Step 10 — teardown

Skipped: `setup-machines` was skipped, so there are no remote machines to tear down.

### Step 13 — compare-literature

Skipped: not in data-analysis `optional_steps`; this task's outputs (clean-manifest size, flag
counts) are not comparable to published baselines.

### Step 6 — research-code

Wrote `research/research_code.md` via the `/research-code` subagent, which deep-dove into
`t0011_v5_data_quality_audit`'s `code/` directory (`audit_audio.py`, `build_manifest.py`,
`plot_histograms.py`, `constants.py`, `paths.py`) and its `creative_thinking.md`/`suggestions.json`.
`verify_research_code` passed with 0 errors and 0 warnings (confirmed independently by the
step-executor). Key finding: this task directly implements t0011's follow-up suggestions S-0011-03
(replace `peak_dbfs > -0.1 dBFS` with a `clipped_fraction > 0.001` metric) and S-0011-01
(LUFS-normalize to -14 LUFS), for which t0011 already left worked-out thresholds and a code
skeleton.

* * *

## Cross-Step Decisions

* * *

## Next Step Notes

Step 6 (`research-code`) is complete; `research/research_code.md` documents that t0011 already
worked out the corrected `clipped_fraction` metric (threshold > 0.1%, S-0011-03) and a LUFS
normalization code skeleton targeting -14.0 LUFS (S-0011-01), plus the reusable pieces of
`audit_audio.py`, `build_manifest.py`, and `plot_histograms.py` to copy into this task's `code/`
directory (no cross-task library import applies here). Proceed to step 7 (`planning`): synthesize
`research/research_code.md` into `plan/plan.md`, covering the corrected clipping metric, the LUFS
normalization pass over all 1557 v5 clips, a post-normalization re-check pass, and the
clean-manifest output required by step 9 (`implementation`).
