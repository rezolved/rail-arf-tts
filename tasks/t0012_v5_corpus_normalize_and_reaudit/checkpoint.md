---
spec_version: "1"
task_id: "t0012_v5_corpus_normalize_and_reaudit"
updated_at: "2026-09-16T06:37:34Z"
completed_steps: 8
next_step_number: 6
next_step_id: "research-code"
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

* * *

## Cross-Step Decisions

* * *

## Next Step Notes

Step 3 completed successfully; task folder structure and the `ctx/` aggregator cache are in place.
Steps 4 and 5 are already marked skipped in `step_tracker.json`. Proceed to step 6
(`research-code`): review t0011's `per_clip_stats.jsonl`, `audit_audio.py`, and its clipping
thresholds as the starting point for the corrected audit and LUFS-normalization code. Use the cached
`ctx/tasks.json` and `ctx/task_types.json` instead of re-running aggregators.
