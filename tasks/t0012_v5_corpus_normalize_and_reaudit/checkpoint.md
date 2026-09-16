---
spec_version: "1"
task_id: "t0012_v5_corpus_normalize_and_reaudit"
updated_at: "2026-09-16T07:30:00Z"
completed_steps: 11
next_step_number: 11
next_step_id: "creative-thinking"
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

Since steps 4 and 5 were skipped, research-code was the only research stage to run. Per the
`execute-task` skill's "Summarize research" step, a subagent then compressed `research_code.md` into
`research/research_summary.md` (101 lines, ~6.98 KB) for downstream planning/implementation agents.

### Step 7 — planning

The `/planning` subagent wrote `plan/plan.md` (spec_version "2", status "complete"), synthesizing
`research/research_summary.md` and t0011's `results/creative_thinking.md` /
`results/suggestions.json` into 19 `REQ-*` requirement items, an approach that copies/adapts t0011's
`audit_audio.py`, `build_manifest.py`, and `plot_histograms.py`, adds a new `clipped_fraction`
metric (bit-depth read via `soundfile.info().subtype`, defaulting to 16-bit for unrecognized
subtypes) and a single decode-normalize-recheck pass (no second disk read), validation gates
(`--limit 20` before the full 1557-clip run, baseline "0 reclassified = STOP"), and pre-registered
rejection criteria (DVC-pull completeness, clean-count must exceed t0011's 1311/1557 baseline, zero
val_96 leakage). `verify_plan` passed with 0 errors/0 warnings, confirmed independently by the
step-executor via `run_with_logs.py`. Caveat for downstream: the subagent initially ran in the main
repo checkout on branch `main` instead of the task worktree, leaving an untracked `plan/plan.md`
there; the step-executor relocated it into the worktree and removed the stray copy before committing
— no main-repo tracked files were touched, so no follow-up cleanup on `main` is needed.

### Step 9 — implementation

The `/implementation` subagent implemented all 19 `REQ-*` items:
`code/{paths,constants, audit_normalize,build_manifest_v2,plot_histograms_v2}.py`,
`data/per_clip_stats_v2.jsonl` (1557 records), `data/flagged_clips_v2.txt` (26 flagged),
`data/train_list_v5_normalized_clean.txt` (1531 clean, 98.3% — vs t0011's 1311/1557 baseline),
`data/flag_counts_v2.json`, `data/analysis_v2.json`, `data/v5_normalized.dvc`, and 3 histograms in
`results/images/`. Genuine pre-existing clipping: 0 clips (of the 224 originally peak-flagged, 220
reclassified clean, 4 remain excluded only for being &lt;1.5s short). 0 val_96 leaks confirmed
independently by the step-executor.

Two notable findings, both self-caught and fixed within the step (not carried forward as open
issues): (1) the plan's own gain formula, copied verbatim from t0011's pseudocode, caused 408 new
clipping cases at full-corpus scale — caught by the plan's own pre-registered `>1311` rejection gate
and fixed with a `PEAK_CEILING_DBFS = -1.0` true-peak limiter; (2) `dvc pull`/`dvc push` hang
indefinitely in this environment (step-executor independently reproduced the hang) — worked around
with a direct Azure Blob SDK upload/download that replicates DVC's content-addressable layout, byte-
verified against the (separately-completing) local `dvc add` hash. `data/v5_normalized/` itself is
gitignored (`data/.gitignore`); only the `.dvc` pointer is committed. `ruff`/`mypy` clean; zero diff
against `tasks/t0011_v5_data_quality_audit/`; zero diff outside the task folder.

**Downstream flag**: the DVC hang is an environment/tooling issue, not specific to this task — a
future infrastructure task should investigate `arf/scripts/utils` DVC config or formally document
the Azure-SDK fallback.

* * *

## Cross-Step Decisions

* DVC `pull`/`push` hang in this environment; use direct Azure Blob SDK (`AzureCliCredential`)
  replicating DVC's content-addressable `.dir` layout as the verified fallback until a framework fix
  lands. Verify byte-for-byte against `dvc add`'s locally-computed manifest before trusting the
  upload.
* Loudness normalization to a fixed LUFS target must cap true peak (this task uses -1 dBFS via
  `PEAK_CEILING_DBFS`) — an unbounded gain formula reintroduces clipping at corpus scale even though
  it looked correct on a small `--limit 20` sample.

* * *

## Next Step Notes

Step 9 (`implementation`) is complete and fully verified — all deliverables exist, all REQ items
done, no open issues. Step 10 (`teardown`) is already marked `skipped` in `step_tracker.json` (no
remote machines were used). Proceed to step 11 (`creative-thinking`): explore the plan's Key
Questions using `data/analysis_v2.json` and `data/flag_counts_v2.json` as the primary data sources —
genuine-clipping count (0), new-clipping-from-normalization count (0, after the peak-ceiling fix),
manifest size vs baselines (1531/1557 vs 1311/1557 and the two suggestions' individual ~1532/~1556
estimates), and the duration/silence-flag interaction cross-tab already computed in
`analysis_v2.json` for Key Question 4.
