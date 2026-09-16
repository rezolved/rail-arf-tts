---
spec_version: "1"
task_id: "t0012_v5_corpus_normalize_and_reaudit"
updated_at: "2026-09-16T07:45:00Z"
completed_steps: 15
next_step_number: null
next_step_id: null
---
# Task Objective

Fix the clipping heuristic that over-flags peak-normalized clips, LUFS-normalize the full v5 corpus
to -14 LUFS, and produce a near-full-corpus clean train manifest.

* * *

## Step History

### Step 1 — create-branch

Branch `task/t0012_v5_corpus_normalize_and_reaudit` created. Task folder structure initialized.

### Step 2 — check-deps

Dependency `t0011_v5_data_quality_audit` confirmed `status: completed`. Passed 0 errors/warnings.

### Step 3 — init-folders

Created mandatory folder structure via `init_task_folders`. Populated gitignored `ctx/` aggregator
cache for downstream subagents (removed at step 15 once no longer needed).

### Step 4 — research-papers

Skipped: LUFS normalization (EBU R128) is standard; no literature validation needed beyond t0011.

### Step 5 — research-internet

Skipped: not in `optional_steps`; task uses only local audio data and prior task code.

### Step 8 — setup-machines

Skipped: CPU-only task (pyloudnorm + soundfile on 1557 clips), no remote compute needed.

### Step 10 — teardown

Skipped: `setup-machines` was skipped, no remote machines to tear down.

### Step 13 — compare-literature

Skipped: not in `optional_steps`; outputs not comparable to published baselines.

### Step 6 — research-code

Wrote `research/research_code.md`, deep-diving t0011's `code/` and its
creative-thinking/suggestions. `verify_research_code` passed, 0 errors/warnings. Key finding: this
task implements t0011's S-0011-03 (`clipped_fraction > 0.001` metric) and S-0011-01 (-14 LUFS
normalization), with thresholds and a code skeleton already worked out. Summarized into
`research/research_summary.md` for downstream agents.

### Step 7 — planning

Wrote `plan/plan.md`: 19 `REQ-*` items, adapts t0011's audit/manifest/histogram code, adds
`clipped_fraction` metric and single decode-normalize-recheck pass, validation gates (`--limit 20`
first, "0 reclassified = STOP"), rejection criteria (DVC completeness, clean-count exceeding t0011's
1311/1557, zero val_96 leakage). `verify_plan` passed, 0 errors/warnings. Caveat: plan was first
written outside the worktree and relocated by the step-executor before committing; no main-repo
tracked files touched.

### Step 9 — implementation

Implemented all 19 `REQ-*` items: audit/normalize/manifest/histogram code,
`data/per_clip_stats_v2.jsonl` (1557 records), `train_list_v5_normalized_clean.txt` (1531/1557,
98.3% vs t0011's 84.2%), `data/v5_normalized.dvc`. 0 genuine pre-existing clipping; 0 val_96 leaks.
Self-caught/fixed: (1) plan's gain formula caused 408 new clipping cases at full scale — caught by
the rejection gate, fixed with `PEAK_CEILING_DBFS = -1.0`; (2) `dvc pull`/`push` hang in this
environment — worked around with a byte-verified direct Azure Blob SDK upload/download.
`ruff`/`mypy` clean; zero diff outside the task folder. **Downstream flag**: the DVC hang needs an
infra fix.

### Step 11 — creative-thinking

Wrote `results/creative_thinking.md`. Headline: the -1 dBFS peak ceiling makes LUFS normalization
effectively one-directional — 850/852 clips needing a boost remain short of -14 LUFS target
(post-LUFS std 1.58, down from 2.34, not ~0). Also flagged the 25 `duration_low` exclusions skew
toward short, high-frequency filler phrases, not truncated audio. No REQ answer changes.

### Step 12 — results

Wrote all five mandatory `results/` files. Clean manifest **1531/1557 (98.3%)** vs t0011's 1311/1557
(84.2%); 220/224 clipping-flagged clips reclassified clean, 0 genuinely clipped, 0 new clipping.
`metrics.json = {}` (no registered metric applies). `verify_task_metrics` and `verify_task_results`
both passed, 0 errors/warnings.

### Step 14 — suggestions

`/generate-suggestions` subagent wrote `results/suggestions.json`: S-0012-01 (duration-aware floor
for excluded filler clips), S-0012-02 (fix/document the DVC hang), S-0012-03 (extract the audit/LUFS
pipeline into a library), S-0012-04 (verify LUFS normalization doesn't hurt GE2E similarity before
full Stage 2). Deduplicated against active suggestions and prior tasks; no overlaps.
`verify_suggestions` passed, 0 errors/warnings.

### Step 15 — reporting

Ran full verificator sweep (`verify_task_file`, `verify_task_dependencies`, `verify_suggestions`,
`verify_task_metrics`, `verify_task_results`, `verify_research_code`, `verify_task_folder`,
`verify_logs`) — all passed, 0 errors (only expected warnings: empty `expected_assets`, empty
`logs/searches/`, no asset subdirs, 3 pre-existing non-zero-exit command logs already documented
above). `compare-literature`/`setup-machines`/`teardown` were skipped so their verificators don't
apply; `corrections/` is empty so `verify_corrections` doesn't apply. Removed a leftover gitignored
`ctx/` dir that tripped `verify_task_folder`'s FD-E016. Ran `capture_task_sessions` (0 transcripts
found; wrote `capture_report.json`). Set `task.json` `status: "completed"`, `end_time` set.

* * *

## Cross-Step Decisions

* DVC `pull`/`push` hang in this environment; use direct Azure Blob SDK (`AzureCliCredential`)
  replicating DVC's content-addressable `.dir` layout as the verified fallback until a framework fix
  lands. Verify byte-for-byte against `dvc add`'s locally-computed manifest before trusting the
  upload.
* Loudness normalization to a fixed LUFS target must cap true peak (this task uses -1 dBFS via
  `PEAK_CEILING_DBFS`) — an unbounded gain formula reintroduces clipping at corpus scale even though
  it looked correct on a small `--limit 20` sample.
* The peak ceiling that prevents new clipping also caps upward loudness correction: 850/852 clips
  (99.8%) that needed a boost to reach -14 LUFS remain below target (post-LUFS std 1.58, down from
  2.34 pre, not ~0). Stage 2 training should treat this residual loudness variance as a known,
  quantified factor if training curves show unexplained instability — see
  `results/creative_thinking.md` §1.
* The 25 `duration_low`-excluded clips are disproportionately short, high-frequency filler phrases
  ("got it", "sure thing", "of course"), not truncated audio. `DURATION_MIN_S` was kept unchanged
  from t0011 per this task's scope; a future task could revisit a duration-aware (transcript-length
  -scaled) floor to recover them — see `results/creative_thinking.md` §3.

* * *

## Next Step Notes

None — this was the final task step. All 15 applicable steps (1-3, 6, 7, 9, 11, 12, 14, 15; steps 4,
5, 8, 10, 13 skipped with documented rationale) are complete. `task.json` is `status: "completed"`.
Ready for the coordinator's Phase 7 (PR creation, `verify_pr_premerge`, merge) and Phase 8 (final
`verify_task_complete`).
