---
spec_version: "1"
task_id: "t0011_v5_data_quality_audit"
updated_at: "2026-09-15T12:33:41Z"
completed_steps: 14
next_step_number: 15
next_step_id: "reporting"
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

### Step 6 — research-code

Reviewed 9 completed tasks; found 2 libraries (`tts_eval_harness`, `t0009_training_safeguards`);
identified 6 reusable code items. Key output: `research/research_code.md` (verified, zero errors)
and `research/research_summary.md` (101-line compressed summary for downstream agents). Critical
finding: t0009 explicitly deferred all audio-level metrics because DVC data was not pulled — t0011
must `dvc pull` first, then reuse the manifest-parsing, histogram, and OOV-detection patterns from
t0003/t0009.

### Step 7 — planning

Plan written to `plan/plan.md` (verified, zero errors). Seven REQ items cover: DVC pull (REQ-1),
per-clip audio metrics (REQ-2), flag thresholds + manifests (REQ-3, REQ-4), OOV audit (REQ-5), four
histograms (REQ-6), and distribution stats (REQ-7). Five scripts specified in `code/`: `paths.py`,
`constants.py`, `audit_audio.py`, `audit_transcripts.py`, `build_manifest.py`, `plot_histograms.py`.
Budget: $0. Remote machines: none.

### Step 8 — setup-machines

Skipped: CPU-only data audit, no GPU required.

### Step 9 — implementation

All audit scripts written and run successfully. 1557 train + 96 val WAVs downloaded via az CLI (DVC
pull failed due to credential chaining). Per-clip stats JSONL: 1557 records, 0 errors. OOV audit: 0
clips with oov_fraction > 0.20. Flagging: 246 clips removed (clipping: 224, duration_low: 25,
silence: 1). Clean manifest: 1311 clips (84.2%). All 4 histogram PNGs generated. Results files
written: metrics.json, costs.json, remote_machines_used.json, results_detailed.md,
results_summary.md. Key threshold adjustment: PEAK_DBFS_MAX changed from -1.0 to -0.1 dBFS
(ElevenLabs corpus is peak-normalized, not hard-clipped).

### Step 10 — teardown

Skipped: no remote machines provisioned.

### Step 11 — creative-thinking

Five alternative strategies explored for the peak-normalization and short-clip findings. Key
recommendation: LUFS-normalize the full 1557-clip corpus to −14 LUFS before Stage 2 training instead
of flagging 224 peak-normalized clips. Output at `results/creative_thinking.md`.

### Step 12 — results

All results files verified: `results_summary.md`, `results_detailed.md`, `metrics.json`,
`costs.json`, `remote_machines_used.json`, 4 PNG charts. Fixed `## Examples` section to include 10
fenced-code-block examples from actual `per_clip_stats.jsonl` data (required by `data-analysis` task
type). Both `verify_task_metrics` and `verify_task_results` pass with zero errors.

### Step 13 — compare-literature

Skipped: data audit produces no quantitative results comparable to published baselines.

### Step 14 — suggestions

Three follow-up suggestions generated and written to `results/suggestions.json`. Key suggestions:
LUFS-normalize all 1557 clips to -14 LUFS and use them for Stage 2 (S-0011-01, high), run
full-corpus Stage 2 on the 1311-clip clean manifest after t0010 validates safeguards (S-0011-02,
high), replace peak_dbfs threshold with clipped-fraction filter for a more principled audit
(S-0011-03, medium). Verificator passed zero errors.

* * *

## Cross-Step Decisions

- Budget confirmed $0 — CPU-only local compute; DVC pull from existing Azure Blob storage.
- val_96 (`data/v4/val/val_list.txt`) must never appear in `train_list_v5_clean.txt` — verified: 0
  leakage found.
- PEAK_DBFS_MAX adjusted to -0.1 dBFS (from -1.0) to avoid flagging peak-normalized TTS output.

* * *

## Next Step Notes

Step 15 (reporting): run all verificators (`verify_task_file`, `verify_task_dependencies`,
`verify_suggestions`, `verify_task_metrics`, `verify_task_results`, `verify_task_folder`,
`verify_logs`), capture session transcripts via `capture_task_sessions`, update `task.json` status
to `completed` with `end_time`, then write `step_log.md` and commit. No asset verificators apply
(expected_assets is `{}`). No remote machines to verify. The creative_thinking.md file should be
included in the reporting verificator pass.
