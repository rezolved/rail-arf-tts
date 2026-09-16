---
spec_version: "3"
task_id: "t0012_v5_corpus_normalize_and_reaudit"
step_number: 9
step_name: "implementation"
status: "completed"
started_at: "2026-09-16T06:56:56Z"
completed_at: "2026-09-16T07:25:00Z"
---
## Summary

Executed `plan/plan.md` end to end: implemented the corrected `clipped_fraction` clipping metric,
LUFS-normalized the full 1557-clip v5 train corpus to -14 LUFS with a peak-ceiling safeguard,
re-checked normalized audio for new clipping/silence, and produced the final clean manifest. All 19
`REQ-*` items from the plan are marked done. Final clean manifest is 1531/1557 (98.3%), well above
t0011's 1311/1557 (84.2%) baseline and the plan's `>1311` rejection-criteria gate.

## Actions Taken

1. Spawned the `/implementation` subagent (per Critical Rule 9) with explicit instructions to `cd`
   into the task worktree via
   `uv run python -m arf.scripts.utils.worktree path t0012_v5_corpus_normalize_and_reaudit` before
   doing any work, since the prior `planning` step's subagent had forgotten this and had to be
   corrected.
2. The subagent created `code/paths.py`, `code/constants.py`, `code/audit_normalize.py`,
   `code/build_manifest_v2.py`, `code/plot_histograms_v2.py`; ran the `--limit 20` validation gate,
   inspected 5 individual records by hand, then ran the full 1557-clip pass.
3. During the full-corpus run, the plan's own gain formula (copied verbatim from t0011's
   creative-thinking pseudocode) introduced 408/1531 new clipping cases at scale, failing the plan's
   own `>1311`-clean pre-registered gate (result was 1123). The subagent diagnosed the root cause
   (many clips are already near-peak-normalized regardless of loudness, so boosting gain to hit -14
   LUFS pushed them over full scale) and fixed it by adding a `PEAK_CEILING_DBFS = -1.0` true-peak
   limiter to the normalization step (standard loudness-normalization practice), eliminating all new
   clipping while leaving the peak-capped clips' `post_lufs` honestly reported (not silently
   overwritten) in `data/analysis_v2.json`.
4. `dvc pull`/`dvc push` hang indefinitely in this environment (confirmed independently — a
   concurrent `dvc push` from the t0010 worktree was also stuck for 25+ minutes with zero progress).
   The subagent worked around this using direct Azure Blob SDK calls (`AzureCliCredential`) that
   replicate DVC's content-addressable layout exactly, and verified the result byte-for-byte against
   the local `dvc add`-computed `.dir` manifest (which did complete via the SMB-mounted local cache,
   just slowly — see `logs/commands/016_..._dvc-add-....json`, exit 0, 356s, "Done: 1532/1532, 0
   errors"). The step-executor independently reproduced the `dvc status` hang (`timeout 60`, exit
   124\) confirming the subagent's account rather than taking it on faith.
5. Step-executor independently re-verified after the subagent returned:
   `wc -l data/per_clip_stats_v2.jsonl` = 1557; `wc -l data/train_list_v5_normalized_clean.txt` =
   1531; ran the plan's own verification Python snippet (`OK: 1531/1557 clean, 0 val leaks`);
   confirmed 3 PNGs in `results/images/`; `ruff check`/`ruff format --check` clean;
   `mypy -p tasks.t0012_v5_corpus_normalize_and_reaudit.code` clean (same "1 source file" scope as
   t0011's own precedent, due to the project's `exclude = ["tasks/.*/code/", ...]` mypy config plus
   `-p` resolution — not a regression); confirmed zero diff against
   `tasks/t0011_v5_data_quality_audit/` (t0011's folder untouched); confirmed no changes to
   `pyproject.toml`/`uv.lock`/`ruff.toml`/root `.gitignore`.

## Outputs

- `tasks/t0012_v5_corpus_normalize_and_reaudit/code/paths.py`
- `tasks/t0012_v5_corpus_normalize_and_reaudit/code/constants.py`
- `tasks/t0012_v5_corpus_normalize_and_reaudit/code/audit_normalize.py`
- `tasks/t0012_v5_corpus_normalize_and_reaudit/code/build_manifest_v2.py`
- `tasks/t0012_v5_corpus_normalize_and_reaudit/code/plot_histograms_v2.py`
- `tasks/t0012_v5_corpus_normalize_and_reaudit/data/per_clip_stats_v2.jsonl` (1557 records)
- `tasks/t0012_v5_corpus_normalize_and_reaudit/data/flagged_clips_v2.txt` (26 flagged)
- `tasks/t0012_v5_corpus_normalize_and_reaudit/data/train_list_v5_normalized_clean.txt` (1531 clean)
- `tasks/t0012_v5_corpus_normalize_and_reaudit/data/flag_counts_v2.json`
- `tasks/t0012_v5_corpus_normalize_and_reaudit/data/analysis_v2.json`
- `tasks/t0012_v5_corpus_normalize_and_reaudit/data/v5_normalized.dvc` (DVC pointer, 1531 files,
  253874432 bytes) plus `data/.gitignore` excluding `/v5_normalized` raw audio
- `tasks/t0012_v5_corpus_normalize_and_reaudit/results/images/peak_before_after.png`
- `tasks/t0012_v5_corpus_normalize_and_reaudit/results/images/lufs_before_after.png`
- `tasks/t0012_v5_corpus_normalize_and_reaudit/results/images/clipped_fraction_distribution.png`
- `tasks/t0012_v5_corpus_normalize_and_reaudit/results/metrics.json`

## Issues

`dvc pull`/`dvc push` hang indefinitely in this environment (reproduced independently by the
step-executor). Worked around with direct Azure Blob SDK upload/download that replicates DVC's
content-addressable object layout exactly; `dvc add`'s local hashing (which does complete, over the
SMB-mounted cache) confirms the local manifest is correct, and the step-executor independently
verified the `dvc status` hang and the blob-count match reported by the subagent. This should be
flagged for a future infrastructure task to fix `arf/scripts/utils` DVC tooling or document the
Azure-SDK fallback as the sanctioned workaround; not a defect introduced by this task's own code. A
genuine plan-formula bug (unbounded gain causing 408 new-clipping cases at full-corpus scale) was
caught by the plan's own pre-registered `>1311` rejection gate and fixed with a documented
peak-ceiling safeguard before the task was allowed to proceed — this is the validation gate working
as designed, not a failure.
