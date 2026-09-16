# Research Summary — t0012_v5_corpus_normalize_and_reaudit

## Key Findings (top 10 insights directly actionable for this task)

1. The corrected clipping metric is fully pre-specified, not a new design decision:
   `clipped_fraction` = fraction of samples within 1 LSB of full scale, flag threshold `> 0.001`
   (0.1%). It replaces `peak_dbfs > -0.1 dBFS` (S-0011-03). Keep `peak_dbfs` in output for
   before/after reporting even though it no longer drives the flag.
2. Normalization target is -14 LUFS (EBU R128) via `pyloudnorm` (S-0011-01):
   `gain = 10 ** ((TARGET_LUFS - loudness) / 20)`, `np.clip(audio * gain, -1.0, 1.0)`. Since t0011
   found corpus LUFS mean -14.69/std 2.34 (all within [-30,-6]), required gains are modest.
3. Run the corrected audit against the **full original 1557-clip v5 manifest**
   (`tasks/t0003_kokoro_v5_phoneme_data/results/v5/train_list.txt`), NOT t0011's reduced 1311-clip
   clean list — the corrected metric is expected to reclassify most of the 224 previously
   peak-flagged clips as clean; starting from the reduced set would silently drop ~200 recoverable
   clips.
4. Post-normalization, re-run clipped-fraction (and silence) checks on the normalized audio only;
   move any clip that becomes genuinely clipped after gain to the flagged list, not the clean
   manifest. This is a mandatory safety net, not optional.
5. Write normalized audio to a fresh directory `data/v5_normalized/`, never mutate source WAVs in
   place (DVC anti-pattern; also keeps t0011's folder immutable per repo rule 5).
6. `dvc pull` is unreliable for this corpus (Azure credential-chaining failures in t0011); budget
   for fallback `az storage blob download --auth-mode login` (16 parallel workers, ~64 min for
   1557+96 clips). Worktree's `data/v4/train/wavs/` and `data/v4/val/wavs/` are currently empty.
7. All needed deps (`pyloudnorm`, `soundfile`, `librosa`, `numpy`, `matplotlib`) already exist in
   top-level `pyproject.toml` and were proven at this corpus scale by t0011 (<60s pure compute) — no
   new dependencies needed.
8. Neither registered library (`tts_eval_harness`, `t0009_training_safeguards`) is relevant — no
   synthesis/scoring, no GPU training. Do not import either.
9. Keep a `MIN_CLEAN_CLIPS`-style intervention floor (t0011 used 1200; expected final size here is
   ~1550+, so ~1400 is a safe floor) — writes `intervention/too_few_clean_clips.md` and exits
   nonzero if too few clips survive, catching threshold/normalization bugs early.
10. Repeat t0011's verification checklist: flagged+clean == 1557 total-count consistency, 0 val_96
    leakage into the clean manifest (CLAUDE.md forbids training on val_96), JSONL required-field
    completeness.

## Best Approaches (top 3 recommended implementation approaches from research)

### Approach 1: Copy and adapt t0011's four-script pipeline

Reuse `constants.py`, `paths.py`, `audit_audio.py` (`compute_clip_stats`), `build_manifest.py`
(`_get_flag_reasons`, `_compute_metric_stats`, `main`), and `plot_histograms.py` verbatim-copied
into `tasks/t0012_.../code/`, then adapt: add `clipped_fraction` field/threshold, add `TARGET_LUFS`,
add pre/post field names, add a normalization module, and add a second flag pass for the
post-normalization re-check. Do not copy `audit_transcripts.py` (OOV out of scope — no change from
gain normalization, already 0 flags in t0011).

### Approach 2: Two-pass audit structure (pre-norm gate, then post-norm safety net)

Pass 1: run corrected audit (clipped_fraction + existing thresholds) on the original 1557 clips to
decide who is eligible for normalization. Pass 2: normalize eligible clips to -14 LUFS, write to
`data/v5_normalized/`, then re-run clipped_fraction + silence checks on the normalized output only,
demoting any newly-clipped clip to flagged. Final clean manifest = eligible clips that also pass
pass 2.

### Approach 3: Before/after reporting via reused histogram plotting

Reuse `plot_histogram(train_vals, val_vals, title, xlabel, out_path, flag_lines)` from
`plot_histograms.py`, but feed (pre-normalization, post-normalization) value pairs instead of
(train, val) and relabel series. Produces `peak_before_after.png`, `lufs_before_after.png`,
`clipped_fraction_distribution.png` (single-series call) as required by task_description.md §6.

## Reusable Code / Assets

* `tasks/t0011_v5_data_quality_audit/code/constants.py` — threshold/field-name constants; copy, swap
  `PEAK_DBFS_MAX` for `CLIPPED_FRACTION_MAX = 0.001`, add `TARGET_LUFS = -14.0`.
* `tasks/t0011_v5_data_quality_audit/code/paths.py` — centralized Path constants; copy, add v2
  output paths and `V5_NORMALIZED_DIR`.
* `tasks/t0011_v5_data_quality_audit/code/audit_audio.py` (`compute_clip_stats`, `_parse_list`) —
  per-clip stats computation; copy, add `clipped_fraction`, make reusable for both pre/post passes.
* `tasks/t0011_v5_data_quality_audit/code/build_manifest.py` (`_get_flag_reasons`,
  `_compute_metric_stats`, `main`, `MIN_CLEAN_CLIPS` intervention pattern) — flagging/reporting;
  copy, swap clipping branch, add post-normalization second pass.
* `tasks/t0011_v5_data_quality_audit/code/plot_histograms.py` (`plot_histogram`) — before/after
  histogram plotting; copy, repurpose train/val series as pre/post series.
* `tasks/t0011_v5_data_quality_audit/data/per_clip_stats.jsonl` (read-only reference) — useful for
  consistency-checking which of the 224 previously peak-flagged clips reclassify as clean; not
  copied, t0011's folder is immutable.

## Key Papers (top 5, with finding most relevant to this task)

(not generated — research-papers step skipped; not applicable to this CPU-only audio data task)

## Risks Flagged in Research

* `dvc pull` likely fails on Azure credential chaining; plan the `az storage blob download` fallback
  up front rather than debugging DVC credentials from scratch (~64 min for full corpus).
* Threshold calibration must be spot-checked against real data before finalizing — t0011's original
  `-1.0 dBFS` threshold failed a 20-clip preflight (90% exceeded it) and had to be adjusted in
  flight; the `clipped_fraction > 0.001` metric is pre-validated by t0011's creative-thinking
  analysis but still warrants a small sanity spot-check.
* Gain normalization can push a previously-safe clip into genuine clipping — the post-normalization
  re-check (task step 4) is mandatory, not a nice-to-have.
* Without a `MIN_CLEAN_CLIPS`-style floor, a miscalibrated threshold or normalization bug could
  silently shrink the clean manifest without being caught by script logic alone.

## Full Detail Available In

* `tasks/t0012_v5_corpus_normalize_and_reaudit/research/research_papers.md` — (not generated — step
  skipped)
* `tasks/t0012_v5_corpus_normalize_and_reaudit/research/research_internet.md` — (not generated —
  step skipped)
* `tasks/t0012_v5_corpus_normalize_and_reaudit/research/research_code.md` — 3 tasks cited, 2
  libraries reviewed (0 relevant)
