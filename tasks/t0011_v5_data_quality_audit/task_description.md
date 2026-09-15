# v5 Training Data Audio Quality Audit and Clean Manifest

## Motivation

t0009's data audit could not compute audio quality metrics (peak amplitude, LUFS, silence rate,
clipping) because the v5 clips are DVC-tracked and were not pulled during forensics. The creative
analysis in t0009 flagged malformed or outlier clips as a plausible secondary contributor to
training instability: a single clipped or near-silent clip can produce NaN gradients that the health
gates stop but do not prevent.

t0010 trains on the existing 250-clip manifest. If t0010 succeeds and the target approaches 0.85,
the next training run will scale to the full 1557-clip corpus. Running that scale-up on unaudited
data risks reproducing the instability t0009 fixed. This task audits the full 1557-clip corpus
first, produces a cleaned manifest, and characterizes the v5 distribution so that the full-corpus
Stage 2 task can reference it.

This task runs no training and needs no GPU.

## Key Questions

1. What fraction of v5 train clips have peak amplitude > −1 dBFS (clipping)?
2. What fraction have silence occupying > 30% of total duration?
3. What is the LUFS distribution, and are there extreme outliers (e.g. > −6 LUFS or < −30 LUFS)?
4. Are there duration outliers (< 1.5 s or > 15 s) that could destabilize the mel extractor?
5. What is the phoneme-per-second rate distribution, and are there frames-per-phoneme outliers?
6. Are there OOV or vocab-invalid tokens in the train transcripts (beyond what t0003 already fixed)?

## Scope

### 1. Pull audio data

Run `dvc pull tasks/t0003_kokoro_v5_phoneme_data/data/v5/` and
`dvc pull tasks/t0003_kokoro_v5_phoneme_data/data/v4/val/` to pull the 1557 train clips and 96 val
clips locally. Confirm clip count matches the manifest before proceeding.

### 2. Per-clip audio stats

For each clip in the v5 train manifest compute:

- **peak_dbfs**: `20 * log10(|max sample|)` — flag if > −1 dBFS.
- **rms_lufs**: integrated loudness via `pyloudnorm` or `ffmpeg -filter_complex ebur128`.
- **silence_fraction**: fraction of frames with RMS < −60 dBFS.
- **duration_s**: clip length in seconds.
- **sample_rate**: confirm all clips are 24 kHz (mismatch → flag).
- **channels**: confirm all mono (stereo → flag).

Save one JSON record per clip to `data/per_clip_stats.jsonl`.

### 3. Flag thresholds and manifest

Flag a clip if any of:

- peak_dbfs > −1 dBFS (clipping)
- silence_fraction > 0.30
- duration_s < 1.5 or > 15.0
- sample_rate ≠ 24000
- channels > 1

Save flagged clip IDs to `data/flagged_clips.txt`. Write a cleaned train manifest (original minus
flagged) to `data/train_list_v5_clean.txt`. Report flag counts by category.

### 4. Transcript audit

For each clip in the train manifest, run the text through the phonemizer used in t0003
(`build_pipeline.py`, British `lang_code="b"`) and count OOV tokens (those that fall back to
character-by-character pronunciation). Flag clips where OOV fraction > 20% of tokens.

### 5. Distribution charts

Produce histograms for:

- Peak dBFS distribution (train vs val).
- LUFS distribution (train vs val).
- Duration distribution (train vs val).
- Phoneme string length (train vs val) — extends t0009's proxy analysis with real per-clip phoneme
  counts.

### 6. Summary

Report total clips, flagged count by category, cleaned manifest size, and distribution stats (mean,
p5, p95, p99 for each metric). Document any systematic patterns (e.g. specific speaker IDs or
recording sessions that cluster in flagged clips).

## Compute and Budget

CPU-only. `pyloudnorm` + `librosa` on 1557 clips: < 30 min on a standard CPU.

Planned total: $0 (local compute only).

Write `results/costs.json` with `total_cost_usd: 0`.

## Expected Outputs

- `data/per_clip_stats.jsonl` — one JSON record per clip with all audio metrics.
- `data/flagged_clips.txt` — clip IDs that failed one or more thresholds.
- `data/train_list_v5_clean.txt` — cleaned train manifest (for use in the next full-corpus Stage 2
  task).
- `results_detailed.md` — flag counts by category, distribution summaries, systematic patterns.
- `results/images/peak_distribution.png`
- `results/images/lufs_distribution.png`
- `results/images/duration_distribution.png`
- `results/images/phoneme_distribution.png`

## Dependencies

- `t0009_stage2_training_failure_forensics` — confirmed v5 train manifest path and flagged the
  missing audio stats gap.
- Independent of t0010; can run in parallel.
