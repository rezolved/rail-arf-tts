# v5 Corpus LUFS Normalization and Clipped-Fraction Re-Audit

## Motivation

t0011 audited all 1557 v5 train clips and found the corpus is largely clean, but its clipping
heuristic (`peak_dbfs > -0.1 dBFS`) flagged 224 clips (14.4%) purely because ElevenLabs peak-
normalizes its output to 0 dBFS by design — not because the audio is actually clipped. t0011's
creative-thinking step confirmed this and recorded two follow-up suggestions:

* **S-0011-01** (high priority): LUFS-normalize the full corpus to -14 LUFS (EBU R128) to recover
  the 224 clips and give the whole corpus consistent loudness.
* **S-0011-03** (medium priority): replace the raw peak-dBFS clipping check with a clipped-fraction
  filter (fraction of samples within 1 LSB of full scale, threshold > 0.1%) so the audit itself
  stops mistaking intentional peak normalization for clipping.

This task implements both together rather than either alone: the corrected audit (S-0011-03) is
what makes it safe to trust that the corpus has no *real* clipping before normalizing gain up
(S-0011-01), and the post-normalization audio must be re-checked with the same corrected clipping
metric in case raising the gain pushes any clip into genuine clipping for the first time. The
result is a single clean, consistently-loud manifest ready for the eventual full-corpus Stage 2 run
(t0010's follow-on, tracked separately as S-0011-02) — this task produces the data, it does not
train.

This task runs no training and needs no GPU, so it can run fully in parallel with t0010.

## Key Questions

1. How many v5 clips are genuinely clipped once measured by `clipped_fraction` (samples within 1
   LSB of full scale) rather than raw `peak_dbfs`? (t0011's creative-thinking step estimated 0-10.)
2. After LUFS-normalizing every non-genuinely-clipped clip to -14 LUFS, does the gain change
   introduce *new* clipping in any clip that passed the corrected check pre-normalization?
3. What is the final clean-manifest size, and how does it compare to t0011's 1311/1557 (84.2%)
   baseline and the two suggestions' individual estimates (~1532 for S-0011-03 alone, ~1556 for
   S-0011-01 alone)?
4. Does loudness normalization change the LUFS/duration/silence distributions in ways that could
   affect training stability (e.g. does it interact with the short-duration or excessive-silence
   flags from t0011)?

## Scope

### 1. Reuse t0011's data and code

`dvc pull` the same v5 train (1557 clips) and v4 val (96 clips) paths t0011 used. Read
`tasks/t0011_v5_data_quality_audit/data/per_clip_stats.jsonl` and
`tasks/t0011_v5_data_quality_audit/code/audit_audio.py` as the starting point — do not modify
t0011's files (completed task folders are immutable); copy what's needed into this task's `code/`.

### 2. Corrected clipping metric (S-0011-03)

Add a `clipped_fraction` metric per clip: the fraction of samples within 1 LSB of full scale.
Replace the `peak_dbfs > -0.1 dBFS` clipping flag with `clipped_fraction > 0.1%`. Keep t0011's other
thresholds unchanged (duration < 1.5s, silence > 30%, LUFS outside [-30, -6], sample rate ≠ 24kHz,
non-mono). Report how many of the original 224 peak-flagged clips are reclassified as not-clipped,
and how many (if any) genuinely clipped clips remain.

### 3. LUFS normalization (S-0011-01)

For every clip that passes the corrected audit (i.e. not genuinely clipped, not too short, not too
silent, not an LUFS/rate/channel outlier), normalize loudness to -14 LUFS (EBU R128) with
`pyloudnorm`, preserving sample rate (24 kHz) and mono channel layout. Write normalized audio to
`data/v5_normalized/`.

### 4. Post-normalization re-check

Re-run the clipped-fraction and silence checks on the *normalized* audio only (LUFS is now fixed by
construction; duration/sample-rate/channels are unaffected by gain). Move any clip that becomes
genuinely clipped after the gain change to the flagged list instead of the clean manifest.

### 5. Manifest and DVC tracking

Write:
- `data/per_clip_stats_v2.jsonl` — per-clip stats including `clipped_fraction`, pre- and
  post-normalization peak/LUFS.
- `data/flagged_clips_v2.txt` — clips still excluded after the corrected audit + normalization.
- `data/train_list_v5_normalized_clean.txt` — final clean manifest, paths pointing into
  `data/v5_normalized/`.

`dvc add data/v5_normalized/` and `dvc push` before marking this task complete, per repo DVC rules
— normalized audio must not be committed to git directly.

### 6. Comparison and distributions

Report flag counts by category, before (t0011) vs after (this task), and the final clean-manifest
size as a fraction of 1557. Produce histograms (peak dBFS, LUFS, clipped_fraction) before vs after
normalization, saved to `results/images/`.

## Compute and Budget

CPU-only: `pyloudnorm` + `soundfile` on 1557 clips, read + write + re-check. Expect a few minutes,
well under the 30 min t0011 needed for its lighter read-only pass.

Planned total: $0 (local compute only). Write `results/costs.json` with `total_cost_usd: 0`.

## Expected Outputs

- `data/per_clip_stats_v2.jsonl`
- `data/v5_normalized/` — DVC-tracked normalized clips
- `data/flagged_clips_v2.txt`
- `data/train_list_v5_normalized_clean.txt` — manifest for the next full-corpus Stage 2 task
- `results_detailed.md` — corrected-audit reclassification counts, before/after distributions,
  final clean-manifest size
- `results/images/peak_before_after.png`
- `results/images/lufs_before_after.png`
- `results/images/clipped_fraction_distribution.png`

## Dependencies

- `t0011_v5_data_quality_audit` — source of the audio stats, thresholds, and the two suggestions
  (S-0011-01, S-0011-03) this task implements together.
- Independent of `t0010_stage2_safeguarded_training` — CPU-only, no GPU pool contention; can run
  while t0010 is still in progress. The full-corpus training run that consumes this task's manifest
  (S-0011-02) is separate and should wait for t0010 to validate the training safeguards first.
