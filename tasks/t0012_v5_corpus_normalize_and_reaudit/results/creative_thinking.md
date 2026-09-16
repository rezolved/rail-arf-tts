---
spec_version: "1"
task_id: "t0012_v5_corpus_normalize_and_reaudit"
---
# Creative Thinking — t0012 v5 Corpus LUFS Normalization and Clipped-Fraction Re-Audit

This step goes beyond the plan's four pre-registered Key Questions (all answered cleanly in
`data/analysis_v2.json`: 0 genuine clipping, 0 new clipping, 1531/1557 clean, gain-invariant
duration/silence cross-tab). The analysis below re-derives a finding the implementation's own code
comments already anticipated but never quantified, flags a representation risk in the 26 excluded
clips that matters specifically for this project's voice-commerce use case, and lists alternative
normalization designs that were not taken and why.

## 1. The headline finding: "-14 LUFS normalization" is one-directional for this corpus

`data/analysis_v2.json`'s `post_lufs` distribution looks, at first glance, like a success: std drops
from 2.34 (pre) to 1.58 (post), and the median lands almost exactly on target (-14.89 vs -14.0
target). But the shape is not a tight cluster around -14 — it is a hard ceiling at -14.0 (p50, p95,
p99, and max are *all* exactly -14.0) with a long left tail down to -21.58 LUFS. That shape is the
signature of a limiter, not a two-sided normalizer, and a direct query of
`data/per_clip_stats_v2.jsonl` confirms why:

* 852/1531 normalized clips (55.6%) have `pre_lufs < -14.0` and therefore need upward gain to reach
  target.
* Of those 852, **850 (99.8%) are ceiling-capped** — their `post_lufs` still falls short of -14.0 —
  because `PEAK_CEILING_DBFS = -1.0` caps the gain before it reaches the LUFS target.
* The five worst-case clips (e.g. `let_me_clarify_the_development_basis_6125a9.wav`, pre_lufs
  -21.70, pre_peak -1.11 dBFS) move by under 0.2 LUFS after "normalization" — the ceiling permits
  almost no boost at all.
* Corpus-wide `pre_peak_dbfs` is tightly clustered at -0.55 ± 0.72 dBFS regardless of `pre_lufs`,
  which spans -26.2 to -8.6 LUFS (a 17.6 LU range). This is the root cause: ElevenLabs peak-
  normalizes every clip's *sample peak* independently of its *integrated loudness*, so peak and
  loudness are almost decoupled in this corpus. A quiet, plosive-heavy technical phrase ("verifying
  the current CEO...", "checking the headquarters location...") already sits at -1.0 to -1.1 dBFS
  peak despite being 7-10 LU quieter than target — there is no more peak headroom to spend on a
  loudness boost.

This is the same root cause the implementation step already hit once, from the other direction: the
plan's original unbounded gain formula caused 408 new-clipping cases at full-corpus scale precisely
because so many quiet clips already sit near 0 dBFS peak. The fix (the -1 dBFS ceiling) correctly
neutralizes the clipping risk, but as a side effect it also neutralizes most of the *upward*
loudness correction the task set out to apply. `code/constants.py`'s comment on `PEAK_CEILING_DBFS`
already predicted this in the abstract ("a clip whose LUFS gain would exceed the ceiling lands
quieter than -14 LUFS rather than being flagged and dropped") — this step is the first place that
result is measured: not a handful of edge cases, but the majority (55.6%) of clips that needed a
boost at all.

**Practical read for this task**: this is not a bug and does not change any REQ-4/REQ-12/REQ-14/
REQ-15 answer — genuine and post-normalization clipping are both still 0, which was the hard
constraint. It does mean "LUFS-normalize to -14" was delivered as "cap loud clips at -14 LUFS,
gently nudge quiet clips, leave already-quiet-and-peaky clips essentially as they were." The
resulting corpus is still measurably more uniform than the source (std 2.34 → 1.58, a 32% reduction)
but is not the tight ±1 LU band the task description's phrasing might suggest.

## 2. Alternative designs not taken, and why the chosen one is still the right lazy call

* **Multiband/soft-knee limiter instead of a hard gain cap.** Would recover a little more headroom
  for the ceiling-capped clips by allowing brief, sub-perceptual inter-sample excursions. Rejected:
  adds a dependency and tuning surface for a corpus where the recoverable gain is at most ~1 dB (the
  ceiling only blocks the last dB or so before 0 dBFS) — not worth the complexity for clips that are
  already 7+ LU short of target.
* **Dynamic-range compression before the LUFS gain stage**, to shrink crest factor and decouple peak
  from loudness so more gain fits under the ceiling. Rejected on the same grounds t0011's
  creative-thinking already used to prefer a flat linear gain over anything more aggressive: a
  compressor changes the spectral/dynamic envelope the GE2E speaker-similarity metric and the mel
  loss are sensitive to, where a uniform per-sample scalar gain provably does not (t0011 §3).
  Trading a clean, auditable gain-only pipeline for a few recovered LU on <1% of clips is not a good
  trade for a corpus this small.
* **Per-clip RMS-based gain instead of LUFS-based gain.** Would not help — RMS and LUFS are both
  loudness proxies and both are blocked by the same peak ceiling; the bottleneck is peak headroom,
  not the loudness metric chosen.
* **True inter-sample ("true") peak measurement via 4x oversampling (ITU-R BS.1770 true-peak),
  instead of the flat per-sample peak actually used** (`audit_normalize.py` computes
  `peak_abs = max(abs(samples))`, no oversampling). Checkpoint.md's step-9 summary calls this a
  "true-peak limiter," which is a slight misnomer — it is a sample-peak limiter with a 1 dB margin.
  For 24 kHz speech content a 1 dB sample-peak margin is generous compared to typical inter-sample
  overshoot (usually a few tenths of a dB) and matches common broadcast practice (EBU R128
  recommends -1 dBTP), so this is very unlikely to matter in practice — flagged here only as a
  terminology correction, not a defect requiring rework.
* **Two-pass normalize-then-reflag instead of the single decode-normalize-recheck pass actually
  used.** Not applicable here — REQ-6 already required exactly the re-check-in-one-pass design that
  was implemented; a second disk read would only add I/O cost for the same result.

## 3. A risk the pre-registered questions didn't ask about: what exactly is in the excluded 26

All 26 remaining exclusions are `duration_low` (25) or `silence` (1) — zero clipping-related
exclusions survive. Inspecting the 25 `duration_low` filenames and durations
(`data/flagged_clips_v2.txt` cross-referenced against `data/per_clip_stats_v2.jsonl`) shows they are
not truncated or corrupted audio — they are short, clean, single-phrase voice-commerce fillers:
`sure_thing_01c5a2.wav` (0.65 s), `got_it_06c73d.wav` (0.74 s), `of_course_fbdfee.wav` (0.88 s),
`right_away_8beee1.wav` (0.93 s), `no_problem_b3095d.wav` (0.93 s), several `hello_how_can_i_help_*`
and `sure_thing*` variants (1.1-1.49 s) — exactly the category of utterance this whole project
exists to synthesize (`project/description.md`: "voice commerce filler synthesis pipeline").

Quantitatively this is a rounding error: 25/1557 = 1.6% of the corpus, and t0011's own
creative-thinking already argued (§2) that legitimate short single-word/short-phrase fillers are the
majority case among short clips, with the 1.5 s floor mainly a safety net against silence-
padding/truncation artifacts (which the silence-fraction and duration checks already catch
directly). Nothing here overturns that reasoning or this task's own scope, which explicitly keeps
`DURATION_MIN_S` unchanged (REQ-3 only touches the clipping flag). But it is worth naming plainly
for whoever consumes `data/train_list_v5_normalized_clean.txt` next: the exclusion set is not a
random 1.6% sample of the corpus — it is concentrated in exactly the shortest, highest-frequency
filler phrases (the ones most likely to be *repeated verbatim* at inference time), so the clean
manifest slightly under-represents the sub-1.5s end of the filler-length spectrum relative to the
full corpus. Given the 1.5 s floor was inherited unchanged from t0011 rather than re-examined here,
a future task revisiting Strategy C from t0011's creative-thinking (a duration-aware floor scaled to
transcript length, e.g. `duration_s > k * phoneme_len`, rather than one fixed cutoff) would recover
most of these 25 clips without reopening the truncation risk the fixed floor exists to guard
against.

## 4. Sanity check: does the genuine-clipping count validate the corrected metric?

t0011's own creative-thinking estimated 0-10 residual clipped clips under `clipped_fraction > 0.001`
(its "Strategy A"), and a ~1532-clip clean manifest. This task measured 0 genuinely clipped clips
and a 1531-clip clean manifest — inside t0011's predicted range and within 1 clip of its point
estimate. That the plan's own literature (i.e., the prior task's worked-out analysis) predicted this
result almost exactly is a useful cross-check that the `clipped_fraction` metric is doing what it
was designed to do, not an artifact of a bug that happens to look plausible. The convergence with
t0011's independent Strategy-A estimate (computed before any of this task's code existed) is
stronger evidence for correctness than internal consistency checks alone would be.

## 5. Risks and insights for downstream Stage 2 training

* **No blocking risk.** 0 genuine clipping, 0 new clipping, 98.3% clean-manifest coverage (vs 84.2%
  baseline) is an unambiguous improvement with no discovered regression. Stage 2 can safely consume
  `data/train_list_v5_normalized_clean.txt`.
* **Residual loudness variance is real but smaller than pre-normalization, not eliminated.** Post-
  normalization LUFS std is 1.58 (vs 2.34 pre, vs a hypothetical 0 for a perfect normalizer). Per
  t0011 §3's own risk analysis (GE2E is amplitude-invariant; mel L1/L2 loss is only mildly amplitude
  -dependent), this is unlikely to materially affect speaker-similarity training, but if Stage 2
  training curves show unexplained instability, this residual variance — concentrated in quiet,
  peaky technical phrases — is now a concrete, quantified, pre-identified suspect rather than an
  unknown.
* **Filler-phrase under-representation is a training-data-composition risk, not an audio-quality
  risk.** If Stage 2 (or a later fine-tune) shows weaker quality specifically on short, high-
  frequency filler phrases ("got it", "sure thing", "of course"), the 25 excluded sub-1.5s clips are
  a specific, named, revisitable cause — see §3 for a concrete unblock (duration-aware floor).
* **Terminology note for `results_detailed.md` / any future reference to this task**: describe the
  -1 dBFS cap as a sample-peak ceiling, not a "true-peak limiter," to avoid overstating the
  inter-sample-overshoot guarantee it provides (see §2).
