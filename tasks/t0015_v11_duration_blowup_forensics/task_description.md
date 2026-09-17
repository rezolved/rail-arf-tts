# v11 Duration-Blowup Forensics and Audible-Speech Gate Hardening

## Motivation

t0014 fixed t0013's diagnosed decoder-init bug and reported the mandatory audible-speech gate
(`code/audio_quality_check.py`'s `is_likely_noise`) as **PASSED** for `kokoro-v11-best`
(`clip_fraction=0.0048`, `spectral_flatness=0.0058`) — a real, correctly-measured improvement over
v10's confirmed clipping failure (0.750-0.807). t0014 also disclosed a "duration anomaly" as a
known limitation: the gate-passing synthesis output was 73.95 seconds for a ~10-word test sentence
(vs. 2.5-4.9s for the control and both v10 checkpoints), and filed it as a follow-up
(S-0014-01) rather than a blocker, since it doesn't affect the pre-registered
`clip_fraction`/`spectral_flatness` criteria.

**The user listened to `results/audio_samples/ft/v11_best.wav` and reports it is trash noise, not
speech.** Claude independently ran per-second waveform analysis on the file and confirmed a pattern
the gate cannot see:

```
sec   0: rms=0.7566 peak=1.0000 flatness=0.0171 clip=0.0786
sec   3: rms=0.2910 peak=0.9610 flatness=0.0070 clip=0.0000
sec   6: rms=0.2979 peak=0.9779 flatness=0.0085 clip=0.0000
...  (RMS stays in 0.25-0.76, flatness stays in 0.004-0.02, clip stays ~0.00, for the ENTIRE 73.9s)
sec  72: rms=0.5404 peak=1.0000 flatness=0.0208 clip=0.0297
```

Every 3-second window across the full clip has speech-like *local* spectral texture (low flatness,
formant-like structure) and low clipping — exactly what the gate checks for, and exactly why it
passed. But real speech for a short sentence has **silence gaps between words and phrases**; this
signal never drops in level for 74 continuous seconds. That is a third, distinct failure mode
neither `spectral_flatness` (tuned to catch literal white noise) nor `clip_fraction` (tuned to
catch v10's rail-to-rail saturation) was ever designed to catch: locally speech-shaped audio with
no linguistic structure, produced by a synthesis pipeline that never stops. This is what a human
hears as droning babble/"trash noise."

**This is the second time the automated gate has needed a blind spot found by a human listening
first** (the first being the entire premise of t0013). This task both fixes the specific v11 defect
and hardens the gate so a third failure mode doesn't need a third human catch.

## Key Questions

1. Where does the blowup actually originate? `infer_styletts2.py`'s `synthesize()` computes
   `pred_dur` from `model.predictor.duration_proj` output, sums it into `pred_aln_trg`, and uses
   that alignment length to drive the decoder. Is `pred_dur` itself absurdly large (a predictor
   calibration problem), or is the bug downstream in how the alignment matrix or frame count is
   constructed (a plumbing bug that would inflate duration regardless of a reasonable `pred_dur`)?
   Log `pred_dur`'s raw values and `pred_aln_trg`'s resulting frame count directly — don't infer
   this from audio length alone.
2. Is this specific to t0014's decoder-init fix (which repointed `first_stage_path` at
   `yl4579/StyleTTS2-LibriTTS`'s `epochs_2nd_00020.pth`)? That pretrained checkpoint's `predictor`
   and `predictor_encoder` weights may be scale-mismatched with David's voice/speaking rate if they
   were loaded but never meaningfully fine-tuned during t0014's 50 epochs (check t0014's
   `ignore_modules` list and Stage 2 training logs for whether `predictor`/`predictor_encoder`
   actually received gradient updates, or effectively rode along at their LibriTTS-scale
   initialization).
3. Does every synthesized text blow up, or only some? t0014's gate only tested a small, specific
   set of reference clips. Run the harness across a wider, varied sample of texts (short and
   longer, from `tasks/t0012_v5_corpus_normalize_and_reaudit/data/train_list_v5_normalized_clean.txt`
   or the filler-phrase prompt set) and characterize whether the blowup is universal, text-length
   dependent, or intermittent.
4. Can this be fixed cheaply, at inference time, with no retraining? Before touching any training
   code, sweep `infer_styletts2.py`'s `alpha`/`beta`/`diffusion_steps`/`embedding_scale` parameters
   (currently `0.3`/`0.7`/`5`/`1.0`, straight from the LibriTTS demo defaults — never tuned for this
   fine-tune) and see whether a different diffusion-sampler configuration produces well-calibrated
   durations. This is the ladder's cheapest rung; only escalate to a training-side fix if it fails.
5. If the inference-side sweep fails to fix it: is targeted fine-tuning of just `predictor`/
   `predictor_encoder` (holding the now-working decoder fixed) a smaller, cheaper remediation than
   a full retrain? Scope this as a recommendation for a follow-up task rather than doing it here —
   this task is forensics + cheap fixes + gate-hardening, not a second GPU training run.
6. What is the cheapest reliable way to make the audible-speech gate catch this class of defect
   going forward? At minimum: a duration-sanity check (synthesized duration vs. a naive
   words-per-second estimate for the input text — e.g. flag anything more than ~3x a reasonable
   upper bound) and a "has silence gaps" check (max contiguous non-silent run, not just overall
   `silence_fraction`) are cheap, fast, and would have caught this immediately. An ASR-based
   intelligibility check (transcribe the output, compare tokens against the input text) is a
   stronger but heavier option worth evaluating — check whether `faster-whisper` (already a
   dependency of `tasks/t0008_tts_eval_harness_baselines`'s environment) is practical to reuse here
   rather than re-deriving a transcription pipeline from scratch.

## Scope

### 1. Reproduce and characterize

Re-run `t0014/code/infer_styletts2.py` (or t0013's, whichever the reusable recipe now is) against
`kokoro-v11-best` across a varied sample of texts (at least 5-10, short and long). For each, log:
input text, token/phoneme count, raw `pred_dur` values, resulting frame count, output audio
duration, and the existing `audio_quality_check.py` metrics. Confirm whether the 73.9s result was
a one-off or the general behavior.

### 2. Localize the bug

Using the logged `pred_dur`/frame-count data from step 1, determine whether the defect is in the
duration prediction itself or in the alignment-matrix/frame-count plumbing downstream of it. This
is a fast, cheap diagnostic step (no GPU) — do this before any inference-parameter sweep.

### 3. Try the cheap fix first

Sweep `alpha`/`beta`/`diffusion_steps`/`embedding_scale` (per Key Question 4) on a fixed short text
and check whether any combination produces a duration in the expected range with an audible-speech
gate pass AND a passing duration-sanity check (step 5). No GPU required for this step.

### 4. If the cheap fix works

Re-synthesize the same texts used in t0014's gate (`lining_up_suggestions_17`,
`lining_up_suggestions_10`, `putting_them_head_to_head_15`) with the corrected inference parameters,
produce fresh paired original-vs-FT audio samples, and document the fix (which parameter(s),
what values, why) in `code/infer_styletts2.py`'s defaults going forward so nobody re-derives this.

### 5. If the cheap fix does not work

Document the negative result with the same rigor as a positive one: which parameter combinations
were tried, what each produced (duration, gate metrics), and why none resolved it. Recommend
targeted `predictor`/`predictor_encoder` fine-tuning (or a full retrain if that's not separable) as
a new follow-up task — do not attempt a GPU training run inside this task.

### 6. Harden the audible-speech gate

Extend `audio_quality_check.py` (or a new sibling function, kept in the same module for one source
of truth per S-0013-04's existing recommendation) with:

* A duration-sanity signal: flag when synthesized duration exceeds a generous multiple of a naive
  words-per-second estimate for the input text.
* A "longest contiguous non-silent run" signal, distinct from aggregate `silence_fraction` — this
  is exactly the signal that would have caught v11's blowup and didn't exist before.
* Evaluate (don't necessarily implement, if it's a bigger lift than this task's forensics scope)
  whether an ASR-round-trip check is worth adding as a stronger, optional third layer.

Re-run this hardened gate against v10 (should still fail, sanity check), v11 as shipped (should now
fail on the new duration/silence-gap signal even though it passes the old one), and whatever
corrected output step 4 produced (should pass all signals) — a clean three-way regression check
proving the hardened gate actually discriminates correctly.

## Expected Outputs

- `code/` — extended `audio_quality_check.py` (or equivalent) with the new duration-sanity and
  silence-gap signals, instrumented `pred_dur`/frame-count logging added to the inference recipe.
- `results/duration_blowup_diagnosis.md` — root cause (predictor calibration vs. plumbing bug),
  whether it's universal or text-dependent, and whether the cheap inference-parameter fix worked.
- `results/results_summary.md` / `results_detailed.md` — outcome and recommendation.
- **Only if the cheap fix in step 3-4 works**: `results/audio_samples/{original,ft}/` with the
  corrected output, and the updated inference recipe with corrected default parameters — the same
  "audio samples + working recipe" deliverable the user has required of every task in this chain.
- **If it does not work**: an honest negative result and a filed follow-up suggestion recommending
  targeted fine-tuning or retraining, matching t0013's and t0014's own standard of disclosing
  failure rather than shipping a false success.
- `results/suggestions.json` — at minimum, the gate-hardening changes made here should be proposed
  for adoption as the project's standard pre-completion check for all future TTS training tasks
  (extends S-0013-02, which only wired in the original, now-shown-insufficient, two-signal gate).

## Dependencies

- `t0013_v10_synthesis_quality_forensics` — source of `audio_quality_check.py` and
  `infer_styletts2.py`, both extended here.
- `t0014_v11_decoder_fix_retrain` — source of `kokoro-v11-best`, the checkpoint and disclosed
  duration anomaly under investigation, and the specific gate-passing texts this task re-tests
  against for direct comparability.
