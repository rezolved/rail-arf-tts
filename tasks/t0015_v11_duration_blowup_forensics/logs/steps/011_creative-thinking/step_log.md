---
spec_version: "3"
task_id: "t0015_v11_duration_blowup_forensics"
step_number: 11
step_name: "creative-thinking"
status: "completed"
started_at: "2026-09-17T10:33:40Z"
completed_at: "2026-09-17T10:55:00Z"
---
## Summary

Read Step 9's forensic outputs closely and surfaced five alternative angles not pursued by the
scripted plan: a concrete cheap ablation that could directly test (not just infer) the
`predictor_encoder` deepcopy-init hypothesis; a previously-unremarked numeric coincidence between
the `dur_loss` plateau ratio and the observed audio-blowup ratio that argues for double-checking
David's ground-truth duration targets before committing GPU budget to a fix; a real
phoneme-density-vs-blowup-severity correlation visible across the smoke test and the 10
characterization records that the word-count-only analysis in `results/duration_blowup_diagnosis.md`
Section 2 did not examine; a specific asymmetry in the new `duration_sanity_pass` signal
(upper-bound only) that leaves under-synthesis/truncation failures uncaught by both the old and
hardened gate; and a concrete list of failure classes where ASR-round-trip and duration/silence-gap
checks are non-overlapping, complementary tools rather than redundant ones.

## Actions Taken

1. Re-read `results/duration_blowup_diagnosis.md`, `results/predictor_tensor_forensics.md`,
   `results/param_sweep.json`, `results/gate_regression.json`, and
   `results/asr_roundtrip_evaluation.md` in full to ground every point below in this task's own
   instrumented measurements, not restated hypotheses.
2. Cross-referenced `results/audio_samples/ft/smoke_test.timing.json` (Step 4's smoke test) against
   `results/duration_characterization.json`'s 10 records and `results/param_sweep.json`'s
   `baseline_default` row, since all three used the identical
   `alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1.0` inference parameters on the same
   `kokoro-v11-best` checkpoint -- making them a clean, confound-free comparison set for isolating a
   text-content effect from a parameter effect.
3. Computed `tokens_per_word` (phonemizer output density) for all 10 characterization records plus
   the smoke test and compared it against each record's `duration_ratio`, since Section 2 of the
   diagnosis only tested word-count and raw token-count for a length correlation, not phoneme
   density as an independent covariate.
4. Worked through what a cheap, forward-pass-only (no training) cross-checkpoint module-swap
   ablation would need to directly test the `predictor_encoder` deepcopy-init hypothesis, since
   Section 1d of the diagnosis explicitly states this task's tensor forensics narrows but does not
   isolate the contributor.
5. Re-derived the numeric relationship between the `dur_loss` plateau (0.53-0.62) and the t0009
   reference convergence value (0.034) and compared that ratio band against the observed
   `duration_ratio` band (8.6x-17.9x) to check whether it points at an alternative or complementary
   root-cause hypothesis about the training data/target pipeline, not just module initialization.
6. Worked through which failure classes the ASR-round-trip check
   (`results/asr_roundtrip_evaluation.md`) would catch that the hardened gate's
   `duration_sanity_pass`/`longest_nonsilent_run_s` signals would not, and vice versa, including a
   direction-of-check asymmetry not previously flagged.

## Outputs

* This step log (`logs/steps/011_creative-thinking/step_log.md`), which is the sole output of this
  step per `arf/skills/execute-task/SKILL.md`'s Phase 5 dispatch table (no dedicated skill/subagent
  is listed for `creative-thinking`; it is an out-of-the-box analysis step, not an asset-producing
  one).
* No files under `results/`, `code/`, or `plan/` were modified -- this step is read-and-think only,
  per the task instructions (no new experiments, no code changes, no GPU/remote work).

### Finding 1 -- A cheap, forward-pass-only ablation to directly test the `predictor_encoder` hypothesis

Section 1d of `results/duration_blowup_diagnosis.md` explicitly says this task's no-GPU scope
"cannot fully isolate the contribution of `predictor_encoder` vs. the diffusion-sampled style vector
vs. David's speaking-rate distribution" and recommends a full GPU retrain as the follow-up. There is
a cheaper intermediate experiment that stays inside a no-GPU, no-training budget and would sharpen
that isolation before committing to a multi-epoch retrain:

* Load `kokoro-v11-best` (`epoch_00048.pth`) and the LibriTTS control checkpoint
  (`epochs_2nd_00020.pth`) via the same raw `torch.load` pattern
  `code/predictor_tensor_forensics.py` already uses.
* Construct a hybrid `state_dict`: v11's `predictor` (including `duration_proj`) combined with the
  **control's** `predictor_encoder` weights (i.e., undo the `copy.deepcopy(style_encoder)`
  contribution by substituting a version of `predictor_encoder` that was never deepcopy-initialized
  from a mismatched module).
* Run `code/infer_styletts2.py`'s already-instrumented `synthesize()` in forward-pass-only mode
  (identical to the smoke test invocation) on the fixed gate text, and log
  `pred_dur`/`pred_dur_sum`.
* If `pred_dur_sum` drops back toward the control's healthy range (~150-200 for the gate text, per
  Section 1c's LibriTTS control run: `pred_dur_sum=186`), that is direct, causal evidence
  `predictor_encoder`'s weights are the dominant contributor -- not merely a correlated shift. If it
  stays elevated, the diffusion-sampled style vector or `bert_encoder`/`text_encoder` output is
  implicated instead, redirecting the follow-up task's fine-tuning scope before any GPU hours are
  spent. This costs roughly one extra CPU inference call (a few seconds, per the existing
  `wall_time_seconds` figures in `smoke_test.timing.json` and `param_sweep.json`) and no training --
  it was in scope for this task's no-GPU budget but was not part of the pre-registered plan, so it
  is flagged here as a recommended first step of the follow-up task rather than retroactively added
  to this one.

### Finding 2 -- The `dur_loss` plateau ratio closely tracks the audio-blowup ratio; check training targets too

Section 1e reports `dur_loss` plateaued at 0.5289-0.6205 across all 50 v11 training epochs, against
a 0.034 reference from t0009. That ratio band is **0.53/0.034 ≈ 15.6x to 0.62/0.034 ≈ 18.2x** --
strikingly close to the **8.6x-17.9x** observed audio-duration-blowup range (Section 2's table, mean
14.79x). The diagnosis's own narrative treats these as two independently corroborating signals of
the same predictor-calibration failure, which is a reasonable reading. But the near-numeric-match
also supports an alternative (or complementary) hypothesis worth ruling out explicitly before a
follow-up task spends GPU budget on `predictor`/`predictor_encoder` fine-tuning alone: if the
ground-truth per-token alignment durations used to compute `dur_loss` for David's corpus were
themselves extracted at a different frame rate/hop-length than the 0.034-reference run assumed (a
forced-alignment or corpus-prep frame-rate mismatch upstream in David's training-data pipeline,
outside this task's `arf/`-adjacent scope), gradient descent could plateau at a loss floor
**proportional to that fixed target-scale error** rather than a floor caused purely by a
hard-to-optimize initialization. This is not contradicted by Section 1d's finding that
`duration_proj`'s own weights barely moved (+0.7%) -- a systematically-scaled target would still
require the *input* distribution (not necessarily `duration_proj`'s weights) to shift to compensate,
which is consistent with what was found. This is flagged as a real alternative worth a 10-minute
check (compare the hop-length/frame-rate assumed by whatever alignment-duration extraction script
produced David's Stage-2 training targets against `config_david_v11.yml`'s
`preprocess_params.spect_params` / hop_length) before the follow-up task assumes the fix is purely
`predictor_encoder` re-initialization -- if the targets themselves are off-scale, re-initializing
`predictor_encoder` alone would not fix it.

### Finding 3 -- Phoneme density (tokens-per-word), not word count, may be the real covariate

Section 2 concludes the blowup is universal with "no meaningful correlation to text length,"
checking word count and (separately) raw token count. It does not examine `tokens_per_word`
(phonemizer output density) as its own variable jointly with `duration_ratio`. Computing it from
`results/duration_characterization.json` plus `results/audio_samples/ft/smoke_test.timing.json`
(same checkpoint, same `alpha/beta/diffusion_steps/embedding_scale` as `param_sweep.json`'s
`baseline_default` row, so these three sources are directly comparable with no parameter confound):

| Source | Text | tokens/word | `duration_ratio` (word_count/2.5 basis) |
| --- | --- | ---: | ---: |
| smoke test | "This is a test." | 4.25 | 5.26 |
| char idx7 | "let me look into how rezolve transforms..." | 6.9 | 8.61 |
| char idx2 | "let me think 04" | 7.5 | 13.12 |
| char idx0 | "pulling that up 03" | 7.25 | 16.42 |
| char idx3 | "cross-checking that 00" | 11.0 | 17.87 |

The two lowest-density texts (smoke test at 4.25 tokens/word, idx7 at 6.9) also produced the two
lowest `duration_ratio` values across all 11 data points (5.26 and 8.61 respectively), and the
highest-density text (idx3 at 11.0) produced the single highest ratio (17.87). This is not a clean
monotonic trend across all 11 points (e.g., idx4 at 10.5 tokens/word only reaches 13.43, below
idx6's 8.56-density/17.83-ratio point) and n=11 informal eyeballing is not a statistical proof, but
it is a visible pattern the original word-count-bucket comparison ("short" 2-4 words vs. "long" 7-10
words) would not surface, since word count and phoneme density are not the same axis --
"double-checking 12" is short by word count (2 words) but dense by phoneme count (10.5 tokens/word).
This matters concretely for the production use case: Rezolve's voice-commerce filler synthesis will
routinely include digits, SKUs, prices, and multi-syllable brand/product terms, which are exactly
the kind of content that inflates tokens-per-word. A follow-up characterization run that holds word
count roughly fixed and deliberately varies phoneme density (e.g., "buy it now" vs. "expedite the
procurement") would give a cleaner test of this hypothesis than this task's original 10-text sample,
which varied length and content simultaneously.

### Finding 4 -- `duration_sanity_pass` is upper-bound-only; under-synthesis would slip past both gates

`results/duration_blowup_diagnosis.md` Section 4 defines `duration_sanity_pass` as
`actual_duration_s <= estimate_naive_duration_bound_s(text) * 3.0` -- an upper bound only. This is
the right design for the specific failure this task investigates (over-long babble), and the
three-way regression proves it closes that blind spot. But it means a *different* regression --
audio that is suspiciously short (e.g., a truncated/cut-off synthesis, or a predictor miscalibrated
in the opposite direction and clamping too low) -- would pass `duration_sanity_pass`
unconditionally, and would very plausibly also pass `longest_nonsilent_run_s <= 12.0` and the
original 3-signal `is_likely_noise` check, since a truncated clip is not clipped, not flat-spectrum,
and not silent for a long contiguous run. This is a distinct failure class from anything found in
this task (nothing in the evidence suggests v11 under-synthesizes), but it is a genuine,
currently-unguarded gap in the now-hardened gate worth naming explicitly for the `suggestions` step:
a symmetric lower bound (e.g., `actual_duration_s >= estimate_naive_duration_bound_s(text) * 0.3` or
similar) would close it cheaply, using the same already-computed `estimate_naive_duration_bound_s`
helper.

### Finding 5 -- ASR-round-trip and duration/silence-gap checks catch disjoint failure classes

`results/asr_roundtrip_evaluation.md` correctly concludes ASR-round-trip is the wrong tool for
*this* failure mode (its own `[0.5, 2.0]` duration pre-gate skips every clip this task's blowup
produces). Stated more generally for the `suggestions` step: the two check families catch disjoint
failure classes, so neither is a superset of the other:

* **Duration-sanity / longest-nonsilent-run catches**: wrong overall clip length or absence of
  expected silence structure (this task's exact failure -- droning babble with no word/phrase gaps)
  -- *without* needing any reference transcript or ASR model, and without caring whether the content
  is linguistically correct.
* **ASR-round-trip would catch, that duration/silence-gap cannot**: a *normal-duration*,
  properly-paced clip that says the wrong words entirely (substitution/mistranscription), mangles a
  brand name or SKU, or produces garbled-but-rhythmically-plausible speech (phoneme confusion that
  keeps the silence-gap structure intact but corrupts intelligibility). None of this task's evidence
  suggests `kokoro-v11-best` has this problem, but it is a real, distinct risk class for a voice
  commerce pipeline (a filler phrase that mispronounces "checkout" or a product name would pass
  every signal this task's hardened gate checks).
* Given Finding 4, a complete future third layer should combine an ASR-round-trip layer's
  content-correctness check with a **symmetric** (both-sided) duration gate rather than treating
  `duration_sanity_pass` as covering everything short of literal noise.

## Issues

No issues encountered. This was a read-only analysis step; no experiments were run, no code was
changed, and no remote/GPU resources were used, consistent with the task instructions for this step.
