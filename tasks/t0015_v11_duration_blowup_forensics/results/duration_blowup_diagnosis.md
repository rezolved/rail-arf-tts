# v11 Duration-Blowup Diagnosis (Milestone E Step 14, REQ-11)

Canonical root-cause diagnosis for `kokoro-v11-best`'s (t0014's `epoch_00048.pth`) 15-18x duration
blowup. Synthesizes Steps 4-13's own instrumented measurements -- every numeric claim below traces
to a specific file this task produced, not to research-phase hypotheses restated as conclusions.

## 1. Root cause

**Verdict: a predictor-calibration failure, upstream of `duration_proj`'s own weights -- NOT a
`pred_aln_trg`/decoder alignment-plumbing bug, and NOT `max_dur`-ceiling saturation.**

### 1a. `pred_dur` is directly measured, not inferred from audio length (REQ-1)

`code/infer_styletts2.py`'s `synthesize()` was instrumented (Step 4) to capture and return
`pred_dur` (per-token sigmoid-sum values), `pred_dur_sum`, and `input_token_count` directly from the
forward pass, before they are consumed by `pred_aln_trg` construction. The smoke-test verification
(`results/audio_samples/ft/smoke_test.timing.json`) confirmed non-null values on the first run:
`pred_dur_sum=337`, `input_token_count=17` for "This is a test." (8.42s output).

### 1b. `pred_dur` is elevated but NOT saturating at the `max_dur=50` ceiling

Across all 10 characterization texts (`results/duration_characterization.json`,
`results/localization_summary.json`), per-token `pred_dur` values range roughly 15-46 frames (mean
per-token values 19.8-36.2 across texts), with a maximum observed value of 46 (item 7) -- below
`config_david_v11.yml:79`'s `model_params.max_dur = 50` ceiling in every single record.
`localization_summary.json`'s aggregate: `mean_fraction_tokens_near_ceiling: 0.0044` (0.44% of all
tokens across all 10 texts are at or above 90% of the ceiling). **This rules out simple
ceiling-saturation as the mechanism** -- the predictor is not just clamping every token to its
maximum; it is systematically over-predicting duration across nearly all tokens, by roughly 2-4x
what natural English phoneme durations require (typical natural speech: ~5-15 frames/phoneme at
12.5ms/frame = 60-190ms; observed here: 20-36 frames/phoneme = 250-450ms).

### 1c. The frame-count-to-output-duration relationship is completely healthy -- NOT a plumbing bug

This is the most decisive finding of this task. Initial analysis of the 10 characterization records
found a striking pattern: `output_duration_s` is consistently **exactly 2.00x** the naive
`pred_dur_sum * hop_length / sample_rate` estimate, for every single record
(`localization_summary.json`: `mean_frame_vs_output_duration_ratio: 1.9998`). This looked, at first,
like a smoking-gun plumbing bug (a duplicated frame count somewhere in `pred_aln_trg` construction
or decoder upsampling).

**It is not a bug.** Two independent checks confirmed this 2x factor is the StyleTTS2-native
decoder's normal, architecture-intrinsic behavior, not a defect:

* **Source reading**: `code/kikiri-tts/StyleTTS2/Modules/hifigan.py` and `istftnet.py` both define
  an identical `Decoder` class whose `decode` `nn.ModuleList` has 4 `AdainResBlk1d` blocks -- the
  first 3 with `upsample="none"`, the last one with `upsample=True`
  (`self.decode.append(AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample=True))`). That last
  block's `UpSample1d` applies `F.interpolate(x, scale_factor=2, mode="nearest")` -- an intrinsic 2x
  time-domain upsample -- *before* the signal reaches the final `Generator`, whose own
  `upsample_rates` product (`10*5*3*2=300`) already equals `hop_length` exactly.
* **Empirical validation run**: the instrumented `synthesize()` was run against the LibriTTS control
  checkpoint (`epochs_2nd_00020.pth`, same decoder architecture,
  `model_params.decoder.type: hifigan` confirmed in its own downloaded `config.yml`) on the gate
  text ("This is a test of the Style T T S two inference harness."). Result: `pred_dur_sum=186`,
  `output_duration_s=4.65` -- `4.65 / (186*300/24000) = 2.00`, exactly matching every v11 record.
  This run and its numbers are reproducible via
  `code/infer_styletts2.py --checkpoint-path code/kikiri-tts/StyleTTS2/Models/LibriTTS/epochs_2nd_00020.pth`.

Since a *known-good, non-blown-up* checkpoint shows the identical 2.00x ratio, this ratio is proven
to be normal model behavior, not a defect. `localization_summary.json`'s
`any_plumbing_deviation_flagged: false` confirms no v11 record deviates from this healthy 2.00x
baseline by more than 5%. **This rules out a `pred_aln_trg`/alignment-matrix/decoder-upsampling
plumbing bug** -- the frame-to-sample relationship downstream of `pred_dur` behaves identically to
the working control.

### 1d. Tensor forensics: `duration_proj`'s own weights barely moved -- the calibration failure is upstream of it

`results/predictor_tensor_forensics.md` (Step 7) compares `kokoro-v11-best`'s `predictor` and
`predictor_encoder` weight norms against the LibriTTS control checkpoint's, via raw `torch.load` (no
inference). Key finding: **`predictor.duration_proj`'s weight norm shifted by only +0.7%**
(v11=83.4531 vs control=82.8593) -- a near-zero shift despite receiving gradients every step of all
50 training epochs (`train_second_v11.py:733-734`'s unconditional `optimizer.step("predictor")`
calls, confirmed in research). For a 25,650-parameter submodule trained for 50 epochs, this is not
consistent with "this exact layer's weights drifted to a badly calibrated region" -- if that were
the mechanism, 50 epochs of gradient updates would be expected to move the weight norm more than
0.7%.

`predictor_encoder`'s *whole-module* weight norm, by contrast, shifted by **-12.7%** (v11=400.6487
vs control=459.0197) -- a much larger, more plausible signal. This is consistent with
`predictor_encoder`'s known initialization path (research finding, corroborated by
`train_second_v11.py:329`): it is excluded from the `first_stage_path` checkpoint load
(`ignore_modules`) and instead initialized as `copy.deepcopy(model.style_encoder)` before Stage-2
training begins -- i.e., it starts from a *different pretrained module's* weights, not from a
duration-calibrated state, and trains from that mismatched starting point.

**Synthesis**: the elevated `pred_dur` values (Section 1b) are best explained not by
`duration_proj`'s own weights being individually miscalibrated, but by a distribution shift in
`duration_proj`'s *input* -- the output of `predictor.lstm`, itself fed by
`predictor.text_encoder(d_en, s, input_lengths, text_mask)`, where `s` comes from the
diffusion-sampled style vector (blended with `predictor_encoder`'s output via `alpha`/`beta`, see
Section 3) and `d_en` from `bert_encoder`. `predictor_encoder`'s substantial, independently-shifted
weights are the most likely single contributor to this upstream distribution shift, though this
task's no-GPU scope cannot fully isolate the contribution of `predictor_encoder` vs. the
diffusion-sampled style vector vs. David's speaking-rate distribution differing from LibriTTS's --
see Section 6 for the recommended follow-up that can.

### 1e. Corroborating (not conclusive on its own) training-log evidence

`tasks/t0014_v11_decoder_fix_retrain/data/run_v11/metrics.jsonl`'s per-epoch `dur_loss` (last-logged
value per epoch, independently re-verified from the raw file by this task) ranges 0.5289-0.6205
across all 50 training epochs (epoch 1 final: 0.6205; epoch 50: 0.5289; no improvement trend after
~epoch 8, per research), roughly 15-18x higher than the project's t0009 reference run's
0.034-by-epoch-6 convergence. `dur_loss` is an L1 loss, in frame units, between
`_dur_pred = sigmoid(duration_proj(x)).sum(axis=1)` (the exact same computation as inference-time
`pred_dur`) and the ground-truth per-token alignment duration (`train_second_v11.py:681-690`). A
plateaued, non-converging `dur_loss` at this scale is consistent with (not proof of, on its own) the
predictor never learning a well-calibrated duration mapping for David's voice/speaking rate -- this
task's own Section 1a-1d instrumentation is what confirms the mechanism, per the task's "confirm
with evidence, not assertion" mandate.

## 2. Universal or text-dependent (REQ-4)

**Verdict: universal. Every one of the 10 characterization texts blew up, with no meaningful
correlation to text length.**

| idx | category | words | tokens | `pred_dur_sum` | output (s) | `duration_ratio` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | short | 4 | 29 | 1051 | 26.27 | 16.42 |
| 1 | short | 4 | 32 | 917 | 22.92 | 14.33 |
| 2 | short | 4 | 30 | 840 | 21.00 | 13.12 |
| 3 | short | 3 | 33 | 858 | 21.45 | 17.87 |
| 4 | short | 2 | 21 | 430 | 10.75 | 13.43 |
| 5 | long | 8 | 74 | 1463 | 36.57 | 11.43 |
| 6 | long | 9 | 77 | 2568 | 64.20 | 17.83 |
| 7 | long | 10 | 69 | 1377 | 34.42 | 8.61 |
| 8 | long | 7 | 66 | 1997 | 49.92 | 17.83 |
| 9 | long | 8 | 62 | 2179 | 54.47 | 17.02 |

Source: `results/duration_characterization.json` (10/10 records, 0 errors -- the
`successful_runs / total_runs < 0.8` Rejection Criteria threshold was cleared trivially, 10/10 =
100%). `duration_ratio` = `output_duration_s / (word_count / 2.5)`, the diagnostic naive estimate
from Step 5 (not the hardened gate's own threshold, see Section 4).

* **Range**: 8.61x-17.87x, mean 14.79x.
* **Short-text mean**: 15.04x (5 texts, 2-4 words each).
* **Long-text mean**: 14.54x (5 texts, 7-10 words each).

Short and long texts show statistically indistinguishable blowup magnitudes -- there is no
length-dependent trend. `all is_likely_noise: False` across all 10 records (original 3-signal gate)
reproduces the exact blind-spot pattern t0014 first found, confirming it is not an isolated, one-off
occurrence.

One text (idx 5, "exploring how agentic conversational ai...", 8 words / 74 tokens, ratio 9.25
tokens/word) was flagged by `analyze_localization.py`'s oversegmentation check
(`input_token_count / word_count > 8`, `localization_summary.json`:
`any_oversegmentation_flagged: true`) -- a phonemizer-density artifact of multi-syllable words
("agentic", "conversational"), not evidence of a systematic tokenization bug: this text's
`duration_ratio` (11.43) is actually the *lowest* of the 10, not an outlier on the high end.

## 3. Cheap fix outcome: did NOT work (REQ-5, REQ-7)

### Pre-registered pass criteria (registered before any sweep run, `code/param_sweep_driver.py`)

A combination counts as "fixed" only if **all three** hold simultaneously:

1. `is_likely_noise == False` (hardened 3-signal check);
2. `duration_ratio <= 3.0`, where `duration_ratio = output_duration_s / (word_count / 1.5)`
   (matching `audio_quality_check.py`'s `estimate_naive_duration_bound_s`, a deliberately slow
   90-words/minute floor);
3. `longest_nonsilent_run_s <= 12.0`.

No partial improvement counts as a pass, per the Rejection Criteria section of `plan/plan.md`.

### Full sweep table (13/13 combinations completed, `results/param_sweep.json`)

Fixed test text: "This is a test of the Style T T S two inference harness." (identical to
t0013's/t0014's own gate text).

| Label | alpha | beta | steps | scale | duration (s) | `duration_ratio` | Passed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_default | 0.3 | 0.7 | 5 | 1.0 | 73.95 | 8.53 | No |
| grid_a0.1_b0.3 | 0.1 | 0.3 | 5 | 1.0 | 58.25 | 6.72 | No |
| grid_a0.1_b0.7 | 0.1 | 0.7 | 5 | 1.0 | 48.00 | 5.54 | No |
| grid_a0.1_b0.9 | 0.1 | 0.9 | 5 | 1.0 | 67.52 | 7.79 | No |
| grid_a0.3_b0.3 | 0.3 | 0.3 | 5 | 1.0 | 71.40 | 8.24 | No |
| grid_a0.3_b0.7 | 0.3 | 0.7 | 5 | 1.0 | 59.70 | 6.89 | No |
| grid_a0.3_b0.9 | 0.3 | 0.9 | 5 | 1.0 | 70.35 | 8.12 | No |
| grid_a0.5_b0.3 | 0.5 | 0.3 | 5 | 1.0 | 69.25 | 7.99 | No |
| grid_a0.5_b0.7 | 0.5 | 0.7 | 5 | 1.0 | 60.75 | 7.01 | No |
| grid_a0.5_b0.9 | 0.5 | 0.9 | 5 | 1.0 | 66.00 | 7.62 | No |
| ext_scale0.5 | 0.1 | 0.7 | 5 | 0.5 | 53.57 | 6.18 | No |
| ext_scale2.0 | 0.1 | 0.7 | 5 | 2.0 | 63.77 | 7.36 | No |
| ext_steps10 | 0.1 | 0.7 | 10 | 1.0 | 60.67 | 7.00 | No |

**0 of 13 combinations passed.** The best result (`grid_a0.1_b0.7`, `duration_ratio=5.54`) is still
1.85x over the `<=3.0` pass threshold -- not a near-miss, a wide margin. Parameter changes produced
real but bounded improvement (duration_ratio range across all 13: 5.54-8.53, vs. the
characterization batch's 8.61-17.87 on different, shorter/longer texts -- direct comparison is
limited since the sweep used a single fixed 12-word text) but never approached passing. No
combination (alpha, beta, embedding_scale, or diffusion_steps, individually or in the tested
combinations) reduced the blowup by the necessary margin. This is consistent with Section 1's
root-cause finding: since the elevated `pred_dur` values trace to a predictor-pathway calibration
issue (Section 1d) rather than to a poorly-conditioned style vector that inference-time sampling
parameters could correct, no amount of `alpha`/`beta`/`embedding_scale`/`diffusion_steps` tuning was
expected to fully resolve it -- the sweep's negative result corroborates, rather than contradicts,
the root-cause verdict.

**No `results/audio_samples/ft/v11_corrected.wav` was produced.** Per the Rejection Criteria ("no
partial improvement may be reported as a pass") and per `task_description.md`'s own instruction to
document a negative result with the same rigor as a positive one, this task does not claim a fix
that does not exist.

### Recommendation

Targeted `predictor`/`predictor_encoder` fine-tuning (holding the now-working decoder fixed) is the
recommended follow-up, not full retrain. Section 1d's tensor forensics narrows this specifically:
`predictor_encoder`'s substantially-shifted-but-still-possibly-miscalibrated weights (and/or its
`copy.deepcopy(style_encoder)` initialization strategy) are the most concrete identified suspect,
more so than `duration_proj` itself (near-zero weight shift). A follow-up task should re-run Stage-2
training with `predictor`/`predictor_encoder` unfrozen and every other module (including the
now-confirmed-working `decoder`) frozen, monitoring `dur_loss` for actual convergence (target:
approaching t0009's 0.034, not plateauing at 0.53-0.62) as the pre-registered success gate -- this
is a GPU training run and is explicitly out of this task's no-GPU scope.

## 4. Gate hardening (REQ-8, REQ-9, REQ-10)

### New signals (`code/audio_quality_check.py`, Step 10)

Two new fields added to `AudioQualityResult` (8 fields total, up from 6, verified by
`code/test_audio_quality_check.py` and the plan's own literal interface check):

* **`duration_sanity_pass: bool | None`** -- `None` when the optional `text` parameter is not
  supplied; otherwise `actual_duration_s <= estimate_naive_duration_bound_s(text) * 3.0`, where
  `estimate_naive_duration_bound_s(text) = len(text.split()) / 1.5` (a deliberately slow
  90-words-per-minute floor, to minimize false positives on genuinely slow, deliberate filler
  narration).
* **`longest_nonsilent_run_s: float`** -- the longest contiguous run of 20ms frames not below
  -40dBFS, computed in the same frame loop that already computes `silence_fraction` (no extra pass
  over the audio). Flagged when it exceeds `LONGEST_NONSILENT_RUN_THRESHOLD_S = 12.0` (a flat,
  generous ceiling -- longer than any single breath group/phrase in normal speech).

**Design decision, made deliberately**: `is_likely_noise` itself is **not** redefined to fold in
either new signal -- it stays exactly the original 3-signal (silence/flatness/clip) computation, for
two reasons: (a) this is what makes the Section 4's three-way regression a real proof (v11's
`v11_best.wav` must keep reading `is_likely_noise=False` post-hardening to demonstrate the OLD gate
genuinely would still ship it -- see the literal assertion in `plan/plan.md`'s Verification
Criteria, `byfixture['v11_as_shipped']['is_likely_noise'] is False`); (b) it preserves exact
backward compatibility for any existing caller that only checks `is_likely_noise`. Callers combine
all three signals into an overall pass/fail via a derived `hardened_gate_pass` field (see
`code/run_gate_regression.py`), computed as
`not (is_likely_noise or duration_sanity_pass is False or longest_nonsilent_run_s > 12.0)`.

### ASR-round-trip evaluation (Step 11, REQ-9): evaluated, not implemented

Full writeup: `results/asr_roundtrip_evaluation.md`. Summary: `faster-whisper`/`jiwer` installed
with no dependency friction (under 6 seconds, ~90MB), and the `base.en` model downloaded
automatically on first use. However, `compute_wer`'s own `DURATION_RATIO_LOW=0.5`/
`DURATION_RATIO_HIGH=2.0` pre-gate skipped **all 3** sampled characterization clips
(`results/asr_roundtrip_raw.json`: `gated_out_by_duration_ratio: true` for every record) --
`duration_ratio` values of 8.6-17.9 fall far outside `[0.5, 2.0]`, so `compute_wer` never even calls
`model.transcribe()` on the exact class of clip this task investigates. **Verdict: not implemented**
-- it is the wrong tool for this specific failure mode (the duration-sanity/ non-silent-run signals
already catch it directly and more cheaply), and wiring it in as a general third layer would need
non-trivial rework (a text-only duration gate instead of the reference-paired one, and a cached
`WhisperModel` instance to avoid the several-second per-call model-reload cost). Documented as a
follow-up recommendation, not force-fit into this task's scope.

### Three-way regression proof (Step 12, REQ-10, [CRITICAL])

`results/gate_regression.json`, produced by `code/run_gate_regression.py` against the hardened gate:

| Fixture | `is_likely_noise` | `duration_sanity_pass` | `longest_nonsilent_run_s` | `hardened_gate_pass` |
| --- | --- | --- | --- | --- |
| v10 (`v10_epoch16_primary.wav`) | **True** | True | 3.42 | **False** |
| v11_as_shipped (`v11_best.wav`) | **False** | **False** | 73.94 | **False** |
| v11_corrected | N/A (REQ-7 path -- fixture does not exist) |  |  |  |

This is the decisive proof this task's REQ-10/[CRITICAL] step exists to produce:

* **v10 still fails** (`is_likely_noise=True`, unchanged) -- the hardened gate has not regressed on
  the failure mode it already caught.
* **v11_as_shipped keeps `is_likely_noise=False`** -- proving the OLD 3-signal gate genuinely would
  still ship this clip today, exactly as t0014 found.
* **v11_as_shipped's `hardened_gate_pass` flips to False** -- driven entirely by the two NEW signals
  (`duration_sanity_pass=False`, `longest_nonsilent_run_s=73.94 > 12.0`). This is the specific
  blind-spot closure this task exists to prove.
* No `v11_corrected` fixture exists to test (REQ-7 path, Section 3) -- documented explicitly rather
  than silently omitted, per the Rejection Criteria's requirement that an incomplete three-way
  regression state its incompleteness rather than substitute a partial proof.

## 5. Metrics

`results/metrics.json` (explicit multi-variant format): `v11-as-shipped` (`rtf=3.18`,
`speaker_sim=0.444`, reproducing t0014's own reported 0.444 to 3 decimal places) and `v10-primary`
(`rtf=5.26`, `speaker_sim=0.351`, reproducing t0013's own reported ~0.351).
`results/speaker_sim_scores.json` holds the raw per-clip values. `ttfb_ms` is correctly omitted --
this offline batch harness has no streaming HTTP endpoint in scope, matching t0013's and t0014's own
precedent.

## 6. Summary and recommendation for follow-up work

* **Root cause**: predictor-pathway calibration failure, most plausibly centered on
  `predictor_encoder`'s mismatched initialization (`copy.deepcopy(style_encoder)`, never loaded from
  a duration-calibrated pretrained state) feeding an elevated, non-saturating `pred_dur`
  distribution into `duration_proj` -- not `duration_proj`'s own weights (near-zero shift from
  control), not a `pred_aln_trg`/decoder alignment-plumbing bug (frame-to-output ratio identical to
  the healthy control), not simple `max_dur`-ceiling saturation (0.44% of tokens near ceiling).
* **Universal**: all 10/10 characterization texts blow up, 8.6x-17.9x, no text-length correlation.
* **Cheap fix**: does not work. 0/13 pre-registered `alpha`/`beta`/`embedding_scale`/
  `diffusion_steps` combinations passed; best result still 1.85x over threshold.
* **Gate hardened and proven**: two new signals, three-way regression confirms the hardened gate now
  catches exactly the failure mode the old gate missed, while not regressing on the already- caught
  v10 failure mode.
* **Recommended follow-up task**: targeted `predictor`/`predictor_encoder` fine-tuning (decoder and
  all other modules frozen), monitoring `dur_loss` convergence toward t0009's 0.034 reference as the
  explicit, pre-registered success gate -- a GPU training run, out of this task's scope.
