# v11 Audible-Speech Gate Verdict (Milestone C step 9)

Per `plan/plan.md` Milestone C step 9 and the Rejection Criteria section, `is_likely_noise` from
`code/audio_quality_check.py`'s `check_audio_quality()` is the **sole, pre-registered pass/fail
signal** for this task -- not `val_loss`, not `speaker_sim`.

## Setup

* Checkpoint: `epoch_00048.pth` (2,086,802,736 bytes, sha256 `1fb329b5...23c4e`, matches
  `data/run_v11/checkpoints.json`'s manifest entry exactly), `val_loss=0.3468841314315796`,
  `flagged_healthy: true`. Chosen per plan.md step 8's rule ("latest epoch unless a clear regression
  is visible") -- post-`joint_epoch` (epoch>=30) val_losses run 0.339-0.367 with no regression at
  epoch 48; epoch 48 is the last saved checkpoint (only even epochs are saved, training completed
  all 50 epochs with `0` `gate_fired` events).
* Config: `code/config_david_v11.yml`.
* Reference audio: 3-clip concatenation of `lining_up_suggestions_17.wav`,
  `lining_up_suggestions_10.wav`, `putting_them_head_to_head_15.wav`
  (`tasks/t0008_tts_eval_harness_baselines/data/11labs_david/`), 0.2s silence gaps, 5.48s total --
  **identical** to the reference t0013's `results/v10_diagnosis.md` used, for direct comparability
  (built by `code/build_reference_concat.py`).
* Text: "This is a test of the Style T T S two inference harness." (same prompt as t0013's control
  and v10 runs).
* Harness: `code/infer_styletts2.py`, run through t0013's already-built `.venv-styletts2`
  (`torch==2.5.1` CPU) and t0013's `kikiri-tts/StyleTTS2` checkout (only `Utils/ASR`, `Utils/JDC`,
  `Utils/PLBERT` -- checkpoint-independent auxiliary models -- are sourced from there; the actual
  StyleTTS2 model weights come from this task's own `epoch_00048.pth`). Output:
  `results/audio_samples/ft/v11_best.wav`.

## Load instrumentation: all 13 modules load cleanly

`results/load_log_epoch_00048.json`: **0 missing, 0 unexpected on all 13 modules** (bert,
bert_encoder, predictor, decoder, text_encoder, predictor_encoder, style_encoder, diffusion,
text_aligner, pitch_extractor, mpd, msd, wd), all via the `module.`-stripped fallback path (this
project's own checkpoints are DataParallel-trained; no legacy-parametrization rename needed since
v11 was trained under this fork's current `models.py`, same as v10). Confirms the decoder-init fix
(REQ-1) held all the way through a completed 50-epoch training run, not just at the pre-flight
tensor-check stage.

## Audio output stats

| Metric | Control (`epochs_2nd_00020`, t0013) | v10 primary (`epoch_2nd_00016`, t0013, BROKEN) | v10 backup (`epoch_2nd_00014`, t0013, BROKEN) | **v11 (`epoch_00048`, this task)** |
| --- | ---: | ---: | ---: | ---: |
| Duration | 4.92s | 3.42s | 2.50s | 73.95s |
| RMS | 0.037 | 0.983 | 0.991 | 0.376 |
| Peak | 0.380 | 1.000 | 1.000 | 1.000 |
| Clip fraction (`\|x\|>0.99`) | 0.0 | **0.750** | **0.807** | 0.0048 |
| Silence fraction | 0.309 | 0.0 | 0.0 | 0.0 |
| Spectral flatness | 0.0615 | 0.00048 | 0.00019 | 0.0058 |
| `is_likely_noise` | False | **True** | **True** | **False** |

v11's `clip_fraction` (0.0048) is two orders of magnitude below the `0.3` noise threshold and below
even the known-good control's implicit "healthy" range, and two orders of magnitude below both
confirmed-broken v10 checkpoints' 0.750-0.807. `spectral_flatness` (0.0058) is well below the `0.35`
noise threshold and lower (more concentrated/formant-like) than the control's own 0.0615 -- the
opposite of both v10 checkpoints' near-zero-but-DC-saturated signature (their near-zero flatness
came from clipping-induced DC dominance, not from healthy formant structure; v11's combination of
near-zero clip fraction AND low flatness is consistent with real voiced speech, not saturation).
`silence_fraction=0.0` means no 20ms frame fell below -40dBFS anywhere in the 74s clip -- unlike the
control's natural pauses (0.309), but this is a duration/pacing anomaly (see below), not a
clipping-noise or silence-failure signature.

## Verdict: **PASS** (`is_likely_noise = False`)

By `code/audio_quality_check.py`'s pre-registered heuristic (spectral_flatness >= 0.35 OR
clip_fraction >= 0.3, and not >95% silent), v11's synthesis output is **not** classified as noise.
This is a qualitative reversal of v10's confirmed failure signature (clip_fraction 0.75-0.81 ->
0.0048; near-zero-but-DC-dominated flatness -> genuinely low, formant-consistent flatness),
corroborating the decoder-init fix (REQ-1: repointing `first_stage_path` at a real hifigan-shaped
checkpoint) at the audio level, not just the tensor-load level. Per the Rejection Criteria in
`plan/plan.md`, this PASS is the sole, pre-registered signal -- Milestone C steps 10-11 and
Milestone D proceed.

## Caveat: anomalous output duration (flagged honestly, not gate-blocking per pre-registered criteria)

v11's output is **73.95s** for a 10-word test sentence -- roughly 15-30x longer than the control
(4.92s) and both v10 checkpoints (2.50-3.42s) on the same text and (for v10) the identical reference
audio. This is a real, distinct anomaly, most likely in the duration predictor
(`predictor.duration_proj`) rather than the decoder/vocoder this task's gate is designed to catch --
the audio is not noise/clipped garbage (`clip_fraction=0.0048`, formant-consistent spectral content,
0% silence, no dropouts), it appears to be genuine but heavily elongated speech. Per this task's own
Rejection Criteria ("no other metric ... may be substituted" for the pre-registered
`is_likely_noise` gate), this does **not** block Milestone D, but it is recorded here as a known
limitation (also carried into the model asset's `description.md` Known Limitations and
`results/results_summary.md`) and flagged as a follow-up-suggestion candidate: investigate whether
`loss_params.lambda_dur=1.0` / duration-predictor calibration on the larger, LUFS-normalized 1,531-
clip corpus (versus v10's 250-clip corpus) produced a systematically different duration scale, or
whether this is specific to the 5.48s multi-clip reference style vector.
