# Predictor Tensor Forensics (Milestone B step 7, REQ-3)

No-GPU tensor-level cross-check of `predictor`/`predictor_encoder`, comparing `kokoro-v11-best`
(`epoch_00048.pth`) against the LibriTTS control checkpoint (`epochs_2nd_00020.pth`) that
`config_david_v11.yml`'s `first_stage_path` loaded from. Produced by
`code/predictor_tensor_forensics.py`.

## Whole-module weight-norm comparison

| Module | Checkpoint | Params | Finite | Weight norm | Mean abs | Max abs |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| predictor | v11 (kokoro-v11-best) | 16,194,612 | yes | 318.2025 | 0.059257 | 3.038462 |
| predictor | control (LibriTTS) | 16,194,612 | yes | 313.9305 | 0.058428 | 3.022892 |
| predictor_encoder | v11 (kokoro-v11-best) | 13,880,813 | yes | 400.6487 | 0.079838 | 2.698808 |
| predictor_encoder | control (LibriTTS) | 13,880,813 | yes | 459.0197 | 0.093326 | 3.320113 |

## `predictor.duration_proj` -- the exact submodule producing `pred_dur`

| Checkpoint | Params | Finite | Weight norm | Mean abs | Max abs |
| --- | ---: | --- | ---: | ---: | ---: |
| v11 (kokoro-v11-best) | 25,650 | yes | 83.4531 | 0.330554 | 3.038462 |
| control (LibriTTS) | 25,650 | yes | 82.8593 | 0.328231 | 3.022892 |

## Verdict

`predictor.duration_proj` weight norm: v11=83.4531, control=82.8593, delta=+0.5938 (+0.7%). This is
a **near-zero shift**: `duration_proj`'s own weights are essentially unchanged from the LibriTTS
control's, despite receiving gradients every step of all 50 epochs (per the `optimizer.step()`
training-log evidence) and despite the flat 0.53-0.62 `dur_loss` plateau in
`tasks/t0014_v11_decoder_fix_retrain/data/run_v11/metrics.jsonl`. This weighs AGAINST attributing
the duration blowup to `duration_proj`'s own weights having drifted to a badly calibrated region --
if that were the mechanism, 50 epochs of gradient updates would be expected to move this small
(25,650-param) submodule's weight norm more than 0.7%. The elevated `pred_dur` values found in
`results/duration_characterization.json` are more consistent with a distribution shift in
`duration_proj`'s INPUT (the output of `predictor.lstm`, itself fed by
`predictor.text_encoder(d_en, s, ...)` where `s` comes from the diffusion-sampled style vector and
`d_en` from `bert_encoder`) than with `duration_proj`'s own weights having miscalibrated.
`predictor_encoder`'s larger whole-module shift (see table above) is a more plausible contributor to
that upstream distribution shift, since it was excluded from the `first_stage_path` checkpoint load
(`ignore_modules`) and instead initialized as `copy.deepcopy(model.style_encoder)` before training,
per `research/research_summary.md` point 5.
