# Research Summary — t0015_v11_duration_blowup_forensics

## Key Findings (top 10 insights directly actionable for this task)

1. Duration computation to instrument: `infer_styletts2.py:synthesize()` lines 342-346 —
   `duration = model.predictor.duration_proj(x)` -> `sigmoid().sum(axis=-1)` ->
   `pred_dur = round().clamp(min=1)` -> `pred_aln_trg = torch.zeros(input_lengths, pred_dur.sum())`.
   Neither `pred_dur` nor the resulting frame count is logged/returned — `synthesize()` only returns
   `(wav, wall_time)`. Add logging/return of `pred_dur.tolist()`, `pred_dur.sum().item()`, and
   `input_lengths.item()` as the first, no-GPU diagnostic step.
2. `model_params.max_dur = 50` in `config_david_v11.yml:79` bounds what a well-formed per-token
   `duration_proj` sigmoid-sum can output. A per-token `pred_dur` > ~50, or a token count far larger
   than the ~10-word input implies (phonemizer over-segmentation), are the two hypotheses to
   discriminate: predictor-calibration vs. plumbing bug.
3. `alpha`/`beta`/`diffusion_steps`/`embedding_scale` are hardcoded LibriTTS-demo defaults
   (0.3/0.7/5/1.0). Only `diffusion_steps` is CLI-exposed. These feed the diffusion sampler that
   produces the style vector consumed by `predictor.text_encoder`/`lstm`/`duration_proj`, so a bad
   style vector could itself skew `pred_dur` — a third hypothesis to instrument.
4. Checkpoint-load path is ruled out: `results/load_log_epoch_00048.json` (t0014) confirms 0
   missing/unexpected keys on all 13 modules. Focus instrumentation on the forward pass only.
5. `predictor`/`predictor_encoder` DID receive gradient updates every step of all 50 epochs
   (`optimizer.step("predictor"/"predictor_encoder")` in `train_second_v11.py:733-734` are
   unconditional, unlike `style_encoder`/`decoder`/`diffusion`, gated on `joint_epoch`/`diff_epoch`)
   — rules out "rode along frozen at init." But `predictor_encoder` is excluded from the
   `first_stage_path` load (`ignore_modules`) and instead copied from `style_encoder`.
6. Despite gradients, per-epoch `dur_loss` in `data/run_v11/metrics.jsonl` plateaus at 0.53-0.62
   across all 50 epochs (epoch 1: 0.620, epoch 50: 0.529) — contrast t0009's v6c run reaching 0.034
   by epoch 6. Aggregate `val_loss` looked healthy (best 0.3394) throughout, masking this. Evidence
   for predictor-calibration failure, not pure plumbing bug, but Finding 1's instrumentation is
   still required to confirm.
7. Same "aggregate loss hides localized failure" pattern t0013 found for the decoder (`val_loss`
   healthy for 17 epochs, output 75-81% clipped noise). Recurring twice — supports output-inspecting
   gate signals over trusting upstream loss curves.
8. Reusable diagnostic patterns, apply before any sweep: `inspect_checkpoint.py` (t0013, no-GPU
   tensor forensics via raw `torch.load`) and `random_decoder_probe.py` (t0013, module-isolation
   falsification probe) — adapt to swap `predictor`/`predictor_encoder` instead of `decoder`.
9. `tts_eval_harness` library (t0008) has `compute_duration_ratio` (reference-paired, flags ratio >
   5.0) and `compute_wer` (faster-whisper + JiWER), importable via
   `from tasks.t0008_tts_eval_harness_baselines.code.scoring import compute_wer, compute_duration_ratio`.
   Neither is drop-in for the new gate signal (must be text-only, words-per-second, no reference
   clip), but `compute_wer` is directly usable for an optional ASR-round-trip layer.
10. Varied-text prompt sets exist: `filler_prompts_100.json`/`val96_prompts.json` (t0008), with
    `ref_duration_s` precomputed. `ref_wav` paths are absolute into t0008's worktree — re-resolve
    repo-relative to `data/11labs_david/<basename>`.

## Best Approaches (top 3 recommended implementation approaches from research)

### Approach 1: Instrument before sweeping (cheap-first ladder)

Copy `t0014`'s `infer_styletts2.py`/`audio_quality_check.py` into `t0015/code/` (not t0013's —
t0014's are checkpoint/config-current). First add `pred_dur`/frame-count logging to `synthesize()`
and run the no-GPU `inspect_checkpoint.py`/`random_decoder_probe.py` variants targeting
`predictor`/`predictor_encoder`. Only after localizing the defect, expose `alpha`/`beta`/
`embedding_scale` as CLI flags and run a small grid sweep via a Python driver calling `synthesize()`
directly. Pre-register "fixed" criteria (duration within Nx of naive words-per-second estimate AND
`is_likely_noise=False`) before sweeping.

### Approach 2: Extend the gate in place, not a new module

Add duration-sanity (text-only words-per-second) and longest-contiguous-non-silent-run signals
directly inside `audio_quality_check.py`'s existing 20ms-frame silence loop (lines 79-88), per the
module's "one source of truth" convention (S-0013-04). Evaluate but don't necessarily implement an
ASR-round-trip layer using `compute_wer` from `tts_eval_harness`.

### Approach 3: Three-way regression proof, mirroring `demo()`'s existing pattern

Reuse `audio_quality_check.py`'s `demo()` self-check pattern, extended from two fixtures to three:
t0013's v10 sample (should still fail), t0014's `v11_best.wav` (should now fail on the new
duration/silence-gap signal despite passing the old flatness/clip signal), and the corrected output
from Approach 1 if the cheap fix works (should pass all signals).

## Reusable Code / Assets

* `tasks/t0014_v11_decoder_fix_retrain/code/infer_styletts2.py` (428 lines) — copy; instrument
  `synthesize()` for `pred_dur`/frame-count logging, add `alpha`/`beta`/`embedding_scale` CLI flags.
* `tasks/t0014_v11_decoder_fix_retrain/code/audio_quality_check.py` (156 lines) — copy; extend with
  duration-sanity and longest-nonsilent-run signals.
* `tasks/t0013_v10_synthesis_quality_forensics/code/inspect_checkpoint.py` (~350 lines) — copy,
  retarget at `kokoro-v11-best` for no-GPU predictor forensics.
* `tasks/t0013_v10_synthesis_quality_forensics/code/random_decoder_probe.py` (~180 lines) — copy,
  adapt probe to `predictor`/`predictor_encoder`.
* `tasks/t0014_v11_decoder_fix_retrain/code/build_reference_concat.py` (53 lines) — copy unchanged
  for re-synthesizing the same t0014 gate texts.
* `tasks/t0008_tts_eval_harness_baselines/code/scoring.py` — import `compute_wer`,
  `compute_duration_ratio` via the `tts_eval_harness` library.
* `tasks/t0008_tts_eval_harness_baselines/data/filler_prompts_100.json`, `data/val96_prompts.json` —
  varied prompts with precomputed `ref_duration_s`; re-resolve `ref_wav` repo-relative.
* `tasks/t0014_v11_decoder_fix_retrain/data/run_v11/metrics.jsonl` — per-epoch `dur_loss`, flat
  0.53-0.62 plateau.
* `tasks/t0014_v11_decoder_fix_retrain/results/load_log_epoch_00048.json` — 0 missing/unexpected
  keys on checkpoint load.

## Key Papers

(not generated — research-papers step skipped)

## Risks Flagged in Research

* `predictor_encoder` is copied from `style_encoder` rather than loaded from `first_stage_path`
  (excluded via `ignore_modules`) — a possible additional scale-mismatch source beyond `predictor`.
* CPU StyleTTS2-native inference is slow (rtf 3.2-5.5x); t0014's 73.95s-audio synthesis took 235.2s
  wall time. A 10-30-item sweep is feasible on CPU but budget time — no GPU should be needed.
* Weight-norm key-naming differs between torch/StyleTTS2-fork versions (`weight_g`/`weight_v` vs.
  `parametrizations.weight.original0/1`); relevant only when loading an external LibriTTS control
  checkpoint's `predictor` — `_rename_legacy_parametrization_keys()` in `infer_styletts2.py` handles
  this already.
* Prior "gate passed" claims in this chain are narrowly pre-registered; don't let a
  partially-improved but still-anomalous duration get silently reported as success — pre-register
  exact pass/fail criteria before the sweep.

## Full Detail Available In

* `tasks/t0015_v11_duration_blowup_forensics/research/research_papers.md` — (not generated — step
  skipped)
* `tasks/t0015_v11_duration_blowup_forensics/research/research_internet.md` — (not generated — step
  skipped)
* `tasks/t0015_v11_duration_blowup_forensics/research/research_code.md` — 8 tasks reviewed, 6 cited,
  2 libraries found (both relevant)
