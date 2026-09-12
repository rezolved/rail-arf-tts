# Package Kokoro v4 voicepack + decoder

## Motivation

t0001 trained a StyleTTS2 Stage 1 + Stage 2 (GAN) fine-tune of Kokoro-82M on the David voice using
the v4 dataset (1557 train / 96 val clips) — 4-6× more data than the v3 fine-tune (266 train clips).
Stage 2 training diverged after epoch 6 (val_loss 0.751 → 1.147 → 1.129 once the GAN discriminators
destabilized), so the raw StyleTTS2 checkpoints from t0001 are not directly production-ready.

The v3 fine-tune (`rail-benchmarks/kokoro-finetune/DAVID_V3_RESULTS.md`) solved the equivalent
problem by not shipping the raw StyleTTS2 checkpoint at all. Instead it packaged two artifacts in
Kokoro-API format:

- `david_v3_best_voicepack.pt` — a `[510, 1, 256]` ref_s style tensor, built by averaging
  `style_encoder` + `predictor_encoder` outputs (from the **Stage 1** checkpoint, i.e. before GAN
  destabilization) over ~200 David reference clips (`scripts/make_voicepack_from_ft.py`).
- `david_v3_best_decoder.pth` — the `decoder` sub-state-dict extracted from the **Stage 2** best
  checkpoint (by val_loss), with DDP/parametrization key remapping applied
  (`scripts/convert_checkpoint.py`), loadable directly by `kokoro.KModel`.

This task reproduces that packaging pipeline for v4 using t0001's checkpoints:

- Style source: `/mnt/kikiri-tts/StyleTTS2/logs/kokoro-david-v4/first_stage.pth` (Stage 1, frozen,
  pre-GAN — never touched by the divergence).
- Decoder source: `epoch_2nd_00005.pth` (t0001's best Stage 2 checkpoint by val_loss, epoch 6,
  val=0.751 — the last checkpoint before GAN divergence began).

A local ad-hoc StyleTTS2-native inference test in t0001 (bypassing the Kokoro packaging, loading
`epoch_2nd_00005.pth`'s own `style_encoder` directly) produced noisy/artifacted audio. That test
used the Stage 2 (post-partial-GAN) style encoder, not the Stage 1 style encoder v3's pipeline
specifies. This task's first concrete question is whether using the Stage-1-sourced style (untouched
by GAN instability) removes or reduces that noise, isolating whether the artifact was a
style-encoder problem or a decoder problem.

## Scope

- Reuse `rail-benchmarks/kokoro-finetune/scripts/make_voicepack_from_ft.py` and
  `convert_checkpoint.py` as reference implementations; adapt paths to v4's checkpoint locations and
  `data/v4/train/wavs/` instead of v3's `data/train/wavs/`.
- Produce exactly one voicepack + one decoder pair from the epoch 6 checkpoint (v4's current best by
  val_loss). Do not re-extract from other epochs unless the epoch 6 pair fails quality inspection.
- Run inference via `kokoro.KModel` + `KPipeline` (the actual production API path), not the raw
  StyleTTS2 `models.py` path used for the earlier ad-hoc listening test in t0001.
- Out of scope: retraining, hyperparameter changes, or diagnosing why Stage 2 v4 diverged (that is
  t0001's concern, already logged there).

## Approach

1. On LLM-T1-NC80, adapt `make_voicepack_from_ft.py`:
   - `CKPT` → `/mnt/kikiri-tts/StyleTTS2/logs/kokoro-david-v4/first_stage.pth`
   - `WAV_DIR` → `/mnt/kikiri-tts/data/v4/train/wavs`
   - `OUT` → `results/david_v4_voicepack.pt`
2. Extract the decoder sub-state-dict from `epoch_2nd_00005.pth`'s `net["decoder"]` key, run through
   `convert_checkpoint.py`'s key-remapping logic → `results/david_v4_decoder.pth`.
3. Load both via `kokoro.KModel(model="results/david_v4_decoder.pth")` +
   `KPipeline(voice="results/david_v4_voicepack.pt")` and synthesize the same 2 test sentences used
   in t0001's ad-hoc listening test, for direct comparison.
4. Listening/quality comparison: t0001's raw-StyleTTS2-style-from-Stage2 audio vs. this task's
   Kokoro-packaged style-from-Stage1 audio. Record a qualitative verdict (noise present/absent) —
   full GE2E speaker-similarity benchmarking against `data/11labs_david/` is a stretch goal, not
   required to close this task.

## Expected Assets

- `results/david_v4_voicepack.pt` — `[510, 1, 256]` ref_s tensor (model asset).
- `results/david_v4_decoder.pth` — Kokoro-API-loadable decoder state dict.
- `results/audio_samples/` — synthesized wavs for the 2 comparison sentences.
- `results/results_summary.md` — qualitative comparison (Stage1-style vs Stage2-style source),
  decision on whether this packaging is usable or whether t0001 needs a re-run with earlier stopping
  / different GAN schedule.

## References

- `tasks/t0001_kokoro_v4_stage2_finetune/` — source checkpoints and training log.
- `rail-benchmarks/kokoro-finetune/DAVID_V3_RESULTS.md` — v3 packaging recipe and results.
- `rail-benchmarks/kokoro-finetune/scripts/make_voicepack_from_ft.py`,
  `scripts/convert_checkpoint.py` — reference implementations.
