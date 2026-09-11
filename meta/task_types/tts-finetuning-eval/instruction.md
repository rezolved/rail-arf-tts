# TTS Fine-tuning Evaluation

## Goal

Evaluate a Kokoro-82M fine-tuned checkpoint on val_96 and the filler corpus, measuring speaker
similarity, TTFB, and RTF. Compare against the ElevenLabs David baseline and Kokoro base.

## Context

Read before starting:

* `project/description.md` — success criteria (speaker_sim ≥ 0.85, TTFB ≤ 300 ms).
* `overview/models/kokoro_82m_v4_finetune.md` — training config and known issues.
* `overview/metrics/` — all three metric definitions.
* `LESSONS.md` — warmup protocol (Lesson 1), smoke-gate (Lesson 2).

Key training config facts (from `configs/config_david_v4.yml`):

* Stage 2 starts from `first_stage.pth` with `load_only_params: true`, `multispeaker: true`.
* `train_LM: false`, `lambda_slm: 0.0` (WavLM discriminator disabled — keep 0.0 or shape
  mismatch crash at GAN epoch).
* Prior checkpoints `epoch_2nd_00000–00003.pth` are ALL corrupted (OOM recovery) — do not use
  them. Only load checkpoints generated after the clean restart from `first_stage.pth`.

## Steps

1. Identify the checkpoint to evaluate (epoch number, val loss from training log).
2. Load checkpoint on LLM-T1-NC80; run smoke-gate synthesis.
3. Discard 50 warmup clips; run N ≥ 96 inference passes on val_96 prompts.
4. Compute per-clip TTFB, RTF, GE2E speaker_sim vs ElevenLabs David reference.
5. Repeat on the 1358-clip filler corpus (N ≥ 100 measured, after 50 warmup).
6. Save per-clip results and aggregate metrics.
7. Compare against t0002 (ElevenLabs baseline) and t0003 (Kokoro base) results.

## Done When

* `results/per_clip_metrics.json` exists with ≥ 96 rows for val_96 and ≥ 100 for filler corpus.
* `results/metrics.json` contains `ttfb_ms`, `speaker_sim`, `rtf` with val_96 and filler splits.
* `results/results_summary.md` includes comparison table vs baseline and base model.
* Verificator passes with no errors.

## Forbidden

* NEVER evaluate a checkpoint from the corrupted `epoch_2nd_00000–00003.pth` series.
* NEVER add `resemblyzer` to main deps (keep in `[speaker-sim]` extra).
* NEVER train or tune on val_96.
* NEVER set `lambda_slm > 0` in inference config — causes WavLM shape mismatch crash.
