# TTS Benchmark Run

## Goal

Benchmark a TTS system on the Rezolve filler corpus or val_96, measuring TTFB, speaker similarity
(GE2E cosine vs ElevenLabs David reference), and real-time factor.

## Context

Read before starting:

* `project/description.md` — success criteria and metric targets.
* `overview/metrics/ttfb_ms.md`, `overview/metrics/speaker_sim.md`, `overview/metrics/rtf.md` —
  metric definitions and targets.
* `overview/datasets/fillers_1358.md` or `overview/datasets/val_96.md` — dataset description.
* `LESSONS.md` — warmup protocol (Lesson 1), smoke-gate before measurement (Lesson 2).

## Steps

1. Run the smoke gate: one synthesis request must succeed before warmup begins.
2. Discard 50 warmup requests (same prompt corpus, same engine session).
3. Run N ≥ 100 measured requests; record per-clip TTFB, RTF, audio duration.
4. Compute GE2E speaker embeddings for synthesized clips and compare to mean embedding of
   `data/11labs_david/` reference set. Install `resemblyzer` via `pip install .[speaker-sim]`
   (do NOT add to main deps — see `overview/metrics/speaker_sim.md`).
5. Report: p50, p95, p99 TTFB; mean ± std speaker_sim; mean ± std RTF.
6. Save per-clip results to `results/per_clip_metrics.json`.

## Done When

* `results/per_clip_metrics.json` exists with at least 100 rows.
* `results/metrics.json` contains `ttfb_ms`, `speaker_sim`, `rtf` keys.
* `results/results_summary.md` includes the aggregate table and per-clip charts.
* Verificator passes with no errors.

## Forbidden

* NEVER skip the warmup protocol (Lesson 1).
* NEVER add `resemblyzer` to main `dependencies` in `pyproject.toml` — keep it in `[speaker-sim]`
  extra only (webrtcvad/pkg_resources breakage on setuptools ≥ 81).
* NEVER train or tune on val_96.
