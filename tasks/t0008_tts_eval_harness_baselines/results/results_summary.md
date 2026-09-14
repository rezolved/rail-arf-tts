---
spec_version: "1"
task_id: "t0008_tts_eval_harness_baselines"
---
## Summary

Built and deployed a reusable TTS evaluation harness (`tts_eval_harness` library) that measures
speaker similarity (GE2E cosine via resemblyzer), TTFB, RTF, WER, and duration ratio across 8 TTS
systems on two prompt sets (96 held-out val96 texts + 100 production fillers). All systems were
benchmarked on LLM-T1-NC80 (2×H100 NVL). ElevenLabs David establishes the baseline; none of the
Kokoro variants reached the 0.85 speaker_sim target, with the best being `kokoro_v3_bundle` at 0.63
(fillers).

## Metrics

Key results (mean speaker_sim GE2E cosine, TTFB p50 ms, all systems on combined prompt sets):

- **ElevenLabs David** (scored vs half-B): speaker_sim=0.832 (fillers), 0.792 (val96);
  TTFB_p50=132ms (fillers), 153ms (val96) — target ≥0.85, ≤300ms
- **kokoro_v3_bundle** (best Kokoro): speaker_sim=0.631 (fillers), 0.588 (val96); TTFB_p50=185ms
  (fillers), 282ms (val96)
- **kokoro_base_v3_voicepack**: speaker_sim=0.603 (fillers), 0.582 (val96); TTFB_p50=196ms
  (fillers), 296ms (val96)
- **kokoro_t0006_v6d**: speaker_sim=0.601 (fillers), 0.482 (val96); TTFB_p50=132ms (fillers, matches
  ElevenLabs latency), 251ms (val96)
- **kokoro_base_george**: speaker_sim=0.563 (fillers), 0.595 (val96); TTFB_p50=233ms (fillers),
  340ms (val96, FAIL)
- **kokoro_t0005_best**: severely degraded — TTFB_p50=688ms/2156ms, most clips duration-explosion
  (sim=nan for val96); this checkpoint is not viable
- **kokoro_floor_control** (female af_heart, control): speaker_sim=0.444, confirming non-David
  baseline separation

Gap to target: best Kokoro (v3_bundle) is 0.63 vs ElevenLabs 0.83 — a gap of ~0.20 GE2E cosine
units. Further fine-tuning is needed.

## Verification

- Library asset `tts_eval_harness` verificator: **PASSED** (0 errors, 3 category warnings — no
  categories defined in `meta/categories/`)
- All 11 unit tests pass (`pytest tasks/t0008_tts_eval_harness_baselines/code/test_harness.py`)
- 1568 total per-clip records (8 systems × 196 prompts each): 1372 Kokoro + 196 ElevenLabs
- Results files verified present: `metrics.json` (16 variants), `tables.json`, 3 chart PNGs,
  `per_clip_metrics.json` (1568 records)
