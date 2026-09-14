---
spec_version: "3"
task_id: "t0008_tts_eval_harness_baselines"
step_number: 5
step_name: "research-internet"
status: "completed"
started_at: "2026-09-14T15:37:00Z"
completed_at: "2026-09-14T15:47:11Z"
---
## Summary

Conducted 11 internet searches covering GE2E speaker embeddings (resemblyzer), Kokoro-82M TTFB/RTF
benchmarks, ElevenLabs streaming API latency, WER thresholds for TTS quality evaluation, and TTS
latency measurement methodology. Found 23 sources and discovered 4 papers for corpus addition.

## Actions Taken

1. Ran 11 targeted web searches covering resemblyzer GE2E usage, Kokoro-82M GPU TTFB benchmarks,
   ElevenLabs streaming TTFB measurements, WER thresholds for neural TTS, and TTFB measurement
   methodology (warmup, p-values, torch.cuda.synchronize).
2. Reviewed community benchmarks for Kokoro-82M on A100/RTX-class hardware: RTF ≈ 0.03, first-chunk
   as low as 28 ms on RTX 5090, 97 ms on consumer GPU. CPU baseline: 3,658 ms (unusable for TTFB
   target).
3. Reviewed ElevenLabs streaming TTFB data: p50 ≈ 264–335 ms (near target), 478 ms average PCM REST.
4. Determined WER thresholds: < 5% clean, ≥ 10% degraded, > 20% hard failure for short filler
   utterances.
5. Identified 4 papers to add: GE2E (Wan2018), StyleTTS2 (Li2023), VERSA evaluation toolkit (2024),
   AnalyzeSim speaker similarity analysis (2025).
6. Wrote research_internet.md (722 lines, status: complete, 11 searches, 23 sources, 4 discovered
   papers).

## Outputs

- `tasks/t0008_tts_eval_harness_baselines/research/research_internet.md`
- `tasks/t0008_tts_eval_harness_baselines/logs/searches/001–011_*.json` (11 search logs)

## Issues

Recovery required: step-executor completed output but did not finish commit/poststep cycle;
recovered inline by coordinator. All output verified correct. Paper additions for 4 discovered
papers will be handled during reporting step to avoid blocking research-code and planning.
