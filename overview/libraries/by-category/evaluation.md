# Libraries: `evaluation`

1 librar(y/ies).

[Back to all libraries](../README.md)

---

<details>
<summary>📦 <strong>TTS Evaluation Harness</strong> (<code>tts_eval_harness</code>)</summary>

| Field | Value |
|---|---|
| **ID** | `tts_eval_harness` |
| **Version** | 0.1.0 |
| **Modules** | `tasks/t0008_tts_eval_harness_baselines/code/harness.py`, `tasks/t0008_tts_eval_harness_baselines/code/adapters.py`, `tasks/t0008_tts_eval_harness_baselines/code/scoring.py`, `tasks/t0008_tts_eval_harness_baselines/code/report.py`, `tasks/t0008_tts_eval_harness_baselines/code/extract_decoder.py`, `tasks/t0008_tts_eval_harness_baselines/code/constants.py`, `tasks/t0008_tts_eval_harness_baselines/code/paths.py`, `tasks/t0008_tts_eval_harness_baselines/code/run_eval.py`, `tasks/t0008_tts_eval_harness_baselines/code/prepare_prompts.py`, `tasks/t0008_tts_eval_harness_baselines/code/score_speaker_sim.py` |
| **Dependencies** | numpy, soundfile, torch, kokoro, requests |
| **Date created** | 2026-09-14 |
| **Categories** | [`evaluation`](../../../meta/categories/evaluation/), [`tts`](../../../meta/categories/tts/), [`benchmark`](../../../meta/categories/benchmark/) |
| **Created by** | [`t0008_tts_eval_harness_baselines`](../../../overview/tasks/task_pages/t0008_tts_eval_harness_baselines.md) |
| **Documentation** | [`description.md`](../../../tasks/t0008_tts_eval_harness_baselines/assets/library/tts_eval_harness/description.md) |

**Entry points:**

* `run_eval` (script) — CLI entry point: run synthesis + timing across any combination of 8
  TTS systems and 2 prompt sets, saving per-clip metrics JSON.
* `score_speaker_sim` (script) — CLI entry point: compute GE2E speaker similarity and WER on
  synthesized audio, updating per-clip metrics JSON in place.
* `report` (script) — CLI entry point: aggregate per-clip metrics into metrics.json (variant
  format), tables.json, and three charts.
* `elevenlabs_david` (function) — Synthesize text via ElevenLabs streaming API, measuring TTFB
  from first non-empty chunk.
* `kokoro_george` (function) — Synthesize via Kokoro-82M base model with bm_george voice
  through build_pipeline.
* `kokoro_v3_bundle` (function) — Synthesize via Kokoro-82M with fine-tuned v3 decoder + David
  voicepack.
* `load_kokoro_model_with_checkpoint` (function) — Load KModel (stock or with a five-module
  packaged checkpoint) ready for build_pipeline.
* `build_centroid` (function) — Compute L2-normalized mean GE2E embedding from a list of
  reference WAV files.
* `compute_speaker_sim` (function) — Score synthesized audio against a reference centroid or
  half-B embeddings using GE2E cosine similarity.
* `compute_wer` (function) — Transcribe synthesized audio with Whisper base.en and compute WER
  vs reference text.
* `compute_duration_ratio` (function) — Compute per-clip synthesis/reference duration ratio;
  flag clips with ratio > 5.0 as explosions.
* `extract` (function) — Extract five-module state dict from a raw StyleTTS2/Kokoro training
  checkpoint.
* `build_reference_split` (function) — Split ElevenLabs reference corpus 679/679 (seed=42),
  compute centroid from half-A, return (centroid, half_b_paths).

Reusable evaluation harness for TTS systems measuring speaker similarity (GE2E cosine), TTFB,
RTF, WER, and duration ratio across multiple prompt sets.

</details>
