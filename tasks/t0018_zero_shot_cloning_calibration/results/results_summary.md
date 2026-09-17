# Results Summary: Zero-Shot Voice-Cloning Calibration

## Summary

Benchmarked three open zero-shot voice-cloning systems (F5-TTS, CosyVoice2, Chatterbox) against
David reference audio on `LLM-T1-NC80` (2xH100) using the t0008 harness. Chatterbox completed both
reference conditions at 100% success; CosyVoice2's `ref_single` condition succeeded fully and beat
the ElevenLabs self-consistency ceiling on `speaker_sim`, but its `ref_concat` condition hard-failed
(0/196, a genuine 30s reference-audio limit in CosyVoice2's own code); F5-TTS produced no data at
all (indefinite hang in model loading across three attempts, most likely a session/VM-level issue
but not conclusively isolated). Total GPU spend was **$58.25** of the **$70** hard cap.

## Metrics

* **CosyVoice2 `ref_single` val96 `speaker_sim` = 0.8627852474649748** — exceeds the ElevenLabs
  self-consistency ceiling of **0.7923378584285578** (val96, re-measured this session) by
  **+0.0704**, and `kokoro_v3_bundle`'s stored **0.588** by **+0.275** (source:
  `results/metrics.json`).
* **CosyVoice2 `ref_single` fillers `speaker_sim` = 0.8420862078666687** vs ElevenLabs
  **0.8324875295162201** (+0.0096) and `kokoro_v3_bundle` **0.631** (+0.211).
* **Chatterbox `ref_single`/`ref_concat` `speaker_sim` = 0.7964-0.8112** across both prompt sets and
  both reference conditions, with **100% success (392/392 clips)** on both conditions — the most
  reliable measured system.
* **No cloning system met the 300 ms TTFB target**: measured p50 TTFB ranged from
  **1344.3423864991928 ms** (Chatterbox `ref_single` fillers) to **2859.2522075005036 ms**
  (CosyVoice2 `ref_single` val96) across the six successful (system, condition, prompt_set)
  combinations, 4-10x over the target and far above `kokoro_v3_bundle`'s stored **185/282 ms**.
* **CosyVoice2 `ref_concat` is NULL (0/196 successful, both prompt sets)**: CosyVoice2's own
  frontend hard-rejects reference audio over 30 s (`assert speech.shape[1] / 16000 <= 30` in
  `cosyvoice/cli/frontend.py`), and this task's shared `ref_concat` clip is 30.57 s — 0.57 s over
  the limit.
* **F5-TTS is NULL for all variants** and `kokoro_v3_bundle` was **not re-measured** this session
  (both hit an identical indefinite-hang failure signature in model loading); `kokoro_v3_bundle`'s
  `speaker_sim` values above are t0008's stored numbers, not fresh measurements.
* **Total GPU cost = $58.25** of the **$70** hard cap (`results/costs.json`), on `LLM-T1-NC80`
  (2xH100 NVL) for **4.17 billable hours** (`results/remote_machines_used.json`).

## Verification

* `uv run python -u -m arf.scripts.verificators.verify_task_metrics t0018_zero_shot_cloning_calibration`
  — **PASSED**, 0 errors, 0 warnings (`results/metrics.json` contains only the three registered
  metric keys `speaker_sim`, `ttfb_ms`, `rtf` in every variant).
* `uv run python -u -m arf.scripts.aggregators.aggregate_metrics --format ids` — confirms the
  project's full registered metric set is exactly `rtf`, `speaker_sim`, `ttfb_ms`, matching what
  `results/metrics.json` uses.
* `uv run python -u -m meta.asset_types.answer.verificator --task-id t0018_zero_shot_cloning_calibration zero-shot-speaker-sim-ceiling`
  — **PASSED** (recorded in step 9/implementation; answer asset at
  `assets/answer/zero-shot-speaker-sim-ceiling/`).
* DVC tracking confirmed for all three audio directories
  (`results/audio_samples/{harness,comparison_set,references}.dvc` present and pushed per step 9's
  implementation log).
* All 4 required charts (`speaker_sim_by_system.png`, `ttfb_vs_speaker_sim.png`,
  `ref_condition_effect.png`, `wer_by_system.png`) present in `results/images/` and non-zero-byte.
