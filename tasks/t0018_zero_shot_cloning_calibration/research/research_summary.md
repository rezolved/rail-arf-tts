# Research Summary — t0018_zero_shot_cloning_calibration

## Key Findings (top 10 insights directly actionable for this task)

1. Measurement task only (no fine-tuning, no val_96 tuning): benchmark F5-TTS (`SWivid/F5-TTS`,
   checkpoint `F5TTS_v1_Base`), CosyVoice 2 (`FunAudioLLM/CosyVoice2-0.5B`), Chatterbox
   (`ResembleAI/chatterbox`, `pip install chatterbox-tts`) via t0008's `tts_eval_harness`, plus
   paired re-scoring of `elevenlabs_david` (self-consistency ceiling) and `kokoro_v3_bundle` (best
   fine-tune, 0.631 fillers speaker_sim).
2. Licensing (Key Question 6): F5-TTS code is MIT but default checkpoint is CC-BY-NC-4.0
   (non-commercial, survives fine-tuning). CosyVoice 2 is Apache-2.0. Chatterbox is MIT end-to-end.
   Record per-system in environment table; flag in `results/suggestions.json` if F5-TTS leads.
3. Streaming/TTFB non-uniform: only CosyVoice 2 has genuine chunked streaming (vendor claims
   ~150ms first-packet, but a GitHub issue reports P99 outliers at concurrency ≥4 — re-measure
   p50/p95/p99). F5-TTS's official path and base Chatterbox are non-streaming — report
   whole-utterance latency, explicitly labelled (t0013 precedent: `ttfb_ms: not measured`).
   Chatterbox-Turbo (<200ms target) is a different checkpoint, not this task's.
4. F5-TTS docs recommend reference clips under ~12s, warn of mid-word truncation above ~20s, cap
   prompt+generated audio at 30s total. This task's `ref_concat` targets ~30s — validate against
   this ceiling for F5-TTS before the full run, or document the deviation.
5. Published similarity numbers use different embedding backbones than this project's GE2E cosine:
   F5-TTS's "SIM/SIM-o" and CosyVoice 2's "SS" are WavLM-based (Seed-TTS-eval); Chatterbox has no
   official paper (maintainer-confirmed), only a vendor-run Podonos AB test (naturalness, not
   similarity). Never merge these into this task's speaker_sim tables — same rule as StyleTTS2's
   CMOS-S.
6. No literature has any reference-clip-duration ablation for zero-shot cloning — `ref_single` vs
   `ref_concat` (Key Question 4) is a first-of-its-kind measurement for this project.
7. Vocoder literature (3 of 4 corpus papers) shows HiFi-GAN-family decoders run 2-3 orders of
   magnitude faster than real-time on GPU, quality driven by discriminator design (MPD ablation:
   removing it drops MOS 4.10→2.28) not generator size. Grounds a hypothesis (not established fact
   for these 3 systems) that a missed 300ms TTFB bar is more likely the backbone than the vocoder.
8. StyleTTS2 (kokoro_v3_bundle's architecture) trades similarity for data efficiency in its own
   zero-shot eval: beats Vall-E on naturalness (CMOS +0.67) but trails on similarity (CMOS-S −0.47)
   using 250x less data. Qualitative context only (different metric) for why kokoro_v3_bundle may
   underperform the three large-corpus systems on speaker_sim (Key Question 2) — never a table entry.
9. Mandatory reuse: `tts_eval_harness` (t0008) for `build_reference_split(seed=42)`,
   `compute_speaker_sim`/`compute_wer`/`compute_duration_ratio`, `SynthResult` contract; **t0015's**
   hardened `audio_quality_check.py` (not t0013/t0014's copies); `build_reference_concat.py`
   (t0014) as a pattern to adapt (5.5s → 30s target, half-A-only source pool).
10. GPU cost discipline: t0010 overspent $272.78 vs a $45-100 plan on this VM pool (watchdog gap +
    full-disk teardown delay); t0014 avoided repeat via proactive disk monitoring. Budget here
    (~$45 planned, $70 cap) is tight for three heavy model stacks — confirm watchdog PID before
    first weight download, monitor `/mnt` disk usage throughout.

## Best Approaches (top 3 recommended implementation approaches from research)

### Approach 1: Adapter-per-system on the existing t0008 harness contract
Write six new adapters (F5-TTS/CosyVoice2/Chatterbox × 2 ref conditions) matching `SynthResult`
in `adapters.py`, models loaded once by the caller (never inside the adapter). Reuse
`harness.run_system_eval` and `report.py`'s aggregation unchanged; write only the four new
required chart functions.

### Approach 2: Streaming-aware TTFB, verified per system rather than assumed
Genuine first-chunk wall time for CosyVoice 2 (mirroring the Kokoro `KPipeline` streaming pattern);
whole-utterance latency, explicitly labelled, for F5-TTS and Chatterbox (t0013 precedent).
Re-measure CosyVoice 2's p50/p95/p99 on H100 rather than citing the vendor's 150ms figure.

### Approach 3: Layered quality gating with mandatory human listening
Run t0015's hardened `audio_quality_check.py` on all ~1568 clips; report gate-filtered and
unfiltered speaker_sim means side by side (t0013: confirmed-garbage audio still scored 0.31-0.35
GE2E cosine). Deliver the mandatory `results/listening_guide.md` + `results/audio_samples/` for
direct human judgment.

## Reusable Code / Assets

* `tasks/t0008_tts_eval_harness_baselines/code/{harness,adapters,scoring,report}.py` — import via
  `tts_eval_harness` library: `build_reference_split`, `compute_speaker_sim`, `compute_wer`,
  `compute_duration_ratio`, `SynthResult`, `elevenlabs_david`/`kokoro_v3_bundle` adapters (as-is),
  `compute_variant_metrics`, `build_metrics_json`, `build_tables_json`, `_percentile`.
* `tasks/t0015_v11_duration_blowup_forensics/code/audio_quality_check.py` (233 lines) — copy into
  task; `check_audio_quality(wav_path, *, text=None)`; use this version, not t0013/t0014's earlier
  copies.
* `tasks/t0014_v11_decoder_fix_retrain/code/build_reference_concat.py` (52 lines) — copy into task;
  adapt `REFERENCE_CLIP_NAMES` (3 clips, ~5.5s) to reach ~30s from half-A-only clips.
* `tasks/t0013_v10_synthesis_quality_forensics/code/infer_styletts2.py` — not directly reusable
  (StyleTTS2-specific); cited only as precedent for instrumented inference wrappers.

## Key Papers (top 5, with finding most relevant to this task)

* **Chen et al. 2024 (F5-TTS)** — Official SIM-o (WavLM-based)/WER benchmarks; not comparable to
  GE2E `speaker_sim`. Discovered, not yet downloaded.
* **Du et al. 2024 (CosyVoice 2)** — Official SS/WER/NMOS tables and streaming architecture behind
  the ~150ms first-packet claim to re-measure. Not yet downloaded.
* **Li et al. 2023 (StyleTTS 2)** — `kokoro_v3_bundle`'s architecture; zero-shot CMOS-S (−0.47 vs
  Vall-E, 250x less data) is qualitative context only, different metric family.
* **Kong et al. 2020 (HiFi-GAN)** — MPD ablation (1.82 MOS-point effect), up to 3,701x real-time
  GPU vocoder speed; background for TTFB-bottleneck attribution if a system misses 300ms.
* **Wan et al. 2018 (GE2E loss)** — Foundational paper behind `speaker_sim`; not yet downloaded,
  not required to execute this task.

## Risks Flagged in Research

* F5-TTS default checkpoint is CC-BY-NC-4.0 — production blocker if it leads on the metrics.
* F5-TTS `ref_concat` (~30s) risks colliding with its own 20s truncation-risk warning / 30s
  generation cap — validate or document the deviation before the full run.
* CosyVoice 2's vendor ~150ms TTFB has documented P99 outliers at concurrency ≥4 — re-measure,
  don't quote the vendor figure.
* No literature precedent exists for reference-clip-duration effects — Key Question 4 is answerable
  only empirically.
* GPU idle billing / disk-full teardown caused a $272.78 overrun on this VM pool before (t0010); the
  $70 cap here is tight for three heavy model stacks — watchdog PID + disk monitoring required.
* `speaker_sim` alone can't distinguish broken from mediocre audio (t0013) — report gate-filtered
  and unfiltered means side by side, plus the human-listening deliverable.
* None of the three target systems have prior integration in this codebase — install, weight
  download, per-system environments are new work.

## Full Detail Available In

* `tasks/t0018_zero_shot_cloning_calibration/research/research_papers.md` — 4 papers
* `tasks/t0018_zero_shot_cloning_calibration/research/research_internet.md` — 18 sources
* `tasks/t0018_zero_shot_cloning_calibration/research/research_code.md` — 5 task references cited
  (15 tasks reviewed)
