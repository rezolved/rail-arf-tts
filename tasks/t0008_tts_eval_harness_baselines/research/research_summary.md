---
task_id: "t0008_tts_eval_harness_baselines"
generated_at: "2026-09-14"
---
# Research Summary — t0008_tts_eval_harness_baselines

## Key Findings (top 10 insights directly actionable for this task)

1. **Five modules required**: loading only `decoder` into KModel causes duration explosions (89 s
   for 9.6 s sentences). The production format packages `bert`, `bert_encoder`, `predictor`,
   `text_encoder`, `decoder`. t0002's `extract_decoder_generic.py` (38 lines) does this correctly.

2. **`build_pipeline(model)` is mandatory** for all Kokoro synthesis: `lang_code="b"` + brand
   lexicon. `lang_code="a"` and bare `KPipeline` are silent failures — wrong phoneme tokens silently
   change synthesis quality. t0006's `infer_v6d.py` bypasses this; do not use it as a template.

3. **All checkpoint binaries are DVC-only** — `dvc pull` required before any synthesis. Four paths:
   t0005 `epoch_2nd_00003.pth` (1.87 GB), t0006 v6d `epoch_2nd_00006.pth` (1.86 GB), v3 decoder (327
   MB), v3 voicepack (523 KB).

4. **GE2E cosine thresholds**: same-speaker per-clip ≈ 0.7; same-speaker vs. centroid ≈ 0.85+. The
   0.85 project target requires centroid comparison, not per-clip. Split 1358 ElevenLabs clips
   679/679 (seed=42); ElevenLabs arm scored against held-out half to avoid self-comparison.

5. **ElevenLabs TTFA from published benchmarks**: Turbo v2.5 p50 = 264 ms, Flash v2.5 p50 = 288 ms,
   Picovoice mean = 335 ms. Already near the 300 ms threshold — direct measurement required. Kokoro
   GPU (A100) RTF ≈ 0.03 → first-chunk ~28–97 ms; H100 should be similar or faster.

6. **WER thresholds**: < 5% consumer-grade; > 10% hard failure; > 20% appropriate hard-flag
   threshold for short filler utterances (single-word error inflates WER on 3-word phrases).

7. **Warmup is required** (LESSONS.md Lesson 1): ≥ 1 discarded synthesis before timing. For
   ElevenLabs: ≥ 1 discarded streaming request with a persistent `requests.Session`.

8. **Version metadata required** (LESSONS.md Lesson 4): capture `torch.__version__`,
   `torch.version.cuda`, `kokoro` version, GPU model (`nvidia-smi`) into `metadata.json`.

9. **VM filler corpus is ephemeral** (LESSONS.md Lesson 10): confirm
   `/mnt/kikiri-tts/data/11labs_david/` exists before planning. Copy it, `dvc add`, `dvc push`
   before teardown.

10. **Resemblyzer minimum clip length**: skip clips < 1.6 s (less than one partial-utterance
    window); log skipped count per system.

* * *

## Best Approaches (top 3 recommended implementation approaches from research)

### Approach 1: Centroid-based speaker_sim with half-split reference

Build the ElevenLabs David centroid from 679 clips (seed=42), L2-normalize the mean embedding. Score
all systems against this centroid. Score ElevenLabs arm against the held-out 679 clips (not the
centroid) to avoid self-comparison. Use `resemblyzer.preprocess_wav` + `embed_utterance` on all
audio (Kokoro output at 24 kHz must be resampled to 16 kHz via resemblyzer's own preprocessing).

### Approach 2: Inline TTFB measurement with per-system pre-warming

For Kokoro: timestamp before `KPipeline.__call__()`, capture time of first yielded `audio_numpy`.
Pre-load all weights outside the timing loop. For ElevenLabs: use `stream=True`, timestamp before
the request, capture arrival of first non-empty chunk via `iter_content(chunk_size=1024)`. Discard ≥
1 warmup synthesis per system. Report p50/p95/p99 over all prompts (val_96 gives 96 data points;
fillers give ≥ 50 more).

### Approach 3: Duration-ratio gate before WER

Compute `len(synth_audio) / 24000 / reference_duration` first. Flag anything > 2.0 or < 0.5 for
manual review without running expensive Whisper transcription. Only clips that pass this gate go to
WER scoring. Use `faster-whisper` (base.en) with JiWER normalization (lowercase, strip punctuation,
expand numbers).

* * *

## Reusable Code / Assets

* `tasks/t0002_kokoro_v4_voicepack_decoder_package/code/extract_decoder_generic.py` — 38 lines,
  five-module extractor for raw `.pth` checkpoints → copy into task, adapt CKPT_DIR arg
* `tasks/t0003_kokoro_v5_phoneme_data/code/build_pipeline.py` — 30 lines, mandatory synthesis entry
  point → import directly (do not copy; lexicon must stay single source of truth)
* `tasks/t0002_kokoro_v4_voicepack_decoder_package/code/test_kokoro_inference_generic.py` — 56
  lines, KModel load + chunk-iteration pattern → copy as template, fix `lang_code="b"`
* `tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/best/` — v3 decoder + voicepack (DVC)
* `tasks/t0005_kokoro_v5_stage2_train/results/checkpoints/epoch_2nd_00003.pth.dvc` — t0005 best
* `tasks/t0006_kokoro_v5_stage2_subset/results/checkpoints/v6d/epoch_2nd_00006.pth.dvc` — t0006
* `tasks/t0003_kokoro_v5_phoneme_data/results/v5/val_list.txt` — 96-line val manifest

* * *

## Key Papers (top 5, with finding most relevant to this task)

* **Wan et al. 2018** — GE2E: 3-LSTM + 256-d d-vector; EER > 10% better than TE2E; this is the
  resemblyzer backbone and the speaker_sim metric foundation
* **Li et al. 2023 (StyleTTS 2)** — Five-module architecture (`bert`+`bert_encoder`+`predictor`
  +`text_encoder`+`decoder`); swapping decoder + voicepack is the correct fine-tune evaluation path
* **VERSA 2024** — WER warn=5%, fail=10% as evaluation starter thresholds; WavLM speaker sim
  correlates better with human perception than GE2E (deliberate divergence for project continuity)
* **AnalyzeSim 2025** — GE2E cosine can give misleading signals when audio is noisy/short; report
  both per-clip distribution and centroid score
* **Baseten 2026 (blog)** — Same-speaker per-clip ≈ 0.7, vs-centroid ≈ 0.85+; validates the
  centroid-comparison design for the 0.85 success criterion

* * *

## Risks Flagged in Research

* **t0005/t0006 raw checkpoints need packaging** (five-module extraction) before they can load into
  KModel. This is a required implementation step before any synthesis can run.
* **VM filler corpus may be absent** if `/mnt` was wiped. Fallback: regenerate from
  `fillers_from_logs.txt` via ElevenLabs API, but this costs ~$5 and takes time.
* **Short filler clips (< 1.6 s) unreliable for resemblyzer** — a non-trivial fraction of the 1358
  filler clips may be sub-threshold; speaker_sim sample size will shrink.
* **ElevenLabs TTFA already near the 300 ms boundary** — measured results could go either way; the
  project's success criterion may not be achievable by ElevenLabs itself.
* **val_loss does not predict perceptual quality** (t0001–t0006): t0005/t0006 best checkpoints
  (val_loss 0.848/0.846) were described as noisy by listening; the harness will provide the first
  objective speaker_sim and WER scores for them.
* **`lang_code` correction in copied code**: `test_kokoro_inference_generic.py` uses
  `lang_code="a"`; any copy must change this to `"b"` before use.

* * *

## Full Detail Available In

* `tasks/t0008_tts_eval_harness_baselines/research/research_papers.md` — 0 papers (corpus empty;
  partial status)
* `tasks/t0008_tts_eval_harness_baselines/research/research_internet.md` — 23 sources, 4 discovered
  papers (Wan2018, Li2023, VERSA2024, AnalyzeSim2025)
* `tasks/t0008_tts_eval_harness_baselines/research/research_code.md` — 5 tasks cited (t0002, t0003,
  t0005, t0006, t0007), 0 registered libraries
