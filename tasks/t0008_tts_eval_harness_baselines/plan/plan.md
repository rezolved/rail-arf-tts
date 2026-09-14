---
spec_version: "2"
task_id: "t0008_tts_eval_harness_baselines"
date_completed: "2026-09-14"
status: "complete"
---
# Plan — t0008_tts_eval_harness_baselines

## Objective

Build the project's first quantitative evaluation harness that scores eight TTS systems on speaker
similarity (GE2E cosine via resemblyzer), TTFB (time to first byte, p50/p95/p99), and RTF (real-time
factor) across two prompt sets (val_96 — 96 held-out clips — and ≥ 50 filler prompts sampled from
the 1358-clip ElevenLabs David corpus). The harness is delivered as a registered library asset
(`tts_eval_harness`) so every future fine-tuning task can reuse it without reimplementing scoring.

Systems scored: ElevenLabs David (API streaming), Kokoro base `bm_george`, Kokoro base `bm_lewis`,
Kokoro base decoder + v3 David voicepack, v3 shipped bundle (five-module decoder + voicepack), t0006
v6d best (epoch 6), t0005 run06 best (epoch 3), and a floor control (`af_heart`, female American).
All Kokoro arms synthesize through the mandatory `build_pipeline.py` entry point (`lang_code="b"`,
brand lexicon). TTFB and RTF for Kokoro are measured on GPU (LLM-T1-NC80 2×H100); speaker_sim and
WER run locally or on the VM.

**Done** when: all 8 systems have `speaker_sim`, `ttfb_ms` (p50/p95/p99), and `rtf` values in a
variant-format `metrics.json`; the main comparison table and two charts (speaker_sim box plot, TTFB
CDF) are in `results_detailed.md`; the library asset verificator passes; and a clear statement
records which systems (if any) pass each project success criterion.

## Task Requirement Checklist

> **Task name**: TTS evaluation harness and baselines
> 
> **Short description**: Build a reusable speaker_sim/TTFB/RTF harness and score ElevenLabs David,
> base Kokoro, v3 and the best t0005/t0006 checkpoints.
> 
> **Long description** (from `task_description.md`): Build the measurement tool once, as a reusable
> library, and use it to answer research questions 1–5 and place every existing checkpoint on the
> same scale. Systems 1–7 + floor control. Prompt sets: val_96 (96 clips) and ≥ 50 fillers. Scoring:
> speaker_sim (GE2E cosine, centroid-half split 679/679 seed=42), ttfb_ms (p50/p95/p99), rtf,
> duration_ratio (sanity), WER (sanity). Compute: H100 for Kokoro TTFB. Expected outputs: library
> asset, metrics.json (variant format), tables, charts, clear pass/fail statement per criterion.

* **REQ-1** — Score all 8 systems on `speaker_sim` (GE2E cosine vs ElevenLabs David centroid).
  Satisfied by: Steps 6–9 (speaker_sim measurement). Evidence: `results/metrics.json` variants,
  speaker_sim box-plot chart.

* **REQ-2** — Score all 8 systems on `ttfb_ms` (p50/p95/p99). ElevenLabs measured via streaming API;
  Kokoro via pipeline first-chunk timing on H100. Satisfied by: Steps 10–13 (TTFB measurement on
  VM). Evidence: `results/metrics.json` variants, TTFB CDF chart.

* **REQ-3** — Score all 8 systems on `rtf` (synthesis wall time / audio duration). Satisfied by:
  Steps 10–13. Evidence: `results/metrics.json` variants.

* **REQ-4** — Score `duration_ratio` (sanity metric) for all 8 systems to detect duration
  explosions. Satisfied by: Steps 6–9. Evidence: `results_detailed.md` table column.

* **REQ-5** — Score WER (sanity metric using Whisper base.en + JiWER) for all 8 systems to detect
  garbled/noisy audio. Satisfied by: Steps 6–9. Evidence: `results_detailed.md` table column.

* **REQ-6** — Answer Q1: ElevenLabs David speaker_sim, TTFB (p50/p95/p99), RTF on ≥ 50 fillers.
  Satisfied by: Steps 6–9, 10–13. Evidence: variant metrics entry `elevenlabs_david_fillers`.

* **REQ-7** — Answer Q2: base Kokoro-82M (`bm_george`, `bm_lewis`) scores on same prompts. Satisfied
  by: Steps 10–13. Evidence: variants `kokoro_base_george_*`, `kokoro_base_lewis_*`.

* **REQ-8** — Answer Q3: v3 shipped bundle speaker_sim vs threshold 0.85. Satisfied by: Step 7.
  Evidence: variant `kokoro_v3_bundle_*`.

* **REQ-9** — Answer Q4: Do t0005/t0006 best checkpoints score measurably worse than v3? Satisfied
  by: Steps 6–9. Evidence: variants `kokoro_t0005_best_*`, `kokoro_t0006_v6d_best_*`.

* **REQ-10** — Answer Q5: Is speaker_sim alone enough to detect broken audio, or are duration_ratio
  and WER also needed? Satisfied by: Step 9 (speaker_sim vs WER scatter plot per clip). Evidence:
  chart `results/images/speaker_sim_wer_scatter.png`.

* **REQ-11** — Copy ElevenLabs 1358 filler corpus from the VM, `dvc add`, `dvc push` (currently not
  DVC-tracked — LESSONS Lesson 10). Satisfied by: Step 2 (VM filler corpus copy). Evidence:
  `tasks/t0008_tts_eval_harness_baselines/data/11labs_david/` with `.dvc` pointer.

* **REQ-12** — Package t0005 and t0006 raw `.pth` checkpoints via five-module extraction before
  evaluation. Satisfied by: Step 3 (checkpoint packaging). Evidence: packaged `.pth` files.

* **REQ-13** — Deliver the harness as a registered library asset `tts_eval_harness` so future tasks
  can reuse it without reimplementing. Satisfied by: Steps 14–15 (library asset files). Evidence:
  `assets/library/tts_eval_harness/details.json` + `description.md`.

* **REQ-14** — Produce `results/metrics.json` in variant format with speaker_sim, ttfb_ms, rtf per
  system. Satisfied by: Step 16. Evidence: `results/metrics.json`.

* **REQ-15** — Produce main comparison table (rows = 8 systems + floor; columns = speaker_sim
  mean/std, ttfb p50/p95/p99, rtf, duration_ratio median, WER, delta vs ElevenLabs) plus
  per-prompt-set breakdown. Satisfied by: Step 16. Evidence: `results_detailed.md`.

* **REQ-16** — Produce two charts: speaker_sim distribution box plot (dashed line at 0.85) and TTFB
  CDF per system (dashed line at 300 ms). Third chart: speaker_sim vs WER scatter (REQ-10).
  Satisfied by: Step 16. Evidence: `results/images/`.

* **REQ-17** — State clearly which systems (if any) already pass each project success criterion
  (speaker_sim ≥ 0.85; TTFB ≤ 300 ms). Satisfied by: Step 16. Evidence: pass/fail summary in
  `results_detailed.md`.

* **REQ-18** — Record which t0005 `.pth` file was actually scored (ambiguity between "epoch 2" and
  "epoch 3" in t0005 logs; DVC pointer is `epoch_2nd_00003.pth`). Satisfied by: Step 3 (record
  actual file path and val_loss at scoring time). Evidence: `results/metadata.json`.

* **REQ-19** — Capture infrastructure versions (torch, CUDA, kokoro, GPU) at benchmark time (LESSONS
  Lesson 4). Satisfied by: Steps 10–13 (metadata capture on VM). Evidence: `results/metadata.json`.

* **REQ-20** — Follow warmup protocol: discard ≥ 1 warmup synthesis per system before recording
  latency (LESSONS Lesson 1). Satisfied by: Steps 10–13. Evidence: documented in
  `results/metadata.json`.

* **REQ-21** — Produce `results/per_clip_metrics.json` with per-clip speaker_sim, duration_ratio,
  WER, and TTFB for every system × prompt. Required by the `tts-benchmark-run` task type spec.
  Satisfied by: Steps 6–9, 10–13. Evidence: `results/per_clip_metrics.json`.

## Approach

### Technical Design

The harness is structured around five responsibilities: (1) prompt-set loading, (2) per-system
synthesis adapters, (3) scoring (speaker_sim, TTFB, RTF, WER, duration_ratio), (4) per-clip results
collection, and (5) metrics.json/chart writing.

All code lives in `tasks/t0008_tts_eval_harness_baselines/code/` as a flat set of Python modules:
`harness.py` (orchestrator + prompt loader), `adapters.py` (8 system adapters), `scoring.py`
(speaker_sim, TTFB, WER, RTF, duration_ratio), `extract_decoder.py` (checkpoint packager, adapted
from t0002), `report.py` (metrics.json + chart generation), and `run_eval.py` (CLI entry point).

**Checkpoint packaging** (from research_code.md): t0005/t0006 raw `.pth` files are multi-GPU
StyleTTS2 checkpoints with DDP prefix and weight-norm parametrization. Loading only the decoder
produces duration explosions. The five-module extraction script from t0002
(`tasks/t0002_kokoro_v4_voicepack_decoder_package/code/extract_decoder_generic.py`) handles this
correctly in 38 lines. It will be copied to `code/extract_decoder.py` and adapted to accept
arbitrary input/output paths as function arguments instead of hardcoded constants.

**Synthesis entry point** (from research_code.md): All Kokoro arms MUST call `build_pipeline(model)`
from `tasks/t0003_kokoro_v5_phoneme_data/code/build_pipeline.py`. This returns a `KPipeline` with
`lang_code="b"` and the brand lexicon (62 OOV words). Bare `KPipeline` or `lang_code="a"` are silent
failures. The harness imports `build_pipeline` directly (no copy needed — this repo is a monorepo
and t0003 is a completed task). The import path is
`from tasks.t0003_kokoro_v5_phoneme_data.code.build_pipeline import build_pipeline`.

**Speaker similarity design** (from research_summary.md): Split the 1358 ElevenLabs reference clips
into two halves of 679 each (seed=42). Half A is used to build the speaker centroid (mean GE2E
embedding). All systems including fine-tuned Kokoro variants are scored against the centroid from
half A. ElevenLabs David is scored against the held-out half B clips (to avoid self-comparison;
same-speaker per-clip cosine ≈ 0.7 but vs-centroid ≈ 0.85+). Clips < 1.6 s are skipped by
resemblyzer (less than one partial-utterance window); log skip count per system.

**TTFB measurement** (from research_summary.md, Lessons 1 & 4): For Kokoro on GPU, timestamp before
`KPipeline.__call__()`, record time of first yielded audio chunk. For ElevenLabs: use `stream=True`,
timestamp before request, capture first non-empty chunk via `iter_content(chunk_size=1024)`. Discard
≥ 1 warmup synthesis per system before recording. Capture `torch.__version__`, `torch.version.cuda`,
`kokoro` package version, and GPU model (`nvidia-smi --query-gpu=name --format=csv,noheader`) into
`results/metadata.json` before measurement begins.

**WER** (from research_summary.md): use `faster-whisper` (`base.en` model), JiWER normalization
(lowercase, strip punctuation, expand numbers). Apply duration-ratio gate first: clips with ratio
> 2.0 or < 0.5 are flagged for manual review and skipped by Whisper to avoid inflating WER from
> duration explosions.

**resemblyzer installation note**: per the `tts-benchmark-run` instruction, `resemblyzer` must NOT
be added to the main `dependencies` in `pyproject.toml`. Install it in a `[speaker-sim]` optional
extra only (webrtcvad/pkg_resources breaks on setuptools ≥ 81). The `[speaker-sim]` extra already
exists in the project if it doesn't, add it; install via `pip install '.[speaker-sim]'` on the VM.

### Alternatives Considered

**Alternative 1 — WavLM-based speaker similarity**: VERSA (2024) reports WavLM correlates better
with human perception than GE2E cosine (AnalyzeSim 2025 supports this). Rejected because the
project's registered `speaker_sim` metric is explicitly defined as GE2E cosine via resemblyzer, and
consistency with the 0.85 success criterion requires the same metric family.

**Alternative 2 — Run TTFB locally (CPU-only)**: cheaper and simpler; avoids the H100 VM cost.
Rejected because the project success criterion explicitly says "local inference, H100"; a CPU number
would not validate the criterion. Both CPU and H100 numbers will be reported, but only H100 counts
toward the success criterion.

**Alternative 3 — Write adapters as a single monolithic script per system**: simpler, avoids a
library structure. Rejected because the task explicitly requires a registered library asset reusable
by future training tasks. A flat script per system would not meet REQ-13.

### Task Types Applied

The task has two registered types: `tts-benchmark-run` and `baseline-evaluation`. The
`tts-benchmark-run` instruction requires a warmup protocol (Lessons 1, 2), per-clip
`per_clip_metrics.json`, and `resemblyzer` in an optional extra only. The `baseline-evaluation`
instruction requires variant-format metrics and efficiency documentation. Both types are applied
here: the warmup and per-clip file come from `tts-benchmark-run`; the variant metrics format and
comprehensive table come from `baseline-evaluation`.

## Cost Estimation

| Item | Quantity | Unit cost | Total |
| --- | --- | --- | --- |
| ElevenLabs API streaming requests (96 val + 100 fillers, 2 prompt sets) | ~200 requests × ~50 chars avg | $0.30/1000 chars | ~$0.30 |
| ElevenLabs API warmup requests (1 per session) | ~5 requests | negligible | ~$0.01 |
| LLM-T1-NC80 H100 VM (provisioning + Kokoro TTFB runs + teardown) | ~1.5 h | ~$13.96/h | ~$21 |
| DVC storage upload (filler audio ~650 MB) | 650 MB | negligible (Azure Blob) | ~$0 |
| **Total estimate** |  |  | **~$21.50** |

Project budget: $5000. Estimated prior spend: ~$370–440 (legacy `total_usd` field means
`aggregate_costs` reports $0, but the real spend is ~$370 recorded + ~$70 unrecorded t0006 GPU).
This task's ≤ $30 budget leaves ≥ $4560 remaining. Comfortable margin.

Note: the task budget spec requires writing `total_cost_usd` (not `total_usd`) in
`results/costs.json`.

## Step by Step

### Milestone 1: Environment Setup and Data Prep (local, no GPU)

1. **[CRITICAL] Confirm and copy the ElevenLabs filler corpus from the VM.** SSH into LLM-T1-NC80
   and check whether `/mnt/kikiri-tts/data/11labs_david/` exists (LESSONS Lesson 10 — `/mnt` is
   ephemeral on Azure ML). If present:
   `rsync -av LLM-T1-NC80:/mnt/kikiri-tts/data/11labs_david/ tasks/t0008_tts_eval_harness_baselines/data/11labs_david/`.
   Count files — expect 1358 WAVs. Then
   `dvc add tasks/t0008_tts_eval_harness_baselines/data/11labs_david/ && dvc push`. If absent:
   regenerate from the ElevenLabs API using the filler text list (`data/11labs_david_texts.txt` if
   it exists, else read from the DVC-tracked manifest from t0003). This fallback costs ~$5 more and
   adds ~30 min. Satisfies REQ-11.

   *Validation gate*: count WAV files under `data/11labs_david/`. If < 1000, halt — the corpus is
   truncated or absent; do not proceed to scoring with an incomplete reference set.

2. **Pull all required DVC-tracked checkpoints.** From the worktree root, run:
   `dvc pull tasks/t0005_kokoro_v5_stage2_train/results/checkpoints/epoch_2nd_00003.pth.dvc`,
   `dvc pull tasks/t0006_kokoro_v5_stage2_subset/results/checkpoints/v6d/epoch_2nd_00006.pth.dvc`,
   `dvc pull tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/best/`. Verify all three
   checkpoint paths exist after pull. Satisfies REQ-12 (prerequisite).

3. **Package t0005 and t0006 checkpoints via five-module extraction.** Copy
   `tasks/t0002_kokoro_v4_voicepack_decoder_package/code/extract_decoder_generic.py` to
   `tasks/t0008_tts_eval_harness_baselines/code/extract_decoder.py`. Adapt: replace hardcoded
   `CKPT_DIR` and `TASK_DIR` constants with a `def extract(ckpt_path: str, out_path: str) -> None:`
   function signature. Run on t0005:
   `python code/extract_decoder.py tasks/t0005_kokoro_v5_stage2_train/results/checkpoints/epoch_2nd_00003.pth tasks/t0008_tts_eval_harness_baselines/data/packaged/t0005_run06_epoch3.pth`
   and on t0006:
   `python code/extract_decoder.py tasks/t0006_kokoro_v5_stage2_subset/results/checkpoints/v6d/epoch_2nd_00006.pth tasks/t0008_tts_eval_harness_baselines/data/packaged/t0006_v6d_epoch6.pth`

   Record the actual filename scored for t0005 in `results/metadata.json` (REQ-18). Satisfies
   REQ-12.

   *Validation gate*: load each packaged `.pth` with
   `torch.load(path, map_location='cpu', weights_only=False)` and confirm keys are exactly
   `['bert', 'bert_encoder', 'predictor', 'text_encoder', 'decoder']`. If any key is missing, halt —
   the extraction is wrong.

4. **Sample filler prompt set (seed=42) and prepare prompt manifest.** Create
   `tasks/t0008_tts_eval_harness_baselines/data/filler_prompts_50.json` by: reading
   `data/11labs_david/` WAV filenames to extract texts (or a companion text file if one exists),
   then sampling exactly 100 clips at random (`random.seed(42)`) to give a generous evaluation set.
   Also read `tasks/t0003_kokoro_v5_phoneme_data/results/v5/val_list.txt` to extract val_96 texts
   (format: `wav_path|phonemes|speaker_id` — the raw text is the corresponding source CSV, so read
   `data/v4/val/` WAV files to find texts or use the manifest CSV). Write `data/val96_prompts.json`
   and `data/filler_prompts_100.json` (JSON array of `{"text": "...", "ref_wav": "path_or_null"}`).
   The fillers sample includes the matching reference WAV path for speaker_sim computation.
   Satisfies REQ-6 (prompt sets ready).

### Milestone 2: Harness Core Code (local)

5. **Write `code/scoring.py`.** This module provides four functions:

   * `compute_speaker_sim(synth_wavs: list[str], ref_embeddings: np.ndarray) -> dict`: takes a list
     of synthesized WAV paths and the precomputed centroid (or half-B embeddings for ElevenLabs),
     returns `{"mean": float, "std": float, "per_clip": list[float]}`. Uses
     `resemblyzer.VoiceEncoder` + `preprocess_wav` + `embed_utterance`. Skips clips < 1.6 s and logs
     skipped count.
   * `compute_wer(synth_wavs: list[str], texts: list[str]) -> dict`: uses `faster-whisper` with
     model `base.en` and JiWER normalization (lowercase, strip punct, expand numbers). Returns
     `{"mean_wer": float, "per_clip": list[float]}`. Only processes clips that pass the
     duration_ratio gate (0.5 ≤ ratio ≤ 2.0).
   * `compute_duration_ratio(synth_wavs: list[str], ref_durations: list[float]) -> dict`: reads each
     synth WAV length via `soundfile.info().duration` and divides by `ref_duration`. Returns
     `{"median": float, "per_clip": list[float]}`.
   * `build_centroid(wav_paths: list[str]) -> np.ndarray`: computes the L2-normalized mean GE2E
     embedding from the provided WAV files using `resemblyzer.VoiceEncoder`. Used in Step 6 to build
     the ElevenLabs reference centroid from half A.

   Satisfies infrastructure for REQ-1, REQ-4, REQ-5.

6. **Build ElevenLabs reference centroid and half-B split.** In `code/harness.py`, implement
   `build_reference_split(corpus_dir: str, seed: int = 42) -> tuple[np.ndarray, list[str]]` that:
   (1) lists all 1358 WAVs in `corpus_dir`, (2) shuffles with `seed=42`, (3) splits 679/679, (4)
   calls `scoring.build_centroid(half_a_wavs)`, (5) returns `(centroid, half_b_wavs)`. The centroid
   is the ElevenLabs David reference embedding used to score all non-ElevenLabs systems.
   `half_b_wavs` are used to score ElevenLabs David (to avoid self-comparison). Satisfies REQ-1
   design.

7. **Write `code/adapters.py` with 8 synthesis adapters.** Each adapter has the signature
   `def synth(text: str, **kwargs) -> tuple[np.ndarray, float, float]` returning
   `(audio_array_16khz, ttfb_s, rtf)`. All Kokoro adapters call `build_pipeline` from t0003:

   * `elevenlabs_david(text, api_key, voice_id, session)`: calls the ElevenLabs streaming endpoint
     `POST /v1/text-to-speech/{voice_id}/stream` with `Accept: audio/mpeg`, records wall time of
     first non-empty `iter_content(chunk_size=1024)` chunk. Returns decoded audio at 22050 Hz
     resampled to 16 kHz for resemblyzer. `voice_id` looked up at startup via `GET /v1/voices`
     filtered by name `David`.
   * `kokoro_george(text, pipeline)`: uses base `KPipeline` with `voice="bm_george"` (already set up
     via `build_pipeline(model=None, voice="bm_george")` or equivalent). Records first-chunk time.
   * `kokoro_lewis(text, pipeline)`: same as above with `voice="bm_lewis"`.
   * `kokoro_base_v3_voicepack(text, pipeline)`: base `KModel` (stock decoder) + loads
     `david_v3_best_voicepack.pt` by setting the `ref_s` parameter. Records first-chunk time.
   * `kokoro_v3_bundle(text, pipeline)`: uses packaged v3 five-module bundle
     (`tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/best/david_v3_best_decoder_kokoro.pth`)
     loaded via `KModel` + `david_v3_best_voicepack.pt`. Records first-chunk time.
   * `kokoro_t0006_v6d(text, pipeline)`: uses packaged t0006 five-module checkpoint
     (`data/packaged/t0006_v6d_epoch6.pth`). Records first-chunk time.
   * `kokoro_t0005_best(text, pipeline)`: uses packaged t0005 five-module checkpoint
     (`data/packaged/t0005_run06_epoch3.pth`). Records first-chunk time.
   * `kokoro_floor_control(text, pipeline)`: Kokoro base `af_heart` (female American). Records
     first-chunk time.

   All Kokoro adapters:
   * Call `build_pipeline` from `tasks.t0003_kokoro_v5_phoneme_data.code.build_pipeline` with
     `lang_code="b"` for David-voice systems. Floor control uses `lang_code="a"` (American).
   * Load model weights via `KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True)` then
     `torch.load(ckpt, map_location="cpu", weights_only=False)` and
     `model.load_state_dict(sd, strict=False)` for each module in the five-module dict.
   * Record `ttfb_s` as `time.perf_counter()` at call start minus time at first chunk yield.
   * Compute `rtf = wall_time_total / audio_duration_seconds`.

   Satisfies REQ-1–REQ-3 infrastructure.

8. **Write `code/run_eval.py` — the CLI entry point.** Arguments: `--systems` (list of system names;
   default all), `--prompt-set` (val96 | fillers | both), `--out-dir` (default `results/`),
   `--n-warmup` (default 1). Orchestrates: (1) load prompts, (2) for each system, run warmup
   synthesis (discard results, REQ-20), (3) run synthesis on all prompts, (4) collect per-clip
   (ttfb_s, rtf, audio_path, text) tuples, (5) call scoring functions, (6) write per-clip JSON to
   `results/per_clip_metrics.json`. Does NOT write `metrics.json` or charts — those are produced by
   `code/report.py` in Step 16.

   Satisfies REQ-21 (per_clip_metrics.json) and REQ-20 (warmup).

   *Validation gate (before full run)*: run
   `--systems elevenlabs_david --prompt-set val96 --limit 3` first. Verify the output WAV sounds
   like David's voice. Verify TTFB values are in the 100–1500 ms range (gross sanity). If TTFB is 0
   or > 5000 ms, halt — the timing is broken.

9. **Write `code/report.py` — metrics.json and chart writer.** Reads `results/per_clip_metrics.json`
   and writes:

   * `results/metrics.json` in explicit variant format: one variant per (system, prompt_set) pair,
     e.g. variant_id `elevenlabs_david_val96`, with
     `dimensions: {"system": "elevenlabs_david", "prompt_set": "val96"}` and
     `metrics: {"speaker_sim": ..., "ttfb_ms_p50": ..., "ttfb_ms_p95": ..., "ttfb_ms_p99": ..., "rtf": ...}`.
     Note: only `speaker_sim`, `ttfb_ms`, and `rtf` are registered project metrics. Duration_ratio
     and WER are sanity metrics reported in results_detailed.md but NOT written to metrics.json
     (only registered keys are allowed).
   * `results/images/speaker_sim_boxplot.png`: box-and-whisker per system, y-axis = GE2E cosine,
     horizontal dashed line at 0.85 (success criterion). One box per (system, prompt_set) or
     grouped. Satisfies REQ-16.
   * `results/images/ttfb_cdf.png`: ECDF per system, x-axis = ms, vertical dashed line at 300 ms.
     Use `matplotlib` step CDF. Satisfies REQ-16.
   * `results/images/speaker_sim_wer_scatter.png`: scatter plot, x = WER, y = speaker_sim, one point
     per synthesized clip, colored by system. Satisfies REQ-10, REQ-16.

   Satisfies REQ-14, REQ-16.

### Milestone 3: ElevenLabs Baseline (local, API)

10. **Run ElevenLabs David evaluation locally.** ElevenLabs requests do not need the H100 VM. Run
    `run_eval.py --systems elevenlabs_david --prompt-set both` locally. This performs ~200 streaming
    requests. Before the full run, validate the voice_id lookup from `/v1/voices` succeeds and
    returns an ID for "David". The warmup adapter discards 1 streaming request. Write ElevenLabs
    per-clip results to `results/per_clip_metrics_elevenlabs.json`. Cost: ~$0.30.

    *Validation gate*: after 10 requests, verify speaker_sim mean > 0.50. ElevenLabs David vs its
    own centroid should score high; a mean < 0.50 indicates a wrong voice_id or broken embedding.
    Halt and debug individual outputs if this fails.

    Satisfies REQ-2 (ElevenLabs TTFB), REQ-6 (Q1), REQ-1 (ElevenLabs speaker_sim).

### Milestone 4: Kokoro GPU Evaluation on H100 (remote)

11. **Set up the VM for Kokoro evaluation.** Via `setup-remote-machine` skill on LLM-T1-NC80: (a)
    SSH into the VM, verify the `kokoro-finetune` conda environment is active or install `kokoro`
    via `pip install kokoro`; (b) verify `resemblyzer` and `faster-whisper` are installed
    (`pip install resemblyzer faster-whisper`); (c) sync task code to VM:
    `rsync -av tasks/t0008_tts_eval_harness_baselines/code/ LLM-T1-NC80:~/t0008/code/`; (d) copy
    packaged checkpoint files and DVC-pull v3 bundle on VM; (e) capture metadata to
    `results/metadata.json`: `torch.__version__`, `torch.version.cuda`, `kokoro.__version__`,
    `nvidia-smi --query-gpu=name --format=csv,noheader`, run date. Satisfies REQ-19.

12. **Smoke-gate before Kokoro measurement (LESSONS Lesson 2).** Run one synthesis request on the VM
    for `kokoro_george` (simplest adapter, no custom checkpoint). If it fails or produces silence,
    halt and write an intervention file — do not proceed to TTFB measurement with a broken pipeline.
    Satisfies REQ-20 prerequisite.

13. **[CRITICAL] Run Kokoro TTFB/RTF measurement on VM for all 7 Kokoro systems.** Execute
    `run_eval.py --systems kokoro_george kokoro_lewis kokoro_base_v3_voicepack kokoro_v3_bundle kokoro_t0006_v6d kokoro_t0005_best kokoro_floor_control --prompt-set both --n-warmup 1`
    on the VM. This covers val_96 (96 prompts) and filler prompts (100 prompts) = ~196 × 7 = 1372
    synthesis calls. Estimated wall time: 1372 × 0.1 s per synthesis = ~140 s synthesis + overhead ≈
    30 min. Download `per_clip_metrics_kokoro.json` to local. Satisfies REQ-2, REQ-3, REQ-7, REQ-8,
    REQ-9.

    *Validation gate*: after v3 bundle runs, verify speaker_sim mean > 0.70. v3 is the gold
    reference checkpoint; a score < 0.70 indicates a loading error or wrong voicepack. Halt if this
    fails — do not continue to t0005/t0006.

    *Failure condition*: if any system produces audio with duration_ratio > 5.0 for > 10% of clips,
    halt that system's run — a duration explosion indicates wrong checkpoint loading (see t0002
    Lesson on five-module packaging).

14. **Run speaker_sim and WER scoring on VM-synthesized audio.** On local machine (or VM if faster):
    run `scoring.compute_speaker_sim` and `scoring.compute_wer` on all downloaded WAV files. Merge
    results into `results/per_clip_metrics.json` (combining ElevenLabs + Kokoro entries). Satisfies
    REQ-1, REQ-4, REQ-5.

### Milestone 5: Library Asset and Final Outputs

15. **Register the library asset.** Write
    `tasks/t0008_tts_eval_harness_baselines/assets/library/tts_eval_harness/details.json`
    (spec_version `"2"`, library_id `tts_eval_harness`) with module_paths listing all 5 code
    modules, entry_points for the main public functions (`run_eval`, `build_centroid`,
    `compute_speaker_sim`, `compute_wer`), and categories `["tts", "evaluation", "library"]`. Write
    `description.md` with all 8 mandatory sections. Run
    `uv run python -m arf.scripts.verificators.verify_library_asset --task-id t0008_tts_eval_harness_baselines tts_eval_harness`
    and fix all errors. Satisfies REQ-13.

16. **Write `results/per_clip_metrics.json` and run `code/report.py`.** Merge all per-clip results
    (ElevenLabs + Kokoro) into a single `results/per_clip_metrics.json` (array of JSON objects with
    fields: `system`, `prompt_set`, `text`, `ttfb_ms`, `rtf`, `speaker_sim`, `duration_ratio`,
    `wer`, `audio_path`). Run `python code/report.py` to produce `results/metrics.json` (explicit
    variant format, registered keys only), all three charts, and write the main comparison table to
    `results/tables.json` for use by results_detailed.md. Satisfies REQ-14, REQ-15, REQ-16, REQ-17,
    REQ-21.

17. **Write test for the harness.** Create `code/test_harness.py` with a `pytest` test that: (a)
    calls `scoring.compute_duration_ratio` on two known-length WAVs and asserts correct ratio; (b)
    calls `scoring.compute_wer` on a trivially simple transcript pair and asserts WER = 0.0; (c)
    verifies `report.py` produces valid variant-format JSON when given a toy per_clip dict. Run
    `uv run pytest code/test_harness.py -v` and confirm all pass. This is the minimum smoke test for
    the harness logic. Satisfies library-quality requirement.

## Remote Machines

Required: **LLM-T1-NC80** (Azure ML, 2×H100 SXM5). Needed for Kokoro TTFB/RTF measurement on GPU
(project success criterion: "local inference, H100"). GPU VRAM: 80 GB × 2; Kokoro-82M fits in < 1
GB. Estimated runtime: ≤ 1.5 h total including setup, synthesis, and teardown.

ElevenLabs evaluation runs locally — no GPU needed.

Pool config at `project/azure_vm.json`. Managed via the `setup-remote-machine` skill. Teardown step
10 follows immediately after Kokoro evaluation to stop billing.

## Assets Needed

* **t0006 v3 reference bundle** (DVC):
  `tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/best/ david_v3_best_decoder_kokoro.pth`
  (327 MB) and `david_v3_best_voicepack.pt` (523 KB). From dependency t0006.
* **t0006 v6d best checkpoint** (DVC):
  `tasks/t0006_kokoro_v5_stage2_subset/results/checkpoints/ v6d/epoch_2nd_00006.pth` (1.86 GB). From
  dependency t0006.
* **t0005 run06 best checkpoint** (DVC):
  `tasks/t0005_kokoro_v5_stage2_train/results/checkpoints/ epoch_2nd_00003.pth` (1.87 GB). From
  t0005 (transitive dependency of t0006).
* **ElevenLabs 1358 filler corpus**: currently on VM at `/mnt/kikiri-tts/data/11labs_david/`. Must
  be copied and DVC-tracked as part of Step 1 (REQ-11). ~650 MB.
* **val_96 prompt texts**: from `tasks/t0003_kokoro_v5_phoneme_data/results/v5/val_list.txt` (96
  lines, format `wav_path|phonemes|speaker_id`) + source manifests from `data/v4/val/`.
* **t0003 `build_pipeline.py`** (import path):
  `tasks/t0003_kokoro_v5_phoneme_data/code/ build_pipeline.py` — imported directly, not copied.
* **t0002 `extract_decoder_generic.py`** (copy as template):
  `tasks/t0002_kokoro_v4_voicepack_decoder_package/ code/extract_decoder_generic.py` — copied to
  `code/extract_decoder.py`.
* **ElevenLabs API key**: from project `.env` (`ELEVENLABS_API_KEY`).

## Expected Assets

* **Library asset** `tts_eval_harness` — the evaluation harness Python library. Produces:
  `tasks/t0008_tts_eval_harness_baselines/assets/library/tts_eval_harness/details.json` and
  `description.md`. Modules: `code/harness.py`, `code/adapters.py`, `code/scoring.py`,
  `code/extract_decoder.py`, `code/report.py`, `code/run_eval.py`. Matches
  `expected_assets: {"library": 1}` in `task.json`.

## Time Estimation

| Phase | Estimated wall time |
| --- | --- |
| Step 1 — VM filler corpus copy + DVC | 30 min (if present) |
| Steps 2–4 — DVC pull, checkpoint packaging, prompt prep | 20 min |
| Steps 5–9 — Core harness code (5 modules) | 3–4 h |
| Step 10 — ElevenLabs evaluation (local) | 30 min |
| Step 11–12 — VM setup + smoke gate | 30 min |
| Step 13–14 — Kokoro GPU runs + scoring | 1.5 h (on VM) |
| Steps 15–17 — Library asset + report + tests | 1 h |
| **Total** | **~7–8 h** |

## Risks & Fallbacks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| VM filler corpus absent (`/mnt` wiped) | Medium | Delays ~30 min + ~$5 | Regenerate from filler text list via ElevenLabs API (Step 1 fallback) |
| Short clips (< 1.6 s) — large fraction of 1358 corpus | Low–Medium | speaker_sim sample shrinks | Log skip count; if > 30% skipped, report in results as limitation |
| ElevenLabs voice_id for "David" not found | Low | Blocks ElevenLabs evaluation | Fall back to listing all voices and picking closest match by name; document voice_id used |
| t0005/t0006 checkpoints not pullable via DVC (remote unavailable) | Low | Cannot evaluate those variants | Mark variants as null, document as limitation, proceed with remaining 6 systems |
| `lang_code` bug in copied code (from `"a"` to `"b"`) | Medium | Silent quality degradation | Code review gate: `grep -r 'lang_code' code/` before running any Kokoro synthesis; must show only `"b"` for David systems |
| Duration explosion on t0005/t0006 packaged checkpoints | Low–Medium | Invalid evaluation | Duration-ratio gate (> 5.0 on > 10% of clips) halts that system; re-check extraction |
| H100 VM quota unavailable | Low | Delays GPU runs | Wait for VM from pool; CPU TTFB run can be reported separately |
| ElevenLabs API rate-limiting during 200 requests | Low | Delays | Add `time.sleep(0.5)` between requests; exponential backoff on 429 |
| `resemblyzer` installation fails (webrtcvad conflict) | Low | Blocks speaker_sim | Install in isolated env: `pip install resemblyzer --no-deps` + `pip install numpy webrtcvad` separately |

## Verification Criteria

* `uv run python -m arf.scripts.verificators.verify_plan t0008_tts_eval_harness_baselines` — exits
  0, zero errors.

* `uv run python -m arf.scripts.verificators.verify_library_asset --task-id t0008_tts_eval_harness_baselines tts_eval_harness`
  — exits 0, zero errors. Confirms REQ-13.

* `python -c "import json; d=json.load(open('tasks/t0008_tts_eval_harness_baselines/results/metrics.json')); v=d['variants']; assert len(v)>=16, f'expected 16+ variants, got {len(v)}'"`
  — confirms at least one variant per (system × prompt_set) pair for 8 systems × 2 prompt sets.
  Confirms REQ-14.

* `python -c "import json; d=json.load(open('tasks/t0008_tts_eval_harness_baselines/results/per_clip_metrics.json')); assert len(d)>=800, f'expected 800+ clips, got {len(d)}'"`
  — confirms ≥ 800 per-clip records (8 systems × 100 prompts; val96+fillers). Confirms REQ-21.

* `ls tasks/t0008_tts_eval_harness_baselines/results/images/*.png | wc -l` — outputs at least 3
  (speaker_sim_boxplot.png, ttfb_cdf.png, speaker_sim_wer_scatter.png). Confirms REQ-16.

* `uv run pytest tasks/t0008_tts_eval_harness_baselines/code/test_harness.py -v` — exits 0, all
  tests pass. Confirms harness logic is correct (REQ-13 quality).

* `python -c "import json; d=json.load(open('tasks/t0008_tts_eval_harness_baselines/results/per_clip_metrics.json')); systems={r['system'] for r in d}; assert 'elevenlabs_david' in systems and 'kokoro_v3_bundle' in systems, f'missing systems: {systems}'"`
  — confirms both ElevenLabs and v3 bundle results are present. Confirms REQ-6 and REQ-8.

## Rejection Criteria

These conditions invalidate results for the named system and must not be retroactively relaxed:

* **Low request success rate**: if `successful_requests / total_requests < 0.8` for any system, that
  system's results are null regardless of measured values (LESSONS Lesson 3).
* **Duration explosion**: if `duration_ratio > 5.0` for > 10% of clips in any Kokoro variant, that
  variant's results are null — the checkpoint was loaded incorrectly.
* **ElevenLabs wrong voice**: if `speaker_sim_mean < 0.50` for ElevenLabs David on half-B reference
  clips, the evaluation used the wrong voice — null that system and re-check voice_id.
* **Insufficient reference corpus**: if fewer than 1000 ElevenLabs reference WAVs are available
  after Step 1, halt the entire evaluation — the centroid cannot be reliably built.
* **Warmup bypass**: if the warmup protocol was not followed (n_warmup = 0), TTFB results are null
  for that run (LESSONS Lesson 1).
