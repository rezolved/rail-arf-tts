---
spec_version: "1"
task_id: "t0008_tts_eval_harness_baselines"
research_stage: "papers"
papers_reviewed: 0
papers_cited: 0
categories_consulted: []
date_completed: "2026-09-14"
status: "partial"
---
## Task Objective

**Status note**: This document is marked `partial` because the project paper corpus is empty at this
stage — no papers have been downloaded and catalogued in prior tasks. The research-internet step
(step 5) will search for and add relevant papers; see `research/research_internet.md` for the
literature findings that inform planning.

This task builds a reusable evaluation harness to score all available TTS systems — ElevenLabs David
(API), Kokoro-82M base voices (bm_george, bm_lewis), the v3 shipped bundle, the t0006 v6d epoch 6
checkpoint, and the t0005 run06 best checkpoint — on three metrics: speaker_sim (GE2E cosine
similarity against the ElevenLabs David reference set via resemblyzer), ttfb_ms (time to first audio
byte, p50/p95/p99), and rtf (real-time factor). It also computes two sanity metrics — duration_ratio
and WER — to detect broken audio that might still produce a plausible speaker_sim. The harness is
packaged as a reusable library asset so future training tasks can report the same metrics without
reimplementing the scoring pipeline.

The task answers five research questions: (1) what are ElevenLabs David's speaker_sim, TTFB, and RTF
baselines? (2) where do base Kokoro voices sit on the same scale? (3) how close is the v3 bundle to
the ≥ 0.85 GE2E cosine success criterion? (4) do the t0005/t0006 checkpoints score measurably worse
than v3, validating the subjective "noisy audio" observation? (5) is speaker_sim alone sufficient to
detect broken audio, or are duration_ratio and WER needed as well?

## Category Selection Rationale

No categories exist in `meta/categories/` at this stage of the project — the category registry is
empty. Consequently no category-filtered paper query could return results, and the
`categories_consulted` list is empty.

Categories that would be relevant if populated include: `text-to-speech`, `speaker-verification`,
`speech-synthesis`, `neural-codec`, `latency-benchmarking`, and `evaluation-methodology`. The
research-internet step (step 5) is expected to discover papers in these areas and add them to the
corpus, after which future tasks can consult them.

Excluded topics that are explicitly out of scope for this project: `speech-recognition` (STT is a
separate project), `multilingual-tts`, `multi-speaker-generalization`, and `streaming-protocol`
changes.

## Key Findings

### No Papers in Corpus

The project paper corpus contains zero downloaded papers. All six prior tasks (t0001-t0006) were
implementation and training tasks that did not include a research-papers step with actual paper
downloads. As a result, there is no literature to synthesize here.

The research-internet step will search for and add papers on the following topics directly relevant
to this task:

* GE2E speaker embedding (Wan et al., 2018) — the speaker_sim metric is defined as GE2E cosine
  similarity; understanding the embedding model is essential for interpreting the metric's range,
  variance, and threshold semantics.
* resemblyzer — the Python library wrapping GE2E for practical speaker similarity computation; its
  embedding dimensionality (256-d), normalization, and batch behavior directly affect the scoring
  pipeline design.
* Kokoro-82M and StyleTTS2 — the base model architecture; architecture details affect which
  components are swapped per evaluation system (decoder weights vs. voicepack vs. both).
* TTFB measurement methodology for streaming TTS — LESSONS Lesson 1 (discard warmup) and Lesson 4
  (record GPU/driver versions) are already project knowledge, but published evaluation protocols
  would ground the methodology.
* WER as a TTS quality proxy — whisper-based WER is used as a sanity metric; published work on WER
  thresholds for detecting garbled synthesis would inform the pass/fail threshold.

## Methodology Insights

Because no papers are available in the corpus, methodology insights are drawn entirely from
project-internal knowledge (LESSONS.md, prior task results, and task_description.md):

* **GE2E centroid split**: the task description specifies splitting the 1358 ElevenLabs David
  reference clips into a centroid half and a held-out half with a fixed seed (seed=42), so the
  ElevenLabs arm is not compared against its own centroid. This is the correct resemblyzer usage
  pattern for fair baseline evaluation.
* **Warmup discards**: LESSONS Lesson 1 mandates discarding warmup requests before measuring TTFB.
  The harness must send at least one warmup request per system before timing begins.
* **Version recording**: LESSONS Lesson 4 mandates recording torch/CUDA/kokoro versions and GPU
  model at measurement time, so TTFB/RTF numbers are reproducible.
* **Pipeline uniformity**: all Kokoro arms must synthesize through
  `tasks/t0003_kokoro_v5_phoneme_data/code/build_pipeline.py` with `lang_code="b"` and the brand
  lexicon. Bare `KPipeline` calls silently change the token stream (established in t0003).
* **Floor control**: including one clearly different Kokoro voice (e.g., `af_heart`, female
  American) as a floor control is essential for interpreting what a "wrong speaker" score looks like
  and contextualising the ≥ 0.85 threshold.
* **Whisper WER**: transcribing synthesized audio with a local Whisper model and comparing against
  prompt text catches garbled audio that would otherwise produce a plausible speaker_sim score.

Once the research-internet step adds papers on GE2E and resemblyzer, the methodology insights
section of `research_internet.md` should be consulted alongside this document during planning.

## Gaps and Limitations

* **No published TTFB benchmarks for Kokoro-82M**: no prior task and no paper in the corpus reports
  TTFB or RTF for Kokoro-82M on H100 hardware. The harness will produce the first such measurements
  for this project. Without published comparators, the 300 ms TTFB threshold comes entirely from the
  project success criterion, not from empirical survey of what is achievable.
* **No GE2E threshold calibration study in corpus**: the 0.85 GE2E cosine threshold is a project
  design choice. Whether 0.85 is a reliable boundary between "perceptually same speaker" and
  "perceptually different speaker" is not grounded in a paper in this corpus. The floor control
  system is the only internal calibration mechanism.
* **Empty category registry**: the lack of categories means cross-task paper discovery by category
  is impossible at this stage. Future tasks that add papers should also add relevant categories to
  `meta/categories/` so the category-based discovery path becomes useful.
* **No WER threshold literature**: the task uses WER as a binary sanity filter for garbled audio,
  but no paper in the corpus establishes what WER threshold distinguishes "acceptable" from "broken"
  synthesis for short filler utterances. The research-internet step should specifically look for
  this.
* **val_loss vs. perceptual quality**: t0003 established that Stage 1 val_loss does not predict
  success. Whether this generalises to Stage 2 val_loss is still unknown; the harness results will
  be the first empirical test of this question for Stage 2.

## Recommendations for This Task

1. **Proceed to research-internet immediately**: the most valuable action is to add GE2E/resemblyzer
   papers and any published TTFB benchmarking methodology papers to the corpus. The planning step
   should wait for research_internet.md before finalising the scoring pipeline design.
2. **Use seed=42 for the centroid/held-out split** as specified in the task description; document
   the exact split indices in the harness library so results are reproducible across tasks.
3. **Record all environment metadata at measurement time**: torch version, CUDA version, Kokoro
   version, GPU model (from `nvidia-smi`), and timestamp. Store these in `results/metadata.json` so
   TTFB/RTF numbers can be reproduced or contextualised against future measurements.
4. **Include a floor control in every run**: `af_heart` (female American) should be synthesised on
   all prompt sets alongside the David-voice systems so the ≥ 0.85 threshold can be contextualised.
5. **Design the library with version-tagging**: the harness will be reused by future training tasks.
   Pin the resemblyzer version and the GE2E model checkpoint so speaker_sim scores are comparable
   across tasks.

## Paper Index

*(Empty — no papers in corpus. See `research/research_internet.md` for literature added during step
5.)*
