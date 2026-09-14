---
spec_version: "1"
task_id: "t0008_tts_eval_harness_baselines"
research_stage: "internet"
searches_conducted: 11
sources_cited: 23
papers_discovered: 4
date_completed: "2026-09-14"
status: "complete"
---
## Task Objective

Build a reusable evaluation harness to score ElevenLabs David (API), Kokoro-82M base voices
(bm_george, bm_lewis), the v3 shipped bundle, the t0006 v6d epoch-6 checkpoint, and the t0005 run06
best checkpoint on three primary metrics — speaker_sim (GE2E cosine via resemblyzer against
ElevenLabs David references), ttfb_ms (time to first audio byte, p50/p95/p99), and rtf (real-time
factor) — plus two sanity metrics (duration_ratio, WER). The harness is packaged as a reusable
library. This task produces the first objective measurements of any success criterion in this
project (GE2E cosine ≥ 0.85, TTFB ≤ 300 ms).

## Gaps Addressed

From `research_papers.md` Gaps and Limitations:

1. **No published TTFB benchmarks for Kokoro-82M** — **Partially resolved**. Community benchmarks
   and a 2025 preprint now exist. GPU TTFB on A100/RTX-class hardware: RTF ≈ 0.03 (i.e., 10 s audio
   synthesized in ~300 ms total), with first-chunk latency as low as 28 ms on RTX 5090 and a
   published 97 ms baseline TTFB [GigaGPU-Kokoro]. CPU baseline (Picovoice benchmark on AMD Ryzen 7
   5700X): 3,658 ms FTTS [Picovoice-TTS] — far above the 300 ms target, confirming H100 is required.
   No peer-reviewed H100-specific numbers exist; the harness will generate the first such
   measurement.

2. **No GE2E threshold calibration study in corpus** — **Resolved**. Multiple sources now establish
   the expected cosine-similarity ranges. Same-speaker per-clip embeddings agree at roughly 0.7
   [Baseten-FineTune]; against the centroid (mean of many clips) this rises to 0.85+
   [Baseten-FineTune]. Different-speaker scores typically fall around 0.3–0.6 [ResemblyzerSearch]. A
   resemblyzer authentication study found an empirical threshold of 0.84 for English voices
   [Resemble-AuthStudy]. The project's 0.85 target aligns with the same-speaker-vs-centroid expected
   range and is a defensible choice.

3. **No WER threshold literature for TTS** — **Resolved**. Industry practice: WER < 5% is
   consumer-grade; WER ≥ 10% is a hard failure threshold indicating broken synthesis
   [Inworld-OpenEval]. WER > 100% signals a fundamentally broken system (excessive
   insertions/deletions) [Deepgram-PER]. The open-tts-eval toolkit uses WER warn=0.05 / fail=0.10 as
   starter thresholds; these are domain-calibrated, not universal [Inworld-OpenEval]. For short
   filler utterances the effective WER threshold may need to be higher due to ASR sensitivity to
   out-of-vocabulary proper nouns.

4. **Empty category registry** — **Unresolved** (out of scope for this task). Categories must be
   added to `meta/categories/` in a separate infrastructure task.

## Search Strategy

**Sources searched**: Google Search (via WebSearch), GitHub repository READMEs and issues, Hugging
Face model cards, arXiv, PapersWithCode, ElevenLabs documentation, Picovoice benchmark
documentation, Preprints.org.

**Queries executed** (11 total):

*Gap-targeted queries:*

1. `GE2E generalized end-to-end loss speaker verification embedding Wan 2018`
2. `resemblyzer speaker similarity cosine threshold calibration verification`
3. `Kokoro-82M TTS TTFB latency benchmark evaluation H100 GPU inference`
4. `WER word error rate threshold TTS quality evaluation broken audio detection`
5. `ElevenLabs streaming API TTFB latency measurement milliseconds 2025 2026`

*Broadening queries:*

6. `StyleTTS2 speaker similarity evaluation cosine GE2E metric benchmark`
7. `hexgrad Kokoro-82M model architecture StyleTTS2 voicepack training evaluation 2025`
8. `streaming TTS TTFB measurement methodology warmup requests benchmark reproducible`
9. `TTS evaluation harness speaker similarity WER latency reproducible open source Python 2025`
10. `speaker similarity evaluation TTS fine-tuning cosine same speaker different speaker range values`

*Snowball queries:*

11. `GE2E speaker verification LSTM 3 layers 256 dimensions d-vector cosine similarity threshold EER`

**Date range**: 2018–2026, no exclusion for foundational GE2E references.

**Inclusion criteria**: Sources providing (a) GE2E/resemblyzer architecture and threshold data, (b)
published TTFB/RTF numbers for Kokoro or ElevenLabs, (c) WER thresholds for TTS quality, (d)
benchmark methodology for streaming TTS latency measurement. Excluded: non-English TTS,
multi-speaker generalization research not applicable to fine-tuning, server infrastructure papers.

**Search iterations**: Queries 6–10 were broadening searches triggered after gap queries confirmed
the paper corpus is empty. Query 11 was a snowball triggered by finding that GE2E architecture
specifics (LSTM layers, d-vector dimensions) were only partially described in the initial results.

**WebFetch deep-reads** (6 fetches; pages fetched): ElevenLabs latency docs, resemblyzer README,
Kokoro-FastAPI stream example, Coval benchmark methodology, Picovoice TTS benchmark docs, Baseten
fine-tuning blog.

## Key Findings

### GE2E Embeddings and Resemblyzer: Architecture and Thresholds

The GE2E model (Wan et al., 2018) introduced in [Wan2018] consists of 3 LSTM layers followed by a
linear projection layer that outputs 256-dimensional L2-normalized d-vectors. It reduces speaker
verification EER by more than 10% versus the prior TE2E loss while cutting training time by 60%
[Wan2018]. Resemblyzer [Resemble-GH] wraps GE2E with a convenient Python API: `preprocess_wav`
normalizes volume, trims silence, and resamples; `embed_utterance` returns the L2-normed mean of
1.6-second partial-utterance embeddings (minimum rate 0.625 partial utterances per second). The
output is a 256-dimensional unit vector.

**Cosine similarity ranges (non-peer-reviewed but empirically grounded)**:

* Same-speaker, per-clip comparison: typically ~0.7 [Baseten-FineTune]
* Same-speaker, clip vs. centroid (mean of many clips): 0.85+ [Baseten-FineTune]
* Different-speaker: typically 0.3–0.6 [ResemblyzerSearch]
* Empirical English-voice authentication threshold: 0.84 [Resemble-AuthStudy] (not peer-reviewed)
* Reliable authentication requires audio clips ≥ 2.63 s and ≥ 495 KB [Resemble-AuthStudy]

**Best practice — centroid comparison**: comparing synthesized audio against the centroid of many
reference clips (rather than a single clip) is more stable. A speaker self-comparison against its
own centroid achieves 0.85+ versus the noisier ~0.7 per-clip agreement [Baseten-FineTune]. This
directly validates the task description's design choice to split the ElevenLabs reference clips into
a centroid half and a held-out half: the centroid half builds a stable reference, and the held-out
half provides the scoring baseline for ElevenLabs itself.

**Hypothesis**: If the project's 0.85 threshold is met on the centroid comparison, per-clip scores
for individual synthesized clips will often fall below 0.85 — the task should report both the mean
against centroid and the per-clip distribution to avoid false pass/fail calls.

**Warning on audio length**: Resemblyzer is unreliable for very short clips. Filler utterances in
the ElevenLabs corpus are typically 1–5 seconds. Clips under 1.6 s (one partial-utterance window)
should be flagged and excluded from speaker_sim scoring or handled with care.

### Kokoro-82M Architecture: StyleTTS2 + ISTFTNet Decoder-Only

Kokoro-82M [Kokoro-HF] is built on StyleTTS2 [Li2023] and ISTFTNet [Kaneko2022]. It uses a
decoder-only architecture (no diffusion module, no encoder release) trained on a few hundred hours
of permissive audio. The G2P pipeline is `misaki`. Voicepacks are style tensors that define the
speaker embedding; swapping the voicepack tensor changes the voice without reloading the decoder
weights. This is directly relevant to evaluation systems 3–7: system 3 swaps only the voicepack
while keeping the base decoder, while systems 5–7 swap both decoder and voicepack.

Audio output is 24 kHz. The KPipeline API yields
`(grapheme_sequence, phoneme_sequence, audio_numpy)` tuples — TTFB measurement must timestamp the
first `audio_numpy` yield, not the return from `KPipeline()`.

**Best practice**: Always instantiate `KPipeline(lang_code="b")` for British English with brand
lexicon (confirmed by t0003). A bare `KPipeline` defaults to American English and silently changes
the phoneme stream.

### Kokoro-82M GPU TTFB and RTF: Published Numbers

Community benchmarks (non-peer-reviewed) consistently report:

* RTF ≈ 0.03 on A100 (10-second clip in ~300 ms total synthesis time) [GigaGPU-Kokoro]
* RTF ≈ 0.03–0.04 on RTX 4090; 30 s audio generated in 2.1 s on RTX 3090 [GigaGPU-TTS]
* First-audio latency as low as 28 ms on RTX 5090 (self-hosted via FastAPI) [Forasoft-Stream]
* 97 ms baseline TTFB cited by Together.ai for their hosted Kokoro endpoint [Together-Kokoro]

CPU (Picovoice benchmark on AMD Ryzen 7 5700X, no GPU, 200 taskmaster2 prompts):

* FTTS (First Token to Speech) = **3,658 ms** [Picovoice-TTS] — approximately 12× above the 300 ms
  target, confirming GPU is required

**Important caveat**: all GPU numbers are for stock voices from a single CUDA process. The project's
fine-tuned checkpoints run from custom `.pth` weights through a modified pipeline; first-chunk
latency for fine-tuned Kokoro may differ from stock Kokoro due to different model load paths and
possible extra preprocessing steps. The harness must measure each system independently.

**Best practice — warmup**: all benchmarks that report reliable TTFB discard at least one warmup
request per system before timing begins [Coval-Methodology], [Picovoice-TTS]. The warmup amortizes
model weight loading, CUDA kernel compilation, and TCP/TLS setup for API-based systems. The harness
should send ≥1 warmup synthesis per system before collecting timing measurements.

### ElevenLabs Streaming API TTFB: Published Numbers

ElevenLabs documentation [ElevenLabs-Latency] distinguishes:

* **Model inference**: ~75 ms for Flash models on typical short inputs (internal, excludes network)
* **TTFA (API streaming)**: ~478 ms average with PCM streaming REST API (includes network
  round-trip)
* **End-to-end for voice agents**: median 1,424 ms, p95 1,768 ms (full voice agent pipeline)
  [Openbenchmarks-EL]

Independent benchmarks [Forasoft-Stream]:

* ElevenLabs Turbo v2.5: **264 ms median TTFA**, 28 ms IQR (Coval benchmark, May 2026)
* ElevenLabs Flash v2.5: **288 ms median TTFA**, 28 ms IQR (same benchmark)

Picovoice benchmark [Picovoice-TTS]:

* ElevenLabs Streaming: **335 ms** average FTTS (AMD Ryzen 7 host, network included)
* ElevenLabs standard (non-streaming): 1,470 ms

**For this project**: ElevenLabs is measured for real-world TTFB via its streaming API to model the
actual production latency the Rezolve voice assistant experiences. The ~264–335 ms range from
published benchmarks is already at the 300 ms target edge — whether specific David voice+Flash model
combinations stay under 300 ms needs direct measurement.

**Best practice — ElevenLabs TTFB measurement**:

1. Use the streaming API with `stream=True`, timestamping the moment the first audio chunk arrives.
2. Use a pre-warmed `requests.Session` to amortize TCP+TLS (excludes ~80–200 ms handshake overhead).
3. Discard 1–2 warmup requests before collecting measurements.
4. Report p50/p95/p99 over ≥30 prompts.

### WER as a Sanity Metric: Thresholds and Caveats

WER, computed by transcribing synthesized audio with Whisper and comparing against the prompt text,
is a standard intelligibility proxy for TTS quality [NVIDIA-Riva], [Inworld-OpenEval]. Key
thresholds from community practice:

* WER < 5% → consumer-grade quality, expected for well-functioning synthesis [Deepgram-PER]
* WER 5–10% → degraded but potentially acceptable
* WER > 10% → hard failure; broken synthesis likely [Inworld-OpenEval]
* WER > 100% → fundamentally broken (insertions + deletions dominate) [Deepgram-PER]

F5-TTS, a modern neural TTS, achieves WER ≈ 2.4% on LibriSpeech-PC test-clean [F5TTS-Paper],
establishing that < 5% is achievable for clean studio-quality synthesis.

**Caveats**:

* WER depends on the ASR model and normalization. Use `faster-whisper` with normalized text (number
  expansion, punctuation stripping) for reproducibility [Inworld-OpenEval].
* Short filler utterances (< 3 s) inflate WER because a single word error is a large proportion of
  the word count. A filler phrase with WER 33% (1 error in 3 words) is not necessarily broken.
* WER cannot distinguish TTS pronunciation errors from ASR errors on edge-case words.
  Cross-reference with duration_ratio to detect silence/length explosions [TaskDescription].

**Best practice**: compute WER per clip, flag clips with WER > 20% for manual review (higher
threshold than the general 10% to account for short-utterance inflation), and never rely on WER
alone to accept or reject a system — treat it as a triage filter alongside duration_ratio.

### Streaming TTS TTFB Measurement: Methodology Best Practices

The Coval benchmark [Coval-Methodology] defines the canonical TTFA measurement for streaming TTS:

```
TTFA = (first audio chunk arrival − synthesis start) + leading silence in the stream
```

The leading-silence correction is computed using an RMS-threshold onset detector on the assembled
PCM. This ensures the metric reflects when the user actually hears audio, not when data bytes
arrive.

For local Kokoro inference, "leading silence" is minimal because Kokoro yields the first audio chunk
only after processing the first phoneme segment — no HTTP headers or codec headers precede the
audio. The time to first `audio_numpy` yield from `KPipeline.__call__` is the correct TTFB for
Kokoro.

The Picovoice benchmark [Picovoice-TTS] uses ~200 simulated voice-assistant interactions
(taskmaster2 dataset), reporting simple mean latencies without p-percentiles. For this project,
reporting p50/p95/p99 is preferable because a single outlier (e.g., a CUDA cache miss on the first
synthesis) can inflate the mean without representing typical performance.

**Best practice — number of trials**: A minimum of 30 prompts per system is sufficient to estimate
p50 and p95 reliably for GPU inference. Use the val_96 set (96 clips) and the filler set (≥50 clips)
as described in the task description for a total of ≥145 timing measurements per system.

### Open-Source TTS Evaluation Frameworks

Two open-source frameworks provide reference implementations for the harness architecture:

**inworld-ai/open-tts-eval** [Inworld-OpenEval]: Uses `faster-whisper` + `JiWER 3.0+` for WER; ECAPA
speaker similarity (requires reference audio); NISQAv2 MOS. Does not measure latency. WER
thresholds: warn=0.05, fail=0.10 (starter bands, domain-calibrated). Speaker similarity: warn <
0.65, fail < 0.55.

**wavlab-speech/versa** [VERSA2024]: 65 metrics including WER, MCD, UTMOS, speaker similarity
(WavLM-based), DNSMOS. Full installation, YAML-driven configuration. Heavy dependency footprint.

Neither framework measures TTFB. The harness in this task must implement timing inline.

For speaker similarity, both frameworks use WavLM or ECAPA embeddings rather than GE2E/resemblyzer.
The project specifically requires GE2E cosine (via resemblyzer) for consistency with the project's
success criterion. This is a deliberate divergence from broader community practice — WavLM-based
speaker similarity correlates better with human perception [AnalyzeSim2025] but GE2E was chosen for
project continuity.

## Methodology Insights

* **Centroid construction**: build the ElevenLabs David centroid by calling `embed_utterance` on
  each reference clip and taking the mean, then L2-normalizing. Do not average the raw waveforms.
  Split 1358 clips 679/679 (seed=42) — centroid half and held-out half. ElevenLabs arm is scored
  against the held-out half; all other arms are scored against the full centroid.

* **Resemblyzer normalization**: `preprocess_wav` already normalizes volume and resamples.
  Synthesized audio should be passed through `preprocess_wav` before embedding to ensure consistent
  sample rate (16 kHz) and amplitude normalization. Raw 24 kHz Kokoro output must be resampled.

* **TTFB for Kokoro**: timestamp immediately before the `KPipeline.__call__()` invocation and
  capture the time of the first yielded `audio_numpy` tuple. Do not include model loading time
  (pre-warm the pipeline). Send ≥1 warmup synthesis (discard its timing) per system session.

* **TTFB for ElevenLabs**: use `stream=True` in the API call with chunk size 1024 bytes. Timestamp
  before the request and capture arrival of the first non-empty chunk. Use a persistent
  `requests.Session`. Discard ≥1 warmup request.

* **RTF**: `synthesis_wall_time / output_audio_duration_seconds`. Use `soundfile.info()` to get
  audio duration from the output numpy array rather than dividing by 24000 directly.

* **WER**: use `faster-whisper` (base.en model for speed; large-v3 for accuracy reference).
  Normalize both hypothesis and reference: lowercase, expand numbers, strip punctuation. Compute
  per-clip WER with JiWER; report mean and % of clips with WER > 20%.

* **Duration ratio**: `len(synth_audio) / 24000 / reference_audio_duration`. Values > 2.0 or < 0.5
  warrant a manual review flag.

* **Hypothesis — WER detects noisy checkpoints**: The t0005/t0006 checkpoints were described as
  producing "noisy audio" on subjective evaluation. If this noise manifests as unintelligible
  phonemes, WER should be elevated (> 10%) for those systems while speaker_sim might still score
  moderately. If WER and speaker_sim are decorrelated across systems, that validates the need for
  both metrics.

* **Best practice — floor control**: Include `af_heart` (female American) as the floor control to
  establish what a deliberately wrong-speaker score looks like on this scale. Expected floor: cosine
  similarity approximately 0.3–0.5 against the ElevenLabs David centroid.

* **Reproducibility**: record `torch.__version__`, `cuda version` (from `torch.version.cuda`),
  Kokoro package version (`pip show kokoro`), `nvidia-smi` GPU model, and timestamp in a
  `metadata.json` alongside each system's results.

## Tool and Library Landscape

| Tool | Purpose | Notes |
| --- | --- | --- |
| `resemblyzer` (resemble-ai) | GE2E speaker embeddings | 256-d, L2-normed; `preprocess_wav` + `embed_utterance` |
| `faster-whisper` | WER transcription | Much faster than openai-whisper; use base.en |
| `jiwer` | WER computation | Standard; normalize with `RemovePunctuation`, `ToLowerCase` |
| `soundfile` | Audio I/O | Reads/writes WAV; get duration with `soundfile.info()` |
| `kokoro` (hexgrad) | Kokoro TTS synthesis | `KPipeline(lang_code="b")` for British English |
| `elevenlabs` Python SDK | ElevenLabs API | Use `generate(..., stream=True)` for TTFB |
| `inworld-ai/open-tts-eval` [Inworld-OpenEval] | Reference eval pipeline | WER + speaker_sim reference implementation |
| `wavlab-speech/versa` [VERSA2024] | Comprehensive audio eval | 65 metrics; heavy but authoritative |
| `Picovoice/tts-latency-benchmark` [Picovoice-TTS] | TTFB benchmark framework | Apache 2.0; ~200 prompts methodology |

## Benchmark Comparison

Published and community numbers for ElevenLabs and Kokoro-82M latency (all non-peer-reviewed unless
noted):

| System | Metric | Value | Hardware | Source |
| --- | --- | --- | --- | --- |
| ElevenLabs Turbo v2.5 | TTFA p50 | 264 ms | Cloud API | [Forasoft-Stream] |
| ElevenLabs Flash v2.5 | TTFA p50 | 288 ms | Cloud API | [Forasoft-Stream] |
| ElevenLabs Streaming | FTTS mean | 335 ms | AMD Ryzen 7 5700X | [Picovoice-TTS] |
| ElevenLabs PCM stream | TTFA avg | 478 ms | Cloud API | [ElevenLabs-Latency] |
| Kokoro-82M | TTFB baseline | 97 ms | GPU (unspecified) | [Together-Kokoro] |
| Kokoro-82M | FTTS | 28 ms | RTX 5090 | [Forasoft-Stream] |
| Kokoro-82M | RTF | ~0.03 | A100 | [GigaGPU-Kokoro] |
| Kokoro-82M | RTF | ~0.04 | RTX 4090 | [GigaGPU-TTS] |
| Kokoro-82M | FTTS (CPU) | 3,658 ms | AMD Ryzen 7 5700X | [Picovoice-TTS] |

**Key insight**: ElevenLabs TTFA (264–335 ms) is already near the 300 ms project target. The ≤ 300
ms criterion is achievable by ElevenLabs only with the Flash/Turbo model and optimized streaming —
not with the standard REST API (1,470 ms). Kokoro on H100 should easily beat this given RTF 0.03 on
A100.

## Discovered Papers

### [Wan2018]

* **Title**: Generalized End-to-End Loss for Speaker Verification
* **Authors**: Wan, L., Wang, Q., Papir, A., Lopez Moreno, I.
* **Year**: 2018
* **DOI**: `10.1109/ICASSP.2018.8462665`
* **URL**: https://arxiv.org/abs/1710.10467
* **Suggested categories**: `speaker-verification`, `speech-embeddings`
* **Why download**: Defines the GE2E loss and the d-vector architecture that resemblyzer wraps. The
  speaker_sim metric in this project is GE2E cosine; understanding the embedding model is essential
  for interpreting metric range, training objective, and threshold semantics. This is the
  foundational reference.

### [Li2023]

* **Title**: StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial
  Training with Large Speech Language Models
* **Authors**: Li, Y., Han, C., Raghavan, V., Mesgarani, N., Yoon, H.
* **Year**: 2023
* **DOI**: `10.5555/3666122.3667032`
* **URL**: https://arxiv.org/abs/2306.07691
* **Suggested categories**: `text-to-speech`, `speech-synthesis`
* **Why download**: Kokoro-82M is built on StyleTTS2. The architecture paper documents the style
  embedding structure, decoder design, and evaluation methodology (MOS, CMOS, WER) used by the
  underlying model. Essential for understanding what is swapped between evaluation systems
  (voicepack vs. decoder vs. both).

### [VERSA2024]

* **Title**: VERSA: A Versatile Evaluation Toolkit for Speech, Audio, and Music
* **Authors**: Shi, J. et al.
* **Year**: 2024
* **DOI**: null
* **URL**: https://arxiv.org/abs/2412.17667
* **Suggested categories**: `evaluation-methodology`, `text-to-speech`
* **Why download**: Comprehensive reference for speech evaluation methodology. Covers WER, MCD,
  UTMOS, speaker similarity metrics, and MOS correlation. Directly relevant to harness design
  decisions: which metrics to use, how to normalize, and how they correlate with human perception.
  65 metrics with 729 variants.

### [AnalyzeSim2025]

* **Title**: Analyzing and Improving Speaker Similarity Assessment for Speech Synthesis
* **Authors**: (Authors not fully captured from search)
* **Year**: 2025
* **DOI**: null
* **URL**: https://arxiv.org/html/2507.02176v1
* **Suggested categories**: `speaker-verification`, `evaluation-methodology`
* **Why download**: Critical analysis of speaker similarity metrics in TTS evaluation. Discusses
  where cosine-based GE2E metrics agree and disagree with human perception, and proposes
  improvements. Directly relevant to interpreting the project's 0.85 GE2E cosine threshold and
  understanding when the metric might give misleading signals.

## Recommendations for This Task

1. **Use centroid-half comparison** (not per-clip) for the scoring design: build the ElevenLabs
   David centroid from 679 clips (seed=42 split), score all synthesis against that centroid.
   ElevenLabs arm is scored against the held-out 679 clips to avoid self-comparison. Expected
   same-system scores: 0.85+ against centroid for a well-matched voice; 0.7 for per-clip comparisons
   [Baseten-FineTune].

2. **Set WER thresholds as**: warn at WER > 10%, hard-flag at WER > 20% (adjusted up from the
   general 10% hard threshold to account for short filler utterance inflation). Report % of clips
   above each threshold per system [Inworld-OpenEval].

3. **Expect ElevenLabs TTFA ≈ 250–340 ms** from published benchmarks [Forasoft-Stream],
   [Picovoice-TTS] — already near the 300 ms boundary. Measure directly; do not rely on these
   numbers.

4. **Expect Kokoro H100 TTFB < 100 ms** based on A100 RTF ≈ 0.03 and 97 ms baseline
   [GigaGPU-Kokoro]. GPU first-chunk time dominates on H100; network overhead is zero for local
   inference.

5. **Minimum 1 warmup synthesis per system** before timing begins [Coval-Methodology],
   [Picovoice-TTS]. Pre-load all model weights before starting the timing loop — exclude model
   loading from TTFB.

6. **Download [Wan2018] and [Li2023]** as the foundational architecture papers. Download [VERSA2024]
   for evaluation methodology reference. Download [AnalyzeSim2025] for speaker similarity metric
   interpretation guidance.

7. **Resemblyzer audio length floor**: skip clips < 1.6 s in speaker_sim scoring (fewer than one
   partial-utterance window). Log the count of skipped clips per system.

8. **Duration ratio is complementary to WER**: the t0002 10× duration explosion would have been
   caught by duration_ratio > 2.0 long before WER was computed. Implement duration_ratio check
   before WER to short-circuit expensive Whisper transcription on obviously broken outputs.

## Source Index

### [Wan2018]

* **Type**: paper
* **Title**: Generalized End-to-End Loss for Speaker Verification
* **Authors**: Wan, L. et al.
* **Year**: 2018
* **DOI**: `10.1109/ICASSP.2018.8462665`
* **URL**: https://arxiv.org/abs/1710.10467
* **Peer-reviewed**: yes (ICASSP 2018)
* **Relevance**: Defines the GE2E embedding model that resemblyzer implements. Architecture: 3 LSTM
  layers + projection to 256-d d-vector. This is the speaker_sim metric foundation for this project.

### [Li2023]

* **Type**: paper
* **Title**: StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial
  Training with Large Speech Language Models
* **Authors**: Li, Y. et al.
* **Year**: 2023
* **DOI**: `10.5555/3666122.3667032`
* **URL**: https://arxiv.org/abs/2306.07691
* **Peer-reviewed**: yes (NeurIPS 2023)
* **Relevance**: Base architecture for Kokoro-82M. Documents style embedding structure (decoder
  components swapped between evaluation systems), evaluation metrics (MOS, WER), and human-level
  performance claims.

### [VERSA2024]

* **Type**: paper
* **Title**: VERSA: A Versatile Evaluation Toolkit for Speech, Audio, and Music
* **Authors**: Shi, J. et al.
* **Year**: 2024
* **DOI**: null
* **URL**: https://arxiv.org/abs/2412.17667
* **Peer-reviewed**: no (preprint, under review)
* **Relevance**: Comprehensive evaluation framework covering WER, MCD, speaker similarity, and MOS
  metrics. Reference for harness architecture and metric selection decisions.

### [AnalyzeSim2025]

* **Type**: paper
* **Title**: Analyzing and Improving Speaker Similarity Assessment for Speech Synthesis
* **Authors**: (First author et al.)
* **Year**: 2025
* **DOI**: null
* **URL**: https://arxiv.org/html/2507.02176v1
* **Peer-reviewed**: no (preprint 2025)
* **Relevance**: Critical analysis of cosine-based speaker similarity metrics in TTS. Important for
  interpreting when GE2E cosine ≥ 0.85 is and is not meaningful as a perceptual quality signal.

### [Resemble-GH]

* **Type**: repository
* **Title**: Resemblyzer — A python package to analyze and compare voices with deep learning
* **Author/Org**: resemble-ai
* **Date**: 2019-12
* **URL**: https://github.com/resemble-ai/Resemblyzer
* **Last updated**: 2024
* **Peer-reviewed**: no
* **Relevance**: Primary library for speaker_sim computation. Wraps GE2E with `preprocess_wav` and
  `embed_utterance`. Outputs 256-d L2-normed embeddings. Partial-utterance window: 1.6 s.

### [Resemble-AuthStudy]

* **Type**: paper
* **Title**: Evaluation of Resemblyzer Ability to Authenticate the Speaker
* **Author/Org**: CEUR Workshop Proceedings (authors not captured)
* **Date**: 2023
* **URL**: https://ceur-ws.org/Vol-4164/paper7.pdf
* **Peer-reviewed**: no (workshop paper)
* **Relevance**: Empirical calibration study of resemblyzer cosine similarity thresholds. Found
  threshold of 0.84 for English voice authentication. Minimum reliable clip: 2.63 s, 495 KB.

### [Kokoro-HF]

* **Type**: documentation
* **Title**: hexgrad/Kokoro-82M model card
* **Author/Org**: hexgrad
* **Date**: 2025-01
* **URL**: https://huggingface.co/hexgrad/Kokoro-82M
* **Last updated**: 2025-09
* **Peer-reviewed**: no
* **Relevance**: Official Kokoro-82M architecture description: StyleTTS2 + ISTFTNet, decoder-only,
  82M params, Apache 2.0, 54 voices, 24 kHz output. Voicepack format and KPipeline API
  documentation.

### [GigaGPU-Kokoro]

* **Type**: blog
* **Title**: Kokoro TTS Latency by GPU
* **Author/Org**: GigaGPU
* **Date**: 2025
* **URL**: https://gigagpu.com/kokoro-tts-latency-by-gpu/
* **Peer-reviewed**: no
* **Relevance**: GPU-specific Kokoro TTFB/RTF benchmarks. RTF ≈ 0.03 on A100; RTX 3050 at 180 ms.
  Important for setting expectations before H100 measurement.

### [GigaGPU-TTS]

* **Type**: blog
* **Title**: TTS Latency Benchmarks
* **Author/Org**: GigaGPU
* **Date**: 2025
* **URL**: https://gigagpu.com/tts-latency-benchmarks/
* **Peer-reviewed**: no
* **Relevance**: Comparative GPU latency benchmarks across TTS models. RTF ~0.03–0.04 for Kokoro on
  RTX 4090. Contextualizes expected Kokoro performance on H100.

### [ElevenLabs-Latency]

* **Type**: documentation
* **Title**: Understanding latency | ElevenLabs Documentation
* **Author/Org**: ElevenLabs
* **Date**: 2026
* **URL**: https://elevenlabs.io/docs/eleven-api/concepts/latency
* **Last updated**: 2026-09
* **Peer-reviewed**: no
* **Relevance**: Official latency documentation. Flash model inference: ~75 ms. PCM streaming REST
  API TTFA: ~478 ms average. Measurement guidance: measure from application, not API benchmarks.

### [Forasoft-Stream]

* **Type**: blog
* **Title**: Streaming TTS — Kokoro, ElevenLabs Turbo, Cartesia, And OpenAI TTS
* **Author/Org**: Forasoft
* **Date**: 2026-06
* **URL**:
  https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/streaming-tts-kokoro-elevenlabs-turbo-openai-tts
* **Peer-reviewed**: no
* **Relevance**: Side-by-side streaming latency comparison. ElevenLabs Turbo: 264 ms median TTFA.
  ElevenLabs Flash: 288 ms median TTFA. Kokoro RTX 4090: ~210× faster than real-time.

### [Picovoice-TTS]

* **Type**: documentation
* **Title**: Open-Source Text-to-Speech Latency Benchmark — Picovoice Docs
* **Author/Org**: Picovoice
* **Date**: 2025
* **URL**: https://picovoice.ai/docs/benchmark/tts/
* **Last updated**: 2026
* **Peer-reviewed**: no
* **Relevance**: Comprehensive TTFB benchmark covering 13+ TTS systems. Kokoro CPU FTTS: 3,658 ms.
  ElevenLabs Streaming: 335 ms. Open-source methodology on GitHub (Apache 2.0). ~200 taskmaster2
  prompts. Reference framework for this task's harness design.

### [Coval-Methodology]

* **Type**: documentation
* **Title**: TTS Benchmark Methodology — coval-ai/benchmarks
* **Author/Org**: Coval AI
* **Date**: 2025-2026
* **URL**: https://github.com/coval-ai/benchmarks/blob/main/docs/methodology.md
* **Last updated**: 2026
* **Peer-reviewed**: no
* **Relevance**: Canonical TTFA definition: (first chunk arrival − synthesis start) + leading
  silence. Warmup methodology: scale-from-zero boot via 6-minute handshake budget; HTTP/2
  multiplexing to amortize TCP+TLS. Reference for this task's TTFB measurement protocol.

### [Inworld-OpenEval]

* **Type**: repository
* **Title**: open-tts-eval — Open, reproducible TTS/STT evaluation toolkit
* **Author/Org**: inworld-ai
* **Date**: 2025
* **URL**: https://github.com/inworld-ai/open-tts-eval
* **Last updated**: 2026
* **Peer-reviewed**: no
* **Relevance**: Reference implementation for WER + speaker similarity. WER thresholds: warn=0.05,
  fail=0.10. Speaker similarity: warn < 0.65, fail < 0.55. Uses faster-whisper + JiWER 3.0+. Audio
  health checks: clipping, silence, tail-clicks.

### [Deepgram-PER]

* **Type**: blog
* **Title**: PER vs WER: New TTS Accuracy Measurement Standard 2026
* **Author/Org**: Deepgram
* **Date**: 2026
* **URL**: https://deepgram.com/learn/per-new-standard-measuring-tts-accuracy
* **Peer-reviewed**: no
* **Relevance**: WER thresholds for TTS quality gates: < 5% consumer-grade; > 100% fundamentally
  broken. Introduces phoneme error rate (PER) as a complementary metric.

### [Baseten-FineTune]

* **Type**: blog
* **Title**: Fine-tuning Qwen3-TTS for high-quality voice cloning
* **Author/Org**: Baseten
* **Date**: 2026
* **URL**: https://www.baseten.co/blog/fine-tuning-qwen3-tts-for-high-quality-voice-cloning/
* **Peer-reviewed**: no
* **Relevance**: Key finding: per-clip cosine similarity for the same speaker ≈ 0.7; with centroid
  (mean of 64 clips) rises to 0.85+. Directly explains why the 0.85 target uses centroid comparison
  and why per-clip scores will typically be lower.

### [ResemblyzerSearch]

* **Type**: blog
* **Title**: Speaker identification, differentiation and verification using resemblyzer
* **Author/Org**: Various (search result compilation)
* **Date**: 2024-2025
* **URL**: https://github.com/resemble-ai/Resemblyzer
* **Peer-reviewed**: no
* **Relevance**: Community-reported cosine similarity ranges: same-speaker 0.8–0.95 (typical),
  different-speaker 0.3–0.6 (typical), 0.75 as common default threshold in prototyping.

### [Kaneko2022]

* **Type**: paper
* **Title**: iSTFTNet: Fast and Lightweight Mel-Spectrogram Vocoder Incorporating Inverse Short-Time
  Fourier Transform
* **Authors**: Kaneko, T., Tanaka, K., Kameoka, H., Seki, S.
* **Year**: 2022
* **DOI**: null
* **URL**: https://arxiv.org/abs/2203.02395
* **Peer-reviewed**: yes (ICASSP 2022)
* **Relevance**: Vocoder component of Kokoro-82M. ISTFTNet converts mel-spectrograms to waveforms.
  Understanding this component informs the audio output format (24 kHz) and synthesis pipeline.

### [Together-Kokoro]

* **Type**: documentation
* **Title**: Kokoro-82M TTS API — Together AI
* **Author/Org**: Together AI
* **Date**: 2025
* **URL**: https://www.together.ai/models/kokoro-82m
* **Peer-reviewed**: no
* **Relevance**: Reports 97 ms baseline TTFB for hosted Kokoro endpoint. Provides context for
  expected GPU TTFB range before H100 measurement.

### [Openbenchmarks-EL]

* **Type**: blog
* **Title**: ElevenLabs voice agent latency: independent TTFAB benchmark (2026)
* **Author/Org**: Openbenchmarks
* **Date**: 2026
* **URL**: https://openbenchmarks.com/voice-agent-latency/elevenlabs
* **Peer-reviewed**: no
* **Relevance**: Independent end-to-end ElevenLabs latency benchmark. Median TTFAB 1,424 ms, p95
  1,768 ms over 429 voice agent turns. Distinguishes voice-agent pipeline latency from raw TTS TTFB.

### [NVIDIA-Riva]

* **Type**: documentation
* **Title**: Evaluate a TTS Pipeline — NVIDIA Riva User Guide
* **Author/Org**: NVIDIA
* **Date**: 2024
* **URL**: https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tutorials/tts-evaluate.html
* **Peer-reviewed**: no
* **Relevance**: Reference methodology for Whisper-based WER computation in TTS evaluation.
  Describes intelligibility evaluation using ASR transcription and WER as a proxy for synthesis
  quality.

### [F5TTS-Paper]

* **Type**: paper
* **Title**: F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching
* **Authors**: Chen, S. et al.
* **Year**: 2024
* **DOI**: null
* **URL**: https://arxiv.org/abs/2410.06885
* **Peer-reviewed**: no (preprint 2024, accepted Interspeech 2025)
* **Relevance**: State-of-the-art flow-matching TTS achieving WER ≈ 2.4% on LibriSpeech-PC
  test-clean. Establishes that WER < 5% is achievable for clean neural TTS, providing a reference
  point for what a well-functioning synthesis pipeline should score.

### [TaskDescription]

* **Type**: documentation
* **Title**: TTS Evaluation Harness and Baselines — Task Description
* **Author/Org**: rail-arf-tts project
* **Date**: 2026-09-14
* **URL**: tasks/t0008_tts_eval_harness_baselines/task_description.md
* **Peer-reviewed**: no
* **Relevance**: Project-internal document specifying evaluation systems, prompt sets, scoring
  methodology, and compute budget. Reference for duration_ratio design choice (catches 10× duration
  explosion from t0002) and the sanity-metric motivation.
