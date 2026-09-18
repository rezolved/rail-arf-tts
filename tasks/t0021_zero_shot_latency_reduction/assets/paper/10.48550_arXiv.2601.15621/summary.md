---
spec_version: "3"
paper_id: "10.48550_arXiv.2601.15621"
citation_key: "Hu2026"
summarized_by_task: "t0021_zero_shot_latency_reduction"
date_summarized: "2026-09-18"
---
# Qwen3-TTS Technical Report

## Metadata

* **File**: `files/hu_2026_qwen3-tts-technical-report.pdf`
* **Published**: 2026
* **Authors**: Hangrui Hu 🇨🇳, Xinfa Zhu 🇨🇳, Ting He 🇨🇳, Dake Guo 🇨🇳, Bin Zhang 🇨🇳, Xiong Wang 🇨🇳,
  Zhifang Guo 🇨🇳, Ziyue Jiang 🇨🇳, Hongkun Hao 🇨🇳, Zishan Guo 🇨🇳, Xinyu Zhang 🇨🇳, Pei Zhang 🇨🇳,
  Baosong Yang 🇨🇳, Jin Xu 🇨🇳, Jingren Zhou 🇨🇳, Junyang Lin 🇨🇳
* **Venue**: preprint (arXiv)
* **DOI**: `10.48550/arXiv.2601.15621`

## Abstract

In this report, we present the Qwen3-TTS series, a family of advanced multilingual, controllable,
robust, and streaming text-to-speech models. Qwen3-TTS supports state-of-the-art 3-second voice
cloning and description-based control, allowing both the creation of entirely novel voices and
fine-grained manipulation over the output speech. Trained on over 5 million hours of speech data
spanning 10 languages, Qwen3-TTS adopts a dual-track LM architecture for real-time synthesis,
coupled with two speech tokenizers: 1) Qwen-TTS-Tokenizer-25Hz is a single-codebook codec
emphasizing semantic content, which offers seamlessly integration with Qwen-Audio and enables
streaming waveform reconstruction via a block-wise DiT. 2) Qwen-TTS-Tokenizer-12Hz achieves extreme
bitrate reduction and ultra-low-latency streaming, enabling immediate first-packet emission (97 ms)
through its 12.5 Hz, 16-layer multi-codebook design and a lightweight causal ConvNet. Extensive
experiments indicate state-of-the-art performance across diverse objective and subjective benchmark
(e.g., TTS multilingual test set, InstructTTSEval, and our long speech test set). To facilitate
community research and development, we release both tokenizers and models under the Apache 2.0
license.

## Overview

Qwen3-TTS is Alibaba's first text-to-speech entry in the Qwen model family, built to be a
production-grade, streaming, multilingual, and controllable TTS system rather than a narrow research
demo. The report frames the work around two coupled bottlenecks that any streaming zero-shot TTS
system must solve simultaneously: token quality (does the discrete speech representation preserve
enough acoustic detail for natural speech without making autoregressive modeling unstable?) and
decoding latency (how quickly can the first audio packet reach the user, and how does that latency
scale under concurrent requests?). The authors address both by shipping two interchangeable speech
tokenizers — a 25 Hz single-codebook, semantic-leaning tokenizer built on a fine-tuned Qwen2-Audio
encoder with a block-wise diffusion-transformer (DiT) vocoder, and a 12.5 Hz multi-codebook
tokenizer (16 residual layers, semantic + acoustic disentangled à la Mimi) decoded by a lightweight
causal ConvNet — and pairing each with a dual-track Qwen3-LM backbone that interleaves text and
speech tokens on the channel axis so speech generation can start before the full input text has been
consumed.

The most operationally relevant contribution for streaming deployment is the explicit,
concurrency-swept latency breakdown in Section 3.4 (Table 2): first-packet latency and steady-state
real-time factor (RTF) are reported separately for the LM's time-to-first-packet (TTFP) and the
tokenizer's per-packet decode time (TPP), at concurrency levels 1, 3, and 6, measured on the
authors' internal vLLM engine (vLLM V0 backend) with `torch.compile` and CUDA-graph acceleration
applied to the tokenizer decode stage. This is a rare case of a TTS technical report publishing
per-stage, per-concurrency latency numbers rather than a single headline figure, which makes it
directly comparable to this project's own TTFB profiling methodology for CosyVoice2 and Chatterbox.

On quality, Qwen3-TTS reports state-of-the-art WER on the Seed-TTS zero-shot cloning benchmark for
its 12Hz-1.7B variant, and claims the highest speaker-similarity scores across all 10 evaluated
languages against MiniMax-Speech and ElevenLabs Multilingual v2 — the same commercial reference the
Rezolve pipeline is trying to replace with Kokoro. The report also documents an instruction-tuned
voice-design/voice-editing capability (InstructTTSEval) and long-form stability testing (>10 minutes
of continuous speech) that are secondary to this project's latency focus but corroborate that the
low-latency 12Hz configuration does not come at a quality cost relative to the semantic-only 25Hz
configuration on most axes.

## Architecture, Models and Methods

**Tokenizers.** Qwen-TTS-Tokenizer-25Hz is trained in two stages on top of Qwen2-Audio: Stage 1
continues ASR pretraining with an added resampling layer and a vector-quantization (VQ) layer
inserted mid-encoder; Stage 2 fine-tunes the full model with a convolutional mel-spectrogram decoder
to inject acoustic detail into the tokens. Its streaming detokenizer is a Diffusion Transformer
(DiT) trained with flow matching, restricted via sliding-window block attention to a 4-block
receptive field (current block + 3-block lookback + 1-block lookahead), followed by a modified
BigVGAN vocoder operating on fixed-size chunks. With a chunk size of 8 tokens at 40 ms/token (25
Hz), the LM must emit 16 tokens before the first mel chunk can be produced, and BigVGAN adds 130 ms
of right-context lookahead, yielding roughly 190 ms of audio in the first packet.
Qwen-TTS-Tokenizer-12Hz instead uses a Mimi-style semantic/acoustic disentangled quantizer at 12.5
Hz: a WavLM-distilled semantic first codebook layer plus a 15-layer residual VQ (RVQ) acoustic
stack, trained end-to-end with a GAN objective (waveform-domain generator/discriminator) and a
multi-scale mel-spectrogram reconstruction loss. Encoder and decoder are fully causal (no
look-ahead), so one 80 ms token can in principle be decoded immediately; the authors batch 4 tokens
per packet (320 ms of audio) to limit scheduling overhead.

**Backbone.** Both tokenizer variants sit under a shared Qwen3-LM backbone with a jointly trained
speaker encoder for identity control. The dual-track design concatenates text and acoustic tokens
along the channel axis so the model predicts acoustic tokens immediately upon receiving each text
token; Qwen3-TTS-25Hz predicts single-level tokens through a linear head into the chunk-wise DiT,
while Qwen3-TTS-12Hz uses a hierarchical scheme — the backbone predicts the zeroth (semantic)
codebook and a Multi-Token Prediction (MTP) module generates the remaining residual codebooks in a
single forward pass, enabling instant per-frame generation.

**Training.** Pretraining has three stages: (S1) general multilingual pretraining on 5M+ hours of
speech; (S2) continual pretraining (CPT) on a quality-stratified subset to reduce hallucination;
(S3) long-context extension from 8,192 to 32,768 max tokens with upsampled long-speech data.
Post-training has three stages: Direct Preference Optimization (DPO) on human-labeled multilingual
preference pairs, rule-based-reward reinforcement learning via GSPO, and lightweight per-speaker
fine-tuning (producing the "CustomVoice" / fine-tuned "Aiden Voice" variants used in later
evaluations). Model sizes evaluated are 0.6B and 1.7B parameters for each tokenizer, i.e. four base
model configurations (Qwen3-TTS-25Hz/12Hz x 0.6B/1.7B), plus VoiceDesign, CustomVoice and
VoiceEditing task-specific variants (Table 1). Evaluation benchmarks used: Seed-TTS test set (WER),
a 10-language multilingual test set from MiniMax-Speech's paper (WER + speaker-similarity cosine),
CV3-Eval for cross-lingual transfer, InstructTTSEval (APS/DSD/RP metrics) for controllability, and
an internal 100-text long-speech set (200-2,000 words per text, transcribed with Qwen3-ASR) for
long-form stability. No GPU count, GPU model, or total training compute (FLOPs/GPU-hours) is
reported anywhere in the paper.

## Results

* First-packet latency as low as **97 ms** (12Hz-0.6B variant) and **101 ms** (12Hz-1.7B variant) at
  concurrency 1, versus **138 ms** (25Hz-0.6B) and **150 ms** (25Hz-1.7B) for the semantic-only
  tokenizer at the same concurrency (Table 2).
* Steady-state RTF at concurrency 1 ranges from **0.234** (25Hz-0.6B) to **0.313** (12Hz-1.7B); at
  concurrency 6 it degrades to **0.709-0.725** for the 25Hz tokenizer but stays far lower, at
  **0.434-0.463**, for the 12Hz tokenizer (Table 2) — the 12Hz decoder's causal ConvNet scales
  better under load than the 25Hz DiT+BigVGAN pipeline.
* Tokenizer decode time per packet (TPP) is only **4-5 ms** for the 12Hz codec versus **25-147 ms**
  for the 25Hz codec across concurrency 1-6, confirming the DiT/BigVGAN stage is the latency-scaling
  bottleneck for the 25Hz variant.
* Zero-shot cloning on Seed-TTS test-en: Qwen3-TTS-12Hz-1.7B-Base reaches **1.24 WER**, beating
  CosyVoice 3 (**1.45**), MiniMax-Speech (**1.65**), and Seed-TTS itself (**2.25**) (Table 5).
* Multilingual generation: Qwen3-TTS reports the **lowest WER in 6 of 10 languages** (Chinese,
  English, Italian, French, Korean, Russian) versus MiniMax-Speech and ElevenLabs Multilingual v2,
  and the **highest speaker-similarity (SIM) score in all 10 evaluated languages** (Table 6).
* Voice design/editing (InstructTTSEval-ZH, Target-Speaker Editing): Qwen3TTS-12Hz-1.7B-CustomVoice
  beats GPT-4o-mini-tts by **+28% APS** in Chinese (Table 8).
* Long-form generation (internal 100-text set, 200-2,000 words): Qwen3-TTS-25Hz-1.7B-CustomVoice
  achieves **1.517 WER (long-zh)** and **1.225 WER (long-en)**, versus VibeVoice's **22.619 WER** on
  long-zh, illustrating the chunk-based baseline's stability collapse on long Chinese text (Table
  10).
* Speech-reconstruction quality of Qwen-TTS-Tokenizer-12Hz on LibriSpeech test-clean (2,620
  utterances) sets a new state of the art across STOI, PESQ, UTMOS, and WavLM-based speaker
  similarity simultaneously (Table 4; exact per-metric values render as a table image in the
  extracted text and are not individually quoted here — see the PDF for the full table).
* Cross-lingual voice transfer (CV3-Eval): Qwen3-TTS outperforms GPT-4o-Audio-Preview in **7 of 10
  languages**, with a notably larger intelligibility gap in Japanese (**3.88 vs. 5.00** for GPT-4o).

## Innovations

### Dual-Track Streaming Architecture

Concatenating text and acoustic tokens along the channel axis lets the LM begin predicting speech
tokens as soon as it receives a text token, rather than waiting for full-sentence buffering. This
directly targets the "sentence-level buffering" latency tax that the project's own research on
CosyVoice2/Chatterbox streaming has identified as a major TTFB contributor.

### Two Interchangeable Tokenizers With an Explicit Latency/Quality Trade-off

Rather than committing to one codec, the authors ship a 25 Hz semantic-leaning tokenizer (better
long-form stability, higher first-packet latency due to DiT lookahead) and a 12.5 Hz multi-codebook
tokenizer (near-instant first packet via a causal ConvNet decoder, slightly worse long-form WER).
This is a directly reusable design pattern: pick the tokenizer per deployment constraint rather than
assuming one codec must win on every axis.

### Multi-Token Prediction (MTP) for Multi-Codebook Decoding

The 12Hz backbone predicts the zeroth (semantic) codebook autoregressively but generates the
remaining 15 residual codebook layers via an MTP module in the same forward step, avoiding the
per-codebook autoregressive cost that typically dominates multi-codebook LLM-TTS decoding latency.

### Per-Stage, Per-Concurrency Latency Reporting

Table 2 decomposes end-to-end first-packet latency into LM TTFP and tokenizer decode TPP, and
reports both at concurrency 1/3/6 on a real serving engine (vLLM V0). Very few TTS technical reports
publish this granularity; it is a template this project's own benchmark write-ups could adopt for
comparing CosyVoice2/Chatterbox/Kokoro under load rather than only at concurrency 1.

## Datasets

* **Training data**: over 5 million hours of multilingual speech spanning 10 languages (exact
  language list, licensing, and per-language hour counts are not reported in the paper).
* **Seed-TTS test set** (Anastassiou et al., 2024): public zero-shot TTS evaluation benchmark,
  test-zh and test-en subsets, used for WER-based content-consistency comparison.
* **TTS multilingual test set** (from Zhang et al., 2025a / MiniMax-Speech paper): 10-language
  zero-shot multilingual evaluation set used for WER and speaker-similarity (SIM) comparison against
  MiniMax-Speech and ElevenLabs Multilingual v2.
* **CV3-Eval** (Du et al., 2025): cross-lingual voice-transfer evaluation benchmark.
* **InstructTTSEval** (Huang et al., 2025): instruction-following / controllable-generation
  benchmark with APS, DSD, and RP metrics, Chinese and English subsets.
* **LibriSpeech test-clean**: 2,620 utterances, used for tokenizer reconstruction quality (STOI,
  PESQ, UTMOS, WavLM speaker similarity).
* **CommonVoice and Fleurs (EN/CN subsets)**: used for ASR-based evaluation of the
  Qwen-TTS-Tokenizer-25Hz's semantic discriminability.
* **Internal long-speech set**: 100 texts (Chinese and English), 200-2,000 words each, not publicly
  released, used for long-form stability evaluation with Qwen3-ASR transcription.
* Model weights and both tokenizers are released under the **Apache 2.0 license** on Hugging Face,
  ModelScope, and GitHub (`QwenLM/Qwen3-TTS`); the training data itself is not released.

## Main Ideas

* The paper's per-concurrency latency table (Table 2) is a directly usable comparison template:
  reporting LM TTFP and tokenizer-decode TPP separately at concurrency 1/3/6 on a real serving stack
  (vLLM) is exactly the kind of per-stage breakdown this project needs to produce for CosyVoice2 and
  Chatterbox to identify where their TTFB budget is spent. Qwen3-TTS-12Hz reaches first-packet
  latency of 97-101 ms at concurrency 1, well under this project's 300 ms TTFB target, which sets a
  useful external reference point for how low a well-engineered streaming multi-codebook tokenizer
  can push latency, even though Qwen3-TTS is not a candidate replacement model itself.
* The 12Hz tokenizer's near-flat TPP scaling (4-5 ms regardless of concurrency 1-6) versus the 25Hz
  tokenizer's steep scaling (25 ms to 147 ms) demonstrates that a causal-ConvNet vocoder scales far
  better under concurrent load than a DiT+BigVGAN pipeline with lookahead — relevant if this project
  ever evaluates non-Kokoro vocoder backends or considers vocoder choice as a lever for the TTFB/RTF
  trade-off.
* The explicit trade-off between the 25 Hz (better long-form WER, higher latency) and 12 Hz (lower
  latency, slightly worse long-form WER) tokenizers is evidence that codec choice, not just serving
  engine or precision, materially shifts the latency/quality Pareto frontier — a factor this
  project's Gap analysis for CosyVoice2/Chatterbox latency reduction should consider alongside
  serving-stack and precision changes.
* No GPU model, GPU count, or absolute training compute is reported, and the exact hardware behind
  Table 2's "single typical computational resource" is not specified in enough detail to reproduce
  precisely — treat the reported latency numbers as directional evidence of what a highly optimized
  multi-codebook streaming TTS stack can achieve, not as a hardware-matched baseline for this
  project's own H100 benchmarks.

## Summary

This report introduces Qwen3-TTS, Alibaba's first TTS entry in the Qwen model family: a multilingual
(10-language), streaming, controllable, zero-shot voice-cloning text-to-speech system built around
two interchangeable discrete speech tokenizers (a 25 Hz semantic-leaning codec and a 12.5 Hz
multi-codebook codec) paired with a shared Qwen3-LM dual-track backbone. The core research question
is how to jointly minimize first-packet latency and preserve long-form stability and speaker
fidelity in an autoregressive, LLM-based TTS system intended for real-time deployment.

Methodologically, the paper separates the tokenizer design problem (semantic-heavy vs.
acoustic-heavy quantization, streaming-capable vocoders) from the LM architecture problem (a
dual-track text/speech channel-concatenation scheme with Multi-Token Prediction for multi-codebook
decoding), and trains both 0.6B and 1.7B parameter variants through a three-stage pretraining
pipeline (general, high-quality CPT, long-context) followed by a three-stage post-training pipeline
(DPO, GSPO reinforcement learning, per-speaker fine-tuning). Evaluation spans zero-shot cloning
(Seed-TTS), multilingual generation, cross-lingual transfer (CV3-Eval), instruction-following
(InstructTTSEval), and long-form stability, with latency and RTF measured separately at concurrency
1, 3, and 6 on an internal vLLM (V0 backend) deployment.

The headline finding is that the 12 Hz multi-codebook tokenizer achieves first-packet latency of
97-101 ms at concurrency 1 — far below typical zero-shot TTS TTFB — while the 25 Hz semantic
tokenizer trades higher first-packet latency (138-150 ms at concurrency 1, scaling to 481-523 ms at
concurrency 6) for better long-form WER stability (1.517/1.225 WER on a 200-2,000-word internal
Chinese/English long-speech set, versus a competing baseline's 22.6 WER collapse in Chinese).
Qwen3-TTS also reports state-of-the-art or near-state-of-the-art WER and the best speaker-similarity
scores across all 10 evaluated languages versus MiniMax-Speech and ElevenLabs Multilingual v2.

For this project, the paper matters less as a candidate replacement model (it is not evaluated
against Kokoro or David-voice fine-tuning, and its training data and hardware are not disclosed
precisely enough to reproduce) and more as (a) an external existence proof that streaming
multi-codebook LLM-TTS can reach sub-150 ms first-packet latency at concurrency 1 on a modern
serving stack, well inside this project's 300 ms TTFB target, and (b) a directly reusable per-stage,
per-concurrency latency reporting template (LM TTFP + tokenizer TPP, measured at concurrency 1/3/6)
that this task should consider adopting when profiling CosyVoice2 and Chatterbox's own TTFB budgets.
It pairs naturally with the already-reviewed Chatterbox-Flash paper as a second recent data point on
how far block/chunked streaming architectures can push zero-shot TTS latency down.
