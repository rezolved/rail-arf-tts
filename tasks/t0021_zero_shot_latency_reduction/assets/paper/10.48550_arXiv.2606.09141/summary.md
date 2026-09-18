---
spec_version: "3"
paper_id: "10.48550_arXiv.2606.09141"
citation_key: "Xie2026"
summarized_by_task: "t0021_zero_shot_latency_reduction"
date_summarized: "2026-09-18"
---
# FlashTTS: Fast Streaming TTS with MTP Acceleration and X-pred Mean Flow Distillation

## Metadata

* **File**: `files/xie_2026_flashtts.pdf`
* **Published**: 2026
* **Authors**: Hanke Xie 🇨🇳, Xiaming Ren 🇨🇳, Dake Guo 🇨🇳, Ruonan You 🇨🇳, Wenhao Li 🇨🇳, Jingbin Hu
  🇨🇳, Guobin Ma 🇨🇳, Huakang Chen 🇨🇳, Kejie Xu 🇨🇳, Rui Huang 🇨🇳, Weiguo Tan 🇨🇳, Xianrong Wang 🇨🇳, Lei
  Xie 🇨🇳
* **Venue**: Interspeech 2026
* **DOI**: `10.48550/arXiv.2606.09141`

## Abstract

Recent progress in speech dialogue systems requires Text-to-Speech (TTS) models to be faster and
more responsive. Modern speech dialogue systems impose two primary requirements on TTS models: low
latency and support for streaming inputs and outputs. However, most existing single-codebook
LLM-based TTS methods rely on multi-stage pipelines that lack native streaming capabilities. These
systems typically suffer from high end-to-end latency due to slow autoregressive prediction and
multi-step flow matching. To address these limitations, we propose FlashTTS, an open-source and
low-latency streaming TTS framework. FlashTTS introduces a lagged multi-track architecture that
natively processes streaming text and speech inputs, thereby eliminating the need for sentence-level
buffering. To accelerate acoustic generation, we integrate parallel Multi-Token Prediction (MTP)
with an X-pred mean flow matching decoder. This configuration achieves high-fidelity token-to-mel
generation in exactly two function evaluations (2-NFE). By jointly optimizing input processing and
decoding efficiency, FlashTTS offers a practical foundation for real-time speech dialogue systems.
Experiments show that FlashTTS substantially reduces First-Packet Latency to 325ms compared to
robust streaming baselines, all while preserving strong zero-shot voice cloning and cross-lingual
intelligibility. Speech samples are available. The model code and checkpoints will be released as
open source.

## Overview

FlashTTS is a Qwen2.5-0.5B-based streaming TTS system built by the ASLP lab at Northwestern
Polytechnical University with Huawei, targeting the specific latency bottlenecks that make LLM-based
TTS unsuitable for real-time speech dialogue. The paper identifies two separate latency sources in
conventional streaming TTS pipelines such as CosyVoice2: (1) input-side latency caused by
sentence-level text buffering before synthesis can begin, and (2) output-side latency caused by slow
autoregressive token prediction combined with multi-step (10+ NFE) flow matching for token-to-mel
decoding. FlashTTS attacks both sources independently: a "stacked and lagged" parallel track input
scheme (speech, text, language tracks) lets the model begin generating audio from a single input
token rather than buffering whole sentences, and a combination of parallel Multi-Token Prediction
(MTP, inspired by DeepSeek-V3) with a distilled 2-NFE "X-pred" mean-flow decoder collapses the
autoregressive-plus-flow-matching decode path into a small, fixed number of steps.

The system is trained in two stages: Stage 1 trains the full decoder-only backbone end to end on the
stacked-track inputs with a standard mean-flow objective; Stage 2 freezes the backbone and trains
only lightweight parallel MTP heads (plus a distilled X-pred mean-flow decoder) to predict multiple
future speech tokens and mel frames per step, using a frozen-backbone verification mechanism
borrowed from Llasa+ to keep speculative multi-token predictions stable. Evaluated against
CosyVoice2 (0.5B, its primary same-scale baseline) and several larger zero-shot TTS systems
(Seed-TTS, MaskGCT, F5-TTS, Llasa-8B, Spark-TTS) plus two commercial systems (MiniMax, ElevenLabs),
FlashTTS's best configuration (MTP-3, 2-NFE) cuts First-Token Latency to 60ms and First-Packet
Latency to 325ms — roughly a 2.6x FPL reduction versus the CosyVoice2 (10-NFE) baseline's 843ms —
while keeping WER and speaker similarity (SIM) broadly competitive, at some cost to SIM relative to
larger, slower models.

## Architecture, Models and Methods

FlashTTS is built on the Qwen2.5-0.5B backbone (hidden size 896, 24 layers, 14 attention heads,
feed-forward dimension 4864). Speech is tokenized with S3Tokenizer v2 (ASR-derived semantic tokens),
chosen for token stability that eases LLM-side prediction and enables a fair token-level comparison
against CosyVoice2. Input is organized into three parallel "stacked and lagged" tracks (Figure 2 of
the paper): a speech track (initialized with a speaker embedding, followed by generated speech
tokens), a text track (ingests text tokens, then pads once text is exhausted), and a continuous
language-conditioning track — this removes the need to buffer a full sentence of text before
acoustic modeling begins.

Stage 2 adds a Multi-Token Prediction (MTP) module set, each comprising a linear projection plus a
Qwen2.5-style decoder block; parallel MTP modules independently transform the frozen backbone's
final hidden states h0(0:t) to predict k = 1 … N-1 future tokens simultaneously (Eq. 1-2 in the
paper), trained with cross-entropy against shifted ground-truth speech tokens while the backbone
stays frozen. A frozen-backbone verification step (following Llasa+) checks the lightweight MTP
modules' speculative token predictions for stability. The acoustic decoder ("X-pred Mean Flow")
combines the Mean Flow identity (Geng et al., 2025) with JIT's explicit data-prediction
parameterization (Li & He, 2025): rather than predicting the velocity field directly, the network
predicts the clean mel-spectrogram x, from which the average velocity is analytically derived (Eq.
5), enabling accurate synthesis in as few as 1-2 NFE. Block-wise chunked attention supports
streaming output. The X-pred MeanFlow module is a 16-layer Diffusion Transformer (hidden dim 768,
~159.25M parameters) followed by a 50M-parameter HiFi-GAN 24kHz vocoder, distilled from a pretrained
conditional flow-matching teacher.

Training: Stage 1 optimizes the full backbone with a dynamic frame-based batch size of 40,000 on 8
A100 GPUs using AdamW, peak learning rate 1e-4 with 20k warmup steps and cosine decay over 1M steps.
Stage 2 trains only the MTP modules on 4 A100 GPUs at peak LR 5e-5, while the X-pred MeanFlow
decoder is distilled separately on 8 RTX 4090 GPUs with batch size 2,000, gradient accumulation of
2, and peak LR 7e-5. Training data totals ~300,000 hours: Emilia, Emilia-Yodas, LibriHeavy, and
WenetSpeech4TTS. Latency is measured on a single RTX 4090 GPU with the textual input stream
generated live by an upstream Qwen2-7B model to simulate a real conversational pipeline, with no
engineering-level latency optimizations applied (to isolate architectural speed). Evaluation
metrics: WER/CER (Paraformer-zh for Chinese, Whisper-large-v3 otherwise), SIM (cosine similarity of
speaker embeddings from a fine-tuned WavLM-large model), CMOS (100 test pairs, 30 listeners, 95%
confidence intervals), Speedup Ratio, RTF, TPS, and FPL.

## Results

* Best FlashTTS configuration (MTP-3, 2-NFE) reaches **First-Packet Latency (FPL) 325 ms** versus
  **843 ms** for the CosyVoice2 (10-NFE) baseline on the Minimax zh/en/ja/ko subset — roughly a 2.6x
  reduction.
* Same MTP-3 (2-NFE) config cuts **First-Token Latency (FTL) to 62 ms** vs. **257 ms** for
  CosyVoice2, and RTF drops to **0.632** vs. CosyVoice2's **0.913**.
* Throughput (TPS) rises from **51** (CosyVoice2) to **73** (MTP-3, 2-NFE) and **75** (MTP-5,
  2-NFE); FlashTTS configurations also improve WER on this Minimax subset, from CosyVoice2's
  **26.2%** down to **18.0-20.8%** across FlashTTS variants.
* MTP-5 (2-NFE) shows diminishing/negative returns: WER worsens to **20.8%**, SIM drops to
  **0.668**, and subjective CMOS turns negative (**-0.08**) versus the CosyVoice2 reference,
  identifying MTP-3 as the better operating point.
* On the six-language MiniMax multilingual test set, FlashTTS achieves **1.08% WER / 0.743 SIM** on
  Chinese and **3.02% WER / 0.662 SIM** on English, versus CosyVoice2's **1.22% WER / 0.773 SIM**
  (zh) and **3.44% WER / 0.765 SIM** (en); CosyVoice2 does not support French/German while FlashTTS
  does (**8.62% WER / 0.564 SIM** fr; **9.96% WER / 0.672 SIM** de).
* On Seed test sets, FlashTTS (stage 1) reaches **1.38% CER / 0.718 SIM** (test-zh) and **2.21% WER
  / 0.572 SIM** (test-en), below larger offline models such as Seed-TTS (**1.12% CER / 0.796 SIM**
  zh; **2.25% WER / 0.762 SIM** en).
* Ablation study: removing X-pred cuts the speed-up ratio from **49.23%** to **12.53%** (WER worsens
  to 2.28%, SIM to 0.691); removing MTP cuts speed-up to **12.52%**; removing the language ID
  conditioning keeps speed-up at **49.28%** but WER worsens sharply to **3.42%**.

## Innovations

### Lagged Multi-Track Streaming Input

A "stacked and lagged" input scheme with parallel speech, text, and language tracks lets the model
begin generation from a single incoming text token instead of buffering an entire sentence, unlike
CosyVoice2 which needs to buffer five text tokens before synthesis starts. This directly targets
input-side latency, which prior streaming-TTS work (interleaved generation schemes) had not fully
eliminated.

### Parallel Multi-Token Prediction (MTP) with Frozen-Backbone Verification

Adapting DeepSeek-V3's MTP idea to TTS, FlashTTS attaches lightweight parallel decoder modules to a
frozen backbone that each predict a different future token offset simultaneously (rather than
sequentially), motivated by the empirical observation that text token frame rate is only 3-5 Hz. A
verification step reusing the frozen backbone's more robust probability distributions checks the
lightweight modules' speculative predictions, addressing the instability that naive parallel token
prediction otherwise introduces.

### X-pred Mean Flow Distillation for 2-NFE Decoding

By combining Mean Flow theory with JIT's explicit data-prediction (x-prediction) parameterization,
the acoustic decoder is trained to predict the clean mel-spectrogram directly rather than a velocity
field, from which the average velocity is analytically derived. This yields stable, high-fidelity
token-to-mel synthesis in as few as 1-2 neural function evaluations, versus the 10+ steps
conventional conditional flow matching (as used in CosyVoice2) requires — directly attacking the
flow-matching term that dominates output-side latency in prior streaming TTS.

## Datasets

* **Training corpus**: ~300,000 hours total, combining Emilia, Emilia-Yodas, LibriHeavy, and
  WenetSpeech4TTS — all open-source, multilingual speech datasets used broadly in the TTS
  literature.
* **Seed-TTS test sets** (`test-zh`, `test-en`): public zero-shot evaluation benchmark released
  alongside Seed-TTS, used here for CER/WER and SIM comparison against MaskGCT, F5-TTS,
  Llasa-8B-250k, Spark-TTS, CosyVoice2, and Seed-TTS itself.
* **MiniMax multilingual test set**: a six-language (Chinese, English, Japanese, Korean, French,
  German) zero-shot evaluation set (`MiniMaxAI/TTS-Multilingual-Test-Set` on Hugging Face), used for
  both the latency table (Minimax zh/en/ja/ko subset) and the multilingual WER/SIM table.
* Speaker similarity is computed against reference speaker embeddings extracted with a fine-tuned
  WavLM-large model — a different SIM extractor than the GE2E model used in this project's
  benchmark, so absolute SIM values are not directly comparable across the two setups.

## Main Ideas

* FlashTTS's achieved **FPL of 325 ms** is very close to this project's own **TTFB ≤ 300 ms**
  target, giving a concrete external reference point for what a modern 0.5B-scale LLM backbone plus
  a distilled few-step flow decoder can achieve end to end — useful context when judging whether
  Kokoro-82M's latency budget is competitive with the current state of the art.
* The paper's core latency lever — collapsing a 10+-NFE flow-matching decoder down to 2-NFE via
  X-pred Mean Flow distillation — is directly analogous to any diffusion/flow decoding step count in
  Kokoro/StyleTTS2's decoder; reducing NFE is a candidate latency-reduction technique worth
  evaluating if the project's decoder currently uses many sampling steps.
* The MTP-5 ablation is a cautionary result: pushing parallel token prediction too aggressively (5
  branches vs. 3) improved throughput only marginally while measurably degrading WER, SIM, and
  subjective CMOS (turning negative). This is a concrete example of a latency/quality Pareto
  frontier that should inform how aggressively any zero-shot latency-reduction technique is pushed
  in this project.
* FlashTTS's own results show latency-optimized configurations trade off speaker similarity (SIM
  0.695-0.714 for its fastest zero-shot configs) against slower, larger models (Seed-TTS reaches
  0.762-0.796 SIM) — reinforcing that the project's dual bar of GE2E cosine ≥ 0.85 and TTFB ≤ 300 ms
  is a genuinely hard joint target, not one where latency wins come for free.
* Because SIM here is computed with WavLM-large embeddings rather than GE2E, none of FlashTTS's or
  the compared commercial systems' (MiniMax, ElevenLabs) SIM numbers can be used as literal targets
  for this project's GE2E-based ≥ 0.85 threshold; they are useful only as relative,
  order-of-magnitude context.

## Summary

FlashTTS addresses a well-defined problem: existing single-codebook LLM-based TTS systems used in
real-time speech dialogue suffer from two compounding latency sources — sentence-level text
buffering before generation can start, and slow autoregressive token prediction combined with
multi-step flow matching for mel decoding. The paper's research question is whether both sources can
be attacked simultaneously without sacrificing the zero-shot voice cloning and multilingual
intelligibility that made LLM-based TTS attractive in the first place.

Methodologically, FlashTTS combines three ideas on a Qwen2.5-0.5B backbone: a stacked/lagged
multi-track input scheme that removes the need for sentence buffering, a DeepSeek-V3-style parallel
Multi-Token Prediction (MTP) module with frozen-backbone verification for stable multi-token
speculative decoding, and an "X-pred" Mean Flow acoustic decoder distilled from a conditional
flow-matching teacher that produces high-fidelity mel-spectrograms in just 2 neural function
evaluations. Training proceeds in two stages — full-backbone pretraining, then frozen-backbone
MTP/X-pred distillation — on ~300,000 hours of open multilingual speech data.

The headline finding is that the best configuration (MTP-3, 2-NFE) cuts First-Packet Latency to 325
ms and First-Token Latency to 60 ms, versus 843 ms and 257 ms for a comparably-sized CosyVoice2
(10-NFE) baseline, while keeping WER and SIM broadly competitive on multilingual zero-shot
benchmarks (Minimax subset, Seed-TTS test sets). The paper also shows the limits of this approach:
pushing MTP branches from 3 to 5 yields diminishing throughput gains and a net-negative CMOS, and
FlashTTS's SIM consistently trails larger, non-streaming models like Seed-TTS.

For this project, FlashTTS is directly relevant as an external latency benchmark close to the 300 ms
TTFB target, and its central technique — distilling a many-step flow-matching decoder down to 2 NFE
— is a transferable idea for any flow/diffusion decoder stage in the Kokoro/StyleTTS2 pipeline. Its
ablations are also a useful cautionary data point: aggressive latency optimization measurably erodes
speaker similarity, which is exactly the tension this project must resolve between the GE2E cosine ≥
0.85 and TTFB ≤ 300 ms success criteria. Its SIM metric (WavLM-large cosine) differs from this
project's GE2E-based metric, so numeric SIM values should be treated as contextual rather than
directly comparable.
