---
spec_version: "3"
paper_id: "10.48550_arXiv.2412.10117"
citation_key: "Du2024"
summarized_by_task: "t0018_zero_shot_cloning_calibration"
date_summarized: "2026-09-17"
---
# CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models

## Metadata

* **File**: `files/du_2024_cosyvoice2-scalable-streaming-speech-synthesis.pdf`
* **Published**: 2024
* **Authors**: Zhihao Du 🇨🇳, Yuxuan Wang 🇨🇳, Qian Chen 🇨🇳, Xian Shi 🇨🇳, Xiang Lv 🇨🇳, Tianyu Zhao 🇨🇳,
  Zhifu Gao 🇨🇳, Yexin Yang 🇨🇳, Changfeng Gao 🇨🇳, Hui Wang 🇨🇳, Fan Yu 🇨🇳, Huadai Liu 🇨🇳, Zhengyan
  Sheng 🇨🇳, Yue Gu 🇨🇳, Chong Deng 🇨🇳, Wen Wang 🇨🇳, Shiliang Zhang 🇨🇳, Zhijie Yan 🇨🇳, Jingren Zhou 🇨🇳
* **Venue**: arXiv preprint (tech report, work in progress)
* **DOI**: `10.48550/arXiv.2412.10117`

## Abstract

In our previous work, we introduced CosyVoice, a multilingual speech synthesis model based on
supervised discrete speech tokens. By employing progressive semantic decoding with two popular
generative models, language models (LMs) and Flow Matching, CosyVoice demonstrated high prosody
naturalness, content consistency, and speaker similarity in speech in-context learning. Recently,
significant progress has been made in multi-modal large language models (LLMs), where the response
latency and real-time factor of speech synthesis play a crucial role in the interactive experience.
Therefore, in this report, we present an improved streaming speech synthesis model, CosyVoice 2,
which incorporates comprehensive and systematic optimizations. Specifically, we introduce
finite-scalar quantization to improve the codebook utilization of speech tokens. For the text-speech
LM, we streamline the model architecture to allow direct use of a pre-trained LLM as the backbone.
In addition, we develop a chunk-aware causal flow matching model to support various synthesis
scenarios, enabling both streaming and non-streaming synthesis within a single model. By training on
a large-scale multilingual dataset, CosyVoice 2 achieves human-parity naturalness, minimal response
latency, and virtually lossless synthesis quality in the streaming mode. We invite readers to listen
to the demos at https://funaudiollm.github.io/cosyvoice2.

## Overview

CosyVoice 2 is Alibaba's second-generation zero-shot streaming TTS system, and its core framing is a
semantic/acoustic decoupling pipeline: a text-speech language model predicts discrete supervised
semantic speech tokens autoregressively, and a separate chunk-aware conditional flow matching (CFM)
model converts those tokens into a Mel spectrogram conditioned on a speaker embedding and reference
(prompt) speech, followed by a pre-trained vocoder that reconstructs the waveform. The report frames
its contribution as four systemic changes over the original CosyVoice: (1) a single unified model
that natively supports both streaming and non-streaming synthesis by changing only how text/speech
tokens are interleaved in the LM's input sequence, not the model architecture; (2) replacing a
custom, randomly initialized LM backbone with a pre-trained textual LLM (Qwen2.5-0.5B), while
dropping the text encoder and the utterance-level speaker embedding from the LM (moving speaker
information entirely into the flow-matching stage); (3) replacing vector quantization (VQ) with
finite scalar quantization (FSQ) in the supervised speech tokenizer to eliminate codebook collapse;
and (4) upgrading instructed generation (emotion, dialect, role-play, fine-grained vocal bursts) to
be integrated into the same zero-shot model rather than a separate instruct-only variant.

The paper reports an extensive ablation program isolating the contribution of each architectural
change (LLM initialization, dropping the speaker embedding, FSQ, streaming LM, streaming flow
matching), evaluates on LibriSpeech test-clean and the SEED benchmark (test-zh/test-en/test-hard),
introduces new self-constructed Japanese and Korean benchmarks, and additionally explores
multi-speaker fine-tuning (mSFT) and reinforcement learning (DPO plus a differentiable ASR reward)
to further improve a single low-performing fine-tuned speaker. A first-package (TTFB-equivalent)
latency model is derived analytically as a function of per-token LM, flow-matching, and vocoder
compute time, explaining why the streaming architecture reduces response latency relative to fully
offline synthesis. The authors are explicit that this is a technical report ("work in progress"),
not a peer-reviewed publication, and that code and pre-trained models are released on GitHub
(FunAudioLLM/CosyVoice).

## Architecture, Models and Methods

* **Text tokenizer**: raw text is tokenized with a BPE-based tokenizer (no grapheme-to-phoneme
  frontend); multi-character Chinese BPE tokens are masked out and each character is encoded
  separately to avoid long/sparse pronunciation units.
* **Supervised semantic speech tokenizer**: FSQ is inserted into the encoder of the SenseVoice-Large
  ASR model. Encoder1 (6 Transformer blocks with rotary position embeddings) produces intermediate
  representations, projected to a D-dimensional low-rank space and bounded-rounded into `[-K, K]`,
  then re-projected up and passed through Encoder2 + ASR decoder for the ASR objective used to train
  the tokenizer. The token index is computed in a `(2K+1)`-ary positional system. Token rate: **25
  Hz** (25 speech tokens/second).
* **Text-speech LM**: **Qwen2.5-0.5B** used directly as the autoregressive backbone (no separate
  text encoder, no speaker embedding). Streaming vs. non-streaming differs only in how the input
  sequence is constructed: non-streaming concatenates `[S, text, T, speech, E]`; streaming
  interleaves text and speech tokens at a fixed ratio of **N:M = 5:15** with a "filling token"
  predicted when the next N text tokens are not yet available.
* **Chunk-aware causal flow matching**: Mel spectrogram acoustic feature at **50 Hz frame rate**,
  **24000 Hz sampling rate**; speech tokens are 2x up-sampled (with a look-ahead convolution, pad
  size P, kernel P+1) to match the Mel frame rate, then passed through causal Transformer blocks. An
  optimal-transport conditional flow matching (OT-CFM) objective is used with an L1 loss between
  predicted and ground-truth ODE vector fields; Mel masking during training randomly masks
  **70%-100%** of final frames; inference uses a cosine noise schedule and classifier-free guidance
  with strength **β = 0.7** and **10 flow-matching steps (NFE)**. Four attention masks (non-causal,
  full-causal, chunk-M, chunk-2M), randomly sampled per training example, let a single flow-matching
  model serve offline and multiple streaming latency/quality trade-offs (implicit self-distillation
  from more- to less-context masks).
* **Latency model**: first-package latency for TTS is modeled as `L_TTS = M·d_lm + M·d_fm + M·d_voc`
  (LM, flow-matching, vocoder per-token compute time), and for LLM voice chat as
  `L_Chat ≤ N·d_llm + L_TTS`.
* **Instructed generation**: 1,500 hours of instructed training data covering natural-language
  instructions (emotion, speaking rate, dialect, role-play, prepended before an `<|endofprompt|>`
  token) and fine-grained markers (`[laughter]`, `[breath]`, `<strong>...</strong>`).
* **Multi-speaker fine-tuning (mSFT)**: fine-tunes on multiple speakers simultaneously (rather than
  one), using speaker-prompt tags (e.g., `"Speaker A<|endofprompt|>"`) to avoid timbre confusion;
  learning rate **1e-5**.
* **RL for SFT**: DPO using speaker similarity (SS) and ASR word-error-rate as reward signals to
  select preferred/rejected sample pairs, plus a differentiable ASR reward (gumbel-softmax sampling
  through the LM, frozen ASR backend) used because full DPO requires 4 forward TTS passes per
  training step and is compute-heavy; 10,000 sample-pairs synthesized for DPO.
* **Training data**: speech tokenizer trained on 200,000 hours (**110,884 h Chinese, 99,918 h
  English**); CosyVoice 2 itself trained on **130,000 h Chinese, 30,000 h English, 4,600 h Japanese,
  2,200 h Korean**, with pseudo-labels from Paraformer/SenseVoice and a force-alignment filter for
  quality/punctuation.
* **Evaluation**: LibriSpeech test-clean (Whisper-large-V3 ASR, ERes2Net for SS, NMOS/DNSMOS-P.835
  for quality) and the SEED benchmark (test-zh ~2,000 samples, test-en ~1,000 samples, test-hard
  ~400 hard cases from CommonVoice, both WavLM-based SV and ERes2Net SV models for SS). New
  self-built Japanese (1,000 samples) and Korean (1,000 samples) test sets from CommonVoice.

## Results

* LibriSpeech test-clean: CosyVoice 2 reaches **2.47% WER**, **3.96 NMOS**, **0.745 SS**, surpassing
  ChatTTS, GPT-SoVITs, OpenVoice, ParlerTTS, EmotiVoice, and CosyVoice v1 on every metric, and
  exceeding the human reference row (2.66% WER, 3.84 NMOS, 0.697 SS) on all three.
* Streaming variant CosyVoice 2-S on the same set: **2.45% WER**, **3.90 NMOS**, **0.751 SS** —
  essentially lossless vs. the offline model.
* SEED test-zh: CosyVoice 2 achieves **1.45% CER** and SS of **0.748 (0.806 with ERes2Net)**,
  beating all open-source baselines (F5-TTS: 1.56% CER, 0.741/0.794 SS; MaskGCT: 2.27% CER;
  CosyVoice v1: 3.63% CER) and trailing only the closed-source Seed-TTS (1.12% CER, 0.796 SS) by a
  small margin.
* SEED test-en: CosyVoice 2 gets **2.57% WER**, SS **0.652 (0.736 ERes2Net)** — ranked 4th on WER
  and 3rd on SS among compared systems, attributed to English being a much smaller share of training
  data than Chinese.
* SEED test-hard: CosyVoice 2 achieves **6.83% WER**, SS **0.724 (0.776 ERes2Net)**, the best result
  among all compared baselines (vs. F5-TTS 8.67% WER, MaskGCT 10.27% WER, Seed-TTS 7.59% WER) —
  demonstrating robustness on tongue-twisters/repetition.
* Ablation (Table 7) isolates each design change on test-zh CER: base CosyVoice v1 **3.63%** →
  + pretrained LLM init **2.96%** (**18.46%** relative WER improvement on test-zh, **15.40%** on
    test-hard) → + drop speaker embedding **2.56%** → + FSQ (= CosyVoice 2) **1.45%** → + pitch loss
    **1.19%**, while SS stays roughly flat (~0.80-0.81) throughout, showing content accuracy is
    driven by the LM/tokenizer path and speaker fidelity by the flow-matching path.
* Streaming-module ablation (Table 8, chunk size 15): fully streaming LM+FM (M4) reaches **1.45% CER
  / 0.812 SS** on test-zh vs. **1.45% CER / 0.806 SS** fully offline (M1) — streaming costs
  essentially nothing on typical cases, with the largest degradation confined to test-hard (6.83%
  offline vs. 8.08% streaming CER).
* Japanese/Korean benchmarks: test-ja **18.79% CER, 0.630 SS, 3.42 NMOS**; test-ko **7.98% CER,
  0.707 SS, 3.73 NMOS** — Japanese is markedly worse due to character-set overlap with Chinese
  causing mispronunciations.
* Instructed generation (in-house 290-sample Chinese test set, 29 instruction types): CosyVoice 2
  reaches **1.52% CER, 0.804 SS, 3.94 NMOS, 4.06 MOS-I** vs. CosyVoice-Instruct's **1.72% CER, 0.797
  SS, 3.94 NMOS, 3.09 MOS-I** — instruction accuracy/naturalness (MOS-I) improves by roughly 1 point
  while content and speaker metrics also improve slightly.
* Speaker fine-tuning + RL (Spk E, hardest in-house speaker): RL with combined differentiable ASR
  reward and DPO (`L_ASR + L_DPO`) reduces the in-house WER from the SFT baseline's **7.15%** to
  **6.64%** while maintaining NMOS **3.97** and SS **0.796**, and on the SEED test-hard subset cuts
  WER from **7.90%** (plain SFT) to **6.66%**.
* Speech tokenizer codebook comparison (Table 4): FSQ achieves **100% codebook utilization**
  (6,561/6,561 codes) vs. VQ's **23%** (963/4,096), and lower ASR error rate across
  CommonVoice/Fleurs EN/CN test sets (e.g., CommonVoice EN: FSQ 10.67% vs. VQ 18.26%).

## Innovations

### Unified Streaming/Non-Streaming Text-Speech LM

A single LM checkpoint serves both modes purely by changing how text and speech tokens are
interleaved in the training/inference sequence (N:M = 5:15 ratio for streaming), rather than
requiring separate streaming and offline models. This removes the deployment complexity of
maintaining two model variants and is shown to cost almost nothing in content accuracy on typical
text (only degrading noticeably on adversarial "hard" text).

### Finite Scalar Quantization (FSQ) Speech Tokenizer

Replaces VQ in the SenseVoice-Large-based supervised speech tokenizer, eliminating codebook collapse
(100% vs. 23% utilization) and improving downstream ASR-measured content preservation, which
propagates into a large WER/CER improvement in the full TTS pipeline (test-zh CER 2.56% → 1.45% from
FSQ alone, per the ablation).

### Chunk-Aware Causal Flow Matching with Four Attention Masks

A single CFM/UNet model is trained with four randomly-sampled causal attention masks (non-causal,
full-causal, chunk-M, chunk-2M), letting one model span the full latency/quality trade-off curve and
implicitly self-distill from high-context to low-context masks. This is presented as generalizable
beyond CosyVoice 2 to other non-autoregressive TTS models seeking streaming support.

### LLM-Backbone Simplification (Drop Text Encoder + Speaker Embedding)

Directly using Qwen2.5-0.5B as the text-speech LM backbone, while removing the previous CosyVoice's
text encoder and utterance-level speaker embedding, both simplifies the architecture and measurably
improves content consistency — the paper attributes this to eliminating information
leakage/entanglement between speaker identity and linguistic content in the LM's conditioning.

### Reinforcement Learning for TTS Fine-Tuning (DPO + Differentiable ASR Reward)

Applies DPO with WER/SS-based preference labeling to a TTS LM, and additionally proposes a cheaper
differentiable ASR reward (gumbel-softmax through the frozen ASR backend of the speech tokenizer)
that avoids the 4x-forward-pass cost of standard DPO sampling, showing better generalization to
out-of-domain SEED test sets than DPO alone.

## Datasets

* **Speech tokenizer training**: 200,000 hours total — 110,884 h Chinese, 99,918 h English, drawn
  from open-source ASR datasets, internal industrial datasets, and TTS-generation datasets.
* **CosyVoice 2 training**: 130,000 h Chinese, 30,000 h English, 4,600 h Japanese, 2,200 h Korean;
  pseudo-labeled via Paraformer (Chinese) / SenseVoice (other languages) with internal
  force-alignment filtering. Not publicly released as a downloadable corpus in this report.
* **Instructed-generation training data**: 1,500 hours covering natural-language and fine-grained
  instruction types (emotion, speaking rate, dialect, role-play, vocal bursts).
* **Evaluation — LibriSpeech test-clean**: standard public English ASR/TTS benchmark subset.
* **Evaluation — SEED test sets** (from Anastassiou et al. 2024, Seed-TTS): test-zh (~2,000 Chinese
  samples), test-en (~1,000 English samples), test-hard (~400 hard cases: text repetition, tongue
  twisters), all built from CommonVoice reference/target pairs.
* **Evaluation — Japanese/Korean**: self-constructed test-ja (1,000 CommonVoice JA samples, 8-32
  character utterances) and test-ko (1,000 CommonVoice KO samples, WER < 5% and no
  deletion/insertion errors per Whisper-Large-V3 filtering); prompt speech, prompt transcriptions,
  and input text lists are released by the authors for reproducibility.
* **Evaluation — instructed generation**: in-house 290-sample Chinese test set (29 instruction types
  x 10 texts), 5 speaker prompts (3 female, 2 male), evaluated by 10 native Chinese speakers for
  MOS-I.
* **Speaker fine-tuning**: target speakers with as few as 400 audio recordings; Spk E dataset is
  Chinese-only with a faster/more complex voice, used as the RL fine-tuning case study.
* All datasets besides the released Japanese/Korean test-set lists and standard public corpora
  (LibriSpeech, CommonVoice) are internal to Alibaba and not publicly released with this report.

## Main Ideas

* CosyVoice 2 is a direct benchmark target for this project's zero-shot cloning calibration work:
  its official SEED test-zh/test-en/test-hard SS and WER/CER numbers, and its LibriSpeech test-clean
  NMOS/SS numbers, give externally reported reference points to compare against our own reproduction
  of CosyVoice 2 (or F5-TTS/Chatterbox) under this project's own benchmark protocol.
* The paper's own latency model (`L_TTS = M·d_lm + M·d_fm + M·d_voc`) is directly relevant to how we
  reason about and decompose TTFB in our own streaming pipeline — it separates LM token generation,
  flow-matching, and vocoder time as the three additive latency components per output chunk.
* CosyVoice 2 reports that WavLM-based and ERes2Net-based speaker-similarity scores are not
  consistent with each other (Section 4.2) — this is a direct, citable caution for our own
  GE2E-cosine-based speaker-similarity metric: cross-system SS numbers are only comparable when
  computed with the same speaker-verification model, so CosyVoice 2's published SS numbers are not
  directly comparable to our GE2E cosine ≥ 0.85 target without recomputing SS with our own scorer.
* The chunk-aware flow matching design (single model spanning multiple causal/latency configurations
  via randomly sampled attention masks) is a candidate architecture pattern if this project ever
  needs a single model that trades off latency vs. quality at inference time, rather than training
  separate streaming/offline checkpoints.
* The "Limitations" section explicitly states CosyVoice 2 cannot control timbre via text
  instructions and performs poorly on singing — worth noting as known failure modes if our benchmark
  stresses those capabilities.

## Summary

This technical report from Alibaba Group presents CosyVoice 2, the second generation of the
CosyVoice zero-shot text-to-speech system, motivated by the need for low-latency, streaming-capable
speech synthesis in interactive LLM-based voice applications where prior zero-shot TTS models
(including CosyVoice v1) operated only in offline (whole-utterance) mode. The paper's research
question is how to unify streaming and non-streaming zero-shot TTS into a single model without
sacrificing the naturalness, content consistency, and speaker similarity that made codec-language-
model-based TTS competitive with human speech.

Methodologically, the system keeps the semantic/acoustic decoupling of the original CosyVoice — an
autoregressive text-speech LM predicts discrete semantic speech tokens, and a separate flow-
matching model converts tokens plus a speaker embedding and reference audio into a Mel spectrogram
for vocoding — but makes four systemic changes: swapping the LM backbone for a pre-trained
Qwen2.5-0.5B model while removing the text encoder and speaker embedding from the LM, replacing
vector quantization with finite scalar quantization in the speech tokenizer to fix codebook
collapse, introducing a chunk-aware causal flow-matching model trained with four randomly sampled
attention masks so one model covers the full offline-to-fully-causal latency spectrum, and folding
instructed generation (emotion, dialect, role-play, vocal bursts) into the same zero-shot model.

The headline findings are that CosyVoice 2 reaches 2.47% WER / 3.96 NMOS / 0.745 SS on LibriSpeech
test-clean (exceeding the human reference on all three metrics), 1.45% CER / 0.806 SS on SEED
test-zh (best among open-source systems, close to closed-source Seed-TTS), and its streaming variant
is described as "nearly lossless" relative to the offline model across typical test cases, with the
largest streaming-mode degradation confined to adversarial "hard" text. Ablations attribute most of
the content-accuracy gain to the pretrained-LLM backbone and FSQ tokenizer, while speaker similarity
is governed mainly by the flow-matching stage.

For this project, CosyVoice 2 is one of three zero-shot voice-cloning systems being directly
benchmarked (alongside F5-TTS and Chatterbox), and this paper is the sole primary source for its
official SS/WER/NMOS tables and the architectural basis of its streaming TTFB claims. Two points are
directly actionable: the paper's own caution that SS scores are not comparable across different
speaker-verification backbones (WavLM vs. ERes2Net) means CosyVoice 2's published SS numbers should
not be compared directly to this project's GE2E-cosine metric without recomputation under a common
scorer, and its explicit latency decomposition (LM + flow-matching + vocoder time per chunk) is a
useful template for analyzing our own pipeline's TTFB budget.
