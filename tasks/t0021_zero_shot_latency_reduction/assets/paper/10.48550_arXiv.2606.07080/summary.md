---
spec_version: "3"
paper_id: "10.48550_arXiv.2606.07080"
citation_key: "Lian2026"
summarized_by_task: "t0021_zero_shot_latency_reduction"
date_summarized: "2026-09-18"
---
# dots.tts Technical Report

## Metadata

* **File**: `files/lian_2026_dots-tts-technical-report.pdf`
* **Published**: 2026
* **Authors**: Shi Lian 🇨🇳, Changtao Li 🇨🇳, Bohan Li 🇨🇳, Hankun Wang 🇨🇳, Da Zheng 🇨🇳, Junfeng Tian
  🇨🇳, Yufeng Ma 🇨🇳, Colin Zhang 🇨🇳, Kai Yu 🇨🇳
* **Venue**: arXiv preprint
* **DOI**: `10.48550/arXiv.2606.07080`

## Abstract

We present dots.tts, a 2B-parameter continuous autoregressive text-to-speech (TTS) foundation model
that models speech in a continuous latent space. Compared with existing continuous autoregressive
models, our key innovations are threefold. First, we train an AudioVAE with multiple objectives to
build a semantically structured and prediction-friendly continuous speech space. Second, we use
full-history conditioning in the flow-matching head to preserve long-range consistency and reduce
drift during generation. Third, we apply reward-free self-corrective post-training to the
flow-matching head to further improve robustness and acoustic quality. After being trained on a
large-scale multilingual corpus, dots.tts achieves the best average performance on Seed-TTS-Eval,
with WERs of 0.94%/1.30%/6.60% and SIM scores of 81.0/77.1/79.5 on the zh/en/zh-hard test sets,
respectively. Across other benchmarks, dots.tts also consistently demonstrates open-source
state-of-the-art performance, exhibiting strong generation stability, voice cloning ability, and
emotional expressiveness. For efficient inference, we further apply CFG-aware MeanFlow distillation,
enabling low-latency speech generation with first-packet latencies of 85/54 ms in output streaming
and dual-streaming modes, respectively. To facilitate reproducible research and practical
deployment, we release the training and inference code, together with the pretrained, post-trained,
and MeanFlow-distilled checkpoints, under the Apache 2.0 license.

## Overview

dots.tts is a joint release from dots / Xiaohongshu Inc. and the X-LANCE Lab at Shanghai Jiao Tong
University that targets two open problems in continuous-latent autoregressive (AR) TTS: long-range
error accumulation during AR rollouts (continuous latents have no quantization buffer to snap
imperfect predictions back onto a valid manifold, unlike discrete-token codecs) and an immature
post-training stack relative to the discrete-token cascade. The backbone decomposes generation into
three specialized modules — a causal semantic encoder, a text LLM initialized from Qwen2.5-1.5B
Base, and an autoregressive flow-matching (AR-FM) head — so that long-range semantic planning and
local acoustic rendering do not compete inside a single module. The LLM never sees the raw 128-dim
VAE latent directly; it only consumes a 6.25 Hz semantic summary produced by a frozen causal
semantic encoder, which the paper reports is necessary to keep continuous-AR rollouts stable.

The paper's central empirical claim is that this decomposition, combined with a two-stage AudioVAE
training recipe (BigVGAN-v2-style reconstruction losses plus a second stage adding a WavLM alignment
loss and multitask ASR/emotion/speaker supervision to make the latent "learnable"), a reward-free
self-corrective post-training stage adapted from SOAR, and CFG-aware MeanFlow distillation together
let a fully continuous AR model reach state-of-the-art zero-shot voice-cloning quality while
remaining deployable in real time. Training used 1.5M hours of speech (1.2M in-house
Chinese/English, ~300K hours open-source multilingual corpora, ~7K hours caption-paired data). The
self-corrective post-training stage ("SOAR") exposes the frozen AR-FM head's own off-trajectory
rollout errors to a supervised correction target without any reward model or external teacher,
directly addressing the multi-step ODE mismatch between teacher-forced pretraining and free-running
inference. MeanFlow distillation then compresses the flow-matching ODE solve from an effective NFE
of 20 (10 Euler steps with classifier-free guidance) down to NFE ∈ {2, 3, 4} with a single
conditional forward pass per step, since CFG is fused into the distillation target.

On efficiency, measured on a single NVIDIA H800 GPU via vLLM with continuous batching and paged-KV
attention (LLM side) plus torch.compile JIT-compilation (AR-FM head and semantic encoder), the
MeanFlow-distilled model at NFE=4 reaches first-packet latency of 85.4 ms in plain (text-prefix)
streaming mode and 54.4 ms in 1T1A (one-text-token-one-audio-step) interleaved dual-streaming mode,
at RTF 0.231 and 0.245 respectively. The paper explicitly reports that all numbers in its accuracy
tables (Tables 1-5) use float32 with torch.compile disabled — the paper does not describe a specific
mixed-precision or quantization scheme (e.g., bf16 for flow-matching/vocoder vs. int8/int4 for the
LLM trunk) for either the accuracy evaluation runs or the reported efficiency numbers in Section
3.4; readers seeking such a precision-split design should treat that as inferred/external
information, not a claim made in this paper's text.

## Architecture, Models and Methods

**AudioVAE**: encodes 48 kHz mono speech to a 128-dimensional continuous latent at 25 Hz (1920x
temporal downsampling), decoded via a causal BigVGAN-v2-style decoder. The causal convolutional
encoder uses downsample strides [2, 2, 2, 4, 6, 10] ending in a posterior projection producing
per-frame mean/log-variance, plus flow regularization on top of the standard KL prior. Stage 1
training (500K steps on 9.6-second crops) uses multi-period + multi-scale sub-band CQT adversarial
loss, multi-scale mel-spectral reconstruction loss, feature-matching loss, and KL+flow
regularization, following the BigVGAN-v2 recipe. Stage 2 (200K further steps) adds a frame-level
cosine alignment loss against the 23rd-layer hidden representation of a frozen WavLM teacher, plus a
multitask downstream encoder-LLM block jointly trained on ASR, emotion, and speaker-classification
objectives (the downstream LLM head is discarded post-training; only the encoder is kept as the
semantic frontend). AdamW optimizer, beta=(0.8, 0.99), eps=1e-6, exponential LR decay 1e-4 to 1e-6.

**LLM backbone**: initialized from Qwen2.5-1.5B Base, consumes BPE text tokens directly (not
phonemes) interleaved or prefixed with a 6.25 Hz audio-semantic embedding stream. Two sequence
layouts: "plain" (full text prefix, standard TTS) and "1T1A" interleaved (one BPE token alternates
with one 6.25 Hz audio step until an eot marker, then audio continues alone) for low-latency dual
streaming with an upstream conversational LLM.

**Semantic encoder**: reused Stage-2 AudioVAE supervision module, transplanted into the backbone
with pretrained weights frozen. A strided causal-convolution projector halves the frame rate,
followed by a 24-layer causal Transformer (hidden dim 1024, FFN dim 4096), grouped in pairs along
time and linearly projected to the LLM embedding dimension — an end-to-end 4x downsampling from 25
Hz latent frames to 6.25 Hz LLM tokens.

**AR flow-matching (AR-FM) head**: an 18-layer Diffusion Transformer (DiT; hidden dim 1024, FFN dim
4096\) with RoPE on every layer, RMSNorm with QK-norm, and adaLN-zero modulation driven by the
diffusion timestep and a speaker-embedding side input (frozen CAM++ x-vector encoder). Trained with
the rectified-flow objective against the straight-line velocity between Gaussian noise and the clean
4-frame (25 Hz) VAE-latent patch. Training uses a block-causal attention mask over a concatenated
cause/generation sequence so a single parallel forward pass reproduces the exact per-step
autoregressive context seen at inference (Figure 2). Classifier-free guidance is applied by dropping
the LLM hidden stream and speaker stream independently with probability p_drop=0.5 during training.

**Post-training**: (1) Self-corrective alignment ("SOAR", reward-free): only the DiT is updated;
global batch of 4 audio-hours, 50K steps, 5K-step linear warmup to peak LR 3e-5, cosine decay to
2e-6, auxiliary correction weight lambda_aux=1.0, self-correction CFG scale gamma_soar=1.2, K_aux=6
auxiliary correction states per sample, logit-normal integration-time sampling. (2) CFG-aware
MeanFlow distillation: freezes the SOAR-corrected DiT as teacher, trains a student DiT (same
backbone + interval-duration embedder) to predict mean velocity over teacher trajectories built with
a 16-step Euler solver and CFG scale gamma_mf fused into the target; 50K steps, 8-hour global batch,
peak LR 1e-4, 5K-step warmup, cosine decay to 2e-6, log-normal interval sampling (mean -0.4, std
1.0), anchor-sample mixing probability 0.5. Pretraining itself follows a 3-stage WSD
(Warmup-Stable-Decay) schedule: Stage 1 modality alignment (frozen LLM, only semantic encoder
+ AR-FM head trained, 100K steps on Emilia only, 0.5-hour batches, reaches ~42% WER on
  Seed-TTS-Eval); Stage 2 general training (all modules unfrozen, full 1.5M-hour data mixture,
  8-hour batches, 700K steps, ~4 epochs); Stage 3 annealing (100K steps, ~1 epoch, higher-quality
  re-filtered subset, LR annealed 2e-4 to 3e-5).

## Results

* Seed-TTS-Eval average across test-en/test-zh/test-zh-hard: dots.tts (SOAR) reaches **2.95% WER**
  and **79.2 SIM**, both best-in-table against CosyVoice 3, F5-TTS, FireRedTTS-2, IndexTTS 2,
  MegaTTS3, MiniMax-Speech, Qwen3-TTS, Seed-TTS, DiTAR, VibeVoice, and VoxCPM 2.
* dots.tts (Pretrain) achieves **2.92% average WER** (best in table); MeanFlow-distilled NFE=4
  variant stays within 0.01 WER of SOAR at every column (**2.94% average WER**).
* On test-zh specifically, dots.tts (SOAR) reaches **0.94% WER / 81.0 SIM**, versus the next-best
  SIM baseline Seed-TTS at 76.2 SIM.
* AudioVAE reconstruction on LibriSpeech test-other: **PESQ-WB 3.95**, **STOI 0.973**, **UTMOS
  3.75**, **SIM 0.969**, **WER 4.14%** at 48 kHz / 25 FPS — the highest input bandwidth among all
  continuous and discrete baselines compared in Table 1.
* MiniMax-Speech 24-language multilingual test set: dots.tts (SOAR) leads average SIM at **83.9**,
  1.6 points above the next-best baseline VoxCPM 2 (82.3), and wins per-language SIM on 19/24
  languages (ties on 2 more).
* CV3-Eval cross-lingual voice cloning: dots.tts (SOAR) leads SIM in both directions — **75.0 SIM
  (en to zh)** and **72.8 SIM (zh to en)**, 6-8 absolute points above CosyVoice 3 (66.9 / 66.4);
  cross-lingual WER trails CosyVoice 3 (5.66% vs 4.32% zh to en; 10.75% vs 8.01% en to zh).
* EmergentTTS-Eval (judged by Gemini-2.5-Pro-0506 vs. gpt-4o-mini-tts reference): dots.tts
  (Pretrain) leads the open-source field on overall win-rate at **49.2%**; SOAR has the lowest
  open-source WER at **10.45%** and the top Syntactic Complexity score across every system in the
  table, open- and closed-source, at **65.7%**.
* Inference efficiency (MeanFlow-distilled, NFE=4, single NVIDIA H800 GPU, vLLM + torch.compile):
  **85.4 ms** time-to-first-packet (TTFP) at **RTF 0.231** in plain streaming mode; **54.4 ms** TTFP
  at **RTF 0.245** in 1T1A interleaved dual-streaming mode.
* The Pretrain to SOAR post-training stage produces the largest single-column accuracy uplift
  observed in the paper: CV3-Eval hard-en WER drops from **5.99% to 4.49%** (Pretrain to SOAR), and
  EmergentTTS-Eval Syntactic Complexity rises **+7.3 points** (58.4% to 65.7%) from the same stage.

## Innovations

### Fully Continuous End-to-End AR TTS at Production Scale

dots.tts removes discrete acoustic tokens entirely, modeling speech as a continuous 128-dim VAE
latent throughout generation. The paper argues this raises the perceptual ceiling relative to
discrete-token cascades (which "snap" imperfect predictions back to a valid codebook entry before
reconstruction) and unifies speech, paralinguistics, singing, and general audio under one
distribution, at the cost of removing that quantization error-correction buffer.

### Semantic/Acoustic Decomposition with a Frozen Semantic Frontend

Inspired by ARDiT, the backbone splits into a semantic encoder (content-aligned summary), an LLM
(long-range text-to-content planning), and a full-context AR flow-matching head (local acoustic
rendering). Critically, the LLM only ever sees the 6.25 Hz semantic-encoder output, never the raw 25
Hz VAE latent — the paper states this separation is necessary to keep continuous-AR rollouts stable
over long generations, since it prevents acoustic detail from leaking into (and destabilizing) the
LLM's autoregressive state.

### Reward-Free Self-Corrective Alignment (SOAR Adaptation)

Adapts the SOAR self-correction idea to the AR flow-matching head specifically: the current DiT
generates its own off-trajectory inference-time states via a detached one-step Euler rollout, then
is supervised to steer those self-generated errors back toward the clean latent endpoint — without
any reward model, human preference data, or external acoustic teacher. This directly targets the
train/inference mismatch of multi-step ODE solving under teacher forcing.

### CFG-Aware MeanFlow Distillation for Low-NFE Streaming

Fuses classifier-free guidance directly into the MeanFlow distillation target, so the distilled
student predicts an already-CFG-guided mean velocity with a single conditional forward pass at
inference (no separate conditional/unconditional evaluation). Combined with the fully causal 1T1A
interleaved layout, this yields the paper's headline low-latency numbers (85 ms / 54 ms first-packet
latency at NFE=4) without a full quality collapse (<=0.01 WER, ~1 SIM point cost vs. the
un-distilled SOAR checkpoint on Seed-TTS-Eval).

## Datasets

* **In-house Chinese/English corpus**: ~1.2M hours after filtering (cross-ASR consistency,
  effective-bandwidth estimation, UTMOS, intra-clip x-vector variance), transcribed via
  Whisper-Large-v3 (non-Mandarin) and Paraformer (Mandarin); not publicly released.
* **Open-source corpora mixture** (~300K hours, non-CJK language coverage): Emilia, LibriTTS-R,
  HiFi-TTS, HiFi-TTS-2, WenetSpeech4TTS, AISHELL-3, Magicdata, MLS, MSR-86K, IndicVoices-R,
  EuroSpeech, WaxalNLP-TTS, FLEURS — all publicly available research TTS/ASR corpora.
* **Caption-paired data** (~7K hours): sampled from AutoACD with natural-language captions, plus an
  in-house subset with Gemini-generated speaker-trait/emotion/delivery/acoustic-environment
  descriptions.
* **Evaluation benchmarks**: Seed-TTS-Eval (test-en/test-zh/test-zh-hard, ~3-second zero-shot
  reference prompts), MiniMax-Speech 24-language multilingual test set (100 utterances/language, 2
  MCV reference speakers/language), CV3-Eval (monolingual + cross-lingual voice cloning subset,
  released with CosyVoice 3), EmergentTTS-Eval (six expressiveness scenarios, judged by
  Gemini-2.5-Pro-0506), and LibriSpeech test-other (AudioVAE reconstruction only).
* AudioVAE trained on 48 kHz audio from the in-house and open-source pools, augmented with a small
  general-audio share for non-speech coverage.

## Main Ideas

* The three-way semantic-encoder/LLM/AR-FM-head decomposition, with the LLM restricted to a compact
  6.25 Hz semantic summary rather than the raw continuous latent, is presented as the key stabilizer
  for continuous-AR rollouts — directly relevant to this project's evaluation of whether a
  Kokoro-style architecture could similarly decouple planning from acoustic rendering to reduce
  drift.
* CFG-aware MeanFlow distillation (NFE 2-4, single conditional forward pass, CFG fused into the
  distillation target) is a concrete, measured recipe for cutting flow-matching-head latency with
  documented cost (<=0.01 WER, ~1 SIM point) — a directly applicable technique if this project ever
  explores flow-matching-based low-latency heads, though it is orthogonal to Kokoro's current
  architecture.
* Reported first-packet latencies (85 ms plain, 54 ms interleaved) and RTF (0.231-0.245) were
  measured on a single H800 with vLLM + torch.compile — useful as an external reference point for
  what "low-latency" looks like on comparable hardware, but not directly transferable to this
  project's ElevenLabs-replacement TTFB target (<=300 ms) without accounting for architecture and
  GPU differences.
* This paper's own reported numbers (Tables 1-5, Section 3.4) do not specify a mixed-precision or
  quantization deployment scheme; the "bf16 acoustic stages / quantized LLM trunk" precision design
  attributed to this paper in prior research notes could not be verified against the paper's text
  and should be treated as unconfirmed pending a re-check against the project's GitHub repository or
  a different paper revision.
* The paper's Stage-1 "modality alignment" pretraining finding — that freezing the LLM while
  training only the semantic encoder and AR-FM head opens a stable channel before the full model is
  unfrozen (full pretraining-mixture rollouts were "severely unstable" otherwise) — is a
  transferable curriculum-design lesson for any staged fine-tuning of a frozen backbone.

## Summary

dots.tts is a 2B-parameter, fully continuous, end-to-end autoregressive TTS foundation model built
jointly by dots/Xiaohongshu Inc. and the X-LANCE Lab at Shanghai Jiao Tong University. The paper's
research question is whether continuous-latent AR generation — which avoids the discrete-token
bottleneck that caps what a TTS language model can express but suffers from unbuffered long-range
error accumulation — can be made production-stable and competitive with discrete-token systems such
as CosyVoice, Qwen3-TTS, and Seed-TTS, while also being efficient enough for real-time,
conversational deployment.

The approach decomposes generation into a frozen causal semantic encoder, a Qwen2.5-1.5B-Base LLM
that only ever sees a 6.25 Hz semantic summary of the audio (never the raw 25 Hz VAE latent), and an
18-layer DiT-based autoregressive flow-matching head trained with a block-causal attention mask that
reproduces exact per-step inference context during parallel training. A two-stage AudioVAE
(BigVGAN-v2-style reconstruction losses, then WavLM-alignment plus multitask ASR/emotion/speaker
supervision) keeps the 128-dim, 25 Hz latent both high-fidelity and learnable. Two post-training
stages follow pretraining: a reward-free self-corrective alignment stage (adapted from SOAR) that
exposes the AR-FM head to its own off-trajectory rollout errors without any reward model, and a
CFG-aware MeanFlow distillation stage that compresses the flow-matching ODE solve to as few as 2-4
function evaluations with CFG fused into the distillation target.

Trained on 1.5M hours of speech, dots.tts reports the best average WER (2.92-2.95%) and SIM
(78.8-79.2) on Seed-TTS-Eval among all compared systems, the highest average SIM (83.9) on the
24-language MiniMax multilingual test set, leading cross-lingual SIM on CV3-Eval (75.0 en to zh,
72.8 zh to en), and the top Syntactic Complexity score on EmergentTTS-Eval (65.7%) across both open-
and closed-source systems. The SOAR post-training stage produces the largest single-stage accuracy
gains observed (e.g., CV3-Eval hard-en WER 5.99% to 4.49%). On efficiency, the MeanFlow-distilled
model reaches 85.4 ms / 54.4 ms first-packet latency (plain / 1T1A-interleaved streaming) at RTF
0.231/0.245 on one NVIDIA H800 GPU.

For this project, the paper is most useful as evidence that decoupling long-range semantic planning
from local acoustic rendering (via a frozen, low-rate semantic bottleneck between the LLM and the
flow-matching head) is a viable, measured way to stabilize continuous generation and that CFG-fused
MeanFlow distillation is a concrete, quantified technique for cutting flow-matching NFE with small,
documented quality cost. It is less directly useful as a precision/quantization reference: despite
research notes suggesting this paper documents an explicit bf16-for-acoustic vs. quantized-LLM-trunk
precision split, the downloaded paper text reports only that its accuracy tables use float32 with
torch.compile disabled and does not describe a deployment quantization scheme — that specific claim
should be re-verified against the project's GitHub repository rather than cited from this arXiv
report.
