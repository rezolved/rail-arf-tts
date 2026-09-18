---
spec_version: "3"
paper_id: "10.48550_arXiv.2509.09631"
citation_key: "Nguyen2025"
summarized_by_task: "t0021_zero_shot_latency_reduction"
date_summarized: "2026-09-18"
---
# DiFlow-TTS: Compact and Low-Latency Zero-Shot Text-to-Speech with Discrete Flow Matching

## Metadata

* **File**: `files/nguyen_2025_diflow-tts-discrete-flow-matching.pdf`
* **Published**: 2025
* **Authors**: Ngoc-Son Nguyen 🇻🇳, Thanh V. T. Tran 🇻🇳, Hieu-Nghia Huynh-Nguyen 🇻🇳, Truong-Son Hy
  🇺🇸, Van Nguyen 🇻🇳
* **Venue**: preprint (arXiv)
* **DOI**: `10.48550/arXiv.2509.09631`

## Abstract

Zero-shot text-to-speech (TTS) has made significant progress in replicating unseen voices, yet
balancing generation quality and inference efficiency remains challenging. Autoregressive models
suffer from high latency, while diffusion-based approaches are constrained by training-time
configurations. Moreover, most flow-based methods operate in continuous space, which introduces
optimization challenges because continuous token spaces are inherently more complex than discrete
ones. To address these limitations, we propose DiFlow-TTS, a novel zero-shot TTS framework based on
discrete flow matching. The model consists of a deterministic Phoneme-Content Mapper for linguistic
modeling and a Factorized Discrete Flow Denoiser that simultaneously generates prosody and acoustic
token streams. Experimental results demonstrate the effectiveness of our approach across multiple
evaluation metrics.

## Overview

DiFlow-TTS is a zero-shot TTS system built entirely on Discrete Flow Matching (DFM) rather than the
continuous-space flow matching used by systems like F5-TTS. The authors argue that continuous flow
representations are unnecessarily high-dimensional and unbounded, making density estimation harder
and increasing the risk of out-of-distribution artifacts, whereas discrete codec tokens live in a
structured, finite space. The system tokenizes speech with a frozen, pre-trained FACodec (from
NaturalSpeech 3), which disentangles a reference utterance into a speaker embedding plus three
factorized discrete-token streams: one prosody codebook, two content codebooks, and three acoustic
codebooks (vocabulary size 1024 each). Two trainable modules operate on top of this fixed tokenizer:
a Phoneme-Content Mapper (PCM) that deterministically converts phonemes into content
tokens/embeddings via duration prediction and length regulation, and a Factorized Discrete Flow
Denoiser (FDFD) that jointly predicts the prosody and acoustic token streams through a discrete
flow-matching process conditioned on the reference speaker embedding and the PCM content embeddings.

The core methodological contribution is reformulating DFM, which prior work (the cited "Discrete
Flow Matching" paper by Gat et al., NeurIPS 2024) applied only to homogeneous discrete sequences,
into a factorized target distribution over structured attribute subspaces (prosody and acoustic). A
single DiT-based denoiser backbone with attribute-type embeddings produces a shared representation
that is then split into two independent prediction heads -- a prosody head and an acoustic head --
each predicting a categorical distribution over its own token vocabulary. This "multi-head"
factorization is presented as the first application of DFM that decomposes the probability velocity
field across multiple discrete subspaces within a single flow process.

Trained on only 470 hours of LibriTTS audio (far less than most baselines, some of which use up to
100K hours), DiFlow-TTS is evaluated on LibriSpeech test-clean cross-sentence generation against
autoregressive (VoiceCraft, VALL-E), continuous flow/diffusion (NaturalSpeech 2, F5-TTS, OZSpeech),
and masked-generative (MaskGCT) baselines. The headline finding is a strong naturalness/efficiency
trade-off: DiFlow-TTS achieves the best UTMOS and WER among all compared systems and is dramatically
smaller and faster (up to 11.7x smaller, up to 34x faster) than most baselines, but it does not lead
on the objective speaker-similarity metric (SIM-O), where it ranks third behind F5-TTS and MaskGCT.
The authors attribute this speaker-similarity gap to FACodec's explicit disentanglement of speaker
identity from content/prosody, which makes speaker conditioning harder to inject effectively
compared to codecs (e.g., EnCodec) that embed speaker information directly in their quantizers, and
they flag improving speaker conditioning as future work.

## Architecture, Models and Methods

* **Speech tokenizer**: pre-trained, frozen FACodec (from NaturalSpeech 3), operating at 16 kHz, 80
  tokens/s. Produces one prosody RVQ codebook, two content RVQ codebooks, three acoustic RVQ
  codebooks (vocabulary size v=1024 each), plus a continuous speaker embedding (D_spk = 256).
* **Phoneme-Content Mapper (PCM)**: hidden dimension 768. Phoneme Encoder: 2 Feed-Forward
  Transformer (FFT) layers, 4 attention heads, hidden size 256, output dim 768, conv filter size
  1024, kernel sizes [9, 1], dropout 0.2, max sequence length 5000. Duration Predictor/variance
  adapter (FastSpeech 2-style): encoder hidden size 256, filter size 1024, kernel size 3, dropout
  0.5, MSE loss on log-scale durations (L_dur). Content Predictor: FFT-based layers producing
  content embeddings (via projection head) and content-token logits (via a separate head), trained
  with cross-entropy loss L_c.
* **Factorized Discrete Flow Denoiser (FDFD)**: source distribution is a point mass on the
  all-[MASK] sequence; target distribution factorizes into independent prosody and acoustic
  components (Definition 3.1 in the paper). Scheduler kappa_t = t^2. Backbone: 12 Diffusion
  Transformer (DiT) blocks, 12 attention heads, hidden size 768, rotary position embeddings (RoPE),
  feed-forward width multiplier 4, AdaLN modulation conditioned on a global vector formed from
  speaker embedding + timestep embedding (each via an MLP), plus a long skip connection. Output is
  split into a prosody head and an acoustic head, each producing per-token categorical logits over
  vocabulary v.
* **Training objective**: total loss L = lambda_dur * L_dur + lambda_c * L_c + lambda_FDFD * L_FDFD,
  with weights 0.5, 1.0, 1.0 respectively; L_FDFD is a cross-entropy loss over masked tokens
  recovered by the posterior denoiser p_{1|t}.
* **Training setup**: 4x NVIDIA A100 GPUs, 315K training steps, batch size 16, AdamW optimizer
  (learning rate 1e-4, weight decay 0.01), 200K warm-up steps.
* **Training data**: 470-hour subset of LibriTTS (multi-speaker English), silence-trimmed, clips
  1.0-16.6 s with >3 words, tokenized via FACodec at 80 tokens/s, phoneme-level durations obtained
  via Montreal Forced Aligner (MFA).
* **Evaluation data/protocol**: LibriSpeech test-clean (2.2 hours, clips 4-10 s), cross-sentence
  generation using two same-speaker samples (one as prompt, one as target text), 3-second audio
  prompts for the main table.
* **Metrics**: UTMOS (naturalness/MOS-prediction), WER via HuBERT-large ASR, SIM-O (WavLM-TDNN
  speaker-embedding cosine similarity), F0/Energy tier accuracy and RMSE (following
  PromptTTS/TextrolSpeech discretization into high/normal/low tiers), RTF (seconds to generate 1s of
  audio on a single A100 80GB GPU), plus a subjective MOS study with 30 listeners rating
  naturalness, intelligibility, and similarity.
* **Model variants**: DiFlow-TTS (164M parameters, 12 heads/12 DiT layers) and DiFlow-TTS-Small
  (122M parameters, 8 heads/8 DiT layers), each evaluated at multiple numbers of function
  evaluations (NFE = 1, 2, 4, 8, 16, 32, 64, 128).

## Results

* DiFlow-TTS achieves the best **UTMOS 3.98** at NFE=128 (128-NFE, 3s prompt), vs. ground truth
  **4.10**, ahead of MaskGCT (**3.83**), F5-TTS-100K (**3.72**), and VALL-E (**3.68**).
* DiFlow-TTS ties for best WER at **0.05** (in [0, 1] scale) alongside OZSpeech, versus ground-truth
  WER **0.02** and worse baselines such as VoiceCraft (**0.18**) and VALL-E (**0.19**).
* DiFlow-TTS trails on speaker similarity: **SIM-O 0.45**, ranking third behind F5-TTS-100K
  (**0.66**) and MaskGCT (**0.67**); the paper explicitly states "DiFlow-TTS offers no clear
  advantage over baselines" on this metric.
* DiFlow-TTS leads prosody reconstruction with **F0 accuracy 0.88** and **F0 RMSE 7.97**, the best
  of all compared systems, and **Energy accuracy 0.73** (second-best, trailing MaskGCT's **0.75** by
  only 0.02) with the best **Energy RMSE 0.007**.
* In the subjective MOS study (3-second prompts, 95% CI), DiFlow-TTS scores highest on every
  dimension: **Naturalness 4.18 +/- 0.16**, **Intelligibility 4.41 +/- 0.13**, and **Similarity 4.42
  +/- 0.12** -- the latter even exceeding the ground-truth similarity rating of **4.29 +/- 0.14**.
* Model size/latency (Table 3): full DiFlow-TTS has **164M parameters** and reaches **RTF 0.07** at
  NFE=16 (**RTF 0.03** at NFE=4); DiFlow-TTS-Small has **122M parameters** and reaches **RTF 0.05**
  at NFE=16 (**RTF 0.03** at NFE=4) -- the paper reports this is **5.2x to 34.0x** faster and **1.2x
  to 11.7x** smaller than the compared baselines (e.g., VoiceCraft 830M params/RTF 1.70;
  NaturalSpeech 2 378M params/RTF 1.66).
* NFE ablation (Table 6) shows UTMOS rising from **2.90** at NFE=1 to **3.98** at NFE=128, with RTF
  rising from **0.022 s** to **0.394 s** over the same range; performance stabilizes around NFE=32
  (**UTMOS 3.92**) with only marginal gains beyond NFE=64 (**UTMOS 3.96**).
* Component ablation (Table 4, NFE=128) shows removing the speaker embedding from FDFD causes the
  largest degradation: SIM-O drops from **0.454** to **0.378** and F0 RMSE worsens from **7.97** to
  **20.87**; removing content embeddings drops UTMOS most sharply, from **3.978** to **3.077**.

## Innovations

### Discrete Flow Matching Applied to Zero-Shot TTS

DiFlow-TTS is presented as the first system to apply Discrete Flow Matching (DFM), previously
demonstrated on domains like language, graphs, and proteins, to zero-shot speech synthesis,
establishing what the authors call "an initial baseline for applying DFM to speech generation." This
avoids the continuous-space optimization difficulties (high dimensionality, unbounded support,
out-of-distribution artifacts) that affect prior flow-matching TTS systems such as F5-TTS and
Matcha-TTS.

### Factorized Discrete Flow Denoiser (FDFD)

Unlike prior DFM applications that model a single, homogeneous discrete sequence, FDFD reformulates
the target distribution as a factorized joint of independent prosody and acoustic components and
predicts their probability velocity fields with dedicated prosody and acoustic heads from a shared
DiT backbone. The paper claims this is "the first work to decompose probability velocity fields
across multiple discrete subspaces within a single discrete flow process," enabling a single unified
model to jointly capture prosody and acoustic-detail dynamics that are usually modeled by separate
diffusion stages (as in NaturalSpeech 3).

### Deterministic Phoneme-Content Mapper (PCM)

Rather than treating text-to-speech alignment as a stochastic or attention-based process, PCM
performs deterministic duration-based alignment directly on discrete content tokens (rather than
continuous mel-spectrogram frames as in FastSpeech-family models), which the authors argue makes
downstream discrete flow modeling more tractable.

### Compact, Data-Efficient, Low-Latency Model

DiFlow-TTS and DiFlow-TTS-Small (122M-164M parameters) reach competitive or superior naturalness
using only 470 hours of training audio -- 1.1x to 212.8x less than baselines -- while running at RTF
as low as 0.03-0.07 for practical NFE settings (4-16), positioning the architecture as a candidate
for "resource-constrained and latency-critical environments," which is directly relevant to this
project's TTFB and cost goals.

## Datasets

* **Training**: 470-hour subset of LibriTTS (English, multi-speaker), 16 kHz, tokenized via FACodec
  at 80 tokens/s; phoneme-level durations obtained via Montreal Forced Aligner (MFA); clips 1.0-16.6
  s retained with silence trimmed and >3 words per utterance. LibriTTS is publicly available.
* **Evaluation**: LibriSpeech test-clean (2.2 hours, clips 4-10 s), used for cross-sentence
  zero-shot generation (one same-speaker sample as prompt, another as target text); publicly
  available.
* **Baseline training data** (for comparison only, not used to train DiFlow-TTS): GigaSpeech (9K
  hours, VoiceCraft), LibriTTS subsets of 500-585 hours (VALL-E, NaturalSpeech 2, F5-TTS
  reproduction, OZSpeech), and Emilia (100K hours, F5-TTS official checkpoint and MaskGCT).
* Additional robustness analyses use LibriSpeech test-clean prompts corrupted with additive noise at
  varying SNR levels (no new dataset introduced).

## Main Ideas

* Discrete flow matching over factorized codec tokens (prosody/content/acoustic, via FACodec) is a
  viable alternative to F5-TTS-style continuous flow matching, and the DiFlow-TTS-Small variant
  (122M params, RTF 0.03-0.05 at NFE 4-16) is directly comparable in latency profile to F5-TTS (336M
  params, RTF 0.26 at NFE 32) -- worth benchmarking against F5-TTS and OZSpeech on this project's
  TTFB/RTF targets given its reported 5.2x-34x speedup over larger baselines.
* DiFlow-TTS's weakest metric is exactly this project's primary success criterion -- speaker
  similarity. It does not match systems such as MaskGCT and F5-TTS (SIM-O 0.66-0.67), a limitation
  the authors attribute to FACodec's explicit speaker/content disentanglement requiring more
  sophisticated speaker conditioning than plain AdaLN summation -- a concrete risk to flag before
  adopting this architecture for a project whose primary success metric is GE2E speaker-similarity
  cosine (target >=0.85 vs. ElevenLabs David).
* No official pre-trained checkpoint or released model weights are confirmed in this paper; the code
  repository (https://github.com/Fsoft-AIC/DiFlowTTS) would need to be checked separately for
  training/inference scripts and any released weights before this architecture could be evaluated
  experimentally in this project.
* The NFE/RTF/UTMOS trade-off curve (Table 6) is a useful reference for choosing operating points:
  NFE=16 (RTF 0.05-0.07) already captures most of the achievable UTMOS gain (3.86-3.89 vs. 3.98 at
  NFE=128), which is a pattern worth replicating when tuning Kokoro-82M sampling steps for
  TTFB-constrained inference.

## Summary

DiFlow-TTS addresses the same core trade-off this project cares about -- zero-shot speaker fidelity
versus inference latency -- by proposing a novel TTS architecture built entirely on Discrete Flow
Matching rather than the continuous-space flow matching used by comparable low-latency systems like
F5-TTS. The research question is whether operating directly in a structured, finite discrete-token
space (rather than a continuous, unbounded one) yields better optimization behavior and, in turn,
better quality/efficiency trade-offs for zero-shot voice cloning.

Methodologically, the system freezes a pre-trained FACodec to factorize reference speech into a
speaker embedding plus prosody, content, and acoustic token streams, then trains two modules: a
deterministic Phoneme-Content Mapper that aligns phonemes to content tokens/embeddings via duration
prediction, and a Factorized Discrete Flow Denoiser (12-layer DiT backbone, 164M/122M parameter
variants) that predicts prosody and acoustic tokens jointly through dedicated prediction heads
within a single discrete flow-matching process -- the first time, per the authors, that a discrete
flow's probability velocity field has been decomposed across multiple discrete subspaces in one
process.

The headline finding is a strong efficiency/naturalness profile: trained on only 470 hours of
LibriTTS (far less data than most baselines), DiFlow-TTS achieves the best UTMOS (**3.98**) and ties
for best WER (**0.05**) on LibriSpeech test-clean cross-sentence generation, best-in-class prosody
metrics (F0 accuracy **0.88**, F0 RMSE **7.97**), and the highest scores across all MOS dimensions
(naturalness, intelligibility, and even similarity, at **4.18-4.42**), while running at RTF as low
as **0.03-0.07** and being up to **11.7x smaller** and **34x faster** than baselines. Its clear
weakness is objective speaker similarity (SIM-O **0.45**, third place behind F5-TTS's **0.66** and
MaskGCT's **0.67**), which the authors attribute to FACodec's explicit disentanglement of speaker
identity requiring more advanced conditioning than the model's current AdaLN-based approach, and
which they flag as future work.

For this project, DiFlow-TTS is a relevant alternative low-latency zero-shot architecture worth
comparing against F5-TTS (already reviewed) on this task's latency-reduction goals, given its
reported RTF/parameter-count advantages at comparable NFE settings. However, its weaker objective
speaker-similarity score is a material risk relative to this project's primary GE2E cosine
similarity target (>=0.85), and no confirmed public checkpoint is available in the reviewed paper --
any adoption would require verifying whether the authors' GitHub repository
(https://github.com/Fsoft-AIC/DiFlowTTS) provides trained weights or only training code before this
architecture could be benchmarked directly against Kokoro-82M or F5-TTS in this project's pipeline.
