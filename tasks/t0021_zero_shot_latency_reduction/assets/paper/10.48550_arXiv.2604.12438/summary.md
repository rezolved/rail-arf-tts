---
spec_version: "3"
paper_id: "10.48550_arXiv.2604.12438"
citation_key: "Su2026"
summarized_by_task: "t0021_zero_shot_latency_reduction"
date_summarized: "2026-09-18"
---
# An Ultra-Low Latency, End-to-End Streaming Speech Synthesis Architecture via Block-Wise

# Generation and Depth-Wise Codec Decoding

## Metadata

* **File**: `files/su_2026_ultra-low-latency-streaming-tts.pdf`
* **Published**: 2026
* **Authors**: Tianhui Su 🇲🇾, Tien-Ping Tan 🇲🇾, Salima Mdhaffar 🇫🇷, Yannick Estève 🇫🇷, Aghilas Sini
  🇫🇷
* **Venue**: preprint (arXiv, eess.AS)
* **DOI**: `10.48550/arXiv.2604.12438`

## Abstract

Real-time speech synthesis requires a rigorous balance between inference latency and acoustic
fidelity to support interactive human-computer applications. Conventional continuous text-to-speech
pipelines necessitate computationally intensive neural vocoders to reconstruct phase information,
creating a significant streaming bottleneck. Furthermore, regression-based speech synthesis acoustic
modeling frequently induces spectral over-smoothing artifacts that degrade perceptual quality. To
address these limitations, this paper proposes a novel end-to-end non-autoregressive architecture
optimized for ultra-low latency block-wise generation, which directly models the highly compressed
discrete latent space of the Mimi neural audio codec. By integrating a modified FastSpeech 2
backbone with a progressive depth-wise sequential decoding strategy, the proposed architecture
dynamically conditions 32 layers of residual vector quantization codes. This mechanism effectively
resolves phonetic alignment degradation and manages the structural complexity of high-fidelity
discrete representations without introducing temporal autoregressive overhead. Experimental
evaluations conducted on English and Malay single-speaker datasets validate the language-independent
deployment capability of the proposed architecture. Compared to conventional continuous regression
models, the proposed architecture demonstrates quantitative improvements in fundamental voicing
accuracy and effectively mitigates high-frequency spectral degradation. Furthermore, it achieves
ultra-low latency inference, translating to a 10.6-fold absolute acceleration over conventional
cascaded continuous pipelines. Crucially, the architecture achieves an average time-to-first-byte
latency of 48.99 milliseconds, falling significantly below the human perception threshold for
real-time interactive streaming. The results firmly establish the proposed discrete architecture as
a highly optimized solution for deploying high-quality, real-time streaming speech interfaces in
resource-constrained environments.

## Overview

This paper targets the same structural bottleneck this project keeps running into: neural vocoders
and autoregressive token generation both add algorithmic latency that a purely architectural fix can
remove. The authors couple a modified FastSpeech 2 non-autoregressive frontend directly to Meta's
Mimi neural audio codec (12.5 Hz frame rate, 32-layer residual vector quantization, RVQ), completely
bypassing both the continuous Mel-spectrogram representation and the heavy vocoder that normally
reconstructs phase information from it. The central engineering contribution is a "depth-wise
sequential decoding" head: rather than predicting all 32 RVQ codebook layers in parallel from one
hidden state (which the paper shows collapses badly), each layer's prediction is conditioned on the
accumulated embeddings of all previously predicted layers for that same frame -- sequential across
the 32-deep codebook axis, but still fully parallel across the time axis. This lets the model
preserve inter-layer acoustic dependencies without paying any autoregressive penalty over time.

The paper trains two single-speaker models -- English (LJSpeech, ~24h) and Malay (Mesolitica
Malaysian-TTS-v2 subset, 13h) -- on a single RTX 4090, and reports both objective acoustic metrics
(MCD, BAP, F0 RMSE, V/UV error, WER) and subjective MOS against VITS and FastSpeech2+
HiFi-GAN/Parallel WaveGAN baselines. It also runs three ablations: codebook depth (32 vs. truncated
16), depth-wise sequential vs. naive parallel decoding, and a failed subword/BPE phoneme-merging
experiment that catastrophically collapsed acoustic quality. The headline result is an average
time-to-first-byte of 48.99 ms and a real-time factor of 0.0033 (English) / 0.0055 (Malay) -- a 6x
to 10.6x speedup over VITS and cascaded FastSpeech2 + vocoder baselines respectively, at some cost
in MOS naturalness (2.51 vs. 2.81 for English) and WER (8.89% vs. 1.64-5.19% for baselines).

This is a fixed-speaker (not zero-shot, not multi-speaker) architecture: there is no reference audio
encoding, no speaker embedding, and no cloning step in the pipeline described, so it is not a
candidate replacement for the project's zero-shot David-voice pipeline as-is. Its value for this
task is as an existence proof of a third architecture family -- non-autoregressive, non-diffusion,
discrete-codec-native -- that removes the vocoder-latency and phase-estimation bottleneck entirely,
at latencies an order of magnitude below anything measured for CosyVoice2 or Chatterbox in
t0018/t0021.

## Architecture, Models and Methods

* **Backbone**: modified FastSpeech 2 (phoneme encoder + variance adaptor predicting log-duration,
  F0, energy) operating over phoneme-level linguistic tokens, feed-forward transformer blocks, fully
  non-autoregressive.
* **Codec**: pre-trained, frozen Mimi neural audio codec (Défossez et al., 2024) -- semantic +
  acoustic dual RVQ, 12.5 Hz frame rate (80 ms hop), 1.1 kbps target bitrate, 32 residual quantizer
  layers, 2048-entry codebook vocabulary per layer, 24 kHz waveform output.
* **Depth-wise cascaded discrete decoder**: for base hidden state `h_t` at frame `t`, the
  conditioned state for the `i`-th quantizer is `h_{t,i} = h_t + sum_{j<i} E_j(y_{t,j})`, where
  `E_j` is a learned embedding lookup for codebook `j`; each layer's token distribution is a softmax
  over a linear projection of `h_{t,i}`. This recursion runs across the 32-deep codebook axis within
  a single frame, not across time, so it adds negligible latency.
* **Auxiliary Mel supervision branch**: a linear layer + 5-layer convolutional PostNet projects
  hidden states to 80-channel Mel-spectrograms (window 1024 samples, hop 256 samples) purely as a
  training-time regularizer against discrete latent collapse; discarded at inference.
* **Alignment**: Montreal Forced Aligner for phoneme durations; a "dummy token" mechanism injects
  placeholder frames for phonemes shorter than the 80 ms Mimi hop to preserve strict one-to-one
  duration-regulator sequence integrity; F0 via YIN (50-500 Hz, log domain), energy via frame-wise
  RMS, both linearly interpolated to the 12.5 Hz codec frame rate; Mel features downsampled to
  codec-frame resolution via average pooling with factor K=8 (Eq. 1).
* **Training objective**: hybrid multi-task loss
  `L_total = L_token + lambda_dur*L_dur + L_pitch + L_energy + lambda_mel*(L_mel + L_postnet)`, with
  `lambda_dur=2.0`, `lambda_mel=10.0`. Token cross-entropy loss is aggregated over all 32 codebook
  layers with staged weighting: codebooks 1-4 weight 1.0, codebooks 5-16 weight 0.5, codebooks 17-32
  weight 0.1.
* **Training setup**: PyTorch, single NVIDIA RTX 4090 (24 GB), Adam optimizer (beta1=0.9,
  beta2=0.98, epsilon 1e-9 as printed), batch size 16, linear warm-up over 4000 steps then
  exponential decay. English model converges at 200,000 steps; Malay model at 90,000 steps.
* **Datasets**: LJSpeech (~24 h, English, single speaker, audiobook style) and a 13-hour subset of
  Mesolitica's Malaysian-TTS-v2 (single native female speaker); all audio resampled to 24 kHz.
* **Evaluation**: objective -- MCD, BAP, F0 RMSE, Voiced/Unvoiced (V/UV) error, WER via Whisper ASR;
  subjective -- MOS from 11 listeners with 95% confidence intervals, rated against ground truth on a
  5-point scale; efficiency -- RTF and TTFB measured on an RTX 4090.

## Results

* Time-to-first-byte: **48.99 ms** average, **22.03 ms** at the 90th percentile, well under the 200
  ms human-perceptible-latency threshold the paper cites.
* Real-time factor: **0.0033** (English, ~303x faster than real time) and **0.0055** (Malay, ~179x
  faster than real time).
* Comparative inference speed: **10.6-fold** speedup over cascaded FastSpeech2 + HiFi-GAN (RTF
  ~0.025) and FastSpeech2 + Parallel WaveGAN (RTF ~0.035); **~6-fold** speedup over VITS (RTF
  ~0.020); proposed method RTF ~0.0033.
* Acoustic fidelity (English, LJSpeech test set): MCD **10.20 dB** vs. VITS **7.31 dB** and
  FastSpeech2 **8.24 dB** (worse than baselines), but V/UV error **2.67%** -- better than both
  FastSpeech2 (**3.62%**) and VITS (**2.82%**).
* Speech intelligibility (English): WER **8.89%** vs. VITS **1.64%** and FastSpeech2 **5.19%**
  (topline Mimi-reconstruction WER is 1.34%) -- the discrete-codec approach trades intelligibility
  for latency.
* Subjective MOS: English **2.51 +/- 0.11** vs. baseline (FastSpeech2 + Parallel WaveGAN) **2.81 +/-
  0.13** and ground truth **4.51 +/- 0.08**; Malay **4.31 +/- 0.11**, close to Malay ground truth
  **4.65 +/- 0.08** -- much stronger relative performance on the phonetically transparent Malay
  language than on English.
* Ablation -- codebook depth: full 32-codebook decoding reaches WER **8.89%** / MCD **10.20 dB** vs.
  a truncated 16-codebook variant at WER **9.07%** / MCD **10.38 dB**.
* Ablation -- decoding strategy: depth-wise sequential decoding (WER **8.89%**, MCD **10.20 dB**)
  substantially beats naive fully-parallel per-frame decoding (WER **14.37%**, MCD **12.49 dB**,
  V/UV error 3.25% vs. 2.67%), confirming the depth-wise conditioning mechanism is load-bearing.
* Ablation -- subword/BPE phoneme aggregation: catastrophic failure -- synthesized audio exhibited
  severe high-frequency mechanical artifacts and total loss of intelligibility, attributed to
  exponential cascading error through the 32-layer depth-wise conditioning chain when the initial
  semantic codebook prediction is corrupted by an ambiguous merged-phoneme hidden state.

## Innovations

### Depth-Wise Sequential Codec Decoding

The core novelty: instead of a single regression/classification head predicting a Mel-spectrogram
frame (as in stock FastSpeech 2) or all RVQ codebook layers independently and in parallel, the
decoder predicts the 32 RVQ layers for a frame one at a time, each conditioned on the sum of learned
embeddings of all previously predicted layers in that same frame. This recovers the hierarchical
dependency structure of residual vector quantization (coarse semantic codes first, fine
acoustic-texture codes later) that naive parallel prediction destroys, while adding the recursion
only along the 32-deep codebook axis rather than along the time axis -- so it costs none of the
sequence-length-scaling latency an autoregressive model would incur.

### Vocoder-Free End-to-End Discrete Pipeline

By directly targeting Mimi's discrete latent tokens and decoding them with Mimi's frozen,
pre-trained decoder, the architecture removes the neural vocoder stage entirely -- no HiFi-GAN, no
Parallel WaveGAN, no flow-matching/ODE-solving stage. This is presented as the primary source of the
reported 6-10.6x speedup over cascaded and VITS-style pipelines, since dense transposed convolutions
and iterative sampling are the dominant compute cost in those baselines.

### Sub-Frame Duration "Dummy Token" Mechanism

At Mimi's aggressively compressed 80 ms hop size, some phonemes or textual boundaries are shorter
than one codec frame and would otherwise receive zero frames from the non-autoregressive length
regulator, breaking the required one-to-one alignment. The paper's fix -- injecting a synthetic
placeholder token with a minimum duration of one frame -- is a small but necessary engineering
detail for making non-autoregressive length regulation compatible with very low-frame-rate discrete
codecs.

### Auxiliary Continuous Regularization Without Inference-Time Cost

The auxiliary Mel-spectrogram supervision branch (with a 5-layer convolutional PostNet) is used only
during training to prevent discrete latent collapse and stabilize spectral envelope learning; it is
fully discarded at inference, so it improves training dynamics without adding any inference latency
-- a pattern applicable to this project's own training/inference latency trade-off decisions.

## Datasets

* **LJSpeech**: ~24 hours, English, single female speaker, audiobook recordings (Ito & Johnson,
  2017). Publicly available.
* **Malaysian-TTS-v2 (Mesolitica)**: 13-hour subset used, Malay, single native female speaker
  (Husein & Mesolitica, 2023). Publicly available on Hugging Face (`mesolitica/Malaysian-TTS-v2`).
* No multi-speaker or zero-shot cloning data is used; both models are trained single-speaker,
  single-language deployments, and neither dataset overlaps with this project's ElevenLabs David or
  reference-clip data.

## Main Ideas

* This is a third architecture family (non-autoregressive, non-diffusion, discrete-codec-native)
  distinct from the autoregressive-LLM-plus-flow-matching designs behind CosyVoice2 and Chatterbox
  that t0018/t0021 have been measuring -- useful evidence for Key Question 5 that sub-300 ms,
  sub-100 ms TTFB is architecturally achievable when the pipeline has no autoregressive token loop
  and no heavy vocoder, not just an engineering optimization of an AR pipeline.
* The reported 48.99 ms average TTFB (RTX 4090) is roughly two orders of magnitude below the 1.3-2.9
  s TTFB this project measured for CosyVoice2 and Chatterbox in t0018, which is consistent with this
  task's hypothesis that the AR-LLM-must-produce-N-tokens-before-any-audio requirement is a large
  structural contributor to the observed latency gap.
* The design is single-speaker with no reference-audio encoding step at all -- it cannot be adopted
  directly for this project's zero-shot David-voice cloning use case, but the depth-wise sequential
  RVQ decoding trick could in principle be grafted onto a codec-based zero-shot model (e.g.,
  replacing a flat/parallel RVQ head) to recover acoustic fidelity without autoregression.
* The paper's own ablation shows naive parallel RVQ-layer prediction is a real failure mode (WER
  14.37% vs. 8.89%) -- a caution for this project if any future architecture evaluation considers
  discrete-codec models with multi-layer RVQ targets: layer-parallel heads need explicit inter-layer
  conditioning or they degrade badly.
* The MOS gap between English (2.51, below the 2.81 baseline) and Malay (4.31, near the 4.65 ground
  truth) illustrates that discrete-codec approaches can trade quality for latency unevenly across
  languages/speaking styles -- a reminder to always report speaker_sim/WER/MOS-equivalent alongside
  any TTFB win, exactly as this task's protocol already requires.

## Summary

This paper addresses the same core problem this project has been fighting since t0017-t0020: how to
get a text-to-speech pipeline's first-audio latency down to tens of milliseconds without destroying
speech quality. The authors' scope is narrower than this project's -- single-speaker,
single-language-at-a-time (English or Malay), no zero-shot cloning -- but the research question is
the same structural one: can architectural choices (not engineering tricks on top of an existing
architecture) remove the dominant latency bottlenecks in TTS synthesis.

Methodologically, the paper fuses a non-autoregressive FastSpeech 2 frontend with Meta's frozen,
pre-trained Mimi neural audio codec, entirely eliminating the neural vocoder stage that normally
reconstructs phase information from a continuous Mel-spectrogram. Its key design decision is the
depth-wise sequential decoding head, which predicts Mimi's 32 residual-vector-quantization codebook
layers one at a time within each frame -- conditioned on previously predicted layers in that same
frame -- instead of independently in parallel. This recovers inter-layer acoustic dependencies that
a naive parallel head destroys, while adding recursion only along the codebook depth axis, not the
time axis, so it does not reintroduce autoregressive latency.

The headline finding is an average time-to-first-byte of 48.99 ms and a real-time factor of
0.0033-0.0055, a 6-10.6x speedup over VITS and cascaded FastSpeech2+vocoder baselines, achieved at
some cost to WER (8.89% vs. 1.64-5.19% for baselines) and English MOS (2.51 vs. 2.81 baseline, 4.51
ground truth), though Malay MOS (4.31) nearly matches ground truth (4.65). Ablations confirm both
that the full 32-codebook depth matters (vs. a truncated 16-codebook variant) and that the
depth-wise sequential conditioning is essential (naive parallel decoding nearly doubles WER).

For this project, the paper matters less as a directly reusable system -- it has no reference-audio
encoding or speaker-cloning mechanism at all -- and more as a data point for Key Question 5: it
demonstrates that removing the autoregressive-LLM-token-loop and the heavy vocoder stage can push
TTFB into the tens-of-milliseconds range on commodity GPU hardware, two orders of magnitude below
what t0018 measured for CosyVoice2 and Chatterbox. This supports treating a meaningful portion of
the CosyVoice2/Chatterbox latency gap as architectural rather than purely an engineering
optimization problem, while the depth-wise RVQ-conditioning technique itself is a candidate building
block if the project ever explores grafting a discrete-codec decoder onto a zero-shot cloning
frontend.
