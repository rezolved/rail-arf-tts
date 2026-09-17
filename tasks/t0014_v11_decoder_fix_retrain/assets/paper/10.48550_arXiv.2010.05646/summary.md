---
spec_version: "3"
paper_id: "10.48550_arXiv.2010.05646"
citation_key: "Kong2020"
summarized_by_task: "t0014_v11_decoder_fix_retrain"
date_summarized: "2026-09-16"
---
## Metadata

* File: `files/kong_2020_hifigan.pdf`
* Published: 2020
* Authors: Jungil Kong, Jaehyeon Kim, Jaekyoung Bae 🇰🇷
* Venue: NeurIPS 2020
* DOI: `10.48550/arXiv.2010.05646`

## Abstract

Several recent work on speech synthesis have employed generative adversarial networks (GANs) to
produce raw waveforms. Although such methods improve the sampling efficiency and memory usage, their
sample quality has not yet reached that of autoregressive and flow-based generative models. In this
work, we propose HiFi-GAN, which achieves both efficient and high-fidelity speech synthesis. As
speech audio consists of sinusoidal signals with various periods, we demonstrate that modeling
periodic patterns of an audio is crucial for enhancing sample quality. A subjective human evaluation
(mean opinion score, MOS) of a single speaker dataset indicates that our proposed method
demonstrates similarity to human quality while generating 22.05 kHz high-fidelity audio 167.9 times
faster than real-time on a single V100 GPU. We further show the generality of HiFi-GAN to the
mel-spectrogram inversion of unseen speakers and end-to-end speech synthesis. Finally, a small
footprint version of HiFi-GAN generates samples 13.4 times faster than real-time on CPU with
comparable quality to an autoregressive counterpart.

## Overview

This is the origin paper for the decoder architecture `config_david_v10.yml` selects via
`model_params.decoder.type: hifigan`, and its "V1" configuration is, field-for-field, the
architecture t0010's config builds (matching `resblock_kernel_sizes: [3, 7, 11]`,
`upsample_initial_channel: 512`, and the same resblock-dilation convention, with only
`upsample_rates`/`upsample_kernel_sizes` adapted from the paper's 256-sample hop to this project's
300-sample hop). The paper's contribution is a GAN-based neural vocoder (mel-spectrogram-to-waveform
model) built around two discriminators designed to capture the periodic structure of speech: a
Multi-Period Discriminator (MPD), which reshapes the 1D waveform into multiple 2D representations at
different periods to expose periodic patterns that a purely 1D convolutional discriminator misses,
and a Multi-Scale Discriminator (MSD), operating on the raw waveform at multiple temporal
resolutions (p. 2-3).

The generator is a fully-convolutional upsampling network: a stack of transposed-convolution
upsampling blocks (structured to reach the target audio sample rate from an 80-band mel-spectrogram
frame rate) each followed by a Multi-Receptive-Field Fusion (MRF) module that sums the outputs of
multiple parallel residual blocks with varying kernel sizes and dilation rates, intended to model
diverse receptive fields simultaneously rather than relying on a single fixed receptive field (p.
3). The paper offers three named configurations, V1 (highest quality, 13.92M parameters), V2 (a
smaller version of V1 with the same receptive field but a narrower hidden dimension, 0.92M
parameters), and V3 (further reduced layer count with carefully selected kernel sizes/dilations,
1.46M parameters) (p. 5), trading model size and inference speed against sample quality. This is the
paper directly cited (`[30]`) by the StyleTTS 2 paper as the source of its `HifiGAN`-based decoder
option.

The paper's central quantitative contribution to this project's Key Question 2 is its training
budget for a HiFi-GAN vocoder trained entirely from random initialization on a comparably-sized
single-speaker corpus: all models in the main LJSpeech/VCTK comparison (Table 1) "were trained until
2.5M steps" (p. 5), and even the smaller ablation-study runs (Table 2) were "updated up to 500k
steps for each configuration" (p. 6-7) — both figures many orders of magnitude larger than a
step-count implied by 20 epochs over a ~1,500-clip corpus.

## Architecture, Models and Methods

* **Generator**: transposed-convolution upsampling network with Multi-Receptive-Field Fusion (MRF)
  residual blocks after each upsampling stage; V1 configuration uses hidden dimension `hu=512`,
  upsample kernel sizes `ku=[16,16,4,4]`, resblock kernel sizes `kr=[3,7,11]`, and resblock dilation
  sizes `Dr=[[1,1],[3,1],[5,1]] x 3` (p. 5, Appendix A). `config_david_v10.yml`'s decoder block uses
  the same `resblock_kernel_sizes: [3,7,11]` and `upsample_initial_channel: 512` as V1, with
  `upsample_rates: [10,5,3,2]` (product 300, matching this project's 300-sample hop length) in place
  of the paper's `[8,8,2,2]` (product 256, matching the paper's 256-sample hop) — confirming
  `config_david_v10.yml`'s decoder is an adapted HiFi-GAN V1, not a novel architecture.
* **Discriminators**: Multi-Period Discriminator (MPD) with periods typically `{2,3,5,7,11}`,
  reshaping the 1D waveform into 2D `(period, T/period)` slices before 2D convolutions, and
  Multi-Scale Discriminator (MSD) operating on the raw waveform plus 2x/4x average-pooled
  downsamples (p. 3-4).
  * An ablation confirms MPD is the dominant quality driver: removing MPD drops MOS from **4.10
    (baseline V3) to 2.28** (a 1.82-point collapse), far larger than removing MSD (**3.74**, a
    0.36-point drop) [Table 2, p. 7].
* **Loss**: adversarial (LSGAN-style) loss on both generator and discriminator, feature-matching
  loss between real/fake discriminator activations, and an L1 mel-spectrogram reconstruction loss
  (p. 4, Eq. 7-8). Removing the mel-spectrogram loss drops MOS from 4.10 to **3.25**
  [Table 2, p. 7], the second-largest ablation effect after MPD removal.
* **Optimizer and schedule**: AdamW, `beta1=0.8, beta2=0.99`, weight decay `0.01`, initial learning
  rate `2e-4`, decayed by a factor of `0.999` every epoch (p. 6).
* **Input conditioning**: 80-band mel-spectrograms, FFT size 1024, window 1024, hop size 256 (p. 6).
* **Training budget**: main-comparison models (Table 1, LJSpeech and VCTK) trained to **2.5M
  steps**; ablation-study models (Table 2, V3 baseline and variants) trained to **500k steps** each
  (p. 5-7). No explicit epoch count or dataset-size-normalized step count is given, but 2.5M steps
  on a ~13,100-clip corpus implies a training budget several orders of magnitude beyond a 20-epoch
  schedule on a corpus of similar or smaller size.

## Results

* HiFi-GAN V1 (13.92M params) achieves **MOS 4.36 (+/- 0.07)**, a gap of only **0.09** from ground
  truth (**4.45 +/- 0.06**), and outperforms WaveNet (MoL) **4.02**, WaveGlow **3.81**, and MelGAN
  **3.79** [Table 1, p. 6].
* V1 synthesizes at **3,701x real-time** on a single V100 GPU (31.74 kHz on CPU); V3 (0.92M params,
  MOS **4.05**) synthesizes at **1,186.80x real-time** on GPU and **13.44x real-time on CPU**
  [Table 1, p. 6].
* Ablation study (V3 baseline, 500k steps each): removing MPD drops MOS **4.10 -> 2.28**; removing
  MSD drops MOS **4.10 -> 3.74**; removing MRF drops MOS **4.10 -> 3.92**; removing the
  mel-spectrogram loss drops MOS **4.10 -> 3.25** [Table 2, p. 7].
* Adding MPD to MelGAN (an unrelated baseline generator) improves its MOS by **+0.47** (**2.88 ->
  3.35**), showing the MPD contribution generalizes beyond HiFi-GAN's own generator [Table 2, p. 7].
* Generalization to 9 unseen VCTK speakers: V1/V2/V3 score **3.77 / 3.69 / 3.61** MOS respectively,
  all exceeding the autoregressive/flow-based baselines [Table 3, p. 7].
* Main-comparison training budget: **2.5M steps** (Table 1 models, p. 5); ablation training budget:
  **500k steps** (Table 2 models, p. 6-7).

## Innovations

### Multi-Period Discriminator (MPD)

Reshapes the 1D raw waveform into 2D slices at several distinct periods before applying 2D
convolutions, exposing periodic structure that a purely 1D discriminator cannot see as directly. The
paper's own toy sinc-function experiment (Appendix B) and the MOS ablation (removing MPD costs 1.82
MOS points) both support MPD as the single most important architectural component for sample
quality.

### Multi-Receptive-Field Fusion (MRF) generator

Instead of a single residual stack per upsampling stage, MRF sums the outputs of several parallel
residual blocks with different kernel sizes and dilation rates, letting the generator model patterns
at multiple receptive-field scales simultaneously without deepening the network.

### Small-footprint variants (V2, V3) for real-time and on-device inference

By tuning hidden dimension and layer count while preserving receptive field, HiFi-GAN offers
configurations from 13.92M down to 0.92M parameters with a MOS degradation of at most ~0.3-0.4
points relative to V1, enabling CPU real-time or faster-than-real-time synthesis (V3: 13.44x
real-time on CPU).

## Datasets

* **LJSpeech**: 13,100 short clips, single speaker, ~24 hours total, 16-bit PCM at 22.05 kHz, used
  unmodified for the main quality/speed comparison (p. 5).
* **VCTK**: ~44,200 clips, 109 native-English speakers with various accents, ~44 hours total,
  originally 44.1 kHz downsampled to 22.05 kHz; nine speakers held out entirely from training to
  evaluate mel-spectrogram inversion generalization to unseen speakers (p. 5).

## Main Ideas

* `config_david_v10.yml`'s `decoder.type: hifigan` block is architecturally V1 from this paper
  (matching `resblock_kernel_sizes`, `upsample_initial_channel`) with only the upsample-rate
  schedule adapted to this project's hop length — meaning this paper's own training budget is the
  single most directly applicable data point for judging whether v10's 17-epoch effective run (or
  v11's planned budget) was realistic for training this exact generator/discriminator pair from
  scratch.
* The paper trains its main comparison models to **2.5M optimizer steps** and even its smaller
  ablation configurations to **500k steps** (p. 5-7) — several orders of magnitude beyond what 20
  epochs on a ~1,500-clip corpus (a few thousand steps at most, depending on batch size) would
  provide, strongly suggesting that v10's failure mode (a HiFi-GAN decoder that never produced valid
  audio) is consistent with severe under-training even independent of the `ignore_modules`
  checkpoint-loading bug t0013 diagnosed.
* MPD is disproportionately responsible for sample quality (a 1.82-MOS-point ablation effect, by far
  the largest of the four ablated components) — if the v11 retrain's discriminators are miswired or
  under-weighted in the loss, this is the first place to check before assuming an epoch-count
  problem.

## Summary

HiFi-GAN is a GAN-based neural vocoder that converts mel-spectrograms to raw waveforms, aiming to
close the sample-quality gap between GAN-based vocoders and slower autoregressive/flow-based models
by explicitly modeling the periodic structure of speech audio (p. 1-2). Its research question is
whether discriminators designed around the multi-periodic nature of speech (rather than generic
waveform or spectrogram discriminators) can deliver autoregressive-level quality at GAN-level
inference speed.

Methodologically, the generator is a transposed-convolution upsampling stack with Multi-Receptive-
Field Fusion residual blocks, trained adversarially against a novel Multi-Period Discriminator and a
Multi-Scale Discriminator, using a combined adversarial, feature-matching, and mel-spectrogram L1
reconstruction loss (p. 2-4). Three generator configurations (V1/V2/V3) trade model size (13.92M
down to 0.92M parameters) against quality and speed, trained with AdamW (initial lr 2e-4, 0.999
per-epoch decay) to 2.5M steps for the main comparisons and 500k steps for ablations (p. 5-7).

The paper finds HiFi-GAN V1 reaches MOS 4.36 versus ground truth 4.45 (a 0.09-point gap), beating
WaveGlow, MoL WaveNet, and MelGAN, while running up to 1,186x real-time on GPU; the ablation study
isolates MPD as the single largest quality contributor, with its removal costing 1.82 MOS points
(Table 1-2, p. 6-7).

For this project, the paper matters directly because `config_david_v10.yml`'s HiFi-GAN decoder is
this paper's V1 architecture adapted to a different hop length, and its 2.5M-step (main) / 500k-step
(ablation) training budgets are the primary published reference point for how much training a
HiFi-GAN decoder needs from random initialization — evidence that v10's actual training budget
(effectively ~17 epochs after the checkpoint-loading bug left the decoder near-random) was almost
certainly insufficient regardless of the bug, and that the v11 retrain should plan for a
substantially larger epoch/step budget than 20 epochs if it wants the decoder to converge from
scratch rather than relying on a genuine pretrained starting point.
