---
spec_version: "3"
paper_id: "10.21437_Interspeech.2021-41"
citation_key: "You2021"
summarized_by_task: "t0014_v11_decoder_fix_retrain"
date_summarized: "2026-09-16"
---
# GAN Vocoder: Multi-Resolution Discriminator Is All You Need

## Metadata

* **File**: `files/you_2021_gan-vocoder-multi-resolution-discriminator.pdf`
* **Published**: 2021
* **Authors**: Jaeseong You 🇰🇷, Dalhyun Kim 🇰🇷, Gyuhyeon Nam 🇰🇷, Geumbyeol Hwang 🇰🇷, Gyeongsu Chae
  🇰🇷
* **Venue**: Interspeech 2021
* **DOI**: `10.21437/Interspeech.2021-41`

## Abstract

Several of the latest GAN-based vocoders show remarkable achievements, outperforming autoregressive
and flow-based competitors in both qualitative and quantitative measures while synthesizing orders
of magnitude faster. In this work, we hypothesize that the common factor underlying their success is
the multi-resolution discriminating framework, not the minute details in architecture, loss
function, or training strategy. We experimentally test the hypothesis by evaluating six different
generators paired with one shared multi-resolution discriminating framework. For all evaluative
measures with respect to text-to-speech syntheses and for all perceptual metrics, their performances
are not distinguishable from one another, which supports our hypothesis.

## Overview

The paper asks a narrow but consequential question: when researchers credit a GAN vocoder's success
to its generator architecture, its loss composition, or its training tricks, are those claims
actually load-bearing? The authors (MoneyBrain Inc., Seoul) test this by holding the discriminator
fixed — always the HiFi-GAN V2 multi-resolution discriminating framework — and swapping in six
structurally different generators: the HiFi-GAN generator itself, MelGAN, Parallel WaveGAN (PWGAN),
Universal MelGAN (UMGAN), VocGAN, and a new generator of their own design. All six are trained from
scratch under identical data, optimizer, and step-budget conditions on LJSpeech, then evaluated with
both a large-scale Mechanical Turk MOS study and mel-cepstral distortion (MCD).

The central finding is a near-total absence of separation: on three of four evaluation metrics (MOS
on ground-truth mels, MOS on Tacotron 2 TTS mels, and MCD on TTS mels), the six generators'
confidence intervals overlap so heavily that no model can be called a clear winner or loser — this
despite generator parameter counts spanning nearly two orders of magnitude (928K for HiFi-GAN to 93M
for UMGAN). Only MCD computed on ground-truth reconstruction shows statistically resolvable
differences, with the heaviest model (UMGAN) doing best and the differences not tracking parameter
count. The authors interpret this as strong evidence that generator capacity and architectural
detail are, past a basic sufficiency threshold established as early as MelGAN, not the deciding
factor in GAN vocoder quality — the multi-resolution discriminator is.

The paper is explicitly a negative/ablation-style result rather than a new state-of-the-art claim:
the authors' own proposed generator is included mainly to widen the diversity of the comparison set,
not to be promoted as a superior architecture. The significance for the field is methodological — it
argues that future vocoder research effort is better spent on discriminator design (and possibly on
adapting the multi-resolution discriminating framework to related tasks such as voice conversion or
speech enhancement) than on further generator engineering.

## Architecture, Models and Methods

All six generators are trained under a strictly shared discriminating framework: the HiFi-GAN V2
multi-resolution discriminator (multi-period discriminator, MPD, plus multi-scale discriminator,
MSD), with the V2 (not V1) configuration chosen because it trains faster and its reported quality is
close to V1's. Data: LJSpeech, 13,100 English utterances from a single female speaker (~24 hours),
split into 12,950 training and 150 held-out validation files, 16-bit PCM at 22.05 kHz, used
unmodified. Mel spectrograms use 80 bands, FFT size 1024, window size 1024, hop size 256.
Optimization: AdamW with β1 = 0.8, β2 = 0.99, weight decay λ = 0.01; learning rate starts at 2e-4
and decays by a factor of 0.999 every epoch; batch size 16; all six models are trained to a fixed
700k-step stoppage (a midpoint of the 400k–2.5M range reported across the five prior GAN vocoders
being replicated). The authors' own proposed generator uses grouped 1D convolution (kernel size 17,
32 groups) for a wide receptive field at low parameter cost, a second conv1d (kernel 3,
exponentially increasing dilation) and 1×1 channel-mixing convolutions inside each residual block,
giving a 12-block residual net with WaveNet-style skip connections into a shared 1×1 postnet;
receptive field reaches 8,369 samples by the 12th block. Upsampling uses alternating
nearest-neighbor interpolation and kernel-3 1D convolution (8×, 8×, 2×, 2×) rather than transposed
convolution, citing reduced checkerboard artifacts. Evaluation: MOS collected via Amazon Mechanical
Turk on a 5-point Likert naturalness scale, 340 shuffled test utterances (240 synthesized: 6
vocoders × 20 GT-mel-based + 20 Tacotron-2-TTS-based syntheses, plus 100 real recordings), 30 unique
raters per sample, 189 total participants. MCD is computed over 13 MFCC coefficients (excluding the
first) with dynamic time warping used to align TTS-based syntheses to their ground-truth pairs; all
clips are loudness-normalized to -21 dB before scoring.

## Results

* Best MOS on ground-truth mel resynthesis: the authors' proposed generator at **4.028 ± .065**, vs.
  real (unsynthesized) audio at **4.132 ± .028**
* Best MOS on Tacotron-2 TTS synthesis: proposed generator again, **3.708 ± .075**
* Full MOS_GT spread across all six models is only **3.938–4.028** (PWGAN lowest, proposed highest)
  — confidence intervals overlap
* Full MOS_TTS spread is **3.580–3.708** (MelGAN lowest, proposed highest) — confidence intervals
  overlap
* MCD on TTS-based synthesis is statistically indistinguishable across models, ranging
  **12.824–13.023**
* MCD on ground-truth reconstruction is the one metric with resolvable differences: UMGAN best at
  **5.365 ± .059**, worst is the proposed generator at **6.680 ± .052**
* Generator parameter counts range **928,514** (HiFi-GAN) to **93,077,506** (UMGAN, roughly 9-10x
  every other model) yet do not predict MOS or MCD_TTS ranking
* Training speed per batch ranges **0.58 s** (MelGAN) to **1.09 s** (UMGAN); inference speed per
  sample ranges **0.0033 s** (MelGAN) to **0.0098 s** (PWGAN)
* MOS study totals: **189** unique AMT raters, **30** raters per audio sample, **340** total
  shuffled utterances judged on a 5-point Likert naturalness scale

## Innovations

### Isolating the Discriminator as the Causal Variable

Rather than proposing a new generator and claiming state-of-the-art MOS, the paper's core
contribution is an ablation methodology: hold the discriminator (HiFi-GAN's multi-resolution
framework) constant and vary only the generator across six structurally distinct designs
(mel-conditioned vs. noise-latent input, MelGAN-style vs. WaveNet-style vs. axial-residual blocks).
This isolates the discriminator, not the generator, as the variable that best explains GAN vocoder
quality — a claim prior single-model papers could not make because they always changed generator and
discriminator together.

### Axial Residual Generator with Grouped Convolution and Nearest-Neighbor Upsampling

The authors' own generator (adapted from their companion voice-conversion work) combines a
wide-receptive-field grouped 1×17 convolution, a dilated 1×3 convolution, and 1×1 channel mixing per
residual block, plus nearest-neighbor (not transposed-convolution) upsampling to avoid checkerboard
and frequency-domain distortion artifacts. It achieves the best MOS in the study at roughly
one-fourth the parameter count of MelGAN and 1/77th that of UMGAN.

### Cross-Representation Generality of the Multi-Resolution Discriminator

The paper shows the shared discriminator framework — which only ever sees final raw waveform —
performs equally well when paired against generators whose *original* designs used different
discrimination targets (UMGAN originally discriminates on spectral + waveform representations;
VocGAN originally discriminates on intermediate and final waveform outputs). Both still reach
MOS/MCD parity with the other four generators, evidence that the multi-resolution discriminating
framework generalizes across representation types, not just architectures.

## Datasets

* **LJSpeech**: 13,100 English utterances from a single female speaker, approximately 24 hours total
  duration, 16-bit PCM audio at 22.05 kHz, publicly available (Ito, keithito.com). Split into 12,950
  training files and 150 held-out validation files (validation set never used in training, reserved
  for MOS/MCD evaluation). Used without modification. No other datasets were used; the paper also
  relies on a public pretrained Tacotron 2 checkpoint (PyTorch implementation and NVIDIA-released
  weights) purely to generate TTS-based mel spectrograms for evaluation, not for training the
  vocoders.

## Main Ideas

* This project's own decoder (Kokoro/StyleTTS2, HiFi-GAN-style MPD/MSD discriminator) sits inside
  exactly the design space this paper studies — the paper's central finding suggests that if v11
  still underperforms after the `ignore_modules` decoder-init fix, the discriminator configuration
  (MPD/MSD setup, resolutions) is a more promising place to look than further decoder/generator
  architecture changes, since generator capacity spanning 928K to 93M parameters did not separate
  MOS or MCD_TTS outcomes here.
* The paper does **not** report GAN training-stability diagnostics (no discriminator warmup
  ablation, no feature-matching-loss pause rule, no discussion of discriminator-generator balance
  during training) — it confirms, rather than fills, the gap noted in `research_papers.md` that no
  reviewed paper yet documents reusable stability diagnostics for t0009's safeguard library. A
  separate source is still needed for that specific need.
* MCD on ground-truth reconstruction was the only one of four metrics in this paper to show
  statistically resolvable differences between vocoder configurations (MOS_GT, MOS_TTS, and MCD_TTS
  all overlapped) — this project could consider tracking GT-reconstruction MCD as a cheap, more
  discriminating proxy signal alongside GE2E speaker-sim and MOS-style checks during decoder
  debugging.
* Nearest-neighbor upsampling (over transposed convolution) is cited by the authors as reducing
  checkerboard and frequency-domain artifacts in their best-scoring generator; worth a low-cost
  sanity check on the current decoder's upsampling layers if v11 shows artifact-like quality
  regressions after the retrain.

## Summary

This paper investigates what actually drives the quality of modern GAN-based neural vocoders,
challenging the common practice of crediting generator architecture, loss function choices, or
training tricks for a given model's success. The authors, from MoneyBrain Inc. (Seoul, Korea),
hypothesize instead that the shared ingredient across recent successful GAN vocoders is the
multi-resolution discriminating framework popularized by HiFi-GAN, and that generator-side details
are comparatively unimportant once a basic capacity threshold is met.

To test this, the authors fix the discriminator to HiFi-GAN's V2 multi-resolution framework and
train six structurally different generators — HiFi-GAN, MelGAN, Parallel WaveGAN, Universal MelGAN,
VocGAN, and a new axial-residual generator of their own design — from scratch on LJSpeech under
identical optimizer, batch size, and 700k-step training budgets. Quality is judged with a large
Amazon Mechanical Turk MOS study (189 raters, 340 utterances, 5-point naturalness scale) covering
both ground-truth-mel resynthesis and Tacotron-2 TTS-based synthesis, plus mel-cepstral distortion
(MCD) as a quantitative counterpart.

The headline finding is that three of four evaluation metrics (MOS_GT, MOS_TTS, MCD_TTS) show no
statistically resolvable difference among the six generators, despite their parameter counts
spanning **928,514** to **93,077,506** — roughly two orders of magnitude — and despite widely
different architectural choices (mel-conditioned upsampling vs. Gaussian-noise-latent input,
MelGAN-style residual stacks vs. WaveNet-like gated units vs. the authors' axial-residual blocks).
Only MCD on ground-truth reconstruction separates the models, and not in proportion to parameter
count. The authors conclude that generator capacity and wiring have been "sufficiently powerful
since MelGAN," and that the multi-resolution discriminating framework is the real decisive factor,
one they expect to generalize to other GAN-based speech tasks.

For this project, the paper is relevant primarily as a caution and a prioritization signal rather
than as a source of reusable training-stability techniques: it does not report discriminator warmup
schedules, feature-matching-loss pause rules, or other GAN stability diagnostics, so it does not
close the gap `research_papers.md` identified for t0009's safeguard library. What it does offer is
evidence that, if v11 still underperforms after the decoder-init (`ignore_modules`) fix, the
project's MPD/MSD discriminator configuration is a more promising lever to inspect than further
decoder architecture tuning, and that ground-truth-reconstruction MCD may be a more discriminating
cheap proxy metric than MOS-style or TTS-based MCD checks during iteration.
