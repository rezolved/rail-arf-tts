---
spec_version: "3"
paper_id: "10.48550_arXiv.2203.02395"
citation_key: "Kaneko2022"
summarized_by_task: "t0014_v11_decoder_fix_retrain"
date_summarized: "2026-09-16"
---
## Metadata

* File: `files/kaneko_2022_istftnet.pdf`
* Published: 2022
* Authors: Takuhiro Kaneko, Kou Tanaka, Hirokazu Kameoka, Shogo Seki 🇯🇵
* Venue: ICASSP 2022
* DOI: `10.48550/arXiv.2203.02395`

## Abstract

In recent text-to-speech synthesis and voice conversion systems, a mel-spectrogram is commonly
applied as an intermediate representation, and the necessity for a mel-spectrogram vocoder is
increasing. A mel-spectrogram vocoder must solve three inverse problems: recovery of the
original-scale magnitude spectrogram, phase reconstruction, and frequency-to-time conversion. A
typical convolutional mel-spectrogram vocoder solves these problems jointly and implicitly using a
convolutional neural network, including temporal upsampling layers, when directly calculating a raw
waveform. Such an approach allows skipping redundant processes during waveform synthesis (e.g., the
direct reconstruction of high-dimensional original-scale spectrograms). By contrast, the approach
solves all problems in a black box and cannot effectively employ the time-frequency structures
existing in a mel-spectrogram. We thus propose iSTFTNet, which replaces some output-side layers of
the mel-spectrogram vocoder with the inverse short-time Fourier transform (iSTFT) after sufficiently
reducing the frequency dimension using upsampling layers, reducing the computational cost from
black-box modeling and avoiding redundant estimations of high-dimensional spectrograms. During our
experiments, we applied our ideas to three HiFi-GAN variants and made the models faster and more
lightweight with a reasonable speech quality. Audio samples are available at
https://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/istftnet/.

## Overview

This paper is directly relevant to t0013's root-cause diagnosis: `first_stage_v3.pth` (the Stage 1
checkpoint `train_second_v10.py:load_checkpoint()` partially loaded into `config_david_v10.yml`'s
HifiGAN-shaped decoder) is istftnet-shaped, i.e. built from this paper's architecture rather than
plain HiFi-GAN. Reading this paper directly explains, at the architectural level, why a partial
shape-match between the two is possible at all: iSTFTNet is explicitly designed as a modification of
existing HiFi-GAN variants, not a fully independent architecture, so many early/middle layers (input
convolution, several upsampling ResBlock stages) keep the same shapes as their HiFi-GAN
counterparts, while only the final output-side layers diverge (magnitude/phase heads plus an iSTFT
operation, versus HiFi-GAN's `conv_post` waveform head) (p. 1-2, Fig. 1). This is exactly the kind
of partial-shape-match failure mode t0013 identified: the loader's `len(matched) == 0` guard only
fires on total mismatch, but a HiFi-GAN-shaped `decoder` and an iSTFTNet-shaped `first_stage_v3.pth`
overlap enough (shared upsampling/ResBlock layers) to pass silently while the true
vocoder-generating output layers are architecturally incompatible and either fail to load or load
onto the wrong tensors.

Methodologically, the paper takes each of three HiFi-GAN generator variants (V1, V2, V3, from Kong
et al. 2020) and replaces some suffix of their upsampling+ResBlock stages with a lightweight head
that predicts log-magnitude and phase spectra, then reconstructs the waveform via inverse STFT
instead of continuing raw-waveform convolution all the way to the output sample rate (p. 1-2). The
key empirical question the paper answers is how many of a HiFi-GAN variant's layers can be replaced
with this iSTFT head before speech quality degrades; the answer differs by variant, since the
lighter variants (V2, V3) have less redundant capacity to spare (p. 3-4).

The paper's training recipe is the second directly relevant data point for Key Question 2, alongside
the original HiFi-GAN paper: it trains each vocoder-only configuration from scratch for **2.5M
iterations** on LJSpeech (p. 3), the identical step count reported in the original HiFi-GAN paper,
and separately fine-tunes combined TTS+vocoder (Conformer-FastSpeech2 + iSTFTNet) end-to-end for
**300k iterations** starting from independently-trained components (p. 4) — a fine-tuning-scale
budget nearly two orders of magnitude smaller than the from-scratch budget, illustrating the same
"pretrained-start vs. random-init" asymmetry this project's Key Question 1 is concerned with.

## Architecture, Models and Methods

* **Core idea**: replace the final output-side convolutional layers of a HiFi-GAN-style generator
  (which continue upsampling all the way to the raw waveform's sample rate) with (1) a small amount
  of remaining upsampling, (2) two parallel convolutional heads predicting log-magnitude and phase
  spectra, and (3) an inverse short-time Fourier transform (iSTFT) that reconstructs the waveform
  from those spectra (p. 1-2, Fig. 1). This "moves" the final frequency-to-time conversion out of
  the learned network and into a fixed, well-understood signal-processing operation.
* **Base architectures modified**: HiFi-GAN V1, V2, and V3 (Kong et al., 2020) — the same three
  configurations as `[Kong2020]` — each with a variable number of trailing ResBlock/upsampling
  stages ("C8", "C8C8", "C8C8C2", etc., in the paper's naming) replaced by the iSTFT head (p. 2-3,
  Fig. 3).
* **Training settings**: reused "the HiFi-GAN configuration provided in the open-source code, the
  parameters of which were tuned for stable training across various datasets" — Adam optimizer,
  initial learning rate **0.0002**, momentum `beta1=0.5, beta2=0.9`, least-squares GAN + mel-
  spectrogram + feature-matching losses, trained for **2.5M iterations** (p. 3).
* **Dataset preprocessing**: 80-dimensional log-mel spectrograms, FFT size 1024, hop length 256,
  window length 1024, at 22.05 kHz (p. 3) — matching the original HiFi-GAN paper's preprocessing
  exactly.
* **End-to-end TTS fine-tuning**: Conformer-FastSpeech2 combined with V1 or V1-C8C8I (an iSTFTNet
  variant), fine-tuned jointly for **300k iterations** after each component was independently
  pretrained, following the fine-tuning-on-HiFi-GAN precedent (p. 4).

## Results

* Replacing an increasing number of layers with iSTFT sped up inference without heavily degrading
  quality up to a point: for V1, `V1-C8C8I` (**MOS 4.26 +/- 0.17**) is comparable to or slightly
  above the original V1 (**MOS 4.22 +/- 0.17**) while running **245.68x real-time on GPU** (1.71x
  the original V1's 143.59x) and **2.33x faster on CPU** (1.74x the original), with **13.26M**
  parameters versus V1's **13.94M** [Table 1, p. 3].
* Replacing too many layers degrades quality sharply: `V1-C8I` drops to **MOS 3.32 (+/- 0.22)** from
  V1's 4.22, despite reaching **609.43x real-time on GPU** [Table 1, p. 3].
* For the lightest variant V2 (0.93M params, original **MOS 3.91**), `V2-C8C8C2I` is comparable
  (**MOS 3.98**) at **732.96x real-time on GPU** (117% of original speed), while `V2-C8I` collapses
  to **MOS 3.21** [Table 1, p. 3].
* For V3 (1.46M params, original **MOS 3.78**), even the mildest tested replacement (`V3-C8C8I`)
  degrades quality to **MOS 3.41**, unlike V1/V2 — the paper attributes this to V3 already being
  "carefully tuned... to reduce the number of layers", leaving little redundant capacity to trade
  away [Table 1, p. 3].
* End-to-end TTS synthesis (Conformer-FS2 + V1-C8C8I) achieves **MOS 4.25 (+/- 0.11)**, matching
  Conformer-FS2 + original V1 (**MOS 4.09 (+/- 0.12)**, actually slightly higher for the iSTFTNet
  variant) and clearly beating the Conformer-FS2 autoregressive-style baseline alone (**MOS 3.66 +/-
  0.15**) [Table 2, p. 3].
* Training budget for all vocoder-only models: **2.5M iterations** from scratch (p. 3); end-to-end
  TTS+vocoder fine-tuning: **300k iterations** (p. 4).

## Innovations

### Partial replacement of learned upsampling with inverse STFT

Rather than learning the entire spectrogram-to-waveform mapping end-to-end, iSTFTNet keeps only the
early upsampling/ResBlock stages as learned layers and hands off the final frequency-to-time
conversion to a closed-form inverse STFT, cutting both parameter count and inference compute for a
comparable quality budget when the number of replaced layers is tuned per base architecture.

### Per-architecture sensitivity to how much can be replaced

The paper's systematic sweep over "how many layers to replace" (Table 1, Nos. 2-15) demonstrates
that the answer is architecture-dependent: heavier variants (V1, V2) tolerate more replacement
before quality drops, while an already-minimal variant (V3) tolerates almost none — a directly
transferable lesson for any project deciding how aggressively to modify a HiFi-GAN-family decoder.

### Demonstrated compatibility with existing TTS front-ends

By plugging iSTFTNet into Conformer-FastSpeech2 as a drop-in vocoder replacement and fine-tuning
end-to-end, the paper shows the swap is compatible with standard non-autoregressive TTS acoustic
models without requiring bespoke integration work.

## Datasets

* **LJSpeech**: 13,100 clips (~24 hours), single female speaker, split into 12,600 training, 250
  validation, and 250 evaluation utterances, sampled at 22.05 kHz (p. 3). The same corpus as the
  original HiFi-GAN paper's main comparison, enabling direct step-count comparability.

## Main Ideas

* `first_stage_v3.pth` (the checkpoint `train_second_v10.py:load_checkpoint()` partially loaded into
  a HifiGAN-shaped decoder) is architecturally an iSTFTNet-family generator per t0013's tensor
  forensics — this paper is the primary source describing exactly which layers such a checkpoint
  would and would not share with a plain HiFi-GAN decoder (early ResBlock/upsampling layers shared,
  final output head diverges), which is the mechanistic reason a partial, silently-accepted shape
  match was possible at all.
* This paper's from-scratch training budget (**2.5M iterations** on a 12,600-clip corpus, matching
  the original HiFi-GAN paper) is a second, independent literature data point corroborating
  `[Kong2020]`'s: any HiFi-GAN-family decoder (HiFi-GAN proper or an iSTFTNet variant of it) trained
  from random initialization in the literature uses a step budget several orders of magnitude beyond
  what 20 epochs on ~1,500-2,000 clips would provide.
* The 300k-iteration end-to-end fine-tuning budget (versus 2.5M for from-scratch vocoder training)
  is a concrete ~8x ratio between "fine-tune a working vocoder" and "train one from scratch" —
  directly relevant to Key Question 1's framing of whether v11 should fine-tune from a genuine
  pretrained hifigan-shaped checkpoint rather than train from random init.

## Summary

iSTFTNet is a mel-spectrogram vocoder architecture that reduces the computational cost of HiFi-GAN-
family generators by replacing their final, purely-learned output-side layers with a closed-form
inverse short-time Fourier transform, aiming to cut inference cost without sacrificing the speech
quality of the original HiFi-GAN variants (p. 1-2). Its research question is how many of a base
HiFi-GAN generator's output-side layers can be replaced by this iSTFT head, and how that answer
varies across model sizes.

Methodologically, the paper applies its replacement scheme to all three published HiFi-GAN
configurations (V1, V2, V3), training each from scratch on LJSpeech with the original HiFi-GAN's
Adam optimizer settings (lr 0.0002, `beta1=0.5`, `beta2=0.9`) and loss (LSGAN + mel-spectrogram +
feature-matching) for 2.5M iterations, then separately fine-tuning an end-to-end TTS pipeline
(Conformer-FastSpeech2 + iSTFTNet) for 300k iterations (p. 3-4).

The paper finds that moderate-depth replacement (`V1-C8C8I`, `V2-C8C8C2I`) matches or slightly
exceeds the quality of the unmodified HiFi-GAN baseline while running 1.2-1.7x faster and with fewer
parameters, but replacing too many layers (`V1-C8I`, `V2-C8I`, any replacement on V3) causes sharp
MOS degradation, showing the technique's benefit is architecture- and depth-dependent (Table 1, p.
3).

For this project, the paper matters because it identifies exactly which layers an iSTFTNet-family
checkpoint (such as `first_stage_v3.pth`) shares with a plain HiFi-GAN decoder and which diverge —
the mechanistic explanation for t0013's partial-shape-match finding — and because its 2.5M-iteration
from-scratch training budget independently corroborates `[Kong2020]`'s equally large figure,
reinforcing that a HiFi-GAN-family decoder trained from random initialization needs a training
budget far larger than v10's effective ~17-epoch run, informing the epoch-count recommendation for
the v11 retrain.
