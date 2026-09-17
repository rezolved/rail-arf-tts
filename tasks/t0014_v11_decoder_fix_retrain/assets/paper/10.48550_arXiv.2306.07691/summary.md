---
spec_version: "3"
paper_id: "10.48550_arXiv.2306.07691"
citation_key: "Li2023"
summarized_by_task: "t0014_v11_decoder_fix_retrain"
date_summarized: "2026-09-16"
---
## Metadata

* File: `files/li_2023_styletts2.pdf`
* Published: 2023
* Authors: Yinghao Aaron Li, Cong Han, Vinay S. Raghavan, Gavin Mischler, Nima Mesgarani 🇺🇸
* Venue: NeurIPS 2023
* DOI: `10.48550/arXiv.2306.07691`

## Abstract

In this paper, we present StyleTTS 2, a text-to-speech (TTS) model that leverages style diffusion
and adversarial training with large speech language models (SLMs) to achieve human-level TTS
synthesis. StyleTTS 2 differs from its predecessor by modeling styles as a latent random variable
through diffusion models to generate the most suitable style for the text without requiring
reference speech, achieving efficient latent diffusion while benefiting from the diverse speech
synthesis offered by diffusion models. Furthermore, we employ large pre-trained SLMs, such as WavLM,
as discriminators with our novel differentiable duration modeling for end-to-end training, resulting
in improved speech naturalness. StyleTTS 2 surpasses human recordings on the single-speaker LJSpeech
dataset and matches it on the multispeaker VCTK dataset as judged by native English speakers.
Moreover, when trained on the LibriTTS dataset, our model outperforms previous publicly available
models for zero-shot speaker adaptation. This work achieves the first human-level TTS on both single
and multispeaker datasets, showcasing the potential of style diffusion and adversarial training with
large SLMs. The audio demos and source code are available at https://styletts2.github.io/.

## Overview

StyleTTS 2 is the architecture this project's `kokoro-finetune` pipeline is built on: `models.py`'s
`build_model()` constructs the same module set this paper describes (text encoder, style diffusion
model, duration/prosody predictors, decoder, and SLM discriminator), and `train_second_v10.py`
implements this paper's two-stage training recipe almost verbatim. The paper's central idea is to
replace StyleTTS's fixed-reference style encoding with a diffusion-sampled latent style variable, so
a target style can be generated from text alone rather than requiring a reference clip at inference
time, while still supporting reference-conditioned style transfer for voice cloning when a reference
is supplied (p. 1-2). The second major contribution is adversarial training against a frozen,
pre-trained WavLM speech-language model (SLM) acting as a discriminator, combined with a novel
differentiable duration upsampler that lets gradients from the SLM discriminator flow end-to-end
back through the duration predictor (p. 3-4).

Architecturally, the paper explicitly documents a two-stage training process: first-stage
acoustic-module pre-training (text encoder, style encoder, decoder, discriminators, text aligner,
pitch extractor), followed by second-stage joint training of the prediction modules with the
first-stage acoustic modules held fixed except for the discriminator (p. 3). This exactly matches
`config_david_v10.yml`'s `first_stage_path` / `load_only_params` / Stage 2 structure that this
project uses. The paper is explicit that the decoder is available in two interchangeable
implementations, HifiGAN-based and iSTFTNet-based (p. 4, p. 24), a design choice this project's
`config_david_v10.yml` exercises directly via `model_params.decoder.type: hifigan`.

The paper's own experimental recipe (Section 4.1, p. 6-7) reports per-dataset training schedules
that are the primary Key-Question-2 evidence this project needs: acoustic-module pre-training for
100/50/30 epochs and joint training for 60/40/25 epochs on LJSpeech/VCTK/LibriTTS respectively,
using a batch size of 16. Critically, the HifiGAN decoder (the same decoder type as
`config_david_v10.yml`) was used for the VCTK and LibriTTS experiments, not LJSpeech — the iSTFTNet
decoder was reserved for LJSpeech "due to its speed and sufficient performance on this dataset" (p.
7). This means the paper's own from-scratch/fine-tuned HifiGAN-decoder configurations used at
minimum 40 joint-training epochs (VCTK, a 43,470-clip multi-speaker corpus comparable in kind,
though far larger in size, to this project's David corpus) on top of 50 pre-training epochs.

## Architecture, Models and Methods

* **Decoder options**: "The decoder is a combination of the original decoder from StyleTTS and
  either iSTFTNet or HifiGAN, with the AdaIN module appended after each activation function" (p.
  24). This confirms the two decoder types are structurally distinct submodules with
  architecture-specific weight shapes, not interchangeable at the tensor level — consistent with
  t0013's finding that an istftnet-shaped `first_stage_v3.pth` and a hifigan-shaped
  `config_david_v10.yml` produce a partial, mismatched checkpoint load.
* **Two-stage training**: acoustic modules (text encoder, style encoder, decoder, discriminators,
  text aligner, pitch extractor) are pre-trained first via `Lmel`, `Ladv`, `Lfm`, and TMA objectives
  "for N epochs where N depends on the size of the training set" (p. 4). The paper explicitly notes
  pre-training is not mandatory: "despite being slower, starting joint training directly from
  scratch also leads to model convergence" (p. 4) — i.e., the decoder can converge without a
  pretrained starting point, but at the cost of more total epochs, not fewer.
* **Joint training objective** (Appendix G.2, p. 27): combines mel-spectrogram L1 reconstruction,
  duration cross-entropy + L1, F0/energy L1, adversarial (LSGAN) and feature-matching losses against
  MPD/MRD, an SLM adversarial loss against WavLM, and the EDM diffusion loss, with loss weights
  `lambda_dur=1, lambda_f0=0.1, lambda_n=1, lambda_ce=1, lambda_s2s=0.2, lambda_mono=5` (p. 27).
  This full multi-term joint objective, including the adversarial decoder losses, only begins to
  apply once joint training starts — in `config_david_v10.yml`'s config terms, this is the
  `joint_epoch` boundary.
  * **Optimizer**: AdamW, `beta1=0, beta2=0.99`, weight decay `1e-4`, learning rate `1e-4`, batch
    size 16, for both pre-training and joint training (p. 7).
  * **Compute**: training was conducted on four NVIDIA A40 GPUs (p. 7); no explicit wall-clock time
    per epoch is reported.
* **Style diffusion**: sampled with 3-5 steps during training (randomly, for speed) and 5 steps at
  inference (p. 7), matching a widely-copied StyleTTS2 configuration default.
* **Datasets used**: LJSpeech (13,100 clips, ~24h, single speaker), VCTK (~44,000 clips, ~44h, 109
  speakers), LibriTTS train-clean-460 (~245h, 1,151 speakers) (p. 6-7) — all substantially larger
  than this project's 1,531-clip David corpus, which is a relevant scale gap for Key Question 2.

## Results

* StyleTTS 2 achieves **CMOS +1.07** (p < 1e-6) over NaturalSpeech on LJSpeech, and is even
  preferred over ground truth with **CMOS +0.28** (p = 0.021) [Table 1, p. 7].
* On VCTK (multi-speaker, the same decoder type as this project's config), StyleTTS 2 is
  statistically indistinguishable from ground truth naturalness (**CMOS -0.02**, p = 0.628) and
  beats VITS with **CMOS-N +0.45** / **CMOS-S +0.43** (both p < 0.05) [Table 1, p. 7].
* LJSpeech in-distribution MOS is **3.83 (+/- 0.08)**, versus ground truth **3.81 (+/- 0.09)**, JETS
  **3.57**, VITS **3.34**, and "StyleTTS + HifiGAN" (an older non-end-to-end pairing) **3.35**
  [Table 2, p. 8].
* Out-of-distribution (OOD) MOS is **3.87 (+/- 0.08)** for StyleTTS 2, versus ground truth **3.70
  (+/-0.11)**, with every baseline model degrading on OOD text while StyleTTS 2 does not
  [Table 2, p. 8].
* In zero-shot speaker adaptation on LibriTTS, StyleTTS 2 beats Vall-E in naturalness (**CMOS
  +0.67**, p < 1e-3) while trailing slightly in similarity (**CMOS -0.47**, p < 1e-3), achieved with
  only 245 hours of training data versus Vall-E's 60,000 hours — a 250x data efficiency advantage
  [Table 1 and accompanying text, p. 8].
* Per-dataset training schedule: acoustic-module pre-training for **100 / 50 / 30 epochs** and joint
  training for **60 / 40 / 25 epochs** on LJSpeech / VCTK / LibriTTS respectively, batch size **16**
  [Section 4.1, p. 6-7]. VCTK and LibriTTS both use the HifiGAN decoder (p. 7) — directly relevant
  as the closest published analogue to `config_david_v10.yml`'s multispeaker, HifiGAN-decoder
  configuration.

## Innovations

### Style diffusion instead of a fixed reference encoder

Rather than encoding a single reference clip into a deterministic style vector (as in StyleTTS 1),
StyleTTS 2 models style as a distribution sampled via diffusion, conditioned on text and,
optionally, a reference for voice-transfer use cases. This removes the requirement for a reference
audio clip at inference while retaining the option to supply one.

### SLM adversarial training with a differentiable duration upsampler

The paper introduces differentiable upsampling so gradients from a frozen, pre-trained WavLM
discriminator can flow end-to-end through the duration predictor during adversarial training — a
mechanism not present in the original StyleTTS or in HiFi-GAN-style vocoder-only adversarial
training.

### Human-level naturalness on single- and multi-speaker data

The paper claims and statistically supports (via CMOS with Wilcoxon p-values) the first
demonstration of TTS output rated as good as or better than human ground-truth recordings on both a
single-speaker (LJSpeech) and multi-speaker (VCTK) benchmark simultaneously.

## Datasets

* **LJSpeech**: 13,100 clips, ~24 hours, single female speaker, split 12,500/100/500 train/val/test
  (p. 6).
* **VCTK**: ~44,000 clips, ~44 hours, 109 native-English speakers with varied accents, split
  43,470/100/500 train/val/test (p. 6).
* **LibriTTS (train-clean-460 subset)**: ~245 hours, 1,151 speakers, utterances 1-30s, split
  98%/1%/1% train/val/test; test-clean used for zero-shot evaluation with 3-second reference clips
  (p. 6).

## Main Ideas

* This project's `train_second_v10.py`/`config_david_v10.yml` pipeline directly implements this
  paper's two-stage (pre-train + joint) training recipe, including the `decoder.type: hifigan` vs.
  `istftnet` choice the paper itself treats as an interchangeable but structurally distinct
  submodule (p. 24) — this is the architectural basis of t0013's diagnosed checkpoint-shape
  mismatch.
* The paper's own from-scratch/HifiGAN-decoder schedule for a comparable multi-speaker corpus (VCTK)
  is 50 pre-training epochs + 40 joint-training epochs — both figures the project's 20-epoch
  (`epochs_2nd: 20`, `joint_epoch: 8`) v10/v11 configs fall well short of, even before accounting
  for VCTK's ~30x larger training set.
* The paper's explicit statement that "starting joint training directly from scratch also leads to
  model convergence" (p. 4) supports treating v11's decoder-random-init retrain as viable in
  principle, but only if given an epoch budget in the same order of magnitude as the paper's own
  from-scratch numbers, not v10's 20.

## Summary

StyleTTS 2 is a text-to-speech model that combines a diffusion-based latent style sampler with
adversarial training against a frozen, pre-trained WavLM speech-language model, aiming to close the
remaining naturalness gap between synthetic and human speech on both single- and multi-speaker
benchmarks (p. 1-2). Its research question is whether removing the need for a fixed reference-audio
style encoding, and adding an SLM-based adversarial signal with differentiable end-to-end duration
modeling, can push TTS naturalness to or beyond human-recording quality.

Methodologically, the model is trained in two stages: acoustic modules (text encoder, style encoder,
decoder — either HifiGAN- or iSTFTNet-based — and discriminators) are pre-trained first, then all
prediction modules are jointly optimized with the pre-trained acoustic modules mostly fixed, using
AdamW (lr 1e-4, batch size 16) and a multi-term loss combining mel reconstruction, duration,
pitch/energy, adversarial, feature-matching, and SLM-adversarial terms (p. 3-4, Appendix G.2).
Per-dataset epoch budgets are 100+60 (LJSpeech), 50+40 (VCTK, HifiGAN decoder), and 30+25 (LibriTTS,
HifiGAN decoder) pre-train+joint epochs (p. 6-7).

The paper finds that StyleTTS 2 matches or exceeds ground-truth human recordings in comparative MOS
on both LJSpeech (CMOS +0.28 over ground truth) and VCTK (CMOS -0.02 vs. ground truth, i.e.
statistically indistinguishable), and outperforms Vall-E in zero-shot naturalness using 250x less
training data (Table 1-2, p. 7-8).

For this project, the paper matters because it is the literal architecture and training recipe
`kokoro-finetune` implements, and it is the primary literature source for Key Question 2: its own
HifiGAN-decoder, multi-speaker (VCTK) configuration used 50 pre-training + 40 joint-training epochs
on a corpus ~28x larger than this project's 1,531-clip David corpus — context essential for judging
whether v10's 20-epoch (`joint_epoch: 8`) budget was realistic for training a HiFi-GAN decoder from
random initialization, and for setting a defensible epoch target for the v11 retrain.
