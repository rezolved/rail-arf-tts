---
spec_version: "3"
paper_id: "10.48550_arXiv.2112.02418"
citation_key: "Casanova2022"
summarized_by_task: "t0018_zero_shot_cloning_calibration"
date_summarized: "2026-09-17"
---
# YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for everyone

## Metadata

* **File**: `files/casanova_2022_yourtts-zero-shot-multi-speaker-tts.pdf`
* **Published**: 2022 (arXiv preprint 2021-12-04; ICML 2022)
* **Authors**: Edresson Casanova 🇧🇷, Julian Weber 🇫🇷, Christopher Shulby 🇺🇸, Arnaldo Candido Junior
  🇧🇷, Eren Golge 🇩🇪, Moacir Antonelli Ponti 🇧🇷
* **Venue**: Proceedings of the 39th International Conference on Machine Learning (ICML 2022), PMLR
  162:2709-2720
* **DOI**: `10.48550/arXiv.2112.02418`

## Abstract

YourTTS brings the power of a multilingual approach to the task of zero-shot multi-speaker TTS. Our
method builds upon the VITS model and adds several novel modifications for zero-shot multi-speaker
and multilingual training. We achieved state-of-the-art (SOTA) results in zero-shot multi-speaker
TTS and results comparable to SOTA in zero-shot voice conversion on the VCTK dataset. Additionally,
our approach achieves promising results in a target language with a single-speaker dataset, opening
possibilities for zero-shot multi-speaker TTS and zero-shot voice conversion systems in low-resource
languages. Finally, it is possible to fine-tune the YourTTS model with less than 1 minute of speech
and achieve state-of-the-art results in voice similarity and with reasonable quality. This is
important to allow synthesis for speakers with a very different voice or recording characteristics
from those seen during training.

## Overview

YourTTS extends the VITS end-to-end TTS architecture (Kim et al. 2021) with modifications that
target zero-shot multi-speaker and multilingual synthesis. The paper runs four progressively larger
training experiments — monolingual (VCTK only), bilingual (VCTK + Portuguese), trilingual (VCTK +
Portuguese + French), and a fourth stage that adds 1,151 additional English speakers from LibriTTS —
and evaluates each with Mean Opinion Score (MOS), Similarity MOS (Sim-MOS), and Speaker Encoder
Cosine Similarity (SECS) computed with a Resemblyzer speaker encoder. This SECS metric, and its
GE2E-trained Resemblyzer backbone, is the paper's most consequential contribution for downstream
speaker-similarity research: it is defined identically to how this project's own `speaker_sim`
metric is computed, and the paper is one of the clearest published precedents for a GE2E-cosine
similarity metric applied specifically to zero-shot voice cloning.

Beyond the core architecture, the paper studies a Speaker Consistency Loss (SCL) that maximizes
cosine similarity, in embedding space, between a pretrained speaker encoder's embeddings of the
generated audio and the ground-truth reference audio during fine-tuning. Critically, the paper
includes a post-publication erratum: the SCL's gradient was never actually propagated during
training due to an implementation bug, discovered by a third party after publication and fixed in
Coqui TTS v0.12.0+. This means every "+SCL" result row in the paper's tables is confounded with
simply training for more steps, not with the loss term itself — a methodological caution relevant to
any downstream comparison against the paper's reported deltas.

The paper's zero-shot voice conversion experiments and its low-resource speaker-adaptation
experiments (fine-tuning on as little as 20-61 seconds of target-speaker audio) are also directly
relevant to this project's zero-shot cloning calibration work, since they quantify how SECS and
Sim-MOS respond to reference-audio duration and to distributional mismatch between training and
target-speaker recording conditions — both open questions for this project's own metric calibration.

## Architecture, Models and Methods

YourTTS builds on VITS (Kim et al., 2021), a conditional VAE with adversarial learning that fuses a
normalizing-flow-based TTS model with an end-to-end HiFi-GAN vocoder, eliminating the need for an
intermediate mel-spectrogram representation. Key architectural changes from vanilla VITS: (1) raw
text is used as input instead of phonemes, to support languages without high-quality
grapheme-to-phoneme tools; (2) the transformer-based text encoder is scaled up to 10 transformer
blocks and 196 hidden channels, with 4-dimensional trainable language embeddings concatenated to
each input character embedding for multilingual training; (3) the flow-based decoder uses a stack of
4 affine coupling layers, each itself a stack of 4 WaveNet residual blocks; (4) the Posterior
Encoder uses 16 non-causal WaveNet residual blocks and receives a linear spectrogram plus external
speaker embeddings; (5) the vocoder is HiFi-GAN v1 with VITS's discriminator modifications; (6) a
stochastic duration predictor (from VITS) enables diverse synthesized rhythms.

Zero-shot multi-speaker capability comes from conditioning every affine coupling layer, the
posterior encoder, and the vocoder on external speaker embeddings extracted by a pretrained, frozen
H/ASP speaker encoder (Heo et al. 2020), trained with Prototypical Angular + Softmax loss on
VoxCeleb2. This encoder reached an average Equal Error Rate (EER) of 1.967 on Multilingual
LibriSpeech, versus 5.244 EER for the speaker encoder used in the authors' prior SC-GlowTTS paper.

Training used three datasets — VCTK (English, 44 hours, 109 speakers, 48kHz), TTS-Portuguese Corpus
(Brazilian Portuguese, ~10 hours, single speaker, denoised with FullSubNet and resampled to 16kHz),
and the French subset of M-AILABS (2 female speakers/104h + 3 male speakers/71h, 16kHz) — plus, in
Experiment 4, 1,151 additional English speakers from LibriTTS train-clean-100 and train-clean-360.
All audio was normalized to -27dB RMS and silence-trimmed with WebRTC VAD. Training used an NVIDIA
Tesla V100 32GB, batch size 64, AdamW optimizer (betas 0.8/0.99, weight decay 0.01), initial
learning rate 0.0002 with exponential decay (gamma 0.999875), and transfer learning from a 1M-step
LJSpeech checkpoint. Each multilingual stage trained for ~140k steps per added language, with a
further 50k-step Speaker Consistency Loss fine-tuning stage (alpha=9) applied per experiment —
though, per the erratum, that loss's gradient was not actually applied.

Evaluation metrics: MOS (naturalness, crowdsourced per Ribeiro et al. 2011's CrowdMOS methodology),
Sim-MOS (perceived speaker similarity to a reference clip), and SECS — cosine similarity between
Resemblyzer speaker-encoder embeddings of a synthesized/converted utterance and a reference
utterance, ranging from -1 to 1. English MOS/Sim-MOS used 276/200 native-English crowdworkers
respectively; Portuguese used 90 native-Portuguese crowdworkers for both. 55 LibriTTS-derived test
sentences (>20 words) were used per language condition, with the VCTK speaker's 5th sentence as the
fixed reference clip for embedding extraction.

## Results

* Ground-truth VCTK SECS is **0.824**; the best YourTTS configurations (Experiment 1 and Experiment
  2+SCL) both reach **0.864** SECS on VCTK, exceeding ground truth.
* Best VCTK Sim-MOS: **4.17 +/- 0.06** (Experiment 2), versus ground-truth Sim-MOS of **4.19 +/-
  0.06**.
* Prior SOTA comparisons on VCTK: SC-GlowTTS reaches **0.804** SECS / **3.99 +/- 0.07** Sim-MOS;
  AttentronZS reaches **0.731** SECS / **3.30 +/- 0.06** Sim-MOS — both below YourTTS's best runs.
* Best LibriTTS (out-of-domain English) SECS is **0.856** (Experiment 4+SCL), versus ground-truth
  SECS of **0.931** — a persistent domain gap even in the best configuration.
* Best Portuguese (MLS-PT) SECS is **0.798** (Experiment 4+SCL), versus ground-truth SECS of
  **0.9018**; best Portuguese Sim-MOS is **3.19 +/- 0.10** (Experiment 3), well below the **4.41 +/-
  0.05** ground-truth Sim-MOS.
* Zero-shot voice conversion, English-to-English (en-en): MOS **4.20 +/- 0.05**, Sim-MOS **4.07 +/-
  0.06** — versus prior SOTA AutoVC (MOS **3.54**, Sim-MOS **1.91**) and NoiseVC (MOS **3.38**,
  Sim-MOS **3.05**) on the same 10 VCTK speakers-not-seen-in-training setup (YourTTS used 8 of those
  10 for gender balance).
* Speaker adaptation (fine-tuning): the Portuguese female speaker's Sim-MOS rose from **2.77 +/-
  0.15** (zero-shot) to **4.43 +/- 0.06** after fine-tuning on just 20 seconds (5 samples) of that
  speaker's audio; the Portuguese male speaker rose from **3.35 +/- 0.12** to **4.19 +/- 0.07**
  after fine-tuning on 31 seconds (7 samples).
* Gender effect in Portuguese (Experiment 4, zero-shot): male Sim-MOS **3.29 +/- 0.14** versus
  female Sim-MOS **2.84 +/- 0.14**, attributed to the absence of any female Portuguese speaker in
  training data.

## Innovations

### Multilingual Zero-Shot Multi-Speaker TTS

The paper claims to be the first work to combine multilingual training with zero-shot multi-speaker
TTS, showing a single VITS-derived model can synthesize unseen speakers' voices across English,
Portuguese, and French, including cross-lingual voice transfer (e.g., an English reference speaker
synthesized in Portuguese).

### Single-Speaker Low-Resource Language Bootstrapping

The model achieves usable zero-shot multi-speaker quality in Portuguese despite that language being
represented by only one training speaker (a single-speaker corpus), demonstrating that multilingual
transfer from richer-resourced languages (English via VCTK/LibriTTS) can substitute for per-language
speaker diversity — relevant to any low-resource voice-cloning scenario.

### Sub-Minute Speaker Adaptation

Fine-tuning the pretrained zero-shot model on 20-61 seconds of a new speaker's audio for 1,500 steps
substantially improves SECS and Sim-MOS versus the zero-shot condition, at some cost to naturalness
(MOS) when the reference clip is under ~45 seconds — giving this project a concrete, publicly
reported precedent for reference-duration effects on speaker similarity.

### Documented SCL Implementation Bug (Erratum)

The paper's own published erratum (Appendix A) discloses that the Speaker Consistency Loss's
gradient never propagated during training due to an implementation mistake, discovered by an
external contributor after publication and fixed in Coqui TTS v0.12.0+. All "+SCL" rows in Tables
1-3 are therefore equivalent to additional plain fine-tuning steps, not evidence that the SCL loss
term itself improves similarity — a transparency precedent worth noting when interpreting the
paper's own ablation claims.

## Datasets

* **VCTK** (Veaux et al. 2016): English, 44 hours, 109 speakers, 48kHz; split into train/dev (same
  speakers) and an 11-speaker test set (7F/4M) held out entirely, selected one per accent. Publicly
  available (CSTR, University of Edinburgh).
* **LibriTTS** (Zen et al. 2019): English; `train-clean-100` and `train-clean-360` subsets used to
  add 1,151 speakers in Experiment 4; `test-clean` subset used for out-of-domain evaluation (10
  speakers, 5F/5M). Publicly available.
* **TTS-Portuguese Corpus** (Casanova et al. 2020): Brazilian Portuguese, ~10 hours, single male
  speaker, non-studio recording with ambient noise; denoised with FullSubNet and resampled to 16kHz.
  Publicly available.
* **M-AILABS (fr_FR)**: French, derived from LibriVox audiobooks; 2 female speakers (104h) + 3 male
  speakers (71h), 16kHz. Publicly available.
* **Multilingual LibriSpeech (MLS) Portuguese subset** (Pratap et al. 2020): used for Portuguese
  out-of-domain evaluation (10 speakers, 5F/5M) and for speaker-encoder EER validation. Publicly
  available.
* **Common Voice** (Ardila et al. 2020): 4 speakers (2 Portuguese, 2 English; 1M/1F each) used for
  the speaker-adaptation fine-tuning experiments, with clips of 20-61 seconds. Publicly available,
  crowdsourced.

## Main Ideas

* The paper's Speaker Encoder Cosine Similarity (SECS) metric — cosine similarity between
  Resemblyzer (GE2E-trained) speaker-encoder embeddings of two utterances — is architecturally
  identical to this project's own `speaker_sim` GE2E-cosine metric, making YourTTS's SECS numbers
  (e.g., ground-truth VCTK SECS of 0.824, best-model SECS of 0.864) the closest published reference
  range for what "good" GE2E-cosine similarity looks like in a zero-shot cloning context, unlike the
  WavLM-based SIM-o/SS metrics reported by F5-TTS and CosyVoice 2.
* Reference-audio duration has a measurable, reproducible effect on both SECS and Sim-MOS: as little
  as 20-31 seconds of target-speaker audio can raise Sim-MOS from the 2.7-3.4 range to the 4.2-4.4
  range after fine-tuning, but sub-45-second clips can simultaneously reduce naturalness (MOS) — a
  duration/similarity/quality trade-off this project's own reference-duration calibration should
  expect to reproduce.
* The published SCL erratum is a direct methodological warning: an ablation term that "improved" a
  metric in a paper's tables may later be confirmed to have been a no-op due to an implementation
  bug — a reminder to independently verify that any loss term or auxiliary signal this project adds
  is actually contributing gradient, not just consuming extra training steps.
* Cross-domain evaluation (VCTK-trained model evaluated on out-of-domain LibriTTS/MLS speakers)
  shows a substantial SECS/Sim-MOS gap versus in-domain ground truth, even in the best experiment —
  a caution against assuming in-domain benchmark similarity scores will transfer to production
  domains with different recording conditions.

## Summary

YourTTS addresses zero-shot multi-speaker TTS — synthesizing speech in a voice unseen during
training, from only seconds of reference audio — and extends it to multilingual and zero-shot
voice-conversion settings. Prior methods (Tacotron-2-based speaker-embedding conditioning,
Attentron, SC-GlowTTS) had each advanced accuracy on English VCTK, but none had jointly attacked
multilingual, low-resource-language, and cross-lingual zero-shot cloning within one architecture.
The paper's scope spans four training configurations that progressively add languages and speaker
count, evaluated on MOS, Sim-MOS, and a GE2E-based SECS metric.

Methodologically, YourTTS builds on the VITS conditional-VAE-plus-flow architecture, adding raw-text
input (no phonemizer), a larger transformer text encoder with concatenated language embeddings, and
external speaker-embedding conditioning throughout the flow decoder, posterior encoder, and vocoder
using a frozen VoxCeleb2-trained H/ASP speaker encoder. A Speaker Consistency Loss was proposed to
further push generated-audio embeddings toward the reference speaker's embedding, though the paper's
own erratum later revealed this loss's gradient never actually propagated due to an implementation
bug.

The headline findings are state-of-the-art VCTK zero-shot multi-speaker SECS/Sim-MOS (matching or
exceeding ground truth: 0.864 vs. 0.824 SECS), voice-conversion results well above the prior AutoVC
and NoiseVC baselines (Sim-MOS 4.07 vs. 1.91 and 3.05), and evidence that even a single-speaker
target-language dataset (Portuguese) can reach usable zero-shot quality when trained jointly with
richer-resourced languages. Sub-minute speaker adaptation was shown to substantially close remaining
similarity gaps, at some naturalness cost for very short (<45s) reference clips.

For this project, YourTTS is significant primarily because its SECS metric is defined identically to
this project's own GE2E-cosine `speaker_sim` metric — unlike the WavLM-TDNN-based SIM-o/SS metrics
reported by newer systems like F5-TTS and CosyVoice 2, which are not directly comparable. YourTTS's
reported SECS ranges (ground truth ~0.82-0.93 depending on dataset; best zero-shot models
~0.75-0.86) provide the most directly applicable published calibration point for interpreting this
project's own speaker_sim cosine values against ElevenLabs David reference clips, and its
reference-duration and cross-domain findings both directly inform this project's own zero-shot
cloning calibration experiments.
