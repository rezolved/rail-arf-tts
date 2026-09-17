---
spec_version: "3"
paper_id: "10.48550_arXiv.2410.06885"
citation_key: "Chen2024"
summarized_by_task: "t0018_zero_shot_cloning_calibration"
date_summarized: "2026-09-17"
---
## Metadata

* File: `files/chen_2024_f5-tts-flow-matching.pdf`
* Published: 2024 (arXiv v1, 2024-10-09; camera-ready as ACL 2025)
* Authors: Yushen Chen 🇨🇳, Zhikang Niu 🇨🇳, Ziyang Ma 🇨🇳, Keqi Deng 🇬🇧, Chunhui Wang 🇨🇳, Jian Zhao
  🇨🇳, Kai Yu 🇨🇳, Xie Chen 🇨🇳
* Venue: Annual Meeting of the Association for Computational Linguistics (ACL 2025)
* DOI: `10.48550/arXiv.2410.06885`

## Abstract

This paper introduces F5-TTS, a fully non-autoregressive text-to-speech system based on flow
matching with Diffusion Transformer (DiT). Without requiring complex designs such as duration model,
text encoder, and phoneme alignment, the text input is simply padded with filler tokens to the same
length as input speech, and then the denoising is performed for speech generation, which was
originally proved feasible by E2 TTS. However, the original design of E2 TTS makes it hard to follow
due to its slow convergence and low robustness. To address these issues, we first model the input
with ConvNeXt to refine the text representation, making it easy to align with the speech. We further
propose an inference-time Sway Sampling strategy, which significantly improves our model's
performance and efficiency. This sampling strategy for flow step can be easily applied to existing
flow matching based models without retraining. Our design allows faster training and achieves an
inference RTF of 0.15, which is greatly improved compared to state-of-the-art diffusion-based TTS
models. Trained on a public 100K hours multilingual dataset, our F5-TTS exhibits highly natural and
expressive zero-shot ability, seamless code-switching capability, and speed control efficiency. We
have released all codes and checkpoints to promote community development, at
https://SWivid.github.io/F5-TTS/.

## Overview

F5-TTS is a fully non-autoregressive (NAR), zero-shot text-to-speech system that frames synthesis as
a text-guided speech-infilling problem solved with conditional flow matching (CFM) and a Diffusion
Transformer (DiT) backbone. It is a direct successor to Microsoft's E2 TTS, which pads a raw
character sequence with filler tokens to the mel-spectrogram length and denoises it end to end with
no phoneme aligner, duration predictor, text encoder, or semantically-infused neural codec. The
authors diagnose E2 TTS's core weakness as a "deep entanglement" of semantic (text) and acoustic
(mel) features caused by directly concatenating a coarse, filler-padded character sequence with the
speech sequence — this produces slow convergence and frequent alignment failures (looping, skipping,
or hallucinated speech) that cannot be fixed by re-ranking multiple samples.

F5-TTS addresses this with two orthogonal contributions. First, ConvNeXt V2 blocks refine the
character sequence in its own representation space before it is concatenated with the noisy/masked
speech features, giving the text a chance to "prepare itself" for in-context learning rather than
being consumed raw. Second, at inference time only (no retraining), a Sway Sampling function
reshapes the distribution of flow-matching steps fed to the ODE solver, biasing sampling toward
earlier steps (sway-to-left, coefficient s < 0). The authors show through a "leak and override"
diagnostic experiment that early flow steps determine the coarse content/alignment of the generated
speech while later steps refine detail — so concentrating solver evaluations early yields large
robustness gains for a fixed number of function evaluations (NFE).

Trained on 95K hours of the public, multilingual, in-the-wild Emilia dataset (English + Mandarin),
the 335.8M-parameter F5-TTS base model reaches a Real-Time Factor (RTF) of 0.15 at 16 NFE and
matches or exceeds much larger autoregressive and non-autoregressive baselines (VALL-E 2, MELLE,
Voicebox, NaturalSpeech 3, DiTTo-TTS, MaskGCT, Seed-TTS) on word error rate (WER) and speaker
similarity (SIM-o) across LibriSpeech-PC test-clean and the Seed-TTS test-en/test-zh benchmarks. A
key significance of the work for zero-shot voice-cloning TTS broadly is that it demonstrates a
duration predictor and phoneme aligner are not necessary for state-of-the-art robustness — duration
is instead estimated by a simple character-count ratio between the reference and target text — and
that an inference-time-only sampling trick (Sway Sampling) is portable to any other CFM-based TTS
model without retraining.

The paper also reports an in-depth ablation program (Section 5) isolating the effect of ConvNeXt
text refinement versus the DiT backbone versus long skip-connections, and a separate ablation of the
Sway Sampling coefficient s, both run on smaller (~155M parameter) models trained on the
WenetSpeech4TTS Premium 945-hour Mandarin subset for tractability. The authors explicitly flag two
limitations: the mel-spectrogram sequence is still much longer than the text sequence (an
unaddressed efficiency ceiling), and F5-TTS lacks fine-grained control over paralinguistic
attributes such as emotion.

## Architecture, Models and Methods

* **Task formulation**: text-guided speech infilling. Training pairs (x, y) are audio x and
  transcript y; the model predicts a masked mel segment m ⊙ x1 given the surrounding unmasked audio
  (1 − m) ⊙ x1 and an extended character sequence z (raw text padded with filler tokens `<F>` to the
  same frame length as the mel spectrogram, Eq. 5).
* **Objective**: Conditional Flow Matching (CFM) with the Optimal-Transport path (OT-CFM, Eq. 3),
  i.e., regressing a neural vector field v_t against (x1 − x0) where x0 is Gaussian noise and x1 is
  the mel target; flow step t is sampled uniformly during training, t ~ U[0, 1].
* **Backbone**: latent Diffusion Transformer (DiT) with zero-initialized adaptive LayerNorm
  (adaLN-zero) blocks — no U-Net-style long skip connections (unlike Voicebox/E2 TTS). Flow step t
  is injected via the adaLN-zero conditioning path rather than concatenated to the input sequence.
* **Text refinement**: ConvNeXt V2 blocks (4 layers, 512/1024 embedding/FFN dimension) process the
  padded character sequence independently before concatenation with the speech features in the
  feature dimension, giving the text its own modeling space prior to in-context learning.
* **Positional encoding**: sinusoidal embedding for the flow step, convolutional position embedding
  on the concatenated input sequence, rotary position embedding (RoPE) for self-attention (instead
  of ALiBi), and absolute sinusoidal position embedding added to the character sequence before the
  ConvNeXt blocks.
* **Base model size**: 335.8M parameters — DiT with 22 layers, 16 attention heads, 1024/2048
  embedding/FFN dimensions, plus the 4-layer ConvNeXt V2 text branch. The reproduced E2 TTS baseline
  is a 333.2M-parameter, 24-layer, 16-head flat U-Net Transformer (1024/4096 embedding/FFN) for
  direct comparison.
* **Training data**: Emilia dataset (He et al., 2024), filtered to ~95K hours of English and
  Mandarin (in-the-wild multilingual speech). Small ablation models are trained on the
  WenetSpeech4TTS Premium subset (945 hours Mandarin).
* **Training regime**: 1.2M updates, batch size 307,200 audio frames (0.91 hours/batch), on 8×
  NVIDIA A100 80GB GPUs for over one week. AdamW optimizer, peak learning rate 7.5e-5, 20K-update
  linear warmup then linear decay, gradient-norm clip of 1. Dropout 0.1 on attention/FFN.
  Classifier- Free Guidance (CFG) training drops the masked-speech condition at rate 0.3 and drops
  masked-speech
  + text jointly at rate 0.2.
* **Text representation**: raw alphabet/symbols for English; full pinyin (via `jieba` + `pypinyin`)
  for Chinese. Character embedding vocabulary size 2546.
* **Audio features**: 100-dimensional log mel-filterbank, 24 kHz sampling rate, hop length 256;
  70-100% of mel frames randomly masked during infilling training.
* **Sway Sampling** (Eq. 7): f_sway(u; s) = u + s·(cos(π/2·u) − 1 + u), monotonic for s ∈
  [−1, 2/(π − 2)]. Applied only at inference to reshape uniformly-sampled u ~ U[0, 1] into a
  non-uniform flow-step schedule; default coefficient s = −1 (sway-to-left) with CFG strength 2.
* **Inference**: Exponential Moving Averaged (EMA) weights, Euler ODE solver for F5-TTS (midpoint
  solver for the E2 TTS baseline), duration estimated by the character-count ratio of target text to
  reference text (no separate duration predictor), Vocos vocoder to convert mel back to waveform.
* **Evaluation metrics**: WER (Whisper-large-v3 for English, Paraformer-zh for Chinese) and speaker
  similarity SIM-o (WavLM-large-based speaker verification embeddings, cosine similarity) for
  objective evaluation; Comparative Mean Opinion Score (CMOS) and Similarity Mean Opinion Score
  (SMOS) for subjective evaluation. RTF measured for 10s of generated speech on an NVIDIA RTX 3090.
* **Test sets**: LibriSpeech-PC test-clean (a new 1127-sample, 4-10s subset released by the
  authors), Seed-TTS test-en (1088 samples from Common Voice), Seed-TTS test-zh (2020 samples from
  DiDiSpeech).

## Results

* F5-TTS (32 NFE) achieves **2.42% WER** on LibriSpeech-PC test-clean with SIM-o of **0.66** and RTF
  of **0.31**; at 16 NFE it achieves **2.53% WER**, SIM-o **0.66**, RTF **0.15**.
* At 16 NFE, F5-TTS's RTF of **0.15** is faster than DiTTo-TTS (RTF **0.162**, 740M params) and CFG-
  doubled inference alternatives, while training on a larger multilingual corpus (100K hours vs.
  DiTTo-TTS's 55K hours English-only).
* On Seed-TTS test-en, F5-TTS (32 NFE) reaches **1.83% WER**, SIM-o **0.67**, CMOS **0.31**, SMOS
  **3.89**, versus ground truth WER **2.06%**/SIM **0.73**; F5-TTS (16 NFE) reaches **1.89% WER**,
  SIM-o **0.67**, CMOS **0.16**, SMOS **3.79**.
* On Seed-TTS test-zh, F5-TTS (32 NFE) reaches **1.56% WER**, SIM-o **0.76**, CMOS **0.21**, SMOS
  **3.83**, versus ground truth WER **1.26%**/SIM **0.76**; F5-TTS (16 NFE) reaches **1.74% WER**,
  SIM-o **0.75**.
* The reproduced E2 TTS (32 NFE) baseline on Seed-TTS test-zh has a notably worse WER (**9.63%**,
  per the paper's discussion of E2 TTS's Mandarin failures) versus F5-TTS's low-single-digit WER,
  while E2 TTS's SIM (**0.53** in that discussion) is also markedly lower than F5-TTS's SIM-o scores
  in the corresponding regime — illustrating E2 TTS's inherent alignment-robustness deficiency
  (consistently failing, WER > 50%, on 7% of Mandarin test samples throughout training).
* CosyVoice (~300M params, 170K hours multilingual) scores **3.59% WER** / SIM-o **0.66** / RTF
  **0.92** on LibriSpeech-PC test-clean — F5-TTS at 16 NFE beats it on both WER and RTF (**0.15**
  vs. **0.92**) despite training on less data (100K vs. 170K hours).
* Small-model ablation (155M params, WenetSpeech4TTS Premium, 945h Mandarin, 800K updates): F5-TTS
  (32 NFE, without Sway Sampling) achieves **4.17% WER** and SIM of **0.54**; removing the ConvNeXt
  text-refinement branch (pure adaLN DiT, "F5-TTS−Conv2Text") fails to learn alignment at all, and
  substituting MMDiT instead trains fast but collapses fast into "severe repeated utterance with
  wild timbre and prosody."
* Adding a symmetric ConvNeXt branch on the audio side ("F5-TTS+Conv2Audio") trades alignment
  robustness for speaker similarity: **+1.61 WER** in exchange for only **+0.01 SIM**.
* Sway Sampling with more negative s further improves performance; the paper's "leak and override"
  diagnostic shows the model can override a leaked ground-truth audio prompt and follow the provided
  text prompt only when Sway Sampling is used — uniform flow-step sampling instead reproduces the
  leaked (duplicated) audio content.

## Innovations

### ConvNeXt-based text refinement

Rather than feeding the raw, filler-padded character sequence directly into the shared
Transformer/U-Net backbone (as E2 TTS does), F5-TTS routes it through ConvNeXt V2 blocks first,
giving the text an independent representation space to "prepare itself" before in-context learning
with the speech features. This is presented as the fix for E2 TTS's entangled semantic/acoustic
representation and is shown (via ablation) to be necessary for the pure-adaLN DiT backbone to learn
alignment at all.

### Sway Sampling (inference-time, training-free)

A closed-form reshaping function (Eq. 7) for the flow-matching step schedule used only at inference.
It requires no retraining and, per the paper, "can be easily applied to existing flow matching based
models without retraining" — a portable technique with implications beyond F5-TTS for any CFM-based
TTS or generative model wanting to trade NFE for accuracy without architecture changes.

### DiT backbone without duration predictor, phoneme aligner, text encoder, or semantic codec

F5-TTS keeps E2 TTS's "text in, speech out" simplicity (no explicit alignment module) while
replacing the U-Net-style backbone with a skip-connection-free DiT using adaLN-zero conditioning,
demonstrating that architecture choice (not just data scale) determines whether such a simple
pipeline is robust or brittle.

### Public LibriSpeech-PC evaluation subset

The authors release a new 1127-sample, 4-10 second LibriSpeech-PC test-clean subset specifically to
enable fair, reproducible community comparisons, noting that prior work evaluated on undisclosed
subsets of LibriSpeech test-clean with unreleased prompt lists.

## Datasets

* **Emilia** (He et al., 2024): in-the-wild, multilingual speech dataset; filtered to ~95K hours of
  English and Chinese data used to train the F5-TTS and reproduced E2 TTS base models. Publicly
  available per the paper's citation.
* **WenetSpeech4TTS Premium subset** (Ma et al., 2024): 945-hour Mandarin corpus used for smaller
  ablation and architecture-search models (~155M parameters).
* **LibriSpeech-PC test-clean**: a new 1127-sample (4-10s), 2-hour English evaluation subset built
  and released by the authors for reproducible zero-shot TTS comparison.
* **Seed-TTS test-en**: 1088 samples drawn from Common Voice (Ardila et al., 2019).
* **Seed-TTS test-zh**: 2020 samples drawn from DiDiSpeech (Guo et al., 2021).
* Licensing/language details beyond dataset names and sizes are not reported in the paper body
  (dataset-specific licenses are governed by the original Emilia/WenetSpeech4TTS/Common
  Voice/DiDiSpeech releases, not restated here).

## Main Ideas

* F5-TTS is one of the three zero-shot voice-cloning systems (alongside CosyVoice 2 and Chatterbox)
  that task t0018 directly benchmarks; this paper is the primary source for F5-TTS's official
  SIM-o/WER numbers (e.g., LibriSpeech-PC test-clean: **2.42% WER** / **0.66 SIM-o** at 32 NFE) and
  architecture, which can anchor apples-to-apples comparisons against Kokoro-82M and the ElevenLabs
  David baseline on this project's own filler corpus.
* The paper's duration-estimation approach (target/reference character-count ratio, no separate
  duration predictor) is a low-complexity technique worth considering if this project ever needs to
  control output duration without training an explicit duration model.
* Sway Sampling is presented as retraining-free and portable to any CFM-based TTS model — if a
  future stage of this project experiments with flow-matching-based vocoders/decoders, this
  technique could be evaluated for latency/robustness trade-offs without additional training cost.
* The paper explicitly reports RTF measured on a single reference GPU (RTX 3090) for 10s of
  generated speech, which is a useful methodological precedent for how this project should report
  its own RTF/TTFB numbers consistently across models.

## Summary

F5-TTS asks whether a fully non-autoregressive, zero-shot TTS system can match state-of-the-art
speech quality and speaker similarity while remaining architecturally simple — with no duration
predictor, phoneme aligner, text encoder, or semantically-infused neural codec — and whether the
robustness problems of its direct predecessor, E2 TTS, stem from entangled text/speech
representations rather than the lack of those components. The paper is motivated by the practical
burden E2 TTS's slow convergence and alignment failures pose for real-world (industrial) TTS
deployment.

Methodologically, F5-TTS keeps E2 TTS's text-guided speech-infilling formulation and Conditional
Flow Matching training objective, but swaps the backbone for a skip-connection-free Diffusion
Transformer with adaLN-zero conditioning, and routes the padded character sequence through ConvNeXt
V2 blocks before concatenation with speech features so text can be refined independently prior to
in-context learning. It further introduces Sway Sampling, an inference-time-only, retraining-free
reshaping of the flow-matching step schedule that biases sampling toward earlier flow steps. The
335.8M-parameter base model is trained on ~95K hours of the multilingual Emilia dataset on 8×A100
GPUs for over a week.

The paper finds that F5-TTS matches or beats much larger autoregressive and non-autoregressive
baselines (VALL-E 2, Voicebox, NaturalSpeech 3, DiTTo-TTS, MaskGCT, CosyVoice) on WER and speaker
similarity across LibriSpeech-PC and Seed-TTS test-en/test-zh, while reaching an inference RTF of
0.15 at 16 NFE — substantially faster than comparable diffusion-based TTS systems. Ablations show
the ConvNeXt text-refinement branch is necessary (its removal causes the pure-adaLN DiT to fail to
learn alignment at all) and that Sway Sampling with a sway-to-left coefficient materially improves
robustness and efficiency without any additional training.

For this project, F5-TTS matters as one of the three reference zero-shot voice-cloning systems that
task t0018_zero_shot_cloning_calibration benchmarks against, and this paper is the only source in
the project's paper corpus for F5-TTS's official architecture and SIM-o/WER numbers — filling a gap
that the research-internet step of t0018 identified as having zero prior coverage. Its
duration-estimation heuristic and portable, training-free Sway Sampling technique are also directly
relevant design patterns this project could borrow when evaluating or tuning Kokoro-82M's own
inference behavior.
