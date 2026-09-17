---
spec_version: "3"
paper_id: "10.48550_arXiv.2506.20190"
citation_key: "Kunesova2025"
summarized_by_task: "t0018_zero_shot_cloning_calibration"
date_summarized: "2026-09-17"
---
## Metadata

* **File**: `files/kunesova_2025_ecapa-tdnn-xvector-zeroshot-tts.pdf`
* **Published**: 2025
* **Authors**: Marie Kunešová 🇨🇿, Zdeněk Hanzlíček 🇨🇿, Jindřich Matoušek 🇨🇿
* **Venue**: International Conference on Text, Speech, and Dialogue (TSD 2025)
* **DOI**: `10.48550/arXiv.2506.20190`

## Abstract

Zero-shot multi-speaker text-to-speech (TTS) systems rely on speaker embeddings to synthesize speech
in the voice of an unseen speaker, using only a short reference utterance. While many speaker
embeddings have been developed for speaker recognition, their relative effectiveness in zero-shot
TTS remains underexplored. In this work, we employ a YourTTS-based TTS system to compare three
different speaker encoders - YourTTS's original H/ASP encoder, x-vector embeddings, and ECAPA-TDNN
embeddings - within an otherwise fixed zero-shot TTS framework. All models were trained on the same
dataset of Czech read speech and evaluated on 24 out-of-domain target speakers using both subjective
and objective methods. The subjective evaluation was conducted via a listening test focused on
speaker similarity, while the objective evaluation measured cosine distances between speaker
embeddings extracted from synthesized and real utterances. Across both evaluations, the original
H/ASP encoder consistently outperformed the alternatives, with ECAPA-TDNN showing better results
than x-vectors. These findings suggest that, despite the popularity of ECAPA-TDNN in speaker
recognition, it does not necessarily offer improvements for speaker similarity in zero-shot TTS in
this configuration. Our study highlights the importance of empirical evaluation when reusing speaker
recognition embeddings in TTS and provides a framework for additional future comparisons.

## Overview

This paper asks a narrow but practically important question: if you swap the speaker encoder inside
a fixed zero-shot multi-speaker TTS system, does a "better" speaker-recognition embedding actually
produce more speaker-similar synthesized speech? The authors use YourTTS (a VITS-based zero-shot
multi-speaker system) as the fixed TTS backbone and vary only the speaker encoder that conditions
it: the system's original H/ASP encoder (trained on VoxCeleb2), a SpeechBrain x-vector model
(trained on VoxCeleb1+2), and a SpeechBrain ECAPA-TDNN model (also VoxCeleb1+2). All three resulting
TTS models are trained from scratch on the same in-house Czech read-speech corpus ("SPT-MGW", 1062
speakers after filtering, ~15 minutes per speaker) for 295 epochs (~1.6M steps), with Speaker
Consistency Loss (SCL) enabled in all three configurations.

The evaluation combines a MUSHRA-style subjective listening test (23 native Czech listeners rating
speaker similarity for 24 held-out target speakers sourced from radio broadcasts) with a fully
separate objective evaluation that measures cosine distance between embeddings of synthesized and
real utterances, repeated across four different, independently-chosen speaker-embedding extractors
(ECAPA-TDNN, x-vector, NVIDIA TitaNet-large, and the Resemblyzer package, which implements the
GE2E/d-vector approach of Wan et al. 2018). Using multiple external, evaluation-only embedding
extractors is a methodological device meant to avoid the circularity of judging an encoder's TTS
output using that same encoder's own embedding space.

The central finding, replicated across both the subjective test and all four objective embedding
extractors used for scoring (with the partial exception of Resemblyzer and TitaNet, where the
alternate encoders' relative ranking is noisier), is negative: neither x-vector nor ECAPA-TDNN
embeddings improve speaker similarity over the original H/ASP encoder baseline, despite ECAPA-TDNN
being regarded as state-of-the-art for speaker verification. Statistical testing (Wilcoxon
signed-rank with Holm-Bonferroni correction) confirms H/ASP is significantly better than both
alternatives (p < 0.001), and ECAPA-TDNN is significantly better than x-vector (p = 0.02-0.03). The
paper positions this as evidence that speaker-recognition embedding quality does not transfer
directly to zero-shot TTS speaker-similarity quality, and that plug-and-play embedding swaps require
empirical validation rather than being assumed to help based on speaker-verification benchmarks
alone.

## Architecture, Models and Methods

* **TTS backbone**: YourTTS (a fork of Coqui-ai/TTS), a VITS-based end-to-end zero-shot
  multi-speaker TTS model, conditioned on a speaker embedding vector instead of a discrete speaker
  ID, enabling synthesis for unseen speakers at inference time.
* **Speaker encoders compared** (three TTS models trained, one per encoder):
  1. "H/ASP TTS" (baseline) — YourTTS's original pretrained H/ASP speaker encoder, embedding
     dimension 512, trained on VoxCeleb2.
  2. "x-vector TTS" — SpeechBrain `spkrec-xvect-voxceleb` pretrained x-vector model, embedding
     dimension 512, trained on VoxCeleb1+VoxCeleb2.
  3. "ECAPA-TDNN TTS" — SpeechBrain ECAPA-TDNN pretrained model (SpeechBrain VoxCeleb/SpeakerRec
     recipe), embedding dimension 192, trained on VoxCeleb1+VoxCeleb2.
* **Speaker Consistency Loss (SCL)**: enabled (`use_speaker_encoder_as_loss = True`) for all three
  models; compares speaker embeddings of generated vs. original audio via cosine distance during
  training.
* **Training data**: in-house "SPT-MGW" Czech read-speech corpus, 1116 speakers recorded at 24 kHz
  with 150-174 utterances (~15 minutes) per speaker; 54 noisy speakers excluded, leaving 1062
  speakers for training. ~25% of prompts identical across speakers, the rest drawn from a larger
  text pool.
* **Training hyperparameters**: 295 epochs (~1.6 million steps) per model, hop length 256, sample
  rate 24000 Hz, `mixed_precision = False`, embedding dim 512 (baseline, x-vector) or 192
  (ECAPA-TDNN).
* **Target speaker data**: 24 out-of-domain speakers (12 men, 12 women) selected from a large
  archive of manually labeled Czech radio broadcasts, none present in the training set; reference
  audio for TTS conditioning was 30-second concatenations of held-out utterances (resampled to 16
  kHz).
* **Subjective evaluation**: MUSHRA-style listening test with 23 native Czech listeners (12
  experienced, 11 non-expert) rating speaker similarity (0-100 scale) for 24 speakers x 3 TTS
  models, using a single fixed test sentence; results reported as raw and per-listener normalized
  ratings, with Wilcoxon signed-rank tests (Holm-Bonferroni corrected) for pairwise significance.
* **Objective evaluation**: cosine distance between embeddings of 21 synthesized utterances and 15
  real utterances per target speaker (24 speakers), repeated independently with four different
  embedding extractors used purely for scoring: ECAPA-TDNN, x-vector, NVIDIA TitaNet-large (via
  NeMo), and Resemblyzer (GE2E/d-vector style, per Wan et al. 2018); embeddings mean- and
  L2-normalized except for Resemblyzer, which already L2-normalizes internally. Reference benchmarks
  (distance to same speaker, to the 2nd-closest speaker, and to the average of all other speakers)
  were computed for context. Significance testing again used Wilcoxon signed-rank with
  Holm-Bonferroni correction, aggregating 24 speakers x 4 extractors = 96 values per TTS model.

## Results

* Subjective listening test (normalized ratings, mean ± std): H/ASP TTS (baseline) **47.31 ±
  20.50**, ECAPA-TDNN TTS **42.61 ± 20.38**, x-vector TTS **40.96 ± 20.85** (0-100 scale, 24
  speakers x 23 listeners).
* Subjective raw ratings (mean ± std, median): H/ASP **47.28 ± 26.19** (median **50.0**), ECAPA-TDNN
  **42.62 ± 26.52** (median **43.0**), x-vector **40.98 ± 26.24** (median **40.5**).
* Subjective significance: H/ASP TTS rated significantly better than both alternatives (**p <
  0.001** for each pairwise comparison); ECAPA-TDNN TTS rated significantly better than x-vector TTS
  (**p = 0.03**).
* Objective cosine-distance results using ECAPA-TDNN as the scoring extractor (mean ± std, lower =
  more similar): H/ASP **0.524 ± 0.052**, ECAPA-TDNN TTS **0.533 ± 0.054**, x-vector TTS **0.696 ±
  0.090**.
* Objective cosine-distance results using Resemblyzer (GE2E-style) as the scoring extractor: H/ASP
  **0.438 ± 0.046**, ECAPA-TDNN TTS **0.494 ± 0.042**, x-vector TTS **0.627 ± 0.086**.
* Objective cosine-distance results using x-vector as the scoring extractor: H/ASP **0.674 ±
  0.070**, ECAPA-TDNN TTS **0.681 ± 0.081**, x-vector TTS **0.649 ± 0.074** (the one case where
  x-vector TTS scores marginally better than the baseline under its own matching extractor).
* Objective cosine-distance results using NVIDIA TitaNet-large as the scoring extractor: H/ASP
  **0.231 ± 0.034**, ECAPA-TDNN TTS **0.254 ± 0.030**, x-vector TTS **0.237 ± 0.027**.
* Objective significance (aggregated across 24 speakers x 4 extractors, 96 values per model):
  "ECAPA-TDNN TTS" and "H/ASP TTS" both significantly better than "x-vector TTS" (**p < 0.001** for
  both); "H/ASP TTS" significantly better than "ECAPA-TDNN TTS" (**p = 0.02**) — consistent with the
  subjective test ranking.

## Innovations

### Isolating the speaker encoder as the sole independent variable

Rather than proposing a new TTS architecture or embedding method, the paper's contribution is purely
experimental design: it holds the TTS system, training data, and training procedure fixed and swaps
only the speaker encoder, letting the effect of encoder choice be measured in isolation — something
prior TTS papers that adopt ECAPA-TDNN embeddings (e.g., for duration prediction or voice-conversion
speaker loss) had not directly benchmarked against alternatives.

### Four independent external embedding extractors for objective scoring

To avoid circularity (scoring a TTS model's output using the very embedding model that trained it),
the objective evaluation is run four separate times with four extractors not all tied to the
encoders being compared (ECAPA-TDNN, x-vector, TitaNet-large, Resemblyzer/GE2E), and it reports
reference distances (same-speaker, 2nd-closest speaker, average-speaker) for scale calibration. This
is presented as a reusable framework for future TTS speaker-encoder comparisons.

### Empirical counter-evidence to a "better encoder = better TTS" assumption

The paper's main scientific contribution is a negative result with a clear practical implication:
strong speaker-verification performance (ECAPA-TDNN's SOTA status) does not automatically transfer
to better zero-shot TTS speaker similarity, at least in the tested plug-and-play configuration
against a well-tuned existing baseline (H/ASP with SCL).

## Datasets

* **SPT-MGW** (in-house, not publicly named as released): Czech read-speech corpus for TTS training,
  1116 speakers originally recorded for ASR research in home environments (24 kHz), 150- 174
  utterances (~15 minutes) per speaker; 54 speakers excluded for noise, 1062 used for training.
  License/availability not stated in the paper (in-house dataset).
* **Czech radio broadcast archive** (in-house): manually labeled archive spanning over a decade,
  used to source 24 out-of-domain target speakers (12 men, 12 women) for evaluation reference and
  test utterances. Availability/license not stated.
* **VoxCeleb1 / VoxCeleb2**: used to pretrain the external speaker encoders (x-vector, ECAPA-TDNN,
  TitaNet-large, H/ASP), not used directly for TTS training in this paper. Publicly available
  research datasets (standard academic license).

## Main Ideas

* This project's own `speaker_sim` metric uses GE2E cosine similarity (via a Resemblyzer-style
  approach) against ElevenLabs reference clips; this paper's finding that Resemblyzer/GE2E,
  x-vector, ECAPA-TDNN, and TitaNet rankings are not fully consistent with each other (Table 3) is a
  direct caution that a single embedding-extractor choice can shift which system "looks" more
  speaker-similar — worth flagging when comparing this project's GE2E-based metric against
  WavLM-based SIM-o/SS numbers reported by F5-TTS/CosyVoice 2.
* The paper's core result — that a purportedly stronger speaker-recognition embedding (ECAPA-TDNN)
  did not outperform an older, already-integrated encoder (H/ASP) when swapped into a fixed TTS
  pipeline — is a caution against assuming Kokoro-82M fine-tuning or evaluation changes driven by
  "state-of-the-art" speaker embeddings will automatically improve measured `speaker_sim` without
  direct empirical A/B testing.
* The paper's objective-evaluation design (multiple independent scoring extractors plus
  same-speaker/2nd-closest/average-speaker reference distances for calibration) is a reusable
  template this project could adapt when validating whether its GE2E-based `speaker_sim` metric
  agrees with alternative embedding extractors on the val_96 / 11labs_david benchmark.
* Because this is a low-priority, tangential background paper (Czech, YourTTS-specific, not directly
  about Kokoro or ElevenLabs), it should inform methodology discussion rather than be treated as a
  benchmark comparison target.

## Summary

This paper investigates whether replacing the speaker encoder in a fixed zero-shot multi-speaker TTS
system with a more modern, higher-performing speaker-recognition embedding model improves speaker
similarity of synthesized speech. The motivation is that many speaker embeddings (x-vector,
ECAPA-TDNN, d-vectors, WavLM-based embeddings) were developed for speaker recognition/verification
rather than TTS, and their relative effectiveness when plugged into a TTS speaker-conditioning
pipeline has been comparatively underexplored, with most prior TTS work adopting a chosen embedding
without directly comparing it against alternatives.

The authors isolate the speaker-encoder variable by training three otherwise-identical YourTTS
(VITS-based) zero-shot multi-speaker models on the same Czech read-speech corpus (1062 speakers, 295
epochs each, with Speaker Consistency Loss enabled), differing only in whether the speaker encoder
is YourTTS's original H/ASP model, a SpeechBrain x-vector model, or a SpeechBrain ECAPA-TDNN model.
They evaluate speaker similarity on 24 out-of-domain target speakers both subjectively (a
23-listener MUSHRA-style test) and objectively (cosine distance between synthesized and real speaker
embeddings, scored independently by four different extractors — ECAPA-TDNN, x-vector, TitaNet-large,
and Resemblyzer/GE2E — to avoid scoring circularity).

Both evaluations converge on the same negative result: the original H/ASP encoder significantly
outperforms both alternatives (p < 0.001 in the subjective test; p < 0.001 in the aggregated
objective test), and ECAPA-TDNN significantly outperforms x-vector (p = 0.02-0.03 in both tests),
despite ECAPA-TDNN's stronger reputation in speaker-verification benchmarks. The objective results
also show that extractor choice matters for ranking consistency — the four scoring extractors mostly
agree but diverge somewhat under Resemblyzer and TitaNet-large scoring.

For this project, the paper is useful background rather than a direct benchmark: it is evidence that
a speaker embedding's speaker-verification quality does not automatically predict its usefulness for
zero-shot TTS speaker-similarity conditioning, reinforcing the need to empirically validate any
embedding-extractor choice (including this project's GE2E-based `speaker_sim` metric versus
WavLM-based alternatives) rather than assuming state-of-the-art speaker-recognition performance
transfers directly to TTS evaluation quality.
