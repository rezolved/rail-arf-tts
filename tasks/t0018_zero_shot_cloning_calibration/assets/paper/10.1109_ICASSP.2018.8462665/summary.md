---
spec_version: "3"
paper_id: "10.1109_ICASSP.2018.8462665"
citation_key: "Wan2018"
summarized_by_task: "t0018_zero_shot_cloning_calibration"
date_summarized: "2026-09-17"
---
# Generalized End-to-End Loss for Speaker Verification

## Metadata

* **File**: `files/wan_2018_ge2e-loss-speaker-verification.pdf`
* **Published**: 2018
* **Authors**: Li Wan 🇺🇸, Quan Wang 🇺🇸, Alan Papir 🇺🇸, Ignacio Lopez Moreno 🇺🇸
* **Venue**: 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
* **DOI**: `10.1109/ICASSP.2018.8462665`

## Abstract

In this paper, we propose a new loss function called generalized end-to-end (GE2E) loss, which makes
the training of speaker verification models more efficient than our previous tuple-based end-to-end
(TE2E) loss function. Unlike TE2E, the GE2E loss function updates the network in a way that
emphasizes examples that are difficult to verify at each step of the training process. Additionally,
the GE2E loss does not require an initial stage of example selection. With these properties, our
model with the new loss function decreases speaker verification EER by more than 10%, while reducing
the training time by 60% at the same time. We also introduce the MultiReader technique, which allows
us to do domain adaptation - training a more accurate model that supports multiple keywords (i.e.,
"OK Google" and "Hey Google") as well as multiple dialects.

## Overview

This paper introduces the generalized end-to-end (GE2E) loss, a training objective for neural
speaker-embedding ("d-vector") models used in speaker verification (SV). It is a direct successor to
the authors' earlier tuple-based end-to-end (TE2E) loss (Heigold et al., 2016), and the GE2E
formulation and evaluation protocol described here is the mechanism underlying the cosine-similarity
speaker-verification metrics ("GE2E cosine similarity") widely cited in later TTS and voice-cloning
literature, including the `speaker_sim` metric this project uses against ElevenLabs David reference
clips.

Where TE2E trains on individual tuples of one evaluation utterance plus a small set of same-speaker
enrollment utterances, requiring an explicit positive/negative tuple selection step, GE2E instead
processes a full batch of `N` speakers times `M` utterances per speaker at once. It builds a
similarity matrix between every utterance's embedding and every speaker's centroid embedding in the
batch, and applies a per-embedding softmax or contrast loss over that matrix. This means a single
GE2E training step implicitly covers a very large number of equivalent TE2E tuple comparisons — the
paper derives a lower bound of `2(N-1)` TE2E-equivalent tuple updates per GE2E step — which is the
core reason GE2E both converges faster and continually emphasizes the hardest (most confusable)
negative speaker at every step, rather than relying on a separately engineered hard-example-mining
procedure.

The paper also introduces "MultiReader," a simple multi-domain training technique that linearly
combines the losses from multiple (possibly very unbalanced) datasets with per-dataset weights,
functioning as a data-driven regularizer instead of an explicit weight-norm penalty. This is used to
train a single production text-dependent SV model that generalizes across both the "OK Google" and
"Hey Google" wake-word datasets, which differ in size by two orders of magnitude.

Experimentally, GE2E is shown to reduce equal error rate (EER) by more than 10% relative to TE2E
while training roughly 3x faster (the paper states both "60% less training time" and, in the TD-SV
section, "about 3x faster" for TI-SV), on production-scale Google speaker-verification datasets
(hundreds of millions of utterances, tens of thousands to hundreds of thousands of speakers). The
significance for the field is that GE2E's batch-similarity-matrix formulation, feature extraction
pipeline (log-mel-filterbank energies, LSTM d-vector encoder), and cosine-similarity evaluation
protocol became a de facto standard reference implementation for computing speaker similarity in
downstream TTS/voice-cloning work, which is why "GE2E cosine similarity" is used as shorthand for
"speaker-embedding cosine similarity computed the way this paper defines it" in much subsequent
literature.

## Architecture, Models and Methods

* **Feature extraction**: raw audio is framed at 25 ms width with a 10 ms step; 40-dimensional
  log-mel-filterbank energies are extracted per frame.
* **Encoder architecture**: a multi-layer LSTM with a linear projection layer on top of the final
  LSTM layer's output at each timestep; the network output is L2-normalized to produce the embedding
  ("d-vector") `e = f(x; w) / ||f(x; w)||_2`.
* **Production model sizes**: for text-dependent SV (TD-SV), the LSTM uses 128 hidden nodes with a
  64-dimensional projection (embedding size 64); for text-independent SV (TI-SV), the LSTM uses 768
  hidden nodes with a 256-dimensional projection (embedding size 256). Both use a 3-layer LSTM with
  projection ("LSTMP").
* **Batch construction**: each training batch contains `N = 64` speakers and `M = 10` utterances per
  speaker (`N x M` total utterances per batch) in the production configuration.
* **Similarity matrix**: for every embedding `e_ji` (speaker `j`, utterance `i`) and every speaker
  centroid `c_k` (mean of that speaker's embeddings in the batch, computed with the query utterance
  excluded from its own speaker's centroid to avoid a trivial solution — Eq. 8), the scaled cosine
  similarity `S_ji,k = w * cos(e_ji, c_k) + b` is computed, with learnable scalars `w` (constrained
  positive) and `b`.
* **Two loss variants**: (1) softmax loss, a cross-entropy-style loss over the row of the similarity
  matrix that pushes `e_ji` toward its own centroid and away from all others; (2) contrast loss,
  which combines a positive term `1 - sigmoid(S_ji,j)` with a hard-negative term
  `max_{k != j} sigmoid(S_ji,k)`, focusing only on the single most confusable negative speaker per
  step. Contrast loss performed better for TD-SV; softmax loss performed slightly better for TI-SV.
* **Optimization**: SGD with initial learning rate 0.01, halved every 30M steps; gradient L2-norm
  clipped at 3; gradient scale for the LSTM projection node set to 0.5; loss-scale parameters
  `(w, b)` initialized to `(10, -5)` with a reduced gradient scale of 0.01 applied to them for
  stable convergence.
* **TD-SV data**: fixed-length ~800 ms segments gated by a keyword-detection system; training sets
  include an "OK Google" set of ~150M utterances / ~630K speakers and a smaller manually collected
  "OK/Hey Google" set of ~1.2M utterances / ~18K speakers (125x and 35x smaller respectively).
  Evaluation: 665 speakers, average 4.5 enrollment and 10 evaluation utterances per speaker (for the
  MultiReader comparison), and a larger evaluation of ~83K speakers with 7.3 enrollment / 5.0
  evaluation utterances per speaker on average (for the loss-function comparison in Table 2).
  Baseline architecture for comparison: single-layer 512-node LSTM with 128-dim embeddings (from the
  prior TE2E paper).
* **TI-SV data and inference**: training uses partial utterances of randomly chosen length in
  `[140, 180]` frames per batch (all partial utterances within one batch share the same length); at
  inference, a sliding window of length `(140+180)/2 = 160` frames with 50% overlap is applied,
  window-level d-vectors are L2-normalized then averaged to form the final utterance embedding.
  Training set: ~36M utterances from 18K speakers; evaluation set: 1,000 speakers with an average of
  6.3 enrollment and 7.2 evaluation utterances per speaker.
* **Evaluation metric**: Equal Error Rate (EER) is the sole reported metric throughout; no other
  metrics (accuracy, minDCF, etc.) are reported.
* **Compute/hardware**: no GPU/TPU model, core count, or wall-clock training time in hours is
  reported; only relative training-time reductions (percentages / multiples) are given.

## Results

* GE2E vs. TE2E (production-scale, larger TD-SV evaluation, Table 2, MultiReader on): GE2E reaches
  **2.38% EER** vs. TE2E's **2.67% EER** with the same 3-layer LSTM (64-dim embedding) architecture
  — about **10%** relative improvement, consistent with the abstract's claim of "more than 10%" EER
  reduction.
* GE2E without MultiReader (Table 2, TD-SV): **3.10% EER**, vs. TE2E without MultiReader at **3.55%
  EER**, vs. the older single-layer 512-node/128-dim TE2E baseline at **3.30% EER**.
* GE2E training speed: the paper reports the GE2E model "took about **60% less training time** than
  TE2E" (TD-SV section) and, separately, that GE2E training "was about **3x faster**" than other
  loss functions in the TI-SV experiments.
* TI-SV loss-function comparison (Table 3, single set of numbers, no MultiReader ablation given):
  Softmax-trained baseline **4.06% EER**, TE2E **4.13% EER**, GE2E **3.55% EER** — the lowest EER of
  the three.
* MultiReader ablation (Table 1, TD-SV, four enroll/verify keyword combinations): "OK Google" to "OK
  Google" improves from **1.16% EER** (mixed data) to **0.82% EER** (MultiReader); "OK Google" to
  "Hey Google" improves from **4.47%** to **2.99%**; "Hey Google" to "OK Google" improves from
  **3.30%** to **2.30%**; "Hey Google" to "Hey Google" improves from **1.69%** to **1.15%** — the
  paper characterizes this as "around **30%** relative improvement on all four cases."
* Theoretical bound: each GE2E update step for a given utterance is shown to be equivalent to at
  least **2(N-1)** TE2E tuple-loss updates for the same utterance (Eq. 11), which the authors use to
  explain the observed convergence-speed gap.

## Innovations

### Generalized End-to-End (GE2E) Loss

Replaces TE2E's single-tuple-at-a-time training with a full similarity-matrix formulation computed
over an entire `N`-speaker x `M`-utterance batch in one forward pass. Every embedding is compared
against every speaker centroid in the batch simultaneously, which implicitly incorporates a very
large number of positive/negative tuple comparisons per gradient step without needing a separate
example-selection or hard-negative-mining stage, unlike TE2E.

### Two interchangeable batch losses: softmax and contrast

The paper offers two ways to turn the similarity matrix into a scalar loss per embedding — a softmax
cross-entropy-style loss (Eq. 6) and a contrast loss focused on the single hardest negative centroid
(Eq. 7) — and empirically shows their relative merits differ by task (TD-SV favors contrast, TI-SV
favors softmax), rather than prescribing one loss for all speaker-verification settings.

### Centroid-exclusion trick for training stability

Excluding the query utterance's own embedding when computing its speaker's centroid (Eq. 8, the
"(-i)" superscript notation) is identified as necessary to avoid a trivial degenerate solution and
to stabilize training — a small but important implementation detail for anyone reproducing
GE2E-style embedding training.

### MultiReader multi-domain training

A generalization of L2 weight regularization that instead regularizes by requiring the model to also
perform well on a second (or `K`-way combination of) data source(s), each weighted by a scalar
`alpha_k`. This lets a single embedding model be trained jointly on wildly unbalanced datasets (a
125x size gap in the paper's case) without the larger dataset drowning out the smaller one, and
without needing to manually balance sampling ratios.

## Datasets

* **"OK Google" TD-SV training set**: ~150M utterances from ~630K speakers, anonymized user query
  logs. Not publicly released (internal Google production data).
* **"OK/Hey Google" TD-SV training set**: ~1.2M utterances from ~18K speakers, manually collected.
  Not publicly released.
* **TD-SV evaluation set (Table 1, MultiReader ablation)**: 665 speakers, manually collected,
  average 4.5 enrollment and 10 evaluation utterances per speaker. Not publicly released.
* **TD-SV evaluation set (Table 2, loss comparison)**: ~83K speakers, from both anonymized logs and
  manual collection, average 7.3 enrollment and 5 evaluation utterances per speaker. Not publicly
  released.
* **TI-SV training set**: ~36M utterances from 18K speakers, extracted from anonymized logs. Not
  publicly released.
* **TI-SV evaluation set**: 1,000 speakers, average 6.3 enrollment and 7.2 evaluation utterances per
  speaker. Not publicly released.
* All datasets are proprietary Google-internal collections; none are publicly available or licensed
  for external use. No dataset names, download links, or license terms are given in the paper.

## Main Ideas

* This paper defines the exact mechanism ("GE2E cosine similarity") that the project's own
  `speaker_sim` benchmark metric is named after: an LSTM d-vector encoder trained with a
  batch-similarity-matrix loss, embeddings L2-normalized, and speaker similarity scored via cosine
  similarity between an evaluation embedding and an enrollment-set centroid. Any GE2E-cosine speaker
  encoder used to compute `speaker_sim` against the 11labs David reference clips should ideally
  follow this feature pipeline (log-mel-filterbank input, LSTM/d-vector embedding, L2 normalization,
  cosine similarity) or clearly document where it deviates, since the metric's interpretability
  depends on matching this reference definition.
* The centroid-based evaluation protocol (average multiple enrollment-utterance embeddings into one
  centroid, then compare a test utterance's embedding to that centroid via cosine similarity) is the
  correct reference procedure for computing `speaker_sim` against a multi-clip reference set like
  `data/11labs_david/` (1358 reference clips) rather than, e.g., averaging pairwise cosine
  similarities against every individual reference clip — the two are not equivalent when embeddings
  are not perfectly linear.
* The MultiReader technique (weighted multi-dataset joint training) is directly relevant if this
  project ever needs to fine-tune a speaker encoder or a voice-cloning model on a small, David-voice
  specific dataset (1557 clips) while also leveraging a larger, more generic corpus — it offers a
  documented alternative to naive dataset mixing or plain weight-decay regularization for avoiding
  overfitting on the smaller set.
* Because GE2E's stated evaluation metric is EER, not raw cosine similarity, this project should be
  explicit that its ≥0.85 GE2E-cosine threshold is a similarity-score criterion inspired by, but not
  identical to, the paper's own EER-based evaluation methodology — the paper does not report or
  calibrate any fixed cosine-similarity threshold value itself.

## Summary

This paper addresses the problem of training neural speaker-verification (SV) embedding models more
efficiently and accurately than the authors' own prior tuple-based end-to-end (TE2E) approach. The
central research question is how to design a loss function that, within each training step,
automatically emphasizes the hardest-to-verify examples without an explicit tuple-selection or
hard-negative-mining stage, while remaining applicable to both text-dependent (TD-SV, e.g. wake-word
verification) and text-independent (TI-SV) speaker verification.

The method, generalized end-to-end (GE2E) loss, processes a full batch of `N` speakers by `M`
utterances at once, computes a similarity matrix between every embedding and every speaker's
centroid (with the query utterance's own embedding excluded from its speaker's centroid to prevent a
trivial solution), and applies either a softmax or contrast-based loss over that matrix. Both
variants are shown to implicitly cover a very large number of TE2E-equivalent tuple comparisons per
step (at least `2(N-1)`), which is offered as the theoretical explanation for GE2E's faster
convergence. The authors also introduce MultiReader, a weighted multi-dataset joint-training
technique for combining unbalanced data sources without one dominating the other.

Empirically, GE2E reduces equal error rate (EER) by roughly 10% relative to TE2E on production-scale
TD-SV data (2.38% vs. 2.67% EER with MultiReader) while cutting training time by 60-70% (stated as
"60% less" in one section and "about 3x faster" in another), and it also outperforms both softmax
and TE2E baselines on TI-SV (3.55% vs. 4.06% and 4.13% EER respectively). MultiReader alone yields
roughly 30% relative EER improvement across four TD-SV enroll/verify keyword combinations compared
to naively mixing the same two datasets. All experiments use proprietary Google-internal datasets
(hundreds of millions to tens of millions of utterances) and are evaluated exclusively via EER; no
public benchmark or dataset is used.

For this project, the paper matters because it is the literal origin of "GE2E cosine similarity" as
a speaker-verification concept, and the project's primary quality metric, `speaker_sim`, is a GE2E
cosine similarity computed against the ElevenLabs David reference clips. Understanding the paper's
feature pipeline (log-mel-filterbank energies into an LSTM d-vector encoder), centroid-based
enrollment/evaluation protocol, and the fact that the paper reports EER rather than a raw cosine
threshold is important context for correctly interpreting, reproducing, or auditing the ≥0.85
GE2E-cosine similarity success criterion defined in `project/description.md`. It also surfaces the
MultiReader technique as a candidate approach if future work needs to combine the small David-voice
dataset with a larger generic corpus during fine-tuning or speaker-encoder training.
