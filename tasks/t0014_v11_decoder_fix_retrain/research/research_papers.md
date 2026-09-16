---
spec_version: "1"
task_id: "t0014_v11_decoder_fix_retrain"
research_stage: "papers"
papers_reviewed: 3
papers_cited: 3
categories_consulted: []
date_completed: "2026-09-16"
status: "complete"
---
## Task Objective

t0014 fixes the `ignore_modules` bug in `train_second_v10.py:load_checkpoint()` that t0013 traced as
the root cause of `kokoro-v10-best` producing 75-81%-clipped, DC-saturated noise instead of speech:
`first_stage_v3.pth`'s decoder is istftnet-shaped while `config_david_v10.yml` builds a
`hifigan`-shaped decoder, and the loader's `ignore_modules` list omits `"decoder"`, so the HiFi-GAN
vocoder's actual generating layers entered Stage 2 training via a partial, architecture-mismatched
load — a state a falsification probe showed is actively worse than pure random init. This task
retrains with the fix applied, on t0012's larger 1,531-clip normalized corpus, and gates its own
completion on an audible-speech check (`audio_quality_check.py`'s `is_likely_noise` heuristic)
rather than on `val_loss`. This research file addresses this task's Key Question 2 specifically: "is
~20 epochs on 1531 clips a realistic budget for a HiFi-GAN decoder to converge from random init?
Check t0009's/general StyleTTS2 literature for typical vocoder-from-scratch convergence epoch
counts... if 20 is clearly insufficient, say so in planning and propose a realistic epoch count." It
also informs Key Question 1 (whether a pretrained hifigan-shaped first-stage checkpoint exists to
fine-tune from instead of training from scratch) and Key Question 3 (verifying the fix actually
leaves `decoder` at genuine random init or genuine pretrained load, not another silent partial
match).

## Category Selection Rationale

`meta/categories/` is currently empty for this entire project —
`uv run python -u -m arf.scripts.aggregators.aggregate_categories --format json` returns
`{"categories": []}` with zero entries, confirmed before and after this task added new paper assets.
No prior task in this project (including t0013, whose own `research_summary.md` explicitly records
"Key Papers ... (not generated -- research-papers step skipped)") has ever populated
`meta/categories/` or run the research-papers step. `categories_consulted` is therefore `[]` in this
file's frontmatter, and every paper asset added by this task carries `categories: []` in its
`details.json` per the paper asset specification's instruction to use an empty list rather than
invent slugs when no category exists in the project.

This is a genuine project-level gap, not a scoping decision: `meta/asset_types/paper/aggregator.py`
also confirmed **zero paper assets existed anywhere in the project** before this task ran
(`uv run python -u -m arf.scripts.aggregators.aggregate_papers --format json --detail short`
returned `"paper_count": 0` both with and without category filters), despite
`project/description.md` naming "StyleTTS2 paper: Hu et al. (2023)" as a Key Reference since the
project's inception. Because this task's own Key Question 2 cannot be answered without literature on
HiFi-GAN-family vocoder convergence, and because the research-papers step exists specifically to
review papers already in the project corpus, this task first added the three papers most directly
implicated by t0013's architectural diagnosis (StyleTTS 2 — the exact architecture and training
recipe this project's `kokoro-finetune` implements; HiFi-GAN — the decoder architecture
`config_david_v10.yml` selects; iSTFTNet — the architecture `first_stage_v3.pth`'s decoder was
actually shaped like) via the `add-paper` skill, verified each with `verify_paper_asset.py` (all
three pass with zero errors/ warnings), and only then proceeded with this review. No category-based
filtering was applicable since none exist; all three papers were selected by direct relevance to the
task's architectural and training-budget questions, confirmed by reading `project/description.md`,
`task_description.md`, and t0013's `results/v10_diagnosis.md` and `research/research_summary.md`
before selecting them.

## Key Findings

### HiFi-GAN-family decoders need step budgets orders of magnitude beyond a 20-epoch schedule

Both vocoder-architecture papers in this review report explicit from-scratch training budgets on
single-speaker corpora comparable in size to this project's 1,531-clip David corpus. The original
HiFi-GAN paper trains its main-comparison models (Table 1, LJSpeech and VCTK, V1/V2/V3 generators)
to **2.5M optimizer steps**, and even its smaller ablation-study configurations to **500k steps**
each [Kong2020]. iSTFTNet, which modifies HiFi-GAN's output-side layers but keeps its core
upsampling/ResBlock structure, independently reports the identical **2.5M-iteration** from-scratch
budget on a 12,600-clip LJSpeech training split, using the same optimizer and loss configuration as
[Kong2020] [Kaneko2022]. Neither paper reports step counts on a corpus close to 1,531 clips, so the
comparison is necessarily budget-vs-corpus-size rather than budget-vs-identical-corpus, but the
scale gap is stark: v10's `epochs_2nd: 20` config, with `joint_epoch: 8` (meaning only 12 of the 20
epochs include adversarial decoder training at all), on a corpus roughly 1/9th to 1/28th the size of
the papers' training sets, produced a decoder-optimizer-step count many orders of magnitude below
either paper's reported budget. This is independent corroboration, from two separate papers with
different authors and slightly different architectures, that a HiFi-GAN-family decoder trained from
random initialization needs a training budget several orders of magnitude larger than what v10
provided — consistent with, though not sufficient on its own to fully explain (see Gaps section),
t0013's finding that v10 never produced valid audio at any observed epoch.

### StyleTTS2's own published recipe also exceeds v10's epoch budget

[Li2023] is more directly comparable than the vocoder-only papers because it reports epoch counts
(not raw optimizer steps) for the exact two-stage StyleTTS2 training procedure this project's
`train_second_v10.py`/`config_david_v10.yml` implements, including the choice between HifiGAN- and
iSTFTNet-based decoders as an explicit, documented configuration switch [Li2023, p. 4, p. 24]. The
paper's own HifiGAN-decoder configurations (used for its VCTK and LibriTTS experiments — LJSpeech
used the iSTFTNet decoder instead) were trained for **50 pre-training epochs + 40 joint-training
epochs** (VCTK, ~44,000 clips / 109 speakers) and **30 pre-training epochs + 25 joint-training
epochs** (LibriTTS, ~245 hours / 1,151 speakers) [Li2023, Section 4.1, p. 6-7]. Both figures are
well above v10/v11's `epochs_2nd: 20` / `joint_epoch: 8` configuration, even before accounting for
VCTK's ~28x and LibriTTS's much larger corpus size relative to this project's 1,531-clip corpus. The
paper does not report a HifiGAN-decoder configuration on a corpus close to this project's scale, so
no published StyleTTS2 recipe directly validates a 20-epoch budget as sufficient; every reported
HifiGAN-decoder configuration in the paper uses more joint-training epochs than v10/v11's total
epoch count.

### Skipping pretraining means more epochs are needed, not fewer

[Li2023] explicitly states that acoustic-module pre-training, while it can "accelerate the training
process," is "not an absolute necessity: despite being slower, starting joint training directly from
scratch also leads to model convergence" [Li2023, p. 4]. This is directly relevant to Key Question
3: if the corrected `load_checkpoint()` leaves `decoder` at genuine `build_model()` random init
(rather than the Frankenstein partial-match state t0013's falsification probe showed is worse than
random), the paper's own claim supports that convergence remains achievable — but the same sentence
frames this as a slower path, not a faster or equal-epoch one. Neither [Li2023] nor
[Kong2020]/[Kaneko2022] report a case where a HiFi-GAN-family decoder was trained from scratch in
fewer epochs/steps than a fine-tuned equivalent; every reported from-scratch budget in this review
is larger than the corresponding fine-tuning budget, not smaller (see next finding).

### Fine-tuning from a pretrained checkpoint uses a fraction of the from-scratch step budget

[Kaneko2022] provides a directly quantifiable fine-tuning-vs-from-scratch ratio within a single
paper: vocoder-only models are trained from scratch for **2.5M iterations**, while the end-to-end
TTS+vocoder fine-tuning experiment (Conformer-FastSpeech2 + iSTFTNet, starting from independently
pretrained components) uses only **300k iterations** — roughly **1/8th** the from-scratch budget
[Kaneko2022, p. 3-4]. This is the paper's own internal evidence for Key Question 1's framing: if a
genuinely hifigan-shaped (or architecturally-compatible) pretrained checkpoint is available, fine-
tuning from it is a substantially cheaper path to a working decoder than training one from
initialization, consistent with why t0013's diagnosis treats the "source a real hifigan-shaped
pretrained checkpoint" branch as preferable to random-init retraining when available. None of the
three papers report whether an istftnet-shaped checkpoint can be usefully fine-tuned into a
hifigan-shaped model (the reverse direction implicated in t0013's bug) — this is a gap, addressed
below.

### Why HiFi-GAN and iSTFTNet share enough shape to pass a partial-match load

[Kaneko2022] is explicit that iSTFTNet is a targeted modification of existing HiFi-GAN variants, not
an independent architecture: "we applied our ideas to three HiFi-GAN variants," replacing only "some
output-side layers" — the final upsampling/output convolutions — with a magnitude/phase prediction
head plus inverse STFT, while retaining the earlier input convolution and upsampling/ResBlock stages
unchanged from the corresponding HiFi-GAN variant [Kaneko2022, p. 1-2, Fig. 1, Fig. 3].
`config_david_v10.yml`'s decoder configuration (`resblock_kernel_sizes: [3, 7, 11]`,
`upsample_initial_channel: 512`) matches [Kong2020]'s HiFi-GAN V1 configuration exactly except for
`upsample_rates`/`upsample_kernel_sizes`, which are adapted from the paper's 256-sample hop length
to this project's 300-sample hop length [Kong2020, p. 5, Appendix A]. Because iSTFTNet shares
early-layer shapes with its HiFi-GAN parent by design, a checkpoint built from the iSTFTNet-style
`first_stage_v3.pth` and a target `hifigan`-shaped `decoder` module can share enough tensor shapes
in early layers to pass a loader's "any match found" check while the true vocoder-generating output
layers (`ups`, `resblocks`/`alphas`, `conv_post`, `noise_convs` in HiFi-GAN's own naming versus
iSTFTNet's magnitude/phase/iSTFT head) remain architecturally incompatible — exactly the partial-
match failure mode t0013's tensor forensics documented at the code level. This is the mechanistic,
paper-level explanation for why the bug was silent rather than a hard failure.

## Methodology Insights

* **Do not accept a 20-epoch budget as sufficient for a from-scratch HiFi-GAN decoder without
  explicit justification.** Both [Kong2020] (2.5M steps main / 500k steps ablation) and [Kaneko2022]
  (2.5M iterations, same architecture family) report from-scratch budgets far beyond what 20 epochs
  on ~1,531 clips implies. If v11 retrains `decoder` from genuine random init (Key Question 1's
  no-pretrained-checkpoint branch), the plan should either substantially raise the epoch budget
  beyond v10's 20, or explicitly document — with evidence, not assumption — why this project's setup
  (much smaller corpus, StyleTTS2's non-adversarial-loss components carrying some of the
  reconstruction burden, `joint_epoch: 8` starting adversarial training partway through) makes a
  shorter schedule viable.
* **Use [Li2023]'s own HifiGAN-decoder epoch counts (50 pretrain + 40 joint on VCTK; 30 pretrain +
  25 joint on LibriTTS) as the closest single published anchor** for a realistic joint-training
  epoch count, since it is the only reviewed paper reporting epoch (not step) counts for this
  project's exact architecture and two-stage recipe. Neither VCTK nor LibriTTS is
  corpus-size-matched to this project (43,470 and ~245h-of-audio respectively, versus 1,531 clips),
  so treat these as an upper anchor for a full from-scratch schedule, not a corpus-size-adjusted
  target — extrapolating a precise epoch count for a ~30x-smaller corpus is not supported by any
  number in these three papers and should not be presented as a paper-derived figure.
* **Before spending any GPU time, verify Key Question 3 as planned**: confirm the fixed
  `load_checkpoint()` leaves `decoder` at pure `build_model()` random init (or a genuine
  architecture-matched pretrained load), not another partial match. [Kaneko2022]'s description of
  exactly which HiFi-GAN-family layers are shape-compatible across variants (p. 1-2, Fig. 1) is a
  useful checklist for what "genuinely random" tensor inspection should look like: early
  input-convolution and upsampling-stage shapes may coincidentally match even between
  architecturally distinct variants, so shape-matching alone is not proof of a correct load —
  cross-reference against known module names (`ups.*`, `resblocks.*`, `alphas.*`, `conv_post` for
  HiFi-GAN) as t0013's `inspect_checkpoint.py` already does.
* **If Key Question 1 locates a genuinely hifigan-shaped pretrained checkpoint**, prefer fine-tuning
  from it over random-init retraining: [Kaneko2022]'s internal ratio (300k fine-tuning iterations
  versus 2.5M from-scratch iterations, roughly 1/8th) is concrete evidence that fine-tuning from a
  working vocoder converges in a small fraction of the from-scratch budget, making it the
  substantially cheaper and lower-risk path if available.
* **MPD is disproportionately important to decoder output quality** [Kong2020]: removing the
  Multi-Period Discriminator alone costs 1.82 MOS points in ablation (4.10 -> 2.28), far more than
  MSD removal (-0.36), MRF removal (-0.18), or mel-loss removal (-0.85) [Kong2020, Table 2, p. 7].
  If v11's retrained decoder still fails the audible-speech gate after the `ignore_modules` fix and
  a larger epoch budget, checking whether MPD is correctly wired and contributing to the loss should
  be an early diagnostic step, not just assuming under-training.

## Gaps and Limitations

* **No reviewed paper reports a from-scratch HiFi-GAN-family training budget on a corpus close to
  1,531 clips.** The smallest reviewed corpus is LJSpeech's 12,600-13,100 training clips
  ([Kong2020], [Kaneko2022]); [Li2023]'s smallest HifiGAN-decoder configuration (LibriTTS) trains on
  ~245 hours across 1,151 speakers, structurally different from this project's single-speaker,
  1,531-clip setup. Extrapolating a specific epoch count for this project's exact corpus size from
  these papers is not supported by any single reported number — any epoch recommendation for v11
  must be presented as an order-of-magnitude inference from these budgets, not a literature-derived
  precise value.
* **None of the three papers report training-from-scratch behavior on a corpus this small at all**,
  so it is unknown from this literature whether a HiFi-GAN decoder on 1,531 clips would converge
  faster (fewer unique examples to cover) or slower (fewer gradient updates per epoch, higher
  overfitting risk) than the papers' much larger corpora would predict per-epoch. This is a genuine
  open question the literature does not resolve, and should be flagged as such in planning rather
  than resolved by assumption.
* **No paper addresses fine-tuning in the reverse direction implicated by t0013's bug** — loading an
  iSTFTNet-shaped checkpoint into a HiFi-GAN-shaped target (or vice versa) as a deliberate
  cross-architecture initialization strategy, rather than as an accidental partial match. If Key
  Question 1 considers deliberately exploiting the shared early-layer shapes [Kaneko2022] documents
  (e.g., initializing only the shared prefix and randomizing the divergent suffix), that would be a
  novel strategy not validated by any of the reviewed literature.
* **None of the three papers report GAN training stability diagnostics** (gradient norms, skip
  counts, discriminator/generator loss ratios) in a form directly reusable for this project's
  `t0009_training_safeguards` health-gate library. The ablation and comparison tables report only
  converged-model MOS, not training dynamics over the course of training, so these papers cannot
  inform what a healthy versus unhealthy mid-training trajectory looks like for this project's
  safeguards.
* **This project's paper corpus was empty before this task** — zero categories and zero papers
  existed anywhere in the project prior to this research step, despite `project/description.md`
  naming the StyleTTS2 paper as a Key Reference since project inception. This gap likely affected
  every prior task that would have benefited from literature grounding (t0001-t0013), not just
  t0014; it is noted here as a process gap worth a follow-up suggestion, not something this task's
  scope can retroactively fix for earlier tasks.

## Recommendations for This Task

1. **Do not treat v10's `epochs_2nd: 20` / `joint_epoch: 8` budget as self-evidently adequate for a
   from-scratch HiFi-GAN decoder.** Both [Kong2020] and [Kaneko2022] report from-scratch budgets
   (2.5M steps/iterations) far beyond what this implies, and even [Li2023]'s own HifiGAN-decoder
   epoch counts (50+40 on VCTK, 30+25 on LibriTTS) exceed v10/v11's total. Planning should either
   substantially increase the epoch budget or explicitly justify keeping it short.
2. **Complete Key Question 1's search for a genuine pretrained hifigan-shaped checkpoint before
   committing to random-init retraining.** [Kaneko2022]'s internal ~8x ratio (300k fine-tune vs.
   2.5M from-scratch) is concrete evidence that fine-tuning is dramatically cheaper than training
   from scratch, when a suitable starting checkpoint exists.
3. **When verifying Key Question 3's tensor-level fix**, check specific HiFi-GAN module names
   (`ups.*`, `resblocks.*`, `alphas.*`, `conv_post`) rather than relying on shape-count matching
   alone — [Kaneko2022]'s description of shared early-layer shapes across the HiFi-GAN/iSTFTNet
   family shows shape overlap alone does not prove a correct, architecture-matched load.
4. **If the retrained decoder still fails the audible-speech gate after the fix and a larger epoch
   budget**, check MPD wiring/loss weighting before assuming pure under-training — [Kong2020]'s
   ablation shows MPD removal alone costs more MOS than any other single component (1.82 points).
5. **File a follow-up suggestion for the project's empty `meta/categories/` and paper corpus**,
   since this gap predates t0014 and likely affected every prior task's research quality, not just
   this one's.
6. **Present any epoch-count recommendation for v11 as an order-of-magnitude judgment call**, not a
   precise literature-derived number — no reviewed paper reports a from-scratch HiFi-GAN-family
   budget on a corpus this project's size (1,531 clips), so the actual planning number should
   combine this literature's budgets with this project's own compute/cost constraints (per
   `task_description.md`'s Compute and Budget section), not literature numbers alone.

## Paper Index

### [Li2023]

* **Title**: StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial
  Training with Large Speech Language Models
* **Authors**: Li, Y. A., Han, C., Raghavan, V. S., Mischler, G., Mesgarani, N.
* **Year**: 2023
* **DOI**: `10.48550/arXiv.2306.07691`
* **Asset**: `assets/paper/10.48550_arXiv.2306.07691/`
* **Categories**: none (project has no categories defined in `meta/categories/`)
* **Relevance**: The exact architecture and two-stage training recipe this project's
  `kokoro-finetune` pipeline implements, including the HifiGAN-vs-iSTFTNet decoder choice
  `config_david_v10.yml` exercises; its own HifiGAN-decoder epoch counts are the most directly
  comparable published data point for Key Question 2.

### [Kong2020]

* **Title**: HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech
  Synthesis
* **Authors**: Kong, J., Kim, J., Bae, J.
* **Year**: 2020
* **DOI**: `10.48550/arXiv.2010.05646`
* **Asset**: `assets/paper/10.48550_arXiv.2010.05646/`
* **Categories**: none (project has no categories defined in `meta/categories/`)
* **Relevance**: Origin paper for `config_david_v10.yml`'s `decoder.type: hifigan` architecture
  (matches HiFi-GAN V1's configuration field-for-field except hop-length-adapted upsample rates);
  reports the 2.5M-step/500k-step from-scratch training budgets central to Key Question 2, and the
  MPD ablation relevant to post-fix debugging if the retrain still fails the audible-speech gate.

### [Kaneko2022]

* **Title**: iSTFTNet: Fast and Lightweight Mel-Spectrogram Vocoder Incorporating Inverse Short-Time
  Fourier Transform
* **Authors**: Kaneko, T., Tanaka, K., Kameoka, H., Seki, S.
* **Year**: 2022
* **DOI**: `10.48550/arXiv.2203.02395`
* **Asset**: `assets/paper/10.48550_arXiv.2203.02395/`
* **Categories**: none (project has no categories defined in `meta/categories/`)
* **Relevance**: Describes the exact architecture family `first_stage_v3.pth`'s decoder belongs to
  (an iSTFTNet-modified HiFi-GAN variant), explaining at the paper level why a partial shape match
  with a `hifigan`-shaped target was possible; independently corroborates [Kong2020]'s 2.5M-step
  from-scratch budget and provides a concrete fine-tuning-vs-from-scratch ratio (300k vs. 2.5M)
  relevant to Key Question 1.
