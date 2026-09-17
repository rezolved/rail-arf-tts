---
spec_version: "1"
task_id: "t0014_v11_decoder_fix_retrain"
research_stage: "internet"
searches_conducted: 12
sources_cited: 17
papers_discovered: 1
date_completed: "2026-09-16"
status: "complete"
---
## Task Objective

t0014 fixes the `ignore_modules` bug that left `kokoro-v10-best`'s HiFi-GAN decoder worse than
random init (t0013: `first_stage_v3.pth` is istftnet-shaped while `config_david_v10.yml` builds a
`hifigan`-shaped decoder), retrains on t0012's 1,531-clip normalized corpus, and gates completion on
an audible-speech check. This step's assignment is Key Question 1: find a genuinely `hifigan`-shaped
pretrained StyleTTS2 first-stage checkpoint (original authors, `semidark/StyleTTS2`, or elsewhere)
before the plan commits to random-init decoder training, and secondarily gather internet guidance on
realistic epoch counts at this project's ~1,531-clip scale (Key Question 2).

## Gaps Addressed

From `research_papers.md` Gaps and Limitations:

1. **No paper reports a from-scratch HiFi-GAN-family budget near 1,531 clips.** — **Partially
   resolved**: community StyleTTS2 fine-tunes at a comparable scale (~1k samples) converge in 50
   epochs starting from a pretrained decoder [StyleTTS2-Disc65], [StyleTTS2-ConfigFT-GH] — not a
   from-scratch number, but supports preferring fine-tuning over training from scratch at this
   scale.
2. **None of the three papers report from-scratch behavior on a corpus this small.** — **Partially
   resolved**: Stage 1 from-scratch training on ~150 files needed 400-1,000 epochs to stabilize
   validation loss [StyleTTS2-Disc144] — order-of-magnitude confirmation 20 epochs is short, but
   this is Stage 1 acoustic data, not decoder-specific.
3. **No paper addresses cross-architecture initialization (the reverse-direction bug).** —
   **Unresolved, but reframed**: upstream `train_second.py`'s own default `ignore_modules` also
   omits `"decoder"` [StyleTTS2-TrainSecond-GH] — nobody needed a deliberate cross-architecture
   trick because upstream assumes `first_stage_path`'s decoder always matches `config.yml`'s
   `decoder.type`.
4. **No paper reports GAN training-stability diagnostics reusable for t0009's safeguards.** —
   **Partially resolved**: a discriminator-warmup (10k steps) and a feature-matching-loss-99%-pause
   rule were found [GAN-Vocoder-MRD-2021], usable though not HiFi-GAN-specific.
5. **The project's paper corpus was empty before this task.** — **Unresolved by design**: a
   project-process gap internet research about vocoder architectures cannot fix; unchanged from
   `research_papers.md`'s framing as a follow-up-suggestion item.

## Search Strategy

**Sources**: GitHub (code, READMEs, `Configs/*.yml`, Discussions, Issues), Hugging Face (model
repos/ cards), arXiv, DeepWiki, general web search.

**Queries executed (12, verbatim)**:

1. `StyleTTS2 pretrained first_stage checkpoint hifigan decoder download`
2. `semidark/StyleTTS2 github pretrained checkpoint`
3. `StyleTTS2 Hugging Face pretrained model LJSpeech LibriTTS checkpoint download`
4. `yl4579 StyleTTS2 github pretrained models release`
5. `yl4579 StyleTTS2 LibriTTS config.yml decoder type hifigan istftnet`
6. `StyleTTS2 fine-tune small single speaker dataset "how many epochs" second stage github issue`
7. `"kokoro-finetune" github first_stage_v3.pth StyleTTS2`
8. `GAN vocoder training health diagnostics discriminator generator loss ratio gradient norm best practices`
9. `HiFi-GAN vocoder fine-tune small dataset single speaker epochs convergence "few hundred" OR "1000 samples"`
10. `yl4579 StyleTTS2 license usage terms commercial LibriTTS pretrained model`
11. `StyleTTS2 "load_only_params" ignore_modules decoder github issue partial checkpoint load`
12. `StyleTTS2 discussion 65 "good fine-tune" epochs dataset size examples`

**Date range**: no hard restriction; StyleTTS2 ecosystem sources span 2023-2026. **Inclusion
criteria**: must bear on (a) a `hifigan`-shaped pretrained StyleTTS2 checkpoint compatible with
`config_david_v10.yml`, (b) epoch/step counts at a corpus scale near 1,531 clips, or (c) GAN vocoder
training-health diagnostics. **Excluded**: unrelated TTS architectures, recipes with no
epoch/architecture detail. **Iterations**: queries 1-4 located the two official `yl4579`
checkpoints; query 5 confirmed the LibriTTS checkpoint's decoder config matches
`config_david_v10.yml` field-for-field. Query 2 (the literal task-named "semidark/StyleTTS2")
resolved to `semidark/kikiri-tts`'s bundled submodule; query 7 and a DeepWiki fetch followed up on
its actual patches. Query 11 snowballed from the checkpoint find, fetching `train_second.py` to
check whether upstream's own `ignore_modules` includes `"decoder"` — it does not. 11 pages were
deep-read (within the 12-page budget).

## Key Findings

### A genuinely `hifigan`-shaped pretrained StyleTTS2 checkpoint exists and field-matches this project

`yl4579/StyleTTS2-LibriTTS`'s `Models/LibriTTS/epochs_2nd_00020.pth` + `config.yml` uses
`decoder.type: hifigan`, `resblock_kernel_sizes: [3, 7, 11]`, `upsample_initial_channel: 512`,
`upsample_rates: [10, 5, 3, 2]` (product 300) [StyleTTS2-LibriTTS-HF] — a field-for-field match to
`config_david_v10.yml`'s own decoder block, and `multispeaker: true` like this project. By contrast
`yl4579/StyleTTS2-LJSpeech` uses `decoder.type: istftnet` [StyleTTS2-LJSpeech-HF],
[StyleTTS2-Config-GH] — the same family t0013 diagnosed `first_stage_v3.pth` as. This directly
answers Key Question 1: yes, a suitable pretrained checkpoint exists — the official LibriTTS
release.

### The official fine-tuning recipe fine-tunes this exact checkpoint, not from scratch

`Configs/config_ft.yml` sets `pretrained_model: "Models/LibriTTS/epochs_2nd_00020.pth"`,
`decoder.type: hifigan`, `epochs: 50`, `batch_size: 8`, `max_len: 400`, `diff_epoch: 10`,
`joint_epoch: 30`, targeting LJSpeech-scale data (~1k samples / ~1h) [StyleTTS2-ConfigFT-GH] — the
closest corpus-scale anchor to 1,531 clips found anywhere. Two independent community fine-tunes
("Aurora", 8h; "Chaos", 10h, single voice) both used 50 epochs with joint training from epoch 10
[StyleTTS2-Disc65], corroborating the shipped default as real-world-validated, not just unused
boilerplate.

### Upstream's own `ignore_modules` default also omits `"decoder"` — an architecture-match assumption, not an oversight

`train_second.py`'s `load_checkpoint()` call uses
`ignore_modules=['bert', 'bert_encoder', 'predictor', 'predictor_encoder', 'msd', 'mpd', 'wd', 'diffusion']`
with `load_only_params=True` [StyleTTS2-TrainSecond-GH] — `"decoder"` is absent, exactly like this
project's current bugged state. Upstream loads `decoder` from `first_stage_path` by design, assuming
whatever produced it used the same `decoder.type`. t0013's root cause (istftnet-shaped
`first_stage_v3.pth` vs. hifigan-shaped config) violates an assumption upstream never guards
against. This **updates** `research_papers.md`'s framing (which treated the missing `"decoder"`
entry as *the* bug): the more upstream-consistent fix is pointing `first_stage_path` at a genuinely
hifigan-shaped checkpoint, not adding `"decoder"` to `ignore_modules` and accepting random init.

### "semidark/StyleTTS2" resolves to a patched submodule, not a checkpoint source

It resolves to a patched `StyleTTS2/` submodule inside `semidark/kikiri-tts` (formerly
`kokoro-deutsch`) [kikiri-tts-GH]. Documented patches: `weight_norm`/`spectral_norm` migration for
PyTorch 2.6+, a `.train()`-mode-after-checkpoint-load fix, and a PL-BERT 510-token guard
[semidark-StyleTTS2-DeepWiki] — none overlap with this project's `ignore_modules`/architecture-match
bug, and it ships no pretrained checkpoint of its own.

### Licensing note on the LibriTTS pretrained weights

Code is MIT [StyleTTS2-Issue37], but the README requires informing listeners the model was
initialized from pretrained weights and cloning only consented voices [StyleTTS2-GH]. Since this
project only uses the checkpoint as a decoder initialization for fine-tuning on Rezolve's own
licensed David data (never shipping a LibriTTS speaker identity), this is very likely compatible but
should be noted in `results_summary.md`.

### GAN training-health diagnostics found (partial gap-4 answer)

A discriminator-only warmup for the first 10k steps, and pausing GAN-loss updates when
feature-matching loss exceeds 99% of total loss, are documented, implementable rules
[GAN-Vocoder-MRD-2021] — not HiFi-GAN-specific, but directly usable for t0009's safeguard library.

## Methodology Insights

* **Prefer architecture-matching `first_stage_path` over forced random init.** Upstream's
  `ignore_modules` default [StyleTTS2-TrainSecond-GH] only works when the checkpoint's decoder
  architecture matches the config. Point `first_stage_path` at the LibriTTS checkpoint
  [StyleTTS2-LibriTTS-HF] before falling back to random init — architecturally correct, and per
  [Kaneko2022]'s ~8x fine-tune-vs-scratch ratio, likely far faster to converge.
* **Hypothesis for Key Question 3**: with `first_stage_path` repointed at the LibriTTS checkpoint,
  `decoder.*`/`ups.*`/`resblocks.*`/`conv_post` should show 100% key/shape overlap, not just a
  partial match — stronger positive evidence this is the upstream-intended fix.
* **Epoch anchor if fine-tuning from the LibriTTS checkpoint**: 50 epochs, `diff_epoch: 10`,
  `joint_epoch: 30`, `batch_size: 8`, `max_len: 400` [StyleTTS2-ConfigFT-GH] — validated by two
  community fine-tunes [StyleTTS2-Disc65] — a closer scale-matched anchor than [Li2023]'s own
  VCTK/LibriTTS numbers already in `research_papers.md`.
* **If random init is still chosen** (e.g., licensing objection), [StyleTTS2-Disc144]'s 400-1,000
  from-scratch Stage 1 epochs on ~150 files reinforces that v10's 20-epoch budget is an order of
  magnitude short — directional, not decoder-specific or precise.
* **Practical tip**: `load_only_params=True` with a matched `first_stage_path` carries over weights
  only, not optimizer/epoch state — v11's epoch counter still starts fresh.

## Discovered Papers

### [GAN-Vocoder-MRD-2021]

* **Title**: GAN Vocoder: Multi-Resolution Discriminator Is All You Need
* **Authors**: You, J., Kim, D., Nam, G., Hwang, G., Chae, G.
* **Year**: 2021
* **DOI**: not confirmed (arXiv preprint; venue not verified during this search)
* **URL**: https://arxiv.org/abs/2103.05236
* **Suggested categories**: none (project has no categories defined in `meta/categories/`)
* **Why download**: Fills `research_papers.md`'s gap that no reviewed paper reports GAN
  training-stability diagnostics reusable for t0009's safeguard library (discriminator warmup,
  feature-matching-loss pause rule); also argues multi-resolution discriminators, not architecture
  minutiae, drive GAN vocoder quality — worth checking against this project's MPD/MSD setup if v11
  still underperforms after the fix.

## Recommendations for This Task

1. **Resolve Key Question 1 as: yes, use the pretrained checkpoint.** Point `first_stage_path` at
   `yl4579/StyleTTS2-LibriTTS`'s `epochs_2nd_00020.pth` [StyleTTS2-LibriTTS-HF] instead of
   `first_stage_v3.pth`. This **updates** `research_papers.md`'s Recommendation 1 (which assumed
   random-init retraining might be necessary) — do not default to random init without trying this.
2. **If adopted, budget ~50 epochs with `joint_epoch` around 10-30** [StyleTTS2-ConfigFT-GH],
   [StyleTTS2-Disc65] — not v10's 20/8. This supersedes `research_papers.md`'s VCTK/LibriTTS-scale
   numbers for the fine-tuning-from-pretrained scenario specifically.
3. **Verify Key Question 3 as zero shape mismatches**, not just "no crash" — full key/shape overlap
   on every `decoder.*` submodule is a stronger correctness signal than partial-match detection.
4. **Document the pretrained-weights usage terms** in `results_summary.md` [StyleTTS2-GH],
   [StyleTTS2-Issue37].
5. **Only fall back to random-init retraining if the LibriTTS-checkpoint path fails** (licensing
   objection, or Key Question 3 reveals unexpected mismatches) — then treat [StyleTTS2-Disc144]'s
   400-1,000-epoch numbers as further evidence v10's 20-epoch budget needs to grow by an order of
   magnitude.
6. **Consider adding [GAN-Vocoder-MRD-2021]'s warmup/pause rules to t0009's safeguard library** as
   an early-instability diagnostic, independent of the decoder-init path chosen.

## Source Index

### [Li2023]

* **Type**: paper
* **Title**: StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial
  Training with Large Speech Language Models
* **Authors**: Li, Y. A., Han, C., Raghavan, V. S., Mischler, G., Mesgarani, N.
* **Year**: 2023
* **DOI**: `10.48550/arXiv.2306.07691`
* **URL**: https://arxiv.org/abs/2306.07691
* **Peer-reviewed**: yes (NeurIPS 2023)
* **Relevance**: Already in the project corpus; this step's findings extend its Section 4.1 recipe
  with corpus-scale-matched internet sources.

### [Kaneko2022]

* **Type**: paper
* **Title**: iSTFTNet: Fast and Lightweight Mel-Spectrogram Vocoder Incorporating Inverse Short-Time
  Fourier Transform
* **Authors**: Kaneko, T., Tanaka, K., Kameoka, H., Seki, S.
* **Year**: 2022
* **DOI**: `10.48550/arXiv.2203.02395`
* **URL**: https://arxiv.org/abs/2203.02395
* **Peer-reviewed**: yes (ICASSP 2022)
* **Relevance**: Already in the project corpus; its ~8x fine-tune-vs-scratch step ratio underlies
  the recommendation to prefer fine-tuning the LibriTTS checkpoint over random-init retraining.

### [StyleTTS2-GH]

* **Type**: repository
* **Title**: StyleTTS2 (official implementation)
* **Author/Org**: yl4579
* **Date**: 2023-06
* **URL**: https://github.com/yl4579/StyleTTS2
* **Peer-reviewed**: no (accompanies peer-reviewed [Li2023])
* **Relevance**: Source of pretrained-checkpoint links and the pretrained-weights usage-terms text.

### [StyleTTS2-LibriTTS-HF]

* **Type**: repository
* **Title**: yl4579/StyleTTS2-LibriTTS (pretrained checkpoint)
* **Author/Org**: yl4579
* **Date**: 2023-06
* **URL**: https://huggingface.co/yl4579/StyleTTS2-LibriTTS/tree/main
* **Peer-reviewed**: no
* **Relevance**: The candidate hifigan-shaped checkpoint answering Key Question 1; `config.yml`
  decoder block matches `config_david_v10.yml` field-for-field.

### [StyleTTS2-LJSpeech-HF]

* **Type**: repository
* **Title**: yl4579/StyleTTS2-LJSpeech (pretrained checkpoint)
* **Author/Org**: yl4579
* **Date**: 2023-06
* **URL**: https://huggingface.co/yl4579/StyleTTS2-LJSpeech/tree/main
* **Peer-reviewed**: no
* **Relevance**: Confirms LJSpeech release uses istftnet, not hifigan — the negative case ruling it
  out and corroborating why `first_stage_v3.pth` was the wrong source.

### [StyleTTS2-Config-GH]

* **Type**: repository
* **Title**: `Configs/config.yml` (base training config)
* **Author/Org**: yl4579
* **Date**: 2023-06
* **URL**: https://github.com/yl4579/StyleTTS2/blob/main/Configs/config.yml
* **Peer-reviewed**: no
* **Relevance**: From-scratch LJSpeech config (`istftnet`, `epochs_2nd: 100`, `joint_epoch: 50`) —
  from-scratch anchor vs. v10/v11's 20/8.

### [StyleTTS2-ConfigFT-GH]

* **Type**: repository
* **Title**: `Configs/config_ft.yml` (official fine-tuning config)
* **Author/Org**: yl4579
* **Date**: 2023-06
* **URL**: https://github.com/yl4579/StyleTTS2/blob/main/Configs/config_ft.yml
* **Peer-reviewed**: no
* **Relevance**: Reference recipe for fine-tuning the LibriTTS hifigan checkpoint on a new
  ~1k-sample voice; source of the 50-epoch / `joint_epoch: 30` anchor.

### [StyleTTS2-TrainSecond-GH]

* **Type**: repository
* **Title**: `train_second.py` (Stage 2 training script)
* **Author/Org**: yl4579
* **Date**: 2023-06
* **URL**: https://github.com/yl4579/StyleTTS2/blob/main/train_second.py
* **Peer-reviewed**: no
* **Relevance**: Shows upstream's default `ignore_modules` also omits `"decoder"`, reframing t0013's
  bug as an architecture-mismatch assumption violation, not a missing exclusion.

### [StyleTTS2-Disc81]

* **Type**: forum
* **Title**: Notes on Finetuning (Discussion #81)
* **Author/Org**: yl4579/StyleTTS2 GitHub Discussions
* **Date**: 2023-10
* **URL**: https://github.com/yl4579/StyleTTS2/discussions/81
* **Peer-reviewed**: no
* **Relevance**: VRAM/batch-size guidance; clarifies `joint_epoch` is 0-indexed.

### [StyleTTS2-Disc128]

* **Type**: forum
* **Title**: Fine tuning guide (Discussion #128)
* **Author/Org**: yl4579/StyleTTS2 GitHub Discussions
* **Date**: 2023-11
* **URL**: https://github.com/yl4579/StyleTTS2/discussions/128
* **Peer-reviewed**: no
* **Relevance**: GPU/wall-clock data for 30-60 minute datasets; no epoch-count prescription beyond
  config defaults.

### [StyleTTS2-Disc65]

* **Type**: forum
* **Title**: Examples of a good fine-tune? (Discussion #65)
* **Author/Org**: yl4579/StyleTTS2 GitHub Discussions
* **Date**: 2023-09
* **URL**: https://github.com/yl4579/StyleTTS2/discussions/65
* **Peer-reviewed**: no
* **Relevance**: Two independent 50-epoch/joint-at-10 fine-tunes corroborate `config_ft.yml`'s
  defaults as real-world-validated.

### [StyleTTS2-Disc144]

* **Type**: forum
* **Title**: StyleTTS2 Training from Scratch Notebooks (Discussion #144)
* **Author/Org**: yl4579/StyleTTS2 GitHub Discussions
* **Date**: 2023-12
* **URL**: https://github.com/yl4579/StyleTTS2/discussions/144
* **Peer-reviewed**: no
* **Relevance**: From-scratch Stage 1 on ~150 files needed 400-1,000 epochs to stabilize —
  order-of-magnitude evidence for the from-scratch fallback scenario.

### [kikiri-tts-GH]

* **Type**: repository
* **Title**: kikiri-tts (formerly kokoro-deutsch)
* **Author/Org**: semidark
* **Date**: 2025-01
* **URL**: https://github.com/semidark/kokoro-deutsch
* **Peer-reviewed**: no
* **Relevance**: Resolves what the task meant by "semidark/StyleTTS2" — a patched submodule, not an
  independent checkpoint source.

### [semidark-StyleTTS2-DeepWiki]

* **Type**: documentation
* **Title**: semidark/StyleTTS2 (DeepWiki auto-generated docs)
* **Author/Org**: DeepWiki (community-generated, unaffiliated with semidark)
* **Date**: checked 2026-09-16
* **URL**: https://deepwiki.com/semidark/StyleTTS2
* **Peer-reviewed**: no
* **Relevance**: Details the fork's actual patches (PyTorch 2.6 migration, `.train()`-mode fix,
  PL-BERT token guard) — none overlap with this project's bug.

### [hifigan-Issue54]

* **Type**: forum
* **Title**: Minimum hours of data required for fine-tuning for a single unseen speaker (Issue #54)
* **Author/Org**: jik876/hifi-gan GitHub Issues
* **Date**: 2021-01
* **URL**: https://github.com/jik876/hifi-gan/issues/54
* **Peer-reviewed**: no
* **Relevance**: The exact question this project needs answered was asked in the official HiFi-GAN
  repo in 2021 and never answered — corroborates that no authoritative small-corpus figure exists.

### [StyleTTS2-Issue37]

* **Type**: forum
* **Title**: Possibly misleading license info (Issue #37)
* **Author/Org**: yl4579/StyleTTS2 GitHub Issues
* **Date**: 2023-08
* **URL**: https://github.com/yl4579/StyleTTS2/issues/37
* **Peer-reviewed**: no
* **Relevance**: Clarifies the MIT code license does not cover pretrained weights, which carry
  separate disclosure/consent terms.

### [GAN-Vocoder-MRD-2021]

* **Type**: paper
* **Title**: GAN Vocoder: Multi-Resolution Discriminator Is All You Need
* **Authors**: You, J., Kim, D., Nam, G., Hwang, G., Chae, G.
* **Year**: 2021
* **DOI**: not confirmed (arXiv preprint)
* **URL**: https://arxiv.org/abs/2103.05236
* **Peer-reviewed**: no (arXiv preprint; venue not verified)
* **Relevance**: Source of the discriminator-warmup and feature-matching-loss-pause GAN training
  diagnostics; also listed in Discovered Papers as a candidate corpus addition.
