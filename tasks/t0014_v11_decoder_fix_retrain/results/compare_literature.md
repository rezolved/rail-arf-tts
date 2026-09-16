---
spec_version: "1"
task_id: "t0014_v11_decoder_fix_retrain"
date_compared: "2026-09-16"
---
# Comparison with Published Results

## Summary

This task's Stage 2 epoch schedule (`epochs=50`, `diff_epoch=10`, `joint_epoch=30`) is compared
against StyleTTS2's own published two-stage recipes, and its decoder-adversarial training volume is
compared against HiFi-GAN-family papers' reported step budgets. The schedule **exactly matches**
StyleTTS2's official fine-tuning config (`config_ft.yml`) on all three epoch parameters, but the
resulting joint-phase step count (**≈3,820 steps**, 20 joint epochs × ≈191 steps/epoch) is **two to
three orders of magnitude below** both HiFi-GAN's **2,500,000**-step from-scratch budget
[Kong2020, Table 1] and iSTFTNet's **300,000**-step fine-tuning budget [Kaneko2022, p. 3-4] — a gap
explained by v11 fine-tuning an already-converged pretrained `hifigan` decoder rather than training
one from scratch or from a mismatched initialization, consistent with the ~8x fine-tune-vs-scratch
ratio [Kaneko2022] itself reports. Against this project's own prior task (t0013), v11's
`speaker_sim` (**0.444**) is a **+0.093 to +0.133** improvement over both confirmed-broken v10
checkpoints (0.311–0.351) and the audible-speech gate reverses from `is_likely_noise=True` to
`False` — but v11 remains **0.376** below the project's 0.85 GE2E-cosine target and **0.038** below
the unrelated-speaker LibriTTS control (0.482), so the fix resolves audibility, not speaker
similarity.

## Comparison Table

| Method / Paper | Metric | Published Value | Our Value | Delta | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| StyleTTS2 official fine-tune recipe (`config_ft.yml`) [StyleTTS2-ConfigFT-GH] | Total epoch budget | 50 | 50 | 0 | Fidelity check, not a performance metric — v11's config directly adopts this published recipe |
| StyleTTS2 official fine-tune recipe (`config_ft.yml`) [StyleTTS2-ConfigFT-GH] | `joint_epoch` (epoch adversarial/joint training begins) | 30 | 30 | 0 | Same recipe; exact match |
| StyleTTS2 official fine-tune recipe (`config_ft.yml`) [StyleTTS2-ConfigFT-GH] | `diff_epoch` (epoch diffusion training begins) | 10 | 10 | 0 | Same recipe; exact match |
| HiFi-GAN (Kong2020) [Kong2020, Table 1] | Decoder-adversarial training steps (from scratch) | 2,500,000 | 3,820 | -2,496,180 | Not directly comparable: v11 fine-tunes a pretrained `hifigan` decoder rather than training from scratch, and on a ~1/9-1/28-scale corpus (1,531 vs. 12,600-13,100 LJSpeech/VCTK clips); step count is v11's derived joint-phase volume (20 joint epochs x floor(1,531/8)=191 steps/epoch), not a logged optimizer counter |
| iSTFTNet (Kaneko2022) [Kaneko2022, p. 3-4] | Fine-tuning steps (from pretrained checkpoint) | 300,000 | 3,820 | -296,180 | Not directly comparable: Kaneko2022's figure is for an end-to-end Conformer-FastSpeech2 + iSTFTNet TTS+vocoder fine-tune, a different architecture/task than StyleTTS2's joint Stage-2 fine-tune; order-of-magnitude context only |
| StyleTTS2 (Li2023) LibriTTS HifiGAN-decoder config [Li2023, Section 4.1, p. 6-7] | Total epoch budget (pretrain + joint) | 55 | 50 | -5 | Li2023's LibriTTS config trains on ~245h/1,151 speakers, structurally different from this project's single-speaker 1,531-clip corpus; closest published paper-level anchor besides the exact-match `config_ft.yml` row above |

## Methodology Differences

* **Fine-tune vs. from-scratch.** [Kong2020] and [Kaneko2022]'s headline step budgets (2.5M / 300k)
  are for training a `hifigan`-family decoder from random initialization (or from independently
  pretrained TTS+vocoder components in Kaneko2022's case); v11 instead fine-tunes
  `yl4579/StyleTTS2-LibriTTS`'s already-converged `epochs_2nd_00020.pth` decoder — the corrected
  version of the "prefer a genuine pretrained checkpoint over random init" path this task's own
  research (`research/research_internet.md`) recommended.
* **Corpus scale.** [Kong2020] and [Kaneko2022] train on LJSpeech/VCTK-scale corpora (12,600-13,100
  clips); [Li2023]'s HifiGAN-decoder configs use VCTK (~44,000 clips/109 speakers) or LibriTTS
  (~245h/1,151 speakers). v11 trains on 1,531 single-speaker clips — roughly 1/9th to 1/160th the
  scale of any paper's reported HifiGAN-decoder configuration, per this task's own
  `research/research_papers.md` gap analysis.
* **Metric availability.** None of the three papers in the project corpus report GE2E cosine speaker
  similarity against an external reference set (this project's `speaker_sim` metric); their quality
  metrics are MOS (Kong2020 Table 2) or PESQ/mel-cepstral distortion style measures, so no
  paper-level `speaker_sim` comparison is possible — only the epoch/step-budget dimension is
  literature-comparable.
* **Step-count derivation.** v11's "Our Value" step counts in the table are derived (`joint_epoch`
  span × clips-per-batch), not read from a training-time optimizer-step counter —
  `data/run_v11/metrics.jsonl` logs epochs, not raw steps, so this is an arithmetic estimate from
  values in `code/config_david_v11.yml` and `results/results_detailed.md`, not a fabricated figure.

## Analysis

The epoch-schedule match to `config_ft.yml` (0 delta on all three parameters) confirms this task
executed the literature-recommended recipe precisely, not an ad hoc variant — direct evidence
against the possibility that v11's audible-speech gate pass was a fluke of an unusually long
schedule. The step-budget rows tell an orthogonal story: even with a schedule matching the published
recipe exactly, the *volume* of decoder-adversarial training (≈3,820 steps) is dwarfed by both
from-scratch papers' budgets. This is not a red flag on its own — [Kaneko2022]'s own internal ratio
(300k fine-tune vs. 2.5M from-scratch, roughly 1/8) already establishes that fine-tuning from a
converged checkpoint needs far fewer steps than training from scratch, and v11's ≈3,820 steps is
itself roughly 1/79th of even that reduced fine-tuning budget. The gate passing despite this small
step count is consistent with the decoder starting from an already-converged state (unlike
Kaneko2022's fine-tune, which still integrates a newly-initialized acoustic model) rather than
evidence the literature's step-budget concerns are moot.

[Kong2020, Table 2, p. 7]'s MPD ablation (removing the Multi-Period Discriminator costs **1.82
MOS**, the single largest-cost ablation in that table) was flagged in `plan/plan.md`'s Risks table
as the first diagnostic to check if the gate failed post-fix. Because the gate passed
(`is_likely_noise=False`), that diagnostic path was not exercised, and this task did not measure MOS
directly — so the ablation number remains contextual risk documentation, not a result this task can
confirm or refute quantitatively.

### Prior Task Comparison

`plan/plan.md` and `research/research_internet.md` cite t0013's confirmed-broken v10 checkpoints
(`speaker_sim` 0.311-0.351, `is_likely_noise=True`) as this task's baseline-to-beat, and t0013's
control run (the unrelated-speaker official LibriTTS checkpoint, `speaker_sim=0.482`) as an
upper-bound sanity reference:

| Prior result (t0013) | Metric | Prior Value | v11 Value | Delta | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| v10 primary, `epoch_2nd_00016` (`tasks/t0013_v10_synthesis_quality_forensics/results/metrics.json`) | speaker_sim | 0.3506 | 0.4442 | +0.0936 | v10 confirmed broken (`is_likely_noise=True`); v11 fixes the decoder-init bug |
| v10 backup, `epoch_2nd_00014` (same file) | speaker_sim | 0.3108 | 0.4442 | +0.1334 | Same; larger of the two improvements |
| Control, `epochs_2nd_00020` official checkpoint (same file) | speaker_sim | 0.4816 | 0.4442 | -0.0374 | Not a valid "improvement" baseline — different (unrelated) speaker identity, included for context only per t0013's own precedent |

This confirms rather than contradicts t0013's diagnosis: both v10 checkpoints scored non-trivially
positive `speaker_sim` despite being confirmed unlistenable noise (`clip_fraction=0.750-0.807`),
which is exactly why `metrics_notes.md` and `plan/plan.md` designate the audible-speech gate, not
`speaker_sim`, as this task's pass/fail criterion — v11's qualitative gate reversal
(`is_likely_noise: True -> False`) is the more load-bearing result than the `speaker_sim` delta. No
prior-task conclusion is contradicted here; v11's results are consistent with, and extend, t0013's
forensics.

## Limitations

* **No paper in the project corpus reports GE2E cosine speaker similarity or an equivalent
  speaker-similarity metric**, so `speaker_sim=0.444` (this task's headline result besides the
  audible-speech gate) cannot be compared to any published number — only against t0013's prior-task
  in-house measurements (see Prior Task Comparison) and the project's own 0.85 target
  (`project/description.md`), neither of which is "published literature."
- **No paper reports a from-scratch or fine-tuning HiFi-GAN-family step/epoch budget on a corpus
  close to 1,531 clips** ([Kong2020]/[Kaneko2022]: 12,600-13,100 clips; [Li2023]: 43,470-clip VCTK
  or ~245h LibriTTS) — the epoch-budget rows above are therefore order-of-magnitude context, not a
  scale-matched benchmark, as `research/research_papers.md`'s own Gaps section already discloses.
* **Step counts in the comparison table are derived, not measured.** `data/run_v11/metrics.jsonl`
  logs per-epoch records, not raw optimizer steps; the ≈191-steps/epoch and ≈3,820-joint-phase-steps
  figures are computed from `batch_size=8`, the 1,531-clip corpus size, and `joint_epoch=30`/
  `epochs=50` in `code/config_david_v11.yml`, all independently confirmed in this task's own result
  files, but this is not a substitute for a direct optimizer-step comparison.
* **The 73.95s duration anomaly (`results/v11_gate_verdict.md`) has no literature comparison
  point.** None of the three papers in the project corpus, nor the internet-research sources, report
  duration-predictor calibration failures or typical single-sentence synthesis durations that could
  contextualize this anomaly quantitatively — it remains a disclosed, unresolved limitation of this
  checkpoint (also noted in `results/results_detailed.md`'s own Limitations section) rather than
  something this comparison can explain.
* **[Kong2020]'s MPD ablation (1.82 MOS) is cited as risk-mitigation context, not a confirmed
  finding for this checkpoint** — this task did not run a MOS evaluation or an MPD-ablation
  diagnostic, since the gate passed and that diagnostic path (per `plan/plan.md`'s Risks table) was
  only scoped to trigger on gate failure.
