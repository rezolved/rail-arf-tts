# Research Summary — t0014_v11_decoder_fix_retrain

## Key Findings (top 10 insights directly actionable for this task)

1. Root cause: `train_second_v10.py:253-259`'s `ignore_modules` omits `"decoder"`, so
   `load_checkpoint()` partially loads istftnet-shaped `first_stage_v3.pth` into
   `config_david_v10.yml`'s hifigan-shaped decoder (shape-matching early layers only). t0013's probe
   showed this partial match is *worse than random init* (`clip_fraction` 0.75-0.81 vs. 0.004).
2. Upstream `train_second.py`'s own default `ignore_modules` **also** omits `"decoder"` — it assumes
   `first_stage_path`'s decoder always matches `config.yml`'s `decoder.type`. Correct fix: repoint
   `first_stage_path` at a genuinely hifigan-shaped checkpoint, not add `"decoder"` to
   `ignore_modules` and accept random init.
3. A genuinely hifigan-shaped pretrained checkpoint exists and field-matches this project:
   `yl4579/StyleTTS2-LibriTTS`'s `Models/LibriTTS/epochs_2nd_00020.pth` — decoder config
   (`resblock_kernel_sizes`, `upsample_initial_channel: 512`, `upsample_rates` product 300,
   `multispeaker: true`) matches `config_david_v10.yml` field-for-field.
4. This exact checkpoint is already harness-validated in this repo: t0013's `infer_styletts2.py`
   used it as `CONTROL_CKPT` (via the gitignored `semidark/kikiri-tts` submodule) and
   `results/control_test.md` proved it produces intelligible, non-noise speech.
5. Official fine-tuning anchor (`Configs/config_ft.yml`): `pretrained_model` = LibriTTS checkpoint,
   `epochs: 50`, `diff_epoch: 10`, `joint_epoch: 30`, at ~1k-sample/~1h scale — closest published
   anchor to this project's 1,531 clips, corroborated by two community fine-tunes (8h/10h
   single-voice) both at 50 epochs / joint-at-10.
6. From-scratch budgets dwarf v10's `epochs_2nd: 20` / `joint_epoch: 8`: HiFi-GAN 2.5M steps main /
   500k ablation; iSTFTNet 2.5M iterations; StyleTTS2's own HifiGAN configs use 50+40 (VCTK) / 30+25
   (LibriTTS) epochs. Fine-tuning uses ~1/8th the from-scratch budget (300k vs. 2.5M, Kaneko2022) —
   evidence for Approach 1 below.
7. LibriTTS pretrained weights: code is MIT, but the README requires disclosing pretrained-weight
   init and restricts cloning to consented voices — note in `results_summary.md`.
8. Verify the fix by module name, not shape-count: check `ups.*`, `resblocks.*`, `alphas.*`,
   `conv_post` for full key/shape overlap, using t0013's `inspect_checkpoint.py` and
   `infer_styletts2.py`'s `load_checkpoint_instrumented()` — shape overlap alone doesn't prove a
   correct load (how the original bug slipped through).
9. Do not reuse t0010's `eval_all_checkpoints.py`: Kokoro-pipeline synthesis can't load a
   hifigan-decoder checkpoint, and `MIN_SPEAKER_SIM_GATE=0.35` wouldn't have caught the bug (garbage
   v10 checkpoints scored 0.35/0.31). Use t0013's `infer_styletts2.py` + `audio_quality_check.py`'s
   `is_likely_noise` gate, per `task_description.md`.
10. `HealthGate` thresholds were calibrated on a single 250-clip run — 6x smaller than v11's corpus
    — and v11 combines the decoder fix *and* more data at once, so treat gate firings as needing
    manual review. If audio still fails, check MPD wiring (removing it costs 1.82 MOS in HiFi-GAN's
    ablation).

## Best Approaches (top 3 recommended implementation approaches from research)

### Approach 1: Fine-tune from the LibriTTS hifigan checkpoint (recommended)

Repoint `first_stage_path` in `config_david_v11.yml` at `yl4579/StyleTTS2-LibriTTS`'s
`epochs_2nd_00020.pth` instead of `first_stage_v3.pth`. No `ignore_modules` decoder exclusion needed
since architectures now match. Budget ~50 epochs, `diff_epoch≈10`, `joint_epoch≈30` per
`config_ft.yml`, validated by community fine-tunes at comparable corpus scale.

### Approach 2: Random-init decoder retrain (fallback only)

If the pretrained-checkpoint path fails verification or hits a licensing objection, add `"decoder"`
to `ignore_modules` and train from genuine random init. Requires an order-of-magnitude larger epoch
budget than v10's 20/8; StyleTTS2 Stage-1 from-scratch on ~150 files needed 400-1,000 epochs.

### Approach 3: Pre-flight tensor-forensics gate before any GPU spend

Run t0013-style `inspect_checkpoint.py` forensics against the new `first_stage_path` target to
confirm full key/shape overlap on every `decoder.*` submodule (not partial match) before training.
Gate completion on `audio_quality_check.py`'s `is_likely_noise=False`, not `val_loss` — v10 had zero
health-gate firings and a plausible `val_loss` curve yet produced 75-81% clipped noise, never
listened to before t0013.

## Reusable Code / Assets

* `tasks/t0010_stage2_safeguarded_training/code/train_second_v10.py` — copy forward as
  `train_second_v11.py`; fix `first_stage_path`/`load_checkpoint()` at lines 244-269; already
  imports the `t0009` safeguard library.
* `tasks/t0009_stage2_training_failure_forensics/code/{checkpoint_manager,health_gates,jsonl_logger,run_config}.py`
  — import via library; widen thresholds if v11's larger corpus produces different loss magnitudes.
* `tasks/t0013_v10_synthesis_quality_forensics/code/inspect_checkpoint.py` — copy into task; adapt
  expected-match target to the LibriTTS checkpoint (`classify_decoder()`).
* `tasks/t0013_v10_synthesis_quality_forensics/code/audio_quality_check.py` — copy into task
  unchanged; the completion-gate deliverable (`is_likely_noise`).
* `tasks/t0013_v10_synthesis_quality_forensics/code/infer_styletts2.py` — copy into task; StyleTTS2
  native harness, `load_checkpoint_instrumented()`; requires re-cloning gitignored `kikiri-tts`.
* `tasks/t0013_v10_synthesis_quality_forensics/code/random_decoder_probe.py` — copy into task as a
  sanity check the new config no longer reproduces the worse-than-random signature.
* `tasks/t0008_tts_eval_harness_baselines/code/scoring.py` — import via library (`build_centroid`,
  `compute_speaker_sim`); **not** `code/adapters.py`'s Kokoro synthesis (can't load a hifigan
  checkpoint).
* `tasks/t0012_v5_corpus_normalize_and_reaudit/data/train_list_v5_normalized_clean.txt` — the
  1,531-clip training manifest; `val_data` stays the unchanged `val_96`.

## Key Papers (top 5, with finding most relevant to this task)

* **Li et al. 2023 (StyleTTS 2)** — HifiGAN-decoder recipe uses 50+40 epochs (VCTK) / 30+25 epochs
  (LibriTTS), both exceeding v10/v11's 20/8 budget; this project's exact two-stage architecture.
* **Kong et al. 2020 (HiFi-GAN)** — From-scratch budget is 2.5M steps (500k ablations); MPD-removal
  costs 1.82 MOS, the largest single-component drop — a useful post-fix diagnostic.
* **Kaneko et al. 2022 (iSTFTNet)** — Fine-tuning uses ~1/8th the from-scratch step budget (300k vs.
  2.5M); explains why iSTFTNet/HiFi-GAN early-layer shapes overlap enough for a silent partial load.
* **You et al. 2021 (Multi-Resolution Discriminator)** — discovered, not yet added to the corpus;
  discriminator-warmup (10k steps) / feature-matching-loss-pause (99%) rules for `t0009`'s
  safeguards.

## Risks Flagged in Research

* No source reports a from-scratch/fine-tuning budget at exactly ~1,531 clips — any epoch count is
  an order-of-magnitude judgment call, not a literature-derived figure.
* `t0009` `HealthGate` thresholds were calibrated on a 250-clip run; v11 changes decoder init and
  corpus size at once, so gate firings need manual interpretation.
* LibriTTS pretrained weights require a disclosure note in results.
* `t0010`'s `eval_all_checkpoints.py`/`MIN_SPEAKER_SIM_GATE=0.35` are invalidated — do not reuse.
* `semidark/kikiri-tts` (control checkpoint + inference harness source) is gitignored and must be
  re-cloned in this task's worktree.
* If audio still fails the gate after the fix and a larger epoch budget, check MPD wiring/loss
  weighting before assuming pure under-training.

## Full Detail Available In

* `tasks/t0014_v11_decoder_fix_retrain/research/research_papers.md` — 3 papers
* `tasks/t0014_v11_decoder_fix_retrain/research/research_internet.md` — 17 sources
* `tasks/t0014_v11_decoder_fix_retrain/research/research_code.md` — 6 tasks cited (13 reviewed)
