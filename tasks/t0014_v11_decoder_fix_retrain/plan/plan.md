---
spec_version: "2"
task_id: "t0014_v11_decoder_fix_retrain"
date_completed: "2026-09-16"
status: "complete"
---
# Plan: v11 Decoder-Init Fix and Full-Corpus Retrain

## Objective

Fix the root cause that left the `kokoro-v10-best` HiFi-GAN decoder producing clipped/saturated
noise instead of speech, retrain Kokoro-82M Stage 2 on the larger, LUFS-normalized 1,531-clip corpus
t0012 produced, and gate this task's completion claim on an objective, automated audible-speech
check rather than on `val_loss` looking healthy (which is exactly what let `kokoro-v10-best` ship
broken in t0010).

**Root cause** (from t0013's forensics,
`tasks/t0013_v10_synthesis_quality_forensics/results/v10_diagnosis.md`):
`train_second_v10.py:load_checkpoint()` loaded `first_stage_v3.pth` (an ISTFTNet-shaped Stage-1
checkpoint) into `config_david_v10.yml`'s HiFi-GAN-shaped `decoder` without excluding `decoder` from
`ignore_modules`. The loader's `len(matched) == 0` guard only raises on a *fully*-zero match; a
partial shape match (architecture-agnostic layers only) passed silently, leaving the HiFi-GAN
generator's real vocoder layers (`ups`, `resblocks`, `alphas`, `conv_post`, `noise_convs`)
effectively randomly initialized — and a follow-up falsification probe
(`tasks/t0013_v10_synthesis_quality_forensics/code/random_decoder_probe.py`) proved this
"Frankenstein" partial load is actively *worse* than pure random init (clip fraction 0.750-0.807 vs.
0.004).

**The fix** (resolved by this task's own research steps 5-6, per
`tasks/t0014_v11_decoder_fix_retrain/research/research_summary.md` points 2-3): repoint
`first_stage_path` at a genuinely HiFi-GAN-shaped pretrained checkpoint —
`yl4579/StyleTTS2-LibriTTS`'s `Models/LibriTTS/epochs_2nd_00020.pth` — instead of adding `"decoder"`
to `ignore_modules` and accepting random init. This checkpoint's decoder config
(`resblock_kernel_sizes: [3,7,11]`, `upsample_initial_channel: 512`, `upsample_rates: [10,5,3,2]`,
`multispeaker: true`) field-matches `config_david_v10.yml` exactly, and is already harness-validated
in this repo: t0013's `infer_styletts2.py` loaded it with 0 missing/unexpected keys on all 13
modules and confirmed it produces real, non-noise speech
(`tasks/t0013_v10_synthesis_quality_forensics/results/control_test.md`, verdict PASS).

**Success criteria for this task**:

1. A tensor-level pre-flight check (before any GPU training spend) confirms the new
   `first_stage_path` target's `decoder` submodule is genuinely HiFi-GAN-shaped and loads with 0
   missing/0 unexpected keys — not a repeat of v10's silent partial match.
2. Training completes (or is stopped early only by a genuine health-gate firing or an unrecoverable
   infrastructure failure, never by disk exhaustion left unmonitored) on t0012's 1,531-clip
   normalized corpus.
3. **The mandatory audible-speech gate**: `audio_quality_check.py`'s `check_audio_quality(...)`
   returns `is_likely_noise=False` on synthesis from the resulting checkpoint, using the same
   `clip_fraction`/`spectral_flatness` heuristic t0013 built and validated against a known-good
   control. If this gate fails, the task stops, documents the failure with the same evidence
   standard t0013 used (control comparison, clip fraction/spectral flatness numbers, root-cause
   hypothesis if identifiable), and does **not** claim the model/audio-samples/inference-recipe
   deliverables as if they succeeded — a follow-up suggestion is filed instead.
4. Only if the gate passes: a `model` asset (`assets/model/kokoro-v11-best/`), paired
   original-vs-fine-tuned audio samples (`results/audio_samples/{original,ft}/`), and a working,
   instrumented inference recipe (`code/infer_styletts2.py`, adapted from t0013's) are produced.

## Task Requirement Checklist

Operative task text, quoted from `task.json` and `task_description.md`:

> **name**: "Kokoro Stage 2 v11: decoder-init fix, full normalized corpus retrain"
> **short_description**: "Fix the ignore_modules bug that left v10's HiFi-GAN decoder worse than
> random, retrain on t0012's 1531-clip normalized corpus, and gate completion on an actual audible-
> speech check." **task_description.md Motivation**: "This task's own completion is gated on proof
> of audible speech, not on val_loss ... If the fix still does not produce audible speech, say so
> plainly and stop — do not report success, do not fabricate an inference recipe or ship audio
> samples that fail the gate." **Scope §1**: "Fix the decoder-init bug ... Immediately verify the
> fix with a tensor-level check ... before launching any GPU training." **Scope §2**: "Switch to the
> t0012 normalized corpus ... Keep `val_data` pointed at the same held-out `val_96` set ... never
> train on it." **Scope §3**: "Train. Reuse t0009's safeguard library ... and t0010's
> `joint_epoch=8` + DP-aware checkpoint loader fix unchanged. Follow the same `setup-remote-machine`
> VM-pool flow t0010 used (`LLM-T1-NC80`), with the idle watchdog deployed before any long-running
> step ... watch disk usage proactively this time." **Scope §4**: "Mandatory audible-speech gate
> before claiming completion ... If it passes ... proceed to step 5. If it fails: stop, document the
> failure ... and do not produce the audio-samples/inference-recipe deliverables ... File a
> follow-up suggestion instead." **Scope §5 Deliverables**: "The best checkpoint ... Audio samples,
> paired original vs. FT ... A working inference recipe." **Compute and Budget**: "GPU training on
> `LLM-T1-NC80` (2xH100) ... budget and plan explicitly for disk monitoring and prompt teardown this
> time. Write `results/costs.json` with the actual total."

Requirement decomposition:

* **REQ-1**: Fix the decoder-init bug by repointing `first_stage_path` at a genuinely
  HiFi-GAN-shaped pretrained checkpoint (per this task's already-resolved Key Question 1), not by
  adding `"decoder"` to `ignore_modules` and accepting random init. Satisfied by Milestone A, step
  1\. Evidence: `code/train_second_v11.py` diff against `train_second_v10.py`, documented inline.
* **REQ-2**: Verify the fix with a tensor-level check *before* any GPU training spend, confirming
  `decoder` loads with full key/shape overlap (not a partial match). Satisfied by Milestone A, step
  2\. Evidence: `code/inspect_checkpoint.py` output, `results/checkpoint_forensics_v11.md`.
* **REQ-3**: Point `train_data` at t0012's 1,531-clip normalized manifest; keep `val_data` unchanged
  at `val_96`; never train on `val_96`. Satisfied by Milestone A, step 3. Evidence:
  `code/config_david_v11.yml` diff, explicit grep confirming no `val_96` clip IDs appear in
  `train_data`.
* **REQ-4**: Train using t0009's safeguard library (JSONL logger, health gates, per-epoch
  `CheckpointManager`) unchanged, on `LLM-T1-NC80`, with the idle watchdog deployed before any
  long-running step and disk usage monitored proactively (not discovered at teardown, per t0010's
  failure). Satisfied by Milestone B. Evidence: `data/run_v11/metrics.jsonl`,
  `results/disk_usage_log.txt`, watchdog PID confirmation in `logs/steps/`.
* **REQ-5**: Resolve the epoch-budget ambiguity explicitly (see Approach section) — anchor on the
  official `Configs/config_ft.yml` recipe (`epochs: 50`, `diff_epoch: 10`, `joint_epoch: 30`) rather
  than v10's `20`/`8`, per this task's own research resolution. Satisfied by Milestone A, step 3
  (config values) and Milestone B (actual epochs run). Evidence: `code/config_david_v11.yml`.
* **REQ-6**: Run the mandatory audible-speech gate (`audio_quality_check.py`'s
  `check_audio_quality(...).is_likely_noise`) against the new best checkpoint before claiming
  completion. Satisfied by Milestone C. Evidence: `results/audio_quality_v11.json`,
  `results/v11_gate_verdict.md`.
* **REQ-7**: If the gate passes, produce the `model` asset (`assets/model/kokoro-v11-best/`), paired
  `results/audio_samples/{original,ft}/`, and the inference recipe `code/infer_styletts2.py`. If the
  gate fails, do **not** produce these — document the failure honestly and file a follow-up
  suggestion instead. Satisfied by Milestone D (conditional). Evidence: asset folder + gate verdict
  cross-reference.
* **REQ-8**: Write `results/costs.json` with the actual total GPU spend, and tear down `LLM-T1-NC80`
  promptly (no idle overnight billing repeat). Satisfied by Milestone E. Evidence:
  `results/costs.json`, `results/remote_machines_used.json` showing `destroyed_at` set.
* **REQ-9**: Measure the registered project metrics that apply (`speaker_sim`, `rtf`) on the gate
  synthesis output, in the explicit multi-variant `results/metrics.json` format, matching t0013's
  precedent; explicitly omit `ttfb_ms` (not applicable — offline batch harness, no streaming
  endpoint) with a stated reason, not a silent gap. Satisfied by Milestone C, step 3. Evidence:
  `results/metrics.json`.

**Ambiguity flagged and resolved** (per plan-spec instruction not to silently merge/discard):
`task_description.md` Scope §3 literally says to reuse t0010's `joint_epoch=8` "unchanged," but the
task's own `checkpoint.md` Cross-Step Decisions and `research/research_summary.md` (written *after*
`task_description.md`) explicitly supersede this: "the retrain epoch budget (anchor on
`Configs/config_ft.yml`'s 50/10/30 schedule per step 5, not v10's 20/8)." This plan follows the
later, research-informed resolution (REQ-5) — the `task_description.md` text predates Key Question 2
being answered and is treated as superseded, not as a contradiction to route around silently.

## Approach

**Root fix — Approach 1 from `research/research_summary.md` (recommended, adopted here)**: repoint
`first_stage_path` in a new `code/config_david_v11.yml` at `yl4579/StyleTTS2-LibriTTS`'s
`Models/LibriTTS/epochs_2nd_00020.pth` instead of `first_stage_v3.pth`. Because both the
checkpoint's decoder and `config_david_v10.yml`'s `model_params.decoder.type: hifigan` block are
HiFi-GAN-shaped, no `ignore_modules` change is needed for `decoder` — the existing
`ignore_modules=["predictor_encoder", "msd", "mpd", "wd", "diffusion"]` list in `load_checkpoint()`
is left unchanged (these five modules are excluded for the same reason as v10: `predictor_encoder`
is deep-copied from `style_encoder` post-load, `msd`/`mpd`/`wd`/`diffusion` start fresh regardless
of source checkpoint). `decoder` is not in that exclusion list, so it now loads for real from a
checkpoint whose architecture actually matches, instead of silently partial-matching a structurally
different one.

**Alternative considered and rejected as primary (Approach 2, fallback only)**: add `"decoder"` to
`ignore_modules` and train the HiFi-GAN decoder from genuine random init. Research found this needs
an order-of-magnitude larger epoch budget than v10's `20`/`8` — StyleTTS2's own from-scratch HifiGAN
configs use 50+40 epochs (VCTK) / 30+25 epochs (LibriTTS), and HiFi-GAN/iSTFTNet train 2.5M steps
from scratch (500k for ablations). This project has no from-scratch budget validated at 1,531-clip
scale, so choosing this path means silently under-training a second time unless the epoch count is
inflated far beyond anything already validated on comparable data. **This plan uses Approach 2 only
as an escape hatch**: if REQ-2's pre-flight tensor check finds the LibriTTS checkpoint's `decoder`
does *not* cleanly match (unexpected licensing block, download failure, architecture drift
discovered at load time), Milestone A step 2 falls back to adding `"decoder"` to `ignore_modules`,
and Milestone A step 3's epoch budget switches to a documented from-scratch estimate (400-1,000
epochs per `research/research_summary.md` point 6) — which, given `LLM-T1-NC80`'s cost, requires
flagging in `intervention/` for human budget sign-off before proceeding, not silently absorbing the
10-50x cost multiplier.

**Epoch budget (REQ-5)**: the official `Configs/config_ft.yml` fine-tuning recipe sets
`pretrained_model: Models/LibriTTS/epochs_2nd_00020.pth`, `epochs: 50`, `diff_epoch: 10`,
`joint_epoch: 30`, at ~1k-sample/~1h corpus scale — the closest published anchor to this project's
1,531 clips, corroborated by two independent community fine-tunes ("Aurora" 8h, "Chaos" 10h,
single-voice) both using 50 epochs with joint training starting at epoch 10
(`research/research_internet.md`, sources `StyleTTS2-ConfigFT-GH`, `StyleTTS2-Disc65`). This plan
adopts `epochs: 50`, `epochs_2nd: 50`, `diff_epoch: 10`, `joint_epoch: 30` for
`config_david_v11.yml`, superseding v10's `20`/`6`/`8`. All other hyperparameters (`batch_size: 8`,
`lr`, `ft_lr`, `bert_lr`, all `lambda_*` weights, `max_len` default of 200) are left unchanged from
v10 — only the epoch-schedule fields research explicitly anchored on are changed, to keep the
surface of change minimal and attributable.

**Licensing note (REQ-1 supporting)**: `yl4579/StyleTTS2-LibriTTS`'s code is MIT-licensed, but per
GitHub issue `StyleTTS2-Issue37` the README separately requires disclosing that a model was
initialized from these pretrained weights and restricts voice-cloning to consented speakers. This
project uses the checkpoint only as a decoder weight initialization for fine-tuning on Rezolve's own
licensed David voice data — it never ships a LibriTTS speaker identity — but this plan requires
`results/results_summary.md` to carry an explicit disclosure sentence per this finding
(`research/research_internet.md` §"Licensing").

**Audible-speech gate (REQ-6, the task's core deliverable per its `short_description`)**: reuse
`tasks/t0013_v10_synthesis_quality_forensics/code/audio_quality_check.py` unchanged — its
`check_audio_quality(wav_path)` returns an `AudioQualityResult` with `clip_fraction`,
`spectral_flatness`, `silence_fraction`, and the boolean `is_likely_noise` (flagged when
`spectral_flatness >= 0.35` OR `clip_fraction >= 0.3`, and the clip is not >95% silent). This
heuristic was purpose-built and validated in this exact repo: it correctly separates a known-good
external control (`clip_fraction=0.0`, `spectral_flatness=0.0615`, `is_likely_noise=False`) from
confirmed-broken v10 output (`clip_fraction=0.750-0.807`, `spectral_flatness=0.00048-0.00019`,
`is_likely_noise=True`). It is a stronger signal than `val_loss` (looked healthy throughout t0010's
17 epochs despite total decoder failure) and stronger than `speaker_sim` (both confirmed-garbage v10
checkpoints still scored a non-trivial 0.31-0.35 GE2E cosine, not a near-zero outlier — see
`research/research_summary.md` point 9 and `results/v10_diagnosis.md` "Metrics note"). This plan
does **not** reuse t0010's `eval_all_checkpoints.py`: it routes through Kokoro's
`KModel`/`KPipeline`, which cannot load a `hifigan`-decoder StyleTTS2-native checkpoint, and its
`MIN_SPEAKER_SIM_GATE=0.35` threshold would not have caught v10's failure (both broken checkpoints
scored 0.31/0.35, straddling that exact threshold).

**Inference harness**: reuse `tasks/t0013_v10_synthesis_quality_forensics/code/infer_styletts2.py`
(StyleTTS2-native, not Kokoro's `KModel`/`KPipeline`) copied into this task as
`code/infer_styletts2.py` and re-pointed at the new checkpoint/config. It already implements
`load_checkpoint_instrumented()` (per-module missing/unexpected key logging, hard-fails on any core
module mismatch) and `_rename_legacy_parametrization_keys()` (needed only for the external LibriTTS
control checkpoint, a no-op for this project's own DataParallel-trained checkpoints). Also copy
`code/inspect_checkpoint.py` (tensor-level pre-flight forensics, REQ-2),
`code/random_decoder_probe.py` (adapted as a v11 sanity check that the fix no longer reproduces the
worse-than-random signature), and `code/score_speaker_sim.py` (REQ-9's `speaker_sim` measurement,
adapted from t0008's `build_centroid`/`compute_speaker_sim` logic per
`research/research_summary.md`'s Reusable Code note — import the scoring functions, not
`adapters.py`'s Kokoro synthesis).

**Recommended task types**: `task.json` already declares `tts-finetuning-eval`, which covers the
evaluation-against-baseline framing (compare v11 vs. ElevenLabs David and vs. v10) and the
`overview/models/`-style documentation convention. This plan additionally recommends `build-model`
(from `meta/task_types/build-model/instruction.md`) as a second applicable type, since this task's
core action is training/fine-tuning a model from a fixed config and producing a `model` asset with
logged hyperparameters, checkpoint-saving cadence, and a comparison against a prior checkpoint (v10)
— `build-model`'s Planning Guidelines (hard budget ceiling, checkpoint-saving frequency,
hyperparameter logging before training starts) are followed throughout this plan. The orchestrator
or a human may want to add `build-model` to `task.json`'s `task_types` list; this plan proceeds
under the guidance of both without requiring that edit first.

**Alternative approach considered and rejected**: skip the tensor pre-flight check (REQ-2) and go
straight to training, relying on the training run's own `load_checkpoint()` print statements to
catch a bad load. Rejected because this is exactly how v10's bug slipped through undetected for 17
epochs — `load_checkpoint()`'s pass condition only raises on a *fully* zero-matched module, so a
partial (Frankenstein) match prints success and looks identical to a correct load in the training
log. `research/research_summary.md` point 8 is explicit: "Verify the fix by module name, not
shape-count... shape overlap alone doesn't prove a correct load." A dedicated pre-flight step
outside the training loop, using `inspect_checkpoint.py`'s `classify_decoder()` (checks for the
`generator.alphas.*` HiFi-GAN-only marker, not just matched-parameter counts), is the only way to
close this gap before spending GPU time.

## Cost Estimation

**Per-task default limit**: $100.00 (from `project/budget.json`). **Project budget remaining**:
$4,692.12 of $5,000.00 (93.8%) as of task start. This task is explicitly expected to exceed the
per-task default, per the task brief — budget headroom is not the constraint; an itemized, justified
estimate is required instead of silent scope truncation.

**GPU compute** (`LLM-T1-NC80`, 2xH100, $13.96/hr, from `project/azure_vm.json`):

* Active training compute estimate, derived from t0010's own measured per-epoch wall time (not a
  guess): t0010's `data/run_v10/metrics.jsonl` timestamps show 250-clip epochs averaged **32.5s**
  pre-`joint_epoch` (epochs 1-8) and **38.6s** post-`joint_epoch` (epochs 9-17), i.e. GAN-phase
  epochs cost ~19% more wall time than non-GAN epochs at this corpus size. Scaling by the
  corpus-size ratio (1,531 / 250 = **6.12x** more batches at the same `batch_size: 8`):
  * Non-joint epochs (1-29 of 50, `joint_epoch: 30`): 29 × 32.5s × 6.12 ≈ 29 × 199s ≈ **96 min**
  * Joint epochs (30-50 of 50, 21 epochs): 21 × 38.6s × 6.12 ≈ 21 × 236s ≈ **83 min**
  * **Estimated active training compute: ~3.0 hours** (this likely *overestimates* slightly, since
    part of each epoch's fixed cost — the 96-clip `val_96` validation loop and the 200-sample
    voicepack extraction — does not scale with train corpus size; kept as the conservative,
    over-not-under estimate).
* Setup + environment verification (Lesson 10/11 preflight, dependency install, checkpoint
  download): **~0.75 hour**.
* Pre-flight tensor check (REQ-2, CPU-only, cheap, can run on the GPU VM or locally — budgeted on
  the VM for simplicity): **~0.1 hour**.
* Audible-speech gate synthesis + `speaker_sim`/`rtf` scoring + audio-sample generation (REQ-6,
  REQ-7, REQ-9): **~0.5 hour**.
* Contingency buffer for restart-on-health-gate-firing or a short reproducible retry (not unlimited
  retries — see Risks table): **~1.5 hours**.
* **Subtotal active/planned VM time: ~5.85 hours ≈ $81.66.**

**Idle-billing risk (the actual driver of t0010's $272.78 overrun, not compute)**: t0010's 19.54
billed hours vs. ~10 minutes of actual measured training compute shows the overrun was **entirely**
idle time after a full disk stalled teardown overnight, not GPU-hour cost of the work itself. This
plan's Milestone B step 5 (proactive disk monitoring) and Step by Step's explicit watchdog
deployment before any long-running step are the direct mitigation. Planned total, assuming
mitigations hold: **~6 hours billed ≈ $84**. **Realistic worst-case if disk/idle mitigations fail
again despite being planned for** (same failure class as t0010, watchdog
`IDLE_THRESHOLD_SECONDS=3600` bounds any single idle gap to ~1 additional hour before
auto-termination): budget up to **$150-200** before this is treated as a repeat-pattern escalation,
not routine retry. **Hard stop**: if total spend for this task exceeds **$300** (more than the full
amount t0010 overran to), halt and write an `intervention/` file rather than continuing to retry —
this is more than 2x the planned estimate and signals a systemic issue, not routine variance.

**Other costs**: $0 — no paid APIs used (LibriTTS checkpoint download is free via `git-lfs clone`
from Hugging Face; `resemblyzer`/`librosa` are open-source, already used by t0013's harness in this
repo's `.venv-styletts2`). Downloading `yl4579/StyleTTS2-LibriTTS`'s ~771MB checkpoint is free
bandwidth, one-time.

**Comparison to budget**: even the $300 hard-stop ceiling is 6% of the total $5,000 project budget
and leaves the project at ~~93.6% headroom remaining ($4,392+ of $5,000) — well under the 80% warn
threshold. The realistic planned total (~~$84-150) is 3x the $100 per-task default limit but in line
with, and well below, t0010's own precedent ($272.78) for materially the same class of work (GPU
Stage-2 training run on this VM pool).

## Step by Step

### Milestone A: Fix + verify before any GPU spend (no GPU compute yet — cheap, local/CPU)

1. **[CRITICAL] Create the corrected training config.** Copy
   `tasks/t0010_stage2_safeguarded_training/data/run_v10/config_david_v10.yml` to
   `tasks/t0014_v11_decoder_fix_retrain/code/config_david_v11.yml`. Change exactly these fields:
   * `first_stage_path`: from `/home/azureuser/kokoro-finetune/first_stage_v3.pth` to the VM-side
     path where the LibriTTS checkpoint will be downloaded, e.g.
     `/mnt/cache/persist/pretrained/StyleTTS2-LibriTTS/Models/LibriTTS/epochs_2nd_00020.pth` (Lesson
     10: persistent, not ephemeral `/mnt`, so a VM stop/restart does not force a 771MB re-download).
   * `data_params.train_data`: from `data/data_list_v5_train_250.txt` to the path (on the VM) that
     mirrors `tasks/t0012_v5_corpus_normalize_and_reaudit/data/train_list_v5_normalized_clean.txt`
     (1,531 clips). `data_params.val_data` stays unchanged (`data/val_list.txt`, the same `val_96`
     set t0008/t0009/t0010 all used).
   * `epochs`: `20` → `50`. `epochs_2nd`: `20` → `50`. `loss_params.diff_epoch`: `6` → `10`.
     `loss_params.joint_epoch`: `8` → `30`.
   * `log_dir`: `logs/v10` → `logs/v11`.
   * Leave every other field (`batch_size: 8`, all `lambda_*`, `optimizer_params`, `model_params`,
     `slmadv_params`) unchanged from v10. Document each change with an inline YAML comment citing
     this plan's Approach section reasoning. Satisfies REQ-1, REQ-3, REQ-5.

2. **[CRITICAL] Copy and adapt the training script; do not modify t0010's completed folder.** Copy
   `tasks/t0010_stage2_safeguarded_training/code/train_second_v10.py` to
   `tasks/t0014_v11_decoder_fix_retrain/code/train_second_v11.py`. The only functional change needed
   is none to `load_checkpoint()`'s `ignore_modules` list itself (it stays
   `["predictor_encoder", "msd", "mpd", "wd", "diffusion"]`, unchanged from v10) — the fix is
   entirely in `config_david_v11.yml`'s `first_stage_path` (step 1). Add a module-level docstring
   note explaining why `decoder` is correctly *not* excluded here (architectures now match, per this
   plan's Approach section), so a future reader does not reintroduce v10's bug by "fixing" this
   list. Update the two `from tasks.t0009...` import lines only if t0009's safeguard module paths
   changed (they have not, per `research/research_code.md`) — otherwise the imports are unchanged.
   Satisfies REQ-1.

3. **[CRITICAL] Download the pretrained checkpoint and run the tensor-level pre-flight check —
   before any training.** On `LLM-T1-NC80` (or locally if disk/bandwidth allow, then `scp` to the
   VM):
   `git lfs install && git clone https://huggingface.co/yl4579/StyleTTS2-LibriTTS /mnt/cache/persist/pretrained/StyleTTS2-LibriTTS`
   (Lesson 10: clone directly onto the verified persistent-disk symlink target, not ephemeral
   `/mnt`). Copy `tasks/t0013_v10_synthesis_quality_forensics/code/inspect_checkpoint.py` into
   `tasks/t0014_v11_decoder_fix_retrain/code/inspect_checkpoint.py`, adapt its `targets` list in
   `main()` to inspect `epochs_2nd_00020.pth` (the new `first_stage_path` target) alongside
   `first_stage_v3.pth` (for a before/after contrast), and its path constants (copy
   `tasks/t0013_v10_synthesis_quality_forensics/code/paths.py` and add a `LIBRITTS_CONTROL_CKPT`
   constant pointing at the downloaded checkpoint). Run it:
   `uv run python tasks/t0014_v11_decoder_fix_retrain/code/inspect_checkpoint.py`. **Validation
   gate**: expect `classify_decoder()` to report `classification: "hifigan"` with
   `has_hifigan_marker: True` for `epochs_2nd_00020.pth`'s decoder (same result t0013 already
   confirmed for this exact checkpoint in `results/checkpoint_forensics.md`). If it does **not**
   report `hifigan`, or if `torch.load` fails, or if any module shows `finite: False` — **STOP**: do
   not proceed to Milestone B. Fall back to Approach 2 (Approach section): add `"decoder"` to
   `ignore_modules` in `code/train_second_v11.py`, write an `intervention/` file explaining why the
   primary approach failed and flagging the epoch-budget implications of the fallback for human
   sign-off, then continue with Milestone A step 4 adapted for random-init decoder training instead.
   Output: `results/checkpoint_forensics_v11.md`, `results/checkpoint_forensics_v11_raw.json`.
   Satisfies REQ-2.

4. **Confirm no `val_96` leakage into the new train manifest.** Run a direct diff/grep: every line
   in `tasks/t0012_v5_corpus_normalize_and_reaudit/data/train_list_v5_normalized_clean.txt` against
   `data/val_list.txt`'s clip IDs (the same file t0008/t0009/t0010 used as `val_data`), confirming
   zero overlap — mirrors t0012's own "0 val leaks" self-check already reported for that task's
   corpus-normalization work. Write the check result to `results/val96_leak_check.txt`. Satisfies
   REQ-3.

### Milestone B: Provision `LLM-T1-NC80` and train (GPU compute begins here)

5. **[CRITICAL] Provision `LLM-T1-NC80` via `/setup-remote-machine`.** Follow
   `arf/skills/setup-remote-machine/SKILL.md` Phases 1-4. `azure_ml_vm.acquire()` runs
   `arf/scripts/utils/remote_preflight.sh` before placing the task lock — this already verifies
   `loginctl show-user azureuser | grep Linger` shows `Linger=yes` (Lesson 11) and that
   `/mnt/cache/persist` resolves to the real Azure Files mount, not ephemeral `/mnt` (Lesson 10).
   Re-verify by hand anyway before launching the long training job (belt-and-suspenders, since this
   is the exact failure class that cost t0010 $272.78):
   ```bash
   ssh LLM-T1-NC80 "loginctl show-user azureuser | grep Linger"
   ssh LLM-T1-NC80 "readlink -f /mnt/cache/persist && df -h /mnt/cache/persist"
   ssh LLM-T1-NC80 "df -h /"
   ```
   **Additionally, proactively check root-disk (`/dev/root`) free space before starting** — this is
   the exact disk that filled up at 100% in t0010 (per
   `tasks/t0010_stage2_safeguarded_training/intervention/eval_deferred_disk_full.md`), separate from
   the `/mnt/cache/persist` Azure Files share. Require at least 30GB free on `/` before proceeding;
   if not, clean `~/.cache/{pip,huggingface,whisper}` first. Deploy the idle watchdog per
   `arf/scripts/utils/idle_watchdog.sh` and CLAUDE.md's "Deploying the watchdog" section
   (`TERMINATE_CMD="az ml compute stop --name LLM-T1-NC80 --workspace-name brainpowa-northeurope --resource-group rezolve-AI"`,
   `IDLE_THRESHOLD_SECONDS=3600`) **before** launching training, per Lesson 8's "watchdog installed
   on every GPU machine, not only ones that will pause" rule. Record the watchdog PID in
   `logs/steps/`. Satisfies REQ-4.

6. **[CRITICAL] Launch training inside `tmux`, monitor disk proactively (not just at teardown).**
   ```bash
   ssh LLM-T1-NC80 "tmux new-session -d -s v11train \
     'cd ~/kokoro-finetune && python tasks/t0014_v11_decoder_fix_retrain/code/train_second_v11.py \
     -p tasks/t0014_v11_decoder_fix_retrain/code/config_david_v11.yml --run-id v11 \
     > /home/azureuser/v11_train.log 2>&1; echo DONE >> /home/azureuser/v11_train.log'"
   ```
   Wrap the launch and every monitoring check in `run_with_logs.py` per CLAUDE.md Key Rule 1:
   `uv run python -m arf.scripts.utils.run_with_logs --task-id t0014_v11_decoder_fix_retrain -- ssh LLM-T1-NC80 "..."`.
   While training runs, poll every ~15-20 minutes (synchronously, within this step-executor's own
   turn — never a fire-and-forget background poller, per `implementation` skill's prohibition and
   Lesson 8): `tmux has-session -t v11train`, tail `/home/azureuser/v11_train.log`, and explicitly
   check `df -h /` and `df -h /mnt/cache/persist` — this is the proactive disk monitoring the task
   brief calls for, closing the exact gap that stalled t0010's teardown. If root-disk usage exceeds
   85%, pause and clean caches (pip/HF/tensorboard event files) before it reaches 100%. Output:
   `data/run_v11/metrics.jsonl` (t0009's `StepLogger`, one record per `log_interval` steps plus one
   per-epoch validation record), `data/run_v11/checkpoints.json` (t0009's `CheckpointManager`
   manifest with SHA-256 per checkpoint), `data/run_v11/config_david_v11.yml` (copied at startup by
   `capture_run_config`). Expected observable output: JSONL `val_loss` records printing every epoch,
   `[HealthGate FIRED]` absent (0 gate events, matching v10's stability) unless a genuine divergence
   occurs. Satisfies REQ-4, REQ-5.

7. **Handle health-gate firings or disconnection per `LESSONS.md`/`setup-remote-machine` protocol.**
   If a `HealthGate` fires (`sys.exit(3)`), the training process exits and the last-known-healthy
   checkpoint is named in the gate's log message
   (`tasks/t0009_stage2_training_failure_forensics/code/health_gates.py`'s `_last_healthy_ckpt`). Do
   not blindly restart from epoch 0 — restart training pointed at that checkpoint only after
   inspecting `data/run_v11/metrics.jsonl` for the specific metric that fired and confirming it is
   not a repeat of the same failure mode (per `build-model` task type's "try at least 3 different
   approaches before creating an intervention file" guidance — but do not silently retry more than 3
   times without writing an `intervention/` file). If SSH disconnects, reconnect and confirm the
   `tmux` session and training log are still progressing (lingering + `tmux` guarantee survival per
   Lesson 11) before assuming failure.

### Milestone C: Mandatory audible-speech gate (the task's actual pass/fail criterion)

8. **[CRITICAL] Package the best checkpoint and copy the inference harness.** Identify the
   best/final checkpoint from `data/run_v11/checkpoints.json` (highest epoch with lowest `val_loss`
   post-`joint_epoch`, following t0010's own precedent of picking the latest completed epoch unless
   a clear regression is visible). Copy
   `tasks/t0013_v10_synthesis_quality_forensics/code/infer_styletts2.py` to
   `tasks/t0014_v11_decoder_fix_retrain/code/infer_styletts2.py`, re-pointing its path constants
   (via a copied-and-adapted `code/paths.py`) at the v11 checkpoint and
   `tasks/t0014_v11_decoder_fix_retrain/code/config_david_v11.yml`'s `model_params`. No functional
   change to `load_checkpoint_instrumented()` or `synthesize()` is needed — v11's checkpoint is
   DataParallel-trained under this fork's current `models.py` exactly like v10's, so the
   `module.`-stripped fallback path (no legacy-parametrization rename needed) applies unchanged.
   Satisfies REQ-6, REQ-7 (recipe).

9. **[CRITICAL] Run the mandatory audible-speech gate.** Using the same 3-clip-concatenated
   `tasks/t0008_tts_eval_harness_baselines/data/11labs_david/` reference audio and the same test
   text t0013 used ("This is a test of the Style T T S two inference harness.") for direct
   comparability, run `code/infer_styletts2.py` against the v11 checkpoint, producing
   `results/audio_samples/ft/v11_best.wav`. Copy
   `tasks/t0013_v10_synthesis_quality_forensics/code/audio_quality_check.py` unchanged into
   `tasks/t0014_v11_decoder_fix_retrain/code/audio_quality_check.py` and run
   `check_audio_quality(Path("results/audio_samples/ft/v11_best.wav"))`. **Baseline / validation
   gate**: the trivial baseline is v10's own confirmed-noise output (`clip_fraction=0.750-0.807`,
   `is_likely_noise=True`) and the known-good control (`clip_fraction=0.0`, `is_likely_noise=False`,
   from `results/control_test.md`). **Failure condition**: if `is_likely_noise=True` for the v11
   output, **STOP** — do not proceed to step 10 or Milestone D. Individual-output inspection
   requirement: read the numeric `clip_fraction`, `spectral_flatness`, and `silence_fraction` values
   directly (not just the boolean) and compare them explicitly against both t0013's confirmed-noise
   numbers and the control's numbers in `results/v11_gate_verdict.md`, so a borderline pass/fail is
   documented with evidence, not just a boolean. Write `results/audio_quality_v11.json` and
   `results/v11_gate_verdict.md` (pass or fail, with the same evidence standard as t0013's
   `v10_diagnosis.md`). If fail: write `intervention/audible_gate_failed.md` documenting the
   failure, do not create Milestone D's deliverables, and record recommended next steps for a
   follow-up task (e.g., MPD-wiring check per HiFi-GAN's ablation finding that removing MPD costs
   1.82 MOS, or the Approach-2 random-init fallback with a much larger epoch budget) so the
   orchestrator's own suggestions step can pick them up. Satisfies REQ-6.

10. **Only if step 9 passes**: also synthesize with the *original* (pre-fine-tune) checkpoint
    referenced by `first_stage_path` for the paired `results/audio_samples/{original,ft}/`
    comparison the task's Scope §5 requires —
    `results/audio_samples/original/librispeech_control.wav` can reuse t0013's already-produced
    `control_epochs_2nd_00020.wav` (same checkpoint, same harness) rather than re-synthesizing
    identical output; note this reuse explicitly wherever this task documents its outcome. Satisfies
    REQ-7.

11. **Measure `speaker_sim` and `rtf` (REQ-9), explicitly omit `ttfb_ms`.** Copy
    `tasks/t0013_v10_synthesis_quality_forensics/code/score_speaker_sim.py` into
    `tasks/t0014_v11_decoder_fix_retrain/code/score_speaker_sim.py`, retarget its `targets` list at
    `results/audio_samples/ft/v11_best.wav`, and run it through the isolated `.venv-styletts2`
    environment (re-created per t0013's Milestone C setup, since `resemblyzer` is kept out of the
    main project `.venv` per `overview/metrics/speaker_sim.md`). It builds the GE2E centroid from
    `tasks/t0008_tts_eval_harness_baselines/data/11labs_david/` (1,358 clips, above the hardcoded
    `MIN_CORPUS_SIZE=1000` assertion) using the same `CENTROID_SEED=42`, `CENTROID_SAMPLE_SIZE=679`
    as t0013, for direct comparability. `rtf` is already emitted by `infer_styletts2.py`'s `main()`
    as `<output_wav>.timing.json` — read it directly, no re-computation needed. Write
    `results/metrics.json` in the explicit multi-variant format (same shape as
    `tasks/t0013_v10_synthesis_quality_forensics/results/metrics.json`):
    ```json
    {
      "variants": [
        {
          "variant_id": "v11-best",
          "label": "kokoro-v11-best (decoder-init fix + 1531-clip corpus)",
          "dimensions": {"checkpoint": "<epoch_NNNNN.pth>", "decoder_type": "hifigan",
                          "role": "fixed_retrain"},
          "metrics": {"rtf": <float>, "speaker_sim": <float>}
        }
      ]
    }
    ```
    Record as a documented caveat, mirroring t0013's own note: `rtf` is measured single-clip on the
    same unbatched-CPU-inference harness as t0013's numbers (corroborating only, not a benchmark
    claim), and `speaker_sim` is a weak signal for the specific defect this task fixes
    (confirmed-broken v10 output still scored 0.31-0.35, non-trivially positive) — the
    audible-speech gate (step 9), not `speaker_sim`, is this task's actual pass/fail criterion.
    State explicitly that `ttfb_ms` is deliberately not measured: this harness is an offline batch
    script writing one complete WAV per invocation, with no streaming/production serving endpoint in
    scope, matching t0013's own precedent and Milestone C's out-of-scope note. Satisfies REQ-9.

### Milestone D: Deliverables (only if Milestone C step 9 passes)

12. **Create the `model` asset.** Following the exact packaging convention of
    `tasks/t0010_stage2_safeguarded_training/assets/model/kokoro-v10-best/`: create
    `assets/model/kokoro-v11-best/details.json` (per `meta/asset_types/model/specification.md`,
    `spec_version: "2"`, `framework: "pytorch"`, `base_model: "kokoro-82m"`,
    `base_model_source: "t0008_tts_eval_harness_baselines"`,
    `training_task_id: "t0014_v11_decoder_fix_retrain"`, `hyperparameters` block mirroring v10's but
    with `joint_epoch: 30`, `diff_epoch: 10`, `epochs: 50`,
    `first_stage_path: "epochs_2nd_00020.pth (yl4579/StyleTTS2-LibriTTS)"`, `training_metrics` block
    including `best_val_loss`, `health_gate_events`, and the gate verdict from step 9 as
    `audible_speech_gate: "pass"`), `description.md` (all mandatory sections: Metadata, Overview ≥80
    words, Architecture, Training, Evaluation, Usage Notes, Main Ideas ≥3 bullets, Summary ≥100
    words — mirror v10's description.md structure), and `files/` containing the extracted state dict
    (`kokoro-v11-best.pth`, DVC-tracked per CLAUDE.md's DVC workflow — `dvc add` then `dvc push`
    before this task's PR merges) and `files/config_david_v11.yml`. Satisfies REQ-7.

13. **Verify the model asset.**
    `uv run python -m arf.scripts.verificators.verify_model_asset --task-id t0014_v11_decoder_fix_retrain kokoro-v11-best`
    — expect 0 errors. Satisfies REQ-7.

## Remote Machines

**Required.** GPU training and inference run on `LLM-T1-NC80` (Azure ML, 2×H100 SXM5,
`gpu_class: H100`, `provider: azure_ml` — the default per
`arf/specifications/remote_machines_specification.md`'s routing rule, matching
`project/azure_vm.json`'s single pool entry, `hourly_cost_usd: 13.96`). This is the same VM t0010
used. Estimated runtime: ~5.85 hours billed (see Cost Estimation) — setup/preflight (~0.75h), tensor
pre-flight check (~0.1h, CPU-only but budgeted on the VM), training (~3.0h active compute for 50
epochs on 1,531 clips), audible-speech gate + metrics (~0.5h), contingency buffer (~1.5h). VRAM
requirement: same as v10 (Kokoro-82M Stage 2, `batch_size: 8`, unchanged) — 2×H100's 160GB total
VRAM is far more than needed but is the only pool entry available. Follow
`arf/skills/setup-remote-machine/SKILL.md` in full, with the Lesson 10 (persistent-storage symlink)
and Lesson 11 (`loginctl enable-linger` + `tmux`) preflight guarantees already enforced by
`azure_ml_vm.acquire()`'s `remote_preflight.sh` call, re-verified by hand per Milestone B step 5.
Deploy `arf/scripts/utils/idle_watchdog.sh` before training starts (Lesson 8), per CLAUDE.md's
"Deploying the watchdog" section. Tear down promptly after Milestone C/D complete via the standard
teardown protocol (`arf/skills/setup-remote-machine/SKILL.md` "Teardown Protocol" section) —
`azure_ml_vm.py`'s `teardown(task_id, deallocate=True)` — and confirm via
`verify_machines_destroyed.py` before this task can be marked complete.

## Assets Needed

* `tasks/t0010_stage2_safeguarded_training/code/train_second_v10.py` — base training script to copy
  forward and adapt (Milestone A step 2).
* `tasks/t0009_stage2_training_failure_forensics/code/{checkpoint_manager,health_gates,jsonl_logger,run_config}.py`
  — imported unchanged as the safeguard library (already wired into `train_second_v10.py`).
* `tasks/t0012_v5_corpus_normalize_and_reaudit/data/train_list_v5_normalized_clean.txt` — the
  1,531-clip normalized training manifest (Milestone A step 1).
* `tasks/t0013_v10_synthesis_quality_forensics/code/{audio_quality_check,infer_styletts2,inspect_checkpoint,random_decoder_probe,score_speaker_sim,paths}.py`
  — copied and adapted for v11 throughout Milestones A and C.
* `tasks/t0008_tts_eval_harness_baselines/code/scoring.py` (`build_centroid`, `compute_speaker_sim`
  logic, already re-derived by t0013's `score_speaker_sim.py`) and
  `tasks/t0008_tts_eval_harness_baselines/data/11labs_david/` (1,358-clip reference corpus for the
  `speaker_sim` centroid).
* External: `yl4579/StyleTTS2-LibriTTS` (`Models/LibriTTS/epochs_2nd_00020.pth`,
  `Models/LibriTTS/config.yml`), downloaded via `git-lfs clone` from Hugging Face — free, MIT code
  license, pretrained-weight disclosure/consent terms noted in the Approach section.
* `data/val_list.txt` (the unchanged `val_96` held-out set) — never trained on.

## Expected Assets

Matches `task.json`'s `expected_assets: {"model": 1}` **conditionally** — only if the audible-speech
gate (Milestone C step 9) passes:

* **`model` asset**: `assets/model/kokoro-v11-best/` — the corrected Stage-2 checkpoint
  (decoder-init fix + 1,531-clip corpus + 50/10/30 epoch schedule), packaged per
  `meta/asset_types/model/specification.md` with `details.json`, `description.md`, and
  `files/kokoro-v11-best.pth` (DVC-tracked) + `files/config_david_v11.yml`.

If the gate fails, no `model` asset is created — `results/results_summary.md` documents the failure
honestly instead (per REQ-7's explicit prohibition on fabricating success), and this deviation from
`task.json`'s `expected_assets` is itself the correct, gate-driven outcome, not a plan defect.

## Time Estimation

* **Research** (steps 4-6, already complete before this plan): ~done, not re-estimated here.
* **Milestone A** (fix + pre-flight verification, mostly CPU/local + one VM-side checkpoint
  download): ~1-1.5 hours wall clock (includes the 771MB checkpoint download).
* **Milestone B** (VM provisioning + training): ~1 hour provisioning/monitoring overhead + ~3.0
  hours active training compute ≈ 4 hours wall clock, plus periodic monitoring checks every 15-20
  minutes throughout.
* **Milestone C** (audible-speech gate + metrics): ~0.5-1 hour wall clock.
* **Milestone D** (model asset packaging, conditional on gate pass): ~0.5-1 hour wall clock.
* **Teardown + verification**: ~0.25 hour.
* **Total estimated wall clock**: ~6-8 hours, close to the ~5.85 billed-hour compute estimate since
  most steps are sequential and VM-bound.

## Risks & Fallbacks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Root disk (`/dev/root`) fills up during training, repeating t0010's exact failure (blocked teardown, no torch env, eval deferred with null metrics) | Medium — this VM already did this once under similar training workload | High — blocks the audible-speech gate, the task's core deliverable | Milestone B step 6's explicit `df -h /` polling every 15-20 min (proactive, not discovered at teardown); 85% threshold triggers cache cleanup (`~/.cache/pip`, `~/.cache/huggingface`, tensorboard event files) before reaching 100%; checkpoints write only to `/mnt/cache/persist` (verified symlink, Lesson 10), never bare `/mnt` or `/` |
| Pretrained LibriTTS checkpoint's `decoder` does not actually match `config_david_v10.yml`'s hifigan block at the tensor level (architecture drift not caught by research) | Low — t0013 already validated this exact checkpoint loads with 0 missing/0 unexpected keys via the same harness | High — the entire fix depends on this; training would silently repeat v10's bug | Milestone A step 3's mandatory pre-flight `inspect_checkpoint.py` run, **before any GPU spend**; hard STOP-and-fallback-to-Approach-2 condition explicitly defined, not a soft warning |
| Idle VM billing recurs (watchdog gap, SSH-session-scope kill per Lesson 11, or a missed teardown) | Medium — this exact failure cost t0010 $272.78 despite a watchdog existing in the framework | High (cost) | Watchdog deployed **before** training starts (not after), per Lesson 8's "every GPU machine, not only ones that will pause"; `loginctl enable-linger` + `tmux` both verified by hand (Milestone B step 5) in addition to the automated `remote_preflight.sh` check; hard $300 spend ceiling with mandatory `intervention/` file if breached |
| Audible-speech gate fails even after the decoder-init fix (e.g., MPD/discriminator wiring issue, under-training despite the larger epoch budget) | Medium — this is a genuinely new architecture/corpus combination, and `research/research_summary.md` risk notes flag `HealthGate` thresholds were calibrated on a 6x-smaller run | High for this task's success claim, but the correct outcome is honest failure reporting, not silent success | REQ-6/REQ-7 explicitly define the fail path: stop, document with the same evidence standard as `v10_diagnosis.md`, do not fabricate deliverables, file a follow-up suggestion (e.g., check MPD wiring per HiFi-GAN's 1.82 MOS ablation finding) |
| `HealthGate` thresholds (calibrated on a 250-clip run) fire spuriously on the 1,531-clip run due to different loss magnitudes, aborting a healthy training run | Medium | Medium — wastes GPU time on a false-positive abort, requires a restart | Milestone B step 7 requires inspecting the specific fired metric in `metrics.jsonl` before restarting (not blind retry); if the same gate fires 3 times against what otherwise looks like healthy convergence, write an `intervention/` file recommending a threshold-widening follow-up task rather than silently disabling the gate |
| `speaker_sim` centroid corpus (`11labs_david`, 1,358 clips) or the `.venv-styletts2` environment is unavailable/broken on re-creation | Low — t0013 already built and validated this exact environment in this repo | Low — only affects REQ-9's corroborating metrics, not the audible-speech gate (REQ-6, the actual pass/fail criterion) | If `score_speaker_sim.py` fails, omit `speaker_sim` from `results/metrics.json` with an explicit documented reason (mirroring the plan's already-explicit `ttfb_ms` omission pattern) rather than blocking the whole task on a non-critical metric |

## Verification Criteria

* `uv run python -u -m arf.scripts.verificators.verify_plan t0014_v11_decoder_fix_retrain` — expect
  0 errors (run at the end of this planning step, before handoff).
* After Milestone A step 3:
  `uv run python tasks/t0014_v11_decoder_fix_retrain/code/inspect_checkpoint.py` produces
  `results/checkpoint_forensics_v11.md` containing `classification: "hifigan"` for the LibriTTS
  checkpoint's decoder — confirms REQ-2 before any GPU spend.
* After Milestone B: `wc -l tasks/t0014_v11_decoder_fix_retrain/data/run_v11/metrics.jsonl` returns
  a nonzero count with at least one `val_loss` record per completed epoch, and
  `grep -c '"gate_fired"' data/run_v11/metrics.jsonl` reports the exact count of any HealthGate
  firings (expected 0, matching v10's stability, unless a genuine divergence is documented) —
  confirms REQ-4, REQ-5.
* After Milestone C step 9: `cat results/audio_quality_v11.json` shows `is_likely_noise: false` (or,
  if the gate failed, `results/v11_gate_verdict.md` explicitly states FAIL with evidence and
  `intervention/audible_gate_failed.md` exists) — confirms REQ-6 either way.
* If the gate passed:
  `uv run python -m arf.scripts.verificators.verify_model_asset --task-id t0014_v11_decoder_fix_retrain kokoro-v11-best`
  — expect 0 errors — confirms REQ-7.
* `uv run python -m arf.scripts.verificators.verify_task_metrics t0014_v11_decoder_fix_retrain` —
  expect 0 errors, confirming `results/metrics.json`'s explicit-variant format only uses registered
  metric keys (`rtf`, `speaker_sim`) and the deliberate `ttfb_ms` omission does not trip a warning —
  confirms REQ-9.
* `uv run python -m arf.scripts.verificators.verify_machines_destroyed --task-id t0014_v11_decoder_fix_retrain`
  — expect 0 errors, confirming `LLM-T1-NC80` was torn down and `results/costs.json` records the
  actual total — confirms REQ-4 and REQ-8.
* Requirement-coverage check: every `REQ-*` ID listed in `## Task Requirement Checklist` above
  appears at least once in `## Step by Step`'s "Satisfies REQ-*" annotations — manually confirmed in
  this plan (REQ-1 through REQ-9 all appear at least once above) and re-checked by the
  implementation agent against the actual produced outputs before claiming the task complete.

## Rejection Criteria

This task does not run a paired-bootstrap benchmark or a high-volume request protocol, so Lesson 3's
`successful_requests / total_requests < 0.8` threshold does not apply directly as a formula, but the
equivalent principle applies to its own pass/fail signal:

* **The audible-speech gate result (`is_likely_noise`) is the sole, pre-registered pass/fail
  condition for this task's deliverables (REQ-6/REQ-7).** No other metric — not `val_loss`, not
  `speaker_sim`, not the absence of `HealthGate` firings — may be substituted as evidence of success
  if the gate fails. This mirrors the exact failure this task exists to correct: `kokoro-v10-best`
  shipped with a healthy-looking `val_loss` curve and a non-trivial `speaker_sim` score despite
  producing pure noise.
* If the tensor-level pre-flight check (Milestone A step 3) does not confirm a clean HiFi-GAN-shaped
  match for `decoder`, training must not proceed under the assumption it will "probably be fine" —
  the fallback path (Approach 2, random-init) is a distinct, documented decision requiring an
  `intervention/` file, not a silent substitution.
* If GPU spend for this task exceeds $300 (the pre-registered hard stop in Cost Estimation) without
  a passing audible-speech gate, the task must halt, write an `intervention/` file, and treat any
  partial results as inconclusive rather than reporting a checkpoint that was never gate-validated.
* If `results/audio_quality_v11.json` and `results/v11_gate_verdict.md` are absent from the final
  task folder, the task cannot be marked `completed` regardless of any other artifact present — the
  audible-speech check is not optional evidence, it is the task's defining success condition per its
  own `short_description`.
