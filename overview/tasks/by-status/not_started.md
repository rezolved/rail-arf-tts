# ⏹ Tasks: Not Started

2 tasks. ⏹ **2 not_started**.

[Back to all tasks](../README.md)

---

## ⏹ Not Started

<details>
<summary>⏹ 0009 — <strong>Stage 2 training failure forensics and safeguards</strong></summary>

| Field | Value |
|---|---|
| **ID** | `t0009_stage2_training_failure_forensics` |
| **Status** | not_started |
| **Effective date** | — |
| **Dependencies** | [`t0005_kokoro_v5_stage2_train`](../../../overview/tasks/task_pages/t0005_kokoro_v5_stage2_train.md), [`t0006_kokoro_v5_stage2_subset`](../../../overview/tasks/task_pages/t0006_kokoro_v5_stage2_subset.md) |
| **Expected assets** | 1 answer, 1 library |
| **Source suggestion** | — |
| **Task types** | [`data-analysis`](../../../meta/task_types/data-analysis/) |
| **Task page** | [Stage 2 training failure forensics and safeguards](../../../overview/tasks/task_pages/t0009_stage2_training_failure_forensics.md) |
| **Task folder** | [`t0009_stage2_training_failure_forensics/`](../../../tasks/t0009_stage2_training_failure_forensics/) |

# Stage 2 Training Failure Forensics and Safeguards

## Motivation

Four training tasks (t0001, t0004, t0005, t0006; 19+ launches, ~$440 of GPU) have not produced
a checkpoint with clean audio. Each task fixed the symptom in front of it and moved on:

* t0001 (v4) diverged at epoch 7. t0003 traced it to 21% grapheme-poisoned manifests.
* t0005 (v5, clean data) crashed five times, then diverged after the GAN switched on
  (`acoustic_norm` 10 → 671, val 0.848 → 2.07). Seven patches were stacked, including
  grad-skip guards and an `exp` clamp in `istftnet.py` that "stops the crash; does NOT stop
  the divergence".
* t0006 stopped divergence with `lambda_gen=0.05`, but changed six variables at once (data
  size, `multispeaker`, Stage 1 checkpoint, `joint_epoch`, `lambda_gen`, LR 3e-5 → 1e-4). The
  audio is still noisy, and the cause is still unknown.

The researcher asked for a systematic review before any more training: read every log, check
the data, check the pipeline, check that checkpoints are saved and loaded correctly, and build
protection so that the next failure is caught early, rolled back from a checkpoint, and
explainable from the logs.

This task runs **no new training** (see Compute for the one optional smoke test).

## Key Questions

1. At which step does each failed run first go wrong, and which logged signal moves first (Dur
   loss, `acoustic_norm`, generator/discriminator loss, grad norms, skip counts)?
2. What separates v3's successful Stage 2 from every failure, once confounds are listed
   explicitly? t0003 found v3's success depended on which `first_stage.pth` happened to be in
   the log directory.
3. Is the training data consistent: sample rate, duration distribution, clipping, silence,
   text length vs audio length, and how does the 1557-clip v5 set differ from v3's 266 clips?
4. Is the training pipeline correct: does `train_second_patched.py` match upstream StyleTTS2
   and v3's patch, do any of the seven t0005 patches mask a real problem, and are DataParallel
   and the loading path (`load_only_params`, `ignore_modules`, `multispeaker`) doing what the
   configs assume?
5. Are checkpoints saved, labeled and loaded correctly? Known inconsistencies to resolve:
   * t0004: `epoch_1st_00007.pth` labeled "epoch 10" (zero-indexed it looks like epoch 8, val
     0.757).
   * t0005: best checkpoint reported as epoch 3 in results but epoch 2 in `logs/README.md`.
   * t0006: `epochs_2nd: 10` instead of the intended 15; v6d metrics report `acoustic_norm`
     8.51 while the text says v3's Stage 1 dropped it to 0.36.
   * Only the top-2 checkpoints by `val_loss` are kept, although `val_loss` is not proven to
     track audio quality.
6. Is `val_loss` 0.506 (v3) even comparable to the v5 numbers — same val set, same loss
   definition?
7. Why did v3's audio go from exploded durations (`v3/audio/v3/`) to clean (`v3b/`) six
   minutes apart with no retraining (open question from t0003)?

## Scope

### 1. Log inventory and collection

Almost no training logs are in git: `.gitignore` excludes `*.log`, and the only committed
training log is `tasks/t0006_kokoro_v5_stage2_subset/logs/run03_v6c_v3_stage1.log`. t0005's
`logs/README.md` describes `crash_01`-`crash_06`, `run06_divergence_full.log` and
`stage2_v5.log`, none of which are in the repo. t0005 also notes the launch command `rm -f`s
the remote log and the config file was edited in place between launches.

Collect every log that still exists, per run: t0001 (`stage2_v4.log`), v3
(`stage2_v3_frozen_lm.log`, `stage2_clean.log`, `stage2_dirty.log`, Stage 1 logs), t0004 Stage
1, t0005 launches 1-6, t0006 runs 01-04, plus any TensorBoard event files and configs. Likely
locations: LLM-T1-NC80 (`/mnt/kikiri-tts/StyleTTS2/logs/`, `kokoro-finetune/`),
`rezolved/rail-benchmarks` (`kokoro-finetune/`), and the original author's machine. LESSONS
Lesson 10: `/mnt` on Azure ML is ephemeral, so some logs may be gone — record what is missing
rather than guessing.

Store logs in this task's `data/logs/<run_id>/` (text logs are small; force-add past `*.log`
or DVC-track if large). Produce an inventory table: run_id, task, log present (y/n), config
present, Stage 1 checkpoint used (with SHA-256 where recoverable), data list, outcome.

### 2. Per-run timelines

Parse every loss field per step into one tidy table (run_id, epoch, step, field, value). Plot
each run aligned on `joint_epoch`. Mark the first anomalous step per run and the leading
indicator.

### 3. Confound table

For every run, list the effective settings: data list and size, `multispeaker`,
`first_stage_path` (and hash), `joint_epoch`, `lambda_gen`, `lambda_slm`,
`lr`/`ft_lr`/`bert_lr`, `batch_size`, GPU count and parallelism mode, `train_LM`, which of the
seven patches were active. Reconstruct t0005's per-launch configs from log headers and git
history, since the file was edited in place.

### 4. Data audit

On the v5 train/val lists and v3's 266-clip list: sample rate, channels, duration histogram,
peak/clipping rate, leading/trailing silence, loudness (LUFS), frames per phoneme outliers,
OOV or vocab-invalid tokens, train/val overlap, and whether v3's val set equals val_96.

### 5. Pipeline audit

Diff `tasks/t0005_kokoro_v5_stage2_train/code/train_second_patched.py` against upstream
StyleTTS2 `train_second.py` and against v3's
`tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/train_second_patch.diff`. Review each
of the seven t0005 patches and state whether it fixes a cause or hides one. Compare against
the published StyleTTS2 fine-tuning recipe (`train_finetune.py`, which starts from a
pretrained multi-speaker model) and against public Kokoro fine-tuning recipes. Check that mel
extraction and the 24 kHz setup match what Kokoro's decoder expects.

### 6. Checkpoint audit

Map every checkpoint file to its epoch, step, `val_loss`, config and launch. Resolve the
inconsistencies in Key Question 5. Verify from code how "best" checkpoints are selected and
overwritten, and whether a healthy checkpoint from before each divergence still exists. Verify
with hashes which `first_stage.pth` each Stage 2 run actually loaded.

### 7. Safeguards (library asset)

Ship a small, tested patch plus a monitor that future training tasks reuse:

* Save a checkpoint every epoch (and every N steps near `joint_epoch`), with a retention
  policy that always keeps the last healthy one. Never select only by `val_loss`.
* Write a structured JSONL metrics log (one record per step: all losses, grad norms, skip
  count, LR, epoch) next to the text log. Never `rm -f` or overwrite logs between launches;
  one directory per launch, with the resolved config and checkpoint hashes saved alongside.
* Health gates checked live, with thresholds derived from Step 2 (for example Dur loss on the
  first Stage 2 step near v3's 0.81, `acoustic_norm` ceiling, val spike limit, consecutive
  skips). When a gate fires: stop, record why, and point to the last healthy checkpoint to
  resume from.
* **Offline replay test**: run the gates over the collected logs. They must fire on t0001,
  t0005 run06 and t0006 run01/run02 at or before divergence, and must not fire on v3's
  successful run.

### 8. Answer asset

Answer: "Why does Kokoro Stage 2 fine-tuning break in this project, and how should the next
run be configured and monitored?" Root causes ranked with evidence and confidence, what is
still unknown, and a single recommended configuration for the next training task, changing one
variable at a time relative to the closest known-good run.

## Compute and Budget

* Analysis runs locally or on CPU: $0.
* VM time only to retrieve logs and checkpoint hashes from LLM-T1-NC80: target ≤ 30 min (~$7),
  with teardown through `setup-remote-machine`.
* Optional: one short GPU smoke test (≤ 1 h, ~$14) to confirm the patched training script
  writes the JSONL log, saves per-epoch checkpoints and triggers a gate. Only if the offline
  replay cannot exercise the code path.
* Planned total: ≤ $25. Write `results/costs.json` with `total_cost_usd` (t0001-t0006 used the
  wrong key and are invisible to `aggregate_costs`).

## Expected Outputs

* **Answer asset** (see Scope 8).
* **Library asset**: safeguard patch, JSONL metrics logger, health-gate monitor, log parser,
  and a `test_*.py` offline replay test.
* `results_detailed.md` tables: log inventory, confound table, checkpoint map, data audit
  summary, patch review (patch, fixes cause or hides symptom, keep/remove).
* Charts in `results/images/`, embedded in `results_detailed.md`:
  * Loss timelines per run aligned on `joint_epoch` (x = step, y = loss, one panel per field)
    — Q1.
  * `acoustic_norm` and grad norm per run on log scale — Q1.
  * Duration and loudness histograms, v5 vs v3 — Q3.
* Suggestions for the next training task.

## Dependencies

* `t0005_kokoro_v5_stage2_train`, `t0006_kokoro_v5_stage2_subset` — the runs under audit.
* Independent of t0008; the two can run in parallel. Audio-quality verdicts for checkpoints
  come from t0008's harness when available, but this task does not wait for it.

</details>

<details>
<summary>⏹ 0008 — <strong>TTS evaluation harness and baselines</strong></summary>

| Field | Value |
|---|---|
| **ID** | `t0008_tts_eval_harness_baselines` |
| **Status** | not_started |
| **Effective date** | — |
| **Dependencies** | [`t0006_kokoro_v5_stage2_subset`](../../../overview/tasks/task_pages/t0006_kokoro_v5_stage2_subset.md) |
| **Expected assets** | 1 library |
| **Source suggestion** | — |
| **Task types** | [`tts-benchmark-run`](../../../meta/task_types/tts-benchmark-run/), [`baseline-evaluation`](../../../meta/task_types/baseline-evaluation/) |
| **Task page** | [TTS evaluation harness and baselines](../../../overview/tasks/task_pages/t0008_tts_eval_harness_baselines.md) |
| **Task folder** | [`t0008_tts_eval_harness_baselines/`](../../../tasks/t0008_tts_eval_harness_baselines/) |

# TTS Evaluation Harness and Baselines

## Motivation

After six tasks (t0001-t0006) the project has not measured a single one of its success
criteria. Every training attempt was judged by StyleTTS2 `val_loss` and by listening. The
success criteria in `project/description.md` are speaker similarity (GE2E cosine ≥ 0.85
against ElevenLabs David) and TTFB ≤ 300 ms. There is no ElevenLabs baseline, no base-Kokoro
baseline, and no speaker-similarity code anywhere in the repo.

t0003 also showed that Stage 1 `val_loss` does not predict success (0.770 preceded the working
v3 Stage 2, 0.545 did not), so `val_loss` alone cannot be trusted as a quality signal.

This task builds the measurement tool once, as a reusable library, and uses it to answer
research questions 1 and 2 and to place every existing checkpoint on the same scale. It runs
no training.

## Key Questions

1. What are speaker_sim, TTFB (p50/p95/p99) and RTF for ElevenLabs David on ≥ 50 filler
   prompts?
2. What do base Kokoro-82M British male voices score on the same prompts?
3. How close is the v3 shipped bundle (the only checkpoint with clean audio) to speaker_sim ≥
   0.85?
4. Do the t0005 and t0006 best checkpoints score measurably worse than v3, i.e. does the
   harness detect the "noisy audio" that listening detected?
5. Is speaker_sim alone enough to reject broken audio, or are the duration and intelligibility
   sanity metrics needed as well?

## Systems to Evaluate

Every system is scored on the same prompt sets with the same harness:

1. **ElevenLabs David** — API (voice ID David). TTFB and RTF measured live through the
   streaming API; speaker_sim computed on held-out reference clips (see Scoring below).
2. **Kokoro-82M base, `bm_george`** — stock voicepack.
3. **Kokoro-82M base, `bm_lewis`** — stock voicepack.
4. **Kokoro-82M base decoder + v3 David voicepack** — `david_v3_best_voicepack.pt` with stock
   decoder/predictor. Tells us how much of the voice a voicepack alone captures, with no
   fine-tuned weights.
5. **v3 shipped bundle** — `david_v3_best_decoder_kokoro.pth` + `david_v3_best_voicepack.pt`
   from `tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/best/` (DVC).
6. **t0006 v6d epoch 6** —
   `tasks/t0006_kokoro_v5_stage2_subset/results/checkpoints/v6d/epoch_2nd_00006.pth`, packaged
   with t0002's five-module extraction (`extract_decoder_generic.py`).
7. **t0005 run06 best** — `tasks/t0005_kokoro_v5_stage2_train/results/checkpoints/` best
   checkpoint, same packaging. Note: t0005's results say epoch 3 but `logs/README.md` says
   epoch 2; record which file was actually scored.

Floor control: one clearly different Kokoro voice (e.g. `af_heart`, female American) to show
what a "wrong speaker" score looks like, so 0.85 can be interpreted.

All Kokoro arms must synthesize through
`tasks/t0003_kokoro_v5_phoneme_data/code/build_pipeline.py` (British `lang_code="b"` plus
brand lexicon), not a bare `KPipeline` — t0003 showed accent and lexicon mismatches silently
change the token stream.

## Prompt Sets and Data

* **val_96**: the 96 held-out clips (`data/v4/val/`, text from
  `tasks/t0003_kokoro_v5_phoneme_data/results/v5/val_list.txt`). Real David audio is available
  for each prompt, so per-prompt paired comparison is possible. Never train on these.
* **Fillers**: ≥ 50 prompts sampled with a fixed seed (seed=42) from the 1358-clip ElevenLabs
  filler corpus. The corpus currently lives only on the VM
  (`kokoro-finetune/data/11labs_david/`, see `overview/datasets/fillers_1358.md`) and is not
  DVC-tracked. First step: copy it off the VM, `dvc add` it under this task's `data/`, and
  `dvc push`. Beware LESSONS Lesson 10 — `/mnt` on Azure ML is ephemeral; confirm the corpus
  still exists before planning around it.

## Scoring

* **speaker_sim** (registered metric): resemblyzer GE2E cosine between each synthesized clip
  and the mean embedding of the ElevenLabs David reference set. Split the reference clips into
  a centroid half and a held-out half with a fixed seed; the ElevenLabs arm is scored on the
  held-out half so it is not compared against itself. Report mean, std and per-clip
  distribution.
* **ttfb_ms** (registered metric): for ElevenLabs, wall time from request to first streamed
  audio byte. For Kokoro, wall time from pipeline call to the first audio chunk yielded.
  Follow LESSONS Lesson 1 (discarded warmup requests before measuring) and Lesson 4 (record
  torch/CUDA/kokoro versions and GPU at measurement time). Report p50/p95/p99.
* **rtf** (registered metric): synthesis wall time divided by output audio duration.
* **duration_ratio** (sanity): synthesized duration divided by reference clip duration.
  Catches the 10× duration explosion seen in t0002 before anyone listens.
* **WER** (sanity): transcribe synthesized audio with a local Whisper model and compare
  against the prompt text. Catches garbled or noisy audio that could still score a plausible
  speaker_sim.

## Compute and Budget

* Quality metrics (speaker_sim, duration_ratio, WER) can run on CPU or a single GPU.
* The success criterion defines TTFB "local inference, H100". Kokoro TTFB/RTF should be
  measured on LLM-T1-NC80 in one short window (target ≤ 1.5 h, ~$21), with setup and teardown
  through the `setup-remote-machine` skill. A CPU TTFB number may be reported in addition,
  clearly labeled.
* ElevenLabs API: ~150 streaming requests of short filler text, well under $5.
* **Budget note**: the project budget was raised to $5000 in t0007. `aggregate_costs` still
  reports $0 spent because t0001-t0006 write `total_usd` instead of `total_cost_usd`; real
  spend so far is ~~$370 recorded plus unrecorded t0006 GPU time (~~$70). This task must write
  `total_cost_usd`. Planned total for this task: ≤ $30.

## Expected Outputs

* **Library asset**: the evaluation harness (prompt-set loader, per-system synth adapters,
  scoring, report writer), reusable by every future training task.
* `results/metrics.json` with speaker_sim, ttfb_ms, rtf per system (as variants).
* Tables in `results_detailed.md`:
  * Main table: rows = systems 1-7 + floor control; columns = speaker_sim mean/std, ttfb_ms
    p50/p95/p99, rtf, duration_ratio median, WER; plus delta vs ElevenLabs.
  * Per-prompt-set breakdown (val_96 vs fillers).
* Charts in `results/images/`, embedded in `results_detailed.md`:
  * speaker_sim distribution per system (box plot; y = GE2E cosine, dashed line at 0.85) —
    answers Q1-Q4.
  * TTFB CDF per system (x = ms, dashed line at 300 ms) — answers Q1-Q2.
  * speaker_sim vs WER scatter per clip — answers Q5.
* A few synthesized examples per system (DVC-tracked, not git).
* A clear statement: which systems (if any) already pass each success criterion.

## Dependencies

* `t0006_kokoro_v5_stage2_subset` — provides the v3 reference bundle and the v6d checkpoint.
* Independent of t0009; the two can run in parallel.

</details>
