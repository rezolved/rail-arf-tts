# ⏳ Tasks: In Progress

1 tasks. ⏳ **1 in_progress**.

[Back to all tasks](../README.md)

---

## ⏳ In Progress

<details>
<summary>⏳ 0009 — <strong>Stage 2 training failure forensics and safeguards</strong></summary>

| Field | Value |
|---|---|
| **ID** | `t0009_stage2_training_failure_forensics` |
| **Status** | in_progress |
| **Effective date** | 2026-09-14 |
| **Dependencies** | [`t0005_kokoro_v5_stage2_train`](../../../overview/tasks/task_pages/t0005_kokoro_v5_stage2_train.md), [`t0006_kokoro_v5_stage2_subset`](../../../overview/tasks/task_pages/t0006_kokoro_v5_stage2_subset.md) |
| **Expected assets** | 1 answer, 1 library |
| **Source suggestion** | — |
| **Task types** | [`data-analysis`](../../../meta/task_types/data-analysis/) |
| **Start time** | 2026-09-14T15:28:39Z |
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
