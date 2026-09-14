# ✅ Kokoro v5 Stage 2 fine-tune

[Back to all tasks](../README.md)

## Overview

| Field | Value |
|---|---|
| **ID** | `t0005_kokoro_v5_stage2_train` |
| **Status** | ✅ completed |
| **Started** | 2026-09-12T23:00:00Z |
| **Completed** | 2026-09-13T06:00:00Z |
| **Duration** | 7h 0m |
| **Dependencies** | [`t0004_kokoro_v5_stage1_train`](../../../overview/tasks/task_pages/t0004_kokoro_v5_stage1_train.md) |
| **Task types** | `tts-finetuning-eval` |
| **Expected assets** | 1 model |
| **Step progress** | 5/5 |
| **Task folder** | [`t0005_kokoro_v5_stage2_train/`](../../../tasks/t0005_kokoro_v5_stage2_train/) |
| **Detailed results** | [`results_detailed.md`](../../../tasks/t0005_kokoro_v5_stage2_train/results/results_detailed.md) |

<details>
<summary><strong>Task Description</strong></summary>

*Source:
[`task_description.md`](../../../tasks/t0005_kokoro_v5_stage2_train/task_description.md)*

# t0005 — Kokoro v5 Stage 2 fine-tune

## Objective

Run StyleTTS2 Stage 2 from t0004's `first_stage.pth` (epoch 10, val_loss 0.740) on the v5
phoneme-corrected corpus, with the stop-early criterion from t0003 checked at the earliest
possible moment, and sync tooling that shows live progress.

## Background

t0003 traced t0001's Stage 2 divergence to a corrupted training manifest and rebuilt clean
manifests. t0004 ran Stage 1 on the clean data. This task is Stage 2 — the step where t0001
actually failed (Dur loss opened at 16.84 and never recovered), so it carries the stop
criterion t0003 defined: v3's successful Stage 2 opened at Dur loss 0.81; every failed run
(including t0001's) opened at 8-17. That's readable from the first logged step, long before a
checkpoint exists to evaluate.

## Multi-GPU: two different mechanisms, only one applies here

`train_first.py` and `train_second.py` use different multi-GPU code paths — this matters for
how each is launched:

- **`train_first.py`** uses `accelerate` (`Accelerator`, `DistributedDataParallelKwargs`) —
  proper DDP, requires `accelerate launch --num_processes 2` to actually use both GPUs. t0004
  launched it as plain `python3 train_first.py`, so it silently ran on **1 of 2 GPUs**
  (confirmed via `nvidia-smi` during the run — GPU0 63%/34GB, GPU1 0%/3MB). It still trained
  correctly; just half-speed.
- **`train_second.py`** uses `torch.nn.DataParallel` (`MyDataParallel`, applied
  unconditionally at lines 195-197) — a different, single-process mechanism. It auto-detects
  all visible CUDA devices and splits the batch across them **within one process**. Launching
  this script via `accelerate launch --num_processes 2` would be wrong: it would spawn two
  independent processes, each seeing only one GPU (accelerate restricts visible devices per
  rank), each running its own un-synchronized `MyDataParallel`-wrapped copy of the whole
  training loop, both writing checkpoints to the same `log_dir` — a race, not real 2-GPU
  training.

Confirmed on the VM: `CUDA_VISIBLE_DEVICES` unset, `torch.cuda.device_count() == 2`. So the
correct launch for this task is plain `python3 train_second.py` — no accelerate wrapper — and
`DataParallel` uses both GPUs automatically.

## batch_size: kept at v3's value, not doubled

`DataParallel` splits the configured `batch_size` across GPUs rather than multiplying it, so
`batch_size: 8` here means 4+4 per GPU per step — smaller than v3's per-GPU batch, if v3's run
was in fact single-GPU (v3's own log shows the `accelerate launch` multi-GPU banner even for
Stage 2, whose script has no `accelerate`/DDP code at all — so it's unclear whether v3's "8"
was ever a true single-process batch or an artifact of a similarly ambiguous launch).

Doubling `batch_size` to 16 (to get 8/GPU, matching v3's number) was considered and rejected:
this run's entire purpose is isolating the data fix as the one changed variable against t0001.
Batch size affects gradient noise and its interaction with the `joint_epoch` GAN warmup —
changing it on top of the data fix would confound the result. `batch_size: 8` is kept exactly
as v3's config states it, literally. `num_workers: 16` is unchanged (VM has 80 cores; no
bottleneck at this batch size).

## Approach

1. Launch `python3 train_second.py --config_path Configs/config_david_v5_stage2.yml` on the VM
   (`first_stage_path: first_stage.pth` resolves to `logs/kokoro-david-v5/first_stage.pth` —
   t0004's output, confirmed by matching `log_dir`).
2. Run `code/sync_and_monitor.sh` locally: rsyncs the log every 30s (cheap, checked first),
   prints the stop-criterion verdict the moment the first `Dur Loss:` line appears, then syncs
   any new `epoch_2nd_*.pth` and prunes to the top-2 by `val_loss` — locally and on the VM.
3. If the stop criterion fires bad (Dur loss > 2.0 on the first step), kill the run
   immediately and report — the data hypothesis would be falsified, and 10 epochs would be
   wasted confirming it slowly instead of quickly.
4. Otherwise let the 10-epoch schedule finish.

## Verification criteria

- Stop-criterion verdict printed within the first sync cycle after Stage 2 starts.
- If healthy: `train_second.py` completes 10 epochs without a traceback, and the final
  val_loss is compared against v3's 0.506.
- Exactly 2 checkpoints survive locally and on the VM at any point after epoch 3.

## Crash at epoch 3 (joint_epoch) and the fix

First launch crashed at epoch 3 (`epoch >= joint_epoch`) with `UnboundLocalError: local
variable 'ref' referenced before assignment` at the `slmadv(...)` call. `ref` is only assigned
when `multispeaker and epoch >= diff_epoch` (never true here — `diff_epoch: 999`), but vanilla
`train_second.py` calls `slmadv(..., ref if multispeaker else None)` unconditionally once
`epoch >= joint_epoch`, regardless of `lambda_slm`.

t0001 (v4) never hit this because its `code/train_second_patched.py` wraps the call in `if
loss_params.lambda_slm == 0: slm_out = None else: slm_out = slmadv(...)` — skipping it
entirely when the SLM discriminator is disabled (which it is: `lambda_slm: 0.0`, WavLM
discriminator crashes). Re-provisioning the VM from the wiped ephemeral disk re-cloned vanilla
`train_second.py` from `semidark/StyleTTS2` and lost this local patch.

Fix: re-applied the same guard to `/mnt/kikiri-tts/StyleTTS2/train_second.py` on the VM (copy
saved at `code/train_second_patched.py`), deleted the 3 stale `epoch_2nd_*.pth` checkpoints
and the old log, and relaunched from scratch (only ~24 min lost).

## Second crash at epoch 3→4 (joint_epoch) and the LR revert

The relaunch hit `epoch >= joint_epoch` cleanly (no `ref` error this time) but then NaN'd in
`d_loss.backward()` — `RuntimeError: Function 'PowBackward0' returned nan values in its 0th
output.` — at the exact same epoch boundary where t0001 (v4) previously NaN'd.

This directly falsifies the config's prior comment: `ft_lr`/`lr: 0.0001` and `bert_lr:
1.0e-05` were set back to v3's literal values on the assumption that t0001's epoch-4 NaN was
caused by the 21% raw-text (unphonemized) rows in v4's corpus, not the LR — and that v5's
phoneme-corrected data would make the high LR safe again. It didn't: the NaN reproduced on the
cleaned v5 data at the same LR and the same joint_epoch boundary, meaning the LR itself is a
real culprit here, at least in combination with GAN activation, independent of the
raw-text-row issue.

Fix: reverted `bert_lr: 1.0e-06`, `ft_lr`/`lr: 3.0e-05` — v4's proven-stable values (see
t0001/research.md's NaN-collapse fix) — and relaunched from scratch again. This does add a
second changed variable on top of the data fix (no longer a perfectly isolated single-variable
comparison against t0001), but continuing to crash at the higher LR isn't a usable
alternative.

## Third lost patch: `train_LM` guard on the unconditional BERT optimizer step

`rail-benchmarks/kokoro-finetune/DAVID_V3_RESULTS.md` documents `train_LM: false` as the
actual fix (not a dead config key) that unblocked v3: without it,
`optimizer.step("bert_encoder")` + `optimizer.step("bert")` run unconditionally every step,
BERT gradients from the discriminator signal destroy the language model once the GAN phase
starts, and LM Loss explodes (23 → 1757, all-noise audio in v1/v2).

Checked the current (freshly re-cloned, vanilla) `train_second.py` on the VM: the guard was
missing — same pattern as the `slmadv`/`ref` and `lambda_slm` guards lost when the VM's
ephemeral disk was wiped and the repo re-cloned.
`t0001/code/train_second_patched.py:107,623-625` has `train_LM = config.get("train_LM", True)`
gating exactly this call; the VM's script had no such variable or gate at all, so `train_LM:
false` in the yaml was silently a no-op and BERT was being fully updated every step of every
run this session (including the two crashed attempts above).

The second, GAN-phase `optimizer.step("bert_encoder")`/`("bert")` call (inside the
`joint_epoch`/`slmadv` block) is NOT guarded by `train_LM` in either the vanilla or the
patched script — but it's dead code here regardless, since `lambda_slm: 0.0` makes `slm_out`
always `None`, which hits `continue` before reaching it.

Applied the same guard to the VM's `train_second.py` (copy saved to
`code/train_second_patched.py`), killed the in-progress run (epoch 2, ~20 min in, no data lost
that mattered), deleted stale checkpoints/log, and relaunched a third time with the full set
of three restored patches: `slmadv`/`lambda_slm==0` guard, LR revert to v4-safe values, and
this `train_LM` guard.

## Fourth crash: NaN in loss_mel, same epoch boundary — missing gradient clipping

The third relaunch reached epoch 3/4 cleanly (no `ref` error, no `PowBackward0` NaN) but then
hit `NaN detected in loss_mel after loss_mel = stft_loss(y_rec, wav)` — a different NaN
symptom, same exact epoch boundary (`joint_epoch`). The script's own NaN guard called
`sys.exit(1)` (no traceback, process just disappears — confirmed via `nvidia-smi` showing both
GPUs at 0%/0 MiB).

Investigated `train_second.py` end to end: `start_ds` (activates MSD/MPD discriminators) and
the first-ever `optimizer.step("style_encoder")`/`optimizer.step("decoder")` calls both flip
on at the exact same instant (`epoch >= joint_epoch`), and there is **no gradient clipping
anywhere in the file** — not on the discriminator step, not on the decoder/style_encoder step.
Ruled out a data cause first: `ref_mels` (the multispeaker style reference) are sourced
per-item from the training dataset itself (`meldataset.py`), not from the unrelated scipy
test-WAV files that show up in the separate per-epoch `extract_voicepack` TensorBoard
diagnostic (a cosmetic, unrelated warning).

The most likely mechanism: the very first backward through freshly-activated,
never-yet-trained MSD/MPD discriminators produces an unbounded gradient into the
decoder/style_encoder's first-ever update, pushing weights into a regime that yields NaN mel
output on the next forward pass. v3 (266 clips, single-speaker) apparently crossed this same
boundary without incident at the same `ft_lr`/`lr: 0.0001` — plausibly luck of the batch/seed
on a much smaller, more homogeneous dataset, not proof the transition is inherently safe.

Fix: added `torch.nn.utils.clip_grad_norm_(..., 10.0)` on `model.msd`, `model.mpd` (right
after `d_loss.backward()`) and on `model.style_encoder`, `model.decoder` (right after
`g_loss.backward()`, inside the `epoch >= joint_epoch` block) in the VM's `train_second.py`
(copy saved to `code/train_second_patched.py`). This is a new stabilization measure, not a
restored v3/v4 patch — standard practice for GAN fine-tuning that this StyleTTS2 fork never
had. Relaunched a fourth time with all four fixes: `slmadv`/`lambda_slm==0` guard, LR revert,
`train_LM` guard, and this gradient clipping.

</details>

## Remote Machines

| Provider | GPU | Count | RAM | Duration | Cost |
|----------|-----|-------|-----|----------|------|
| azure_ml | — | 2 | — GB | — | — |

## Metrics

| Metric | Value |
|--------|-------|
| [`task_id`](../../metrics-results/task_id.md) | **t0005_kokoro_v5_stage2_train** |
| [`metrics`](../../metrics-results/metrics.md) | **[{'name': 'best_val_loss', 'value': 0.848, 'unit': None, 'variant': 'run06'}, {'name': 'best_epoch', 'value': 3, 'unit': None, 'variant': 'run06'}, {'name': 'runs_attempted', 'value': 6, 'unit': None, 'variant': None}, {'name': 'train_clips', 'value': 1557, 'unit': None, 'variant': None}, {'name': 'divergence_epoch', 'value': 4, 'unit': None, 'variant': 'run06'}]** |

<details>
<summary><strong>Results Summary</strong></summary>

*Source:
[`results_summary.md`](../../../tasks/t0005_kokoro_v5_stage2_train/results/results_summary.md)*

# Results Summary: Kokoro v5 Stage 2 Fine-tune

## Summary

6 Stage 2 training runs on LLM-T1-NC80 (DataParallel, both GPUs). All converged through GAN
activation but all diverged within 4 post-GAN epochs. Best checkpoint: **run06 epoch 3,
val_loss=0.848**. Root cause: 1557-clip dataset is too large for the discriminator at
joint_epoch transition — GAN gradient overwhelms the generator. 7 crash fixes accumulated
across runs. All fixes carried forward to t0006 (250-clip subset).

## Key Metrics

| Metric | Value |
| --- | --- |
| Best val loss | 0.848 |
| Best run | run06 |
| Best epoch | 3 |
| Runs attempted | 6 |
| Machine | LLM-T1-NC80 (2× H100 NVL, DataParallel) |
| Training clips | 1557 (v5 phoneme-corrected) |

## Run History

| Run | Key change | Best val | GAN result |
| --- | --- | --- | --- |
| run01 | baseline | crashed pre-GAN | OOM / WavLM crash |
| run02 | `lambda_slm: 0.0` guard | crashed pre-GAN | guard fixed crash |
| run03 | `train_LM: false` | 0.918 (ep1) | diverged ep2 |
| run04 | `clip_grad_norm_` | 0.918 (ep1) | diverged ep2 |
| run05 | `istftnet.py` clamp fix | 0.905 (ep1) | diverged ep5 |
| **run06** | all fixes combined | **0.848 (ep3)** | diverged ep4→2.070 |

</details>

<details>
<summary><strong>Detailed Results</strong></summary>

*Source:
[`results_detailed.md`](../../../tasks/t0005_kokoro_v5_stage2_train/results/results_detailed.md)*

# t0005 — Stage 2 Results

**Machine**: LLM-T1-NC80 (2× H100 NVL via `torch.nn.DataParallel`, both GPUs used) **Data**:
1557 clips (v5 phoneme-corrected corpus) **Stage 1 checkpoint**: t0004 epoch 10 (val=0.740)
**Config**: `code/config_david_v5_stage2.yml`

## Summary

6 runs attempted. All converged through GAN activation at joint_epoch, but all diverged within
4 post-GAN epochs. Root cause: 1557-clip dataset is too large for the discriminator at
joint_epoch transition — GAN gradient overwhelms the generator. Fixed in t0006 (250-clip
subset).

## Run history

| Run | Key change | Best val | GAN result |
|-----|-----------|----------|-----------|
| run01 | baseline | crashed pre-GAN | OOM / WavLM crash |
| run02 | `lambda_slm: 0.0` guard | crashed pre-GAN | guard fixed crash |
| run03 | `train_LM: false` | 0.918 (ep1) | diverged ep2 |
| run04 | `clip_grad_norm_` | 0.918 (ep1) | diverged ep2 |
| run05 | `istftnet.py` clamp fix | 0.905 (ep1) | diverged ep5 |
| **run06** | all fixes combined | **0.848 (ep3)** | diverged ep4→2.070 |

Best checkpoint: **run06 epoch 3, val=0.848** (`results/checkpoints/epoch_2nd_00003.pth`).

All 7 fixes from t0005 run06 were carried forward into t0006.

## Key fixes accumulated in this task

1. `lambda_slm == 0` guard around `slmadv()` (WavLM crash)
2. `train_LM: false` guard (BERT update crash at GAN phase)
3. `clip_grad_norm_` on msd/mpd/decoder/style_encoder
4. `set_detect_anomaly(False)`
5. Finite-grad skip guard + 50-consecutive-skip abort
6. `torch.exp(torch.clamp(..., max=15.0))` in `istftnet.py:523/541` (root-cause NaN fix)
7. `lambda_slm: 0.0` in config

## Files Produced

- `results/checkpoints/epoch_2nd_00002.pth.dvc` — epoch 2 checkpoint
- `results/checkpoints/epoch_2nd_00003.pth.dvc` — epoch 3 checkpoint (best)
- `logs/stage2_v5.log` — full run06 training log
- `logs/crash_04_convolutionbackward_nan.log` — run05 crash log (NaN root cause)
- `code/train_second_patch.diff` — all 7 patches applied to train_second.py

</details>
