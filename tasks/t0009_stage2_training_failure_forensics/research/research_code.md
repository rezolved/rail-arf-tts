---
spec_version: "1"
task_id: "t0009_stage2_training_failure_forensics"
research_stage: "code"
tasks_reviewed: 7
tasks_cited: 6
libraries_found: 0
libraries_relevant: 0
date_completed: "2026-09-14"
status: "complete"
---
## Task Objective

Audit every Kokoro StyleTTS2 Stage 2 training run's logs, data, pipeline, and checkpoints to explain
why Stage 2 fine-tuning consistently fails across four tasks (t0001, t0004, t0005, t0006; 19+
launches, ~$440 GPU spend). No checkpoint with clean audio has been produced. The task will produce
a root-cause answer asset explaining the failures, a library asset containing safeguards (JSONL
metrics logger, health-gate monitor, per-epoch checkpoint retention, offline replay test), and the
artefact tables required by the task description (log inventory, confound table, checkpoint map,
data audit summary, pipeline patch review).

## Library Landscape

No registered libraries exist in the project yet (`aggregate_libraries` returned
`{"library_count": 0, "libraries": []}`). All reusable code is currently embedded in task `code/`
directories and must be copied rather than imported across tasks. The project has no shared library
infrastructure yet. The safeguard library that t0009 will produce — containing the JSONL metrics
logger, health-gate monitor, checkpoint retention policy, and log parser — is expected to become the
first registered library asset, enabling future training tasks to import it directly.

## Key Findings

### The Seven Patch Stack in train_second_patched.py

Six crash-inducing bugs were discovered and patched across t0005's six training launches. The final
patched script lives at `tasks/t0005_kokoro_v5_stage2_train/code/train_second_patched.py` (995
lines). An earlier version with slightly fewer patches is at
`tasks/t0001_kokoro_v4_stage2_finetune/code/train_second_patched.py` (954 lines). The patches, in
order of discovery:

1. **`lambda_slm == 0` guard** [t0005]: Vanilla `train_second.py` called `slmadv()` unconditionally
   once `epoch >= joint_epoch`, regardless of `lambda_slm`. When `lambda_slm: 0.0`, the `ref`
   variable is never assigned (only set under `multispeaker and epoch >= diff_epoch`), causing
   `UnboundLocalError` at the first GAN epoch. t0001's patched script already had this guard at
   lines 682-695; it was lost when the VM's ephemeral disk was wiped and `StyleTTS2` was re-cloned.
   The v3 reference patch
   (`tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/train_second_patch.diff`) shows a
   slightly different formulation: `if loss_params.lambda_slm > 0:` wrapping the entire `slmadv`
   block.

2. **LR revert to v4-safe values** [t0005]: t0005 initially tried `lr/ft_lr: 1e-4` and
   `bert_lr: 1e-5`, assuming t0001's NaN at joint_epoch was data-caused. The NaN reproduced on clean
   v5 data at the same LR. Fix: `lr/ft_lr: 3e-5`, `bert_lr: 1e-6`. t0006 v6d reverted to
   `lr/ft_lr: 1e-4` again — this is an inconsistency to investigate in forensics.

3. **`train_LM: false` guard** [t0005]: Without this guard, `optimizer.step("bert_encoder")` and
   `optimizer.step("bert")` fire unconditionally every step. When the GAN phase starts, BERT
   gradients from the discriminator signal destroy the language model (LM Loss 23 → 1757). The guard
   at t0005 lines 657-659 gates these steps on `train_LM`. t0001's script has the same guard at line
   636\. The config key `train_LM: false` appears in both t0001/code/config_david_v4.yml and
   t0005/code/config_david_v5_stage2.yml but NOT in any t0006 config — the guard is present in the
   patched script that t0006 ran, so it was silently applied.

4. **`clip_grad_norm_` on discriminators and generator** [t0005]: Added at t0005 lines 584-585
   (msd/mpd) and 667-668 (style_encoder/decoder). The first backward through freshly-activated,
   never-trained MSD/MPD discriminators at `joint_epoch` produces unbounded gradients. t0001's
   script has no gradient clipping at all — this is a real difference between t0001 and t0005.

5. **`set_detect_anomaly(False)`** [t0005]: `set_detect_anomaly(True)` aborts `backward()` before
   gradient clipping runs. Disabling it (t0005 line 22) allows `backward()` to complete so the
   finite-grad checks below can skip the bad step instead of crashing.

6. **Finite-grad skip guard with 50-consecutive-skip abort** [t0005]: At lines 583-589 and 598-606,
   non-finite grads skip the optimizer step; after 50 consecutive skips, the process calls
   `sys.exit(2)`. The abort was necessary because an unbounded skip loop (no abort) let training
   spin for hours with no optimizer updates (409 consecutive skips observed in run05).

7. **`torch.exp(torch.clamp(..., max=15.0))` in istftnet.py** [t0005]: The root-cause NaN fix.
   `istftnet.py:523/541` contained `torch.exp(x)` without clamping. The first GAN gradient at
   `joint_epoch` shifts `conv_post` by tens in log space; unbounded `exp()` produces ~1e22
   amplitude; mel spectrogram squares it to 1e44, exceeding fp32's 3.4e38 → inf;
   `SpectralConvergengeLoss` then computes `inf - inf` → NaN. The patched version is at
   `tasks/t0005_kokoro_v5_stage2_train/code/istftnet_patched.py`.

All seven patches are present in `tasks/t0005_kokoro_v5_stage2_train/code/train_second_patched.py`,
which t0006 also ran. None address the divergence; they only prevent crashes.

### Checkpoint Loading: Strict vs Silent-Failure

t0001's `train_second_patched.py` adds a custom `load_checkpoint` function (lines 49-80) that fails
loudly on key mismatch. The upstream `utils.py` `load_checkpoint` uses `strict=False`, which means a
checkpoint saved from a DataParallel-wrapped model (every key prefixed `module.`) loads ZERO
parameters while printing `"<key> loaded"` — silently training from scratch. t0001's version strips
the `module.` prefix and raises `RuntimeError` if 0 params matched:

```python
ckpt_sd = {k.removeprefix("module."): v for k, v in params[key].items()}
matched = {k: v for k, v in ckpt_sd.items() if k in model_sd and model_sd[k].shape == v.shape}
if len(matched) == 0:
    raise RuntimeError(f"{key}: 0/{len(model_sd)} params matched ...")
```

t0005's script does NOT have this guard — it uses the upstream `load_checkpoint` from `utils.py`.
t0006 run01 and run02 used `first_stage.pth` from t0004 (multispeaker:true) but ran with
`multispeaker: false` — this mismatch means the style encoder's shape may differ, causing silent
parameter mismatch in the upstream loader. This is one of the key inconsistencies to verify.

### The Checkpoint Overwrite Problem

Both t0001 and t0005 use `sync_and_monitor.sh` to keep only the top-2 checkpoints by `val_loss`. The
checkpoint naming uses `epoch_2nd_%05d.pth` where the index is the 0-based epoch number (t0005 line
956). When training resumes from scratch (deleted checkpoints between runs) the naming restarts,
making it impossible to distinguish epoch 3 from a fresh run vs epoch 3 from a previous run without
comparing timestamps. The log `rm -f`s itself between launches (noted in t0005 logs README),
destroying the audit trail for runs 1-5.

The `saving_epoch` variable (t0005 line 106) shadows `save_freq`, so checkpoints are saved every
`save_freq` epochs (set to 1 in v5/v6 configs), but the top-2 pruning means only the 2 best by
`val_loss` survive. This policy has two problems: (a) `val_loss` has not been verified to track
audio quality, and (b) a checkpoint from just before divergence may be pruned if a pre-GAN
checkpoint has a lower `val_loss` — losing the only healthy GAN-phase weights.

### Config-Code Mismatches and Inconsistencies

**t0006 v6d config bug**: `epochs: 15, epochs_1st: 10, epochs_2nd: 10`. The script reads
`epochs_2nd` for training length, so training stopped at epoch 10 despite the intent of 15 epochs.
The `epochs: 15` key is ignored.

**v6b checkpoint mismatch**: t0006 run02 (v6b) used the t0004 `first_stage.pth` (multispeaker:true)
with `multispeaker: false` in the config. The v6c/v6d runs correctly switched to v3's
`first_stage_v3.pth` (multispeaker:false), causing baseline `acoustic_norm` to drop from 10 to 0.36
— a clear signal that checkpoint-config alignment is critical.

**t0004 checkpoint label**: `epoch_1st_00007.pth` is described in the results as "epoch 10,
val=0.740". The file is 0-indexed (epoch 7 in 0-based = the 8th epoch). t0005 relied on this
checkpoint; the off-by-one in labeling needs forensic resolution.

**t0005 best checkpoint discrepancy**: `logs/README.md` says best is "epoch 2", while
`results_detailed.md` says "epoch 3, val=0.848". The checkpoint file is `epoch_2nd_00003.pth`
(0-based index 3 = the 4th epoch). The results summary says "epoch 3" which matches the file name,
but the README says "epoch 2" — the 0-based vs 1-based confusion propagates.

### Data Scale and GAN Instability

The central hypothesis emerging from t0005 and t0006 is that 1557 clips produce ~194 steps/epoch
while v3's 266 clips produced ~31 steps/epoch. More steps means more diverse discriminator batches
at `joint_epoch`, stronger GAN gradient at transition. t0006 tested 250 clips (31 steps/epoch,
matching v3). With `lambda_gen: 0.05` and `joint_epoch: 6`, run04 (v6d) achieved no divergence but
`acoustic_norm=8.51` and noisy audio. This is still 5.85× v3's baseline acoustic_norm of 0.36. The
data-scale hypothesis is partially confirmed but insufficient on its own.

### The v3 Mystery and Comparison Gap

v3 achieved `val_loss: 0.506` (t0006 results say `0.569` at epoch 1, then `0.506` overall). Every
current run starts pre-GAN validation loss around 1.6-1.65 (from run03_v6c_v3_stage1.log), while v3
opened at 0.569. This 3× gap before the GAN even activates is unresolved. t0006 notes "If v3 used
`joint_epoch: 0` (GAN active from ep1), this explains the gap." No v3 config has been committed to
git; only the Stage 1 checkpoint and patch diff are available as reference assets in
`tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/`.

### Manifest Validation Gate Pattern

t0003 produced a clean manifest validation framework in
`tasks/t0003_kokoro_v5_phoneme_data/code/prepare_v5_data.py` (255 lines) with:

* `load_kokoro_vocab()` — loads the 114-symbol Kokoro vocab from `config.json`
* `rejection_reason(text, vocab)` — returns the first failing gate: `"raw_text_leaked"`,
  `"double_phonemized"`, `"unknown_word"`, `"empty"`, `"out_of_vocab:<char>"`, or `None` if clean
* `slug(text)` — reproduces `prepare_v4_data.py`'s filename convention for text matching

Tests in `tasks/t0003_kokoro_v5_phoneme_data/code/test_gates.py` (58 lines) verify these gates
against real examples from broken manifests. This is a concrete, tested pattern for data-quality
gating that t0009's forensics will reuse.

### Log Format and Monitoring Scripts

The training log format (from `run03_v6c_v3_stage1.log`) is fixed-structure lines like:

```
Epoch [1/10], Step [10/31], Loss: 0.15010, Disc Loss: 0.00000, Dur Loss: 9.66929, CE Loss: 0.26508, ...
Validation loss: 1.643, Dur loss: 0.716, F0 loss: 5.058
voicepack/acoustic_norm=0.3604  prosodic_norm=0.3604
```

The `sync_and_monitor.sh` in t0004/t0005 provides log-polling, checkpoint pruning, and stop-early
criterion checking. The stop criterion: Dur Loss at the first logged step of Stage 2 must be < 2.0
(v3 opened at 0.81; failures opened at 8-17). The monitor does NOT produce structured JSONL — it
reads the text log. t0009 must add a JSONL metrics logger.

## Reusable Code and Assets

### 1. `train_second_patched.py` (t0005) — copy into task

**Source**: `tasks/t0005_kokoro_v5_stage2_train/code/train_second_patched.py` **What it does**:
StyleTTS2 Stage 2 training script with all 7 crash fixes applied. Includes `MyDataParallel`,
`_grads_finite()`, `_CONSECUTIVE_SKIPS` guard, `train_LM` gate, and `clip_grad_norm_` calls. **Reuse
method**: **copy into task** — this is non-library task code **Key functions**:

* `_grads_finite(*modules) -> bool` — checks all `.grad` tensors are finite across module list
* `MyDataParallel(torch.nn.DataParallel)` — passes `__getattr__` through to `.module`
* `main(config_path)` — full training loop, ~900 lines

**Adaptation needed for t0009**: Add a JSONL metrics writer (one record per step), replace the
`epoch_2nd_%05d.pth` naming + top-2-only policy with per-epoch saves + health-gate checkpoint
retention, and add the health gates inline. **Line count**: 995 lines

### 2. `load_checkpoint` with key-mismatch guard (t0001) — copy into task

**Source**: `tasks/t0001_kokoro_v4_stage2_finetune/code/train_second_patched.py` lines 49-80 **What
it does**: DP-aware checkpoint loader that strips `module.` prefix and raises on 0-param match,
preventing silent from-scratch training. **Reuse method**: **copy into task** **Key signature**:
`load_checkpoint(model, optimizer, path, load_only_params=True, ignore_modules=[])` **Adaptation
needed**: Adopt this version over t0005's upstream loader for the safeguard library. **Line count**:
31 lines

### 3. `prepare_v5_data.py` gate functions (t0003) — copy into task

**Source**: `tasks/t0003_kokoro_v5_phoneme_data/code/prepare_v5_data.py` **What it does**: Manifest
validation gates (`rejection_reason`, `load_kokoro_vocab`, `slug`) and test corpus in
`test_gates.py`. **Reuse method**: **copy into task** — the data audit will replay these gates
against v5 and v3 manifests **Key signatures**:

* `load_kokoro_vocab() -> frozenset[str]`
* `rejection_reason(text: str, vocab: frozenset[str]) -> str | None`
* `slug(text: str) -> str`

**Line count**: 255 lines (prepare_v5_data.py) + 58 lines (test_gates.py)

### 4. `sync_and_monitor.sh` checkpoint prune logic (t0004/t0005) — copy into task

**Source**: `tasks/t0004_kokoro_v5_stage1_train/code/sync_and_monitor.sh` (139 lines),
`tasks/t0005_kokoro_v5_stage2_train/code/sync_and_monitor.sh` **What it does**: Embeds a Python
snippet that loads checkpoint val_loss via `torch.load` and prunes all but the top-K. The remote
pruning uses `scp` + `ssh rm -f`. **Reuse method**: **copy into task** — the log-parser will need
this logic to reconstruct which checkpoints survived each run **Key pattern**: The `load_val_loss`
embedded Python reads `ckpt["val_loss"]` to rank checkpoints.

### 5. `infer_v6d.py` DataParallel strip pattern (t0006) — copy into task

**Source**: `tasks/t0006_kokoro_v5_stage2_subset/code/infer_v6d.py` lines 56-61 **What it does**:
Strips `module.` prefix from checkpoint state dict before `load_state_dict`. **Reuse method**:
**copy into task** **Key snippet**:
`clean = {re.sub(r"^module\.", "", key): v for key, v in raw.items()}` **Line count**: 141 lines
total; the relevant pattern is 5 lines

### 6. v3 reference assets (t0006) — read directly

**Source**: `tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/` Contains:
`train_second_patch.diff` (144 lines), `stage1/` (DVC-tracked), `best/` (DVC-tracked), `audio/`
(DVC-tracked). The diff is committed to git and is the primary source for the v3 patch comparison in
the pipeline audit.

## Lessons Learned

### Patches Mask Causes Rather Than Fix Them

The `istftnet.py exp()` clamp (patch 7) stops the NaN crash but explicitly does not stop the
divergence (`t0005/logs/README.md`: "Stops the crash; does NOT stop the divergence"). The
consecutive-skip abort (patch 6) prevents endless spinning but only because the training was already
in a degenerate state the optimizer could not recover from. Each patch addresses a crash symptom;
none addresses why the GAN gradient is too large at `joint_epoch`.

### The `rm -f` Log Deletion Pattern Is Catastrophic

`sync_and_monitor.sh` for t0005 uses `REMOTE_LOG="/mnt/kikiri-tts/logs/stage2_v5.log"` and the
launch command `rm -f`s the remote log before each run. This destroyed the per-run logs for runs
1-5. Only one Stage 2 log is committed
(`tasks/t0006_kokoro_v5_stage2_subset/logs/run03_v6c_v3_stage1.log`). The t0009 task description
notes: "the launch command `rm -f`s the remote log and the config file was edited in place between
launches." This is the primary obstacle for forensic reconstruction of runs 1-5.

### Config-in-Place Editing Creates an Audit Trail Gap

t0005's config was edited between launches without versioning. The t0009 forensic must reconstruct
per-launch configs from log headers and git history. The safeguard library must copy the resolved
config alongside each run's log directory.

### val_loss Is a Proxy Metric, Not a Quality Metric

The top-2 checkpoint pruning policy uses `val_loss` as the sole ranking criterion, but no evidence
links `val_loss` to audio quality. v3 achieved 0.506; t0006 v6d achieved 0.846 with noisy audio. The
`acoustic_norm` logged by the TensorBoard voicepack extract is a better quality proxy (v3's 0.36 vs
t0006 v6c's 6.07 at best), but it is not used for checkpoint selection.

### DataParallel and Multi-GPU Confusion

t0001 launched `train_second.py` with `accelerate launch --num_processes 2` — wrong for DataParallel
(which is single-process). This spawned two independent processes each with one GPU, each writing
checkpoints to the same `log_dir` — a race condition. t0004's Stage 1 used plain
`python3 train_first.py` (which uses `accelerate`/DDP), so it ran single-GPU silently. These launch
errors are independent confounds that must be in the confound table.

### The v3 Config Gap Is the Largest Unknown

No v3 training config is committed. The task description notes the open question: "Why did v3's
audio go from exploded durations to clean six minutes apart with no retraining?" t0003 found v3's
success depended on which `first_stage.pth` happened to be in the log directory. The v3 patch diff
only shows the `lambda_slm > 0` guard and a `utils.py` path fix — no gradient clipping, no
`train_LM`, no `istftnet` clamp — yet v3 succeeded. Understanding what v3's `joint_epoch` actually
was is the highest-leverage question.

## Recommendations for This Task

1. **Start the pipeline audit from `train_second_patch.diff`**: The v3 reference diff in
   `tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/train_second_patch.diff` is the only
   committed record of what v3 actually ran. Diff this against t0005's `train_second_patched.py` to
   enumerate exactly which of the 7 patches v3 had vs did not have. This directly answers pipeline
   audit question 5.

2. **Use `load_checkpoint` from t0001 in the safeguard library**: It is strictly safer than the
   upstream loader and already covers the `module.` prefix problem that invalidated t0006 run02.
   Copy it into `tasks/t0009_stage2_training_failure_forensics/code/`.

3. **Copy and extend `_grads_finite` and the skip-guard from t0005**: These are the most mature
   patterns from the crash experience. The safeguard library's health gate can build on top of them
   by adding `acoustic_norm` and `val_loss` spike checks alongside the existing grad-finiteness
   check.

4. **Gate the offline replay test on `rejection_reason` from t0003**: The data audit (scope 4)
   should reuse `prepare_v5_data.py`'s gates verbatim against the v5 and v3 manifests to confirm
   zero rejected lines. Copy the file rather than importing it, per the cross-task code rule.

5. **The JSONL logger must write a record per step, not per `log_interval`**: The text log only
   captures every 10 steps. Loss fields to include: all those in the `logger.info` call at t0005
   line 758 plus `grad_norm_msd`, `grad_norm_mpd`, `grad_norm_decoder`, `grad_norm_style_encoder`,
   `skip_count`, `lr` per optimizer group.

6. **Health gates to implement**: Based on the observed failure signals in the logs:
   * Dur Loss at epoch 1 step 1: must be < 2.0 (v3 opened at 0.81; failures at 8-17)
   * `acoustic_norm` after any epoch: must be < 20 (v6 run01 hit 27 at divergence)
   * Val loss spike: must not exceed previous epoch by > 0.05 after `joint_epoch`
   * Consecutive non-finite grad skips: must not exceed 50

7. **Checkpoint retention policy**: Keep a checkpoint every epoch regardless of `val_loss`, plus
   flag and never prune the last checkpoint saved before `joint_epoch`. Add SHA-256 hash to each
   checkpoint record. Never `rm -f` the log between runs; use a per-run subdirectory instead.

## Dataset Landscape

* **v4 train/val lists** (`data/v4/train/train_list.txt`, `data/v4/val/val_list.txt`): 1557 train,
  96 val, 21% raw-text contamination. Used by t0001. Root cause of t0001's Stage 2 divergence.
* **v5 train/val lists** (`results/v5/train_list.txt`, `results/v5/val_list.txt`): 1557/96, all
  gate-clean. Produced by t0003. Used by t0005 (full 1557) and t0006 (250-clip subset via
  `code/train_list_250.txt`, seed=42 random sample).
* **v3 manifests** (referenced in t0003 as source of original texts): committed to `rail-benchmarks`
  repo, not this one. 266 train clips + 96 val clips.
* **Val set overlap**: Whether v3's val set equals `val_96` (the 96-clip held-out regression set) is
  an open forensic question from the task description.

## Architecture Overview

The training pipeline is:

```
Stage 1 checkpoint (first_stage.pth)
  → load_checkpoint (load_only_params=True, ignore_modules=[predictor_encoder, msd, mpd, wd, diffusion])
  → DataParallel wrapping (after load, to avoid module. prefix mismatch)
  → Pre-GAN epochs: train bert, bert_encoder, predictor, predictor_encoder only
  → joint_epoch: activate MSD/MPD discriminators, add decoder/style_encoder updates
  → val_loss evaluated per epoch → top-2 checkpoints retained by sync_and_monitor.sh
```

Key epoch boundaries in configs:
* `diff_epoch: 999` — diffusion sampler never activates (deactivated in all runs)
* `joint_epoch: 3` (v5, v6, v6b, v6c) or `6` (v6d) — GAN switch-on
* All configs: `lambda_slm: 0.0` → WavLM discriminator path is dead code

## Task Index

### [t0001]

* **Task ID**: `t0001_kokoro_v4_stage2_finetune`
* **Name**: Kokoro v4 Stage 2 fine-tune
* **Status**: completed
* **Relevance**: First failed Stage 2 run. Contains the most mature patched training script (with
  the crucial `load_checkpoint` key-mismatch guard absent from t0005). Its v4 data had 21% raw-text
  contamination — the original root cause of divergence.

### [t0003]

* **Task ID**: `t0003_kokoro_v5_phoneme_data`
* **Name**: Regenerate v5 phoneme manifests
* **Status**: completed
* **Relevance**: Produced the data validation gates (`rejection_reason`, `load_kokoro_vocab`) and
  the clean v5 manifests that all subsequent training tasks used. The gate pattern is directly
  reusable for the data audit.

### [t0004]

* **Task ID**: `t0004_kokoro_v5_stage1_train`
* **Name**: Kokoro v5 Stage 1 fine-tune
* **Status**: completed
* **Relevance**: Produced the `first_stage.pth` that t0005 used for Stage 2. Contains the
  `sync_and_monitor.sh` with the checkpoint pruning pattern. The 0-based epoch labeling discrepancy
  (`epoch_1st_00007.pth` described as "epoch 10") is a key forensic question.

### [t0005]

* **Task ID**: `t0005_kokoro_v5_stage2_train`
* **Name**: Kokoro v5 Stage 2 fine-tune
* **Status**: completed
* **Relevance**: Direct dependency. Six training runs, seven crash patches accumulated. The final
  `train_second_patched.py` is the primary source for the pipeline audit and safeguard library
  implementation.

### [t0006]

* **Task ID**: `t0006_kokoro_v5_stage2_subset`
* **Name**: Kokoro v5 Stage 2: 250-clip subset, multispeaker: false
* **Status**: completed
* **Relevance**: Direct dependency. Contains the only committed Stage 2 training log
  (`logs/run03_v6c_v3_stage1.log`), the v3 reference patch diff, and v3's Stage 1 checkpoint. Four
  config variants with multiple changed variables at once are the primary confound to untangle.

### [t0007]

* **Task ID**: `t0007_brainstorm_results_1`
* **Name**: Brainstorm results session 1
* **Status**: completed
* **Relevance**: Commissioned t0009. Documents the decision to prioritize forensics over more
  training. Notes: only one training log in git, checkpoint labeling inconsistencies, `total_usd`
  cost key bug in t0001-t0006, and the six-variable-change problem in t0006.
