# Research Summary — t0009_stage2_training_failure_forensics

## Key Findings (top 10 insights directly actionable for this task)

1. **Seven crash patches exist but none fix divergence.** All 7 fixes in
   `tasks/t0005_kokoro_v5_stage2_train/code/train_second_patched.py` prevent crashes at
   `joint_epoch`; the underlying GAN-gradient instability causing val_loss to diverge remains
   unexplained.

2. **The v3 patch diff is the ground truth for what v3 actually ran.** Only `lambda_slm > 0` guard
   and a `utils.py` path fix are in v3's committed diff — no grad clipping, no `train_LM` guard, no
   `istftnet` clamp. Yet v3 succeeded (val 0.506). This is the central forensic question.

3. **Log destruction is the biggest audit obstacle.** The launch script `rm -f`s the remote log
   before each run. Only one Stage 2 log is committed to git
   (`tasks/t0006_kokoro_v5_stage2_subset/logs/run03_v6c_v3_stage1.log`). Logs for t0005 runs 1-5 are
   gone from both VM and git.

4. **t0005's `load_checkpoint` is silent-failure-prone; t0001's is not.** The upstream
   `strict=False` loader loads 0 params without raising when a DataParallel-wrapped checkpoint
   encounters a non-wrapped model. t0001's custom loader (lines 49-80) strips `module.` prefix and
   raises `RuntimeError` on 0-param match. t0005 and t0006 used the upstream loader — t0006 run02
   (v6b) may have trained from scratch silently due to the multispeaker:true/false mismatch.

5. **Checkpoint epoch labeling is 0-based, causing persistent off-by-one confusion.** t0004's
   `epoch_1st_00007.pth` is "epoch 10, val=0.740" in the results. t0005's best is
   `epoch_2nd_00003.pth` (file name = epoch 3, README says "epoch 2"). All forensic analysis must
   re-derive epoch from file name, not from prose descriptions.

6. **Top-2 val_loss pruning may delete the only healthy GAN-phase checkpoint.** A pre-GAN epoch (low
   val_loss, easy reconstruction) can outrank a post-GAN epoch. The last checkpoint before
   divergence is the most valuable one and the most likely to be pruned.

7. **t0006 v6d accidentally trained only 10 epochs (intended: 15) due to config key confusion.** The
   script reads `epochs_2nd`; the config has `epochs: 15, epochs_2nd: 10`. The extra key is silently
   ignored. Best checkpoint (val=0.846) may simply be undertrained.

8. **t0006 v6b used a multispeaker:true Stage 1 checkpoint with multispeaker:false config.** The
   switch to v3's `first_stage_v3.pth` in v6c dropped baseline acoustic_norm from 10 to 0.36 — a 28×
   improvement from correct checkpoint alignment alone.

9. **v3's pre-GAN val_loss (~0.57 at epoch 1) is 3× better than any v5/v6 run (~1.64).** This gap
   exists before the GAN activates and likely stems from v3's unknown `joint_epoch` setting — if v3
   used `joint_epoch: 0`, GAN was active from epoch 1 and the loss definition differs.

10. **Data scale partially confirmed but insufficient.** 250-clip v6 still diverged at
    `lambda_gen=1.0` and `0.2`; only `lambda_gen=0.05` with `joint_epoch=6` stabilised it. The true
    isolating variable may be `lambda_gen`, not data size.

## Best Approaches (top 3 recommended implementation approaches from research)

### Approach 1: Log and config archaeology before any new training

Collect every surviving log and config from the VM (`/mnt/kikiri-tts/`) and git history. Use
`train_second_patch.diff` as the v3 baseline. Reconstruct per-launch configs by diffing git
snapshots and reading log headers. This is zero-cost and unblocks all subsequent analysis.

### Approach 2: Build the JSONL logger and health gates from t0005's existing guards

Copy `_grads_finite()`, `_CONSECUTIVE_SKIPS`, and `clip_grad_norm_` calls from
`tasks/t0005_kokoro_v5_stage2_train/code/train_second_patched.py`. Extend with: one JSONL record per
step (all loss fields + grad norms + skip count + LR), health gate checks after each epoch
(`acoustic_norm < 20`, val spike < 0.05), and per-epoch checkpoint saves that never prune the last
pre-`joint_epoch` checkpoint.

### Approach 3: Offline replay to validate gates against committed logs

Use the single committed log (`run03_v6c_v3_stage1.log`) plus any logs recovered from the VM to test
that health gates fire on t0001/t0005-run06/t0006-run01/run02 divergences and do not fire on v3's
successful run. The replay can run locally with no GPU.

## Reusable Code / Assets

* `tasks/t0005_kokoro_v5_stage2_train/code/train_second_patched.py` (995 lines) — full Stage 2
  script with all 7 crash patches; copy into task for safeguard library base
* `tasks/t0001_kokoro_v4_stage2_finetune/code/train_second_patched.py` lines 49-80 — safe
  `load_checkpoint` with `module.` strip and 0-param raise; copy into safeguard library
* `tasks/t0003_kokoro_v5_phoneme_data/code/prepare_v5_data.py` — `rejection_reason()`,
  `load_kokoro_vocab()`, `slug()`; copy for data audit
* `tasks/t0003_kokoro_v5_phoneme_data/code/test_gates.py` (58 lines) — gate unit tests; copy as
  model for offline replay test
* `tasks/t0004_kokoro_v5_stage1_train/code/sync_and_monitor.sh` — checkpoint prune logic (embedded
  Python reads `ckpt["val_loss"]`); copy for log parser
* `tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/train_second_patch.diff` — v3's actual
  patch; the forensic baseline for pipeline diff
* `tasks/t0006_kokoro_v5_stage2_subset/logs/run03_v6c_v3_stage1.log` — only committed Stage 2 log;
  primary offline replay input

## Key Papers (top 5, with finding most relevant to this task)

*(not generated — research-papers step skipped)*

## Risks Flagged in Research

* **Most t0005 logs are unrecoverable**: if the VM's `/mnt` was wiped, crash logs 1-5 are gone.
  Record what is missing rather than guessing (per LESSONS Lesson 10).
* **v3 config is unknown**: without it, the confound table has a gap that may prevent root-cause
  attribution.
* **`val_loss` does not track audio quality**: the safeguard library's checkpoint retention must not
  rely on `val_loss` as the sole criterion.
* **t0006 v6d LR revert**: v6d used `lr/ft_lr: 1e-4` (reverted from t0005's safe `3e-5`), adding a
  new unstudied variable on top of the `lambda_gen=0.05` fix.
* **DataParallel launch confusion**: t0001 was launched with `accelerate launch --num_processes 2`,
  which is wrong for DataParallel and creates a checkpoint-write race. Any results from t0001 may
  reflect this race.

## Full Detail Available In

* `tasks/t0009_stage2_training_failure_forensics/research/research_papers.md` — (not generated —
  step skipped)
* `tasks/t0009_stage2_training_failure_forensics/research/research_internet.md` — (not generated —
  step skipped)
* `tasks/t0009_stage2_training_failure_forensics/research/research_code.md` — 6 tasks cited, 5
  reusable code items, full patch stack analysis
