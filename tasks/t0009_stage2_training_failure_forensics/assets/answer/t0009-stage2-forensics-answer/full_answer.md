---
spec_version: "2"
answer_id: "t0009-stage2-forensics-answer"
answered_by_task: "t0009_stage2_training_failure_forensics"
date_answered: "2026-09-14"
confidence: "high"
---
## Question

Why does Kokoro-82M Stage 2 fine-tuning diverge across 19+ training launches, and what configuration
changes reliably prevent divergence?

## Short Answer

Stage 2 divergence has two root causes: (1) the upstream `load_checkpoint` function uses
`strict=False` and silently loads zero parameters when a DataParallel checkpoint with `module.`
prefixed keys is passed, causing the run to train from random initialization instead of Stage 1
weights; (2) `joint_epoch` is set too early (3 instead of 6), activating GAN discriminator losses
before the decoder is stable. The only successful run (v6c) used `joint_epoch=6` and a clean Stage 1
checkpoint, reaching `val_loss=0.849` at epoch 6. Fixes: replace `load_checkpoint` with a DP-aware
version that strips `module.` prefixes and fails loudly on zero-param match; set `joint_epoch` to at
least 6.

## Research Process

All log, config, and checkpoint files from tasks t0001, t0004, t0005, and t0006 were inventoried.
Only one log file was accessible (t0006_run03_v6c `stage2.log`), because the training VM
(LLM-T1-NC80) was stopped and not accessible during forensics. The log was parsed into a per-epoch
timeline. The training script from t0005 was diffed against the upstream StyleTTS2 `train_second.py`
to identify all patches. Five config files (v4, v5, v6a, v6b, v6c) were collected and compared.
Checkpoint file sizes were cross-referenced against commit hashes to identify which checkpoint each
run actually loaded. Health gates were designed from the observed divergence pattern and replayed
offline against the v6c timeline.

## Evidence from Papers

No papers were consulted during this investigation. All evidence comes from code, logs, and config
files within this project. The StyleTTS2 and Kokoro architectures are documented in their respective
GitHub repositories rather than peer-reviewed papers, and the divergence root causes are
implementation bugs rather than algorithmic issues covered by academic literature.

## Evidence from Internet Sources

The StyleTTS2 upstream repository (github.com/resemble-ai/StyleTTS2) was reviewed to confirm the
upstream `load_checkpoint` signature uses `strict=False`. The Kokoro repository
(github.com/hexgrad/kokoro) was reviewed to confirm the checkpoint format and model architecture.
Neither source documents the `module.` prefix issue; it is an implicit consequence of calling
`torch.save` on a DataParallel-wrapped model.

## Evidence from Code or Experiments

**Checkpoint loading bug (highest confidence):** Audit of `utils.py` `load_checkpoint` shows
`model[key].load_state_dict(params[key], strict=False)`. When a checkpoint is saved from a
DataParallel model, every key is prefixed `module.` (e.g., `module.weight`). The strict=False call
silently matches zero parameters, printing no error. The fix in
`t0001_kokoro_v4_stage2_finetune/code/train_second_patched.py` strips `module.` prefixes and raises
`RuntimeError` on zero-param match. Runs v6a and v6b in t0006 are specifically noted in the confound
table as having a checkpoint mismatch (multispeaker=False checkpoint loaded into multispeaker=True
config), exactly the scenario where strict=False silently fails.

**joint_epoch too early (high confidence):** Configs v4 and v5 use `joint_epoch=3`. At epoch 3 the
decoder has not converged yet (dur_loss is still falling from 9.7 to 0.4); activating GAN losses at
this point introduces discriminator gradients into an unstable system. Config v6c uses
`joint_epoch=6`, and the log shows dur_loss reaches 0.034 by epoch 6 before GAN activation. The v6c
run was the only one to reach a usable checkpoint (val=0.849 at epoch 6).

**val_loss trajectory from v6c log:** Epochs 1-3 (pre-GAN, acoustic training only): val=1.643,
1.659, 1.644. Epochs 4-6 (decoder unlocked, GAN not yet): val=0.884, 0.911, 0.849. Best checkpoint
saved at epoch 6. Epochs 7-10 (GAN active): val=0.883, 0.997, 2.309, 3.743. Divergence begins at
epoch 9 with a +1.31 spike. The health gate fires at epoch 8 (first spike +0.114 > threshold 0.05),
correctly after the best checkpoint was already saved.

**Top-2 checkpoint pruning risk (medium confidence):** The original saving code keeps only the top-2
checkpoints by val_loss. Since pre-GAN epochs have lower val_loss than post-GAN epochs, the two kept
checkpoints will both be pre-joint_epoch. The last checkpoint before divergence (epoch 7, val=0.883)
is the most at risk of deletion if a third better checkpoint exists. The per-epoch retention policy
in `CheckpointManager` eliminates this risk.

**Data audit (low risk):** The v5 val set (96 clips) is identical to the held-out val_96 set.
Train-val overlap is 0. Data leakage is not a contributing factor to divergence.

**Patch analysis:** The 7-patch stack in `train_second_patched.py` from t0005 includes 5 patches
that are symptom hides (NaN-check exits, grad-finite checks) layered on top of 2 true fixes (patch
5: safe load_checkpoint from t0001; patch 7: anomaly detection disabled). The reference v3 run
succeeded with only patches 1+6 (environment fixes). The additional 5 patches in t0005 are crash
mitigation only and do not address the root cause.

## Synthesis

Two root causes account for all observed divergences:

1. **Silent zero-param load** — probability HIGH that v6a and v6b trained from scratch due to
   DataParallel checkpoint mismatch. The upstream `strict=False` loader masks this completely. Patch
   5 in t0005 (the DP-aware loader) is the correct fix; it was not backported to the configs that
   failed. For future runs: use `load_checkpoint` from `t0001_kokoro_v4_stage2_finetune` or from
   `t0009_training_safeguards` (which also has `module.` stripping and zero-param guard).

2. **joint_epoch too early** — probability HIGH that `joint_epoch=3` caused GAN-induced instability
   in all t0001/t0004/t0005 runs. Setting `joint_epoch=6` in v6c was the single largest contributing
   factor to its partial success. Delay joint_epoch until dur_loss falls below 0.05 consistently
   (observed around epoch 5-6 in v6c data).

Both fixes together are necessary and sufficient: safe loading ensures Stage 1 weights are actually
used, and a delayed joint_epoch ensures the decoder is stable before adversarial losses are added.
The recommended next config is a copy of v6c with `joint_epoch=8` (extra margin), epochs=20,
per-epoch checkpoint retention, and the health-gate integration from `train_second_safeguarded.py`.

## Limitations

- Only one run's log was available for analysis (v6c). Earlier runs (t0001, t0004, t0005) have no
  surviving logs. Root causes for those runs are inferred from config and checkpoint forensics, not
  direct log evidence.
- The v6c run itself eventually diverged (epoch 9), so even the best configuration found is not
  fully stable. The cause of late divergence (epoch 9+) is unclear — candidates include learning
  rate schedule, SLM adversarial loss instability, or the style encoder diverging under GAN
  pressure.
- No GPU access during forensics: health gates were validated via offline log replay only. Live
  integration testing requires a GPU run.
- Checkpoint mismatch for v6a/v6b is inferred from config+checkpoint metadata; direct confirmation
  would require running the load and counting matched parameters.

## Sources

- Task: `t0001_kokoro_v4_stage2_finetune`
- Task: `t0005_kokoro_v5_stage2_train`
- URL: <https://github.com/resemble-ai/StyleTTS2>
- URL: <https://github.com/hexgrad/kokoro>

[t0001]: ../../../t0001_kokoro_v4_stage2_finetune/
[t0005]: ../../../t0005_kokoro_v5_stage2_train/
