---
spec_version: "2"
answer_id: "t0009-stage2-forensics-answer"
answered_by_task: "t0009_stage2_training_failure_forensics"
date_answered: "2026-09-14"
---
## Question

Why does Kokoro-82M Stage 2 fine-tuning diverge across 19+ training launches, and what configuration
changes reliably prevent divergence?

## Answer

Stage 2 divergence has two root causes: (1) the upstream `load_checkpoint` function uses
`strict=False` and silently loads zero parameters when a DataParallel checkpoint with `module.`
prefixed keys is passed, causing the run to train from a random initialization instead of Stage 1
weights; (2) `joint_epoch` is set too early (3 instead of 6), activating GAN discriminator losses
before the decoder is stable, which triggers gradient explosions. The only successful run (v6c) used
`joint_epoch=6` and loaded a clean Stage 1 checkpoint, achieving `val_loss=0.849` at epoch 6 before
diverging at epoch 9. Fixes are: replace `load_checkpoint` with the DP-aware version that strips
`module.` prefixes and raises on zero-param match, and set `joint_epoch` to at least 6 in every new
config.

## Sources

* Task: `t0001_kokoro_v4_stage2_finetune`
* Task: `t0005_kokoro_v5_stage2_train`
* URL: <https://github.com/resemble-ai/StyleTTS2>
* URL: <https://github.com/hexgrad/kokoro>
