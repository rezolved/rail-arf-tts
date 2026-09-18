---
spec_version: "2"
answer_id: "v3-recipe"
answered_by_task: "t0016_v3_recipe_recovery"
date_answered: "2026-09-17"
---
## Question

What exactly did Kokoro v3's Stage 2 training recipe consist of, and which parts of that are
confirmed versus inferred?

## Answer

v3 fine-tuned StyleTTS2 Stage 2 from `first_stage_v3.pth` (confirmed by name only; byte-identity
unrecoverable) for at least 10 epochs (confirmed from the epoch-numbered audio samples, epochs 0-9),
using the ISTFTNet decoder architecture and `max_dur=50`/`style_dim=128`/`n_token=178` (confirmed
from the shipped bundle's `config.json`), with `multispeaker: false` (inferred from checkpoint-shape
forensics on the Stage 1 diffusion module, resolving the standing three-way contradiction in favor
of `false`), `lambda_slm: 0.0` (inferred from the surviving `train_second_patch.diff`), and all five
bundle modules (`bert`, `bert_encoder`, `predictor`, `text_encoder`, `decoder`) showing substantial
weight-norm shifts from Stage 1, meaning the decoder itself changed during Stage 2 rather than the
speaker-similarity gain coming from `predictor`/ `text_encoder` alone. The 266-clip training list,
the exact `joint_epoch`/`lambda_gen`/`lr`, and byte-identity to the checkpoint v6c/v6d also loaded
could not be recovered: the VM's home directory turned out to be a later task's (t0014's) StyleTTS2
clone, not a preserved v3-era environment, so no literal launch config or clip list survived. This
is a low-confidence reconstruction by design — most fields are `inferred` or `unknown`, honestly
labelled, not `confirmed`.

## Sources

* Task: `t0002_kokoro_v4_voicepack_decoder_package`
* Task: `t0006_kokoro_v5_stage2_subset`
* Task: `t0008_tts_eval_harness_baselines`
* Task: `t0009_stage2_training_failure_forensics`
* Task: `t0013_v10_synthesis_quality_forensics`
* Task: `t0015_v11_duration_blowup_forensics`
