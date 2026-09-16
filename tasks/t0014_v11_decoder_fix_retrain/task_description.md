# Kokoro Stage 2 v11: Decoder-Init Fix, Full Normalized Corpus Retrain

## Motivation

t0013 proved that `kokoro-v10-best` never produced audible speech at any epoch: the HiFi-GAN
decoder was silently partial-loaded from an ISTFTNet-shaped first-stage checkpoint because
`train_second_v10.py:load_checkpoint()`'s `ignore_modules` list (lines 244-260) omits `"decoder"`.
A falsification probe showed this partial load is actively **worse** than leaving the decoder at
pure random init (clip fraction 0.750-0.807 vs. 0.004). `val_loss` alone never caught it — it
looked healthy throughout t0010's 17 epochs — and `speaker_sim` wouldn't have caught it either
(confirmed-garbage v10 audio still scored 0.31-0.35 GE2E cosine, not a near-zero outlier).

This task retrains with the fix, and — since t0012 has since produced a much larger clean
manifest (1531/1557 clips, LUFS-normalized to -14, up from the 250-clip subset t0010 trained on) —
combines the decoder fix with the corpus expansion in one run rather than two, per S-0011-02's
original full-corpus follow-on plan.

**This task's own completion is gated on proof of audible speech, not on val_loss.** t0013's
`code/audio_quality_check.py` (`is_likely_noise` heuristic: clip fraction + spectral flatness) must
run against the new best checkpoint and return `is_likely_noise=False` before this task may claim
`status: completed`. If the fix still does not produce audible speech, say so plainly and stop —
do not report success, do not fabricate an inference recipe or ship audio samples that fail the
gate. This is the exact process fix t0013 filed as S-0013-02, applied here rather than deferred.

## Key Questions

1. Is there a `hifigan`-shaped pretrained StyleTTS2 first-stage checkpoint available (from the
   original StyleTTS2 authors, `semidark/StyleTTS2`, or elsewhere) that `config_david_v10.yml`'s
   `first_stage_path` could point at instead? If yes, that gives genuine fine-tuning of a converged
   vocoder rather than training one from scratch in ~20 epochs — check this **before** committing
   to the random-init path.
2. If no suitable pretrained `hifigan` checkpoint exists: is ~20 epochs on 1531 clips a realistic
   budget for a HiFi-GAN decoder to converge from random init? Check t0009's/general StyleTTS2
   literature for typical vocoder-from-scratch convergence epoch counts on a comparably-sized
   corpus; if 20 is clearly insufficient, say so in planning and propose a realistic epoch count
   rather than silently under-training a second time.
3. Does the corrected `load_checkpoint()` call (fixed `ignore_modules`) actually leave `decoder` at
   `build_model()`'s own random init, with zero partial overlap from `first_stage_v3.pth`? Verify
   this directly with `t0013/code/inspect_checkpoint.py`-style tensor inspection immediately after
   the fix, before spending any GPU time training on it.
4. Does training on the larger, louder (-14 LUFS normalized), cleaner 1531-clip corpus change
   convergence behavior (batch composition, epoch wall-clock, GAN-phase stability) relative to
   t0010's 250-clip run? `joint_epoch=8` and the DP-aware loader fix both carry over unchanged from
   t0010/t0009 — only the corpus and the decoder-init fix change.
5. Does the resulting checkpoint pass `audio_quality_check.py`'s `is_likely_noise` gate? This is the
   task's actual pass/fail criterion — everything else is diagnostic.

## Scope

### 1. Fix the decoder-init bug

In a copy of `train_second_v10.py` under this task's `code/` (do not modify t0010's completed task
folder — copy forward and adapt, per the corrections convention), add `"decoder"` to the
`ignore_modules` list at the `load_checkpoint()` call site, **unless** Key Question 1 finds a
`hifigan`-shaped pretrained checkpoint worth fine-tuning from instead — in that case, point
`first_stage_path` at it and leave `decoder` fine-tuning from those weights (not random init).

Immediately verify the fix with a tensor-level check (reuse/adapt
`t0013/code/inspect_checkpoint.py`) confirming `decoder` is either genuinely random-initialized or
genuinely loaded from a `hifigan`-shaped source — not partially overwritten by mismatched shapes.
Do this before launching any GPU training.

### 2. Switch to the t0012 normalized corpus

Point `train_data` at `tasks/t0012_v5_corpus_normalize_and_reaudit/data/train_list_v5_normalized_clean.txt`
(1531 clips) instead of t0010's `data/data_list_v5_train_250.txt` subset. Keep `val_data` pointed
at the same held-out `val_96` set t0008/t0009/t0010 all used — never train on it.

### 3. Train

Reuse t0009's safeguard library (JSONL logger, health gates, per-epoch `CheckpointManager`) and
t0010's `joint_epoch=8` + DP-aware checkpoint loader fix unchanged. Follow the same
`setup-remote-machine` VM-pool flow t0010 used (`LLM-T1-NC80`), with the idle watchdog deployed
before any long-running step — t0010's own teardown got stuck on a full root disk; watch disk usage
proactively this time rather than discovering it at teardown.

### 4. Mandatory audible-speech gate before claiming completion

Run `t0013/code/infer_styletts2.py` (or a copy adapted for this task) against the new best
checkpoint, synthesizing the same texts t0013 used for direct comparability. Run
`audio_quality_check.py`'s `is_likely_noise` heuristic on the output.

* **If it passes** (`is_likely_noise=False`): proceed to step 5.
* **If it fails**: stop, document the failure the same way t0013 did (control-checkpoint
  comparison, clip fraction / spectral flatness numbers, root-cause if identifiable), and do **not**
  produce the audio-samples/inference-recipe deliverables below as if they succeeded. File a
  follow-up suggestion instead.

### 5. Deliverables (only once the gate in step 4 passes)

The user explicitly asked for three things out of this task, not just a checkpoint:

1. **The best checkpoint** — the standard `model` asset (`expected_assets: {"model": 1}`), same
   packaging convention as `kokoro-v10-best`.
2. **Audio samples**, paired original vs. FT, same convention as t0013's
   `results/audio_samples/{original,ft}/` — real files the user can listen to, not just metrics.
3. **A working inference recipe** — the exact script and steps used to go from checkpoint to audio
   in step 4, committed under `code/`, instrumented the same way t0013's `infer_styletts2.py` is
   (missing/unexpected key logging on load) so nobody has to reconstruct this from scratch again.

## Compute and Budget

GPU training on `LLM-T1-NC80` (2xH100), same pool as t0010. t0010 planned ~$45-100 and actually
spent $272.78 (19.54h, mostly idle overnight due to a watchdog gap and a full-disk teardown delay)
— budget and plan explicitly for disk monitoring and prompt teardown this time. Write
`results/costs.json` with the actual total.

## Expected Outputs

- `code/train_second_v11.py` (or similarly named) — the corrected training script, copied forward
  from t0010's, with the `ignore_modules` fix (or pretrained-hifigan-checkpoint fix) applied and
  explained.
- `code/inspect_checkpoint.py`-derived pre-flight check output confirming the decoder-init fix
  actually took effect, before training started.
- `results/audio_samples/{original,ft}/` — only if the step 4 gate passes.
- `code/infer_styletts2.py` (or equivalent) — the inference recipe, only if the step 4 gate passes.
- `results/results_summary.md` / `results_detailed.md` — training outcome, gate result (pass or
  fail), and — if the gate failed — an honest account of what was tried and what's recommended
  next, matching t0013's standard of evidence over assertion.
- `assets/model/<new-checkpoint-id>/` — only if the step 4 gate passes.

## Dependencies

- `t0010_stage2_safeguarded_training` — source of `train_second_v10.py`, the `joint_epoch=8` fix,
  and the DP-aware checkpoint loader, all carried forward unchanged except the decoder-init bug.
- `t0012_v5_corpus_normalize_and_reaudit` — source of the 1531-clip normalized training manifest.
- `t0013_v10_synthesis_quality_forensics` — source of the root-cause diagnosis, the
  `audio_quality_check.py` noise gate, and the `infer_styletts2.py` inference recipe this task
  reuses and extends.
