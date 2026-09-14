# Brainstorm Session 1 — Full Transcript

Structured summary. The session was held partly in Russian at the researcher's request; the
exchanges are recorded here in English.

## Project State Presented

Six completed tasks, zero suggestions, zero answers. Cost aggregator reported $0 spent (all six
tasks skipped because costs.json uses `total_usd` instead of `total_cost_usd`).

| Task | Finding |
| --- | --- |
| t0001 v4 Stage 2 | 8 epochs, best val 0.751 at ep6, GAN diverged ep7 (1.147). ~$147. |
| t0002 v4 packaging | Five-module extraction bug fixed; pipeline validated on v3. v4 checkpoint broken: 10× duration explosion, 3-3.5 kHz band. |
| t0003 v5 phonemes | Root cause: grapheme fallback in 21% of v4 manifests (329/1557). v5 1557/1557 and 96/96 clean. $0. |
| t0004 v5 Stage 1 | Best val 0.740; single GPU only. ~$126. |
| t0005 v5 Stage 2 | 6 launches, 7 fixes; all diverged within 4 post-GAN epochs. Best 0.848. ~$98. |
| t0006 250-clip subset | 4 runs; `lambda_gen=0.05` stable, audio still noisy. Best 0.846 vs v3 0.506. |

Problems raised:

1. Budget: ~$370 recorded plus ~$70 unrecorded (t0006) ≈ 88% of $500, invisible to the aggregator.
2. t0006 v6d changed six variables (data, multispeaker, Stage 1 checkpoint, joint_epoch, lambda_gen,
   LR 3e-5 → 1e-4); t0004's Stage 1 was bypassed.
3. `val_loss` comparison against v3's 0.506 may not be like-for-like, and is not a project metric.
4. Stale `description.md`; t0004 checkpoint label mismatch; v5 data outside `data/`; no suggestions
   recorded.

Proposed: eval harness first (ElevenLabs, Kokoro base, v3, v6d); voicepack-only test; train again
only if a gap remains; infra fixes.

## Clarification Questions

AI: Any notes? Budget (~$60 left)? Pause training until harness exists? Success bar still ≥ 0.85 and
≤ 300 ms? Fix costs.json first?

Researcher: Asked for the analysis again in plain Russian: what was done, what worked, what did not.

AI: Gave a plain-language Russian summary: goal, the six tasks, what worked (data bug found and
fixed, packaging fixed, Stage 2 no longer crashes or diverges), what did not (no clean-audio model,
no success metrics ever measured, budget nearly spent while the tracker shows $0, t0006 changed too
much at once). Proposed: learn to measure first, check whether v3 is already close, retrain only if
needed.

## Discussion — Round 1: New Tasks

Researcher: Agrees about measurement and wants one task for it. Did not understand what v3 is. Also
wants to understand how to stop training from breaking: go through the logs, understand what broke,
check whether training and checkpointing were done correctly, check the data and the pipeline, check
checkpoint saving, protect against breakage through checkpoints, and use logs to verify and explain
why it broke.

AI: Explained v3: an earlier fine-tune from before this repo, 266 clips, the only run with clean
audio (val 0.506), artifacts in `t0006/data/reference/v3`. Summarized what the logs already show
(Dur loss on the first Stage 2 step separates v3 success 0.81 from failures 8-17; v3 success
depended on which `first_stage.pth` was present; `exp` overflow in `istftnet.py`; unexplained v3 →
v3b fix). Listed checkpoint/log suspicions (t0004 label, missing t0006 logs, epochs_2nd 10 vs 15,
six variables changed, top-2-by-val_loss retention). Proposed two tasks:

* A — evaluation harness and baselines.
* B — training failure forensics: collect all logs, per-run loss timelines, data audit, pipeline
  audit vs upstream StyleTTS2 and the seven patches, checkpoint audit, safeguard recipe (per-epoch
  checkpoints, auto-stop and rollback on health gates, structured logs).

Flagged that the disk was full and commands could not run.

Researcher: Agreed — create the two tasks; the researcher will launch them in separate branches.

Researcher (after freeing the disk): Continue.

Researcher (mid-work): Increase the budget to $5000.

## Discussion — Round 2: Suggestion Cleanup

No active suggestions existed (`aggregate_suggestions --uncovered` returned 0), so there was nothing
to reject or reprioritize.

## Discussion — Round 3: Confirmation

AI: Plan — t0007 brainstorm record, t0008 evaluation harness, t0009 training forensics, PR and merge
to main; researcher launches t0008 and t0009 separately.

Researcher: Confirmed ("agreed — make 2 tasks and I will launch them in separate branches").

## Decisions Summary

1. Create t0008_tts_eval_harness_baselines.
2. Create t0009_stage2_training_failure_forensics.
3. Raise project total budget from $500 to $5000.
4. No suggestion corrections (backlog empty).
