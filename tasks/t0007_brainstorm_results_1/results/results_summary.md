# Brainstorm Results Session 1

## Summary

First brainstorm after six Kokoro fine-tuning tasks (t0001-t0006) that produced no clean-audio
checkpoint and measured none of the project's success criteria. The session created **2 new tasks**
(t0008 evaluation harness and baselines, t0009 Stage 2 training failure forensics and safeguards)
and raised the project budget from $500 to **$5000**. No suggestions existed, so none were rejected
or reprioritized.

## Session Overview

* Date: 2026-09-14.
* Context: t0001-t0006 were all Kokoro StyleTTS2 fine-tuning attempts. t0003 found and fixed a
  grapheme fallback that poisoned 21% of the v4 manifests. t0005 fixed seven crashes but every Stage
  2 run diverged after the GAN switched on. t0006 stabilized the GAN with `lambda_gen=0.05` but the
  audio is still noisy (best val 0.846 vs v3's 0.506).
* Prompt: researcher asked to analyze the repo and decide next steps.
* Key findings presented:
  * No speaker_sim, TTFB or RTF measurement exists; no ElevenLabs or base-Kokoro baseline.
  * `aggregate_costs` reports $0 because costs.json files use `total_usd`; real spend is ~$370
    recorded plus ~$70 unrecorded for t0006.
  * t0006's final run changed six variables at once, so the cause of divergence is unknown.
  * Only one training log is committed to git (`*.log` is gitignored).
  * Checkpoint labeling inconsistencies in t0004, t0005 and t0006.

## Decisions

1. **Create t0008 `tts_eval_harness_baselines`.** Researcher agreed that measurement must come
   first. Scores ElevenLabs David, Kokoro base (`bm_george`, `bm_lewis`), base decoder + v3
   voicepack, the v3 shipped bundle, t0006 v6d ep6 and t0005 run06 best on speaker_sim, TTFB, RTF,
   plus duration-ratio and WER sanity checks.
2. **Create t0009 `stage2_training_failure_forensics`.** Researcher's own request: read all logs,
   check data, pipeline and checkpoint saving/loading, understand what broke, and add checkpoint and
   log-based protection against future breakage. No new training.
3. **Raise `project/budget.json` `total_budget` from 500 to 5000 USD.** Researcher's instruction.
   `per_task_default_limit` left at $100.
4. **No suggestion changes.** The suggestions backlog is empty.

## Metrics

| Item | Count |
| --- | --- |
| Tasks created | 2 |
| Tasks cancelled | 0 |
| Tasks updated | 0 |
| Suggestions rejected | 0 |
| Suggestions reprioritized | 0 |
| New suggestions | 0 |
| Corrections written | 0 |

## Verification

* verify_task_file (t0007, t0008, t0009) — PASSED, 0 errors
* verify_corrections — PASSED, 0 errors
* verify_suggestions — PASSED, 0 errors
* verify_logs — PASSED, 0 errors; session capture warnings explained in results_detailed.md

## Next Steps

* Wave 1 (parallel, separate branches, launched by the researcher): t0008 and t0009.
* Wave 2: next training task, configured from t0009's answer and judged with t0008's harness.
* Infrastructure follow-ups noted but not decided in this session: fix `total_usd` →
  `total_cost_usd` in t0001-t0006 costs (via corrections), and refresh the stale "Current Phase" and
  Nebius reference in `project/description.md`.
