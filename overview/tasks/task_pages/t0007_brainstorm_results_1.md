# ✅ Brainstorm results session 1

[Back to all tasks](../README.md)

## Overview

| Field | Value |
|---|---|
| **ID** | `t0007_brainstorm_results_1` |
| **Status** | ✅ completed |
| **Started** | 2026-09-14T13:00:00Z |
| **Completed** | 2026-09-14T14:30:00Z |
| **Duration** | 1h 30m |
| **Dependencies** | [`t0001_kokoro_v4_stage2_finetune`](../../../overview/tasks/task_pages/t0001_kokoro_v4_stage2_finetune.md), [`t0002_kokoro_v4_voicepack_decoder_package`](../../../overview/tasks/task_pages/t0002_kokoro_v4_voicepack_decoder_package.md), [`t0003_kokoro_v5_phoneme_data`](../../../overview/tasks/task_pages/t0003_kokoro_v5_phoneme_data.md), [`t0004_kokoro_v5_stage1_train`](../../../overview/tasks/task_pages/t0004_kokoro_v5_stage1_train.md), [`t0005_kokoro_v5_stage2_train`](../../../overview/tasks/task_pages/t0005_kokoro_v5_stage2_train.md), [`t0006_kokoro_v5_stage2_subset`](../../../overview/tasks/task_pages/t0006_kokoro_v5_stage2_subset.md) |
| **Task types** | `brainstorming` |
| **Step progress** | 4/4 |
| **Task folder** | [`t0007_brainstorm_results_1/`](../../../tasks/t0007_brainstorm_results_1/) |
| **Detailed results** | [`results_detailed.md`](../../../tasks/t0007_brainstorm_results_1/results/results_detailed.md) |

<details>
<summary><strong>Task Description</strong></summary>

*Source:
[`task_description.md`](../../../tasks/t0007_brainstorm_results_1/task_description.md)*

# Brainstorm Results Session 1

First human brainstorming session for rail-arf-tts. Reviews the six completed tasks
(t0001-t0006), all of which were Kokoro fine-tuning attempts, and decides what to do next. The
researcher agreed that the project cannot measure its own success criteria yet and that the
repeated Stage 2 training failures need a systematic forensic review before more GPU is spent.

</details>

## Research

* [`research_code.md`](../../../tasks/t0007_brainstorm_results_1/research/research_code.md)
* [`research_internet.md`](../../../tasks/t0007_brainstorm_results_1/research/research_internet.md)
* [`research_papers.md`](../../../tasks/t0007_brainstorm_results_1/research/research_papers.md)

<details>
<summary><strong>Results Summary</strong></summary>

*Source:
[`results_summary.md`](../../../tasks/t0007_brainstorm_results_1/results/results_summary.md)*

# Brainstorm Results Session 1

## Summary

First brainstorm after six Kokoro fine-tuning tasks (t0001-t0006) that produced no clean-audio
checkpoint and measured none of the project's success criteria. The session created **2 new
tasks** (t0008 evaluation harness and baselines, t0009 Stage 2 training failure forensics and
safeguards) and raised the project budget from $500 to **$5000**. No suggestions existed, so
none were rejected or reprioritized.

## Session Overview

* Date: 2026-09-14.
* Context: t0001-t0006 were all Kokoro StyleTTS2 fine-tuning attempts. t0003 found and fixed a
  grapheme fallback that poisoned 21% of the v4 manifests. t0005 fixed seven crashes but every
  Stage 2 run diverged after the GAN switched on. t0006 stabilized the GAN with
  `lambda_gen=0.05` but the audio is still noisy (best val 0.846 vs v3's 0.506).
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
   voicepack, the v3 shipped bundle, t0006 v6d ep6 and t0005 run06 best on speaker_sim, TTFB,
   RTF, plus duration-ratio and WER sanity checks.
2. **Create t0009 `stage2_training_failure_forensics`.** Researcher's own request: read all
   logs, check data, pipeline and checkpoint saving/loading, understand what broke, and add
   checkpoint and log-based protection against future breakage. No new training.
3. **Raise `project/budget.json` `total_budget` from 500 to 5000 USD.** Researcher's
   instruction. `per_task_default_limit` left at $100.
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
  `total_cost_usd` in t0001-t0006 costs (via corrections), and refresh the stale "Current
  Phase" and Nebius reference in `project/description.md`.

</details>

<details>
<summary><strong>Detailed Results</strong></summary>

*Source:
[`results_detailed.md`](../../../tasks/t0007_brainstorm_results_1/results/results_detailed.md)*

# Brainstorm Results Session 1 — Detailed

## Summary

Reviewed t0001-t0006, created t0008 (evaluation harness and baselines) and t0009 (Stage 2
training failure forensics and safeguards), and raised the project budget from $500 to $5000.
See `results_summary.md` for decisions and rationale.

## Methodology

1. Ran `aggregate_tasks`, `aggregate_suggestions --uncovered`, `aggregate_answers`,
   `aggregate_costs`.
2. Read `results_summary.md`, `results_detailed.md`, `metrics.json`, `costs.json` and
   `task_description.md` for all six completed tasks, plus t0005 `logs/README.md` and t0003's
   Stage 1 duration probe.
3. Diffed the t0005 and t0006 Stage 2 configs; checked which logs, checkpoints and reference
   audio are tracked in git or DVC.
4. Presented state, answered the researcher's questions (in Russian), agreed on two tasks and
   a budget change.
5. Created the task folders, wrote the records, ran verificators.

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

## Limitations

Planning task, no experiments run. Cost figures for t0001-t0006 quoted in the session are read
from their `costs.json` `total_usd` fields plus an estimate for t0006's unrecorded GPU time
(~5 h at $13.96/h); they were not re-derived from Azure billing.

## Files Created

* `tasks/t0007_brainstorm_results_1/` — this task.
* `tasks/t0008_tts_eval_harness_baselines/task.json`, `task_description.md`.
* `tasks/t0009_stage2_training_failure_forensics/task.json`, `task_description.md`.
* `project/budget.json` — `total_budget` 500.0 → 5000.0.
* `overview/` — regenerated by the materializer.

## Verification

* `verify_task_file` t0007 — PASSED, 0 errors, 1 warning (TF-W005, expected: no assets).
* `verify_task_file` t0008, t0009 — PASSED, 0 errors, 0 warnings.
* `verify_corrections` t0007 — PASSED, 0 errors.
* `verify_suggestions` t0007 — PASSED, 0 errors.
* `verify_logs` t0007 — PASSED, 0 errors, 3 warnings. LG-W007 and LG-W008:
  `capture_task_sessions` crashed with `UnicodeDecodeError` in
  `_jsonl_content_has_worktree_cwd` (`arf/scripts/utils/capture_task_sessions.py:362` opens
  JSONL as strict UTF-8). A transcript on disk contains invalid UTF-8, most likely truncated
  when the local disk filled up during this session. LG-W004 is that failed capture command.
  Framework fix (`errors="replace"`) belongs in a separate infrastructure change.

</details>
