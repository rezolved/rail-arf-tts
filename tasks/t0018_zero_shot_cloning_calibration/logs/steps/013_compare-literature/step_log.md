---
spec_version: "3"
task_id: "t0018_zero_shot_cloning_calibration"
step_number: 13
step_name: "compare-literature"
status: "completed"
started_at: "2026-09-17T19:52:42Z"
completed_at: "2026-09-17T20:05:00Z"
---
## Summary

Compared this task's measured GE2E-cosine `speaker_sim`/WER results against the closest published
numbers for the same three named systems (F5-TTS, CosyVoice2, Chatterbox) using the paper corpus
landed in steps 5-6, producing `results/compare_literature.md` while explicitly flagging the
GE2E-cosine vs. WavLM/ERes2Net-cosine metric mismatch on every row.

## Actions Taken

1. Read `tasks/t0018_zero_shot_cloning_calibration/checkpoint.md` in full, plus
   `results/results_detailed.md`, `results/metrics.json`, and `results/tables.json`, to identify the
   measured values and the metric-mismatch constraint repeated across prior steps.
2. Spawned a dedicated subagent (in the task worktree, on
   `task/t0018_zero_shot_cloning_calibration`) to execute `/compare-literature` per
   `arf/skills/compare-literature/SKILL.md`, with an explicit instruction that published
   SIM-o/SS/MOS numbers from the corpus papers (WavLM-large, ERes2Net, or WavLM-ECAPA-TDNN based)
   must never be merged numerically with this project's GE2E-cosine `speaker_sim` — any comparison
   must be framed as order-of-magnitude/qualitative only.
3. The subagent traced every published value to an exact table in the source PDF
   ([Chen2024, Table 1/2], [Du2024, Table 5/6], [Seo2026, Table 1]) rather than relying on the
   research summaries, and produced an 8-row comparison table, each row carrying an explicit
   **METRIC MISMATCH** flag in its Notes column.
4. Reviewed the produced `results/compare_literature.md` in full: confirmed the metric-mismatch
   framing is present and unambiguous throughout `## Summary`, every table row,
   `## Methodology Differences`, `## Analysis`, and `## Limitations`; confirmed no row presents a
   numeric ranking as apples-to-apples.
5. Ran `uv run flowmark --inplace --nobackup` on the file (no-op; already correctly formatted) via
   `run_with_logs`.
6. Ran `verify_compare_literature.py` via `run_with_logs` — **PASSED, 0 errors, 0 warnings**.

## Outputs

* `tasks/t0018_zero_shot_cloning_calibration/results/compare_literature.md` (new) — YAML
  frontmatter, `## Summary`, `## Comparison Table` (8 data rows across F5-TTS/CosyVoice2/Chatterbox,
  each flagged with the GE2E-vs-WavLM/ERes2Net metric mismatch), `## Methodology Differences`,
  `## Analysis` (including a `### Prior Task Comparison` subsection reconciling this task's
  re-measured `elevenlabs_david` baseline against t0008's cited numbers, and flagging that
  CosyVoice2's `ref_single` result contradicts the implicit prior assumption that 0.85 was
  structurally unreachable under GE2E-cosine scoring), `## Limitations`.
* `tasks/t0018_zero_shot_cloning_calibration/logs/commands/080_*`, `081_*` — command logs for the
  flowmark and verificator runs.

## Issues

No issues encountered. The subagent's self-run verificator check and this step-executor's
independent `run_with_logs`-wrapped re-run both confirm 0 errors/0 warnings.
