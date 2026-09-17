---
spec_version: "3"
task_id: "t0018_zero_shot_cloning_calibration"
step_number: 4
step_name: "research-papers"
status: "completed"
started_at: "2026-09-17T13:51:57Z"
completed_at: "2026-09-17T13:58:30Z"
---
## Summary

Reviewed the entire existing paper corpus for material relevant to F5-TTS, CosyVoice 2, Chatterbox,
and GE2E speaker-similarity evaluation, and wrote `research/research_papers.md` covering the four
papers currently in the corpus (all vocoder/StyleTTS2 papers added by an unrelated prior task), with
an explicit documented gap that none of the three target zero-shot cloning systems have corpus
papers yet.

## Actions Taken

1. Spawned a dedicated subagent to execute the `/research-papers` skill per
   `arf/skills/research-papers/SKILL.md`, after running prestep for the step.
2. The subagent ran `aggregate_categories` (found the project has no defined categories yet) and
   `aggregate_papers` (found 4 papers total in the corpus, none covering F5-TTS, CosyVoice 2,
   Chatterbox, or GE2E), then wrote `research/research_papers.md` with all seven mandatory sections,
   citing `[Kong2020]`, `[Kaneko2022]`, `[You2021]`, and `[Li2023]`, and documenting the corpus gap
   in `## Gaps and Limitations` and `## Recommendations for This Task`.
3. Ran `uv run flowmark --inplace --nobackup` on the research file and fixed two heading-length
   style violations it flagged.
4. Verified the output with
   `uv run python -m arf.scripts.utils.run_with_logs --task-id t0018_zero_shot_cloning_calibration -- uv run python -m arf.scripts.verificators.verify_research_papers t0018_zero_shot_cloning_calibration`:
   0 errors, 1 warning (`RP-W003`, expected — the project has no category taxonomy, so Paper Index
   `Categories` fields are empty for all four entries).

## Outputs

* `tasks/t0018_zero_shot_cloning_calibration/research/research_papers.md`
* `tasks/t0018_zero_shot_cloning_calibration/logs/commands/004_20260917T135705Z_uv-run-python.*`
  (verificator run log)

## Issues

No issues encountered. The one verificator warning (`RP-W003`) is expected and documented in the
research file's Category Selection Rationale — the project has not yet defined a category taxonomy
in `meta/categories/`, so no paper in the corpus has categories assigned.
