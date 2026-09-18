---
spec_version: "3"
task_id: "t0018_zero_shot_cloning_calibration"
step_number: 5
step_name: "research-internet"
status: "completed"
started_at: "2026-09-17T13:58:23Z"
completed_at: "2026-09-17T14:12:04Z"
---
## Summary

Spawned a subagent to execute the `/research-internet` skill, which ran 19 documented web searches
plus 4 targeted fetches and wrote `research/research_internet.md` covering install/inference
requirements, checkpoints, licensing, and streaming/TTFB behavior for F5-TTS, CosyVoice 2, and
Chatterbox — the primary source for these three systems since the paper corpus has zero coverage of
them. The verificator passed with zero errors and zero warnings.

## Actions Taken

1. Ran `prestep` for `research-internet`, then spawned a dedicated subagent to execute the
   `/research-internet` skill per Rule 9 (never execute skill logic inline).
2. The subagent conducted 19 web search queries and 4 targeted page fetches (F5-TTS/CosyVoice
   2/Chatterbox GitHub repos, inference READMEs, licensing discussions, streaming-latency issue
   threads) and wrote `research/research_internet.md` with all 8 mandatory sections, 18 cited
   sources, and a 7-entry `## Discovered Papers` section.
3. Verified the output file exists and ran
   `uv run python -m arf.scripts.utils.run_with_logs --task-id t0018_zero_shot_cloning_calibration -- uv run python -m arf.scripts.verificators.verify_research_internet t0018_zero_shot_cloning_calibration`,
   which passed with 0 errors and 0 warnings.
4. Cross-checked the 7 discovered papers against `aggregate_papers` (4 existing corpus papers:
   You2021, Kong2020, Kaneko2022, Li2023) — confirmed none overlap, so all 7 are genuinely new.
5. Spawned the first batch of 3 `/add-paper` subagents (max 3 concurrent, per the paper-addition
   protocol) for the highest-priority discovered papers: `Chen2024-F5TTS` (F5-TTS paper),
   `Du2024-CosyVoice2` (CosyVoice 2 paper), and `Wan2018-GE2E` (foundational GE2E
   speaker-verification paper behind this project's own `speaker_sim` metric). These run in parallel
   with subsequent steps and are tracked in `checkpoint.md` for the `compare-literature`/`reporting`
   steps to check on.

## Outputs

* `tasks/t0018_zero_shot_cloning_calibration/research/research_internet.md` — 8/8 mandatory
  sections, 18 sources cited, 7 papers discovered, `status: "complete"`.
* `tasks/t0018_zero_shot_cloning_calibration/logs/commands/005_*` — command logs for the verificator
  run.
* Three in-flight `/add-paper` subagents (not yet committed; their asset files will land in
  `assets/paper/` and be picked up by a later step's `git status` check once they complete).

## Issues

No issues encountered. 4 of the 7 discovered papers (`Du2024-CosyVoice1`, `Casanova2022-YourTTS`,
`Zhang2025-ECAPA`, `ChatterboxFlash2026`) remain queued for `/add-paper` dispatch — not yet started,
to respect the max-3-concurrent limit — and are recorded in `checkpoint.md` as pending.
