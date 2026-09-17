---
spec_version: "3"
task_id: "t0018_zero_shot_cloning_calibration"
step_number: 6
step_name: "research-code"
status: "completed"
started_at: "2026-09-17T14:19:02Z"
completed_at: "2026-09-17T14:32:00Z"
---
## Summary

Spawned a subagent to execute the `/research-code` skill, which reviewed the `tts_eval_harness`
library from t0008 plus code and lessons from t0013, t0014, t0015, and t0010, and wrote
`research/research_code.md` (5 tasks cited, 2 libraries surveyed). In parallel, dispatched the 4
remaining queued `/add-paper` subagents from step 5 (respecting the max-3-concurrent cap, with the
4th dispatched once a slot freed) — all 4 succeeded, bringing the paper corpus to 11 papers (4
pre-existing + 7 discovered by `research-internet`). After all research steps completed, spawned a
`/research-summarize` subagent that wrote `research/research_summary.md`.

## Actions Taken

1. Ran `prestep` for `research-code`, then in a single batch spawned 4 subagents concurrently: 3
   `/add-paper` subagents for the still-queued discovered papers (`Du2024-CosyVoice1`,
   `Casanova2022-YourTTS`, `Zhang2025-ECAPA`) and 1 `/research-code` subagent, per Rule 9 (never
   execute skill logic inline) and the paper-addition protocol's max-3-concurrent cap.
2. The `/research-code` subagent reviewed the library aggregator (2 libraries: `tts_eval_harness`
   relevant, `t0009_training_safeguards` not relevant), the answer aggregator (1 answer, not
   relevant), and the task aggregator (15 completed tasks), deep-diving into t0008
   (`tts_eval_harness`: `adapters.py`, `harness.py`, `scoring.py`, `report.py`), t0013/t0014/t0015
   (audible-speech gate hardening lineage and `build_reference_concat.py`), and t0010 (GPU
   cost-overrun precedent). Wrote `research/research_code.md` with all 7 mandatory sections.
3. Independently verified `research/research_code.md` exists and ran
   `uv run python -m arf.scripts.utils.run_with_logs --task-id t0018_zero_shot_cloning_calibration -- uv run python -m arf.scripts.verificators.verify_research_code t0018_zero_shot_cloning_calibration`,
   which passed with 0 errors and 0 warnings.
4. All 3 first-batch `/add-paper` subagents completed successfully: `Du2024-CosyVoice1` (commit
   `0268535`, paper_id `10.48550_arXiv.2407.05407`), `Casanova2022-YourTTS` (commit `a305347`,
   paper_id `10.48550_arXiv.2112.02418`), and `Zhang2025-ECAPA` — which, on download, turned out to
   be misidentified by the research-internet snippet: the real paper is "An Exploration of
   ECAPA-TDNN and x-vector Speaker Representations in Zero-shot Multi-speaker TTS" by Kunešová,
   Hanzlíček, and Matoušek (TSD 2025), correctly re-labeled `Kunesova2025` (paper_id
   `10.48550_arXiv.2506.20190`, commit `4bb35d8`).
5. Once a concurrency slot freed, dispatched the 4th queued paper, `ChatterboxFlash2026`. It also
   completed successfully but was likewise misidentified by the snippet: the real title is
   "Chatterbox-Flash: Prior-Calibrated Block Diffusion for Streaming Zero-Shot TTS" by Seo, Park,
   and Nam (paper_id `10.48550_arXiv.2605.30748`, commit `a5f7160`).
6. Confirmed via `aggregate_papers --format json --detail short` that the corpus now holds 11 papers
   total (up from 4 before this task started), including all 7 papers discovered in
   `research/research_internet.md`'s `## Discovered Papers` section. No paper additions failed, so
   no inline fallback or `intervention/` file was needed.
7. Spawned a `/research-summarize` subagent, which compressed `research_papers.md`,
   `research_internet.md`, and `research_code.md` into `research/research_summary.md` (112 lines, ~8
   KB, all mandatory sections present).

## Outputs

* `tasks/t0018_zero_shot_cloning_calibration/research/research_code.md` — 7/7 mandatory sections, 5
  tasks cited, `status: "complete"`.
* `tasks/t0018_zero_shot_cloning_calibration/research/research_summary.md` — compact summary of all
  three research files for downstream planning/implementation subagents.
* `tasks/t0018_zero_shot_cloning_calibration/assets/paper/10.48550_arXiv.2407.05407/` — CosyVoice
  (predecessor) paper asset.
* `tasks/t0018_zero_shot_cloning_calibration/assets/paper/10.48550_arXiv.2112.02418/` — YourTTS
  paper asset.
* `tasks/t0018_zero_shot_cloning_calibration/assets/paper/10.48550_arXiv.2506.20190/` — Kunesova2025
  (ECAPA-TDNN/x-vector speaker representations) paper asset.
* `tasks/t0018_zero_shot_cloning_calibration/assets/paper/10.48550_arXiv.2605.30748/` —
  ChatterboxFlash2026 paper asset.
* `tasks/t0018_zero_shot_cloning_calibration/logs/commands/*` — command logs for the verificator run
  and each `/add-paper` subagent's downloads/checks (each subagent staged and committed its own
  command logs alongside its asset).

## Issues

Two of the four dispatched papers (`Zhang2025-ECAPA`, `ChatterboxFlash2026`) had inaccurate
titles/authorship in the `research-internet` search-snippet-derived dispatch metadata (expected —
step 5 flagged these as "authorship not independently verified"). Both `/add-paper` subagents
independently re-verified against the actual downloaded PDF/arXiv metadata, corrected the
citation_key and title in `details.json`/`summary.md` to match reality, and both verificators still
passed. No paper addition failed outright, so the inline-fallback path in the paper-addition
protocol was not needed. Separately, two subagents independently reported that the `/add-paper`
skill document's Phase 6 verificator command (`arf.scripts.verificators.verify_paper_asset`)
references a module path that does not exist in this repo — the real module is
`meta.asset_types.paper.verificator`. This is a documentation bug in a skill/spec file and is out of
scope for this task per CLAUDE.md Rule 0 (framework changes are not task work); noting it here for a
future `self-improvement` pass.
