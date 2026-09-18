---
spec_version: "3"
task_id: "t0021_zero_shot_latency_reduction"
step_number: 4
step_name: "research-papers"
status: "completed"
started_at: "2026-09-18T10:17:55Z"
completed_at: "2026-09-18T10:30:00Z"
---
## Summary

Reviewed the project's existing paper corpus for literature relevant to CosyVoice2/Chatterbox TTFB
optimization and wrote `research/research_papers.md`, synthesizing findings on zero-shot cloning
architectures, streaming/chunked decoding, flow-matching vocoder latency, and LLM-serving
acceleration to ground this task's per-stage profiling methodology.

## Actions Taken

1. Spawned a dedicated subagent (per Critical Rule 9) to execute the `/research-papers` skill for
   `t0021_zero_shot_latency_reduction`, unrestricted, per the skill's own instructions.
2. The subagent found `meta/categories/` empty for this project (only `.gitkeep`), confirmed via
   `aggregate_categories`, so triaged the full 11-paper corpus manually against t0021's five stated
   research areas instead of filtering by category.
3. The subagent wrote `research/research_papers.md` (10 papers cited of 11 reviewed), organized Key
   Findings into 7 topic subsections (additive latency model, AR-vs-NAR architectural floor,
   streaming/chunking quality cost, RTF-vs-TTFB distinction, LM-serving acceleration as the likely
   dominant lever, reference-encoding caching/encoder-swap caution, cross-encoder metric
   incompatibility), plus two additional sections (`## Benchmark Comparison`,
   `## Gaps in Latency-Specific Published Data`).
4. Ran `uv run flowmark --inplace --nobackup` on the produced markdown file.
5. As step-executor, independently re-ran the verificator via `run_with_logs` to confirm the
   subagent's self-reported PASSED result:
   `uv run python -m arf.scripts.verificators.verify_research_papers t0021_zero_shot_latency_reduction`
   — result: PASSED, 0 errors, 1 warning (`RP-W003`, expected: this project has no registered
   categories, so the frontmatter placeholder does not match any real category slug).

## Outputs

* `tasks/t0021_zero_shot_latency_reduction/research/research_papers.md` — 10 papers cited (Du2024
  CosyVoice2, Chen2024 F5-TTS, Seo2026 Chatterbox-Flash, Wan2018 GE2E, Casanova2022 YourTTS,
  Kunesova2025 speaker-encoder study, Du2024a CosyVoice v1, Kong2020 HiFi-GAN, Kaneko2022 iSTFTNet,
  Li2023 StyleTTS2).
* `tasks/t0021_zero_shot_latency_reduction/logs/commands/006_20260918T102558Z_uv-run-python.*` —
  command log for the step-executor's independent verificator run.

## Issues

No issues encountered. The one verificator warning (`RP-W003`) is an expected, unavoidable artifact
of this project having no `meta/categories/` entries registered yet, not a defect in the research
document.
