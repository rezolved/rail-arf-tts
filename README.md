# TTS Research — Kokoro-82M vs ElevenLabs for Rezolve Fillers

## Goal

Replace ElevenLabs David voice (currently used for filler synthesis in Rezolve's voice commerce
assistant) with a self-hosted Kokoro-82M fine-tuned model that matches ElevenLabs speaker
similarity and latency at a fraction of the cost. The current ElevenLabs integration costs $0.30
per 1000 characters; the target is a local model with TTFB ≤ 300 ms and GE2E cosine similarity
≥ 0.85 against the ElevenLabs David reference.

## Research Questions

1. What is the TTFB, speaker similarity (GE2E cosine), and RTF of the ElevenLabs David baseline
   on the 1358-clip filler corpus?
2. How does Kokoro-82M base (no fine-tuning) compare to ElevenLabs David on speaker similarity
   and RTF?
3. Does fine-tuning Kokoro-82M on the David voice dataset (1557 clips, StyleTTS2 Stage 2) close
   the speaker-similarity gap to ≥ 0.85 GE2E cosine while keeping TTFB ≤ 300 ms?
4. What is the minimum viable checkpoint (epoch) for the fine-tuned Kokoro v4 that meets the
   quality bar, and does further training help?
5. Is Kokoro-82M v4 a viable ElevenLabs replacement for Rezolve fillers on cost, quality, and
   latency grounds?

## Success Criteria

- ElevenLabs David baseline benchmarked: TTFB, speaker_sim, RTF on ≥ 50 filler prompts.
- Kokoro-82M base benchmarked on the same prompt set.
- Fine-tuned Kokoro v4 achieves speaker_sim (GE2E cosine) ≥ 0.85 on val_96.
- Fine-tuned Kokoro v4 achieves TTFB ≤ 300 ms on the filler corpus (local inference, H100).
- Viability report produced with a clear go / no-go recommendation.

## Current Phase

Setup complete. Next: run ElevenLabs baseline benchmark (t0002) and Kokoro base eval (t0003) in
parallel, then evaluate fine-tuned Kokoro v4 once Stage 2 training completes (t0004).

## Results Dashboard

See [`overview/README.md`](overview/README.md) for aggregated metrics, task status, and the
per-model results breakdown.

## Getting Started

```bash
git clone https://github.com/rezolved/rail-arf-tts.git
cd rail-arf-tts
uv sync                         # Install Python deps
uv run pre-commit install       # Activate git hooks
cp .dvc/config.local.example .dvc/config.local   # Fill in Azure connection string
dvc pull                        # Download training audio and tracked results
uv run python3 doctor.py        # Validate environment
```

Do **not** re-run `/setup-project` — that is a one-time initialization for fresh forks.

## Daily Workflow

```bash
# Plan a new task
/create-task

# Execute a planned task
/execute-task t0002_elevenlabs_baseline

# After a task completes, generate next task suggestions
/human-brainstorm

# Regenerate the results dashboard
uv run python -m arf.scripts.overview.materialize
```

## Key Rules

These are enforced by verificators — violations block commits.

- **Every CLI call** is wrapped in `uv run python -m arf.scripts.utils.run_with_logs -- <cmd>` so
  logs are captured.
- **Tasks only modify files inside their own folder.** The only top-level files a task may touch
  are `pyproject.toml`, `uv.lock`, `ruff.toml`, and `.gitignore`.
- **Every task stage and every action is a separate, well-described commit.**
- **Completed task folders are immutable.** Fix mistakes via correction files in a new task, never
  by editing past folders.
- **Read through aggregators, never walk task folders directly.** Raw globs miss the corrections
  overlay.
- **Metrics must be registered in `meta/metrics/` before a task reports them.**
- **Audio and model files go in DVC, never in git.** Run `dvc add` then `dvc push` before merging.
- **NEVER train or tune on val_96** — it is a held-out regression set only.

## Project Structure

```text
arf/            Framework code: scripts, skills, specifications, styleguide, docs, tests
meta/           Project metadata: categories/, metrics/, task_types/
tasks/          One folder per research task (created by the create-task skill)
overview/       Materialized aggregator dashboard (regenerated, committed)
project/        Project-level files: description.md, budget.json, azure_vm.json
data/           Training and eval audio (DVC-tracked, not in git)
.dvc/           DVC config and cache pointers
.claude/        Claude Code config (settings.json, rules/, skills/ symlinks)
.codex/         Codex CLI config (agents/, skills/ symlinks)
CLAUDE.md       Project overview loaded at Claude Code session start
pyproject.toml  Python deps and tooling config
doctor.py       Environment validation script
```

## Datasets

- **val_96** — 96 held-out clips from the David voice corpus. Primary regression set. Never
  used for training.
- **fillers_1358** — 1358 ElevenLabs David reference clips. Speaker-similarity ground truth for
  all evaluations.

## Metrics

- 🎯 `speaker_sim` — Speaker Similarity (GE2E cosine vs ElevenLabs David reference) · `cosine`
- ⚡ `ttfb_ms` — Time to First Audio Byte · `ms` (lower is better)
- 📊 `rtf` — Real-Time Factor · `ratio` (lower is better)

## Task Types

**Project-specific:**

- `tts-benchmark-run` — TTS Benchmark Run
- `tts-finetuning-eval` — TTS Fine-Tuning Evaluation

**Generic** (built-in ARF types):

- `answer-question`, `baseline-evaluation`, `brainstorming`, `build-model`, `comparative-analysis`,
  `correction`, `data-analysis`, `experiment-run`, `infrastructure-setup`, `literature-survey`

## Budget and Services

Total budget: **$500 USD** · per-task default limit: $100 · Services: `anthropic_api`,
`openai_api`, `elevenlabs_api`

GPU: **LLM-T1-NC80** — Azure ML H100 NVL ×2 (`brainpowa-northeurope`, $13.96/hr). Managed via
`arf/scripts/utils/azure_ml_vm.py` + `project/azure_vm.json`.

## Documentation

- [`arf/docs/explanation/safety.md`](arf/docs/explanation/safety.md) — autonomy and safety risks
- [`arf/docs/tutorial/`](arf/docs/tutorial/) — walkthrough from empty fork to first results
- [`arf/docs/reference/`](arf/docs/reference/) — glossary, task folder structure, verificators,
  aggregators, skills
- [`arf/styleguide/python_styleguide.md`](arf/styleguide/python_styleguide.md) — Python style guide
- [`arf/styleguide/markdown_styleguide.md`](arf/styleguide/markdown_styleguide.md) — Markdown style
  guide

## License

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for details.
