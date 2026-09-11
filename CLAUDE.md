# rail-arf-tts — Rezolve TTS Research

Autonomous-research project to evaluate and fine-tune Kokoro-82M as a drop-in replacement for
ElevenLabs David voice in Rezolve's voice commerce filler synthesis pipeline. Goal: match
ElevenLabs speaker similarity (GE2E cosine ≥ 0.85) and latency (TTFB ≤ 300 ms) at a fraction of
the cost.

This repo is a private fork of `rezolved/rail-arf` (Rezolve's canonical ARF fork-base). See
`project/description.md` for the full goal and success criteria.

## Commands

```bash
# Setup
uv sync                                            # Install deps
uv run pre-commit install                          # Activate git hooks
python3 doctor.py                                  # Validate environment

# Development
uv run python -u <script.py>                                       # Run a script
uv run python -m arf.scripts.utils.run_with_logs --task-id <id> -- <cmd>   # ARF logging

# Quality
uv run flowmark --inplace --nobackup <path.md>      # Format markdown
uv run ruff check --fix . && uv run ruff format .   # Lint and format Python
uv run mypy .                                        # Type check
uv run pytest                                        # Run framework tests in arf/tests

# DVC (large data files)
dvc pull                                            # Download all tracked data files
dvc push                                            # Upload new/updated data files
dvc add <file-or-dir>                               # Track a new large file with DVC
```

## GPU Machine

GPU training runs on **LLM-T1-NC80** — an Azure ML 2×H100 SXM5 VM. Connection via SSH alias
`LLM-T1-NC80` defined in `~/.ssh/config`. Pool config: `project/azure_vm.json`.

The VM pool manager lives in `arf/scripts/utils/azure_ml_vm.py`; the `setup-remote-machine` skill
drives it. Fill in the `workspace` field in `project/azure_vm.json` before first use.

Stage 2 finetune environment: `kokoro-finetune/` repo on the VM, config at
`configs/config_david_v4.yml`. Always start Stage 2 from `first_stage.pth` with
`load_only_params: true` and `multispeaker: true`.

## Idle VM Prevention

Azure ML H100 instances bill by the hour even when idle. Two layers prevent runaway spend:

**Layer 1 — Orchestrator** (`arf/scripts/utils/azure_ml_vm.py`): `teardown(task_id,
deallocate=True)` clears the task lock, kills stray processes, and calls `stop_compute(vm=target_vm)`
when no other task lock remains. This is the normal shutdown path when an ARF step completes.

**Layer 2 — VM-side dead-man's switch** (`arf/scripts/utils/idle_watchdog.sh`): runs on the VM,
polls `nvidia-smi` every 60 s. If all GPUs stay at ≤ 5% utilization for 60 min it executes
`TERMINATE_CMD` — protecting against missed wakeups, crashed orchestrators, or fire-and-forget
handoffs.

### Deploying the watchdog

SSH into the VM, copy `arf/scripts/utils/idle_watchdog.sh` if not present, then:

```bash
# Set TERMINATE_CMD for LLM-T1-NC80 (fill in workspace from project/azure_vm.json)
export TERMINATE_CMD="az ml compute stop --name LLM-T1-NC80 \
  --workspace-name brainpowa-northeurope --resource-group rezolve-AI"

export IDLE_THRESHOLD_SECONDS=3600   # 60 min idle → terminate
export POLL_INTERVAL_SECONDS=60
export IDLE_UTIL_PERCENT=5           # ≤5% GPU util = idle
export GRACE_SECONDS=600             # 10 min arming delay after boot
export WATCHDOG_LOG=/var/log/arf_idle_watchdog.log

nohup bash /path/to/arf/scripts/utils/idle_watchdog.sh >> "$WATCHDOG_LOG" 2>&1 &
echo "Watchdog PID $!"
# Verify: ps aux | grep idle_watchdog && tail -f "$WATCHDOG_LOG"
```

If `az` is not authenticated on the VM, run `az login --use-device-code` first. On `TERMINATE_CMD`
failure the watchdog logs a warning and keeps retrying — it does NOT exit.

Full command: `az ml compute stop --name LLM-T1-NC80 --workspace-name brainpowa-northeurope --resource-group rezolve-AI`

## DVC data workflow

Large data files (audio clips, model checkpoints) are tracked by DVC and stored in Azure Blob
Storage at `azure://ml-dvc-datasets/datasets/rail-arf-tts` (account: `mldvcstorerezolve`). Git
commits only the small `.dvc` pointer files — not the data bytes.

**After `git pull`, always run `dvc pull` to sync data.**

Key rules for task agents:

* Any task that produces audio files or model checkpoints MUST gitignore the data and track it
  with `dvc add`.
* Run `dvc push` before merging the task PR so teammates can `dvc pull` the data.
* NEVER commit raw audio blobs to git — only `.dvc` pointer files are committed.

## Key References

* Project description and goals: `project/description.md`
* ARF architecture and glossary: `arf/README.md`
* Lessons from prior Rezolve projects: `LESSONS.md`
* Python style guide: `arf/styleguide/python_styleguide.md`
* Markdown style guide: `arf/styleguide/markdown_styleguide.md`
* Models overview: `overview/models/`
* Datasets overview: `overview/datasets/`
* Metrics overview: `overview/metrics/`

## Benchmark

Primary benchmark: 96 held-out val clips (`data/v4/val/val_list.txt`) + 1358 ElevenLabs David
reference clips (`data/11labs_david/`) as speaker-similarity ground truth.

Primary metrics: `ttfb_ms` (time to first audio byte), `speaker_sim` (GE2E cosine vs 11labs ref),
`rtf` (real-time factor).

NEVER train on val_96 — it is a held-out regression set only.

## Rezolve conventions

* Always write **brainpowa** in lowercase — never "Brainpowa" or "BrainPowa".
* Use the `gh` CLI for GitHub operations. Never paste credentials into commits, logs, or agent
  prompts.
* Never add a "Generated with Claude Code" promo line to commit messages or PR descriptions.
* External communications (PRs, Jira, Confluence, Slack) go in English even when prompting in
  Russian.
* The default GPU provider is **Azure ML** (configured via `project/azure_vm.json`). vast.ai is
  supported as a fallback if a project declares it in `available_services`.

## Key Rules

0. Framework / infrastructure / specification / skill / verificator / aggregator / materializer
   changes in `arf/`, generic `meta/`, and generic boilerplate are not task work. Do not create a
   `tasks/tXXXX_*` folder for such changes.
1. All CLI tool calls MUST be wrapped in `arf/scripts/utils/run_with_logs.py`.
2. One task = one folder = one branch = one PR.
3. **NEVER** modify files outside the task folder. Only top-level tooling files may change:
   `pyproject.toml`, `uv.lock`, `ruff.toml`, `.gitignore`.
4. Each task stage and each action is a separate well-described commit.
5. Nothing in a completed task folder may be changed; use the corrections mechanism in later tasks.
6. Run `uv run flowmark --inplace --nobackup <changed.md>` on edited markdown files, then run
   `uv run ruff check --fix . && uv run ruff format . && uv run mypy .` before commit.
7. Framework tests live in `arf/tests/`. Task-specific tests live in
   `tasks/$TASK_ID/code/test_*.py`. Do not create or use a top-level `tests/` directory.
8. Full data normalization; no duplication across task folders.
9. **Always use aggregators to enumerate cross-task data.** Never walk `tasks/` with
   Glob/Grep/find/Explore.
10. Read `LESSONS.md` — and `project/LESSONS.md` if it has one — before planning a task that
    involves latency benchmarks, GPU provisioning, or paired-bootstrap analysis.

## Task Workflow

* Tasks live in `tasks/tXXXX_slug/` (t prefix + 4-digit ID + underscore slug).
* Each task runs in its own git worktree on branch `task/<task_id>`.
* New tasks branch: `new_tasks/<first_index>-<last_index>`.
* Mandatory stages: research → planning → implementation → analysis → reporting.
* Every step must be logged in `logs/`; verificators enforce this.

## Provenance

rail-arf-tts is a fork of `rezolved/rail-arf`. There is no `upstream` remote. Modify `arf/`
freely — those are Rezolve's now.
