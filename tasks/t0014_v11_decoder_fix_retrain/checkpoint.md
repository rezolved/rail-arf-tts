---
spec_version: "1"
task_id: "t0014_v11_decoder_fix_retrain"
updated_at: "2026-09-16T15:16:11Z"
completed_steps: 5
next_step_number: 5
next_step_id: "research-internet"
---
# Task Objective

Fix the ignore_modules bug that left v10's HiFi-GAN decoder worse than random, retrain on t0012's
1531-clip normalized corpus, and gate completion on an actual audible-speech check.

* * *

## Step History

### Step 1 — create-branch

Branch `task/t0014_v11_decoder_fix_retrain` created. Initial folder structure initialized in
`tasks/t0014_v11_decoder_fix_retrain/`. Step 1 is a mechanical setup step with no research output.

### Step 2 — check-deps

Ran `verify_task_dependencies.py` (via prestep, then again wrapped in `run_with_logs` for the audit
trail) — PASSED with no errors or warnings. All three declared dependencies
(`t0010_stage2_safeguarded_training`, `t0012_v5_corpus_normalize_and_reaudit`,
`t0013_v10_synthesis_quality_forensics`) are `completed`. Wrote
`logs/steps/002_check-deps/deps_report.json`.

### Step 3 — init-folders

Ran `init_task_folders` (wrapped in `run_with_logs`), creating the mandatory directory structure
(`plan/`, `research/`, `results/`, `results/images/`, `corrections/`, `intervention/`, `code/`,
`logs/commands/`, `logs/searches/`, `logs/sessions/`, `logs/steps/`, `assets/model/`) plus
`__init__.py` and `code/__init__.py`. Wrote `logs/steps/003_init-folders/folders_created.txt`.
Populated the local aggregator cache at `tasks/t0014_v11_decoder_fix_retrain/ctx/` (task_types,
costs, tasks, metrics, suggestions) — gitignored, not committed.

### Step 4 — research-papers

The project corpus had zero paper assets and zero `meta/categories/` entries across all 13 prior
tasks, so the `/research-papers` subagent added three papers via `add-paper` before writing
findings: StyleTTS 2 (`10.48550/arXiv.2306.07691`), HiFi-GAN (`10.48550/arXiv.2010.05646`), and
iSTFTNet (`10.48550/arXiv.2203.02395` — the architecture `first_stage_v3.pth`'s decoder actually is,
per t0013's diagnosis). Key finding for Key Question 2: v10/v11's `epochs_2nd: 20` /
`joint_epoch: 8` budget is well below every published HiFi-GAN-family from-scratch schedule reviewed
(HiFi-GAN/ iSTFTNet train to 2.5M steps from scratch; StyleTTS2's own HifiGAN-decoder configs use
50+40 epochs on VCTK, 30+25 on LibriTTS) — no paper matches this project's exact 1,531-clip scale,
so the gap is reported as order-of-magnitude, not a precise epoch count (see Gaps and Limitations in
`research/research_papers.md`). `verify_research_papers.py` passed with 0 errors, 1 warning
(`RP-W003`, pre-existing project-wide absence of `meta/categories/`, documented in the file).

### Step 11 — creative-thinking

Skipped: task scope is a well-defined diagnostic fix (decoder-init bug) plus corpus-expansion
retrain with an explicit audible-speech gate; the Key Questions that call for alternative approaches
(pretrained-checkpoint search, epoch-count sizing) are already covered by the research and planning
steps.

* * *

## Cross-Step Decisions

* Corpus had no paper assets or categories before this task; three papers (StyleTTS 2, HiFi-GAN,
  iSTFTNet) were added under this task's `assets/paper/` to ground Key Question 2 and 3 findings.
  `categories_consulted` is `[]` project-wide — not a fixable gap within this task's scope (Key Rule
  0).

* * *

## Next Step Notes

Step 4 (`research-papers`) completed — findings on file at `research/research_papers.md` show the
20-epoch/8-joint-epoch v10/v11 budget is likely undersized versus published HiFi-GAN-family
from-scratch schedules (order-of-magnitude gap, no exact match at this corpus scale). Proceed to
step 5 (`research-internet`) per step_tracker.json: search for a hifigan-shaped pretrained StyleTTS2
first-stage checkpoint (Key Question 1) before committing to the random-init decoder path.
