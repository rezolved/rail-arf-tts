---
spec_version: "1"
task_id: "t0014_v11_decoder_fix_retrain"
updated_at: "2026-09-16T15:29:51Z"
completed_steps: 6
next_step_number: 6
next_step_id: "research-code"
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

### Step 5 — research-internet

Resolved Key Question 1: `yl4579/StyleTTS2-LibriTTS`'s `epochs_2nd_00020.pth` is a genuinely
`hifigan`-shaped pretrained checkpoint that field-matches `config_david_v10.yml`'s decoder block
(`resblock_kernel_sizes: [3,7,11]`, `upsample_initial_channel: 512`, `upsample_rates: [10,5,3,2]`,
`multispeaker: true`). Findings on file at `research/research_internet.md`
(`verify_research_internet.py` passed, 0 errors/warnings). One new paper discovered ("GAN Vocoder:
Multi-Resolution Discriminator Is All You Need", arXiv:2103.05236); its `/add-paper` subagent was
spawned and runs in parallel with subsequent steps.

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
* **Decoder-init fix direction changed by internet research.** Step 5 found that upstream
  `train_second.py`'s own default `ignore_modules` also omits `"decoder"` — t0013's bug is better
  framed as an architecture-mismatch assumption violation (istftnet-shaped `first_stage_v3.pth` vs.
  a hifigan-shaped config) than a missing-exclusion bug. The recommended fix is now to repoint
  `first_stage_path` at `yl4579/StyleTTS2-LibriTTS`'s `epochs_2nd_00020.pth` (a real hifigan-shaped
  checkpoint), not to add `"decoder"` to `ignore_modules` and accept random init. Planning must
  resolve licensing use-terms disclosure (MIT code license; pretrained weights carry separate
  consent/disclosure terms per `research_internet.md`) and treat random-init retraining only as a
  fallback if this checkpoint path fails.
* **Epoch-count anchor updated.** If fine-tuning from the LibriTTS checkpoint, the official
  `Configs/config_ft.yml` recipe (50 epochs, `diff_epoch: 10`, `joint_epoch: 30`, ~1k-sample scale,
  corroborated by two independent community fine-tunes) is a closer corpus-scale anchor than
  `research_papers.md`'s VCTK/LibriTTS from-scratch numbers, and should supersede v10's 20/8 budget
  in planning — for the from-scratch fallback, `research_internet.md` also found 400-1,000 Stage-1
  epochs on ~150 files as order-of-magnitude confirmation the old budget was short.

* * *

## Next Step Notes

Step 5 (`research-internet`) completed — `research/research_internet.md` resolves Key Question 1:
use `yl4579/StyleTTS2-LibriTTS`'s `epochs_2nd_00020.pth` (hifigan-shaped, field-matches
`config_david_v10.yml`) instead of random-init decoder training. Proceed to step 6 (`research-code`)
per step_tracker.json: review t0009's safeguard library, t0010's `train_second_v10.py`/`joint_epoch`
fix/DP-aware loader, and t0013's `inspect_checkpoint.py`/
`audio_quality_check.py`/`infer_styletts2.py` for reuse — and check whether any of those scripts
already assume the old random-init/`ignore_modules` fix path, since that assumption is now
superseded. One `/add-paper` subagent (arXiv:2103.05236) is still running in the background; confirm
its completion before `compare-literature` or `reporting`.
