---
spec_version: "1"
task_id: "t0014_v11_decoder_fix_retrain"
updated_at: "2026-09-16T16:52:00Z"
completed_steps: 9
next_step_number: 9
next_step_id: "implementation"
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

### Step 6 — research-code

Reviewed `t0009`'s safeguard library, `t0010`'s `train_second_v10.py`, and `t0013`'s
`inspect_checkpoint.py`/`audio_quality_check.py`/`infer_styletts2.py`, writing
`research/research_code.md` (6 tasks cited of 13 reviewed, both project libraries assessed;
`verify_research_code.py` passed 0 errors/0 warnings). Pinpointed the bug to
`tasks/t0010_stage2_safeguarded_training/code/train_second_v10.py:253-259`'s `ignore_modules` list
omitting `"decoder"`. Confirmed t0010's `eval_all_checkpoints.py` (routes through Kokoro
`KModel`/`KPipeline`, which cannot load a `hifigan`-decoder checkpoint) and t0013's
`v10_diagnosis.md` (framed the fix as random-init retrain) both bake in the now-superseded
random-init assumption; also found `t0013`'s `infer_styletts2.py` already validated
`yl4579/StyleTTS2-LibriTTS`'s `epochs_2nd_00020.pth` as a known-good control, corroborating step 5's
repoint recommendation. Also ran the mandatory `research-summarize` step, producing
`research/research_summary.md` for planning/implementation to consume instead of the full research
files.

### Step 7 — planning

Spawned the `/planning` subagent with the budget summary from `ctx/costs.json` and the full step 1-6
context. It wrote `plan/plan.md` (11 mandatory sections, `spec_version: "2"`, `status: "complete"`);
`verify_plan.py` passed 0 errors, 0 warnings on first re-run. Chosen approach: repoint
`first_stage_path` at `yl4579/StyleTTS2-LibriTTS`'s `epochs_2nd_00020.pth` in a new
`config_david_v11.yml` (leaving `ignore_modules` unchanged), gated by a mandatory pre-flight tensor
check (`inspect_checkpoint.py`'s `classify_decoder()`) before any GPU spend, with random-init as a
documented fallback requiring an `intervention/` file. Epoch budget set to
`epochs: 50, diff_epoch: 10, joint_epoch: 30` (the `Configs/config_ft.yml` anchor), explicitly
overriding `task_description.md`'s literal "joint_epoch=8 unchanged" text as a called-out, resolved
ambiguity. The audible-speech gate is `audio_quality_check.py`'s `check_audio_quality()`
(`clip_fraction`/`spectral_flatness` heuristic) as the sole pre-registered pass/fail signal, not
`val_loss` or `speaker_sim`. Cost estimate: ~$84-150 (itemized from t0010's measured per-epoch
wall-clock, scaled 6.12x for corpus size and epoch count), explicitly flagged as exceeding the $100
per-task default, with a pre-registered $300 hard-stop escalation threshold.

### Step 8 — setup-machines

Provisioned `LLM-T1-NC80` (2xH100 NVL, Azure ML) via a `/setup-remote-machine` subagent. First
`acquire` hit `pool_busy` because the VM was mid an in-flight `az ml compute stop` raced by the
attempt (not a stale lock); the subagent's first hand-off claimed an untracked "background poller"
was watching for it — the exact fire-and-forget pattern Lesson 8 forbids — so it was resumed and
corrected to block synchronously instead. The retry succeeded (`ready_at: 2026-09-16T16:45:14Z`).
GPU/CUDA verified (2x H100 NVL, CUDA 12.2), idle watchdog installed and confirmed active with a live
PID (`watchdog_active: true`, 3600s idle timeout), `/mnt/cache/persist` verified as a real symlink
to the live Azure Files CIFS mount (Lesson 10), and `loginctl` linger confirmed enabled for
`azureuser` (Lesson 11) — all independently re-verified from raw command-log stdout, not just the
subagent's summary. `machine_log.json` at `logs/steps/008_setup-machines/machine_log.json`;
`verify_step` passes 0 errors/0 warnings. The project's `kokoro-finetune` env symlink was found
wiped (ephemeral `/mnt/tmp`, same Lesson 10 class of failure on the code checkout) — a generic conda
env satisfied the mandatory smoke test instead; re-establishing `kokoro-finetune` is deferred to
`implementation`.

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
* **Prior code confirmed to still carry the superseded random-init assumption.** `t0010`'s
  `eval_all_checkpoints.py` and `t0013`'s `v10_diagnosis.md` both assume the fix is "add
  `\"decoder\"` to `ignore_modules`" (random init); planning must not copy that framing forward
  as-is. `t0010`'s `train_second_v10.py` (with the `t0009` safeguards already wired in) and
  `t0013`'s
  `inspect_checkpoint.py`/`audio_quality_check.py`/`infer_styletts2.py`/`random_decoder_probe.py`
  are otherwise directly reusable (copy into task; only `tts_eval_harness`'s scoring functions, not
  its Kokoro-based synthesis adapters, are usable pre-v11).
* **Plan finalized and verified.** `plan/plan.md` is `status: "complete"`, `verify_plan.py` passes 0
  errors/0 warnings. The plan supersedes `task_description.md`'s literal `joint_epoch=8 unchanged`
  instruction with `epochs: 50, diff_epoch: 10, joint_epoch: 30`, documented explicitly as a
  resolved ambiguity per the plan spec's Task Requirement Checklist rules — implementation must use
  the plan's schedule, not the task description's literal text. Cost estimate (~$84-150) is expected
  to exceed the $100 per-task default like t0010 did; this is pre-authorized in the plan with a $300
  hard-stop escalation threshold, and project budget headroom (99.9% left) is not a constraint.
* **`LLM-T1-NC80` is live and billing.** Acquired and `ready` since 2026-09-16T16:45:14Z, watchdog
  active (3600s idle timeout, confirmed PID), `/mnt/cache/persist` verified as a real Azure Files
  mount. `implementation` must write all checkpoints under `/mnt/cache/persist/`, never bare `/mnt`.
  The `kokoro-finetune` conda/code environment on the VM needs to be re-created from scratch — its
  prior symlink target on ephemeral `/mnt/tmp` was wiped by an earlier VM stop, so
  `implementation`'s Milestone A/B setup cannot assume it still exists.
* **A subagent fire-and-forget near-miss occurred during setup-machines** (Lesson 8 pattern:
  claiming an untracked "background poller" instead of blocking synchronously or registering a
  `ScheduleWakeup`). Caught and corrected within the same step via `SendMessage` before any VM was
  billing idle. Future long-running steps on this task (especially `implementation`'s multi-hour
  training run) must use the sanctioned `paused_waiting` + `ScheduleWakeup` mechanism, never an
  agent-described background watcher, when a wait needs to outlive the current turn.

* * *

## Next Step Notes

`LLM-T1-NC80` (2xH100 NVL) is provisioned and `ready`: SSH/GPU/CUDA verified (CUDA 12.2), idle
watchdog active with a confirmed PID (3600s timeout), `/mnt/cache/persist` verified resolving to the
live Azure Files CIFS mount (Lesson 10), and `loginctl` linger confirmed enabled for `azureuser`
(Lesson 11). `machine_log.json` is at `logs/steps/008_setup-machines/machine_log.json`,
`verify_step` passes 0 errors/0 warnings. Proceed to step 9 (`implementation`) per
`step_tracker.json`: fix `ignore_modules` per plan Milestone A, re-create the `kokoro-finetune` env
on the VM (its prior symlink to `/mnt/tmp/kikiri-tts/StyleTTS2` was found wiped — build fresh, do
not assume it exists), run the mandatory pre-flight `classify_decoder()` tensor check on the
LibriTTS checkpoint **before any GPU training spend**, then train on t0012's 1,531-clip corpus with
the 50/10/30 epoch schedule, polling `df -h /` every 15-20 min per the plan's disk-usage risk
mitigation (t0010's stuck-teardown precedent), and finally gate completion on
`audio_quality_check.py`'s `check_audio_quality()` per REQ-6/REQ-7 — document a gate failure
honestly rather than fabricating a `model` asset.
