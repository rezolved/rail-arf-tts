---
spec_version: "1"
task_id: "t0014_v11_decoder_fix_retrain"
updated_at: "2026-09-16T23:39:04Z"
completed_steps: 15
next_step_number: null
next_step_id: null
---
# Task Objective

Fix the ignore_modules bug that left v10's HiFi-GAN decoder worse than random, retrain on t0012's
1531-clip normalized corpus, and gate completion on an actual audible-speech check.

* * *

## Step History

### Step 1 — create-branch

Trimmed to stay within 10 KB limit.

### Step 2 — check-deps

Trimmed to stay within 10 KB limit.

### Step 3 — init-folders

Trimmed to stay within 10 KB limit.

### Step 4 — research-papers

Trimmed to stay within 10 KB limit.

### Step 5 — research-internet

Trimmed to stay within 10 KB limit.

### Step 6 — research-code

Trimmed to stay within 10 KB limit.

### Step 7 — planning

Trimmed to stay within 10 KB limit.

### Step 8 — setup-machines

Trimmed to stay within 10 KB limit.

### Step 11 — creative-thinking

Skipped: task scope is a well-defined diagnostic fix (decoder-init bug) plus corpus-expansion
retrain with an explicit audible-speech gate; the Key Questions that call for alternative approaches
(pretrained-checkpoint search, epoch-count sizing) are already covered by the research and planning
steps.

### Step 9 — implementation

Resumed from `paused_waiting`; `resume_check`'s `job_dead` was independently confirmed over SSH as a
false-negative (training actually finished cleanly: `DONE` marker, 50/50 epochs, 0 gate firings).
Milestone C's mandatory audible-speech gate **passed** (`is_likely_noise=False`, `clip_fraction`
0.0048 vs. v10's confirmed-broken 0.750-0.807) against `epoch_00048.pth`, so Milestone D's
`kokoro-v11-best` model asset was created and independently verified (0 errors). See
`results/v11_gate_verdict.md` for full evidence; `LLM-T1-NC80` is left running for `teardown`.

### Step 10 — teardown

Confirmed no job was running, then a subagent executed the Teardown Protocol on `LLM-T1-NC80`: no
further downloads needed (step 9 already pulled the verified model asset), cleaned up this task's
own `/mnt/tmp/t0014_venv` scratch, and ran `azure_ml_vm teardown`. `machine_log.json` now has
`destroyed_at: 2026-09-16T23:11:25Z`, final `total_duration_hours: 6.436`, `total_cost_usd: 89.85`
(supersedes the interim ~$86.46 estimate). `results/remote_machines_used.json`/`costs.json` updated
to match. `verify_machines_destroyed.py` — PASSED, 0 errors, 2 non-blocking warnings.

### Step 12 — results

Finalized `results/results_summary.md`/`results_detailed.md` (already largely written during step 9)
against the full `task_results_specification.md`: corrected the stale pre-teardown cost prose (REQ-8
now `Done`, $89.85/6.436h), added `## Methodology` runtime/timestamps, and added a
`## Visualizations` section with two new charts (`results/images/val_loss_by_epoch.png`,
`audible_gate_comparison.png`). `verify_task_results`/`verify_task_metrics`/`verify_step` all
PASSED, 0 errors/0 warnings.

### Step 13 — compare-literature

Wrote `results/compare_literature.md`: v11's Stage 2 epoch schedule (50/10/30) exactly matches
StyleTTS2's official `config_ft.yml` fine-tune recipe, while its decoder-adversarial step volume
(~3,820 steps) is orders of magnitude below HiFi-GAN's/iSTFTNet's from-scratch/fine-tune budgets —
explained by fine-tuning an already-converged checkpoint. A `### Prior Task Comparison` table shows
v11's `speaker_sim` (0.444) beats both confirmed-broken v10 checkpoints (0.311-0.351, +0.093/+0.133)
but trails the unrelated-speaker control (0.482, -0.037) and the project's 0.85 target — confirming
the audible-speech gate, not `speaker_sim`, is the load-bearing pass/fail signal.
`verify_compare_literature.py` PASSED, 0 errors/0 warnings.

### Step 14 — suggestions

A subagent ran `/generate-suggestions`, cross-checking candidates against all 20 open project
suggestions and all 14 tasks via the aggregators before writing `results/suggestions.json` (4
entries, `S-0014-01`..`S-0014-04`): the duration-predictor calibration anomaly (high), adapting the
batch eval harness to the StyleTTS2-native inference path and scoring `kokoro-v11-best` on the real
val_96 + 1358-clip benchmark (high), porting You2021's discriminator-warmup/feature-matching-pause
rules into the t0009 safeguard library (medium), and a from-scratch decoder run to drop the LibriTTS
pretrained-weight dependency (low). `verify_suggestions.py` PASSED, 0 errors/0 warnings.

### Step 15 — reporting

Ran all remaining verificators (`verify_task_file`, `verify_task_dependencies`,
`verify_suggestions`, `verify_task_metrics`, `verify_task_results`, `verify_task_folder`,
`verify_logs`, `meta.asset_types.model.verificator kokoro-v11-best`, `verify_machines_destroyed`,
`verify_compare_literature`, `verify_research_papers`, `verify_research_internet`) — all PASSED, 0
errors. Removed a leftover gitignored `ctx/` aggregator-cache directory that tripped
`verify_task_folder`'s `FD-E016`. Ran `capture_task_sessions` (0 transcripts matched out of 360
candidates — recorded, non-blocking). Set `task.json` `status: "completed"`,
`end_time: "2026-09-16T23:39:04Z"`.

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
* **Plan finalized and verified.** `plan/plan.md` is `status: "complete"`, `verify_plan.py` passes 0
  errors/0 warnings. The plan supersedes `task_description.md`'s literal `joint_epoch=8 unchanged`
  instruction with `epochs: 50, diff_epoch: 10, joint_epoch: 30`. Cost estimate (~$84-150) was
  expected to exceed the $100 per-task default, pre-authorized with a $300 hard-stop threshold.
* **Two real bugs fixed during `implementation` (step 9), beyond the plan's original scope.** (1)
  The LibriTTS checkpoint's classic `torch.nn.utils.{weight_norm,spectral_norm}` key naming didn't
  match this fork's `parametrizations.*`-based `models.py`, so the first training launch silently
  partial-loaded despite the tensor-level pre-flight passing (which only checks decoder shape
  classification, not key-name load compatibility) — fixed via
  `_rename_legacy_parametrization_keys()` ported from t0013's `infer_styletts2.py`. (2) The training
  venv could not be built on `/mnt/cache/persist` (CIFS stalled on PyTorch's ~14k files), so it was
  built at `/mnt/tmp/t0014_venv/venv` instead (local ext4, fully regenerable, not the root disk, not
  a Lesson-10 violation since all actual data/checkpoints stayed on `/mnt/cache/persist`). A
  disk-space near-miss (root disk 98%/3GB free, t0010's exact failure mode recurring) was also found
  and mitigated (→87%/16GB free) before training started.
* **`LLM-T1-NC80` fully torn down at teardown (step 10).** Final measured cost is $89.85 over 6.436
  hours (`created_at` 16:45:14Z → `destroyed_at` 23:11:25Z) — the authoritative total for
  `results/costs.json` and any budget-reporting step downstream.

* * *

## Next Step Notes

Task complete. All 15 steps in `step_tracker.json` are `completed` or `skipped`. `task.json` has
`status: "completed"` with `start_time`/`end_time` both set. The `kokoro-v11-best` model asset
(satisfying `expected_assets: {"model": 1}`) is verified with 0 errors. Next: the coordinator
proceeds to Phase 7 (PR and merge), Phase 8 (final verification via `verify_task_complete.py`), and
Phase 9 (overview sync on `main`) — none of which are step-executor scope.
