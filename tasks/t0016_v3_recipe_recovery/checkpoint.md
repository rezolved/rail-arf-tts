---
spec_version: "1"
task_id: "t0016_v3_recipe_recovery"
updated_at: "2026-09-17T15:45:00Z"
completed_steps: 11
next_step_number: 10
next_step_id: "teardown"
---
# Task Objective

Reconstruct the exact Kokoro v3 Stage 2 recipe (config, data list, patches, epochs, environment)
from VM files, DVC artifacts and checkpoint forensics, without access to the original author.

* * *

## Step History

### Step 1 — create-branch

Branch `task/t0016_v3_recipe_recovery` created. Initial folder structure initialized in
`tasks/t0016_v3_recipe_recovery/`. Step 1 is a mechanical setup step with no research output.

### Step 2 — check-deps

Ran `verify_task_dependencies.py` (via `prestep` and again via `run_with_logs` for the log record):
both declared dependencies, `t0006_kokoro_v5_stage2_subset` and
`t0009_stage2_training_failure_forensics`, have `status: "completed"` in their `task.json`, and the
verificator reported PASSED with 0 errors and 0 warnings. Result recorded in
`logs/steps/002_check-deps/deps_report.json`. No caveats.

### Step 3 — init-folders

Created the mandatory task folder structure via `init_task_folders` (12 directories with `.gitkeep`,
plus `__init__.py` and `code/__init__.py`), recorded in
`logs/steps/003_init-folders/folders_created.txt`. Populated the local, gitignored aggregator
context cache at `tasks/t0016_v3_recipe_recovery/ctx/` (task_types, costs, tasks, metrics,
suggestions) for reuse by downstream step-executors. No caveats.

### Step 4 — research-papers

Skipped, per `step_tracker.json`: this task is checkpoint/VM forensics, not literature-driven — no
papers in the corpus bear on Kokoro/StyleTTS2 recipe recovery from local artifacts.

### Step 5 — research-internet

Skipped, per `step_tracker.json`: `task_description.md`'s evidence sources are entirely internal (VM
home dir, DVC artifacts, prior task checkpoints/code); no external internet research is required or
listed as an evidence source.

### Step 6 — research-code

Wrote `research/research_code.md` (13 tasks cited, 2 libraries documented; verificator PASSED, 0
errors/0 warnings). Confirmed the four tasks named in `task_description.md` are directly reusable:
`t0006`'s `code/config_david_v6c_stage2.yml` is the reconstruction template; `t0015`'s
`code/predictor_tensor_forensics.py` and `code/audio_quality_check.py` are the checkpoint-diff and
audio-gate scripts to copy into this task's `code/`; `t0002`'s `extract_decoder_generic.py` confirms
the packaging recipe; `t0009`'s `checkpoint_manager.py`/`health_gates.py`/`confound_table.md` supply
the forensics conventions and the open confound ledger. Caveat for downstream: found a genuine
three-way `multispeaker` contradiction across `best/config.json` (true), `t0009`'s confound table
(true, assumed), and `t0006`'s `config_david_v6c_stage2.yml` (false, inline-commented) — none
carries a SHA-256, so only checkpoint-shape forensics in the implementation step can resolve it.

### Step 11 — creative-thinking

Skipped, per `step_tracker.json`: `task_description.md` fully specifies the evidence sources,
priority order, and reconstruction methodology; no exploratory alternative-approach analysis is
called for.

### Step 13 — compare-literature

Skipped, per `step_tracker.json`: this task reconstructs an internal training recipe from artifacts;
it does not produce results comparable to published external baselines.

### Step 7 — planning

Wrote `plan/plan.md` (17 `REQ-*` items, 6 milestones, 20 numbered steps; verificator PASSED, 0
errors/0 warnings). The plan schedules: bounded VM inspection with a hard 90-minute wall-clock timer
and unconditional teardown step regardless of progress; checkpoint forensics generalizing `t0015`'s
`predictor_tensor_forensics.py` to all five v3 bundle modules to compute weight-norm deltas and
resolve the `multispeaker` three-way contradiction via module presence/shape (not source
preference); per-epoch sample gating with `t0015`'s `audio_quality_check.py`; the annotated
`data/config_david_v3_reconstructed.yml` with confirmed/inferred/unknown provenance on every line;
the mandatory human-listenable audio set (`v3_shipped`, `v3_per_epoch`, `elevenlabs_reference`,
`listening_guide.md`); the `v3_module_weight_delta.png` chart; and the `v3-recipe` answer asset.
Cost itemized at ~$21 VM + $0 local/API against the task's $30 cap. Caveat for downstream: while
planning, a live `dvc pull` against the v3 reference checkpoint failed with a transient Azure
`DefaultAzureCredential` auth error (cross-checked against `t0015`'s command logs as a known,
retry-resolvable failure) — the plan's Step 1 and Step 13 both build in bounded retry-with-backoff
for `dvc pull`/`dvc push` rather than treating one failure as a hard blocker.

### Step 8 — setup-machines

Acquired `LLM-T1-NC80` (2xH100 NVL, the project's sole Azure ML pool entry) via
`/setup-remote-machine` through Phase 5. GPU/CUDA verified (`2x NVIDIA H100 NVL`, CUDA 12.2), idle
watchdog installed and confirmed alive (PID 6143, 3600s idle timeout), and environment
sanity-checked (`~/kokoro-finetune/` present, `/mnt/cache/persist` resolves to the real Azure Files
share, SSH lingering enabled). Result recorded in `logs/steps/008_setup-machines/machine_log.json`.
Caveat for downstream: the first `acquire` attempt hit a one-time boot-timing race (VM's own Azure
`Start` operation hadn't finished within the tool's 480s SSH-readiness window), burning ~9.5 minutes
of the 90-minute VM cap and writing `intervention/pool_busy_llm-t1-nc80.md`; a retry succeeded in
~16 seconds with no wasted cost. The VM is left **running** with the watchdog active and locked for
this task — the `implementation` step must complete its read-only inventory and hand off to
`teardown` within the remaining budget (~80 of the original 90 minutes left).

### Step 9 — implementation

Executed `plan/plan.md`'s full 20-step, 6-milestone Step by Step. **Milestone 1 (VM inventory):**
completed the read-only SSH inventory of `~/kokoro-finetune`/`/mnt/cache/persist` in 13.47 minutes
of VM wall time (well under the 90-minute cap) and called `azure_ml_vm teardown` itself immediately
afterward rather than leaving the VM running for step 10 (see Cross-Step Decisions) —
`~/bash_history` was found to contain unrelated other-project credentials on this shared VM pool and
was deliberately redacted before committing, per CLAUDE.md's "never paste credentials" rule
(`data/vm_inventory/raw_dump.txt` documents the redaction). Key negative finding:
`~/kokoro-finetune` is t0014's own later StyleTTS2 clone, not a preserved v3-era environment — no
`first_stage_v3.pth`, no v3 launch config, no 266-clip list survived (REQ-4, REQ-10 unresolved,
honestly documented). VM's `models.py` did survive and served as the tie-breaker for `multispeaker`.
**Milestones 2-6 (local CPU):** checkpoint-shape forensics across all 5 bundle modules plus
`net["diffusion"]` resolved `multispeaker` to `inferred false` (all 5 modules show substantial
weight-norm shifts, ruling out "decoder frozen"); scored 55 per-epoch/variant samples plus 8
freshly-synthesized shipped-bundle samples with the audio-quality gate (`v3` epoch 6-8 all noise,
`v3b` clean, explaining t0002's `v3b` citation); wrote the 93-field annotated
`data/config_david_v3_reconstructed.yml`; generated `results/images/v3_module_weight_delta.png`;
wrote and DVC-pushed the full mandatory human-listening set (`v3_shipped`, `v3_per_epoch`,
`elevenlabs_reference`, `listening_guide.md`, all links verified resolving); wrote the `v3-recipe`
answer asset (confidence `"low"`, verificator PASSED 0/0). Caveat for downstream: `dvc pull`/`push`
hit the documented transient `DefaultAzureCredential` failure twice and was resolved by minting a
short-lived Azure CLI user-delegation SAS token into the gitignored `.dvc/config.local` (not
committed) rather than waiting out the ~9-minute transient window again; this local workaround is
not portable and a future task hitting the same error should either retry-with-backoff (as
originally planned) or repeat this SAS workaround, not assume `.dvc/config.local` persists across
worktrees. Locally re-measured `speaker_sim=0.566` fell outside the ±0.02 tolerance against t0008's
recorded numbers; investigated and attributed to a reference-corpus/pre-filter mismatch (documented
in `results/v3_checkpoint_forensics.md`), not a bundle regression — treat this task's audio-quality
conclusions as resting on the checkpoint forensics and audio gate, not this speaker_sim cross-check.

* * *

## Cross-Step Decisions

* Planning (step 7) fixed the VM inspection budget at a hard 90-minute wall-clock cap with an
  unconditional teardown step (Step 7 of the plan) — this overrides "keep investigating" instincts
  if the cap is hit before all VM evidence sources are covered.

* Implementation (step 9) deviated from the canonical step-lifecycle assumption that the dedicated
  `teardown` step (step 10) makes the `azure_ml_vm teardown` call: instead, the implementation
  step-executor called `azure_ml_vm teardown t0016_v3_recipe_recovery` itself immediately after the
  SSH inventory copy finished (`vm_teardown_called_at: 2026-09-17T14:47:56Z`,
  `vm_teardown_by: "implementation step (this step), not a later teardown step"` recorded in
  `data/vm_inventory/inventory.json`), per plan.md Milestone 1 Step 7's explicit instruction and the
  orchestrator's own time-pressure guidance not to hold the VM open through the CPU-only forensics
  work. Step 10 (`teardown`) should expect to find `LLM-T1-NC80` already deallocated and the task
  lock already cleared — its job is to confirm/reconcile that state (and write
  `remote_machines_used.json` / cost reconciliation per `remote_machines_specification.md`), not to
  issue a fresh stop call.

* * *

## Next Step Notes

Step 9 (`implementation`) is complete. **VM usage: 13.47 minutes of the 90-minute cap**
(`vm_billing_started_at: 2026-09-17T14:34:28.703Z` →
`vm_teardown_called_at: 2026-09-17T14:47:56.755Z`), roughly $3 of the $21 VM sub-cap — the VM is
**already stopped/deallocated**, not left running. Step 10 (`teardown`) should verify this (e.g.
`az ml compute show`/pool-lock state) rather than assume the VM is still up, and write its own step
artifacts (`remote_machines_used.json`, cost reconciliation) against the already-completed teardown
recorded in `tasks/t0016_v3_recipe_recovery/data/vm_inventory/inventory.json` and
`logs/steps/008_setup-machines/machine_log.json`.

Key findings from implementation, for the `results`/`suggestions`/`reporting` steps downstream:
`~/kokoro-finetune` on the VM turned out to be a symlink to t0014's own later StyleTTS2 clone, not a
preserved v3-era environment — no `first_stage_v3.pth`, no v3 launch config, and no 266-clip list
survived anywhere on the VM or `/mnt/cache/persist` (REQ-4 unrecoverable, documented in
`data/v3_train_list_UNRECOVERED.md`; REQ-10 byte-identity also unconfirmed). The VM's `models.py`
DID survive and was the tie-breaker for the `multispeaker` contradiction: checkpoint-shape forensics
on `net["diffusion"]` plus `models.py:808`'s `build_model()` branch landed on a verdict of
**`inferred false`** (not `confirmed` — the external diffusion-package class shapes were not fully
recovered), independently corroborating v6c's uncited inline claim over `best/config.json` and
t0009's confound table. All five bundle modules (including the decoder) show substantial weight-norm
shifts Stage 1 → best, ruling out "decoder frozen" as the explanation for v3's speaker_sim gain. A
secondary finding worth flagging to `suggestions`: cross-checking t0009's confound table against
v6c's actual committed file caught a factual error in the confound table (`lambda_gen` claimed
`1.0`, actual v6c value is `0.2`) — the confound table's other "assumed" values should be treated
with corresponding skepticism. The `v3` audio variant's epoch 6-8 samples are all
`is_likely_noise=True` while the parallel `v3b` variant (same epochs) is clean, explaining why t0002
cites `v3b/ep9_p5.wav` specifically. This task's own local re-synthesis of the shipped v3 bundle
scored `speaker_sim=0.566`, outside the ±0.02 tolerance against t0008's recorded `0.631`/`0.588` —
investigated and attributed to a documented reference-centroid-building deviation (t0008's own
`MIN_CLIP_DURATION_S=1.6` pre-filter now rejects nearly the entire current `11labs_david` corpus),
not treated as a bundle-quality regression; flagged as weak evidence either way in
`results/v3_checkpoint_forensics.md`.

All expected assets exist and pass verification: the `v3-recipe` answer asset
(`assets/answer/v3-recipe/`) passes `meta.asset_types.answer.verificator` with 0 errors/0 warnings,
confidence `"low"` (honestly reflects REQ-4/REQ-6/REQ-8/REQ-10 landing short of "confirmed"/"high").
REQ-1, REQ-2, REQ-3, REQ-5, REQ-7, REQ-9, REQ-11, REQ-12, REQ-13, REQ-16, REQ-17 are `done`; REQ-6
and REQ-8 are `partial` (verdict reached but not at the `confirmed` bar); REQ-4 and REQ-10 are
`blocked` (evidence genuinely does not exist, documented rather than fabricated) — all per the
task's own Rejection Criteria, which pre-registers an all-`inferred`/`unknown` reconstruction as a
valid outcome. `code/`, `data/`, and `results/` all pass `ruff check`, `ruff format`, and `mypy`
with 0 issues. `results/audio_samples/` is DVC-tracked and pushed (`dvc status` clean, no pending
push). Proceed to step 10, `teardown`.
