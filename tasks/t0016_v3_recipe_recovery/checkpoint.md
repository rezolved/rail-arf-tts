---
spec_version: "1"
task_id: "t0016_v3_recipe_recovery"
updated_at: "2026-09-17T16:27:09Z"
completed_steps: 15
next_step_number: null
next_step_id: null
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

### Step 10 — teardown

`LLM-T1-NC80` was already deallocated and this task's lock already cleared by step 9's own
`azure_ml_vm teardown` call, so this step verified and reconciled rather than re-stopping. That
earlier call omitted the billing-anchor flags and recorded a bogus `$0.00`/`0h`; this step
recomputed the real figures from Azure's own activity log against the correct billing window
(`14:34:28.703377Z` → `14:47:56.755481Z`) and wrote `total_duration_hours = 0.2245` (~13.47 min),
`total_cost_usd = $3.13` into `logs/steps/008_setup-machines/machine_log.json` (also correcting
`provider` from `"azure-ml"` to the spec-enum `"azure_ml"`), plus new
`results/remote_machines_used.json` and `results/costs.json`. A `az ml compute show` check found the
VM `Running` again at verification time — this is `t0018_zero_shot_cloning_calibration` legitimately
re-acquiring the shared pool VM afterward (confirmed via its live lock file and the Azure
activity-log timeline), not a failed teardown; no action was taken against it.
`verify_machines_destroyed` passed with 0 errors, 2 non-blocking warnings (legacy `spec_version`,
API-unreachable-from-sandbox), independently confirmed by both the subagent and this step-executor.
See `logs/steps/010_teardown/step_log.md` for full detail.

* * *

### Step 12 — results

Wrote `results/results_summary.md` and `results/results_detailed.md` (`spec_version: "2"`) directly
from the implementation step's already-produced outputs — no new forensics were run. Covered all 17
`REQ-*` items from `plan/plan.md`'s Task Requirement Checklist in the final
`## Task Requirement Coverage` section (12 `Done`, 3 `Partial` — REQ-6, REQ-8, REQ-14 — 2 `Not done`
— REQ-4, REQ-10, genuinely unrecoverable evidence, not a shortfall of effort), embedded and
described `results/images/v3_module_weight_delta.png`, wrote a 12-item `## Examples` section with
real fenced code/YAML/diff/JSON blocks (mandatory per
`meta/task_types/data-analysis/description.json`'s `requires_result_examples: true`), and documented
two plan-assumption contradictions under `## Analysis`: t0009's confound table is independently
wrong about `t0006_run03_v6c`'s own `lambda_gen` (claims `1.0`, actual committed value `0.2`), and
the VM's home directory being *present but from the wrong era* (t0014's later clone) is a distinct,
arguably worse failure mode than the plan's anticipated "directory gone" risk. `verify_task_metrics`
and `verify_task_results` both PASSED (0 errors/0 warnings) before and after `flowmark`.
`results/metrics.json`, `results/costs.json`, and `results/remote_machines_used.json` were verified
against `checkpoint.md`'s authoritative figures and left unmodified (all already correct). No caveat
for downstream — all cited facts trace to files already committed by `implementation`/`teardown`.

### Step 14 — suggestions

A dedicated subagent ran `/generate-suggestions`, gathering all task context and deduplicating
against 29 existing uncovered suggestions and 18 tasks, and wrote `results/suggestions.json` (5
suggestions, IDs `S-0016-01`..`S-0016-05`). Covers both REQ-14 mandatory seeds — the seed-42
stratified 266-clip fallback resample disjoint from `data/v4/val_list.txt` (`S-0016-01`) and the
two-arm `multispeaker` ablation to move REQ-6 from `inferred false` to confirmed (`S-0016-02`) —
plus the t0009 confound-table audit prompted by the `lambda_gen` discrepancy (`S-0016-03`), and two
self-surfaced findings: a `build_centroid()` duration-filter regression that now rejects the whole
`11labs_david` corpus in t0008's own harness (`S-0016-04`), and a VM working-directory
provenance-stamping convention to prevent a repeat of this task's "later clone mistaken for
preserved environment" failure mode (`S-0016-05`). `verify_suggestions` PASSED, 0 errors/0 warnings.
No caveat for downstream — all five suggestions are `status: "active"` and independently
verificator-clean.

### Step 15 — reporting

Ran the full reporting verification pass with no new forensics/results/suggestions work.
`verify_task_folder` initially failed on `FD-E016` (stray gitignored `ctx/` aggregator-cache
directory left over from step 3); removed the untracked `ctx/` dir and it passed clean. All other
verificators (`verify_task_file`, `verify_task_dependencies`, `verify_suggestions`,
`verify_task_metrics`, `verify_task_results`, `verify_logs`, the `v3-recipe` answer asset
verificator, `verify_machines_destroyed`) passed with 0 errors, only pre-existing/documented
warnings. `capture_task_sessions` ran and found 0 matching transcripts (recorded in
`logs/sessions/capture_report.json`), clearing `verify_logs`'s `LG-W007`/`LG-W008`. `task.json` set
to `status: "completed"`, `end_time: "2026-09-17T16:27:09Z"`. No caveat for downstream — this is the
final step-executor step.

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

* Teardown (step 10) found that step 9's own `azure_ml_vm teardown` call had omitted the
  `--billing-started-at`/`--billing-anchor` flags, so its self-reported `$0.00`/`0h` in
  `logs/commands/026_*` is wrong and must not be trusted by any downstream step reading that log —
  the authoritative figures are `total_duration_hours = 0.22445891777777777` and
  `total_cost_usd = 3.133446492177778`, now recorded in
  `logs/steps/008_setup-machines/machine_log.json` and
  `results/costs.json`/`results/remote_machines_used.json`. Step 10 also found `LLM-T1-NC80` back in
  `Running` state at verification time; this is `t0018_zero_shot_cloning_calibration` legitimately
  re-acquiring the shared pool VM afterward, not a teardown failure — downstream steps should not
  re-open this.

## Next Step Notes

Step 15 (`reporting`) is complete — this was the last step-executor step. All steps 1-15 are
`completed` or `skipped` in `step_tracker.json`; `task.json` now has `status: "completed"` and
`end_time: "2026-09-17T16:27:09Z"`. Every reporting-phase verificator passed with 0 errors
(`verify_task_file`, `verify_task_dependencies`, `verify_suggestions`, `verify_task_metrics`,
`verify_task_results`, `verify_task_folder`, `verify_logs`, the `v3-recipe` answer asset
verificator, `verify_machines_destroyed`); remaining warnings are all pre-existing and individually
documented in earlier Step History entries. `logs/sessions/capture_report.json` exists (0 matched
transcripts — expected, no local Claude Code/Codex session on this machine matched this task's
worktree `cwd`). The coordinator should now proceed directly to Phase 7 (push branch, create PR),
Phase 8 (`verify_task_complete` after merge), and Phase 9 (overview sync on `main`) — no further
step-executor work is needed for this task.
