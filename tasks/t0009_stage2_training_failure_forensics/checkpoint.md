---
spec_version: "1"
task_id: "t0009_stage2_training_failure_forensics"
updated_at: "2026-09-14T17:15:00Z"
completed_steps: 14
next_step_number: 15
next_step_id: "reporting"
---
# Task Objective

Audit every Kokoro training run's logs, data, pipeline and checkpoints to explain why Stage 2
breaks, and ship checkpoint and log safeguards.

* * *

## Step History

### Step 1 — create-branch

Branch `task/t0009_stage2_training_failure_forensics` created. Initial folder structure initialized
in `tasks/t0009_stage2_training_failure_forensics/`. Step 1 is a mechanical setup step with no
research output.

### Step 2 — check-deps

Both dependencies verified as completed: `t0005_kokoro_v5_stage2_train` and
`t0006_kokoro_v5_stage2_subset`. Verification passed with zero errors and zero warnings. Output:
`logs/steps/002_check-deps/deps_report.json`.

### Step 4 — research-papers

Skipped: no papers in the corpus are directly relevant to forensic analysis of local Kokoro training
logs.

### Step 5 — research-internet

Skipped: all inputs (logs, checkpoints, code) are local; no external data or new documentation
needed.

### Step 10 — setup-machines

Skipped: analysis runs locally on stored logs and checkpoints; no GPU required.

### Step 11 — teardown

Skipped: no remote machines provisioned.

### Step 3 — init-folders

Mandatory task folder structure created by `init_task_folders`; 13 directories with `.gitkeep` files
plus `__init__.py` files for the code package. Aggregator cache written to
`tasks/t0009_stage2_training_failure_forensics/ctx/` (gitignored).

### Step 13 — compare-literature

Skipped: this is a forensic audit task; results are not comparable to published quantitative
baselines.

### Step 6 — research-code

Reviewed 7 completed tasks; cited 6. No registered libraries found. Key output:
`tasks/t0009_stage2_training_failure_forensics/research/research_code.md` documenting the 7-patch
stack in `train_second_patched.py`, the checkpoint loading silent-failure bug in t0005/t0006, the
top-2 val_loss pruning risk, and all config-code inconsistencies. Research summary also produced at
`tasks/t0009_stage2_training_failure_forensics/research/research_summary.md`.

### Step 7 — planning

Full forensic plan written at `tasks/t0009_stage2_training_failure_forensics/plan/plan.md` — 15
steps across 8 milestones covering all 8 REQ items. `verify_plan` passed with 0 errors. Budget
confirmed at ≤ \$25 (local/CPU + ≤ 30 min VM for log retrieval). The safeguard library base is
`train_second_patched.py` from t0005; health gate thresholds derived from research (Dur Loss step-1
< 2.0, `acoustic_norm` < 20, val spike ≤ 0.05, consecutive skips ≤ 50).

### Step 9 — creative-thinking

Six alternative failure hypotheses explored and stress-tested against the step 8 findings. Four new
safeguards identified (startup param-count assertion, per-loss gradient norm logging, warm-up epoch
calibration, audio quality pre-filter) — all are recommendations for the next training task. Key
caveat: the causal independence of joint\_epoch=3 from the checkpoint mismatch is weaker than stated
in full\_answer.md; the two factors were never crossed in a controlled experiment.

### Step 8 — implementation

All forensic analysis complete. Produced: log inventory (1 of 13 runs has a surviving log), confound
table, data audit (v5 val == val\_96, no overlap), pipeline diff (7 patches classified), checkpoint
audit. Library asset `t0009_training_safeguards` (4 modules: jsonl\_logger, health\_gates,
checkpoint\_manager, run\_config) and answer asset `t0009-stage2-forensics-answer` both produced and
verified. Root cause: DP checkpoint mismatch + joint\_epoch=3 too early.

### Step 12 — results

All results files written and verified. `results_summary.md`, `results_detailed.md` (spec\_version
"2"), `metrics.json` (`{}`), `costs.json` (`$0`), `remote_machines_used.json` (`[]`). Two new charts
generated: `results/images/log_availability.png` (log survival rate, 1/13) and
`results/images/confound_heatmap.png` (normalised hyperparameter heatmap). `verify_task_metrics`
PASS; `verify_task_results` PASS (1 expected warning TR-W013 for data-analysis task type). Both
assets (answer, library) present and confirmed. `## Task Requirement Coverage` section lists REQ-1
through REQ-8 with Done/Partial status. REQ-2 and REQ-4 are Partial due to 12/13 logs being
permanently deleted and audio DVC not pulled.

### Step 14 — suggestions

Five follow-up suggestions generated in `results/suggestions.json` and verified by
`verify_suggestions` (0 errors). Suggestions cover: controlled Stage 2 run with safeguards applied
(high priority), controlled joint_epoch ablation to test causal independence (high priority), audio
quality pre-filter for v5 train set (medium priority), v3 config recovery to resolve val_loss
comparability gap (medium priority), and safeguard library extension with startup param-count
assertion and per-loss gradient norm logging (medium priority). No duplicate suggestions or existing
tasks cover these objectives.

* * *

## Cross-Step Decisions

* All reusable code must be copied into `tasks/t0009_stage2_training_failure_forensics/code/`; no
  libraries exist to import.
* v3's `train_second_patch.diff` is the forensic baseline — use it as the starting point for the
  pipeline audit in planning.
* Root cause finding: DataParallel checkpoint mismatch (missing "module." prefix) + joint\_epoch=3
  too early; recommended fix: DP-aware loader + joint\_epoch≥6.
* Health gate thresholds from log analysis: Dur Loss step-1 < 2.0, acoustic\_norm < 20, val spike ≤
  0.05, consecutive skips ≤ 50.

* * *

## Next Step Notes

Step 14 (suggestions) complete. Proceed to step 15 (reporting): run all relevant verificators,
capture session transcripts, update task.json status to completed, write final step log, and commit.
The main deliverables (answer asset, library asset, results files, suggestions) are all verified.
REQ-2 and REQ-4 are Partial (documented in results_detailed.md). Report the $0 cost in costs.json.
