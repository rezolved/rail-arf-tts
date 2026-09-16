---
spec_version: "1"
task_id: "t0013_v10_synthesis_quality_forensics"
updated_at: "2026-09-16T13:58:05Z"
completed_steps: 15
next_step_number: null
next_step_id: null
---
# Task Objective

Determine whether kokoro-v10-best produces noise instead of speech due to a bug in an ad hoc
inference reproduction, or a real checkpoint defect, and act accordingly.

* * *

## Step History

### Step 1 — create-branch

Trimmed to stay within 10 KB limit.

### Step 2 — check-deps

Trimmed to stay within 10 KB limit.

### Step 4 — research-papers

Trimmed to stay within 10 KB limit.

### Step 5 — research-internet

Trimmed to stay within 10 KB limit.

### Step 8 — setup-machines

Trimmed to stay within 10 KB limit.

### Step 10 — teardown

Trimmed to stay within 10 KB limit.

### Step 13 — compare-literature

Trimmed to stay within 10 KB limit.

### Step 3 — init-folders

Trimmed to stay within 10 KB limit.

### Step 6 — research-code

Trimmed to stay within 10 KB limit.

### Step 7 — planning

Trimmed to stay within 10 KB limit. See `plan/plan.md`.

### Step 9 — implementation

Trimmed to stay within 10 KB limit. Verdict (real training defect, `ignore_modules` omits `decoder`)
preserved in full in Cross-Step Decisions below and in `results/v10_diagnosis.md`.

### Step 11 — creative-thinking

Trimmed to stay within 10 KB limit. See `results/creative_thinking.md` and Cross-Step Decisions
below.

### Step 12 — results

Wrote `results/results_summary.md` and `results/results_detailed.md` (`spec_version: "2"`),
synthesizing `v10_diagnosis.md`, `control_test.md`, and `creative_thinking.md`; both pass
`verify_task_results.py` and `verify_task_metrics.py` with 0 errors/0 warnings. Discovered
`results/costs.json` and `results/remote_machines_used.json` were never written during
implementation (despite the earlier caveat assuming they existed) and created them
(`{"total_cost_usd": 0, "breakdown": {}}` and `[]`). Tightened one paragraph in
`results/v10_diagnosis.md` per the prior Cross-Step Decision, citing the step-11 probe's
`clip_fraction=0.004` finding explicitly, without changing the verdict or Recommendation.
`## Examples` (mandatory for `code-reproduction` task types) has 12 instances, all copied verbatim
from existing result files.

### Step 14 — suggestions

Spawned a dedicated `/generate-suggestions` subagent, instructed to carry forward both the retrain
follow-up and the eval-harness pre-completion regression-check follow-up from
`results/v10_diagnosis.md`'s Recommendation section. It wrote `results/suggestions.json` with four
suggestions: S-0013-01 (retrain t0010/a corrected variant with `ignore_modules` fixed for the
decoder handoff, high priority), S-0013-02 (mandatory synthesis noise/clipping smoke gate via
`code/audio_quality_check.py` before any Stage 2 training task can claim `completed`, explicitly
framed as a significant process/framework gap, high priority), S-0013-03 (preflight
decoder-architecture consistency check, medium priority), and S-0013-04 (promote
`audio_quality_check.py` to a registered library, low priority). `verify_suggestions` passed with 0
errors/0 warnings. No caveats.

### Step 15 — reporting

Ran all applicable verificators (`verify_task_file`, `verify_task_dependencies`,
`verify_suggestions`, `verify_task_metrics`, `verify_task_results`, `verify_task_folder`,
`verify_logs`) with 0 errors, captured session transcripts via `capture_task_sessions` (0 matched),
and set `task.json` `status` to `"completed"` with `end_time`. Deleted the gitignored `ctx/`
aggregator cache directory that tripped `verify_task_folder`'s `FD-E016`. No caveats — task is
complete; remaining work is the coordinator's Phase 7-9 (PR/merge, `verify_task_complete`, overview
sync).

* * *

## Cross-Step Decisions

* Leading root-cause hypothesis for v10 noise (from `research-code`, step 6):
  `config_david_v10.yml`'s `decoder.type: hifigan` vs. Stage 1's ISTFTNet-shaped weights in
  `first_stage_v3.pth`, silently partially loaded by `train_second_v10.py:load_checkpoint()`'s
  zero-match-only failure guard. Later steps (planning, implementation) should test this first via a
  cheap checkpoint-tensor check before building the full inference harness.
* `plan/plan.md` (step 7) sequences work into milestones: Milestone A (venv/checkpoint pull setup),
  Milestone B (cheap checkpoint-tensor falsifier — run first, before any inference code), a hard
  control-validation gate, then Milestones C-E (instrumented StyleTTS2-native harness, control test,
  v10 diagnosis). Implementation must follow this order and must not skip Milestone B.
* **Root cause confirmed** (step 9): `kokoro-v10-best` never produced working audio, at any epoch —
  a real training defect from the start caused by `train_second_v10.py`'s `ignore_modules` list
  omitting `decoder`, silently leaving the HiFi-GAN vocoder near-randomly-initialized. Both
  `epoch_2nd_00016.pth` (primary) and `epoch_2nd_00014.pth` (backup) fail identically. Recommended
  action (per `results/v10_diagnosis.md`): fix `train_second_v10.py`'s `ignore_modules` and retrain,
  or discard `kokoro-v10-best`; do not promote it as a production replacement candidate. Downstream
  steps (`results`, `suggestions`, `reporting`) must reflect this verdict and must not repeat the
  earlier informal "sounds like noise" framing as if unresolved — it is now resolved with evidence.
* `results/suggestions.json` was deliberately **not** written during `implementation` (reserved for
  the orchestrator's `suggestions` step). The eval-harness regression-check follow-up
  (`code/audio_quality_check.py` as a required pre-completion gate for future training tasks) is
  documented in `results/v10_diagnosis.md` and must be carried into `results/suggestions.json` at
  step 14.
* **Verdict mechanism refined, not overturned, by step 11 (`creative-thinking`).** A new
  falsification probe (`code/random_decoder_probe.py`, `results/random_decoder_probe.json`,
  `results/creative_thinking.md`) loaded v10 primary's real checkpoint into every module except
  `decoder` (left at pure `build_model()` random init, verified untouched). Result:
  `clip_fraction=0.004`, `is_likely_noise=False` — **not** the 75-81% clipping signature of the real
  v10 checkpoints, and qualitatively close to the control instead. Two consequences for the
  `results` step: (1) this rules out `diffusion`/`predictor_encoder` as independently sufficient
  causes of the clipping (decoder's specific state is necessary for it), strengthening the existing
  root-cause attribution; (2) it shows the real v10 decoder's partial architecture-mismatched load
  is an actively **worse** initialization than plain random — not merely "near-randomly-initialized"
  as `results/v10_diagnosis.md` currently phrases it. The `results` step-executor should consider
  tightening that phrasing (and optionally citing the probe as corroborating evidence) when writing
  `results_summary.md`/`results_detailed.md`, but should NOT treat this as changing the verdict's
  bottom line (real training defect, not a reproduction bug) or the Recommendation section — both
  stand, and are further supported. Full detail and a methodology caveat (probe used a different
  3-clip reference selection than `v10_diagnosis.md`) are in `results/creative_thinking.md`.
* **`results/costs.json` and `results/remote_machines_used.json` were not actually written during
  `implementation`** (step 9's checkpoint entry implied all deliverables were complete alongside
  `results/metrics.json`, but only `metrics.json` existed on disk). The `results` step (step 12)
  created both as zero-cost/no-machines records (`{"total_cost_usd": 0, "breakdown": {}}` and `[]`).
  Future step-executors should verify a prior step's "all deliverables written" claim against the
  filesystem directly rather than trusting the checkpoint summary alone.

* * *

## Next Step Notes

Task complete. Step 15 (`reporting`) ran all applicable verificators (`verify_task_file`,
`verify_task_dependencies`, `verify_suggestions`, `verify_task_metrics`, `verify_task_results`,
`verify_task_folder`, `verify_logs`) — all PASSED, 0 errors. `verify_machines_destroyed` and asset
verificators were not applicable (no remote machine, `expected_assets: {}`). One fix applied: the
local aggregator cache `tasks/t0013_v10_synthesis_quality_forensics/ctx/` (gitignored, populated at
step 3) was deleted because `verify_task_folder` flags any unexpected root-level directory
(`FD-E016`) — it is local-only scratch data, never committed, so deletion has no effect on the
committed history. `capture_task_sessions` found 0 matching transcripts (331 candidate Claude Code
files scanned, none matched this task's worktree `cwd`) and wrote
`logs/sessions/capture_report.json` recording the scan; per SKILL.md this is acceptable when no
transcript matches. `task.json` `status` is now `"completed"` with
`end_time: "2026-09-16T13:58:05Z"`. Remaining work is the coordinator's: Phase 7 (push branch, open
PR, `verify_pr_premerge`, merge), Phase 8 (`verify_task_complete` from main), and Phase 9 (overview
sync) — none of which are step-executor scope.
