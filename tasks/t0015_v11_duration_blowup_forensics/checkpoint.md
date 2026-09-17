---
spec_version: "1"
task_id: "t0015_v11_duration_blowup_forensics"
updated_at: "2026-09-17T10:52:00Z"
completed_steps: 14
next_step_number: 15
next_step_id: "reporting"
---
# Task Objective

Root-cause why kokoro-v11-best's synthesis runs 15-20x too long and sounds like a droning babble to
a human, despite passing the clipping/flatness noise gate, then close that gate's blind spot.

* * *

## Step History

### Step 1 — create-branch

Branch `task/t0015_v11_duration_blowup_forensics` created. Initial folder structure initialized in
`tasks/t0015_v11_duration_blowup_forensics/`. Step 1 is a mechanical setup step with no research
output.

### Step 2 — check-deps

Ran `verify_task_dependencies.py` (via prestep and again via `run_with_logs`) against both
dependencies: `t0013_v10_synthesis_quality_forensics` and `t0014_v11_decoder_fix_retrain`. Both have
`status: "completed"` in their `task.json`, so the check passed with 0 errors and 0 warnings. Result
recorded in `logs/steps/002_check-deps/deps_report.json`.

### Step 3 — init-folders

Ran `init_task_folders` to create the mandatory folder structure (`plan/`, `research/`, `results/`,
`results/images/`, `corrections/`, `intervention/`, `code/`,
`logs/commands|searches|sessions|steps/`, `assets/`), each with `.gitkeep`; `expected_assets` is
`{}` so no asset-type subdirectory was added. Populated the aggregator cache at
`tasks/t0015_v11_duration_blowup_forensics/ctx/` (task_types.json, costs.json, tasks.json,
metrics.json, suggestions.json) for downstream subagents to reuse; `ctx/` is gitignored and not
committed.

### Step 4 — research-papers

Skipped: this is an empirical debugging/forensics task about this project's own checkpoints and
code, and no published-paper evidence bears on the specific duration-blowup root cause. StyleTTS2
background is already present in the corpus from prior dependency tasks.

### Step 5 — research-internet

Skipped: `task_description.md` already specifies the exact reproduction recipe, the parameters to
sweep, and the ASR library (faster-whisper) already vendored in t0008's environment, so no new
external research is needed.

### Step 8 — setup-machines

Skipped: `task_description.md` states no GPU is required for the diagnostic, localization, and
inference-parameter-sweep work; GPU-based predictor retraining is explicitly scoped out as a
follow-up task.

### Step 10 — teardown

Skipped: no remote machine was provisioned (`setup-machines` was skipped), so there is nothing to
tear down.

### Step 13 — compare-literature

Skipped: this task produces internal forensic evidence about this project's own checkpoint and code,
not quantitative results comparable to a published baseline.

### Step 6 — research-code

Wrote `research/research_code.md` (verificator: 0 errors, 0 warnings) documenting the reusable
inference/diagnostic code and hard evidence on the predictor-gradient question. Key finding: t0014's
`train_second_v11.py` shows `predictor`/`predictor_encoder` DID receive unconditional
`optimizer.step()` calls for all 50 epochs (lines 733-734), refuting the "rode along frozen"
hypothesis — but `data/run_v11/metrics.jsonl` shows `dur_loss` plateaued at 0.53-0.62 the entire run
versus t0009's reference run converging to 0.034, pointing instead at a predictor-calibration
failure. Also identified the exact instrumentation point in `infer_styletts2.py`'s `synthesize()`
(t0014, lines 282-371) for logging `pred_dur`/`pred_aln_trg`, plus reusable diagnostic scripts
(`inspect_checkpoint.py`, `random_decoder_probe.py`, `build_reference_concat.py`) and varied-text
prompt sets from t0008, all labeled "copy into task" (no library registration). Since this was the
only research step executed, also ran `/research-summarize` to produce
`research/research_summary.md` for downstream subagents.

### Step 7 — planning

Wrote `plan/plan.md` (11 mandatory sections plus an added `## Rejection Criteria` section);
verificator passed with 0 errors and 0 warnings on the first attempt. The plan defines 14 numbered,
milestone-grouped steps (env setup, instrument `synthesize()` for `pred_dur`/frame-count logging, a
10-text varied characterization sweep, localization analysis, tensor-level forensics, an
alpha/beta/diffusion_steps/embedding_scale parameter sweep with pre-registered pass criteria, a
conditional cheap-fix-or-documented-negative-result branch, gate hardening with two new signals
(duration-sanity, silence-gap), an ASR round-trip evaluation via faster-whisper, a three-way
v10/v11/corrected regression check, `results/metrics.json` in explicit multi-variant format, and the
canonical `results/duration_blowup_diagnosis.md`), with 3 steps marked `[CRITICAL]`. Cost estimate
is $0.00 (CPU-only, no paid APIs, no GPU); estimated implementation time ~3.5-5.5 hours. The plan
corrects a `task_description.md` assumption during research: the ElevenLabs David reference corpus
actually lives at `tasks/t0008_tts_eval_harness_baselines/data/11labs_david`, not a top-level
`data/11labs_david`, and notes that DVC-tracked checkpoint/reference data has not yet been pulled in
this worktree — `dvc pull` is an explicit early step.

### Step 9 — implementation

Executed all 14 `plan/plan.md` Step-by-Step items via a dedicated `/implementation` subagent; none
were skipped or blocked, and no `intervention/` file was needed. Root-cause verdict
(`results/duration_blowup_diagnosis.md`): a predictor-pathway calibration failure — NOT
`duration_proj`'s own weights (only +0.7% weight-norm shift vs. the LibriTTS control,
`results/predictor_tensor_forensics.md`), NOT `pred_aln_trg`/decoder plumbing (the observed "exactly
2.00x" frame-to-output-duration ratio was confirmed, via source-reading `hifigan.py`/`istftnet.py`
and an empirical control-checkpoint validation run, to be the decoder's normal
architecture-intrinsic upsample, not a bug), and NOT `max_dur=50` ceiling saturation (only 0.44% of
tokens near ceiling). Most likely culprit: `predictor_encoder`'s -12.7% weight-norm shift from being
initialized as `copy.deepcopy(style_encoder)` rather than loaded from a duration-calibrated state.
All 10/10 characterization texts blew up 8.6x-17.9x with no text-length correlation (universal, not
text-dependent) — `results/duration_characterization.json`. The cheap inference-parameter fix does
NOT work: 0/13 pre-registered `alpha`/`beta`/`embedding_scale`/`diffusion_steps` combinations passed
(best `duration_ratio` 5.54x vs. the required ≤3.0x) — `results/param_sweep.json`; no
`v11_corrected.wav` was fabricated, and `results/gate_regression.json`'s `v11_corrected` fixture is
correctly recorded as `null`/skipped with an explanatory note. `code/audio_quality_check.py` was
hardened with `duration_sanity_pass` and `longest_nonsilent_run_s` signals and proven, via the
three-way regression, to flag `v11_best.wav` (`hardened_gate_pass=False`) while the original
`is_likely_noise` signal stays unchanged and v10 still fails (`results/gate_regression.json`,
matches the plan's own literal verification assertions exactly). `results/metrics.json` (explicit
variant format) records `rtf`/`speaker_sim` for `v11-as-shipped` and `v10-primary`;
`verify_task_metrics` passes. Ruff, mypy (task-code package form), and pytest
(`code/test_audio_quality_check.py`) all pass clean. A mid-task `dvc pull` auth failure
(`DefaultAzureCredential`'s `ManagedIdentityCredential` aborting the chain) was fixed with a local,
gitignored `dvc remote modify --local azureblob exclude_managed_identity_credential true` — not a
repo modification, no intervention file needed. 55 MB of new audio samples were `dvc add`ed and
`dvc push`ed (27 files, `logs/commands/031_...json`/`032_...json`, both exit 0). All 11 `REQ-*`
items are `done` except REQ-6, which is correctly `n/a` (the REQ-7 negative-result path was taken
instead, per the pre-registered Rejection Criteria — no partial fix was reported as a pass).

### Step 11 — creative-thinking

Wrote `logs/steps/011_creative-thinking/step_log.md` (verificator: 0 errors, 0 warnings; no results/
code/plan files touched, read-only analysis step). Five grounded findings for the `results`/
`suggestions` steps to draw on: (1) a concrete, cheap, forward-pass-only cross-checkpoint
module-swap ablation (v11's `predictor` + control's `predictor_encoder`) that would directly test,
not just infer, the `predictor_encoder` deepcopy-init hypothesis before any GPU retrain is funded;
(2) the `dur_loss` plateau ratio (0.53-0.62 / 0.034 ≈ 15.6x-18.2x) is suspiciously close to the
observed audio-blowup ratio (8.6x-17.9x) — worth a 10-minute check of David's forced-alignment/
duration-target frame-rate pipeline before assuming the fix is purely `predictor_encoder`
re-initialization; (3) cross-referencing the smoke test (`smoke_test.timing.json`, identical
inference params to `param_sweep.json`'s `baseline_default`) against the 10 characterization records
shows `duration_ratio` may track phoneme density (`tokens_per_word`) rather than word count — the
two lowest-density texts (smoke test 4.25, char idx7 6.9) gave the two lowest ratios (5.26, 8.61) —
a covariate the original word-count-bucket analysis in `results/duration_blowup_diagnosis.md`
Section 2 did not test, and directly relevant to voice-commerce filler text containing digits/SKUs/
brand names; (4) the hardened gate's `duration_sanity_pass` is upper-bound-only — an
under-synthesis/truncation regression (a different failure class than anything found in this task)
would currently pass both the old and hardened gate undetected; (5) ASR-round-trip and duration/
silence-gap checks catch disjoint failure classes (content-correctness vs. structural-plausibility),
so a future third gate layer should pair ASR-round-trip with a symmetric (both-sided) duration
check, not treat either as a superset of the other.

### Step 12 — results

Wrote `results/results_summary.md` and `results/results_detailed.md` (`spec_version: "2"`, 11
concrete `## Examples` instances drawn from `duration_characterization.json`/`param_sweep.json`, and
a final `## Task Requirement Coverage` section covering all 11 `REQ-*` items). Filled in the two
results files step 9 correctly left for this step: `results/costs.json` (`$0.00`, CPU-only) and
`results/remote_machines_used.json` (`[]`). Generated `results/images/duration_ratio_by_text.png`
and `results/images/param_sweep_duration_ratio.png` from already-produced JSON data (none existed
before this step). `## Analysis` documents the plan's contradicted assumption explicitly: the
inference-parameter sweep was framed as the cheapest fix to try, but 0/13 combinations came close to
passing. `verify_task_metrics` and `verify_task_results` both pass with 0 errors/0 warnings.

* * *

## Cross-Step Decisions

* The v11 duration blowup is NOT caused by frozen predictor/predictor_encoder modules during t0014's
  finetune — gradients were applied every epoch. The leading hypothesis going into planning is a
  duration-predictor calibration failure evidenced by the stalled `dur_loss` (0.53-0.62 vs. a 0.034
  reference), which should steer the planning step's localization approach toward `pred_dur`/frame
  ratio instrumentation rather than a plumbing/freeze-bug search.

* No GPU is used anywhere in the plan: all 14 implementation steps run CPU-only. Estimated cost is
  $0.00. Any escalation to targeted `predictor`/`predictor_encoder` fine-tuning is explicitly
  deferred to a follow-up task, per `task_description.md`'s scope and the `setup-machines` skip
  rationale from step 8.

* The ElevenLabs David reference corpus path used throughout downstream steps is
  `tasks/t0008_tts_eval_harness_baselines/data/11labs_david` (confirmed during planning research),
  not the top-level `data/11labs_david` implied by `CLAUDE.md`'s benchmark description —
  implementation must use the t0008 path and run `dvc pull` early since the checkpoint and reference
  audio are DVC-tracked and not yet present in this worktree.

* Root cause is confirmed as a predictor-pathway calibration failure (most likely
  `predictor_encoder`'s `copy.deepcopy(style_encoder)` init, not `duration_proj`'s own weights or a
  plumbing bug), the blowup is universal across all 10 characterization texts, and the cheap
  inference-parameter fix does not work (0/13 combinations passed). This means the task's
  recommendation (Step 9's evidence, folded into `results/duration_blowup_diagnosis.md`'s
  Recommendation section) is a follow-up task scoped to targeted `predictor`/`predictor_encoder`
  fine-tuning — downstream `results`/`suggestions` steps should reflect this recommendation and must
  not describe the gate-hardening work as also having "fixed" the underlying model.

* `code/audio_quality_check.py`'s hardened two-signal gate (`duration_sanity_pass`,
  `longest_nonsilent_run_s`) is proven via `results/gate_regression.json` to correctly flag
  `v11_best.wav` while leaving the original `is_likely_noise` signal and the v10 known-broken
  fixture's verdict unchanged — this is the blind-spot closure the task exists to prove, and
  downstream steps can cite `results/gate_regression.json` directly rather than re-deriving it.

### Step 14 — suggestions

Spawned a dedicated `/generate-suggestions` subagent (per Critical Rule 9), passing the task ID plus
the root-cause diagnosis and step 11's five findings as raw context, without restricting the skill's
own process (Critical Rule 10). The subagent independently reviewed
`results/duration_blowup_diagnosis.md`, `results/asr_roundtrip_evaluation.md`, and the
creative-thinking step log, then deduplicated against 23 existing open suggestions and 15 existing
tasks before writing `results/suggestions.json` (7 suggestions, `S-0015-01`-`S-0015-07`):
`S-0015-01`/`S-0015-02` (high) cover the targeted `predictor`/`predictor_encoder` Stage 2 fine-tune
and the cheap cross-checkpoint module-swap ablation that should precede it; `S-0015-03`/`S-0015-04`
(medium) cover the forced-alignment frame-rate audit and a phoneme-density characterization;
`S-0015-05`/`S-0015-07` (medium) cover the symmetric lower-bound duration check and wiring
ASR-round-trip as an optional third gate layer; `S-0015-06` (high) proposes adopting the hardened
gate as the project's mandatory pre-completion check, explicitly superseding S-0013-02. Re-ran
`verify_suggestions` independently via `run_with_logs`: PASSED, 0 errors, 0 warnings.

* * *

## Next Step Notes

Step 14 (`suggestions`) is complete. The next and final pending step is step 15 (`reporting`). It
should run all relevant verificators (`verify_task_file`, `verify_task_dependencies`,
`verify_suggestions`, `verify_task_metrics`, `verify_task_results`, `verify_task_folder`,
`verify_logs`; no asset-type or remote-machine verificators apply since `expected_assets` is `{}`
and no remote machine was used), capture session transcripts via `capture_task_sessions`, set
`task.json`'s `status` to `"completed"` and `end_time`, finalize `checkpoint.md` with
`next_step_number`/`next_step_id` set to `null`, write the step log, commit, and run poststep. After
that the coordinator handles Phase 7-9 (PR, merge, overview sync) inline.
