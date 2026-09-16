---
spec_version: "1"
task_id: "t0013_v10_synthesis_quality_forensics"
updated_at: "2026-09-16T13:56:30Z"
completed_steps: 14
next_step_number: 15
next_step_id: "reporting"
---
# Task Objective

Determine whether kokoro-v10-best produces noise instead of speech due to a bug in an ad hoc
inference reproduction, or a real checkpoint defect, and act accordingly.

* * *

## Step History

### Step 1 — create-branch

Branch `task/t0013_v10_synthesis_quality_forensics` created. Initial folder structure initialized in
`tasks/t0013_v10_synthesis_quality_forensics/`. Step 1 is a mechanical setup step with no research
output.

### Step 2 — check-deps

Ran `verify_task_dependencies.py`, which passed with no errors or warnings; the aggregator confirms
`t0010_stage2_safeguarded_training` (the task that produced the kokoro-v10-best checkpoints under
investigation) has `status: "completed"`. Result recorded in
`logs/steps/002_check-deps/deps_report.json`. No caveats — the checkpoint under investigation is
confirmed available for step 6 (`research-code`) and step 9 (`implementation`).

### Step 4 — research-papers

Skipped: this is an empirical debugging/forensics task (checkpoint loading, weight inspection, audio
inference), not a literature question. No published-paper evidence bears on the root cause.

### Step 5 — research-internet

Skipped: `task_description.md` already specifies the exact reproduction recipe (torch pin, StyleTTS2
demo notebook, dependency list) discovered in the prior ad hoc session, so no new external research
is needed to execute it.

### Step 8 — setup-machines

Skipped: `task_description.md` specifies a CPU-only venv reproduction (torch==2.5.1 CPU, espeak-ng,
local StyleTTS2 inference) with no GPU training or large-scale inference involved.

### Step 10 — teardown

Skipped: no remote machine was provisioned (`setup-machines` not included), so there is nothing to
tear down.

### Step 13 — compare-literature

Skipped: this task produces internal forensic evidence (key-load diagnostics, weight-norm
comparisons, audio checks) about one project's own checkpoints, not quantitative results comparable
to a published baseline.

### Step 3 — init-folders

Created the mandatory task folder structure (`plan/`, `research/`, `results/`, `results/images/`,
`corrections/`, `intervention/`, `code/`, `logs/commands/`, `logs/searches/`, `logs/sessions/`,
`logs/steps/`, `assets/`) via `init_task_folders`, logged to
`logs/steps/003_init-folders/folders_created.txt`. Populated the local aggregator cache under
`tasks/t0013_v10_synthesis_quality_forensics/ctx/` (task_types, costs, tasks, metrics, suggestions)
for reuse by downstream subagents; `ctx/` is gitignored and not committed. No caveats.

### Step 6 — research-code

Reviewed 12 completed tasks (deep-diving into t0001, t0002, t0005, t0006, t0008, t0009, t0010) and 2
registered libraries, wrote `research/research_code.md` (verificator: PASSED, no errors/warnings),
then spawned `/research-summarize` to produce `research/research_summary.md`. Central new finding:
`config_david_v10.yml` is the project's only Stage 2 config with
`model_params.decoder.type: hifigan` (all others use `istftnet`), and `train_second_v10.py`'s
checkpoint loader does not exclude `decoder` when loading the ISTFTNet-shaped `first_stage_v3.pth`,
making a silent architecture mismatch the leading noise-output hypothesis.

### Step 7 — planning

Spawned a dedicated subagent to execute `/planning`, which wrote `plan/plan.md` (verificator:
PASSED, no errors/warnings) sequencing a cheap, venv-free checkpoint-tensor falsifier (direct
`net["decoder"]` key-name / NaN-Inf / weight-norm inspection against a known-good `istftnet`
control) strictly before the full instrumented StyleTTS2-native inference harness build. Traced
`first_stage_v3.pth` to a concrete DVC-tracked path
(`tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/stage1/first_stage.pth`) and located a
reusable speaker-sim scoring pattern
(`tasks/t0008_tts_eval_harness_baselines/code/score_speaker_sim.py`) and the 11labs David reference
corpus. Caveat: `ctx/task_types.json` shows `has_external_costs: true` for this task's types
(`tts-benchmark-run`, `code-reproduction`), not `false` — did not block planning since the budget
gate only fires at `create-branch`, but implementation should be aware the type declares external
costs even though this task's actual compute is CPU-only local work with $0 real cost.

### Step 9 — implementation

Spawned a dedicated subagent to execute `/implementation`, which followed `plan/plan.md`'s
milestones in order: `dvc pull` + `kikiri-tts` clone (Milestone A), the cheap tensor-level falsifier
`code/inspect_checkpoint.py` run **before** any inference code (Milestone B) — confirmed the v10
`decoder` module is HiFi-GAN-shaped while `first_stage_v3.pth` is ISTFTNet-shaped, all tensors
finite; the isolated CPU `torch==2.5.1` venv and instrumented harness `code/infer_styletts2.py`
(Milestone C); the control-validation gate against the base pretrained StyleTTS2 LibriTTS checkpoint
(Milestone D), which caught and fixed a real harness bug (`weight_norm`/`spectral_norm`
parametrization key-naming drift) before trusting v10 output, then passed; and the v10
primary/backup diagnosis (Milestone E). **Verdict** (`results/v10_diagnosis.md`): **real defect from
the start** — `train_second_v10.py`'s `ignore_modules` list omits `decoder`, so the HiFi-GAN vocoder
was left effectively randomly initialized entering training; both `epoch_2nd_00016.pth` and
`epoch_2nd_00014.pth` produce clipped/saturated garbage (75-81% samples pinned at +-1.0), not a
reproduction bug. `results/audio_samples/` DVC-tracked and pushed (7 files); `code/kikiri-tts/` and
`.venv-styletts2/` correctly gitignored, not committed; `results/metrics.json` has all three
variants' `rtf`/`speaker_sim`, `ttfb_ms` explicitly omitted with reasoning. Caveat: the plan's step
16 mentions naming the regression check as a follow-up in `results/suggestions.json`, but that file
is reserved for the orchestrator's `suggestions` step per the implementation skill's forbidden-file
list — the subagent documented the follow-up in `results/v10_diagnosis.md`'s Recommendation section
instead; step 14 (`suggestions`) must pick this up.

### Step 11 — creative-thinking

Ran a new falsification probe (`code/random_decoder_probe.py`) isolating `decoder` as the only
module differing from the real v10 primary checkpoint (everything else loaded normally); pure
random-init decoder gave `clip_fraction=0.004` (not clipped), unlike the real checkpoints'
0.750-0.807. This rules out `diffusion`/`predictor_encoder` as independent causes of the clipping
and refines the failure mechanism (worse than random, not merely undertrained). Full write-up:
`results/creative_thinking.md`; probe outputs: `results/random_decoder_probe.json`,
`results/audio_samples/probe_random_decoder.wav`. Caveat: probe used a different 3-clip reference
selection than `v10_diagnosis.md`, so its numbers are corroborating, not directly merged into that
table.

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

* * *

## Next Step Notes

Step 14 (`suggestions`) completed. `results/suggestions.json` contains 4 suggestions (S-0013-01
through S-0013-04) and passes `verify_suggestions` with 0 errors/0 warnings; the process-gap
follow-up (S-0013-02) is explicitly flagged as significant framework feedback, not a routine
suggestion. Step 15 (`reporting`) is the final step: run all relevant verificators (including
`verify_suggestions`, `verify_task_results`, `verify_task_metrics`, `verify_task_folder`,
`verify_logs`, `verify_task_file`, `verify_task_dependencies`), capture session transcripts via
`capture_task_sessions`, set `task.json` `status` to `"completed"` with `end_time`, do the final
`checkpoint.md` update (`next_step_number`/`next_step_id` to `null`), commit, and run poststep.
