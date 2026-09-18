---
spec_version: "1"
task_id: "t0021_zero_shot_latency_reduction"
updated_at: "2026-09-18T23:20:00Z"
completed_steps: 14
next_step_number: 15
next_step_id: "reporting"
---
# Task Objective

Profile where CosyVoice2 and Chatterbox spend their 1.3-2.9 s TTFB and test streaming, chunking,
vLLM/TensorRT backends and precision to find the best reachable TTFB at unchanged speaker_sim.

* * *

## Step History

### Step 1 — create-branch

Branch `task/t0021_zero_shot_latency_reduction` created. Initial folder structure initialized in
`tasks/t0021_zero_shot_latency_reduction/`. Step 1 is a mechanical setup step with no research
output.

### Step 2 — check-deps

`verify_task_dependencies.py` passed with no errors or warnings. The sole dependency,
`t0018_zero_shot_cloning_calibration`, is `completed` and provides the F5-TTS/CosyVoice2/Chatterbox
benchmark, adapters, reference clips, and prompt sets this task will reuse. Result recorded in
`logs/steps/002_check-deps/deps_report.json`.

### Step 3 — init-folders

Ran `init_task_folders` to create the mandatory task folder structure (`plan/`, `research/`,
`results/`, `results/images/`, `corrections/`, `intervention/`, `code/`,
`logs/{commands,searches,sessions,steps}/`, `assets/answer/`), recording
`logs/steps/003_init-folders/folders_created.txt`. Populated the local `ctx/` aggregator cache
(`task_types.json`, `costs.json`, `tasks.json`, `metrics.json`, `suggestions.json`) for downstream
subagents to reuse instead of re-running aggregators; `ctx/` is gitignored and not committed.

### Step 4 — research-papers

Reviewed the project's 11-paper corpus (no `meta/categories/` entries exist yet, so triage was
manual by topic) and wrote `research/research_papers.md` (10 papers cited). Key takeaway for
downstream steps: [Du2024] (CosyVoice2) gives an additive latency model
`L_TTS = M*d_lm + M*d_fm + M*d_voc` to use as the `results/latency_breakdown.json` schema; published
vocoder speeds (150x-3700x real time, [Kong2020, Kaneko2022]) mean the LM/decoder stage is the most
likely dominant TTFB term, so LM-serving acceleration (vLLM, TensorRT-LLM, fp16/bf16, torch.compile)
should be prioritized over vocoder-stage work. [Seo2026] (Chatterbox-Flash) is the only paper with
paired TTFP/RTF for a streaming zero-shot system (103-118 ms TTFP on H100) but required fine-tuning
a new decoding objective — evidence the 300 ms gap is not purely architectural but may not be fully
closeable by engineering-only (no-fine-tuning) levers. Caveat: no paper benchmarks vLLM/TensorRT-LLM
for CosyVoice2's backbone or reports a measured per-stage ms breakdown on H100 — this task must
establish those numbers empirically. Verificator passed (0 errors, 1 expected `RP-W003` warning from
the empty categories registry).

### Step 5 — research-internet

Wrote `research/research_internet.md` (21 queries, 15 sources, 8 discovered papers) addressing all
six gaps from `research_papers.md`. Key new evidence: a `vllm-project/vllm-omni` RFC (issue #6870,
non-peer-reviewed, GPU unstated) gives the closest available per-stage TTFA decomposition (357 ms
median: 40.5 ms prefill, 142.3 ms AR decode, 129.0 ms first flow chunk) and shows the flow-matching
stage is architecturally batch-1 — Gap 2 narrowed, not closed. Gap 6 (does acceleration shift
`speaker_sim`?) remains fully unresolved; no source measures this for any zero-shot TTS system, so
t0021's own paired measurement will be novel. All 8 discovered papers were added to the corpus via
`/add-paper` subagents and pass their verificator. Caveat for downstream steps: one citation
(`[Lian2026]`, dots.tts) was corrected in this step after full-text review showed the paper does not
support the bf16-acoustic/quantized-LLM-trunk precision claim originally attributed to it — the
bf16-decoder recommendation now rests solely on `[Chatterbox-TTS-Server-GH]`, and int8-LM-trunk
evidence should be treated as weak/unverified (`[LlasaQuant2026]`'s full text could not be
retrieved).

### Step 6 — research-code

Wrote `research/research_code.md` (5 tasks cited, 2 libraries surveyed: `tts_eval_harness` from
t0008 relevant, `t0009_training_safeguards` not relevant) documenting what t0021 can reuse from
t0018/t0008/t0015/t0014. Key output for planning/implementation: import `tts_eval_harness` (t0008)
via library for `SynthResult`, scoring, and prompt loading; copy (do not import) t0018's
`adapters_zeroshot.py`, `run_eval_zeroshot.py`, `constants.py`, `paths.py`, `build_references.py`,
`run_gate_check.py`, `track_cost.py`, and `report_zeroshot.py` into t0021's own `code/`; also copy
t0015's `audio_quality_check.py` (233 lines) directly rather than repeating t0018's disallowed
cross-task import of it. Caveat: none of t0018's three adapters (F5-TTS, CosyVoice2, Chatterbox)
records intermediate per-stage timestamps, so the per-stage latency breakdown this task needs is
genuinely new instrumentation; CosyVoice2's adapter already exposes `load_jit`/`load_trt`/`fp16`
flags (all currently `False`) ready to flip per variant; isolated venvs already diverge on
torch/CUDA (Chatterbox `2.6.0+cu124` vs CosyVoice2 `2.3.1+cu121`), so any vLLM/TensorRT install must
respect per-venv isolation; set `REF_CONCAT_TARGET_DURATION_S = 29.5` (from t0018's `30.0`) to fix
the CosyVoice2 30 s hard-limit failure (S-0018-02). `verify_research_code` passed (0 errors, 0
warnings). Also ran `/research-summarize` afterward, compressing all three research files into
`research/research_summary.md` (119 lines, 8191 bytes).

### Step 7 — planning

Wrote `plan/plan.md` (all 11 mandatory sections plus dedicated `## Owner Correction` and
`## Rejection Criteria` sections); `verify_plan` passed with 0 errors/warnings. The plan defines a
6-variant cumulative-stack acceleration matrix per system, new `StageTiming` instrumentation, and a
17-item `REQ-*` checklist where `REQ-11`..`REQ-17` map every owner-correction sub-requirement to a
concrete Step by Step action. Caveat: the intervention file and the corrected references/centroid
are created in implementation steps 1-2 (CPU-only, before GPU provisioning) — not yet on disk.

### Step 8 — setup-machines

Provisioned `LLM-T1-NC80` (2x NVIDIA H100 NVL, CUDA 12.2) via `/setup-remote-machine`; armed and
confirmed the idle watchdog (PID 6807) before any build started, verified `/mnt/cache/persist`
resolves to the real Azure Files mount (not ephemeral `/mnt`), and confirmed both t0018's
`.venv-cosyvoice2` and `.venv-chatterbox` are intact and reusable. CosyVoice2's `load_jit`/
`load_trt` export succeeded (`flow.encoder.fp32.zip`/`flow.encoder.fp16.zip` under
`/mnt/cache/persist/t0021_zero_shot_latency_reduction/pretrained/cosyvoice2/`). The new
`.venv-cosyvoice2-vllm` install hit its pre-authorized 20-minute cutoff mid-unpack (torch installed,
vllm itself not reached) — documented in `intervention/cosyvoice2_vllm_install_timeout.md`, and the
`vllm_backend` CosyVoice2 acceleration variant is null for this task per the plan's own
pre-registered fallback; this does not affect the other 5 CosyVoice2 variants or any Chatterbox
variant. `machine_log.json` recorded in `logs/steps/008_setup-machines/` with all required fields
(watchdog_active, watchdog_pid, gpu_verified, cuda_version, smoke_test_output); `destroyed_at`
remains `null` — the VM stays up for `implementation`. Caveat for the implementation step: the
provisioning subagent left two long-running remote jobs (the TRT export and the vLLM install)
running without registering any way to be resumed after ending its own turn — this step-executor
caught and closed that gap directly via bounded synchronous SSH polling rather than leaving the step
unmonitored; the watchdog protected the VM throughout so no idle-billing risk materialized, but
future steps on this task should not assume a subagent's self-reported "I'll wait for the
notification" actually corresponds to a real wakeup mechanism for remote (non-harness-tracked) work.

### Step 9 — implementation

Ran the full 12-cell CosyVoice2/Chatterbox acceleration sweep plus both closures against the
corrected `data/v4/val/wavs` references; two real bugs surfaced and were fixed mid-sweep — a
`transcribe_references.py` Whisper truncation bug that fed CosyVoice2 a 1/15th-length `prompt_text`
(all `ref_single` variants redone after the fix) and a duplicate `merge_and_score` process from a
confused resume (killed before causing damage). All owner-correction requirements (dual-centroid
scoring with a `speaker_sim_radiohost_control` column, gate-necessary-not-sufficient caveat, 3-way
comparison set) and all REQ-1..REQ-17 items were independently re-verified against real verificator
runs (`verify_task_metrics`, `aggregate_metrics --format ids`, the answer verificator, ruff/mypy) by
this closeout turn, not just trusted from the prior report. Final confirmed cost: $98.25 of the $100
cap (a documented, self-corrected track_cost.py double-counting bug is explained in
`cost_tracking.json`'s own final entry).

### Step 10 — teardown

Confirmation-only: `LLM-T1-NC80` was already deallocated mid-implementation, so this step
re-confirmed via `az ml compute show` (state=Stopped) without any SSH or re-provisioning, then
closed out `logs/steps/008_setup-machines/machine_log.json`'s previously-null `destroyed_at` (
`2026-09-18T20:39:00.626630Z`), `total_duration_hours` (7.0378h), and `total_cost_usd` ($98.25)
using the 3 billing windows already itemized in `results/cost_tracking.json`. Wrote the
previously-missing `results/remote_machines_used.json` and `results/costs.json`.
`verify_machines_destroyed` passed (0 errors, 3 expected warnings: legacy `spec_version`, Azure API
check called without workspace/resource-group context so reports unreachable, and no
`checkpoint_path` on a non-training GPU job).

### Step 11 — creative-thinking

Wrote a four-part creative-thinking analysis inline in
`logs/steps/011_creative-thinking/step_log.md` (matching t0018's established convention of no
separate `research/creative_thinking.md`). Key findings for `results`/`suggestions`: (1) the
LM-decode "floor" may partly reflect a non-pipelined call shape — CosyVoice2 never got a
chunk-size-tuned streaming variant despite research recommending one; (2) batch-1
memory-bandwidth-bound decode explains why precision/JIT levers left LM decode flat-or-worse and
predicts `vllm_backend`/speculative decoding as the correct next levers; (3) stripping the LM stage,
CosyVoice2's non-LM stages alone already total ≈152.5 ms; (4) `ttfb_ms_p95` (e.g. Chatterbox
`ref_cache` val96 p95=4,003 ms) tells a different story than the p50 numbers used to pick "best"
variants, plus a product-architecture aside on caching the finite filler catalog. No code or
committed measurement was changed.

### Step 12 — results

Wrote `results/results_summary.md` and `results/results_detailed.md` (spec_version "2") against the
already-produced `metrics.json`/`tables.json`/`latency_breakdown.json` — no regeneration, every
quoted number cross-checked exactly against its JSON source. Folded in all four of step 11's
creative-thinking angles into `## Analysis`, embedded and captioned all 3 existing charts, and wrote
the mandatory 17-row `## Task Requirement Coverage` (REQ-1..REQ-17: 15 `Done`, 2 `Partial` — REQ-3
for the 2 pre-registered-but-never-run levers, REQ-16 for the listening guide's missing
`t0018_old_ref` column, both pre-existing documented gaps, not new). Caveat:
`verify_task_results.py` has an undocumented `TR-E020` rule requiring at least one fenced code block
in `## Examples` for experiment-type tasks (not stated in `task_results_specification.md`'s prose) —
fixed by adding raw per-clip JSON excerpts; future `results` steps for experiment-type tasks should
include fenced JSON/ code blocks in `## Examples` from the start, not just bulleted prose.

### Step 13 — compare-literature

A subagent wrote `results/compare_literature.md` (8 comparison-table rows plus a Prior Task
Comparison subsection) comparing this task's own measured latency numbers against [Du2024]'s
additive latency model (structurally confirmed: LM stage dominates at 814-1,004 ms vs. 151-785 ms
combined flow-matching+vocoder), [Seo2026]'s 103-118 ms Chatterbox-Flash TTFP (framed as a
different, disallowed fine-tuning lever, not a shortfall), [Kong2020]/[Kaneko2022]'s vocoder-only
benchmarks (framed as a scope mismatch, not underperformance), and the `vllm-omni-6870` RFC's 182.8
ms LM-stage figure (an unreconciled ~4.5-5x gap versus this task's 814-904 ms measurement, addressed
with three explicit, unconfirmed hypotheses rather than silently dropped).
`verify_compare_literature` passed (0 errors, 0 warnings) both as reported by the subagent and on
this step-executor's own independent re-run. Caveat: one cosmetic typo (a stray space in an
`intervention/` file path reference) was found and fixed during independent review; no other issues.

### Step 14 — suggestions

A subagent executed `/generate-suggestions`, reading `task_description.md`'s Expected Outputs
together with `results/results_summary.md`, `results/results_detailed.md`,
`results/compare_literature.md`, and the `cosyvoice2_vllm_install_timeout.md` intervention, then
wrote `results/suggestions.json` (7 suggestions, `S-0021-01`..`S-0021-07`) after deduplicating
against 41 existing project suggestions and 21 project tasks via the aggregators (no overlaps).
Since neither system reached the 300 ms TTFB target, the suggestions lead with the
distillation/smaller-backbone lever (`S-0021-05`) rather than a production-integration task, per
`task_description.md`'s Expected Outputs, plus 6 other untested engineering-lever candidates this
task's own analysis surfaced. `verify_suggestions` passed 0 errors/0 warnings on both the subagent's
run and this step-executor's independent re-run.

* * *

## Cross-Step Decisions

### Owner correction (2026-09-18, injected by coordinator before step 7)

The project owner found on 2026-09-18 that the ElevenLabs account has two voices named "David".
Production (`brainpowa-voice-gateway` FluxCD) and the Kokoro training corpus `data/v4` use
`voice_id=rWV5HleMkWb5oluMwkA7` ("David - narrator and newsreader", `model_id=eleven_flash_v2_5`,
`output_format=pcm_24000`). t0008's harness resolved the voice by name and picked
`5gLuKtB16QIQv1vuSas1` ("David - British Radio Host") instead, so `data/11labs_david` is the WRONG
David, and t0018's `ref_single`/`ref_concat` clips (built from `data/11labs_david` half-A) cloned
the wrong voice. t0018 is completed and immutable — this is not corrected retroactively there; it
applies to t0021 from step 7 onward only.

Binding requirements for every remaining t0021 step (planning, implementation, results, reporting):

1. Build `ref_single` (~10 s) and `ref_concat` (<= 29.5 s) for CosyVoice2/Chatterbox from
   `data/v4/val/wavs` (val_96, held-out newsreader voice) — NOT from `data/11labs_david`. Record
   exact source filenames used.
2. Build the speaker-similarity centroid from `data/v4/val/wavs` (the half not used for references),
   not from `data/11labs_david`. Keep a second scoring column against the OLD `data/11labs_david`
   centroid, labelled `"radiohost (wrong voice) control"`, so numbers stay comparable with t0018.
3. Any ElevenLabs API call must pin `voice_id=rWV5HleMkWb5oluMwkA7`, `model_id=eleven_flash_v2_5`,
   `output_format=pcm_24000`, `stability=0.5`, `similarity_boost=0.75`. Never resolve a voice by
   name.
4. Latency work (per-stage TTFB breakdown, acceleration variants) is unaffected in scope, but the
   t0018 baseline setting must be RE-RUN with the NEW references in the same session as every
   variant, so speed and similarity stay paired (Lesson 1).
5. Write an `intervention/` file documenting this as an owner correction; cite it in `plan.md` and
   in `results/`; explicitly note that t0018's `speaker_sim` values were measured against the wrong
   voice.
6. `results/audio_samples/` must place, per comparison text: the new-reference output, the old t0018
   (wrong-voice-reference) output, and the `val_96` original of the production voice, side by side,
   indexed in `results/listening_guide.md`.
7. Gate caveat: t0015's `audio_quality_check.py` passes clips a human hears as "voice plus strong
   noise" — treat its PASS as necessary, not sufficient. The owner will listen; do not report a
   variant as clean on the automated gate alone.

### `dvc push` still pending for this task's audio (2026-09-18, noted at implementation closeout)

This task's 5 `.dvc` pointer files (`data/references/{ref_single,ref_concat}.wav.dvc`,
`results/audio_samples/{comparison_set,harness,references}.dvc`) are committed to git, but the
actual audio bytes have NOT been pushed to `azure://ml-dvc-datasets/datasets/rail-arf-tts` —
`dvc push` hangs in this environment (Azure credential-chain issue, not a data/methodology problem;
full detail in `intervention/dvc_push_pull_credential_failure.md`). Whichever later step handles the
task PR/merge (per `CLAUDE.md`'s "Run `dvc push` before merging the task PR" rule) must either retry
`dvc push` from a session with working Azure Blob Storage credentials, or explicitly flag this as an
open item in the PR description — do not silently merge with the data undurable.

* * *

## Next Step Notes

Step 15 (`reporting`, mandatory, final task step) is next, per `step_tracker.json`. Run
`uv run python -m arf.scripts.utils.prestep t0021_zero_shot_latency_reduction reporting`, then run
ALL relevant verificators with `run_with_logs.py` per `execute-task/SKILL.md` Phase 6's `reporting`
section: `verify_task_file.py`, `verify_task_dependencies.py`, `verify_suggestions.py`,
`verify_task_metrics.py`, `verify_task_results.py`, `verify_task_folder.py`, `verify_logs.py`,
`verify_compare_literature.py` (compare-literature ran in step 13), `verify_machines_destroyed.py`
(remote machines were used in steps 8-10) — no paper/predictions/model/library asset verificators
apply (this task produced no such assets) and `corrections/` is empty so `verify_corrections.py` is
not needed. Then run `capture_task_sessions` via `run_with_logs.py` to populate `logs/sessions/`.
Update `task.json`: set `status` to `"completed"` and set `end_time` (do not touch `start_time`).
This is the final checkpoint update — set `next_step_number` and `next_step_id` to `null` and
`completed_steps` to 15.

Two carried-forward items the reporting step (or the coordinator's Phase 7) must not silently drop:
(1) the `dvc push` gap documented in Cross-Step Decisions above — this task's 5 `.dvc` pointer files
are committed but the actual audio bytes have not been pushed to
`azure://ml-dvc-datasets/datasets/rail-arf-tts` (Azure credential-chain issue, see
`intervention/dvc_push_pull_credential_failure.md`) — either retry `dvc push` from a session with
working Azure Blob Storage credentials, or explicitly flag this as an open item in the PR
description; (2) `results/suggestions.json` now exists with 7 candidates (`S-0021-01`..`S-0021-07`,
verified 0 errors/0 warnings) — the strongest, per `task_description.md`'s Expected Outputs, is
`S-0021-05` (evaluate a distilled/smaller AR backbone), since neither CosyVoice2 nor Chatterbox
reached the 300 ms TTFB target in this task and the gap is framed as architectural
(LM-decode-bound); the other 6 (`vllm_backend` retry, chunk-size-tuned streaming, a
fine-tuning-permitted Chatterbox-Flash follow-up, flow-matching/vocoder stage-split instrumentation,
speculative decoding, and a Chatterbox `ref_cache` tail-latency repeat-run) are untested
engineering-lever candidates worth surfacing in the final report's forward-looking section.
