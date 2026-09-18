---
spec_version: "1"
task_id: "t0021_zero_shot_latency_reduction"
updated_at: "2026-09-18T23:05:00Z"
completed_steps: 10
next_step_number: 11
next_step_id: "creative-thinking"
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

Step 11 (`creative-thinking`, optional) is next, per `step_tracker.json` — an out-of-the-box
analysis step for alternative levers beyond the planned acceleration matrix (distillation, smaller
models, etc.) toward the 300 ms TTFB target, described in `step_tracker.json` step 11's
`description`. All GPU work is done and the machine is fully torn down (`machine_log.json` now has
`destroyed_at=2026-09-18T20:39:00.626630Z`, `total_duration_hours=7.0378`, `total_cost_usd=98.25`;
`results/remote_machines_used.json` and `results/costs.json` now exist and both verify clean). No
further remote-machine work is expected for the rest of this task (steps 11-15 are all
local/CPU-only: creative-thinking, results, compare-literature, suggestions, reporting) — do not
provision anything new.

Key inputs this step should build on: `results/latency_breakdown*.json` (12+ per-variant files) and
`results/metrics.json`/`results/tables.json` for the achieved TTFB/speaker_sim numbers per
acceleration variant across CosyVoice2 and Chatterbox; `research/research_summary.md` for the
architectural latency model (`L_TTS = M*d_lm + M*d_fm + M*d_voc`) and prior evidence (e.g.
Chatterbox-Flash's 103-118 ms TTFP required fine-tuning, not just engineering levers) that should
ground whatever alternative levers this step proposes.

Carried-forward, not this step's job: the `dvc push` gap documented in Cross-Step Decisions above
(this task's 5 `.dvc` pointer files are committed but the actual audio bytes have not been pushed to
`azure://ml-dvc-datasets/datasets/rail-arf-tts` — Azure credential-chain issue, see
`intervention/dvc_push_pull_credential_failure.md`) — that belongs to whichever step handles the
task PR/merge, not creative-thinking.
