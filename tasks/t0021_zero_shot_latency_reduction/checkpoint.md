---
spec_version: "1"
task_id: "t0021_zero_shot_latency_reduction"
updated_at: "2026-09-18T10:57:18Z"
completed_steps: 5
next_step_number: 6
next_step_id: "research-code"
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

* * *

## Cross-Step Decisions

* * *

## Next Step Notes

Steps 4-5 established: (a) the additive latency-model schema `L_TTS = M*d_lm + M*d_fm + M*d_voc`
([Du2024]) with LM/decoder-stage acceleration prioritized over vocoder work, and (b) the closest
available per-stage TTFA breakdown (vllm-omni issue #6870: ~11% prefill / ~40% AR decode / ~36%
flow) plus a validated bf16-decoder lever (~40% throughput, H100-confirmed,
[Chatterbox-TTS-Server-GH]). Proceed to step 6 (`research-code`): review t0018's adapters, harness
wiring, reference clips, and prompt sets in `code/` and `data/references/` for reuse, per the step
description in `step_tracker.json`. Read `research/research_internet.md` first for the
acceleration-lever priority order and the caveat on the `[Lian2026]` citation correction.
