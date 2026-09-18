---
spec_version: "1"
task_id: "t0021_zero_shot_latency_reduction"
updated_at: "2026-09-18T10:30:00Z"
completed_steps: 4
next_step_number: 5
next_step_id: "research-internet"
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

* * *

## Cross-Step Decisions

* * *

## Next Step Notes

Step 4 established the additive latency-model schema (`L_TTS = M*d_lm + M*d_fm + M*d_voc`, [Du2024])
and the prioritization order (LM/decoder-stage acceleration first, vocoder last) that should carry
into planning. Proceed to step 5 (`research-internet`): research CosyVoice2 and Chatterbox
acceleration paths — `load_jit`/`load_trt`, the vLLM LLM backend, `torch.compile`, streaming APIs,
and chunking strategies — per the step description in `step_tracker.json`. Read
`research/research_papers.md` first; it flags that no reviewed paper covers vLLM/TensorRT-LLM
benchmarks for CosyVoice2's Qwen2.5-0.5B backbone, so this is the gap research-internet must fill.
