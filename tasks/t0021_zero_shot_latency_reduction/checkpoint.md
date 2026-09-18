---
spec_version: "1"
task_id: "t0021_zero_shot_latency_reduction"
updated_at: "2026-09-18T19:56:31Z"
completed_steps: 8
next_step_number: 9
next_step_id: "implementation"
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

* * *

## Next Step Notes

Step 9 (`implementation`) is `paused_waiting` (pause_count=3, resume_after `2026-09-18T21:00:00Z`,
`watchdog_active=true`, `current_owner=null`). This note supersedes the pause_count=2 note below,
which described a since-resolved local CPU job; keep the history below for context but trust this
paragraph first on resume.

**What happened between pause_count=2 and pause_count=3:** the local `merge_and_score.py` job (PID
591111) did finish and wrote `results/per_clip_metrics.json` at 19:22:10Z, but a post-hoc inspection
of that output (aggregate WER=0.9352, catastrophically high) found a real bug, not a scoring
artifact: `code/transcribe_references.py` used `WhisperModel("small.en", beam_size=5)` to build
`ref_single`'s `prompt_text`, which silently truncated the transcript to only the clip's first
sentence despite the returned segment's timestamps correctly spanning the full 15.08s clip. A
`prompt_text` describing 1/15th of the actual reference audio confused CosyVoice2's zero-shot
conditioning badly enough that every `ref_single` CosyVoice2 variant (`baseline_new_ref` included)
synthesized audio unrelated to the requested text (confirmed by direct listening: e.g. "checking for
the latest press release" produced "A wolf profferpate."). This is fully written up in
`intervention/cosyvoice2_ref_single_prompt_text_truncated.md`. Fix: switched
`WHISPER_MODEL_SIZE` to `"base.en"` with plain defaults (matches t0008's own setting); re-ran
`transcribe_references.py`; `data/references/manifest.json`'s `ref_single_transcript` now holds the
full multi-sentence text. `chatterbox`'s `ref_single` and cosyvoice2's `ref_concat` were NOT affected
(different transcript / never anomalous WER) and are not being re-run.

The VM (`LLM-T1-NC80`) had been correctly idle-stopped by the watchdog during the long CPU-only
`merge_and_score` wait (not an incident — see `code/constants.py`'s `VM_CONFIRMED_DOWNTIME_SECONDS`
Gap 2 comment) and was re-acquired at 19:28:26Z specifically to redo the 5 affected CosyVoice2
`ref_single` variants (`baseline_new_ref`, `ref_cache`, `fp16`, `load_jit`, `load_trt`) against the
corrected transcript. This step-executor turn directly confirmed via SSH: VM state Running, idle
watchdog armed and PID-confirmed (PID 5695), and a real tmux session `cosyvoice2_rerun` running
`run_cosyvoice2_sweep.sh` (a simple sequential-loop script at
`/mnt/cache/persist/t0021_zero_shot_latency_reduction/repo/run_cosyvoice2_sweep.sh`, logging to
`tasks/t0021_zero_shot_latency_reduction/logs/cosyvoice2_sweep.log` on the VM, echoing
`ALL_COSYVOICE2_VARIANTS_DONE` at the very end). As of this pause: variant 1/5 (`baseline_new_ref`)
finished (exit=0, 19:51:55Z, 196/196); its corrected
`results/per_clip_metrics_cosyvoice2_baseline_new_ref_ref_single.json` already resynced to this
local worktree (a live sync mechanism mirrors the VM's persist dir back here — no manual rsync
needed) and is committed. Variant 2/5 (`ref_cache`, PID 7756) is in progress; this step-executor
directly verified it is NOT stuck via two spaced checks (CPU time 00:02:14→00:04:10, GPU memory
3MiB→3889MiB, log advancing through model-load/onnxruntime-init) — it is genuinely progressing
through a slow model-load off the Azure Files persist mount, the same slow-I/O pattern noted in step
8. 3 variants remain after `ref_cache` (`fp16`, `load_jit`, `load_trt`), each expected ~11-13 minutes
based on variant 1's timing, so completion is expected roughly 20:40-20:50Z; `resume_after` is set to
21:00Z for buffer.

**Caveat carried forward and now resolved by construction:** the earlier concern (pause_count=2
note, below) about `merge_and_score.py` possibly having read a stale pre-16:47:42Z
`chatterbox_precision_bf16_or_fp16_ref_single` source file no longer needs separate handling —
`merge_and_score.py`'s `main()` globs and `read_text()`s every `per_clip_metrics_*.json` file fresh
at call time (confirmed by reading its source this turn), so the FULL re-run required below
(triggered by the cosyvoice2 fix) will naturally pick up both the already-corrected chatterbox file
and the freshly redone cosyvoice2 files in one pass. No separate re-run-just-one-variant step is
needed.

**On resume (after `ALL_COSYVOICE2_VARIANTS_DONE` appears in the VM's `cosyvoice2_sweep.log`, checked
via the recorded `liveness_probe`):**

1. Confirm all 5 `results/per_clip_metrics_cosyvoice2_<variant>_ref_single.json` files resynced
   locally with post-fix data (spot-check: `text` field's WER should no longer show the ~0.94
   gibberish signature once scored).
2. Re-run `uv run python -m tasks.t0021_zero_shot_latency_reduction.code.merge_and_score` in full
   (fast, CPU-only, no GPU needed — just re-reads existing JSON files and re-scores ~2156 clips).
3. Run `code/run_gate_check.py` (writes `hardened_gate_pass` per clip + `results/gate_failures.json`
   — remember the owner-correction caveat: PASS is necessary, not sufficient).
4. Run `code/build_final_reports.py` (writes `results/metrics.json`, `results/tables.json`,
   `results/latency_breakdown.json`, and the 3 required charts under `results/images/`).
5. Run `code/build_comparison_set.py` then `code/build_listening_guide.py` (the 3-way audio
   comparison set + `results/listening_guide.md`; `build_comparison_set.py` already documents a
   real, already-resolved deviation — t0018's old-ref audio is unreachable via `dvc pull` due to an
   Azure credential-chain mismatch specific to `dvc`'s constrained credential chain, so the
   `t0018_old_ref` column is intentionally `-`; this is expected, not a new bug to chase).
6. Write the answer asset: `assets/answer/zero-shot-ttfb-floor/{details.json,short_answer.md,
   full_answer.md}` per `meta/asset_types/answer/specification.md` v2 (already read in full this
   turn — v2 requires `short_answer_path`/`full_answer_path` in `details.json`; `meta/categories/` is
   currently empty, so an empty `categories` list is expected, matching the `RP-W003`-style warning
   precedent from step 4).
7. Run the implementation step's closing verificators per `plan/plan.md`'s Verification Criteria
   section: `verify_task_metrics`, `aggregate_metrics --format ids` (expect exactly `rtf`,
   `speaker_sim`, `ttfb_ms`), the answer-asset verificator, and the file-existence/REQ-coverage
   checks.
8. Do NOT kill/restart PID 7756 or the `cosyvoice2_rerun` tmux session on resume unless a fresh check
   shows it is actually stuck (no CPU-time/GPU-memory/log movement across two checks spaced >=60s
   apart) — it was healthy as of this pause.

Caveat (re-affirmed, this is now the second time this exact lesson mattered on this task): a bare
background shell command (local `run_in_background` Bash, or a remote `nohup`/tmux job) does not by
itself notify the coordinator — only `heartbeat.pause_step` with a concrete `resume_after` makes the
wait visible and safe. Ending a turn on an unregistered background poll, even a well-intentioned
bounded one, leaves the tracker in a half-consistent state (`paused_waiting` status with a stale
`current_owner` still set) exactly as the coordinator caught this turn. Every pause from here on must
go through `heartbeat.pause_step` before the turn ends, with `current_owner` verified `null`
afterward.

* * *

### Superseded note (pause_count=2, kept for history only — see paragraph above for current state)

Step 9 (`implementation`) is `paused_waiting` (pause_count=2, resume_after `2026-09-18T19:45:00Z`).
GPU work is 100% done and the VM is torn down (`cost_tracking.json` final entry $68.77/4.93h at
16:51Z); all remaining work is CPU-only on this local machine. The 11-variant sweep produced all
`per_clip_metrics_<variant>.json` and `latency_breakdown_<variant>.json` files in `results/`. A
local scoring job (`uv run -m tasks.t0021_zero_shot_latency_reduction.code.merge_and_score`, PID
591111, started 2026-09-18T16:23:11Z) is merging all 2156 per-clip rows and computing dual-centroid
speaker_sim + duration_ratio + WER (faster-whisper `base.en` int8 on CPU). As of this pause (18:44Z,
elapsed 2:21) it is still alive and healthy (182% CPU across 2 hot compute threads, no crash) but
`results/per_clip_metrics.json` has not appeared yet. This is the THIRD executor turn to encounter
this same job still running — see the resume_sentinel recorded on the step tracker
(`tasks/t0021_zero_shot_latency_reduction/step_tracker.json`, step 9) for the full check procedure.

On resume: `ps -p 591111` and check `results/per_clip_metrics.json` is non-empty.

* If present: merge_and_score succeeded. **Before trusting it**, verify the
  `chatterbox_precision_bf16_or_fp16_ref_single` rows are not stale — that source file
  (`results/per_clip_metrics_chatterbox_precision_bf16_or_fp16_ref_single.json`) was rewritten at
  `16:47:42Z` (a bug-fix re-run; T3Cond dtype mismatch, see `results/cost_tracking.json` 16:32/16:51
  entries) which is 24 minutes AFTER merge_and_score's glob+read at process start (16:23:11Z). If
  the merged output shows null/suspicious `speaker_sim`/`wer` for that variant's 196 rows, re-run
  `merge_and_score` (fast — no GPU, just re-reads the now-final JSON files and re-scores ~2156
  clips) before proceeding. Once confirmed good, proceed to `plan/plan.md` Milestone 5:
  `run_gate_check`, `report_zeroshot` charts, `build_comparison_set`, `build_listening_guide`, the
  `answer` asset, and the implementation step's closing verificators.
* If PID 591111 is gone and `per_clip_metrics.json` is still absent/empty: the job crashed. Read
  `/tmp/claude-1000/-home-azureuser-rail-metarepo-real-repos-rail-arf-tts/632781c0-3a57-4e0f-825f-dc71a72b011c/tasks/bbnr9u1zw.output`
  for stderr — note this file legitimately reads EMPTY even for a healthy live job because it is fed
  via a plain `tail -40` (no `-f`) that buffers everything until the upstream process hits EOF;
  emptiness alone is not evidence of a hang. Diagnose and either fix-and-rerun or write an
  `intervention/` file.
* If PID 591111 is still alive and healthy: do NOT kill/restart it. Poll with real bounded sleeps
  for up to ~15-20 minutes; if it finishes, proceed as above. If not, `pause_step` again with a new
  concrete `resume_after` (this is pause_count=2 already — a third pause without a materially better
  ETA basis would trip `ST-E010`; use elapsed time and thread CPU activity, not a guess).

Caveat carried from step 8 (now doubly confirmed): never end a turn claiming "the background job
will notify me" — there is no such mechanism for a bare shell PID. Either drive it to completion
with real synchronous polling (bounded, e.g. via repeated `sleep`-loop Bash calls) or transition to
`paused_waiting` via `heartbeat.pause_step` with a concrete `resume_after` and commit. Per the owner
correction above (still binding), the `intervention/` file for the wrong-ElevenLabs-voice correction
and the new `ref_single`/`ref_concat` references + corrected centroid must already exist from
implementation steps 1-2 — confirm this on resume if starting fresh context, but do not redo it if
already present.
