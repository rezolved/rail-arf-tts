---
spec_version: "3"
task_id: "t0018_zero_shot_cloning_calibration"
step_number: 9
step_name: "implementation"
status: "completed"
started_at: "2026-09-17T17:27:59Z"
completed_at: "2026-09-17T19:25:27Z"
---
## Summary

Executed `plan/plan.md` Steps 1-18 (reference audio, install/smoke-gate, the 6 cloning variants plus
2 paired baselines, scoring, the audible-speech gate, metrics/tables/charts, the comparison audio
set, the listening guide, the answer asset, and DVC push) via a dedicated `/implementation`
subagent. Chatterbox completed both reference conditions at 100% success; CosyVoice2 completed
`ref_single` at 100% success but `ref_concat` hard-failed (system-level 30 s reference-audio limit,
exceeded by 0.57 s); F5-TTS and the `kokoro_v3_bundle` re-synthesis both hung indefinitely on this
VM session and are documented as failures, not silently substituted. Total GPU spend for the whole
task (setup-machines + this step, shared billing anchor) reached **~$56.15 of the $70 hard cap**,
leaving ~$13.85 for the teardown step.

## Actions Taken

1. Read `arf/skills/implementation/SKILL.md`, `plan/plan.md` (all 11 sections), and `checkpoint.md`;
   ran `prestep` with a 300 s heartbeat cadence and a 4 h expected duration given the multi-hour
   GPU-bound scope.
2. Spawned a dedicated `/implementation` subagent with the full worktree path, branch, budget ledger
   (elapsed spend at dispatch time, hourly VM rate, hard cap, and a priority order for
   scope-trimming: build references -> install/smoke-gate all 3 systems -> `ref_single` for all
   systems + both baselines -> `ref_concat` only if budget allows), and all carry-forward facts from
   `checkpoint.md` (venv paths, `HF_HOME`, watchdog state, F5-TTS torch pin, F5-TTS auto-crop
   behavior, metrics-routing rules).
3. When the coordinator flagged a stale-heartbeat (`ST-E007`) false alarm mid-run (real work was
   progressing; the step-executor's own turn had ended without a live heartbeat-refresh mechanism),
   corrected it by launching a self-terminating background loop
   (`heartbeat write t0018_zero_shot_cloning_calibration 9 step-executor/implementation` every ~250
   s, exiting automatically once step 9 leaves `in_progress`) so the heartbeat could not go stale
   again for the remainder of the wait, and confirmed the fix directly against `step_tracker.json`'s
   `last_heartbeat_at`.
4. After the nested subagent returned, verified its work independently rather than trusting its
   self-report: re-ran the answer-asset verificator (`PASSED`, 0 errors/warnings), checked
   `results/metrics.json` programmatically for registered-metric-keys-only compliance (12/12
   variants clean), counted `results/per_clip_metrics.json` rows (980, matching 5 non-null
   system/condition/prompt-set combinations x ~196), confirmed all 4 required PNG charts exist and
   are non-zero-byte, ran `dvc status` on the three tracked audio dirs
   (`Data and pipelines are up to date`), ran the 5 task-specific pytest tests (all pass),
   `ruff check`/`ruff format` (clean), and `mypy -p tasks.t0018_zero_shot_cloning_calibration.code`
   (clean — noting the repo-wide `exclude = ["tasks/.*/code/", ...]` mypy config, pre-existing on
   `main` and unrelated to this task, means task code is not actually strict-type-checked by this
   invocation; not a defect introduced here).
5. Read both intervention files (`f5_tts_smoke_gate_failed.md`,
   `kokoro_v3_bundle_not_remeasured.md`) in full and the `ref_single`-is-a-concatenation deviation
   documented in `code/build_references.py`'s module docstring and `data/references/manifest.json`;
   confirmed each is a genuine, well-evidenced technical finding (not a silent substitution or
   budget-driven cut) and that `results/tables.json`'s `notes` array and the answer asset's
   short/full answers carry the same caveats forward.
6. Ran `flowmark --inplace --nobackup` on every `.md` file the subagent created (both intervention
   files, `results/listening_guide.md`, `results/smoke_gate_log.md`, and the answer asset's
   `short_answer.md`/`full_answer.md`); re-ran the answer verificator afterward to confirm it still
   passes.
7. Reconciled cumulative GPU cost directly from `results/cost_tracking.json` (shared billing anchor
   `2026-09-17T15:21:13Z` with setup-machines) against wall-clock time computed independently
   (`date -u`): final ledger entry $56.15 at 19:22:32Z is consistent with an independent
   recomputation at verification time (~19:23-19:25Z), confirming no further drift and that the task
   remains under the $70 hard cap with the shared billing anchor already accounting for
   setup-machines' own spend.

## Outputs

* `tasks/t0018_zero_shot_cloning_calibration/code/{paths,constants,adapters_zeroshot,build_references,run_eval_zeroshot,merge_and_score,run_gate_check,report_zeroshot,build_comparison_set,build_listening_guide,build_smoke_gate_log,build_final_reports,score_elevenlabs_baseline,run_kokoro_v3_local_ckpt,transcribe_references,track_cost,test_zeroshot}.py`
* `tasks/t0018_zero_shot_cloning_calibration/data/references/` (`ref_single.wav`, `ref_concat.wav`,
  `manifest.json`, half-A centroid)
* `tasks/t0018_zero_shot_cloning_calibration/results/{metrics,tables,per_clip_metrics, gate_failures,environment,cost_tracking,smoke_gate_results}.json`,
  `results/{smoke_gate_log,listening_guide}.md`, `results/images/*.png` (4 charts),
  `results/audio_samples/{harness,comparison_set,references}/` (DVC-tracked, pushed)
* `tasks/t0018_zero_shot_cloning_calibration/assets/answer/zero-shot-speaker-sim-ceiling/`
  (`details.json`, `short_answer.md`, `full_answer.md`) — verificator `PASSED`, 0 errors/warnings
* `tasks/t0018_zero_shot_cloning_calibration/intervention/{f5_tts_smoke_gate_failed, kokoro_v3_bundle_not_remeasured}.md`

## Issues

* **F5-TTS null for all variants (REQ-7)**: three independent smoke-gate attempts hung indefinitely
  inside model loading with zero progress signal, despite the model weights being confirmed present
  in the HF cache from an earlier successful bare-class-load test. Documented in
  `intervention/f5_tts_smoke_gate_failed.md`; the F5-TTS x `ref_concat` auto-crop pre-check the plan
  required could not run as a result, since the system never got past its `ref_single` smoke gate.
* **CosyVoice2 `ref_concat` null (REQ-6 rejection rule)**: 0/196 successful — a genuine system-level
  hard limit (CosyVoice2 rejects reference audio over 30 s; this task's `ref_concat` clip is 30.57
  s, 0.57 s over), not a corpus, harness, or budget problem. This is itself a reportable Key
  Question 4 finding.
* **`kokoro_v3_bundle` not re-measured (REQ-3 partial)**: two independent re-synthesis attempts
  (including one after copying the checkpoint to local disk to rule out network-filesystem slowness)
  hung with the same signature as the F5-TTS failure. Falls back to t0008's stored `speaker_sim`
  numbers with `ttfb_ms`/`rtf` omitted (Lesson 1), per `task_description.md`'s own documented
  fallback pattern. Documented in `intervention/kokoro_v3_bundle_not_remeasured.md`. Both hangs
  point at the same underlying VM-pool environment issue (large-file/HF-Hub loads on `LLM-T1-NC80`
  intermittently hanging with zero progress) and are flagged as a follow-up recommendation, not
  attributed to this task's own code.
* **`ref_single` deviation (REQ-2 partial)**: the real half-A David corpus has no single clip near
  10 s (longest clip in the full 1364-clip corpus is 1.67 s), so `ref_single` was built as a short
  concatenation (same method as `ref_concat`, ~10 s target) rather than a literal single utterance.
  Caught during Phase 1.5 preflight inspection, not discovered downstream; documented in
  `data/references/manifest.json`, the code docstring, `results/tables.json`, and the answer asset —
  never silently presented as a natural single-utterance condition.
* No budget overrun: final GPU spend ~$56.15 of the $70 hard cap (setup-machines + this step, shared
  billing anchor), leaving ~$13.85 headroom for the teardown step. The step-executor's own heartbeat
  went stale once mid-run (coordinator-flagged `ST-E007`, confirmed a false alarm — real work was
  progressing) because the step-executor ended a turn without an active refresh mechanism after
  spawning the async subagent; corrected with a self-terminating background heartbeat loop for the
  remainder of the wait, and this is being called out here rather than omitted so the pattern is
  visible for future step-executors on long GPU-bound steps.
