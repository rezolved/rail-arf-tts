# F5-TTS retry (S-0018-01): hung again, null verdict confirmed with a concrete stack trace

## What happened

Per `plan/plan.md` Step 12, this task retried F5-TTS's smoke gate against the NEW
`data/references/ref_single.wav` (built in Milestone 1 from `data/v4/val/wavs`, per the owner
correction), in a fresh shell session on the freshly-restarted `LLM-T1-NC80` VM (not reusing any
process/venv state from t0018's three prior hangs).

* First installed `py-spy` into `.venv-f5tts` (was missing) — succeeded, no issues.
* Launched
  `.venv-f5tts/bin/python -m tasks.t0021_zero_shot_latency_reduction.code.run_eval_zeroshot --system f5_tts --acceleration-variant baseline_new_ref --conditions ref_single --prompt-set fillers --n-warmup 0 --limit 1 --hf-cache-dir /mnt/cache/persist/t0018_zero_shot_cloning_calibration/hf-cache`
  under `timeout 2700` (45-minute hard cap per plan.md), on `CUDA_VISIBLE_DEVICES=1`, in a tmux
  session (`f5tts_retry`) at `2026-09-18T15:55:40Z`.
* The log printed exactly one line (`Prompts loaded: {'fillers': 100}`) plus a harmless
  `google.api_core` deprecation warning, then produced **zero further output** for 8+ minutes.
  `ps aux` showed the process in state `DLl` (uninterruptible sleep, i.e. blocked on I/O) the
  entire time, with almost no CPU time accumulated (0:29 over ~8 minutes wall-clock) — the same
  "true hang, not slow progress" signature t0018 documented three times in
  `tasks/t0018_zero_shot_cloning_calibration/intervention/f5_tts_smoke_gate_failed.md`.
* Per the plan's explicit protocol ("if it hangs again past 5 minutes with zero incremental log
  output, run `py-spy dump --pid <pid>`"), `py-spy dump` was run against the live process
  (PID 26378) at `2026-09-18T16:02Z` — **unlike t0018's attempt, this one worked** (t0018 could
  only get `Permission denied` on `/proc/<pid>/stack`/`/proc/<pid>/syscall`; `py-spy dump` used a
  different, working introspection path) and produced a full Python stack trace.

## Stack trace (the concrete new evidence t0018 could not obtain)

```text
Thread 26378 (idle): "MainThread"
    get_data (<frozen importlib._bootstrap_external>:1073)
    get_code (<frozen importlib._bootstrap_external>:975)
    exec_module (<frozen importlib._bootstrap_external>:879)
    ...
    <module> (datasets/packaged_modules/__init__.py:21)
    ...
    <module> (datasets/load.py:69)
    ...
    <module> (datasets/inspect.py:26)
    ...
    <module> (datasets/__init__.py:26)
    ...
    <module> (model/dataset.py:7)
    ...
    <module> (model/trainer.py:19)
    ...
    <module> (model/__init__.py:5)
    ...
    <module> (utils_infer.py:31)
    ...
    <module> (api.py:11)
    ...
    load_f5tts_model (code/adapters_zeroshot.py:88)
    main (code/run_eval_zeroshot.py:353)
    <module> (code/run_eval_zeroshot.py:528)
```

The hang is inside CPython's own module-import machinery (`importlib._bootstrap_external.get_data`,
i.e. reading a `.py` **source file's bytes off disk** to compile/exec it), triggered transitively by
`from f5_tts.api import F5TTS` (`adapters_zeroshot.py:88`, `load_f5tts_model()`) pulling in
HuggingFace's `datasets` package (`f5_tts/api.py` → `model/__init__.py` → `model/trainer.py` →
`model/dataset.py` → `datasets/__init__.py` → `datasets/inspect.py` → `datasets/load.py` →
`datasets/packaged_modules/__init__.py`). The process is not computing anything and not making a
network call — it is blocked reading a source file from the venv's `site-packages` directory, which
lives on the VM's Azure Files SMB mount (`/mnt/batch/tasks/shared/LS_root/mounts/clusters/...`, the
same mount t0018's `wchan` inspection (`wait_for_response`) already implicated, now pinned to the
exact import statement).

This rules out every hypothesis this task's plan considered as "the first thing to try":

* **Not** a stale lock file — nothing in this trace touches a lock, cache, or download path; it is a
  plain `importlib` source-file read.
* **Not** a network/HF-Hub stall — `HF_HUB_OFFLINE`/network round-trips play no role in loading a
  local `.py` file from the installed package.
* **Not** fixed by a fresh shell session or the VM restart — this run used a brand-new tmux session
  on a VM that had been fully stopped and re-provisioned since t0018's three hangs, and it hung in
  the identical way.

The proximate cause is therefore the Azure Files SMB mount itself intermittently stalling on reads of
this specific dependency chain's source files (a `datasets`-package import that neither CosyVoice2
nor Chatterbox's load paths happen to trigger) — an infrastructure-level flakiness in the VM's file
mount, not a bug in this task's own code, and not something a `.venv-f5tts` reinstall (t0018's other
candidate fix, not reattempted here given this concrete new evidence directly implicates the mount
rather than the venv's package state) would be expected to resolve.

## Resolution (REQ-6)

Per plan.md Step 12's explicit instruction ("do not retry a third time past the 45-minute cap") and
the Rejection Criteria section ("F5-TTS's third hang (if it recurs) is a null result for F5-TTS, not
a task failure"): F5-TTS is marked **null** for all conditions/prompt-sets in
`results/metrics.json`/`results/tables.json`, with this file and t0018's own
`f5_tts_smoke_gate_failed.md` as the combined evidence trail. The process (PID 26377/26378) was
killed (`kill -9`) at `2026-09-18T16:02Z`, well inside the 45-minute cap (actual attempt duration:
~7 minutes before the py-spy dump, consistent with t0018's own "hangs indefinitely with zero
progress" pattern rather than "slow but eventually succeeding").

F5-TTS is a cheap closure item (S-0018-01), not a critical-path deliverable of this task — CosyVoice2
and Chatterbox's per-stage latency breakdown and acceleration-variant sweep (this task's actual
subject) are unaffected.

## Cost impact

~7 minutes of GPU wall-clock on `LLM-T1-NC80` (~$1.63 at $13.96/hr), plus the `py-spy` install
(seconds). Well under the plan's 0.75 h / $10.47 budget line for this closure item.

## Recommendation for any future retry

Do not retry F5-TTS on this VM's current Azure Files mount without first isolating whether the mount
stall is reproducible outside this task's own code path (e.g. `time python3 -c "import datasets"` in
a bare venv, independent of F5-TTS). If it reproduces there too, this is squarely an infrastructure
ticket (VM image / mount configuration), not a task-level fix — consistent with `CLAUDE.md` Key Rule
0 (framework/infrastructure issues are not task work).
