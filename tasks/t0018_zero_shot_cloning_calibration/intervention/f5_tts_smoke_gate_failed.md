# F5-TTS smoke gate: failed (timeout / hang), null variant per REQ-7

## What happened

F5-TTS's `ref_single` smoke gate (one synthesis call via
`run_eval_zeroshot.py --system f5_tts --conditions ref_single --limit 1`) was attempted three times
between 2026-09-17T18:05Z and 2026-09-17T18:19Z (~14 minutes of wall-clock across the three
attempts, plus an earlier successful bare model-class load at 2026-09-17T17:57Z that took ~1.5
minutes and printed `F5TTS_LOADED_OK`). All three `run_eval_zeroshot.py`-driven attempts hung
indefinitely inside `F5TTS.__init__` (the constructor call in `code/adapters_zeroshot.py`'s
`load_f5tts_model`), producing **zero further log output** after `"Loading F5-TTS model..."` — no
vocoder-download print, no traceback, no progress — for the full duration of each attempt (up to ~15
minutes on the longest single attempt) before being killed.

## What was tried

1. **Attempt 1** (`CUDA_VISIBLE_DEVICES=0`, default HF caching): hung ~15 min, killed.
2. **Attempt 2** (`CUDA_VISIBLE_DEVICES=0`, `HF_HUB_OFFLINE=1` to force cached-only, no network
   round-trips): hung ~7 min, killed. `HF_HUB_OFFLINE` should have made this instant, since the
   model weights and Vocos vocoder were already confirmed present in
   `/mnt/cache/persist/t0018_zero_shot_cloning_calibration/hf-cache` from the successful bare-load
   test. It did not help, which rules out a network/etag-verification-latency explanation on its own
   (that explanation *did* account for CosyVoice2's and Chatterbox's multi-minute — but eventually
   successful — load times; F5-TTS's failure mode is qualitatively different: a true hang with no
   forward progress at all, not just slow progress).
3. **Attempt 3** (`CUDA_VISIBLE_DEVICES=1`, `HF_HUB_OFFLINE=1`, run concurrently with CosyVoice2's
   debug script on the same GPU): hung ~7 min, killed.

`nvidia-smi` showed 0 MiB used on the target GPU for the full duration of every attempt — the
process never even reached the point of allocating a CUDA context, let alone loading model weights
onto the device. Kernel wait-channel inspection (`/proc/<pid>/wchan`) showed `wait_for_response`
(the CIFS/SMB client's wait state for a pending network-filesystem RPC) throughout, consistent with
the VM's `/mnt/batch/.../code` mount (an Azure Files SMB share) being the proximate blocking point,
but the exact stuck syscall could not be identified (`/proc/<pid>/stack` and `/proc/<pid>/syscall`
both returned `Permission denied` for this non-root user).

## Why this is being treated as a genuine failure, not "still loading"

Every other system's load (CosyVoice2, Chatterbox, and F5-TTS's own first bare-class-load test)
showed **visible incremental progress** in logs (download progress bars, HTTP request logs, RSS
growth) even when slow (multi-minute). All three post-fix F5-TTS attempts showed **zero**
incremental signal for their entire duration, including the offline-mode attempt where no network
calls should have occurred at all. This asymmetry is the basis for concluding it is a hang specific
to `run_eval_zeroshot.py`'s F5-TTS code path, not generic filesystem slowness (which would still
show *some* progress, as it did for the other two systems).

## Resolution (REQ-7)

Per REQ-7 and the plan's Rejection Criteria: F5-TTS is marked **null for all variants**
(`f5_tts_ref_single_*`, `f5_tts_ref_concat_*`) in `results/metrics.json` / `results/tables.json`,
with this file as the recorded exact-error evidence. No substitute system was added (REQ-7 permits a
substitute only after all three named systems are attempted and only if time/budget remain; given
the tight $70 budget already largely consumed by setup + the other two systems' full runs, no
substitute was pursued).

**Given the successful bare `F5TTS(model="F5TTS_v1_Base", hf_cache_dir=...)` load at 17:57Z**, this
is very likely an environment/concurrency issue specific to this VM session (e.g., a stale lock file
left by an interrupted earlier attempt, or SMB-mount contention specific to `run_eval_zeroshot.py`'s
import order) rather than a fundamental F5-TTS incompatibility — a fresh VM session or a
`.venv-f5tts` reinstall would be the first thing to try in any follow-up task.

## Cost impact

Approximately 3.0 GPU-hours-equivalent of wall-clock time (~$4-5 at $13.96/hr, spread across three
attempts plus diagnostic overhead) was spent pursuing F5-TTS before this cutoff decision, given the
task's overall $70 hard cap and the setup step alone having already consumed ~$26-28 of it. This is
recorded in `results/cost_tracking.json` and `results/tables.json`'s notes.
