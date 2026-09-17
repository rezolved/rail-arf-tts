# kokoro_v3_bundle: not re-measured this session (stored t0008 numbers cited instead)

## What happened

Two attempts to re-synthesize `kokoro_v3_bundle` fresh in this task's GPU session (REQ-3) both hung
indefinitely inside model loading, with the exact same failure signature already documented for
F5-TTS in `intervention/f5_tts_smoke_gate_failed.md`:

1. **Attempt 1**: loaded `V3_DECODER` directly from
   `/mnt/cache/persist/t0018_zero_shot_cloning_calibration/repo/tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/best/david_v3_best_decoder_kokoro.pth`
   (a path that, per its own printed log line, actually resolves through
   `/mnt/batch/tasks/shared/LS_root/mounts/clusters/llm-t1-nc80/code/...` -- the same slow Azure
   Files SMB mount implicated in the F5-TTS hang). Hung ~11 minutes with zero progress, killed.
2. **Attempt 2**: worked around the suspected network-filesystem `torch.load` slowness by first
   plain-`cp`-ing the 313 MB decoder checkpoint and the voicepack to local disk (`/tmp/kokoro_ckpt/`
   -- confirmed fast, seconds not minutes, proving raw sequential I/O on this mount is fine and the
   slowness is specific to how `torch.load`/model-loading code accesses the file) and loading from
   there instead (`code/run_kokoro_v3_local_ckpt.py`). This **still hung** (~90+ seconds with
   negligible CPU time accumulation, kernel state consistent with a blocked I/O wait), so the
   local-copy workaround that this exact reasoning predicted would fix it did not fully resolve the
   issue -- something else in the kokoro/misaki loading path (e.g., `hf_hub_download` for
   `hexgrad/Kokoro-82M`'s `config.json`, needed by `build_pipeline`'s `install_lexicon`, going out
   to HuggingFace Hub under the same rate-limiting/latency conditions observed for every other
   system's HF Hub calls this session) is also implicated, not only the checkpoint file read.

Given two independent attempts failed and the task's remaining budget, no further attempts were
made.

## Resolution

Per `task_description.md`'s own explicit fallback pattern (used identically for `elevenlabs_david`
in Step 8a): **t0008's stored `kokoro_v3_bundle` numbers are cited instead of a fresh
re-measurement**, with an explicit `"source": "t0008 stored, not re-measured this session"` field,
and **`ttfb_ms`/`rtf` are omitted** rather than falsely presenting a different session's latency
numbers as if they were measured in this task's own engine session (Lesson 1). Source values (from
`tasks/t0008_tts_eval_harness_baselines/results/metrics.json`, verified directly, not from memory):

* `kokoro_v3_bundle_fillers`: `speaker_sim = 0.6309629625082016`
* `kokoro_v3_bundle_val96`: `speaker_sim = 0.5876460997387767`

These are the same numbers already used throughout `plan/plan.md` (rounded to 0.631/0.588) as the
comparison baseline this task is calibrating the three zero-shot systems against, so this does not
change any of this task's headline conclusions -- it only means `kokoro_v3_bundle`'s row in
`results/tables.json`/`results/metrics.json` carries a `speaker_sim` value with a "not re-measured"
provenance note and `null` `ttfb_ms`/`rtf`, rather than a freshly measured row.

## Cost impact

Approximately 2 GPU-hours-equivalent of wall-clock time (~$3-4, spread across two attempts plus the
checkpoint-copy diagnostic) was spent pursuing `kokoro_v3_bundle` before this cutoff decision.
Recorded in `results/cost_tracking.json`.

## Follow-up recommendation

Both this hang and F5-TTS's hang point at the same underlying environment issue: model-loading code
paths that read large files or hit HuggingFace Hub from this specific Azure ML VM pool
(`LLM-T1-NC80`) intermittently hang for many minutes with zero progress, while plain sequential file
copies and some other model loads (CosyVoice2, Chatterbox) eventually succeed (slowly). A follow-up
task should investigate the VM's `/mnt/batch/tasks/shared/...` SMB mount and HF Hub connectivity
specifically (e.g., set `HF_HUB_OFFLINE=1` and pre-stage ALL weights before any of the 4 model loads
in a session, or provision `HF_TOKEN` to escape the "unauthenticated requests" rate limit warning
that appeared for every HF Hub call this session) before relying on this VM pool for other
multi-model TTS benchmark tasks.
