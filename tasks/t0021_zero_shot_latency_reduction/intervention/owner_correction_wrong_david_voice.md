# Owner correction: t0008/t0018 used the wrong ElevenLabs "David" voice

## What happened

On 2026-09-18, the project owner found that the ElevenLabs account has two voices named "David".
Production (`brainpowa-voice-gateway` FluxCD) and the Kokoro training corpus `data/v4` use
`voice_id=rWV5HleMkWb5oluMwkA7` ("David - narrator and newsreader", `model_id=eleven_flash_v2_5`,
`output_format=pcm_24000`). `t0008_tts_eval_harness_baselines`'s harness resolved the voice by
fuzzy name match and picked `voice_id=5gLuKtB16QIQv1vuSas1` ("David - British Radio Host") instead.

As a result, `data/11labs_david/` (the corpus t0008 built and t0018 reused) is the WRONG David
voice. `t0018_zero_shot_cloning_calibration`'s `ref_single`/`ref_concat` reference clips and its
speaker-similarity centroid (`data/references/half_a_centroid.npy`) were built from
`data/11labs_david` half-A, so they clone and score against the wrong voice.

This correction was injected by the coordinator before the planning step (step 7) of
`t0021_zero_shot_latency_reduction` and is recorded verbatim in that task's `checkpoint.md`
"Cross-Step Decisions" section and in `plan/plan.md`'s "Owner Correction" section.

## Why

**Root cause, located precisely**: `get_elevenlabs_voice_id()` in
`tasks/t0008_tts_eval_harness_baselines/code/run_eval.py` (lines ~110-137) resolves the ElevenLabs
voice by calling the ElevenLabs `/voices` API and matching by `name == ELEVENLABS_DAVID_VOICE_NAME`
(falling back to a fuzzy case-insensitive substring match on `"david"` if the exact name is not
found). Because the account has two voices whose names both contain "David", this name-based
resolution is inherently ambiguous and picked the "David - British Radio Host" voice
(`5gLuKtB16QIQv1vuSas1`) instead of the production "David - narrator and newsreader" voice
(`rWV5HleMkWb5oluMwkA7`). No part of the original t0008 code pinned a `voice_id` directly.

## Resolution

* `t0018_zero_shot_cloning_calibration` is **completed and immutable**. This correction is **not**
  applied retroactively there — t0018's own `results/metrics.json` and `results/tables.json` stand
  as originally recorded, but their `speaker_sim` numbers were measured against the wrong voice.
  For the record, the exact values (read directly from
  `tasks/t0018_zero_shot_cloning_calibration/results/metrics.json`, not from memory):
  * CosyVoice2 `ref_single`: `speaker_sim` = `0.8627852474649748` (val96),
    `0.8420862078666687` (fillers) — against `voice_id=5gLuKtB16QIQv1vuSas1`.
  * Chatterbox `ref_single`: `speaker_sim` = `0.8073432354987422` (val96),
    `0.8112275004386902` (fillers) — against `voice_id=5gLuKtB16QIQv1vuSas1`.
  * Chatterbox `ref_concat`: `speaker_sim` = `0.809679239988327` (val96),
    `0.79637620680862` (fillers) — against `voice_id=5gLuKtB16QIQv1vuSas1`.
  * CosyVoice2 `ref_concat` is `null` for both prompt sets in t0018 (S-0018-02, the 30s
    hard-limit failure this task's Step 11/Milestone 4 closes).

* From this task (`t0021_zero_shot_latency_reduction`) onward, every reference clip and every
  speaker-similarity centroid is built from `data/v4/val/wavs` (the val_96 held-out set, which IS
  the correct production voice, since `data/v4` is the Kokoro training corpus recorded against
  `voice_id=rWV5HleMkWb5oluMwkA7`), never from `data/11labs_david`. See
  `data/references/manifest.json` for the exact source filenames used.

* This task's own numbers are therefore the **first correct-voice measurement** of
  CosyVoice2/Chatterbox zero-shot cloning quality, and are **not directly comparable** to t0018's
  headline numbers without the explicit "radiohost (wrong voice) control" column
  (`speaker_sim_radiohost_control` in `results/per_clip_metrics.json` and `results/tables.json`),
  scored against `data/references/old_wrongvoice_centroid.npy` — a verbatim copy of t0018's
  `half_a_centroid.npy`, never rebuilt from `data/11labs_david` in this task.

* **Standing constraint (never an action item — see `plan/plan.md`'s Approach section):** this task
  makes zero new ElevenLabs API calls. If any future task calls the ElevenLabs API for the "David"
  voice, it MUST pin `voice_id=rWV5HleMkWb5oluMwkA7`, `model_id=eleven_flash_v2_5`,
  `output_format=pcm_24000`, `stability=0.5`, `similarity_boost=0.75` explicitly, and MUST NEVER
  resolve the voice by name (i.e., must never call `get_elevenlabs_voice_id()` or any equivalent
  fuzzy name-matching helper). This task's own `code/` never calls that function — confirmed by
  `grep -rq get_elevenlabs_voice_id tasks/t0021_zero_shot_latency_reduction/code/` returning no
  match.

* **Gate caveat**: t0015's `audio_quality_check.py` gate (`hardened_gate_pass` in this task's
  `results/per_clip_metrics.json`) is known to pass clips a human listener would describe as "voice
  plus strong noise." A PASS on this automated gate is necessary, not sufficient, for a
  production-quality claim. The owner will listen to `results/listening_guide.md`'s clips directly
  before any production-readiness conclusion is drawn from this task's outputs.

## Cost impact

None. This is a paperwork/methodology correction, resolved entirely with CPU-only, local work
(reference/centroid rebuild from already-local DVC-tracked `data/v4/val/wavs`) before the GPU VM's
billing clock was touched further. No ElevenLabs API spend, no GPU spend attributable to this
correction itself.
