# Zero-Shot Voice-Cloning Calibration

## Motivation

The project's success criterion is `speaker_sim >= 0.85` (GE2E cosine vs ElevenLabs David). t0008
measured ElevenLabs David against **itself** (half-A centroid vs half-B clips) at 0.832 on fillers
and 0.792 on val96. The target is therefore above the reference voice's own self-consistency, and
the best Kokoro fine-tune (v3) sits at 0.631. After fifteen tasks nobody knows whether the 0.2 gap
is a Kokoro limitation, a fine-tuning-recipe limitation, or a ceiling of the metric on this voice.

Zero-shot voice-cloning models answer that question without any training: given a few seconds of
David audio and a text, they synthesize in that voice directly. Running three of them through the
existing t0008 harness gives, in one GPU session, an empirical envelope of what `speaker_sim` is
reachable on this voice and at what latency. That number recalibrates the success criterion (project
`description.md`), tells us whether to keep investing in Kokoro Stage 2, and gives a fallback
production candidate if one of them lands in the latency budget.

This is a measurement task, not a model-selection task. No fine-tuning, no tuning on val_96.

## Systems

Three open-weight zero-shot cloning models, chosen for being current, widely used, and installable
from public weights:

1. **F5-TTS** (`SWivid/F5-TTS`, base English checkpoint) — flow-matching, 10-30 s reference audio,
   non-autoregressive so RTF should be low but TTFB is a whole-utterance latency.
2. **CosyVoice 2** (`FunAudioLLM/CosyVoice2-0.5B`) — LLM + flow-matching, supports streaming output,
   so TTFB is meaningful.
3. **Chatterbox** (`ResembleAI/chatterbox`) — LLM-based, reference-clip cloning, widely deployed.

If one of the three cannot be installed or fails its smoke gate within 45 minutes of effort, mark
that system null with the exact error (Lesson 2), do not substitute silently, and continue with the
other two. A substitute (for example XTTS-v2 or Fish Speech) may be added only if time and budget
remain after the three named systems are done, and must be named in the results as an addition.

Also re-run in the same session, for paired comparison (Lesson 1):

4. `elevenlabs_david` scored against half-B (the t0008 protocol), as the self-consistency ceiling.
5. `kokoro_v3_bundle`, the current best fine-tune.

## Reference-audio conditions

Zero-shot quality depends heavily on the reference clip. Use two conditions per cloning system, both
drawn **only from the half-A split** (`build_reference_split(seed=42)` from the harness) so the
half-B centroid used for scoring never contains the reference:

* `ref_single`: one fixed ~10 s half-A clip (pick the longest clean half-A clip; record its
  filename).
* `ref_concat`: a fixed ~30 s concatenation of half-A clips (reuse
  `tasks/t0014_v11_decoder_fix_retrain/code/build_reference_concat.py`'s approach; record the
  filenames and order).

That is 3 systems × 2 conditions = 6 cloning variants, plus the two baselines.

## Prompt sets and protocol

Same as t0008 so the numbers are directly comparable: the 96 val96 texts
(`tasks/t0008_tts_eval_harness_baselines/data/val96_prompts.json`) and the 100 filler texts
(`data/filler_prompts_100.json`). Per system: smoke gate (one synthesis succeeds), 50 discarded
warmup requests, then all 196 prompts measured in one engine session. Record per clip: TTFB (first
audio chunk for streaming systems; whole-utterance latency for non-streaming, labelled as such),
RTF, audio duration, duration ratio vs the ElevenLabs reference clip, WER (faster-whisper `base.en`,
harness `compute_wer`), and GE2E cosine vs the half-B centroid.

Run the hardened audible-speech gate
(`tasks/t0015_v11_duration_blowup_forensics/code/audio_quality_check.py`) on every synthesized clip
and report the failure count per system; exclude gate-failing clips from `speaker_sim` means only if
that is stated in the table caption and the unfiltered mean is shown alongside.

Rejection: any system with `successful_prompts / total_prompts < 0.8` on a prompt set has null
metrics for that set (Lesson 3).

## Audio for human listening (mandatory)

The point of this task is for the owner to hear what each system does with David's voice, not only
to read a number. DVC-track (`dvc add`, `dvc push` before the PR) and index:

* `results/audio_samples/harness/<system>_<condition>/` — all 196 synthesized clips for every
  variant, as t0008 did with `synth_audio.dvc`. Nothing is discarded, including gate-failing clips.
* `results/audio_samples/comparison_set/` — a fixed side-by-side set: the three gate texts plus
  seven val96 prompts (seed 42, the same ten texts for every system), with the ElevenLabs original
  and the v3 bundle output for each, named `<text_id>__<system>_<condition>.wav` so one folder sorts
  by text.
* `results/audio_samples/references/` — the exact `ref_single` and `ref_concat` clips fed to the
  cloning models.
* `results/listening_guide.md` — one row per comparison text, one column per system/condition plus
  ElevenLabs and v3, each cell a clickable relative link with the per-clip `speaker_sim`, WER, and
  gate verdict; a "what to listen for" line per row (timbre match, accent drift, brand-name
  pronunciation, artifacts).

## Key Questions

1. What is the highest fillers and val96 `speaker_sim` any zero-shot system reaches on David, and
   how far is it from the ElevenLabs self-consistency ceiling (0.832 / 0.792)?
2. Does any zero-shot system beat the best Kokoro fine-tune (0.631 fillers) — and by how much?
3. Which of them, if any, meets TTFB p50 ≤ 300 ms on H100, and what is the RTF? For non-streaming
   systems, report whole-utterance latency and state that TTFB equals it.
4. How much does the reference condition matter (`ref_single` vs `ref_concat`) per system?
5. Is WER on val96 (brand names, product terms) acceptable, or do the cloning models trade
   intelligibility for similarity?
6. Given 1-5, should the project's success criterion be restated (for example as a fraction of
   ElevenLabs self-consistency), and is Kokoro Stage 2 still the right main line?

## Expected Outputs

* `assets/answer/zero-shot-speaker-sim-ceiling/` — one answer asset for the question "What
  speaker_sim and latency envelope is reachable on the David voice by zero-shot cloning, and what
  does that imply for the Kokoro fine-tuning line?" Short answer states the numbers; full answer
  carries the tables and the recommendation on the success criterion.
* `results/per_clip_metrics.json` — ≥ 100 rows per system (8 variants × 196 prompts ≈ 1568 rows).
* `results/metrics.json` — explicit variants: each system/condition × prompt set, with
  `speaker_sim`, `ttfb_ms`, `rtf`, plus WER, duration ratio, gate-failure count, and
  `efficiency_inference_time_per_item_seconds`, `efficiency_inference_cost_per_item_usd` (machine
  hourly price × wall-clock / clips).
* `results/audio_samples/{harness,comparison_set,references}/` (DVC) and
  `results/listening_guide.md` — see "Audio for human listening".
* Charts in `results/images/`, embedded in `results_detailed.md`: `speaker_sim_by_system.png`
  (grouped bars, fillers vs val96, with horizontal lines at the ElevenLabs ceiling and the v3 score;
  Q1, Q2), `ttfb_vs_speaker_sim.png` (scatter, one point per variant, x: TTFB p50 ms with a vertical
  line at 300, y: fillers speaker_sim; Q3), `ref_condition_effect.png` (paired bars per system; Q4),
  `wer_by_system.png` (Q5).
* Tables: per variant × prompt set (speaker_sim mean ± std, TTFB p50/p95/p99, RTF mean ± std, WER,
  explosions, gate failures, n successful); environment table (model commit/version, torch, CUDA,
  Lesson 4).
* `results/suggestions.json` — including a proposed restated success criterion for
  `project/description.md` and, if a system meets both bars, a production-integration feasibility
  task.

## Compute and budget

* `LLM-T1-NC80` (H100). Zero-shot LLM-based models are heavier than Kokoro: budget ≈ 0.5 h setup and
  downloads (weights to `/mnt/cache/persist/pretrained/`, Lesson 10), ≈ 0.4 h per cloning variant ×
  6, ≈ 0.3 h for the two baselines and teardown. **≈ 3.2 h × $13.96 ≈ $45.** Hard cap: **$70.**
  ElevenLabs API: none (reference clips already exist; the ElevenLabs baseline is re-scored from
  t0008's stored synthesized audio if `synth_audio.dvc` is pulled, otherwise skipped and t0008's
  numbers quoted with the session caveat).
* Watchdog armed and PID confirmed before the first model download (Lesson 8).

## Forbidden

* No fine-tuning of any system. No tuning of reference selection on val_96.
* No `resemblyzer` in main dependencies (keep the `[speaker-sim]` extra).
* No substitution of a named system without a null result and the recorded reason.

## Dependencies

* `t0008_tts_eval_harness_baselines` — `tts_eval_harness` library, prompt sets, half-A/half-B split,
  and the paired baseline numbers. Everything else needed is public.
