# ⏹ TTS evaluation harness and baselines

[Back to all tasks](../README.md)

## Overview

| Field | Value |
|---|---|
| **ID** | `t0008_tts_eval_harness_baselines` |
| **Status** | ⏹ not_started |
| **Dependencies** | [`t0006_kokoro_v5_stage2_subset`](../../../overview/tasks/task_pages/t0006_kokoro_v5_stage2_subset.md) |
| **Task types** | `tts-benchmark-run`, `baseline-evaluation` |
| **Expected assets** | 1 library |
| **Task folder** | [`t0008_tts_eval_harness_baselines/`](../../../tasks/t0008_tts_eval_harness_baselines/) |

<details>
<summary><strong>Task Description</strong></summary>

*Source:
[`task_description.md`](../../../tasks/t0008_tts_eval_harness_baselines/task_description.md)*

# TTS Evaluation Harness and Baselines

## Motivation

After six tasks (t0001-t0006) the project has not measured a single one of its success
criteria. Every training attempt was judged by StyleTTS2 `val_loss` and by listening. The
success criteria in `project/description.md` are speaker similarity (GE2E cosine ≥ 0.85
against ElevenLabs David) and TTFB ≤ 300 ms. There is no ElevenLabs baseline, no base-Kokoro
baseline, and no speaker-similarity code anywhere in the repo.

t0003 also showed that Stage 1 `val_loss` does not predict success (0.770 preceded the working
v3 Stage 2, 0.545 did not), so `val_loss` alone cannot be trusted as a quality signal.

This task builds the measurement tool once, as a reusable library, and uses it to answer
research questions 1 and 2 and to place every existing checkpoint on the same scale. It runs
no training.

## Key Questions

1. What are speaker_sim, TTFB (p50/p95/p99) and RTF for ElevenLabs David on ≥ 50 filler
   prompts?
2. What do base Kokoro-82M British male voices score on the same prompts?
3. How close is the v3 shipped bundle (the only checkpoint with clean audio) to speaker_sim ≥
   0.85?
4. Do the t0005 and t0006 best checkpoints score measurably worse than v3, i.e. does the
   harness detect the "noisy audio" that listening detected?
5. Is speaker_sim alone enough to reject broken audio, or are the duration and intelligibility
   sanity metrics needed as well?

## Systems to Evaluate

Every system is scored on the same prompt sets with the same harness:

1. **ElevenLabs David** — API (voice ID David). TTFB and RTF measured live through the
   streaming API; speaker_sim computed on held-out reference clips (see Scoring below).
2. **Kokoro-82M base, `bm_george`** — stock voicepack.
3. **Kokoro-82M base, `bm_lewis`** — stock voicepack.
4. **Kokoro-82M base decoder + v3 David voicepack** — `david_v3_best_voicepack.pt` with stock
   decoder/predictor. Tells us how much of the voice a voicepack alone captures, with no
   fine-tuned weights.
5. **v3 shipped bundle** — `david_v3_best_decoder_kokoro.pth` + `david_v3_best_voicepack.pt`
   from `tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/best/` (DVC).
6. **t0006 v6d epoch 6** —
   `tasks/t0006_kokoro_v5_stage2_subset/results/checkpoints/v6d/epoch_2nd_00006.pth`, packaged
   with t0002's five-module extraction (`extract_decoder_generic.py`).
7. **t0005 run06 best** — `tasks/t0005_kokoro_v5_stage2_train/results/checkpoints/` best
   checkpoint, same packaging. Note: t0005's results say epoch 3 but `logs/README.md` says
   epoch 2; record which file was actually scored.

Floor control: one clearly different Kokoro voice (e.g. `af_heart`, female American) to show
what a "wrong speaker" score looks like, so 0.85 can be interpreted.

All Kokoro arms must synthesize through
`tasks/t0003_kokoro_v5_phoneme_data/code/build_pipeline.py` (British `lang_code="b"` plus
brand lexicon), not a bare `KPipeline` — t0003 showed accent and lexicon mismatches silently
change the token stream.

## Prompt Sets and Data

* **val_96**: the 96 held-out clips (`data/v4/val/`, text from
  `tasks/t0003_kokoro_v5_phoneme_data/results/v5/val_list.txt`). Real David audio is available
  for each prompt, so per-prompt paired comparison is possible. Never train on these.
* **Fillers**: ≥ 50 prompts sampled with a fixed seed (seed=42) from the 1358-clip ElevenLabs
  filler corpus. The corpus currently lives only on the VM
  (`kokoro-finetune/data/11labs_david/`, see `overview/datasets/fillers_1358.md`) and is not
  DVC-tracked. First step: copy it off the VM, `dvc add` it under this task's `data/`, and
  `dvc push`. Beware LESSONS Lesson 10 — `/mnt` on Azure ML is ephemeral; confirm the corpus
  still exists before planning around it.

## Scoring

* **speaker_sim** (registered metric): resemblyzer GE2E cosine between each synthesized clip
  and the mean embedding of the ElevenLabs David reference set. Split the reference clips into
  a centroid half and a held-out half with a fixed seed; the ElevenLabs arm is scored on the
  held-out half so it is not compared against itself. Report mean, std and per-clip
  distribution.
* **ttfb_ms** (registered metric): for ElevenLabs, wall time from request to first streamed
  audio byte. For Kokoro, wall time from pipeline call to the first audio chunk yielded.
  Follow LESSONS Lesson 1 (discarded warmup requests before measuring) and Lesson 4 (record
  torch/CUDA/kokoro versions and GPU at measurement time). Report p50/p95/p99.
* **rtf** (registered metric): synthesis wall time divided by output audio duration.
* **duration_ratio** (sanity): synthesized duration divided by reference clip duration.
  Catches the 10× duration explosion seen in t0002 before anyone listens.
* **WER** (sanity): transcribe synthesized audio with a local Whisper model and compare
  against the prompt text. Catches garbled or noisy audio that could still score a plausible
  speaker_sim.

## Compute and Budget

* Quality metrics (speaker_sim, duration_ratio, WER) can run on CPU or a single GPU.
* The success criterion defines TTFB "local inference, H100". Kokoro TTFB/RTF should be
  measured on LLM-T1-NC80 in one short window (target ≤ 1.5 h, ~$21), with setup and teardown
  through the `setup-remote-machine` skill. A CPU TTFB number may be reported in addition,
  clearly labeled.
* ElevenLabs API: ~150 streaming requests of short filler text, well under $5.
* **Budget note**: the project budget was raised to $5000 in t0007. `aggregate_costs` still
  reports $0 spent because t0001-t0006 write `total_usd` instead of `total_cost_usd`; real
  spend so far is ~~$370 recorded plus unrecorded t0006 GPU time (~~$70). This task must write
  `total_cost_usd`. Planned total for this task: ≤ $30.

## Expected Outputs

* **Library asset**: the evaluation harness (prompt-set loader, per-system synth adapters,
  scoring, report writer), reusable by every future training task.
* `results/metrics.json` with speaker_sim, ttfb_ms, rtf per system (as variants).
* Tables in `results_detailed.md`:
  * Main table: rows = systems 1-7 + floor control; columns = speaker_sim mean/std, ttfb_ms
    p50/p95/p99, rtf, duration_ratio median, WER; plus delta vs ElevenLabs.
  * Per-prompt-set breakdown (val_96 vs fillers).
* Charts in `results/images/`, embedded in `results_detailed.md`:
  * speaker_sim distribution per system (box plot; y = GE2E cosine, dashed line at 0.85) —
    answers Q1-Q4.
  * TTFB CDF per system (x = ms, dashed line at 300 ms) — answers Q1-Q2.
  * speaker_sim vs WER scatter per clip — answers Q5.
* A few synthesized examples per system (DVC-tracked, not git).
* A clear statement: which systems (if any) already pass each success criterion.

## Dependencies

* `t0006_kokoro_v5_stage2_subset` — provides the v3 reference bundle and the v6d checkpoint.
* Independent of t0009; the two can run in parallel.

</details>
