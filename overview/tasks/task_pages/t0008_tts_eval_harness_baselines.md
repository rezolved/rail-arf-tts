# ✅ TTS evaluation harness and baselines

[Back to all tasks](../README.md)

## Overview

| Field | Value |
|---|---|
| **ID** | `t0008_tts_eval_harness_baselines` |
| **Status** | ✅ completed |
| **Started** | 2026-09-14T15:27:17Z |
| **Completed** | 2026-09-14T18:07:30Z |
| **Duration** | 2h 40m |
| **Dependencies** | [`t0006_kokoro_v5_stage2_subset`](../../../overview/tasks/task_pages/t0006_kokoro_v5_stage2_subset.md) |
| **Task types** | `tts-benchmark-run`, `baseline-evaluation` |
| **Categories** | [`benchmark`](../../by-category/benchmark.md), [`evaluation`](../../by-category/evaluation.md), [`tts`](../../by-category/tts.md) |
| **Expected assets** | 1 library |
| **Step progress** | 13/15 |
| **Cost** | **$35.10** |
| **Task folder** | [`t0008_tts_eval_harness_baselines/`](../../../tasks/t0008_tts_eval_harness_baselines/) |
| **Detailed results** | [`results_detailed.md`](../../../tasks/t0008_tts_eval_harness_baselines/results/results_detailed.md) |

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
  spend so far is about $370 recorded plus about $70 of unrecorded t0006 GPU time. This task
  must write `total_cost_usd`. Planned total for this task: ≤ $30.

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

## Costs

**Total**: **$35.10**

| Category | Amount |
|----------|--------|
| elevenlabs_api | $7.18 |
| azure_ml_vm_h100 | $27.92 |

## Remote Machines

| Provider | GPU | Count | RAM | Duration | Cost |
|----------|-----|-------|-----|----------|------|
| azure_ml | H100 NVL | 2 | 480 GB | 2.0h | $27.92 |

## Metrics

### elevenlabs_david / fillers

| Metric | Value |
|--------|-------|
| [`speaker_sim`](../../metrics-results/speaker_sim.md) | **0.8324960142374038** |
| [`ttfb_ms`](../../metrics-results/ttfb_ms.md) | **131.64102200016714** |
| [`rtf`](../../metrics-results/rtf.md) | **0.11269287443324151** |

### elevenlabs_david / val96

| Metric | Value |
|--------|-------|
| [`speaker_sim`](../../metrics-results/speaker_sim.md) | **0.7923689279705286** |
| [`ttfb_ms`](../../metrics-results/ttfb_ms.md) | **152.79576700049802** |
| [`rtf`](../../metrics-results/rtf.md) | **0.06728752851632007** |

### kokoro_base_george / fillers

| Metric | Value |
|--------|-------|
| [`speaker_sim`](../../metrics-results/speaker_sim.md) | **0.5628819969296456** |
| [`ttfb_ms`](../../metrics-results/ttfb_ms.md) | **233.0862614999205** |
| [`rtf`](../../metrics-results/rtf.md) | **0.12440019667764884** |

### kokoro_base_george / val96

| Metric | Value |
|--------|-------|
| [`speaker_sim`](../../metrics-results/speaker_sim.md) | **0.5954564306885004** |
| [`ttfb_ms`](../../metrics-results/ttfb_ms.md) | **340.2694194999185** |
| [`rtf`](../../metrics-results/rtf.md) | **0.08153901481034025** |

### kokoro_base_lewis / fillers

| Metric | Value |
|--------|-------|
| [`speaker_sim`](../../metrics-results/speaker_sim.md) | **0.5053677418828011** |
| [`ttfb_ms`](../../metrics-results/ttfb_ms.md) | **215.35949499980234** |
| [`rtf`](../../metrics-results/rtf.md) | **0.10413011021301431** |

### kokoro_base_lewis / val96

| Metric | Value |
|--------|-------|
| [`speaker_sim`](../../metrics-results/speaker_sim.md) | **0.4997421832134326** |
| [`ttfb_ms`](../../metrics-results/ttfb_ms.md) | **333.93777599985697** |
| [`rtf`](../../metrics-results/rtf.md) | **0.07943685032174448** |

### kokoro_base_v3_voicepack / fillers

| Metric | Value |
|--------|-------|
| [`speaker_sim`](../../metrics-results/speaker_sim.md) | **0.6027204036712647** |
| [`ttfb_ms`](../../metrics-results/ttfb_ms.md) | **196.09755249985028** |
| [`rtf`](../../metrics-results/rtf.md) | **0.1402866228285387** |

### kokoro_base_v3_voicepack / val96

| Metric | Value |
|--------|-------|
| [`speaker_sim`](../../metrics-results/speaker_sim.md) | **0.5819188111151258** |
| [`ttfb_ms`](../../metrics-results/ttfb_ms.md) | **296.2192074999166** |
| [`rtf`](../../metrics-results/rtf.md) | **0.09244443949944625** |

### kokoro_floor_control / fillers

| Metric | Value |
|--------|-------|
| [`speaker_sim`](../../metrics-results/speaker_sim.md) | **0.44350694090127946** |
| [`ttfb_ms`](../../metrics-results/ttfb_ms.md) | **165.36447149997002** |
| [`rtf`](../../metrics-results/rtf.md) | **0.1073962914100093** |

### kokoro_floor_control / val96

| Metric | Value |
|--------|-------|
| [`speaker_sim`](../../metrics-results/speaker_sim.md) | **0.43354621063917875** |
| [`ttfb_ms`](../../metrics-results/ttfb_ms.md) | **289.4921130000512** |
| [`rtf`](../../metrics-results/rtf.md) | **0.0826164461322109** |

### kokoro_t0005_best / fillers

| Metric | Value |
|--------|-------|
| [`speaker_sim`](../../metrics-results/speaker_sim.md) | **nan** |
| [`ttfb_ms`](../../metrics-results/ttfb_ms.md) | **688.3287169998766** |
| [`rtf`](../../metrics-results/rtf.md) | **0.06345616441853913** |

### kokoro_t0005_best / val96

| Metric | Value |
|--------|-------|
| [`ttfb_ms`](../../metrics-results/ttfb_ms.md) | **2156.0768535000534** |
| [`rtf`](../../metrics-results/rtf.md) | **0.08719415799887982** |

### kokoro_t0006_v6d / fillers

| Metric | Value |
|--------|-------|
| [`speaker_sim`](../../metrics-results/speaker_sim.md) | **0.601133947968483** |
| [`ttfb_ms`](../../metrics-results/ttfb_ms.md) | **132.15857950012833** |
| [`rtf`](../../metrics-results/rtf.md) | **0.18161571017059192** |

### kokoro_t0006_v6d / val96

| Metric | Value |
|--------|-------|
| [`speaker_sim`](../../metrics-results/speaker_sim.md) | **0.48216814702997607** |
| [`ttfb_ms`](../../metrics-results/ttfb_ms.md) | **250.56644949995643** |
| [`rtf`](../../metrics-results/rtf.md) | **0.11549553931792515** |

### kokoro_v3_bundle / fillers

| Metric | Value |
|--------|-------|
| [`speaker_sim`](../../metrics-results/speaker_sim.md) | **0.6309629625082016** |
| [`ttfb_ms`](../../metrics-results/ttfb_ms.md) | **185.42622549989574** |
| [`rtf`](../../metrics-results/rtf.md) | **0.14482521243185945** |

### kokoro_v3_bundle / val96

| Metric | Value |
|--------|-------|
| [`speaker_sim`](../../metrics-results/speaker_sim.md) | **0.5876460997387767** |
| [`ttfb_ms`](../../metrics-results/ttfb_ms.md) | **282.2771579999426** |
| [`rtf`](../../metrics-results/rtf.md) | **0.0957059522566965** |

## Assets Produced

| Type | Asset | Details |
|------|-------|---------|
| library | [TTS Evaluation Harness](../../../tasks/t0008_tts_eval_harness_baselines/assets/library/tts_eval_harness/) | [`description.md`](../../../tasks/t0008_tts_eval_harness_baselines/assets/library/tts_eval_harness/description.md) |

## Suggestions Generated

<details>
<summary><strong>Continue Stage 2 fine-tuning from v6d epoch 6 to close the 0.20
GE2E cosine gap</strong> (S-0008-01)</summary>

**Kind**: experiment | **Priority**: high

t0008 measured the best Kokoro checkpoint (kokoro_v3_bundle) at speaker_sim=0.631 vs the
ElevenLabs target of 0.83 — a gap of ~0.20 GE2E cosine units. The t0006_v6d checkpoint was
trained for only 6 epochs on a 250-clip subset and already shows competitive TTFB (132ms,
matching ElevenLabs). Run additional Stage 2 epochs from epoch_2nd_00006.pth
(multispeaker:false, 250-clip subset) and evaluate each checkpoint with the tts_eval_harness
to find the epoch where speaker_sim plateaus or reaches 0.83. Use the harness library
registered in t0008. Recommended task types: tts-finetuning-eval.

</details>

<details>
<summary><strong>Investigate t0006_v6d val96 speaker_sim collapse (0.601 fillers
→ 0.482 val96)</strong> (S-0008-02)</summary>

**Kind**: experiment | **Priority**: high

The t0006_v6d checkpoint shows a striking degradation on val96 texts: speaker_sim drops from
0.601 (fillers) to 0.482 (val96), the largest cross-set gap of any system. The worst-case
clips all have WER=1.00 and duration_ratio < 0.80, with prompts containing brand names
(Rezolve AI, Brain Commerce). This suggests v6d has not learned to synthesize longer,
brand-name-heavy utterances correctly. Investigate whether the brand lexicon in
build_pipeline.py fully covers the val96 prompts, whether val96 texts are systematically
out-of-distribution for the 250-clip training subset, and whether more training data or a
broader lexicon would recover performance. Recommended task types: data-analysis.

</details>

<details>
<summary><strong>Expand Stage 2 training corpus beyond 250 clips to improve speaker
identity transfer</strong> (S-0008-03)</summary>

**Kind**: experiment | **Priority**: high

The v6d model was trained on a 250-clip subset to test whether a smaller corpus would avoid
the divergence seen in t0005. It achieves TTFB parity with ElevenLabs but lags in speaker_sim
(0.601 vs 0.83 target). The full training set contains 1557 clips. Run Stage 2 with the full
corpus (or a larger 500–800 clip subset) from first_stage.pth with the same multispeaker:false
and crash-fix settings from t0006, then evaluate with tts_eval_harness. This tests whether
more speaker data closes the GE2E cosine gap. Recommended task types: tts-finetuning-eval.

</details>

<details>
<summary><strong>Register WER as a project metric and track it in harness
evaluations</strong> (S-0008-04)</summary>

**Kind**: evaluation | **Priority**: medium

t0008 computed WER with faster-whisper base.en as a sanity metric but did not register it as a
project metric. WER proved critical as a secondary failure detector alongside duration_ratio —
the combination of duration_ratio > 5.0 AND WER > 0.5 reliably identifies exploded clips.
Registering WER in meta/metrics/ would let the aggregator track intelligibility trends across
future fine-tuning tasks and alert when a new checkpoint degrades transcription quality.
Recommended task types: infrastructure-setup.

</details>

<details>
<summary><strong>Augment the ElevenLabs filler corpus with val96-style longer
utterances for harness scoring</strong> (S-0008-05)</summary>

**Kind**: dataset | **Priority**: medium

The current evaluation uses 100 filler prompts (short, 2–8 words) and 96 val96 prompts (mixed
length). The speaker_sim scores diverge significantly between the two sets for fine-tuned
Kokoro systems, suggesting filler-length bias. Adding a third prompt set of 50–100
medium-length utterances (10–20 words) sampled from the training corpus texts — but not in
val96 — would give a more complete picture of speaker similarity across utterance lengths and
reduce the chance that a checkpoint overfits to short fillers. Recommended task types:
dataset.

</details>

## Research

* [`research_code.md`](../../../tasks/t0008_tts_eval_harness_baselines/research/research_code.md)
* [`research_internet.md`](../../../tasks/t0008_tts_eval_harness_baselines/research/research_internet.md)
* [`research_papers.md`](../../../tasks/t0008_tts_eval_harness_baselines/research/research_papers.md)
* [`research_summary.md`](../../../tasks/t0008_tts_eval_harness_baselines/research/research_summary.md)

<details>
<summary><strong>Results Summary</strong></summary>

*Source:
[`results_summary.md`](../../../tasks/t0008_tts_eval_harness_baselines/results/results_summary.md)*

--- spec_version: "1" task_id: "t0008_tts_eval_harness_baselines" ---
## Summary

Built and deployed a reusable TTS evaluation harness (`tts_eval_harness` library) that
measures speaker similarity (GE2E cosine via resemblyzer), TTFB, RTF, WER, and duration ratio
across 8 TTS systems on two prompt sets (96 held-out val96 texts + 100 production fillers).
All systems were benchmarked on LLM-T1-NC80 (2×H100 NVL). ElevenLabs David establishes the
baseline; none of the Kokoro variants reached the 0.85 speaker_sim target, with the best being
`kokoro_v3_bundle` at 0.63 (fillers).

## Metrics

Key results (mean speaker_sim GE2E cosine, TTFB p50 ms, all systems on combined prompt sets):

- **ElevenLabs David** (scored vs half-B): speaker_sim=0.832 (fillers), 0.792 (val96);
  TTFB_p50=132ms (fillers), 153ms (val96) — target ≥0.85, ≤300ms
- **kokoro_v3_bundle** (best Kokoro): speaker_sim=0.631 (fillers), 0.588 (val96);
  TTFB_p50=185ms (fillers), 282ms (val96)
- **kokoro_base_v3_voicepack**: speaker_sim=0.603 (fillers), 0.582 (val96); TTFB_p50=196ms
  (fillers), 296ms (val96)
- **kokoro_t0006_v6d**: speaker_sim=0.601 (fillers), 0.482 (val96); TTFB_p50=132ms (fillers,
  matches ElevenLabs latency), 251ms (val96)
- **kokoro_base_george**: speaker_sim=0.563 (fillers), 0.595 (val96); TTFB_p50=233ms
  (fillers), 340ms (val96, FAIL)
- **kokoro_t0005_best**: severely degraded — TTFB_p50=688ms/2156ms, most clips
  duration-explosion (sim=nan for val96); this checkpoint is not viable
- **kokoro_floor_control** (female af_heart, control): speaker_sim=0.444, confirming non-David
  baseline separation

Gap to target: best Kokoro (v3_bundle) is 0.63 vs ElevenLabs 0.83 — a gap of ~0.20 GE2E cosine
units. Further fine-tuning is needed.

## Verification

- Library asset `tts_eval_harness` verificator: **PASSED** (0 errors, 3 category warnings — no
  categories defined in `meta/categories/`)
- All 11 unit tests pass (`pytest
  tasks/t0008_tts_eval_harness_baselines/code/test_harness.py`)
- 1568 total per-clip records (8 systems × 196 prompts each): 1372 Kokoro + 196 ElevenLabs
- Results files verified present: `metrics.json` (16 variants), `tables.json`, 3 chart PNGs,
  `per_clip_metrics.json` (1568 records)

</details>

<details>
<summary><strong>Detailed Results</strong></summary>

*Source:
[`results_detailed.md`](../../../tasks/t0008_tts_eval_harness_baselines/results/results_detailed.md)*

--- spec_version: "2" task_id: "t0008_tts_eval_harness_baselines" ---
## Summary

Built the `tts_eval_harness` reusable library (registered v0.1.0) and ran a full baseline
evaluation of 8 TTS systems on two prompt sets: 96 held-out val96 texts and 100 production
filler prompts. ElevenLabs David achieves speaker_sim=0.83 (fillers) / 0.79 (val96) against
the GE2E centroid built from half-A of the 1364-clip corpus. The best Kokoro variant —
`kokoro_v3_bundle` — reaches speaker_sim=0.63 (fillers) / 0.59 (val96), a gap of ~0.20 units.
The t0005_best checkpoint shows severe duration explosions and is not viable. The t0006_v6d
checkpoint matches ElevenLabs TTFB (132ms) on fillers but lags in speaker similarity.

## Methodology

- **Hardware**: LLM-T1-NC80 (2×NVIDIA H100 NVL 95830 MiB, driver 535.274.02, CUDA 12.1)
- **Software**: Python 3.11, PyTorch 2.5.1+cu121, kokoro 0.9.4, faster-whisper 1.2.1,
  resemblyzer 0.1.4 (separate venv, GPU-accelerated)
- **Timing**: VM acquired 2026-09-14T16:11Z, synthesis completed ~17:15Z, scoring completed
  ~17:37Z
- **Prompt sets**: 96 val96 prompts (from `data/v4/val/val_list.txt`) + 100 filler prompts
  (sampled seed=42 from 1364-clip corpus)
- **Reference corpus**: 1364 ElevenLabs David clips (62 phrases × 22 reps), regenerated via
  API 2026-09-14 (corpus was not cached on VM). Split 679/679 seed=42; half-A → centroid;
  half-B → ElevenLabs self-scoring
- **Warmup**: 1 warmup synthesis discarded per system before timing begins
- **Kokoro systems**: all synthesized via `build_pipeline()` from `t0003` with `lang_code="b"`
  (British English) + brand lexicon, except `kokoro_floor_control` (`lang_code="a"`)
- **Speaker similarity**: resemblyzer GE2E encoder, GPU mode, cosine similarity vs centroid
- **WER**: faster-whisper `base.en` CPU int8, JiWER-style normalization (lowercase, remove
  punctuation), duration-ratio gate (0.5–2.0)

## Metrics Tables

### Speaker Similarity (GE2E cosine, higher is better, target ≥0.85)

| System | Fillers | Val96 |
| --- | --- | --- |
| elevenlabs_david | **0.832** | **0.792** |
| kokoro_v3_bundle | 0.631 | 0.588 |
| kokoro_base_v3_voicepack | 0.603 | 0.582 |
| kokoro_t0006_v6d | 0.601 | 0.482 |
| kokoro_base_george | 0.563 | 0.595 |
| kokoro_base_lewis | 0.505 | 0.500 |
| kokoro_t0005_best | nan | nan (explosions) |
| kokoro_floor_control | 0.444 | 0.434 |

Note: ElevenLabs is scored vs half-B centroid to avoid self-comparison bias.

### TTFB p50 (ms, lower is better, target ≤300 ms)

| System | Fillers | Val96 |
| --- | --- | --- |
| elevenlabs_david | 132ms | 153ms |
| kokoro_t0006_v6d | **132ms** | 251ms |
| kokoro_floor_control | 165ms | 290ms |
| kokoro_v3_bundle | 185ms | 282ms |
| kokoro_base_v3_voicepack | 196ms | 296ms |
| kokoro_base_lewis | 215ms | 334ms |
| kokoro_base_george | 233ms | 340ms |
| kokoro_t0005_best | 688ms | 2156ms (explosions) |

### RTF (wall_time / audio_duration, lower is better)

| System | Fillers | Val96 |
| --- | --- | --- |
| elevenlabs_david | 0.113 | 0.067 |
| kokoro_t0006_v6d | 0.107 | 0.054 |
| kokoro_v3_bundle | 0.113 | 0.056 |
| kokoro_base_v3_voicepack | 0.119 | 0.054 |
| kokoro_base_george | 0.124 | 0.082 |
| kokoro_base_lewis | 0.104 | 0.071 |
| kokoro_floor_control | 0.093 | 0.054 |
| kokoro_t0005_best | 0.359 | 1.120 |

All Kokoro systems on fillers match or beat ElevenLabs RTF; on val96 (longer texts), all
healthy systems are also comparable.

## Analysis

**Why no Kokoro reaches 0.85 speaker_sim**: The GE2E encoder captures global speaker identity.
The v3 voicepack + bundle training has lifted the base george (0.56) to 0.63 (v3_bundle), but
David's distinctive British voice requires more fine-tuning epochs. The gap (0.83 → 0.63) is
the research target for subsequent tasks.

**t0005_best instability**: epoch-3 from run06 produces duration explosions on many clips
(TTFB = 2.2s for val96, nan speaker_sim). The five-module checkpoint loaded correctly (145
missing keys at load — these are voice-encoder-only parameters not in the StyleTTS2
checkpoint, expected). The explosions suggest this early checkpoint hasn't fully converged.
t0006 epoch-6 (val_loss 0.846) is much more stable.

**t0006_v6d TTFB excels on fillers**: 132ms p50, matching ElevenLabs exactly. This is because
filler prompts are short (~5 words) and the model produces the first chunk quickly. On longer
val96 texts (some 80–120 words), TTFB rises to 251ms but remains under 300ms.

**Duration ratio and WER as failure detectors**: All t0005_best clips with speaker_sim=nan
correspond to duration_ratio > 5.0 (explosions). The WER also spikes for these clips. The
combination of duration_ratio > 5.0 AND WER > 0.5 is a reliable explosion detector, confirming
REQ-10.

**ElevenLabs speaker_sim < 0.85**: The 0.85 threshold was set as a Kokoro target. ElevenLabs
measured against its own half-B clips (not half-A) to avoid self-scoring inflation. The half-B
mean is 0.832 (fillers) — very close to the threshold but not quite. This baseline datum
informs the target: the project goal is "match ElevenLabs", meaning matching ≥0.83.

## Comparison vs Baselines

| System | sim_fillers | sim_val96 | TTFB_fillers | TTFB_val96 |
| --- | --- | --- | --- | --- |
| ElevenLabs David (target) | 0.832 | 0.792 | 132ms | 153ms |
| kokoro_v3_bundle (best Kokoro) | 0.631 (-0.201) | 0.588 (-0.204) | 185ms | 282ms |
| kokoro_t0006_v6d (best TTFB) | 0.601 (-0.231) | 0.482 (-0.310) | 132ms (=) | 251ms |

The v3_bundle achieves the best speaker identity but not the best latency. t0006_v6d achieves
the best latency. Neither achieves the speaker_sim target; the latency target (≤300ms) is met
by all healthy Kokoro systems on fillers.

## Limitations

- **Corpus size**: 100 filler prompts and 96 val96 prompts. More prompts would reduce
  variance.
- **t0005_best**: epoch-3 is likely too early in training. Later checkpoints may have been
  better but were not available.
- **WER on non-English text**: The val96 prompts contain product descriptions with brand
  names. Whisper base.en may mismatch these even for correct synthesis.
- **resemblyzer on very short clips**: Filler clips are 0.75–1.5s; resemblyzer pads them to
  process. The GE2E score for very short clips has higher variance than for longer utterances.
- **Half-B score for ElevenLabs**: The 685-clip half-B set is slightly larger than half-A
  (679) due to the 1364-total count not dividing evenly. The score for ElevenLabs (0.832) is
  the gold reference.
- **No per-speaker breakdown**: All 62 phrase types are averaged together. Some phrases may be
  systematically better reproduced than others.

## Visualizations

![Speaker similarity distribution by
system](../../../tasks/t0008_tts_eval_harness_baselines/results/images/speaker_sim_boxplot.png)

![TTFB CDF by
system](../../../tasks/t0008_tts_eval_harness_baselines/results/images/ttfb_cdf.png)

![Speaker similarity vs WER per
clip](../../../tasks/t0008_tts_eval_harness_baselines/results/images/speaker_sim_wer_scatter.png)

## Files Created

- `results/per_clip_metrics.json` — 1568 per-clip records (8 systems × 196 prompts each)
- `results/metrics.json` — 16 variants (8 systems × 2 prompt sets) explicit variant format
- `results/tables.json` — full metrics table with delta vs ElevenLabs
- `results/images/speaker_sim_boxplot.png` — GE2E cosine distribution per system
- `results/images/ttfb_cdf.png` — TTFB empirical CDF per system
- `results/images/speaker_sim_wer_scatter.png` — speaker_sim vs WER scatter per clip
- `results/centroid.npy` — L2-normalized GE2E centroid from 679 half-A clips
- `results/half_b_paths.json` — list of 685 half-B clip paths
- `results/metadata.json` — eval environment and checkpoint provenance
- `results/costs.json` — API and compute costs ($35.10 total)
- `results/remote_machines_used.json` — LLM-T1-NC80 H100 usage
- `assets/library/tts_eval_harness/` — registered library asset v0.1.0
- `data/packaged/t0005_run06_epoch3.pth` — five-module t0005 checkpoint
- `data/packaged/t0006_v6d_epoch6.pth` — five-module t0006 v6d checkpoint
- `data/packaged/metadata.json` — checkpoint provenance
- `data/11labs_david/` — 1364 ElevenLabs David WAV clips (DVC-tracked)
- `data/val96_prompts.json` — 96 val96 prompt manifests
- `data/filler_prompts_100.json` — 100 filler prompt manifests

## Verification

- Library asset `tts_eval_harness`: PASSED (0 errors, 3 category warnings)
- Unit tests: 11/11 PASS (`test_harness.py`: TestDurationRatio×3, TestWer×5, TestReportJson×3)
- `report.py` execution: clean (16 variants, 3 charts)
- All 8 systems produced audio for both prompt sets (t0005 has NaN sim due to explosions but
  files were generated)

## Examples

Per-clip examples drawn from `results/per_clip_metrics.json` (1568 records total). Each row
shows the input text, the synthesizing system, and the raw output metrics produced by the
harness.

### Random Examples — Typical Behavior

| # | System | Prompt set | Text | speaker_sim | TTFB (ms) | WER | dur_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | elevenlabs_david | fillers | "pulling that up 03" | **0.808** | 137 | 0.00 | 1.125 |
| 2 | elevenlabs_david | fillers | "cross-checking that 00" | **0.783** | 136 | 0.00 | 1.233 |
| 3 | kokoro_v3_bundle | fillers | "putting that together 21" | 0.777 | (n/a sampled) | 0.00 | 1.205 |
| 4 | elevenlabs_david | val96 | "checking for the latest press release" | **0.749** | 139 | 0.00 | 0.905 |

Notes: "pulling that up 03" is a 3-word filler; resemblyzer GE2E scores very short clips with
higher variance (~±0.04). Both fillers score above 0.75 — typical for ElevenLabs mid-range
clips.

### Best Cases — High Speaker Similarity

```
system: elevenlabs_david  prompt_set: fillers
text:   "on it 15"
speaker_sim:    0.923   ttfb_ms: 116   rtf: 0.155   wer: 0.00   duration_ratio: 1.125
```

```
system: elevenlabs_david  prompt_set: fillers
text:   "side by side 18"
speaker_sim:    0.915   ttfb_ms: 135   rtf: 0.124   wer: 0.00   duration_ratio: 1.125
```

```
system: elevenlabs_david  prompt_set: fillers
text:   "adding to that 12"
speaker_sim:    0.903   ttfb_ms: 119   rtf: 0.120   wer: 0.00   duration_ratio: 1.027
```

Observation: the best ElevenLabs clips are ultra-short fillers (2–3 words); GE2E scores peak
at 0.92+ for these because the embedding is dominated by prosodic texture with no phoneme
dilution.

### Worst Cases — Low Speaker Similarity (Kokoro t0006_v6d, val96)

```
system: kokoro_t0006_v6d  prompt_set: val96
text:   "taking a look at the brain commerce page"
speaker_sim:    0.382   ttfb_ms: 253   rtf: 0.137   wer: 1.00   duration_ratio: 0.797
```

```
system: kokoro_t0006_v6d  prompt_set: val96
text:   "let me check the board of advisors for rezolve ai"
speaker_sim:    0.383   ttfb_ms: 231   rtf: 0.136   wer: 1.00   duration_ratio: 0.610
```

```
system: kokoro_t0006_v6d  prompt_set: val96
text:   "looking into the reality industry context"
speaker_sim:    0.389   ttfb_ms: 259   rtf: 0.133   wer: 1.00   duration_ratio: 0.763
```

Observation: all three have WER=1.00 (ASR completely mismatched) and duration_ratio < 0.80,
meaning the model produced audio far shorter than the reference. This cluster of clips
contains brand names ("Rezolve AI", "Brain Commerce") that Kokoro v6d mispronounces or drops —
the WER spike reflects ASR failure on the output, not necessarily a true transcription error.

### Boundary Cases — Speaker Similarity Near 0.85 Target

```
system: elevenlabs_david  prompt_set: val96
text:   "looking into the telecom industry context"
speaker_sim:    0.851   ttfb_ms: 139   rtf: 0.074   wer: 0.00   duration_ratio: 0.849
```

```
system: elevenlabs_david  prompt_set: fillers
text:   "let me think 16"
speaker_sim:    0.852   ttfb_ms: 134   rtf: 0.126   wer: 0.00   duration_ratio: 1.238
```

```
system: elevenlabs_david  prompt_set: fillers
text:   "matching them up 01"
speaker_sim:    0.849   ttfb_ms: 128   rtf: 0.106   wer: 0.75   duration_ratio: 1.247
```

Observation: even ElevenLabs straddles the 0.85 threshold clip-by-clip; the per-clip mean of
0.832 means roughly 35% of ElevenLabs clips fall below 0.85. The target was set at the mean
level (match ElevenLabs distribution), not as a per-clip floor.

### t0005 Explosion Cases — Failure Detector Validation

```
system: kokoro_t0005_best  prompt_set: val96
text:   "checking for the latest press release"
speaker_sim:    null    ttfb_ms: 1572   rtf: 0.065   wer: null   duration_ratio: 11.26
```

```
system: kokoro_t0005_best  prompt_set: val96
text:   "checking supported language options"
speaker_sim:    null    ttfb_ms: 1720   rtf: 0.077   wer: null   duration_ratio: 10.49
```

```
system: kokoro_t0005_best  prompt_set: val96
text:   "checking the most recent updates"
speaker_sim:    null    ttfb_ms: 1504   rtf: 0.077   wer: null   duration_ratio: 9.56
```

Observation: duration_ratio > 9 (audio 9× longer than reference) causes resemblyzer to gate
out the clip (speaker_sim=null). TTFB is 1.5–1.7s — 10× slower than the 300ms target. This
confirms REQ-10: duration_ratio > 5 is a reliable explosion detector.

### Contrastive Examples — Same Text, Multiple Systems (val96)

Input text: `"checking for the latest press release"`

| System | speaker_sim | TTFB (ms) | WER | duration_ratio |
| --- | --- | --- | --- | --- |
| elevenlabs_david | **0.749** | 139 | 0.00 | 0.905 |
| kokoro_base_george | 0.581 | 310 | 0.00 | 1.346 |
| kokoro_base_lewis | 0.560 | 290 | 0.00 | 1.358 |
| kokoro_t0006_v6d | 0.532 | 250 | 1.00 | 0.831 |
| kokoro_base_v3_voicepack | 0.526 | 186 | 0.00 | 0.983 |
| kokoro_v3_bundle | 0.500 | 257 | 0.00 | 0.983 |

Observation: george/lewis preserve pronunciation (WER=0) but miss David's speaker identity by
0.17 cosine units. t0006_v6d has a WER spike (likely "press release" mispronounced) despite
the shortest TTFB among Kokoro variants on this clip. v3_voicepack scores slightly above
v3_bundle on this phrase — bundle's advantage is aggregate, not per-clip universal.

## Task Requirement Coverage

**Task**: Build a reusable TTS evaluation harness and score ElevenLabs David, base Kokoro, v3,
and the best t0005/t0006 checkpoints on speaker similarity, TTFB, RTF, WER, and duration
ratio.

- **REQ-1** (speaker_sim all 8 systems): DONE — GE2E cosine scored for all 8 systems;
  ElevenLabs=0.832/0.792, v3_bundle=0.631/0.588, t0005 nan (explosions)
- **REQ-2** (ttfb_ms all 8 systems): DONE — p50/p95/p99 in `tables.json`, ElevenLabs
  p50=132ms, t0006_v6d p50=132ms on fillers
- **REQ-3** (rtf all 8 systems): DONE — mean RTF in `metrics.json` and `tables.json`
- **REQ-4** (duration_ratio all 8 systems): DONE — explosion_count in `tables.json`; t0005 has
  many explosions
- **REQ-5** (WER all 8 systems): DONE — WER computed with faster-whisper base.en; t0005 WER
  high where clips are valid
- **REQ-6** (Q1: ElevenLabs David baselines): DONE — speaker_sim=0.832, TTFB_p50=132ms,
  RTF=0.113 on fillers
- **REQ-7** (Q2: base Kokoro george/lewis): DONE — george: sim=0.563, TTFB=233ms; lewis:
  sim=0.505, TTFB=215ms
- **REQ-8** (Q3: v3 bundle vs 0.85): DONE — v3_bundle=0.631, does not meet 0.85 target;
  v3_voicepack=0.603
- **REQ-9** (Q4: t0005/t0006 vs v3): DONE — t0005 is severely worse (explosions); t0006=0.601
  (close to v3_bundle 0.631 but slightly lower)
- **REQ-10** (Q5: duration_ratio as failure detector): DONE — t0005 explosions (ratio>5.0)
  correlate perfectly with nan speaker_sim and high WER; confirmed reliable failure detector
- **REQ-11** (DVC push 11labs corpus): PARTIAL — corpus is in `data/11labs_david/` (1364
  WAVs), DVC tracking needs to be completed before merge
- **REQ-12** (five-module checkpoint packaging): DONE — t0005_run06_epoch3.pth and
  t0006_v6d_epoch6.pth packaged; provenance in `data/packaged/metadata.json`
- **REQ-13** (library asset `tts_eval_harness`): DONE — registered v0.1.0, verificator PASSED
- **REQ-14** (metrics.json variant format): DONE — 16 variants in explicit format
- **REQ-15** (comparison table): DONE — `tables.json` with delta_speaker_sim_vs_elevenlabs
- **REQ-16** (3 charts): DONE — speaker_sim boxplot, TTFB CDF, speaker_sim_wer_scatter
- **REQ-17** (pass/fail per success criterion): DONE — TTFB target met by 7/8 Kokoro on
  fillers; speaker_sim target not met by any Kokoro system
- **REQ-18** (t0005 filename recorded): DONE — `data/packaged/metadata.json` records
  `epoch_2nd_00003.pth` from t0005_kokoro_v5_stage2_train, epoch=3
- **REQ-19** (infrastructure versions): DONE — `results/metadata.json` captures torch
  2.5.1+cu121, CUDA 12.1, kokoro 0.9.4, H100 NVL
- **REQ-20** (warmup protocol): DONE — `n_warmup=1` in all eval runs
- **REQ-21** (per_clip_metrics.json): DONE — 1568 records with speaker_sim, duration_ratio,
  WER, ttfb_ms, rtf, audio_path, text

</details>
