# Project Dashboard

<p align="center">
  <a href="papers/"><img src="https://img.shields.io/badge/Papers-0-4169E1" alt="Papers"></a>
  <a href="datasets/"><img src="https://img.shields.io/badge/Datasets-0-2E8B57" alt="Datasets"></a>
  <a href="models/"><img src="https://img.shields.io/badge/Models-0-FF8C00" alt="Models"></a>
  <a href="predictions/"><img src="https://img.shields.io/badge/Predictions-0-9370DB" alt="Predictions"></a>
  <a href="libraries/"><img src="https://img.shields.io/badge/Libraries-1-20B2AA" alt="Libraries"></a>
  <a href="answers/"><img src="https://img.shields.io/badge/Answers-0-CD853F" alt="Answers"></a>
</p>

<p align="center">
  <a href="news/"><img src="https://img.shields.io/badge/News-0-FF6347" alt="News"></a>
  <a href="tasks/"><img src="https://img.shields.io/badge/Tasks-9-4682B4" alt="Tasks"></a>
  <a href="suggestions/"><img src="https://img.shields.io/badge/Suggestions-5-DAA520" alt="Suggestions"></a>
  <a href="llm-context/"><img src="https://img.shields.io/badge/LLM%20Contexts-8-8B4513" alt="LLM Contexts"></a>
  <a href="metrics/"><img src="https://img.shields.io/badge/Metrics-3-708090" alt="Metrics"></a>
  <a href="metrics-results/"><img src="https://img.shields.io/badge/Results-6-DC143C" alt="Results"></a>
  <a href="task-types/"><img src="https://img.shields.io/badge/Task%20Types-19-708090" alt="Task%20Types"></a>
</p>

🏷️ **Categories**:

**[LLM Contexts](llm-context/README.md)**: [overview](llm-context/project-overview.xml) (3K) |
[full](llm-context/full.xml) (17K) | [roadmap](llm-context/roadmap.xml) (7K) |
[results](llm-context/results-deep-dive.xml) (14K) |
[assets](llm-context/literature-and-assets.xml) (4K)

*Last updated: 2026-09-14 18:11 UTC*

* **Budget**: **$35** spent of $5000
* **Remaining**: **$4965**
* **Usage**: `░░░░░░░░░░░░░░░░░░░░` 0.7%
* **GPU Machines**: **1** provisioned across 1 tasks · **$28** GPU spend
  ([details](machines/))

---

## [Daily News (0)](news/)

No daily news yet.

---

## [In Progress (1)](tasks/by-status/in_progress.md)

| # | Task | Started |
|---|------|---------|
| 0009 | [Stage 2 training failure forensics and safeguards](../overview/tasks/task_pages/t0009_stage2_training_failure_forensics.md) | 2026-09-14 15:28 |

---

## [Ready to Start (0)](tasks/by-status/not_started.md)

No tasks ready to start.

---

## [Blocked Tasks (0)](tasks/)

No blocked tasks.

---

## [Recently Completed (8 total)](tasks/by-status/completed.md)

| # | Task | Results | Completed |
|---|------|---------|-----------|
| 0008 | [TTS evaluation harness and baselines](../overview/tasks/task_pages/t0008_tts_eval_harness_baselines.md) | [`results`](../tasks/t0008_tts_eval_harness_baselines/results/results_detailed.md) | 2026-09-14 18:07 |
| 0007 | [Brainstorm results session 1](../overview/tasks/task_pages/t0007_brainstorm_results_1.md) | [`results`](../tasks/t0007_brainstorm_results_1/results/results_detailed.md) | 2026-09-14 14:30 |
| 0005 | [Kokoro v5 Stage 2 fine-tune](../overview/tasks/task_pages/t0005_kokoro_v5_stage2_train.md) | [`results`](../tasks/t0005_kokoro_v5_stage2_train/results/results_detailed.md) | 2026-09-13 06:00 |
| 0006 | [Kokoro v5 Stage 2: 250-clip subset, multispeaker: false](../overview/tasks/task_pages/t0006_kokoro_v5_stage2_subset.md) | [`results`](../tasks/t0006_kokoro_v5_stage2_subset/results/results_detailed.md) | 2026-09-13 00:00 |
| 0004 | [Kokoro v5 Stage 1 fine-tune](../overview/tasks/task_pages/t0004_kokoro_v5_stage1_train.md) | [`results`](../tasks/t0004_kokoro_v5_stage1_train/results/results_detailed.md) | 2026-09-12 23:00 |
| 0003 | [Regenerate v5 phoneme manifests](../overview/tasks/task_pages/t0003_kokoro_v5_phoneme_data.md) | [`results`](../tasks/t0003_kokoro_v5_phoneme_data/results/results_detailed.md) | 2026-09-12 14:00 |
| 0002 | [Package Kokoro v4 voicepack + decoder](../overview/tasks/task_pages/t0002_kokoro_v4_voicepack_decoder_package.md) | [`results`](../tasks/t0002_kokoro_v4_voicepack_decoder_package/results/results_detailed.md) | 2026-09-12 10:00 |
| 0001 | [Kokoro v4 Stage 2 fine-tune](../overview/tasks/task_pages/t0001_kokoro_v4_stage2_finetune.md) | [`results`](../tasks/t0001_kokoro_v4_stage2_finetune/results/results_detailed.md) | 2026-09-11 19:00 |

---

## [Recent Suggestions (5 open)](suggestions/)

<details>
<summary>🧪 <strong>Continue Stage 2 fine-tuning from v6d epoch 6 to close the 0.20
GE2E cosine gap</strong> (S-0008-01)</summary>

**Kind**: experiment | **Priority**: high | **Date**: 2026-09-14 | **Source**:
[t0008_tts_eval_harness_baselines](../tasks/t0008_tts_eval_harness_baselines/)

t0008 measured the best Kokoro checkpoint (kokoro_v3_bundle) at speaker_sim=0.631 vs the
ElevenLabs target of 0.83 — a gap of ~0.20 GE2E cosine units. The t0006_v6d checkpoint was
trained for only 6 epochs on a 250-clip subset and already shows competitive TTFB (132ms,
matching ElevenLabs). Run additional Stage 2 epochs from epoch_2nd_00006.pth
(multispeaker:false, 250-clip subset) and evaluate each checkpoint with the tts_eval_harness
to find the epoch where speaker_sim plateaus or reaches 0.83. Use the harness library
registered in t0008. Recommended task types: tts-finetuning-eval.

</details>

<details>
<summary>🧪 <strong>Investigate t0006_v6d val96 speaker_sim collapse (0.601 fillers
→ 0.482 val96)</strong> (S-0008-02)</summary>

**Kind**: experiment | **Priority**: high | **Date**: 2026-09-14 | **Source**:
[t0008_tts_eval_harness_baselines](../tasks/t0008_tts_eval_harness_baselines/)

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
<summary>🧪 <strong>Expand Stage 2 training corpus beyond 250 clips to improve
speaker identity transfer</strong> (S-0008-03)</summary>

**Kind**: experiment | **Priority**: high | **Date**: 2026-09-14 | **Source**:
[t0008_tts_eval_harness_baselines](../tasks/t0008_tts_eval_harness_baselines/)

The v6d model was trained on a 250-clip subset to test whether a smaller corpus would avoid
the divergence seen in t0005. It achieves TTFB parity with ElevenLabs but lags in speaker_sim
(0.601 vs 0.83 target). The full training set contains 1557 clips. Run Stage 2 with the full
corpus (or a larger 500–800 clip subset) from first_stage.pth with the same multispeaker:false
and crash-fix settings from t0006, then evaluate with tts_eval_harness. This tests whether
more speaker data closes the GE2E cosine gap. Recommended task types: tts-finetuning-eval.

</details>

<details>
<summary>📊 <strong>Register WER as a project metric and track it in harness
evaluations</strong> (S-0008-04)</summary>

**Kind**: evaluation | **Priority**: medium | **Date**: 2026-09-14 | **Source**:
[t0008_tts_eval_harness_baselines](../tasks/t0008_tts_eval_harness_baselines/)

t0008 computed WER with faster-whisper base.en as a sanity metric but did not register it as a
project metric. WER proved critical as a secondary failure detector alongside duration_ratio —
the combination of duration_ratio > 5.0 AND WER > 0.5 reliably identifies exploded clips.
Registering WER in meta/metrics/ would let the aggregator track intelligibility trends across
future fine-tuning tasks and alert when a new checkpoint degrades transcription quality.
Recommended task types: infrastructure-setup.

</details>

<details>
<summary>📂 <strong>Augment the ElevenLabs filler corpus with val96-style longer
utterances for harness scoring</strong> (S-0008-05)</summary>

**Kind**: dataset | **Priority**: medium | **Date**: 2026-09-14 | **Source**:
[t0008_tts_eval_harness_baselines](../tasks/t0008_tts_eval_harness_baselines/)

The current evaluation uses 100 filler prompts (short, 2–8 words) and 96 val96 prompts (mixed
length). The speaker_sim scores diverge significantly between the two sets for fine-tuned
Kokoro systems, suggesting filler-length bias. Adding a third prompt set of 50–100
medium-length utterances (10–20 words) sampled from the training corpus texts — but not in
val96 — would give a more complete picture of speaker similarity across utterance lengths and
reduce the chance that a checkpoint overfits to short fillers. Recommended task types:
dataset.

</details>

---

## [High Priority Suggestions (3)](suggestions/)

<details>
<summary>🧪 <strong>Continue Stage 2 fine-tuning from v6d epoch 6 to close the 0.20
GE2E cosine gap</strong> (S-0008-01)</summary>

**Kind**: experiment | **Priority**: high | **Date**: 2026-09-14 | **Source**:
[t0008_tts_eval_harness_baselines](../tasks/t0008_tts_eval_harness_baselines/)

t0008 measured the best Kokoro checkpoint (kokoro_v3_bundle) at speaker_sim=0.631 vs the
ElevenLabs target of 0.83 — a gap of ~0.20 GE2E cosine units. The t0006_v6d checkpoint was
trained for only 6 epochs on a 250-clip subset and already shows competitive TTFB (132ms,
matching ElevenLabs). Run additional Stage 2 epochs from epoch_2nd_00006.pth
(multispeaker:false, 250-clip subset) and evaluate each checkpoint with the tts_eval_harness
to find the epoch where speaker_sim plateaus or reaches 0.83. Use the harness library
registered in t0008. Recommended task types: tts-finetuning-eval.

</details>

<details>
<summary>🧪 <strong>Investigate t0006_v6d val96 speaker_sim collapse (0.601 fillers
→ 0.482 val96)</strong> (S-0008-02)</summary>

**Kind**: experiment | **Priority**: high | **Date**: 2026-09-14 | **Source**:
[t0008_tts_eval_harness_baselines](../tasks/t0008_tts_eval_harness_baselines/)

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
<summary>🧪 <strong>Expand Stage 2 training corpus beyond 250 clips to improve
speaker identity transfer</strong> (S-0008-03)</summary>

**Kind**: experiment | **Priority**: high | **Date**: 2026-09-14 | **Source**:
[t0008_tts_eval_harness_baselines](../tasks/t0008_tts_eval_harness_baselines/)

The v6d model was trained on a 250-clip subset to test whether a smaller corpus would avoid
the divergence seen in t0005. It achieves TTFB parity with ElevenLabs but lags in speaker_sim
(0.601 vs 0.83 target). The full training set contains 1557 clips. Run Stage 2 with the full
corpus (or a larger 500–800 clip subset) from first_stage.pth with the same multispeaker:false
and crash-fix settings from t0006, then evaluate with tts_eval_harness. This tests whether
more speaker data closes the GE2E cosine gap. Recommended task types: tts-finetuning-eval.

</details>

---

## [Recent Answers (0 total)](answers/)

No answers yet.

---

## [Latest Papers (0 total)](papers/)

No papers yet.

---

## [Latest Datasets (0 total)](datasets/)

No datasets yet.

---

## [Latest Models (0 total)](models/)

No models yet.

---

## [Latest Predictions (0 total)](predictions/)

No predictions yet.

---

## [Latest Libraries (1 total)](libraries/)

| Name | Source | Created |
|------|--------|---------|
| [TTS Evaluation Harness](../tasks/t0008_tts_eval_harness_baselines/assets/library/tts_eval_harness/description.md) | [8](../overview/tasks/task_pages/t0008_tts_eval_harness_baselines.md) | 2026-09-14 |

---

## [Cost Leaders (1 tasks with spend)](costs/)

| Task | Cost | Date |
|------|------|------|
| [TTS evaluation harness and baselines](../overview/tasks/task_pages/t0008_tts_eval_harness_baselines.md) | [`$35.10`](../tasks/t0008_tts_eval_harness_baselines/results/costs.json) | 2026-09-14 18:07 |
