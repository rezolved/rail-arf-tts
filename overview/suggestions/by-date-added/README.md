# Suggestions by Date Added

5 suggestion(s) grouped by derived added date.

[Back to all suggestions](../README.md)

---

## 2026-09-14 (5)

## High Priority

<details>
<summary>🧪 <strong>Continue Stage 2 fine-tuning from v6d epoch 6 to close the 0.20
GE2E cosine gap</strong> (S-0008-01)</summary>

| Field | Value |
|---|---|
| **ID** | `S-0008-01` |
| **Kind** | experiment |
| **Date added** | 2026-09-14 |
| **Source task** | [`t0008_tts_eval_harness_baselines`](../../../overview/tasks/task_pages/t0008_tts_eval_harness_baselines.md) |
| **Source paper** | — |
| **Categories** | — |

t0008 measured the best Kokoro checkpoint (kokoro_v3_bundle) at speaker_sim=0.631 vs the
ElevenLabs target of 0.83 — a gap of ~0.20 GE2E cosine units. The t0006_v6d checkpoint was
trained for only 6 epochs on a 250-clip subset and already shows competitive TTFB (132ms,
matching ElevenLabs). Run additional Stage 2 epochs from epoch_2nd_00006.pth
(multispeaker:false, 250-clip subset) and evaluate each checkpoint with the tts_eval_harness
to find the epoch where speaker_sim plateaus or reaches 0.83. Use the harness library
registered in t0008. Recommended task types: tts-finetuning-eval.

</details>

<details>
<summary>🧪 <strong>Expand Stage 2 training corpus beyond 250 clips to improve
speaker identity transfer</strong> (S-0008-03)</summary>

| Field | Value |
|---|---|
| **ID** | `S-0008-03` |
| **Kind** | experiment |
| **Date added** | 2026-09-14 |
| **Source task** | [`t0008_tts_eval_harness_baselines`](../../../overview/tasks/task_pages/t0008_tts_eval_harness_baselines.md) |
| **Source paper** | — |
| **Categories** | — |

The v6d model was trained on a 250-clip subset to test whether a smaller corpus would avoid
the divergence seen in t0005. It achieves TTFB parity with ElevenLabs but lags in speaker_sim
(0.601 vs 0.83 target). The full training set contains 1557 clips. Run Stage 2 with the full
corpus (or a larger 500–800 clip subset) from first_stage.pth with the same multispeaker:false
and crash-fix settings from t0006, then evaluate with tts_eval_harness. This tests whether
more speaker data closes the GE2E cosine gap. Recommended task types: tts-finetuning-eval.

</details>

<details>
<summary>🧪 <strong>Investigate t0006_v6d val96 speaker_sim collapse (0.601 fillers
→ 0.482 val96)</strong> (S-0008-02)</summary>

| Field | Value |
|---|---|
| **ID** | `S-0008-02` |
| **Kind** | experiment |
| **Date added** | 2026-09-14 |
| **Source task** | [`t0008_tts_eval_harness_baselines`](../../../overview/tasks/task_pages/t0008_tts_eval_harness_baselines.md) |
| **Source paper** | — |
| **Categories** | — |

The t0006_v6d checkpoint shows a striking degradation on val96 texts: speaker_sim drops from
0.601 (fillers) to 0.482 (val96), the largest cross-set gap of any system. The worst-case
clips all have WER=1.00 and duration_ratio < 0.80, with prompts containing brand names
(Rezolve AI, Brain Commerce). This suggests v6d has not learned to synthesize longer,
brand-name-heavy utterances correctly. Investigate whether the brand lexicon in
build_pipeline.py fully covers the val96 prompts, whether val96 texts are systematically
out-of-distribution for the 250-clip training subset, and whether more training data or a
broader lexicon would recover performance. Recommended task types: data-analysis.

</details>

## Medium Priority

<details>
<summary>📂 <strong>Augment the ElevenLabs filler corpus with val96-style longer
utterances for harness scoring</strong> (S-0008-05)</summary>

| Field | Value |
|---|---|
| **ID** | `S-0008-05` |
| **Kind** | dataset |
| **Date added** | 2026-09-14 |
| **Source task** | [`t0008_tts_eval_harness_baselines`](../../../overview/tasks/task_pages/t0008_tts_eval_harness_baselines.md) |
| **Source paper** | — |
| **Categories** | — |

The current evaluation uses 100 filler prompts (short, 2–8 words) and 96 val96 prompts (mixed
length). The speaker_sim scores diverge significantly between the two sets for fine-tuned
Kokoro systems, suggesting filler-length bias. Adding a third prompt set of 50–100
medium-length utterances (10–20 words) sampled from the training corpus texts — but not in
val96 — would give a more complete picture of speaker similarity across utterance lengths and
reduce the chance that a checkpoint overfits to short fillers. Recommended task types:
dataset.

</details>

<details>
<summary>📊 <strong>Register WER as a project metric and track it in harness
evaluations</strong> (S-0008-04)</summary>

| Field | Value |
|---|---|
| **ID** | `S-0008-04` |
| **Kind** | evaluation |
| **Date added** | 2026-09-14 |
| **Source task** | [`t0008_tts_eval_harness_baselines`](../../../overview/tasks/task_pages/t0008_tts_eval_harness_baselines.md) |
| **Source paper** | — |
| **Categories** | — |

t0008 computed WER with faster-whisper base.en as a sanity metric but did not register it as a
project metric. WER proved critical as a secondary failure detector alongside duration_ratio —
the combination of duration_ratio > 5.0 AND WER > 0.5 reliably identifies exploded clips.
Registering WER in meta/metrics/ would let the aggregator track intelligibility trends across
future fine-tuning tasks and alert when a new checkpoint degrades transcription quality.
Recommended task types: infrastructure-setup.

</details>
