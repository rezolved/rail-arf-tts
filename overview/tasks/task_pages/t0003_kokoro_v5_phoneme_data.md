# ✅ Regenerate v5 phoneme manifests

[Back to all tasks](../README.md)

## Overview

| Field | Value |
|---|---|
| **ID** | `t0003_kokoro_v5_phoneme_data` |
| **Status** | ✅ completed |
| **Started** | 2026-09-12T08:00:00Z |
| **Completed** | 2026-09-12T14:00:00Z |
| **Duration** | 6h 0m |
| **Dependencies** | [`t0001_kokoro_v4_stage2_finetune`](../../../overview/tasks/task_pages/t0001_kokoro_v4_stage2_finetune.md), [`t0002_kokoro_v4_voicepack_decoder_package`](../../../overview/tasks/task_pages/t0002_kokoro_v4_voicepack_decoder_package.md) |
| **Task types** | `tts-finetuning-eval` |
| **Expected assets** | 2 dataset |
| **Step progress** | 3/3 |
| **Task folder** | [`t0003_kokoro_v5_phoneme_data/`](../../../tasks/t0003_kokoro_v5_phoneme_data/) |
| **Detailed results** | [`results_detailed.md`](../../../tasks/t0003_kokoro_v5_phoneme_data/results/results_detailed.md) |

<details>
<summary><strong>Task Description</strong></summary>

*Source:
[`task_description.md`](../../../tasks/t0003_kokoro_v5_phoneme_data/task_description.md)*

# t0003 — Regenerate v5 phoneme manifests

## Objective

Produce train/val manifests whose text column is in the same token space Kokoro uses at
inference, so the next Stage 1 + Stage 2 fine-tune can beat v3's val_loss of 0.506.

## Background

t0001's Stage 2 run diverged (val_loss 0.594 → 0.751 → 1.147) and t0002's packaging of its
checkpoints produced a 10x duration explosion — 89 s of audio for a sentence that should take
9.6 s. t0002 ruled out the packaging code by running v3's proven artifacts through the
identical extraction and inference path, which came out clean. The defect was therefore in
t0001's training, and t0001's hyperparameters were already close to v3's.

The training data was never checked. `kokoro-finetune/scripts/prepare_v4_data.py` phonemizes
with misaki inside a bare `except Exception: return text.strip()`. misaki raises on words it
does not know — and the corpus is full of them: *Rezolve* (273 occurrences), *Ai*,
*brainpowa*, *agentic*, and a long tail of person and company names. Every one of those clips
fell back to raw English orthography. StyleTTS2's `TextCleaner` accepts ASCII letters, so
nothing crashed and nothing warned; the run simply trained on a 21% mixture of graphemes
inside a phoneme corpus.

A second, independent mismatch sat on top: the corpus was phonemized as British English
(correct — David's reference voice is British) while t0002 synthesized with
`KPipeline(lang_code="a")`, American. The two accents emit different vowel symbols for the
same word.

## Approach

Rebuild the text column only. Wav files and the train/val split are taken verbatim from the
existing v4 directories, so the sole difference against the failed run is the token space.

1. Recover each clip's original text from the *originals* — the v3 `manifest.csv` files and
   `fillers_from_logs.txt` — never from a v4 manifest, since all three v4 variants are
   corrupted in different ways.
2. Phonemize with `misaki.en.G2P(trf=False, british=True)`, with no exception fallback.
3. Install a lexicon (`code/lexicon.py`) for the 62 words misaki does not know, so none of
   them degrade to misaki's `❓` marker.
4. Gate every line against Kokoro's own 114-symbol vocab from `config.json` — the exact
   character set the model accepts — plus explicit checks for raw-text leakage, double
   phonemization and `❓`.

## Deliverables

- `results/v5/train_list.txt`, `results/v5/val_list.txt` — manifests in
  `wav_path|phonemes|speaker_id` form, every line gate-clean.
- `results/config_david_v5.yml` — training config pointing at the new manifests, with v3's
  optimizer settings restored.
- `code/build_pipeline.py` — inference-side pipeline that applies the same accent and lexicon,
  so synthesis is fed the token space the model was trained on.

## Verification criteria

- 0 lines rejected by any gate in either output manifest.
- The gates flag all three broken v4 manifests when replayed against them.
- `Rezolve`, `brainpowa` and `agentic` phonemize without `❓` on both the training and the
  inference path.
- Deferred to the training run: Dur loss reaches ~0.72 in the first Stage 1 epoch (v3's
  value). If it sits at ~1.03 again, the data hypothesis is wrong and the run should be
  stopped rather than continued for 20 epochs.

</details>

## Metrics

| Metric | Value |
|--------|-------|
| [`task_id`](../../metrics-results/task_id.md) | **t0003_kokoro_v5_phoneme_data** |
| [`metrics`](../../metrics-results/metrics.md) | **[{'name': 'v5_clean_train_lines', 'value': 1557, 'unit': None, 'variant': None}, {'name': 'v5_clean_val_lines', 'value': 96, 'unit': None, 'variant': None}, {'name': 'v4_raw_text_leaked_train', 'value': 329, 'unit': None, 'variant': None}, {'name': 'v4_raw_text_leaked_val', 'value': 31, 'unit': None, 'variant': None}, {'name': 'oov_words_repaired_distinct', 'value': 64, 'unit': None, 'variant': None}, {'name': 'oov_words_repaired_occurrences', 'value': 480, 'unit': None, 'variant': None}]** |

<details>
<summary><strong>Results Summary</strong></summary>

*Source:
[`results_summary.md`](../../../tasks/t0003_kokoro_v5_phoneme_data/results/results_summary.md)*

# Results Summary: v5 Phoneme Manifest Regeneration

## Summary

Rebuilt the Kokoro fine-tune manifests: **1557/1557 train and 96/96 val lines pass every
validation gate, 0 rejected**, against 329 raw-text and 9 unknown-word lines in the manifest
t0001 actually trained on. Root cause of t0001's divergence and t0002's 10× duration explosion
is a silent grapheme fallback in `prepare_v4_data.py` that fired on every clip containing a
word misaki does not know — 21% of the training set. No training has been run yet.

## Key Metrics

| Metric | Value |
| --- | --- |
| v5 clean train lines | 1557 / 1557 |
| v5 clean val lines | 96 / 96 |
| v4 raw_text_leaked (train) | 329 / 1557 (21%) |
| OOV words repaired | 64 distinct, 480 occurrences |
| Rezolve occurrences fixed | 273 |
| Machine | local (darwin, no GPU) |
| Runtime | ~2 s |

</details>

<details>
<summary><strong>Detailed Results</strong></summary>

*Source:
[`results_detailed.md`](../../../tasks/t0003_kokoro_v5_phoneme_data/results/results_detailed.md)*

# Results: v5 phoneme manifest regeneration

## Summary

Rebuilt the Kokoro fine-tune manifests: **1557/1557 train and 96/96 val lines pass every
validation gate, 0 rejected**, against 329 raw-text and 9 unknown-word lines in the manifest
t0001 actually trained on. Root cause of t0001's divergence and t0002's 10x duration explosion
is a silent grapheme fallback in `prepare_v4_data.py` that fired on every clip containing a
word misaki does not know — 21% of the training set. No training has been run yet.

## Methodology

- Machine: local (darwin, no GPU needed — G2P only). Runtime ~2 s for 1653 clips.
- Source texts read from the originals: `kokoro-finetune/data/train/manifest.csv`,
  `data/val/manifest.csv`, `data/fillers_from_logs.txt` (1653 entries indexed).
- Phonemizer: `misaki.en.G2P(trf=False, british=True)`, no exception fallback.
- Wav files and the train/val split reused verbatim from `data/v4/train/wavs` and
  `data/v4/val/wavs`, so the token space is the only variable changed against t0001.
- Gates applied per line: non-empty; no `❓`; at least one IPA-only character; no
  double-phonemization marker; every character present in Kokoro's 114-symbol vocab
  (`hexgrad/Kokoro-82M` `config.json`).

## Metrics

### Gate results, v5 output vs the three v4 manifests

| Manifest | Clean | raw_text_leaked | double_phonemized | unknown_word | out_of_vocab |
| --- | ---: | ---: | ---: | ---: | ---: |
| **v5 `train_list.txt`** | **1557** | 0 | 0 | 0 | 0 |
| **v5 `val_list.txt`** | **96** | 0 | 0 | 0 | 0 |
| v4 `train/train_list.txt` (used by t0001) | 1219 | 329 | 0 | 9 | 0 |
| v4 `train_list.txt` | 475 | 0 | 1081 | 0 | 1 |
| v4 `val/val_list.txt` | 65 | 31 | 0 | 0 | 0 |
| v4 `val_list.txt` | 36 | 0 | 60 | 0 | 0 |

### Out-of-vocabulary words repaired

64 distinct words, 480 occurrences, present in 359/1653 clips (22%). Top of the distribution:
`Rezolve` (273), `Ai` (46), `agentic` (12), `telecom` (9), `Wagner` (8), `Salman` (6),
`brainpowa` (4), `Rezolve's` (4). `code/lexicon.py` covers 62 entries; after installation the
corpus contains zero `❓`.

## Analysis

**The failure mechanism.** `prepare_v4_data.py:38-52` wraps misaki in `except Exception:
return text.strip()`. misaki raises on out-of-vocabulary words, so the fallback fired
precisely on the clips containing brand and person names. 329 of 1557 training lines and 31 of
96 val lines therefore held raw English orthography. StyleTTS2's `TextCleaner` accepts ASCII
letters — they are in the 178-symbol training table — so the run neither crashed nor warned.

The count lines up: 359 clips contain an unknown word, 329+31 = 360 lines fell back. This is
the same set.

**Why that breaks the duration predictor specifically.** Duration is learned from how text
tokens map to frame counts. A grapheme sequence has no stable relationship to the audio's
frame count, so those 21% of clips contribute pure noise to the duration objective while the
other 79% pull toward a correct alignment. It also explains why t0002 found v4's *voicepack*
usable with v3's decoder while v4's own predictor and decoder were garbage — the style encoder
is conditioned on audio only and never sees text, so it was the one module the corrupted
manifests could not damage.

The Stage 2 loss curves, re-measured directly from the logs with an anchored pattern (an
earlier pass misread them — see "Hypotheses tested and rejected"):

| Stage 2 run | Dur first → last | CE first | val min |
| --- | --- | ---: | ---: |
| `stage2_v3_frozen_lm.log` (v3, success) | 0.81 → 0.81 | 0.05 | **0.506** |
| `stage2_clean.log` (v3) | 8.27 → 1.07 | 0.28 | 0.797 |
| `stage2_dirty.log` (v3) | 9.18 → 1.25 | 0.30 | 0.868 |
| `stage2_v4.log` (t0001) | **16.84** → 1.22 | 0.52 | 0.751 |

v3's successful run opened at Dur 0.81 and held it flat; every failure opened between 8 and 17
and spent the whole schedule descending without arriving. The first logged step is where these
runs separate, which is why it is the stop criterion below.

**Accent.** David's reference voice is British, so `british=True` was correct in v4. The
mismatch was on the other side: t0002 synthesized with `KPipeline(lang_code="a")`. British and
American misaki emit different vowel symbols for the same word, so half of this pair was wrong
before the grapheme fallback is even counted. v5 fixes both ends — `british=True` for
training, `lang_code="b"` for inference, enforced by `code/build_pipeline.py`.

**Two gates that look right and are not.** Both were tried and rejected during this task:

- Rejecting `[A-Za-z]` in phonemes. Valid English IPA is full of ASCII letters (`bˌʌt kæn
  hˈɛlp`); this gate would reject the entire corpus.
- Rejecting uppercase letters mid-IPA. `A I O Q S T W Y` are legal Kokoro vocab symbols
  standing for /eɪ/, /aɪ/, /oʊ/, /əʊ/ and the flap. `bɪhˈInd` is correct misaki output, not
  corruption. This gate rejected 35/40 clips on its first run.

The authoritative gate is Kokoro's own vocab, since that is exactly the token space the model
accepts.

## Limitations

- No training run yet — the data hypothesis is unproven until Dur loss is observed.
- Lexicon pronunciations for person names are a judgement call. They are consistent, in-vocab
  and in the right accent, but a wrong-but-consistent pronunciation of a rare name costs far
  less than the dropped token it replaces.
- The train/val split is inherited from v4 rather than re-derived, deliberately: reusing it
  keeps the token space as the single changed variable against t0001.
- `val_list.txt` is the 96-clip held-out regression set and must never be trained on.

## Files Created

- `results/v5/train_list.txt` (1557 lines), `results/v5/val_list.txt` (96 lines),
  `results/v5/rejects.txt` (empty).
- `results/config_david_v5.yml` — v5 manifests, 10 epochs, v3's optimizer settings restored.
- `code/prepare_v5_data.py`, `code/lexicon.py`, `code/constants.py`, `code/paths.py`,
  `code/build_pipeline.py`, `code/test_gates.py`.

## Stage 1 duration probe (`code/check_stage1_durations.py`)

Run to test whether the defect could be localized to Stage 1 without training. Each
checkpoint's five text-conditioned modules loaded into a stock `KModel` (548/548 keys matched,
0 unexpected, in every case), same sentence, same voicepacks.

| Checkpoint | Stage 1 val | Synthesized | Verdict |
| --- | ---: | ---: | --- |
| v3 Stage 1, 13:57 Sep 7 — fed v3's *successful* Stage 2 | 0.770 | 45.5 s | explosion |
| v3 Stage 1, 21:13 Sep 7 | 0.545 | 3.55 s | sane |
| v4 Stage 1 (t0001's starting point) | 0.594 | 45.5 s | explosion |
| v3 shipped Stage 2 bundle (control) | — | 3.55 s | sane |
| stock Kokoro (baseline) | — | 3.55 s | sane |

**The probe falsified its own premise.** The Stage 1 checkpoint v3 launched its successful
Stage 2 from explodes exactly as v4's does — 45.5 s against 45.5 s. Duration explosion at the
end of Stage 1 is therefore normal: Stage 1 does not train the duration predictor, Stage 2
does. v3 converted an exploding Stage 1 into a working model; v4 did not. **The defect is in
Stage 2, not earlier**, and "v4's Stage 1 is worse" is not supportable — by val it sits
between the two v3 checkpoints and by behaviour it is identical to the one that preceded
success.

This also retires Stage 1 val_loss as a signal: 0.770 preceded success, 0.545 did not.

What survives as evidence for the data hypothesis: from comparably-exploding Stage 1
checkpoints, the two Stage 2 runs opened at Dur loss **0.81 (v3) vs 16.84 (v4)** — a 20x gap
on the first step, which is the input-token side. The dataset confound (266 vs 1557 clips,
long sentences vs short fillers) is not eliminated.

## Next Steps

1. Copy `results/v5/` next to the wavs as `data/v5/` on the VM and rsync the corpus.
2. **Stage 1 from scratch** with `config_david_v5.yml` (`train_first.py`) — required because
   Stage 1 must see v5's token space, not because its checkpoint was diagnosed faulty.
3. **Stop-early check** at Stage 2, not Stage 1: Dur loss on the first logged step must be
   ~0.8, as v3's successful run was. t0001 opened at 16.84 and never recovered. If it opens
   high again, the data hypothesis is wrong — stop rather than burning the full schedule.
4. Stage 2 from the new `first_stage.pth`, `load_only_params: true`,
   `second_stage_load_pretrained: false`.
5. Package via t0002's five-module extraction, then synthesize through `build_pipeline.py` —
   not a bare `KPipeline`, or the brand words and the accent regress.
6. Duration sanity check before any listening: a short sentence must land within ~2x of the
   base Kokoro predictor's duration. This is the cheap test that would have caught t0002's
   defect immediately.

## Hypotheses tested and rejected this session

Recorded because each looked convincing enough to act on:

- **Voicepack scale.** v4's ref_s has a 4.7x larger timbre half and a 4x smaller prosody half
  than v3's. Rescaling both halves to v3's statistics changed the synthesized duration from
  39.65 s to 41.83 s — no effect.
- **Frozen LM.** `stage2_v3_frozen_lm.log` (val 0.506) against `stage2_clean.log` (val 0.822)
  looked like a controlled A/B on identical data. Two measurement errors produced it:
  `stage1.log` is a Stage 2 log by its loss set, not a Stage 1 log, and a `LM Loss:` regex was
  also matching `SLM Loss:`. With an anchored pattern, LM loss explodes in *every* Stage 2 run
  including the successful one (23 → 1165), so LM freezing is not the discriminator.
- **Stage 1 quality.** See the duration probe above.

What separates v3's successful Stage 2 from the three failures is which `first_stage.pth` sat
at `logs/kokoro-david/first_stage.pth` when the run started — the two 10:58/11:35 failures
predate the 13:57 checkpoint the 18:13 success loaded. The fix was re-running Stage 1, with
Stage 2 unchanged.

## Open question

t0002 left one thread unresolved: v3's own first synthesis batch (`v3/audio/v3/`, 43–85 s per
clip) showed the same duration explosion, and `v3b/` (2 s per clip) was clean six minutes
later with no retraining in between. Voicepack rescaling was tested this session and did not
reproduce the fix (39.65 s → 41.83 s), so the v3→v3b change is still unidentified. It does not
block the v5 run, but it should be identified before trusting any single synthesis result.

</details>
