# t0003 — Regenerate v5 phoneme manifests

## Objective

Produce train/val manifests whose text column is in the same token space Kokoro uses at inference,
so the next Stage 1 + Stage 2 fine-tune can beat v3's val_loss of 0.506.

## Background

t0001's Stage 2 run diverged (val_loss 0.594 → 0.751 → 1.147) and t0002's packaging of its
checkpoints produced a 10x duration explosion — 89 s of audio for a sentence that should take 9.6 s.
t0002 ruled out the packaging code by running v3's proven artifacts through the identical extraction
and inference path, which came out clean. The defect was therefore in t0001's training, and t0001's
hyperparameters were already close to v3's.

The training data was never checked. `kokoro-finetune/scripts/prepare_v4_data.py` phonemizes with
misaki inside a bare `except Exception: return text.strip()`. misaki raises on words it does not
know — and the corpus is full of them: *Rezolve* (273 occurrences), *Ai*, *brainpowa*, *agentic*,
and a long tail of person and company names. Every one of those clips fell back to raw English
orthography. StyleTTS2's `TextCleaner` accepts ASCII letters, so nothing crashed and nothing warned;
the run simply trained on a 21% mixture of graphemes inside a phoneme corpus.

A second, independent mismatch sat on top: the corpus was phonemized as British English (correct —
David's reference voice is British) while t0002 synthesized with `KPipeline(lang_code="a")`,
American. The two accents emit different vowel symbols for the same word.

## Approach

Rebuild the text column only. Wav files and the train/val split are taken verbatim from the existing
v4 directories, so the sole difference against the failed run is the token space.

1. Recover each clip's original text from the *originals* — the v3 `manifest.csv` files and
   `fillers_from_logs.txt` — never from a v4 manifest, since all three v4 variants are corrupted in
   different ways.
2. Phonemize with `misaki.en.G2P(trf=False, british=True)`, with no exception fallback.
3. Install a lexicon (`code/lexicon.py`) for the 62 words misaki does not know, so none of them
   degrade to misaki's `❓` marker.
4. Gate every line against Kokoro's own 114-symbol vocab from `config.json` — the exact character
   set the model accepts — plus explicit checks for raw-text leakage, double phonemization and `❓`.

## Deliverables

- `results/v5/train_list.txt`, `results/v5/val_list.txt` — manifests in
  `wav_path|phonemes|speaker_id` form, every line gate-clean.
- `results/config_david_v5.yml` — training config pointing at the new manifests, with v3's optimizer
  settings restored.
- `code/build_pipeline.py` — inference-side pipeline that applies the same accent and lexicon, so
  synthesis is fed the token space the model was trained on.

## Verification criteria

- 0 lines rejected by any gate in either output manifest.
- The gates flag all three broken v4 manifests when replayed against them.
- `Rezolve`, `brainpowa` and `agentic` phonemize without `❓` on both the training and the inference
  path.
- Deferred to the training run: Dur loss reaches ~0.72 in the first Stage 1 epoch (v3's value). If
  it sits at ~1.03 again, the data hypothesis is wrong and the run should be stopped rather than
  continued for 20 epochs.
