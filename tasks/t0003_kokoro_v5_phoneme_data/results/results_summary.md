# Results Summary: v5 Phoneme Manifest Regeneration

## Summary

Rebuilt the Kokoro fine-tune manifests: **1557/1557 train and 96/96 val lines pass every validation
gate, 0 rejected**, against 329 raw-text and 9 unknown-word lines in the manifest t0001 actually
trained on. Root cause of t0001's divergence and t0002's 10× duration explosion is a silent grapheme
fallback in `prepare_v4_data.py` that fired on every clip containing a word misaki does not know —
21% of the training set. No training has been run yet.

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
