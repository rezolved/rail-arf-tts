# Results Summary: v5 Training Data Audio Quality Audit

## Summary

Audited all **1557** v5 training clips across six quality dimensions. The corpus is predominantly
high-quality — no LUFS outliers, no OOV flags, no sample-rate or channel mismatches, and zero file
errors. The primary issue is peak normalization: 224 clips (14.4%) have peak amplitude within 0.1 dB
of full scale. After removing all flagged clips, **1311 clean clips** remain for the full-corpus
Stage 2 fine-tuning run.

## Metrics

- **Total clips audited**: **1557** (train) + **96** (val)
- **Clean manifest**: **1311** clips (84.2% of corpus)
- **Flagged clips**: **246** (15.8%)
  - Clipping (peak_dbfs > -0.1 dBFS): **224**
  - Duration too short (< 1.5 s): **25**
  - Excessive silence (> 30%): **1**
  - LUFS outliers, OOV, sample rate mismatch, errors: **0**
- **LUFS distribution**: mean **-14.69 LUFS**, range [-26.19, -8.55] (all within [-30, -6])
- **Duration distribution**: mean **3.42 s**, median **2.51 s**, p95 **11.58 s**
- **Silence fraction**: mean **3.4%**, max **33.5%** (only 1 clip exceeds threshold)
- **Processing time**: < 60 seconds for 1557 clips (soundfile + librosa + pyloudnorm)

## Verification

- `wc -l per_clip_stats.jsonl` → 1557 — PASSED
- Consistency check: 246 flagged + 1311 clean = 1557 — PASSED
- All JSONL records have required fields (including `oov_fraction`, `oov_count`) — PASSED
- val_96 leak check: 0 paths from val set appear in clean manifest — PASSED
- 4 PNG histograms generated (28-38 KB each) — PASSED
- `ruff check`: 0 errors — PASSED
- `mypy`: 0 errors — PASSED
