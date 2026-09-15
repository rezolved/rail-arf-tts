# Data Audit Summary

## v5 Dataset

- Train clips: **1557**
- Val clips: **96**
- val_96 (held-out, from data/v4/val/): **96**

## Overlap Checks

- Train/val overlap: **0 clips**
- val set = val_96: **True**
- val/val_96 overlap: **96 clips**
- train/val_96 overlap: **0 clips**

## Audio Stats

Audio files are DVC-tracked and were not pulled in this task. Peak amplitude, LUFS, and silence rate
could not be computed. Manifest-level checks (filename overlap, phoneme string length) were
performed.

## Charts

![Duration histogram](images/duration_histogram.png)

Phoneme string length distribution as a proxy for clip duration.

![Loudness histogram](images/loudness_histogram.png)

Loudness distribution (placeholder — audio not available locally).
