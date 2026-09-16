# Results Summary: v5 Corpus LUFS Normalization and Clipped-Fraction Re-Audit

## Summary

Replaced t0011's over-flagging `peak_dbfs > -0.1 dBFS` clipping check with a
`clipped_fraction > 0.1%` metric, LUFS-normalized every clip that passed the corrected audit to -14
LUFS (with a -1 dBFS sample-peak ceiling to prevent new clipping), and re-checked the normalized
audio. 220 of the 224 clips t0011 flagged purely for peak-normalization are now correctly classified
clean, 0 clips are genuinely clipped, and the final clean train manifest
(`data/train_list_v5_normalized_clean.txt`) covers **1531/1557 clips (98.3%)**, up from t0011's
1311/1557 (84.2%) baseline.

## Metrics

* **Clean-manifest coverage**: **1531/1557 (98.3%)** vs t0011's **1311/1557 (84.2%)** baseline — a
  **+220-clip** recovery.
* **Reclassification of the original 224 peak-flagged clips**: **220 now clean (98.2%)**, **0 still
  genuinely clipped**, 4 excluded only for being `<1.5s` (`duration_low`).
* **Genuine pre-existing clipping** (`clipped_fraction > 0.001`): **0/1557 clips**, inside t0011's
  pre-registered estimate of 0-10.
* **New clipping introduced by the -14 LUFS gain change**: **0/1531** normalized clips
  (`clipping_after_normalization` count is 0).
* **Post-normalization loudness spread**: mean LUFS moves from **-14.69 (std 2.34)** pre to **-15.44
  (std 1.58)** post — a 32% std reduction, though the distribution is ceiling-capped (median, p95,
  p99, and max `post_lufs` are all exactly -14.0), not a tight two-sided cluster (see
  `results/creative_thinking.md` §1).
* **Remaining exclusions**: 26 clips (25 `duration_low`, 1 `silence`), none clipping-related.
* **Total cost**: **$0.00** (CPU-only; no paid APIs or GPU compute).

No registered project metric (`rtf`, `speaker_sim`, `ttfb_ms`) applies to this corpus-normalization
task, so `results/metrics.json` is `{}` — this is a deliberate omission per the plan's Metrics and
cost note, not an oversight.

## Verification

* `verify_plan.py t0012_v5_corpus_normalize_and_reaudit` — PASSED (0 errors, 0 warnings), confirmed
  independently during the planning step.
* `verify_research_code.py` — PASSED (0 errors, 0 warnings).
* Step-executor's independent re-run of the plan's verification snippet during implementation:
  `OK: 1531/1557 clean, 0 val leaks` (confirms REQ-12 and REQ-19: zero `val_96` leakage into the
  clean manifest).
* `ruff check` / `ruff format --check` / `mypy -p tasks.t0012_v5_corpus_normalize_and_reaudit.code`
  — all PASSED (0 errors) on `code/`.
* `wc -l data/per_clip_stats_v2.jsonl` = 1557; `ls results/images/*.png | wc -l` = 3 — both match
  plan expectations (REQ-7, REQ-13).
* `uv run python -u -m arf.scripts.aggregators.aggregate_metrics --format ids` — confirms only
  `rtf`, `speaker_sim`, `ttfb_ms` are registered project metrics, none applicable here, justifying
  `results/metrics.json = {}`.
