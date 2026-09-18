# Results Summary: Recover the v3 Training Recipe From Artifacts

## Summary

Reconstructed Kokoro v3's Stage 2 StyleTTS2 recipe from a bounded VM inventory, checkpoint-shape
forensics, and per-epoch audio gating, since the original author is unavailable and the recipe was
never written down. The VM's home directory turned out to be a later task's (t0014's) StyleTTS2
clone rather than a preserved v3-era environment, so the 266-clip training list and byte-identity of
`first_stage_v3.pth` remain unrecoverable, but checkpoint-shape forensics independently resolved the
standing `multispeaker` three-way contradiction to **`inferred false`** and showed all five bundle
modules (including the decoder) changed substantially during Stage 2. The `v3-recipe` answer asset
records the full recipe at `confidence: "low"`, honestly reflecting how much of the config is
`inferred`/`unknown` rather than `confirmed`.

## Metrics

* **VM session cost**: **$3.13** for **0.2245 h** (~13.47 min) of `LLM-T1-NC80` time, well under the
  $21 VM sub-cap and the $30 total task cap.
* **Checkpoint module deltas (Stage 1 -> best)**: `decoder` **+224.96%**, `predictor` **+167.23%**,
  `bert_encoder` **+253.70%**, `text_encoder` **+20.17%**, `bert` **-6.55%** relative weight-norm —
  all 5 of 5 modules exceed the 5% "changed" threshold, ruling out a frozen decoder.
* **Audio-quality gate**: **55** per-epoch/variant samples scored; **15/55** (`v3/ep6-8`) flagged
  `is_likely_noise=True`, **0/20** `epoch0-3` and **0/20** `v3b/ep6-9` flagged — confirms `v3b`
  supersedes a broken `v3` variant at the same epochs.
* **Local re-synthesis cross-check**: `speaker_sim=0.566`, `rtf=1.9361`, `ttfb_ms=6349.0` (8 clips,
  CPU) — outside the ±0.02 tolerance against t0008's recorded `0.631`/`0.588`, attributed to a
  reference-corpus pre-filter mismatch, not a bundle regression (see `results/metrics.json` and
  `results/v3_checkpoint_forensics.md`).
* **Requirement coverage**: **11/17** `REQ-*` items `Done`, **2/17** `Partial`, **2/17** `Blocked`
  (evidence genuinely unrecoverable), **2/17** orchestrator-scoped (see
  `results/results_detailed.md`'s `## Task Requirement Coverage`).

## Verification

* `uv run python -m arf.scripts.verificators.verify_task_metrics t0016_v3_recipe_recovery` — PASSED
  (0 errors, 0 warnings): `speaker_sim`, `rtf`, `ttfb_ms` are all registered project metrics with
  scalar float values.
* `uv run python -m arf.scripts.verificators.verify_step t0016_v3_recipe_recovery results` — PASSED.
* `uv run python -m arf.scripts.verificators.verify_answer_asset t0016_v3_recipe_recovery v3-recipe`
  (run during `implementation`, step 9) — PASSED (0 errors, 0 warnings), `confidence: "low"`.
* `verify_machines_destroyed` (run during `teardown`, step 10) — PASSED (0 errors, 2 non-blocking
  warnings: legacy `spec_version`, API-unreachable-from-sandbox).
