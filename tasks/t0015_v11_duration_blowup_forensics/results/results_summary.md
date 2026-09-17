# Results Summary: v11 Duration-Blowup Forensics and Audible-Speech Gate Hardening

## Summary

`kokoro-v11-best`'s 15-20x synthesis duration blowup is root-caused to a predictor-pathway
calibration failure — most plausibly `predictor_encoder`'s `copy.deepcopy(style_encoder)`
initialization — not a `pred_aln_trg`/decoder plumbing bug and not `max_dur`-ceiling saturation. The
blowup is universal (10/10 characterization texts) and the cheap inference-parameter fix does not
work (0/13 combinations passed). The audible-speech gate (`code/audio_quality_check.py`) was
hardened with two new signals and proven, via a three-way regression, to now catch this exact defect
while leaving its original signals and v10's known-broken verdict unchanged.

## Metrics

* **Universal duration blowup**: all **10/10** characterization texts blew up, range
  **8.61x-17.87x** (mean **14.79x**), with no separation between short-text (mean 15.04x) and
  long-text (mean 14.54x) categories (`results/duration_characterization.json`,
  `results/duration_blowup_diagnosis.md` Section 2).
* **Cheap inference-parameter fix failed**: **0 of 13** pre-registered `alpha`/`beta`/
  `diffusion_steps`/`embedding_scale` combinations passed all three pass criteria simultaneously;
  the best result (`grid_a0.1_b0.7`) reached `duration_ratio=5.54`, still **1.85x** over the
  required `<=3.0` threshold (`results/param_sweep.json`).
* **Tensor forensics narrows the culprit**: `predictor.duration_proj`'s own weight norm shifted only
  **+0.7%** from the LibriTTS control (83.4531 vs. 82.8593), while `predictor_encoder`'s
  whole-module weight norm shifted **-12.7%** (400.6487 vs. 459.0197) — evidence the calibration
  failure is upstream of `duration_proj`'s own weights (`results/predictor_tensor_forensics.md`).
* **Frame-to-output plumbing is healthy, not buggy**: the observed 2.00x frame-to-output-duration
  ratio (`mean_frame_vs_output_duration_ratio=1.9998`) is confirmed to be the decoder's normal
  architecture-intrinsic upsample (source-read + reproduced identically on the LibriTTS control
  checkpoint: `pred_dur_sum=186`, `output_duration_s=4.65`, ratio 2.00) — not a plumbing defect
  (`results/localization_summary.json`).
* **Hardened gate proven via three-way regression**: v10 still fails (`is_likely_noise=True`),
  v11-as-shipped keeps `is_likely_noise=False` (proving the old gate genuinely still ships it) but
  now fails the hardened gate (`hardened_gate_pass=False`, `longest_nonsilent_run_s=73.94` vs. the
  `12.0`s threshold) (`results/gate_regression.json`).
* **Metrics reproduced**: `v11-as-shipped` `rtf=3.18`, `speaker_sim=0.444`; `v10-primary`
  `rtf=5.26`, `speaker_sim=0.351` (`results/metrics.json`).

## Verification

* `uv run python -u -m arf.scripts.verificators.verify_task_metrics t0015_v11_duration_blowup_forensics`
  — PASSED (step 9, re-confirmed this step; see `results/results_detailed.md` ## Verification for
  the exact command output).
* `python3 -c "import json; d = json.load(open('results/duration_characterization.json')); assert len(d) == 10"`
  — PASSED, exactly 10 characterization records with `pred_dur`/`pred_dur_sum`/`input_token_count`
  populated.
* `python3 -c "import json; d = json.load(open('results/gate_regression.json'))..."` (the plan's
  REQ-10 discriminator assertion: v10 `is_likely_noise=True`, v11-as-shipped `is_likely_noise=False`
  and `hardened_gate_pass=False`) — PASSED.
* `uv run ruff check . && uv run ruff format --check .` and
  `uv run mypy -p tasks.t0015_v11_duration_blowup_forensics.code` — PASSED clean (step 9).
* `uv run pytest tasks/t0015_v11_duration_blowup_forensics/code/` — PASSED (step 9's
  `test_audio_quality_check.py`).
