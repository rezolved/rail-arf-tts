# Results Summary: v10 Checkpoint Synthesis Quality Forensics

## Summary

`kokoro-v10-best` produces noise because of a **real training defect**, not a broken reproduction
script. An instrumented, control-validated StyleTTS2 harness (`code/infer_styletts2.py`) loads both
`epoch_2nd_00016.pth` (primary) and `epoch_2nd_00014.pth` (backup) with 0 missing/0 unexpected keys
on all 13 modules, yet both produce 75-81%-clipped, DC-dominated audio. The root cause, traced to a
specific line, is `train_second_v10.py:load_checkpoint()`'s `ignore_modules` list omitting
`"decoder"`, so a HiFi-GAN vocoder was initialized by a partial, architecture-mismatched load from
an ISTFTNet-shaped checkpoint (`first_stage_v3.pth`) — an initialization a follow-up falsification
probe shows is actively **worse** than pure random init (clip fraction 0.004 vs. 0.750-0.807).

## Metrics

* **Clip fraction (`|sample| > 0.99`)**: control **0.0**, v10 primary (`epoch_2nd_00016`) **0.750**,
  v10 backup (`epoch_2nd_00014`) **0.807**, random-decoder falsification probe **0.004** — the
  decisive noise signature (`results/v10_diagnosis.md`, `results/creative_thinking.md`).
* **Spectral flatness**: control **0.0615** (speech-like, concentrated formant structure) vs. v10
  primary **0.00048** and v10 backup **0.00019** (near-zero flatness with a dominant 0 Hz/DC
  component — vocoder saturation, not white noise).
* **`speaker_sim`** (GE2E cosine vs. 11labs David reference): control **0.48155468702316284**, v10
  primary **0.3506302833557129**, v10 backup **0.3108280301094055** — flagged as a **weak, not
  reliable** signal for this defect: confirmed-garbage v10 audio still scores non-trivially
  positive, so `speaker_sim` alone would not have caught the failure.
* **`rtf`** (CPU, same hardware across variants): control **4.536100560509557**, v10 primary
  **5.262690890033129**, v10 backup **5.485033271459472** — corroborating only, not indicative of
  the defect (all three land in the same unbatched-CPU-inference range).
* **Checkpoint load instrumentation**: **0 missing, 0 unexpected** keys on all 13 modules for both
  v10 checkpoints and the external control (`results/load_log_epoch_2nd_00016.json`,
  `results/load_log_epoch_2nd_00014.json`, `results/load_log_epochs_2nd_00020.json`) — rules out a
  reproduction/loader bug.

## Verification

* `verify_task_dependencies.py` — PASSED (0 errors, 0 warnings; step 2,
  `logs/steps/002_check-deps/deps_report.json`).
* `verify_research_code.py` — PASSED (0 errors, 0 warnings; step 6).
* `verify_plan.py` — PASSED (0 errors, 0 warnings; step 7).
* `verify_task_results.py` — PASSED (0 errors, 0 warnings).
* `verify_task_metrics.py` — PASSED (0 errors, 0 warnings).
* Harness self-validation (Milestone D control gate): **PASS** — `code/audio_quality_check.py`'s
  `is_likely_noise=False` on the known-good external control checkpoint, confirming the harness
  itself is not the source of the noise before ever diagnosing v10 (`results/control_test.md`).
