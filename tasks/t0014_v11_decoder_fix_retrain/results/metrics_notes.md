# Notes on `results/metrics.json` (REQ-9)

`results/metrics.json` uses the explicit multi-variant format and, per
`arf/specifications/task_results_specification.md`, may only contain registered metric keys under
`metrics` -- caveats live here instead.

* **`ttfb_ms` deliberately not measured.** This is an offline batch harness
  (`code/infer_styletts2.py`) writing one complete WAV per invocation via CPU inference; there is no
  streaming/production serving endpoint in scope for this task, matching t0013's own precedent
  (`tasks/t0013_v10_synthesis_quality_forensics/results/metrics.json` also omits it).
* **`rtf` caveat.** Measured single-clip on the same unbatched-CPU-inference harness as t0013's
  numbers (corroborating only, not a benchmark claim). `rtf=3.18` means synthesis wall time (235.2s)
  was 3.18x the output audio duration (73.9s) -- both figures are elevated by the duration-predictor
  anomaly documented in `results/v11_gate_verdict.md` ("Caveat: anomalous output duration"), not
  solely by CPU inference cost.
* **`speaker_sim` caveat.** 0.444 is higher than both confirmed-broken v10 checkpoints (0.311-0.351,
  `tasks/t0013_v10_synthesis_quality_forensics/results/metrics.json`) and close to (slightly below)
  the LibriTTS control's 0.482 (different reference/content, not directly comparable). `speaker_sim`
  is a weak signal for the specific defect this task fixes -- both confirmed-broken v10 outputs
  still scored non-trivially positive -- so **the audible-speech gate
  (`results/v11_gate_verdict.md`), not `speaker_sim`, is this task's actual pass/fail criterion**
  (plan.md step 11).
