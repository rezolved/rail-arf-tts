# Results Summary: Zero-Shot Latency Reduction (CosyVoice2 / Chatterbox TTFB Floor)

## Summary

This task instrumented CosyVoice2 and Chatterbox with per-stage `StageTiming` timers and swept 5
CosyVoice2 and 6 Chatterbox acceleration variants (reference-embedding caching, precision, JIT,
TensorRT, `torch.compile`, sentence chunking) across 196 prompts per variant, all measured against a
corrected (production-voice) reference and centroid per a binding owner correction. Neither system
reaches the 300 ms TTFB target: CosyVoice2's floor is **826 ms p50** (`load_trt`) and Chatterbox's
is **837 ms p50** (`torch_compile`), both roughly 2.7-2.8x the target, with the gap traced to an
autoregressive LM decode/prefill stage that stayed at 815-1,004 ms across every tested lever. Every
variant preserved `speaker_sim` within 0.007 of its paired baseline, so the tested levers are safe,
free wins even though they do not close the 300 ms gap.

## Metrics

* CosyVoice2 best TTFB (fillers, `load_trt`): **826.315 ms p50** / **1,035.226 ms p95** vs.
  **1,176.701 ms p50** baseline (`baseline_new_ref`) — a **29.8%** p50 cut, `speaker_sim` 0.857 vs.
  0.854 baseline (unchanged).
* Chatterbox best TTFB (fillers, `torch_compile`): **837.034 ms p50** / **1,032.547 ms p95** vs.
  **1,039.240 ms p50** baseline (`baseline_new_ref`) — a **19.4%** p50 cut, `speaker_sim` 0.838 vs.
  0.834 baseline (unchanged).
* LM prefill/decode stage cost across all 10 measured variants: **814.3-904.4 ms** (CosyVoice2),
  **838.98-1,003.84 ms** (Chatterbox) — flat or slightly worse under every precision/JIT/TensorRT
  lever, versus a 62% cut to the flow-matching+vocoder stage from `load_trt` alone (397.4 ms to
  151.1 ms).
* `meets_ttfb_target_300ms` is `false` for all 22 measured variant rows in `results/tables.json`; no
  variant of either system reaches 300 ms p50.
* Automated `hardened_gate_pass` rate: **2,045/2,156** scored clips (**94.85%**) — reported as
  necessary-not-sufficient per the owner's explicit correction (owner has not yet performed a manual
  listening pass as of this writing).
* Total GPU cost: **$98.25** of the **$100.00** hard cap (`results/costs.json`), 7.0378 hours on
  `LLM-T1-NC80` (2x H100 NVL).

## Verification

* `verify_task_metrics.py` — PASSED (0 errors) at implementation closeout (step 9); re-verified this
  step, no changes to `metrics.json`.
* `aggregate_metrics --format ids` — confirmed `speaker_sim`, `ttfb_ms`, `rtf` are the only 3
  registered project metrics, and `metrics.json` uses exactly these keys.
* `assets/answer/zero-shot-ttfb-floor/` answer-asset verificator — PASSED at implementation
  closeout.
* `verify_task_results.py` — run at the end of this step against `results_summary.md` /
  `results_detailed.md`; see `logs/steps/012_results/step_log.md` for the exact invocation and
  outcome.
