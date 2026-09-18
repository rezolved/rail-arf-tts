---
spec_version: "2"
answer_id: "zero-shot-ttfb-floor"
answered_by_task: "t0021_zero_shot_latency_reduction"
date_answered: "2026-09-18"
---
## Question

What is the lowest reachable TTFB for CosyVoice2 and Chatterbox on our hardware without losing
speaker_sim, and does either reach 300 ms?

## Answer

No, neither CosyVoice2 nor Chatterbox reaches the 300 ms TTFB target, and the remaining gap is
architectural rather than closeable by engineering-only levers. CosyVoice2's lowest reachable TTFB
is 826 ms p50 (TensorRT/`load_trt`), a 30% cut from its 1,177 ms baseline with `speaker_sim`
unchanged (0.857 vs 0.854); Chatterbox's is 837 ms p50 (`torch_compile`), a 19% cut from 1,039 ms,
also with `speaker_sim` unchanged (0.838 vs 0.834). Both floors are still 2.7-2.8x the target. The
per-stage breakdown explains why: the autoregressive LM decode/prefill stage costs 815-1,004 ms in
every variant of both systems regardless of caching, precision, JIT, or TensorRT, because that cost
comes from the model's own token-by-token decoding scheme, not a serving-stack inefficiency the
tested levers could remove. This is based on a paired 196-prompt measurement of 10 acceleration
variants across both systems on the project's own hardware.

## Sources

* Task: `t0021_zero_shot_latency_reduction`
* Task: `t0018_zero_shot_cloning_calibration`
* Paper: `10.48550_arXiv.2412.10117` (CosyVoice 2)
* Paper: `10.48550_arXiv.2605.30748` (Chatterbox-Flash)
