# Time to First Byte (TTFB)

**Key**: `ttfb_ms`

**Unit**: milliseconds

**Definition**: wall-clock time from the moment a TTS synthesis request is issued until the first
audio byte (or first audio chunk) is received by the client. Measured at the inference layer, not
the network layer (local loopback or same-machine calls).

**Target**: ≤ 300 ms for the fine-tuned Kokoro v4 on LLM-T1-NC80 (H100).

**Baseline**: ElevenLabs David TTFB measured in t0002 (preliminary ~300–600 ms from WEB-761).

**Measurement protocol**: 50 warmup requests discarded, then N=100 measured requests, same
prompt corpus, same engine session. Report p50, p95, p99.

**Lower is better.**
