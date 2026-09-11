# ElevenLabs David

**Role**: production baseline — current system used for filler synthesis in Rezolve's voice commerce
assistant.

**Provider**: ElevenLabs API (voice ID: David).

**Reference audio**: `data/11labs_david/` — 1358 WAV clips synthesized from the `fillers_from_logs.txt`
prompt set.

**Cost**: ~$0.30 / 1000 characters (ElevenLabs Starter plan).

**Key metrics (to be measured in t0002)**:

| Metric | Expected | Source |
|--------|----------|--------|
| `ttfb_ms` | ~300–600 ms | WEB-761 preliminary (RTF=1.55×, C=1=911 ms) |
| `speaker_sim` | 1.0 (self-reference) | GE2E cosine against own clips |
| `rtf` | >1.0 (streaming) | WEB-761 |

**Benchmark corpus**: `data/11labs_david/` (1358 clips). Used as speaker-similarity ground truth for
all Kokoro evaluations.
