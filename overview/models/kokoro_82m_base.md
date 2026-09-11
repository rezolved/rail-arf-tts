# Kokoro-82M Base

**Role**: zero-shot TTS baseline — Kokoro without domain fine-tuning, evaluated before committing
to fine-tuning effort.

**Architecture**: StyleTTS2 (82M parameters). Open-weight model by hexgrad.

**Checkpoint**: `kokoro-v1_0.pth` (public release, no fine-tuning).

**Key characteristics**:

* Multi-speaker zero-shot synthesis via style embedding conditioning.
* 24 kHz output, iSTFTNet decoder.
* No domain-specific David voice data — speaker similarity expected to be lower than fine-tuned.

**Key metrics (to be measured in t0003)**:

| Metric | Expected | Notes |
|--------|----------|-------|
| `ttfb_ms` | < 200 ms | Local H100, batch_size=1 |
| `speaker_sim` | 0.5–0.7 | GE2E cosine vs 11labs ref — pre-finetune |
| `rtf` | < 0.2 | H100 real-time factor |

**Benchmark corpus**: val_96 (96 held-out clips from `data/v4/val/val_list.txt`).
