# Real-Time Factor (RTF)

**Key**: `rtf`

**Unit**: ratio (dimensionless)

**Definition**: `synthesis_wall_time_seconds / output_audio_duration_seconds`. RTF < 1.0 means
the model synthesizes faster than real time. RTF > 1.0 means synthesis is slower than playback.

**Target**: < 0.2 for Kokoro on LLM-T1-NC80 (H100).

**Baseline**: ElevenLabs is measured RTF > 1.0 (streaming, network-bound) per WEB-761 (RTF=1.55×
at C=1). Local Kokoro inference should far exceed this.

**Note**: RTF is measured end-to-end (model forward pass + vocoder), not just the model forward.
For batched inference, report per-clip RTF (total_time / total_audio_duration across batch).

**Lower is better.**
