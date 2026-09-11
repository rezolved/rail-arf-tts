# Kokoro-82M v4 Fine-Tune (David)

**Role**: primary TTS candidate — Kokoro fine-tuned on David voice data via StyleTTS2 Stage 2.

**Architecture**: StyleTTS2 (82M parameters), same as base but with domain-adapted weights.

**Training data**: 1557 clips from `data/v4/train/train_list.txt` (David voice, 24 kHz WAV).

**Training config**: `kokoro-finetune/configs/config_david_v4.yml`.

**Key training parameters**:

* Stage 2 only (Stage 1 complete: `first_stage.pth`, val=0.594, epoch 9).
* `multispeaker: true`, `load_only_params: true` (fresh Stage 2 from Stage 1).
* `train_LM: false`, `lambda_slm: 0.0` (WavLM discriminator disabled).
* `batch_size: 2`, `epochs_2nd: 20`, `lr: 3e-5`.
* GPU: LLM-T1-NC80 (Nebius 2×H100 SXM5).

**Status**: Stage 2 training in progress (restarted from `first_stage.pth` — prior
`epoch_2nd_00000–00003.pth` checkpoints were all corrupted via OOM recovery).

**Key metrics (to be measured in t0004)**:

| Metric | Target | Notes |
|--------|--------|-------|
| `speaker_sim` | ≥ 0.85 | GE2E cosine vs 11labs ref on val_96 |
| `ttfb_ms` | ≤ 300 ms | Local H100 inference |
| `rtf` | < 0.2 | H100 real-time factor |
