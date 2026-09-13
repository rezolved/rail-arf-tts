# Research: Kokoro-82M v4 Stage 2 Fine-tune

## Objective

Run StyleTTS2 Stage 2 fine-tuning of Kokoro-82M on the ElevenLabs David voice dataset to produce a
self-hosted checkpoint that matches ElevenLabs David speaker similarity (GE2E cosine ≥ 0.85) and
inference latency (TTFB ≤ 300 ms).

## Background

Prior work in `rail-benchmarks/kokoro-finetune` (Kokoro finetune v3/v4 experiments):

- Stage 1 complete: `best_v4/first_stage.pth` (epoch 10, val=0.594).
- Stage 2 reached epoch 4 with val=1.539; epoch 5 hit OOM before checkpoint save.
- Training data: 1557 clips (`data/v4/train/train_list.txt`), val: 96 clips (`data/v4/val/val_list.txt`).
- ElevenLabs David reference audio: 1358 WAVs (`data/11labs_david/`).

## Known Issues from v3/v4 Experiments

### OOM at GAN epoch

- `batch_size: 4` caused CUDA OOM at epoch 5 when GAN discriminator activates (`joint_epoch: 3`).
- Fix 1: `batch_size: 2` (already set in config).
- Fix 2: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — prevents fragmentation during
  discriminator forward passes.
- Fix 3: patched `train_second.py` guards `slmadv()` call when `lambda_slm == 0`. Without patch:
  `RuntimeError: Given groups=1, weight of size [64, 9984, 1], expected input[4, 9926400, 62] to have 9984 channels`.

### NaN collapse

- NaN at epoch 4 with `lr: 1e-4`; reduced to `lr: 3e-05`, `ft_lr: 3e-05`, `bert_lr: 1e-06`.
- `train_LM: false` — freeze BERT; `train_LM: true` causes LM Loss explosion (23→400+).

### WavLM discriminator

- `lambda_slm: 0.0` — WavLM shape mismatch crash at GAN activation. Must stay 0.

## Multi-GPU Strategy

Two H100 NVL GPUs (95 GB each) on LLM-T1-NC80. Using DDP via `accelerate launch --num_processes 2`:

- DDP doubles throughput (2× faster epochs) but does NOT reduce per-GPU peak memory — each GPU
  holds a full model copy + optimizer state.
- OOM fix is `expandable_segments` + `batch_size: 2`, not DDP itself.
- DDP benefit: faster iteration over 20 epochs.

## Key Config Parameters

| Parameter | Value | Why |
| --- | --- | --- |
| `batch_size` | 2 | OOM fix — 4 crashes at GAN phase |
| `lr`, `ft_lr` | 3e-05 | NaN fix — 1e-4 collapsed at epoch 4 |
| `bert_lr` | 1e-06 | Stable BERT fine-tuning |
| `train_LM` | false | LM Loss explosion fix |
| `lambda_slm` | 0.0 | WavLM shape mismatch crash fix |
| `joint_epoch` | 3 | GAN activates from epoch 4 onward |
| `load_only_params` | true | Fresh Stage 2 — no corrupted optimizer state |
| `second_stage_load_pretrained` | true | Load from first_stage.pth |

## References

- StyleTTS2: Hu et al. (2023) — "StyleTTS 2: Towards Human-Level Text-to-Speech through Style
  Diffusion and Adversarial Training with Large Speech Language Models".
- Kokoro-82M: hexgrad/kokoro (open-weight StyleTTS2 variant).
- kikiri-tts: semidark/kikiri-tts — training harness used in prior experiments.
- Prior experiment notes: `rail-benchmarks/kokoro-finetune/RESUME_TOMORROW.md`.
