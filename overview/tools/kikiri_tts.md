# kikiri-tts

**Role**: training framework for Kokoro-82M fine-tuning on David voice.

**Repo**: `github.com/semidark/kikiri-tts` (semidark fork of StyleTTS2).

**Location on VM**: `/mnt/kikiri-tts/StyleTTS2/` (LLM-T1-NC80, Azure ML 2×H100 SXM5).

**Key scripts**:

| Script | Purpose |
|--------|---------|
| `train_second.py` | Stage 2 StyleTTS2 training (patched — guards `slmadv()` when `lambda_slm == 0`) |
| `scripts/convert_checkpoint.py` | Convert `epoch_2nd_*.pth` → Kokoro API format (strips `module.`, fixes weight_g/weight_v keys) |
| `scripts/extract_voicepack.py` | Extract David voicepack `.pt` from best Stage 2 checkpoint |

**Venv**: `/mnt/kikiri-tts/StyleTTS2/.venv` (Python 3.10, torch 2.6+cu124).

**Critical patch in `train_second.py`** (WavLM shape mismatch fix):

```python
if train_LM:
    optimizer.step("bert_encoder")
    optimizer.step("bert")
```

Without this guard, `slmadv()` crashes at GAN epoch with shape mismatch when `lambda_slm == 0`.

**Config**: `kokoro-finetune/configs/config_david_v4.yml` (tracked in this repo; actual training
copy at `/tmp/config_david_v4b.yml` on the VM).

**Setup reference**: `rail-benchmarks/kokoro-finetune/SETUP_FROM_SCRATCH.md`.
