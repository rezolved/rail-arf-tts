#!/bin/bash
# Sets up kikiri-tts training environment on a fresh LLM-T1-NC80 VM.
# Run as: ssh LLM-T1-NC80 "bash /mnt/setup_vm.sh"
set -euo pipefail

INSTALL_DIR=/mnt/kikiri-tts
LOG_DIR=/mnt/kikiri-tts/logs

echo "=== Step 1: System deps ==="
sudo apt-get update -qq
sudo apt-get install -y espeak-ng libsndfile1 git git-lfs ffmpeg

echo "=== Step 2: Clone kikiri-tts ==="
if [ ! -d "$INSTALL_DIR" ]; then
    git clone https://github.com/semidark/kikiri-tts "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    git submodule update --init --recursive
else
    echo "Already cloned, pulling latest..."
    cd "$INSTALL_DIR" && git pull
fi

echo "=== Step 3: Python venv ==="
cd "$INSTALL_DIR/StyleTTS2"
python3.10 -m venv .venv
source .venv/bin/activate

echo "=== Step 4: PyTorch + training deps ==="
pip install --quiet torch==2.6.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
pip install --quiet \
    accelerate==1.14.0 \
    transformers==5.16.1 \
    huggingface_hub==1.30.0 \
    librosa==0.11.0 \
    soundfile==0.14.0 \
    phonemizer==3.4.0 \
    munch==4.0.0 \
    pyyaml einops einops-exts tqdm matplotlib pydub
pip install --quiet git+https://github.com/resemble-ai/monotonic_align.git

echo "=== Step 5: Utils models (ASR, JDC, PLBERT) ==="
cd "$INSTALL_DIR/StyleTTS2"
# Download from HuggingFace if not already present
if [ ! -f "Utils/ASR/epoch_00080.pth" ]; then
    echo "Downloading ASR model..."
    python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os
for path in ['Utils/ASR/epoch_00080.pth', 'Utils/ASR/config.yml',
             'Utils/JDC/bst.t7', 'Utils/PLBERT/config.yml']:
    os.makedirs(os.path.dirname(path), exist_ok=True)
hf_hub_download('yl4579/StyleTTS2-LJSpeech', 'Models/LJSpeech/Utils/ASR/epoch_00080.pth',
                local_dir='.')
" 2>/dev/null || echo "Manual download needed — see RESUME_TOMORROW.md for instructions"
fi

echo "=== Step 6: Create dirs ==="
mkdir -p "$LOG_DIR"
mkdir -p "$INSTALL_DIR/StyleTTS2/logs/kokoro-david-v4"
mkdir -p "$INSTALL_DIR/data/v4/train" "$INSTALL_DIR/data/v4/val"

echo "=== Step 7: Verify GPUs ==="
python3 -c "import torch; n=torch.cuda.device_count(); print(f'Setup complete. GPUs: {n}'); assert n == 2, f'Expected 2 GPUs, got {n}'"

echo "=== Setup complete ==="
