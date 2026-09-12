#!/usr/bin/env python3
"""
Synthesize the two t0001 comparison sentences via the production kokoro.KModel/KPipeline
path, using the packaged v4 voicepack (Stage 1 style) + decoder (Stage 2 epoch 6).

Usage:
    uv run --with kokoro --with torchaudio --with soundfile python3 code/test_kokoro_inference.py
"""

from pathlib import Path

import soundfile as sf
import torch

TASK_DIR = Path(__file__).parent.parent
VOICEPACK = TASK_DIR / "results" / "david_v4_voicepack.pt"
DECODER = TASK_DIR / "results" / "david_v4_decoder.pth"
OUT_DIR = TASK_DIR / "results" / "audio_samples"

TEXTS = {
    "sample1": (
        "That's great! Our conversational AI can personalise vehicle recommendations, "
        "streamline checkout and provide real-time support for luxury auto retailers."
    ),
    "sample2": "Understood. If you need anything else later, feel free to reach out.",
}


def main() -> None:
    from kokoro import KModel, KPipeline

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading decoder from {DECODER}")
    decoder_sd = torch.load(str(DECODER), map_location="cpu", weights_only=False)["decoder"]

    model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True)
    missing, unexpected = model.decoder.load_state_dict(decoder_sd, strict=False)
    print(f"decoder loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()

    pipe = KPipeline(lang_code="a", model=model, repo_id="hexgrad/Kokoro-82M")

    for name, text in TEXTS.items():
        chunks = []
        for _, _, chunk in pipe(text, voice=str(VOICEPACK), speed=1.0):
            if chunk is not None:
                chunks.append(chunk if isinstance(chunk, torch.Tensor) else torch.tensor(chunk))
        if chunks:
            audio = torch.cat(chunks)
            out_path = OUT_DIR / f"{name}.wav"
            sf.write(str(out_path), audio.numpy(), 24000)
            print(f"{name}: saved {out_path} ({audio.shape[0]/24000:.2f}s)")
        else:
            print(f"{name}: no audio generated")


if __name__ == "__main__":
    main()
