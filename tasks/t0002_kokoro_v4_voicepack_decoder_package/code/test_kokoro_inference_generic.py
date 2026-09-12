#!/usr/bin/env python3
"""Synthesize test sentences with a given decoder file + fixed Stage1 voicepack.
Usage: script.py <decoder_filename> <out_subdir>
"""

import sys
from pathlib import Path

import soundfile as sf
import torch

TASK_DIR = Path(__file__).parent.parent
VOICEPACK = TASK_DIR / "results" / "david_v4_voicepack.pt"

TEXTS = {
    "sample1": (
        "That's great! Our conversational AI can personalise vehicle recommendations, "
        "streamline checkout and provide real-time support for luxury auto retailers."
    ),
    "sample2": "Understood. If you need anything else later, feel free to reach out.",
}


def main() -> None:
    decoder_name, out_subdir = sys.argv[1], sys.argv[2]
    decoder_path = TASK_DIR / "results" / decoder_name
    out_dir = TASK_DIR / "results" / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    from kokoro import KModel, KPipeline

    state = torch.load(str(decoder_path), map_location="cpu", weights_only=False)

    model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True)
    for mod_name, sd in state.items():
        target = getattr(model, mod_name)
        missing, unexpected = target.load_state_dict(sd, strict=False)
        print(f"  {mod_name}: missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()

    pipe = KPipeline(lang_code="a", model=model, repo_id="hexgrad/Kokoro-82M")

    for name, text in TEXTS.items():
        chunks = []
        for _, _, chunk in pipe(text, voice=str(VOICEPACK), speed=1.0):
            if chunk is not None:
                chunks.append(chunk if isinstance(chunk, torch.Tensor) else torch.tensor(chunk))
        if chunks:
            audio = torch.cat(chunks)
            out_path = out_dir / f"{name}.wav"
            sf.write(str(out_path), audio.numpy(), 24000)
            print(f"{name}: saved {out_path} ({audio.shape[0]/24000:.2f}s)")


if __name__ == "__main__":
    main()
