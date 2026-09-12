#!/usr/bin/env python3
"""
Extract decoder sub-state-dict from v4 Stage 2 best checkpoint, remapped to Kokoro-API format.
Adapted from rail-benchmarks/kokoro-finetune/scripts/convert_checkpoint.py.

Usage:
    uv run python3 code/extract_decoder_v4.py
"""

from pathlib import Path

import torch

TASK_DIR = Path(__file__).parent.parent
STAGE2_CKPT = (
    TASK_DIR.parent
    / "t0001_kokoro_v4_stage2_finetune"
    / "results"
    / "checkpoints"
    / "epoch_2nd_00005.pth"  # epoch 6, val_loss=0.751 — best pre-divergence checkpoint
)
OUT = TASK_DIR / "results" / "david_v4_decoder.pth"


def convert_key(k: str) -> str:
    if k.startswith("module."):
        k = k[7:]
    k = k.replace(".parametrizations.weight.original0", ".weight_g")
    k = k.replace(".parametrizations.weight.original1", ".weight_v")
    return k


MODULES = ["bert", "bert_encoder", "predictor", "text_encoder", "decoder"]


def main() -> None:
    print(f"Loading Stage 2 checkpoint: {STAGE2_CKPT}")
    raw = torch.load(str(STAGE2_CKPT), map_location="cpu", weights_only=False)
    net = raw["net"]
    print(f"val_loss={raw.get('val_loss')}, epoch={raw.get('epoch')}")

    out = {mod: {convert_key(k): v for k, v in net[mod].items()} for mod in MODULES}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(OUT))
    print(f"Saved {OUT}")
    for mod in MODULES:
        print(f"  {mod}: {len(out[mod])} params")


if __name__ == "__main__":
    main()
