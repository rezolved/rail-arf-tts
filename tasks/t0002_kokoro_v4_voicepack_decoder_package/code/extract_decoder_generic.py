#!/usr/bin/env python3
"""Extract+convert decoder from any checkpoint path to Kokoro format. Usage: script.py <ckpt> <out_name>"""

import sys
from pathlib import Path

import torch

TASK_DIR = Path(__file__).parent.parent
CKPT_DIR = TASK_DIR.parent / "t0001_kokoro_v4_stage2_finetune" / "results" / "checkpoints"


def convert_key(k: str) -> str:
    if k.startswith("module."):
        k = k[7:]
    k = k.replace(".parametrizations.weight.original0", ".weight_g")
    k = k.replace(".parametrizations.weight.original1", ".weight_v")
    return k


MODULES = ["bert", "bert_encoder", "predictor", "text_encoder", "decoder"]


def main() -> None:
    ckpt_name, out_name = sys.argv[1], sys.argv[2]
    raw = torch.load(str(CKPT_DIR / ckpt_name), map_location="cpu", weights_only=False)
    net = raw["net"]
    print(f"{ckpt_name}: val_loss={raw.get('val_loss')}, epoch={raw.get('epoch')}")
    out_sd = {mod: {convert_key(k): v for k, v in net[mod].items()} for mod in MODULES}
    out = TASK_DIR / "results" / out_name
    torch.save(out_sd, str(out))
    print(f"Saved {out}")
    for mod in MODULES:
        print(f"  {mod}: {len(out_sd[mod])} params")


if __name__ == "__main__":
    main()
