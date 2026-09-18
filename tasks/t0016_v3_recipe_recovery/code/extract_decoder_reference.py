#!/usr/bin/env python3
"""Read-only reference copy of
`tasks/t0002_kokoro_v4_voicepack_decoder_package/code/extract_decoder_generic.py`, kept verbatim
(per plan.md Milestone 1 Step 2) to document the five-module/key-remapping contract that produced
`best/david_v3_best_decoder_kokoro.pth` from a raw StyleTTS2 checkpoint -- `convert_key()` below is
exactly why `checkpoint_forensics.py`'s `load_raw_net()` must tolerate two different top-level
checkpoint shapes (raw `{"net": {...}}` vs. packaged `{module: state_dict}`) and why a raw
checkpoint's parametrized `.parametrizations.weight.original{0,1}` keys become `.weight_{g,v}` in
the packaged bundle. This module is NOT re-run in t0016 -- its hardcoded `CKPT_DIR` points at
t0001's own checkpoint directory, which does not exist in this task's context. Do not invoke
`main()` here; read `convert_key()`/`MODULES` for reference only.

Original docstring: "Extract+convert decoder from any checkpoint path to Kokoro format. Usage:
script.py <ckpt> <out_name>"
"""

from pathlib import Path

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
    raise NotImplementedError(
        "extract_decoder_reference.py is a read-only reference copy for t0016 (see module "
        "docstring) -- it is not re-run. Use tasks.t0016_v3_recipe_recovery.code."
        "checkpoint_forensics for this task's own forensics."
    )


if __name__ == "__main__":
    main()
