"""Extract and convert a five-module decoder checkpoint from a raw StyleTTS2 .pth file.

Adapted from tasks/t0002_kokoro_v4_voicepack_decoder_package/code/extract_decoder_generic.py
(38-line original). The original used hardcoded CKPT_DIR constants; this version exposes
a clean function API accepting arbitrary input/output paths.

Five-module requirement: loading only 'decoder' into KModel causes duration explosions
(89 s for 9.6 s sentences). The production format is a dict with exactly these keys:
['bert', 'bert_encoder', 'predictor', 'text_encoder', 'decoder']. Weight-norm layers use
parametrizations (.parametrizations.weight.original0/1 → .weight_g/.weight_v).
DDP prefix ('module.') is stripped from all keys.
"""

from pathlib import Path

import torch

from tasks.t0008_tts_eval_harness_baselines.code.constants import CHECKPOINT_MODULES


def _convert_key(k: str) -> str:
    if k.startswith("module."):
        k = k[7:]
    k = k.replace(".parametrizations.weight.original0", ".weight_g")
    k = k.replace(".parametrizations.weight.original1", ".weight_v")
    return k


def extract(*, ckpt_path: Path, out_path: Path) -> dict[str, int]:
    """Extract five-module state dict from a raw StyleTTS2 checkpoint.

    Args:
        ckpt_path: Path to the raw .pth file (StyleTTS2 multi-GPU training checkpoint).
        out_path: Destination .pth file for the packaged five-module dict.

    Returns:
        Mapping of module name → parameter count for verification.

    Raises:
        KeyError: if expected modules are missing from the checkpoint.
        AssertionError: if output keys don't match CHECKPOINT_MODULES exactly.
    """
    raw: dict[str, object] = torch.load(
        str(ckpt_path),
        map_location="cpu",
        weights_only=False,
    )
    net: dict[str, object] = raw["net"]  # type: ignore[index]
    val_loss = raw.get("val_loss")
    epoch = raw.get("epoch")
    print(f"{ckpt_path.name}: val_loss={val_loss}, epoch={epoch}")

    out_sd: dict[str, dict[str, object]] = {
        mod: {_convert_key(k): v for k, v in net[mod].items()}  # type: ignore[union-attr]
        for mod in CHECKPOINT_MODULES
    }
    assert sorted(out_sd.keys()) == sorted(CHECKPOINT_MODULES), (
        f"Extracted keys {sorted(out_sd.keys())} != expected {sorted(CHECKPOINT_MODULES)}"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_sd, str(out_path))
    print(f"Saved → {out_path}")

    counts: dict[str, int] = {}
    for mod, sd in out_sd.items():
        counts[mod] = len(sd)
        print(f"  {mod}: {len(sd)} params")
    return counts


def validate_packaged(ckpt_path: Path) -> None:
    """Load a packaged .pth and assert keys are exactly CHECKPOINT_MODULES.

    Raises:
        AssertionError: if keys don't match.
    """
    sd: object = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    assert isinstance(sd, dict), f"Expected dict, got {type(sd)}"
    actual_keys = sorted(sd.keys())
    expected_keys = sorted(CHECKPOINT_MODULES)
    assert actual_keys == expected_keys, (
        f"Packaged checkpoint has wrong keys:\n  actual={actual_keys}\n  expected={expected_keys}"
    )
    print(f"Validation OK: {ckpt_path.name} keys = {actual_keys}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract five-module decoder checkpoint.")
    parser.add_argument("ckpt_path", type=Path, help="Input raw .pth checkpoint")
    parser.add_argument("out_path", type=Path, help="Output packaged .pth path")
    args = parser.parse_args()
    extract(ckpt_path=args.ckpt_path, out_path=args.out_path)
    validate_packaged(args.out_path)
