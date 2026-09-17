"""Milestone 2 (REQ-2, REQ-6, REQ-7, REQ-10): no-GPU checkpoint-shape forensics on the v3 bundle.

Adapted from `tasks/t0015_v11_duration_blowup_forensics/code/predictor_tensor_forensics.py`'s
`module_forensics()` pattern, generalized from 2 target modules (`predictor`, `predictor_encoder`)
to the 5 v3 Kokoro-API bundle modules (`bert`, `bert_encoder`, `predictor`, `text_encoder`,
`decoder`), and extended to enumerate every top-level key in the raw Stage 1 checkpoint's `net`
dict (not only the 5 bundle modules) so the `multispeaker` contradiction
(`best/config.json`=true vs t0009's confound table=true(assumed) vs t0006's
`config_david_v6c_stage2.yml` line 76=false) can be attacked with checkpoint-shape evidence rather
than picked by source preference.

Reads raw checkpoint tensors directly via `torch.load` -- no StyleTTS2 import, no inference run.
Compares `stage1/first_stage.pth` (label `stage1`, raw StyleTTS2 shape:
`{"net": {module_name: state_dict, ...}, "epoch": ..., "val_loss": ...}`) against
`best/david_v3_best_decoder_kokoro.pth` (label `best`, packaged Kokoro-API bundle shape:
`{module_name: state_dict, ...}` directly, keys already remapped by t0002's
`extract_decoder_generic.py::convert_key()` -- no `"net"` wrapper, no `module.` DP prefix).

Usage::

    uv run python -m tasks.t0016_v3_recipe_recovery.code.checkpoint_forensics
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from tasks.t0016_v3_recipe_recovery.code.paths import (
    CHECKPOINT_FORENSICS_MD,
    MODULE_WEIGHT_DELTA_RAW_JSON,
    V3_BEST_DECODER_CKPT,
    V3_STAGE1_CKPT,
)

TARGET_MODULES: tuple[str, ...] = ("bert", "bert_encoder", "predictor", "text_encoder", "decoder")

# NEAR_ZERO_SHIFT_THRESHOLD: the same 5% near-zero-shift cutoff
# `predictor_tensor_forensics.py` documents and uses -- well below what gradient updates would be
# expected to produce over a real Stage 2 finetune if the loss were actively improving.
NEAR_ZERO_SHIFT_THRESHOLD = 0.05


@dataclass(frozen=True, slots=True)
class ModuleForensics:
    label: str
    num_params: int
    finite: bool
    weight_norm: float | None
    mean_abs: float | None
    max_abs: float | None
    dp_prefix_present: bool
    nonfinite_keys: tuple[str, ...]


def sha256_file(path: Path) -> str:
    """Standalone SHA-256 helper, extracted from
    `tasks/t0009_stage2_training_failure_forensics/code/checkpoint_manager.py`'s
    `CheckpointManager._sha256()` static method (hashing logic only, not the full class, which is
    training-loop-oriented and not needed here)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def module_forensics(label: str, module_sd: dict[str, torch.Tensor]) -> ModuleForensics:
    num_params = 0
    sq_norm_sum = 0.0
    abs_sum = 0.0
    max_abs = 0.0
    nonfinite_keys: list[str] = []
    dp_prefix_present = any(k.startswith("module.") for k in module_sd)
    for raw_key, tensor in module_sd.items():
        key = raw_key.removeprefix("module.")
        if not torch.is_tensor(tensor):
            continue
        num_params += tensor.numel()
        finite_tensor = torch.isfinite(tensor)
        if not bool(finite_tensor.all()):
            nonfinite_keys.append(key)
            continue
        t = tensor.float()
        sq_norm_sum += t.norm().item() ** 2
        abs_sum += t.abs().sum().item()
        max_abs = max(max_abs, t.abs().max().item())
    finite = len(nonfinite_keys) == 0
    weight_norm = sq_norm_sum**0.5 if finite else None
    mean_abs = abs_sum / num_params if finite and num_params > 0 else None
    return ModuleForensics(
        label=label,
        num_params=num_params,
        finite=finite,
        weight_norm=weight_norm,
        mean_abs=mean_abs,
        max_abs=max_abs if finite else None,
        dp_prefix_present=dp_prefix_present,
        nonfinite_keys=tuple(sorted(nonfinite_keys)),
    )


def load_raw_net(checkpoint_path: Path) -> dict[str, dict[str, torch.Tensor]]:
    """Load a checkpoint and return its module-name -> state_dict mapping.

    Tolerates two different top-level checkpoint shapes (confirmed by direct inspection of
    `tasks/t0002_kokoro_v4_voicepack_decoder_package/code/extract_decoder_generic.py`):

    (a) a raw StyleTTS2 checkpoint: `{"net": {module_name: state_dict, ...}, "epoch": ...,
        "val_loss": ...}` -- `stage1/first_stage.pth`'s shape.
    (b) a packaged Kokoro-API bundle: `{module_name: state_dict, ...}` directly, no `"net"`
        wrapper -- `best/david_v3_best_decoder_kokoro.pth`'s shape.
    """
    raw = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    net = raw["net"] if isinstance(raw, dict) and "net" in raw else raw
    assert isinstance(net, dict), f"Unexpected checkpoint shape at {checkpoint_path}: {type(raw)}"
    return net


def all_top_level_keys(checkpoint_path: Path) -> dict[str, int]:
    """Every top-level key in the checkpoint's net dict, with each module's raw param count.

    Used for the multispeaker resolution: StyleTTS2's `build_model()` returns a `nets` Munch with
    more top-level keys than the 5 bundled modules (`predictor_encoder`, `style_encoder`,
    `diffusion`, `text_aligner`, `pitch_extractor`, `mpd`, `msd`, `wd`), and `multispeaker` only
    affects `diffusion.diffusion.net` (see `## Multispeaker Resolution` below) -- entirely outside
    the 5-module bundle.
    """
    net = load_raw_net(checkpoint_path)
    counts: dict[str, int] = {}
    for key, sd in net.items():
        if not isinstance(sd, dict):
            continue
        n = 0
        for tensor in sd.values():
            if torch.is_tensor(tensor):
                n += tensor.numel()
        counts[key] = n
    return counts


def render_markdown(
    stage1_results: dict[str, ModuleForensics],
    best_results: dict[str, ModuleForensics],
    stage1_sha256: str,
    best_sha256: str,
    stage1_all_keys: dict[str, int],
) -> str:
    lines: list[str] = []
    lines.append("## Module-by-Module Weight-Norm Comparison (REQ-2, REQ-7)")
    lines.append("")
    lines.append(
        "| Module | Params | Finite | Weight norm (stage1) | Weight norm (best) "
        "| Relative delta | Changed (yes/no) | DP prefix present |"
    )
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | --- | --- |")
    for module_name in TARGET_MODULES:
        s1 = stage1_results.get(module_name)
        b = best_results.get(module_name)
        if s1 is None or b is None:
            lines.append(f"| {module_name} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        finite_str = "yes" if (s1.finite and b.finite) else "**NO**"
        if s1.weight_norm is not None and b.weight_norm is not None and s1.weight_norm != 0:
            rel_delta = (b.weight_norm - s1.weight_norm) / s1.weight_norm
            rel_str = f"{rel_delta:+.2%}"
            changed = "yes" if abs(rel_delta) >= NEAR_ZERO_SHIFT_THRESHOLD else "no"
        else:
            rel_str = "N/A"
            changed = "N/A"
        dp_str = "yes" if s1.dp_prefix_present else "no"
        wn_s1 = f"{s1.weight_norm:.4f}" if s1.weight_norm is not None else "N/A"
        wn_b = f"{b.weight_norm:.4f}" if b.weight_norm is not None else "N/A"
        lines.append(
            f"| {module_name} | {s1.num_params:,} | {finite_str} | {wn_s1} | {wn_b} "
            f"| {rel_str} | {changed} | {dp_str} |"
        )
    lines.append("")

    lines.append("## Byte-Identity Check (REQ-10)")
    lines.append("")
    lines.append(f"* `stage1/first_stage.pth` local SHA-256: `{stage1_sha256}`")
    lines.append(f"* `best/david_v3_best_decoder_kokoro.pth` local SHA-256: `{best_sha256}`")
    lines.append(
        "* VM `first_stage_v3.pth` SHA-256: see `data/vm_inventory/inventory.json` "
        "`first_stage_v3_remote_sha256` field."
    )
    lines.append("")

    lines.append("## All Top-Level `net` Keys in `stage1/first_stage.pth` (multispeaker scope)")
    lines.append("")
    lines.append("| Top-level key | Raw param count | In 5-module bundle? |")
    lines.append("| --- | ---: | --- |")
    for key, n in sorted(stage1_all_keys.items()):
        in_bundle = "yes" if key in TARGET_MODULES else "no"
        lines.append(f"| {key} | {n:,} | {in_bundle} |")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    stage1_net = load_raw_net(V3_STAGE1_CKPT)
    best_net = load_raw_net(V3_BEST_DECODER_CKPT)

    stage1_results = {
        name: module_forensics(f"stage1.{name}", stage1_net[name])
        for name in TARGET_MODULES
        if name in stage1_net
    }
    best_results = {
        name: module_forensics(f"best.{name}", best_net[name])
        for name in TARGET_MODULES
        if name in best_net
    }

    stage1_sha256 = sha256_file(V3_STAGE1_CKPT)
    best_sha256 = sha256_file(V3_BEST_DECODER_CKPT)
    stage1_all_keys = all_top_level_keys(V3_STAGE1_CKPT)

    print("stage1 modules:")
    for name, r in stage1_results.items():
        print(f"  {name}: {r}")
    print("best modules:")
    for name, r in best_results.items():
        print(f"  {name}: {r}")
    print(f"stage1 sha256: {stage1_sha256}")
    print(f"best sha256: {best_sha256}")
    print(f"stage1 all top-level net keys: {stage1_all_keys}")

    relative_deltas: dict[str, float] = {}
    for module_name in TARGET_MODULES:
        s1 = stage1_results.get(module_name)
        b = best_results.get(module_name)
        if s1 is not None and b is not None and s1.weight_norm and s1.weight_norm != 0:
            assert b.weight_norm is not None
            relative_deltas[module_name] = (b.weight_norm - s1.weight_norm) / s1.weight_norm

    raw: dict[str, object] = {
        "stage1_modules": {k: asdict(v) for k, v in stage1_results.items()},
        "best_modules": {k: asdict(v) for k, v in best_results.items()},
        "stage1_sha256": stage1_sha256,
        "best_sha256": best_sha256,
        "stage1_all_top_level_net_keys": stage1_all_keys,
        "near_zero_shift_threshold": NEAR_ZERO_SHIFT_THRESHOLD,
        "target_modules": list(TARGET_MODULES),
        "relative_deltas": relative_deltas,
    }

    MODULE_WEIGHT_DELTA_RAW_JSON.parent.mkdir(parents=True, exist_ok=True)
    MODULE_WEIGHT_DELTA_RAW_JSON.write_text(json.dumps(raw, indent=2) + "\n")

    md = render_markdown(stage1_results, best_results, stage1_sha256, best_sha256, stage1_all_keys)
    CHECKPOINT_FORENSICS_MD.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FORENSICS_MD.write_text(
        "# v3 Checkpoint Forensics (Milestone 2, REQ-2/REQ-6/REQ-7/REQ-10)\n\n"
        "No-GPU tensor-level cross-check of the v3 Stage 2 bundle, comparing "
        "`stage1/first_stage.pth` against `best/david_v3_best_decoder_kokoro.pth` across all 5 "
        "Kokoro-API bundle modules. Produced by `code/checkpoint_forensics.py`.\n\n" + md
    )
    print(f"\nWrote {MODULE_WEIGHT_DELTA_RAW_JSON}")
    print(f"Wrote {CHECKPOINT_FORENSICS_MD}")


if __name__ == "__main__":
    main()
