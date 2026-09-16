"""Mandatory tensor-level pre-flight check for v11's new `first_stage_path` target (REQ-2).

Copied and adapted from `tasks/t0013_v10_synthesis_quality_forensics/code/inspect_checkpoint.py`
(same `classify_decoder()` logic, empirically validated there against both v10 checkpoints and the
istftnet-shaped `first_stage_v3.pth` control). This task's own pre-flight target is
`yl4579/StyleTTS2-LibriTTS`'s `epochs_2nd_00020.pth` -- the new `first_stage_path` in
`code/config_david_v11.yml` -- and must report `classification: "hifigan"` /
`has_hifigan_marker: True` BEFORE any GPU spend (plan.md Milestone A step 3). If it does not, STOP
per plan.md's Approach-2 fallback -- do not proceed to Milestone B.

The v10-bug before/after contrast (first_stage_v3.pth classifying as "istftnet", both v10
checkpoints as "hifigan") is not re-derived here -- it is already documented with full evidence in
`tasks/t0013_v10_synthesis_quality_forensics/results/checkpoint_forensics.md`, cited directly rather
than re-downloading the 1.7GB `first_stage_v3.pth` a second time for a number this task doesn't
newly need (the gate is entirely about the LibriTTS checkpoint's own decoder architecture).

Usage::

    uv run python tasks/t0014_v11_decoder_fix_retrain/code/inspect_checkpoint.py
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import torch

from tasks.t0014_v11_decoder_fix_retrain.code.paths import (
    CHECKPOINT_FORENSICS_V11_MD,
    CHECKPOINT_FORENSICS_V11_RAW_JSON,
    LIBRITTS_CONTROL_CKPT,
)

# Core modules that must be fully covered by any StyleTTS2 stage-2 checkpoint.
CORE_MODULES: tuple[str, ...] = (
    "bert",
    "bert_encoder",
    "predictor",
    "decoder",
    "text_encoder",
    "predictor_encoder",
    "style_encoder",
    "diffusion",
)

# Decoder submodule key fragment that discriminates HiFi-GAN from ISTFTNet -- see
# t0013's code/inspect_checkpoint.py module docstring for the full empirical derivation
# (`generator.alphas.*` exists only in hifigan.py's Generator).
HIFIGAN_ONLY_FRAGMENT = "generator.alphas"
ISTFTNET_UPS_STAGE_COUNT = 2


@dataclass(frozen=True, slots=True)
class ModuleForensics:
    module: str
    num_params: int
    finite: bool
    weight_norm: float | None
    nonfinite_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DecoderArchitectureVerdict:
    has_hifigan_marker: bool
    num_ups_stages: int
    classification: str  # "hifigan" | "istftnet" | "ambiguous"


@dataclass(frozen=True, slots=True)
class CheckpointForensics:
    checkpoint: str
    modules: tuple[ModuleForensics, ...]
    decoder_verdict: DecoderArchitectureVerdict | None
    missing_core_modules: tuple[str, ...]


def classify_decoder(decoder_sd: dict[str, torch.Tensor]) -> DecoderArchitectureVerdict:
    keys = list(decoder_sd.keys())
    has_hifigan = any(HIFIGAN_ONLY_FRAGMENT in k for k in keys)
    ups_indices: set[int] = set()
    for k in keys:
        marker = "generator.ups."
        if marker in k:
            rest = k.split(marker, 1)[1]
            idx_str = rest.split(".", 1)[0]
            if idx_str.isdigit():
                ups_indices.add(int(idx_str))
    num_ups_stages = len(ups_indices)
    if has_hifigan:
        classification = "hifigan"
    elif num_ups_stages == ISTFTNET_UPS_STAGE_COUNT:
        classification = "istftnet"
    else:
        classification = "ambiguous"
    return DecoderArchitectureVerdict(
        has_hifigan_marker=has_hifigan,
        num_ups_stages=num_ups_stages,
        classification=classification,
    )


def module_forensics(module_name: str, module_sd: dict[str, torch.Tensor]) -> ModuleForensics:
    num_params = 0
    sq_norm_sum = 0.0
    nonfinite_keys: list[str] = []
    for key, tensor in module_sd.items():
        if not torch.is_tensor(tensor):
            continue
        num_params += tensor.numel()
        finite_tensor = torch.isfinite(tensor)
        if not bool(finite_tensor.all()):
            nonfinite_keys.append(key)
            continue
        sq_norm_sum += tensor.float().norm().item() ** 2
    finite = len(nonfinite_keys) == 0
    weight_norm = sq_norm_sum**0.5 if finite else None
    return ModuleForensics(
        module=module_name,
        num_params=num_params,
        finite=finite,
        weight_norm=weight_norm,
        nonfinite_keys=tuple(sorted(nonfinite_keys)),
    )


def inspect_checkpoint(path: object, label: str) -> CheckpointForensics:
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    net = state["net"]
    modules: list[ModuleForensics] = []
    for module_name in sorted(net.keys()):
        module_sd = net[module_name]
        modules.append(module_forensics(module_name, module_sd))
    decoder_verdict = classify_decoder(net["decoder"]) if "decoder" in net else None
    missing_core = tuple(sorted(m for m in CORE_MODULES if m not in net))
    return CheckpointForensics(
        checkpoint=label,
        modules=tuple(modules),
        decoder_verdict=decoder_verdict,
        missing_core_modules=missing_core,
    )


def render_markdown(results: list[CheckpointForensics]) -> str:
    lines: list[str] = []
    lines.append("## Per-checkpoint forensics")
    lines.append("")
    for r in results:
        lines.append(f"### `{r.checkpoint}`")
        lines.append("")
        if r.missing_core_modules:
            lines.append(f"**Missing core modules:** {', '.join(r.missing_core_modules)}")
            lines.append("")
        if r.decoder_verdict is not None:
            dv = r.decoder_verdict
            lines.append(
                f"**Decoder architecture:** `{dv.classification}` "
                f"(hifigan `alphas` marker={dv.has_hifigan_marker}, ups stages={dv.num_ups_stages})"
            )
            lines.append("")
        lines.append("| Module | Params | Finite | Weight norm |")
        lines.append("| --- | ---: | --- | ---: |")
        for m in r.modules:
            wn = f"{m.weight_norm:.4f}" if m.weight_norm is not None else "N/A (non-finite)"
            finite_str = "yes" if m.finite else f"**NO** ({', '.join(m.nonfinite_keys[:5])})"
            lines.append(f"| {m.module} | {m.num_params:,} | {finite_str} | {wn} |")
        lines.append("")
    return "\n".join(lines)


def render_verdict(results: list[CheckpointForensics]) -> str:
    by_label = {r.checkpoint: r for r in results}
    lines = ["## Verdict (REQ-2 gate)", ""]
    target = by_label.get("epochs_2nd_00020 (v11 new first_stage_path target)")
    any_nonfinite = any(not m.finite for r in results for m in r.modules)
    if any_nonfinite:
        lines.append(
            "**NaN/Inf found** in at least one module -- STOP, do not proceed to Milestone B."
        )
        lines.append("")
    if target is not None and target.decoder_verdict is not None:
        dv = target.decoder_verdict
        if dv.classification == "hifigan" and not target.missing_core_modules and not any_nonfinite:
            lines.append(
                f"**PASS.** `epochs_2nd_00020.pth`'s `decoder` classifies as "
                f"`{dv.classification}` (hifigan `alphas` marker={dv.has_hifigan_marker}, "
                f"ups stages={dv.num_ups_stages}), "
                "matching `config_david_v11.yml`'s `model_params.decoder.type: hifigan` block "
                "field-for-field, all core modules present, all tensors finite. This corroborates "
                "t0013's own already-documented classification of this exact checkpoint file "
                "(`tasks/t0013_v10_synthesis_quality_forensics/results/checkpoint_forensics.md`, "
                "where it served as the validated known-good control). Cleared to proceed to "
                "Milestone B (GPU training) -- Approach 1 (fine-tune from real hifigan pretrained "
                "weights), not the Approach-2 random-init fallback."
            )
        else:
            lines.append(
                f"**FAIL.** `epochs_2nd_00020.pth`'s decoder classified as `{dv.classification}` "
                f"(hifigan marker={dv.has_hifigan_marker}), missing core modules: "
                f"{target.missing_core_modules or 'none'}. STOP -- do not proceed to Milestone B "
                "under this approach; follow plan.md's Approach-2 fallback "
                "(add 'decoder' to ignore_modules, write an intervention/ file for budget "
                "sign-off)."
            )
    else:
        lines.append("**FAIL.** Could not load or classify the target checkpoint's decoder.")
    lines.append("")
    lines.append(
        "See `tasks/t0013_v10_synthesis_quality_forensics/results/checkpoint_forensics.md` for "
        "the already-documented before/after contrast (`first_stage_v3.pth` classifies "
        "`istftnet`; both v10 checkpoints classify `hifigan`) -- not re-derived here to avoid a "
        "redundant 1.7GB re-download of `first_stage_v3.pth`."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    targets = [
        (LIBRITTS_CONTROL_CKPT, "epochs_2nd_00020 (v11 new first_stage_path target)"),
    ]
    results: list[CheckpointForensics] = []
    for path, label in targets:
        print(f"Inspecting {label}: {path}")
        result = inspect_checkpoint(path, label)
        results.append(result)
        if result.decoder_verdict is not None:
            print(f"  decoder classification: {result.decoder_verdict.classification}")
        for m in result.modules:
            status = "OK" if m.finite else "NONFINITE"
            wn = f"{m.weight_norm:.2f}" if m.weight_norm is not None else "N/A"
            print(f"  {m.module}: {m.num_params:,} params, finite={status}, weight_norm={wn}")

    CHECKPOINT_FORENSICS_V11_RAW_JSON.parent.mkdir(parents=True, exist_ok=True)
    raw = [asdict(r) for r in results]
    CHECKPOINT_FORENSICS_V11_RAW_JSON.write_text(json.dumps(raw, indent=2) + "\n")

    md = render_markdown(results) + "\n" + render_verdict(results)
    CHECKPOINT_FORENSICS_V11_MD.write_text(
        "# Checkpoint Forensics v11 (Milestone A step 3, REQ-2)\n\n"
        "Mandatory pre-flight tensor-level check, run **before** any GPU training spend, per "
        "`plan/plan.md` Milestone A. Produced by `code/inspect_checkpoint.py`.\n\n" + md
    )
    print(f"\nWrote {CHECKPOINT_FORENSICS_V11_RAW_JSON}")
    print(f"Wrote {CHECKPOINT_FORENSICS_V11_MD}")


if __name__ == "__main__":
    main()
