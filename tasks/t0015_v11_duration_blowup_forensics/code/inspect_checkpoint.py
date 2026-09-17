"""Cheap tensor-level forensics for t0013 Milestone B.

Reads raw StyleTTS2 checkpoint tensors directly via ``torch.load`` -- no StyleTTS2 source is
imported and no inference is run, so this needs nothing beyond the main project's existing
``.venv`` (already pins ``torch>=2.12.1``). It confirms or refutes the architecture-mismatch
hypothesis (``config_david_v10.yml``'s ``hifigan`` decoder vs. ``first_stage_v3.pth``'s
``istftnet``-shaped first-stage checkpoint) in minutes, before the CPU inference harness
(Milestone C) is ever built. See ``research/research_summary.md`` points 1-3 and
``plan/plan.md`` Milestone B (steps 4-6).

Usage::

    uv run python tasks/t0013_v10_synthesis_quality_forensics/code/inspect_checkpoint.py
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import torch

from tasks.t0013_v10_synthesis_quality_forensics.code.paths import (
    CHECKPOINT_FORENSICS_MD,
    CHECKPOINT_FORENSICS_RAW_JSON,
    FIRST_STAGE_V3_CKPT,
    V10_EPOCH14_CKPT,
    V10_EPOCH16_CKPT,
)

# Core modules that must be fully covered by any StyleTTS2 stage-2 checkpoint (research_summary.md
# point 4 / plan.md step 8): a nonzero missing/unexpected count on any of these is a hard failure.
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

# Decoder submodule key fragments that discriminate HiFi-GAN from ISTFTNet, confirmed by DIRECT
# source inspection of code/kikiri-tts/StyleTTS2/Modules/{hifigan,istftnet}.py during this task's
# implementation step (plan.md step 4). The research-phase guess ("ups.*/resblocks.* only in
# HiFi-GAN") was WRONG -- both architectures register `ups`, `resblocks`, `conv_post`, `m_source`,
# `noise_convs`, `noise_res` under identical names. The real, empirically-confirmed discriminator:
#   * `generator.alphas.*` -- HiFi-GAN Generator ONLY (a ParameterList of per-stage AdaIN alphas;
#     istftnet.py's Generator has no `self.alphas` at all). Confirmed against both raw checkpoints
#     in this task: `first_stage_v3.pth`'s decoder has 375 keys, zero `alphas`; v10's decoder has
#     678 keys, 5 `alphas.{0..4}` keys.
# `istftnet.py`'s Generator ALSO registers `self.stft = TorchSTFT(...)` and
# `self.reflection_pad = nn.ReflectionPad1d(...)`, which looked like a second, symmetric
# discriminator during source reading -- but neither shows up in `state_dict()` at all:
# `TorchSTFT.window` is assigned via plain `self.window = torch.from_numpy(...)`, NOT
# `register_buffer`, and `ReflectionPad1d` has no learnable parameters. Confirmed empirically (zero
# `stft`/`reflection` keys in either checkpoint) -- this fragment is documented, not used for
# classification, so a future reader does not repeat the same wrong assumption.
HIFIGAN_ONLY_FRAGMENT = "generator.alphas"
# Upsample stage counts observed in this project: hifigan's upsample_rates=[10,5,3,2] -> 4 stages
# (config_david_v10.yml); every istftnet config's upsample_rates=[10,6] -> 2 stages.
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
    lines.append("## Ground truth: decoder key naming by architecture")
    lines.append("")
    lines.append(
        "Read directly from `code/kikiri-tts/StyleTTS2/Modules/{hifigan,istftnet}.py` during "
        "implementation (plan.md step 4), **not** assumed from research. Both `Generator` classes "
        "register `self.ups`, `self.resblocks`, `self.conv_post`, `self.m_source`, "
        "`self.noise_convs`, `self.noise_res` under identical names -- the research-phase guess "
        "that `ups.*`/`resblocks.*` alone would discriminate the two architectures was wrong. The "
        "actual discriminators found by reading both files' `Generator.__init__`:"
    )
    lines.append("")
    lines.append(
        "* `generator.alphas.*` (a `nn.ParameterList`) exists **only** in `hifigan.py`'s "
        "`Generator` -- `istftnet.py`'s `Generator` has no `self.alphas` at all. **This is the "
        "primary classification signal** and it is decisive: empirically, `first_stage_v3.pth`'s "
        "`decoder` state dict has 375 keys and zero `alphas.*`; both v10 checkpoints' `decoder` "
        "state dicts have 678 keys including `alphas.{0..4}`."
    )
    lines.append(
        "* A second, symmetric discriminator initially looked plausible while reading source: "
        "`istftnet.py`'s `Generator` also registers `self.stft = TorchSTFT(...)` and "
        "`self.reflection_pad = nn.ReflectionPad1d(...)`, entirely absent from `hifigan.py`. "
        "**This turned out not to work** -- neither shows up in `state_dict()` at all, confirmed "
        "empirically (zero `stft`/`reflection` keys in either checkpoint): `TorchSTFT.window` is "
        "assigned via plain `self.window = torch.from_numpy(...)`, not `register_buffer`, so it is "
        "never tracked as module state; `ReflectionPad1d` has no learnable parameters to begin "
        "with. Recorded here so a future reader does not repeat the same wrong assumption."
    )
    lines.append(
        "* Both architectures share `ups`/`resblocks`/`conv_post`/`noise_convs` key **names**, but "
        "the **shapes** differ: `conv_post` outputs 1 channel (raw waveform) in HiFi-GAN vs. "
        "`gen_istft_n_fft + 2` channels (magnitude+phase) in ISTFTNet; `noise_convs[0]` takes 1 "
        "input channel in HiFi-GAN vs. `gen_istft_n_fft + 2` in ISTFTNet. The **number** of "
        "`ups`/`resblocks` stages also differs with `upsample_rates` length -- empirically 4 "
        "stages (`generator.ups.{0..3}`) in both v10 checkpoints (matching "
        "`config_david_v10.yml`'s `[10,5,3,2]`) vs. 2 stages (`generator.ups.{0,1}`) in "
        "`first_stage_v3.pth` (matching every istftnet config's `[10,6]` in this project) -- a "
        "second, independent, corroborating "
        "signal used as a fallback classifier when the `alphas` marker is absent."
    )
    lines.append("")
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
    lines = ["## Verdict", ""]
    control = by_label.get("first_stage_v3 (control, istftnet-trained)")
    v16 = by_label.get("epoch_2nd_00016 (v10 primary)")
    v14 = by_label.get("epoch_2nd_00014 (v10 backup)")
    any_nonfinite = any(not m.finite for r in results for m in r.modules)
    if any_nonfinite:
        lines.append(
            "**NaN/Inf found** in at least one module -- see the per-checkpoint tables above for "
            "which module(s). This alone is sufficient to explain noise output, regardless of the "
            "architecture-mismatch finding below."
        )
        lines.append("")
    if control is not None and v16 is not None and v14 is not None:
        control_arch = control.decoder_verdict.classification if control.decoder_verdict else "?"
        v16_arch = v16.decoder_verdict.classification if v16.decoder_verdict else "?"
        v14_arch = v14.decoder_verdict.classification if v14.decoder_verdict else "?"
        if control_arch == "istftnet" and v16_arch == "hifigan" and v14_arch == "hifigan":
            lines.append(
                "**(a) Architecture-mismatch hypothesis CONFIRMED at the tensor level.** "
                f"`first_stage_v3.pth` (control) decoder classifies as `{control_arch}`, while "
                f"both v10 checkpoints (`epoch_2nd_00016`={v16_arch}, "
                f"`epoch_2nd_00014`={v14_arch}) classify as `hifigan`, matching "
                "`config_david_v10.yml`'s "
                "`model_params.decoder.type: hifigan`. This is consistent with the "
                "`train_second_v10.py:load_checkpoint()` finding (`decoder` is not in "
                "`ignore_modules`, so `first_stage_v3.pth`'s istftnet-shaped decoder weights were "
                "loaded with a partial shape match, leaving the HiFi-GAN generator's `ups`/"
                "`resblocks`/`alphas`/`conv_post` layers randomly initialized at the start of "
                "training) -- the HiFi-GAN vocoder proper had only 17 epochs to learn from "
                "scratch. This is a working hypothesis carried into Milestone E, not the task's "
                "final answer -- Milestones C-E still run regardless."
            )
        else:
            lines.append(
                "**(b) Architecture classification does not cleanly confirm the mismatch "
                f"hypothesis** (control={control_arch}, v10-epoch16={v16_arch}, "
                f"v10-epoch14={v14_arch}). Investigate other explanations (NaN/Inf, weight-norm "
                "collapse, or a harness bug found later in Milestone C-E)."
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    targets = [
        (V10_EPOCH16_CKPT, "epoch_2nd_00016 (v10 primary)"),
        (V10_EPOCH14_CKPT, "epoch_2nd_00014 (v10 backup)"),
        (FIRST_STAGE_V3_CKPT, "first_stage_v3 (control, istftnet-trained)"),
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

    CHECKPOINT_FORENSICS_RAW_JSON.parent.mkdir(parents=True, exist_ok=True)
    raw = [asdict(r) for r in results]
    CHECKPOINT_FORENSICS_RAW_JSON.write_text(json.dumps(raw, indent=2) + "\n")

    md = render_markdown(results) + "\n" + render_verdict(results)
    CHECKPOINT_FORENSICS_MD.write_text(
        "# Checkpoint Forensics (Milestone B)\n\n"
        "Cheap tensor-level falsifier, run **before** any inference harness code, per "
        "`plan/plan.md` Milestone B. Produced by `code/inspect_checkpoint.py`.\n\n" + md
    )
    print(f"\nWrote {CHECKPOINT_FORENSICS_RAW_JSON}")
    print(f"Wrote {CHECKPOINT_FORENSICS_MD}")


if __name__ == "__main__":
    main()
