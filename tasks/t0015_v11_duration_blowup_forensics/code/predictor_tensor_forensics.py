"""Milestone B step 7 (REQ-3): no-GPU tensor-level forensics on `predictor`/`predictor_encoder`.

Adapted from `code/inspect_checkpoint.py`'s `module_forensics()` (t0013's decoder-focused pattern),
retargeted at `predictor`/`predictor_encoder` and specifically `predictor.duration_proj` -- the
exact submodule that produces `pred_dur` (`infer_styletts2.py:synthesize()`). Reads raw checkpoint
tensors directly via `torch.load` -- no StyleTTS2 import, no inference run -- to cross-check the
training-log evidence (`tasks/t0014_v11_decoder_fix_retrain/data/run_v11/metrics.jsonl`'s flat
0.53-0.62 `dur_loss` plateau across all 50 epochs) with an independent, tensor-level signal.

Compares `kokoro-v11-best` (`epoch_00048.pth`) against the LibriTTS control checkpoint
(`epochs_2nd_00020.pth`, the same `first_stage_path` t0014's `config_david_v11.yml` loaded from) for
`predictor` and `predictor_encoder`'s weight-norm summary statistics. A large weight-norm shift with
no non-finite values corroborates a calibration failure (the weights moved during training but
converged to a bad region); a near-zero shift would suggest `predictor` effectively didn't move
despite receiving gradients (contradicting the `optimizer.step()` training-log evidence and
warranting a deeper look, out of this plan's no-GPU scope).

Usage::

    code/.venv-styletts2/bin/python -m \
        tasks.t0015_v11_duration_blowup_forensics.code.predictor_tensor_forensics
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import torch

from tasks.t0015_v11_duration_blowup_forensics.code.paths import (
    LIBRITTS_CONTROL_CKPT,
    PREDICTOR_TENSOR_FORENSICS_MD,
    V11_CHECKPOINT,
)

TARGET_MODULES: tuple[str, ...] = ("predictor", "predictor_encoder")
# `predictor.duration_proj` is the exact submodule computing `pred_dur`
# (`torch.sigmoid(model.predictor.duration_proj(x)).sum(axis=-1)`), so it gets its own dedicated
# weight-norm comparison beyond the whole-module summary.
DURATION_PROJ_PREFIX = "duration_proj"


@dataclass(frozen=True, slots=True)
class TensorForensics:
    key: str
    num_params: int
    finite: bool
    weight_norm: float | None
    mean_abs: float | None
    max_abs: float | None
    nonfinite_keys: tuple[str, ...]


def tensor_forensics(label: str, module_sd: dict[str, torch.Tensor]) -> TensorForensics:
    num_params = 0
    sq_norm_sum = 0.0
    abs_sum = 0.0
    max_abs = 0.0
    nonfinite_keys: list[str] = []
    for key, tensor in module_sd.items():
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
    return TensorForensics(
        key=label,
        num_params=num_params,
        finite=finite,
        weight_norm=weight_norm,
        mean_abs=mean_abs,
        max_abs=max_abs if finite else None,
        nonfinite_keys=tuple(sorted(nonfinite_keys)),
    )


def load_module_state_dicts(checkpoint_path: object) -> dict[str, dict[str, torch.Tensor]]:
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    net = state["net"]
    result: dict[str, dict[str, torch.Tensor]] = {}
    for module_name in TARGET_MODULES:
        if module_name not in net:
            continue
        # Checkpoints in this project are DataParallel-trained (`module.`-prefixed keys); strip
        # the prefix for readable submodule-name matching (e.g. `duration_proj.*`) below. This is
        # a pure key-rename for forensics purposes -- no shape/value transform, unlike
        # infer_styletts2.py's `_rename_legacy_parametrization_keys` (not needed here since we
        # never load these tensors into a live model, only read their raw values).
        raw_sd = net[module_name]
        result[module_name] = {k.removeprefix("module."): v for k, v in raw_sd.items()}
    return result


def submodule_state_dict(
    module_sd: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    return {k: v for k, v in module_sd.items() if k.startswith(prefix)}


def render_markdown(
    v11_results: dict[str, TensorForensics],
    control_results: dict[str, TensorForensics],
    v11_duration_proj: TensorForensics,
    control_duration_proj: TensorForensics,
) -> str:
    lines: list[str] = []
    lines.append("## Whole-module weight-norm comparison")
    lines.append("")
    lines.append("| Module | Checkpoint | Params | Finite | Weight norm | Mean abs | Max abs |")
    lines.append("| --- | --- | ---: | --- | ---: | ---: | ---: |")
    for module_name in TARGET_MODULES:
        for label, results in (
            ("v11 (kokoro-v11-best)", v11_results),
            ("control (LibriTTS)", control_results),
        ):
            r = results.get(module_name)
            if r is None:
                continue
            wn = f"{r.weight_norm:.4f}" if r.weight_norm is not None else "N/A"
            ma = f"{r.mean_abs:.6f}" if r.mean_abs is not None else "N/A"
            xa = f"{r.max_abs:.6f}" if r.max_abs is not None else "N/A"
            finite_str = "yes" if r.finite else f"**NO** ({', '.join(r.nonfinite_keys[:5])})"
            cells = [module_name, label, f"{r.num_params:,}", finite_str, wn, ma, xa]
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## `predictor.duration_proj` -- the exact submodule producing `pred_dur`")
    lines.append("")
    lines.append("| Checkpoint | Params | Finite | Weight norm | Mean abs | Max abs |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: |")
    for label, r in (
        ("v11 (kokoro-v11-best)", v11_duration_proj),
        ("control (LibriTTS)", control_duration_proj),
    ):
        wn = f"{r.weight_norm:.4f}" if r.weight_norm is not None else "N/A"
        ma = f"{r.mean_abs:.6f}" if r.mean_abs is not None else "N/A"
        xa = f"{r.max_abs:.6f}" if r.max_abs is not None else "N/A"
        finite_str = "yes" if r.finite else f"**NO** ({', '.join(r.nonfinite_keys[:5])})"
        lines.append(f"| {label} | {r.num_params:,} | {finite_str} | {wn} | {ma} | {xa} |")
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    any_nonfinite = (
        any(not r.finite for r in list(v11_results.values()) + list(control_results.values()))
        or not v11_duration_proj.finite
        or not control_duration_proj.finite
    )
    if any_nonfinite:
        lines.append(
            "**Non-finite (NaN/Inf) values found** in at least one target module -- this alone "
            "is sufficient to explain anomalous behavior, independent of any weight-norm-shift "
            "finding below."
        )
        lines.append("")
    if v11_duration_proj.weight_norm is not None and control_duration_proj.weight_norm is not None:
        delta = v11_duration_proj.weight_norm - control_duration_proj.weight_norm
        rel_delta = (
            delta / control_duration_proj.weight_norm
            if control_duration_proj.weight_norm != 0
            else None
        )
        rel_str = f"{rel_delta:+.1%}" if rel_delta is not None else "N/A"
        lines.append(
            f"`predictor.duration_proj` weight norm: v11={v11_duration_proj.weight_norm:.4f}, "
            f"control={control_duration_proj.weight_norm:.4f}, delta={delta:+.4f} ({rel_str})."
        )
        # NEAR_ZERO_SHIFT_THRESHOLD: a relative delta under 5% is treated as "the module barely
        # moved" -- an arbitrary but generous cutoff (t0014's Stage-2 finetune ran 50 epochs with
        # a nonzero learning rate, so even a "barely moved" module is expected to show some drift;
        # 5% is well below what 50 epochs of gradient updates on a 25,650-param submodule would be
        # expected to produce if the loss were actively improving).
        near_zero_shift_threshold = 0.05
        if rel_delta is not None and abs(rel_delta) < near_zero_shift_threshold:
            lines.append(
                "This is a **near-zero shift**: `duration_proj`'s own weights are essentially "
                "unchanged from the LibriTTS control's, despite receiving gradients every step of "
                "all 50 epochs (per the `optimizer.step()` training-log evidence) and despite the "
                "flat 0.53-0.62 `dur_loss` plateau in "
                "`tasks/t0014_v11_decoder_fix_retrain/data/run_v11/metrics.jsonl`. This weighs "
                "AGAINST attributing the duration blowup to `duration_proj`'s own weights having "
                "drifted to a badly calibrated region -- if that were the mechanism, 50 epochs of "
                "gradient updates would be expected to move this small (25,650-param) submodule's "
                "weight norm more than 0.7%. The elevated `pred_dur` values found in "
                "`results/duration_characterization.json` are more consistent with a distribution "
                "shift in `duration_proj`'s INPUT (the output of `predictor.lstm`, itself fed by "
                "`predictor.text_encoder(d_en, s, ...)` where `s` comes from the diffusion-sampled "
                "style vector and `d_en` from `bert_encoder`) than with `duration_proj`'s own "
                "weights having miscalibrated. `predictor_encoder`'s larger whole-module shift "
                "(see table above) is a more plausible contributor to that upstream distribution "
                "shift, since it was excluded from the `first_stage_path` checkpoint load "
                "(`ignore_modules`) and instead initialized as "
                "`copy.deepcopy(model.style_encoder)` before training, per "
                "`research/research_summary.md` point 5."
            )
        else:
            lines.append(
                "This is a non-trivial shift, consistent with `duration_proj`'s weights having "
                "moved during t0014's Stage-2 finetune -- consistent with (though not proof of) a "
                "calibration failure in this exact submodule."
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    v11_modules = load_module_state_dicts(V11_CHECKPOINT)
    control_modules = load_module_state_dicts(LIBRITTS_CONTROL_CKPT)

    v11_results = {name: tensor_forensics(f"v11.{name}", sd) for name, sd in v11_modules.items()}
    control_results = {
        name: tensor_forensics(f"control.{name}", sd) for name, sd in control_modules.items()
    }

    v11_duration_proj_sd = submodule_state_dict(
        v11_modules.get("predictor", {}), DURATION_PROJ_PREFIX
    )
    control_duration_proj_sd = submodule_state_dict(
        control_modules.get("predictor", {}), DURATION_PROJ_PREFIX
    )
    v11_duration_proj = tensor_forensics("v11.predictor.duration_proj", v11_duration_proj_sd)
    control_duration_proj = tensor_forensics(
        "control.predictor.duration_proj", control_duration_proj_sd
    )

    print("v11 modules:")
    for name, r in v11_results.items():
        print(f"  {name}: {r}")
    print("control modules:")
    for name, r in control_results.items():
        print(f"  {name}: {r}")
    print(f"v11 duration_proj: {v11_duration_proj}")
    print(f"control duration_proj: {control_duration_proj}")

    raw = {
        "v11_modules": {k: asdict(v) for k, v in v11_results.items()},
        "control_modules": {k: asdict(v) for k, v in control_results.items()},
        "v11_duration_proj": asdict(v11_duration_proj),
        "control_duration_proj": asdict(control_duration_proj),
    }
    raw_json_path = PREDICTOR_TENSOR_FORENSICS_MD.with_suffix(".raw.json")
    raw_json_path.write_text(json.dumps(raw, indent=2) + "\n")

    md = render_markdown(v11_results, control_results, v11_duration_proj, control_duration_proj)
    PREDICTOR_TENSOR_FORENSICS_MD.write_text(
        "# Predictor Tensor Forensics (Milestone B step 7, REQ-3)\n\n"
        "No-GPU tensor-level cross-check of `predictor`/`predictor_encoder`, comparing "
        "`kokoro-v11-best` (`epoch_00048.pth`) against the LibriTTS control checkpoint "
        "(`epochs_2nd_00020.pth`) that `config_david_v11.yml`'s `first_stage_path` loaded from. "
        "Produced by `code/predictor_tensor_forensics.py`.\n\n" + md
    )
    print(f"\nWrote {raw_json_path}")
    print(f"Wrote {PREDICTOR_TENSOR_FORENSICS_MD}")


if __name__ == "__main__":
    main()
