"""Instrumented StyleTTS2-native inference harness for t0013 Milestone C.

Adapted from `code/kikiri-tts/StyleTTS2/Demo/Inference_LibriTTS.ipynb` (cells 2-16), run through
`code/.venv-styletts2` (isolated CPU venv, `torch==2.5.1` -- see `plan/plan.md` Milestone C step
7). Unlike `kokoro.KModel`/`KPipeline` (confirmed broken for a `hifigan`-decoder checkpoint, per
`research/research_summary.md` point 5), this goes through StyleTTS2's own native `models.py`,
which dispatches `Decoder` construction on `model_params.decoder.type` (`istftnet` or `hifigan`,
`models.py:build_model`) -- the correct path for both the v10 checkpoints and any control.

The checkpoint loader here deliberately does NOT reuse the notebook's own load loop (cell 12, which
silently swallows any exception into a `strict=False` fallback with no missing/unexpected
reporting) or `train_second_v10.py:load_checkpoint()`'s pass condition (raises only on
zero-matched). Per `research/research_summary.md` point 4 and `plan/plan.md` step 8: log
missing/unexpected key counts PER MODULE for both the primary (raw) load attempt and the
`module.`-stripped fallback, and hard-fail on any nonzero missing/unexpected on a core module --
this project has already hit the silent-partial-match anti-pattern twice (t0009's training loader,
t0008's `adapters.load_kokoro_model_with_checkpoint`).

Usage::

    .venv-styletts2/bin/python code/infer_styletts2.py \\
        --checkpoint-path code/kikiri-tts/StyleTTS2/Models/LibriTTS/epochs_2nd_00020.pth \\
        --config-path code/kikiri-tts/StyleTTS2/Models/LibriTTS/config.yml \\
        --text "This is a test." \\
        --reference-audio code/kikiri-tts/StyleTTS2/Demo/reference_audio/1221-135767-0014.wav \\
        --output-wav results/audio_samples/control_epochs_2nd_00020.wav
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from munch import Munch

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))
from tasks.t0013_v10_synthesis_quality_forensics.code.paths import (  # noqa: E402
    RESULTS_DIR,
    STYLETTS2_DIR,
)

sys.path.insert(0, str(STYLETTS2_DIR))

# StyleTTS2-native imports -- must happen after sys.path.insert and after chdir (below), matching
# the notebook's own `%cd ..` convention, since several call sites resolve config-relative paths
# (Utils/ASR/*, Utils/JDC/*, Utils/PLBERT/) against the process cwd, not against this file's
# location.
# ruff: noqa: E402

# Core modules that must be fully covered by any StyleTTS2 stage-2 checkpoint (same list used by
# code/inspect_checkpoint.py -- kept in sync manually, both are small task-local scripts).
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

MEL_MEAN = -4.0
MEL_STD = 4.0
SAMPLE_RATE = 24000


@dataclass(frozen=True, slots=True)
class AttemptStats:
    matched: int
    missing: int
    unexpected: int


@dataclass(frozen=True, slots=True)
class ModuleLoadResult:
    module: str
    is_core: bool
    num_model_params: int
    num_ckpt_params: int
    primary: AttemptStats
    fallback: AttemptStats
    used_fallback: bool
    final_missing: int
    final_unexpected: int


def _rename_legacy_parametrization_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename classic `torch.nn.utils.{weight_norm,spectral_norm}` state dict keys to the newer
    `torch.nn.utils.parametrizations.{weight_norm,spectral_norm}` naming that `models.py` actually
    constructs (its top-level import, line 13, uses the `parametrizations` module).

    Discovered during this task's own control-checkpoint run: the officially released 2023
    `epochs_2nd_00020.pth` was saved under the classic (pre-parametrize-API) normalization
    wrappers; this fork's `models.py` imports the newer `parametrizations` versions, which
    register different key names for the mathematically identical reparametrization. Confirmed
    empirically both renames are pure (shape-preserving, no numerical transform needed):

    * weight_norm: `{name}.weight_g`/`{name}.weight_v` -> `{name}.parametrizations.weight.
      original0`/`original1`. E.g. `predictor.F0.1.conv1`: control's `weight_g`/`weight_v` are
      `(256,1,1)`/`(256,512,3)`, exactly matching v10's `original0`/`original1` shapes.
    * spectral_norm: `{name}.weight_orig`/`{name}.weight_u`/`{name}.weight_v` ->
      `{name}.parametrizations.weight.original`/`{name}.parametrizations.weight.0._u`/
      `{name}.parametrizations.weight.0._v`. E.g. `predictor_encoder.shared.0`: control's
      `weight_orig`/`weight_u`/`weight_v` are `(64,1,3,3)`/`(64,)`/`(9,)`, exactly matching v10's
      `parametrizations.weight.original`/`.0._u`/`.0._v` shapes.

    v10's own checkpoints were trained under this fork's current `models.py` already, so they
    already use the new naming and both renames are a no-op for them -- this exists purely to make
    the external control checkpoint loadable through the same harness code, unmodified.
    """
    # `.weight_v` is ambiguous on its own -- weight_norm and spectral_norm both use that suffix
    # for a DIFFERENT tensor (direction vector vs. power-iteration buffer, different shapes,
    # different new-API targets). Disambiguate by prefix: a prefix with a `.weight_orig` sibling
    # key is spectral_norm; a prefix with a `.weight_g` sibling key is weight_norm.
    spectral_norm_prefixes = {k[: -len(".weight_orig")] for k in sd if k.endswith(".weight_orig")}
    renamed: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        prefix = k.rsplit(".", 1)[0] if "." in k else ""
        if k.endswith(".weight_g"):
            new_key = k[: -len(".weight_g")] + ".parametrizations.weight.original0"
        elif k.endswith(".weight_orig"):
            new_key = k[: -len(".weight_orig")] + ".parametrizations.weight.original"
        elif k.endswith(".weight_u"):
            new_key = k[: -len(".weight_u")] + ".parametrizations.weight.0._u"
        elif k.endswith(".weight_v") and prefix in spectral_norm_prefixes:
            new_key = k[: -len(".weight_v")] + ".parametrizations.weight.0._v"
        elif k.endswith(".weight_v"):
            new_key = k[: -len(".weight_v")] + ".parametrizations.weight.original1"
        else:
            new_key = k
        renamed[new_key] = v
    return renamed


def _match_stats(
    ckpt_sd: dict[str, torch.Tensor], model_sd: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], AttemptStats]:
    matched = {k: v for k, v in ckpt_sd.items() if k in model_sd and model_sd[k].shape == v.shape}
    stats = AttemptStats(
        matched=len(matched),
        missing=len(model_sd) - len(matched),
        unexpected=len(ckpt_sd) - len(matched),
    )
    return matched, stats


def load_checkpoint_instrumented(model: Munch, checkpoint_path: Path) -> list[ModuleLoadResult]:
    """DP-aware loader that logs missing/unexpected key counts per module, per attempt.

    Every checkpoint observed in this project so far is DataParallel-trained (all `net[key]` keys
    are `module.`-prefixed), so the "primary" (raw) attempt is expected to show near-100%
    missing/unexpected for every module on every checkpoint -- that alone is not a defect signal,
    it is why the fallback exists. The diagnostic signal this task cares about is the FINAL
    (best-of-two) missing/unexpected count: nonzero there, on a core module, means the checkpoint's
    tensor shapes genuinely do not match this model's architecture.
    """
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    params = state["net"]
    results: list[ModuleLoadResult] = []
    for key in model:
        if key not in params:
            continue
        raw_sd = params[key]
        model_sd = model[key].state_dict()

        primary_matched, primary_stats = _match_stats(raw_sd, model_sd)
        stripped_sd = {k.removeprefix("module."): v for k, v in raw_sd.items()}
        stripped_sd = _rename_legacy_parametrization_keys(stripped_sd)
        fallback_matched, fallback_stats = _match_stats(stripped_sd, model_sd)

        fallback_total_bad = fallback_stats.missing + fallback_stats.unexpected
        primary_total_bad = primary_stats.missing + primary_stats.unexpected
        used_fallback = fallback_total_bad < primary_total_bad
        final_matched = fallback_matched if used_fallback else primary_matched
        final_stats = fallback_stats if used_fallback else primary_stats

        model[key].load_state_dict(final_matched, strict=False)

        is_core = key in CORE_MODULES
        result = ModuleLoadResult(
            module=key,
            is_core=is_core,
            num_model_params=len(model_sd),
            num_ckpt_params=len(raw_sd),
            primary=primary_stats,
            fallback=fallback_stats,
            used_fallback=used_fallback,
            final_missing=final_stats.missing,
            final_unexpected=final_stats.unexpected,
        )
        results.append(result)
        status = "OK" if final_stats.missing == 0 and final_stats.unexpected == 0 else "MISMATCH"
        print(
            f"{key}: primary(matched={primary_stats.matched},missing={primary_stats.missing},"
            f"unexpected={primary_stats.unexpected}) "
            f"fallback(matched={fallback_stats.matched},missing={fallback_stats.missing},"
            f"unexpected={fallback_stats.unexpected}) "
            f"used_fallback={used_fallback} [{status}]"
        )
        if is_core and (final_stats.missing > 0 or final_stats.unexpected > 0):
            raise RuntimeError(
                f"{key}: hard failure on core module -- final missing={final_stats.missing} "
                f"unexpected={final_stats.unexpected} (checkpoint {checkpoint_path})"
            )
    _ = [model[key].eval() for key in model]
    return results


def length_to_mask(lengths: torch.Tensor) -> torch.Tensor:
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    return torch.gt(mask + 1, lengths.unsqueeze(1))


def build_harness(config_path: Path, checkpoint_path: Path) -> tuple[Munch, Munch, object]:
    """Build the StyleTTS2 model from `config_path` and load `checkpoint_path` into it.

    Must run with cwd == STYLETTS2_DIR (caller's responsibility) so `Utils/ASR/*`, `Utils/JDC/*`,
    `Utils/PLBERT/*` relative paths inside the config resolve correctly, matching upstream's own
    `%cd ..` notebook convention.
    """
    from models import build_model, load_ASR_models, load_F0_models
    from utils import recursive_munch
    from Utils.PLBERT.util import load_plbert

    with open(config_path) as f:
        config = yaml.safe_load(f)
    text_aligner = load_ASR_models(config["ASR_path"], config["ASR_config"])
    pitch_extractor = load_F0_models(config["F0_path"])
    plbert = load_plbert(config["PLBERT_dir"])

    model_params = recursive_munch(config["model_params"])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to("cpu") for key in model]

    load_results = load_checkpoint_instrumented(model, checkpoint_path)

    load_log = {
        "checkpoint": str(checkpoint_path),
        "modules": [asdict(r) for r in load_results],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / f"load_log_{checkpoint_path.stem}.json"
    log_path.write_text(json.dumps(load_log, indent=2) + "\n")
    print(f"Wrote {log_path}")

    return model, model_params, log_path


def compute_style(model: Munch, path: Path) -> torch.Tensor:
    import librosa
    import torchaudio

    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80, n_fft=2048, win_length=1200, hop_length=300
    )
    wave, sr = librosa.load(str(path), sr=SAMPLE_RATE)
    audio, _ = librosa.effects.trim(wave, top_db=30)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    wave_tensor = torch.from_numpy(audio).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - MEL_MEAN) / MEL_STD
    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))
    return torch.cat([ref_s, ref_p], dim=1)


def synthesize(
    model: Munch,
    model_params: Munch,
    text: str,
    ref_s: torch.Tensor,
    *,
    alpha: float = 0.3,
    beta: float = 0.7,
    diffusion_steps: int = 5,
    embedding_scale: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Returns (waveform, synthesis_wall_time_seconds). Adapted from notebook cell 16."""
    import phonemizer
    from Modules.diffusion.sampler import ADPM2Sampler, DiffusionSampler, KarrasSchedule
    from nltk.tokenize import word_tokenize
    from text_utils import TextCleaner

    global_phonemizer = phonemizer.backend.EspeakBackend(
        language="en-us", preserve_punctuation=True, with_stress=True
    )
    text_cleaner = TextCleaner()
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False,
    )

    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = " ".join(ps)
    tokens = text_cleaner(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).unsqueeze(0)

    start = time.perf_counter()
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]])
        text_mask = length_to_mask(input_lengths)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(
            noise=torch.randn((1, 256)).unsqueeze(1),
            embedding=bert_dur,
            embedding_scale=embedding_scale,
            features=ref_s,
            num_steps=diffusion_steps,
        ).squeeze(1)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]
        ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
        s = beta * s + (1 - beta) * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0)
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        f0_pred, n_pred = model.predictor.F0Ntrain(en, s)

        asr = t_en @ pred_aln_trg.unsqueeze(0)
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr, f0_pred, n_pred, ref.squeeze().unsqueeze(0))
    wall_time = time.perf_counter() - start
    wav = out.squeeze().cpu().numpy()[..., :-50]  # trailing-pulse trim, per upstream comment
    return wav, wall_time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--output-wav", type=Path, required=True)
    parser.add_argument("--diffusion-steps", type=int, default=5)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path.resolve()
    config_path = args.config_path.resolve()
    reference_audio = args.reference_audio.resolve()
    output_wav = args.output_wav.resolve()

    torch.manual_seed(0)
    np.random.seed(0)

    os.chdir(STYLETTS2_DIR)
    model, model_params, _log_path = build_harness(config_path, checkpoint_path)

    ref_s = compute_style(model, reference_audio)
    wav, wall_time = synthesize(
        model, model_params, args.text, ref_s, diffusion_steps=args.diffusion_steps
    )

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_wav), wav, SAMPLE_RATE)
    duration_s = len(wav) / SAMPLE_RATE
    rtf = wall_time / duration_s if duration_s > 0 else None
    print(
        f"Wrote {output_wav} ({duration_s:.2f}s audio, synth wall time {wall_time:.2f}s, rtf={rtf})"
    )

    timing_path = output_wav.with_suffix(".timing.json")
    timing_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "wall_time_seconds": wall_time,
                "duration_seconds": duration_s,
                "rtf": rtf,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
