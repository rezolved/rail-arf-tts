"""Step 11 (creative-thinking) falsification probe.

Milestone E's verdict (`results/v10_diagnosis.md`) attributes v10's clipped/saturated audio to the
HiFi-GAN `decoder` starting Stage 2 at a bad (partially shape-matched, effectively random)
initialization and never recovering in 17 epochs. This script tests a sharper, independently
falsifiable version of that claim directly, instead of only reasoning about it from weight norms:

    If v10's `decoder` is loaded from `epoch_2nd_00016.pth` as usual, but every OTHER module
    (diffusion, predictor_encoder, predictor, style_encoder, ...) is ALSO loaded from that same
    checkpoint (i.e. this is the real v10 primary checkpoint, unmodified) EXCEPT `decoder` is left
    at its fresh `build_model()` random initialization (never touched by any checkpoint at all) --
    does the resulting audio show the same clip_fraction / DC-dominant signature as the real v10
    output?

If yes: this shows 17 epochs of Stage-2 training moved the decoder so little from pure random
init that its output is statistically indistinguishable from never having been trained at all --
sharpening Verdict point 4 ("never produced a working vocoder at any point") from "undertrained"
to "training had ~no detectable effect on the decoder's output validity." It also directly answers
this step's assignment: since every module OTHER than decoder here is the real, checkpoint-loaded
(trained) diffusion/predictor_encoder/predictor/style_encoder, any resulting clipping cannot be
attributed to those modules -- isolating decoder as sufficient on its own to reproduce the failure
signature, independent of whether diffusion/predictor_encoder are well- or under-trained.

If no (random-decoder output is a different, non-clipped kind of bad): this would be a real update
to the verdict and must be flagged prominently for the `results` step.

Usage (through the isolated CPU venv, matching all other Milestone C-E runs)::

    code/.venv-styletts2/bin/python code/random_decoder_probe.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))
from tasks.t0013_v10_synthesis_quality_forensics.code.audio_quality_check import (  # noqa: E402
    check_audio_quality,
)
from tasks.t0013_v10_synthesis_quality_forensics.code.infer_styletts2 import (  # noqa: E402
    _match_stats,
    _rename_legacy_parametrization_keys,
    compute_style,
    synthesize,
)
from tasks.t0013_v10_synthesis_quality_forensics.code.paths import (  # noqa: E402
    ELEVENLABS_DAVID_DIR,
    RESULTS_AUDIO_DIR,
    RESULTS_DIR,
    STYLETTS2_DIR,
    V10_CONFIG_YML,
    V10_EPOCH16_CKPT,
)

TEXT = "This is a test of the Style T T S two inference harness."


def build_model_with_decoder_skipped(config_path: Path, checkpoint_path: Path):
    import yaml
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

    # Snapshot decoder's fresh build_model() random init BEFORE loading anything, so we can prove
    # afterward it was never overwritten.
    decoder_init_sd = {k: v.clone() for k, v in model.decoder.state_dict().items()}

    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    params = state["net"]
    load_report = []
    for key in model:
        if key not in params:
            continue
        if key == "decoder":
            load_report.append({"module": key, "action": "SKIPPED (left at random init)"})
            continue
        raw_sd = params[key]
        model_sd = model[key].state_dict()
        stripped_sd = {k.removeprefix("module."): v for k, v in raw_sd.items()}
        stripped_sd = _rename_legacy_parametrization_keys(stripped_sd)
        matched, stats = _match_stats(stripped_sd, model_sd)
        model[key].load_state_dict(matched, strict=False)
        load_report.append(
            {
                "module": key,
                "action": "loaded",
                "missing": stats.missing,
                "unexpected": stats.unexpected,
            }
        )
    _ = [model[key].eval() for key in model]

    # Verify decoder truly untouched.
    unchanged = all(
        torch.equal(decoder_init_sd[k], model.decoder.state_dict()[k]) for k in decoder_init_sd
    )
    return model, model_params, load_report, unchanged


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    os.chdir(STYLETTS2_DIR)

    model, model_params, load_report, decoder_unchanged = build_model_with_decoder_skipped(
        V10_CONFIG_YML, V10_EPOCH16_CKPT
    )
    assert decoder_unchanged, "decoder state dict was modified -- probe is invalid"

    ref_wavs = sorted(ELEVENLABS_DAVID_DIR.glob("*.wav"))[:3]
    # Reuse the same 3-clip concatenation approach as Milestone E (single David clips are too
    # short for StyleEncoder, see results/v10_diagnosis.md Key Question 6). Build one concat file.
    import soundfile as sfio

    audio_parts = []
    sr = None
    for p in ref_wavs:
        data, sr = sfio.read(str(p))
        audio_parts.append(data)
        audio_parts.append(np.zeros(int(0.2 * sr)))
    concat = np.concatenate(audio_parts)
    concat_path = RESULTS_DIR / "_probe_reference_concat.wav"
    sfio.write(str(concat_path), concat, sr)

    ref_s = compute_style(model, concat_path)
    wav, wall_time = synthesize(model, model_params, TEXT, ref_s)

    out_path = RESULTS_AUDIO_DIR / "probe_random_decoder.wav"
    sf.write(str(out_path), wav, 24000)
    concat_path.unlink()

    quality = check_audio_quality(out_path)
    result = {
        "description": (
            "v10 primary checkpoint (epoch_2nd_00016.pth) with decoder module left at "
            "build_model()'s fresh random init -- every other module (diffusion, "
            "predictor_encoder, predictor, style_encoder, bert, text_encoder, ...) loaded "
            "normally from the real trained checkpoint."
        ),
        "decoder_confirmed_untouched_by_checkpoint": decoder_unchanged,
        "load_report": load_report,
        "wall_time_seconds": wall_time,
        "audio_quality": {
            "rms": quality.rms,
            "peak": quality.peak,
            "silence_fraction": quality.silence_fraction,
            "spectral_flatness": quality.spectral_flatness,
            "clip_fraction": quality.clip_fraction,
            "is_likely_noise": quality.is_likely_noise,
        },
    }
    out_json = RESULTS_DIR / "random_decoder_probe.json"
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Wrote {out_path} and {out_json}")


if __name__ == "__main__":
    main()
