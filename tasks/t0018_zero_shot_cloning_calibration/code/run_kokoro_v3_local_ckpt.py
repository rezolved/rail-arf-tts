"""One-off runner for `kokoro_v3_bundle` using LOCALLY COPIED checkpoint files.

**Discovered during implementation:** loading the v3 decoder checkpoint (313 MB) via
`torch.load` directly from the Azure Files-backed `/mnt/cache/persist/.../repo/tasks/t0006.../data/`
path hangs indefinitely (mirrors the F5-TTS hang -- see
`intervention/f5_tts_smoke_gate_failed.md`): `torch.load`'s many small reads over this SMB mount
are pathologically slow, even though a **plain sequential `cp` of the same file completes in
seconds**. Workaround: copy the checkpoint files to local disk (`/tmp/kokoro_ckpt/`, confirmed fast
via `cp`) first, then load from there.

Usage (from the VM, in the venv with kokoro installed)::

    python -m tasks.t0018_zero_shot_cloning_calibration.code.run_kokoro_v3_local_ckpt \
        --decoder-path /tmp/kokoro_ckpt/david_v3_best_decoder_kokoro.pth \
        --voicepack-path /tmp/kokoro_ckpt/david_v3_best_voicepack.pt \
        --prompt-set both --n-warmup 50
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--decoder-path", type=Path, required=True)
    p.add_argument("--voicepack-path", type=Path, required=True)
    p.add_argument("--prompt-set", default="both")
    p.add_argument("--n-warmup", type=int, default=50)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    from tasks.t0003_kokoro_v5_phoneme_data.code.build_pipeline import build_pipeline
    from tasks.t0008_tts_eval_harness_baselines.code.adapters import (
        kokoro_v3_bundle,
        load_kokoro_model_with_checkpoint,
    )
    from tasks.t0008_tts_eval_harness_baselines.code.harness import get_prompts_by_set
    from tasks.t0018_zero_shot_cloning_calibration.code.paths import (
        RESULTS_AUDIO_HARNESS_DIR,
        RESULTS_DIR,
    )
    from tasks.t0018_zero_shot_cloning_calibration.code.run_eval_zeroshot import (
        _run_one_condition,
        _write_environment_record,
    )

    logger.info("Loading kokoro_v3_bundle model from LOCAL copies: %s", args.decoder_path)
    kmodel = load_kokoro_model_with_checkpoint(args.decoder_path)
    pipeline = build_pipeline(model=kmodel)

    def synth(text: str) -> object:
        return kokoro_v3_bundle(text, pipeline=pipeline, voicepack_path=args.voicepack_path)

    _write_environment_record("kokoro_v3_bundle")

    prompts_by_set = get_prompts_by_set(args.prompt_set)
    logger.info("Prompts loaded: %s", {k: len(v) for k, v in prompts_by_set.items()})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    audio_out_dir = RESULTS_AUDIO_HARNESS_DIR / "kokoro_v3_bundle"

    _run_one_condition(
        system="kokoro_v3_bundle",
        condition=None,
        synth=synth,
        is_streaming=False,
        prompts_by_set=prompts_by_set,  # type: ignore[arg-type]
        n_warmup=args.n_warmup,
        limit=args.limit,
        audio_out_dir=audio_out_dir,
        out_dir=RESULTS_DIR,
    )


if __name__ == "__main__":
    main()
