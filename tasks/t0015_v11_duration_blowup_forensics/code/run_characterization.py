"""Milestone B step 5 (REQ-4): varied-text duration-blowup characterization batch.

Runs the instrumented `synthesize()` (code/infer_styletts2.py, plan.md Milestone B step 4) against
`kokoro-v11-best` for 10 varied texts -- 5 short filler phrases (first 5 entries of
`tasks/t0008_tts_eval_harness_baselines/data/filler_prompts_100.json`, by list order, for
determinism) and 5 longer sentences (the 5 longest `"text"` values by character length in
`tasks/t0008_tts_eval_harness_baselines/data/val96_prompts.json`, for determinism and maximum
"long text" coverage) -- to determine whether the duration blowup first found in
`tasks/t0014_v11_decoder_fix_retrain/results/audio_samples/ft/v11_best.wav` (73.95s for a ~10-word
sentence) is universal, text-length-dependent, or intermittent.

Each text is synthesized with the fixed style reference (`data/reference_concat.wav`, built by
`code/build_reference_concat.py`) and default inference parameters
(`alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1.0`) -- the same defaults
`tasks/t0014_v11_decoder_fix_retrain/code/infer_styletts2.py` used to produce the original 73.95s
finding. Only `--limit N` is supported, for the plan's required 2-item validation gate before the
full 10-item run.

Usage::

    code/.venv-styletts2/bin/python -m \
        tasks.t0015_v11_duration_blowup_forensics.code.run_characterization
    code/.venv-styletts2/bin/python -m \
        tasks.t0015_v11_duration_blowup_forensics.code.run_characterization --limit 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import soundfile as sf
import torch
from munch import Munch

from tasks.t0015_v11_duration_blowup_forensics.code.audio_quality_check import (
    check_audio_quality,
)
from tasks.t0015_v11_duration_blowup_forensics.code.infer_styletts2 import (
    build_harness,
    compute_style,
    synthesize,
)
from tasks.t0015_v11_duration_blowup_forensics.code.paths import (
    DURATION_CHARACTERIZATION_JSON,
    FILLER_PROMPTS_100_JSON,
    REFERENCE_CONCAT_WAV,
    RESULTS_CHARACTERIZATION_AUDIO_DIR,
    STYLETTS2_DIR,
    V11_CHECKPOINT,
    V11_CONFIG,
    VAL96_PROMPTS_JSON,
)

SAMPLE_RATE = 24000
NAIVE_WORDS_PER_SECOND = 2.5  # diagnostic-only ratio for this step's log, per plan.md step 5(e).
NUM_SHORT_TEXTS = 5
NUM_LONG_TEXTS = 5


@dataclass(frozen=True, slots=True)
class CharacterizationRecord:
    index: int
    slug: str
    category: str  # "short" | "long"
    text: str
    word_count: int
    input_token_count: int
    phoneme_string: str
    pred_dur: list[float]
    pred_dur_sum: int
    output_duration_s: float
    wall_time_s: float
    rtf: float | None
    expected_duration_s: float
    duration_ratio: float | None
    rms: float
    peak: float
    silence_fraction: float
    spectral_flatness: float
    clip_fraction: float
    is_likely_noise: bool
    error: str | None


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:max_len] if len(slug) > max_len else slug


def load_texts() -> list[tuple[str, str]]:
    """Returns a list of (category, text) tuples: 5 short then 5 long."""
    with FILLER_PROMPTS_100_JSON.open() as f:
        filler_prompts = json.load(f)
    short_texts = [entry["text"] for entry in filler_prompts[:NUM_SHORT_TEXTS]]

    with VAL96_PROMPTS_JSON.open() as f:
        val96_prompts = json.load(f)
    longest_first = sorted(val96_prompts, key=lambda e: -len(e["text"]))
    long_texts = [entry["text"] for entry in longest_first[:NUM_LONG_TEXTS]]

    return [("short", t) for t in short_texts] + [("long", t) for t in long_texts]


def run_one(
    model: Munch,
    model_params: Munch,
    text: str,
    ref_s: torch.Tensor,
    index: int,
    category: str,
) -> CharacterizationRecord:
    slug = slugify(text)
    word_count = len(text.split())
    expected_duration_s = word_count / NAIVE_WORDS_PER_SECOND

    out_wav = RESULTS_CHARACTERIZATION_AUDIO_DIR / f"v11_{index:02d}_{slug}.wav"
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = synthesize(model, model_params, text, ref_s)
    except Exception as exc:  # noqa: BLE001 -- record the failure per-text, don't abort the batch
        return CharacterizationRecord(
            index=index,
            slug=slug,
            category=category,
            text=text,
            word_count=word_count,
            input_token_count=0,
            phoneme_string="",
            pred_dur=[],
            pred_dur_sum=0,
            output_duration_s=0.0,
            wall_time_s=0.0,
            rtf=None,
            expected_duration_s=expected_duration_s,
            duration_ratio=None,
            rms=0.0,
            peak=0.0,
            silence_fraction=1.0,
            spectral_flatness=0.0,
            clip_fraction=0.0,
            is_likely_noise=False,
            error=str(exc),
        )

    sf.write(str(out_wav), result.wav, SAMPLE_RATE)
    output_duration_s = len(result.wav) / SAMPLE_RATE
    rtf = result.wall_time_seconds / output_duration_s if output_duration_s > 0 else None
    duration_ratio = output_duration_s / expected_duration_s if expected_duration_s > 0 else None

    quality = check_audio_quality(out_wav)

    return CharacterizationRecord(
        index=index,
        slug=slug,
        category=category,
        text=text,
        word_count=word_count,
        input_token_count=result.input_token_count,
        phoneme_string=result.phoneme_string,
        pred_dur=result.pred_dur,
        pred_dur_sum=result.pred_dur_sum,
        output_duration_s=output_duration_s,
        wall_time_s=result.wall_time_seconds,
        rtf=rtf,
        expected_duration_s=expected_duration_s,
        duration_ratio=duration_ratio,
        rms=quality.rms,
        peak=quality.peak,
        silence_fraction=quality.silence_fraction,
        spectral_flatness=quality.spectral_flatness,
        clip_fraction=quality.clip_fraction,
        is_likely_noise=quality.is_likely_noise,
        error=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    sys.path.insert(0, str(STYLETTS2_DIR))
    os.chdir(STYLETTS2_DIR)

    model, model_params, _log_path = build_harness(V11_CONFIG, V11_CHECKPOINT)
    ref_s = compute_style(model, REFERENCE_CONCAT_WAV)

    texts = load_texts()
    if args.limit is not None:
        texts = texts[: args.limit]

    records: list[CharacterizationRecord] = []
    for i, (category, text) in enumerate(texts):
        print(f"[{i + 1}/{len(texts)}] ({category}) {text!r}")
        start = time.perf_counter()
        record = run_one(model, model_params, text, ref_s, index=i, category=category)
        elapsed = time.perf_counter() - start
        print(
            f"  -> output_duration_s={record.output_duration_s:.2f} "
            f"pred_dur_sum={record.pred_dur_sum} input_token_count={record.input_token_count} "
            f"is_likely_noise={record.is_likely_noise} error={record.error} "
            f"(wall {elapsed:.1f}s)"
        )
        records.append(record)
        # Write incrementally so a crash mid-batch does not lose completed records.
        DURATION_CHARACTERIZATION_JSON.parent.mkdir(parents=True, exist_ok=True)
        DURATION_CHARACTERIZATION_JSON.write_text(
            json.dumps([asdict(r) for r in records], indent=2) + "\n"
        )

    print(f"Wrote {DURATION_CHARACTERIZATION_JSON} ({len(records)} records)")


if __name__ == "__main__":
    main()
