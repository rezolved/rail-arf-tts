"""Task-specific tests for t0018 zero-shot voice-cloning calibration (plan Step 9).

Run with:
    uv run pytest tasks/t0018_zero_shot_cloning_calibration/code/test_zeroshot.py -v
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import soundfile as sf

from tasks.t0018_zero_shot_cloning_calibration.code.build_references import _build_concat
from tasks.t0018_zero_shot_cloning_calibration.code.constants import (
    COMPARISON_SET_SEED,
    COMPARISON_SET_VAL96_COUNT,
    GATE_TEXT_NAMES,
)
from tasks.t0018_zero_shot_cloning_calibration.code.report_zeroshot import (
    compute_variant_metrics_zeroshot,
    variant_id_for,
)


def test_build_concat_duration_in_range(tmp_path: Path) -> None:
    """A synthetic fixture corpus concatenated to a ~30s target lands in [28, 34]s."""
    sr = 24000
    clip_len_s = 1.2
    n_needed = int(30.0 / (clip_len_s + 0.2)) + 3
    paths: list[Path] = []
    rng = np.random.default_rng(0)
    for i in range(n_needed):
        data = (rng.standard_normal(int(clip_len_s * sr)) * 0.01).astype(np.float32)
        p = tmp_path / f"fixture_clip_{i:02d}.wav"
        sf.write(str(p), data, sr)
        paths.append(p)

    concat, out_sr, used_names = _build_concat(paths, target_duration_s=30.0, exclude_names=set())
    duration_s = len(concat) / out_sr

    assert 28.0 <= duration_s <= 34.0, f"duration {duration_s} out of [28, 34]"
    assert len(used_names) > 0


def test_gate_text_names_resolve_to_nonempty_text() -> None:
    """All 3 fixed gate texts resolve to non-empty text and their source clip exists on disk.

    None of the 3 fixed `GATE_TEXT_NAMES` happen to be in the 100-item filler sample (verified:
    only 100 of 1364 corpus clips are sampled there); their transcript is instead derived via the
    same filename-ground-truth rule `transcribe_references.py` uses (verified against the filler
    manifest for every clip actually used in `ref_single`/`ref_concat` — see that module's
    docstring), and their audio existence is checked directly against the 11labs_david corpus.
    """
    from tasks.t0018_zero_shot_cloning_calibration.code.paths import DATA_11LABS_DAVID_DIR
    from tasks.t0018_zero_shot_cloning_calibration.code.transcribe_references import (
        _filename_to_text,
    )

    for name in GATE_TEXT_NAMES:
        clip_path = DATA_11LABS_DAVID_DIR / f"{name}.wav"
        assert clip_path.exists(), f"gate text clip {clip_path} does not exist"
        text = _filename_to_text(f"{name}.wav")
        assert len(text) > 0


def test_comparison_set_sampler_deterministic() -> None:
    """random.Random(COMPARISON_SET_SEED).sample(...) is deterministic across two calls."""
    val96_texts = [f"text_{i}" for i in range(96)]

    sample_1 = random.Random(COMPARISON_SET_SEED).sample(val96_texts, COMPARISON_SET_VAL96_COUNT)
    sample_2 = random.Random(COMPARISON_SET_SEED).sample(val96_texts, COMPARISON_SET_VAL96_COUNT)

    assert sample_1 == sample_2
    assert len(sample_1) == COMPARISON_SET_VAL96_COUNT


def test_variant_metrics_builder_three_dimensions() -> None:
    """compute_variant_metrics_zeroshot produces the (system, condition, prompt_set) variant_id."""
    records: list[dict[str, object]] = [
        {
            "system": "f5_tts",
            "condition": "ref_single",
            "prompt_set": "fillers",
            "ttfb_ms": 100.0,
            "rtf": 0.5,
            "speaker_sim": 0.7,
            "duration_ratio": 1.0,
            "wer": 0.0,
        }
        for _ in range(10)
    ]
    metrics = compute_variant_metrics_zeroshot(records, "f5_tts", "ref_single", "fillers")

    assert metrics["speaker_sim"] == 0.7
    assert metrics["n_clips"] == 10.0
    assert variant_id_for("f5_tts", "ref_single", "fillers") == "f5_tts_ref_single_fillers"
    assert variant_id_for("elevenlabs_david", None, "fillers") == "elevenlabs_david_fillers"


def test_variant_metrics_rejection_rule_below_80_percent() -> None:
    """A variant with < 80% successful requests is nulled with a rejected_reason."""
    ok_records = [
        {
            "system": "chatterbox",
            "condition": "ref_concat",
            "prompt_set": "val96",
            "ttfb_ms": 200.0,
            "rtf": 0.4,
            "speaker_sim": 0.6,
            "duration_ratio": 1.0,
            "wer": 0.1,
        }
        for _ in range(5)
    ]
    fail_records = [
        {
            "system": "chatterbox",
            "condition": "ref_concat",
            "prompt_set": "val96",
            "ttfb_ms": None,
            "rtf": None,
            "speaker_sim": None,
            "duration_ratio": None,
            "wer": None,
        }
        for _ in range(20)
    ]
    metrics = compute_variant_metrics_zeroshot(
        ok_records + fail_records, "chatterbox", "ref_concat", "val96"
    )

    assert metrics["speaker_sim"] is None
    assert metrics["rejected_reason"] == "successful_requests/total_requests < 0.8"
