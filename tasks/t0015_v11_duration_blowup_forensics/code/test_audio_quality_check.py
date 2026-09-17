"""Task-specific test for Milestone D step 10 (REQ-8): confirm the hardened gate's public interface.

Verifies `check_audio_quality()` accepts an optional `text` parameter and `AudioQualityResult` has
the two new fields (`duration_sanity_pass`, `longest_nonsilent_run_s`) -- not just described in
prose. Per `arf/skills/implementation/SKILL.md`, task-specific tests live in
`tasks/$TASK_ID/code/test_*.py`.

Run::

    uv run pytest tasks/t0015_v11_duration_blowup_forensics/code/test_audio_quality_check.py -v
"""

from __future__ import annotations

import inspect

from tasks.t0015_v11_duration_blowup_forensics.code.audio_quality_check import (
    AudioQualityResult,
    check_audio_quality,
    estimate_naive_duration_bound_s,
)


def test_check_audio_quality_accepts_text_parameter() -> None:
    sig = inspect.signature(check_audio_quality)
    assert "text" in sig.parameters
    assert sig.parameters["text"].default is None


def test_audio_quality_result_has_hardened_fields() -> None:
    fields = AudioQualityResult.__dataclass_fields__
    assert "duration_sanity_pass" in fields
    assert "longest_nonsilent_run_s" in fields
    # Pre-existing signals must still be present (backward compatibility).
    assert "rms" in fields
    assert "peak" in fields
    assert "silence_fraction" in fields
    assert "spectral_flatness" in fields
    assert "clip_fraction" in fields
    assert "is_likely_noise" in fields


def test_estimate_naive_duration_bound_scales_with_word_count() -> None:
    short = estimate_naive_duration_bound_s("one two three")
    long = estimate_naive_duration_bound_s("one two three four five six seven eight nine ten")
    assert long > short
    assert short > 0.0
