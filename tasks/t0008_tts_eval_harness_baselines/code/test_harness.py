"""Smoke tests for the TTS evaluation harness.

Tests:
a) compute_duration_ratio on known-length synthetic WAVs
b) compute_wer on trivial transcript pair (expects WER = 0.0)
c) report.py build_metrics_json produces valid variant-format JSON

Run with: uv run pytest tasks/t0008_tts_eval_harness_baselines/code/test_harness.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def _make_wav(path: Path, duration_s: float, sample_rate: int = 16000) -> None:
    """Write a silent float32 WAV of exactly duration_s seconds."""
    n_samples = int(duration_s * sample_rate)
    sf.write(str(path), np.zeros(n_samples, dtype=np.float32), sample_rate, subtype="FLOAT")


class TestDurationRatio:
    def test_exact_ratio(self) -> None:
        """2-second synth vs 1-second ref → ratio = 2.0."""
        from tasks.t0008_tts_eval_harness_baselines.code.scoring import compute_duration_ratio

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            synth = p / "synth.wav"
            _make_wav(synth, duration_s=2.0)

            result = compute_duration_ratio(
                synth_wavs=[synth],
                ref_durations_s=[1.0],
            )
            assert abs(result.median - 2.0) < 0.05, f"Expected ratio ~2.0, got {result.median}"
            assert result.explosion_count == 0

    def test_explosion_detected(self) -> None:
        """6-second synth vs 1-second ref → explosion count = 1."""
        from tasks.t0008_tts_eval_harness_baselines.code.scoring import compute_duration_ratio

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            synth = p / "long.wav"
            _make_wav(synth, duration_s=6.0)

            result = compute_duration_ratio(
                synth_wavs=[synth],
                ref_durations_s=[1.0],
            )
            assert result.explosion_count == 1

    def test_three_clips(self) -> None:
        """Three clips with known ratios: median should be 1.5."""
        from tasks.t0008_tts_eval_harness_baselines.code.scoring import compute_duration_ratio

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            paths: list[Path] = []
            for i, dur in enumerate([1.0, 1.5, 2.0]):
                wav_path = p / f"s{i}.wav"
                _make_wav(wav_path, duration_s=dur)
                paths.append(wav_path)

            result = compute_duration_ratio(
                synth_wavs=paths,
                ref_durations_s=[1.0, 1.0, 1.0],
            )
            # Ratios: 1.0, 1.5, 2.0 → median = 1.5
            assert abs(result.median - 1.5) < 0.05, f"Expected median 1.5, got {result.median}"


class TestWer:
    def test_identical_transcripts(self) -> None:
        """Identical reference and hypothesis → WER = 0.0 (manual fallback)."""
        from tasks.t0008_tts_eval_harness_baselines.code.scoring import _manual_wer

        assert _manual_wer("hello world", "hello world") == 0.0

    def test_one_substitution(self) -> None:
        """One-word substitution in 2-word ref → WER = 0.5."""
        from tasks.t0008_tts_eval_harness_baselines.code.scoring import _manual_wer

        wer = _manual_wer("hello world", "hello there")
        assert abs(wer - 0.5) < 1e-9, f"Expected 0.5, got {wer}"

    def test_normalize_text(self) -> None:
        """_normalize_text removes punctuation and lowercases."""
        from tasks.t0008_tts_eval_harness_baselines.code.scoring import _normalize_text

        result = _normalize_text("Hello, World!")
        assert result == "hello world", f"Got {result!r}"

    def test_empty_reference(self) -> None:
        """Empty reference and non-empty hypothesis → WER = 1.0."""
        from tasks.t0008_tts_eval_harness_baselines.code.scoring import _manual_wer

        assert _manual_wer("", "hello") == 1.0

    def test_both_empty(self) -> None:
        """Both empty → WER = 0.0."""
        from tasks.t0008_tts_eval_harness_baselines.code.scoring import _manual_wer

        assert _manual_wer("", "") == 0.0


class TestReportJson:
    def test_variant_format(self) -> None:
        """build_metrics_json produces valid variant-format JSON with required keys."""
        from tasks.t0008_tts_eval_harness_baselines.code.report import build_metrics_json

        records: list[dict[str, object]] = [
            {
                "system": "test_system",
                "prompt_set": "val96",
                "ttfb_ms": 250.0,
                "rtf": 0.05,
                "speaker_sim": 0.88,
                "duration_ratio": 1.1,
                "wer": 0.02,
            },
            {
                "system": "test_system",
                "prompt_set": "val96",
                "ttfb_ms": 280.0,
                "rtf": 0.06,
                "speaker_sim": 0.86,
                "duration_ratio": 1.0,
                "wer": 0.05,
            },
        ]

        result = build_metrics_json(
            records=records,
            systems=["test_system"],
            prompt_sets=["val96"],
        )

        assert "variants" in result, "Missing 'variants' key"
        variants: list[dict[str, object]] = result["variants"]  # type: ignore[assignment]
        assert len(variants) == 1, f"Expected 1 variant, got {len(variants)}"

        v = variants[0]
        assert v["variant_id"] == "test_system_val96"
        assert "metrics" in v
        metrics = v["metrics"]

        # Only registered keys: speaker_sim, ttfb_ms, rtf
        assert "speaker_sim" in metrics
        assert "ttfb_ms" in metrics
        assert "rtf" in metrics
        # Non-registered keys must NOT be in metrics
        assert "wer_mean" not in metrics
        assert "duration_ratio_median" not in metrics

        # Values
        assert metrics["speaker_sim"] is not None
        assert abs(float(metrics["speaker_sim"]) - 0.87) < 0.01

    def test_empty_records(self) -> None:
        """Empty records produce empty variants list."""
        from tasks.t0008_tts_eval_harness_baselines.code.report import build_metrics_json

        result = build_metrics_json(
            records=[],
            systems=["no_system"],
            prompt_sets=["val96"],
        )
        assert result["variants"] == []

    def test_json_serializable(self) -> None:
        """Output is JSON-serializable."""
        from tasks.t0008_tts_eval_harness_baselines.code.report import build_metrics_json

        records: list[dict[str, object]] = [
            {
                "system": "sys_a",
                "prompt_set": "fillers",
                "ttfb_ms": 120.0,
                "rtf": 0.03,
                "speaker_sim": 0.92,
                "duration_ratio": 0.9,
                "wer": None,
            }
        ]
        result = build_metrics_json(
            records=records,
            systems=["sys_a"],
            prompt_sets=["fillers"],
        )
        # Should not raise
        json.dumps(result)
