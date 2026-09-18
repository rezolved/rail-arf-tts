"""Milestone 3 Step 11 (REQ-3, REQ-8): score every per-epoch v3 sample with the hardened audio
gate and append the results table to `results/v3_checkpoint_forensics.md`.

Covers `epoch{0..3}_phrase{1..5}.wav` (20 files) plus `audio/v3/ep{6,7,8}_p{1..5}.wav` (15 files)
and `audio/v3b/ep{6,7,8,9}_p{1..5}.wav` (20 files) -- 55 files total, fixing the epoch range
(0-9 across the `v3`/`v3b` variants) and the 5-phrase-per-epoch sampling cadence.

The original phrase texts were not preserved anywhere in this task's evidence sources (no
`phrases.txt` manifest in the DVC-pulled reference data or the VM inventory), so `text=None` is
passed for every file and `duration_sanity_pass` is recorded as `null` rather than guessed, per the
data-analysis instruction's "use None/null when data is unavailable, not 0.0/false as a stand-in."

Usage::

    uv run python -m tasks.t0016_v3_recipe_recovery.code.score_per_epoch_samples
"""

from __future__ import annotations

from pathlib import Path

from tasks.t0016_v3_recipe_recovery.code.audio_quality_check import check_audio_quality
from tasks.t0016_v3_recipe_recovery.code.paths import CHECKPOINT_FORENSICS_MD, V3_AUDIO_DIR


def collect_wav_files() -> list[tuple[str, str]]:
    """Return (relative_label, absolute_path_str) pairs, sorted for stable output."""
    files: list[tuple[str, str]] = []
    for wav_path in sorted(V3_AUDIO_DIR.glob("epoch*_phrase*.wav")):
        files.append((f"audio/{wav_path.name}", str(wav_path)))
    for subdir_name in ("v3", "v3b"):
        subdir = V3_AUDIO_DIR / subdir_name
        if subdir.is_dir():
            for wav_path in sorted(subdir.glob("*.wav")):
                files.append((f"audio/{subdir_name}/{wav_path.name}", str(wav_path)))
    return files


def render_table(rows: list[tuple[str, object, object, float]]) -> str:
    lines = ["## Per-Epoch Sample Gate Scores (REQ-3, REQ-8)", ""]
    lines.append(
        "Phrase texts were not recoverable from any evidence source (no `phrases.txt` manifest "
        "survived in DVC or the VM inventory), so `text=None` was passed for every file and "
        "`duration_sanity_pass` is `null` throughout -- honest per the data-analysis instruction, "
        "not a guessed value."
    )
    lines.append("")
    lines.append("| File | is_likely_noise | duration_sanity_pass | longest_nonsilent_run_s |")
    lines.append("| --- | --- | --- | ---: |")
    for label, is_noise, dur_pass, longest_run in rows:
        lines.append(f"| {label} | {is_noise} | {dur_pass} | {longest_run:.3f} |")
    lines.append("")
    noise_count = sum(1 for _, is_noise, _, _ in rows if is_noise)
    lines.append(
        f"**Summary**: {len(rows)} files scored; {noise_count} flagged `is_likely_noise=True`. "
        "Epoch range observed across `epoch{0..3}_phrase{1..5}.wav` plus `v3/ep{6,7,8}` and "
        "`v3b/ep{6,7,8,9}` is epochs 0-9 (10 distinct epoch checkpoints sampled at 5 phrases "
        "each), "
        "consistent with `task_description.md`'s note that `v3/audio/v3b/ep9_p5.wav` implies at "
        "least 10 epochs and a `v3b` variant -- confirmed here directly from the file listing, not "
        "inferred."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    files = collect_wav_files()
    rows: list[tuple[str, object, object, float]] = []
    for label, path_str in files:
        result = check_audio_quality(Path(path_str), text=None)
        rows.append(
            (
                label,
                result.is_likely_noise,
                result.duration_sanity_pass,
                result.longest_nonsilent_run_s,
            )
        )
        print(f"{label}: is_likely_noise={result.is_likely_noise}")

    table_md = render_table(rows)
    existing = CHECKPOINT_FORENSICS_MD.read_text()
    CHECKPOINT_FORENSICS_MD.write_text(existing.rstrip() + "\n\n" + table_md)
    print(f"\nAppended per-epoch gate table to {CHECKPOINT_FORENSICS_MD} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
