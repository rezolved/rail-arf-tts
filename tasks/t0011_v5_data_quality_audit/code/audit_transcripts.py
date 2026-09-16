"""OOV transcript audit for t0011_v5_data_quality_audit.

Reads the v5 train manifest and merges oov_fraction / oov_count /
has_double_phonemize fields into per_clip_stats.jsonl (rewrites in place).

Run via run_with_logs:
    uv run python -m arf.scripts.utils.run_with_logs \
        --task-id t0011_v5_data_quality_audit -- \
        uv run python -u tasks/t0011_v5_data_quality_audit/code/audit_transcripts.py
"""

import json
from pathlib import Path

from tasks.t0011_v5_data_quality_audit.code.constants import (
    DOUBLE_PHONEMIZED_MARKERS,
    FIELD_HAS_DOUBLE_PHONEMIZE,
    FIELD_OOV_COUNT,
    FIELD_OOV_FRACTION,
    FIELD_WAV_PATH,
    OOV_MARKER,
)
from tasks.t0011_v5_data_quality_audit.code.paths import (
    PER_CLIP_STATS_JSONL,
    V5_TRAIN_LIST,
)


def compute_oov_stats(phonemes: str) -> dict[str, object]:
    """Compute OOV and double-phonemize stats from a phoneme string."""
    total_len = max(len(phonemes), 1)
    oov_count = phonemes.count(OOV_MARKER)
    oov_fraction = oov_count / total_len
    has_double_phonemize = any(marker in phonemes for marker in DOUBLE_PHONEMIZED_MARKERS)
    return {
        FIELD_OOV_COUNT: oov_count,
        FIELD_OOV_FRACTION: oov_fraction,
        FIELD_HAS_DOUBLE_PHONEMIZE: has_double_phonemize,
    }


def _parse_list(list_path: Path) -> list[dict[str, str]]:
    """Parse a pipe-delimited manifest: wav_path|phonemes|speaker_id."""
    entries: list[dict[str, str]] = []
    if not list_path.exists():
        print(f"  MISSING: {list_path}")
        return entries
    with list_path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if len(line) == 0:
                continue
            parts = line.split("|")
            wav_path = parts[0] if len(parts) > 0 else ""
            phonemes = parts[1] if len(parts) > 1 else ""
            speaker_id = parts[2] if len(parts) > 2 else "0"
            entries.append({"wav_path": wav_path, "phonemes": phonemes, "speaker_id": speaker_id})
    return entries


def main() -> None:
    print("=== audit_transcripts.py: OOV audit for v5 train manifest ===")

    # Build OOV lookup dict
    train_entries = _parse_list(V5_TRAIN_LIST)
    print(f"  Loaded {len(train_entries)} train manifest entries")

    oov_map: dict[str, dict[str, object]] = {}
    for entry in train_entries:
        oov_map[entry["wav_path"]] = compute_oov_stats(phonemes=entry["phonemes"])

    # Merge OOV fields into per_clip_stats.jsonl
    if not PER_CLIP_STATS_JSONL.exists():
        print(f"ERROR: {PER_CLIP_STATS_JSONL} not found — run audit_audio.py first")
        raise FileNotFoundError(str(PER_CLIP_STATS_JSONL))

    # Write to a temp file then rename (atomic-ish)
    tmp_path = PER_CLIP_STATS_JSONL.parent / (PER_CLIP_STATS_JSONL.name + ".tmp")
    merged_count = 0
    missing_count = 0

    try:
        with (
            PER_CLIP_STATS_JSONL.open(encoding="utf-8") as in_f,
            tmp_path.open("w", encoding="utf-8") as out_f,
        ):
            for line in in_f:
                line = line.strip()
                if len(line) == 0:
                    continue
                record: dict[str, object] = json.loads(line)
                wav_path_val = str(record.get(FIELD_WAV_PATH, ""))
                if wav_path_val in oov_map:
                    record.update(oov_map[wav_path_val])
                    merged_count += 1
                else:
                    # Clip not in manifest — set defaults
                    record[FIELD_OOV_COUNT] = 0
                    record[FIELD_OOV_FRACTION] = 0.0
                    record[FIELD_HAS_DOUBLE_PHONEMIZE] = False
                    missing_count += 1
                out_f.write(json.dumps(record) + "\n")

        tmp_path.replace(PER_CLIP_STATS_JSONL)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    print(f"  Merged OOV fields for {merged_count} clips ({missing_count} not in manifest)")
    print(f"  Updated: {PER_CLIP_STATS_JSONL}")

    # Verify a sample record has the OOV fields
    with PER_CLIP_STATS_JSONL.open(encoding="utf-8") as f:
        sample = json.loads(next(iter(f)))
    assert FIELD_OOV_FRACTION in sample, "oov_fraction field missing in sample record"
    assert FIELD_OOV_COUNT in sample, "oov_count field missing in sample record"
    print("  Verification: oov_fraction and oov_count present in first record")
    print("Done.")


if __name__ == "__main__":
    main()
