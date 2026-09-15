"""Regenerate v5 train/val manifests with validated British-English IPA phonemes.

t0001's Stage 2 run trained on `data/v4/train/train_list.txt`, whose text column held raw
English orthography: `prepare_v4_data.py` swallowed every misaki failure with
`except Exception: return text.strip()`. StyleTTS2's TextCleaner accepts ASCII letters, so
training never crashed -- it silently learned a grapheme token space while Kokoro's
KPipeline feeds IPA at inference. Result: Dur loss pinned at ~1.03 and a 10x duration
explosion at synthesis time.

This script rebuilds the text column only. Wav files and the train/val split are taken
verbatim from the existing v4 directories, so the sole difference against the failed run is
the token space.

Usage:
    uv run python3 code/prepare_v5_data.py
    uv run python3 code/prepare_v5_data.py --limit 20   # smoke run
"""

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from tasks.t0003_kokoro_v5_phoneme_data.code import constants as C
from tasks.t0003_kokoro_v5_phoneme_data.code import paths as P
from tasks.t0003_kokoro_v5_phoneme_data.code.lexicon import IPA_PREFIX, RESPELLINGS


@dataclass(frozen=True, slots=True)
class Entry:
    """One manifest line: a wav file, its source text and its phonemization."""

    wav_rel_path: str
    text: str
    phonemes: str


@dataclass(frozen=True, slots=True)
class Rejection:
    """A clip excluded from the manifest, with the gate that caught it."""

    wav_rel_path: str
    reason: str
    phonemes: str


def slug(text: str) -> str:
    """Reproduce prepare_v4_data.py's filler-to-filename slug so texts can be matched back."""
    clean = C.SLUG_STRIP_RE.sub("", text.lower())
    clean = C.SLUG_SPACE_RE.sub("_", clean.strip())[: C.SLUG_MAX_LEN]
    digest = hashlib.md5(text.encode()).hexdigest()[: C.SLUG_HASH_LEN]
    return f"{clean}_{digest}"


def load_manifest_texts(manifest_path: Path) -> dict[str, str]:
    """Read a v3 `filename|text|duration` CSV into {filename: text}."""
    texts: dict[str, str] = {}
    with manifest_path.open() as handle:
        for row in csv.DictReader(handle, delimiter=C.FIELD_SEP):
            texts[row[C.MANIFEST_FILENAME_COLUMN]] = row[C.MANIFEST_TEXT_COLUMN].strip()
    return texts


def build_text_index() -> dict[str, str]:
    """Map every known wav stem/filename to its original orthographic text.

    Sources are the *originals* -- the v3 manifests and the filler log -- never a v4
    manifest, since all three v4 variants are corrupted in different ways.
    """
    index: dict[str, str] = {}
    index.update(load_manifest_texts(P.V3_TRAIN_MANIFEST))
    index.update(load_manifest_texts(P.V3_VAL_MANIFEST))

    fillers = [
        line.strip() for line in P.FILLERS_FILE.read_text().splitlines() if len(line.strip()) > 0
    ]
    for text in fillers:
        index[f"{slug(text)}.wav"] = text
    return index


def phonemize(text: str, g2p: object) -> str:
    """British-English IPA for `text`. Raises on any G2P failure -- no silent fallback."""
    repaired = C.MISSING_SPACE_RE.sub(r"\1 \2", text)
    result, _ = g2p(repaired)  # type: ignore[operator]
    if isinstance(result, str):
        return result.strip()
    parts: list[str] = []
    for token in result:
        phoneme = token.phonemes if token.phonemes is not None else token.text
        parts.append(phoneme + (token.whitespace if token.whitespace else ""))
    return "".join(parts).strip()


def load_kokoro_vocab() -> frozenset[str]:
    """Kokoro's 114 token symbols -- the exact character set the model accepts at inference."""
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(repo_id=C.KOKORO_REPO_ID, filename=C.KOKORO_CONFIG_FILENAME)
    config = json.loads(Path(config_path).read_text())
    return frozenset(config[C.KOKORO_VOCAB_KEY].keys())


def install_lexicon(g2p: object, vocab: frozenset[str]) -> int:
    """Teach misaki the corpus's brand and person names. Returns how many were added.

    Every entry is validated before install: a respelling must not itself be unknown, and any
    literal IPA must use only Kokoro vocab symbols. Failures are collected and raised together
    so a bad lexicon is fixed in one pass rather than one crash at a time.
    """
    lexicon = g2p.lexicon  # type: ignore[attr-defined]
    failures: list[str] = []
    for word, spelling in RESPELLINGS.items():
        if spelling.startswith(IPA_PREFIX):
            phonemes = spelling[len(IPA_PREFIX) :]
        else:
            phonemes = phonemize(spelling, g2p)
            if C.OOV_MARKER in phonemes:
                failures.append(f"{word!r}: respelling {spelling!r} is itself unknown")
                continue
        illegal = set(phonemes) - vocab
        if len(illegal) > 0:
            failures.append(f"{word!r}: {''.join(sorted(illegal))!r} not in Kokoro vocab")
            continue
        # misaki resolves proper nouns by the token's original casing, so a lowercase key
        # alone is hit only when spaCy does not tag the word PROPN -- "Rezolve supports" works
        # but "Checking if Rezolve supports" does not. Install every casing.
        for key in (word, word.capitalize(), word.title(), word.upper()):
            lexicon.golds[key] = phonemes
    if len(failures) > 0:
        raise ValueError("Lexicon entries rejected:\n  " + "\n  ".join(failures))
    return len(RESPELLINGS)


def rejection_reason(phonemes: str, vocab: frozenset[str]) -> str | None:
    """Return the name of the gate `phonemes` trips, or None if clean."""
    if len(phonemes) == 0:
        return "empty"
    if C.OOV_MARKER in phonemes:
        return "unknown_word"
    if len(set(phonemes) & C.IPA_MARKERS) == 0:
        return "raw_text_leaked"
    for marker in C.DOUBLE_PHONEMIZED_MARKERS:
        if marker in phonemes:
            return "double_phonemized"
    if len(set(phonemes) - vocab) > 0:
        return f"out_of_vocab:{''.join(sorted(set(phonemes) - vocab))}"
    return None


def collect_split(
    wav_dir: Path,
    wav_prefix: str,
    text_index: dict[str, str],
    g2p: object,
    vocab: frozenset[str],
    limit: int | None,
) -> tuple[list[Entry], list[Rejection]]:
    """Phonemize every wav in `wav_dir`, preserving the existing v4 split membership."""
    wav_files = sorted(wav_dir.glob("*.wav"))
    if limit is not None:
        wav_files = wav_files[:limit]

    entries: list[Entry] = []
    rejects: list[Rejection] = []
    for wav_file in tqdm(wav_files, desc=wav_dir.parent.name, unit="clip"):
        text = text_index.get(wav_file.name)
        if text is None:
            raise KeyError(
                f"No source text for {wav_file.name}. The v3 manifests and the filler log are "
                f"the only trusted text sources; do not fall back to a v4 manifest."
            )
        phonemes = phonemize(text, g2p)
        rel_path = f"{wav_prefix}/{wav_file.name}"
        reason = rejection_reason(phonemes, vocab)
        if reason is not None:
            rejects.append(Rejection(wav_rel_path=rel_path, reason=reason, phonemes=phonemes))
            continue
        entries.append(Entry(wav_rel_path=rel_path, text=text, phonemes=phonemes))
    return entries, rejects


def write_list(entries: list[Entry], out_path: Path) -> None:
    lines = [
        C.FIELD_SEP.join((entry.wav_rel_path, entry.phonemes, C.SPEAKER_ID)) for entry in entries
    ]
    out_path.write_text("\n".join(lines) + "\n")


def write_rejects(rejects: list[Rejection], out_path: Path) -> None:
    lines = [
        C.FIELD_SEP.join((reject.reason, reject.wav_rel_path, reject.phonemes))
        for reject in rejects
    ]
    out_path.write_text("\n".join(lines) + ("\n" if len(lines) > 0 else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only the first N clips per split."
    )
    args = parser.parse_args()

    from misaki import en

    g2p = en.G2P(trf=C.G2P_TRF, british=C.G2P_BRITISH)
    print(
        f"misaki G2P: british={C.G2P_BRITISH} (inference must use lang_code={C.KOKORO_LANG_CODE!r})"
    )
    vocab = load_kokoro_vocab()
    print(f"Lexicon: +{install_lexicon(g2p, vocab)} words")

    text_index = build_text_index()
    print(f"Kokoro vocab: {len(vocab)} symbols | Text index: {len(text_index)} entries")

    P.V5_DIR.mkdir(parents=True, exist_ok=True)

    train_entries, train_rejects = collect_split(
        wav_dir=P.V4_TRAIN_WAVS,
        wav_prefix=C.TRAIN_WAV_PREFIX,
        text_index=text_index,
        g2p=g2p,
        vocab=vocab,
        limit=args.limit,
    )
    val_entries, val_rejects = collect_split(
        wav_dir=P.V4_VAL_WAVS,
        wav_prefix=C.VAL_WAV_PREFIX,
        text_index=text_index,
        g2p=g2p,
        vocab=vocab,
        limit=args.limit,
    )

    write_list(train_entries, P.TRAIN_LIST_OUT)
    write_list(val_entries, P.VAL_LIST_OUT)
    write_rejects(train_rejects + val_rejects, P.REJECTS_OUT)

    print(
        f"\nTrain: {len(train_entries)} kept, {len(train_rejects)} rejected -> {P.TRAIN_LIST_OUT}"
    )
    print(f"Val:   {len(val_entries)} kept, {len(val_rejects)} rejected -> {P.VAL_LIST_OUT}")
    if len(train_rejects) + len(val_rejects) > 0:
        print(f"Rejects -> {P.REJECTS_OUT}")
    if len(train_entries) > 0:
        print(f"\nSample: {train_entries[0].phonemes[:110]}")


if __name__ == "__main__":
    main()
