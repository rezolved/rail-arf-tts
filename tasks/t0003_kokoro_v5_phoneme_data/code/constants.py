"""Named constants for t0003 manifest regeneration."""

import re
from typing import Final

# Manifest line format consumed by StyleTTS2's meldataset: wav_path|phonemes|speaker_id.
# Paths stay relative to the repo root the way t0001's working config resolved them
# (config root_path: '..', cwd StyleTTS2/), so wav files need no moving or re-copying.
TRAIN_WAV_PREFIX: Final[str] = "data/v4/train/wavs"
VAL_WAV_PREFIX: Final[str] = "data/v4/val/wavs"
SPEAKER_ID: Final[str] = "0"
FIELD_SEP: Final[str] = "|"

MANIFEST_FILENAME_COLUMN: Final[str] = "filename"
MANIFEST_TEXT_COLUMN: Final[str] = "text"

# David's reference voice is British English, so training phonemizes with british=True and
# inference MUST use KPipeline(lang_code="b"). The two must agree: British and American misaki
# emit different vowel symbols for the same word, and mixing them is a token-space mismatch.
# v4 phonemized british=True but t0002 synthesized with lang_code="a" -- half of this pair was
# already wrong before the raw-text fallback is even considered.
G2P_BRITISH: Final[bool] = True
G2P_TRF: Final[bool] = False

# lang_code passed to kokoro.KPipeline at inference; must match G2P_BRITISH.
KOKORO_LANG_CODE: Final[str] = "b"

# --- Corruption gates -------------------------------------------------------
# The authoritative gate is Kokoro's own 114-symbol vocab (config.json "vocab"): it is exactly
# the token space the model sees at inference, so any character outside it is either dropped
# by StyleTTS2's TextCleaner or mistrained.
#
# Two tempting gates are WRONG and deliberately not used:
#   - blanket [A-Za-z] rejection: valid English IPA is full of ASCII letters ("bˌʌt kæn hˈɛlp").
#   - uppercase rejection: A I O Q S T W Y are legal Kokoro symbols and misaki's American
#     output uses them for /eɪ/, /aɪ/, /oʊ/, /aʊ/ and the flap. "bɪhˈInd" is correct, not corrupt.
#
# What actually broke v4, and what these gates catch:
#   1. raw orthography leaked through (prepare_v4_data.py's silent except-fallback)
#   2. espeak re-read already-IPA text as Unicode character names (double phonemization)
#   3. `❓`, misaki's out-of-vocabulary marker, which is not in Kokoro's vocab

KOKORO_REPO_ID: Final[str] = "hexgrad/Kokoro-82M"
KOKORO_CONFIG_FILENAME: Final[str] = "config.json"
KOKORO_VOCAB_KEY: Final[str] = "vocab"

OOV_MARKER: Final[str] = "❓"

# Characters that only appear in IPA output, never in the source orthography.
IPA_MARKERS: Final[frozenset[str]] = frozenset("ˈˌːɹɪʊəɐɒɑɔɛɜæʌʒʃθðŋɡɾʔᵊᵻ")

# Literal fragments espeak emits when fed IPA: "SECONDARY STRESS", "SMALL CAPITAL", ...
DOUBLE_PHONEMIZED_MARKERS: Final[tuple[str, ...]] = (
    "sˌɛkəndɹɪstɹˌɛs",
    "smˈɔːlkˌap",
    "smˌɔːlkˌap",
    "pɹˈaɪmɚɹistɹˈɛs",
    "pɹˈaɪməɹɪstɹˈɛs",
    "mˈɒdɪfaɪɚlˈɛtɚ",
    "lˈatɪnsmˈɔːl",
)

# Sentence-boundary repair applied before G2P: "Rezolve.Let me know" -> "Rezolve. Let me know".
MISSING_SPACE_RE: Final[re.Pattern[str]] = re.compile(r"([.!?])([A-Za-z])")

SLUG_STRIP_RE: Final[re.Pattern[str]] = re.compile(r"[^\w\s-]")
SLUG_SPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
SLUG_MAX_LEN: Final[int] = 60
SLUG_HASH_LEN: Final[int] = 6
