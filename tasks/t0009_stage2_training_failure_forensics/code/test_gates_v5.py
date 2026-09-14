"""Gate tests: the checks that would have caught the v4 manifest corruption.

Each constant below is a real line lifted from one of the three broken v4 manifests, plus a
counter-example that must NOT be rejected.
"""

from tasks.t0003_kokoro_v5_phoneme_data.code.prepare_v5_data import (
    load_kokoro_vocab,
    rejection_reason,
    slug,
)

VOCAB = load_kokoro_vocab()

# data/v4/train/train_list.txt -- raw orthography, the bug that trained t0001 on graphemes.
RAW_TEXT_LEAK = "I'm sorry, but I can only help with questions about Rezolve."

# data/v4/train_list.txt -- espeak re-read already-IPA text as Unicode character names.
DOUBLE_PHONEMIZED = "sˌɛkəndɹɪstɹˌɛssmˌɔːlkˌapˈaɪtˌiːˈɛs ðˈiː tˈiː ˌəʊpənˈɛː"

# misaki's out-of-vocabulary marker for a brand name it does not know.
UNKNOWN_WORD = "ʧˈɛkɪŋ ɪf ❓ səpˈɔːts ˈʌðə lˈaŋɡwɪʤɪz"

# Uppercase mid-IPA is CORRECT misaki output: A I O Q S T W Y are Kokoro vocab symbols
# standing for /eɪ/, /aɪ/, /oʊ/, /əʊ/ and friends. Rejecting these would have thrown away
# most of the corpus.
UPPERCASE_IS_LEGAL = "bɪhˈInd ðə kˈɒnvəsˈAʃənᵊl ˈAʤᵊnt"

CLEAN_IPA = "ˈaksɛsɪŋ ðə bɹˈAn kˈɒməːs pˈAʤ…"


def test_gates() -> None:
    assert rejection_reason(RAW_TEXT_LEAK, VOCAB) == "raw_text_leaked"
    assert rejection_reason(DOUBLE_PHONEMIZED, VOCAB) == "double_phonemized"
    assert rejection_reason(UNKNOWN_WORD, VOCAB) == "unknown_word"
    assert rejection_reason("", VOCAB) == "empty"
    assert rejection_reason(CLEAN_IPA, VOCAB) is None
    assert rejection_reason(UPPERCASE_IS_LEGAL, VOCAB) is None


def test_out_of_vocab_char_is_named() -> None:
    """A symbol outside Kokoro's 114 must be rejected and reported, not silently kept."""
    reason = rejection_reason("ˈaksɛsɪŋ ðə bɹˈAn §", VOCAB)
    assert reason is not None
    assert reason.startswith("out_of_vocab:")
    assert "§" in reason


def test_slug_matches_v4_filenames() -> None:
    """Slug must reproduce prepare_v4_data.py's naming or texts cannot be matched back."""
    assert slug("Accessing the Brain Commerce page…") == "accessing_the_brain_commerce_page_cddbb5"


if __name__ == "__main__":
    test_gates()
    test_out_of_vocab_char_is_named()
    test_slug_matches_v4_filenames()
    print("gates ok")
