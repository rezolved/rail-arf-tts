"""Pronunciations for corpus words misaki's British lexicon does not know.

misaki emits `❓` for out-of-vocabulary words. `❓` is not one of Kokoro's 114 vocab symbols,
so StyleTTS2's TextCleaner drops it: the phoneme sequence loses a whole word that is still
present in the audio, and the text/audio alignment the duration predictor learns from is wrong
for that clip. 22% of the corpus (359/1653 clips) hit this -- overwhelmingly brand and person
names.

Entries come in two forms:

* **Respelling** -- a phrase of ordinary English words misaki already knows. Preferred: misaki
  derives the IPA itself, so the result is guaranteed in-vocab and in the right accent.
* **Literal IPA**, marked with the `IPA_PREFIX`. Used only where no respelling produces the
  right sounds. Validated character-by-character against Kokoro's vocab at install time.

All phonemes are British English (`en.G2P(british=True)`), matching David's reference voice.
"""

from typing import Final

IPA_PREFIX: Final[str] = "/"

# Unknown word (lowercased) -> respelling, or IPA_PREFIX + literal British IPA.
RESPELLINGS: Final[dict[str, str]] = {
    # --- Rezolve brand vocabulary ---------------------------------------------------
    "rezolve": "resolve",
    "rezolve's": "resolves",
    "brainpowa": "brain power",
    "powa": "power",
    "ai": "A.I.",
    "agentic": "/Aʤˈɛntɪk",
    "ermecarnatrezolve": "/ˈɜːmi kˈɑːnə tˈiː ɹɪzˈɒlv",
    # --- Common words missing from the lexicon ---------------------------------------
    "telecom": "tele com",
    "walkthrough": "walk through",
    "verified": "vairy fide",
    "personalisation": "personalization",
    "optimisation": "optimization",
    "shortlist": "short list",
    "replatforming": "ree platforming",
    "frontend": "front end",
    "auditability": "audit ability",
    "poseable": "pose able",
    "multimodal": "/mˌʌltɪmˈQdᵊl",
    "geo-fencing": "gee oh fencing",
    # --- Person names ----------------------------------------------------------------
    "salman": "/sˈalmɑːn",
    "ahmad": "ah mahd",
    "ahmed": "ah med",
    "amad": "/ˈamad",
    "urmee": "ur mee",
    "ermy": "ur mee",
    "ermei": "ur may",
    "michelle": "/mɪʃˈɛl",
    "ingram": "in gram",
    "burchill": "bur chill",
    "ferrero": "ferry row",
    "derek": "derrick",
    "anthony": "/ˈantəni",
    "banerjee": "banner jee",
    "binlaurie": "bin lorry",
    "salvik": "/sˈalvɪk",
    "kahn": "khan",
    "creighton": "cray ton",
    "presley": "press lee",
    "sackville": "sack vill",
    "wavell": "way vell",
    "wagner": "/vˈaɡnə",
    "barrell": "barrel",
    # Keyed with its corpus casing: capitalize()/title() cannot produce an internal capital.
    "McElla": "/məkˈɛlə",
    "efisher": "/ˈiːfɪʃə",
    "reebug": "ree bug",
    # --- Company and product names ---------------------------------------------------
    "microsoft": "micro soft",
    "microsoft's": "micro softs",
    "magento": "/məʤˈɛntQ",
    "magento's": "/məʤˈɛntQz",
    "rakuten": "/ɹˌakˈuːtɛn",
    "mahindra": "/məhˈɪndɹə",
    "mintra": "/mˈɪntɹə",
    "subsquid": "sub squid",
    "shoeby": "shoe bee",
    "azos": "/ˈAzɒs",
    "agio": "/ˈaʤiQ",
    # --- Fragments left by ordinals, contractions and truncated speech ----------------
    "th": "tee aitch",
    "st": "ess tee",
    "ll": "ell",
    "pq": "P.Q.",
    "mv": "M.V.",
    "unif-": "uniform",
}
