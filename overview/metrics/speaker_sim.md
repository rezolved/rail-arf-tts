# Speaker Similarity (GE2E Cosine)

**Key**: `speaker_sim`

**Unit**: ratio (0–1)

**Definition**: cosine similarity between the GE2E speaker embedding of a synthesized clip and the
mean GE2E embedding of the ElevenLabs David reference set (`data/11labs_david/`). Computed using
`resemblyzer` (install with `pip install rezolve-arf-tts[speaker-sim]`).

**Target**: ≥ 0.85 for fine-tuned Kokoro v4 on val_96.

**Baseline**: ElevenLabs David self-similarity ≈ 1.0 (same voice, different clips).

**Note on deps**: `resemblyzer` depends on `webrtcvad 2.0.10` which imports `pkg_resources`
(removed in setuptools ≥ 81). Keep resemblyzer in the optional `[speaker-sim]` extra — never add
it to main dependencies.

**Higher is better.**
