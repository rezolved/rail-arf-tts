# ElevenLabs Filler Corpus (1358 clips)

**Path**: `data/11labs_david/` (in `kokoro-finetune/` working tree on LLM-T1-NC80).

**Size**: 1358 WAV clips, 24 kHz mono.

**Source**: synthesized by ElevenLabs David voice from `fillers_from_logs.txt` — real filler
prompts extracted from Rezolve voice commerce session logs.

**Role**:

* Speaker-similarity ground truth for all Kokoro evaluations (GE2E cosine computed against these
  clips as the "David reference").
* Benchmark corpus for ElevenLabs baseline latency/RTF measurement (t0002).
* Prompt set for Kokoro inference benchmark (t0003, t0004).

**Format**: `<prompt_text>` → `<clip_id>.wav`. Clip duration 0.5–4 s (filler length).

**DVC**: to be tracked after first task runs `dvc add data/11labs_david/`.
