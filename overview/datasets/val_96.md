# Val-96 Held-Out Evaluation Set

**Path**: `data/v4/val/val_list.txt` (manifest) + `data/v4/val/wavs/` (audio).

**Size**: 96 clips, 24 kHz mono WAV.

**Source**: held-out split from the David voice training corpus (manually curated, not overlapping
with the 1557-clip training set).

**Format** (train_list.txt): `wav_path|phonemes|speaker_id`

**Role**:

* Primary regression set for all Kokoro evaluations (base and fine-tuned).
* NEVER used for training or hyperparameter tuning.
* Ground truth: ElevenLabs David clips for the same prompts used as speaker-similarity reference.

**DVC**: to be tracked after first task runs `dvc add data/v4/val/`.
