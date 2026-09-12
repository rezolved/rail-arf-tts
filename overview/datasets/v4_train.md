# V4 Training Corpus (1557 clips)

**Path**: `data/v4/train_list.txt` (manifest) + `data/v4/train/wavs/` (audio).

**Size**: 1557 clips, 24 kHz mono WAV.

**Source**: David voice filler synthesis sessions — real Rezolve voice commerce prompts
synthesized via ElevenLabs David. Phonemized with eSpeak en-gb in t0003.

**Format** (`train_list.txt`): `wav_path|phonemes|speaker_id`

- `wav_path` — relative to repo root (`data/v4/train/wavs/<name>.wav`)
- `phonemes` — IPA string produced by eSpeak en-gb (via t0003 pipeline)
- `speaker_id` — always `0` (single-speaker corpus)

**Splits**:

| Split | Manifest | Clips | Notes |
|-------|----------|-------|-------|
| Train | `data/v4/train_list.txt` | 1557 | Full training set |
| Val | `data/v4/val_list.txt` | 96 | Held-out; NEVER train on this |
| Subset 250 | `tasks/t0006_kokoro_v5_stage2_subset/code/train_list_250.txt` | 250 | seed=42 random sample; used in t0006 |

**DVC**: `data/v4/train/wavs/` and `data/v4/val/wavs/` tracked via DVC
(`azure://ml-dvc-datasets/datasets/rail-arf-tts`). Run `dvc pull` to download.

**Phonemization**: produced by `tasks/t0003_kokoro_v5_phoneme_data/code/prepare_v5_data.py`.
Canonical lists live in `tasks/t0003_kokoro_v5_phoneme_data/results/v5/`; `data/v4/*.txt`
are copies kept at repo root for convenience.

**Role**: primary fine-tuning corpus for all Kokoro Stage 1 and Stage 2 experiments (t0004–t0006).
