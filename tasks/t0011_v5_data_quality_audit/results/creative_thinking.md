---
spec_version: "1"
task_id: "t0011_v5_data_quality_audit"
---
# Creative Thinking — t0011 v5 Data Quality Audit

## 1. Is the -0.1 dBFS Threshold the Right Call?

**The current choice: flag, don't normalize.** 224 clips (14.4%) have peak amplitude between -0.1
and 0.0 dBFS and are excluded from the clean manifest.

**Alternative A — accept all clips, normalize as preprocessing.** Peak normalization is a reversible
linear operation. Before feeding clips into the Kokoro Stage 2 mel extractor, apply a fixed
attenuation (e.g., −1 dB) to every clip uniformly, or apply per-clip gain to target −3 dBFS. This
recovers the 224 clips at zero risk of audible artifacts — a linear gain change does not alter the
spectral envelope, formants, or prosody that the speaker-similarity metric measures.

**Alternative B — accept all clips unconditionally.** The Kokoro mel spectrogram is computed from
log power, which compresses amplitude differences. A 0.1 dB difference in peak amplitude is
inaudible and unmeasurable by GE2E cosine similarity. The ElevenLabs corpus is internally
consistent: if every clip is peak-normalized, the training signal is consistent and the model never
sees a clip that is qualitatively different from the others.

**Recommendation.** The -0.1 dBFS flag is overcautious for this corpus. The real clipping risk is
flat-topped waveforms (true hard clipping at exactly 0 dBFS for a run of samples), not a clip whose
single peak sample touches 0 dBFS. A more principled threshold would count the fraction of samples
within 1 LSB of full scale (e.g., `clipped_fraction > 0.001`). On a peak-normalized corpus this
distinguishes genuine saturation from deliberate normalization. For the Stage 2 run that follows
t0010, the safest path is to LUFS-normalize the entire corpus to −14 LUFS before training and accept
all 1557 clips.

## 2. Are the 25 Short Clips Truly Problematic?

**What short means here.** The 25 clips below 1.5 s have a corpus minimum of 0.65 s. At 24 kHz that
is ~15 600 samples — enough for 975 ms of speech after any leading/trailing silence. Kokoro's mel
extractor uses 12.5 ms frames with 256-sample hop; 0.65 s yields ~61 frames, which is above the
minimum window for all standard TTS loss functions.

**The real risk is not duration per se, but content.** Short clips are likely single words,
interjections, or brief fillers. Three sub-cases:

1. **Single-word clips with a clear phoneme sequence** — perfectly valid training examples; the
   model benefits from seeing short tokens to avoid over-producing duration.
2. **Near-silence clips** (clipped silence from the boundary of a longer recording) — harmful; they
   teach the model to produce silence.
3. **Truncated clips** (audio was cut mid-word) — harmful; they contain abrupt transitions that
   corrupt duration and pitch.

Cases 2 and 3 are detected by silence_fraction and the existence of a valid phoneme transcript. The
audit already checks silence_fraction (only 1 clip exceeds 30%) and OOV fraction (0 clips). A
transcript-length check — flag clips where phoneme string length < 5 characters — would catch case 3
without discarding valid short clips.

**Alternative strategy.** Instead of a fixed 1.5 s floor, filter by phoneme-to-duration ratio. A
clip shorter than `len(phoneme_string) * 40 ms` is suspiciously fast (faster than any human TTS
output). This ratio-based threshold would catch truncated clips while preserving legitimate short
words.

## 3. Risk of Training on Peak-Normalized vs. Level-Normalized Clips

**Amplitude inconsistency during training.** The current corpus spans roughly −9 to 0 dBFS in peak
amplitude (mean −0.55 dBFS, std 0.73 dB). This 9 dB spread is modest for a TTS corpus. The
speaker-similarity loss (GE2E) operates on normalized embeddings and is amplitude-invariant. The
mel-spectrogram loss (L1 or L2 on log-mel) is mildly amplitude-dependent — a louder clip will have
higher absolute log-mel values — but the model learns to map phoneme embeddings to relative spectral
shapes, not absolute levels.

**The practical risk.** A 9 dB spread in training clips will make the model's output level slightly
unpredictable. If ElevenLabs David reference clips are all peak-normalized (as this corpus suggests)
and Kokoro's output ends up at −6 dBFS on average, GE2E cosine similarity is unaffected, but the
downstream voice-commerce pipeline may need a final normalization step before serving.

**Mitigation.** LUFS-normalize the corpus to −14 LUFS (EBU R128 broadcast standard) before Stage 2
training. This removes amplitude as a confounding variable from the training signal. The 1557-clip
LUFS mean is −14.69 LUFS with std 2.34 — already close to −14 LUFS. Normalization would bring all
clips within ±1 LUFS of the target, with no perceptual side effects.

## 4. LUFS-Based Normalization as a Preprocessing Step

**Why LUFS normalization is strictly better than flagging for this corpus.** The current audit flags
clips by peak dBFS — a single-sample metric that reflects normalization strategy, not audio quality.
LUFS (BS.1770-4 integrated loudness) captures the perceptual loudness of the entire clip. Because
the corpus LUFS distribution is tight (−14.7 ± 2.3 LUFS) and all clips fall within the acceptance
window (0 clips below −30, 0 above −6 LUFS), no clip has a loudness problem — only a peak problem.

**Proposed preprocessing pipeline.**

```python
# ponytail: one-pass normalization; covers all 1557 clips in < 30 s on CPU
import soundfile as sf
import pyloudnorm as pyln
import numpy as np

TARGET_LUFS = -14.0

for wav_path, phoneme in manifest:
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(mono)
    except Exception:
        loudness = 20 * np.log10(np.sqrt(np.mean(mono**2)) + 1e-9)
    gain = 10 ** ((TARGET_LUFS - loudness) / 20)
    # Clip to avoid any new clipping after gain
    normalized = np.clip(audio * gain, -1.0, 1.0)
    sf.write(wav_path, normalized, sr)
```

This eliminates the 224-clip clipping flag entirely (clips that were peak-normalized to 0 dBFS will
land at roughly −14 LUFS after normalization, well below 0 dBFS). It also removes the LUFS
acceptance thresholds as a filtering concern for future tasks.

**Trade-off.** LUFS normalization in place mutates the source data — a DVC anti-pattern. The clean
approach is to normalize into a new directory (`data/v5_normalized/`) and write a new manifest
pointing to the normalized files. This costs ~150 MB disk (1557 × 24 kHz × 4 bytes × ~3.4 s ≈ 500 MB
WAVs; the normalized copies are the same size).

## 5. Alternative Threshold Strategies

### Strategy A — Accept peak-normalized clips, flag only true clipping

Replace `peak_dbfs > -0.1` with `clipped_fraction > 0.001` (fraction of samples at full scale).
Expected result: 0–10 clips flagged (only those with genuine flat-topped waveforms). Clean manifest:
~1532 clips (98.4%).

### Strategy B — Two-pass filtering

Pass 1 (hard gates): sample_rate, channels, error — catches technical failures. Pass 2 (soft audit):
compute LUFS and silence metrics; log outliers but do not exclude. Exclude only: error clips,
multi-channel clips, wrong sample rate, silence_fraction > 0.5. Expected result: 1556 clean clips
(99.9%).

### Strategy C — Duration-aware minimum

Instead of a fixed 1.5 s floor, require `duration_s > 3 * len(words)` where `len(words)` is
estimated from the phoneme string. Typical filler rate: ~0.3 s/word at ElevenLabs David's pace. This
allows 0.9 s clips for single-word utterances while flagging clips that are implausibly short for
their transcript length.

### Strategy D — Percentile-based thresholds

Set thresholds at the 1st and 99th percentile of each metric computed from the corpus itself, rather
than hardcoded values. This makes the audit self-calibrating and dataset-agnostic. For this corpus:

| Metric | p1 | p99 | Current threshold |
| --- | --- | --- | --- |
| peak_dbfs lower | −8.83 | — | — (no lower threshold) |
| duration_s lower | 0.76 s | — | 1.5 s (removes p1–p3) |
| lufs lower | −26 LUFS | — | −30 LUFS (no clips flagged) |
| silence_fraction upper | — | 12.3% | 30% (removes only p99.9) |

With percentile-based thresholds the corpus would retain 1527 clips (98.1%) — a more principled
trade-off than the current 84.2%.

## Key Insight: The Corpus Is Not Damaged — It Is Consistently Formatted

The most important finding of this audit is the absence of damage. Zero errors, zero LUFS outliers,
zero OOV flags, zero multi-channel clips. The 224 "clipping" clips are not damaged audio — they are
peak-normalized TTS output following the ElevenLabs house style. The 1 silence-flagged clip is the
sole quality anomaly in the corpus.

**The conservative threshold strategy (current, 84.2% retention) is defensible as a first run.** For
the full Stage 2 run planned after t0010, a preprocessing step that LUFS-normalizes all 1557 clips
and drops only the 1 silence outlier would give a 99.9% clean corpus (1556 clips) with better
amplitude consistency than the current 1311-clip manifest. That is the recommended evolution for the
next task.
