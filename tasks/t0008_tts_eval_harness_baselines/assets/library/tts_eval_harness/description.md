---
spec_version: "2"
library_id: "tts_eval_harness"
documented_by_task: "t0008_tts_eval_harness_baselines"
date_documented: "2026-09-14"
---
# TTS Evaluation Harness

Reusable evaluation harness for TTS systems. Measures speaker similarity (GE2E cosine via
resemblyzer), time-to-first-byte (TTFB), real-time factor (RTF), word error rate (WER), and duration
ratio across multiple TTS backends.

## Metadata

- **Name**: TTS Evaluation Harness
- **Version**: 0.1.0
- **Task**: `t0008_tts_eval_harness_baselines`
- **Dependencies**: numpy, soundfile, torch, kokoro, requests
- **Modules**: `code/harness.py`, `code/adapters.py`, `code/scoring.py`, `code/report.py`,
  `code/extract_decoder.py`, `code/constants.py`, `code/paths.py`, `code/run_eval.py`,
  `code/prepare_prompts.py`, `code/score_speaker_sim.py`

## Overview

Built for `t0008_tts_eval_harness_baselines`, this library provides a single, consistent measurement
tool for all Rezolve TTS research. It wraps eight TTS backends behind a uniform `SynthResult`
interface and computes five metrics: GE2E cosine speaker similarity, time-to-first- byte (TTFB),
real-time factor (RTF), word error rate (WER), and duration ratio. Future fine-tuning tasks import
this library to avoid re-implementing scoring logic.

The harness supports ElevenLabs David (streaming API) and seven Kokoro-82M variants: base models
with `bm_george` and `bm_lewis` voices, the stock decoder with a custom voicepack, the full v3
fine-tuned bundle (five-module decoder + voicepack), and two task-specific checkpoints from t0005
and t0006. A floor-control variant uses the `af_heart` female American voice. All British-English
Kokoro systems run through `build_pipeline()` from
`tasks.t0003_kokoro_v5_phoneme_data.code. build_pipeline` with `lang_code="b"` and the brand
lexicon.

Speaker similarity uses the GE2E encoder from resemblyzer to compute cosine distance between
synthesized audio and a centroid built from 679 ElevenLabs reference clips (seed=42 split). The
other 679 reference clips score ElevenLabs itself to avoid self-comparison. WER uses faster-whisper
`base.en` with JiWER normalization; clips with duration ratio outside 0.5–2.0 are excluded from WER
scoring. Duration ratio flags clips above 5.0 as "explosions" — a known failure mode when only the
decoder module is loaded without the full five-module state dict.

## API Reference

### Synthesis Adapters (`code/adapters.py`)

```python
from tasks.t0008_tts_eval_harness_baselines.code.adapters import (
    SynthResult,
    elevenlabs_david,
    kokoro_george,
    kokoro_lewis,
    kokoro_base_v3_voicepack,
    kokoro_v3_bundle,
    kokoro_t0006_v6d,
    kokoro_t0005_best,
    kokoro_floor_control,
    load_kokoro_model_with_checkpoint,
    save_wav,
)
```

**`SynthResult`** (frozen dataclass):

- `audio_array_16khz: np.ndarray` — float32, 16 kHz resampled audio for resemblyzer
- `audio_array_native: np.ndarray` — float32 at native sample rate for WAV saving
- `native_sample_rate: int`
- `ttfb_s: float` — wall time in seconds from request start to first non-empty audio chunk
- `rtf: float` — `wall_time / audio_duration`
- `audio_duration_s: float`

**`elevenlabs_david(text, *, api_key, voice_id, session)`** — Synthesize via ElevenLabs streaming
API. Returns `SynthResult`. TTFB measured as wall time to first non-empty response chunk. `session`
must be a `requests.Session`.

**`kokoro_george(text, *, pipeline)`**, **`kokoro_lewis(text, *, pipeline)`**,
**`kokoro_floor_control(text, *, pipeline)`** — Synthesize using a pre-built `KPipeline` with a
named built-in voice (`bm_george`, `bm_lewis`, `af_heart`). Return `SynthResult`.

**`kokoro_base_v3_voicepack(text, *, pipeline, voicepack_path)`**,
**`kokoro_v3_bundle(text, *, pipeline, voicepack_path)`**,
**`kokoro_t0006_v6d(text, *, pipeline, voicepack_path)`**,
**`kokoro_t0005_best(text, *, pipeline, voicepack_path)`** — Synthesize using an already-loaded
pipeline and a `.pt` voicepack file path. The distinction between these adapters is which model
weights were loaded into the pipeline; each adapter is a thin wrapper over
`_kokoro_synth_via_ pipeline`.

**`load_kokoro_model_with_checkpoint(ckpt_path)`** — Load a `KModel` with `disable_complex=True`. If
`ckpt_path` is not `None`, load a five-module packaged checkpoint (keys: `bert`, `bert_encoder`,
`predictor`, `text_encoder`, `decoder`). Returns the model ready for `build_pipeline()`.

**`save_wav(audio, sample_rate, path)`** — Write float32 audio to a WAV file via soundfile.

### Scoring (`code/scoring.py`)

**`build_centroid(wav_paths)`** — Compute the L2-normalized mean GE2E embedding from a list of WAV
file paths. Requires the `[speaker-sim]` extra (resemblyzer). Returns `np.ndarray`.

**`compute_speaker_sim(synth_wavs, ref_embeddings)`** — Score synthesized WAV files against a
reference centroid or array of half-B embeddings using GE2E cosine similarity. Returns a
`SpeakerSimResult` with fields `mean`, `std`, `per_clip`.

**`compute_wer(synth_wavs, texts, duration_ratios)`** — Transcribe synthesized audio with
faster-whisper `base.en` and compute WER against reference `texts` using JiWER normalization. Skips
clips where `duration_ratio` is outside 0.5–2.0. Returns `WerResult` with `mean`, `std`, `per_clip`.

**`compute_duration_ratio(synth_wavs, ref_durations_s)`** — Compute per-clip ratio of synthesized to
reference audio duration. Flags clips with ratio > 5.0 as explosions. Returns `DurationResult` with
`median`, `mean`, `std`, `explosion_count`, `per_clip`.

### Harness (`code/harness.py`)

**`build_reference_split(wav_dir, seed)`** — Split the ElevenLabs reference corpus into two equal
halves (679/679), compute a GE2E centroid from half-A, and return `(centroid, half_b_paths)`.
Default `seed=42`.

**`run_system_eval(system_name, adapter_fn, prompts, ref_durations, centroid, half_b_paths, ...)`**
— Run synthesis for one system across all prompts, measuring TTFB and RTF per clip, then score
speaker similarity, WER, and duration ratio. Returns a list of per-clip metric dicts.

### CLI Entry Points

**`code/run_eval.py`** — Full evaluation CLI. Runs synthesis + timing for any combination of systems
and prompt sets, saves per-clip JSON.

**`code/score_speaker_sim.py`** — Speaker similarity CLI. Runs in the resemblyzer venv to compute
GE2E scores and WER on pre-synthesized audio, updating per-clip JSON in place.

**`code/report.py`** — Aggregation CLI. Reads per-clip JSON, writes `metrics.json` (variant format),
`tables.json`, and three PNG charts.

**`code/extract_decoder.py`** — Checkpoint extraction. Converts raw StyleTTS2/Kokoro training
checkpoints to the five-module packaged format.

## Usage Examples

### Full Evaluation Run

```bash
python -m tasks.t0008_tts_eval_harness_baselines.code.run_eval \
    --systems elevenlabs_david kokoro_base_george kokoro_v3_bundle \
    --prompt-set both \
    --n-warmup 1 \
    --out-dir results/ \
    --elevenlabs-api-key sk_... \
    --v3-voicepack-path data/packaged/david_v3_best_voicepack.pt \
    --v3-decoder-path data/packaged/david_v3_best_decoder_kokoro.pth
```

### Programmatic Synthesis

```python
from pathlib import Path
import tasks.t0003_kokoro_v5_phoneme_data.code.build_pipeline as bp
from tasks.t0008_tts_eval_harness_baselines.code.adapters import (
    load_kokoro_model_with_checkpoint,
    kokoro_george,
    save_wav,
)

# Load stock model (pass a checkpoint path for fine-tuned variants)
model = load_kokoro_model_with_checkpoint(ckpt_path=None)
pipeline = bp.build_pipeline(model)

result = kokoro_george("Good morning, this is a test.", pipeline=pipeline)
print(f"TTFB: {result.ttfb_s * 1000:.1f} ms  RTF: {result.rtf:.3f}")
save_wav(result.audio_array_native, result.native_sample_rate, Path("out.wav"))
```

### Speaker Similarity Scoring

```python
from pathlib import Path
from tasks.t0008_tts_eval_harness_baselines.code.scoring import (
    build_centroid,
    compute_speaker_sim,
)

# Build reference centroid from half-A clips
half_a_paths = [Path("data/11labs_david") / f for f in half_a_files]
centroid = build_centroid(wav_paths=half_a_paths)

# Score synthesized clips
result = compute_speaker_sim(synth_wavs=synth_paths, ref_embeddings=centroid)
print(f"Speaker sim: {result.mean:.4f} ± {result.std:.4f}")
```

### Checkpoint Extraction

```python
from pathlib import Path
from tasks.t0008_tts_eval_harness_baselines.code.extract_decoder import extract

counts = extract(
    ckpt_path=Path("epoch_2nd_00006.pth"),
    out_path=Path("data/packaged/t0006_v6d_epoch6.pth"),
)
# counts: {"bert": 25, "bert_encoder": 2, "predictor": 122, "text_encoder": 44, "decoder": 371}
```

## Dependencies

- **numpy** — array operations throughout adapters and scoring
- **soundfile** — WAV read/write for all audio I/O
- **torch** — Kokoro model loading and inference; also for checkpoint manipulation in
  `extract_decoder.py`
- **kokoro** — `KModel`, `KPipeline` for all Kokoro synthesis adapters
- **requests** — ElevenLabs streaming API HTTP client

Optional (not in main dependencies to avoid `webrtcvad`/`pkg_resources` breakage on setuptools ≥
81):

- **resemblyzer** — GE2E speaker encoder for `build_centroid` and `compute_speaker_sim`. Install via
  `pip install rail-arf-tts[speaker-sim]` or use the dedicated venv.
- **faster-whisper** — ASR for WER scoring in `compute_wer`.
- **soxr** — high-quality resampling to 16 kHz (falls back to scipy `resample_poly`).
- **pydub** — MP3 decoding for ElevenLabs audio (falls back to soundfile).

## Testing

Unit tests live in `tasks/t0008_tts_eval_harness_baselines/code/test_harness.py`. Run with:

```bash
uv run pytest tasks/t0008_tts_eval_harness_baselines/code/test_harness.py -v
```

Tests cover:

- `TestDurationRatio` (3 tests) — normal ratio, explosion detection (ratio > 5.0), reference-absent
  handling
- `TestWer` (5 tests) — WER computation, duration-ratio gating (clips outside 0.5–2.0 excluded),
  normalization behavior
- `TestReportJson` (3 tests) — per-clip JSON round-trip, variant-format `metrics.json` output,
  `tables.json` structure

The speaker similarity, ElevenLabs API, and Kokoro pipeline tests require optional infrastructure
(resemblyzer venv, API key, GPU) and are exercised by the integration run in `run_eval.py` rather
than in the unit test suite.

## Main Ideas

* **Adapter pattern**: each TTS system is a pure function `(text, **kwargs) -> SynthResult` so
  callers treat all backends uniformly. Models are loaded by the caller, not by the adapter,
  avoiding repeated checkpoint loading across hundreds of prompts.

* **Reference centroid split**: 1358–1364 ElevenLabs David clips are split 679/679 (seed=42). Half-A
  builds the centroid used to score every system including ElevenLabs itself. Half-B clips are used
  when scoring ElevenLabs to avoid self-comparison bias.

* **Five-module checkpoint requirement**: Loading only the `decoder` state dict from a raw StyleTTS2
  checkpoint causes duration explosions (synthesized audio ~10× longer than reference). The
  `extract_decoder.py` utility always packages all five modules (`bert`, `bert_encoder`,
  `predictor`, `text_encoder`, `decoder`) and strips DDP prefixes and weight-norm reparametrization
  keys for clean loading.

* **resemblyzer kept optional**: resemblyzer depends on `webrtcvad`, which uses `pkg_resources` from
  setuptools. On setuptools ≥ 81 this crashes the main pyproject install. Speaker similarity runs in
  a separate venv or is invoked via `code/score_speaker_sim.py` with an explicit `PYTHONPATH`.

* **Warmup discarding**: at least one warmup synthesis is always run and discarded before timing
  begins. This prevents JIT compilation and model cache warm-up effects from inflating TTFB and RTF
  numbers.

* **Torch.Tensor compatibility**: kokoro 0.9.x on Python 3.11 returns `torch.Tensor` from the
  pipeline iterator. All Kokoro adapters explicitly convert to `np.ndarray` via
  `.detach().cpu().numpy()` before concatenation.

## Summary

The TTS Evaluation Harness provides a reusable, backend-agnostic framework for comparing TTS systems
on the metrics that matter for production filler synthesis: speaker identity (GE2E cosine), latency
(TTFB), throughput (RTF), intelligibility (WER), and duration fidelity (duration ratio). It was
built as part of `t0008_tts_eval_harness_baselines` and is designed for import by future fine-tuning
tasks such that evaluation logic does not need to be re-implemented.

The harness handles the significant environment complexity of Rezolve's Kokoro-82M stack: five-
module checkpoint packaging, `build_pipeline` integration with the brand lexicon, Python 3.11
tensor-vs-array incompatibilities, and the resemblyzer isolation requirement. By centralizing all of
this in one library, downstream tasks can focus on model development rather than evaluation
plumbing.

All synthesis adapters share a common `SynthResult` dataclass and a single internal
`_kokoro_synth_via_pipeline` helper. Adding a new Kokoro variant requires only a one-line wrapper
function. The CLI entry points (`run_eval.py`, `score_speaker_sim.py`, `report.py`) are thin
argument-parsing shells that delegate entirely to the library functions, keeping them easily
testable independently of the command line.
