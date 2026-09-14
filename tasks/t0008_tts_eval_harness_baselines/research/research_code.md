---
spec_version: "1"
task_id: "t0008_tts_eval_harness_baselines"
research_stage: "code"
tasks_reviewed: 7
tasks_cited: 5
libraries_found: 0
libraries_relevant: 0
date_completed: "2026-09-14"
status: "complete"
---
## Task Objective

Build the project's first quantitative measurement tool: a reusable evaluation harness that scores
speaker similarity (GE2E cosine), TTFB, and RTF for eight TTS systems (ElevenLabs David, two base
Kokoro voicepacks, four fine-tuned checkpoints, and a floor-control voice) across two prompt sets
(val_96 and ≥ 50 filler prompts). The harness is delivered as a registered library asset so every
future training task can reuse it. No training is run; this task measures what already exists.

* * *

## Library Landscape

Zero libraries have been registered in `assets/library/` across the seven completed tasks
(t0001–t0007). All seven tasks were fine-tuning, data-prep, or brainstorming tasks; none produced a
reusable library asset. The library aggregator confirms `library_count: 0`. No cross-task library
import is therefore available, and all reusable code from prior tasks must be **copied** into
`tasks/t0008_tts_eval_harness_baselines/code/`.

The absence of a library layer means this task is the first to register one. The harness produced
here should be registered so downstream training tasks (t0009 and later) can call it directly.

* * *

## Key Findings

### Checkpoint Packaging: Five-Module Extraction Is Required

t0002 discovered that v3's `david_v3_best_decoder_kokoro.pth` packages **five** modules, not just
the decoder: `bert`, `bert_encoder`, `predictor`, `text_encoder`, and `decoder` [t0002]. The file
name is a misnomer inherited from the v3 packaging script. The `extract_decoder_generic.py` (38
lines) captures all five modules with correct key remapping:

* Strips the `module.` DDP prefix (`k.removeprefix("module.")` pattern is in `make_voicepack_v4.py`
  but the generic extractor uses `k[7:]` conditionally)
* Remaps `parametrizations.weight.original0` → `weight_g` and `.original1` → `weight_v`
* Saves `{mod: {converted_key: tensor}}` for all five modules

t0002 also confirmed that loading fewer than five modules into `KModel` produces either a duration
explosion (Stage 2 predictor with base weights produces 89 s for a 9.6 s sentence) or persistent
tonal artifacts (~3–3.5 kHz band in silent pauses). For t0008, the v3 voicepack and decoder bundle
at `tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/best/` (DVC-tracked) already packages all
five correctly; the t0005 and t0006 raw checkpoints (`epoch_2nd_*.pth`) must be packaged by copying
and adapting `extract_decoder_generic.py`.

### Synthesis Entry Point: build_pipeline.py Is Mandatory for All Kokoro Arms

t0003 established that all Kokoro synthesis must go through
`tasks/t0003_kokoro_v5_phoneme_data/code/build_pipeline.py` [t0003]. The function
`build_pipeline(model)` returns a `KPipeline` with:

* `lang_code="b"` (British English — matches the corpus G2P)
* The full brand lexicon installed on `pipeline.g2p` (62 OOV words including `rezolve`, `brainpowa`,
  `agentic`, all Rezolve brand names)

Using bare `KPipeline(lang_code="a")` (American) or omitting the lexicon are both silent failures:
misaki emits `❓` for unknown words (not in Kokoro's 114-symbol vocab), and American vowel symbols
don't match what the v5-trained model saw during training. t0006's raw inference scripts
(`infer_v6c.py`, `infer_v6d.py`, ~150 lines each) bypass this path entirely — they use raw StyleTTS2
model objects and espeak G2P [t0006]. The harness must NOT use those scripts for evaluation; it must
use `build_pipeline.py` for any Kokoro arm that loads a packaged `KModel` + voicepack.

Note: `build_pipeline` imports from `tasks.t0003_kokoro_v5_phoneme_data.code` — this is a cross-task
import that works only because `build_pipeline.py` is not a library (no `assets/library`
registration). The harness must either copy the full `build_pipeline.py` + `constants.py` +
`lexicon.py` + the two functions from `prepare_v5_data.py` (`install_lexicon`, `load_kokoro_vocab`)
into `tasks/t0008_tts_eval_harness_baselines/code/`, or import them directly (acceptable since t0003
is a dependency's sibling and this repo is a monorepo). The cleaner option is to import directly,
preserving the single source of truth for the lexicon.

### KModel Loading Pattern for Fine-Tuned Checkpoints

t0002's `test_kokoro_inference_generic.py` (56 lines) establishes the canonical `KModel` loading
pattern [t0002]:

```python
model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True)
state = torch.load(decoder_path, map_location="cpu", weights_only=False)
for mod_name, sd in state.items():
    target = getattr(model, mod_name)
    target.load_state_dict(sd, strict=False)
model.eval()
pipe = KPipeline(lang_code="b", model=model, repo_id="hexgrad/Kokoro-82M")
```

Key details: `disable_complex=True` selects the real-FFT ISTFT path in `istftnet.py` (more
numerically stable on CPU); `strict=False` is required because the five-module bundle may have
slightly different key sets than the base `KModel`. The `lang_code` here should be `"b"` (British)
for David fine-tuned models — `test_kokoro_inference_generic.py` uses `"a"` (American), which is
wrong for David-voice checkpoints and must be corrected when copying.

### Checkpoint Locations and Disambiguation

Both t0005 and t0006 have DVC-tracked checkpoints but the actual `.pth` files are not checked into
git — only `.dvc` pointer files. Running `dvc pull` before evaluation is required.

* **t0005 run06 best**: `tasks/t0005_kokoro_v5_stage2_train/results/checkpoints/epoch_2nd_00003.pth`
  (val_loss=0.848, epoch 3). The results_summary says "epoch 3" but the task_description logs say
  "epoch 2 or 3"; the DVC pointer file is named `epoch_2nd_00003.pth.dvc`, confirming epoch 3.
  Record which file was actually scored (from t0008's task description: "record which file was
  actually scored") [t0005].

* **t0006 v6d best**:
  `tasks/t0006_kokoro_v5_stage2_subset/results/checkpoints/v6d/epoch_2nd_00006.pth` (val_loss=0.846,
  epoch 6) [t0006].

* **v3 bundle** (gold reference): two files at
  `tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/best/`:
  * `david_v3_best_decoder_kokoro.pth` (327 MB, five-module format, DVC-tracked)
  * `david_v3_best_voicepack.pt` (523 KB, `[510, 1, 256]` ref_s tensor, DVC-tracked) A third file
    `david_v3_voicepack_ft_enc.pt` exists but is a variant; use `david_v3_best_voicepack.pt`.

* **Base Kokoro voicepacks**: `bm_george` and `bm_lewis` use the stock `KModel` (no custom decoder)
  with built-in voicepack names — no file path needed, just `voice="bm_george"` in the pipeline.

* **Base decoder + v3 voicepack** (system 4): stock `KModel` (no decoder replacement) +
  `david_v3_best_voicepack.pt`. This tests how much of the voice the voicepack alone captures with
  the base Kokoro decoder.

### Val_96 Text Extraction Pattern

t0003 produces `tasks/t0003_kokoro_v5_phoneme_data/results/v5/val_list.txt` (96 lines) [t0003]. The
format is `wav_path|phonemes|speaker_id`. The harness needs the original text, not the
pre-phonemized form. The best approach is to parse the wav filename (which encodes a slug of the
original text) or, more reliably, trace back to the source manifests. However, `build_pipeline.py`
accepts raw text and phonemizes itself — so the harness can just feed the text from the original
manifest CSV (at `data/train/manifest.csv` and `data/val/manifest.csv` on the benchmark machine).
Alternatively, the val audio files can be synthesized from the phoneme strings directly via a custom
pipeline; both approaches are valid.

### ElevenLabs API Streaming Pattern

No prior task has used the ElevenLabs streaming API in this repo. The research_internet step found
that ElevenLabs' streaming endpoint for TTFB measurement uses `text-to-speech/{voice_id}/stream`
with an `Accept: audio/mpeg` header; TTFB is wall time from request start to first audio chunk
(`iter_content(chunk_size=1)`). Voice ID for David: the project description references "ElevenLabs
David voice" but does not record the voice_id; the harness must look it up from the ElevenLabs
`/voices` API using the API key in the project's `.env`.

### DVC Pull Is Required Before Any Evaluation

All checkpoint and reference bundle paths have `.dvc` pointer files only; no binary data is in git.
The harness or its setup step must run `dvc pull` for the relevant paths before synthesis begins.
The 11labs filler corpus also needs to be copied from the VM and DVC-added as part of this task's
setup (it is not yet tracked by DVC, per `overview/datasets/fillers_1358.md`).

### LESSONS.md Constraints That Affect Harness Design

Lesson 1: Warmup before any latency measurement. For Kokoro TTFB, at least one discarded synthesis
call per system before recording p50/p95/p99. For ElevenLabs, at least one discarded streaming
request before measurement.

Lesson 4: Capture torch, CUDA, kokoro, and whisper versions plus GPU identity at the start of every
TTFB measurement run. Write these to the results metadata.

Lesson 10: `/mnt/kikiri-tts/` is ephemeral on the Azure ML VM. The filler corpus must be confirmed
present before planning around it. All output audio from TTFB runs must be synced to the persistent
mount or DVC-pushed before the VM is torn down.

* * *

## Reusable Code and Assets

### `extract_decoder_generic.py` — Five-Module Checkpoint Extractor

* **Source**: `tasks/t0002_kokoro_v4_voicepack_decoder_package/code/extract_decoder_generic.py`
* **What it does**: Loads a raw StyleTTS2 Stage 2 checkpoint (`.pth` with `net` dict), extracts the
  five modules (`bert`, `bert_encoder`, `predictor`, `text_encoder`, `decoder`) with key remapping
  for DDP prefix and weight-norm parametrization, and saves a five-module state dict loadable by
  `KModel`.
* **Reuse method**: **copy into task** (not a library). Copy to
  `tasks/t0008_tts_eval_harness_baselines/code/extract_decoder.py` and adapt: the `CKPT_DIR`
  constant is hardcoded to t0001; replace with a function argument.
* **Key function**: `main()` reads `sys.argv[1]` (checkpoint name in `CKPT_DIR`) and `sys.argv[2]`
  (output name in `TASK_DIR/results/`). For the harness, adapt to accept arbitrary paths.
* **Line count**: 38 lines. Minimal adaptation needed.

### `build_pipeline.py` + `constants.py` + `lexicon.py` + `install_lexicon` / `load_kokoro_vocab`

* **Source**: `tasks/t0003_kokoro_v5_phoneme_data/code/build_pipeline.py` (30 lines),
  `tasks/t0003_kokoro_v5_phoneme_data/code/constants.py` (55 lines),
  `tasks/t0003_kokoro_v5_phoneme_data/code/lexicon.py` (93 lines),
  `tasks/t0003_kokoro_v5_phoneme_data/code/prepare_v5_data.py:99-140` (`load_kokoro_vocab`,
  `install_lexicon`, `phonemize` — ~60 lines total)
* **What it does**: Returns a `KPipeline` with British G2P (`lang_code="b"`) and the brand lexicon
  pre-installed. This is the mandatory entry point for all Kokoro synthesis.
* **Reuse method**: **import directly** (acceptable for this monorepo — t0003 is already a completed
  dependency of t0006, which this task depends on). The harness calls
  `from tasks.t0003_kokoro_v5_phoneme_data.code.build_pipeline import build_pipeline`. No copy
  needed; the single lexicon source of truth is preserved.
* **Function signature**: `build_pipeline(model: Any) -> Any` — takes a `KModel` instance, returns a
  `KPipeline`.
* **Adaptation needed**: None. Use as-is.

### `test_kokoro_inference_generic.py` — KModel Loading Pattern

* **Source**:
  `tasks/t0002_kokoro_v4_voicepack_decoder_package/code/test_kokoro_inference_generic.py`
* **What it does**: Loads a five-module bundle into `KModel`, wraps it in `KPipeline`, synthesizes
  text, writes WAV. Shows the correct `strict=False` load pattern and chunk-iteration over the
  pipeline generator.
* **Reuse method**: **copy into task** as a template. The harness adapter for fine-tuned systems
  follows this pattern exactly, with two corrections: `lang_code="b"` (not `"a"`), and the pipeline
  must be created via `build_pipeline(model)` rather than bare `KPipeline(...)`.
* **Key pattern**: iterating `for _, _, chunk in pipe(text, voice=voicepack, speed=1.0)` to collect
  audio chunks, `torch.cat(chunks)` to assemble.
* **Line count**: 56 lines.

* * *

## Dataset Landscape

* **val_96** (96 clips): `data/v4/val/wavs/` + `data/v4/val/val_list.txt` (manifest, not yet
  DVC-tracked). Text is recoverable from the clip filename slug or from the val manifest CSV.
* **Filler corpus** (1358 clips): on the VM at `/mnt/kikiri-tts/data/11labs_david/`. Not yet
  DVC-tracked. Must be copied off the VM and tracked in this task's `data/11labs_david/`. Confirm it
  still exists (Lesson 10: `/mnt` is ephemeral).
* **t0005 run06 best checkpoint**:
  `tasks/t0005_kokoro_v5_stage2_train/results/checkpoints/epoch_2nd_00003.pth.dvc` (1.87 GB).
  Requires `dvc pull`.
* **t0006 v6d epoch 6 checkpoint**:
  `tasks/t0006_kokoro_v5_stage2_subset/results/checkpoints/v6d/epoch_2nd_00006.pth.dvc` (1.86 GB).
  Requires `dvc pull`.
* **v3 reference bundle**: `tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/best/` — two files
  DVC-tracked (`david_v3_best_decoder_kokoro.pth` 327 MB, `david_v3_best_voicepack.pt` 523 KB).

* * *

## Lessons Learned

### val_loss Does Not Predict Perceptual Quality

t0001–t0006's combined experience shows that `val_loss` alone is not a reliable quality indicator
[t0005][t0006]. t0003's v3 gold-standard bundle has val_loss=0.506; t0005 run06 achieved 0.848 and
t0006 v6d achieved 0.846 — both considered "close" by loss but described as noisy by listening. The
harness must measure GE2E cosine and WER; the training pipeline's loss signal is not a substitute.

### The Three Common Silent Failure Modes in This Codebase

1. **Wrong lang_code at inference**: `lang_code="a"` (American) vs `"b"` (British) produces
   different IPA tokens silently. t0003 caught this; t0006's raw inference scripts still used espeak
   en-gb rather than the misaki+lexicon path [t0003][t0006].
2. **Fewer than five modules loaded**: loading only `decoder` leaves `predictor` as base Kokoro,
   causing duration explosions (89 s for 9.6 s sentences) [t0002].
3. **Missing lexicon**: `Rezolve` phonemizes to `❓` (not in Kokoro vocab, silently dropped) without
   the brand lexicon [t0003].

### DVC Pull Must Precede Every Evaluation Run

t0005 and t0006 checkpoints are DVC-tracked only; the `.pth` files are not in git. Any harness run
without a prior `dvc pull` for the relevant paths will fail with "file not found" rather than a
clear error.

### Ephemeral VM Storage (LESSONS.md Lesson 10)

The filler corpus lives on the VM's `/mnt/kikiri-tts/` mount, which is wiped on every stop. This
task's first action on the VM must be to confirm the corpus is present, copy it to a persistent
location, and `dvc push` before teardown. Do not assume the corpus survives from t0003's
description.

* * *

## Recommendations for This Task

1. **Copy `extract_decoder_generic.py` into task code, adapt for arbitrary paths.** The CKPT_DIR
   hardcode is the only adaptation needed. Use this to package t0005 run06 and t0006 v6d checkpoints
   as five-module bundles before synthesis.

2. **Import `build_pipeline` directly from t0003; do not copy.** The lexicon is a single source of
   truth with 62 entries that must not diverge. Import as
   `from tasks.t0003_kokoro_v5_phoneme_data.code.build_pipeline import build_pipeline`.

3. **Use `test_kokoro_inference_generic.py` as the template for the fine-tuned synthesis adapter.**
   Fix the two bugs: `lang_code="b"` and delegate to `build_pipeline(model)`.

4. **Correct `lang_code` in every Kokoro synthesis call.** `"b"` (British) is mandatory for all
   David-voice arms. `"a"` (American) is wrong and produces token-space mismatch with the v5
   training corpus.

5. **DVC-pull t0005 and t0006 checkpoints and v3 bundle before synthesis.** Add this as an explicit
   step in the implementation plan. The paths are:
   * `tasks/t0005_kokoro_v5_stage2_train/results/checkpoints/epoch_2nd_00003.pth`
   * `tasks/t0006_kokoro_v5_stage2_subset/results/checkpoints/v6d/epoch_2nd_00006.pth`
   * `tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/best/david_v3_best_decoder_kokoro.pth`
   * `tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/best/david_v3_best_voicepack.pt`

6. **Confirm VM filler corpus before teardown.** On setup-machines:
   `ls /mnt/kikiri-tts/data/11labs_david/` and verify count ≈ 1358 WAVs. If absent, fall back to
   regenerating from `fillers_from_logs.txt` via ElevenLabs API. If present, `rsync` to a local
   path, `dvc add`, `dvc push` before teardown.

7. **Record version metadata at TTFB measurement time (LESSONS.md Lesson 4).** Capture: `torch`
   version, `kokoro` version, CUDA version (`nvidia-smi`), GPU model. Write to the harness results
   metadata JSON.

8. **Warmup before TTFB measurement (LESSONS.md Lesson 1).** Run ≥ 1 discarded synthesis call per
   system before recording. ElevenLabs needs ≥ 1 discarded streaming request.

9. **Register the harness as a library asset.** The task expects exactly one library asset. Define
   the entry points (prompt loader, per-system synthesizer, scorer, report writer) so future tasks
   can import via `from tasks.t0008_tts_eval_harness_baselines.code.<module> import ...` through the
   registered library path.

* * *

## Task Index

### [t0002]

* **Task ID**: t0002_kokoro_v4_voicepack_decoder_package
* **Name**: Package Kokoro v4 voicepack + decoder
* **Status**: completed
* **Relevance**: Contains `extract_decoder_generic.py` — the five-module checkpoint extractor
  required to package t0005 and t0006 raw StyleTTS2 checkpoints into Kokoro-loadable format. Also
  validates the full `KModel` + `KPipeline` loading pipeline and documents the duration explosion
  caused by loading fewer than five modules.

### [t0003]

* **Task ID**: t0003_kokoro_v5_phoneme_data
* **Name**: Regenerate v5 phoneme manifests
* **Status**: completed
* **Relevance**: Provides `build_pipeline.py` — the mandatory synthesis entry point for all Kokoro
  arms in this task. Also provides the val_96 manifest (`results/v5/val_list.txt`) and the brand
  lexicon (`lexicon.py`) that must be active during all David-voice inference.

### [t0005]

* **Task ID**: t0005_kokoro_v5_stage2_train
* **Name**: Kokoro v5 Stage 2 fine-tune
* **Status**: completed
* **Relevance**: Provides the t0005 run06 best checkpoint
  (`results/checkpoints/epoch_2nd_00003.pth`, val_loss=0.848), one of the two checkpoints to be
  scored in this task. Documents that epoch 3 is the DVC-tracked best, not epoch 2.

### [t0006]

* **Task ID**: t0006_kokoro_v5_stage2_subset
* **Name**: Kokoro v5 Stage 2: 250-clip subset, multispeaker: false
* **Status**: completed
* **Relevance**: Provides the t0006 v6d epoch 6 checkpoint
  (`results/checkpoints/v6d/epoch_2nd_00006.pth`, val_loss=0.846) and the v3 gold-standard reference
  bundle (`data/reference/v3/best/`) that is the primary system-4/system-5 baseline. Also provides
  raw inference scripts (`infer_v6d.py`) whose patterns the harness must NOT follow (wrong synthesis
  path, uses bare espeak not `build_pipeline`).

### [t0007]

* **Task ID**: t0007_brainstorm_results_1
* **Name**: Brainstorm results session 1
* **Status**: completed
* **Relevance**: Established the requirement for this evaluation harness task, confirmed the eight
  systems to score, and documented the cost tracking bug (`total_usd` vs `total_cost_usd`) that
  t0008 must fix in its own `results/costs.json`.
