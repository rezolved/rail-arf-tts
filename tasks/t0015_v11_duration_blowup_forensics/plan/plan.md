---
spec_version: "2"
task_id: "t0015_v11_duration_blowup_forensics"
date_completed: "2026-09-17"
status: "complete"
---
# Plan: v11 Duration-Blowup Forensics and Audible-Speech Gate Hardening

## Objective

`kokoro-v11-best` (produced by `t0014_v11_decoder_fix_retrain`) passed the mandatory audible-speech
gate (`audio_quality_check.py`'s `is_likely_noise=False`, `clip_fraction=0.0048`,
`spectral_flatness=0.0058`) yet its synthesis output for a ~10-word test sentence is **73.95
seconds** long — 15-30x the 2.5-4.9s produced by the control checkpoint and both v10 checkpoints on
comparable inputs — and a human listener (the user) confirmed the file
(`tasks/t0014_v11_decoder_fix_retrain/results/audio_samples/ft/v11_best.wav`) is droning babble, not
intelligible speech. Per-second waveform analysis already run outside this plan (documented in
`task_description.md`) shows the clip has locally speech-like spectral texture (low flatness, low
clipping) for the *entire* 73.9 seconds with zero silence gaps — a third failure mode
(locally-speech-shaped, structurally-unbounded audio) that neither `spectral_flatness` (tuned for
white noise) nor `clip_fraction` (tuned for vocoder saturation) was ever designed to catch.

This plan produces:

1. A root-cause diagnosis of the duration blowup — whether it originates in the duration predictor's
   calibration (`model.predictor.duration_proj`'s output scale) or in downstream
   alignment/frame-count plumbing (`pred_aln_trg` construction) — backed by direct instrumentation
   logging, not inference from audio length alone.
2. A characterization of whether the blowup is universal or text-dependent, across at least 10
   varied texts (short filler phrases and longer sentences).
3. A cheap, no-retrain fix attempt: an inference-parameter sweep
   (`alpha`/`beta`/`diffusion_steps`/`embedding_scale`) against pre-registered pass criteria, with
   an honest negative result and a follow-up-task recommendation if no combination fixes it.
4. A hardened `audio_quality_check.py` with two new signals — a text-only duration-sanity check and
   a longest-contiguous-non-silent-run check — proven, via a three-way v10/v11/corrected regression,
   to catch this exact failure mode that the original two-signal gate missed.

**Success criteria ("done" looks like)**: `results/duration_blowup_diagnosis.md` states a
definitive, evidence-backed root cause (predictor-calibration vs. plumbing) with the raw
`pred_dur`/frame-count numbers to back it; the parameter sweep's outcome (worked or didn't) is
documented with the same rigor either way; `audio_quality_check.py` has two new signals that are
proven, by a reproducible three-way regression run, to flag `v11_best.wav` as failing while still
passing the known-good control; and `results/metrics.json` records `rtf` (and `speaker_sim` wherever
new audio samples are produced) for every synthesis condition measured. No claim of "fixed" or "gate
hardened" is made without the corresponding evidence file backing it.

## Task Requirement Checklist

Verbatim from `task.json`:

> **name**: "v11 duration-blowup forensics and audible-speech gate hardening" **short_description**:
> "Root-cause why kokoro-v11-best's synthesis runs 15-20x too long and sounds like a droning babble
> to a human, despite passing the clipping/flatness noise gate, then close that gate's blind spot."

Verbatim key passages from `task_description.md` (the resolved long description):

> "Where does the blowup actually originate? ... Is `pred_dur` itself absurdly large (a predictor
> calibration problem), or is the bug downstream in how the alignment matrix or frame count is
> constructed (a plumbing bug ...)? Log `pred_dur`'s raw values and `pred_aln_trg`'s resulting frame
> count directly — don't infer this from audio length alone."

> "Is this specific to t0014's decoder-init fix ...? ... check t0014's `ignore_modules` list and
> Stage 2 training logs for whether `predictor`/`predictor_encoder` actually received gradient
> updates ..."

> "Does every synthesized text blow up, or only some? ... Run the harness across a wider, varied
> sample of texts (short and longer ...) and characterize whether the blowup is universal,
> text-length dependent, or intermittent."

> "Can this be fixed cheaply, at inference time, with no retraining? ... sweep
> `infer_styletts2.py`'s `alpha`/`beta`/`diffusion_steps`/`embedding_scale` parameters ... and see
> whether a different diffusion-sampler configuration produces well-calibrated durations."

> "If the inference-side sweep fails to fix it: is targeted fine-tuning of just
> `predictor`/`predictor_encoder` ... a smaller, cheaper remediation ...? Scope this as a
> recommendation for a follow-up task rather than doing it here ..."

> "What is the cheapest reliable way to make the audible-speech gate catch this class of defect
> going forward? At minimum: a duration-sanity check ... and a 'has silence gaps' check (max
> contiguous non-silent run ...) ... An ASR-based intelligibility check ... is a stronger but
> heavier option worth evaluating ..."

Requirement IDs (used throughout this plan and in the implementation agent's completion notes):

| ID | Requirement | Satisfied by | Evidence |
| --- | --- | --- | --- |
| REQ-1 | Instrument `synthesize()` to log/return `pred_dur` per-token values, `pred_dur.sum()` (frame count), and `input_lengths` (token count) — no-GPU, first diagnostic step. | Step 3 | `code/infer_styletts2.py`'s modified `synthesize()`; `results/duration_characterization.json` |
| REQ-2 | Localize the defect: predictor-calibration vs. alignment/frame-count plumbing bug, using the logged data (not inferred from audio length). | Steps 3-5 | `results/duration_blowup_diagnosis.md` "Localization" section |
| REQ-3 | Check whether `predictor`/`predictor_encoder` rode along at LibriTTS-scale init vs. were meaningfully fine-tuned; cross-check the training-log evidence with a no-GPU tensor-level weight comparison. | Step 4 | `results/predictor_tensor_forensics.md` |
| REQ-4 | Characterize whether the blowup is universal, text-length-dependent, or intermittent across >=10 varied (short + long) texts, logging text, token/phoneme count, `pred_dur`, frame count, output duration, and `audio_quality_check.py` metrics for each. | Step 5 | `results/duration_characterization.json`, `results/duration_blowup_diagnosis.md` |
| REQ-5 | Sweep `alpha`/`beta`/`diffusion_steps`/`embedding_scale` on a fixed short text; pre-register pass criteria before sweeping; no GPU required. | Steps 6-7 | `results/param_sweep.json`, `results/duration_blowup_diagnosis.md` "Cheap fix" section |
| REQ-6 | If the cheap fix works: re-synthesize the 3 t0014 gate texts (`lining_up_suggestions_17`, `lining_up_suggestions_10`, `putting_them_head_to_head_15`) with corrected parameters, produce fresh paired original-vs-FT audio samples, and update `infer_styletts2.py`'s defaults with documentation of why. | Step 8 (conditional) | `results/audio_samples/{original,ft}/`, updated `synthesize()` defaults |
| REQ-7 | If the cheap fix does not work: document the negative result with the same rigor (every combination tried, its outcome, and why none resolved it) and recommend targeted `predictor`/`predictor_encoder` fine-tuning (or full retrain) as a follow-up — do not attempt GPU training in this task. | Step 8 (conditional) | `results/duration_blowup_diagnosis.md` "Cheap fix did not work" section |
| REQ-8 | Extend `audio_quality_check.py` with a duration-sanity signal (synthesized duration vs. a naive words-per-second estimate; flag > ~3x a generous upper bound) and a longest-contiguous-non-silent-run signal, in the same module (one source of truth). | Step 9 | `code/audio_quality_check.py` |
| REQ-9 | Evaluate (not necessarily implement) whether an ASR-round-trip check (`faster-whisper`/`compute_wer` from `tts_eval_harness`) is worth adding as an optional third layer. | Step 10 | `results/asr_roundtrip_evaluation.md` |
| REQ-10 | Re-run the hardened gate as a three-way regression: v10 (should still fail), v11-as-shipped (should now fail on the new signals despite passing the old ones), and the corrected output from Step 8 if it exists (should pass all signals) — proving the hardened gate discriminates correctly. | Step 11 | `results/gate_regression.json`, `results/duration_blowup_diagnosis.md` |
| REQ-11 | Produce `results/duration_blowup_diagnosis.md`: root cause (predictor calibration vs. plumbing), universal-vs-text-dependent verdict, and whether the cheap fix worked. | Step 12 | `results/duration_blowup_diagnosis.md` |

Ambiguity notes:

* The task text says "at least 5-10, short and long" texts for characterization (Scope step 1) but
  also "a wider, varied sample of texts (short and longer ...)" (Key Question 3) and Key Findings
  from research recommend the pre-built `filler_prompts_100.json`/`val96_prompts.json` prompt sets.
  This plan resolves it as exactly 10 texts (5 short filler phrases + 5 longer sentences), the
  intersection of "at least 5-10" and "varied" that keeps CPU wall-clock time bounded (Time
  Estimation section) while still being definitive about universal-vs-text-dependent.
* "results/results_summary.md / results_detailed.md — outcome and recommendation" and
  "results/suggestions.json" from `task_description.md`'s Expected Outputs are **not** included as
  Step by Step items in this plan — per `arf/specifications/plan_specification.md`, these are
  orchestrator-managed outputs produced by `execute-task`'s reporting/generate-suggestions stages,
  not implementation-agent steps. `results/duration_blowup_diagnosis.md`, by contrast, is a
  specific, named diagnostic deliverable this task's own Scope demands, so it is a Step by Step item
  (Step 12).

## Approach

**Grounding in research** (`tasks/t0015_v11_duration_blowup_forensics/research/research_summary.md`
and `research/research_code.md`):

* The exact instrumentation point is already located:
  `tasks/t0014_v11_decoder_fix_retrain/code/infer_styletts2.py`'s `synthesize()` function, lines
  282-371. The duration computation is `duration = model.predictor.duration_proj(x)` ->
  `torch.sigmoid(duration).sum(axis=-1)` ->
  `pred_dur = torch.round(duration.squeeze()).clamp(min=1)` (line 344) ->
  `pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))` (line 346), filled by a
  `for i in range(pred_aln_trg.size(0))` loop (lines 347-350). Neither `pred_dur` nor the frame
  count is currently logged or returned; `synthesize()` only returns `(wav, wall_time)`.
  `config_david_v11.yml:79` sets `model_params.max_dur = 50`, bounding what a well-formed per-token
  `duration_proj` output can be — a per-token `pred_dur` value exceeding ~50, or a token count far
  larger than the ~10-word input implies (phonemizer over-segmentation), are the two concrete
  hypotheses to discriminate between predictor-calibration and plumbing-bug root causes.
* Checkpoint-load correctness is already ruled out:
  `tasks/t0014_v11_decoder_fix_retrain/results/load_log_epoch_00048.json` shows 0 missing/0
  unexpected keys on all 13 modules including `predictor`/`predictor_encoder`, so this task's
  instrumentation should focus on the forward pass in `synthesize()`, not the load path.
* Training-log evidence already partially answers the "did predictor train" question (Key Question
  2): `tasks/t0014_v11_decoder_fix_retrain/code/train_second_v11.py` lines 733-734 show
  `optimizer.step("predictor")` and `optimizer.step("predictor_encoder")` execute
  **unconditionally** every training step (unlike `style_encoder`/`decoder`, gated on
  `epoch >= joint_epoch`, and `diffusion`, gated on `epoch >= diff_epoch`) — this rules out "rode
  along completely frozen." But `predictor_encoder` is excluded from the `first_stage_path`
  checkpoint load via `ignore_modules` (line ~316-321) and instead initialized as
  `copy.deepcopy(model.style_encoder)` — a second possible scale-mismatch source. Despite receiving
  gradients, `tasks/t0014_v11_decoder_fix_retrain/data/run_v11/metrics.jsonl`'s per-epoch `dur_loss`
  **plateaus at 0.53-0.62 across all 50 epochs** (epoch 1: 0.620, epoch 50: 0.529, no clear downward
  trend after ~epoch 8) — contrast `t0009`'s reference run converging to 0.034 by epoch 6. This is
  evidence *for* a predictor-calibration failure (the loss never converged), not conclusive proof by
  itself — REQ-1's instrumentation is what confirms whether the failure is in `duration_proj`'s
  output scale itself or in `pred_aln_trg` construction from otherwise-plausible per-token values.
  This plan additionally adds a no-GPU tensor-level weight-scale comparison (Step 4) as an
  independent cross-check of the training-log evidence, following the same "confirm with evidence,
  not assertion" discipline `t0013`'s `inspect_checkpoint.py` established.
* `alpha`/`beta`/`diffusion_steps`/`embedding_scale` (currently hardcoded `0.3`/`0.7`/`5`/`1.0`,
  straight LibriTTS-demo defaults) feed the diffusion sampler that produces the style vector
  consumed by `predictor.text_encoder`/`lstm`/`duration_proj` — i.e., they are upstream of the
  duration computation, so a poorly-conditioned style vector could itself skew `pred_dur` even if
  `duration_proj`'s weights are individually reasonable. Only `diffusion_steps` is currently
  CLI-exposed; `alpha`/`beta`/`embedding_scale` require either new CLI flags or a Python driver
  script calling `synthesize()` directly.
* The gate module's own docstring documents its "add a new signal when one signal misses a failure
  mode" design history (`spectral_flatness` alone missed v10's clipping; `clip_fraction` was added).
  This task repeats that same pattern for the duration/silence-gap failure mode, in the same module,
  matching the project's S-0013-04-derived "one source of truth" convention.
* `tts_eval_harness` (t0008 library) already has `compute_duration_ratio` (reference-paired,
  `ratio > 5.0` flag) and `compute_wer` (faster-whisper + JiWER). Neither is a drop-in for the new
  gate signal (the new signal must be text-only, no paired reference recording exists for arbitrary
  synthesized text), but `compute_wer` is directly reusable for the optional ASR-round-trip
  evaluation (REQ-9).

**Alternatives considered**:

* *Skip the no-GPU tensor-forensics cross-check (Step 4) and rely solely on the training-log
  evidence already in hand.* Rejected: the training-log evidence (flat `dur_loss`) is consistent
  with a predictor-calibration failure but does not, by itself, distinguish it from a plumbing bug
  that would produce a flat `dur_loss` for an unrelated reason (e.g., if the loss computation itself
  references the buggy frame count). REQ-1's forward-pass instrumentation and REQ-3's tensor check
  are cheap (no GPU, minutes of CPU time) and remove this ambiguity directly rather than by
  inference.
* *Attempt targeted `predictor`/`predictor_encoder` fine-tuning inside this task if the cheap fix
  fails.* Rejected per `task_description.md`'s own Scope step 5 explicitly: "do not attempt a GPU
  training run inside this task" — this is scoped as forensics + cheap fixes + gate-hardening only;
  a retrain is a recommended follow-up task, not part of this plan.
* *Build the ASR-round-trip check as a mandatory third gate signal now, not just an evaluation.*
  Rejected: `task_description.md` explicitly frames this as "evaluate (don't necessarily implement,
  if it's a bigger lift than this task's forensics scope)". `compute_wer` requires downloading a
  `faster-whisper` model and running ASR inference per clip — a real cost/time increment beyond the
  two cheap signals REQ-8 already mandates. This plan evaluates it (Step 10) and implements it only
  if Step 10 finds it is a small, low-risk addition; if not, it is deferred as a documented
  recommendation rather than force-fit into this task's scope.
* *Use `train_list_v5_normalized_clean.txt`'s pipe-delimited manifest directly for varied-text
  characterization, per the task description's literal file suggestion.* Rejected in favor of
  `filler_prompts_100.json`/`val96_prompts.json`: the manifest's second column is already
  IPA-phonemized text (`ˈaksɛsɪŋ ðə bɹˈAn kˈɒməːs pˈAʤ…`), not raw English text, and
  `infer_styletts2.py`'s `synthesize()` expects raw text (it phonemizes internally via
  `phonemizer.backend.EspeakBackend`). The prompt-set JSONs store raw `"text"` fields directly, are
  already varied in length (`filler_prompts_100.json` entries are 2-5 words; `val96_prompts.json`
  entries range from short phrases to full sentences), and match this task's own research
  recommendation.

**Task types**: `task.json` declares `["tts-benchmark-run", "code-reproduction"]`.
`meta/task_types/code-reproduction/instruction.md`'s Planning Guidelines shaped this plan's
structure directly — instrument and re-run the existing recipe faithfully before changing anything,
mark the instrumentation/characterization step `[CRITICAL]`, and do not substitute a different
approach if a critical step is blocked. `meta/task_types/tts-benchmark-run/instruction.md`'s full
protocol (N>=100 measured requests, p50/p95/p99 latency, warmup discipline) does **not** apply at
that scale here: this is a forensics task running 10-20 one-off CPU synthesis calls for root-cause
diagnosis, not a production latency benchmark. This plan applies that type's *metrics discipline*
(report `rtf` per synthesis condition, never claim TTFB it did not measure) without running its full
protocol — consistent with `t0013`'s and `t0014`'s own precedent of not measuring `ttfb_ms` for this
offline, non-streaming harness. If a human reviewer judges `tts-benchmark-run` too heavy a label for
this diagnostic task, `answer-question` or `data-analysis` would be closer fits for the
characterization/localization work; this plan does not request a `task.json` change since the two
declared types already permit the approach taken here without contradiction.

## Cost Estimation

**Total: $0.00.**

* No GPU/remote compute: all work runs on the local CPU-only `.venv-styletts2` environment (torch
  2.5.1 CPU build), matching `t0013`'s and `t0014`'s own precedent of building this environment
  twice already for the same reason (StyleTTS2-native inference does not require a GPU; CPU `rtf` is
  3.2-5.5x, acceptable for one-off diagnostic synthesis calls, not a production benchmark).
* No paid API calls: `faster-whisper` (used only if Step 10 concludes the ASR-round-trip layer is
  worth evaluating) runs a local, already-downloaded-once-per-machine open-weights model (`base.en`,
  per `tts_eval_harness`'s `constants.WHISPER_MODEL_SIZE`) — no API key, no per-call cost.
* Network egress: `git clone --depth 1` of `github.com/semidark/kikiri-tts` (+ submodules) and `pip`
  package downloads for the venv are free, standard package-index/GitHub bandwidth — $0, per
  `t0013`'s plan's identical cost-estimation precedent for the same clone.
* `dvc pull` for `tasks/t0014_v11_decoder_fix_retrain/data/run_v11/epoch_00048.pth` (the
  `kokoro-v11-best` raw checkpoint, ~2.09 GB) and
  `tasks/t0008_tts_eval_harness_baselines/data/11labs_david/` (reference corpus) from Azure Blob
  Storage (`azure://ml-dvc-datasets/datasets/rail-arf-tts`) — storage egress within an existing
  project account, not a new billable resource for this task.
* This matches the check-deps assessment already recorded for this task: `tts-benchmark-run` and
  `code-reproduction` task types were evaluated with `has_external_costs: false` for this specific
  task's scope. Compared against `project/budget.json`'s `per_task_default_limit` of $100.00 and the
  project's `total_budget` of $5000.00 (currently 7.95% spent, $397.73), a $0.00 estimate leaves the
  full per-task limit unused and poses no risk to project budget thresholds (`warn_at_percent: 80`,
  `stop_at_percent: 100`).

## Step by Step

### Milestone A: Environment setup (no GPU)

1. **[CRITICAL] Build the CPU StyleTTS2-native inference environment.** Create `code/kikiri-tts/`
   and clone:
   `uv run python -m arf.scripts.utils.run_with_logs --task-id t0015_v11_duration_blowup_forensics -- git clone --depth 1 https://github.com/semidark/kikiri-tts.git tasks/t0015_v11_duration_blowup_forensics/code/kikiri-tts`,
   then
   `git -C tasks/t0015_v11_duration_blowup_forensics/code/kikiri-tts submodule update --init --recursive`.
   Create `code/.gitignore` containing `kikiri-tts/` and `.venv-styletts2/` (neither is committed,
   mirroring `t0013`'s and `t0014`'s precedent). Build the venv:
   `uv venv tasks/t0015_v11_duration_blowup_forensics/code/.venv-styletts2 --python 3.10`, then
   `uv pip install --python tasks/t0015_v11_duration_blowup_forensics/code/.venv-styletts2/bin/python torch==2.5.1 torchaudio==2.5.1 soundfile munch pydub pyyaml librosa nltk matplotlib accelerate transformers einops einops-exts tqdm typing-extensions phonemizer "resemblyzer>=0.1.4"`
   and
   `uv pip install --python tasks/t0015_v11_duration_blowup_forensics/code/.venv-styletts2/bin/python git+https://github.com/resemble-ai/monotonic_align.git`.
   Install the phonemizer backend: `sudo apt-get install -y espeak-ng` if not already present (check
   `espeak-ng --version` first; both t0013 and t0014 already installed this system-wide, so it may
   already be present). Verify:
   `uv run --python tasks/t0015_v11_duration_blowup_forensics/code/.venv-styletts2/bin/python -c "import torch; print(torch.__version__)"`
   prints `2.5.1`. This is `[CRITICAL]` because every downstream diagnostic step depends on a
   working inference harness — this task's core identity is running the harness with new
   instrumentation, not inferring the bug from documentation alone.

2. **Fetch DVC-tracked data.** Run
   `dvc pull tasks/t0014_v11_decoder_fix_retrain/data/run_v11/epoch_00048.pth` to materialize the
   raw 13-module `kokoro-v11-best` checkpoint (2,086,802,736 bytes, sha256 `1fb329b5...23c4e` per
   `tasks/t0014_v11_decoder_fix_retrain/data/run_v11/checkpoints.json` — verify the pulled file's
   size matches after pulling). Run
   `dvc pull tasks/t0008_tts_eval_harness_baselines/data/11labs_david` to materialize the ElevenLabs
   David reference corpus (needed for `build_reference_concat.py` and, if Step 8/11 need
   `speaker_sim`, `score_speaker_sim.py`'s centroid). Also confirm
   `tasks/t0013_v10_synthesis_quality_forensics/results/audio_samples/` is present (needed as the
   v10 fixture for Step 11's three-way regression) — run
   `dvc pull tasks/t0013_v10_synthesis_quality_forensics/results/audio_samples` if the directory is
   empty or missing files. Expected output:
   `ls -la tasks/t0014_v11_decoder_fix_retrain/data/run_v11/epoch_00048.pth` shows a ~2.1 GB file
   (not a small `.dvc` pointer-sized file);
   `find tasks/t0008_tts_eval_harness_baselines/data/11labs_david -name '*.wav' | wc -l` returns >=
   1000 (per `score_speaker_sim.py`'s own `MIN_CORPUS_SIZE = 1000` assertion).

3. **Copy the reusable code files into this task.** Copy (verbatim, then adapt per later steps):
   `tasks/t0014_v11_decoder_fix_retrain/code/infer_styletts2.py` ->
   `tasks/t0015_v11_duration_blowup_forensics/code/infer_styletts2.py` (t0014's version, not t0013's
   — it is checkpoint/config-current per research finding);
   `tasks/t0014_v11_decoder_fix_retrain/code/audio_quality_check.py` ->
   `tasks/t0015_v11_duration_blowup_forensics/code/audio_quality_check.py`;
   `tasks/t0014_v11_decoder_fix_retrain/code/build_reference_concat.py` ->
   `tasks/t0015_v11_duration_blowup_forensics/code/build_reference_concat.py` (copy unchanged — same
   3 reference clips, same 5.48s concatenation, for direct comparability with t0013/t0014's
   numbers); `tasks/t0014_v11_decoder_fix_retrain/code/config_david_v11.yml` ->
   `tasks/t0015_v11_duration_blowup_forensics/code/config_david_v11.yml`;
   `tasks/t0013_v10_synthesis_quality_forensics/code/inspect_checkpoint.py` ->
   `tasks/t0015_v11_duration_blowup_forensics/code/inspect_checkpoint.py`;
   `tasks/t0013_v10_synthesis_quality_forensics/code/random_decoder_probe.py` ->
   `tasks/t0015_v11_duration_blowup_forensics/code/random_decoder_probe.py`;
   `tasks/t0014_v11_decoder_fix_retrain/code/score_speaker_sim.py` ->
   `tasks/t0015_v11_duration_blowup_forensics/code/score_speaker_sim.py`. Create
   `tasks/t0015_v11_duration_blowup_forensics/code/paths.py` with path constants for this task
   (`TASK_ROOT`, `REPO_ROOT`, `V11_CHECKPOINT` pointing at
   `tasks/t0014_v11_decoder_fix_retrain/data/run_v11/epoch_00048.pth`, `V11_CONFIG` pointing at this
   task's copied `config_david_v11.yml`, `ELEVENLABS_DAVID_DIR` pointing at
   `tasks/t0008_tts_eval_harness_baselines/data/11labs_david`, `RESULTS_DIR`, `RESULTS_AUDIO_DIR`,
   `T0013_V10_AUDIO_DIR` pointing at
   `tasks/t0013_v10_synthesis_quality_forensics/results/audio_samples`, `T0014_V11_AUDIO_DIR`
   pointing at `tasks/t0014_v11_decoder_fix_retrain/results/audio_samples`), following
   `arf/styleguide/python_styleguide.md`'s centralized-paths convention. Build the reference concat:
   `.venv-styletts2/bin/python code/build_reference_concat.py` (or equivalent invocation matching
   the copied script's CLI) writing to
   `tasks/t0015_v11_duration_blowup_forensics/data/reference_concat.wav`. Expected output: all 7
   files present in `code/`; `code/reference_concat.wav` (or `data/reference_concat.wav`) exists and
   is ~5.48s. No REQ satisfied directly by this step (setup only), but it is a prerequisite for all
   of Steps 4-11.

### Milestone B: Instrument and reproduce (REQ-1, REQ-2, REQ-3, REQ-4)

4. **[CRITICAL] Instrument `synthesize()` to log `pred_dur` and frame count.** Edit
   `code/infer_styletts2.py`'s `synthesize()` function (copied at lines ~282-371). After line
   `pred_dur = torch.round(duration.squeeze()).clamp(min=1)` (original line 344), add: capture
   `pred_dur_list = pred_dur.detach().cpu().tolist()` (list of per-token floats),
   `pred_dur_sum = int(pred_dur.sum().item())` (total frame count),
   `input_token_count = int(input_lengths.item())`. Change `synthesize()`'s return type from
   `tuple[np.ndarray, float]` to a `@dataclass(frozen=True, slots=True)` `SynthesisResult` (or an
   equivalent named tuple) with fields: `wav: np.ndarray`, `wall_time_seconds: float`,
   `pred_dur: list[float]`, `pred_dur_sum: int`, `input_token_count: int`, `phoneme_string: str`
   (the `ps` variable already computed at line 313, `" ".join(ps)`). Update `main()`'s call site
   accordingly and add these fields to the existing `<output>.timing.json` write. Verify with one
   manual run on a trivial text:
   `.venv-styletts2/bin/python code/infer_styletts2.py --checkpoint-path tasks/t0014_v11_decoder_fix_retrain/data/run_v11/epoch_00048.pth --config-path code/config_david_v11.yml --text "This is a test." --reference-audio data/reference_concat.wav --output-wav results/audio_samples/ft/smoke_test.wav`
   and confirm the printed/`.timing.json` output includes non-null `pred_dur`, `pred_dur_sum`, and
   `input_token_count` values. This is `[CRITICAL]`: REQ-1's entire premise is that duration must be
   measured directly, not inferred from audio length — without this instrumentation, no later step
   in this plan can distinguish predictor-calibration from plumbing-bug causes. Satisfies REQ-1.

5. **Run the varied-text characterization batch (10 texts).** Build `code/run_characterization.py`:
   a driver script that (a) loads 5 short texts from
   `tasks/t0008_tts_eval_harness_baselines/data/filler_prompts_100.json` (first 5 entries by list
   order, for determinism) and 5 longer texts from
   `tasks/t0008_tts_eval_harness_baselines/data/val96_prompts.json` (the 5 longest `"text"` values
   by character length, for determinism and to maximize "long text" coverage), extracting only the
   `"text"` field (ignore each entry's own `ref_wav`/`ref_duration_s` — this task uses the single
   fixed `data/reference_concat.wav` style reference built in Step 3 for every text, so duration
   differences are attributable to input text only, not to a confounding style-reference change);
   (b) for each of the 10 texts, calls the instrumented `synthesize()` from Step 4 with default
   parameters (`alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1.0`) against the v11
   checkpoint; (c) writes each output WAV to
   `results/audio_samples/characterization/v11_<index>_<slug>.wav`; (d) for each output, runs
   `check_audio_quality()` from the copied `audio_quality_check.py` (pre-hardening, original
   3-signal version at this point — hardening happens in Step 9); (e) computes a naive
   `expected_duration_s = word_count / 2.5` (word_count via `len(text.split())`, 2.5 words/second as
   the naive average-conversational-rate denominator) purely as a diagnostic ratio for this step's
   log (the pre-registered gate threshold itself is defined in Step 9); (f) writes one JSON record
   per text to `results/duration_characterization.json`:
   `{text, word_count, input_token_count, phoneme_string, pred_dur, pred_dur_sum, output_duration_s, wall_time_s, rtf, expected_duration_s, duration_ratio (output_duration_s / expected_duration_s), rms, peak, silence_fraction, spectral_flatness, clip_fraction, is_likely_noise}`.
   **Validation gate**: run the driver with `--limit 2` first (the first short text and the first
   long text). Baseline: the control checkpoint's duration for a comparable ~10-word sentence was
   4.92s (`tasks/t0014_v11_decoder_fix_retrain/results/v11_gate_verdict.md`'s comparison table) — if
   either of the first 2 outputs crashes, produces a zero-length WAV, or produces NaN/Inf in
   `pred_dur`, halt and debug (read the raw `pred_dur` list and the phoneme string for that specific
   input) before running the remaining 8. After the 2-item validation run, read both individual JSON
   records and confirm `pred_dur` is a non-empty list of finite positive numbers,
   `input_token_count` roughly matches the phoneme string's token count, and `output_duration_s` is
   a positive float — then run the full 10-item batch:
   `.venv-styletts2/bin/python code/run_characterization.py`. Expected output:
   `results/duration_characterization.json` has exactly 10 records; at least the same ~10-word-class
   text used in `t0014`'s gate (if present in the sample) reproduces a duration in the 60-90s range,
   consistent with the previously observed 73.95s. Satisfies REQ-4 (universal-vs-text-dependent
   characterization) and contributes raw data to REQ-2 (localization, Step 6 analyzes this data).

6. **Analyze the characterization data for localization (predictor-calibration vs. plumbing).**
   Write `code/analyze_localization.py` (or inline analysis captured in the diagnosis doc) that
   reads `results/duration_characterization.json` and, for each record, computes: (a) mean per-token
   `pred_dur` value (`pred_dur_sum / input_token_count`); (b) whether any single per-token
   `pred_dur` value exceeds `model_params.max_dur = 50` (from `config_david_v11.yml:79` — a value at
   or near 50 for many tokens indicates the sigmoid-sum is saturating, i.e., a calibration failure,
   since 50 is the per-token ceiling a well-formed `duration_proj` output can reach); (c) whether
   `input_token_count` is wildly disproportionate to `word_count` (phonemizer over-segmentation, a
   plumbing-adjacent hypothesis) — flag if `input_token_count / word_count > 8` (a generous
   multiple; English IPA phonemization typically produces 3-6 phoneme-tokens per word). Write the
   findings to a dedicated results section (folded into Step 12's `duration_blowup_diagnosis.md`,
   not a separate file). Expected finding, to be confirmed empirically rather than assumed:
   per-token `pred_dur` values clustered near or at the `max_dur=50` ceiling across most/all tokens
   (consistent with the flat 0.53-0.62 `dur_loss` plateau found in research) would confirm
   predictor-calibration as the root cause; a normal per-token `pred_dur` distribution with an
   anomalously large `input_token_count` would instead point to a plumbing/tokenization bug.
   Satisfies REQ-2.

7. **[CRITICAL if Step 6 is inconclusive] Cross-check with no-GPU tensor forensics on
   `predictor`/`predictor_encoder`.** Adapt the copied `inspect_checkpoint.py` (originally targeted
   `decoder`'s `hifigan` vs. `istftnet` key-presence and weight-norm/`isfinite` checks) to instead
   report per-module weight-norm summary statistics (mean absolute value, max absolute value,
   `torch.isfinite` check) for `predictor` and `predictor_encoder`'s parameter tensors, loaded via
   raw `torch.load` on `tasks/t0014_v11_decoder_fix_retrain/data/run_v11/epoch_00048.pth` (no
   StyleTTS2 import, no inference — this is the same cheap, no-GPU pattern `t0013` used for the
   decoder). Compare against the same statistics computed for the LibriTTS control checkpoint's
   `predictor`/`predictor_encoder` (`tasks/t0014_v11_decoder_fix_retrain/code/paths.py`'s
   `LIBRITTS_CONTROL_CKPT`, already local per `t0013`'s clone at
   `tasks/t0013_v10_synthesis_quality_forensics/code/kikiri-tts/StyleTTS2/Models/LibriTTS/epochs_2nd_00020.pth`
   if present, else the freshly-cloned `code/kikiri-tts/` equivalent from Step 1). Write
   `results/predictor_tensor_forensics.md` documenting per-module weight-norm deltas between v11 and
   the LibriTTS control for `predictor.duration_proj` specifically (the exact submodule producing
   `pred_dur`). A large weight-norm shift with no `isfinite` failures corroborates a calibration
   failure (the weights changed during training but converged to a bad region); near-zero shift
   would suggest `predictor` effectively didn't move despite receiving gradients (contradicting the
   `optimizer.step()` evidence and warranting a deeper look at gradient magnitudes, which is out of
   this plan's no-GPU scope and would become a follow-up recommendation). Satisfies REQ-3.

### Milestone C: Cheap fix attempt (REQ-5, REQ-6, REQ-7)

8. **Sweep `alpha`/`beta`/`diffusion_steps`/`embedding_scale`; pre-register pass criteria before
   running.** Before running anything, write the pre-registered pass criteria into
   `results/param_sweep.json`'s top-level `"pass_criteria"` field (and restate in
   `duration_blowup_diagnosis.md`): a parameter combination counts as "the cheap fix worked" **only
   if ALL of** (a) `check_audio_quality()`'s (hardened, per Step 9 — run Step 9 before this step if
   not already done, or apply the same duration-sanity/non-silent-run logic ad hoc here and
   formalize in Step 9) `is_likely_noise == False`; (b) `duration_ratio` (output duration / naive
   `word_count / 1.5`-based upper bound, see Step 9's exact formula) `<= 3.0`; (c)
   `longest_nonsilent_run_s <= 12.0`. No partial improvement (e.g., duration cut in half but still
   10x too long) may be reported as "fixed" — per the Rejection Criteria section below and the
   research-documented pattern of this project chain's prior "no other metric may be substituted"
   discipline. Expose `alpha`, `beta`, `embedding_scale` as new CLI flags on
   `code/infer_styletts2.py` alongside the existing `--diffusion-steps` (or write a small
   `code/param_sweep_driver.py` calling `synthesize()` directly with different keyword arguments —
   simpler than shelling out per combination since `synthesize()` is already a Python function). Use
   the fixed test text "This is a test of the Style T T S two inference harness." (identical to
   `t0013`'s/`t0014`'s gate text, for direct comparability) and the same `data/reference_concat.wav`
   reference. Grid: `alpha` in `{0.1, 0.3, 0.5}` x `beta` in `{0.3, 0.7, 0.9}` at fixed
   `diffusion_steps=5, embedding_scale=1.0` (9 combinations); then, using whichever `(alpha, beta)`
   pair from that grid produced the shortest `output_duration_s` (even if not yet passing), test
   `embedding_scale` in `{0.5, 2.0}` and `diffusion_steps` in `{10}` (3 more combinations) — 12
   total runs. **Validation gate**: run the first 2 grid combinations first; baseline is the
   default-parameter duration already measured in Step 5 (record it for the identical text if
   present in the characterization batch, else run it once more with defaults as the explicit
   baseline). If both of the first 2 combinations still produce >60s output (no improvement at all
   over baseline), read the `pred_dur` values for those 2 runs individually before continuing — a
   completely flat response to parameter changes is itself a diagnostic finding (evidence the bug is
   not upstream-style-vector-sensitive) worth confirming before spending the remaining 10 runs'
   wall-clock time. Then run all 12 combinations. Write every combination's full result
   (`alpha, beta, diffusion_steps, embedding_scale, output_duration_s, pred_dur_sum, is_likely_noise, longest_nonsilent_run_s, duration_ratio, pass: bool`)
   to `results/param_sweep.json`. Expected output: a JSON array of exactly 12 records (13 if the
   explicit baseline re-run was needed), each with all fields populated (no nulls from crashed runs
   — if any combination crashes, record the error message in that record's `error` field rather than
   omitting the record). Satisfies REQ-5.

9. **Branch on the sweep outcome (REQ-6 or REQ-7).**
   * **If any combination in `results/param_sweep.json` has `pass: true`** (REQ-6 path): select the
     combination with the lowest `duration_ratio` among passing combinations as the new default.
     Update `code/infer_styletts2.py`'s `synthesize()` default keyword arguments (`alpha=`, `beta=`,
     `embedding_scale=`, and `--diffusion-steps`'s CLI default if changed) to this combination, with
     an inline comment citing `results/param_sweep.json` and explaining why (e.g., "alpha=0.1,
     beta=0.3 selected: lowest duration_ratio among passing combinations, see
     results/param_sweep.json"). Re-synthesize the exact 3 gate texts from `t0014`'s
     `v11_gate_verdict.md` (`lining_up_suggestions_17`, `lining_up_suggestions_10`,
     `putting_them_head_to_head_15` — read their exact text content from
     `tasks/t0008_tts_eval_harness_baselines/data/filler_prompts_100.json` or
     `tasks/t0014_v11_decoder_fix_retrain/code/build_reference_concat.py`'s source clip list) with
     the corrected parameters, writing to `results/audio_samples/ft/v11_corrected.wav`. Also copy
     (not re-synthesize, per `t0014`'s own precedent for the control) the existing control sample:
     `tasks/t0013_v10_synthesis_quality_forensics/results/audio_samples/control_epochs_2nd_00020.wav`
     -> `results/audio_samples/original/librispeech_control.wav`. Run `score_speaker_sim.py` (copied
     in Step 3) against `v11_corrected.wav`, writing `results/speaker_sim_scores.json`. Satisfies
     REQ-6.
   * **If no combination passes** (REQ-7 path): do not modify `synthesize()`'s defaults. Write a
     dedicated "Cheap fix did not work" section into `results/duration_blowup_diagnosis.md` (Step
     12\) listing every one of the 12+ combinations tried, its `output_duration_s`/`duration_ratio`,
     and why each failed the pre-registered criteria (e.g., "all 12 combinations produced
     `duration_ratio` between 8.2 and 14.6 — parameter changes reduced but did not eliminate the
     blowup, consistent with a predictor-weight-scale problem that inference-time sampling
     parameters cannot correct"). Recommend targeted `predictor`/`predictor_encoder` fine-tuning
     (holding the working decoder fixed) as a follow-up task, citing the specific evidence from
     Steps 6-7 that narrows the recommendation (predictor-calibration vs. plumbing). Do not produce
     `results/audio_samples/ft/v11_corrected.wav` in this branch — do not claim a fix that does not
     exist. Satisfies REQ-7.

### Milestone D: Harden the gate (REQ-8, REQ-9, REQ-10)

10. **Add duration-sanity and longest-contiguous-non-silent-run signals to
    `audio_quality_check.py`.** Edit `code/audio_quality_check.py` (copied in Step 3) in place —
    same module, per the "one source of truth" convention already documented in its own docstring.
    Add two new named constants alongside the existing three thresholds:
    `NAIVE_MIN_WORDS_PER_SECOND = 1.5` (a deliberately slow floor — 90 words/minute — chosen to
    minimize false positives on genuinely slow, deliberate filler narration; document this reasoning
    in a code comment) and `DURATION_SANITY_MULTIPLIER = 3.0` (per `task_description.md`'s own
    suggested "~3x a reasonable upper bound"), plus `LONGEST_NONSILENT_RUN_THRESHOLD_S = 12.0` (a
    flat, generous ceiling — longer than any single breath group/phrase in normal speech; document
    in a comment that this is intentionally generous to avoid false positives, with a note that a
    text-length-scaled threshold is a possible future refinement). Add a new function
    `estimate_naive_duration_bound_s(text: str) -> float` returning
    `len(text.split()) / NAIVE_MIN_WORDS_PER_SECOND`. Extend `check_audio_quality()`'s signature to
    accept an optional `text: str | None = None` parameter; when provided, compute
    `duration_sanity_bound_s = estimate_naive_duration_bound_s(text) * DURATION_SANITY_MULTIPLIER`
    and `duration_sanity_pass = (actual_duration_s <= duration_sanity_bound_s)` where
    `actual_duration_s` is derived from `len(y) / sr` (already computed via `librosa.load` at the
    top of the function). Modify the existing 20ms-frame silence loop (lines 79-88) to also track a
    running counter of consecutive non-silent frames and its running maximum, converting the final
    max frame count to seconds
    (`longest_nonsilent_run_s = max_consecutive_nonsilent_frames * frame_len / sr`) — a small, local
    change to the existing loop, not a new pass over the audio. Add
    `duration_sanity_pass: bool | None` (`None` when `text` was not provided) and
    `longest_nonsilent_run_s: float` fields to the `AudioQualityResult` dataclass. Update
    `is_likely_noise`'s computation to also flag True when
    `longest_nonsilent_run_s > LONGEST_NONSILENT_RUN_THRESHOLD_S` (the duration-sanity signal
    requires the caller to pass `text`, so it is reported as a separate field rather than folded
    into `is_likely_noise`, to preserve backward compatibility for callers that don't have the
    source text at hand — document this design choice in the module docstring). Expected output:
    `AudioQualityResult` now has 8 fields instead of 6; running
    `.venv-styletts2/bin/python code/audio_quality_check.py results/audio_samples/ft/v11_best.wav`
    (or the equivalent v11 fixture) with a `--text` CLI flag prints a result with
    `longest_nonsilent_run_s` far exceeding 12.0 and (if `--text` given)
    `duration_sanity_pass=False`. Satisfies REQ-8.

11. **Evaluate the ASR-round-trip check as an optional third layer.** Import `compute_wer` from
    `tasks.t0008_tts_eval_harness_baselines.code.scoring` (the `tts_eval_harness` library) and run
    it once against 2-3 of the Step 5 characterization outputs (a mix of a normal- duration one, if
    any, and the blown-up v11 default-parameter ones) to observe its behavior in practice: does it
    complete without new dependency friction (`faster-whisper`'s `base.en` model download), does its
    `DURATION_RATIO_LOW=0.5`/`DURATION_RATIO_HIGH=2.0` gating skip the blown-up clips entirely
    (likely, since a 74s clip against a ~4s expected duration is far outside that gate), and how
    long ASR transcription takes per clip on CPU. Write findings to
    `results/asr_roundtrip_evaluation.md`: whether it is a small, low-risk addition worth wiring
    into `audio_quality_check.py` as a permanent third layer, or a heavier lift (new dependency,
    CPU-only ASR inference latency, and the duration-ratio pre-gating already means it would rarely
    even run on the exact class of clip this task cares about) better left as a documented
    recommendation for a future task rather than implemented now. If the evaluation concludes it is
    trivially easy to wire in as a fully optional (default-off) parameter to
    `check_audio_quality()`, add it in this step; if not, do not implement it — state the
    recommendation clearly instead of half-implementing it. Satisfies REQ-9.

12. **[CRITICAL] Run the hardened gate as a three-way regression (v10, v11-as-shipped,
    corrected-if-any).** Run the hardened `check_audio_quality()` from Step 10 (with `text=` passed
    for the duration-sanity signal) against exactly three fixtures: (a)
    `tasks/t0013_v10_synthesis_quality_forensics/results/audio_samples/v10_epoch16_primary.wav` (the
    confirmed-broken v10 checkpoint's output, text: reuse the known gate text "This is a test of the
    Style T T S two inference harness." since that is what produced this fixture) — expect
    `is_likely_noise=True` (unchanged from before, via the original clip/flatness signals — this
    fixture was never a duration problem); (b)
    `tasks/t0014_v11_decoder_fix_retrain/results/audio_samples/ft/v11_best.wav` (the exact 73.95s
    clip this whole task investigates, same gate text) — expect `is_likely_noise=False` under the
    *original* 3-signal logic (as t0014 already found) but `duration_sanity_pass=False` and
    `longest_nonsilent_run_s > 12.0` under the *new* signals, i.e., the hardened gate's overall
    pass/fail (now defined as
    `is_likely_noise or not duration_sanity_pass or longest_nonsilent_run_s > LONGEST_NONSILENT_RUN_THRESHOLD_S`)
    flips to **fail** for this fixture — this is the specific blind-spot closure this task exists to
    prove; (c) `results/audio_samples/ft/v11_corrected.wav` from Step 9 **only if it exists** (the
    REQ-6 path was taken) — expect it to pass all signals; if the REQ-7 path was taken instead (no
    fix), skip fixture (c) and note explicitly in the results that no corrected fixture exists to
    test. Write all results (each fixture's full `AudioQualityResult`, plus a
    `hardened_gate_pass: bool` derived field) to `results/gate_regression.json`. This is
    `[CRITICAL]`: proving the hardened gate actually discriminates the exact failure mode it was
    built for (not just adding fields that are never exercised) is the task's other core identity
    alongside root-causing the blowup — without this regression proof, "hardened the gate" is an
    unverified claim. Expected output: `results/gate_regression.json` shows fixture (a)
    `is_likely_noise=True`, fixture (b) `is_likely_noise=False` AND `hardened_gate_pass=False` (the
    discriminating proof), and fixture (c) (if present) `hardened_gate_pass=True`. Satisfies REQ-10.

### Milestone E: Metrics and diagnosis document

13. **Compute and record `rtf` and `speaker_sim` metrics.** For every synthesis run that produced a
    kept WAV file (Step 5's 10 characterization clips, Step 8's sweep clips, and Step 9's
    `v11_corrected.wav` if produced), `rtf` is already computed by `synthesize()`'s
    `wall_time_seconds / output_duration_s` (already logged per-record in
    `results/duration_characterization.json` and `results/param_sweep.json`). Compute `speaker_sim`
    (GE2E cosine vs. the `tasks/t0008_tts_eval_harness_baselines/data/11labs_david/` centroid, via
    `score_speaker_sim.py`, copied in Step 3) for: the v11-as-shipped fixture (`v11_best.wav`,
    reused from `t0014`, already has a recorded value of 0.444 in
    `tasks/t0014_v11_decoder_fix_retrain/results/metrics.json` — recompute here for a self-contained
    record rather than only citing it), the v10 fixture (already 0.351/0.311 per `t0013`'s
    `results/speaker_sim_scores.json` — recompute or cite explicitly), and `v11_corrected.wav` if it
    exists. Do **not** measure `ttfb_ms`: this task, like `t0013` and `t0014` before it, runs an
    offline batch harness with no streaming HTTP endpoint in scope — state this omission explicitly
    rather than writing a placeholder value. Write `results/metrics.json` using the **explicit
    multi-variant format** (per `arf/specifications/metrics_specification.md` and
    `arf/specifications/task_results_specification.md` — this task compares multiple conditions, so
    the explicit format is required, not the legacy flat format), with one variant per synthesis
    condition measured: at minimum `v11-as-shipped` (dimensions: `checkpoint: epoch_00048.pth`,
    `params: default`, `role: diagnosed`), `v10-primary` (dimensions:
    `checkpoint: epoch_2nd_00016.pth`, `role: known_broken_control`), and, if Step 9 took the REQ-6
    path, `v11-corrected` (dimensions: `checkpoint: epoch_00048.pth`,
    `params: <the selected alpha/beta/embedding_scale/diffusion_steps>`, `role: cheap_fix`). Each
    variant's `metrics` object includes `rtf` and `speaker_sim` (only registered metric keys from
    `tasks/t0015_v11_duration_blowup_forensics/ctx/metrics.json`: `rtf`, `speaker_sim`, `ttfb_ms` —
    `ttfb_ms` is correctly omitted per the reasoning above, not encoded as a placeholder). Expected
    output: `results/metrics.json` validates against the explicit-variant schema with 2-3 variants,
    each containing non-null `rtf` and `speaker_sim` float values.

14. **[CRITICAL] Write `results/duration_blowup_diagnosis.md`.** Synthesize all prior steps'
    findings into the single canonical diagnosis document `task_description.md`'s Expected Outputs
    names explicitly. Required sections: (1) **Root cause** — predictor-calibration vs. plumbing-bug
    verdict, citing the exact `pred_dur`/frame-count numbers from Steps 5-7 (not just a restatement
    of the research's prior hypothesis — the actual measured values from this task's own
    instrumented runs); (2) **Universal or text-dependent** — summary table of all 10
    characterization texts' `duration_ratio` values from Step 5, stating whether every text blew up,
    only long/short ones did, or it was intermittent; (3) **Cheap fix outcome** — REQ-6 or REQ-7
    result, with the full sweep table from Step 8 either inlined or referenced by exact path; (4)
    **Gate hardening** — the two new signals added in Step 10, the ASR-round-trip evaluation verdict
    from Step 11, and the three-way regression proof from Step 12 (`results/gate_regression.json`
    summarized inline); (5) **Recommendation** — if REQ-7's path was taken, the specific recommended
    follow-up (targeted `predictor`/`predictor_encoder` fine-tuning vs. full retrain, with reasoning
    drawn from Steps 6-7's localization evidence). This is `[CRITICAL]`: it is the single named
    deliverable that makes the entire investigation legible to a human reader without re-deriving it
    from 6+ raw JSON/MD files — without it, the task's forensics work exists but is not usable.
    Expected output: `results/duration_blowup_diagnosis.md` exists, is >= 400 words, and every
    numeric claim in it traces to a specific file produced by Steps 5-12 (cite exact paths, e.g.
    "see `results/duration_characterization.json` record index 3"). Satisfies REQ-11.

## Remote Machines

None required. All work in this plan runs on CPU: the StyleTTS2-native inference harness
(`.venv-styletts2`, torch 2.5.1 CPU build) already validated twice by `t0013` and `t0014` for
one-off/small-batch synthesis calls, the no-GPU tensor-forensics scripts (`inspect_checkpoint.py`,
`random_decoder_probe.py` adaptations), and the gate-hardening code changes. `task_description.md`'s
own Key Question 4 and Scope steps 2-3 explicitly frame the localization and cheap-fix-sweep work as
"no GPU required," and Scope step 5 explicitly scopes any GPU-requiring retrain out of this task as
a follow-up recommendation only.

## Assets Needed

* `kokoro-v11-best` raw checkpoint:
  `tasks/t0014_v11_decoder_fix_retrain/data/run_v11/epoch_00048.pth` (DVC-tracked, ~2.09 GB) — from
  dependency `t0014_v11_decoder_fix_retrain`, fetched via `dvc pull` in Step 2.
* `epoch_2nd_00016.pth` / `epoch_2nd_00014.pth` (v10 checkpoints) — not re-loaded directly; this
  task reuses `t0013`'s already-synthesized `results/audio_samples/v10_epoch16_primary.wav` fixture
  for the three-way regression (Step 12), not the raw checkpoint.
* `infer_styletts2.py`, `audio_quality_check.py`, `build_reference_concat.py`,
  `score_speaker_sim.py`, `config_david_v11.yml` — code files from `t0014_v11_decoder_fix_retrain`'s
  `code/`, copied in Step 3.
* `inspect_checkpoint.py`, `random_decoder_probe.py` — code files from
  `t0013_v10_synthesis_quality_forensics`'s `code/`, copied in Step 3.
* `filler_prompts_100.json`, `val96_prompts.json` — varied-text prompt data from
  `t0008_tts_eval_harness_baselines`'s `data/`, read directly (not copied) in Step 5.
* `tts_eval_harness` library (registered project library, created by
  `t0008_tts_eval_harness_baselines`) — `compute_wer` imported via
  `from tasks.t0008_tts_eval_harness_baselines.code.scoring import compute_wer` in Step 11.
* `tasks/t0008_tts_eval_harness_baselines/data/11labs_david/` — ElevenLabs David reference corpus
  (DVC-tracked), fetched via `dvc pull` in Step 2, used for `build_reference_concat.py`'s reference
  clips and `score_speaker_sim.py`'s centroid.
* `tasks/t0013_v10_synthesis_quality_forensics/results/audio_samples/` — v10 and control audio
  fixtures (DVC-tracked), fetched via `dvc pull` in Step 2, used as the "still fails" fixture in
  Step 12's three-way regression and the reused control sample in Step 9.
* `github.com/semidark/kikiri-tts` (external GitHub repository) — cloned fresh in Step 1 for
  StyleTTS2 source code (`Utils/ASR`, `Utils/JDC`, `Utils/PLBERT` auxiliary models); public, no
  authentication required, same source `t0013`/`t0014` already used.

## Expected Assets

`task.json`'s `expected_assets` field is `{}` — this task produces no new registered asset-type
entries (no `model`, `predictions`, `dataset`, `paper`, or `answer` assets under `assets/`). All
outputs are result documents, code files, and (conditionally) audio samples under `results/`:

* `code/infer_styletts2.py` (instrumented, and with corrected defaults if REQ-6's path is taken),
  `code/audio_quality_check.py` (hardened with 2 new signals), `code/inspect_checkpoint.py`,
  `code/random_decoder_probe.py`, `code/build_reference_concat.py`, `code/score_speaker_sim.py`,
  `code/paths.py`, `code/run_characterization.py`, `code/analyze_localization.py` (or equivalent),
  `code/param_sweep_driver.py` (or CLI flags on `infer_styletts2.py`).
* `results/duration_characterization.json` — 10-text characterization data (REQ-4).
* `results/predictor_tensor_forensics.md` — no-GPU weight-scale comparison (REQ-3).
* `results/param_sweep.json` — 12+ parameter-combination sweep results (REQ-5).
* `results/audio_samples/original/librispeech_control.wav`,
  `results/audio_samples/ft/v11_corrected.wav` — **only if the cheap fix (REQ-6) works**.
* `results/asr_roundtrip_evaluation.md` — ASR-round-trip evaluation verdict (REQ-9).
* `results/gate_regression.json` — three-way v10/v11/corrected regression proof (REQ-10).
* `results/duration_blowup_diagnosis.md` — the canonical root-cause diagnosis document (REQ-11).
* `results/metrics.json` — `rtf`/`speaker_sim` in explicit multi-variant format.
* `results/speaker_sim_scores.json` — raw per-clip speaker-sim scores backing `metrics.json`.

## Time Estimation

* Research (already done): ~1 hour (recorded in `research/research_code.md`).
* Milestone A (environment setup, Steps 1-3): ~30-45 minutes (venv build + `dvc pull` of ~2.1 GB
  checkpoint + reference corpus, network-speed dependent).
* Milestone B (instrument + reproduce, Steps 4-7): ~60-90 minutes (10 CPU synthesis calls at
  observed 73.95s-audio / 235s-wall-time worst case, so up to ~40 minutes of raw synthesis time for
  the characterization batch alone, plus tensor-forensics scripts which run in seconds).
* Milestone C (cheap fix sweep, Steps 8-9): ~60-90 minutes (12+ CPU synthesis calls, each up to ~4
  minutes wall time in the worst case where the parameter change doesn't help).
* Milestone D (gate hardening + regression, Steps 10-12): ~30-45 minutes (code changes plus 3 fast
  `check_audio_quality()` calls on existing WAV files, no new synthesis).
* Milestone E (metrics + diagnosis doc, Steps 13-14): ~30-45 minutes.
* **Total implementation estimate: ~3.5-5.5 hours wall-clock**, dominated by CPU synthesis time in
  Milestones B and C, not by code-writing time.
* Validation (verificators, self-review): ~15-20 minutes.

## Risks & Fallbacks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| `.venv-styletts2` build fails on a dependency (e.g., `monotonic_align` git install, `espeak-ng` apt install without passwordless sudo) | Low | Blocking (Milestone A is `[CRITICAL]`) | `t0013` and `t0014` both already built this exact environment successfully with the same commands; if a version pin has since drifted (e.g., a PyPI package removed an old release), pin to the last-known-good version recorded in `t0013`'s `logs/commands/014_...json` and document the substitution. If `espeak-ng` requires sudo and it fails, create an intervention file requesting human assistance — this blocks all downstream steps. |
| CPU synthesis for the 10-text characterization batch or the 12-combination sweep takes far longer than estimated (e.g., every combination reproduces the ~235s wall-time-per-74s-audio ratio) | Medium | Delays implementation, not a blocker | Time Estimation already budgets for the worst case (up to ~40 min for characterization, ~48 min for the sweep). If wall-clock exceeds double the estimate, reduce the sweep grid (drop the `embedding_scale`/`diffusion_steps` extension combinations, keep only the core 9 `alpha`x`beta` grid) and document the reduction explicitly in `duration_blowup_diagnosis.md` rather than silently truncating results. |
| `dvc pull` for `epoch_00048.pth` (~2.1 GB) or `11labs_david/` fails (network, Azure Blob credentials, or the blob no longer exists) | Low | Blocking (Step 2 is a prerequisite for all synthesis steps) | Retry once. If it persists, check `dvc status` and `dvc remote list` for a misconfigured remote; if the blob is genuinely gone, this is a permanent failure requiring an intervention file — do not substitute a different checkpoint or silently skip the diagnosis. |
| The parameter sweep (Step 8) finds no passing combination (REQ-7 path), and the task is perceived as having "failed" to fix the bug | Medium | None to task completion — this is an anticipated, valid outcome | `task_description.md`'s own Scope step 5 explicitly anticipates and permits this outcome ("if it does not work... Document the negative result with the same rigor as a positive one"). This is not a task failure; Step 9's REQ-7 branch and Step 14 make this an equally complete, equally rigorous deliverable. |
| The no-GPU predictor tensor-forensics (Step 7) is inconclusive (e.g., weight-norm deltas are ambiguous, neither clearly "shifted" nor clearly "frozen") | Medium | Weakens REQ-2/REQ-3's localization confidence | Step 6's forward-pass instrumentation data (REQ-1) is the primary, decisive evidence source; Step 7 is an independent cross-check, not the sole basis for the root-cause verdict. If Step 7 is inconclusive, state that explicitly in `duration_blowup_diagnosis.md` rather than forcing a false-confidence verdict, and rely on Step 6's `max_dur`-saturation / token-count-ratio findings as the primary evidence. |
| A parameter combination in the sweep (Step 8) appears to "pass" on `is_likely_noise`/`duration_sanity_pass`/`longest_nonsilent_run_s` but produces audibly degraded speech (e.g., unnaturally fast/garbled) that no automated signal catches | Low-Medium | Would produce a false "fixed" claim (REQ-6) | Before finalizing the REQ-6 path, a human-audible spot-check of `v11_corrected.wav` should be requested/noted as a recommended manual verification step in `duration_blowup_diagnosis.md`, given this exact project chain's own history (t0013's initial premise) of automated gates missing what a human immediately hears. This plan cannot force a human listen during automated implementation, but it must not claim unconditional success without flagging this residual risk. |

## Verification Criteria

* `uv run python -u -m arf.scripts.verificators.verify_plan t0015_v11_duration_blowup_forensics`
  exits 0 with no errors (run at the end of this planning stage, before implementation begins).
* `test -f tasks/t0015_v11_duration_blowup_forensics/results/duration_blowup_diagnosis.md && test -f tasks/t0015_v11_duration_blowup_forensics/results/duration_characterization.json && test -f tasks/t0015_v11_duration_blowup_forensics/results/param_sweep.json && test -f tasks/t0015_v11_duration_blowup_forensics/results/gate_regression.json`
  exits 0 (all four core deliverable files exist) — confirms REQ-1/2/4
  (`duration_characterization.json`), REQ-5 (`param_sweep.json`), REQ-10 (`gate_regression.json`),
  and REQ-11 (`duration_blowup_diagnosis.md`) all produced their evidence artifacts.
* `python3 -c "import json; d = json.load(open('tasks/t0015_v11_duration_blowup_forensics/results/duration_characterization.json')); assert len(d) == 10, len(d); assert all('pred_dur' in r and 'pred_dur_sum' in r and 'input_token_count' in r for r in d)"`
  exits 0 with no `AssertionError` — confirms Step 5 produced exactly 10 characterization records,
  each with the REQ-1 instrumentation fields populated (not silently omitted).
* `python3 -c "import json; d = json.load(open('tasks/t0015_v11_duration_blowup_forensics/results/gate_regression.json')); byfixture = {r['fixture']: r for r in d}; assert byfixture['v10']['is_likely_noise'] is True; assert byfixture['v11_as_shipped']['is_likely_noise'] is False; assert byfixture['v11_as_shipped']['hardened_gate_pass'] is False"`
  exits 0 — confirms REQ-10's core discriminating proof: the hardened gate correctly still fails v10
  (original signals) and now fails v11-as-shipped (new signals) despite v11 passing the old signals.
* `.venv-styletts2/bin/python -c "from tasks.t0015_v11_duration_blowup_forensics.code.audio_quality_check import check_audio_quality, AudioQualityResult; import inspect; sig = inspect.signature(check_audio_quality); assert 'text' in sig.parameters; assert 'duration_sanity_pass' in AudioQualityResult.__dataclass_fields__; assert 'longest_nonsilent_run_s' in AudioQualityResult.__dataclass_fields__"`
  (run from the repo root with `PYTHONPATH` set appropriately, or as an equivalent `pytest` test in
  `code/test_audio_quality_check.py`) exits 0 with no `AssertionError` — confirms REQ-8's two new
  signals are actually present in the hardened module's public interface, not just described in
  prose.
* `uv run python -u -m arf.scripts.verificators.verify_task_metrics t0015_v11_duration_blowup_forensics`
  (or the project's current metrics verificator entry point, matching
  `arf/specifications/task_results_specification.md`) exits 0 — confirms `results/metrics.json` uses
  only registered metric keys (`rtf`, `speaker_sim`, correctly omitting `ttfb_ms`) in valid
  explicit-variant format.
* Requirement-coverage check:
  `grep -c "REQ-" tasks/t0015_v11_duration_blowup_forensics/results/duration_blowup_diagnosis.md`
  returns a nonzero count, and a manual read confirms all 11 `REQ-*` items from the Task Requirement
  Checklist above are each traceable to a specific step's output file — no `REQ-*` item is left
  unaddressed by any produced artifact.

## Rejection Criteria

Pre-registered before any implementation runs, per `LESSONS.md` Lesson 3 ("pre-register a
failure-rate rejection threshold before running") applied to this task's small-scale CPU synthesis
runs rather than a high-throughput paired-bootstrap benchmark:

* **Characterization batch (Step 5)**: if fewer than 8 of the 10 planned characterization syntheses
  complete successfully (produce a non-empty WAV with finite `pred_dur` values) — i.e.,
  `successful_runs / total_runs < 0.8` — the "universal vs. text-dependent" verdict (REQ-4) is
  **null**. Report exactly which texts failed and why in `duration_blowup_diagnosis.md`; do not draw
  a universal/text-dependent conclusion from a majority-failed batch.
* **Parameter sweep (Step 8)**: if fewer than 80% of the planned 12 (or reduced, per the Risks
  table's fallback) combinations complete successfully, the "cheap fix worked / did not work"
  verdict (REQ-5, REQ-6, REQ-7) is **null** for any combinations that did not run — report only on
  the combinations that actually completed, explicitly stating the incomplete coverage, rather than
  treating a partial sweep as if it covered the full pre-registered grid.
* **"Cheap fix worked" claim (REQ-6)**: no partial improvement may be reported as a pass. A
  combination counts as passing **only if it satisfies all three pre-registered criteria
  simultaneously** (Step 8's `is_likely_noise=False` AND `duration_ratio <= 3.0` AND
  `longest_nonsilent_run_s <= 12.0`) — a combination that improves duration but still fails any one
  of the three signals is a **null** result for REQ-6 purposes, not a qualified or partial success.
* **Three-way regression (Step 12)**: if the v10 fixture, the v11-as-shipped fixture, or (when
  applicable) the corrected fixture cannot be loaded/scored (missing file after `dvc pull`,
  corrupted audio, `check_audio_quality()` exception), the entire three-way regression proof
  (REQ-10) is **null** — a two-way or one-way partial regression does not substitute for the
  pre-registered three-way proof `task_description.md` explicitly requires ("a clean three-way
  regression check").
* These thresholds are stated here, before implementation, and may not be loosened after seeing
  results.
