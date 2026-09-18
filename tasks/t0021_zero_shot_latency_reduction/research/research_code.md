---
spec_version: "1"
task_id: "t0021_zero_shot_latency_reduction"
research_stage: "code"
tasks_reviewed: 17
tasks_cited: 5
libraries_found: 2
libraries_relevant: 1
date_completed: "2026-09-18"
status: "complete"
---
## Task Objective

t0021 profiles where CosyVoice2 and Chatterbox spend their 1.3-2.9 s p50 TTFB (measured in
`t0018_zero_shot_cloning_calibration`) and tests streaming, chunking, vLLM/TensorRT backends and
precision changes to find the lowest TTFB reachable on `LLM-T1-NC80` (H100) without losing
`speaker_sim` versus the t0018 baseline. Per `task_description.md`, it must produce a per-stage
latency breakdown (reference encoding, text frontend, LM prefill/decode, flow-matching, vocoder), a
paired same-session re-measurement of every acceleration variant against the t0018 baseline setting,
and an answer asset stating whether 300 ms TTFB is architecturally or only engineerably reachable.
Its sole declared dependency is `t0018_zero_shot_cloning_calibration`, whose adapters, reference
clips, prompt sets and harness wiring this task must reuse rather than rebuild; this document
surveys what t0018 and its own predecessors actually built so planning knows what to import, what to
copy, and what is genuinely new work (per-stage instrumentation, acceleration variants).

## Library Landscape

The library aggregator (`aggregate_libraries --format json --detail short`) returns exactly 2
registered libraries project-wide:

* **`tts_eval_harness`** (v0.1.0, created by `t0008_tts_eval_harness_baselines`, categories
  `evaluation`/`tts`/`benchmark`) — the reusable speaker_sim/TTFB/RTF/WER/duration-ratio evaluation
  harness. **Highly relevant**: this is the harness t0021's own `task_description.md` "Protocol"
  section explicitly requires ("Same harness protocol as t0018 ... Use the corrected harness from
  t0019 if it has merged"). Import path:
  `from tasks.t0008_tts_eval_harness_baselines.code.<module> import <name>` (e.g. `code.adapters`,
  `code.harness`, `code.scoring`, `code.report`). No correction/replacement overlay is present on
  this library asset as of this pass.
* **`t0009_training_safeguards`** (v0.1.0, created by `t0009_stage2_training_failure_forensics`, no
  categories) — a JSONL step logger, per-epoch checkpoint manager, health gates, and run-config
  capture for Kokoro **StyleTTS2 Stage 2 training**. **Not relevant**: t0021 does no training (its
  own Forbidden section bars fine-tuning) and this library's entry points (checkpoint health gates,
  training-loop logging) have no analog in a zero-shot inference-latency task. Not imported.

No aggregator correction, replacement, or category filter changed either result; both are the raw,
current registered set (`meta/categories/` has no registered categories in this project, consistent
with `research_papers.md`'s own finding on this same aggregator's empty category list).

## Key Findings

### The `SynthResult` adapter pattern is the load-bearing contract for every system t0021 touches

t0008's `tts_eval_harness` (`tasks/t0008_tts_eval_harness_baselines/code/adapters.py`, 330 lines)
defines a frozen `SynthResult` dataclass (`audio_array_16khz`, `audio_array_native`,
`native_sample_rate`, `ttfb_s`, `rtf`, `audio_duration_s`) that every synthesis adapter across the
project returns, and `_kokoro_synth_via_pipeline`/`elevenlabs_david` establish the pattern of
measuring `ttfb_s` as wall time to the first non-empty output chunk [t0008]. t0018's own
`code/adapters_zeroshot.py` (184 lines) follows this contract exactly for the three systems t0021
must accelerate: `f5_tts_synth`, `cosyvoice2_synth`, and `chatterbox_synth`, each returning
`SynthResult` and importing `SynthResult`/`_resample_to_16k` directly from t0008's `adapters.py`
[t0018]. Critically, **none of the three existing adapters records any intermediate timestamp** —
`cosyvoice2_synth` is the only one with an internal loop (over streamed chunks) and it captures only
one intermediate point, `ttfb_s` at the first non-empty chunk; `f5_tts_synth` and `chatterbox_synth`
wrap a single opaque library call (`model.infer(...)`, `model.generate(...)`) with
`time.perf_counter()` before and after and nothing in between [t0018]. This means t0021's Key
Question 1 (a per-stage breakdown of reference encoding, text frontend, LM prefill/decode,
flow-matching, vocoder) requires genuinely new instrumentation — either patching into each library's
internal call sites (e.g. CosyVoice2's `frontend_zero_shot`, its LM's `.inference()`, its
flow-matching model's forward call) or wrapping them with `py-spy`/manual `time.perf_counter()`
checkpoints added around each library sub-call, not something copyable from t0018's own adapters as
they stand.

### CosyVoice2's own adapter already documents the exact bug and API shape t0021 will touch for `load_jit`/`load_trt`/`fp16`

`load_cosyvoice2_model` in `adapters_zeroshot.py` calls
`CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False)` [t0018] — this is the exact
constructor signature `task_description.md`'s Key Question 3 names for t0021's acceleration variants
(`load_jit`, `load_trt`, `fp16`), already wired and just needing the flags flipped per variant.
`cosyvoice2_synth` also documents, in a code comment, a bug found during t0018's implementation:
CosyVoice2's `frontend_zero_shot` calls `load_wav(prompt_wav, N)` internally at three different
sample rates and crashes with `TypeError: Invalid file: tensor(...)` if a pre-loaded tensor rather
than a file path is passed as `prompt_wav` [t0018] — this must be preserved (pass `ref_wav_path`,
not a loaded array) in any adapter variant t0021 writes, including a future vLLM-backed one.
`cosyvoice2_synth` also shows
`model.inference_zero_shot(text, prompt_text, str(ref_wav_path), stream=True)` already streams, and
confirms `task_description.md`'s own finding that streaming alone still measured 2.86 s p50 TTFB on
val96 (matching the stored `cosyvoice2_ref_single_val96` variant, `ttfb_ms=2859.25`, in
`tasks/t0018_zero_shot_cloning_calibration/results/metrics.json`) [t0018].

### Isolated per-system venvs are the established pattern, and their pinned `torch`/CUDA versions already diverge

t0018's plan (`plan/plan.md`, "Isolated venvs, not shared `pyproject.toml`") created `.venv-f5tts`,
`.venv-cosyvoice2`, `.venv-chatterbox` on `LLM-T1-NC80` rather than adding these packages to the
shared `pyproject.toml`, explicitly to avoid `torch`/`torchaudio` conflicts across three research
repos, following the single-purpose-venv convention already used by
`tasks/t0014_v11_decoder_fix_retrain/code/build_reference_concat.py` (`.venv-styletts2`)
[t0014, t0018]. `results/environment.json` confirms the divergence is real, not hypothetical:
Chatterbox ran `torch==2.6.0+cu124`, CosyVoice2 ran `torch==2.3.1+cu121`, both on the same physical
H100 NVL [t0018]. This is a direct constraint on t0021's vLLM/TensorRT-LLM work (Key Question 3): a
vLLM install for CosyVoice2's Qwen2.5-0.5B backbone must be added inside `.venv-cosyvoice2` (or a
fresh `.venv-cosyvoice2-vllm`) and independently version-checked against CosyVoice2's pinned
`torch 2.3.1+cu121`, since vLLM itself pins specific `torch` versions per release and a mismatch is
exactly the kind of dependency conflict t0018's plan flagged as a medium-likelihood risk to its
45-minute smoke-gate time-box.

### The project's own lessons (Lessons 1/2/3/4/8/10/11) are already operationalized in t0018's code, not just prose

`task_description.md`'s Protocol section cites Lessons 1, 3, and 4 by number, and t0018's
`plan/plan.md` shows exactly how each was implemented in code: Lesson 1 (cold/warm cache pairing) as
`N_WARMUP=50` discarded requests before every measured run in `_run_one_condition`
(`run_eval_zeroshot.py`) [t0018]; Lesson 3 (pre-registered rejection threshold) as
`SUCCESS_RATE_THRESHOLD=0.8` in `constants.py`, enforced in `report_zeroshot.py`'s
`compute_variant_metrics_zeroshot` (`rejected = success_rate < 0.8`, which nulls all three metrics
and records `"rejected_reason": "successful_requests/total_requests < 0.8"`) [t0018]; Lesson 4
(capture infra versions at measurement time) as `_write_environment_record` in
`run_eval_zeroshot.py`, writing `torch_version`/`torch_cuda_version`/`gpu_name`/`package_version`
per system into `results/environment.json` [t0018]. Lessons 8, 10, and 11
(watchdog-before-first-download, `/mnt/cache/persist` symlink verification,
`loginctl enable-linger`) are infrastructure-level and enforced by
`arf/scripts/utils/remote_preflight.sh` during VM acquisition rather than in task code [LESSONS.md];
t0021's own budget section already names the watchdog-armed-before-first-build requirement, so no
new code is needed there beyond following `/setup-remote-machine`.

### The hardened audible-speech gate is imported cross-task by t0018 — a pattern t0021 cannot repeat under the current spec

t0018's `code/run_gate_check.py` imports `check_audio_quality` and
`LONGEST_NONSILENT_RUN_THRESHOLD_S` **directly** from
`tasks.t0015_v11_duration_blowup_forensics.code.audio_quality_check` (not copied), citing a "Key
Rule 5: never copy/modify another task's module" in its own module docstring and reusing exactly the
`hardened_gate_pass` combination
(`is_likely_noise OR duration_sanity_pass is False OR longest_nonsilent_run_s > 12.0`) that t0015's
own `run_gate_regression.py` uses [t0015, t0018]. `audio_quality_check.py`
(`tasks/t0015_v11_duration_blowup_forensics/code/audio_quality_check.py`, 233 lines) is itself a
lineage: unchanged logic copied forward from `t0013_v10_synthesis_quality_forensics`, extended in
place by t0015 with two new signals (`duration_sanity_pass`, `longest_nonsilent_run_s`) alongside
the original three (`silence_fraction`, `spectral_flatness`, `clip_fraction`) [t0015]. **This
project's current `arf/specifications/research_code_specification.md` explicitly forbids cross-task
`code/` imports** ("Tasks MUST NOT import from other tasks' `code/` directories. The only cross-task
import mechanism is libraries") — t0015's `audio_quality_check.py` was never registered as a
library, so t0018's direct-import pattern, whatever rule justified it at the time, is not reusable
as-is for new t0021 code; the function itself must be **copied into `t0021`'s own `code/`** (see
Reusable Code below).

### Reference-audio construction and CosyVoice2's 30 s hard limit are already characterized, with the exact fix t0021 needs

`task_description.md`'s "What t0018 already established" section states CosyVoice2's `ref_concat`
failed only because the shared reference clip was 30.57 s against CosyVoice2's own
`assert speech.shape[1] / 16000 <= 30` hard limit (documented as finding S-0018-02) — confirmed in
t0018's `results/metrics.json`, where `cosyvoice2_ref_concat_val96` and
`cosyvoice2_ref_concat_fillers` are both `null` for every metric [t0018]. t0018's
`code/build_references.py` (264 lines, not read in full but referenced throughout `plan/plan.md`)
builds `ref_concat` by concatenating half-A clips with 0.2 s silence gaps in filename-sorted order
"stopping once total duration first exceeds 30.0 s" — i.e. `REF_CONCAT_TARGET_DURATION_S=30.0` in
`constants.py` is the exact parameter that produced the 30.57 s clip that broke the hard limit
[t0018]. t0018's own `code/paths.py` already exposes `REF_CONCAT_WAV`/`REF_SINGLE_WAV`/
`REFERENCES_MANIFEST` at fixed, importable paths, and `build_reference_concat.py`
(`tasks/t0014_v11_decoder_fix_retrain/code/`, 52 lines) is the earlier, simpler pattern
`build_references.py` was adapted from (fixed 3-clip concatenation with the same 0.2 s gaps,
sourcing `ELEVENLABS_DAVID_DIR`) [t0014]. Both give t0021 the exact concatenation logic to copy and
lower the target from 30.0 s to 29.5 s per `task_description.md`'s "re-run CosyVoice2 `ref_concat`
with a 29.5 s reference" instruction.

### Cost tracking and variant-metrics reporting follow an established, directly extensible pattern

`code/track_cost.py` (t0018, 65 lines) appends milestone-keyed cost checkpoints to
`results/cost_tracking.json`, computing `elapsed_h * VM_HOURLY_COST_USD` from a fixed
`VM_BILLING_ANCHOR_ISO` [t0018] — directly reusable for t0021's own `$100` hard cap with updated
constants. `code/report_zeroshot.py`'s
`compute_variant_metrics_zeroshot(records, system, condition, prompt_set)` (t0018) extends t0008's
two-dimensional `report.compute_variant_metrics(records, system, prompt_set)` to a third `condition`
dimension, already importing `REGISTERED_METRIC_KEYS` and `_percentile` from t0008's `report.py`
[t0008, t0018] — t0021 needs a fourth dimension (`acceleration_variant`) on top of
`system`/`condition`(if kept)/`prompt_set`, which is a straightforward extension of the same
filter-and-aggregate pattern, not a rewrite.

## Reusable Code and Assets

* **Source**: `tts_eval_harness` library (registered, created by
  `t0008_tts_eval_harness_baselines`). **What it does**: `SynthResult` dataclass, `save_wav`,
  prompt-set loading (`harness.get_prompts_by_set`, `harness.build_reference_split`), GE2E scoring
  (`scoring.build_centroid`, `scoring.compute_speaker_sim`), WER (`scoring.compute_wer`), duration
  sanity (`scoring.compute_duration_ratio`), and reporting primitives
  (`report.REGISTERED_METRIC_KEYS`, `report._percentile`, `report.compute_variant_metrics`). **Reuse
  method**: **import via library** —
  `from tasks.t0008_tts_eval_harness_baselines.code.adapters import SynthResult, save_wav, _resample_to_16k`;
  `from tasks.t0008_tts_eval_harness_baselines.code.harness import get_prompts_by_set, build_reference_split, PromptItem`;
  `from tasks.t0008_tts_eval_harness_baselines.code.scoring import build_centroid, compute_speaker_sim, compute_wer, compute_duration_ratio`.
  **Signatures**:
  `build_reference_split(wav_dir: Path, seed: int) -> tuple[np.ndarray, list[Path]]`;
  `compute_speaker_sim(synth_wavs: list[Path], ref_embeddings: np.ndarray) -> SpeakerSimResult`
  (`.mean`, `.std`, `.per_clip`); `get_prompts_by_set(name: str) -> dict[str, list[PromptItem]]`.
  **Adaptation**: none — use as-is. **Line count**: n/a (library import, not copied).

* **Source**: `tasks/t0018_zero_shot_cloning_calibration/code/adapters_zeroshot.py` (184 lines).
  **What it does**: `load_f5tts_model`, `f5_tts_synth`, `load_cosyvoice2_model`, `cosyvoice2_synth`
  (already wired for `load_jit`/`load_trt`/`fp16` flags), `load_chatterbox_model`,
  `chatterbox_synth` — all matching the `SynthResult` contract. **Reuse method**: **copy into task**
  (not a library; `t0018` registered only an `answer` asset, no `library` asset). **Signatures**:
  `f5_tts_synth(text: str, *, ref_wav_path: Path, ref_text: str, model: object) -> SynthResult`;
  `cosyvoice2_synth(text: str, *, ref_wav_path: Path, prompt_text: str, model: object) -> SynthResult`;
  `chatterbox_synth(text: str, *, ref_wav_path: Path, model: object) -> SynthResult`. **Adaptation
  needed**: add intermediate `time.perf_counter()` checkpoints inside each adapter (or around each
  library sub-call) to populate the per-stage `latency_breakdown.json` schema (reference encoding,
  text frontend, LM prefill/decode, flow-matching, vocoder) — none of the three functions currently
  records anything between call-start and call-end; add acceleration-variant parameters (`fp16`,
  `load_jit`, `load_trt`, dtype, chunk size) to the two `load_*` functions and thread a
  `variant_id`/`is_streaming` field through. **Line count**: ~184 lines to copy, plus new
  instrumentation.

* **Source**: `tasks/t0018_zero_shot_cloning_calibration/code/run_eval_zeroshot.py` (346 lines).
  **What it does**: CLI runner: loads a model once per invocation, runs `N_WARMUP` discarded
  warmups, synthesizes all prompts in `prompts_by_set`, saves WAVs and per-clip JSON records, writes
  `results/environment.json` via `_write_environment_record`. **Reuse method**: **copy into task**.
  **Signatures**: `main() -> None` (argparse CLI);
  `_run_one_condition(*, system, condition, synth, is_streaming, prompts_by_set, n_warmup, limit, audio_out_dir, out_dir) -> None`.
  **Adaptation needed**: add an `--acceleration-variant` CLI flag and thread it into
  `variant_slug`/output paths (currently only `system`/`condition` compose `variant_slug`); extend
  the per-clip record schema with per-stage timing fields. **Line count**: ~346 lines to copy and
  extend.

* **Source**: `tasks/t0018_zero_shot_cloning_calibration/code/{constants.py, paths.py}` (59 + 75
  lines). **What it does**: named constants (`N_WARMUP=50`, `SUCCESS_RATE_THRESHOLD=0.8`,
  `GATE_TEXT_NAMES`, `VM_HOURLY_COST_USD=13.96`, stored t0018 baseline numbers for chart reference
  lines) and path constants (`REF_SINGLE_WAV`, `REF_CONCAT_WAV`, `RESULTS_AUDIO_HARNESS_DIR`, chart
  paths) re-exporting several from t0008's `paths.py`. **Reuse method**: **copy into task**.
  **Adaptation needed**: update `BUDGET_HARD_CAP_USD` (70.0 to t0021's 100.0),
  `REF_CONCAT_TARGET_DURATION_S` (30.0 to 29.5 per S-0018-02), add acceleration-variant identifiers
  and a `latency_breakdown.json` path constant. **Line count**: ~134 lines combined.

* **Source**: `tasks/t0018_zero_shot_cloning_calibration/code/build_references.py` (264 lines) and
  `tasks/t0014_v11_decoder_fix_retrain/code/build_reference_concat.py` (52 lines, the earlier,
  simpler pattern it was adapted from). **What it does**: builds `ref_single`/`ref_concat` reference
  clips from the half-A David split, with 0.2 s silence gaps between concatenated clips. **Reuse
  method**: **copy into task** (t0018's version; t0014's is superseded lineage, cite for context
  only). **Adaptation needed**: lower the concat stopping threshold from 30.0 s to 29.5 s
  (S-0018-02). **Line count**: ~264 lines.

* **Source**: `tasks/t0018_zero_shot_cloning_calibration/code/run_gate_check.py` (92 lines) **and**
  `tasks/t0015_v11_duration_blowup_forensics/code/audio_quality_check.py` (233 lines). **What it
  does**: runs the hardened audible-speech gate (`is_likely_noise` OR
  `duration_sanity_pass is False` OR `longest_nonsilent_run_s > 12.0`) over every synthesized clip
  and writes `gate_failures.json`. **Reuse method**: `run_gate_check.py` structure — **copy into
  task**; `check_audio_quality`/`AudioQualityResult`/`LONGEST_NONSILENT_RUN_THRESHOLD_S` from
  `audio_quality_check.py` — **must also be copied into task** (t0018 imported this cross-task
  directly from `t0015`'s `code/`, which the current `research_code_specification.md` disallows for
  new code since `t0015` registered no library asset). **Signatures**:
  `check_audio_quality(wav_path: Path, *, text: str | None = None) -> AudioQualityResult`.
  **Adaptation needed**: none to the logic itself; change the import source to the copied local
  module. **Line count**: 92 + 233 = ~325 lines.

* **Source**: `tasks/t0018_zero_shot_cloning_calibration/code/track_cost.py` (65 lines). **What it
  does**: appends milestone-keyed elapsed-time/cost checkpoints to `cost_tracking.json` from a fixed
  billing anchor. **Reuse method**: **copy into task**. **Adaptation needed**: update
  `VM_BILLING_ANCHOR_ISO` to t0021's own session start and `BUDGET_HARD_CAP_USD` to 100.0. **Line
  count**: ~65 lines.

* **Source**: `tasks/t0018_zero_shot_cloning_calibration/code/report_zeroshot.py` (390 lines,
  `compute_variant_metrics_zeroshot` at lines 27-100+). **What it does**: aggregates per-clip
  records into variant-level `speaker_sim`/`ttfb_ms` (p50/p95/p99)/`rtf`, applying the
  `SUCCESS_RATE_THRESHOLD` rejection rule, and (per `plan/plan.md`) generates the four t0018 charts.
  **Reuse method**: **copy into task**, extending the pattern. **Adaptation needed**: add a fourth
  grouping dimension (`acceleration_variant`) alongside `system`/`condition`/`prompt_set`; add new
  chart functions for `latency_breakdown_stacked.png` (new chart type, not present in t0018) and
  `ttfb_p50_p95_by_variant.png` (t0018 only had `ttfb_vs_speaker_sim.png`, a scatter — the new bar
  chart is new code). **Line count**: ~390 lines as a starting pattern; substantial new charting
  code required on top.

## Lessons Learned

* **Warmup discarding and paired-baseline re-measurement work as designed**: t0018's 50-warmup
  protocol and same-session baseline re-measurement (`kokoro_v3_bundle`, `elevenlabs_david`
  re-scored from stored audio) produced numbers t0021's `task_description.md` treats as trustworthy
  ground truth to build on, not re-derive from scratch [t0018].
* **Isolated venvs prevented cross-system breakage but did not prevent within-system version
  drift**: Chatterbox (`torch 2.6.0+cu124`) and CosyVoice2 (`torch 2.3.1+cu121`) coexisted safely on
  one VM because of `.venv-*` isolation, but this means t0021 cannot assume a single shared
  vLLM/TensorRT install serves both systems — each needs its own acceleration-stack install,
  doubling the setup surface area the task's 1-hour setup budget must cover [t0018].
* **F5-TTS failed to load three times with an unresolved hang** [t0018, per `task_description.md`] —
  `research_internet.md`'s [F5TTS-GH-Issues] finding (jieba dictionary init / stale HF cache as a
  candidate cause) is the only lead; t0018's own code shows no special handling for this failure
  mode beyond the standard 45-minute smoke-gate time-box and null-with-error convention, so t0021's
  `py-spy dump`-on-hang retry (per its own task description) is genuinely new diagnostic work.
* **A hard, undocumented input limit caused a full null variant**: CosyVoice2's `ref_concat` at
  30.57 s tripped an internal `assert speech.shape[1] / 16000 <= 30` with no graceful degradation,
  nulling both `cosyvoice2_ref_concat_val96` and `cosyvoice2_ref_concat_fillers` in
  `results/metrics.json` — a reminder that any new acceleration variant (e.g. a different chunk size
  or a TensorRT export with its own shape constraints) needs its own smoke-gate check before the
  full 196-prompt run, exactly as Lesson 2 already prescribes [t0018].
* **The `successful_prompts / total_prompts < 0.8` rejection rule and the `hardened_gate_pass`
  three-signal gate are both mature, tested code** (11 unit tests in `t0008`'s `test_harness.py`;
  `t0015`'s gate logic proven against a real known-bad checkpoint in `t0013`/`t0015`'s own
  regression) — safe to reuse verbatim rather than re-deriving thresholds
  [t0008, t0013 via t0015, t0015].
* **Cross-task code imports (t0018 importing `t0015`'s `audio_quality_check.py` directly) are a
  pattern the current specification no longer permits for new code** — the fix is mechanical (copy
  the 233-line module) but must not be silently skipped, since the current
  `research_code_specification.md`'s Cross-Task Code Reuse Rule is unambiguous about library-only
  cross-task imports.

## Recommendations for This Task

1. **Import `tts_eval_harness` via library** for `SynthResult`, `save_wav`, `get_prompts_by_set`,
   `build_reference_split`, and all scoring functions (`build_centroid`, `compute_speaker_sim`,
   `compute_wer`, `compute_duration_ratio`) — do not reimplement any of these.
2. **Copy, not import, t0018's `code/adapters_zeroshot.py`, `run_eval_zeroshot.py`, `constants.py`,
   `paths.py`, `build_references.py`, `run_gate_check.py`, `track_cost.py`, and
   `report_zeroshot.py`** into `t0021`'s own `code/` directory, since `t0018` registered no library
   asset. Preserve the `SynthResult` contract, the `is_streaming` labeling convention, and the
   `SUCCESS_RATE_THRESHOLD`/rejection-reason pattern exactly.
3. **Copy `t0015`'s `audio_quality_check.py` (233 lines) into `t0021`'s `code/`** rather than
   reproducing t0018's direct cross-task import of it — the current `research_code_specification.md`
   forbids importing from another task's `code/` outside a registered library, and `t0015` has no
   library asset.
4. **Add per-stage timestamp instrumentation as new code**, structured around `research_papers.md`'s
   adopted `L_TTS = M·d_lm + M·d_fm + M·d_voc` model plus explicit reference-encoding and
   text-frontend stages — none of t0018's three adapters currently records any intermediate timing,
   so this is the task's primary new implementation surface, not a reuse gap to fill from prior
   code.
5. **Set `REF_CONCAT_TARGET_DURATION_S = 29.5`** (down from t0018's `30.0`) in the copied
   `build_references.py`/`constants.py` to close S-0018-02, and smoke-test the resulting clip
   against CosyVoice2's `assert speech.shape[1] / 16000 <= 30` before the full run.
6. **Install any vLLM/TensorRT-LLM acceleration stack inside each system's own isolated venv**
   (`.venv-cosyvoice2`, a new `.venv-chatterbox`-adjacent venv if needed), never in a shared
   environment — `torch 2.3.1+cu121` (CosyVoice2) and `torch 2.6.0+cu124` (Chatterbox) already
   diverge on the same VM per `results/environment.json`, and a vLLM/TensorRT install pins its own
   `torch` version that must be checked against each.
7. **Reuse t0018's stored numbers as fixed chart reference lines**
   (`ELEVENLABS_SPEAKER_SIM_FILLERS =0.832`, `KOKORO_V3_BUNDLE_SPEAKER_SIM_FILLERS=0.631`,
   CosyVoice2/Chatterbox `speaker_sim`/`ttfb_ms` per variant in
   `tasks/t0018_zero_shot_cloning_calibration/results/metrics.json`) but still re-measure each
   system's t0018-equivalent baseline setting fresh, in the same session as every acceleration
   variant, per `task_description.md`'s explicit pairing requirement and Lesson 1.
8. **Extend `report_zeroshot.py`'s variant-metrics pattern with an `acceleration_variant`
   dimension** rather than writing a new aggregation function from scratch, and add the two
   genuinely new charts (`latency_breakdown_stacked.png`, `ttfb_p50_p95_by_variant.png`) as new
   plotting functions in the same module, following its existing `matplotlib` setup conventions
   inherited from `t0008`'s `report.py`.

## Task Index

### [t0008]

* **Task ID**: `t0008_tts_eval_harness_baselines`
* **Name**: TTS evaluation harness and baselines
* **Status**: completed
* **Relevance**: Created the `tts_eval_harness` library t0021 imports for `SynthResult`, prompt-set
  loading, GE2E/WER/duration scoring, and the reporting primitives extended by `report_zeroshot.py`.

### [t0014]

* **Task ID**: `t0014_v11_decoder_fix_retrain`
* **Name**: Kokoro Stage 2 v11: decoder-init fix, full normalized corpus retrain
* **Status**: completed
* **Relevance**: Its `build_reference_concat.py` is the earlier, simpler pattern t0018's
  `build_references.py` (which t0021 copies) was adapted from, including the isolated-venv
  invocation convention t0018 also follows.

### [t0015]

* **Task ID**: `t0015_v11_duration_blowup_forensics`
* **Name**: v11 duration-blowup forensics and audible-speech gate hardening
* **Status**: completed
* **Relevance**: Owns `audio_quality_check.py`, the hardened audible-speech gate t0021 must copy
  (not cross-task-import, as t0018 did) to keep running the same `hardened_gate_pass` check on every
  synthesized clip.

### [t0009]

* **Task ID**: `t0009_stage2_training_failure_forensics`
* **Name**: Stage 2 training failure forensics and safeguards
* **Status**: completed
* **Relevance**: Registered the project's only other library, `t0009_training_safeguards`, reviewed
  and confirmed not relevant — it targets StyleTTS2 Stage 2 training health/checkpointing, which
  t0021's no-fine-tuning latency task never touches.

### [t0018]

* **Task ID**: `t0018_zero_shot_cloning_calibration`
* **Name**: Zero-shot voice-cloning calibration: reachable speaker_sim for David
* **Status**: completed
* **Relevance**: t0021's sole declared dependency. Its adapters (`adapters_zeroshot.py`), harness
  runner (`run_eval_zeroshot.py`), reference-audio builder (`build_references.py`), gate checker
  (`run_gate_check.py`), cost tracker (`track_cost.py`), report writer (`report_zeroshot.py`),
  constants/paths, and the stored `results/metrics.json`/`results/environment.json` baseline numbers
  are the direct starting point for every piece of t0021's implementation.
