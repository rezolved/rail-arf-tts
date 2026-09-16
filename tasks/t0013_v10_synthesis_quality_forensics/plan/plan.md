---
spec_version: "2"
task_id: "t0013_v10_synthesis_quality_forensics"
date_completed: "2026-09-16"
status: "complete"
---
# Plan: v10 Checkpoint Synthesis Quality Forensics

## Objective

Determine, with evidence rather than opinion, whether `kokoro-v10-best` (the Stage 2 checkpoints
`epoch_2nd_00016.pth` and `epoch_2nd_00014.pth` produced by `t0010_stage2_safeguarded_training`)
produces noise instead of speech because of (a) a bug in the ad hoc local inference reproduction
performed outside the ARF pipeline earlier this session, or (b) a real defect in the checkpoint
itself, and then act on whichever is true. "Done" means: a cheap tensor-level falsifier has been run
first and its verdict recorded; a reusable, instrumented, StyleTTS2-native inference harness exists
under `code/` and has been validated against a known-good control checkpoint before ever being
trusted on v10; real audio files exist under `results/audio_samples/` for the control checkpoint and
both v10 checkpoints; `results/control_test.md` and `results/v10_diagnosis.md` state a definitive
root cause (reproduction bug vs. late-training defect vs. defect-from-the-start) with a concrete
recommended action; and a short automatable noise-detection check exists under `code/` for future
training tasks to run before claiming completion. No GPU, no remote machine, and no paid API calls
are required — this is a CPU-only local forensics task.

## Task Requirement Checklist

Operative task text, quoted from `task.json` and `task_description.md`:

> **name**: "v10 checkpoint synthesis quality forensics" **short_description**: "Determine whether
> kokoro-v10-best produces noise instead of speech due to a bug in an ad hoc inference reproduction,
> or a real checkpoint defect, and act accordingly."

> **Key Questions** (`task_description.md`):
> 1. Does the ad hoc inference script actually load every module's weights correctly? (Log
>    missing/unexpected key counts explicitly, per module, for both the primary load attempt and the
>    `module.`-stripped fallback.)
> 2. With a known-good control (base pretrained StyleTTS2 LibriTTS checkpoint), does the same
>    inference harness produce intelligible speech?
> 3. If the harness is validated correct: does `epoch_2nd_00016.pth` (primary) produce noise while
>    `epoch_2nd_00014.pth` (backup) produces speech, or do both fail?
> 4. Do any loaded modules contain NaN/Inf weights, or weight-norm statistics wildly divergent from
>    a known-good reference?
> 5. Does `metrics.jsonl` show any per-step anomaly the epoch-level val_loss average could have
>    masked?
> 6. Is `compute_style`'s reference-clip-length requirement itself a sign of mismatch, or expected
>    StyleTTS2 behavior?
> 
> **Scope**: (1) reproduce the ad hoc setup properly, with instrumentation, under `code/`; (2) run a
> control test on the base pretrained checkpoint before touching v10 again; (3) diagnose v10
> specifically (primary vs. backup, NaN/Inf, weight-norm, metrics.jsonl cross-reference); (4)
> root-cause and recommend a fix, and produce a short automatable regression check.
> 
> **Expected Outputs**: (1) a reusable inference recipe — committed, documented, instrumented script
> under `code/`, not a throwaway `/tmp` script; (2) audio samples saved as real files under
> `results/audio_samples/` for the control checkpoint and v10 primary/backup.

Requirement decomposition:

| ID | Requirement | Satisfied by step(s) | Evidence |
| --- | --- | --- | --- |
| REQ-1 | Cheap tensor-level falsifier runs **first**, before any inference/venv code, checking `net["decoder"]` key naming, NaN/Inf, and per-module weight-norm | Steps 4-6 | `results/checkpoint_forensics.md`, `code/inspect_checkpoint.py` |
| REQ-2 | Instrumented harness logs missing/unexpected key counts per module, for primary and `module.`-stripped fallback load paths (Key Question 1) | Steps 8-9 | `code/infer_styletts2.py` console/log output captured in `logs/` |
| REQ-3 | Control test against base pretrained StyleTTS2 LibriTTS checkpoint through the identical harness, **before** touching v10 (Key Question 2) | Step 10 | `results/control_test.md`, `results/audio_samples/control_*.wav` |
| REQ-4 | Compare v10 primary (`epoch_2nd_00016.pth`) vs. backup (`epoch_2nd_00014.pth`) outputs (Key Question 3) | Steps 12-13 | `results/audio_samples/v10_epoch16_*.wav`, `v10_epoch14_*.wav`, `results/v10_diagnosis.md` |
| REQ-5 | NaN/Inf + per-module weight-norm comparison across v10 checkpoints and control (Key Question 4) | Steps 4-6, 13 | `results/checkpoint_forensics.md` |
| REQ-6 | Cross-reference `data/run_v10/metrics.jsonl` for per-step signal missed by epoch-level val_loss (Key Question 5) | Step 14 | `results/v10_diagnosis.md` section citing `tasks/t0010_stage2_safeguarded_training/data/run_v10/metrics.jsonl` |
| REQ-7 | Document whether `compute_style`'s reference-clip-length requirement is a mismatch signal or expected behavior (Key Question 6) | Step 9 | `results/v10_diagnosis.md` note |
| REQ-8 | Reusable inference recipe: kikiri-tts clone + submodules under `code/` (gitignored), instrumented script committed | Steps 2, 7-9 | `code/infer_styletts2.py`, `code/.gitignore` |
| REQ-9 | Audio sample deliverables under `results/audio_samples/`, DVC-tracked | Steps 10, 12, 15 | `results/audio_samples/*.wav`, `.dvc` files |
| REQ-10 | `results/control_test.md` with pass/fail + objective stats (duration, RMS, peak) | Step 10 | file exists, non-empty |
| REQ-11 | `results/v10_diagnosis.md` with root cause and recommended action (reproduction bug / late-epoch defect / defect-from-start) | Step 14 | file exists, states one of the three verdicts explicitly |
| REQ-12 | Short, objective, automatable regression check (spectral flatness / silence fraction / tiny ASR round-trip) saved to `code/` for future training tasks | Step 16 | `code/audio_quality_check.py` with a runnable self-check |
| REQ-13 | Applicable registered metrics (`speaker_sim`, `rtf`) measured and written to `results/metrics.json`; `ttfb_ms` explicitly omitted with reasoning | Step 17 | `results/metrics.json` (explicit variant format) |

Ambiguity note: `task_description.md`'s Scope §1 says `brew install espeak-ng` for the phonemizer
backend — that instruction was written during a prior session on a different (macOS) machine. This
plan's execution environment is the Linux Azure VM worktree, so Step 7 substitutes
`sudo apt-get install -y espeak-ng` (passwordless `sudo` is available in this environment, confirmed
during planning). This is a mechanical substitution, not a scope change.

## Approach

**Grounded in `research/research_summary.md` and direct inspection of the checkpoint/config files
during planning.** The leading hypothesis, confirmed by reading
`tasks/t0010_stage2_safeguarded_training/code/train_second_v10.py` directly during planning:
`config_david_v10.yml` is the only Stage 2 config in the whole project with
`model_params.decoder.type: hifigan` (no `gen_istft_*` keys, four-stage fully-learned upsampler
`upsample_rates: [10, 5, 3, 2]`) — every other config (`config_david_v4.yml`,
`config_david_v5_stage2.yml`, `config_david_v6_stage2.yml`/`v6b`/`v6c`/`v6d`) uses `istftnet`
(two-stage upsampler `[10, 6]` plus an inverse-STFT stage, `gen_istft_hop_size: 5`,
`gen_istft_n_fft: 20`). `train_second_v10.py:load_checkpoint()` (lines 87-118) loads
`first_stage_path` (`first_stage_v3.pth`) into every module in `model` whose key is not in
`ignore_modules`. The call site at lines 244-260 passes
`ignore_modules=["predictor_encoder", "msd", "mpd", "wd", "diffusion"]` — **`decoder` is not
excluded**. The loader's failure guard only raises `RuntimeError` when `len(matched) == 0` (line
106); a *partial* match (some architecture-agnostic conv layers overlap, the HiFi-GAN
generator/vocoder layers proper do not) satisfies `len(matched) > 0` and passes silently, printing
`"{key} loaded: N/M params"` with no indication that the ratio is poor. `first_stage_v3.pth` is
confirmed to be an ISTFTNet-shaped checkpoint: it is the exact file at
`tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/stage1/first_stage.pth` (1,724,784,307 bytes,
DVC-tracked; `task_description.md` for t0006 states "v6c/v6d used v3's first_stage.pth" and both
v6c/v6d configs use `istftnet`). This makes a randomly-initialized-then-barely-touched HiFi-GAN
vocoder, trained for only 17 epochs (8 pre-GAN + 9 post-GAN), the leading explanation for "pure
noise, no voice at all."

**Why the cheap falsifier runs first (Milestone B, before any venv/inference code):** confirming or
refuting this hypothesis does not require running inference at all — it only requires (a) reading
`net["decoder"]`'s key names out of the raw `.pth` file with `torch.load`, which the **existing**
main project `.venv` (already pinning `torch>=2.12.1`) can do without any new environment, and (b)
reading the StyleTTS2 source (`Modules/hifigan.py` vs `Modules/istftnet.py`, obtained by a plain
`git clone`, no `pip install`) to see which submodule attribute names (`self.ups`, `self.resblocks`,
etc.) each decoder type registers. This can confirm or refute the architecture-mismatch hypothesis
in minutes. Building the full StyleTTS2 CPU inference harness (pinned `torch==2.5.1`, `espeak-ng`,
`monotonic_align` compiled from source, ~10+ Python packages) takes materially longer and should not
gate the cheap check.

**Why the full harness (Milestone C-E) is still required even if Milestone B is conclusive:** the
task's Expected Outputs explicitly demand a reusable inference recipe and real audio-file
deliverables under `results/audio_samples/`, not just a tensor-level verdict. A tensor-level
falsifier can explain *why* the checkpoint sounds like noise, but it cannot itself produce the audio
evidence the user asked for, and it does not exercise the `compute_style` reference-clip behavior
(Key Question 6), which only manifests at inference time. Milestones C-E are executed regardless of
Milestone B's outcome.

**Alternatives considered:**

* *Route inference through `kokoro.KModel`/`KPipeline` (the pip package) instead of StyleTTS2's
  native `models.py`.* Rejected: research confirms `kokoro.KModel` only supports the ISTFTNet
  decoder shape; loading a `hifigan`-decoder checkpoint into it fails with tensor shape mismatches
  on `decoder.generator.*` (this is literally how the noise was first discovered, per
  `checkpoint.md`). `t0008`'s `adapters.load_kokoro_model_with_checkpoint()` and `t0010`'s
  `eval_all_checkpoints.py` both route through this same broken path — reference-only, not usable
  here.
* *Trust `t0010`'s aggregate parameter-match assertion and epoch-level `val_loss` as sufficient
  evidence of a healthy checkpoint, skip tensor/audio forensics.* Rejected: `val_loss` is an
  aggregate reconstruction loss that would not necessarily surface a defect isolated to the
  adversarial decoder, and the parameter-count assertion in `train_second_v10.py` only checks
  aggregate match rate, not per-module correctness — exactly the anti-pattern this project has hit
  twice before (`t0009`'s training loader, `t0008`'s `adapters.py`). Root cause must come from
  evidence closer to the actual defect (checkpoint tensors, real audio).
* *Import `tts_eval_harness` (from `t0008`) or `t0009_training_safeguards` (from `t0009`)
  wholesale.* Rejected: both registered libraries are coupled to the wrong things for this task —
  `tts_eval_harness` is `kokoro.KModel`/ISTFTNet-coupled (`overview/libraries` confirms
  `import_path: null`, i.e. not directly importable as a package; its `score_speaker_sim.py` pattern
  is reused by copy, see Step 17), and `t0009_training_safeguards` is training-loop-coupled.
  Reference patterns only, per `research/research_summary.md` point 8.

**Recommended task type(s):** `task.json` already declares
`["tts-benchmark-run", "code-reproduction"]`. Both apply only partially, and this plan documents the
mismatch explicitly rather than silently ignoring the type-specific guidance:

* `tts-benchmark-run` (`meta/task_types/tts-benchmark-run/instruction.md`): the `speaker_sim`
  computation method (GE2E cosine vs. `data/11labs_david/` reference set via `resemblyzer`, kept in
  the `[speaker-sim]` optional extra per `overview/metrics/speaker_sim.md`, never added to main
  `pyproject.toml` dependencies) is reused exactly as specified (Step 17). The full protocol —
  smoke-gate, 50 discarded warmup requests, N >= 100 measured requests, `ttfb_ms` — does **not**
  apply: this task synthesizes a handful of forensic clips from an offline batch script, not a
  production serving endpoint, so there is no "time to first audio byte" to measure and no benefit
  to a 100-clip warmed-up run. This is a deliberate, documented omission (see Step 17 and the
  metrics discussion below), not an oversight.
* `code-reproduction` (`meta/task_types/code-reproduction/instruction.md`): the environment-pinning
  discipline (exact library versions, documented substitutions such as the `espeak-ng` install
  command change above) and the "preserve full logging, document every modification" guidance are
  followed. The type's core guidance — train a model from scratch, compare to a published paper
  within 2 F1 points, save a `model` asset and a `predictions` asset — does **not** apply: this task
  performs no training at all; it "reproduces" a prior ad hoc *inference* session with
  instrumentation added, which is evaluation/forensics, not code reproduction of a training
  procedure. `task.json`'s `expected_assets` is `{}` (empty), confirming no `model`/`predictions`
  asset was intended for this task. No task-type change is requested in this plan — noting the
  mismatch here satisfies the "recommend types" guidance without altering `task.json`.

## Cost Estimation

**Total estimated cost: $0.00.** This is a CPU-only local forensics task with no GPU provisioning
and no paid API calls:

* Remote compute: **$0** — no GPU machine is provisioned (see `## Remote Machines`). All work runs
  in the local worktree
  (`/home/azureuser/rail-metarepo/real-repos/rail-arf-tts-worktrees/t0013_v10_synthesis_quality_forensics`)
  on CPU.
* LLM/API calls: **$0** — no `anthropic_api`, `openai_api`, or `elevenlabs_api` usage is needed;
  synthesis uses the local StyleTTS2 checkpoints, not any paid TTS API.
* Network egress: **$0** — `git clone` of `github.com/semidark/kikiri-tts` (+ submodules) and the
  base pretrained StyleTTS2 LibriTTS checkpoint download are both free public downloads. `dvc pull`
  for the two v10 checkpoints (~2.09 GB each), `first_stage_v3.pth` (1.72 GB,
  `tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/stage1/first_stage.pth`), and the
  `11labs_david` reference corpus (`tasks/t0008_tts_eval_harness_baselines/data/11labs_david/`)
  pulls from the project's existing Azure Blob Storage remote (`azure://ml-dvc-datasets/...`),
  already paid for as part of project infrastructure, not a new task-attributable cost.
* Disk: local worktree has 26 GB available on `/` at planning time; the two v10 checkpoints (~4.2
  GB), `first_stage_v3.pth` (1.72 GB), the `11labs_david` corpus, and an isolated CPU venv with
  `torch==2.5.1` (~1-2 GB) fit comfortably; checkpoints can be deleted from local disk (re-pullable
  via DVC) after Milestone E if space becomes tight.

Compared against `project/budget.json` (`total_budget: $5000`, current spend $307.88 / 6.2%,
$4692.12 remaining, `stop_at_percent: 100`, `warn_at_percent: 80`): this task adds effectively $0
and does not move the project closer to either threshold.

## Step by Step

### Milestone A — Environment and data staging (steps 1-3)

1. **Stage the checkpoints and reference data via DVC.** From the worktree root
   (`/home/azureuser/rail-metarepo/real-repos/rail-arf-tts-worktrees/t0013_v10_synthesis_quality_forensics`),
   run (wrapped per project convention):
   `uv run python -m arf.scripts.utils.run_with_logs --task-id t0013_v10_synthesis_quality_forensics -- dvc pull tasks/t0010_stage2_safeguarded_training/data/run_v10/epoch_2nd_00016.pth.dvc tasks/t0010_stage2_safeguarded_training/data/run_v10/epoch_2nd_00014.pth.dvc tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/stage1/first_stage.pth.dvc tasks/t0008_tts_eval_harness_baselines/data/11labs_david.dvc`.
   Inputs: existing `.dvc` pointer files (already committed). Outputs: the four files/dirs
   materialize locally (`epoch_2nd_00016.pth` = 2,086,820,872 bytes, `epoch_2nd_00014.pth` =
   2,086,820,872 bytes, `first_stage.pth` = 1,724,784,307 bytes, `11labs_david/` with >= 1000 WAVs).
   Expected output: `dvc pull` reports 4 files fetched, no errors. Satisfies REQ-1, REQ-4, REQ-13
   (data prerequisite).

2. **Clone the inference harness source.** Create `code/kikiri-tts/` and clone:
   `uv run python -m arf.scripts.utils.run_with_logs --task-id t0013_v10_synthesis_quality_forensics -- git clone --depth 1 https://github.com/semidark/kikiri-tts.git tasks/t0013_v10_synthesis_quality_forensics/code/kikiri-tts`
   then
   `git -C tasks/t0013_v10_synthesis_quality_forensics/code/kikiri-tts submodule update --init --recursive`
   (also wrapped in `run_with_logs`). This pulls `StyleTTS2` (`github.com/semidark/StyleTTS2`) and
   `kokoro` (`github.com/semidark/kokoro`) as submodules, including
   `StyleTTS2/Utils/{ASR,JDC,PLBERT}/` pretrained weights (~134 MB, shipped in the repo, no separate
   download). Create `code/kikiri-tts/.gitignore`... actually create
   `tasks/t0013_v10_synthesis_quality_forensics/code/.gitignore` containing `kikiri-tts/` and
   `.venv-styletts2/` so the clone and the venv (step 7) are never committed. Expected output:
   `code/ kikiri-tts/StyleTTS2/Modules/hifigan.py` and
   `code/kikiri-tts/StyleTTS2/Modules/istftnet.py` exist. Satisfies REQ-8.

3. **Record path constants.** Create `code/paths.py` (pattern copied from
   `tasks/t0010_stage2_safeguarded_training/code/paths.py`, 26 lines) defining `TASK_ROOT`,
   `V10_EPOCH16_CKPT`, `V10_EPOCH14_CKPT` (pointing at
   `tasks/t0010_stage2_safeguarded_training/data/run_v10/epoch_2nd_0001{6,4}.pth`),
   `FIRST_STAGE_V3_CKPT` (pointing at
   `tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/stage1/first_stage.pth`),
   `ELEVENLABS_DAVID_DIR` (`tasks/t0008_tts_eval_harness_baselines/data/11labs_david/`),
   `KIKIRI_TTS_DIR` (`code/kikiri-tts/`), `RESULTS_AUDIO_DIR` (`results/audio_samples/`), and
   `RESULTS_DIR`. No REQ satisfied alone; used by every later script.

### Milestone B — Cheap checkpoint-tensor falsifier [CRITICAL] (steps 4-6)

This milestone runs **before** Milestone C and uses only the existing main project `.venv` (already
has `torch>=2.12.1`) — no new environment is built yet. This is the fast, cheap test the parent
research explicitly asked to run first.

4. **[CRITICAL] Determine ground-truth decoder key naming from source.** Without executing any
   StyleTTS2 code, read `code/kikiri-tts/StyleTTS2/Modules/hifigan.py` and
   `code/kikiri-tts/StyleTTS2/Modules/istftnet.py` (both files now present from step 2). Identify,
   for each file's `Generator`/decoder class `__init__`, every `self.<name> = ...` submodule
   assignment (expect patterns resembling `self.ups`, `self.resblocks`, `self.conv_pre`,
   `self.conv_post` for HiFi-GAN, and a structurally different set involving an inverse-STFT stage
   for ISTFTNet — record the **actual** names found, do not assume the exact `ups.*`/`resblocks.*`
   pattern from research without confirming it against this session's source checkout, since
   upstream naming can drift between StyleTTS2 forks). Write findings to
   `results/checkpoint_forensics.md` under a "Ground truth: decoder key naming by architecture"
   section. Satisfies part of REQ-1.

5. **[CRITICAL] Write and run `code/inspect_checkpoint.py`.** New script, run via the main project
   venv
   (`uv run python -m arf.scripts.utils.run_with_logs --task-id t0013_v10_synthesis_quality_forensics -- uv run python tasks/t0013_v10_synthesis_quality_forensics/code/inspect_checkpoint.py`).
   For each of `epoch_2nd_00016.pth`, `epoch_2nd_00014.pth`, and `first_stage.pth` (the control):
   call `torch.load(path, map_location="cpu", weights_only=False)`, then for every top-level module
   key in `state["net"]` (expect the 13 modules named in `research/research_summary.md` point 4:
   `bert`, `bert_encoder`, `predictor`, `decoder`, `text_encoder`, `predictor_encoder`,
   `style_encoder`, `diffusion`, plus `text_aligner`, `pitch_extractor`, `mpd`, `msd`, `wd` if
   present) print: (a) the full sorted list of state-dict key names for `decoder` specifically,
   compared against the ground truth from step 4 to classify as HiFi-GAN-shaped or ISTFTNet-shaped;
   (b) `torch.isfinite(tensor) .all()` for every tensor in every module — flag any module with NaN
   or Inf; (c) per-module weight-norm:
   `sum(t.float().norm().item() ** 2 for t in module_state_dict.values()) ** 0.5`. Write all results
   as structured JSON to `results/checkpoint_forensics_raw.json` (module name -> {finite: bool,
   weight_norm: float, num_params: int}) and a human-readable table to
   `results/checkpoint_forensics.md`. Expected output: the script runs to completion in well under a
   minute (checkpoints are already local from step 1; no GPU, no model instantiation, just
   `torch.load` + dict inspection). Satisfies REQ-1, REQ-5.

6. **[CRITICAL] Interpret the falsifier result.** In `results/checkpoint_forensics.md`, add a
   "Verdict" section stating explicitly one of: (a) `epoch_2nd_00016.pth`'s `decoder` key names
   match the HiFi-GAN pattern from step 4 while `first_stage.pth`'s `decoder` key names match the
   ISTFTNet pattern — **architecture-mismatch hypothesis confirmed at the tensor level**; (b) the
   key names match the same architecture in both files — **hypothesis refuted, investigate other
   explanations** (NaN/Inf, weight-norm collapse, or a harness bug found later in Milestone C-E);
   (c) NaN/Inf found in any module — record which, regardless of (a)/(b). This verdict is a working
   hypothesis to carry into Milestone E's final diagnosis, not the task's final answer — Milestones
   C-E still run regardless, per the Approach section. Satisfies REQ-1, REQ-5, REQ-13
   (falsifier-first ordering).

### Milestone C — Build the instrumented StyleTTS2-native inference harness [CRITICAL] (steps 7-9)

7. **[CRITICAL] Build an isolated CPU venv.** Do **not** modify the main project's `pyproject.toml`
   or `uv.lock` (main project pins `torch>=2.12.1`; StyleTTS2's `models.py` and `utils.py` call
   `torch.load` without `weights_only=` flags, which breaks under torch >= 2.6's new default — pin
   `torch==2.5.1` instead of patching call sites, since pinning is the smaller diff). Run (each
   wrapped in `run_with_logs`):
   `uv venv tasks/t0013_v10_synthesis_quality_forensics/code/.venv-styletts2 --python 3.10`, then
   `uv pip install --python tasks/t0013_v10_synthesis_quality_forensics/code/.venv-styletts2/bin/python torch==2.5.1 torchaudio==2.5.1 soundfile munch pydub pyyaml librosa nltk matplotlib accelerate transformers einops einops-exts tqdm typing-extensions phonemizer "resemblyzer>=0.1.4"`.
   Then install the aligner dependency:
   `uv pip install --python tasks/t0013_v10_synthesis_quality_forensics/code/.venv-styletts2/bin/python git+https://github.com/resemble-ai/monotonic_align.git`.
   If this build fails (it requires a C/Cython toolchain), install `build-essential` first
   (`sudo apt-get install -y build-essential`) and retry once; if it still fails, create a stub
   Python module named `monotonic_align` in the venv's site-packages exposing a no-op
   `maximum_path(...)` function — `StyleTTS2/utils.py` imports this module at load time but it is
   unused at inference time (confirmed in `research/research_code.md`), so a stub is sufficient to
   unblock imports without affecting synthesis output. Install the phonemizer backend:
   `sudo apt-get install -y espeak-ng` (see the Ambiguity Note in the Task Requirement Checklist for
   why this replaces `task_description.md`'s `brew install` instruction). Expected output:
   `uv run --python tasks/t0013_v10_synthesis_quality_forensics/code/.venv-styletts2/bin/python -c "import torch; print(torch.__version__)"`
   prints `2.5.1`. No REQ alone; enables steps 8-14.

8. **[CRITICAL] Write the instrumented inference script.** Create `code/infer_styletts2.py`, adapted
   from `code/kikiri-tts/StyleTTS2/Demo/Inference_LibriTTS.ipynb` (convert notebook cells to a
   script with `--checkpoint-path`, `--text`, `--reference-audio`, `--output-wav`, `--config-path`
   CLI args via `argparse`). Copy the DP-aware shape-matching structure from
   `tasks/t0010_stage2_safeguarded_training/code/train_second_v10.py:87-118` (`load_checkpoint`) as
   a reference for how this project's own loader strips `module.` prefixes and matches by key+shape,
   but **do not reuse its pass condition**: change the per-module success condition from
   "`len(matched) == 0` raises" to "log `missing = len(model_sd) - len(matched)` and
   `unexpected = len(ckpt_sd) - len(matched)` per module explicitly, treat any nonzero `missing` or
   `unexpected` on `bert`, `bert_encoder`, `predictor`, `decoder`, `text_encoder`,
   `predictor_encoder`, `style_encoder`, or `diffusion` as a **hard failure** (raise, do not
   silently continue) — per `research/research_summary.md` point 4, this is the same
   `strict=False`/aggregate-only anti-pattern that has already caused two prior undiagnosed bugs in
   this project (t0009's training loader, t0008's `adapters.load_kokoro_model_with_checkpoint`).
   Print, per module: parameter count, missing count, unexpected count, and whether the primary or
   the `module.`-stripped fallback path was used, to both stdout and a JSON log file
   `results/load_log_<checkpoint_stem>.json`. This directly answers Key Question 1 (REQ-2).
   Satisfies REQ-2, REQ-8.

9. **[CRITICAL] Handle the short reference-clip crash (Key Question 6).** While building step 8's
   script, use a reference audio clip of at least several seconds for `compute_style` (e.g. one from
   `code/kikiri-tts/StyleTTS2/Demo/reference_audio/` for the control, and a longer clip from
   `tasks/t0008_tts_eval_harness_baselines/data/11labs_david/` — not a short filler clip under ~2s —
   for the v10 runs). Document in `results/v10_diagnosis.md` (populated fully in step 14, but note
   it here as it is discovered during harness construction) whether this is StyleTTS2's documented/
   expected behavior (its conv-kernel-based style encoder architecturally requires a minimum input
   length — check `code/kikiri-tts/StyleTTS2/models.py`'s `StyleEncoder`/`compute_style` for an
   explicit kernel-size-driven minimum) or a symptom unrelated to the noise problem. Satisfies
   REQ-7.

### Milestone D — Control validation gate [CRITICAL] (step 10)

10. **[CRITICAL] Run the harness against the base pretrained control checkpoint — validation gate.**
    Download the base pretrained StyleTTS2 LibriTTS checkpoint per the download instructions
    embedded in `code/kikiri-tts/StyleTTS2/Demo/Inference_LibriTTS.ipynb`'s own cells (destination
    `code/kikiri-tts/StyleTTS2/Models/LibriTTS/epochs_2nd_00020.pth`; this file is a large
    third-party download, gitignored via step 2's `code/.gitignore`, not DVC-tracked since it is
    freely re-downloadable from upstream). Run:
    `uv run python -m arf.scripts.utils.run_with_logs --task-id t0013_v10_synthesis_quality_forensics -- tasks/t0013_v10_synthesis_quality_forensics/code/.venv-styletts2/bin/python tasks/t0013_v10_synthesis_quality_forensics/code/infer_styletts2.py --checkpoint-path code/kikiri-tts/StyleTTS2/Models/LibriTTS/epochs_2nd_00020.pth --text "This is a test of the Style T T S two inference harness." --reference-audio code/kikiri-tts/StyleTTS2/Demo/reference_audio/<a_shipped_reference_clip>.wav --output-wav results/audio_samples/control_epochs_2nd_00020.wav`.
    **Validation gate — trivial baseline: silence/pure-noise.** After the run, compute RMS, peak
    amplitude, duration, and silence fraction (fraction of 20ms frames below -40dBFS) of the output
    WAV using `librosa`/`soundfile` (baseline: a silent or pure-noise clip has silence_fraction
    close to 1.0 or near-uniform high-frequency energy with no formant structure; an
    intelligible-speech clip has silence_fraction well below 0.5 and concentrated low/mid-frequency
    energy). **Failure condition: if the control output is silent, clipped to near-zero duration, or
    fails the same noise heuristic built in step 16, STOP — do not proceed to v10 (steps 11-14). The
    bug is in the harness/environment (torch version, phonemizer setup, sampler parameters), not in
    any project checkpoint. Debug the harness against the control and re-run this step until it
    passes**, per `task_description.md` Scope §2 verbatim. Write `results/control_test.md` with the
    load-log summary from step 8's instrumentation, the pass/fail verdict, and the
    RMS/peak/duration/silence stats as the objective substitute for "I listened and it sounded like
    X." Individual-output inspection: read the printed missing/unexpected key counts for all modules
    and confirm 0/0 for every module before trusting the audio stats. Satisfies REQ-3, REQ-9, REQ-2
    (control side).

### Milestone E — v10 diagnosis [CRITICAL] (steps 11-14)

11. **[CRITICAL] Gate check.** Confirm step 10 passed (control produced audio that fails the
    silence/noise heuristic used as the "sounds like noise" baseline, i.e. sounds like speech). If
    step 10 did not pass, do not execute steps 12-14 — see `## Rejection Criteria`.

12. **[CRITICAL] Run the harness against both v10 checkpoints.** Two runs, same text prompt and
    reference audio as step 10 for comparability:
    `.../code/.venv-styletts2/bin/python code/infer_styletts2.py --checkpoint-path tasks/t0010_stage2_safeguarded_training/data/run_v10/epoch_2nd_00016.pth --text "This is a test of the Style T T S two inference harness." --reference-audio <same clip as step 10> --output-wav results/audio_samples/v10_epoch16_primary.wav`
    and the same with `epoch_2nd_00014.pth` -> `results/audio_samples/v10_epoch14_backup.wav`. Both
    wrapped in `run_with_logs`. Capture each run's `results/load_log_epoch_2nd_0001{6,4}.json` (from
    step 8's instrumentation). Expected output: two WAV files plus two load-log JSON files.
    Satisfies REQ-4, REQ-9.

13. **[CRITICAL] Compute objective audio stats and compare to Milestone B's tensor verdict.** For
    both v10 outputs, compute the same RMS/peak/duration/silence-fraction stats as step 10. Cross-
    reference against `results/checkpoint_forensics.md`'s per-checkpoint verdict from step 6: if
    both v10 checkpoints load with 0/0 missing/unexpected on every module (step 12's load logs) and
    both still produce noise by the audio-stats heuristic, the defect is in training/architecture,
    not loading — consistent with (or inconsistent with, report either way) Milestone B's
    tensor-level verdict. If `epoch_2nd_00016.pth` and `epoch_2nd_00014.pth` differ sharply in
    weight-norm (Milestone B) or in audio-stats pass/fail (this step), that points at a
    late-training-specific event between epoch 15 and epoch 17. Satisfies REQ-4, REQ-5.

14. **[CRITICAL] Cross-reference training logs and write the final diagnosis.** Read
    `tasks/t0010_stage2_safeguarded_training/data/run_v10/metrics.jsonl` and `per_epoch_summary.csv`
    directly (both already present locally, not DVC-tracked, per the file listing observed during
    planning). Per `research/research_code.md`'s existing finding, expect every record to have
    `"step": 0` (validation-epoch-only) with `disc_loss`, `grad_norm_*`, `loss_total`, `skip_count`,
    `lr` null throughout, and `acoustic_norm` smooth (6.4-8.5) across epochs 9-17 with no divergence
    spike — confirm this still holds for this task's own read of the file (do not just cite the
    prior research finding without re-checking), and state explicitly that no new per-step signal is
    available beyond what research already found (Key Question 5). Write `results/v10_diagnosis.md`
    synthesizing steps 4-6 (tensor verdict), 8-9 (load-log key counts), 12-13 (audio evidence), and
    this step's log cross-reference into one of three explicit verdicts, per `task_description.md`
    Scope §4:
    * **Reproduction bug**: harness/loader was wrong; state the fix and confirm re-running produces
      correct-sounding audio; explicitly flag that the earlier ad hoc audio comparison delivered to
      the user this session was invalid.
    * **Real defect isolated to late epochs**: recommend using `epoch_2nd_00014.pth` (or an earlier
      checkpoint) instead of the "best" pick, or recommend a retraining run with the root cause
      (e.g. the `decoder` exclusion bug in `train_second_v10.py`'s `ignore_modules`) fixed.
    * **Real defect from the start**: state plainly that `config_david_v10.yml`'s training run never
      produced working audio at any epoch, and that `val_loss` alone is insufficient evidence of
      training success — a synthesis smoke check must run before a training task can claim
      `completed`. Whichever verdict applies, state the recommended action explicitly (retrain with
      a corrected config, fix `train_second_v10.py:load_checkpoint()`'s `ignore_modules` list, or
      discard v10). Satisfies REQ-6, REQ-10, REQ-11.

### Milestone F — Regression check and metrics (steps 15-17)

15. **DVC-track the audio deliverables.** Per `CLAUDE.md`'s DVC workflow rule ("any task that
    produces audio files ... MUST gitignore the data and track it with `dvc add`"), run:
    `uv run python -m arf.scripts.utils.run_with_logs --task-id t0013_v10_synthesis_quality_forensics -- dvc add tasks/t0013_v10_synthesis_quality_forensics/results/audio_samples`.
    This creates `results/audio_samples.dvc` and a `.gitignore` entry for the raw WAVs; the `.dvc`
    pointer file is what gets committed to git. Run `dvc push` for this path before the task PR is
    opened so teammates can `dvc pull` the audio. Satisfies REQ-9.

16. **Write the automatable noise-detection regression check.** Create
    `code/audio_quality_check.py`: a small CLI/importable script exposing
    `check_audio_quality(wav_path: Path) -> AudioQualityResult` where `AudioQualityResult` is a
    `@dataclass(frozen=True, slots=True)` with `rms: float`, `peak: float`,
    `silence_fraction: float`, `spectral_flatness: float`, and `is_likely_noise: bool` (the same
    heuristic used as the pass/fail gate in steps 10 and 13, now generalized into a reusable
    function: high `spectral_flatness` (close to 1.0, indicating energy spread evenly across
    frequencies like white noise, vs. speech's concentrated formant structure) combined with low
    `silence_fraction` flags "sounds like noise, not silence, not speech"). Use
    `librosa.feature.spectral_flatness` and `librosa.feature.rms`. Per the ponytail convention
    already used elsewhere in this project's code (e.g. `train_second_v10.py`'s `# ponytail:`
    comment), add a `demo()` / `if __name__ == "__main__":` self-check block with `assert`
    statements that runs `check_audio_quality` on one of this task's own known-noise v10 outputs and
    one of the known-good control outputs, asserting the heuristic classifies them correctly — this
    is the "one runnable check" for this script's non-trivial branch logic. Add a module docstring
    describing this script as a reusable regression check for the project's eval-harness pipeline,
    so a later orchestrator-managed reporting step can point to it as a follow-up. Satisfies REQ-12.

17. **Measure and record applicable registered metrics.** Read
    `tasks/t0013_v10_synthesis_quality_forensics/ctx/metrics.json` (already fetched during planning:
    registered metrics are `rtf`, `speaker_sim`, `ttfb_ms`).
    * `rtf` (`synthesis_wall_time_seconds / output_audio_duration_seconds`, lower is better,
      registered target "< 0.2 for Kokoro on LLM-T1-NC80 (H100)" per `overview/metrics/rtf.md`):
      **applicable and measured.** Wrap each of `infer_styletts2.py`'s synthesis calls in steps 10
      and 12 with `time.perf_counter()` around the model forward + vocoder call only (exclude
      checkpoint loading and file I/O), divide by the output WAV's duration. Record per-checkpoint.
      This measurement is on CPU, not the registered target's H100 — state explicitly in
      `results/metrics.json` and `results/v10_diagnosis.md` that this is a corroborating
      same-hardware-across-variants comparison (control vs. v10, all on the same CPU), not a claim
      against the H100 target.
    * `speaker_sim` (GE2E cosine vs. `11labs_david` mean embedding, higher is better, registered
      target "≥ 0.85 for fine-tuned Kokoro v4 on val_96"): **applicable and measured**, as a
      corroborating diagnostic signal (does v10's output even resemble David's voice, vs. being pure
      noise) — not a claim against the val_96-specific 0.85 target, since this task's 1-2 test clips
      are not val_96. Reuse the centroid-building and cosine-scoring pattern from
      `tasks/t0008_tts_eval_harness_baselines/code/score_speaker_sim.py` (do not import — copy and
      adapt the `VoiceEncoder`/`preprocess_wav`/centroid logic into `code/score_speaker_sim.py`,
      since the source script's per-clip JSON record schema from `run_eval.py` is not present here):
      build a mean GE2E centroid from `tasks/t0008_tts_eval_harness_baselines/data/11labs_david/`
      (>= 1000 WAVs expected, same >= 1000 assertion as the source script), then compute cosine
      similarity for `results/audio_samples/control_epochs_2nd_00020.wav`,
      `v10_epoch16_primary.wav`, and `v10_epoch14_backup.wav` against that centroid using
      `resemblyzer` (already installed into `code/.venv-styletts2` in step 7, per
      `overview/metrics/speaker_sim.md`'s instruction to keep it out of main dependencies).
    * `ttfb_ms` (time to first audio byte, lower is better): **explicitly not measured** — this
      task's harness is an offline batch script that writes a complete WAV file per invocation;
      there is no streaming/production serving endpoint in scope, so "time to first audio byte" is
      not a meaningful quantity here. This is a deliberate omission, not an oversight. Write
      `results/metrics.json` using the **explicit multi-variant format**
      (`arf/specifications/metrics_specification.md`), one variant per checkpoint tested:
      `variant_id: "control-epochs-2nd-00020"`, `"v10-epoch16-primary"`, `"v10-epoch14-backup"`,
      each with `dimensions: {checkpoint: "<filename>", decoder_type: "istftnet"|"hifigan"}` and
      `metrics: {rtf: <float>, speaker_sim: <float>}` (use `null` for either value if that
      checkpoint's run failed to produce a scoreable WAV, per the Rejection Criteria below — do not
      fabricate a number for a failed run). Satisfies REQ-13.

## Remote Machines

**None required.** This is an explicitly CPU-only local forensics task (`task_description.md` Scope
§1: "Build a CPU venv"). All checkpoint inspection, harness construction, and inference runs execute
in the local worktree
(`/home/azureuser/rail-metarepo/real-repos/rail-arf-tts-worktrees/t0013_v10_synthesis_quality_forensics`)
on the machine already running this task. No Azure ML VM (`LLM-T1-NC80` or otherwise) is
provisioned, so `setup-machines` and `teardown` were already correctly skipped per `checkpoint.md`'s
Step 8/10 notes.

## Assets Needed

* `epoch_2nd_00016.pth` and `epoch_2nd_00014.pth` — from dependency task
  `t0010_stage2_safeguarded_training`, DVC-tracked at
  `tasks/t0010_stage2_safeguarded_training/data/run_v10/epoch_2nd_0001{6,4}.pth.dvc`.
* `first_stage.pth` (a.k.a. `first_stage_v3.pth`) — DVC-tracked at
  `tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/stage1/first_stage.pth.dvc`, from prior
  task `t0006_kokoro_v5_stage2_subset` (not a formal task dependency of `t0013`, but a required
  read-only reference input for Milestone B's tensor comparison; reading it does not violate the
  don't-modify-other-task-folders rule).
* `11labs_david` reference corpus — DVC-tracked at
  `tasks/t0008_tts_eval_harness_baselines/data/11labs_david.dvc`, from prior task
  `t0008_tts_eval_harness_baselines` (read-only reference input for step 17's `speaker_sim`).
* `kikiri-tts` GitHub repository (`github.com/semidark/kikiri-tts`) and its `StyleTTS2`/`kokoro`
  submodules — external, cloned fresh in step 2.
* Base pretrained StyleTTS2 LibriTTS checkpoint — external download, URL embedded in
  `StyleTTS2/Demo/Inference_LibriTTS.ipynb`'s own cells, fetched in step 10.

## Expected Assets

`task.json`'s `expected_assets` field is `{}` (empty) — no formal ARF asset-type registration
(`meta/asset_types/{dataset,library,model,predictions,paper,answer}/`) is required for this task,
and none is planned. This task's tangible outputs are `code/` artifacts and `results/` files,
exactly as enumerated in `task_description.md`'s "Expected Outputs" section:

* `code/infer_styletts2.py`, `code/inspect_checkpoint.py`, `code/score_speaker_sim.py`,
  `code/audio_quality_check.py`, `code/paths.py` — committed, documented Python scripts (the
  `code/kikiri-tts/` clone and `code/.venv-styletts2/` are gitignored, not committed).
* `results/audio_samples/*.wav` (control, v10 primary, v10 backup) — DVC-tracked per step 15.
* `results/checkpoint_forensics.md`, `results/checkpoint_forensics_raw.json` — Milestone B output.
* `results/control_test.md` — Milestone D output.
* `results/v10_diagnosis.md` — Milestone E output, the task's core deliverable.
* `results/metrics.json` — explicit multi-variant `rtf`/`speaker_sim` measurements.
* `results/load_log_*.json` — per-checkpoint missing/unexpected key logs from the instrumented
  loader.

## Time Estimation

* Research (already done): ~2-3 hours (completed in step 6, `research-code`).
* Planning (this step): ~1 hour.
* Milestone A (data staging, clone): ~15-30 min (dominated by ~7 GB of `dvc pull` transfer).
* Milestone B (cheap falsifier): ~15-30 min — the fast path this plan front-loads deliberately.
* Milestone C (venv + harness build): ~45-90 min — dominated by `torch==2.5.1` install and
  `monotonic_align` compilation, with a documented stub fallback if compilation fails.
* Milestone D (control validation gate): ~15-20 min, plus a possible debug loop if the control fails
  (unbounded in the worst case — see Risks).
* Milestone E (v10 diagnosis): ~20-30 min (two inference runs + write-up).
* Milestone F (regression check + metrics): ~30-45 min.
* **Total estimated implementation time: ~3-4.5 hours**, assuming the control gate passes on the
  first or second attempt.

## Risks & Fallbacks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| `monotonic_align` fails to compile from source (missing C/Cython toolchain) in the isolated venv | Medium | Blocks Milestone C entirely | Install `build-essential` first and retry once (step 7); if still failing, install a no-op stub module (confirmed unused at inference time per `research/research_code.md`) — do not let a build failure block the harness |
| Control checkpoint (step 10) also produces noise/garbled audio | Medium | Stops Milestone E from running at all; would otherwise misattribute a harness bug to the checkpoint | This is the explicit validation gate (step 10-11): STOP, debug the harness/environment against the control, do not touch v10 until the control passes. If the harness cannot be made to work on the control after reasonable debugging, create an `intervention/` file describing the blocker rather than silently concluding anything about v10 |
| Base pretrained StyleTTS2 LibriTTS checkpoint download URL (embedded in the notebook) is stale, gated, or blocked by network egress restrictions in this environment | Low-Medium | Blocks the control test (step 10), which gates all of Milestone E | Retry with an alternate mirror if the notebook lists one; if genuinely unreachable, create an `intervention/` file requesting the checkpoint be staged manually — do not substitute a different, unvalidated control |
| `torch==2.5.1` conflicts with something already cached/pinned globally by `uv`, or CPU wheel is unavailable for the local platform | Low | Blocks Milestone C | Use an isolated `uv venv` (not the main project env) exactly as specified in step 7; if the CPU wheel is unavailable, fall back to the nearest compatible 2.5.x patch release and document the substitution |
| The cheap tensor falsifier (Milestone B) and the full-harness diagnosis (Milestone E) disagree (e.g. key names look architecture-matched but audio is still noise) | Low-Medium | Diagnosis less clear-cut than hoped | Not a blocker — report both findings honestly in `results/v10_diagnosis.md` rather than forcing a single tidy verdict; a genuinely ambiguous result (e.g. loader is clean but audio is still noise) is itself a valid, reportable outcome pointing at a different root cause (e.g. GAN divergence not visible in `val_loss`) |
| `dvc pull` for the ~7 GB of checkpoints/corpus is slow or hits a transient Azure Blob Storage error | Low | Delays Milestone A | Re-run `dvc pull` (idempotent); it resumes/re-verifies rather than re-downloading already-valid local files |

## Verification Criteria

* Run
  `uv run python -m arf.scripts.utils.run_with_logs --task-id t0013_v10_synthesis_quality_forensics -- uv run python -m arf.scripts.verificators.verify_plan t0013_v10_synthesis_quality_forensics`
  and confirm it reports 0 errors (this plan itself).
* After implementation, run
  `uv run python -m arf.scripts.verificators.verify_task_results t0013_v10_synthesis_quality_forensics`
  and confirm 0 errors — checks that `results/results_summary.md`, `results/results_detailed.md`
  exist and are well-formed once the orchestrator's reporting steps run.
* After implementation, run
  `uv run python -m arf.scripts.verificators.verify_task_metrics t0013_v10_synthesis_quality_forensics`
  and confirm 0 errors — checks `results/metrics.json` uses only registered metric keys (`rtf`,
  `speaker_sim`) in valid explicit-variant format.
* Confirm REQ-1 through REQ-13 coverage:
  `grep -c "REQ-" tasks/t0013_v10_synthesis_quality_forensics/plan/plan.md` should show every
  `REQ-*` ID from the Task Requirement Checklist appearing at least twice (once in the checklist,
  once in a Step by Step entry) — a manual read-through confirming no `REQ-*` ID is orphaned
  (appears only once).
* File existence:
  `test -f tasks/t0013_v10_synthesis_quality_forensics/results/v10_diagnosis.md && test -f tasks/t0013_v10_synthesis_quality_forensics/results/control_test.md && ls tasks/t0013_v10_synthesis_quality_forensics/results/audio_samples/*.wav`
  — all must succeed (nonzero exit from any of these means an Expected Output is missing).
* `results/v10_diagnosis.md` must contain one of the three literal phrases "reproduction bug", "real
  defect isolated to late epochs", or "real defect from the start" (or an equally unambiguous
  paraphrase) —
  `grep -iE "reproduction bug|isolated to late epochs|defect from the start" tasks/t0013_v10_synthesis_quality_forensics/results/v10_diagnosis.md`
  must match at least once, confirming a definitive verdict was actually reached rather than left
  open-ended.

## Rejection Criteria

Pre-registered before implementation begins, so these cannot be loosened after seeing results:

* **If the control test (step 10) does not pass the noise/silence heuristic** (Milestone D's
  validation gate), any `speaker_sim`/`rtf` numbers computed for v10 checkpoints in that same
  session are **null and non-diagnostic** — write `null` for those metric values in
  `results/metrics.json` and state plainly in `results/v10_diagnosis.md` that the harness itself was
  not validated, so no claim about v10's checkpoint quality can be made from that run. This directly
  operationalizes `task_description.md` Scope §2's "if this produces garbled noise too, the bug is
  in the harness/ environment... before ever touching v10 again."
  * `results/checkpoint_forensics.md` (also `_raw.json`) must state which checkpoints were dropped
    and why (e.g. an unreadable/corrupt `.pth` after `dvc pull`, or a `torch.load` exception),
    per-module — do not average over a partial checkpoint read.
* **If `dvc pull` (step 1) reports an MD5 hash mismatch on any checkpoint**, treat that specific
  checkpoint's forensics as null (do not proceed to run inference on a possibly-corrupt file) —
  retry the pull once, and if the mismatch persists, file an `intervention/` request rather than
  proceeding on unverified bytes.
* This task does not run a >= N-request benchmark protocol (see the `tts-benchmark-run`
  applicability note in `## Approach`), so Lesson 3's `successful_requests / total_requests < 0.8`
  threshold does not apply in its literal form; the equivalent adaptation for this task's scale (a
  handful of forensic inference runs, not a 100+ request benchmark) is the control-gate rule above.
