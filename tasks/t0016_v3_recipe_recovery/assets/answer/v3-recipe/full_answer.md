---
spec_version: "2"
answer_id: "v3-recipe"
answered_by_task: "t0016_v3_recipe_recovery"
date_answered: "2026-09-17"
confidence: "low"
---
## Question

What exactly did Kokoro v3's Stage 2 training recipe consist of, and which parts of that are
confirmed versus inferred?

## Short Answer

v3 fine-tuned StyleTTS2 Stage 2 from `first_stage_v3.pth` (confirmed by name only; byte-identity
unrecoverable) for at least 10 epochs (confirmed from the epoch-numbered audio samples, epochs 0-9),
using the ISTFTNet decoder architecture and `max_dur=50`/`style_dim=128`/`n_token=178` (confirmed
from the shipped bundle's `config.json`), with `multispeaker: false` (inferred from checkpoint-shape
forensics on the Stage 1 diffusion module, resolving the standing three-way contradiction in favor
of `false`), `lambda_slm: 0.0` (inferred from the surviving `train_second_patch.diff`), and all five
bundle modules (`bert`, `bert_encoder`, `predictor`, `text_encoder`, `decoder`) showing substantial
weight-norm shifts from Stage 1, meaning the decoder itself changed during Stage 2 rather than the
speaker-similarity gain coming from `predictor`/ `text_encoder` alone. The 266-clip training list,
the exact `joint_epoch`/`lambda_gen`/`lr`, and byte-identity to the checkpoint v6c/v6d also loaded
could not be recovered: the VM's home directory turned out to be a later task's (t0014's) StyleTTS2
clone, not a preserved v3-era environment, so no literal launch config or clip list survived. This
is a low-confidence reconstruction by design — most fields are `inferred` or `unknown`, honestly
labelled, not `confirmed`.

## Research Process

Per `task_description.md`'s prescribed evidence-priority order: (1) a bounded, 90-minute, read-only
SSH inventory of the project's sole Azure ML pool VM (`LLM-T1-NC80`), searching
`~/kokoro-finetune/`, `/mnt/cache/persist/` (the one Azure Files share proven to survive VM
stop/start), and the shared account's `~/.bash_history`; (2) no-GPU checkpoint-shape forensics
comparing `stage1/first_stage.pth` against `best/david_v3_best_decoder_kokoro.pth` across all five
Kokoro-API bundle modules plus every other top-level key in the Stage 1 checkpoint's `net` dict; (3)
an audio-quality gate over every recoverable per-epoch sample; (4) reconstructing an annotated
config in the exact schema of the closest surviving relative
(`tasks/t0006_kokoro_v5_stage2_subset/code/config_david_v6c_stage2.yml`), diffing it against that
template and against t0009's recommended next-run config, and cross-checking every claim against its
cited primary source rather than trusting secondary tables at face value — this last step surfaced a
factual error in `t0009`'s own `results/confound_table.md` (see Evidence from Code or Experiments).
The VM inventory used 13.47 minutes of its 90-minute budget (well under the hard cap) before an
unconditional teardown; all further work ran on local CPU.

## Evidence from Papers

The papers method was not used. This task reconstructs an internal, project-specific training recipe
from first-party checkpoints, VM filesystem state, and prior-task code and documents, not from
published literature — no paper in this project's corpus addresses recovering a lost training
configuration from surviving artifacts, so none was consulted or cited here.

## Evidence from Internet Sources

The internet method was not used — `task_description.md`'s evidence sources are entirely internal
(VM home directory, DVC-tracked reference artifacts, prior tasks' checkpoints and documents); no
external internet research bore on this reconstruction.

## Evidence from Code or Experiments

* **VM inventory** (`data/vm_inventory/inventory.json`,
  `data/vm_inventory/kokoro_finetune_current/`): `~/kokoro-finetune` on `LLM-T1-NC80` is a symlink
  to `/mnt/cache/persist/t0014_v11_decoder_fix_retrain/kokoro-finetune` — t0014's own later
  StyleTTS2 clone for the v11 retrain (2026-09-16), not a preserved v3-era environment. No
  `first_stage_v3.pth`, no v3-named `logs/` directory, and no 266-clip list exist anywhere on the
  VM's home directory or its `/mnt/cache/persist` share (which does hold other tasks' artifacts —
  t0009, t0010, t0014 — plus several directories from an unrelated LLM fine-tuning project sharing
  this VM pool, none v3-related). `models.py` (StyleTTS2's `build_model()` source) DID survive and
  was copied; grepping it located the exact `multispeaker` branch (`models.py:808`) and confirmed
  the flag only affects `nets.diffusion`'s internal transformer choice (`StyleTransformer1d` vs.
  `Transformer1d`) — a top-level checkpoint key outside the five Kokoro-API bundle modules, but
  still inspectable.
* **Checkpoint-shape forensics** (`code/checkpoint_forensics.py`,
  `results/v3_module_weight_delta.raw.json`, `results/v3_checkpoint_forensics.md`): all five bundle
  modules show weight-norm changes well above the 5% near-zero-shift threshold from Stage 1 to best
  — `bert` -6.6%, `bert_encoder` +253.7%, `predictor` +167.2%, `text_encoder` +20.2%, `decoder`
  +225.0% — meaning Stage 2 updated every bundled module including the decoder, not a frozen subset.
  Inspecting `net["diffusion"]`'s tensor shapes directly (a top-level checkpoint key outside the
  five bundle modules) found exactly one cross-attention conditioning pathway per transformer block,
  with no second, distinctly-shaped pathway that `StyleTransformer1d`'s extra `context_features`
  argument would require — combined with `models.py`'s located branch, this yields
  `multispeaker: inferred false`, resolving the standing three-way contradiction between
  `best/config.json` (`true`), t0009's confound table (`true`, assumed), and `t0006`'s
  `config_david_v6c_stage2.yml` (`false`, inline-commented, no hash) in favor of the latter — now
  independently corroborated rather than merely repeated.
* **Cross-checking a secondary source against its own cited primary caught a real error**: while
  diffing the reconstructed config against v6c, this task read `config_david_v6c_stage2.yml`'s
  actual `lambda_gen` value (`0.2`, line 27) rather than trusting
  `tasks/t0009_stage2_training_failure_forensics/results/confound_table.md`'s claim that v6c's
  `lambda_gen` was `1.0`. The confound table is demonstrably wrong about v6c's own row — which
  correspondingly weakens confidence in that same table's separate, unhedged-looking claim that v3's
  `lambda_gen` was "1.0 (assumed)". `data/config_david_v3_reconstructed.yml` carries v6c's real
  value (`0.2`) forward as the least-bad default instead of propagating the confound table's number.
* **Audio-quality gate** (`code/audio_quality_check.py` copied from
  `tasks/t0015_v11_duration_blowup_forensics/code/audio_quality_check.py`,
  `code/score_per_epoch_samples.py`): scored all 55 recoverable per-epoch samples
  (`epoch{0..3}_phrase{1..5}.wav`, `audio/v3/ep{6,7,8}_p{1..5}.wav`,
  `audio/v3b/ep{6,7,8,9}_p{1..5}.wav`). All 20 `epoch0-3` samples and all 20 `v3b` samples pass
  (`is_likely_noise=False`); all 15 `v3/ep{6,7,8}` samples are flagged `is_likely_noise=True` —
  consistent with `t0002`'s own note that `v3b/ep9_p5.wav` is the "clean" reference sample, implying
  `v3b` supersedes a broken `v3` variant at the same epoch range.
* **Human-listenable audio**
  (`results/audio_samples/{v3_shipped,v3_per_epoch,elevenlabs_reference}/`,
  `results/listening_guide.md`): the shipped bundle was re-synthesized on CPU through
  `kokoro.KModel`, using a locally-adapted copy of `tasks/t0003_kokoro_v5_phoneme_data`'s
  `build_pipeline()`/lexicon (not a registered library, so copied rather than imported per Critical
  Rule 8), on the 3 fixed gate texts, plus 5 seed-42 val96 prompts. All 8 clips pass the audio gate.
  Locally-measured `speaker_sim=0.566` is below t0008's own recorded `0.631` (fillers) outside a
  ±0.02 tolerance, but this used a different (smaller, mixed) clip set and a locally-rebuilt
  reference centroid (t0008's own `build_centroid()` could not be reused unmodified: the current
  `data/11labs_david/` corpus's clips average 1.04s, under `scoring.py`'s own 1.6s minimum-length
  filter, causing it to reject all 1364 clips — documented in `results/v3_checkpoint_forensics.md`'s
  Metrics Cross-Check as a reproducibility finding in its own right, not silently routed around).
* **266-clip list**: not recoverable from any source (`data/v3_train_list_UNRECOVERED.md` documents
  the full search). No project task in this codebase's 15-task history has ever recovered or
  referenced v3's own list.
* **Packaging recipe** (read-only reference, not re-run):
  `tasks/t0002_kokoro_v4_voicepack_decoder_package/code/extract_decoder_generic.py`'s
  `convert_key()` is the exact five-module extraction/key-remapping contract that produced
  `best/david_v3_best_decoder_kokoro.pth` from a raw checkpoint — any future reproduction (t0017,
  out of this task's scope) must package its own output through this same unchanged contract to be
  loadable by `kokoro.KModel` the way v3's bundle is.

## Synthesis

v3's architecture (ISTFTNet decoder, `max_dur=50`, `style_dim=128`, `n_token=178`) is solidly
confirmed from the shipped bundle's own `config.json`. Its two training-loop patches
(`lambda_slm > 0` guard, `monotonic_align` `sys.path` fix) are confirmed from the surviving diff.
Everything else in the Stage 2 hyperparameter surface — `joint_epoch`, `lr`, `batch_size`,
`train_LM`, and the exact epoch count/selection criterion — has no surviving primary source and is
carried from `v6c`'s template as an honest `unknown`, not upgraded to `confirmed` or even `inferred`
without real evidence. The one genuine resolution this task achieves beyond restating prior findings
is `multispeaker`: independent checkpoint-shape forensics now agrees with `v6c`'s own (previously
uncorroborated) claim of `false`, against the shipped bundle's and the confound table's `true`. The
checkpoint forensics also answer Key Question 3 directly: the decoder changed substantially (the
largest relative shift of the five modules, +225%), so v3's speaker-similarity gain over "base
Kokoro + v3 voicepack" cannot be attributed to `predictor`/`text_encoder` alone.

## Limitations

* The 266-clip training list is unrecoverable; whether it overlaps `data/v4/val_list.txt`'s 96
  held-out clips could not be checked, since the list itself was never seen by this or any prior
  task.
* `multispeaker: false` is `inferred`, not `confirmed` — the checkpoint-shape evidence is suggestive
  (absence of a second conditioning pathway) but the exact `StyleTransformer1d`/ `Transformer1d`
  parameter-shape signature could not be confirmed because their class source lives in an external
  pip package not recovered from the VM within the bounded inventory window.
* Byte-identity between `stage1/first_stage.pth`, the VM's (unfound) `first_stage_v3.pth`, and
  whatever `v6c`/`v6d` actually loaded remains unconfirmed — the VM search came back empty.
* `joint_epoch`, `lr`, `batch_size`, `train_LM`, and the exact epoch-selection criterion for "best"
  are all `unknown`, carried from `v6c`'s template with no evidence of their own.
* This project has never run a controlled ablation isolating data-scale, `multispeaker`, or any
  single hyperparameter as the actual cause of v3's success versus every other run's failure — this
  reconstruction states what v3 most likely did, not why it worked where nine other configurations
  did not.
* The locally re-measured `speaker_sim=0.566` used a different reference-centroid construction than
  t0008's own recorded numbers (documented explicitly in `results/v3_checkpoint_forensics.md`), so
  it is weak corroborating evidence at best, not an independent confirmation of t0008's numbers.

## Sources

* Task: `t0002_kokoro_v4_voicepack_decoder_package` — packaging/extraction contract confirmed here
  as [t0002]
* Task: `t0006_kokoro_v5_stage2_subset` — owner of the DVC-tracked v3 reference bundle and the `v6c`
  template, cited throughout as [t0006]
* Task: `t0008_tts_eval_harness_baselines` — `tts_eval_harness` library and recorded
  speaker_sim/rtf/ttfb baselines, cited as [t0008]
* Task: `t0009_stage2_training_failure_forensics` — confound table (including the `lambda_gen` error
  found in this task) and recommended next-run config, cited as [t0009]
* Task: `t0013_v10_synthesis_quality_forensics` — original audio-quality gate design, cited as
  [t0013]
* Task: `t0015_v11_duration_blowup_forensics` — checkpoint-forensics and hardened audio-quality gate
  code copied and adapted here, cited as [t0015]

[t0002]: ../../../t0002_kokoro_v4_voicepack_decoder_package/
[t0006]: ../../../t0006_kokoro_v5_stage2_subset/
[t0008]: ../../../t0008_tts_eval_harness_baselines/
[t0009]: ../../../t0009_stage2_training_failure_forensics/
[t0013]: ../../../t0013_v10_synthesis_quality_forensics/
[t0015]: ../../../t0015_v11_duration_blowup_forensics/
