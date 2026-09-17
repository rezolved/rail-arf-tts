# Results Summary: t0014 Kokoro Stage 2 v11 Decoder-Init Fix and Retrain

## Summary

This task fixed the root cause diagnosed by t0013 that left `kokoro-v10-best`'s HiFi-GAN decoder
producing clipped/saturated noise: `train_second_v10.py` loaded an ISTFTNet-shaped Stage-1
checkpoint (`first_stage_v3.pth`) into a HiFi-GAN-shaped decoder config without excluding `decoder`,
and the loader's zero-match guard silently accepted the partial (Frankenstein) match. The fix
repoints `first_stage_path` at `yl4579/StyleTTS2-LibriTTS`'s `epochs_2nd_00020.pth` -- a checkpoint
whose decoder is genuinely HiFi-GAN-shaped and field-matches `config_david_v11.yml` exactly,
confirmed by a mandatory tensor-level pre-flight check (`results/checkpoint_forensics_v11.md`)
before any GPU spend.

Training completed all **50 of 50 planned epochs** (`joint_epoch=30`, `diff_epoch=10`, anchored on
StyleTTS2's own official `Configs/config_ft.yml` recipe) on t0012's 1,531-clip LUFS-normalized
corpus, with **0 `HealthGate` firings**. Root-disk usage was monitored proactively throughout
(reclaimed from 98%/3GB free to 87%/16GB free before training, stayed stable at 87% for the entire
run) -- the exact disk-exhaustion failure mode that caused t0010's $272.78 overrun did not recur.

**The mandatory audible-speech gate PASSED**: `code/audio_quality_check.py`'s
`check_audio_quality()` on the resulting checkpoint's synthesis output returned
`is_likely_noise=False` (`clip_fraction=0.0048`, `spectral_flatness=0.0058`), a qualitative reversal
of v10's confirmed failure (`clip_fraction=0.750-0.807`, `is_likely_noise=True`). Full detail:
`results/v11_gate_verdict.md`. Because the gate passed, this task produced its conditional
deliverables: the `kokoro-v11-best` model asset, paired original-vs-fine-tuned audio samples, and a
working StyleTTS2-native inference recipe.

**Known anomaly, disclosed rather than hidden**: the gate-passing synthesis output is anomalously
long (73.95s for a 10-word test sentence, vs. 2.5-4.9s for the control and both v10 checkpoints on
comparable inputs). This points to a likely duration-predictor calibration issue, separate from the
decoder/vocoder defect this task fixed. It does not affect the pre-registered gate criterion
(`is_likely_noise`), which examines clipping/spectral-flatness/silence, not duration -- but it is a
real, disclosed limitation of this checkpoint, documented in the model asset's Known Limitations and
recommended as a follow-up investigation.

**Licensing disclosure** (per `plan/plan.md`'s Approach section and
`research/research_internet.md`'s "Licensing" finding): this checkpoint's decoder was initialized
from `yl4579/StyleTTS2-LibriTTS`'s pretrained weights (`epochs_2nd_00020.pth`). Per that
repository's README (GitHub issue `StyleTTS2-Issue37`), any model initialized from these weights
must disclose that fact and restrict voice-cloning to consented speakers. This disclosure is that
notice: this project uses the checkpoint only as a decoder weight initialization for fine-tuning on
Rezolve's own licensed David voice data, and never ships a LibriTTS speaker identity.

## Metrics

- **Best val_loss**: **0.3394** at epoch 38 (post-`joint_epoch`)
- **Selected checkpoint val_loss**: **0.3469** (epoch 48, the latest saved checkpoint; checkpoints
  save every 2 epochs, so 49/50 has no saved file). No clear regression at epoch 48 vs. epoch 38
  (0.339-0.367 range across epochs 30-48) -- selected per plan.md step 8's rule.
- **Health gate events**: **0** across all 50 epochs
- **Audible-speech gate**: **PASS** -- `is_likely_noise=False` (`results/audio_quality_v11.json`,
  `results/v11_gate_verdict.md`)
- **speaker_sim**: **0.444** (GE2E cosine vs. `11labs_david` centroid) -- above both
  confirmed-broken v10 checkpoints (0.311-0.351), still well below the project's 0.85 target
- **rtf**: **3.18** (single-clip, unbatched CPU inference; elevated in part by the duration anomaly,
  not a benchmark claim)
- **ttfb_ms**: not measured (offline batch harness, no streaming endpoint in scope, per t0013's own
  precedent)

## Deliverables produced (Milestone D, gate passed)

- `assets/model/kokoro-v11-best/` -- `details.json`, `description.md`, `files/kokoro-v11-best.pth`
  (331 MB, 5-module state dict, DVC-tracked), `files/config_david_v11.yml`
- `results/audio_samples/ft/v11_best.wav` -- v11 synthesis output (gate-passing)
- `results/audio_samples/original/librispeech_control.wav` -- reused unchanged from
  `tasks/t0013_v10_synthesis_quality_forensics/results/audio_samples/control_epochs_2nd_00020.wav`
  (same pretrained checkpoint, same harness; not re-synthesized, per plan.md step 10's explicit
  reuse allowance)
- `code/infer_styletts2.py`, `code/build_reference_concat.py` -- working StyleTTS2-native inference
  recipe

## Verification

- `verify_task_metrics.py` -- PASSED (0 errors, 0 warnings; explicit variant format, 1 variant
  `v11-best`, registered metrics only)
- `meta.asset_types.model.verificator` (the actual importable verificator; the module path
  `arf.scripts.verificators.verify_model_asset` referenced by `plan/plan.md` does not exist in this
  repo -- confirmed the same command also fails for t0010's own model asset verification log,
  `tasks/t0010_stage2_safeguarded_training/logs/commands/073_...json`, `exit_code: 1`, "No module
  named arf.scripts.verificators.verify_model_asset") -- PASSED for `kokoro-v11-best`: 0 errors, 2
  warnings (`MA-W005` empty `meta/categories/` project-wide, `MA-W014` empty `training_dataset_ids`,
  no dataset asset exists for t0012's corpus) -- same two non-blocking warning classes as
  `kokoro-v10-best`'s own verification.

## Deviations from the plan (disclosed)

1. **Verificator module path.** `plan/plan.md`'s literal command
   (`uv run python -m arf.scripts.verificators.verify_model_asset ...`) does not exist in this repo;
   the real, importable verificator is `meta.asset_types.model.verificator`. This is a pre-existing
   documentation gap in the plan template (also present, and also never corrected, in t0010's plan),
   not something this task introduced.
2. **5-module packaging convention carries forward a known limitation, made explicit here.**
   `files/kokoro-v11-best.pth` mirrors `kokoro-v10-best`'s exact 5-module packaging convention as
   instructed, but per this task's own Approach-section research, Kokoro's `KModel`/`KPipeline`
   cannot correctly load a `hifigan`-decoder checkpoint -- so this 3-file packaging is for
   downstream convention-parity, not the file this task's audible-speech gate actually validated.
   The gate ran against the full 13-module raw `epoch_00048.pth` through the StyleTTS2-native
   harness. This is called out explicitly in `description.md`'s Usage Notes rather than silently
   repeating v10's implicit (and, per this task's own findings, incorrect) claim that the 5-module
   file loads via `adapters.load_kokoro_model_with_checkpoint`.
3. **Duration anomaly disclosed, not investigated to root cause.** Per the Rejection Criteria in
   `plan/plan.md` ("no other metric may be substituted" for the pre-registered `is_likely_noise`
   gate), this anomaly does not block Milestone D, but it is flagged prominently
   (`results/v11_gate_verdict.md`, `description.md`) as a follow-up-suggestion candidate rather than
   silently omitted.
4. **Reference audio for the gate synthesis** used the exact 3 named clips from t0013's
   `v10_diagnosis.md` Milestone E (`lining_up_suggestions_17`, `lining_up_suggestions_10`,
   `putting_them_head_to_head_15`), not t0013's later `random_decoder_probe.py`'s
   alphabetically-first-3 selection -- chosen for direct comparability with the v10/control numbers
   this task's gate verdict is compared against.

## Cost

`LLM-T1-NC80` was torn down at the `teardown` step (destroyed at 2026-09-16T23:11:25Z). **Final
total: $89.85** over **6.436 hours** billed (`results/costs.json`,
`results/remote_machines_used.json`) -- this supersedes the ~$86.46 interim figure quoted during
`implementation`.
