# Checkpoint Forensics v11 (Milestone A step 3, REQ-2)

Mandatory pre-flight tensor-level check, run **before** any GPU training spend, per `plan/plan.md`
Milestone A. Produced by `code/inspect_checkpoint.py`.

## Per-checkpoint forensics

### `epochs_2nd_00020 (v11 new first_stage_path target)`

**Decoder architecture:** `hifigan` (hifigan `alphas` marker=True, ups stages=4)

| Module | Params | Finite | Weight norm |
| --- | ---: | --- | ---: |
| bert | 6,292,480 | yes | 178.4446 |
| bert_encoder | 393,728 | yes | 37.6189 |
| decoder | 54,289,492 | yes | 832.0366 |
| diffusion | 50,653,952 | yes | 896.4111 |
| mpd | 41,105,770 | yes | 254.8386 |
| msd | 280,902 | yes | 31.7925 |
| mwd | 1,173,634 | yes | 21.6887 |
| pitch_extractor | 5,251,148 | yes | 3271270.6511 |
| predictor | 16,194,612 | yes | 313.9305 |
| predictor_encoder | 13,880,813 | yes | 459.0197 |
| style_encoder | 13,880,813 | yes | 399.4790 |
| text_aligner | 7,868,452 | yes | 892.3647 |
| text_encoder | 5,606,400 | yes | 401.4438 |
| wd | 1,173,634 | yes | 23.1127 |

## Verdict (REQ-2 gate)

**PASS.** `epochs_2nd_00020.pth`'s `decoder` classifies as `hifigan` (hifigan `alphas` marker=True,
ups stages=4), matching `config_david_v11.yml`'s `model_params.decoder.type: hifigan` block
field-for-field, all core modules present, all tensors finite. This corroborates t0013's own
already-documented classification of this exact checkpoint file
(`tasks/t0013_v10_synthesis_quality_forensics/results/checkpoint_forensics.md`, where it served as
the validated known-good control). Cleared to proceed to Milestone B (GPU training) -- Approach 1
(fine-tune from real hifigan pretrained weights), not the Approach-2 random-init fallback.

See `tasks/t0013_v10_synthesis_quality_forensics/results/checkpoint_forensics.md` for the
already-documented before/after contrast (`first_stage_v3.pth` classifies `istftnet`; both v10
checkpoints classify `hifigan`) -- not re-derived here to avoid a redundant 1.7GB re-download of
`first_stage_v3.pth`.
