# Pipeline Audit

## v3 Diff vs t0005 Patch Stack

v3's committed diff (`train_second_patch.diff`) contains only 2 hunks: the `lambda_slm > 0` guard
and the `utils.py` monotonic_align path fix. t0005's `train_second_patched.py` has all 7 patches
below.

| ID | Patch Name | In v3 Diff | In t0005 | Classification |
| --- | --- | --- | --- | --- |
| 1 | lambda_slm > 0 guard | YES | YES | **hides symptom** |
| 2 | consecutive skip guard (NaN loss skips optimizer step) | NO | YES | **hides symptom** |
| 3 | gradient norm clipping | NO | YES | **hides symptom** |
| 4 | istftnet exp() clamp | NO | YES | **hides symptom** |
| 5 | train_LM guard (freeze BERT) | NO | YES | **fixes cause** |
| 6 | monotonic_align sys.path fix | YES (in utils.py) | YES | **neutral crash fix** |
| 7 | DataParallel safe load_checkpoint | NO | NO (t0001 only) | **fixes cause** |

## Patch Details

### Patch 1: lambda_slm > 0 guard

**Classification**: hides symptom

Skips SLM backward pass when lambda_slm=0. Prevents KeyError on `wd` optimizer. v3 committed this
guard; t0005 copied it. This is a crash fix, not a divergence fix.

### Patch 2: consecutive skip guard (NaN loss skips optimizer step)

**Classification**: hides symptom

If NaN losses accumulate, the optimizer step is skipped. Prevents training from crashing on NaN.
Caps consecutive skips at MAX_CONSECUTIVE_SKIPS (50). Hides the root cause (NaN gradient from GAN)
rather than fixing it.

### Patch 3: gradient norm clipping

**Classification**: hides symptom

clip_grad_norm_ applied to decoder, MSD, and MPD. Limits gradient explosion symptomatically. v3 did
NOT have this — yet v3 succeeded. This is a symptom fix; removing it while keeping the correct
checkpoint is likely safe.

### Patch 4: istftnet exp() clamp

**Classification**: hides symptom

Clamps the exp() argument in istftnet.py to prevent overflow NaN. t0005 patched istftnet.py
directly. v3 did not need this patch — implies v3 trained in a regime where exp() did not overflow
(lower lambda_gen? different LR?).

### Patch 5: train_LM guard (freeze BERT)

**Classification**: fixes cause

When train_LM=false, BERT is frozen and LM Loss does not backpropagate. Prevents LM Loss explosion
(23 → 400+) seen with train_LM=true. This is a genuine cause fix — train_LM=true destabilizes
training.

### Patch 6: monotonic_align sys.path fix

**Classification**: neutral crash fix

Adds monotonic_align to sys.path so import works from the finetune directory. This is an environment
fix; does not affect training dynamics.

### Patch 7: DataParallel safe load_checkpoint

**Classification**: fixes cause

t0001's custom loader strips `module.` prefix and raises RuntimeError on 0-param match. t0005/t0006
use the upstream strict=False loader — silently loads 0 params when a DataParallel-saved checkpoint
meets a non-wrapped model. v6b (multispeaker mismatch) almost certainly trained from scratch due to
this. This is the highest-confidence root cause fix missing from t0005/t0006.

## Key Finding

v3 succeeded with only patches 1 and 6 (both safe, environment-level fixes). The 5 additional
patches in t0005 (2-5, 7) are all crash mitigation, not root-cause fixes. The most likely missing
root cause fix is **patch 7** (safe load_checkpoint) — t0001 had it; t0005 and t0006 did not, and
v6b almost certainly trained from scratch silently.

## Mel Extraction Parameters

sr=24000 (Kokoro expects 24000 Hz) — OK n_mels=80 (Kokoro model_params expects 80) — OK
hop_length=300 (Kokoro expects 300 @ 24 kHz) — OK n_fft=2048 (Kokoro expects 2048) — OK

All mel extraction parameters match Kokoro decoder expectations.
