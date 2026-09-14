---
spec_version: "3"
task_id: "t0009_stage2_training_failure_forensics"
step_number: 9
step_name: "creative-thinking"
status: "completed"
started_at: "2026-09-14T16:31:24Z"
completed_at: "2026-09-14T16:45:00Z"
---
## Summary

Creative-thinking pass over the forensic findings from step 8 to identify alternative failure
hypotheses, stress-test the two root-cause claims, surface unresolved late-divergence questions, and
propose safeguard strategies not captured in the main analysis. Six distinct hypothesis threads were
explored; three yield actionable changes to the next training task.

## Actions Taken

1. Challenged the primary root-cause claims (DP checkpoint mismatch and joint\_epoch=3) against the
   available evidence, asking whether either claim could be falsified or is over-attributed.
2. Explored alternative mechanisms for the late divergence observed in v6c (epochs 9+) that the main
   analysis left unexplained.
3. Examined the v3 reference run as a counter-case — it succeeded with fewer patches and an unknown
   joint\_epoch — and asked what this implies about the necessity of each fix.
4. Questioned the val\_loss comparability assumption between v3 (val=0.506) and v6c (val=0.849) to
   determine whether they are measuring the same thing.
5. Reviewed the health gate thresholds for over-fit to a single run (v6c) and proposed a
   threshold-derivation method that is robust to dataset and batch-size change.
6. Identified a missing safeguard class: startup-time parameter-count assertion, which catches the
   zero-param load before any training happens rather than after the run diverges.

## Analysis

### Hypothesis 1: DP checkpoint mismatch may be over-attributed

The main analysis assigns HIGH probability to silent zero-param loading for runs v6a and v6b. The
evidence is: (a) the configs explicitly set `multispeaker=false` while the checkpoint
(`epoch_1st_00007.pth`) was saved with `multispeaker=true`, and (b) the upstream loader uses
`strict=False` which silently matches zero parameters. This is the strongest inference chain in the
analysis. However, there is a lurking alternative: the checkpoint itself may not be a DataParallel
checkpoint (DP wrapping only happens if `torch.nn.DataParallel` was used at save time). The confound
table notes that t0001 used `accelerate launch` rather than plain `python train_second.py`, which
means DP wrapping is plausible for that checkpoint — but v6a/v6b loaded `epoch_1st_00007.pth` from
t0004, which may or may not have been saved with DP.

**What this changes**: The conclusion is still directionally correct (strict=False is dangerous
regardless of whether DP wrapping happened), but the "trained from scratch" claim for v6a/v6b cannot
be confirmed without loading the checkpoint and counting matched parameters. The fix
(parameter-count assertion at startup) is the right safeguard independent of which specific
mechanism was active.

**New safeguard (not in implementation)**: Add a startup assertion that logs the exact number of
parameters loaded from the checkpoint and raises an error if fewer than 80% of the expected
parameters are matched. This is a one-line addition after `load_checkpoint` returns and does not
require access to the VM.

### Hypothesis 2: joint\_epoch=3 may not be an independent cause — it may be a multiplicative risk

The main analysis treats `joint_epoch=3` as a separate cause from the checkpoint mismatch. But
consider: if v6a/v6b really did train from scratch (due to zero-param load), then `joint_epoch`
becomes irrelevant — the run diverges regardless because the starting point is random noise, not a
trained Stage 1 model. The only run where `joint_epoch` was the isolating variable is v6c vs the
t0001/t0005 runs, but those runs also differed on the checkpoint (v6c used `first_stage_v3.pth`, the
others used `epoch_1st_00007.pth`). The two factors were never crossed in a controlled way: there is
no run with `joint_epoch=6` + wrong checkpoint or `joint_epoch=3` + correct checkpoint.

**What this changes**: The recommendation to set `joint_epoch>=6` is still correct as defensive
advice, but the evidence that `joint_epoch=3` is an independent cause (vs. merely correlated with
the successful run) is weaker than stated. The answer asset should note this confound rather than
asserting HIGH confidence for both causes independently.

**Alternative hypothesis**: The v3 reference run (val=0.506) is consistent with `joint_epoch` being
very small or even 0 (no GAN phase at all). If v3 succeeded without a late joint\_epoch, it suggests
the checkpoint fix alone may be sufficient, and `joint_epoch` tuning is secondary. This is not
explored in the main analysis.

### Hypothesis 3: Late divergence in v6c (epoch 9+) is likely SLM loss instability

The main analysis labels the epoch 9+ divergence "unclear" and lists three candidates: learning rate
schedule, SLM adversarial loss instability, and style encoder divergence. The val\_loss trajectory
from v6c (`val=0.849 → 0.883 → 0.997 → 2.309 → 3.743`) is an exponential blow-up starting at epoch 9
— the epoch immediately after GAN activation (`joint_epoch=6`, so GAN active from epoch 7, blow-up
at epoch 9). The two-epoch lag between GAN activation and divergence onset is consistent with the
SLM discriminator accumulating gradient signal before destabilizing the acoustic model.

**Alternative mechanism**: The StyleTTS2 SLM loss uses a pre-trained SLM (WavLM or similar) as a
discriminator. If the SLM's gradient scaling is not clipped, it can overwhelm the acoustic loss.
Patch 3 in the t0005 7-patch stack adds an iSTFTNet output clamp — this addresses numerical overflow
in the iSTFTNet output but not the gradient magnitude from the SLM head. The missing safeguard is a
gradient-norm clip specifically on the SLM loss path, not a global gradient clip.

**New safeguard**: Add a per-loss gradient norm check: after the backward pass, compute the L2 norm
of the SLM discriminator gradient and log it. If it exceeds `N×` the acoustic gradient norm, flag it
as a warning at epoch start. Threshold calibration requires a new run, but the logging
infrastructure (StepLogger) can carry the raw values with a one-line addition.

### Hypothesis 4: val\_loss 0.506 (v3) vs 0.849 (v6c) are not measuring the same thing

The main analysis references v3's val=0.506 as a comparison point for v6c's val=0.849. But v3 used a
266-clip dataset; v6c used 250 clips (different split, different phoneme distribution). More
critically: the v3 config is unknown — if v3 used `joint_epoch=0` (GAN never activated), then its
val\_loss is a pure reconstruction loss, while v6c's best checkpoint at epoch 6 was saved just
before GAN activation. These are not the same loss function over the same distribution.

**What this changes**: The targets for the next task should not be stated as "beat v3's 0.506" on
val\_loss. The correct target is: reproduce v6c's behaviour (no early divergence, stable through
epoch 6) and then run additional epochs with a reduced SLM gradient. Val\_loss comparisons across
runs with different `joint_epoch` and different datasets are not meaningful without recomputing on a
fixed held-out set.

**Recommendation**: Define a held-out evaluation set separate from the training-time val set, and
run inference on it with a speaker-similarity metric (GE2E cosine) rather than relying on training
val\_loss for cross-run comparison.

### Hypothesis 5: Health gate thresholds are calibrated from a single run — they may over-fire on

correct training

The gate thresholds (dur\_loss\_step1 < 2.0, acoustic\_norm < 20.0, val\_spike < 0.05) were designed
by observing v6c's divergence at epoch 9. The val\_spike threshold (0.05) is tight: a 5% val\_loss
increase triggers an abort. In the v6c log, the best epoch transition (epoch 5→6) shows a val
increase from 0.884 to... (no data for epoch 5 step-level; only that epoch 6 val=0.849 is better
than epoch 4 val=0.884). So the trajectory is not monotone in the pre-GAN phase either.

**Risk**: If future runs have natural val oscillation > 0.05 in pre-GAN epochs (plausible with a
different dataset or batch size), the gate fires as a false positive and aborts a healthy run.

**Recommendation**: The val\_spike gate should only activate after `joint_epoch`, not during pre-GAN
training. The current implementation checks `after joint_epoch` already (gate 3 fires when
`val_loss increases by more than 0.05 in a single epoch after joint_epoch`) — this is correct. But
the absolute thresholds for dur\_loss and acoustic\_norm should be documented as calibrated to the
v6c 250-clip subset at lr=1e-4. If the next run uses a different dataset or learning rate, a warm-up
epoch with no gates active would help establish baseline values before any gate fires.

### Hypothesis 6: The 7-patch stack in t0005 masks a third root cause — data normalization

The analysis classifies the 7 t0005 patches as "crash mitigation" on top of the two root causes.
Patch 2 (phoneme guard: skip batch if any phoneme sequence is empty) and patch 4 (skip-step guard:
if mel-spec reconstruction fails, skip the batch) are data-path safeguards. These would be
irrelevant if all inputs were well-formed. Their presence implies that the v5 data pipeline produced
malformed batches at non-trivial frequency.

**Alternative hypothesis**: A third contributing factor — malformed or silent clips in the v5
training data — caused instability independent of the checkpoint issue. If enough batches are
skipped due to bad data, the effective batch size drops, gradient variance increases, and the
model's loss landscape becomes noisier. This would interact with `joint_epoch=3` to produce earlier
divergence than expected.

**What this changes**: The data audit confirmed 0 train/val overlap and no immediate manifest-level
errors, but it did not examine audio quality (peak amplitude, silence fraction, clipping). A
pre-training audio quality gate — flag clips below -40 dBFS peak, or clips where >30% of frames are
silence — would eliminate this risk factor before training starts, not during it.

**Proposed next task**: Run a full audio quality audit on the v5 train list (1557 clips) and the
ElevenLabs reference set. Discard clips with silence > 30%, clipping (>0 dBFS peak), or duration <
1.5 s before any future training run.

## Alternative Safeguard Strategies Not in Main Analysis

The following safeguards were not included in the `t0009_training_safeguards` library but are
actionable for the next training task:

1. **Startup parameter-count assertion**: After `load_checkpoint`, assert that matched parameters
   > = 80% of expected. One-line addition to `train_second_safeguarded.py`. Catches both the
   > DataParallel prefix issue and any future architecture mismatch at the moment of load, not after
   > divergence.

2. **Per-loss gradient norm logging**: After the backward pass, log the L2 norm of the SLM
   discriminator gradient alongside the acoustic gradient norm. No abort threshold needed yet — just
   observability data for the next run. The StepLogger already accepts freeform dict records, so
   this is a one-line addition to the training loop.

3. **Warm-up epoch with gates inactive**: Run the first epoch without any health gates active, log
   baseline values for dur\_loss, acoustic\_norm, and val\_loss, then dynamically set the gate
   thresholds as `N × baseline_mean` rather than the v6c-calibrated constants. This makes the gates
   dataset-agnostic and eliminates the false-positive risk on new data.

4. **Audio quality pre-filter**: Before starting training, run a one-pass audio quality check on the
   manifest: discard clips with silence rate > 30%, peak amplitude > -1 dBFS (clipping), or duration
   < 1.5 s. Log the discarded clip count. This addresses the Hypothesis 6 concern about malformed
   batch noise independently of the checkpoint and joint\_epoch fixes.

## Outputs

- `logs/steps/009_creative-thinking/step_log.md` (this file)

## Issues

No issues encountered. All analysis is based on artifacts committed in step 8. The four alternative
safeguard strategies identified here are recommendations for future tasks — they do not require
changes to the step 8 deliverables.
