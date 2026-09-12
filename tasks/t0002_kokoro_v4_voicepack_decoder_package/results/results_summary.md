# Results: Kokoro v4 Voicepack + Decoder Packaging

## Summary

Reproduced v3's Kokoro-API packaging recipe (voicepack from Stage 1 style encoder, decoder+
predictor+text_encoder+bert from Stage 2) for v4's checkpoints. Audio remained noisy across every
variant tested (Stage 1 decoder, Stage 2 epoch 3 pre-GAN, Stage 2 epoch 6 best-by-val_loss). Root
cause isolated to two independent bugs — one in this task's extraction script (fixed), one in
t0001's underlying checkpoint (not fixable by repackaging). **v4 Stage 2 has no usable checkpoint
for production packaging as-is; a re-run with a revised training schedule is required.**

## Methodology

- Extracted `voicepack` (ref_s, `[510,1,256]`) from `first_stage.pth`'s `style_encoder` +
  `predictor_encoder`, averaged over 200 David train clips — matches v3's
  `make_voicepack_from_ft.py`.
- Extracted `decoder` module state dict from three Stage 2 checkpoints
  (`epoch_2nd_00005.pth`/epoch 6/val=0.751, `epoch_2nd_00002.pth`/epoch 3/val=0.770 pre-GAN,
  `first_stage.pth`/Stage 1 pure-reconstruction) via key remapping
  (`parametrizations.weight.originalN` → `weight_g`/`weight_v`), matching
  `convert_checkpoint.py`.
- Loaded into `kokoro.KModel` + `KPipeline`, synthesized 2 fixed test sentences, inspected
  spectrograms (`scipy.signal.spectrogram`) against a known-clean v3 sample
  (`kokoro-finetune/v3/audio/v3b/ep9_p5.wav`) and the real v4 reference clip.

## Key Findings

1. **Bug found and fixed in this task**: the original extraction only pulled `net["decoder"]`,
   leaving `predictor`/`text_encoder`/`bert`/`bert_encoder` as base Kokoro-82M generic weights.
   Inspecting v3's actual production artifact (`david_v3_best_decoder.pth`) showed it packages
   **five** modules (`bert`, `bert_encoder`, `predictor`, `text_encoder`, `decoder`), not just the
   decoder — the file name is a misnomer. Fixed in `code/extract_decoder_v4.py` /
   `code/extract_decoder_generic.py` to extract all five.
2. **`disable_complex=True` is not the fix it first appeared to be**: hypothesized from a first
   background-agent pass (timestamps + `kokoro/istftnet.py` code path), but re-listening after
   applying it showed noise unchanged. Not load-bearing for this issue; keeping it is still
   correct practice (real-FFT ISTFT path is more numerically stable on CPU) but it does not
   address the root cause.
3. **Pipeline validated correct, independent of v4's checkpoint**: running the exact same
   five-module extraction + `KModel`/`KPipeline` load on v3's proven-good artifacts
   (`v3/best/david_v3_best_decoder_kokoro.pth` + `v3/best/david_v3_best_voicepack.pt`) produced a
   clean spectrogram (formant structure, silence in pauses) and a normal 3.4 s duration for a
   short sentence — confirming the extraction/inference code itself is not the source of the v4
   noise.
4. **v4's Stage 2 predictor is broken**: loading v4's five-module bundle (any of the three epochs
   tested) into `KModel` produced a **duration explosion** — 89 s / 39 s synthesized audio for
   sentences that should run ~9.6 s / ~3.6 s (confirmed via the decoder-only test, which uses
   Kokoro's base predictor and produces sane durations). This did not happen with v3's
   checkpoint using the identical code path, isolating the defect to v4's own training run, not
   the packaging.
5. **Persistent tonal artifact (~3–3.5 kHz band, present even in silent pauses)** appears in the
   spectrograms of the decoder-only v4 tests (Stage 1 decoder and Stage 2 epoch 6 decoder,
   both), a pattern absent in v3's clean sample and the real v4 reference. This band is present
   even in the Stage 1 (pre-Stage-2, no-GAN) decoder, meaning it is not a GAN-instability
   artifact specific to the epoch-7 divergence documented in t0001 — it points to something
   wrong earlier, in Stage 1 training or in the v4 training data itself.

## Limitations

- No GE2E speaker-similarity or WER benchmarking was run — this task was a qualitative
  listening/spectrogram check only, per its reduced scope.
- Root-caused the *symptom* (broken predictor durations, persistent decoder-side artifact); did
  not diagnose *why* v4's Stage 1/Stage 2 training produced these specific defects — that
  requires re-examining t0001's Stage 1 training log and data, out of this task's scope.

## Files Created

- `results/david_v4_voicepack.pt` — Stage 1 style, ref_s std=1.208.
- `results/david_v4_decoder.pth`, `david_v4_decoder_epoch3.pth`, `david_v4_decoder_stage1.pth` —
  five-module bundles (bert/bert_encoder/predictor/text_encoder/decoder) from three Stage 2/1
  checkpoints.
- `results/audio_samples/`, `audio_samples_epoch3/`, `audio_samples_stage1/` — synthesized test
  sentences per variant (decoder-only swap; predictor swap not usable, see Key Finding 4).
- `code/make_voicepack_v4.py`, `code/extract_decoder_v4.py`, `code/extract_decoder_generic.py`,
  `code/test_kokoro_inference.py`, `code/test_kokoro_inference_generic.py`.

## Next Steps (recommendation only — not executed)

1. **Re-run t0001's Stage 2 training with a revised schedule** before attempting packaging again:
   - Longer pre-GAN warmup (raise `joint_epoch` from 3 to ~6-8) so the predictor/decoder have
     more stable reconstruction-only epochs before adversarial signal is introduced — v3 used the
     same `joint_epoch: 3` but only ran 10 total epochs on 266 clips; v4's 1557 clips and 20-epoch
     schedule may need a proportionally longer warmup, not the same absolute epoch count.
   - Lower discriminator LR relative to generator LR (asymmetric GAN LR), to reduce the
     epoch-7-style divergence seen in t0001 (val_loss 0.751 → 1.147 in one epoch).
   - Investigate whether the ~3–3.5 kHz artifact traces back to Stage 1 (test Stage 1's decoder
     module in isolation on a v3-style small held-out set, listen before touching Stage 2 again)
     or to a data preprocessing issue specific to v4's 1557-clip set (sample rate, clipping,
     mismatched mel normalization vs the encoder used to build the training manifest).
2. **Before the next packaging attempt, verify predictor duration sanity first**, as a cheap
   smoke test: load only `predictor` (not decoder) into `KModel` and check synthesized duration
   is within ~2x of the base-Kokoro-predictor duration for the same sentence. This is far cheaper
   than a full listen/spectrogram pass and would have caught Key Finding 4 immediately.
3. Once a Stage 2 re-run produces a checkpoint that passes the duration sanity check, repeat this
   task's five-module extraction + spectrogram comparison against v3b before doing any
   GE2E/speaker-similarity benchmarking.
4. Consider tracking `first_stage.pth` and the Stage 2 checkpoints in DVC now that they live in
   `t0001/results/checkpoints/` (copied from `rail-benchmarks/kokoro-finetune/best_v4/` this
   session) — currently untracked by git and undocumented as a versioned asset.
