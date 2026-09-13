# Results Detailed: Kokoro v4 Voicepack + Decoder Packaging

See `results_summary.md` for full methodology, key findings, and analysis. This task's
`results_summary.md` contains complete detail (methodology, spectrogram analysis, bug findings,
limitations, next steps) and serves as the primary results document.

## Files Created

- `results/david_v4_voicepack.pt` — Stage 1 style, ref_s std=1.208 (DVC-tracked)
- `results/david_v4_decoder.pth` — five-module bundle from Stage 2 epoch 6 (DVC-tracked)
- `results/david_v4_decoder_epoch3.pth` — five-module bundle from Stage 2 epoch 3 (DVC-tracked)
- `results/david_v4_decoder_stage1.pth` — five-module bundle from Stage 1 (DVC-tracked)
- `results/audio_samples/` — synthesized audio, Stage 2 epoch 6 decoder (DVC-tracked)
- `results/audio_samples_epoch3/` — synthesized audio, Stage 2 epoch 3 decoder (DVC-tracked)
- `results/audio_samples_stage1/` — synthesized audio, Stage 1 decoder (DVC-tracked)
- `code/make_voicepack_v4.py`, `code/extract_decoder_v4.py`, `code/extract_decoder_generic.py`
- `code/test_kokoro_inference.py`, `code/test_kokoro_inference_generic.py`
