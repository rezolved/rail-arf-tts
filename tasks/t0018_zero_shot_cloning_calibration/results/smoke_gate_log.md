# Smoke Gate Log

Per-system install time, `ref_single` smoke-gate pass/fail, and (F5-TTS only) the dedicated
`ref_concat` pre-check result (plan Step 4).

| System | Install Time | Smoke Gate (ref_single) | Notes |
| --- | --- | --- | --- |
| chatterbox | pre-installed in setup-machines step (t0018 step 8) | PASS | ref_single: 1/1 successful clip, wall_clock=45.7s (first-call CUDA kernel compile overhead, absorbed by the 50-warmup protocol in the full run). Audio file size matches expected float32 WAV size for the synthesized duration. |
| cosyvoice2 | pre-installed in setup-machines step (t0018 step 8); weights downloaded this step via huggingface_hub snapshot_download | FAIL then FIXED then PASS | First attempt failed: TypeError: Invalid file: tensor(...) inside CosyVoice2's own frontend_zero_shot -> _extract_speech_feat -> load_wav(prompt_wav, 24000), because prompt_wav must be a file PATH (CosyVoice2 loads it internally at 3 different sample rates), not the pre-loaded 16kHz tensor adapters_zeroshot.py originally passed. Fixed in code/adapters_zeroshot.py's cosyvoice2_synth (pass str(ref_wav_path) directly). Re-verified working via a standalone debug script before the fix was trusted. |
| f5_tts | pre-installed in setup-machines step (t0018 step 8); weights downloaded this step | FAIL (timeout/hang, REQ-7 null variant) | See intervention/f5_tts_smoke_gate_failed.md for full detail. Three attempts (default caching, HF_HUB_OFFLINE=1, alternate GPU) all hung indefinitely inside the F5TTS constructor with zero progress signal, despite the model class successfully loading once earlier via a bare test outside run_eval_zeroshot.py. Marked null for all f5_tts variants. The dedicated ref_concat pre-check (Step 4) never ran, since F5-TTS never got past the ref_single smoke gate. |

## F5-TTS `ref_concat` pre-check detail

Never reached: F5-TTS's `ref_single` smoke gate itself hung indefinitely across all three attempts
(see `intervention/f5_tts_smoke_gate_failed.md`), so the dedicated `ref_concat` pre-check this
section documents (Step 4) was never executed. This is a documented consequence of the REQ-7
smoke-gate failure, not a separate omission.
