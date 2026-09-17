---
spec_version: "2"
answer_id: "zero-shot-speaker-sim-ceiling"
answered_by_task: "t0018_zero_shot_cloning_calibration"
date_answered: "2026-09-17"
---
## Question

What speaker_sim and latency envelope is reachable on the David voice by zero-shot cloning, and what
does that imply for the Kokoro fine-tuning line?

## Answer

Yes: CosyVoice2 with a ~10.8s reference (`ref_single`) reached 0.863 (val96) / 0.842 (fillers) GE2E
cosine speaker_sim, exceeding both the 0.792/0.832 ElevenLabs self-consistency ceiling and
`kokoro_v3_bundle`'s 0.588/0.631, while Chatterbox reached a solid 0.807-0.811 on both reference
conditions -- but CosyVoice2's number carries a real caveat, since its hardened audible-speech gate
failed on 30% of val96 clips and its ~30s `ref_concat` condition hard-errored on 0/196 clips
(CosyVoice2 rejects reference audio over 30s outright), and F5-TTS could not be measured at all
(environment hang). No system came close to the 300ms TTFB target: p50 latency ranged 1.3-2.9
seconds across all three measured variants, 5-15x slower than `kokoro_v3_bundle`'s 185-282ms. Given
a genuine but unreliable zero-shot path clears speaker_sim ≥0.85, the project's success criterion
should be restated as reachable-but-fragile without fine-tuning, and Kokoro's fine-tuning line
should continue since it remains the only production-viable-latency option.

## Sources

* Task: `t0008_tts_eval_harness_baselines`
* URL: https://github.com/SWivid/F5-TTS
* URL: https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B
* URL: https://github.com/resemble-ai/chatterbox
