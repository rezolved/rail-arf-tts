# TTS Research — Kokoro-82M vs ElevenLabs for Rezolve Fillers

## Goal

Replace ElevenLabs David voice (currently used for filler synthesis in Rezolve's voice commerce
assistant) with a self-hosted Kokoro-82M fine-tuned model that matches ElevenLabs speaker
similarity and latency at a fraction of the cost. The current ElevenLabs integration costs $0.30
per 1000 characters; the target is a local model with TTFB ≤ 300 ms and GE2E cosine similarity
≥ 0.85 against the ElevenLabs David reference.

## Scope

### In Scope

* Benchmarking ElevenLabs David baseline: TTFB, speaker similarity, RTF on the 1358-clip filler
  corpus.
* Evaluating Kokoro-82M base (no fine-tuning) on the same filler corpus.
* Fine-tuning Kokoro-82M on David voice data (Stage 2 StyleTTS2, 1557 training clips from
  `data/v4/train/train_list.txt`).
* Evaluating the fine-tuned Kokoro v4 model on val_96 (96 held-out clips) and the 1358 filler
  prompts.
* Producing a viability report: can Kokoro replace ElevenLabs for Rezolve fillers within budget?

### Out of Scope

* Full pipeline integration and A/B testing (follow-up after PoC).
* Non-David voices or multi-speaker generalization.
* Streaming TTS protocol changes.
* STT — see `rezolved/rail-arf-stt` for that project.

## Research Questions

1. What is the TTFB, speaker similarity (GE2E cosine), and RTF of the ElevenLabs David baseline
   on the 1358-clip filler corpus?
2. How does Kokoro-82M base (no fine-tuning) compare to ElevenLabs David on speaker similarity
   and RTF?
3. Does fine-tuning Kokoro-82M on the David voice dataset (1557 clips, StyleTTS2 Stage 2) close
   the speaker-similarity gap to ≥ 0.85 GE2E cosine while keeping TTFB ≤ 300 ms?
4. What is the minimum viable checkpoint (epoch) for the fine-tuned Kokoro v4 that meets the
   quality bar, and does further training help?
5. Is Kokoro-82M v4 a viable ElevenLabs replacement for Rezolve fillers on cost, quality, and
   latency grounds?

## Success Criteria

* ElevenLabs David baseline benchmarked: TTFB, speaker_sim, RTF on ≥ 50 filler prompts.
* Kokoro-82M base benchmarked on the same prompt set.
* Fine-tuned Kokoro v4 achieves speaker_sim (GE2E cosine) ≥ 0.85 on val_96.
* Fine-tuned Kokoro v4 achieves TTFB ≤ 300 ms on the filler corpus (local inference, H100).
* Viability report produced with a clear go / no-go recommendation.

## Key References

* Training data: `data/v4/train/train_list.txt` (1557 clips), `data/v4/val/val_list.txt` (96
  clips).
* ElevenLabs reference audio: `data/11labs_david/` (1358 WAVs).
* Finetune config: `kokoro-finetune/configs/config_david_v4.yml` (Stage 2, starts from
  `first_stage.pth`).
* GPU machine: LLM-T1-NC80 (Nebius, 2×H100 SXM5) — see `project/nebius_vm.json`.
* StyleTTS2 paper: Hu et al. (2023) — "StyleTTS 2: Towards Human-Level Text-to-Speech through
  Style Diffusion and Adversarial Training with Large Speech Language Models".
* Kokoro-82M: open-weight StyleTTS2-based model by hexgrad.

## Current Phase

Setup complete. Next: run ElevenLabs baseline benchmark (t0002) and Kokoro base eval (t0003)
in parallel, then evaluate fine-tuned Kokoro v4 once Stage 2 training completes (t0004).
