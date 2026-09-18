# Research Summary — t0021_zero_shot_latency_reduction

## Key Findings (top 10 insights directly actionable for this task)

1. Adopt CosyVoice2's additive latency model as the profiling schema:
   `L_TTS = M·d_lm + M·d_fm + M·d_voc`, extended with reference-encoding and text-frontend stages to
   cover Key Question 1 [Du2024].
2. Vocoder inference is already 150x-3,700x real time unoptimized [Kong2020, Kaneko2022], so `d_lm`
   (AR LM decode) is the likely dominant TTFB term — prioritize LM/decoder acceleration (vLLM,
   TensorRT-LLM, fp16/bf16, torch.compile, paged-KV-cache) over vocoder work.
3. Community RFC `vllm-project/vllm-omni#6870` (non-peer-reviewed, different stack) gives the only
   per-stage ms breakdown: median TTFA 357 ms = prefill 40.5 ms (11%) + AR decode 142.3 ms (40%) +
   first flow chunk 129.0 ms (36%); flow-matching asserts batch size 1 (4x concurrency → only 2.2x
   throughput); 44% of the LM per-token gap is blocking device-to-host syncs (possible cheap win).
4. Streaming removes the *quality* penalty of chunking, not the *latency* — the LM must still
   generate enough tokens before flow-matching runs (explains t0018's 2.86s p50 with `stream=True`).
   Chatterbox streaming is chunk-level, not token-level: a single short filler utterance gets zero
   TTFB benefit unless pre-split into 2+ chunks.
5. RTF is not TTFB: both systems are already faster than real time (RTF 0.40-0.54) while TTFB is
   1.3-2.9s; never treat RTF as a latency proxy.
6. Reference-audio speaker embedding is a deterministic function of the fixed clip — caching it per
   voice is zero-cost/zero-risk and should land first, before other variants [Wan2018].
7. bf16 on the LM/decoder stage is low-risk, H100-confirmed (~40% throughput gain per
   Chatterbox-TTS-Server); use bf16-everywhere by default, treat int8/int4 as LM-only, unverified (a
   prior draft mis-cited dots.tts [Lian2026] for a bf16/int8 split — retracted; float32 only, no
   precision ablation).
8. AR decoding is not *strictly* an architectural floor: Chatterbox-Flash fine-tunes the same T3
   decoder into a block-diffusion decoder, reaching TTFP 103-118 ms / RTF 0.076-0.107 on H100 — but
   needs a new decoding-objective fine-tune, out of scope here. Engineering-only levers should close
   much, not necessarily all, of the gap to 300ms.
9. Cross-system/cross-encoder speaker-similarity numbers are not comparable. Re-score every variant
   with the same GE2E scorer as the t0018 baseline, same session — never benchmark against a paper's
   own published number.
10. No source measures whether acceleration shifts `speaker_sim` at fixed content — fully open; the
    paired same-session measurement here is novel.

## Best Approaches (top 3 recommended implementation approaches from research)

### Approach 1: Instrument first, then cheap/free wins before serving-stack rework

Add `time.perf_counter()` checkpoints in copied adapters around each sub-call (ref encoding, text
frontend, LM prefill/decode, flow-matching, vocoder) to populate `latency_breakdown.json` against
the `L_TTS` schema. Land reference-embedding caching (zero cost) and LM-stage bf16 (~40% gain,
H100-confirmed) before heavier serving-stack changes.

### Approach 2: LM/decoder-stage acceleration as the primary lever

Vocoder cost is negligible and `d_lm` likely dominates — prioritize CosyVoice2's already-wired
`load_jit`/`load_trt`/`fp16` flags and a vLLM-adapted Qwen2.5-0.5B backbone
(`swulling/CosyVoice2-0.5B-vllm`, unbenchmarked) in its own isolated venv, version-checked against
`torch==2.3.1+cu121`. Report the flow-matching batch-1 ceiling as architectural, not a bug.

### Approach 3: Chunking/streaming tuned per system, typical vs. hard prompts

For CosyVoice2, evaluate smaller streaming chunk sizes: near-zero quality cost on typical text but
real cost on hard text (1.45% vs. 6.83%→8.08% CER, [Du2024] Table 8). For Chatterbox, pre-split
short filler text into 2+ chunks before `stream=True`; single-chunk text gets no TTFB benefit.

## Reusable Code / Assets

* `tts_eval_harness` library (t0008) —
  `tasks.t0008_tts_eval_harness_baselines.code.{adapters,harness,scoring,report}`: `SynthResult`,
  `save_wav`, `get_prompts_by_set`, `build_reference_split`, `build_centroid`,
  `compute_speaker_sim`, `compute_wer`, `compute_duration_ratio`, `compute_variant_metrics`. Import
  as-is.
* `t0018/code/adapters_zeroshot.py` (184 lines) — copy; `load_cosyvoice2_model`/`cosyvoice2_synth`
  already wired for `load_jit`/`load_trt`/`fp16`. No adapter records intermediate timing — add
  per-stage checkpoints. Bug: `frontend_zero_shot` crashes on a pre-loaded tensor instead of a file
  path for `prompt_wav`.
* `t0018/code/{run_eval_zeroshot.py, constants.py, paths.py, build_references.py, run_gate_check.py, track_cost.py, report_zeroshot.py}`
  — copy (no library registered). Add `--acceleration-variant` flag/dimension; lower
  `REF_CONCAT_TARGET_DURATION_S` 30.0→29.5 (S-0018-02); update billing anchor/cap to 100.0; add
  stacked-latency and ttfb-by-variant charts (new). `build_references.py` was adapted from `t0014`'s
  simpler `build_reference_concat.py` (context only).
* `t0015/code/audio_quality_check.py` (233 lines) — copy (t0018's cross-task import is now
  disallowed) for `check_audio_quality`/`AudioQualityResult`/`LONGEST_NONSILENT_RUN_THRESHOLD_S`;
  `t0018/code/run_gate_check.py` (92 lines) is the calling pattern.
* Stored t0018 baselines for paired comparison/chart lines: `t0018/results/metrics.json` (e.g.
  `cosyvoice2_ref_single_val96` ttfb_ms=2859.25), `t0018/results/environment.json` (Chatterbox
  torch==2.6.0+cu124 vs. CosyVoice2 torch==2.3.1+cu121 — isolated venvs needed for any vLLM/TensorRT
  install).

## Key Papers (top 5, with finding most relevant to this task)

* **Du et al. 2024 (CosyVoice 2)** — Additive TTS latency model
  (`L_TTS = M·d_lm + M·d_fm + M·d_voc`), the profiling schema to adopt; streaming costs near-zero
  quality on typical text, real cost on hard text.
* **Seo et al. 2026 (Chatterbox-Flash)** — Only paper with paired TTFP/RTF for a streaming zero-shot
  system (103-118 ms TTFP, RTF 0.076-0.107, H100); gap closeable only via fine-tuning.
* **Chen et al. 2024 (F5-TTS)** — Establishes RTF-is-not-TTFB; NAR flow-matching, RTF 0.15, no
  streaming/TTFB story.
* **Wan et al. 2018 (GE2E)** — Defines the GE2E methodology `speaker_sim` implements; confirms
  reference-embedding caching is safe and free.
* **Kunešová et al. 2025** — Speaker-encoder choice materially affects similarity; don't treat
  reference-encoding as a free substitution point.

## Risks Flagged in Research

* No benchmark exists for vLLM/TensorRT-LLM on CosyVoice2's Qwen2.5-0.5B backbone or its TensorRT
  flow-matching export — must be established empirically; community resources are unbenchmarked or
  on a different serving stack.
* No source isolates flow-matching/vocoder precision (fp16/bf16/int8); only LM-stage bf16 is
  confirmed. Don't assume vocoder precision changes are quality-free; the retracted dots.tts
  bf16/int8-split citation must not be reintroduced.
* No peer-reviewed source addresses Chatterbox directly; only community GitHub data on other
  hardware (RTX 4090/3090, not H100) — directional only.
* Whether acceleration shifts `speaker_sim` at fixed content is untested — measure empirically,
  paired, per variant. F5-TTS load-hang cause is also unconfirmed (candidate: `jieba` dict init /
  stale HF cache) for the S-0018-01 retry.
* Current spec forbids direct cross-task `code/` imports (t0018's import of t0015's
  `audio_quality_check.py` no longer valid) — copy, don't import.
* vLLM/TensorRT installs pin their own `torch` version, which may conflict with CosyVoice2's
  `torch==2.3.1+cu121` — check compatibility against the ~1h setup budget.

## Full Detail Available In

* `research/research_papers.md` — 11 papers reviewed, 10 cited.
* `research/research_internet.md` — 30 sources consulted, 15 cited; 21 searches; 8 papers
  discovered.
* `research/research_code.md` — 17 prior tasks reviewed, 5 cited; 2 libraries found, 1 relevant
  (`tts_eval_harness`).
