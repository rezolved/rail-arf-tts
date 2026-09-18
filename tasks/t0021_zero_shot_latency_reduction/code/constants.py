"""Named constants for t0021 zero-shot latency reduction."""

from typing import Final

# ── System identifiers (cloning systems under test) ───────────────────────────
SYSTEM_F5_TTS: Final[str] = "f5_tts"
SYSTEM_COSYVOICE2: Final[str] = "cosyvoice2"
SYSTEM_CHATTERBOX: Final[str] = "chatterbox"

CLONING_SYSTEMS: Final[list[str]] = [SYSTEM_F5_TTS, SYSTEM_COSYVOICE2, SYSTEM_CHATTERBOX]

# ── Reference-audio conditions ────────────────────────────────────────────────
CONDITION_REF_SINGLE: Final[str] = "ref_single"
CONDITION_REF_CONCAT: Final[str] = "ref_concat"

REFERENCE_CONDITIONS: Final[list[str]] = [CONDITION_REF_SINGLE, CONDITION_REF_CONCAT]

# ── Acceleration variants (cumulative stack per system; Step 8) ──────────────
VARIANT_BASELINE_NEW_REF: Final[str] = "baseline_new_ref"
VARIANT_REF_CACHE: Final[str] = "ref_cache"
VARIANT_FP16: Final[str] = "fp16"
VARIANT_LOAD_JIT: Final[str] = "load_jit"
VARIANT_LOAD_TRT: Final[str] = "load_trt"
VARIANT_VLLM_BACKEND: Final[str] = "vllm_backend"

COSYVOICE2_VARIANTS: Final[list[str]] = [
    VARIANT_BASELINE_NEW_REF,  # load_jit=False, load_trt=False, fp16=False, no ref-cache
    VARIANT_REF_CACHE,  # + cached speaker embedding/prompt tokens (computed once)
    VARIANT_FP16,  # + fp16=True
    VARIANT_LOAD_JIT,  # + load_jit=True
    VARIANT_LOAD_TRT,  # + load_trt=True (full local-optimization stack)
    VARIANT_VLLM_BACKEND,  # separate stack: ref_cache + vLLM Qwen2.5-0.5B LM backend
]

VARIANT_PRECISION_BF16_OR_FP16: Final[str] = "precision_bf16_or_fp16"
VARIANT_TORCH_COMPILE: Final[str] = "torch_compile"
VARIANT_SENTENCE_CHUNKING: Final[str] = "sentence_chunking"
VARIANT_STREAMING_API: Final[str] = "streaming_api"

CHATTERBOX_VARIANTS: Final[list[str]] = [
    VARIANT_BASELINE_NEW_REF,  # whole-utterance, no cache, fp32, no compile, no chunking
    VARIANT_REF_CACHE,  # + cached voice-conditioning embedding
    VARIANT_PRECISION_BF16_OR_FP16,  # + bf16 (fallback fp16 if bf16 unsupported)
    VARIANT_TORCH_COMPILE,  # + torch.compile on the T3 decoder
    VARIANT_SENTENCE_CHUNKING,  # + pre-split text, first chunk returned first (full stack)
    VARIANT_STREAMING_API,  # separate: native streaming API at pinned 0.1.7, ref_cache applied
]

# ── Smoke gate / rejection thresholds ─────────────────────────────────────────
SMOKE_GATE_TIMEBOX_MINUTES: Final[int] = 45
SUCCESS_RATE_THRESHOLD: Final[float] = 0.8

# ── Fixed gate texts (shared with t0014/t0015/t0018 for cross-task comparability) ──
GATE_TEXT_NAMES: Final[tuple[str, ...]] = (
    "lining_up_suggestions_17",
    "lining_up_suggestions_10",
    "putting_them_head_to_head_15",
)

# ── Comparison-set sampling (same seed/count as t0018 for continuity) ────────
COMPARISON_SET_SEED: Final[int] = 42
COMPARISON_SET_VAL96_COUNT: Final[int] = 7

# ── Cost tracking ──────────────────────────────────────────────────────────────
# Source: tasks/t0021_zero_shot_latency_reduction/logs/steps/008_setup-machines/machine_log.json
VM_HOURLY_COST_USD: Final[float] = 13.96
VM_BILLING_ANCHOR_ISO: Final[str] = "2026-09-18T11:29:10.875712Z"
BUDGET_HARD_CAP_USD: Final[float] = 100.0
BUDGET_NEW_VARIANT_STOP_USD: Final[float] = 90.0
# Confirmed non-billing gap: the idle watchdog stopped LLM-T1-NC80 after the mid-implementation
# agent crash, and it was not billing from this stop until the successful re-acquire. Naive
# (now - VM_BILLING_ANCHOR_ISO) wall-clock elapsed time overcounts cost by this amount unless
# subtracted. Source: tasks/t0021_zero_shot_latency_reduction/intervention/
# vm_idle_after_agent_crash_and_watchdog_fix.md ("13:53:29Z" stop -> "14:19:58Z" successful
# re-acquire).
VM_CONFIRMED_DOWNTIME_SECONDS: Final[float] = 1589.0

# ── Warmup ─────────────────────────────────────────────────────────────────────
N_WARMUP: Final[int] = 50
WARMUP_TEXT: Final[str] = "Checking the latest press release."

# ── Reference-audio construction (Milestone 1; val_96-sourced, owner correction) ──
# Seed itself is imported from tasks.t0008_tts_eval_harness_baselines.code.constants.RANDOM_SEED
# (same seed t0018 used) rather than redefined here, to keep exactly one source of truth.
CENTROID_HALF_SIZE: Final[int] = 48  # disjoint from ref_source_half (owner correction #2)
REF_SINGLE_MAX_DURATION_S: Final[float] = 10.0
REF_CONCAT_TARGET_DURATION_S: Final[float] = 29.5  # S-0018-02 fix (was 30.0 in t0018)
# Hard safety ceiling, strictly below CosyVoice2's internal 30.0s assertion (S-0018-02's actual
# failure mode: t0018's "stop once total first EXCEEDS target" method overshot 30.0 -> 30.57s.
# Simply lowering the target to 29.5 does not, by itself, guarantee staying under 30s if a
# single clip is long enough to push the running total past it in one step — this happened
# during this task's own preflight run against val_96's real (longer, ~4.3s avg) clips. The
# concat builder therefore does a look-ahead check against this ceiling, not just the target.
REF_CONCAT_HARD_CEILING_S: Final[float] = 29.8
REF_CONCAT_SILENCE_GAP_S: Final[float] = 0.2

# ── Baseline numbers from t0018 (measured against the WRONG David voice; for chart
# continuity/reference lines ONLY — never presented as this task's own numbers) ──
# Source: tasks/t0018_zero_shot_cloning_calibration/results/metrics.json
T0018_COSYVOICE2_SPEAKER_SIM_VAL96: Final[float] = 0.8627852474649748
T0018_COSYVOICE2_SPEAKER_SIM_FILLERS: Final[float] = 0.8420862078666687
T0018_COSYVOICE2_TTFB_MS_VAL96: Final[float] = 2859.2522075005036
T0018_COSYVOICE2_TTFB_MS_FILLERS: Final[float] = 1617.0282700004464
T0018_CHATTERBOX_SPEAKER_SIM_VAL96: Final[float] = 0.8073432354987422
T0018_CHATTERBOX_SPEAKER_SIM_FILLERS: Final[float] = 0.8112275004386902
T0018_CHATTERBOX_TTFB_MS_VAL96: Final[float] = 1438.52208250064
T0018_CHATTERBOX_TTFB_MS_FILLERS: Final[float] = 1344.3423864991928

# ── Product target ────────────────────────────────────────────────────────────
TTFB_TARGET_MS: Final[float] = 300.0
