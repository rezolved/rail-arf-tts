---
spec_version: "3"
paper_id: "10.48550_arXiv.2604.16395"
citation_key: "Bachkaniwala2026"
summarized_by_task: "t0021_zero_shot_latency_reduction"
date_summarized: "2026-09-18"
---
# Stream2LLM: Overlap Context Streaming and Prefill for Reduced Time-to-First-Token (TTFT)

## Metadata

* **File**: `files/bachkaniwala_2026_stream2llm.pdf`
* **Published**: 2026
* **Authors**: Rajveer Bachkaniwala 🇺🇸, Chengqi Luo 🇺🇸, Richard So 🇺🇸, Divya Mahajan 🇺🇸, Kexin Rong
  🇺🇸
* **Venue**: 9th MLSys Conference (MLSys 2026)
* **DOI**: `10.48550/arXiv.2604.16395`

## Abstract

Context retrieval systems for LLM inference face a critical challenge: high retrieval latency
creates a fundamental tension between waiting for complete context (poor time-to-first-token) and
proceeding without it (reduced quality). Streaming context incrementally-overlapping retrieval with
inference-can mitigate this latency, but doing so with concurrent requests introduces new
challenges: requests contend for GPU compute and memory, and scheduling must adapt to dynamic
context arrivals. We present STREAM2LLM, a streaming-aware LLM serving system for concurrent
prefill-decode disaggregated deployments. STREAM2LLM introduces adaptive scheduling and preemption
for two distinct retrieval patterns: append-mode (progressive context accumulation) and update-mode
(iterative refinement with cache invalidation). It decouples scheduling decisions from resource
acquisition, enabling flexible preemption strategies guided by hardware-specific cost models, and
uses longest common prefix matching to minimize redundant computation when input changes
dynamically. To evaluate STREAM2LLM, we collect two large-scale, real-world streaming workloads
based on web crawling and approximate nearest neighbor search. Our evaluation demonstrates that
streaming architecture delivers up to 11x TTFT improvements, with cost-aware scheduling providing
critical benefits under memory pressure, all while maintaining throughput parity with non-streaming
baselines.

## Overview

Stream2LLM targets a mismatch that shows up in any pipeline where an LLM (or, by extension, any
autoregressive token-predicting model) must wait on an external, latency-heavy retrieval step before
it can begin producing output. The authors observe that retrieval latency — web crawling (200 ms to
several seconds per document) or disk-based approximate nearest-neighbor search (ANNS, 100 ms to
several seconds) — is frequently the dominant term in time-to-first-token (TTFT), not model compute.
Rather than choosing between "wait for full context" (bad TTFT) or "start with partial context" (bad
quality), the paper streams retrieved chunks into the prefill stage as they arrive, so prefill
computation overlaps with retrieval instead of following it serially.

The paper's core contribution is not the streaming idea itself (prior single-request systems PipeRAG
and AquaPipe already did this) but a serving-system redesign that makes streaming safe and effective
under *concurrent* multi-request load on shared GPUs. The authors identify three
concurrency-specific problems: GPU KV-cache memory contention across simultaneously streaming
requests, the need to dynamically re-prioritize requests as chunks arrive at different times (which
classic FCFS/progress-based schedulers cannot express), and two structurally different retrieval
patterns — append-mode (monotonically growing context, e.g. web crawling) and update-mode (context
is replaced/refined over time, e.g. iterative ANNS) — that demand different cache-invalidation and
preemption behavior.

Methodologically, Stream2LLM is built as an extension of the vLLM v1 engine's scheduler, targeting
the prefill instance of a prefill-decode disaggregated deployment (so TPOT/decode latency is out of
scope; only TTFT and prefill-stage throughput are evaluated). It introduces a two-phase scheduler
(priority ordering + feasibility check, then resource acquisition + adaptive preemption) decoupled
from a longest-common-prefix (LCP) cache-invalidation mechanism that avoids discarding KV-cache
blocks unaffected by an input update, and a hardware-profiled cost model that chooses between
recomputing or CPU-swapping evicted KV blocks. Four scheduling policies (DEFAULT vLLM, FCFS, MCPS,
LCAS) are implemented and compared under this architecture.

The evaluation is run on two real, large-scale streaming traces (a 4,322-query SimpleQA web-crawl
workload and a 500-query SQuAD-over-Fineweb-edu DiskANN workload) on NVIDIA H100/H200 GPUs with
Llama-3.1-8B-Instruct, and shows that streaming alone yields up to 11x TTFT improvement at high
load, but that scheduler and eviction-strategy choice becomes decisive once GPU memory is under
pressure — a poorly chosen policy can make streaming *worse* than not streaming at all (down to
0.18-0.19x of the non-streaming baseline at P99 for the default vLLM scheduler under memory
pressure).

## Architecture, Models and Methods

Stream2LLM extends vLLM v1's `EngineCoreRequest` with three flags — `is_streaming_prompt`,
`is_streaming_prompt_finished`, and `is_prompt_update` — plus driver-level `new_stream`, `append`,
and `update` calls, so a client can either append new chunks to a growing sequence (append mode) or
replace parts of the sequence as retrieval results are refined (update mode). Internally, the
scheduler is restructured into two phases: Phase 1 computes a priority ordering over all unfinished
requests via a pluggable scheduling algorithm and performs a feasibility check against the token
budget (2048-8192 tokens/step in the experiments) and free GPU KV-cache blocks, without mutating any
state. Phase 2 performs actual GPU block allocation for the requests selected in Phase 1, and — if
allocation fails due to memory exhaustion — preempts the lowest-priority request from a
`not_scheduled_reqs` list, choosing between recomputation and CPU swap using a per-GPU cost model:
`recomputation_latency(T)` is a piecewise-linear fit of measured prefill latency for 1K-128K tokens
on each target GPU (A40, H100, H200), and `swap_latency(C)` is derived from measured PCIe transfer
throughput for the model's KV block size (2 MB blocks for Llama-3.1-8B at the default block size of
16 tokens). The system selects whichever strategy has lower predicted latency, comparing
`Crecomp(r)` against `2 x Cswap(r)` to account for bidirectional transfer.

Cache invalidation on input updates uses longest-common-prefix (LCP) matching between the old and
new token sequence: only KV-cache blocks for tokens beyond the LCP are freed and marked for
recomputation; blocks within the LCP are preserved. Four scheduling policies are compared: DEFAULT
vLLM (FIFO arrival order, LIFO eviction, ignores chunk-arrival timing), FCFS (two-tier: full vs.
partial requests, eviction in reverse scheduling order), MCPS ("Most Chunks Processed Scheduling",
priority = `num_computed_tokens`, which collapses badly in update mode when an update resets a
request's progress via a short LCP), and LCAS ("Last Chunk Arrival Scheduling", priority = most
recent chunk arrival time, with a two-tier complete/partial split).

Evaluation hardware: NVIDIA H200 (141 GB) and H100 (80 GB) GPUs, tensor parallelism = 2, GPU memory
utilization target 80%. Model: Llama-3.1-8B-Instruct. Update-mode workload: Fineweb-edu corpus (372
GB text index / 279 GB vectors) embedded with e5-base-v2, indexed with DiskANN (L2 distance, beam
width W=8, search list size L=10000), queried with SQuAD questions (500 queries) using AquaPipe's
recall-aware prefetching for partial-result emission. Append-mode workload: crawl4ai-based web
crawling to depth 2 with BM25 content filtering, driven by 4,327 SimpleQA fact-seeking queries
(4,322 used). Baselines: vLLM-NS (no streaming) and vLLM-S (default vLLM scheduler with streaming
support). Memory-pressure experiments scale chunk delays 10x (crawler) and 30x (ANNS) to saturate
the KV-cache pool. Scheduler sorting overhead is measured directly: 15-16 microseconds for FCFS/LCAS
and 12-13 microseconds for MCPS at 50 concurrent requests, with P99 below 165 microseconds at 500
requests.

## Results

* Streaming improves crawler-workload (append-mode) TTFT by **3.9-4.3x** at low load (QPS 0.5-1.0)
  and by **10.8-11.0x** at high load (QPS 4.0), median latency, vs. non-streaming.
* ANNS-workload (update-mode) streaming achieves **2.49-2.63x** P95 TTFT speedup at QPS 1.0, and
  **1.30-1.42x** P50 advantage across all streaming schedulers even with >10% of requests
  invalidating over 10,000 tokens.
* At QPS 2.0 on the ANNS workload, FCFS reaches **2.26x** P95 speedup over non-streaming vs.
  **1.81x** for LCAS and **1.53x** for MCPS.
* Under crawler memory pressure (4.0 QPS, 10x chunk delays), the default vLLM streaming scheduler
  degrades to **0.71x** (worse than non-streaming) at P99 tail latency, while FCFS with cost-based
  eviction achieves **8.62x** and LCAS achieves **9.14x** at P99.
* Under ANNS memory pressure (2.0 QPS, 30x chunk delays), default vLLM streaming collapses to
  **0.19x** at P99 (catastrophic tail-latency degradation), while FCFS with cost-based eviction
  still reaches **2.04x**.
* Eviction-strategy choice alone shifts P99 speedup substantially: on the crawler workload, FCFS
  achieves **10.03x** with recompute-only eviction vs. only **6.69x** with swap-only eviction;
  cost-based eviction balances the two at **8.62x**.
* Scheduler sorting overhead is negligible: **15-16 microseconds** per scheduling step for FCFS/LCAS
  at 50 concurrent requests, with P99 staying below **165 microseconds** even at 500 requests.
* Streaming introduces essentially **no throughput cost**: trace completion time curves for all
  scheduler variants (including non-streaming) are visually indistinguishable across QPS levels.
* Crawler-workload preemption counts range from **779 to 12,564** total preemptions depending on
  scheduler; cost-based eviction allocates **17-21%** to swap and **79-83%** to recompute.
* ANNS-workload cost-based eviction favors recompute almost exclusively (**98-100%** of
  preemptions), confirming that after context invalidation the remaining valid KV cache is small
  enough that recomputation is nearly always cheaper than swapping.

## Innovations

### Two-Phase Scheduling Architecture

Decouples the decision of *which request to run* (priority ordering and feasibility analysis, no
state mutation) from *how to allocate resources* (GPU block allocation and preemption mechanics).
This separation lets the system plug in different scheduling policies (FCFS, MCPS, LCAS) without
hardcoding preemption behavior into each one, and lets cache invalidation happen at a precise
boundary between the two phases rather than being tangled inside a monolithic scheduling loop.

### Longest-Common-Prefix (LCP) Cache Invalidation for Dynamic Inputs

A single, unified mechanism handles both append-mode growth and update-mode replacement of the input
sequence: compute the LCP between the old and new token sequence, invalidate only KV-cache blocks
beyond that point, and preserve everything within it. This avoids both failure modes of naive
alternatives — invalidating everything on every update (wasteful) or reusing cache without
verification (correctness risk from stale attention over outdated key-value pairs).

### Hardware-Profiled Cost-Based Preemption

Rather than a fixed swap-vs-recompute policy, the system profiles per-GPU recomputation and PCIe
swap latency curves offline (piecewise-linear fits over 1K-128K tokens, ~5 minutes per GPU
configuration) and picks whichever strategy is cheaper at preemption time. This is shown to
materially change outcomes: a fixed recompute-only or swap-only policy loses several points of P99
speedup relative to the cost-based choice, and the optimal strategy differs sharply between
append-mode (balanced 17-21% swap) and update-mode (98-100% recompute) workloads.

### Concurrent Support for Two Retrieval Patterns Under One Framework

Prior systems (PipeRAG for append-style pipeline retrieval, AquaPipe for update-style recall-aware
ANNS prefetching) each handled only one retrieval pattern and only single-request (B=1) settings.
Stream2LLM is the first system, per the authors, to support both patterns concurrently with multiple
simultaneous requests contending for shared GPU resources.

## Datasets

* **Update-mode workload**: Fineweb-edu corpus (text index 372 GB, vector index 279 GB), embedded
  with e5-base-v2, indexed with DiskANN (L2 distance, beam width W=8, search list size L=10000);
  queries drawn from SQuAD (Rajpurkar et al., 2016), 500 queries replayed at 0.25-2.0 QPS. Mean 13K
  tokens/query (P95 31K); mean retrieval latency 4.5 s (P95 8.5 s).
* **Append-mode workload**: web pages retrieved via the crawl4ai library (depth-2 crawl, BM25
  content filtering/deduplication); queries drawn from OpenAI's SimpleQA dataset (Wei et al., 2024),
  4,322 of 4,327 fact-seeking questions used, replayed at 0.5-4.0 QPS. Mean 9.1K tokens/query (P95
  28.9K); mean retrieval latency 9.9 s (P95 16.7 s).
* Both traces are collected by the authors specifically because, per the paper, no large-scale
  public streaming-retrieval workloads with full response and timing information existed; the traces
  themselves (and pre-computed run logs) are released as a HuggingFace dataset
  (`rbachkaniwala3/stream2llm-data`) under CC-BY-4.0, alongside an MIT-licensed code artifact
  (`github.com/rajveerb/stream2llm`), archived at `doi.org/10.5281/zenodo.18906769`.

## Main Ideas

* The core lesson — overlap a slow, external, chunked-arrival latency source with model compute
  instead of serializing them — is a general TTFT-reduction pattern, not specific to text LLMs or to
  retrieval. For this project it reframes what "TTFB" bottlenecks might look like in a Kokoro-82M
  serving pipeline: any stage that currently blocks the vocoder/decoder from starting (e.g. text
  normalization, phonemization, reference-embedding lookup, or a first-chunk style-vector
  computation) is a candidate for the same append-mode streaming treatment.
* Streaming without policy-aware scheduling can be actively harmful under load: the default
  scheduler in this paper degraded to 0.18-0.19x of the non-streaming baseline at P99 under memory
  pressure. If any zero-shot-latency work in this project pipelines TTS stages, it must be paired
  with an explicit admission/priority policy, not just "start early," or tail latency can regress.
* The paper demonstrates that streaming/overlap optimizations can be throughput-neutral (trace
  completion time curves were indistinguishable across all scheduler variants) — useful evidence
  that TTFB-focused latency work does not have to trade off against RTF/throughput if designed
  carefully, which matches this project's dual TTFB/RTF success criteria.
* The hardware-profiled cost model for recompute-vs-swap preemption (piecewise-linear fit, ~5
  minutes of offline profiling per GPU) is a lightweight, reusable pattern that could inform how
  this project profiles H100-specific latency curves (e.g. for chunked inference or KV-cache style
  decisions) when characterizing Kokoro TTFB under load.
* This is a general-purpose LLM-serving/RAG paper with no TTS, speech-token, or speaker-similarity
  evaluation; none of its quantitative results (11x TTFT, cost-model numbers) transfer directly to
  this project's ElevenLabs/Kokoro benchmark — it is relevant only as a transferable systems
  technique, not as a directly comparable baseline or dataset.

## Summary

Stream2LLM addresses the problem of high time-to-first-token (TTFT) in LLM serving pipelines that
depend on slow, external context retrieval (web crawling or disk-based ANNS), where retrieval
latency — hundreds of milliseconds to multiple seconds — typically dominates total TTFT more than
model compute does. The motivating tension is that waiting for complete context hurts responsiveness
while starting generation on partial context can hurt output quality; the paper's premise is that
streaming retrieved context incrementally into inference, so retrieval and prefill overlap in time,
can resolve this tension without sacrificing either.

The authors build Stream2LLM as an extension to the vLLM v1 serving engine's scheduler, targeting
the prefill stage of a prefill-decode disaggregated deployment. The key design decisions are a
two-phase scheduling architecture that separates priority-based request ordering from GPU memory
allocation and preemption (enabling pluggable scheduling policies without hardcoded preemption
rules), a longest-common-prefix (LCP) KV-cache invalidation scheme that handles both
monotonically-growing (append-mode) and iteratively-refined (update-mode) retrieval patterns with a
single mechanism, and a hardware-profiled cost model that chooses between recomputing or
CPU-swapping evicted KV-cache blocks based on measured per-GPU latency curves. Four scheduling
policies (DEFAULT vLLM, FCFS, MCPS, LCAS) are implemented and compared.

The headline finding is that streaming delivers up to 11x TTFT improvement over non-streaming
baselines at high load on a real, 4,322-query web-crawl workload, and 2.49-2.63x P95 speedup on a
500-query ANNS workload, with essentially zero throughput cost. The more consequential finding,
however, is that scheduler and eviction-policy choice is largely irrelevant when GPU memory is
abundant but becomes decisive under memory pressure: the naive default vLLM streaming scheduler
suffers catastrophic P99 tail-latency degradation (down to 0.18-0.19x of the non-streaming baseline)
under contention, while cost-aware scheduling (FCFS/LCAS with hardware-profiled recompute-vs-swap
decisions) sustains 8-9x P99 speedups under the same conditions. This establishes that streaming
alone is necessary but not sufficient — it requires concurrency-aware scheduling and cache
management to be safe in production.

For this project, Stream2LLM is not a directly applicable baseline or dataset (it targets text-LLM
serving with disaggregated prefill-decode, not TTS or speech-token generation, and reports no
speaker-similarity, TTFB, or RTF numbers comparable to the ElevenLabs/Kokoro benchmark), but its
central technique — overlapping a slow, chunked-arrival latency source with the start of model
computation, governed by a concurrency- and memory-aware scheduler — is a transferable pattern for
zero-shot latency reduction work in this project. If any stage of the Kokoro-82M inference pipeline
(e.g., style/reference embedding extraction, phonemization, or chunked decoding) can be restructured
to begin before an upstream input is fully available, this paper is evidence that the gain can be
large (order-of-magnitude TTFB reduction) but only if paired with an explicit scheduling and
preemption policy, since naive early-start approaches can regress tail latency under load rather
than improving it.
