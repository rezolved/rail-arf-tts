---
spec_version: "3"
task_id: "t0021_zero_shot_latency_reduction"
step_number: 5
step_name: "research-internet"
status: "completed"
started_at: "2026-09-18T10:27:19Z"
completed_at: "2026-09-18T10:57:18Z"
---
## Summary

Ran the `/research-internet` skill in a dedicated subagent to fill the six literature gaps flagged
by `research_papers.md`, producing `research/research_internet.md` (21 search queries, 15 sources, 8
discovered papers). All 8 discovered papers were added to the corpus via 8 parallel `/add-paper`
subagents (max 3 concurrent), and one factual error surfaced by a paper-addition agent's full-text
read was corrected in the research file before this step was closed.

## Actions Taken

1. Ran `prestep` for `research-internet`, then spawned a dedicated subagent to execute the
   `/research-internet` skill unrestricted, per Critical Rule 9/10. It produced
   `research/research_internet.md` addressing all six gaps from `research_papers.md` (vLLM/TensorRT
   serving benchmarks, per-stage ms breakdown, precision ablations, direct Chatterbox data, F5-TTS
   hang diagnosis, and acceleration's effect on speaker_sim), citing a `vllm-project/vllm-omni` RFC
   (issue #6870) as the closest available per-stage TTFA decomposition (357 ms median: 40.5 ms
   prefill, 142.3 ms AR decode, 129.0 ms first flow chunk) — narrowing but not closing Gap 2.
2. Ran `verify_research_internet` — passed with zero errors and zero warnings.
3. Parsed the `## Discovered Papers` section (8 papers) and cross-checked each against the 11-paper
   corpus via `aggregate_papers.py` (no title/DOI collisions). Spawned 8 `/add-paper` subagents in 3
   batches (max 3 concurrent, as required), covering FlashTTS, DiFlow-TTS, VocalNet-MDM, an
   ultra-low-latency block-wise/depth-wise-codec architecture, Stream2LLM, Qwen3-TTS, dots.tts, and
   an IEEE INT8 quantization paper (the last downloaded with `download_status: "failed"` since IEEE
   Xplore blocked automated access — metadata resolved via CrossRef/OpenAlex/Semantic Scholar
   instead, per the skill's no-fabrication fallback). One agent (block-wise-codec paper) self-
   committed its asset (`085ae65`); the other 7 were left untracked for this step's commit, as
   instructed.
4. Verified all 8 paper assets individually with
   `meta.asset_types.paper.verificator --task-id t0021_zero_shot_latency_reduction` — all passed (7
   with zero errors/warnings; the IEEE paper passed with one expected `PA-W002` warning, an empty
   Results section, because its full text could not be retrieved).
5. The dots.tts (`Lian2026`) add-paper agent flagged, after reading the full 22-page PDF, that the
   paper contains no bf16-acoustic/quantized-LLM-trunk precision claim — contradicting what
   `research_internet.md` had cited it for in Gap 3, Methodology Insights, and Recommendation 3.
   Independently confirmed via `pdftotext` + grep on the downloaded PDF (only `float32` /
   `torch.compile disabled` appear; no `bf16` or precision-ablation text). Corrected all four
   affected passages in `research_internet.md` (Gap 3, the Methodology Insights bf16 bullet,
   Recommendation 3, the Discovered-Papers "why download" line, and the `[Lian2026]` Source Index
   relevance line) to state the citation was wrong and remove the unsupported claim, then re-ran
   `flowmark` and `verify_research_internet` — still passes with zero errors/warnings.

## Outputs

* `tasks/t0021_zero_shot_latency_reduction/research/research_internet.md` — 8 mandatory sections, 21
  queries, 15 sources, 8 discovered papers; corrected post-hoc for the `Lian2026` citation error.
* `tasks/t0021_zero_shot_latency_reduction/assets/paper/10.48550_arXiv.2606.09141/` — FlashTTS.
* `tasks/t0021_zero_shot_latency_reduction/assets/paper/10.48550_arXiv.2509.09631/` — DiFlow-TTS.
* `tasks/t0021_zero_shot_latency_reduction/assets/paper/10.48550_arXiv.2602.08607/` — VocalNet-MDM.
* `tasks/t0021_zero_shot_latency_reduction/assets/paper/10.48550_arXiv.2604.12438/` — block-wise
  streaming TTS architecture (committed separately as `085ae65` by its own add-paper agent).
* `tasks/t0021_zero_shot_latency_reduction/assets/paper/10.48550_arXiv.2604.16395/` — Stream2LLM.
* `tasks/t0021_zero_shot_latency_reduction/assets/paper/10.48550_arXiv.2601.15621/` — Qwen3-TTS.
* `tasks/t0021_zero_shot_latency_reduction/assets/paper/10.48550_arXiv.2606.07080/` — dots.tts.
* `tasks/t0021_zero_shot_latency_reduction/assets/paper/10.1109_ICMI68585.2026.11539760/` — IEEE
  quantization paper (download failed, metadata-only).

## Issues

One factual error was found and corrected: an earlier draft of `research_internet.md` attributed a
bf16-acoustic/quantized-LLM-trunk precision design to `[Lian2026]` (dots.tts) that the paper does
not contain. Caught by a downstream `/add-paper` subagent's full-text read, independently confirmed
via `pdftotext`, and fixed in this step before commit — see Actions Taken item 5. No other issues
encountered.
