---
spec_version: "2"
answer_id: "zero-shot-speaker-sim-ceiling"
answered_by_task: "t0018_zero_shot_cloning_calibration"
date_answered: "2026-09-17"
confidence: "medium"
---
## Question

What speaker_sim and latency envelope is reachable on the David voice by zero-shot cloning, and what
does that imply for the Kokoro fine-tuning line?

## Short Answer

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

## Research Process

Ran the t0008 `tts_eval_harness` protocol (smoke gate -> 50 discarded warmup -> 196 measured
prompts, val96 + fillers) against three named zero-shot voice-cloning systems (F5-TTS, CosyVoice2,
Chatterbox) on `LLM-T1-NC80` (2xH100), each under two reference-audio conditions built from half-A
of the ElevenLabs David corpus only (`ref_single` ~10.8s, `ref_concat` ~30.6s), plus two paired
baselines re-measured in the same session where possible (`elevenlabs_david`, `kokoro_v3_bundle`).
All measured speaker_sim/duration_ratio/WER values were computed locally from the actual synthesized
audio (`code/merge_and_score.py`), not estimated. Two systems hit hard environment/model limits
during the run (documented in `intervention/`, not silently substituted): F5-TTS's model loading
hung indefinitely across three independent attempts (including one with `HF_HUB_OFFLINE=1` to rule
out network causes), and `kokoro_v3_bundle`'s re-synthesis hit the identical hang pattern even after
copying its checkpoint to local disk to rule out network-filesystem I/O -- both fall back to
documented, non-silent resolutions rather than guessed numbers.

## Evidence from Papers

Not used as primary evidence for this answer (`answer_methods` excludes `papers`): this task's paper
corpus (F5-TTS/CosyVoice2 papers, discovered during planning) reports SIM-o/SS metrics on a
WavLM-based embedding, not this project's GE2E cosine `speaker_sim` -- the two are never comparable,
so no paper number is cited as if it were a `speaker_sim` measurement.

## Evidence from Internet Sources

Research (task's own `research/research_internet.md`, step 5) established before the run that:
F5-TTS's official checkpoint is CC-BY-NC-4.0 (non-commercial, survives fine-tuning) while CosyVoice2
is Apache-2.0 and Chatterbox is MIT; only CosyVoice2 documents genuine chunked streaming (vendor
claim ~150ms first packet); F5-TTS's standard pipeline auto-crops reference audio over ~15s. This
run's own measurement (below) shows CosyVoice2's real single-stream TTFB p50 (1.6-2.9s) is nowhere
near the vendor's ~150ms claim under this task's harness and hardware, and that F5-TTS's
crop-vs-error question was never reached because the system could not be loaded at all.

## Evidence from Code or Experiments

### Speaker similarity (GE2E cosine vs the half-A centroid; ElevenLabs vs half-B)

| Variant | speaker_sim (fillers) | speaker_sim (val96) | vs ElevenLabs ceiling | vs kokoro_v3_bundle |
| --- | --- | --- | --- | --- |
| ElevenLabs (self-consistency ceiling) | 0.832 | 0.792 | -- | +0.201 / +0.204 |
| CosyVoice2 `ref_single` | 0.842 | **0.863** | +0.010 / **+0.071** | +0.211 / +0.275 |
| CosyVoice2 `ref_concat` | NULL (0/196) | NULL (0/196) | -- | -- |
| Chatterbox `ref_single` | 0.811 | 0.807 | -0.021 / +0.015 | +0.180 / +0.219 |
| Chatterbox `ref_concat` | 0.796 | 0.810 | -0.036 / +0.018 | +0.165 / +0.222 |
| F5-TTS (both conditions) | NULL | NULL | -- | -- |
| `kokoro_v3_bundle` (t0008 stored, not re-measured) | 0.631 | 0.588 | -0.201 / -0.204 | -- |

Source: `results/tables.json` (this task); ElevenLabs/`kokoro_v3_bundle` baseline numbers
cross-checked directly against `tasks/t0008_tts_eval_harness_baselines/results/metrics.json`.

**CosyVoice2's val96 number needs a load-bearing caveat.** Its hardened audible-speech gate
(`results/gate_failures.json`) failed 29/96 (30%) val96 clips -- vs 0/96 for both Chatterbox
conditions and 2/96 for ElevenLabs. The gate-passed-only speaker_sim mean is 0.859 (n=67, barely
below the 0.863 unfiltered mean), so the headline number is not simply an artifact of counting
broken clips as high-similarity -- when CosyVoice2 works, it is genuinely a strong voice match. But
`duration_ratio_median` for CosyVoice2 val96 is **2.76** (should be ~1.0; Chatterbox's val96 medians
are 0.86-0.97) with a 2% explosion fraction, and mean WER among the clips that passed the
duration-ratio gate is 3.47 -- both signs of frequent, severe content-duplication/garbling on longer
(val96) sentences specifically, distinct from the fillers condition (median ratio 4.13 is even
higher, but with a smaller 15% explosion fraction and no WER computed there since t0008's own
duration-ratio-before-WER gate skipped nearly everything). **Read this as: CosyVoice2 can beat the
ElevenLabs ceiling on voice similarity for the clips it renders correctly, but is measurably less
reliable at rendering correctly than either Chatterbox or ElevenLabs on this corpus.**

### Latency and RTF

(All whole-utterance or single-stream first-chunk; TTFB/RTF measured in this task's own session, per
Lesson 1.)

| Variant | TTFB p50 (ms) | TTFB p95 (ms) | RTF (mean) | Streaming? |
| --- | --- | --- | --- | --- |
| CosyVoice2 `ref_single` (val96) | 2859 | 3392 | 0.479 | Yes (measured first-chunk) |
| CosyVoice2 `ref_single` (fillers) | 1617 | 2000 | 0.405 | Yes |
| Chatterbox `ref_single` (val96) | 1439 | 2879 | 0.569 | No (whole-utterance) |
| Chatterbox `ref_single` (fillers) | 1344 | 1958 | 0.934 | No |
| Chatterbox `ref_concat` (val96) | 1823 | 3506 | 0.634 | No |
| Chatterbox `ref_concat` (fillers) | 1480 | 2014 | 0.904 | No |
| `kokoro_v3_bundle` (t0008 stored) | 185 (fillers) / 282 (val96) | -- | -- | Yes |

Source: `results/tables.json`. **None of the three cloning variants measured here meet the 300ms
TTFB target** -- even CosyVoice2, the one genuinely-streaming system among them, posts a p50 of
1.3-2.9 seconds under this harness's warmup-then-measure protocol on this VM, 9-19x the target and
far above its own vendor's ~150ms marketing claim (a gap this task's research phase flagged as a
real risk before running: "re-measure, don't quote the vendor figure"). Chatterbox and F5-TTS are
non-streaming by design (a single `generate()` call / non-autoregressive full-pass respectively) so
their TTFB is definitionally whole-utterance latency, labelled as such throughout
`results/tables.json` via `is_streaming: false`.

### Reference-condition effect (Key Question 4: `ref_single` vs `ref_concat`)

Only Chatterbox has real data for both conditions (CosyVoice2's `ref_concat` is a hard-error null;
F5-TTS never ran). Chatterbox's speaker_sim moves by less than 0.02 between conditions in either
direction (0.811->0.796 fillers, 0.807->0.810 val96) -- **no material effect of reference-audio
duration** for Chatterbox in this measurement. CosyVoice2 gives a different, more informative
signal: its `frontend_zero_shot` code path (`cosyvoice/cli/frontend.py`) contains a hard assertion,
`assert speech.shape[1] / 16000 <= 30, 'do not support extract speech token for audio longer than 30s'`
-- this task's `ref_concat` clip is 30.57s, 0.57s over that limit, so **CosyVoice2 rejected 100% of
`ref_concat` prompts outright** (0/196 successful on both prompt sets, well under the 80% REQ-6
threshold, marked null). This is a genuine system limitation discovered by this task, not a corpus
or harness bug: unlike F5-TTS (which the plan's own preflight check found silently crops long
references to ~15s) or Chatterbox (which consumed the full 30.57s without issue), CosyVoice2 cannot
use anywhere-near-30s reference audio *at all*. A follow-up re-run with a `ref_concat` clip trimmed
to <30s (e.g. 29.5s) would be needed to get a real CosyVoice2 duration-effect data point; this task
did not spend further GPU budget attempting that retry (see `results/cost_tracking.json`).

Separately, `ref_single` itself required a documented deviation: the real `11labs_david` corpus has
no single clip anywhere near 10s (max 1.67s across all 1364 clips), so `ref_single` was built as a
~10.8s concatenation of 10 half-A clips using the same method as `ref_concat`, not a literal single
utterance (`code/build_references.py`'s module docstring has the full preflight-inspection finding).

### WER and brand-name pronunciation (Key Question 5)

Overall WER is high across every system measured (Chatterbox 0.36-0.43 mean, ElevenLabs itself
0.23-0.33), driven mostly by short, casual filler phrasing and Whisper/jiwer normalization noise on
this project's own short utterances rather than by any single failure mode. Brand-name-bearing texts
(containing "rezolve" or "resolve") do **not** show elevated WER relative to non-brand texts for any
system measured -- if anything, the opposite:

| System / condition | Brand-text WER (n) | Non-brand WER (n) |
| --- | --- | --- |
| Chatterbox `ref_single` | 0.321 (18) | 0.391 (155) |
| Chatterbox `ref_concat` | 0.256 (18) | 0.334 (150) |
| ElevenLabs (ground truth) | 0.245 (18) | 0.277 (161) |

Source: `results/per_clip_metrics.json`, filtered by substring match on the prompt text. Since even
the ElevenLabs ground-truth recordings show the same pattern (lower WER on brand-name sentences than
on general fillers), this reads as a property of the sentence templates and Whisper/jiwer scoring,
not a cloning-specific brand-pronunciation weakness -- **brand names are not a special risk factor**
for either measured cloning system on this evidence.

## Synthesis

Putting the four load-bearing findings together: (1) a genuine `speaker_sim ≥0.85` ceiling is
reachable zero-shot (CosyVoice2 `ref_single`, val96), which directly falsifies a strict reading of
"fine-tuning is required to reach 0.85"; (2) that same system is unreliable enough (30% hardened-
gate failure, severe duration/WER problems on the failing fraction) that "reachable" cannot be read
as "production-ready"; (3) latency for every zero-shot system measured is 5-19x over the 300ms
target, which `kokoro_v3_bundle` alone meets; and (4) one of three named systems (F5-TTS) produced
no usable data at all due to an environment issue independent of the model itself. The project's
`≥0.85` speaker_sim success criterion should be restated to explicitly separate the similarity bar
from the latency and reliability bars it was implicitly bundled with -- a system can now be shown to
clear 0.85 alone, so 0.85 alone is not a sufficient production-readiness gate. Kokoro's Stage 2
fine-tuning line should remain the project's main line: it is the only system in any comparison run
so far (this task plus t0008) that meets the latency target, and closing its speaker_sim gap to the
now-demonstrated 0.86 zero-shot ceiling is a smaller, better-understood problem than making a
1.3-2.9-second-TTFB, occasionally-unreliable zero-shot system fast and robust enough to ship.

## Limitations

* F5-TTS contributes zero data (environment hang, not a model verdict) -- this answer cannot say
  anything about F5-TTS's actual speaker_sim/latency, only that it could not be measured this
  session under this VM's conditions.
* `kokoro_v3_bundle` numbers are t0008's stored values, not re-measured in this task's own session;
  its TTFB/RTF are omitted here rather than paired with this session's numbers (Lesson 1).
* CosyVoice2's `ref_concat` failure is a corpus-vs-model-limit mismatch (30.57s clip vs a 30s hard
  cap) that a trivial re-trim could likely resolve -- this task did not spend further budget
  re-running it, so the true CosyVoice2 `ref_concat` result is unknown, not confirmed-bad.
  `chatterbox_ref_single`'s fillers speaker_sim (n=9, 91/100 clips skipped as <1.6s output) and
  `ref_concat` fillers (n=36) rest on small surviving samples after resemblyzer's minimum-duration
  filter; treat those two cells as lower-confidence than the val96 numbers (n=79-96).
* Published SIM-o/SS/MOS numbers from the F5-TTS/CosyVoice2/Chatterbox literature use different
  embedding backbones than this project's GE2E `speaker_sim` and are never merged into the tables
  above.
* This is one measurement session on one VM; the TTFB numbers in particular (well above vendor
  claims for CosyVoice2) may partly reflect this specific hardware/software environment rather than
  a universal property of the models.

## Sources

* Task: [t0008_tts_eval_harness_baselines][t0008]
* URL: [F5-TTS (SWivid/F5-TTS)][f5tts]
* URL: [CosyVoice2-0.5B (FunAudioLLM)][cosyvoice2]
* URL: [Chatterbox (resemble-ai)][chatterbox]

[t0008]: ../../../t0008_tts_eval_harness_baselines/
[f5tts]: https://github.com/SWivid/F5-TTS
[cosyvoice2]: https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B
[chatterbox]: https://github.com/resemble-ai/chatterbox
