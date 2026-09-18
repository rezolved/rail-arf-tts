# Listening Guide

Human-listenable evidence for the v3 recipe recovery (task_description.md Scope §4, REQ-11). Every
filename below is a clickable relative link from this file.

## Fixed Gate Texts (ElevenLabs Original vs v3 Shipped Bundle)

| Text | ElevenLabs Original | v3 Shipped | Gate Verdict | Listening Note |
| --- | --- | --- | --- | --- |
| lining up suggestions 17 | [wav](audio_samples/elevenlabs_reference/lining_up_suggestions_17.wav) | [wav](audio_samples/v3_shipped/lining_up_suggestions_17.wav) | `is_likely_noise=False` | Compare prosody and breathiness on sibilants; the shipped bundle uses the British-accent pipeline (`lang_code="b"`) matching David's training accent. |
| lining up suggestions 10 | [wav](audio_samples/elevenlabs_reference/lining_up_suggestions_10.wav) | [wav](audio_samples/v3_shipped/lining_up_suggestions_10.wav) | `is_likely_noise=False` | Listen for timbre similarity to the reference speaker; this is one of the three fixed gate texts referenced throughout the project's benchmarks. |
| putting them head to head 15 | [wav](audio_samples/elevenlabs_reference/putting_them_head_to_head_15.wav) | [wav](audio_samples/v3_shipped/putting_them_head_to_head_15.wav) | `is_likely_noise=False` | Compare pacing on the longer clause; this text has the longest duration of the three gate texts. |

## Val96 Seed-42 Samples (v3 Shipped Bundle Only — No Matching ElevenLabs Clip)

| Text | v3 Shipped | Gate Verdict | Listening Note |
| --- | --- | --- | --- |
| noted looking into relevant retail tie options | [wav](audio_samples/v3_shipped/val96_seed42_00.wav) | `is_likely_noise=False` | Generic filler phrase; no ElevenLabs clip shares this exact text in the corpus. |
| i hear you let me continue from where i left off | [wav](audio_samples/v3_shipped/val96_seed42_01.wav) | `is_likely_noise=False` | Listen for continuity of prosody across the two clauses. |
| checking the one-click checkout capability | [wav](audio_samples/v3_shipped/val96_seed42_02.wav) | `is_likely_noise=False` | Brand/product term ("one-click checkout") exercises the installed pronunciation lexicon. |
| taking a look at what resolve ai is | [wav](audio_samples/v3_shipped/val96_seed42_03.wav) | `is_likely_noise=False` | "Rezolve" is spelled "resolve" via the brand lexicon respelling — listen for whether it reads naturally. |
| llm sess 1f4bbf4db2604037 resp 3d246884a2e44045 | [wav](audio_samples/v3_shipped/val96_seed42_04.wav) | `is_likely_noise=False` | **Not natural language** — this val96 filename is a session-log identifier, not a spoken phrase; its derived "text" is a string of hex characters read aloud. Included honestly (seed-42 sampling landed on this filename) rather than silently re-rolled; listen only to confirm the model does not crash or produce noise on out-of-distribution input, not for naturalness. |

## Per-Epoch Samples (v3 and v3b Variants, Not Re-Synthesized)

Copied directly from `tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/audio/`, unchanged. 20
`epoch{0-3}_phrase{1-5}` samples plus 15 `v3/ep{6,7,8}` and 20 `v3b/ep{6,7,8,9}` samples (55 total).
Full per-file gate scores are in `v3_checkpoint_forensics.md`'s "Per-Epoch Sample Gate Scores"
table; summarized here:

| Variant / Epoch Range | Sample Files (first phrase of each epoch shown) | Gate Verdict | Listening Note |
| --- | --- | --- | --- |
| `epoch0-3` (phrase 1 of 5 shown) | [ep0](audio_samples/v3_per_epoch/epoch0_phrase1.wav), [ep1](audio_samples/v3_per_epoch/epoch1_phrase1.wav), [ep2](audio_samples/v3_per_epoch/epoch2_phrase1.wav), [ep3](audio_samples/v3_per_epoch/epoch3_phrase1.wav) | All `is_likely_noise=False` | Early-training progression; listen for improving intelligibility/prosody from epoch 0 to 3. Original phrase texts are unrecoverable (see `code/score_per_epoch_samples.py`'s docstring). |
| `v3/ep6-8` (phrase 1 of 5 shown) | [ep6](audio_samples/v3_per_epoch/v3/ep6_p1.wav), [ep7](audio_samples/v3_per_epoch/v3/ep7_p1.wav), [ep8](audio_samples/v3_per_epoch/v3/ep8_p1.wav) | **All 15 `v3/ep6-8` files flagged `is_likely_noise=True`** | This `v3` variant's epoch 6-8 samples are broken/noisy — listen to confirm. This is exactly why the `v3b` variant below (same epoch range) exists as the corrected/superseding samples, per `t0002`'s own reference to `v3b/ep9_p5.wav` as the "clean" sample. |
| `v3b/ep6-9` (phrase 1 of 5 shown) | [ep6](audio_samples/v3_per_epoch/v3b/ep6_p1.wav), [ep7](audio_samples/v3_per_epoch/v3b/ep7_p1.wav), [ep8](audio_samples/v3_per_epoch/v3b/ep8_p1.wav), [ep9](audio_samples/v3_per_epoch/v3b/ep9_p1.wav) | All `is_likely_noise=False` | Clean samples across epochs 6-9, confirming `v3b` supersedes the broken `v3` variant at the same epochs. `ep9_p5.wav` (below) is the specific clip `t0002` cites as "what good sounds like". |
| `v3b/ep9_phrase5` (the exact clip t0002 cites) | [ep9_p5](audio_samples/v3_per_epoch/v3b/ep9_p5.wav) | `is_likely_noise=False` | The reference "known good" clip per `t0002`'s own documentation. |

All 55 per-epoch files and all 8 shipped-bundle files' full gate scores (`is_likely_noise`,
`duration_sanity_pass`, `longest_nonsilent_run_s`) are tabulated in
[`v3_checkpoint_forensics.md`](v3_checkpoint_forensics.md).
