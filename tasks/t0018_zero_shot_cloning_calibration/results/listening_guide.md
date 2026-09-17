# Listening Guide -- Zero-Shot Voice-Cloning Calibration

One row per comparison text (3 fixed gate texts + 7 seeded val96 prompts), one column per
system/condition variant that produced audio, plus ElevenLabs (original reference) and
`kokoro_v3_bundle`. Each cell links to the clip, with `speaker_sim` / `wer` / gate verdict in the
cell text. Clips are relative to `results/audio_samples/comparison_set/`.

**Deviation, documented (not silent):** the 3 fixed `GATE_TEXT_NAMES` (`lining_up_suggestions_17`,
`lining_up_suggestions_10`, `putting_them_head_to_head_15`) are **not** among the 100 filler prompts
`get_prompts_by_set("both")` actually returned and synthesized this session -- verified directly:
none of these 3 texts appear anywhere in `results/per_clip_metrics.json`. t0008's
`filler_prompts_100.json` samples only 100 of the 1364 corpus phrases, and these 3 happen not to be
in that sample. Their rows below are therefore all `-` (no clip exists to link). `kokoro_v3_bundle`
is also entirely absent from every row: it was not re-synthesized this session (see
`intervention/kokoro_v3_bundle_not_remeasured.md`), so no comparison clips exist for it either.
`cosyvoice2_ref_concat` is absent for the same structural reason as its column never appearing: it
produced 0 successful clips (REQ-6 null, see `results/tables.json`).

| Text | cosyvoice2_ref_single | chatterbox_ref_single | chatterbox_ref_concat | elevenlabs_david |
| --- | --- | --- | --- | --- |
| **lining up suggestions 17** | - | - | - | - |
| **lining up suggestions 10** | - | - | - | - |
| **putting them head to head 15** | - | - | - | - |
| **notedlooking into relevant retail tie options** | [listen](audio_samples/comparison_set/notedlooking_into_relevant_retail_tie_op__cosyvoice2_ref_single.wav) sim=0.839 wer=n/a gate=PASS | [listen](audio_samples/comparison_set/notedlooking_into_relevant_retail_tie_op__chatterbox_ref_single.wav) sim=0.771 wer=0.33 gate=PASS | [listen](audio_samples/comparison_set/notedlooking_into_relevant_retail_tie_op__chatterbox_ref_concat.wav) sim=0.854 wer=0.33 gate=PASS | [listen](audio_samples/comparison_set/notedlooking_into_relevant_retail_tie_op__elevenlabs_david.wav) sim=0.706 wer=0.33 gate=PASS |
| **i hear youlet me continue from where i left off** | [listen](audio_samples/comparison_set/i_hear_youlet_me_continue_from_where_i_l__cosyvoice2_ref_single.wav) sim=0.860 wer=n/a gate=PASS | [listen](audio_samples/comparison_set/i_hear_youlet_me_continue_from_where_i_l__chatterbox_ref_single.wav) sim=0.782 wer=0.20 gate=PASS | [listen](audio_samples/comparison_set/i_hear_youlet_me_continue_from_where_i_l__chatterbox_ref_concat.wav) sim=0.822 wer=0.40 gate=PASS | [listen](audio_samples/comparison_set/i_hear_youlet_me_continue_from_where_i_l__elevenlabs_david.wav) sim=0.800 wer=0.20 gate=PASS |
| **checking the one-click checkout capability** | [listen](audio_samples/comparison_set/checking_the_one-click_checkout_capabili__cosyvoice2_ref_single.wav) sim=0.870 wer=n/a gate=PASS | [listen](audio_samples/comparison_set/checking_the_one-click_checkout_capabili__chatterbox_ref_single.wav) sim=0.835 wer=0.80 gate=PASS | [listen](audio_samples/comparison_set/checking_the_one-click_checkout_capabili__chatterbox_ref_concat.wav) sim=0.844 wer=0.40 gate=PASS | [listen](audio_samples/comparison_set/checking_the_one-click_checkout_capabili__elevenlabs_david.wav) sim=0.802 wer=0.80 gate=PASS |
| **taking a look at what resolve ai is** | [listen](audio_samples/comparison_set/taking_a_look_at_what_resolve_ai_is__cosyvoice2_ref_single.wav) sim=0.871 wer=n/a gate=PASS | [listen](audio_samples/comparison_set/taking_a_look_at_what_resolve_ai_is__chatterbox_ref_single.wav) sim=0.751 wer=0.12 gate=PASS | [listen](audio_samples/comparison_set/taking_a_look_at_what_resolve_ai_is__chatterbox_ref_concat.wav) sim=0.834 wer=0.12 gate=PASS | [listen](audio_samples/comparison_set/taking_a_look_at_what_resolve_ai_is__elevenlabs_david.wav) sim=0.663 wer=0.12 gate=PASS |
| **llm sess 1f4bbf4db2604037 resp 3d246884a2e44045** | [listen](audio_samples/comparison_set/llm_sess_1f4bbf4db2604037_resp_3d246884a__cosyvoice2_ref_single.wav) sim=0.860 wer=3.40 gate=FAIL | [listen](audio_samples/comparison_set/llm_sess_1f4bbf4db2604037_resp_3d246884a__chatterbox_ref_single.wav) sim=0.807 wer=n/a gate=PASS | [listen](audio_samples/comparison_set/llm_sess_1f4bbf4db2604037_resp_3d246884a__chatterbox_ref_concat.wav) sim=0.808 wer=1.20 gate=PASS | [listen](audio_samples/comparison_set/llm_sess_1f4bbf4db2604037_resp_3d246884a__elevenlabs_david.wav) sim=0.796 wer=n/a gate=PASS |
| **let me see how that transformation is being implemented** | [listen](audio_samples/comparison_set/let_me_see_how_that_transformation_is_be__cosyvoice2_ref_single.wav) sim=0.866 wer=n/a gate=PASS | [listen](audio_samples/comparison_set/let_me_see_how_that_transformation_is_be__chatterbox_ref_single.wav) sim=0.843 wer=0.22 gate=PASS | [listen](audio_samples/comparison_set/let_me_see_how_that_transformation_is_be__chatterbox_ref_concat.wav) sim=0.845 wer=0.00 gate=PASS | [listen](audio_samples/comparison_set/let_me_see_how_that_transformation_is_be__elevenlabs_david.wav) sim=0.859 wer=0.00 gate=PASS |
| **let me get the details on the advisory board** | [listen](audio_samples/comparison_set/let_me_get_the_details_on_the_advisory_b__cosyvoice2_ref_single.wav) sim=0.839 wer=n/a gate=PASS | [listen](audio_samples/comparison_set/let_me_get_the_details_on_the_advisory_b__chatterbox_ref_single.wav) sim=0.736 wer=0.22 gate=PASS | [listen](audio_samples/comparison_set/let_me_get_the_details_on_the_advisory_b__chatterbox_ref_concat.wav) sim=0.727 wer=0.22 gate=PASS | [listen](audio_samples/comparison_set/let_me_get_the_details_on_the_advisory_b__elevenlabs_david.wav) sim=0.817 wer=0.00 gate=PASS |

## What to listen for

* **lining up suggestions 17** (lining_up_suggestions_17): not synthesized this session -- this
  fixed gate text is not among the 100 filler prompts actually sampled by t0008's
  filler_prompts_100.json (see the deviation note above).
* **lining up suggestions 10** (lining_up_suggestions_10): not synthesized this session -- this
  fixed gate text is not among the 100 filler prompts actually sampled by t0008's
  filler_prompts_100.json (see the deviation note above).
* **putting them head to head 15** (putting_them_head_to_head_15): not synthesized this session --
  this fixed gate text is not among the 100 filler prompts actually sampled by t0008's
  filler_prompts_100.json (see the deviation note above).
* **notedlooking into relevant retail tie options** (notedlooking_into_relevant_retail_tie_op): High
  WER on this text for: chatterbox, elevenlabs_david -- listen for mispronounced words.
* **i hear youlet me continue from where i left off** (i_hear_youlet_me_continue_from_where_i_l):
  High WER on this text for: chatterbox -- listen for mispronounced words.
* **checking the one-click checkout capability** (checking_the_one-click_checkout_capabili): High
  WER on this text for: chatterbox, elevenlabs_david -- listen for mispronounced words.
* **taking a look at what resolve ai is** (taking_a_look_at_what_resolve_ai_is): Compare timbre
  match and accent drift against the ElevenLabs original.
* **llm sess 1f4bbf4db2604037 resp 3d246884a2e44045** (llm_sess_1f4bbf4db2604037_resp_3d246884a):
  High WER on this text for: chatterbox, cosyvoice2, elevenlabs_david -- listen for mispronounced
  words.
* **let me see how that transformation is being implemented**
  (let_me_see_how_that_transformation_is_be): Compare timbre match and accent drift against the
  ElevenLabs original.
* **let me get the details on the advisory board** (let_me_get_the_details_on_the_advisory_b):
  Compare timbre match and accent drift against the ElevenLabs original.
