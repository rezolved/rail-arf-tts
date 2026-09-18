# Listening Guide -- Zero-Shot TTFB Latency Reduction (Corrected Voice)

One row per comparison text (3 fixed gate texts + 7 seeded val96 prompts). Three columns: this
task's own best-setting output (built from the CORRECTED `data/v4/val/wavs` reference), t0018's
wrong-voice-reference output (continuity only, NOT comparable as a quality baseline), and the actual
production-voice val96 original.

**Necessary, not sufficient (owner correction #7):** the automated `hardened_gate_pass` check (this
task's own copy of t0015's `audio_quality_check.py`) is known to pass clips a human listener would
describe as 'voice plus strong noise.' A PASS here does not certify audio quality -- the owner will
listen to this guide's clips directly before any production-readiness conclusion is drawn.

**Deviation, documented (not silent):** the `t0018_old_ref` column is entirely absent below (`-` in
every cell) because `dvc pull` for t0018's `results/audio_samples/comparison_set.dvc` failed in this
task's environment (an Azure credential-chain issue outside this task's control -- see
`code/build_comparison_set.py`'s module docstring for the full explanation). The 3 fixed
`GATE_TEXT_NAMES` (`lining_up_suggestions_17`, `lining_up_suggestions_10`,
`putting_them_head_to_head_15`) are also not among the 100 sampled filler prompts this session
actually synthesized (same structural gap t0018 itself documented) -- their `new_ref` cells are `-`
too, and they have no val96 original by construction (they are filler-corpus phrases, not val96
prompts).

| Text | cosyvoice2 (new ref) | cosyvoice2 (t0018 old ref) | chatterbox (new ref) | chatterbox (t0018 old ref) | val96 original |
| --- | --- | --- | --- | --- | --- |
| **lining up suggestions 17** | - | - | - | - | - |
| **lining up suggestions 10** | - | - | - | - | - |
| **putting them head to head 15** | - | - | - | - | - |
| **notedlooking into relevant retail tie options** | [listen](audio_samples/comparison_set/notedlooking_into_relevant_retail_tie_op__cosyvoice2_load_trt__new_ref.wav) sim=0.930 wer=0.33 gate=PASS | - | [listen](audio_samples/comparison_set/notedlooking_into_relevant_retail_tie_op__chatterbox_torch_compile__new_ref.wav) sim=0.851 wer=0.33 gate=PASS | - | [listen](audio_samples/comparison_set/notedlooking_into_relevant_retail_tie_op__val96_original.wav) |
| **i hear youlet me continue from where i left off** | [listen](audio_samples/comparison_set/i_hear_youlet_me_continue_from_where_i_l__cosyvoice2_load_trt__new_ref.wav) sim=0.918 wer=0.20 gate=PASS | - | [listen](audio_samples/comparison_set/i_hear_youlet_me_continue_from_where_i_l__chatterbox_torch_compile__new_ref.wav) sim=0.860 wer=0.20 gate=PASS | - | [listen](audio_samples/comparison_set/i_hear_youlet_me_continue_from_where_i_l__val96_original.wav) |
| **checking the one-click checkout capability** | [listen](audio_samples/comparison_set/checking_the_one-click_checkout_capabili__cosyvoice2_load_trt__new_ref.wav) sim=0.841 wer=0.80 gate=PASS | - | [listen](audio_samples/comparison_set/checking_the_one-click_checkout_capabili__chatterbox_torch_compile__new_ref.wav) sim=0.850 wer=0.40 gate=PASS | - | [listen](audio_samples/comparison_set/checking_the_one-click_checkout_capabili__val96_original.wav) |
| **taking a look at what resolve ai is** | [listen](audio_samples/comparison_set/taking_a_look_at_what_resolve_ai_is__cosyvoice2_load_trt__new_ref.wav) sim=0.862 wer=0.12 gate=PASS | - | [listen](audio_samples/comparison_set/taking_a_look_at_what_resolve_ai_is__chatterbox_torch_compile__new_ref.wav) sim=0.877 wer=0.00 gate=PASS | - | [listen](audio_samples/comparison_set/taking_a_look_at_what_resolve_ai_is__val96_original.wav) |
| **llm sess 1f4bbf4db2604037 resp 3d246884a2e44045** | [listen](audio_samples/comparison_set/llm_sess_1f4bbf4db2604037_resp_3d246884a__cosyvoice2_load_trt__new_ref.wav) sim=0.937 wer=1.60 gate=FAIL | - | [listen](audio_samples/comparison_set/llm_sess_1f4bbf4db2604037_resp_3d246884a__chatterbox_torch_compile__new_ref.wav) sim=0.929 wer=1.40 gate=PASS | - | [listen](audio_samples/comparison_set/llm_sess_1f4bbf4db2604037_resp_3d246884a__val96_original.wav) |
| **let me see how that transformation is being implemented** | [listen](audio_samples/comparison_set/let_me_see_how_that_transformation_is_be__cosyvoice2_load_trt__new_ref.wav) sim=0.903 wer=0.00 gate=PASS | - | [listen](audio_samples/comparison_set/let_me_see_how_that_transformation_is_be__chatterbox_torch_compile__new_ref.wav) sim=0.846 wer=0.00 gate=PASS | - | [listen](audio_samples/comparison_set/let_me_see_how_that_transformation_is_be__val96_original.wav) |
| **let me get the details on the advisory board** | [listen](audio_samples/comparison_set/let_me_get_the_details_on_the_advisory_b__cosyvoice2_load_trt__new_ref.wav) sim=0.888 wer=0.00 gate=PASS | - | [listen](audio_samples/comparison_set/let_me_get_the_details_on_the_advisory_b__chatterbox_torch_compile__new_ref.wav) sim=0.875 wer=0.00 gate=PASS | - | [listen](audio_samples/comparison_set/let_me_get_the_details_on_the_advisory_b__val96_original.wav) |

## What to listen for

* **lining up suggestions 17** (lining_up_suggestions_17): not synthesized this session (see the
  deviation note above).
* **lining up suggestions 10** (lining_up_suggestions_10): not synthesized this session (see the
  deviation note above).
* **putting them head to head 15** (putting_them_head_to_head_15): not synthesized this session (see
  the deviation note above).
* **notedlooking into relevant retail tie options** (notedlooking_into_relevant_retail_tie_op): High
  WER on this text for: chatterbox, cosyvoice2 -- listen for mispronounced words.
* **i hear youlet me continue from where i left off** (i_hear_youlet_me_continue_from_where_i_l):
  Compare timbre match against the production-voice val96 original and listen for the t0015 gate's
  known 'voice plus strong noise' failure mode even on a PASS.
* **checking the one-click checkout capability** (checking_the_one-click_checkout_capabili): High
  WER on this text for: chatterbox, cosyvoice2 -- listen for mispronounced words.
* **taking a look at what resolve ai is** (taking_a_look_at_what_resolve_ai_is): Compare timbre
  match against the production-voice val96 original and listen for the t0015 gate's known 'voice
  plus strong noise' failure mode even on a PASS.
* **llm sess 1f4bbf4db2604037 resp 3d246884a2e44045** (llm_sess_1f4bbf4db2604037_resp_3d246884a):
  High WER on this text for: chatterbox, cosyvoice2 -- listen for mispronounced words.
* **let me see how that transformation is being implemented**
  (let_me_see_how_that_transformation_is_be): Compare timbre match against the production-voice
  val96 original and listen for the t0015 gate's known 'voice plus strong noise' failure mode even
  on a PASS.
* **let me get the details on the advisory board** (let_me_get_the_details_on_the_advisory_b):
  Compare timbre match against the production-voice val96 original and listen for the t0015 gate's
  known 'voice plus strong noise' failure mode even on a PASS.
