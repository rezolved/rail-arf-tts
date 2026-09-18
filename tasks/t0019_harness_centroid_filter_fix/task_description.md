# Fix the TTS Harness: Right David Reference, Ear-Calibrated Metrics, Re-baseline

## Motivation

On 2026-09-18 the owner listened to every audio the project has produced and three things came out,
all verified afterwards with measurements (details in `data/owner_listening_set/labels.json` and the
memory notes behind it):

1. **The project has been measuring against the wrong David.** The ElevenLabs account holds two
   voices named David. Production (`brainpowa-voice-gateway` in FluxCD) uses `rWV5HleMkWb5oluMwkA7`
   "David - narrator and newsreader" with `eleven_flash_v2_5`. That is also the voice of the
   training corpus `data/v4` (WavLM-SV 0.94 against it). t0008's harness resolved the voice **by
   name**, fuzzy-matched the first hit, and got `5gLuKtB16QIQv1vuSas1` "David - British Radio Host &
   Storyteller". The reference corpus `data/11labs_david` (1364 fillers) is that other voice
   (0.93-0.95 against it, only 0.86 against the training voice). So every `speaker_sim` from t0008
   to t0018 scored Kokoro against a voice it never trained on, and t0018 gave CosyVoice2/Chatterbox
   reference clips of the wrong David.
2. **The metric does not agree with the owner's ear.** Against a training-voice centroid,
   resemblyzer GE2E gives stock `bm_george` 0.670, CosyVoice2 0.648 and v3 0.709 (David self 0.890):
   no separation. WavLM-base-plus-sv separates better (george 0.706, v3 0.82, ElevenLabs 0.85,
   zero-shot 0.88, David self 0.97) but still rated a stock-timbre mix at 0.89 that the owner says
   is "not David". A metric that cannot rank like the owner cannot drive training.
3. **The gate cannot hear "voice plus noise".** Every v3b and v3-shipped clip the owner calls "voice
   OK, strong noise" passes t0015's hardened gate (`is_likely_noise=False`, flatness 0.02-0.04) and
   is indistinguishable from clean clips on level and HF-energy statistics. The gate catches total
   breakdown (v3 non-b: all 15 flagged) and nothing subtler.

Also inherited from t0016: `build_centroid()`'s `MIN_CLIP_DURATION_S = 1.6` rejects all 1364 filler
clips (mean 1.04 s); and from t0018: the three fixed gate texts are absent from the harness prompt
sets, and val96 prompt texts are derived from filenames, producing garbage like `notedlooking into`
and `llm sess 1f4bbf4d...`.

The owner has decided the Kokoro training line continues on the existing (correct-voice) corpus.
Nothing in that line can be judged until the harness measures the right voice with a metric the
owner trusts. That is this task.

## Owner-labelled calibration set (already in this task)

`data/owner_listening_set/` (DVC, 254 wav files, 83 MB) with `labels.json`: every clip group the
owner listened to on 2026-09-18, with their verdict on noise
(`none | background | strong | no_voice`) and identity
(`david_newsreader | david_radiohost | not_david | unknown`). Groups: zero-shot outputs, v3 shipped
and per-epoch, training-data sample, stock Kokoro with and without the v3 voicepack, voicepack
rescale/blend/search variants, and both Davids synthesized on flash and turbo next to corpus
originals. This is the ground truth for Key Questions 2 and 3. Do not re-label it; extend it only by
asking the owner.

## Key Questions

1. **Reference.** With the reference centroid built from `data/v4/val/wavs` (val_96: 96 held-out
   clips of the production voice, never trained on), what are the same-session speaker-similarity
   numbers for every system below, and how do they compare with the wrong-reference numbers t0008
   and t0018 recorded?
2. **Speaker metric.** Which embedding model ranks the calibration set most like the owner? Test at
   least: resemblyzer GE2E (t0008 baseline), `microsoft/wavlm-base-plus-sv`, SpeechBrain ECAPA-TDNN
   (`speechbrain/spkrec-ecapa-voxceleb`), WeSpeaker ResNet34 or ReDimNet, NVIDIA TitaNet-L. Score:
   for each model, the ordering of group means against the owner's `identity` labels
   (`david_newsreader` must beat `not_david` for every pair; `david_radiohost` must sit between),
   AUC of clip-level "is production David" vs the owner label, and the gap between David-self and
   the best non-David. Pick one, justify it, and keep the runner-up as a secondary column.
3. **Noise metric.** Which perceptual-quality predictor separates the owner's `none` from
   `background` from `strong` from `no_voice`? Test at least UTMOS (`utmos22_strong`), DNSMOS
   (`sig`, `bak`, `ovrl`), and NISQA. Report per-group means and the threshold that reproduces the
   owner's labels with the fewest errors. Wire the chosen one into the audible-speech gate as a
   third layer next to t0015's signals; the old signals stay.
4. **Corpus filter.** Was `MIN_CLIP_DURATION_S = 1.6` ever binding in t0008's environment, what is
   the chosen embedder's real minimum, and what rule replaces the filter?
5. **Prompts.** Where do the true transcripts for val_96 and the fillers live (the phoneme manifests
   were made from text in t0003; find that text), and how many of the 196 current prompts are wrong?

## Scope

### 1. Pin the voice, rebuild the reference

* Constants: `ELEVENLABS_DAVID_VOICE_ID = "rWV5HleMkWb5oluMwkA7"`,
  `ELEVENLABS_MODEL_ID = "eleven_flash_v2_5"`, `output_format=pcm_24000`, stability 0.5,
  similarity_boost 0.75. Remove the name lookup; a name may be logged, never used for resolution.
* Reference set for similarity: `data/v4/val/wavs` (96 clips). ElevenLabs self-consistency is
  training clips (seed-42 sample of 96) vs the val_96 centroid, same voice and model.
* Keep `data/11labs_david` in the harness as an explicitly named **wrong-voice control**
  (`reference=radiohost`) so old numbers can be reproduced, never as the default.
* Filler prompts for duration ratio: synthesize the 100 harness filler texts and the three gate
  texts once with the pinned voice and model (about 2000 characters, under $1), store under
  `data/fillers_newsreader/` (DVC) as the paired ElevenLabs reference.

### 2. Calibrate the speaker metric (Q2) and the noise metric (Q3)

CPU-only, on `data/owner_listening_set/`. Produce the two comparison tables and charts listed under
Outputs. The decision rule is pre-registered: the metric that violates the fewest owner-ordered
pairs wins; ties go to the smaller model.

### 3. Fix the harness and ship it

`assets/library/tts_eval_harness/` version 0.2.0, code copied from t0008 and changed only where this
task says: pinned voice, val_96 reference with the radiohost control, the chosen speaker metric as
`speaker_sim` (resemblyzer kept as `speaker_sim_ge2e` for continuity), the MOS gate, the replaced
duration filter, the three gate texts as an always-included `gate_texts` prompt set, val96 and
filler prompts read from real transcripts. Public function signatures stay backward-compatible. Unit
tests for every change next to t0008's eleven. A `corrections/` entry of type `replace` points the
`tts_eval_harness` library asset at this version.

### 4. Re-baseline every system against the right David (Q1)

Same session, fixed harness, prompt sets val96 + 100 fillers + gate texts:

1. `elevenlabs_david_newsreader` (self-consistency, train sample vs val_96 centroid)
2. `elevenlabs_david_radiohost` (t0008's stored 44.1 kHz synth audio, `synth_audio.dvc`)
3. `kokoro_base_bm_george`, `kokoro_base_bm_lewis`
4. `kokoro_base_v3_voicepack`
5. `kokoro_v3_bundle`
6. `kokoro_t0006_v6d` (t0008's packaged checkpoint)
7. `kokoro_v10_best`, `kokoro_v11_best` (t0013/t0014 StyleTTS2-native harness, stored samples)
8. `cosyvoice2_ref_single`, `chatterbox_ref_single`, `chatterbox_ref_concat` from t0018's stored
   harness audio (`results/audio_samples/harness.dvc`), scored as-is and labelled "cloned the wrong
   David", so the number is a floor, not their capability.

Report `speaker_sim` (chosen), `speaker_sim_ge2e`, `ttfb_ms`, `rtf`, WER, duration ratio, MOS,
gate-failure count, for each system × prompt set, as explicit variants. File a `corrections/`
`update` on t0008's and t0018's `results/metrics.json` `speaker_sim` values, stating they were
measured against the Radio Host voice, rather than leaving contradicting tables.

### 5. Answer asset

`assets/answer/harness-v2-baselines/`: which David is production and how that was established, the
chosen metrics and why, the corrected baseline table, and what the old numbers meant.

## Audio for human listening (mandatory)

DVC-track and index in `results/listening_guide.md`: the gate texts and five val96 prompts through
every Kokoro system in section 4, the newsreader ElevenLabs originals for the same texts, and the
calibration-set clips referenced by the metric tables so the owner can check the chosen metric's
worst disagreements by ear. One row per text, one column per system, each cell a clickable link with
`speaker_sim`, MOS and gate verdict.

## Expected Outputs

* `assets/library/tts_eval_harness/` 0.2.0, `assets/answer/harness-v2-baselines/`, `corrections/`.
* `data/fillers_newsreader/` (DVC), `data/owner_listening_set/` (already present).
* `results/metric_calibration.md` — speaker-metric table (model × group mean, pair violations, AUC,
  David-self minus best non-David) and noise-metric table (predictor × owner noise class).
* `results/filter_diagnosis.md`, `results/prompt_audit.md` (how many prompts were wrong, the fixed
  source).
* `results/metrics.json`, `results/per_clip_metrics.json`, `results/costs.json`.
* Charts in `results/images/`, embedded in `results_detailed.md`:
  `speaker_metric_vs_owner_labels.png` (one panel per embedder: group means with owner identity
  colour; Q2), `noise_metric_vs_owner_labels.png` (box per owner noise class per predictor; Q3),
  `baseline_wrong_vs_right_david.png` (per system: t0008/t0018 recorded vs this task; Q1),
  `embedding_stability_vs_length.png` (Q4).
* `results/suggestions.json` — at minimum: what the corrected baselines imply for t0017/t0020's
  success thresholds, and a voicepack-by-reconstruction experiment if the noise metric confirms the
  voicepack (not the decoder) as the noise source.

## Rejection criteria

* A metric may not be adopted if it ranks any `not_david` group above any `david_newsreader` group
  in the calibration set.
* No number is "the baseline" unless produced by the fixed harness in this task's own session.
* If no tested predictor separates `background` from `none`, say so and ship the best available as
  advisory (not blocking) with the owner's ear recorded as the blocking check.

## Compute and budget

* CPU for calibration, harness fix, and scoring of stored audio. Kokoro synthesis for section 4 on
  CPU is feasible (t0016 did it) but slow for 200 prompts × 6 systems; use `LLM-T1-NC80` for at most
  1.5 h (about $21) if needed. ElevenLabs: about 2000 characters, under $1. Hard cap: **$30**.

## Dependencies

* `t0008_tts_eval_harness_baselines` — the library being corrected, stored ElevenLabs and Kokoro
  audio, prompt sets.
* `t0016_v3_recipe_recovery` — shipped-v3 synthesis path and the duration-filter failure.
* `t0018_zero_shot_cloning_calibration` — stored zero-shot audio to re-score, and the adapters t0021
  keeps using.
