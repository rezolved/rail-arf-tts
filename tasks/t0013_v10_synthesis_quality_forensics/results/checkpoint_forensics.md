# Checkpoint Forensics (Milestone B)

Cheap tensor-level falsifier, run **before** any inference harness code, per `plan/plan.md`
Milestone B. Produced by `code/inspect_checkpoint.py`.

## Ground truth: decoder key naming by architecture

Read directly from `code/kikiri-tts/StyleTTS2/Modules/{hifigan,istftnet}.py` during implementation
(plan.md step 4), **not** assumed from research. Both `Generator` classes register `self.ups`,
`self.resblocks`, `self.conv_post`, `self.m_source`, `self.noise_convs`, `self.noise_res` under
identical names -- the research-phase guess that `ups.*`/`resblocks.*` alone would discriminate the
two architectures was wrong. The actual discriminators found by reading both files'
`Generator.__init__`:

* `generator.alphas.*` (a `nn.ParameterList`) exists **only** in `hifigan.py`'s `Generator` --
  `istftnet.py`'s `Generator` has no `self.alphas` at all. **This is the primary classification
  signal** and it is decisive: empirically, `first_stage_v3.pth`'s `decoder` state dict has 375 keys
  and zero `alphas.*`; both v10 checkpoints' `decoder` state dicts have 678 keys including
  `alphas.{0..4}`.
* A second, symmetric discriminator initially looked plausible while reading source: `istftnet.py`'s
  `Generator` also registers `self.stft = TorchSTFT(...)` and
  `self.reflection_pad = nn.ReflectionPad1d(...)`, entirely absent from `hifigan.py`. **This turned
  out not to work** -- neither shows up in `state_dict()` at all, confirmed empirically (zero
  `stft`/`reflection` keys in either checkpoint): `TorchSTFT.window` is assigned via plain
  `self.window = torch.from_numpy(...)`, not `register_buffer`, so it is never tracked as module
  state; `ReflectionPad1d` has no learnable parameters to begin with. Recorded here so a future
  reader does not repeat the same wrong assumption.
* Both architectures share `ups`/`resblocks`/`conv_post`/`noise_convs` key **names**, but the
  **shapes** differ: `conv_post` outputs 1 channel (raw waveform) in HiFi-GAN vs.
  `gen_istft_n_fft + 2` channels (magnitude+phase) in ISTFTNet; `noise_convs[0]` takes 1 input
  channel in HiFi-GAN vs. `gen_istft_n_fft + 2` in ISTFTNet. The **number** of `ups`/`resblocks`
  stages also differs with `upsample_rates` length -- empirically 4 stages (`generator.ups.{0..3}`)
  in both v10 checkpoints (matching `config_david_v10.yml`'s `[10,5,3,2]`) vs. 2 stages
  (`generator.ups.{0,1}`) in `first_stage_v3.pth` (matching every istftnet config's `[10,6]` in this
  project) -- a second, independent, corroborating signal used as a fallback classifier when the
  `alphas` marker is absent.

## Per-checkpoint forensics

### `epoch_2nd_00016 (v10 primary)`

**Decoder architecture:** `hifigan` (hifigan `alphas` marker=True, ups stages=4)

| Module | Params | Finite | Weight norm |
| --- | ---: | --- | ---: |
| bert | 6,292,480 | yes | 186.5264 |
| bert_encoder | 393,728 | yes | 13.0890 |
| decoder | 54,289,492 | yes | 213.2920 |
| diffusion | 50,653,952 | yes | 899.2298 |
| mpd | 41,105,770 | yes | 95.4354 |
| msd | 280,902 | yes | 18.0580 |
| pitch_extractor | 5,251,148 | yes | 86263.8580 |
| predictor | 16,194,612 | yes | 142.0331 |
| predictor_encoder | 13,880,813 | yes | 41.8930 |
| style_encoder | 13,880,813 | yes | 41.8430 |
| text_aligner | 7,868,452 | yes | 825.3249 |
| text_encoder | 5,606,400 | yes | 310.7809 |
| wd | 1,173,634 | yes | 21.6820 |

### `epoch_2nd_00014 (v10 backup)`

**Decoder architecture:** `hifigan` (hifigan `alphas` marker=True, ups stages=4)

| Module | Params | Finite | Weight norm |
| --- | ---: | --- | ---: |
| bert | 6,292,480 | yes | 186.5264 |
| bert_encoder | 393,728 | yes | 13.0890 |
| decoder | 54,289,492 | yes | 213.2222 |
| diffusion | 50,653,952 | yes | 899.2322 |
| mpd | 41,105,770 | yes | 95.4225 |
| msd | 280,902 | yes | 18.0486 |
| pitch_extractor | 5,251,148 | yes | 86098.1160 |
| predictor | 16,194,612 | yes | 141.9925 |
| predictor_encoder | 13,880,813 | yes | 41.8415 |
| style_encoder | 13,880,813 | yes | 41.7635 |
| text_aligner | 7,868,452 | yes | 825.3249 |
| text_encoder | 5,606,400 | yes | 310.7809 |
| wd | 1,173,634 | yes | 21.6820 |

### `first_stage_v3 (control, istftnet-trained)`

**Decoder architecture:** `istftnet` (hifigan `alphas` marker=False, ups stages=2)

| Module | Params | Finite | Weight norm |
| --- | ---: | --- | ---: |
| bert | 6,292,480 | yes | 186.5264 |
| bert_encoder | 393,728 | yes | 13.0890 |
| decoder | 53,276,190 | yes | 196.9789 |
| diffusion | 43,836,160 | yes | 900.6127 |
| mpd | 41,105,770 | yes | 95.7515 |
| msd | 280,902 | yes | 18.0478 |
| pitch_extractor | 5,251,148 | yes | 84856.2849 |
| predictor | 16,194,612 | yes | 141.6734 |
| predictor_encoder | 13,880,813 | yes | 41.4237 |
| style_encoder | 13,880,813 | yes | 41.5421 |
| text_aligner | 7,868,452 | yes | 825.3249 |
| text_encoder | 5,606,400 | yes | 310.7809 |
| wd | 1,173,634 | yes | 21.7019 |

## Verdict

**(a) Architecture-mismatch hypothesis CONFIRMED at the tensor level.** `first_stage_v3.pth`
(control) decoder classifies as `istftnet`, while both v10 checkpoints (`epoch_2nd_00016`=hifigan,
`epoch_2nd_00014`=hifigan) classify as `hifigan`, matching `config_david_v10.yml`'s
`model_params.decoder.type: hifigan`. This is consistent with the
`train_second_v10.py:load_checkpoint()` finding (`decoder` is not in `ignore_modules`, so
`first_stage_v3.pth`'s istftnet-shaped decoder weights were loaded with a partial shape match,
leaving the HiFi-GAN generator's `ups`/`resblocks`/`alphas`/`conv_post` layers randomly initialized
at the start of training) -- the HiFi-GAN vocoder proper had only 17 epochs to learn from scratch.
This is a working hypothesis carried into Milestone E, not the task's final answer -- Milestones C-E
still run regardless.
