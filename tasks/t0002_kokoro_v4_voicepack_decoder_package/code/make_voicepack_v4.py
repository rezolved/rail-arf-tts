#!/usr/bin/env python3
"""
Extract Kokoro-API voicepack from v4 Stage 1 checkpoint (style_encoder + predictor_encoder).
Adapted from rail-benchmarks/kokoro-finetune/scripts/make_voicepack_from_ft.py for v4 paths.

Usage:
    uv run --with torchaudio --with soundfile python3 code/make_voicepack_v4.py
"""

from pathlib import Path

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

TASK_DIR = Path(__file__).parent.parent
STAGE1_CKPT = (
    TASK_DIR.parent / "t0001_kokoro_v4_stage2_finetune" / "results" / "checkpoints" / "first_stage.pth"
)
WAV_DIR = (
    Path(__file__).parent.parent.parent.parent.parent
    / "rail-benchmarks"
    / "kokoro-finetune"
    / "data"
    / "v4"
    / "train"
    / "wavs"
)
OUT = TASK_DIR / "results" / "david_v4_voicepack.pt"


def clean_state(raw: dict) -> dict:
    """Strip module. prefix and flatten parametrizations.weight.original -> weight."""
    out = {}
    for k, v in raw.items():
        k = k.removeprefix("module.")
        if ".parametrizations.weight.original" in k:
            k = k.replace(".parametrizations.weight.original", ".weight")
        elif ".parametrizations.weight.0._u" in k or ".parametrizations.weight.0._v" in k:
            continue
        out[k] = v
    return out


class DownsampleRes(nn.Module):
    def __init__(self, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(1, out_ch, 3, 1, 1)

    def forward(self, mel_orig, target_hw):
        mel_ds = F.adaptive_avg_pool2d(mel_orig, target_hw)
        return self.conv(mel_ds)


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.downsample_res = DownsampleRes(dim_in)
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if dim_in != dim_out:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
        self.actv = nn.LeakyReLU(0.2)

    def forward(self, x, mel_orig):
        skip = self.downsample_res(mel_orig, x.shape[-2:])
        h = x + skip
        sc = F.avg_pool2d(self.conv1x1(h) if hasattr(self, "conv1x1") else h, 2)
        r = self.actv(self.conv1(h))
        r = F.avg_pool2d(r, 2)
        r = self.actv(self.conv2(r))
        return sc + r


class KikiriEncoder(nn.Module):
    def __init__(self, style_dim=128):
        super().__init__()
        self.shared = nn.ModuleList(
            [
                nn.Conv2d(1, 64, 3, 1, 1),
                ResBlk(64, 128),
                ResBlk(128, 256),
                ResBlk(256, 512),
                ResBlk(512, 512),
                nn.Identity(),
                nn.Conv2d(512, 512, 5, 1, 0),
                nn.Identity(),
                nn.Identity(),
            ]
        )
        self.unshared = nn.Linear(512, style_dim)

    def forward(self, x):
        mel = x
        h = self.shared[0](mel)
        for i in range(1, 5):
            h = self.shared[i](h, mel)
        h = F.leaky_relu(h, 0.2)
        h = self.shared[6](h)
        h = F.adaptive_avg_pool2d(h, 1)
        h = F.leaky_relu(h, 0.2)
        return self.unshared(h.flatten(1))


mel_fn = T.MelSpectrogram(
    sample_rate=24000, n_fft=2048, win_length=1200, hop_length=300, n_mels=80, f_min=0, f_max=8000
)
_MEL_MEAN, _MEL_STD = -4, 4


def wav_to_mel(path):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.tensor(data.mean(axis=1)).unsqueeze(0)
    if sr != 24000:
        import torchaudio

        wav = torchaudio.functional.resample(wav, sr, 24000)
    mel = mel_fn(wav)
    mel = (torch.log(1e-5 + mel) - _MEL_MEAN) / _MEL_STD
    return mel.unsqueeze(0)


def load_encoder(net, key):
    enc = KikiriEncoder()
    sd = clean_state(net[key])
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    loaded = len(sd) - len(unexpected)
    print(f"  {key}: {loaded}/{len(sd)} keys loaded, missing={len(missing)}, unexpected={len(unexpected)}")
    return enc.eval()


def main():
    print(f"Loading Stage 1 checkpoint: {STAGE1_CKPT}")
    net = torch.load(str(STAGE1_CKPT), map_location="cpu", weights_only=False)["net"]

    style_enc = load_encoder(net, "style_encoder")
    print("style_encoder loaded")
    prosody_enc = load_encoder(net, "predictor_encoder")
    print("predictor_encoder loaded")

    wavs = sorted(WAV_DIR.glob("*.wav"))
    print(f"Encoding {min(len(wavs), 200)} / {len(wavs)} clips...")

    timbre_list, prosody_list = [], []
    for wp in wavs[:200]:
        try:
            mel = wav_to_mel(wp)
            with torch.no_grad():
                timbre_list.append(style_enc(mel).squeeze(0).cpu())
                prosody_list.append(prosody_enc(mel).squeeze(0).cpu())
        except Exception as e:
            print(f"  skip {wp.name}: {e}")

    ref_s = torch.cat(
        [
            torch.stack(timbre_list).mean(0),
            torch.stack(prosody_list).mean(0),
        ]
    )
    print(f"ref_s std={ref_s.std():.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    vp = ref_s.unsqueeze(0).unsqueeze(0).expand(510, 1, 256).clone()
    torch.save(vp, str(OUT))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
