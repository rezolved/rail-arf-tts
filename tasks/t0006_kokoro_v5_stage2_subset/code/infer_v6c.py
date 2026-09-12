"""Synthesise British English test phrases from v6c epoch_5."""
import re
import sys, os, logging
from pathlib import Path
import torch, soundfile as sf
logging.basicConfig(level=logging.WARNING)

BASE = Path("/mnt/kikiri-tts/StyleTTS2")
CONFIG = BASE / "Configs/config_david_v6c_stage2.yml"
CKPT   = BASE / "logs/kokoro-david-v6c/epoch_2nd_00005.pth"
AUDIO  = Path("/mnt/kikiri-tts/data/v4/train/wavs")
OUT    = Path("/mnt/kikiri-tts/infer_v6c"); OUT.mkdir(exist_ok=True)

PHRASES = [
    "Welcome to Rezolve. How can I help you today?",
    "Your order has been placed successfully.",
    "I'm sorry, I didn't quite catch that. Could you repeat?",
    "The total for your basket is twenty-three pounds and fifty pence.",
    "Would you like to add anything else before we proceed?",
    "Thank you for shopping with us. Have a wonderful day.",
]

sys.path.insert(0, str(BASE))
os.chdir(BASE)
os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/kikiri-tts/hf_cache")
os.environ.setdefault("HF_HOME", "/mnt/kikiri-tts/hf_cache")

import yaml
from munch import Munch
from models import load_ASR_models, load_F0_models, build_model, load_checkpoint
from utils import recursive_munch
from Utils.PLBERT.util import load_plbert

from text_utils import TextCleaner
from kokoro_tb_utils import extract_voicepack

with open(CONFIG) as f:
    config = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load all required sub-models (same as train_second.py)
text_aligner   = load_ASR_models(config["ASR_path"], config["ASR_config"])
pitch_extractor = load_F0_models(config["F0_path"])
plbert          = load_plbert(config["PLBERT_dir"])

model_params = recursive_munch(config["model_params"])
model = build_model(model_params, text_aligner, pitch_extractor, plbert)
_ = [model[k].to(device) for k in model]

# Load epoch_5 checkpoint
print(f"Loading {CKPT.name} ...")
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
net = ckpt["net"]
for k in model:
    if k in net:
        raw = net[k]
        # Strip DataParallel module. prefix
        clean = {re.sub(r"^module\.", "", key): v for key, v in raw.items()}
        model[k].load_state_dict(clean, strict=False)
        print(f"  loaded: {k}")

_ = [model[k].eval() for k in model]

# Wrap into a simple namespace for extract_voicepack / inference
class M:
    def __init__(self, d):
        for k, v in d.items(): setattr(self, k, v)

m = M(model)

# Extract voicepack
print("\nExtracting voicepack...")
vp, a_norm, p_norm = extract_voicepack(m, AUDIO, device, n_samples=100)
print(f"  acoustic_norm={a_norm:.4f}  prosodic_norm={p_norm:.4f}")
if vp is not None:
    torch.save(vp.cpu(), OUT / "v6c_ep5_voicepack.pt")

# Synthesise
text_cleaner = TextCleaner()
try:
    from misaki import espeak
    g2p = espeak.EspeakG2P(language="en-gb")
    print("G2P: en-gb loaded")
except Exception as e:
    print(f"G2P en-gb failed ({e}), trying en-us")
    from misaki import espeak
    g2p = espeak.EspeakG2P(language="en-us")

print("\nSynthesising...")
for i, phrase in enumerate(PHRASES, 1):
    try:
        g2p_out = g2p(phrase)
        ipa = g2p_out[0] if isinstance(g2p_out, tuple) else g2p_out
        token_ids = text_cleaner(ipa)
        if not token_ids:
            print(f"  p{i}: no tokens"); continue

        vp_t = vp.to(device)
        ref_a = vp_t[:128].unsqueeze(0)
        ref_p = vp_t[128:].unsqueeze(0)

        with torch.no_grad():
            ids = torch.LongTensor([[0, *token_ids, 0]]).to(device)
            lens = torch.LongTensor([ids.shape[-1]]).to(device)
            mask = torch.gt(torch.arange(lens.max()).unsqueeze(0).expand(1,-1)
                            .type_as(lens)+1, lens.unsqueeze(1)).to(device)

            bd = model["bert"](ids, attention_mask=(~mask).int())
            den = model["bert_encoder"](bd).transpose(-1,-2)
            d = model["predictor"].text_encoder(den, ref_p, lens, mask)
            x, _ = model["predictor"].lstm(d)
            dur = torch.sigmoid(model["predictor"].duration_proj(x)).sum(-1)
            pred_dur = torch.round(dur.squeeze()).clamp(min=1).long()
            if pred_dur.dim()==0: pred_dur = pred_dur.unsqueeze(0)

            n_tok = ids.shape[1]
            tot = int(pred_dur.sum())
            aln = torch.zeros(n_tok, tot).to(device)
            c = 0
            for j in range(n_tok):
                dj = int(pred_dur[j])
                aln[j, c:c+dj] = 1; c += dj
            aln = aln.unsqueeze(0)

            en = d.transpose(-1,-2) @ aln
            F0, N = model["predictor"].F0Ntrain(en, ref_p)
            ten = model["text_encoder"](ids, lens, mask)
            asr = ten @ aln
            audio = model["decoder"](asr, F0, N, ref_a)
            wav = audio.squeeze().cpu().numpy()
            dur_s = len(wav)/24000
            out_f = OUT / f"v6c_ep5_p{i:02d}.wav"
            sf.write(str(out_f), wav, 24000)
            print(f"  p{i}: {dur_s:.1f}s | {phrase[:50]}")
    except Exception as e:
        print(f"  p{i} FAILED: {e}")

print(f"\nFiles in {OUT}")
