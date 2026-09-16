"""Kokoro Stage 2 training script with safeguards (REQ-3, REQ-6, REQ-7, REQ-8).

Derived from tasks/t0005_kokoro_v5_stage2_train/code/train_second_patched.py.
Added safeguards:
  - Safe load_checkpoint: strips 'module.' prefix, raises on 0-param match (REQ-3).
  - StepLogger: per-step JSONL metrics log (REQ-6).
  - CheckpointManager: per-epoch retention, SHA-256 manifest (REQ-6).
  - HealthGate: dur_loss / acoustic_norm / val_spike / skip-storm gates (REQ-7).
  - run_config capture at startup (REQ-8).

Usage (same CLI as the original):
    python train_second_safeguarded.py -p Configs/config_david_v6c.yml
"""

# ruff: noqa: E402,F405  -- kokoro training codebase: torch patch before imports; wildcard symbols from losses/models/utils

# stdlib
import copy
import logging
import os  # noqa: F401 -- used via wildcard namespace (utils.*)
import os.path as osp
import random  # noqa: F401
import shutil
import sys
import time
import traceback
import warnings
from logging import StreamHandler
from pathlib import Path

# third-party
import click
import librosa  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio  # noqa: F401
import yaml
from munch import Munch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# project safeguards
from tasks.t0009_stage2_training_failure_forensics.code.checkpoint_manager import CheckpointManager
from tasks.t0009_stage2_training_failure_forensics.code.health_gates import HealthGate
from tasks.t0009_stage2_training_failure_forensics.code.jsonl_logger import StepLogger
from tasks.t0009_stage2_training_failure_forensics.code.run_config import capture_run_config

_CONSECUTIVE_SKIPS = {"n": 0}
_MAX_CONSECUTIVE_SKIPS = 50


def _grads_finite(*modules):
    for m in modules:
        for p in m.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return False
    return True


torch.autograd.set_detect_anomaly(False)

# ponytail: torch.load patch must run before kokoro imports resolve weights_only default
if getattr(torch, "_original_load", None) is None:
    torch._original_load = torch.load
    torch.load = lambda *args, **kwargs: torch._original_load(
        *args, **{**kwargs, "weights_only": False}
    )

warnings.simplefilter("ignore")

# kokoro training modules — wildcard imports are intentional (upstream convention)
from kokoro_symbols import TextCleaner  # noqa: F401
from kokoro_tb_utils import extract_voicepack, prepare_test_tokens, run_kokoro_inference
from losses import *  # noqa: F401,F403
from meldataset import build_dataloader
from models import *  # noqa: F401,F403
from Modules.diffusion.sampler import ADPM2Sampler, DiffusionSampler, KarrasSchedule
from Modules.slmadv import SLMAdversarialLoss
from optimizers import build_optimizer
from utils import *  # noqa: F401,F403
from Utils.ASR.models import ASRCNN  # noqa: F401
from Utils.JDC.model import JDCNet  # noqa: F401
from Utils.PLBERT.util import load_plbert


def load_checkpoint(model, optimizer, path, load_only_params=True, ignore_modules=None):
    """DP-aware loader: strips 'module.' prefix, raises on 0-param match.

    Upstream loads with strict=False, so a DataParallel checkpoint (keys prefixed
    'module.') loads ZERO parameters while printing success — silently training from
    scratch. This version fails loudly instead. (REQ-3)
    """
    if ignore_modules is None:
        ignore_modules = []
    state = torch.load(path, map_location="cpu")
    params = state["net"]
    for key in model:
        if key not in params or key in ignore_modules:
            continue
        ckpt_sd = {k.removeprefix("module."): v for k, v in params[key].items()}
        model_sd = model[key].state_dict()
        matched = {
            k: v for k, v in ckpt_sd.items() if k in model_sd and model_sd[k].shape == v.shape
        }
        if len(matched) == 0:
            raise RuntimeError(
                f"{key}: 0/{len(model_sd)} params matched checkpoint {path} — "
                "key or shape mismatch, refusing to train from scratch silently"
            )
        model[key].load_state_dict(matched, strict=False)
        print(f"{key} loaded: {len(matched)}/{len(model_sd)} params")
    _ = [model[key].eval() for key in model]

    if load_only_params:
        return model, optimizer, 0, 0
    optimizer.load_state_dict(state["optimizer"])
    return model, optimizer, state["epoch"], state["iters"]


# simple fix for dataparallel that allows access to class attributes
class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


@click.command()
@click.option("-p", "--config_path", default="Configs/config.yml", type=str)
@click.option("--run-id", default="", type=str, help="Run ID for safeguard logs (e.g. v6c_run01)")
def main(config_path, run_id):
    with open(config_path) as _fh:
        config = yaml.safe_load(_fh)

    log_dir = config["log_dir"]
    # Path object required by safeguard modules (capture_run_config, StepLogger, CheckpointManager)
    _log_dir_path = Path(log_dir)
    if not osp.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, "train.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(levelname)s:%(asctime)s: %(message)s"))
    logger.addHandler(file_handler)

    # ── Capture run config at startup (REQ-8) ─────────────────────────────────
    _run_id = run_id or osp.basename(log_dir)
    capture_run_config(
        config_path=Path(config_path),
        log_dir=_log_dir_path,
        run_id=_run_id,
    )
    # ──────────────────────────────────────────────────────────────────────────

    batch_size = config.get("batch_size", 10)

    epochs = config.get("epochs_2nd", 200)
    config.get("save_freq", 2)
    log_interval = config.get("log_interval", 10)
    saving_epoch = config.get("save_freq", 2)

    data_params = config.get("data_params", None)
    sr = config["preprocess_params"].get("sr", 24000)
    train_path = data_params["train_data"]
    val_path = data_params["val_data"]
    root_path = data_params["root_path"]
    min_length = data_params["min_length"]
    OOD_data = data_params["OOD_data"]
    num_workers = data_params.get("num_workers", 2)

    max_len = config.get("max_len", 200)

    loss_params = Munch(config["loss_params"])
    diff_epoch = loss_params.diff_epoch
    joint_epoch = loss_params.joint_epoch
    train_LM = config.get("train_LM", True)

    optimizer_params = Munch(config["optimizer_params"])

    train_list, val_list = get_data_path_list(train_path, val_path)
    device = "cuda"

    train_dataloader = build_dataloader(
        train_list,
        root_path,
        OOD_data=OOD_data,
        min_length=min_length,
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_config={},
        device=device,
    )

    val_dataloader = build_dataloader(
        val_list,
        root_path,
        OOD_data=OOD_data,
        min_length=min_length,
        batch_size=batch_size,
        validation=True,
        num_workers=0,
        device=device,
        dataset_config={},
    )

    # load pretrained ASR model
    ASR_config = config.get("ASR_config", False)
    ASR_path = config.get("ASR_path", False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    # load pretrained F0 model
    F0_path = config.get("F0_path", False)
    pitch_extractor = load_F0_models(F0_path)

    # load PL-BERT model
    BERT_path = config.get("PLBERT_dir", False)
    plbert = load_plbert(BERT_path)

    # build model
    model_params = recursive_munch(config["model_params"])
    multispeaker = model_params.multispeaker
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].to(device) for key in model]

    start_epoch = 0
    iters = 0

    load_pretrained = config.get("pretrained_model", "") != "" and config.get(
        "second_stage_load_pretrained", False
    )

    if not load_pretrained:
        if config.get("first_stage_path", "") != "":
            first_stage_path = osp.join(log_dir, config.get("first_stage_path", "first_stage.pth"))
            print(f"Loading the first stage model at {first_stage_path} ...")
            model, _, start_epoch, iters = load_checkpoint(
                model,
                None,
                first_stage_path,
                load_only_params=True,
                ignore_modules=[
                    "predictor_encoder",
                    "msd",
                    "mpd",
                    "wd",
                    "diffusion",
                ],
            )  # keep starting epoch for tensorboard log

            # these epochs should be counted from the start epoch
            diff_epoch += start_epoch
            joint_epoch += start_epoch
            epochs += start_epoch

            model.predictor_encoder = copy.deepcopy(model.style_encoder)
        else:
            raise ValueError("You need to specify the path to the first stage model.")

    gl = GeneratorLoss(model.mpd, model.msd).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
    wl = WavLMLoss(model_params.slm.model, model.wd, sr, model_params.slm.sr).to(device)

    gl = MyDataParallel(gl)
    dl = MyDataParallel(dl)
    wl = MyDataParallel(wl)

    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(
            sigma_min=0.0001, sigma_max=3.0, rho=9.0
        ),  # empirical parameters
        clamp=False,
    )

    scheduler_params = {
        "max_lr": optimizer_params.lr,
        "pct_start": float(0),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }
    scheduler_params_dict = {key: scheduler_params.copy() for key in model}
    scheduler_params_dict["bert"]["max_lr"] = optimizer_params.bert_lr * 2
    scheduler_params_dict["decoder"]["max_lr"] = optimizer_params.ft_lr * 2
    scheduler_params_dict["style_encoder"]["max_lr"] = optimizer_params.ft_lr * 2

    optimizer = build_optimizer(
        {key: model[key].parameters() for key in model},
        scheduler_params_dict=scheduler_params_dict,
        lr=optimizer_params.lr,
    )

    # adjust BERT learning rate
    for g in optimizer.optimizers["bert"].param_groups:
        g["betas"] = (0.9, 0.99)
        g["lr"] = optimizer_params.bert_lr
        g["initial_lr"] = optimizer_params.bert_lr
        g["min_lr"] = 0
        g["weight_decay"] = 0.01

    # adjust acoustic module learning rate
    for module in ["decoder", "style_encoder"]:
        for g in optimizer.optimizers[module].param_groups:
            g["betas"] = (0.0, 0.99)
            g["lr"] = optimizer_params.ft_lr
            g["initial_lr"] = optimizer_params.ft_lr
            g["min_lr"] = 0
            g["weight_decay"] = 1e-4

    # load models if there is a model
    if load_pretrained:
        model, optimizer, start_epoch, iters = load_checkpoint(
            model,
            optimizer,
            config["pretrained_model"],
            load_only_params=config.get("load_only_params", True),
        )
        # Ensure all modules are in train() mode after loading
        _ = [model[key].train() for key in model]

        # Initialize predictor_encoder from trained style_encoder
        model.predictor_encoder = copy.deepcopy(model.style_encoder)

    # DP — must happen AFTER load_checkpoint so state dict keys match
    for key in model:
        if key != "mpd" and key != "msd" and key != "wd":
            model[key] = MyDataParallel(model[key])

    n_down = model.text_aligner.n_down

    best_loss = float("inf")
    list([])
    list([])
    iters = 0

    nn.L1Loss()  # F0 loss (regression)
    torch.cuda.empty_cache()

    stft_loss = MultiResolutionSTFTLoss().to(device)

    print("BERT", optimizer.optimizers["bert"])
    print("decoder", optimizer.optimizers["decoder"])

    start_ds = False

    running_std = []

    slmadv_params = Munch(config["slmadv_params"])
    slmadv = SLMAdversarialLoss(
        model,
        wl,
        sampler,
        slmadv_params.min_len,
        slmadv_params.max_len,
        batch_percentage=slmadv_params.batch_percentage,
        skip_update=slmadv_params.iter,
        sig=slmadv_params.sig,
        diffusion_enabled=(diff_epoch < epochs),
    )

    # ── Kokoro-faithful TensorBoard inference setup ────────────────────────────
    text_cleaner = TextCleaner()
    _test_tokens = prepare_test_tokens(text_cleaner)
    if _test_tokens:
        logger.info(f"TensorBoard inference: prepared {len(_test_tokens)} test sentences")
    else:
        logger.warning("TensorBoard inference: no test sentences available")

    # Generate Stage 1 baseline before any training updates
    _ = [model[key].eval() for key in model]
    logger.info("Extracting Stage 1 baseline voicepack for TensorBoard...")
    _baseline_voicepack, _baseline_acoustic_norm, _baseline_prosodic_norm = extract_voicepack(
        model, root_path, device, n_samples=200
    )
    if _baseline_voicepack is not None:
        writer.add_scalar("baseline/acoustic_norm", _baseline_acoustic_norm, 0)
        writer.add_scalar("baseline/prosodic_norm", _baseline_prosodic_norm, 0)
        logger.info(
            f"  acoustic_norm={_baseline_acoustic_norm:.4f}"
            f"  prosodic_norm={_baseline_prosodic_norm:.4f}"
        )
        _baseline_audio = run_kokoro_inference(
            model, _test_tokens, _baseline_voicepack, device, text_cleaner
        )
        for i, (text, audio) in enumerate(_baseline_audio):
            writer.add_audio(f"baseline/test_{i + 1:02d}", audio, 0, sample_rate=sr)
            logger.info(f"  baseline/test_{i + 1:02d}: {text[:60]}")
    _ = [model[key].train() for key in model]
    # ──────────────────────────────────────────────────────────────────────────

    # ── Safeguard initialisation ───────────────────────────────────────────────
    # joint_epoch is 1-based in HealthGate (val records use 1-based epoch numbers)
    _step_logger = StepLogger(log_dir=_log_dir_path, run_id=_run_id)
    _ckpt_mgr = CheckpointManager(log_dir=_log_dir_path, run_id=_run_id, joint_epoch=joint_epoch)
    _gate = HealthGate(joint_epoch=joint_epoch, logger=_step_logger)
    # ──────────────────────────────────────────────────────────────────────────

    for epoch in range(start_epoch, epochs):
        running_loss = 0
        start_time = time.time()

        _ = [model[key].eval() for key in model]
        _ = [model[key].train() for key in model]

        if epoch >= joint_epoch:
            start_ds = True

        for i, batch in enumerate(train_dataloader):
            waves = batch[0]
            batch = [b.to(device) for b in batch[1:]]
            (
                texts,
                input_lengths,
                ref_texts,
                ref_lengths,
                mels,
                mel_input_length,
                ref_mels,
            ) = batch

            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2**n_down)).to(device)
                length_to_mask(mel_input_length).to(device)
                text_mask = length_to_mask(input_lengths).to(texts.device)

                try:
                    _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                    s2s_attn = s2s_attn.transpose(-1, -2)
                    s2s_attn = s2s_attn[..., 1:]
                    s2s_attn = s2s_attn.transpose(-1, -2)
                except Exception:
                    continue

                mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2**n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                t_en = model.text_encoder(texts, input_lengths, text_mask)
                if t_en is not None and torch.isnan(t_en).any():
                    print("NaN detected in t_en")
                    sys.exit(1)
                asr = t_en @ s2s_attn_mono
                if asr is not None and torch.isnan(asr).any():
                    print("NaN detected in asr")
                    sys.exit(1)

                d_gt = s2s_attn_mono.sum(axis=-1).detach()

                if multispeaker and epoch >= diff_epoch:
                    ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                    if ref_ss is not None and torch.isnan(ref_ss).any():
                        print("NaN detected in ref_ss")
                        sys.exit(1)
                    ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                    ref = torch.cat([ref_ss, ref_sp], dim=1)

            ss = []
            gs = []
            for bib in range(len(mel_input_length)):
                mel = mels[bib, :, : mel_input_length[bib]]
                s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                ss.append(s)
                s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                gs.append(s)

            s_dur = torch.stack(ss).squeeze(1)
            gs = torch.stack(gs).squeeze(1)
            s_trg = torch.cat([gs, s_dur], dim=-1).detach()

            bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
            if bert_dur is not None and torch.isnan(bert_dur).any():
                print("NaN detected in bert_dur")
                sys.exit(1)
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
            if d_en is not None and torch.isnan(d_en).any():
                print("NaN detected in d_en")
                sys.exit(1)

            if epoch >= diff_epoch:
                num_steps = np.random.randint(3, 5)

                if model_params.diffusion.dist.estimate_sigma_data:
                    model.diffusion.module.diffusion.sigma_data = s_trg.std(axis=-1).mean().item()
                    running_std.append(model.diffusion.module.diffusion.sigma_data)

                if multispeaker:
                    s_preds = sampler(
                        noise=torch.randn_like(s_trg).unsqueeze(1).to(device),
                        embedding=bert_dur,
                        embedding_scale=1,
                        features=ref,
                        embedding_mask_proba=0.1,
                        num_steps=num_steps,
                    ).squeeze(1)
                    loss_diff = model.diffusion(
                        s_trg.unsqueeze(1), embedding=bert_dur, features=ref
                    ).mean()
                    loss_sty = F.l1_loss(s_preds, s_trg.detach())
                else:
                    s_preds = sampler(
                        noise=torch.randn_like(s_trg).unsqueeze(1).to(device),
                        embedding=bert_dur,
                        embedding_scale=1,
                        embedding_mask_proba=0.1,
                        num_steps=num_steps,
                    ).squeeze(1)
                    loss_diff = model.diffusion.module.diffusion(
                        s_trg.unsqueeze(1), embedding=bert_dur
                    ).mean()
                    loss_sty = F.l1_loss(s_preds, s_trg.detach())
            else:
                loss_sty = 0
                loss_diff = 0

            d, p = model.predictor(d_en, s_dur, input_lengths, s2s_attn_mono, text_mask)
            if p is not None and torch.isnan(p).any():
                print("NaN detected in p")
                sys.exit(1)
            if d is not None and torch.isnan(d).any():
                print("NaN detected in d")
                sys.exit(1)

            mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
            mel_len_st = int(mel_input_length.min().item() / 2 - 1)
            en = []
            gt = []
            st = []
            p_en = []
            wav = []

            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)

                random_start = np.random.randint(0, mel_length - mel_len)
                en.append(asr[bib, :, random_start : random_start + mel_len])
                p_en.append(p[bib, :, random_start : random_start + mel_len])
                gt.append(mels[bib, :, (random_start * 2) : ((random_start + mel_len) * 2)])

                y = waves[bib][(random_start * 2) * 300 : ((random_start + mel_len) * 2) * 300]
                wav.append(torch.from_numpy(y).to(device))

                random_start = np.random.randint(0, mel_length - mel_len_st)
                st.append(mels[bib, :, (random_start * 2) : ((random_start + mel_len_st) * 2)])

            wav = torch.stack(wav).float().detach()
            en = torch.stack(en)
            p_en = torch.stack(p_en)
            gt = torch.stack(gt).detach()
            st = torch.stack(st).detach()

            if gt.size(-1) < 80:
                continue

            F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
            N_real = log_norm(gt.unsqueeze(1)).squeeze(1)

            s_dur = model.predictor_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))
            s = model.style_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))

            y_rec_gt = wav.unsqueeze(1)
            y_rec_gt_pred = model.decoder(en, F0_real, N_real, s)

            wav = y_rec_gt if epoch >= joint_epoch else y_rec_gt_pred

            F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s_dur)
            y_rec = model.decoder(en, F0_fake, N_fake, s)

            loss_F0_rec = (F.smooth_l1_loss(F0_real, F0_fake)) / 10
            loss_norm_rec = F.smooth_l1_loss(N_real, N_fake)

            if start_ds:
                optimizer.zero_grad()
                d_loss = dl(wav.detach(), y_rec.detach()).mean()
                d_loss.backward()
                if _grads_finite(model.msd, model.mpd):
                    torch.nn.utils.clip_grad_norm_(model.msd.parameters(), 10.0)
                    torch.nn.utils.clip_grad_norm_(model.mpd.parameters(), 10.0)
                    optimizer.step("msd")
                    optimizer.step("mpd")
                else:
                    print(f"[skip] non-finite grad in msd/mpd at epoch {epoch} step {i}")
                    optimizer.zero_grad()
            else:
                d_loss = 0

            optimizer.zero_grad()

            loss_mel = stft_loss(y_rec, wav)
            if loss_mel is not None and torch.isnan(loss_mel).any():
                print(f"[skip] NaN in loss_mel at epoch {epoch} step {i}")
                optimizer.zero_grad()
                _CONSECUTIVE_SKIPS["n"] += 1
                # ── Gate: consecutive skip storm ───────────────────────────────
                skip_result = _gate.record_skip()
                if skip_result is not None and skip_result.fired:
                    sys.exit(3)
                # ──────────────────────────────────────────────────────────────
                if _CONSECUTIVE_SKIPS["n"] >= _MAX_CONSECUTIVE_SKIPS:
                    print(f"[ABORT] {_CONSECUTIVE_SKIPS['n']} consecutive skipped batches")
                    sys.exit(2)
                continue

            loss_gen_all = gl(wav, y_rec).mean() if start_ds else 0
            loss_lm = wl(wav.detach().squeeze(), y_rec.squeeze()).mean()

            loss_ce = 0
            loss_dur = 0
            for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths, strict=False):
                _s2s_pred = _s2s_pred[:_text_length, :]
                _text_input = _text_input[:_text_length].long()
                _s2s_trg = torch.zeros_like(_s2s_pred)
                for p in range(_s2s_trg.shape[0]):
                    _s2s_trg[p, : _text_input[p]] = 1
                _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)

                loss_dur += F.l1_loss(
                    _dur_pred[1 : _text_length - 1], _text_input[1 : _text_length - 1]
                )
                loss_ce += F.binary_cross_entropy_with_logits(
                    _s2s_pred.flatten(), _s2s_trg.flatten()
                )

            loss_ce /= texts.size(0)
            loss_dur /= texts.size(0)

            g_loss = (
                loss_params.lambda_mel * loss_mel
                + loss_params.lambda_F0 * loss_F0_rec
                + loss_params.lambda_ce * loss_ce
                + loss_params.lambda_norm * loss_norm_rec
                + loss_params.lambda_dur * loss_dur
                + loss_params.lambda_gen * loss_gen_all
                + loss_params.lambda_slm * loss_lm
                + loss_params.lambda_sty * loss_sty
                + loss_params.lambda_diff * loss_diff
            )

            _CONSECUTIVE_SKIPS["n"] = 0
            _gate.reset_skips()
            running_loss += loss_mel.item()
            g_loss.backward()

            gen_modules = [
                model.bert_encoder,
                model.bert,
                model.predictor,
                model.predictor_encoder,
                model.diffusion,
                model.style_encoder,
                model.decoder,
            ]
            if not _grads_finite(*gen_modules):
                print(f"[skip] non-finite grad in generator at epoch {epoch} step {i}")
                optimizer.zero_grad()
            else:
                if train_LM:
                    optimizer.step("bert_encoder")
                    optimizer.step("bert")
                optimizer.step("predictor")
                optimizer.step("predictor_encoder")

                if epoch >= diff_epoch:
                    optimizer.step("diffusion")

                if epoch >= joint_epoch:
                    torch.nn.utils.clip_grad_norm_(model.style_encoder.parameters(), 10.0)
                    torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 10.0)
                    optimizer.step("style_encoder")
                    optimizer.step("decoder")

                    use_ind = np.random.rand() < 0.5

                    if use_ind:
                        ref_lengths = input_lengths
                        ref_texts = texts

                    if loss_params.lambda_slm == 0:
                        slm_out = None
                    else:
                        slm_out = slmadv(
                            i,
                            y_rec_gt,
                            y_rec_gt_pred,
                            waves,
                            mel_input_length,
                            ref_texts,
                            ref_lengths,
                            use_ind,
                            s_trg.detach(),
                            ref if multispeaker else None,
                        )

                    if slm_out is None:
                        continue

                    d_loss_slm, loss_gen_lm, y_pred = slm_out

                    optimizer.zero_grad()
                    loss_gen_lm.backward()

                    total_norm = {}
                    for key in model:
                        total_norm[key] = 0
                        parameters = [
                            p
                            for p in model[key].parameters()
                            if p.grad is not None and p.requires_grad
                        ]
                        for p in parameters:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm[key] += param_norm.item() ** 2
                        total_norm[key] = total_norm[key] ** 0.5

                    if total_norm["predictor"] > slmadv_params.thresh:
                        for key in model:
                            for p in model[key].parameters():
                                if p.grad is not None:
                                    p.grad *= 1 / total_norm["predictor"]

                    for p in model.predictor.duration_proj.parameters():
                        if p.grad is not None:
                            p.grad *= slmadv_params.scale

                    for p in model.predictor.lstm.parameters():
                        if p.grad is not None:
                            p.grad *= slmadv_params.scale

                    for p in model.diffusion.parameters():
                        if p.grad is not None:
                            p.grad *= slmadv_params.scale

                    optimizer.step("bert_encoder")
                    optimizer.step("bert")
                    optimizer.step("predictor")
                    optimizer.step("diffusion")

                    if d_loss_slm != 0:
                        optimizer.zero_grad()
                        d_loss_slm.backward(retain_graph=True)
                        optimizer.step("wd")

                else:
                    d_loss_slm, loss_gen_lm = 0, 0

            iters = iters + 1

            # ── Gate: dur_loss at first step of epoch ≥ 2 (REQ-7) ─────────────
            if (i + 1) == 1:  # first logged step in this epoch
                _gate_result = _gate.check(
                    epoch=epoch + 1,  # 1-based
                    step=1,
                    metrics={"dur_loss": float(loss_dur) if not isinstance(loss_dur, int) else 0.0},
                )
                if _gate_result.fired:
                    sys.exit(3)
            # ──────────────────────────────────────────────────────────────────

            if (i + 1) % log_interval == 0:
                logger.info(
                    "Epoch [%d/%d], Step [%d/%d], Loss: %.5f, Disc Loss: %.5f, Dur Loss: %.5f, CE Loss: %.5f, Norm Loss: %.5f, F0 Loss: %.5f, LM Loss: %.5f, Gen Loss: %.5f, Sty Loss: %.5f, Diff Loss: %.5f, DiscLM Loss: %.5f, GenLM Loss: %.5f"  # noqa: E501,UP031
                    % (
                        epoch + 1,
                        epochs,
                        i + 1,
                        len(train_list) // batch_size,
                        running_loss / log_interval,
                        d_loss,
                        loss_dur,
                        loss_ce,
                        loss_norm_rec,
                        loss_F0_rec,
                        loss_lm,
                        loss_gen_all,
                        loss_sty,
                        loss_diff,
                        d_loss_slm,
                        loss_gen_lm,
                    )
                )

                # ── JSONL step log (REQ-6) ────────────────────────────────────
                _step_logger.log(
                    {
                        "epoch": epoch + 1,
                        "step": i + 1,
                        "loss_total": running_loss / log_interval,
                        "disc_loss": float(d_loss) if not isinstance(d_loss, int) else 0.0,
                        "dur_loss": float(loss_dur) if not isinstance(loss_dur, int) else 0.0,
                        "ce_loss": float(loss_ce) if not isinstance(loss_ce, int) else 0.0,
                        "mel_loss": float(loss_norm_rec),
                        "f0_loss": float(loss_F0_rec),
                    }
                )
                # ──────────────────────────────────────────────────────────────

                writer.add_scalar("train/mel_loss", running_loss / log_interval, iters)
                writer.add_scalar("train/gen_loss", loss_gen_all, iters)
                writer.add_scalar("train/d_loss", d_loss, iters)
                writer.add_scalar("train/ce_loss", loss_ce, iters)
                writer.add_scalar("train/dur_loss", loss_dur, iters)
                writer.add_scalar("train/slm_loss", loss_lm, iters)
                writer.add_scalar("train/norm_loss", loss_norm_rec, iters)
                writer.add_scalar("train/F0_loss", loss_F0_rec, iters)
                writer.add_scalar("train/sty_loss", loss_sty, iters)
                writer.add_scalar("train/diff_loss", loss_diff, iters)
                writer.add_scalar("train/d_loss_slm", d_loss_slm, iters)
                writer.add_scalar("train/gen_loss_slm", loss_gen_lm, iters)

                running_loss = 0

                print("Time elapsed:", time.time() - start_time)

        loss_test = 0
        loss_align = 0
        loss_f = 0
        _ = [model[key].eval() for key in model]

        with torch.no_grad():
            iters_test = 0
            for _batch_idx, batch in enumerate(val_dataloader):
                optimizer.zero_grad()

                try:
                    waves = batch[0]
                    batch = [b.to(device) for b in batch[1:]]
                    (
                        texts,
                        input_lengths,
                        ref_texts,
                        ref_lengths,
                        mels,
                        mel_input_length,
                        ref_mels,
                    ) = batch
                    with torch.no_grad():
                        mask = length_to_mask(mel_input_length // (2**n_down)).to("cuda")
                        text_mask = length_to_mask(input_lengths).to(texts.device)

                        _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                        s2s_attn = s2s_attn.transpose(-1, -2)
                        s2s_attn = s2s_attn[..., 1:]
                        s2s_attn = s2s_attn.transpose(-1, -2)

                        mask_ST = mask_from_lens(
                            s2s_attn, input_lengths, mel_input_length // (2**n_down)
                        )
                        s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                        t_en = model.text_encoder(texts, input_lengths, text_mask)
                        asr = t_en @ s2s_attn_mono

                        d_gt = s2s_attn_mono.sum(axis=-1).detach()

                    ss = []
                    gs = []

                    for bib in range(len(mel_input_length)):
                        mel = mels[bib, :, : mel_input_length[bib]]
                        s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                        ss.append(s)
                        s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                        gs.append(s)

                    s = torch.stack(ss).squeeze(1)
                    gs = torch.stack(gs).squeeze(1)
                    s_trg = torch.cat([s, gs], dim=-1).detach()

                    bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
                    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
                    d, p = model.predictor(d_en, s, input_lengths, s2s_attn_mono, text_mask)

                    mel_len = int(mel_input_length.min().item() / 2 - 1)
                    en = []
                    gt = []
                    p_en = []
                    wav = []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item() / 2)

                        random_start = np.random.randint(0, mel_length - mel_len)
                        en.append(asr[bib, :, random_start : random_start + mel_len])
                        p_en.append(p[bib, :, random_start : random_start + mel_len])
                        gt.append(mels[bib, :, (random_start * 2) : ((random_start + mel_len) * 2)])

                        y = waves[bib][
                            (random_start * 2) * 300 : ((random_start + mel_len) * 2) * 300
                        ]
                        wav.append(torch.from_numpy(y).to(device))

                    wav = torch.stack(wav).float().detach()
                    en = torch.stack(en)
                    p_en = torch.stack(p_en)
                    gt = torch.stack(gt).detach()

                    s = model.predictor_encoder(gt.unsqueeze(1))
                    F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s)

                    loss_dur = 0
                    for _s2s_pred, _text_input, _text_length in zip(
                        d, (d_gt), input_lengths, strict=False
                    ):
                        _s2s_pred = _s2s_pred[:_text_length, :]
                        _text_input = _text_input[:_text_length].long()
                        _s2s_trg = torch.zeros_like(_s2s_pred)
                        for bib in range(_s2s_trg.shape[0]):
                            _s2s_trg[bib, : _text_input[bib]] = 1
                        _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)
                        loss_dur += F.l1_loss(
                            _dur_pred[1 : _text_length - 1],
                            _text_input[1 : _text_length - 1],
                        )

                    loss_dur /= texts.size(0)

                    s = model.style_encoder(gt.unsqueeze(1))
                    y_rec = model.decoder(en, F0_fake, N_fake, s)
                    loss_mel = stft_loss(y_rec.squeeze(), wav.detach())

                    F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))
                    loss_F0 = F.l1_loss(F0_real, F0_fake) / 10

                    loss_test += (loss_mel).mean()
                    loss_align += (loss_dur).mean()
                    loss_f += (loss_F0).mean()

                    iters_test += 1
                except Exception as e:
                    print("run into exception", e)
                    traceback.print_exc()
                    continue

        print("Epochs:", epoch + 1)
        logger.info(
            f"Validation loss: {loss_test / iters_test:.3f},"
            f" Dur loss: {loss_align / iters_test:.3f},"
            f" F0 loss: {loss_f / iters_test:.3f}" + "\n\n\n"
        )
        print("\n\n\n")
        writer.add_scalar("eval/mel_loss", loss_test / iters_test, epoch + 1)
        writer.add_scalar("eval/dur_loss", loss_align / iters_test, epoch + 1)
        writer.add_scalar("eval/F0_loss", loss_f / iters_test, epoch + 1)

        _val_loss = float(loss_test / iters_test)

        if epoch % saving_epoch == 0:
            if _val_loss < best_loss:
                best_loss = _val_loss
            print("Saving..")
            state = {
                "net": {key: model[key].state_dict() for key in model},
                "optimizer": optimizer.state_dict(),
                "iters": iters,
                "val_loss": _val_loss,
                "epoch": epoch,
            }
            save_path = osp.join(log_dir, f"epoch_2nd_{epoch:05d}.pth")
            torch.save(state, save_path)

            # ── CheckpointManager: per-epoch retention + SHA-256 (REQ-6) ──────
            _ckpt_path = _ckpt_mgr.save(
                model={key: model[key].state_dict() for key in model},
                optimizer=optimizer.state_dict(),
                epoch=epoch,
                step=iters,
                val_loss=_val_loss,
            )
            _gate.update_last_checkpoint(str(_ckpt_path))
            # ──────────────────────────────────────────────────────────────────

            if model_params.diffusion.dist.estimate_sigma_data:
                config["model_params"]["diffusion"]["dist"]["sigma_data"] = float(
                    np.mean(running_std)
                )
                with open(osp.join(log_dir, osp.basename(config_path)), "w") as outfile:
                    yaml.dump(config, outfile, default_flow_style=True)

        # ── Voicepack extraction + HealthGate val checks (REQ-7) ──────────────
        _ = [model[key].eval() for key in model]
        logger.info(f"Epoch {epoch}: extracting voicepack for TensorBoard inference...")
        _vp, _acoustic_norm, _prosodic_norm = extract_voicepack(
            model, root_path, device, n_samples=200
        )
        writer.add_scalar("voicepack/acoustic_norm", _acoustic_norm, epoch + 1)
        writer.add_scalar("voicepack/prosodic_norm", _prosodic_norm, epoch + 1)
        logger.info(f"  acoustic_norm={_acoustic_norm:.4f}  prosodic_norm={_prosodic_norm:.4f}")

        # Log val metrics to JSONL
        _step_logger.log(
            {
                "epoch": epoch + 1,
                "step": 0,
                "val_loss": _val_loss,
                "dur_loss": float(loss_align / iters_test),
                "f0_loss": float(loss_f / iters_test),
                "acoustic_norm": float(_acoustic_norm),
            }
        )

        # Check acoustic_norm and val_spike gates
        _gate_result = _gate.check(
            epoch=epoch + 1,  # 1-based
            step=0,
            metrics={"val_loss": _val_loss, "acoustic_norm": float(_acoustic_norm)},
        )
        if _gate_result.fired:
            sys.exit(3)

        if _vp is not None and _test_tokens:
            _epoch_audio = run_kokoro_inference(model, _test_tokens, _vp, device, text_cleaner)
            for i, (_text, audio) in enumerate(_epoch_audio):
                writer.add_audio(f"inference/test_{i + 1:02d}", audio, epoch + 1, sample_rate=sr)
        _ = [model[key].train() for key in model]
        # ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    main()
