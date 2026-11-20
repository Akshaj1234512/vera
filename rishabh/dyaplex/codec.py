"""Causal part-aware RVQ-VAE (App. D.3), built from the pinned public GestureLSM modules.

The encoder, decoder and residual quantizer are GestureLSM's own classes
(third_party/GestureLSM/models/vq/{encdec,residual_vq,quantizer}.py @ f52ac2f). The two
changes DyaPlex states [P D.3] are constructor arguments, not code edits:
  (i)  causal=True  -> GestureLSM's CausalConv1d (left padding only) in encoder and decoder
  (ii) down_t=1     -> a single stride-2 stage, 25 fps -> 12.5 Hz
The face codec uses 4 quantizers instead of 6 [P D.3 table] (review X-5).
One portability patch: QuantizeEMAReset.reset_codebook hard-codes ``.cuda()``; it is
replaced by a device-agnostic copy with identical logic.
"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C
from .vendor import use_gesturelsm

use_gesturelsm()
from models.vq import quantizer as _quantizer  # noqa: E402  (GestureLSM)
from models.vq.encdec import Decoder, Encoder  # noqa: E402
from models.vq.residual_vq import ResidualVQ  # noqa: E402


def _reset_codebook(self):  # identical to GestureLSM except for the hard-coded .cuda()
    self.init = False
    self.code_sum = None
    self.code_count = None
    self.register_buffer("codebook", torch.zeros(self.nb_code, self.code_dim, requires_grad=False))


_quantizer.QuantizeEMAReset.reset_codebook = _reset_codebook


class PartCodec(nn.Module):
    """One body-part RVQ-VAE. Mirrors GestureLSM RVQVAE (models/vq/model.py) with causal convs."""

    def __init__(self, part: str, cfg: C.CodecConfig):
        super().__init__()
        self.part, self.cfg = part, cfg
        self.input_dim = C.PART_DIM[part]
        self.num_quantizers = C.PART_QUANTIZERS[part]
        self.encoder = Encoder(self.input_dim, cfg.code_dim, cfg.down_t, cfg.stride_t, cfg.width, cfg.depth,
                               cfg.dilation_growth_rate, activation=cfg.activation, norm=None, causal=cfg.causal)
        self.decoder = Decoder(self.input_dim, cfg.code_dim, cfg.down_t, cfg.stride_t, cfg.width, cfg.depth,
                               cfg.dilation_growth_rate, activation=cfg.activation, norm=None, causal=cfg.causal)
        self.quantizer = ResidualVQ(num_quantizers=self.num_quantizers, shared_codebook=False,
                                    quantize_dropout_prob=cfg.quantize_dropout_prob, quantize_dropout_cutoff_index=0,
                                    nb_code=cfg.nb_code, code_dim=cfg.code_dim, args=Namespace(mu=cfg.mu))

    # x: [B, T, D] normalized part features at 25 fps
    def forward(self, x: torch.Tensor):
        latent = self.encoder(x.permute(0, 2, 1).float())
        quantized, _, commit, perplexity = self.quantizer(latent, sample_codebook_temp=self.cfg.sample_codebook_temp)
        return self.decoder(quantized).contiguous(), commit, perplexity  # GestureLSM Decoder returns [B, T, D]

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, D] -> codes [B, T // 2, Q] (local indices 0..1023)."""
        latent = self.encoder(x.permute(0, 2, 1).float())
        return self.quantizer.quantize(latent)

    @torch.no_grad()
    def latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.permute(0, 2, 1).float())

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes [B, T', Q] -> [B, 2 T', D]."""
        latent = self.quantizer.get_codes_from_indices(codes).sum(dim=0).permute(0, 2, 1)
        return self.decoder(latent).contiguous()


def device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _windows(features: list[np.ndarray], valid: list[np.ndarray], window: int) -> list[tuple[int, int]]:
    """Start indices of fully valid windows (invalid frames are excluded from codec input [P C.1])."""
    starts = []
    for i, (x, ok) in enumerate(zip(features, valid)):
        bad = np.concatenate([[0], np.cumsum(~ok)])
        for s in range(0, len(x) - window + 1):
            if bad[s + window] - bad[s] == 0:
                starts.append((i, s))
    return starts


def train_codec(part: str, cfg: C.CodecConfig, train: list[np.ndarray], train_valid: list[np.ndarray],
                held: list[np.ndarray], held_valid: list[np.ndarray], seed: int = 123, log=print,
                evaluate_every: int = 100) -> tuple[PartCodec, dict]:
    """GestureLSM recipe (rvq_beatx_train.py:287-370): AdamW, linear warmup, MultiStepLR,
    smooth-L1 reconstruction + 0.02 commitment. Checkpoint = lowest held-out error [P D.3]."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = device()
    model = PartCodec(part, cfg).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(cfg.milestones), gamma=cfg.gamma)
    starts = _windows(train, train_valid, cfg.window)
    held_starts = _windows(held, held_valid, cfg.window)
    if not starts or not held_starts:
        raise ValueError(f"{part}: no fully valid {cfg.window}-frame windows")
    held_idx = rng.choice(len(held_starts), size=min(256, len(held_starts)), replace=False)
    held_batch = torch.tensor(np.stack([held[i][s:s + cfg.window] for i, s in (held_starts[k] for k in held_idx)]),
                              device=dev)
    def loss_fn(pred, target):
        """nn.SmoothL1Loss() (beta=1, mean), written out: the MPS kernel fails on some shapes."""
        diff = (pred - target).abs()
        return torch.where(diff < 1.0, 0.5 * diff * diff, diff - 0.5).mean()
    best, best_state, history = math.inf, None, []
    began = time.time()
    for it in range(1, cfg.total_iter + 1):
        model.train()
        if it < cfg.warmup_iter:
            for group in optimizer.param_groups:
                group["lr"] = cfg.lr * (it + 1) / (cfg.warmup_iter + 1)
        pick = rng.integers(0, len(starts), size=cfg.batch_size)
        batch = torch.tensor(np.stack([train[i][s:s + cfg.window] for i, s in (starts[k] for k in pick)]), device=dev)
        recon, commit, perplexity = model(batch)
        loss = loss_fn(recon, batch) + cfg.commit * commit + cfg.loss_vel * 0.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if it >= cfg.warmup_iter:
            scheduler.step()
        if it % evaluate_every == 0 or it == cfg.total_iter:
            model.eval()
            with torch.no_grad():
                held_loss = float(loss_fn(model.decode(model.encode(held_batch)), held_batch))
            history.append({"iter": it, "train_loss": float(loss.detach()), "held_recon": held_loss,
                            "perplexity": float(perplexity), "seconds": round(time.time() - began, 1)})
            log(f"  codec {part} it {it}: train {float(loss):.4f} held {held_loss:.4f} ppl {float(perplexity):.1f}")
            if held_loss < best:
                best = held_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.cpu().eval()
    return model, {"part": part, "best_held_recon": best, "history": history, "config": asdict(cfg)}


def load_codec(part: str, cfg: C.CodecConfig, state: dict) -> PartCodec:
    model = PartCodec(part, cfg)
    model.load_state_dict(state, strict=True)   # codebooks are registered buffers
    for layer in model.quantizer.layers:
        layer.init = True
    return model.eval()


def save_state(model: PartCodec) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# --------------------------------------------------------------------------- #
# Token bands: local codes <-> shared vocabulary [P D.3]
# --------------------------------------------------------------------------- #
def to_global(part_codes: dict[str, np.ndarray]) -> np.ndarray:
    """{part: [T', Q_part]} local codes -> [T', 22] global ids in canonical slot order."""
    return np.concatenate([part_codes[p] + C.PART_BAND[p] for p in C.PARTS], axis=1).astype(np.int64)


def to_local(global_codes: np.ndarray) -> dict[str, np.ndarray]:
    out, col = {}, 0
    for part in C.PARTS:
        q = C.PART_QUANTIZERS[part]
        out[part] = global_codes[:, col:col + q] - C.PART_BAND[part]
        col += q
    return out


def encode_all(codecs: dict[str, PartCodec], parts: dict[str, np.ndarray]) -> np.ndarray:
    frames = min(len(v) for v in parts.values()) // 2 * 2
    local = {p: codecs[p].encode(torch.tensor(parts[p][None, :frames])).squeeze(0).numpy() for p in C.PARTS}
    return to_global(local)


def decode_all(codecs: dict[str, PartCodec], global_codes: np.ndarray) -> dict[str, np.ndarray]:
    local = to_local(global_codes)
    return {p: codecs[p].decode(torch.tensor(local[p][None])).squeeze(0).numpy() for p in C.PARTS}
