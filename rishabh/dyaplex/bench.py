"""Runtime measurement at the two published motion contexts (§5 'Runtime performance').

Paper (RTX A6000 Ada, 12.5 Hz): speech tower 30 ms/frame, motion tower 173 ms/frame at 4096 motion
tokens and 80 ms/frame at 1024 tokens (speech context kept at 7.1 s = 89 frames), RVQ-VAE
decoders 0.8 ms/frame. Here: reference-size motion tower (random weights; latency does not
depend on the values), agent-only mode, steady state after filling the window, fp32 [A precision
unstated]. The 7B speech tower needs the gated checkpoint and CUDA, so it is not measured here.
"""

from __future__ import annotations

from dataclasses import replace
import json
import platform
import time

import numpy as np
import torch

from . import config as C
from . import layout
from .codec import PartCodec
from .generate import Session
from .tower import MotionTower


def _random_codes(frames: int, rng) -> np.ndarray:
    return np.stack([rng.integers(layout.BAND_LO[layout.SLOT_PART], layout.BAND_HI[layout.SLOT_PART])
                     for _ in range(frames)])


def _sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def motion_tower(dev: torch.device, window_tokens: int, speech_frames: int, timed: int = 5, seed: int = 0) -> dict:
    cfg = replace(C.REFERENCE["tower"], window_tokens=window_tokens, speech_window_frames=speech_frames)
    torch.manual_seed(seed)
    model = MotionTower(cfg).to(dev).eval()
    fill = window_tokens // C.L_STEP
    total = fill + timed + 1
    rng = np.random.default_rng(seed)
    speech = [torch.randn(1, total, cfg.speech_dim) for _ in range(cfg.layers)]
    session = Session(model, speech, C.REFERENCE["sample"], seed, use_cache=True)
    session.prefill(_random_codes(fill, rng), _random_codes(fill, rng))
    partner = _random_codes(timed + 1, rng)
    session.step_frame(partner[0])                       # warm-up frame (kernels, caches)
    _sync(dev)
    times = []
    for f in range(1, timed + 1):
        t0 = time.perf_counter()
        session.step_frame(partner[f])
        _sync(dev)
        times.append(1000 * (time.perf_counter() - t0))
    kv = session.caches[0]["k"].shape[2]
    del model, session
    if dev.type == "mps":
        torch.mps.empty_cache()
    return {"window_tokens": window_tokens, "speech_window_frames": speech_frames, "kv_tokens_at_end": int(kv),
            "ms_per_frame_median": float(np.median(times)), "ms_per_frame": [round(t, 1) for t in times],
            "tokens_per_frame": 46, "sampled_per_frame": 22}


@torch.no_grad()
def decoders(dev: torch.device, history: int = 16, repeats: int = 20) -> dict:
    """All four part decoders for one new 12.5 Hz frame, decoding a `history`-token causal window."""
    codecs = {}
    for part in C.PARTS:
        m = PartCodec(part, C.REFERENCE["codec"]).to(dev)
        m.train()
        m(torch.randn(2, 64, C.PART_DIM[part], device=dev))     # initialize EMA codebooks
        codecs[part] = m.eval()
    codes = {p: torch.randint(0, 1024, (1, history, C.PART_QUANTIZERS[p]), device=dev) for p in C.PARTS}
    for p in C.PARTS:
        codecs[p].decode(codes[p])
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(repeats):
        for p in C.PARTS:
            codecs[p].decode(codes[p])[:, -2:]
    _sync(dev)
    return {"history_tokens": history, "ms_per_frame": 1000 * (time.perf_counter() - t0) / repeats}


def run(device: str = "auto", timed: int = 5) -> dict:
    dev = torch.device(("mps" if torch.backends.mps.is_available() else "cpu") if device == "auto" else device)
    out = {"device": str(dev), "machine": platform.machine(), "torch": torch.__version__,
           "tower_params_M": sum(p.numel() for p in MotionTower(C.REFERENCE["tower"]).parameters()) / 1e6,
           "paper_ms": {"speech_tower": 30, "motion_4096": 173, "motion_1024": 80, "decoders": 0.8},
           "motion": [motion_tower(dev, 4096, 89, timed), motion_tower(dev, 1024, 89, timed)],
           "decoders": decoders(dev),
           "speech_tower": "not measured: needs nvidia/personaplex-7b-v1 (gated) on a CUDA GPU"}
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
