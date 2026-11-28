"""Paper-scale codec training and token caching (§3.2, App. D.3).

    python -m dyaplex train-codec --prepared P --part upper --out runs/codecs        # one GPU per part
    python -m dyaplex tokenize --prepared P --codecs runs/codecs --out P/tokens

Each part codec is trained independently on one GPU with GestureLSM's optimizer and loss
(config.CodecConfig: AdamW 2e-4 betas 0.9/0.99, warmup 400, MultiStepLR(200k, 0.05), 300k iters,
batch 256 x 64 frames, smooth-L1 + 0.02 commitment) and the checkpoint with the lowest held-out
(val split) reconstruction error is kept [P D.3]. Face codec: pairs passing the 20% zero-face rule.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from . import config as C
from .codec import PartCodec, load_codec, save_state
from .prepare import load_meta, load_norm, load_parts, load_valid, split_pairs


class WindowSampler:
    """Random fully-valid windows from memory-mapped per-participant part arrays."""

    def __init__(self, prepared: Path, part: str, file_ids: list[str], window: int, norm: dict):
        self.part, self.window, self.norm = part, window, norm
        self.items = []
        for fid in file_ids:
            x = load_parts(prepared, fid)[part]
            ok = load_valid(prepared, fid)["face" if part == "face" else "body"]
            n = min(len(x), len(ok))
            if n >= window:
                self.items.append((fid, x, np.concatenate([[0], np.cumsum(~ok[:n])])))
        if not self.items:
            raise ValueError(f"{part}: no participant with >= {window} frames")
        lengths = np.array([len(b) - 1 - window + 1 for _, _, b in self.items], dtype=np.float64)
        self.p = lengths / lengths.sum()

    def batch(self, rng, size: int) -> np.ndarray:
        out = np.empty((size, self.window, C.PART_DIM[self.part]), dtype=np.float32)
        for i in range(size):
            for _ in range(100):
                fid, x, bad = self.items[rng.choice(len(self.items), p=self.p)]
                s = int(rng.integers(0, len(bad) - 1 - self.window + 1))
                if bad[s + self.window] - bad[s] == 0:
                    break
            out[i] = (np.asarray(x[s:s + self.window], dtype=np.float32) - self.norm["mean"]) / self.norm["std"]
        return out


def participants(prepared: Path, split: str, face_only: bool) -> list[str]:
    import csv
    rows = {r["pair"]: r for r in csv.DictReader((prepared / "manifest.csv").open())}
    out = []
    for key, a, b in split_pairs(prepared, split):
        if face_only and rows[key]["face_codec_ok"] != "True":
            continue
        out += [a, b]
    return out


def train(prepared: Path, part: str, out: Path, cfg: C.CodecConfig, device: str = "auto", seed: int = 123,
          eval_every: int = 1000, held_windows: int = 1024, log=print) -> dict:
    dev = torch.device(("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
                       if device == "auto" else device)
    out.mkdir(parents=True, exist_ok=True)
    norm = load_norm(prepared)[part]
    train_ids = participants(prepared, "train", part == "face")
    val_ids = participants(prepared, "val", False)      # 20% face rule applies to training only [P C.2]; windows are face-valid
    sampler = WindowSampler(prepared, part, train_ids, cfg.window, norm)
    held_sampler = WindowSampler(prepared, part, val_ids, cfg.window, norm)
    rng = np.random.default_rng(seed)
    held = torch.tensor(held_sampler.batch(np.random.default_rng(seed + 1), held_windows), device=dev)
    torch.manual_seed(seed)
    model = PartCodec(part, cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(cfg.milestones), gamma=cfg.gamma)
    start, best = 1, math.inf
    last = out / f"{part}_last.pt"
    if last.exists():                                            # resume
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        for layer, (code_sum, code_count) in zip(model.quantizer.layers, ck["ema"]):
            layer.init = True                                   # EMA accumulators are plain attributes
            layer.code_sum, layer.code_count = code_sum.to(dev), code_count.to(dev)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best = ck["iter"] + 1, ck["best"]
        rng = np.random.default_rng(seed + start)
        log(f"resumed {part} at iteration {start}")

    def loss_fn(pred, target):     # nn.SmoothL1Loss() written out (MPS kernel issue); identical values
        diff = (pred - target).abs()
        return torch.where(diff < 1.0, 0.5 * diff * diff, diff - 0.5).mean()

    history_path = out / f"{part}_history.jsonl"
    began = time.time()
    for it in range(start, cfg.total_iter + 1):
        model.train()
        if it < cfg.warmup_iter:
            for group in opt.param_groups:
                group["lr"] = cfg.lr * (it + 1) / (cfg.warmup_iter + 1)
        batch = torch.tensor(sampler.batch(rng, cfg.batch_size), device=dev)
        recon, commit, ppl = model(batch)
        loss = loss_fn(recon, batch) + cfg.commit * commit
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it >= cfg.warmup_iter:
            sched.step()
        if it % eval_every == 0 or it == cfg.total_iter:
            model.eval()
            with torch.no_grad():
                held_loss = float(sum(loss_fn(model.decode(model.encode(held[k:k + 256])), held[k:k + 256])
                                      * len(held[k:k + 256]) for k in range(0, len(held), 256)) / len(held))
            row = {"iter": it, "train_loss": float(loss.detach()), "held_recon": held_loss, "perplexity": float(ppl),
                   "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - began, 1)}
            with history_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            log(f"{part} it {it}: train {row['train_loss']:.4f} held {held_loss:.4f} ppl {row['perplexity']:.0f}")
            if held_loss < best:
                best = held_loss
                torch.save({"part": part, "config": asdict(cfg), "state": save_state(model), "iter": it,
                            "held_recon": held_loss}, out / f"{part}.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "iter": it, "best": best,
                        "ema": [(l.code_sum.detach().cpu(), l.code_count.detach().cpu()) for l in model.quantizer.layers]},
                       last)
    return {"part": part, "best_held_recon": best, "checkpoint": str(out / f"{part}.pt"),
            "train_participants": len(train_ids), "val_participants": len(val_ids)}


def load_codecs(directory: Path, device: str = "cpu") -> dict[str, PartCodec]:
    codecs = {}
    for part in C.PARTS:
        ck = torch.load(directory / f"{part}.pt", map_location="cpu", weights_only=False)
        cfg = C.CodecConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["config"].items()})
        codecs[part] = load_codec(part, cfg, ck["state"]).to(device).eval()
    return codecs


@torch.no_grad()
def tokenize(prepared: Path, codec_dir: Path, out: Path, device: str = "auto", shard: str = "0/1", log=print) -> dict:
    """Encode every participant of every kept pair once; cache (T, 22) uint16 [P D.3 Tokenization]."""
    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
    codecs = load_codecs(codec_dir, dev)
    norm = load_norm(prepared)
    out.mkdir(parents=True, exist_ok=True)
    ids = sorted({pid for split in ("train", "val", "test") for _, a, b in split_pairs(prepared, split) for pid in (a, b)})
    i, n = map(int, shard.split("/"))
    ids = ids[i::n]
    for k, fid in enumerate(ids):
        target = out / f"{fid}.npy"
        if target.exists():
            continue
        parts = load_parts(prepared, fid, mmap=False)
        frames = min(len(v) for v in parts.values()) // 2 * 2
        cols = []
        for part in C.PARTS:
            x = (parts[part][:frames].astype(np.float32) - norm[part]["mean"]) / norm[part]["std"]
            codes = codecs[part].encode(torch.tensor(x[None], device=dev))[0].cpu().numpy()
            cols.append(codes + C.PART_BAND[part])
        np.save(target, np.concatenate(cols, axis=1).astype(np.uint16))
        if (k + 1) % 500 == 0:
            log(f"  tokenized {k + 1}/{len(ids)}")
    return {"participants": len(ids), "out": str(out)}
