"""Distributed motion-tower training (§4.2-4.3, App. D.2).

    torchrun --nproc_per_node 8 [--nnodes 8 ...] -m dyaplex.train_tower \
        --prepared P --tokens P/tokens --out runs/tower-body --variant body

Reference [P D.2]: AdamW lr 3e-4, betas (0.9, 0.95), weight decay 0.1, grad-clip 1.0,
effective batch 512 windows of 4096 tokens, 30k iterations, 64 H100. Not in the paper [A]:
constant LR (no warmup / decay), bf16 autocast, DDP (the 0.67B tower fits on one GPU; the
frozen 7B speech tower is replicated per rank in bf16), checkpoint every 500 steps.
Effective batch = world_size x micro_batch x grad_accum; grad_accum is derived.
Variants: --variant body (18 supervised codes, Table 2 'Ours') | face (22 codes, face demos);
--ablation no_self_attn | no_partner | no_cross_attn | no_cross_rope (Table 2).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist

from . import config as C
from . import speech as S
from . import streams as ST
from .tower import MotionTower, loss_fn, masked_logits
from .tower_data import PairData, SpeechTower, make_batch


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--tokens", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", choices=("reference", "tiny"), default="reference")
    p.add_argument("--variant", choices=("body", "face"), default="body")
    p.add_argument("--ablation", choices=("none", "no_self_attn", "no_partner", "no_cross_attn", "no_cross_rope"),
                   default="none")
    p.add_argument("--iterations", type=int)
    p.add_argument("--batch-size", type=int, help="effective batch (reference 512)")
    p.add_argument("--micro-batch", type=int, default=8, help="windows per rank per forward")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--val-windows", type=int, default=64)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--grad-checkpoint", action="store_true", help="activation checkpointing per block (memory)")
    p.add_argument("--limit-pairs", type=int, help="smoke tests")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def setup():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        dev = torch.device("cuda", local)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        dev = torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl" if dev.type == "cuda" else "gloo")
    return rank, world, dev


def configs(args):
    base = C.REFERENCE if args.config == "reference" else C.TINY
    tower = replace(base["tower"], supervise="body" if args.variant == "body" else "all",
                    supervise_speakers="B" if args.ablation == "no_partner" else base["tower"].supervise_speakers)
    train = base["tower_train"]
    if args.iterations:
        train = replace(train, iterations=args.iterations)
    if args.batch_size:
        train = replace(train, batch_size=args.batch_size)
    return base["speech"], tower, train


def speech_lm(config: str, dev):
    if config == "reference":
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders
        from .vendor import PERSONAPLEX_CHECKPOINT
        repo, revision = PERSONAPLEX_CHECKPOINT
        lm = loaders.get_moshi_lm(hf_hub_download(repo, loaders.MOSHI_NAME, revision=revision), device=dev)
    else:
        lm = S.build_lm(C.TINY["speech"], seed=0).to(dev)
    for p in lm.parameters():
        p.requires_grad_(False)
    return lm.eval()


def main(argv=None):
    args = parse(argv)
    rank, world, dev = setup()
    scfg, tcfg, train_cfg = configs(args)
    micro = args.micro_batch
    if train_cfg.batch_size % (world * micro):
        raise SystemExit(f"batch {train_cfg.batch_size} is not divisible by world {world} x micro {micro}")
    accum = train_cfg.batch_size // (world * micro)
    out = args.out
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    prompt = ST.prompt(args.prepared)
    if (args.config == "reference") != (prompt["mode"] == "real"):
        raise SystemExit(f"streams were built in mode {prompt['mode']!r}; use matching --config")
    lm = speech_lm(args.config, dev)
    speech = SpeechTower(lm, scfg, prompt["text_prompt"])
    window = tcfg.window_tokens // C.L_STEP
    data = PairData(args.prepared, "train", args.tokens, speech, window, args.limit_pairs)
    val = PairData(args.prepared, "val", args.tokens, speech, window, args.limit_pairs)
    torch.manual_seed(args.seed)
    model = MotionTower(tcfg)
    if args.ablation != "none":
        model.ablate(args.ablation)
    model.to(dev)
    if args.grad_checkpoint:
        _enable_checkpointing(model)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, betas=train_cfg.betas,
                            weight_decay=train_cfg.weight_decay)
    start = 1
    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"] + 1
    ddp = model
    if world > 1:
        ddp = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dev.index] if dev.type == "cuda" else None,
            find_unused_parameters=args.ablation in ("no_self_attn", "no_cross_attn"))
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)
    meta = {"args": {k: str(v) for k, v in vars(args).items()}, "world": world, "micro_batch": micro, "grad_accum": accum,
            "tower": asdict(tcfg), "train": asdict(train_cfg), "speech": asdict(scfg), "streams_mode": prompt["mode"],
            "train_items": len(data.items), "val_items": len(val.items)}
    if rank == 0:
        (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps({k: meta[k] for k in ("world", "micro_batch", "grad_accum", "train_items", "val_items")}),
              flush=True)
    log_path = out / "log.jsonl"
    began = time.time()
    for step in range(start, train_cfg.iterations + 1):
        rng = np.random.default_rng(args.seed + 1_000_003 * rank + step)
        stats = {"ce": 0.0, "va": 0.0, "token_acc": 0.0}
        for k in range(accum):
            batch = make_batch(data.sample(rng, micro), tcfg, dev)
            tokens, labels, bands, vas, pos, qf, sp, s_pos = batch
            sync = k == accum - 1 or world == 1
            ctx = ddp.no_sync() if (world > 1 and not sync) else _null()
            with ctx, amp:
                logits, va = ddp(tokens, pos, qf, sp, s_pos)
                loss, st = loss_fn(logits.float(), va.float(), labels, bands, vas, tcfg.beta_va)
            (loss / accum).backward()
            for key in stats:
                stats[key] += st[key] / accum
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip))
        opt.step()
        opt.zero_grad(set_to_none=True)
        if rank == 0 and (step % args.log_every == 0 or step == start):
            row = {"step": step, **stats, "grad_norm": grad_norm, "seconds": round(time.time() - began, 1)}
            with log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if rank == 0 and (step % args.val_every == 0 or step == train_cfg.iterations):
            row = {"step": step, "val": validate(model, val, tcfg, dev, args.val_windows, amp)}
            with log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if rank == 0 and (step % args.ckpt_every == 0 or step == train_cfg.iterations):
            state = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "tower": asdict(tcfg),
                     "variant": args.variant, "ablation": args.ablation}
            torch.save(state, out / "last.tmp")
            os.replace(out / "last.tmp", last)
            torch.save({"model": model.state_dict(), "tower": asdict(tcfg), "variant": args.variant,
                        "ablation": args.ablation, "step": step}, out / f"tower_{step:06d}.pt")
        if world > 1 and step % args.ckpt_every == 0:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()


@torch.no_grad()
def validate(model, val: PairData, cfg, dev, windows: int, amp) -> dict:
    model.eval()
    correct = total = 0
    ce_sum = 0.0
    for it, c0, starts in val.fixed_windows(windows):
        batch = make_batch(val.windows(it, c0, starts), cfg, dev)
        tokens, labels, bands, vas, pos, qf, sp, s_pos = batch
        with amp:
            logits, va = model(tokens, pos, qf, model.project_speech(sp, s_pos), s_pos)
        sup = labels >= 0
        masked = masked_logits(logits.float()[sup], bands[sup])
        ce_sum += float(torch.nn.functional.cross_entropy(masked, labels[sup], reduction="sum"))
        correct += int((masked.argmax(-1) == labels[sup]).sum())
        total += int(sup.sum())
    model.train()
    return {"ce": ce_sum / max(total, 1), "token_acc": correct / max(total, 1), "targets": total}


def _enable_checkpointing(model: MotionTower) -> None:
    from torch.utils.checkpoint import checkpoint
    for block in model.blocks:
        forward = block.forward

        def wrapped(*a, _f=forward, **kw):
            if kw.get("cache") is None and torch.is_grad_enabled():
                return checkpoint(_f, *a, use_reentrant=False, **kw)
            return _f(*a, **kw)
        block.forward = wrapped


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
