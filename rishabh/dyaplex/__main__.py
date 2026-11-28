"""python -m dyaplex <command>  (run from rishabh/)"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="dyaplex")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch-upstream", help="clone PersonaPlex and GestureLSM at the pinned commits into third_party/")
    e2e = sub.add_parser("e2e", help="smallest end-to-end test (tiny config, real Seamless pairs, Samhita's rig)")
    e2e.add_argument("--out", type=Path, default=Path("runs/dyaplex-e2e"))
    e2e.add_argument("--pairs", type=int, default=10)
    e2e.add_argument("--overfit-pairs", type=int, default=2)
    e2e.add_argument("--seed", type=int, default=0)
    e2e.add_argument("--no-video", action="store_true")
    e2e.add_argument("--resume", action="store_true", help="reuse codecs and continue the tower in --out")
    e2e.add_argument("--tower-iterations", type=int, help="override tiny tower iterations")
    sub.add_parser("config", help="print the reference and tiny configurations")
    eb = sub.add_parser("export-baselines", help="prepared data in DualTalk (69-D) / A2P (132-D) formats")
    eb.add_argument("--prepared", type=Path, required=True)
    eb.add_argument("--out", type=Path, required=True)
    eb.add_argument("--splits", nargs="+", default=["train", "val", "test330"])
    fx = sub.add_parser("make-official-fixture", help="SMOKE TESTS: copy local pairs into the official download layout")
    fx.add_argument("--src", type=Path, required=True)
    fx.add_argument("--out", type=Path, required=True)
    fx.add_argument("--pairs", type=int, default=2)
    inf = sub.add_parser("infer", help="trained checkpoint -> agent motion on Samhita's rig for a recorded pair")
    inf.add_argument("--prepared", type=Path, required=True)
    inf.add_argument("--tokens", type=Path, required=True)
    inf.add_argument("--codecs", type=Path, required=True)
    inf.add_argument("--tower", type=Path, required=True)
    inf.add_argument("--pair", required=True)
    inf.add_argument("--agent", required=True)
    inf.add_argument("--start-s", type=float, default=0.0)
    inf.add_argument("--seconds", type=float, default=20.0)
    inf.add_argument("--config", choices=("reference", "tiny"), default="reference")
    inf.add_argument("--top-k", type=int, default=200)
    inf.add_argument("--out", type=Path, required=True)
    inf.add_argument("--no-video", action="store_true")
    inf.add_argument("--device", default="auto")
    ev = sub.add_parser("evaluate", help="Table 2 protocol: FGD, Diversity, BeatAlign, P-FD, Delta-User")
    ev.add_argument("--prepared", type=Path, required=True)
    ev.add_argument("--tokens", type=Path, required=True)
    ev.add_argument("--out", type=Path, required=True)
    ev.add_argument("--split", default="test330")
    ev.add_argument("--codecs", type=Path)
    ev.add_argument("--tower", type=Path)
    ev.add_argument("--predictions", type=Path, help="score a baseline's <pair>__<agent>.npz outputs instead")
    ev.add_argument("--smplx", type=Path, help="SMPLX_NEUTRAL.npz")
    ev.add_argument("--synthetic-skeleton", action="store_true", help="SMOKE TESTS ONLY")
    ev.add_argument("--config", choices=("reference", "tiny"), default="reference")
    ev.add_argument("--seconds", type=float, default=20.0)
    ev.add_argument("--top-k", type=int, default=200, help="0 = no truncation (App. D.2)")
    ev.add_argument("--roles", choices=("first", "both"), default="first")
    ev.add_argument("--limit", type=int)
    ev.add_argument("--device", default="auto")
    bs = sub.add_parser("build-streams", help="Mimi codes, inner monologue and voice prompt per participant")
    bs.add_argument("--prepared", type=Path, required=True)
    bs.add_argument("--mode", choices=("real", "tiny"), required=True)
    bs.add_argument("--device", default="cuda")
    bs.add_argument("--shard", default="0/1", help="i/n: every n-th participant from i")
    sub.add_parser("train-tower", help="motion tower training (use torchrun -m dyaplex.train_tower for multi-GPU)",
                   add_help=False)
    tc = sub.add_parser("train-codec", help="train one part RVQ-VAE (GestureLSM recipe, one GPU)")
    tc.add_argument("--prepared", type=Path, required=True)
    tc.add_argument("--part", choices=("upper", "hands", "lower_trans", "face"), required=True)
    tc.add_argument("--out", type=Path, required=True)
    tc.add_argument("--iters", type=int, help="override total iterations (reference 300000)")
    tc.add_argument("--batch", type=int, help="override batch size (reference 256)")
    tc.add_argument("--eval-every", type=int, default=1000)
    tc.add_argument("--device", default="auto")
    tk = sub.add_parser("tokenize", help="encode all kept participants with the trained codecs -> (T,22) uint16")
    tk.add_argument("--prepared", type=Path, required=True)
    tk.add_argument("--codecs", type=Path, required=True)
    tk.add_argument("--out", type=Path, required=True)
    tk.add_argument("--device", default="auto")
    tk.add_argument("--shard", default="0/1", help="i/n: every n-th participant from i")
    pr = sub.add_parser("prepare", help="Seamless download -> filtered, normalized DyaPlex features + splits")
    pr.add_argument("--root", type=Path, required=True, help="official download root or per-participant folders")
    pr.add_argument("--out", type=Path, required=True)
    pr.add_argument("--layout", choices=("auto", "official", "feature"), default="auto")
    pr.add_argument("--face-dir", type=Path, help="SPECTRE outputs from face-extract (<file_id>.npz)")
    pr.add_argument("--face-mode", choices=("auto", "spectre", "standin"), default="auto")
    pr.add_argument("--video-check", choices=("local", "remote", "skip"), default="local")
    pr.add_argument("--workers", type=int, default=8)
    pr.add_argument("--limit", type=int, help="first N pairs only (smoke tests)")
    pr.add_argument("--remap-splits", action="store_true", help="SMOKE TESTS ONLY: hash pairs into train/val/test")
    bench = sub.add_parser("bench", help="runtime of the reference-size motion tower at 4096 / 1024 tokens")
    bench.add_argument("--device", default="auto")
    bench.add_argument("--frames", type=int, default=5)
    bench.add_argument("--out", type=Path)
    ex = sub.add_parser("extract-states", help="PersonaPlex per-layer states for Seamless pairs (both role orders)")
    mode = ex.add_mutually_exclusive_group(required=True)
    mode.add_argument("--real", action="store_true", help="PersonaPlex-7B + Mimi + tokenizer (CUDA host, gated weights)")
    mode.add_argument("--tiny", action="store_true", help="local stand-ins, same code path")
    ex.add_argument("--pairs-dir", type=Path, default=Path("data/seamless-motion/naturalistic-dev"))
    ex.add_argument("--pairs", type=int, default=2)
    ex.add_argument("--label", default="naturalistic")
    ex.add_argument("--split", default="dev")
    ex.add_argument("--device", default="cuda")
    ex.add_argument("--out", type=Path, required=True)
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train-tower":
        from .train_tower import main as train_tower
        return train_tower(sys.argv[2:])
    args = parser.parse_args()
    if args.command == "fetch-upstream":
        from .vendor import fetch_upstream
        print(json.dumps(fetch_upstream(), indent=2))
    elif args.command == "config":
        from . import config as C
        print(json.dumps({"reference": C.describe(C.REFERENCE), "tiny": C.describe(C.TINY)}, indent=2))
    elif args.command == "prepare":
        from .prepare import run as prepare
        print(json.dumps(prepare(args.root, args.out, args.layout, args.face_dir, args.face_mode, args.workers,
                                 args.video_check, args.limit, args.remap_splits), indent=2))
    elif args.command == "train-codec":
        from dataclasses import replace
        from . import config as C
        from .codec_train import train
        cfg = C.REFERENCE["codec"]
        if args.iters:
            cfg = replace(cfg, total_iter=args.iters, milestones=(int(args.iters * 2 / 3),),
                          warmup_iter=min(cfg.warmup_iter, max(1, args.iters // 10)))
        if args.batch:
            cfg = replace(cfg, batch_size=args.batch)
        print(json.dumps(train(args.prepared, args.part, args.out, cfg, args.device, eval_every=args.eval_every),
                         indent=2))
    elif args.command == "tokenize":
        from .codec_train import tokenize
        print(json.dumps(tokenize(args.prepared, args.codecs, args.out, args.device, args.shard), indent=2))
    elif args.command == "build-streams":
        from .streams import build
        print(json.dumps(build(args.prepared, args.mode, args.device, args.shard), indent=2))
    elif args.command == "evaluate":
        from .evaluate import run as evaluate
        result = evaluate(args.prepared, args.tokens, args.out, args.split, args.codecs, args.tower, args.smplx,
                          args.synthetic_skeleton, args.config, args.seconds, args.top_k or None, args.roles,
                          args.predictions, limit=args.limit, device=args.device)
        print(json.dumps(result["rows"], indent=2, default=float))
    elif args.command == "infer":
        from .infer import run as infer
        print(json.dumps(infer(args.prepared, args.tokens, args.codecs, args.tower, args.pair, args.agent, args.out,
                               args.start_s, args.seconds, args.config, args.top_k or None, device=args.device,
                               video=not args.no_video), indent=2))
    elif args.command == "export-baselines":
        from .baselines import run as export
        print(json.dumps(export(args.prepared, args.out, tuple(args.splits)), indent=2))
    elif args.command == "make-official-fixture":
        from .sources import make_official_fixture
        print(json.dumps(make_official_fixture(args.src, args.out, args.pairs), indent=2))
    elif args.command == "bench":
        from .bench import run as bench_run
        result = bench_run(args.device, args.frames)
        text = json.dumps(result, indent=2)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n")
        print(text)
    elif args.command == "extract-states":
        from .extract import run as extract
        print(json.dumps(extract(args.pairs_dir, args.out, args.pairs, args.real, args.label, args.split, args.device),
                         indent=2))
    elif args.command == "e2e":
        from .e2e import run
        report = run(args.out, args.pairs, args.overfit_pairs, seed=args.seed, video=not args.no_video,
                     resume=args.resume, tower_iterations=args.tower_iterations)
        raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
