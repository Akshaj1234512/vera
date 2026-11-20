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
    args = parser.parse_args()
    if args.command == "fetch-upstream":
        from .vendor import fetch_upstream
        print(json.dumps(fetch_upstream(), indent=2))
    elif args.command == "config":
        from . import config as C
        print(json.dumps({"reference": C.describe(C.REFERENCE), "tiny": C.describe(C.TINY)}, indent=2))
    elif args.command == "e2e":
        from .e2e import run
        report = run(args.out, args.pairs, args.overfit_pairs, seed=args.seed, video=not args.no_video,
                     resume=args.resume, tower_iterations=args.tower_iterations)
        raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
