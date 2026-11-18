"""Command-line entry points for VERA speech AI experiments."""

import argparse
from pathlib import Path

from .audio import extract, load_tracks
from .cues import PROFILES, RULE_THRESHOLD, build_timeline, rule_scores
from .fixtures import correction, make_corpus
from .replay import plot_timeline, write_timeline


def run_replay(args: argparse.Namespace) -> None:
    features = extract(*load_tracks(args.input, args.agent, args.stereo),
                       threshold_db=args.vad_db)
    threshold = RULE_THRESHOLD
    policy = "rules"
    if args.checkpoint:
        from .training import checkpoint_scores
        scores, threshold = checkpoint_scores(args.checkpoint, features, args.embeddings, args.vad_db)
        policy = args.checkpoint.stem
    else:
        if args.embeddings:
            raise ValueError("--embeddings requires --checkpoint")
        scores = rule_scores(features)
    timeline = build_timeline(features, scores, threshold, args.profile, args.source or args.input.stem, policy)
    write_timeline(timeline, args.out)
    if args.plot:
        plot_timeline(timeline, args.out / "timeline.png")
    print(f'{args.out} | {len(timeline["events"])} cues | '
          f'{len(timeline["overlap_cessations"])} outgoing-audio cessations after overlap')


def run_demo(args: argparse.Namespace) -> None:
    partner, agent = correction(args.out / "audio")
    for profile in ("linked-eyes", "nod-busy", "no-social-motion"):
        run_replay(argparse.Namespace(
            input=partner, agent=agent, stereo=False, source="Synthetic interruption fixture",
            vad_db=-40, profile=profile, out=args.out / profile, plot=True, checkpoint=None, embeddings=None,
        ))


def run_training(args: argparse.Namespace) -> None:
    from .training import train

    report = train(args.manifest, args.out, args.epochs, args.seed, args.tolerance, args.vad_db, args.embeddings)
    print(f"Held-out event F1 ({', '.join(report['origins'])}): rules {report['rules_test']['f1']:.3f}")
    for model in report["models"]:
        print(f"  {model['name']:<20} {model['test']['f1']:.3f}")
    print(args.out / "evaluation.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    replay = commands.add_parser("replay", help="turn aligned WAV tracks into timestamped cue events")
    replay.add_argument("input", type=Path, help="partner PCM WAV, or stereo PCM WAV with --stereo")
    channels = replay.add_mutually_exclusive_group()
    channels.add_argument("--agent", type=Path, help="aligned outgoing PCM WAV")
    channels.add_argument("--stereo", action="store_true", help="partner left, agent right")
    replay.add_argument("--source", help="label for the audio provenance")
    replay.add_argument("--vad-db", type=float, default=-40)
    replay.add_argument("--profile", choices=PROFILES, default="linked-eyes")
    replay.add_argument("--out", type=Path, required=True)
    replay.add_argument("--plot", action="store_true")
    replay.add_argument("--checkpoint", type=Path, help="trained pause scorer")
    replay.add_argument("--embeddings", type=Path, help="timestamped causal speech features (.npz)")
    replay.set_defaults(run=run_replay)

    demo = commands.add_parser("demo", help="render a deterministic three-profile interruption replay")
    demo.add_argument("--out", type=Path, default=Path("runs/demo"))
    demo.set_defaults(run=run_demo)

    corpus = commands.add_parser("synthetic-data", help="write labeled grouped fixtures for training")
    corpus.add_argument("--out", type=Path, default=Path("data/synthetic"))
    corpus.add_argument("--groups", type=int, default=24)
    corpus.add_argument("--seed", type=int, default=12)
    corpus.set_defaults(run=lambda args: print(make_corpus(args.out, args.groups, args.seed)))

    training = commands.add_parser("train", help="compare cue rules, linear and GRU models by held-out group")
    training.add_argument("manifest", type=Path)
    training.add_argument("--out", type=Path, default=Path("runs/training"))
    training.add_argument("--epochs", type=int, default=24)
    training.add_argument("--seed", type=int, default=17)
    training.add_argument("--tolerance", type=float, default=0.30, help="event match tolerance in seconds")
    training.add_argument("--vad-db", type=float, default=-40)
    training.add_argument("--embeddings", action="store_true", help="also compare models with embeddings")
    training.set_defaults(run=run_training)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
