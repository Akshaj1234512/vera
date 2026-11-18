"""Command-line entry points for VERA speech AI experiments."""

import argparse
from pathlib import Path

from .audio import extract, load_tracks
from .cues import PROFILES, RULE_THRESHOLD, build_timeline, rule_scores
from .fixtures import correction
from .replay import plot_timeline, write_timeline


def run_replay(args: argparse.Namespace) -> None:
    features = extract(*load_tracks(args.input, args.agent, args.stereo),
                       threshold_db=args.vad_db)
    scores = rule_scores(features)
    timeline = build_timeline(features, scores, RULE_THRESHOLD, args.profile, args.source or args.input.stem, "rules")
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
            vad_db=-40, profile=profile, out=args.out / profile, plot=True,
        ))


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
    replay.set_defaults(run=run_replay)

    demo = commands.add_parser("demo", help="render a deterministic three-profile interruption replay")
    demo.add_argument("--out", type=Path, default=Path("runs/demo"))
    demo.set_defaults(run=run_demo)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
