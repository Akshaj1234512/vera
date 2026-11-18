"""Command-line entry points for the two-model G/R workflow."""

import argparse
from pathlib import Path


def make_fixtures(args) -> None:
    from .gr_fixtures import make_fixtures as generate

    print(generate(args.out, args.groups, args.seed))


def prepare(args) -> None:
    from .gr_data import prepare_stream

    prepare_stream(args.input, args.agent, args.stereo, args.teacher, args.embeddings, args.out, args.teacher_map)
    print(args.out)


def trial(args) -> None:
    from .gr_data import pack_trial

    pack_trial(args.stream, args.realized, args.start_s, args.out)
    print(args.out)


def train(args) -> None:
    from .gr_train import train as fit

    report = fit(args.manifest, args.out, args.seed, args.epochs, args.refine_epochs,
                 args.meaning_weight, args.smooth_weight, residual=args.residual,
                 imitation_only=args.imitation_only)
    print(args.out / "evaluation.json")
    if "observer_test" in report:
        print(f'R held-out majority agreement {report["observer_test"]["majority_vote_accuracy"]:.1%}'
              f' | {", ".join(report["origins"])}')


def replay(args) -> None:
    from .gr_runtime import run

    if args.audio and not args.video:
        raise ValueError("--audio requires --video")
    report = run(args.g, args.r, args.stream, args.profile, args.out, args.video,
                 args.audio, args.calibration, args.source)
    print(f'{args.out} | {report["duration_s"]:.2f} s | {report["self_contact_steps"]} self-contact steps')
    if not report["finite_state"] or report["self_contact_steps"]:
        raise SystemExit(1)


def add_commands(commands: argparse._SubParsersAction) -> None:
    fixtures = commands.add_parser("gr-fixture", help="generate synthetic motion, context and observer votes")
    fixtures.add_argument("--out", type=Path, default=Path("data/gr-synthetic"))
    fixtures.add_argument("--groups", type=int, default=16)
    fixtures.add_argument("--seed", type=int, default=31)
    fixtures.set_defaults(run=make_fixtures)

    sources = commands.add_parser("gr-prepare", help="align DyaPlex motion and PersonaPlex features")
    sources.add_argument("input", type=Path, help="audio that defines the recording duration and clock")
    channels = sources.add_mutually_exclusive_group()
    channels.add_argument("--agent", type=Path)
    channels.add_argument("--stereo", action="store_true")
    sources.add_argument("--teacher", type=Path, required=True, help="DyaPlex motion export")
    sources.add_argument("--teacher-map", type=Path, help="calibrated projection for decoded DyaPlex features")
    sources.add_argument("--embeddings", type=Path, required=True, help="PersonaPlex state export")
    sources.add_argument("--out", type=Path, required=True)
    sources.set_defaults(run=prepare)

    packed = commands.add_parser("gr-trial", help="package a recorded 500 ms robot chunk for labeling")
    packed.add_argument("stream", type=Path)
    packed.add_argument("realized", type=Path)
    packed.add_argument("--start-s", type=float, required=True)
    packed.add_argument("--out", type=Path, required=True)
    packed.set_defaults(run=trial)

    training = commands.add_parser("gr-train", help="train R, then refine G with R and motion losses")
    training.add_argument("manifest", type=Path)
    training.add_argument("--out", type=Path, default=Path("runs/gr-training"))
    training.add_argument("--seed", type=int, default=17)
    training.add_argument("--epochs", type=int, default=36)
    training.add_argument("--refine-epochs", type=int, default=8)
    training.add_argument("--meaning-weight", type=float, default=0.04)
    training.add_argument("--smooth-weight", type=float, default=0.15)
    training.add_argument("--residual", action="store_true", help="learn an adjustment to projected teacher motion")
    training.add_argument("--imitation-only", action="store_true", help="train G from demonstrations to collect ratings")
    training.set_defaults(run=train)

    simulation = commands.add_parser("gr-replay", help="drive the MuJoCo head with G and rate realized motion with R")
    simulation.add_argument("stream", type=Path)
    simulation.add_argument("--g", type=Path, required=True)
    simulation.add_argument("--r", type=Path, help="optional observer checkpoint for evaluation")
    simulation.add_argument("--out", type=Path, required=True)
    simulation.add_argument("--profile", choices=("full", "fixed-neck-pitch", "fixed-eyes"), default="full")
    simulation.add_argument("--calibration", type=Path, help="custom capability descriptor JSON")
    simulation.add_argument("--source", help="recording title")
    simulation.add_argument("--video", type=Path)
    simulation.add_argument("--audio", type=Path)
    simulation.set_defaults(run=replay)
