"""VERA speech AI: the G (expression generator) / R (interpretation model) workflow."""

import argparse
import sys
from pathlib import Path


def run_samples(args: argparse.Namespace) -> None:
    from .sources import prepare_personaplex

    print(prepare_personaplex(args.out))


def run_personaplex(args: argparse.Namespace) -> None:
    from .personaplex import run

    print(run(args.input, args.out, args.python, args.device, args.voice, args.prompt,
              args.seed, args.cpu_offload, args.capture_states))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    samples = commands.add_parser("samples", help="download NVIDIA's published PersonaPlex conversation audio")
    samples.add_argument("--out", type=Path, default=Path("data/personaplex"))
    samples.set_defaults(run=run_samples)

    from .personaplex import DEFAULT_PROMPT
    model = commands.add_parser("personaplex", help="run NVIDIA PersonaPlex in its installed environment")
    model.add_argument("input", type=Path)
    model.add_argument("--out", type=Path, required=True)
    model.add_argument("--python", default=sys.executable, help="Python where NVIDIA Moshi is installed")
    model.add_argument("--device", default="cuda")
    model.add_argument("--voice", default="NATF2.pt")
    model.add_argument("--prompt", default=DEFAULT_PROMPT)
    model.add_argument("--seed", type=int, default=42424242)
    model.add_argument("--cpu-offload", action="store_true")
    model.add_argument("--capture-states", action="store_true", help="save timestamped states.npz for gr-prepare")
    model.set_defaults(run=run_personaplex)

    from .gr_cli import add_commands
    add_commands(commands)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
