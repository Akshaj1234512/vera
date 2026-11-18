"""Command-line entry points for VERA speech AI experiments."""

import argparse
import json
import sys
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
        features, scores, threshold = checkpoint_scores(args.checkpoint, features, args.embeddings, args.vad_db)
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


def run_audit(args: argparse.Namespace) -> None:
    from .training import audit

    report = audit(args.manifest, args.vad_db, args.seed, args.tolerance, args.candidates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for split, rows in report["splits"].items():
        matched = sum(row["matches_after_speech_gate"] for row in rows)
        labels = sum(row["labels"] for row in rows)
        print(f"{split:<10} {matched}/{labels} labels reachable within {args.tolerance:g} s")
    print(args.out)


def run_training(args: argparse.Namespace) -> None:
    from .training import train

    report = train(args.manifest, args.out, args.epochs, args.seed, args.tolerance, args.vad_db, args.embeddings, args.candidates)
    print(f"Held-out event F1 ({', '.join(report['origins'])}): rules {report['rules_test']['f1']:.3f}")
    for model in report["models"]:
        print(f"  {model['name']:<20} {model['test']['f1']:.3f}")
    print(args.out / "evaluation.json")


def run_samples(args: argparse.Namespace) -> None:
    from .sources import prepare_samples, prepare_seamless

    output = args.out or Path("data") / args.collection
    if args.collection == "seamless":
        print(prepare_seamless(output))
    else:
        print(prepare_samples(args.collection, output))


def run_personaplex(args: argparse.Namespace) -> None:
    from .personaplex import run

    print(run(args.input, args.out, args.python, args.device, args.voice, args.prompt,
              args.seed, args.cpu_offload, args.capture_states))


def run_simulation(args: argparse.Namespace) -> None:
    from .simulate import replay

    if args.audio and not args.video:
        raise ValueError("--audio requires --video")
    report = replay(args.timeline, args.out, args.video, args.audio)
    print(f'{args.out} | {report["duration_s"]:.2f} s | {report["self_contact_steps"]} self-contact steps')
    if not report["finite_state"] or report["self_contact_steps"]:
        raise SystemExit(1)


def run_pose(args: argparse.Namespace) -> None:
    from .pose import analyze_manifest

    print(analyze_manifest(args.manifest, args.model, args.out, args.side, args.hz,
                           args.minimum_deg, args.export_proxy))


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
    training.add_argument("--candidates", choices=("pause", "cadence"), default="pause")
    training.add_argument("--embeddings", action="store_true", help="also compare models with embeddings")
    training.set_defaults(run=run_training)

    samples = commands.add_parser("samples", help="download and prepare small public published examples")
    samples.add_argument("collection", choices=("personaplex", "vico", "seamless"))
    samples.add_argument("--out", type=Path)
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
    model.add_argument("--capture-states", action="store_true")
    model.set_defaults(run=run_personaplex)

    sim = commands.add_parser("simulate", help="replay cue targets on VERA's existing MuJoCo head")
    sim.add_argument("timeline", type=Path)
    sim.add_argument("--out", type=Path, required=True)
    sim.add_argument("--video", type=Path, help="optional MP4 output (requires a graphics context)")
    sim.add_argument("--audio", type=Path, help="optional aligned audio track to mux into the video")
    sim.set_defaults(run=run_simulation)

    pose = commands.add_parser("pose", help="extract head and jaw tracks from listener videos")
    pose.add_argument("manifest", type=Path, help="sample annotation template with video paths")
    pose.add_argument("--out", type=Path, required=True)
    pose.add_argument("--model", type=Path, default=Path("data/models/face_landmarker.task"))
    pose.add_argument("--side", choices=("full", "left", "right"), default="full")
    pose.add_argument("--hz", type=float, default=10)
    pose.add_argument("--minimum-deg", type=float, default=5)
    pose.add_argument("--export-proxy", action="store_true", help="export unreviewed pitch peaks for a weak-label check")
    pose.set_defaults(run=run_pose)

    audit = commands.add_parser("audit", help="check whether pause candidates can reach the labels before training")
    audit.add_argument("manifest", type=Path)
    audit.add_argument("--out", type=Path, default=Path("runs/coverage.json"))
    audit.add_argument("--seed", type=int, default=17)
    audit.add_argument("--vad-db", type=float, default=-40)
    audit.add_argument("--tolerance", type=float, default=0.30)
    audit.add_argument("--candidates", choices=("pause", "cadence"), default="pause")
    audit.set_defaults(run=run_audit)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
