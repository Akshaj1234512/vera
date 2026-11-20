"""Command-line entry points for the two-model G/R workflow."""

import argparse
import json
from pathlib import Path

from .embodiment import PROFILES

PROFILE_CHOICES = tuple(PROFILES)


def _print(value) -> None:
    print(json.dumps(value, indent=2) if isinstance(value, dict) else value)


def add_commands(commands: argparse._SubParsersAction) -> None:
    def command(name, help_, run):
        parser = commands.add_parser(name, help=help_)
        parser.set_defaults(run=run)
        return parser

    def audio_inputs(parser):
        parser.add_argument("input", type=Path, help="partner WAV (or stereo with --stereo) that defines the clock")
        channels = parser.add_mutually_exclusive_group()
        channels.add_argument("--agent", type=Path, help="aligned robot/agent speech WAV")
        channels.add_argument("--stereo", action="store_true", help="partner left, agent right")

    def models(parser, required_g=False):
        parser.add_argument("--g", type=Path, required=required_g, help="G checkpoint (g.pt or g-imitation.pt)")
        parser.add_argument("--r", type=Path, help="R checkpoint; scores realized motion, never drives the robot")
        parser.add_argument("--policy", choices=("g", "projected"), default="g",
                            help="'projected' replays constrained DyaPlex motion as the baseline")
        parser.add_argument("--profile", choices=PROFILE_CHOICES, default="full")
        parser.add_argument("--calibration", type=Path, help="custom capability JSON {channel: [enabled, min, max, rad/s]}")
        parser.add_argument("--robot", choices=("mujoco", "kinematic", "network"), default="mujoco")
        parser.add_argument("--session", help="session id recorded with the motion")
        parser.add_argument("--group", help="participant/session split group")
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--video", type=Path)
        parser.add_argument("--title")

    def seamless(args):
        from .seamless import prepare as fetch
        _print(fetch(args.label, args.split, args.count, args.out, args.seed, args.include))

    p = command("gr-seamless", "download real human motion + audio (Seamless Interaction) as teacher streams",
                seamless)
    p.add_argument("--label", choices=("naturalistic", "improvised"), default="naturalistic")
    p.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    p.add_argument("--count", type=int, default=30, help="interactions (one per session)")
    p.add_argument("--include", nargs="*", default=[], help="interaction keys that must be included")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--out", type=Path, required=True)

    def dataset(args):
        from .gr_dataset import build
        _print(build(args.streams, args.out, args.seed))

    p = command("gr-dataset", "build G/R trials from real motion streams (votes: SYNTHETIC rule observer)", dataset)
    p.add_argument("streams", type=Path, nargs="+", help="streams.json from gr-seamless")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out", type=Path, required=True)

    def demo(args):
        from .demo import compose
        _print(compose(args.stream, args.partner_wav, args.agent_wav, args.source_video, args.keypoints,
                       args.start_s, args.duration_s, args.g, args.r, args.out, args.label, args.locked))

    p = command("gr-demo", "render a labeled grid: real human, projected DyaPlex and G on the rig", demo)
    p.add_argument("stream", type=Path, help="stream.npz of a held-out participant (from gr-seamless)")
    p.add_argument("--partner-wav", type=Path, required=True)
    p.add_argument("--agent-wav", type=Path, required=True)
    p.add_argument("--source-video", type=Path, required=True, help="the participant's Seamless video")
    p.add_argument("--keypoints", type=Path, required=True, help="Seamless boxes_and_keypoints/keypoints .npy")
    p.add_argument("--start-s", type=float, required=True)
    p.add_argument("--duration-s", type=float, default=40.0)
    p.add_argument("--g", type=Path, required=True)
    p.add_argument("--r", type=Path)
    p.add_argument("--locked", choices=PROFILE_CHOICES, default="fixed-neck-pitch")
    p.add_argument("--label", nargs="+", default=["Human participant", "Seamless Interaction", "measured motion",
                                                   "= teacher input M_h"], help="short lines (240 px column)")
    p.add_argument("--out", type=Path, required=True)

    def fit_map(args):
        from .gr_data import fit_teacher_map
        _print(fit_teacher_map(args.pairs, args.out, args.seed))

    p = command("gr-fit-map", "fit decoded DyaPlex features (e.g. FLAME) to the 21 human-motion channels", fit_map)
    p.add_argument("pairs", type=Path, help="JSON list of {id, group, features, reference}")
    p.add_argument("--out", type=Path, required=True, help="calibration.npz for --teacher-map")
    p.add_argument("--seed", type=int, default=7)

    def prepare(args):
        from .gr_data import prepare_stream
        _print(prepare_stream(args.input, args.agent, args.stereo, args.teacher, args.embeddings, args.out,
                              args.teacher_map, args.teacher_offset_s, args.states_offset_s))

    p = command("gr-prepare", "align DyaPlex motion, audio features and PersonaPlex states", prepare)
    audio_inputs(p)
    p.add_argument("--teacher", type=Path, required=True, help="DyaPlex motion export")
    p.add_argument("--teacher-map", type=Path, help="calibrated projection for decoded DyaPlex features")
    p.add_argument("--embeddings", type=Path, help="PersonaPlex states.npz (omit for audio features only)")
    p.add_argument("--teacher-offset-s", type=float, default=0.0, help="add to teacher clock to reach audio clock")
    p.add_argument("--states-offset-s", type=float, default=0.0, help="add to PersonaPlex clock")
    p.add_argument("--out", type=Path, required=True)

    def trial(args):
        from .gr_data import pack_recording, pack_trial
        if args.all:
            _print(pack_recording(args.stream, args.realized, args.out, args.group, args.condition,
                                  rating_video=args.rating_video))
        else:
            if args.start_s is None:
                raise SystemExit("--start-s or --all is required")
            pack_trial(args.stream, args.realized, args.start_s, args.out)
            _print(args.out)

    p = command("gr-trial", "package recorded 500 ms robot chunks with the inputs G saw", trial)
    p.add_argument("stream", type=Path)
    p.add_argument("realized", type=Path)
    p.add_argument("--start-s", type=float)
    p.add_argument("--all", action="store_true", help="package every chunk into --out (a directory)")
    p.add_argument("--group")
    p.add_argument("--condition")
    p.add_argument("--rating-video", type=Path,
                   help="text-free camera recording of the robot with conversation audio, on the session clock")
    p.add_argument("--out", type=Path, required=True)

    def manifest(args):
        from .gr_eval import build_manifest
        _print(build_manifest(args.fragments, args.labels, args.out, args.intents, args.synthetic))

    p = command("gr-manifest", "merge packaged trials into one manifest with experimenter intents", manifest)
    p.add_argument("fragments", type=Path, nargs="+", help="trials.json from gr-trial --all")
    p.add_argument("--labels", nargs="+", default=["unclear", "neutral_ack", "enthusiastic_ack"])
    p.add_argument("--intents", type=Path, help="CSV with pair,intent written by the experimenter")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", type=Path, required=True)

    def train(args):
        from .gr_train import train as fit
        report = fit(args.manifest, args.out, args.seed, args.epochs, args.refine_epochs,
                     args.meaning_weight, args.smooth_weight, residual=args.residual,
                     imitation_only=args.imitation_only, demo_weight=args.demo_weight)
        print(args.out / "evaluation.json")
        if not report["data"]["human_observer_evidence"]:
            print("NOTE: no human observer votes in this manifest; R and G results are not human evidence.")
        if "observer_test" in report:
            test = report["observer_test"]
            print(f"R held-out cross-entropy {test['soft_cross_entropy']:.3f} | "
                  f"majority agreement {test['majority_vote_accuracy']:.1%} | {', '.join(report['origins'])}")

    p = command("gr-train", "train R on votes, then refine G through frozen R", train)
    p.add_argument("manifest", type=Path)
    p.add_argument("--out", type=Path, default=Path("runs/gr-training"))
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--epochs", type=int, default=36)
    p.add_argument("--refine-epochs", type=int, default=20)
    p.add_argument("--demo-weight", type=float, default=1.0, help="lambda_demo")
    p.add_argument("--meaning-weight", type=float, default=3.0, help="lambda_meaning (selected on validation groups)")
    p.add_argument("--smooth-weight", type=float, default=0.15, help="lambda_temporal")
    p.add_argument("--residual", action="store_true", help="M_r = M_direct + learned adjustment")
    p.add_argument("--imitation-only", action="store_true", help="bootstrap G from approved demonstrations")

    def score_models(args):
        from .gr_train import score
        report = score(args.manifest, args.g, args.r, args.out)
        _print({"macro_p_intent": report["macro_p_intent"], "observer": report["observer"]})

    p = command("gr-score", "evaluate frozen G and R checkpoints on a held-out manifest", score_models)
    p.add_argument("manifest", type=Path)
    p.add_argument("--g", type=Path, required=True)
    p.add_argument("--r", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)

    def replay(args):
        from .gr_runtime import replay as run
        report = run(args.stream, args.g, args.r, args.profile, args.out, args.calibration, args.policy,
                     args.video, args.audio, args.title, args.robot, args.session, args.group, args.robot_address)
        print(f"{args.out} | {report['policy']} / {report['profile']} | {json.dumps(report['metrics'])}")

    p = command("gr-replay", "drive a robot adapter from a prepared stream (simulator check)", replay)
    p.add_argument("stream", type=Path)
    models(p)
    p.add_argument("--audio", type=Path, help="audio to mux into --video")
    p.add_argument("--robot-address", help="HOST:PORT of the robot bridge for --robot network")

    def live(args):
        if args.listen:
            from .gr_link import serve
            offsets = {"teacher": args.teacher_offset_s, "states": args.states_offset_s, "audio": 0.0}
            report = serve(args.listen, args.duration_s, args.g, args.r, args.profile, args.out, args.calibration,
                           args.policy, args.robot, args.robot_address, args.state_dim, args.teacher_map, offsets,
                           args.session, args.group, args.context_id, args.video, args.conversation_audio, args.title,
                           ready=lambda address: print(f"listening on {address[0]}:{address[1]}", flush=True),
                           wait_for_producers=args.wait_for_producers)
        else:
            if args.input is None or args.teacher is None:
                raise SystemExit("Give --listen HOST:PORT for live inputs, or INPUT and --teacher to rehearse from files")
            from .gr_runtime import live as run
            report = run(args.input, args.agent, args.stereo, args.teacher, args.teacher_map, args.embeddings,
                         args.g, args.r, args.profile, args.out, args.calibration, args.policy, args.session,
                         args.group, args.context_id, args.audio_latency_s, args.teacher_offset_s,
                         args.states_offset_s, args.robot, args.video, args.title)
        if report.get("aborted"):
            print(f"ABORTED: {report['aborted']}")
        print(f"{args.out} | {report['origin']} | {report['policy']} / {report['profile']} | "
              f"{json.dumps(report['metrics'])}")

    p = command("gr-live", "run G live on arriving inputs (--listen) or rehearse from recorded files", live)
    p.add_argument("input", type=Path, nargs="?", help="rehearsal only: partner WAV defining the clock")
    channels = p.add_mutually_exclusive_group()
    channels.add_argument("--agent", type=Path)
    channels.add_argument("--stereo", action="store_true")
    p.add_argument("--listen", help="HOST:PORT for producer JSON lines (live mode)")
    p.add_argument("--duration-s", type=float, default=60.0, help="live session length")
    p.add_argument("--wait-for-producers", type=int, default=0,
                   help="start the session clock only once this many producers have connected")
    p.add_argument("--robot-address", help="HOST:PORT of the robot bridge for --robot network")
    p.add_argument("--state-dim", type=int, help="PersonaPlex width when the policy has no checkpoint")
    p.add_argument("--conversation-audio", type=Path, help="session-clock WAV to mux into the videos")
    p.add_argument("--teacher", type=Path, help="rehearsal only: recorded teacher NPZ")
    p.add_argument("--teacher-map", type=Path)
    p.add_argument("--embeddings", type=Path)
    p.add_argument("--teacher-offset-s", type=float, default=0.0)
    p.add_argument("--states-offset-s", type=float, default=0.0)
    p.add_argument("--audio-latency-s", type=float, default=0.0)
    p.add_argument("--context-id", help="shared id for paired sessions over the same conversation")
    models(p)

    def publish(args):
        from .gr_link import publish_files
        _print(publish_files(args.address, args.input, args.agent, args.stereo, args.teacher, args.embeddings,
                             args.latency_s))

    p = command("gr-publish", "dry run: stream recorded inputs into gr-live --listen in real time", publish)
    p.add_argument("address", help="HOST:PORT printed by gr-live --listen")
    audio_inputs(p)
    p.add_argument("--teacher", type=Path, help="teacher NPZ in the DyaPlex motion format")
    p.add_argument("--embeddings", type=Path, help="PersonaPlex states.npz")
    p.add_argument("--latency-s", type=float, default=0.0, help="simulated delivery delay")

    def packet(args):
        from .gr_eval import rating_packet
        _print(rating_packet(args.manifest, args.out, args.context_s, seed=args.seed, key_path=args.key))

    p = command("gr-rating-packet", "cut blinded clips and a rating sheet for observers", packet)
    p.add_argument("manifest", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--context-s", type=float, default=3.0)
    p.add_argument("--seed", type=int)
    p.add_argument("--key", type=Path, help="where to write the code key (default: <out>-key.json, outside --out)")

    def votes(args):
        from .gr_eval import merge_votes
        _print(merge_votes(args.manifest, args.key, args.responses, args.out))

    p = command("gr-votes", "merge observer rating sheets into manifest vote counts", votes)
    p.add_argument("manifest", type=Path)
    p.add_argument("--key", type=Path, required=True)
    p.add_argument("responses", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)

    def evaluate(args):
        from .gr_eval import evaluate as run
        _print(run(args.manifest, args.out, args.sessions, args.r))

    p = command("gr-evaluate", "report intended-vote share and unclear rate per condition", evaluate)
    p.add_argument("manifest", type=Path)
    p.add_argument("--sessions", type=Path, nargs="*", help="replay/live output directories")
    p.add_argument("--r", type=Path)
    p.add_argument("--out", type=Path, required=True)
