"""PersonaPlex states for G only: G sees PCA-reduced speech states, R stays audio-only.

The synthetic votes depend only on motion, so R gains nothing real from speech context, and with
4096 raw dims it overfits to it. This keeps R on the 8 audio features and gives G
audio + PCA(K) of the PersonaPlex states + their availability bit. Training reuses
gr_train.train unchanged; only the model constructors are swapped.

Streams must come from `gr-prepare --embeddings` (context = 8 audio + 4096 states + 1 available).

Needs Rishabh's G/R code from the `vera-gr-model` branch (its `rishabh/` folder); point RISHABH_CODE at it:

  export RISHABH_CODE=/path/to/vera-gr-model-checkout/rishabh
  python richard/personaplex/gr_pp.py train data/dev/manifest.json --components 64 --out runs/pp-g
  python richard/personaplex/gr_pp.py score data/test/manifest.json --judge r.pt \
      --audio-g g.pt --pp-g runs/pp-g/g.pt --out score.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch import nn

if not os.environ.get("RISHABH_CODE"):
    sys.exit("Set RISHABH_CODE to the rishabh/ folder of a vera-gr-model checkout (see the docstring).")
sys.path.insert(0, os.environ["RISHABH_CODE"])
from speech_ai import gr_train  # noqa: E402
from speech_ai.gr_data import load_manifest, split_groups  # noqa: E402
from speech_ai.gr_model import Generator, Observer  # noqa: E402

AUDIO = 8
STATES = 4096
AVAILABLE = AUDIO + STATES  # index of the PersonaPlex availability bit


class SpeechContext(nn.Module):
    """Full context -> audio + standardized PCA(K) of the states (zeroed when unavailable) + availability."""

    def __init__(self, components: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(STATES))
        self.register_buffer("basis", torch.zeros(STATES, components))
        self.register_buffer("scale", torch.ones(components))

    def forward(self, context):
        available = context[:, AVAILABLE:AVAILABLE + 1]
        speech = (context[:, AUDIO:AVAILABLE] - self.mean) @ self.basis / self.scale
        return torch.cat([context[:, :AUDIO], speech * available, available], dim=1)


class PPGenerator(nn.Module):
    def __init__(self, components: int, residual: bool = False):
        super().__init__()
        self.speech = SpeechContext(components)
        self.g = Generator(AUDIO + components + 1, residual=residual)

    def forward(self, human, context, limits, state):
        return self.g(human, self.speech(context), limits, state)


class AudioOnly(nn.Module):
    """Wrap an audio-context model (Rishabh's G or R) so it accepts the full context."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, first, context, *rest):  # G(human, context, limits, state) / R(motion, context, limits)
        return self.model(first, context[:, :AUDIO], *rest)


def check_context(trials):
    width = len(trials[0].context)
    if width != AVAILABLE + 1:
        raise ValueError(f"expected {AVAILABLE + 1} context dims (8 audio + 4096 states + 1), got {width}")


def fit_pca(trials, components: int, seed: int) -> SpeechContext:
    rows = torch.stack([torch.as_tensor(t.context[AUDIO:AVAILABLE]) for t in trials if t.context[AVAILABLE] > 0.5])
    if len(rows) > 20000:
        rows = rows[torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed))[:20000]]
    rows = rows.double()
    mean = rows.mean(0)
    _, singular, v = torch.linalg.svd(rows - mean, full_matrices=False)
    basis = v[:components].T
    scale = ((rows - mean) @ basis).std(0)
    explained = float((singular[:components] ** 2).sum() / (singular ** 2).sum())
    speech = SpeechContext(components)
    speech.mean.copy_(mean.float()); speech.basis.copy_(basis.float()); speech.scale.copy_(scale.float())
    print(f"PCA on {len(rows)} train-split rows: {components} components explain {explained:.1%} of state variance")
    return speech, explained


def train(args):
    labels, trials = load_manifest(args.manifest)
    check_context(trials)
    splits = split_groups(trials, args.seed)  # same split gr_train.train makes with this seed
    speech, explained = fit_pca(splits["train"], args.components, args.seed)
    del trials, splits

    def make_generator(context_dim, residual=False):
        model = PPGenerator(args.components, residual)
        model.speech.load_state_dict(speech.state_dict())
        return model

    original = gr_train.Generator, gr_train.Observer
    gr_train.Generator = make_generator
    gr_train.Observer = lambda context_dim, n_labels: AudioOnly(Observer(AUDIO, n_labels))
    try:
        report = gr_train.train(args.manifest, args.out, args.seed, args.epochs, args.refine_epochs,
                                args.meaning_weight, args.smooth_weight, demo_weight=args.demo_weight)
    finally:
        gr_train.Generator, gr_train.Observer = original
    extra = {"personaplex_g_only": {"components": args.components, "explained_variance": explained,
                                    "g_context": f"{AUDIO} audio + {args.components} PCA + 1 available",
                                    "r_context": f"{AUDIO} audio"}}
    path = args.out / "evaluation.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **extra}, indent=2) + "\n")
    print(json.dumps(report["validation_macro_p_intent"]))


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def score(args):
    labels, trials = load_manifest(args.manifest)
    check_context(trials)
    judge = load(args.judge)
    observer = Observer(judge["context_dim"], len(judge["labels"]))
    if judge["context_dim"] == AUDIO:
        observer.load_state_dict(judge["weights"])
        observer = AudioOnly(observer)
    else:  # an AudioOnly-wrapped R saved by `train`
        observer = AudioOnly(Observer(AUDIO, len(judge["labels"])))
        observer.load_state_dict(judge["weights"])
    observer.eval()
    if list(labels) != list(judge["labels"]):
        raise ValueError("Manifest labels differ from the judge")
    generators = {}
    if args.audio_g:
        data = load(args.audio_g)
        model = Generator(data["context_dim"], residual=data["residual"])
        model.load_state_dict(data["weights"])
        generators["audio_g"] = AudioOnly(model).eval()
    if args.pp_g:
        data = load(args.pp_g)
        model = PPGenerator(data["weights"]["speech.basis"].shape[1], residual=data["residual"])
        model.load_state_dict(data["weights"])
        generators["pp_g"] = model.eval()
    rows = [t for t in trials if t.intent >= 0]
    data = gr_train.batch(rows)
    reviewed = [t for t in trials if t.votes.any()]
    report = {"manifest": str(args.manifest.resolve()), "judge": str(args.judge),
              "observer": gr_train.observer_result(observer, gr_train.batch(reviewed)), "policies": {}}
    with torch.no_grad():
        for name, generator in generators.items():
            table = gr_train.intent_table(rows, data, observer, generator, labels)
            report["policies"][name] = {
                "by_profile_and_intent": table,
                "macro_p_intent": {p: sum(c[p] for c in table.values()) / len(table)
                                   for p in ("projected_streaming_upper_bound", "projected_chunk_hold", "g_refined")}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v["macro_p_intent"] for k, v in report["policies"].items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(required=True)
    p = commands.add_parser("train")
    p.add_argument("manifest", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--components", type=int, default=64)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--epochs", type=int, default=36)
    p.add_argument("--refine-epochs", type=int, default=20)
    p.add_argument("--demo-weight", type=float, default=1.0)
    p.add_argument("--meaning-weight", type=float, default=3.0)
    p.add_argument("--smooth-weight", type=float, default=0.15)
    p.set_defaults(run=train)
    p = commands.add_parser("score")
    p.add_argument("manifest", type=Path)
    p.add_argument("--judge", type=Path, required=True, help="R checkpoint used to score every policy")
    p.add_argument("--audio-g", type=Path)
    p.add_argument("--pp-g", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=score)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
