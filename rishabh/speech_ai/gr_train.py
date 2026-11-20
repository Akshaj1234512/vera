"""Fit the observer, pretrain the generator, then train G with frozen R."""

from copy import deepcopy
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .audio import FRAME_S
from .gr_data import Trial, file_digest, load_manifest, split_groups
from .gr_model import CHANNELS, PROFILES, SCALE, Generator, Observer, capability, constrain_motion, hold_last_valid
from .human import RETARGET_NORM

CHECKPOINT_VERSION = 2  # v2: 21-channel human input, 27-channel Samhita rig output


def batch(trials: list[Trial]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([torch.as_tensor(getattr(trial, key)) for trial in trials])
            for key in ("human", "context", "state", "limits", "robot", "demonstration", "votes", "intent")}


def generator_output(model: Generator, data: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(data["human"], data["context"], data["limits"], data["state"])


def observation(model: Observer, motion: torch.Tensor, data: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(motion, data["context"], data["limits"])


def imitation_error(motion: torch.Tensor, data: dict[str, torch.Tensor], *, absolute: bool = False) -> torch.Tensor:
    mask = data["limits"][:, None, :, 0] * data["demonstration"][:, None, None]
    delta = motion - data["robot"]
    error = delta.abs() * torch.as_tensor(SCALE) if absolute else delta.square()
    return (error * mask).sum() / (mask.sum() * motion.shape[1]).clamp_min(1)


def smoothness(motion: torch.Tensor, data: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean squared change in per-step velocity, including the step across the chunk boundary.

    The measured velocity at the boundary anchors the first difference, so a chunk
    that starts with an abrupt reversal is penalized as much as one inside a chunk.
    """
    first_position = data["state"][:, :1, :]
    velocity = torch.diff(torch.cat([first_position, motion], dim=1), dim=1)
    previous_step = data["state"][:, 1:2, :] * FRAME_S
    delta = torch.diff(torch.cat([previous_step, velocity], dim=1), dim=1)
    return delta.square().mean()


def fit(model, train, valid, loss_fn, epochs: int, seed: int, learning_rate: float) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    random = torch.Generator().manual_seed(seed)
    best_loss = float("inf")
    best = deepcopy(model.state_dict())
    for _ in range(epochs):
        model.train()
        for ids in torch.randperm(len(train["context"]), generator=random).split(256):
            optimizer.zero_grad(set_to_none=True)
            value = loss_fn(model, {key: values[ids] for key, values in train.items()})
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            value = float(loss_fn(model, valid))
        if value < best_loss:
            best_loss = value
            best = deepcopy(model.state_dict())
    if best_loss == float("inf"):
        raise ValueError("All validation losses were non-finite")
    model.load_state_dict(best)


def subsets(splits: dict[str, list[Trial]], predicate, purpose: str, *, required: bool = True) -> dict:
    output = {}
    for name, trials in splits.items():
        rows = [trial for trial in trials if predicate(trial)]
        if rows:
            output[name] = batch(rows)
        elif required:
            raise ValueError(f"{name} needs at least one {purpose} trial")
    return output


def observer_loss(model, data):
    return F.cross_entropy(observation(model, data["robot"], data), data["votes"])


def observer_result(model, data):
    logits = observation(model, data["robot"], data)
    correct = logits.argmax(dim=1) == data["votes"].argmax(dim=1)
    return {"soft_cross_entropy": float(F.cross_entropy(logits, data["votes"])),
            "majority_vote_accuracy": float(correct.float().mean()), "trials": len(correct)}


def projected(data: dict[str, torch.Tensor]) -> torch.Tensor:
    """The projected-DyaPlex baseline in chunk form: newest human frame retargeted, held, constrained."""
    target = (hold_last_valid(data["human"]) @ torch.from_numpy(RETARGET_NORM))[:, None].expand(-1, 10, -1)
    return constrain_motion(target, data["limits"], data["state"][:, 0])


def generator_result(motion: torch.Tensor, observer: Observer, data: dict[str, torch.Tensor]) -> dict:
    logits = observation(observer, motion, data)
    labeled = data["intent"] >= 0
    intended = data["intent"][labeled]
    probabilities = logits[labeled].softmax(dim=1).gather(1, intended[:, None]).flatten()
    return {
        "demonstration_mae": (float(imitation_error(motion, data, absolute=True))
                                  if data["demonstration"].any() else None),
        "observer_mean_intended_probability": float(probabilities.mean()) if len(intended) else None,
        "observer_intended_top1": (float((logits[labeled].argmax(dim=1) == intended).float().mean())
                                    if len(intended) else None),
        "smoothness": float(smoothness(motion, data)),
        "trials": len(motion),
    }


def train(manifest: Path, output: Path, seed: int, epochs: int, refine_epochs: int,
          meaning_weight: float, smooth_weight: float, *, residual: bool = False,
          imitation_only: bool = False, demo_weight: float = 1.0) -> dict:
    if (epochs < 1 or refine_epochs < 1 or not math.isfinite(meaning_weight) or meaning_weight <= 0
            or not math.isfinite(smooth_weight) or smooth_weight < 0
            or not math.isfinite(demo_weight) or demo_weight < 0):
        raise ValueError("Use positive training lengths and meaning weight and nonnegative motion weights")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    labels, trials = load_manifest(manifest)
    splits = split_groups(trials, seed)
    demonstrated = subsets(splits, lambda trial: trial.demonstration, "demonstrated",
                           required=imitation_only)
    context_dim = len(trials[0].context)
    origins = sorted({trial.origin for trial in trials})
    metadata = {"version": CHECKPOINT_VERSION, "channels": list(CHANNELS),
                "context_dim": context_dim, "labels": labels, "origins": origins,
                "residual": residual}
    from .gr_eval import trial_evidence

    profiles = sorted({trial.profile for trial in trials})
    spec = json.loads(manifest.read_text())
    tiers = {}
    for row in spec["trials"]:
        tier = trial_evidence(row, bool(spec.get("synthetic")))
        tiers[tier] = tiers.get(tier, 0) + 1
    report = {
        "data": {"manifest": str(manifest.resolve()), "sha256": file_digest(manifest),
                 "synthetic": bool(spec.get("synthetic", False)), "vote_evidence_trials": tiers,
                 "human_observer_evidence": any(t.startswith("human_") for t in tiers),
                 "labels": labels, "trials": len(trials)},
        "origins": origins, "seed": seed, "epochs": epochs,
        "generator_mode": "residual" if residual else "direct",
        "loss_weights": {"demo": demo_weight, "meaning": meaning_weight, "temporal": smooth_weight},
        "calibration": {profile: dict(zip(CHANNELS, capability(profile).tolist()))
                        for profile in profiles if profile in PROFILES},
        "custom_calibrations": [profile for profile in profiles if profile not in PROFILES],
        "splits": {name: {"groups": sorted({trial.group for trial in rows}), "trials": len(rows),
                          "rated_trials": sum(trial.raters > 0 for trial in rows),
                          "observer_votes": sum(trial.raters for trial in rows)}
                   for name, rows in splits.items()},
    }

    def imitation_loss(model, data):
        motion = generator_output(model, data)
        return imitation_error(motion, data) + smooth_weight * smoothness(motion, data)

    generator = Generator(context_dim, residual=residual)
    imitation = None
    if {"train", "validation"} <= demonstrated.keys():
        fit(generator, demonstrated["train"], demonstrated["validation"], imitation_loss, epochs, seed, 0.001)
        imitation = deepcopy(generator).eval()
    models = {}
    if imitation is not None:
        models["g-imitation"] = imitation
    if imitation_only:
        with torch.no_grad():
            test = demonstrated["test"]
            report["demonstration_test_mae"] = float(
                imitation_error(generator_output(generator, test), test, absolute=True))
        write_results(output, report, metadata, models)
        return report

    reviewed = subsets(splits, lambda trial: trial.votes.any(), "observer-rated")
    optimized = subsets(splits, lambda trial: trial.intent >= 0, "intent-labeled")
    if (reviewed["train"]["votes"].sum(dim=0) == 0).any():
        raise ValueError("Training observer votes must cover every interpretation label")
    observer = Observer(context_dim, len(labels))
    fit(observer, reviewed["train"], reviewed["validation"], observer_loss, epochs, seed, 0.003)
    observer.eval().requires_grad_(False)

    # Keep demonstration-only examples in the refinement set as an imitation anchor.
    def refinement_loss(model, data):
        named = data["intent"] >= 0
        motion = generator_output(model, data)
        logits = observation(observer, motion, data)
        estimates = F.cross_entropy(logits, data["intent"], ignore_index=-1, reduction="none")
        meaning = estimates.sum() / named.sum().clamp_min(1)
        return (demo_weight * imitation_error(motion, data) + smooth_weight * smoothness(motion, data)
                + meaning_weight * meaning)

    refinement = subsets(splits, lambda trial: trial.demonstration or trial.intent >= 0, "G refinement")
    fit(generator, refinement["train"], refinement["validation"], refinement_loss,
        refine_epochs, seed, 0.0007)

    with torch.no_grad():
        test = optimized["test"]
        direct = projected(test)
        report.update({
            "refine_epochs": refine_epochs, "meaning_weight": meaning_weight, "smooth_weight": smooth_weight,
            "observer_test": observer_result(observer, reviewed["test"]),
            "trajectory_test": {
                "projected_dyaplex": generator_result(direct, observer, test),
                "g_refined": generator_result(generator_output(generator, test), observer, test),
            },
        })
        if imitation is not None:
            report["trajectory_test"]["g_imitation"] = generator_result(generator_output(imitation, test),
                                                                         observer, test)
        # The substitution question is per embodiment: compare policies under each profile.
        profiles = [trial.profile for trial in splits["test"] if trial.intent >= 0]
        by_profile = {}
        for profile in sorted(set(profiles)):
            ids = torch.tensor([k for k, name in enumerate(profiles) if name == profile])
            subset = {key: values[ids] for key, values in test.items()}
            by_profile[profile] = {
                "projected_dyaplex": generator_result(projected(subset), observer, subset),
                "g_refined": generator_result(generator_output(generator, subset), observer, subset),
            }
        report["trajectory_test_by_profile"] = by_profile
        report["trajectory_test_by_profile_and_intent"] = intent_table(splits["test"], optimized["test"], observer, generator, labels)
        validation = intent_table(splits["validation"], optimized["validation"], observer, generator, labels)
        report["trajectory_validation_by_profile_and_intent"] = validation
        # Model selection uses validation groups only: macro mean of G's P(intent) over cells.
        report["validation_macro_p_intent"] = {
            policy: sum(cell[policy] for cell in validation.values()) / len(validation)
            for policy in ("projected_streaming_upper_bound", "projected_chunk_hold", "g_refined")}
    models.update({"g": generator, "r": observer})
    write_results(output, report, metadata, models)
    return report


def intent_table(trials: list, data: dict, observer: Observer, generator: Generator, labels: list) -> dict:
    """R's P(intended label) per capability profile x intent, one row per chunk and profile.

    Uses the "retarget" candidate rows, whose recorded motion is the human's own next 500 ms
    retargeted with zero latency: an upper bound on the streaming projected-DyaPlex baseline.
    """
    rows = [trial for trial in trials if trial.intent >= 0]
    table = {}
    for profile in sorted({trial.profile for trial in rows}):
        for intent in sorted({trial.intent for trial in rows}):
            ids = torch.tensor([k for k, trial in enumerate(rows) if trial.profile == profile
                                and trial.intent == intent and trial.condition in (None, "retarget")])
            if not len(ids):
                continue
            subset = {key: values[ids] for key, values in data.items()}
            score = lambda motion: generator_result(motion, observer, subset)["observer_mean_intended_probability"]
            table[f"{profile} / {labels[intent]}"] = {
                "chunks": len(ids), "projected_streaming_upper_bound": score(subset["robot"]),
                "projected_chunk_hold": score(projected(subset)), "g_refined": score(generator_output(generator, subset)),
            }
    return table


def write_results(output: Path, report: dict, metadata: dict, models: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        torch.save({**metadata, "model": name, "weights": model.state_dict()}, output / f"{name}.pt")
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def load_models(generator_path: Path, observer_path: Path | None = None) -> tuple:
    data = torch.load(generator_path, map_location="cpu", weights_only=True)
    if (data["version"] != CHECKPOINT_VERSION or tuple(data["channels"]) != CHANNELS
            or data["model"] not in {"g", "g-imitation"}):
        raise ValueError(f"{generator_path}: expected a compatible G checkpoint")
    generator = Generator(data["context_dim"], residual=data["residual"])
    generator.load_state_dict(data["weights"])
    observer = None
    if observer_path is not None:
        judge = torch.load(observer_path, map_location="cpu", weights_only=True)
        keys = ("version", "channels", "context_dim", "labels", "origins", "residual")
        if judge["model"] != "r" or any(judge[key] != data[key] for key in keys):
            raise ValueError("G and R checkpoint metadata does not match")
        observer = Observer(judge["context_dim"], len(judge["labels"]))
        observer.load_state_dict(judge["weights"])
        observer.eval()
    return generator.eval(), observer, data


def score(manifest: Path, generator_path: Path, observer_path: Path, output: Path) -> dict:
    """Evaluate frozen checkpoints on a manifest that played no part in training or selection."""
    labels, trials = load_manifest(manifest)
    generator, observer, metadata = load_models(generator_path, observer_path)
    if list(labels) != list(metadata["labels"]):
        raise ValueError("Manifest labels differ from the checkpoints")
    rows = [trial for trial in trials if trial.intent >= 0]
    with torch.no_grad():
        table = intent_table(rows, batch(rows), observer, generator, labels)
        reviewed = [trial for trial in trials if trial.votes.any()]
        report = {"manifest": str(manifest.resolve()), "sha256": file_digest(manifest),
                  "g": str(generator_path), "r": str(observer_path),
                  "groups": sorted({trial.group for trial in trials}),
                  "observer": observer_result(observer, batch(reviewed)) if reviewed else None,
                  "by_profile_and_intent": table,
                  "macro_p_intent": {policy: sum(cell[policy] for cell in table.values()) / len(table)
                                     for policy in ("projected_streaming_upper_bound", "projected_chunk_hold",
                                                    "g_refined")}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report
