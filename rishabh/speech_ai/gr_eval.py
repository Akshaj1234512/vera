"""Observer studies: blinded rating packets, vote merging and per-condition reports.

Intended interpretations live in the manifest's ``intent`` field, written by the
experimenter. Observer votes are counted separately and never shown with intent.
"""

import csv
import json
import os
from pathlib import Path
import secrets
import subprocess

import numpy as np

UNCLEAR = "unclear"


def build_manifest(fragments: list[Path], labels: list[str], out: Path, intents: Path | None = None,
                   synthetic: bool = False) -> Path:
    """Merge ``gr-trial --all`` fragments into one manifest; paths become relative to it."""
    if len(set(labels)) != len(labels) or len(labels) < 2:
        raise ValueError("Provide at least two unique labels")
    intent_by_pair = {}
    if intents is not None:
        with intents.open() as file:
            for row in csv.DictReader(file):
                if row["intent"] not in labels:
                    raise ValueError(f"Unknown intent {row['intent']!r} for {row['pair']}")
                intent_by_pair[row["pair"]] = row["intent"]
    rows, seen = [], set()
    for fragment in fragments:
        data = json.loads(fragment.read_text())
        for row in data["trials"]:
            if row["id"] in seen:
                raise ValueError(f"Duplicate trial id {row['id']}")
            seen.add(row["id"])
            row = dict(row)
            row["file"] = os.path.relpath((fragment.parent / row["file"]).resolve(), out.parent.resolve())
            row["video"] = data["source"].get("video") and str(Path(data["source"]["video"]).resolve())
            if row.get("intent") is None and row["pair"] in intent_by_pair:
                row["intent"] = intent_by_pair[row["pair"]]
            rows.append(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"labels": labels, "synthetic": synthetic, "trials": rows}, indent=2) + "\n")
    return out


def rating_packet(manifest: Path, out: Path, context_s: float = 3.0, tail_s: float = 0.5,
                  seed: int | None = None) -> dict:
    """Cut a clip per trial (prior conversation context + the 500 ms chunk), with opaque codes.

    Raters get ``clips/`` and ``rating-sheet.csv``. ``key.json`` maps codes to trials and
    conditions and stays with the experimenter.
    """
    data = json.loads(manifest.read_text())
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data["trials"]))
    (out / "clips").mkdir(parents=True, exist_ok=True)
    key, sheet = {}, []
    for position in order:
        row = data["trials"][position]
        code = secrets.token_hex(4)
        while code in key:
            code = secrets.token_hex(4)
        key[code] = {"id": row["id"], "condition": row.get("condition"), "profile": row.get("profile"),
                     "pair": row.get("pair")}
        video = row.get("video")
        if video:
            start = max(0.0, row["start_s"] - context_s)
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", video,
                            "-t", f"{row['start_s'] + 0.5 + tail_s - start:.3f}", "-c:v", "libx264",
                            "-c:a", "aac", str(out / "clips" / f"{code}.mp4")], check=True)
        sheet.append({"clip_code": code, "rater_id": "", "label": "", "notes": ""})
    with (out / "rating-sheet.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["clip_code", "rater_id", "label", "notes"])
        writer.writeheader()
        writer.writerows(sheet)
    (out / "labels.txt").write_text("\n".join(data["labels"]) + "\n")
    (out / "key.json").write_text(json.dumps({"manifest": str(manifest.resolve()), "clips": key}, indent=2) + "\n")
    return {"clips": len(key), "packet": str(out)}


def merge_votes(manifest: Path, key: Path, responses: list[Path], out: Path) -> dict:
    """Count one vote per (rater, clip) into each trial's ``votes``."""
    data = json.loads(manifest.read_text())
    labels = data["labels"]
    clips = json.loads(key.read_text())["clips"]
    by_id = {row["id"]: row for row in data["trials"]}
    counts: dict[str, dict[str, int]] = {}
    raters, seen = set(), set()
    for path in responses:
        with path.open() as file:
            for line in csv.DictReader(file):
                code, rater, label = line["clip_code"].strip(), line["rater_id"].strip(), line["label"].strip()
                if not rater or not label:
                    continue
                if code not in clips or label not in labels:
                    raise ValueError(f"{path}: unknown clip {code!r} or label {label!r}")
                if (rater, code) in seen:
                    raise ValueError(f"{path}: rater {rater} rated clip {code} twice")
                seen.add((rater, code))
                raters.add(rater)
                trial = clips[code]["id"]
                counts.setdefault(trial, {}).setdefault(label, 0)
                counts[trial][label] += 1
    for trial, votes in counts.items():
        row = by_id[trial]
        merged = dict(row.get("votes") or {})
        for label, n in votes.items():
            merged[label] = merged.get(label, 0) + n
        row["votes"] = merged
    data.setdefault("rating", {})
    data["rating"]["raters"] = sorted(set(data["rating"].get("raters", [])) | raters)
    out.write_text(json.dumps(data, indent=2) + "\n")
    return {"trials_with_votes": len(counts), "raters": len(raters), "ratings": len(seen)}


def evaluate(manifest: Path, out: Path, sessions: list[Path] | None = None, observer: Path | None = None) -> dict:
    """Per condition: intended-vote share, unclear rate, and counts; plus R and motion metrics."""
    data = json.loads(manifest.read_text())
    labels = data["labels"]
    evidence = "synthetic" if data.get("synthetic") else "human_observer_votes"
    conditions: dict[str, dict] = {}
    for row in data["trials"]:
        votes = row.get("votes") or {}
        if not votes or row.get("intent") is None:
            continue
        name = f"{row.get('condition') or 'unspecified'} / {row.get('profile') or 'custom'}"
        entry = conditions.setdefault(name, {"trials": 0, "ratings": 0, "intended_votes": 0, "unclear_votes": 0,
                                             "majority_matches_intent": 0, "pairs": {}})
        total = sum(votes.values())
        entry["trials"] += 1
        entry["ratings"] += total
        entry["intended_votes"] += votes.get(row["intent"], 0)
        entry["unclear_votes"] += votes.get(UNCLEAR, 0)
        majority = max(votes, key=votes.get)
        entry["majority_matches_intent"] += majority == row["intent"]
        entry["pairs"][row.get("pair")] = votes.get(row["intent"], 0) / total
    report = {"evidence": evidence, "labels": labels,
              "raters": len(data.get("rating", {}).get("raters", [])) or None, "conditions": {}}
    for name, entry in conditions.items():
        report["conditions"][name] = {
            "trials": entry["trials"], "ratings": entry["ratings"],
            "intended_vote_share": entry["intended_votes"] / entry["ratings"],
            "unclear_rate": entry["unclear_votes"] / entry["ratings"],
            "majority_matches_intent": entry["majority_matches_intent"] / entry["trials"],
        }
    # Paired comparison: same conversation moment, different condition.
    names = sorted(conditions)
    comparisons = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(conditions[a]["pairs"]) & set(conditions[b]["pairs"]) - {None}
            if shared:
                diff = [conditions[b]["pairs"][p] - conditions[a]["pairs"][p] for p in shared]
                comparisons.append({"a": a, "b": b, "shared_contexts": len(shared),
                                    "mean_intended_share_b_minus_a": float(np.mean(diff)),
                                    "b_better": int(sum(d > 0 for d in diff)),
                                    "a_better": int(sum(d < 0 for d in diff))})
    report["paired_comparisons"] = comparisons
    if observer is not None:
        report["r_predictions"] = r_predictions(manifest, observer)
    if sessions:
        report["sessions"] = {}
        for path in sessions:
            session = json.loads((path / "interpretations.json").read_text())
            probabilities = [c["r_probabilities_on_realized_motion"] for c in session["chunks"]
                             if c.get("r_probabilities_on_realized_motion")]
            mean = ({label: float(np.mean([p[label] for p in probabilities])) for label in probabilities[0]}
                    if probabilities else None)
            report["sessions"][str(path)] = {
                "evidence": "simulator_replay" if session.get("synthetic_or_replay") else "recorded",
                "policy": session["policy"], "profile": session["profile"], "metrics": session["metrics"],
                "r_mean_probabilities_on_realized_motion": mean}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def r_predictions(manifest: Path, observer: Path) -> dict:
    """R's mean P(intended label) per condition on the manifest's recorded motion."""
    import torch

    from .gr_data import load_manifest
    from .gr_model import Observer

    labels, trials = load_manifest(manifest)
    judge = torch.load(observer, map_location="cpu", weights_only=True)
    if judge["labels"] != labels:
        raise ValueError("R was trained with a different label vocabulary")
    model = Observer(judge["context_dim"], len(labels))
    model.load_state_dict(judge["weights"])
    model.eval()
    result: dict[str, list] = {}
    with torch.no_grad():
        for trial in trials:
            if trial.intent < 0:
                continue
            logits = model(torch.from_numpy(trial.robot[None]), torch.from_numpy(trial.context[None]),
                           torch.from_numpy(trial.limits[None]))
            p = float(logits.softmax(1)[0, trial.intent])
            result.setdefault(f"{trial.condition or 'unspecified'} / {trial.profile}", []).append(p)
    return {name: {"mean_p_intended": float(np.mean(v)), "trials": len(v)} for name, v in result.items()}
