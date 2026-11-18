"""Observer studies: blinded rating packets, vote merging and per-condition reports.

Intended interpretations live in the manifest's ``intent`` field, written by the
experimenter. Observer votes are counted separately and never shown with intent.

Evidence is labeled per trial and never upgraded implicitly:

* ``unrated``: no votes.
* ``synthetic_votes``: votes that did not come through ``gr-votes`` rating sheets
  (fixtures, hand-entered counts) or a manifest marked synthetic.
* ``human_votes_on_simulated_motion``: observer sheets, motion from MuJoCo/kinematic runs.
* ``human_votes_on_robot_motion``: observer sheets, motion measured on the physical robot.

Separately, ``teacher`` says whether G's input was a real DyaPlex export or a scripted stand-in.
"""

import csv
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess

import numpy as np

UNCLEAR = "unclear"
SHEET = "observer_rating_sheets"
SIMULATED_MOTION = {"simulator_replay", "live_rehearsal_from_files", "live_simulated_robot",
                    "live_network_unknown_embodiment", "synthetic_gr_fixture"}
SYNTHETIC_ORIGINS = {"synthetic_gr_fixture"}
SCRIPTED_TEACHERS = {"synthetic_listener_script_not_dyaplex"}


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
        rating_video = data["source"].get("rating_video")
        for row in data["trials"]:
            if row["id"] in seen:
                raise ValueError(f"Duplicate trial id {row['id']}")
            seen.add(row["id"])
            row = dict(row)
            row["file"] = os.path.relpath((fragment.parent / row["file"]).resolve(), out.parent.resolve())
            # Only the text-free rating render may be shown to observers.
            row["rating_video"] = str(Path(rating_video).resolve()) if rating_video else None
            row.pop("video", None)
            if row.get("intent") is None and row["pair"] in intent_by_pair:
                row["intent"] = intent_by_pair[row["pair"]]
            rows.append(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"labels": labels, "synthetic": synthetic, "trials": rows}, indent=2) + "\n")
    return out


def rating_packet(manifest: Path, out: Path, context_s: float = 3.0, tail_s: float = 0.5,
                  seed: int | None = None, key_path: Path | None = None) -> dict:
    """Cut one blinded clip per trial: prior conversation context + the 500 ms chunk.

    Raters get ``out`` (``clips/``, ``rating-sheet.csv``, ``labels.txt``). The key that maps
    codes to trials, conditions and profiles is written *outside* that folder.
    Every trial must have a text-free rating video; otherwise nothing is written.
    """
    data = json.loads(manifest.read_text())
    missing = [row["id"] for row in data["trials"]
               if not row.get("rating_video") or not Path(row["rating_video"]).is_file()
               or row.get("start_s") is None]
    if missing:
        raise ValueError(f"{len(missing)} trial(s) have no rating video or start time; render them with "
                         f"--video and rebuild the manifest first: {', '.join(missing[:5])}"
                         + (" ..." if len(missing) > 5 else ""))
    key_path = key_path or out.parent / f"{out.name}-key.json"
    if key_path.resolve().is_relative_to(out.resolve()):
        raise ValueError("The key must live outside the folder given to raters")
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
        start = max(0.0, row["start_s"] - context_s)
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", row["rating_video"],
                        "-t", f"{row['start_s'] + 0.5 + tail_s - start:.3f}", "-map_metadata", "-1",
                        "-c:v", "libx264", "-c:a", "aac", str(out / "clips" / f"{code}.mp4")], check=True)
        sheet.append({"clip_code": code, "rater_id": "", "label": "", "notes": ""})
    with (out / "rating-sheet.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["clip_code", "rater_id", "label", "notes"])
        writer.writeheader()
        writer.writerows(sheet)
    (out / "labels.txt").write_text("\n".join(data["labels"]) + "\n")
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps({"manifest": str(manifest.resolve()), "clips": key}, indent=2) + "\n")
    return {"clips": len(key), "packet": str(out), "key": str(key_path)}


def merge_votes(manifest: Path, key: Path, responses: list[Path], out: Path) -> dict:
    """Count one vote per (rater, clip) from observer sheets into each trial's ``votes``."""
    data = json.loads(manifest.read_text())
    labels = data["labels"]
    clips = json.loads(key.read_text())["clips"]
    by_id = {row["id"]: row for row in data["trials"]}
    counts: dict[str, dict[str, int]] = {}
    raters, seen, digests = set(), set(), []
    for path in responses:
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
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
        if row.get("votes") and row.get("vote_source") != SHEET:
            raise ValueError(f"{trial}: already has votes that did not come from rating sheets")
        merged = dict(row.get("votes") or {})
        for label, n in votes.items():
            merged[label] = merged.get(label, 0) + n
        row["votes"], row["vote_source"] = merged, SHEET
    data.setdefault("rating", {})
    data["rating"]["raters"] = sorted(set(data["rating"].get("raters", [])) | raters)
    data["rating"]["sheet_sha256"] = sorted(set(data["rating"].get("sheet_sha256", [])) | set(digests))
    out.write_text(json.dumps(data, indent=2) + "\n")
    return {"trials_with_votes": len(counts), "raters": len(raters), "ratings": len(seen)}


def trial_evidence(row: dict, synthetic_manifest: bool) -> str:
    if not row.get("votes"):
        return "unrated"
    if synthetic_manifest or row.get("vote_source") != SHEET or row.get("origin") in SYNTHETIC_ORIGINS:
        return "synthetic_votes"
    motion = row.get("motion_source") or row.get("origin")
    return "human_votes_on_simulated_motion" if motion in SIMULATED_MOTION else "human_votes_on_robot_motion"


def teacher_kind(row: dict) -> str:
    source = row.get("teacher_source")
    if source in SCRIPTED_TEACHERS or row.get("origin") in SYNTHETIC_ORIGINS:
        return "scripted_or_synthetic"
    return "unverified" if source in (None, "unrecorded", "external_export_unverified") else source


def evaluate(manifest: Path, out: Path, sessions: list[Path] | None = None, observer: Path | None = None) -> dict:
    """Per condition and evidence tier: intended-vote share, unclear rate and counts."""
    data = json.loads(manifest.read_text())
    labels = data["labels"]
    synthetic = bool(data.get("synthetic"))
    conditions: dict[str, dict] = {}
    unrated: dict[str, int] = {}
    for row in data["trials"]:
        tier = trial_evidence(row, synthetic)
        name = f"{row.get('condition') or 'unspecified'} / {row.get('profile') or 'custom'}"
        if tier == "unrated" or row.get("intent") is None:
            unrated[name] = unrated.get(name, 0) + 1
            continue
        key = f"{name} [{tier}]"
        entry = conditions.setdefault(key, {"condition": name, "evidence": tier, "teacher": set(), "trials": 0,
                                            "ratings": 0, "intended_votes": 0, "unclear_votes": 0,
                                            "majority_matches_intent": 0, "pairs": {}})
        votes = row["votes"]
        total = sum(votes.values())
        entry["teacher"].add(teacher_kind(row))
        entry["trials"] += 1
        entry["ratings"] += total
        entry["intended_votes"] += votes.get(row["intent"], 0)
        entry["unclear_votes"] += votes.get(UNCLEAR, 0)
        entry["majority_matches_intent"] += max(votes, key=votes.get) == row["intent"]
        entry["pairs"][row.get("pair")] = votes.get(row["intent"], 0) / total
    tiers = sorted({entry["evidence"] for entry in conditions.values()})
    human_tiers = [t for t in tiers if t.startswith("human_")]
    report = {
        "evidence_tiers_present": tiers or ["unrated"],
        "human_observer_evidence": bool(human_tiers),
        "human_evidence_on_physical_robot": "human_votes_on_robot_motion" in tiers,
        "labels": labels,
        "raters": len(data.get("rating", {}).get("raters", [])) if human_tiers else 0,
        "unrated_trials_by_condition": unrated,
        "conditions": {},
    }
    for key, entry in conditions.items():
        report["conditions"][key] = {
            "condition": entry["condition"], "evidence": entry["evidence"], "teacher": sorted(entry["teacher"]),
            "trials": entry["trials"], "ratings": entry["ratings"],
            "intended_vote_share": entry["intended_votes"] / entry["ratings"],
            "unclear_rate": entry["unclear_votes"] / entry["ratings"],
            "majority_matches_intent": entry["majority_matches_intent"] / entry["trials"],
        }
    # Paired comparison: same conversation moment and evidence tier, different condition.
    comparisons = []
    names = sorted(conditions)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if conditions[a]["evidence"] != conditions[b]["evidence"]:
                continue
            shared = set(conditions[a]["pairs"]) & set(conditions[b]["pairs"]) - {None}
            if shared:
                diff = [conditions[b]["pairs"][p] - conditions[a]["pairs"][p] for p in shared]
                comparisons.append({"a": a, "b": b, "evidence": conditions[a]["evidence"],
                                    "shared_contexts": len(shared),
                                    "mean_intended_share_b_minus_a": float(np.mean(diff)),
                                    "b_better": int(sum(d > 0 for d in diff)),
                                    "a_better": int(sum(d < 0 for d in diff))})
    report["paired_comparisons"] = comparisons
    if observer is not None:
        report["r_predictions"] = {"evidence": "model_proxy_not_human", **r_predictions(manifest, observer)}
    if sessions:
        report["sessions"] = {}
        for path in sessions:
            session = json.loads((path / "interpretations.json").read_text())
            probabilities = [c["r_probabilities_on_realized_motion"] for c in session["chunks"]
                             if c.get("r_probabilities_on_realized_motion")]
            mean = ({label: float(np.mean([p[label] for p in probabilities])) for label in probabilities[0]}
                    if probabilities else None)
            report["sessions"][str(path)] = {
                "motion_source": session.get("origin"), "teacher_source": session.get("teacher_source"),
                "policy": session["policy"], "profile": session["profile"], "metrics": session["metrics"],
                "r_mean_probabilities_on_realized_motion": mean, "r_evidence": "model_proxy_not_human"}
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
