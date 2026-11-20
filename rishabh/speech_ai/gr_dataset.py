"""Build G/R training trials from real human motion streams.

Inputs are gr-prepare streams of measured human motion (Seamless Interaction, see
seamless.py) with real conversation audio. For every 500 ms chunk:

* M_h = the human's previous 500 ms, x_t = audio features at the boundary.
* z* (intent) = what the human's *next* 500 ms conveyed (``human.human_intent``).
* q_t = the robot state that following the retarget would have produced.
* Robot candidates under each capability profile: the retargeted human motion,
  projected onto the profile (a demonstration only under ``full``), plus one variant
  (look-down / eyebrow-flash substitutions, damped or jerky motion) so R sees motion
  that differs from the teacher.
* Votes come from ``human.observer_probabilities``, a SYNTHETIC rule-based observer.
  The manifest is marked ``synthetic: true`` and every row ``vote_source:
  synthetic_rule_observer``; nothing built here is human-observer evidence.
"""

import json
from pathlib import Path

import numpy as np

from .audio import FRAME_S
from .embodiment import CHANNELS, PROFILES, SCALE
from .gr_data import load_stream
from .gr_model import STEPS, capability
from .human import HEAD_SCALE_DEG, HUMAN_CHANNELS, LABELS, human_intent, observer_probabilities, retarget

VARIANTS = ("eye_dip", "brow_flash", "damped", "jerky")
EYE_PITCH = [CHANNELS.index(c) for c in ("left_eye_pitch", "right_eye_pitch")]
LIDS = [CHANNELS.index(c) for c in ("left_eyelid_close", "right_eyelid_close")]
BROWS = [CHANNELS.index(c) for c in ("left_brow_raise", "right_brow_raise")]
HEAD_PITCH = HUMAN_CHANNELS.index("head_pitch")


def constrain(target: np.ndarray, limits: np.ndarray, start: np.ndarray) -> np.ndarray:
    """numpy twin of gr_model.constrain_motion in physical units."""
    enabled, low, high, speed = limits.T
    previous = np.clip(start, low, high)
    out = []
    for step in target:
        previous = np.clip(previous + np.clip(step - previous, -speed * FRAME_S, speed * FRAME_S), low, high) * enabled
        out.append(previous)
    return np.asarray(out, dtype=np.float32)


def variant(kind: str, target: np.ndarray, future: np.ndarray, start: np.ndarray, rng) -> np.ndarray:
    result = target.copy()
    pitch = future[:, HEAD_PITCH] * HEAD_SCALE_DEG
    dip = np.clip(np.maximum.accumulate(pitch) - pitch, 0, None)  # how far the head has dropped
    shape = dip / max(dip.max(), 1.0)
    if kind == "eye_dip":  # gaze drops with the nod and lids lower: a look-down acknowledgment
        result[:, EYE_PITCH] -= np.radians(1.6 * dip)[:, None]
        result[:, LIDS] -= 0.006 * shape[:, None]
    elif kind == "brow_flash":
        result[:, BROWS] += 0.008 * np.maximum(shape, 0.6 * np.sin(np.linspace(0, np.pi, len(shape))))[:, None]
    elif kind == "damped":
        result = start + 0.35 * (target - start)
    elif kind == "jerky":
        result = target + rng.normal(0, 0.18, target.shape) * SCALE
    return result


def participant_groups(streams: list[dict]) -> dict[str, str]:
    """Union participants that ever share an interaction, so no person spans splits."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for row in streams:
        a, b = row["participants"]
        parent[find(a)] = find(b)
    return {row["stream"]: "people-" + find(row["participants"][0]) for row in streams}


def build(indices: list[Path], out: Path, seed: int = 3, keep_no_signal: float = 0.3,
          raters: tuple[int, int] = (5, 10)) -> dict:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    rows, counts = [], {label: 0 for label in LABELS}
    intents = {label: 0 for label in LABELS}
    limits = {profile: capability(profile) for profile in PROFILES}
    for index_path in indices:
        index = json.loads(index_path.read_text())
        groups = participant_groups(index["streams"])
        for meta in index["streams"]:
            times, human, context = load_stream(index_path.parent / meta["stream"])
            arrays = {"human": [], "human_valid": [], "context": [], "state_q": [], "robot_q": []}
            name = f"trials-{meta['agent']}.npz"
            for k in range(STEPS, len(times) - STEPS, STEPS):
                history, future = human[k - STEPS:k], human[k:k + STEPS]
                if future[:, -1].mean() < 0.9 or history[:, -1].mean() < 0.5 or not human[k - 2:k, -1].all():
                    continue
                intent = human_intent(future[:, :-1])
                if intent == "no_signal" and rng.random() > keep_no_signal:
                    continue
                intents[intent] += 1
                full = retarget(future[:, :-1])
                previous, before = retarget(human[k - 1, :-1]), retarget(human[k - 2, :-1])
                pair = f"{meta['agent']}-{k * 50:06d}ms"
                for profile, limit in limits.items():
                    low, high, enabled = limit[:, 1], limit[:, 2], limit[:, 0]
                    position = np.clip(previous, low, high) * enabled
                    velocity = (np.clip(previous, low, high) - np.clip(before, low, high)) / FRAME_S * enabled
                    state = np.stack([position, velocity]).astype(np.float32)
                    kinds = ["retarget", VARIANTS[rng.integers(len(VARIANTS))]]
                    for kind in kinds:
                        target = full if kind == "retarget" else variant(kind, full, future[:, :-1], position, rng)
                        motion = constrain(target, limit, position)
                        p = observer_probabilities(motion, position)
                        votes = rng.multinomial(int(rng.integers(*raters)), p)
                        for label, n in zip(LABELS, votes):
                            counts[label] += int(n)
                        rows.append({"id": f"{pair}-{profile}-{kind}", "group": groups[meta["stream"]],
                                     "origin": "seamless_human_motion", "file": name,
                                     "index": len(arrays["human"]), "pair": pair, "profile": profile,
                                     "condition": kind, "intent": intent,
                                     "votes": {label: int(n) for label, n in zip(LABELS, votes) if n},
                                     "vote_source": "synthetic_rule_observer",
                                     "demonstration": kind == "retarget" and profile == "full"})
                        arrays["human"].append(history[:, :-1])
                        arrays["human_valid"].append(history[:, -1])
                        arrays["context"].append(context[k - 1])
                        arrays["state_q"].append(state)
                        arrays["robot_q"].append(motion)
            if arrays["human"]:
                np.savez_compressed(out / name, **{key: np.asarray(v, dtype=np.float32) for key, v in arrays.items()})
    manifest = out / "manifest.json"
    manifest.write_text(json.dumps({
        "labels": list(LABELS), "synthetic": True,
        "note": "Real human motion and audio (Seamless Interaction); observer votes are SYNTHETIC (rule).",
        "sources": [str(p) for p in indices], "trials": rows}, indent=1) + "\n")
    report = {"trials": len(rows), "chunks_by_intent": intents, "synthetic_votes_by_label": counts,
              "groups": len({r["group"] for r in rows})}
    (out / "dataset.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
