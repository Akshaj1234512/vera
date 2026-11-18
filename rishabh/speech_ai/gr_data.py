"""Training trials and causal alignment for the G/R models."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .audio import FRAME_S, aligned_embeddings, extract, load_tracks
from .gr_model import CHANNELS, SCALE_RAD, STEPS, capability, normalized_limits


@dataclass
class Trial:
    id: str
    group: str
    origin: str
    profile: str
    human: np.ndarray
    context: np.ndarray
    state: np.ndarray
    limits: np.ndarray
    robot: np.ndarray
    demonstration: bool
    votes: np.ndarray
    intent: int


def array(data, key: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    values = np.asarray(data[key], dtype=np.float32)
    if (shape is not None and values.shape != shape) or not np.isfinite(values).all():
        raise ValueError(f"{key} must be finite and have shape {shape}")
    return values


def clock(data, key: str = "time_s") -> np.ndarray:
    times = np.asarray(data[key], dtype=np.float64)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or times[0] < 0 or (np.diff(times) <= 0).any()):
        raise ValueError(f"{key} needs nonnegative, finite and strictly increasing times")
    return times


def load_manifest(path: Path) -> tuple[list[str], list[Trial]]:
    data = json.loads(path.read_text())
    labels = data["labels"]
    if (not isinstance(labels, list) or len(labels) < 2 or not all(isinstance(x, str) and x for x in labels)
            or len(labels) != len(set(labels))):
        raise ValueError("Provide at least two unique interpretation labels")
    channels = len(CHANNELS)
    trials = []
    seen = set()
    for row in data["trials"]:
        trial_id, group, origin = (row[key] for key in ("id", "group", "origin"))
        if not all(isinstance(x, str) and x for x in (trial_id, group, origin)) or trial_id in seen:
            raise ValueError(f"Each trial needs a unique ID, a split group and an origin: {trial_id}")
        seen.add(trial_id)
        profile = row.get("profile", "full")
        limits = capability(profile, row.get("capabilities"))
        with np.load(path.parent / row["file"], allow_pickle=False) as payload:
            human = array(payload, "human", (STEPS, channels))
            valid = array(payload, "human_valid", (STEPS,)) if "human_valid" in payload else np.ones(STEPS)
            if (np.abs(human) > 1).any() or not np.isin(valid, [0, 1]).all():
                raise ValueError(f"{trial_id}: human motion needs normalized channels and binary validity")
            human = np.column_stack([human * valid[:, None], valid]).astype(np.float32)
            context = array(payload, "context")
            if context.ndim != 1 or not len(context):
                raise ValueError(f"{trial_id}: context must be a nonempty vector")
            state = array(payload, "state_rad", (2, channels)) / SCALE_RAD
            robot = array(payload, "robot_rad", (STEPS, channels)) if "robot_rad" in payload else None

        votes = row.get("votes", {})
        demonstration = row.get("demonstration", False)
        if not isinstance(demonstration, bool) or set(votes) - set(labels):
            raise ValueError(f"{trial_id}: invalid demonstration flag or unknown observer labels")
        if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in votes.values()):
            raise ValueError(f"{trial_id}: each observer vote count must be a positive integer")
        if (demonstration or votes) and robot is None:
            raise ValueError(f"{trial_id}: demonstrations and observer votes require robot_rad")
        distribution = np.array([votes.get(label, 0) for label in labels], dtype=np.float32)
        distribution /= max(float(distribution.sum()), 1)
        intent = labels.index(row["intent"]) if row.get("intent") is not None else -1
        if robot is None:
            robot = np.zeros((STEPS, channels), dtype=np.float32)
        trials.append(Trial(trial_id, group, origin, profile, human, context, state,
                            normalized_limits(limits), robot / SCALE_RAD, demonstration, distribution, intent))
    if not trials or len({trial.context.shape for trial in trials}) != 1:
        raise ValueError("Use a nonempty manifest with consistent context dimensions")
    return labels, trials


def past_indices(times: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.searchsorted(times, np.round(query, 8) + 1e-9, side="right") - 1


def load_stream(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        times = clock(data)
        if not np.allclose(times, np.arange(1, len(times) + 1) * FRAME_S, atol=1e-7, rtol=0):
            raise ValueError("The stream time_s must start at 0.05 and advance by 0.05 seconds")
        human = array(data, "human", (len(times), len(CHANNELS)))
        valid = array(data, "human_valid", (len(times),))
        context = array(data, "context")
    if not np.isin(valid, [0, 1]).all() or (np.abs(human) > 1).any():
        raise ValueError("Invalid human channel values or validity mask")
    if context.ndim != 2 or context.shape[0] != len(times) or not context.shape[1]:
        raise ValueError("The context stream must have one nonempty feature vector per frame")
    return times, np.column_stack([human * valid[:, None], valid]).astype(np.float32), context


def prepare_stream(partner: Path, agent: Path | None, stereo: bool, teacher: Path,
                   embeddings: Path, out: Path, teacher_map: Path | None = None) -> None:
    """Align exported teacher channels and speech features on the 20 Hz grid."""
    features = extract(*load_tracks(partner, agent, stereo))
    with np.load(teacher, allow_pickle=False) as data:
        times = clock(data)
        if teacher_map is None:
            if tuple(data["channels"].tolist()) != CHANNELS:
                raise ValueError(f"Teacher channels must be named and ordered as {CHANNELS}")
            human = array(data, "human", (len(times), len(CHANNELS)))
            if (np.abs(human) > 1).any():
                raise ValueError("The teacher motion must be normalized to [-1, 1]")
        else:
            decoded = array(data, "features")
            if decoded.ndim != 2 or len(decoded) != len(times) or not decoded.shape[1]:
                raise ValueError("The decoded export must contain one feature vector per frame")
            with np.load(teacher_map, allow_pickle=False) as calibration:
                weights = array(calibration, "weights", (decoded.shape[1], len(CHANNELS)))
                bias = array(calibration, "bias", (len(CHANNELS),))
            human = np.clip(decoded @ weights + bias, -1, 1)
        valid = array(data, "human_valid", (len(times),)) if "human_valid" in data else np.ones(len(times))
        if not np.isin(valid, [0, 1]).all():
            raise ValueError("Teacher validity must be binary")
    indices = past_indices(times, features.time_s)
    observed = (indices >= 0) & (features.time_s - times[np.maximum(indices, 0)] <= 0.15 + 1e-8)
    observed &= valid[np.maximum(indices, 0)].astype(bool)
    aligned = np.zeros((len(features.time_s), len(CHANNELS)), dtype=np.float32)
    aligned[observed] = human[indices[observed]]
    context = aligned_embeddings(embeddings, features.time_s)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as file:
        np.savez_compressed(file, time_s=features.time_s, human=aligned, human_valid=observed,
                            context=context)


def window_at(human: np.ndarray, context: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
    history = np.zeros((STEPS, human.shape[1]), dtype=np.float32)
    start = max(0, index - STEPS)
    if index > start:
        history[-(index - start):] = human[start:index]
    latest = context[index - 1] if index else np.zeros(context.shape[1], dtype=np.float32)
    return history, latest


def pack_trial(stream: Path, realized: Path, start_s: float, out: Path) -> None:
    """Pair a recorded robot chunk with exactly the inputs G saw at its start."""
    times, human, context = load_stream(stream)
    index = round(start_s / FRAME_S)
    if index < 0 or index + STEPS > len(times) or not np.isclose(index * FRAME_S, start_s, atol=1e-8, rtol=0):
        raise ValueError("Choose a complete 500 ms chunk on the 20 Hz grid")
    with np.load(realized, allow_pickle=False) as data:
        recorded = clock(data)
        if (recorded.shape != times.shape or not np.allclose(recorded, times, atol=1e-8, rtol=0)
                or tuple(data["channels"].tolist()) != CHANNELS):
            raise ValueError("Realized robot samples must match the stream clock and channels")
        starts = clock(data, "chunk_start_s")
        selected = np.flatnonzero(np.isclose(starts, start_s, atol=1e-8, rtol=0))
        if len(selected) != 1:
            raise ValueError("Start time does not match a recorded chunk")
        states = array(data, "initial_state_rad", (len(starts), 2, len(CHANNELS)))
        motion = array(data, "robot_rad", (len(times), len(CHANNELS)))
    history, latest = window_at(human, context, index)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as file:
        np.savez_compressed(file, human=history[:, :-1], human_valid=history[:, -1],
                            context=latest, state_rad=states[selected[0]],
                            robot_rad=motion[index:index + STEPS])
