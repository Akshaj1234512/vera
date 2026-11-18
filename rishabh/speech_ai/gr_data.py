"""Training trials, grouped splits and causal alignment for the G/R models.

Every input reaches the 20 Hz chunk grid through the same rule: a sample may be
used at boundary t only if it had *arrived* by t (``arrival_s``, defaulting to
its own timestamp) and its timestamp is no older than a source-specific limit.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .audio import FEATURE_NAMES, FRAME_S, extract, load_tracks, normalized
from .gr_model import CHANNELS, SCALE_RAD, STEPS, capability, normalized_limits

TEACHER_MAX_AGE_S = 0.15
STATES_MAX_AGE_S = 0.25


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
    raters: int = 0
    pair: str | None = None
    condition: str | None = None


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


def file_digest(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


# --------------------------------------------------------------------------- #
# Manifest and grouped splits
# --------------------------------------------------------------------------- #
def split_groups(trials: list, seed: int) -> dict[str, list]:
    """Hold out complete groups so related sessions never leak across splits."""
    groups = sorted({trial.group for trial in trials})
    if len(groups) < 3:
        raise ValueError("At least three disjoint session/participant groups are required")
    shuffled = np.random.default_rng(seed).permutation(groups).tolist()
    count = max(1, round(0.2 * len(groups)))
    validation = set(shuffled[:count])
    test = set(shuffled[count:2 * count])
    split = {"train": [], "validation": [], "test": []}
    for trial in trials:
        destination = "validation" if trial.group in validation else "test" if trial.group in test else "train"
        split[destination].append(trial)
    return split


def load_manifest(path: Path) -> tuple[list[str], list[Trial]]:
    data = json.loads(path.read_text())
    labels = data["labels"]
    if (not isinstance(labels, list) or len(labels) < 2 or not all(isinstance(x, str) and x for x in labels)
            or len(labels) != len(set(labels))):
        raise ValueError("Provide at least two unique interpretation labels")
    channels = len(CHANNELS)
    trials = []
    seen = set()
    pair_groups: dict[str, str] = {}
    for row in data["trials"]:
        trial_id, group, origin = (row[key] for key in ("id", "group", "origin"))
        if not all(isinstance(x, str) and x for x in (trial_id, group, origin)) or trial_id in seen:
            raise ValueError(f"Each trial needs a unique ID, a split group and an origin: {trial_id}")
        seen.add(trial_id)
        pair = row.get("pair")
        if pair is not None:
            # Paired variants share conversation context, so they must share a split.
            if pair_groups.setdefault(pair, group) != group:
                raise ValueError(f"{trial_id}: paired trials of {pair!r} must share one group")
        profile = row.get("profile", "full" if row.get("capabilities") is None else "custom")
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
        raters = int(distribution.sum())
        distribution /= max(float(distribution.sum()), 1)
        intent = labels.index(row["intent"]) if row.get("intent") is not None else -1
        if robot is None:
            robot = np.zeros((STEPS, channels), dtype=np.float32)
        trials.append(Trial(trial_id, group, origin, profile, human, context, state,
                            normalized_limits(limits), robot / SCALE_RAD, demonstration, distribution,
                            intent, raters, pair, row.get("condition")))
    if not trials or len({trial.context.shape for trial in trials}) != 1:
        raise ValueError("Use a nonempty manifest with consistent context dimensions")
    return labels, trials


# --------------------------------------------------------------------------- #
# Clock alignment
# --------------------------------------------------------------------------- #
@dataclass
class TimedSource:
    """Samples on the common recording clock, with the time each became usable."""
    time_s: np.ndarray
    arrival_s: np.ndarray
    values: np.ndarray
    valid: np.ndarray


def timed_source(data, values: np.ndarray, offset_s: float, name: str) -> TimedSource:
    times = clock(data) + offset_s
    arrival = (np.asarray(data["arrival_s"], dtype=np.float64) + offset_s if "arrival_s" in data
               else times.copy())
    if arrival.shape != times.shape or not np.isfinite(arrival).all() or np.any(arrival < times - 1e-6):
        raise ValueError(f"{name}: arrival_s must match time_s and never precede its sample")
    if np.any(np.diff(arrival) < 0):
        raise ValueError(f"{name}: arrival_s must be nondecreasing")
    valid = array(data, "human_valid", (len(times),)) if "human_valid" in data else np.ones(len(times))
    if not np.isin(valid, [0, 1]).all():
        raise ValueError(f"{name}: validity must be binary")
    if len(values) != len(times):
        raise ValueError(f"{name}: one value row is required per timestamp")
    return TimedSource(times, arrival, values, valid.astype(bool))


def align(source: TimedSource, query: np.ndarray, max_age_s: float,
          deadline: np.ndarray | float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Latest sample stamped at or before each query time that had arrived by its deadline.

    ``deadline`` defaults to the query time. Returns values, an availability mask and age.
    """
    query = np.asarray(query, dtype=float)
    deadline = query if deadline is None else np.broadcast_to(np.asarray(deadline, dtype=float), query.shape)
    arrived = np.searchsorted(source.arrival_s, np.round(deadline, 8) + 1e-9, side="right")
    stamped = np.searchsorted(source.time_s, np.round(query, 8) + 1e-9, side="right")
    index = np.minimum(arrived, stamped) - 1
    safe = np.maximum(index, 0)
    age = query - source.time_s[safe]
    observed = (index >= 0) & (age <= max_age_s + 1e-8) & source.valid[safe]
    aligned = np.zeros((len(query), source.values.shape[1]), dtype=np.float32)
    aligned[observed] = source.values[index[observed]]
    return aligned, observed, np.where(index >= 0, age, np.inf)


def teacher_source(path: Path) -> str:
    """Provenance tag carried by a teacher file; scripted stand-ins mark themselves."""
    with np.load(path, allow_pickle=False) as data:
        return str(data["source"]) if "source" in data else "external_export_unverified"


def load_teacher(path: Path, teacher_map: Path | None, offset_s: float = 0.0) -> TimedSource:
    with np.load(path, allow_pickle=False) as data:
        if teacher_map is None:
            if "channels" not in data or tuple(data["channels"].tolist()) != CHANNELS:
                raise ValueError(f"Teacher channels must be named and ordered as {CHANNELS}")
            human = array(data, "human")
            if human.ndim != 2 or human.shape[1] != len(CHANNELS) or (np.abs(human) > 1).any():
                raise ValueError("The teacher motion must be normalized to [-1, 1] with shape [frames, 8]")
        else:
            human = project_features(array(data, "features"), teacher_map)
        return timed_source(data, human, offset_s, str(path))


def project_features(decoded: np.ndarray, teacher_map: Path) -> np.ndarray:
    if decoded.ndim != 2 or not decoded.shape[1]:
        raise ValueError("The decoded export must contain one feature vector per frame")
    with np.load(teacher_map, allow_pickle=False) as calibration:
        weights = array(calibration, "weights", (decoded.shape[1], len(CHANNELS)))
        bias = array(calibration, "bias", (len(CHANNELS),))
        mean = array(calibration, "feature_mean", (decoded.shape[1],)) if "feature_mean" in calibration else 0
        scale = array(calibration, "feature_scale", (decoded.shape[1],)) if "feature_scale" in calibration else 1
    return np.clip(((decoded - mean) / scale) @ weights + bias, -1, 1).astype(np.float32)


def load_states(path: Path, offset_s: float = 0.0) -> TimedSource:
    with np.load(path, allow_pickle=False) as data:
        features = array(data, "features")
        if features.ndim != 2 or not features.shape[1]:
            raise ValueError(f"{path}: expected finite [frames, dim] PersonaPlex states")
        return timed_source(data, features, offset_s, str(path))


def aligned_states(path: Path, query: np.ndarray, offset_s: float = 0.0,
                   max_age_s: float = STATES_MAX_AGE_S) -> np.ndarray:
    """PersonaPlex states plus one availability column; zeros when stale or missing."""
    values, observed, _ = align(load_states(path, offset_s), query, max_age_s)
    return np.column_stack([values, observed]).astype(np.float32)


def prepare_stream(partner: Path, agent: Path | None, stereo: bool, teacher: Path,
                   embeddings: Path | None, out: Path, teacher_map: Path | None = None,
                   teacher_offset_s: float = 0.0, states_offset_s: float = 0.0) -> dict:
    """Align teacher motion, audio-encoder features and PersonaPlex states on the 20 Hz grid.

    x_t = [8 normalized causal audio features] (+ [PersonaPlex state, availability]).
    """
    audio = extract(*load_tracks(partner, agent, stereo))
    times = audio.time_s
    source = load_teacher(teacher, teacher_map, teacher_offset_s)
    # A history frame is consumed at the next chunk boundary, so late teacher samples
    # still count if they arrived by then (the same rule LiveInputs applies online).
    consumed = np.ceil(np.round(times / FRAME_S, 6) / STEPS) * STEPS * FRAME_S
    human, observed, age = align(source, times, TEACHER_MAX_AGE_S, deadline=consumed)
    context = [normalized(audio.values)]
    names = list(FEATURE_NAMES)
    report = {
        "frames": len(times), "duration_s": float(times[-1]),
        "teacher": {"file": str(teacher), "sha256": file_digest(teacher), "source": teacher_source(teacher),
                    "mapped_with": str(teacher_map) if teacher_map else None,
                    "clock_offset_s": teacher_offset_s, "max_age_s": TEACHER_MAX_AGE_S,
                    "coverage": float(observed.mean()),
                    "median_age_s": float(np.median(age[observed])) if observed.any() else None},
    }
    if embeddings is not None:
        states = load_states(embeddings, states_offset_s)
        values, available, state_age = align(states, times, STATES_MAX_AGE_S)
        context.append(np.column_stack([values, available]).astype(np.float32))
        names += [f"personaplex_{i}" for i in range(values.shape[1])] + ["personaplex_available"]
        intervals = np.diff(states.time_s)
        report["personaplex"] = {
            "file": str(embeddings), "sha256": file_digest(embeddings), "feature_width": int(values.shape[1]),
            "clock_offset_s": states_offset_s, "max_age_s": STATES_MAX_AGE_S,
            "median_interval_s": float(np.median(intervals)) if len(intervals) else None,
            "median_arrival_delay_s": float(np.median(states.arrival_s - states.time_s)),
            "coverage": float(available.mean()),
        }
    context = np.concatenate(context, axis=1)
    report["context_dim"] = int(context.shape[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as file:
        np.savez_compressed(file, time_s=times, human=human, human_valid=observed,
                            context=context, context_names=np.array(names),
                            teacher_source=np.array(teacher_source(teacher)))
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


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


def window_at(human: np.ndarray, context: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Inputs known at boundary ``index``: frames [index-10, index) and the last context.

    Frame k is stamped at (k + 1) * 50 ms, so frame index-1 ends exactly at the boundary.
    Before 500 ms of history exists, missing frames are zero with validity 0 (cold start).
    """
    history = np.zeros((STEPS, human.shape[1]), dtype=np.float32)
    start = max(0, index - STEPS)
    if index > start:
        history[-(index - start):] = human[start:index]
    latest = context[index - 1] if index else np.zeros(context.shape[1], dtype=np.float32)
    return history, latest


# --------------------------------------------------------------------------- #
# Trial packaging
# --------------------------------------------------------------------------- #
def _recording(stream: Path, realized: Path):
    times, human, context = load_stream(stream)
    with np.load(realized, allow_pickle=False) as data:
        recorded = clock(data)
        if (recorded.shape != times.shape or not np.allclose(recorded, times, atol=1e-8, rtol=0)
                or tuple(data["channels"].tolist()) != CHANNELS):
            raise ValueError("Realized robot samples must match the stream clock and channels")
        starts = clock(data, "chunk_start_s")
        states = array(data, "initial_state_rad", (len(starts), 2, len(CHANNELS)))
        motion = array(data, "robot_rad", (len(times), len(CHANNELS)))
        meta = json.loads(str(data["session"])) if "session" in data else {}
    return times, human, context, starts, states, motion, meta


def _write_trial(out: Path, human, context, index, state, motion) -> None:
    history, latest = window_at(human, context, index)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as file:
        np.savez_compressed(file, human=history[:, :-1], human_valid=history[:, -1],
                            context=latest, state_rad=state, robot_rad=motion[index:index + STEPS])


def pack_trial(stream: Path, realized: Path, start_s: float, out: Path) -> None:
    """Pair a recorded robot chunk with exactly the inputs G saw at its start."""
    times, human, context, starts, states, motion, _ = _recording(stream, realized)
    index = round(start_s / FRAME_S)
    if index < 0 or index + STEPS > len(times) or not np.isclose(index * FRAME_S, start_s, atol=1e-8, rtol=0):
        raise ValueError("Choose a complete 500 ms chunk on the 20 Hz grid")
    selected = np.flatnonzero(np.isclose(starts, start_s, atol=1e-8, rtol=0))
    if len(selected) != 1:
        raise ValueError("Start time does not match a recorded chunk")
    _write_trial(out, human, context, index, states[selected[0]], motion)


def pack_recording(stream: Path, realized: Path, out_dir: Path, group: str | None = None,
                   condition: str | None = None, skip_cold_start: bool = True,
                   rating_video: Path | None = None) -> Path:
    """Package every complete chunk and write an unrated manifest fragment for annotation."""
    times, human, context, starts, states, motion, meta = _recording(stream, realized)
    if rating_video is not None:
        meta["rating_video"] = str(Path(rating_video).resolve())
    session = meta.get("session_id") or realized.parent.name
    group = group or meta.get("participant_group") or session
    profile = meta.get("profile", "full")
    rows = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for chunk, start in enumerate(starts):
        index = round(start / FRAME_S)
        if index + STEPS > len(times) or (skip_cold_start and index < STEPS):
            continue
        name = f"{session}-{index * 50:06d}ms.npz"
        _write_trial(out_dir / name, human, context, index, states[chunk], motion)
        row = {"id": f"{session}-{index * 50:06d}ms", "group": group, "origin": meta.get("origin", "recorded"),
               "file": name, "pair": f"{meta.get('context_id', session)}-{index * 50:06d}ms",
               "condition": condition or meta.get("policy"), "start_s": round(float(start), 3),
               "motion_source": meta.get("origin"), "teacher_source": meta.get("teacher_source"),
               "intent": None, "votes": {}}
        if meta.get("capabilities") is not None and profile not in ("full", "fixed-neck-pitch", "fixed-eyes"):
            row["capabilities"] = meta["capabilities"]
        else:
            row["profile"] = profile
        rows.append(row)
    path = out_dir / "trials.json"
    path.write_text(json.dumps({"source": {"stream": str(stream), "realized": str(realized), **meta},
                                "trials": rows}, indent=2) + "\n")
    return path


# --------------------------------------------------------------------------- #
# DyaPlex decoded-feature -> eight-channel calibration
# --------------------------------------------------------------------------- #
def fit_teacher_map(pairs_path: Path, out: Path, seed: int = 7,
                    ridges: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1, 10)) -> dict:
    """Fit a ridge projection from decoded DyaPlex features to the eight robot-facing channels.

    ``pairs_path`` lists clips: {"id", "group", "features": NPZ(time_s, features),
    "reference": NPZ(time_s, channels, human)}. Reference motion is interpolated onto
    the feature clock. Groups are split so validation and test clips are unseen.
    """
    rows = json.loads(pairs_path.read_text())
    clips = []
    for row in rows:
        with np.load(pairs_path.parent / row["features"], allow_pickle=False) as data:
            times, features = clock(data), array(data, "features")
        with np.load(pairs_path.parent / row["reference"], allow_pickle=False) as data:
            ref_times = clock(data)
            if tuple(data["channels"].tolist()) != CHANNELS:
                raise ValueError(f"{row['id']}: reference channels must be {CHANNELS}")
            reference = array(data, "human", (len(ref_times), len(CHANNELS)))
        inside = (times >= ref_times[0]) & (times <= ref_times[-1])
        target = np.column_stack([np.interp(times[inside], ref_times, reference[:, k])
                                  for k in range(len(CHANNELS))])
        clips.append(type("Clip", (), {"id": row["id"], "group": row["group"],
                                       "x": features[inside], "y": target})())
    if len({clip.x.shape[1] for clip in clips}) != 1:
        raise ValueError("All decoded exports need the same feature width")
    split = split_groups(clips, seed)

    def stack(name):
        return (np.concatenate([c.x for c in split[name]]), np.concatenate([c.y for c in split[name]]))

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = map(stack, ("train", "validation", "test"))
    mean, scale = x_train.mean(0), x_train.std(0) + 1e-6

    def solve(ridge):
        x = (x_train - mean) / scale
        y_mean = y_train.mean(0)
        weights = np.linalg.solve(x.T @ x + ridge * len(x) * np.eye(x.shape[1]), x.T @ (y_train - y_mean))
        return weights, y_mean

    def predict(params, x):
        return np.clip(((x - mean) / scale) @ params[0] + params[1], -1, 1)

    def metrics(params, x, y):
        error = predict(params, x) - y
        variance = ((y - y.mean(0)) ** 2).mean(0)
        r2 = 1 - (error ** 2).mean(0) / np.where(variance > 1e-9, variance, np.nan)
        return {"mae": dict(zip(CHANNELS, np.abs(error).mean(0).round(4).tolist())),
                "r2": dict(zip(CHANNELS, [None if not np.isfinite(v) else round(float(v), 4) for v in r2])),
                "frames": int(len(x))}

    scored = [(float(((predict(solve(r), x_valid) - y_valid) ** 2).mean()), r) for r in ridges]
    ridge = min(scored)[1]
    params = solve(ridge)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, weights=params[0].astype(np.float32), bias=params[1].astype(np.float32),
             feature_mean=mean.astype(np.float32), feature_scale=scale.astype(np.float32))
    report = {"ridge": ridge, "feature_width": int(x_train.shape[1]),
              "splits": {name: sorted({c.group for c in clips_}) for name, clips_ in split.items()},
              "validation": metrics(params, x_valid, y_valid), "test": metrics(params, x_test, y_test)}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report
