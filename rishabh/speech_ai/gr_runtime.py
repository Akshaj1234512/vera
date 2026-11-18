"""The page-3 loop: conversational inputs -> G -> actuator commands -> measured state.

One loop (``run_session``) serves both offline replay of a prepared stream and the
live path, where teacher motion, PersonaPlex states and audio features arrive
over time. Any robot that implements ``RobotInterface`` can be driven; this repo
ships the MuJoCo adapter in ``simulate.py`` and a kinematic test adapter here.
"""

from abc import ABC, abstractmethod
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .audio import FEATURE_NAMES, FRAME_S, normalized
from .gr_data import (STATES_MAX_AGE_S, TEACHER_MAX_AGE_S, TimedSource, align, file_digest,
                      load_stream, window_at)
from .gr_model import CHANNELS, SCALE_RAD, STEPS, capability, constrain_motion, normalized_limits
from .gr_train import load_models


# --------------------------------------------------------------------------- #
# Robot interface
# --------------------------------------------------------------------------- #
class RobotInterface(ABC):
    """What G needs from any embodiment, in the eight-channel joint space (radians).

    A hardware adapter must: map each channel to its actuators (``eye_yaw`` drives
    both eyes), read encoder position and velocity in radians and rad/s at chunk
    boundaries, accept position targets at 20 Hz, and enforce its own firmware
    limits. Calibration (enabled, min, max, max speed) is passed separately.
    """

    @abstractmethod
    def state(self) -> np.ndarray:
        """Measured [2, 8] position (rad) and velocity (rad/s)."""

    @abstractmethod
    def command(self, target_rad: np.ndarray) -> None:
        """Set an [8] position target to be reached by the end of the current 50 ms frame."""

    @abstractmethod
    def advance(self, seconds: float) -> None:
        """Let time pass: step physics, or wait on the wall clock for hardware."""

    def close(self) -> dict:
        return {}


class KinematicRobot(RobotInterface):
    """Test adapter: first-order tracking with per-channel speed limits, no physics."""

    def __init__(self, limits: np.ndarray, lag: float = 0.6):
        self.limits = limits
        self.lag = lag
        self.position = np.zeros(len(CHANNELS))
        self.velocity = np.zeros(len(CHANNELS))
        self.target = np.zeros(len(CHANNELS))

    def state(self) -> np.ndarray:
        return np.stack([self.position, self.velocity]).astype(np.float32)

    def command(self, target_rad: np.ndarray) -> None:
        self.target = np.asarray(target_rad, dtype=float)

    def advance(self, seconds: float) -> None:
        step = np.clip((self.target - self.position) * self.lag, -self.limits[:, 3] * seconds,
                       self.limits[:, 3] * seconds)
        self.velocity = step / seconds
        self.position = self.position + step


# --------------------------------------------------------------------------- #
# Inputs available at a chunk boundary
# --------------------------------------------------------------------------- #
class StreamInputs:
    """A prepared 20 Hz stream (gr-prepare output). Alignment was done offline."""

    def __init__(self, path: Path):
        self.time_s, self.human, self.context = load_stream(path)

    def window(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return window_at(self.human, self.context, index)


class LiveInputs:
    """Buffers samples as they arrive; a boundary only sees what arrived by then.

    ``push_*`` take the sample timestamp on the common recording clock and the
    arrival time on the same clock. For each history frame, the latest teacher
    sample with time <= frame time, arrival <= boundary and age <= 150 ms is used.
    """

    def __init__(self, frames: int, state_dim: int | None):
        self.frames = frames
        self.time_s = np.round(np.arange(1, frames + 1) * FRAME_S, 6)
        self.state_dim = state_dim
        self.buffers = {"teacher": [], "audio": [], "states": []}
        # What G actually saw, per frame; written as the session's stream.npz.
        self.seen_human = np.zeros((frames, len(CHANNELS)), dtype=np.float32)
        self.seen_valid = np.zeros(frames, dtype=np.float32)
        width = len(FEATURE_NAMES) + (state_dim + 1 if state_dim else 0)
        self.seen_context = np.zeros((frames, width), dtype=np.float32)

    def push(self, name: str, time_s: float, arrival_s: float, values: np.ndarray, valid: bool = True) -> None:
        buffer = self.buffers[name]
        if buffer and (time_s <= buffer[-1][0] or arrival_s < buffer[-1][1]):
            raise ValueError(f"{name}: samples must arrive in timestamp order")
        if arrival_s < time_s - 1e-9:
            raise ValueError(f"{name}: a sample cannot arrive before its timestamp")
        buffer.append((time_s, arrival_s, np.asarray(values, dtype=np.float32), bool(valid)))

    def _source(self, name: str, width: int) -> TimedSource | None:
        rows = self.buffers[name]
        if not rows:
            return None
        return TimedSource(np.array([r[0] for r in rows]), np.array([r[1] for r in rows]),
                           np.stack([r[2] for r in rows]).reshape(len(rows), width),
                           np.array([r[3] for r in rows]))

    def _aligned(self, name, width, query, deadline, max_age):
        source = self._source(name, width)
        if source is None:
            return np.zeros((len(query), width), dtype=np.float32), np.zeros(len(query), dtype=bool)
        values, observed, _ = align(source, query, max_age, deadline=deadline)
        return values, observed

    def window(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        boundary = index * FRAME_S
        start = max(0, index - STEPS)
        frames = np.arange(start, index)
        query = self.time_s[frames]
        human, valid = self._aligned("teacher", len(CHANNELS), query, boundary, TEACHER_MAX_AGE_S)
        self.seen_human[frames], self.seen_valid[frames] = human * valid[:, None], valid
        if index:
            last = np.array([self.time_s[index - 1]])
            audio, heard = self._aligned("audio", len(FEATURE_NAMES), last, boundary, FRAME_S)
            context = [normalized(audio) * heard[:, None]]
            if self.state_dim:
                states, available = self._aligned("states", self.state_dim, last, boundary, STATES_MAX_AGE_S)
                context.append(np.column_stack([states, available]))
            self.seen_context[index - 1] = np.concatenate(context, axis=1)[0]
        stacked = np.column_stack([self.seen_human, self.seen_valid]).astype(np.float32)
        return window_at(stacked, self.seen_context, index)

    def save_stream(self, path: Path) -> None:
        with path.open("wb") as file:
            np.savez_compressed(file, time_s=self.time_s, human=self.seen_human,
                                human_valid=self.seen_valid, context=self.seen_context)


class FileFeed:
    """Replays recorded sources into ``LiveInputs`` as their arrival times pass."""

    def __init__(self, inputs: LiveInputs, sources: dict[str, TimedSource]):
        self.inputs, self.sources = inputs, sources
        self.cursor = dict.fromkeys(sources, 0)

    def deliver(self, now: float) -> None:
        for name, source in self.sources.items():
            while self.cursor[name] < len(source.time_s) and source.arrival_s[self.cursor[name]] <= now + 1e-9:
                k = self.cursor[name]
                self.inputs.push(name, float(source.time_s[k]), float(source.arrival_s[k]),
                                 source.values[k], bool(source.valid[k]))
                self.cursor[name] += 1


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #
class GPolicy:
    name = "g"

    def __init__(self, generator: Path, observer: Path | None = None):
        self.g, self.r, self.metadata = load_models(generator, observer)
        self.name = self.metadata["model"]
        self.context_dim = self.metadata["context_dim"]
        self.digest = file_digest(generator)

    def __call__(self, history, context, limits, state):
        return self.g(history, context, limits, state)


class ProjectedPolicy:
    """Baseline: replay the last 500 ms of projected DyaPlex motion within the robot's limits."""
    name = "projected-dyaplex"
    r = None
    context_dim = None
    digest = None
    metadata = {"labels": None}

    def __call__(self, history, context, limits, state):
        return constrain_motion(history[:, :, :-1], limits, state[:, 0])


# --------------------------------------------------------------------------- #
# The chunk loop
# --------------------------------------------------------------------------- #
class Session:
    def __init__(self, policy, inputs, frames: int, limits: np.ndarray, observer=None, labels=None):
        self.policy, self.inputs, self.frames, self.limits = policy, inputs, frames, limits
        self.limit_tensor = torch.from_numpy(normalized_limits(limits)[None])
        self.observer, self.labels = observer, labels
        self.measured = np.zeros((frames, len(CHANNELS)), dtype=np.float32)
        self.commanded = np.zeros((frames, len(CHANNELS)), dtype=np.float32)
        self.chunks = []

    @torch.inference_mode()
    def plan(self, index: int, state: np.ndarray) -> np.ndarray:
        count = min(STEPS, self.frames - index)
        history, context = self.inputs.window(index)
        started = perf_counter()
        if index == 0 or not history[:, -1].any():
            # Cold start or no teacher motion in the last 500 ms: ease toward neutral.
            fallback = "cold_start" if index == 0 else "no_teacher_motion"
            target = constrain_motion(torch.zeros(1, STEPS, len(CHANNELS)), self.limit_tensor,
                                      torch.from_numpy(state[None, 0] / SCALE_RAD))[0].numpy()
        else:
            fallback = None
            target = self.policy(torch.from_numpy(history[None]), torch.from_numpy(context[None]),
                                 self.limit_tensor, torch.from_numpy((state / SCALE_RAD)[None]))[0].numpy()
        self.chunks.append({"start_s": round(index * FRAME_S, 6), "initial_state_rad": state.tolist(),
                            "inference_ms": round((perf_counter() - started) * 1000, 3),
                            "teacher_frames_valid": int(history[:, -1].sum()), "fallback": fallback,
                            "context": context})
        return target[:count] * SCALE_RAD

    @torch.inference_mode()
    def score(self, completed_frames: int | None = None) -> None:
        """R scores the motion the robot actually reached, once each chunk is complete."""
        completed_frames = self.frames if completed_frames is None else completed_frames
        for chunk in self.chunks:
            if "r_probabilities_on_realized_motion" in chunk:
                continue
            index = round(chunk["start_s"] / FRAME_S)
            if index + STEPS > completed_frames and completed_frames < self.frames:
                continue
            motion = self.measured[index:index + STEPS]
            if self.observer is None or len(motion) < STEPS or chunk["fallback"] == "cold_start":
                chunk["r_probabilities_on_realized_motion"] = None
                continue
            logits = self.observer(torch.from_numpy((motion / SCALE_RAD)[None]),
                                   torch.from_numpy(chunk["context"][None]), self.limit_tensor)
            chunk["r_probabilities_on_realized_motion"] = dict(zip(self.labels, logits[0].softmax(0).tolist()))


def run_session(session: Session, robot: RobotInterface, feed: FileFeed | None = None,
                on_frame=None) -> None:
    target = None
    for index in range(session.frames):
        boundary = index * FRAME_S
        if feed is not None:
            feed.deliver(boundary)
        if index % STEPS == 0:
            session.score(index)
            anchor = index
            target = session.plan(index, robot.state())
        session.commanded[index] = target[index - anchor]
        robot.command(target[index - anchor])
        robot.advance(FRAME_S)
        session.measured[index] = robot.state()[0]
        if on_frame is not None:
            on_frame(index, session)
    if feed is not None:
        feed.deliver(session.frames * FRAME_S)
    session.score()


def motion_metrics(session: Session) -> dict:
    """Joint-limit compliance, tracking, smoothness and chunk timing for a session."""
    enabled, low, high, speed = session.limits.T
    commanded, measured = session.commanded, session.measured
    # Each chunk is planned from the measured position at its boundary (q_t), so the
    # speed check anchors every chunk there rather than at the previous command.
    previous = np.vstack([commanded[:1], commanded[:-1]])
    for chunk in session.chunks:
        start = np.clip(np.asarray(chunk["initial_state_rad"])[0], low, high)  # as constrain_motion does
        previous[round(chunk["start_s"] / FRAME_S)] = start
    step = commanded - previous
    tol = 1e-5
    velocity = np.diff(measured, axis=0) / FRAME_S
    accel = np.diff(velocity, axis=0)
    starts = [round(c["start_s"] / FRAME_S) for c in session.chunks[1:]]
    boundary = [i - 1 for i in starts if 1 <= i - 1 < len(accel)]
    inside = np.setdiff1d(np.arange(len(accel)), boundary)
    timings = np.array([c["inference_ms"] for c in session.chunks])
    return {
        "commanded_within_position_limits": bool(((commanded >= low - tol) & (commanded <= high + tol)).all()),
        "commanded_within_speed_limits": bool((np.abs(step) <= speed * FRAME_S + tol).all()),
        "disabled_channels_max_abs_measured_rad": float(np.abs(measured[:, enabled == 0]).max(initial=0)),
        "measured_outside_limits_frames": int(((measured < low - 0.01) | (measured > high + 0.01)).any(1).sum()),
        "mean_abs_tracking_error_rad": dict(zip(CHANNELS, np.abs(commanded - measured).mean(0).round(5).tolist())),
        "mean_sq_accel_inside_chunks": float((accel[inside] ** 2).mean()) if len(inside) else None,
        "mean_sq_accel_at_boundaries": float((accel[boundary] ** 2).mean()) if boundary else None,
        "inference_ms": {"mean": float(timings.mean()), "max": float(timings.max())},
        "fallback_chunks": {kind: sum(c["fallback"] == kind for c in session.chunks)
                            for kind in ("cold_start", "no_teacher_motion")},
    }


def write_session(session: Session, output: Path, meta: dict) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    metrics = motion_metrics(session)
    chunks = [{k: v for k, v in c.items() if k != "context"} for c in session.chunks]
    meta = {**meta, "capabilities": dict(zip(CHANNELS, session.limits.tolist()))}
    np.savez_compressed(output / "realized-motion.npz", time_s=np.round(np.arange(1, session.frames + 1) * FRAME_S, 6),
                        channels=np.array(CHANNELS), robot_rad=session.measured, commanded_rad=session.commanded,
                        initial_state_rad=np.array([c["initial_state_rad"] for c in session.chunks], dtype=np.float32),
                        chunk_start_s=np.array([c["start_s"] for c in session.chunks]),
                        inference_ms=np.array([c["inference_ms"] for c in session.chunks]),
                        session=np.array(json.dumps(meta)))
    report = {**meta, "metrics": metrics, "chunks": chunks}
    (output / "interpretations.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def make_policy(generator: Path | None, observer: Path | None, policy: str):
    if policy == "projected":
        judge = None
        if observer is not None:
            # R can still score the baseline's realized motion.
            judge = torch.load(observer, map_location="cpu", weights_only=True)
            from .gr_model import Observer
            model = Observer(judge["context_dim"], len(judge["labels"]))
            model.load_state_dict(judge["weights"])
            return ProjectedPolicy(), model.eval(), judge["labels"], judge["context_dim"]
        return ProjectedPolicy(), None, None, None
    if generator is None:
        raise ValueError("--g is required for the G policy")
    chosen = GPolicy(generator, observer)
    return chosen, chosen.r, chosen.metadata["labels"], chosen.context_dim


def replay(stream: Path, generator: Path | None, observer: Path | None, profile: str, output: Path,
           calibration: Path | None = None, policy: str = "g", video: Path | None = None,
           audio: Path | None = None, title: str | None = None, robot: str = "mujoco",
           session_id: str | None = None, group: str | None = None) -> dict:
    """Offline integration check: drive a robot adapter from a prepared stream."""
    torch.set_num_threads(1)
    limits = capability(profile, json.loads(calibration.read_text()) if calibration else None)
    chosen, judge, labels, context_dim = make_policy(generator, observer, policy)
    inputs = StreamInputs(stream)
    if context_dim is not None and inputs.context.shape[1] != context_dim:
        raise ValueError(f"Stream context width {inputs.context.shape[1]} differs from training ({context_dim})")
    session = Session(chosen, inputs, len(inputs.time_s), limits, judge, labels)
    meta = {"session_id": session_id or output.name, "participant_group": group, "origin": "simulator_replay",
            "context_id": stream.stem, "profile": profile if calibration is None else "custom",
            "policy": chosen.name, "g_sha256": chosen.digest, "stream": str(stream), "robot": robot,
            "synthetic_or_replay": True, "video": str(video) if video else None}
    run_robot(session, robot, limits, output, video, audio, title or f"{chosen.name} / {profile}")
    report = write_session(session, output, meta)
    return report


def live(partner: Path, agent: Path | None, stereo: bool, teacher: Path, teacher_map: Path | None,
         embeddings: Path | None, generator: Path | None, observer: Path | None, profile: str,
         output: Path, calibration: Path | None = None, policy: str = "g", session_id: str | None = None,
         group: str | None = None, context_id: str | None = None, audio_latency_s: float = 0.0,
         teacher_offset_s: float = 0.0, states_offset_s: float = 0.0, robot: str = "mujoco",
         video: Path | None = None, title: str | None = None) -> dict:
    """Live-path rehearsal: sources arrive over time and G only sees what has arrived.

    Sources are recorded files carrying ``arrival_s``; a real deployment pushes the
    same samples into ``LiveInputs`` from the DyaPlex, PersonaPlex and microphone
    processes and swaps the MuJoCo adapter for the robot's ``RobotInterface``.
    """
    from .audio import extract, load_tracks
    from .gr_data import load_states, load_teacher

    torch.set_num_threads(1)
    limits = capability(profile, json.loads(calibration.read_text()) if calibration else None)
    chosen, judge, labels, context_dim = make_policy(generator, observer, policy)
    features = extract(*load_tracks(partner, agent, stereo))
    frames = len(features.time_s)
    states = load_states(embeddings, states_offset_s) if embeddings else None
    inputs = LiveInputs(frames, states.values.shape[1] if states is not None else None)
    width = inputs.seen_context.shape[1]
    if context_dim is not None and width != context_dim:
        raise ValueError(f"Live context width {width} differs from training ({context_dim})")
    sources = {"teacher": load_teacher(teacher, teacher_map, teacher_offset_s),
               "audio": TimedSource(features.time_s, features.time_s + audio_latency_s, features.values,
                                    np.ones(frames, dtype=bool))}
    if states is not None:
        sources["states"] = states
    session = Session(chosen, inputs, frames, limits, judge, labels)
    run_robot(session, robot, limits, output, video, None, title or f"live {chosen.name} / {profile}",
              FileFeed(inputs, sources))
    output.mkdir(parents=True, exist_ok=True)
    inputs.save_stream(output / "stream.npz")
    meta = {"session_id": session_id or output.name, "participant_group": group,
            "context_id": context_id or partner.stem, "origin": "live_rehearsal_from_files",
            "profile": profile if calibration is None else "custom", "policy": chosen.name,
            "g_sha256": chosen.digest, "robot": robot, "audio_latency_s": audio_latency_s,
            "teacher": str(teacher), "personaplex": str(embeddings) if embeddings else None,
            "synthetic_or_replay": True, "video": str(video) if video else None}
    return write_session(session, output, meta)


def run_robot(session: Session, robot: str, limits: np.ndarray, output: Path, video, audio, title, feed=None):
    if robot == "kinematic":
        if video is not None:
            raise ValueError("Video requires the MuJoCo adapter")
        run_session(session, KinematicRobot(limits), feed)
        return
    from .simulate import MujocoRobot

    adapter = MujocoRobot(limits, video=video, audio=audio, title=title, output=output)
    try:
        run_session(session, adapter, feed, on_frame=adapter.annotate)
    finally:
        report = adapter.close()
    output.mkdir(parents=True, exist_ok=True)
    (output / "simulation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["finite_state"] or report["self_contact_steps"]:
        raise RuntimeError(f"Simulation unstable or self-contact: {report}")
