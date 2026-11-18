"""Run G from measured simulator state and score realized motion with R."""

import json
from pathlib import Path

import numpy as np
import torch

from .audio import FRAME_S
from .gr_data import load_stream, window_at
from .gr_model import ACTUATORS, CHANNELS, SCALE_RAD, STEPS, capability, normalized_limits
from .gr_train import load_models


class MotionPlanner:
    def __init__(self, generator: Path, observer: Path | None, stream: Path, profile: str,
                 calibration: dict | None = None):
        torch.set_num_threads(1)
        self.g, self.r, self.metadata = load_models(generator, observer)
        self.times, self.human, self.context = load_stream(stream)
        if self.context.shape[1] != self.metadata["context_dim"]:
            raise ValueError("Stream context width differs from training")
        self.profile = profile
        self.limits = capability(profile, calibration)
        self.limit_tensor = torch.from_numpy(normalized_limits(self.limits)[None])
        self.groups = tuple(ACTUATORS.values())
        self.rates = {joint: float(row[3])
                      for group, row in zip(self.groups, self.limits) for joint in group}
        self.frame = -1
        self.anchor = 0
        self.target = np.empty((0, len(CHANNELS)), dtype=np.float32)
        self.positions = []
        self.initial_states = []
        self.chunks = []
        self.current_context = None

    def joint_vector(self, values: dict[str, float]) -> np.ndarray:
        return np.array([np.mean([values[joint] for joint in group]) for group in self.groups], dtype=np.float32)

    def timeline(self, source: str | None = None) -> dict:
        return {"version": 1, "time_s": self.times.tolist(), "profile": self.profile,
                "source": source or f"G/R replay ({', '.join(self.metadata['origins'])})",
                "policy": self.metadata["model"], "speech_state": ["trajectory"] * len(self.times),
                "jaw_aperture": [0.0] * len(self.times), "events": []}

    @torch.inference_mode()
    def _start_chunk(self, frame: int, positions: dict, velocities: dict) -> None:
        self.anchor = frame
        state = np.stack([self.joint_vector(positions), self.joint_vector(velocities)])
        self.initial_states.append(state)
        count = min(STEPS, len(self.times) - frame)
        if frame == 0:
            position = np.clip(state[0], self.limits[:, 1], self.limits[:, 2])
            self.target = np.repeat(position[None], count, axis=0)
            return
        history, context = window_at(self.human, self.context, frame)
        self.current_context = torch.from_numpy(context[None])
        samples = self.g(torch.from_numpy(history[None]), self.current_context, self.limit_tensor,
                         torch.from_numpy((state / SCALE_RAD)[None]))
        self.target = samples[0].numpy()[:count] * SCALE_RAD

    @torch.inference_mode()
    def _close_chunk(self) -> None:
        motion = np.asarray(self.positions[self.anchor:], dtype=np.float32)
        if not len(motion):
            return
        score = None
        if self.r is not None and self.current_context is not None and len(motion) == STEPS:
            scores = self.r(torch.from_numpy((motion / SCALE_RAD)[None]),
                            self.current_context, self.limit_tensor)[0].softmax(0).tolist()
            score = dict(zip(self.metadata["labels"], scores))
        self.chunks.append({"start_s": round(self.anchor * FRAME_S, 6),
                            "end_s": float(self.times[self.anchor + len(motion) - 1]),
                            "r_probabilities_on_realized_motion": score})

    def targets(self, now: float, positions: dict, velocities: dict) -> dict:
        index = min(int((now + 1e-8) // FRAME_S), len(self.times) - 1)
        if index != self.frame:
            if self.frame >= 0:
                self.positions.append(self.joint_vector(positions))
            if index % STEPS == 0:
                self._close_chunk()
                self._start_chunk(index, positions, velocities)
            self.frame = index
        return {joint: float(value)
                for group, value in zip(self.groups, self.target[index - self.anchor]) for joint in group}

    def finish(self, positions: dict, output: Path) -> None:
        self.positions.append(self.joint_vector(positions))
        if len(self.positions) != len(self.times):
            raise RuntimeError("Simulator did not reach every expected 20 Hz frame")
        self._close_chunk()
        np.savez_compressed(output / "realized-motion.npz", time_s=self.times,
                            channels=np.array(CHANNELS), robot_rad=np.asarray(self.positions),
                            initial_state_rad=np.asarray(self.initial_states),
                            chunk_start_s=np.array([chunk["start_s"] for chunk in self.chunks]))
        (output / "interpretations.json").write_text(json.dumps({
            "training_origins": self.metadata["origins"], "profile": self.profile,
            "capabilities": dict(zip(CHANNELS, self.limits.tolist())), "chunks": self.chunks,
        }, indent=2, allow_nan=False) + "\n")


def run(generator: Path, observer: Path | None, stream: Path, profile: str, output: Path,
        video: Path | None, audio: Path | None, calibration: Path | None,
        source: str | None) -> dict:
    from .simulate import replay

    limits = json.loads(calibration.read_text()) if calibration is not None else None
    planner = MotionPlanner(generator, observer, stream, profile, limits)
    output.mkdir(parents=True, exist_ok=True)
    timeline_path = output / "timeline.json"
    timeline_path.write_text(json.dumps(planner.timeline(source), indent=2) + "\n")
    return replay(timeline_path, output, video, audio, planner)
