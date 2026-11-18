"""Exercise the full G/R path with a synthetic gesture and vote protocol."""

import json
from pathlib import Path

import numpy as np

from .audio import FRAME_S
from .gr_model import CHANNELS, PROFILES, SCALE_RAD, STEPS

LABELS = ("unclear", "neutral_ack", "enthusiastic_ack")


def make_fixtures(out: Path, groups: int, seed: int) -> Path:
    if groups < 5:
        raise ValueError("Use at least five fixture groups")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    phase = np.arange(1, STEPS + 1) / STEPS
    pulse = np.sin(np.pi * phase) ** 2
    quick_pulse = np.sin(np.pi * phase ** 0.72) ** 2
    state = np.zeros((2, len(CHANNELS)), dtype=np.float32)
    rows = []

    def inputs(intensity, noise):
        human = np.zeros((STEPS, len(CHANNELS)), dtype=np.float32)
        human[:, 1] = intensity * pulse
        human[:, 6:] = intensity * 0.1 * pulse[:, None]
        context = np.array([0.72, 0.41 + intensity * 0.18, 0.83, 1, 0.02, 0, 0, 0], dtype=np.float32)
        context += noise
        return human, context

    def movement(label, profile, amplitude):
        motion = np.zeros((STEPS, len(CHANNELS)), dtype=np.float32)
        shape = quick_pulse if label == 2 else pulse
        shape = np.roll(shape, 1)
        shape[0] = 0
        if profile == "fixed-neck-pitch":
            motion[:, 3] = amplitude * shape * 0.9
            motion[:, 6:] = (amplitude * (0.52 if label == 2 else 0.22) * shape)[:, None]
        else:
            motion[:, 1] = amplitude * shape
            motion[:, 6:] = (amplitude * (0.36 if label == 2 else 0.07) * shape)[:, None]
        return motion * SCALE_RAD

    for group in range(groups):
        scale = rng.uniform(0.88, 1.08)
        turn_bias = rng.normal(0, 0.035, 8).astype(np.float32)
        for take in range(2):
            levels = [0, rng.uniform(0.34, 0.49) * scale, rng.uniform(0.70, 0.82) * scale]
            for intent, amplitude in enumerate(levels):
                human, context = inputs(amplitude, turn_bias)
                for profile in PROFILES:
                    for observed in range(len(LABELS)):
                        trial_id = f"{group:02d}-{take}-{intent}-{profile}-{observed}"
                        name = f"{trial_id}.npz"
                        robot = movement(observed, profile, levels[observed])
                        np.savez_compressed(out / name, human=human, context=context, state_rad=state,
                                            robot_rad=robot)
                        votes = {LABELS[observed]: 9, LABELS[(observed + 1) % len(LABELS)]: 1}
                        rows.append({"id": trial_id, "group": f"synthetic-{group:02d}", "file": name,
                                     "origin": "synthetic_gr_fixture", "profile": profile,
                                     "intent": LABELS[intent], "votes": votes,
                                     "demonstration": observed == intent})
    path = out / "manifest.json"
    path.write_text(json.dumps({"labels": LABELS, "trials": rows}, indent=2) + "\n")

    human, context = [], []
    for label in (0, 1, 0, 2, 0, 1, 0, 2, 0):
        track, feature = inputs([0, 0.44, 0.78][label], 0)
        human.append(track)
        context.append(np.repeat(feature[None], STEPS, axis=0))
    human = np.concatenate(human)
    np.savez_compressed(out / "replay.npz", time_s=np.arange(1, len(human) + 1) * FRAME_S,
                        human=human, human_valid=np.ones(len(human)), context=np.concatenate(context))
    return path
