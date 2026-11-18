"""Deterministic audio fixtures for replay and offline training smoke tests."""

from pathlib import Path

import numpy as np

from .audio import write_wav

RATE = 16000


def add_voice(track: np.ndarray, start: float, end: float, pitch: float, slope: float = 0,
              level: float = 0.23, phase: float = 0.0) -> None:
    """Simulate a voiced phrase with a controllable pitch contour and envelope."""
    lo, hi = round(start * RATE), round(end * RATE)
    t = np.arange(hi - lo) / RATE
    ramp = np.minimum(t / 0.025, 1) * np.minimum((end - start - t) / 0.022, 1)
    angle = 2 * np.pi * (pitch * t + slope * t**2 / (2 * (end - start)))
    pulse = 0.64 + 0.36 * np.sin(2 * np.pi * 4.2 * t + phase) ** 2
    track[lo:hi] += level * ramp * pulse * (np.sin(angle) + 0.22 * np.sin(2 * angle))


def correction(output: Path) -> tuple[Path, Path]:
    """A scripted signal that ends agent speech during a partner interruption."""
    output.mkdir(parents=True, exist_ok=True)
    partner, agent = np.zeros((2, 8 * RATE), dtype=np.float32)
    add_voice(partner, 0.40, 1.38, 176, -35)
    add_voice(partner, 2.00, 2.77, 189, -26)
    add_voice(agent, 3.20, 4.80, 132, 10)
    add_voice(partner, 4.43, 5.42, 212, -52)
    add_voice(agent, 6.30, 7.40, 128, -21)
    partner_path, agent_path = (output / name for name in ("partner.wav", "agent.wav"))
    write_wav(partner_path, partner, RATE)
    write_wav(agent_path, agent, RATE)
    return partner_path, agent_path


def make_corpus(output: Path, groups: int, seed: int) -> Path:
    """Create grouped examples with noisy rising/falling phrase endings."""
    import json

    if groups < 5:
        raise ValueError("Use at least five synthetic groups")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    clips = []
    for group in range(groups):
        pitch = rng.uniform(115, 250)
        level = rng.uniform(0.13, 0.29)
        for take in range(3):
            duration = 12
            partner, agent = rng.normal(0, rng.uniform(0.0005, 0.004), (2, duration * RATE)).astype(np.float32)
            annotations = []
            for i in range(5):
                end = 1.55 + 2.18 * i + rng.uniform(-0.22, 0.22)
                label = bool(rng.random() < 0.5)
                overlap = bool(rng.random() < 0.12)
                length = rng.uniform(0.57, 1.15)
                slope = rng.normal(-65 if label else 70, 39)
                add_voice(partner, end - length, end, pitch + rng.uniform(-15, 15),
                          slope, level, rng.uniform(0, 6))
                if overlap:
                    add_voice(agent, end - 0.26, end + 0.35, 155, 15, 0.18)
                elif label:
                    annotations.append(round(end + rng.uniform(0.12, 0.32), 3))
            name = f"group-{group:02d}-{take}"
            partner_path, agent_path = f"{name}-partner.wav", f"{name}-agent.wav"
            write_wav(output / partner_path, partner, RATE)
            write_wav(output / agent_path, agent, RATE)
            clips.append({"id": name, "group": f"synthetic-{group:02d}", "origin": "synthetic",
                          "partner": partner_path, "agent": agent_path, "ack_s": annotations})
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(clips, indent=2) + "\n")
    return manifest
