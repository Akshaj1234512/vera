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
