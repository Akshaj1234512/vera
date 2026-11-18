"""Capability-conditioned trajectory generation and interpretation."""

import numpy as np
import torch
from torch import nn

from .audio import FRAME_S

STEPS = 10
CHANNELS = (
    "neck_yaw", "neck_pitch", "jaw", "eye_yaw",
    "eyelid_l", "eyelid_r", "brow_l", "brow_r",
)
ACTUATORS = {name: (name,) for name in CHANNELS}
ACTUATORS["eye_yaw"] = ("eye_yaw_l", "eye_yaw_r")
SCALE_RAD = np.array([0.25, 0.25, 0.30, 0.40, 0.55, 0.55, 0.025, 0.025], dtype=np.float32)
# Each row holds enabled, lower/upper position in radians, and radians per second.
SIM_LIMITS = np.array([
    [1, -0.25, 0.25, 1.20], [1, -0.20, 0.24, 1.60], [1, 0, 0.32, 2.00],
    [1, -0.38, 0.38, 2.30], [1, 0, 0.55, 5.00], [1, 0, 0.55, 5.00],
    [1, -0.015, 0.030, 0.24], [1, -0.015, 0.030, 0.24],
], dtype=np.float32)
PROFILES = {"full": (), "fixed-neck-pitch": ("neck_pitch",), "fixed-eyes": ("eye_yaw",)}


def capability(profile: str, custom: dict | None = None) -> np.ndarray:
    """Resolve a simulator profile or an explicit per-channel calibration."""
    if custom is None:
        limits = SIM_LIMITS.copy()
        for name in PROFILES[profile]:
            limits[CHANNELS.index(name)] = 0
    else:
        if set(custom) != set(CHANNELS):
            raise ValueError("Capability calibration must cover all eight channels")
        limits = np.array([custom[name] for name in CHANNELS], dtype=np.float32)
    if limits.shape != (len(CHANNELS), 4) or not np.isfinite(limits).all():
        raise ValueError("Each channel needs enabled, minimum, maximum and speed")
    enabled, low, high, speed = limits.T
    if (not np.isin(enabled, [0, 1]).all() or np.any(low > 0) or np.any(high < 0)
            or np.any(high < low) or np.any(speed[enabled == 1] <= 0)):
        raise ValueError("Invalid capability bounds around the zero pose")
    limits[enabled == 0] = 0
    return limits


def normalized_limits(limits: np.ndarray) -> np.ndarray:
    result = limits.copy()
    result[:, 1:] /= SCALE_RAD[:, None]
    return result


def constrain_motion(target: torch.Tensor, limits: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    """Bound joint positions and per-step velocity, starting at the measured pose."""
    enabled, low, high, speed = limits.unbind(-1)
    previous = position.clamp(low, high)
    delta = speed * FRAME_S
    result = []
    for step in target.unbind(1):
        previous = (previous + (step - previous).clamp(-delta, delta)).clamp(low, high) * enabled
        result.append(previous)
    return torch.stack(result, dim=1)


class Generator(nn.Module):
    """G maps a causal human-motion history and current context to a robot chunk."""

    def __init__(self, context_dim: int, width: int = 48, *, residual: bool = False):
        super().__init__()
        channels = len(CHANNELS)
        self.residual = residual
        self.motion = nn.Linear(channels + 1, width)
        self.conditioning = nn.Linear(context_dim + channels * 6, width)
        self.position = nn.Parameter(torch.randn(1, STEPS, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.output = nn.Linear(width, STEPS * channels)
        # Start near the neutral pose so the output starts inside the slew bounds.
        nn.init.normal_(self.output.weight, std=0.003)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("causal_mask", torch.ones(STEPS, STEPS, dtype=torch.bool).triu(1),
                             persistent=False)

    def forward(self, human, context, limits, state):
        conditioning = self.conditioning(torch.cat([context, limits.flatten(1), state.flatten(1)], dim=1))
        inputs = self.motion(human) + self.position + conditioning[:, None]
        encoded = self.encoder(inputs, mask=self.causal_mask)
        target = self.output(encoded[:, -1]).reshape(-1, STEPS, len(CHANNELS)).tanh() * 1.5
        if self.residual:
            target = human[:, :, :-1] + 0.5 * target
        return constrain_motion(target, limits, state[:, 0])


class Observer(nn.Module):
    """R predicts observer vote proportions from realized motion and context."""

    def __init__(self, context_dim: int, labels: int):
        super().__init__()
        motion_dim = STEPS * len(CHANNELS)
        self.network = nn.Sequential(
            nn.Linear(motion_dim + context_dim + len(CHANNELS) * 4, 64),
            nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 48), nn.GELU(), nn.Linear(48, labels),
        )

    def forward(self, motion, context, limits):
        return self.network(torch.cat([motion.flatten(1), context, limits.flatten(1)], dim=1))
