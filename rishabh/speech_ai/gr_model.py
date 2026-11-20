"""Capability-conditioned trajectory generation and interpretation."""

import numpy as np
import torch
from torch import nn

from .audio import FRAME_S

from .embodiment import CHANNELS, PROFILES, SCALE, SIM_LIMITS
from .human import HUMAN_CHANNELS, RETARGET_NORM

STEPS = 10  # 500 ms at 20 Hz
HUMAN_DIM, ROBOT_DIM = len(HUMAN_CHANNELS), len(CHANNELS)


def capability(profile: str, custom: dict | None = None) -> np.ndarray:
    """Resolve a simulator profile or an explicit per-channel calibration."""
    if custom is None:
        limits = SIM_LIMITS.copy()
        for name in PROFILES[profile]:
            limits[CHANNELS.index(name)] = 0
    else:
        if set(custom) != set(CHANNELS):
            raise ValueError(f"Capability calibration must cover all {len(CHANNELS)} rig channels")
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
    result[:, 1:] /= SCALE[:, None]
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


def hold_last_valid(human: torch.Tensor) -> torch.Tensor:
    """The most recent valid human frame [B, HUMAN_DIM] (zeros when none is valid)."""
    valid = human[:, :, -1] > 0.5
    index = torch.where(valid, torch.arange(human.shape[1], device=human.device), -1).max(dim=1).values
    frame = human[torch.arange(len(human)), index.clamp_min(0), :-1]
    return frame * (index >= 0)[:, None]


class Generator(nn.Module):
    """G: M_r = G(M_h, x_t, b_m, q_t), a small causal transformer over the 500 ms history.

    Input M_h is [B, 10, HUMAN_DIM + 1] (human channels + validity). Output is the next
    500 ms of robot motion [B, 10, ROBOT_DIM] in normalized units, projected onto the
    capability descriptor. With ``residual`` the network predicts an adjustment to the
    projected teacher motion M_direct (the fixed retarget of the latest human frame).
    """

    def __init__(self, context_dim: int, width: int = 64, *, residual: bool = False):
        super().__init__()
        self.residual = residual
        self.motion = nn.Linear(HUMAN_DIM + 1, width)
        self.conditioning = nn.Linear(context_dim + ROBOT_DIM * 6, width)
        self.position = nn.Parameter(torch.randn(1, STEPS, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.output = nn.Linear(width, STEPS * ROBOT_DIM)
        # Start near the neutral pose so the output starts inside the slew bounds.
        nn.init.normal_(self.output.weight, std=0.003)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("causal_mask", torch.ones(STEPS, STEPS, dtype=torch.bool).triu(1),
                             persistent=False)
        self.register_buffer("retarget", torch.from_numpy(RETARGET_NORM), persistent=False)

    def direct(self, human: torch.Tensor) -> torch.Tensor:
        """M_direct: projected teacher motion, the latest human frame retargeted and held."""
        return (hold_last_valid(human) @ self.retarget)[:, None].expand(-1, STEPS, -1)

    def forward(self, human, context, limits, state):
        conditioning = self.conditioning(torch.cat([context, limits.flatten(1), state.flatten(1)], dim=1))
        inputs = self.motion(human) + self.position + conditioning[:, None]
        encoded = self.encoder(inputs, mask=self.causal_mask)
        target = self.output(encoded[:, -1]).reshape(-1, STEPS, ROBOT_DIM).tanh() * 1.5
        if self.residual:
            target = self.direct(human) + 0.5 * target
        return constrain_motion(target, limits, state[:, 0])


class Observer(nn.Module):
    """R predicts observer vote proportions from realized motion, capabilities and context."""

    def __init__(self, context_dim: int, labels: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(STEPS * ROBOT_DIM + context_dim + ROBOT_DIM * 4, 128),
            nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, labels),
        )

    def forward(self, motion, context, limits):
        return self.network(torch.cat([motion.flatten(1), context, limits.flatten(1)], dim=1))
