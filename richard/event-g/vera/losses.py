"""Training losses for Model G.

pred:    [B, T, H, 11] from ModelG.forward
targets: dict of tensors, H = number of cue heads
    enable   [B, T, H]     float, 1 if the head should fire at this frame
    role     [B, T, H]     long, role index (only used where enable == 1)
    params   [B, T, H, 4]  float
    onset    [B, T, H]     float, seconds
    duration [B, T, H]     float, seconds
b_m:     [B, H, 5] capability descriptor; feature 0 = enabled
"""
import torch
import torch.nn.functional as F

from vera.config import ENABLE, ROLE, PARAMS, ONSET, DURATION


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def event_loss(pred: torch.Tensor, targets: dict, b_m: torch.Tensor) -> dict:
    """Supervised event loss. Returns the total plus each term for logging."""
    B, T, H, _ = pred.shape
    head_on = (b_m[..., 0] > 0.5)[:, None, :].expand(B, T, H)   # robot can use this head
    fires = (targets["enable"] > 0.5) & head_on                  # head should fire here

    # Enable: binary cross-entropy, only on heads the robot actually has.
    enable = _masked_mean(
        F.binary_cross_entropy_with_logits(
            pred[..., ENABLE], targets["enable"], reduction="none"),
        head_on)

    # Everything else only matters where the head fires.
    role = _masked_mean(
        F.cross_entropy(pred[..., ROLE].reshape(-1, pred[..., ROLE].shape[-1]),
                        targets["role"].reshape(-1), reduction="none").view(B, T, H),
        fires)
    params = _masked_mean((pred[..., PARAMS] - targets["params"]).abs().mean(-1), fires)
    timing = _masked_mean(
        (pred[..., ONSET] - targets["onset"]).abs()
        + (pred[..., DURATION] - targets["duration"]).abs(),
        fires)

    total = enable + role + params + timing
    return {"total": total, "enable": enable, "role": role, "params": params, "timing": timing}


def smooth_loss(pred: torch.Tensor, b_m: torch.Tensor) -> torch.Tensor:
    """Penalize acceleration of cue parameters over time (anti-tremor term)."""
    p = pred[..., PARAMS]                                  # [B, T, H, 4]
    accel = p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2]          # 2nd difference along time
    head_on = (b_m[..., 0] > 0.5)[:, None, :].expand_as(accel[..., 0])
    return _masked_mean(accel.pow(2).mean(-1), head_on)


def total_loss(pred, targets, b_m, smooth_weight: float = 0.1) -> dict:
    terms = event_loss(pred, targets, b_m)
    terms["smooth"] = smooth_loss(pred, b_m)
    terms["total"] = terms["total"] + smooth_weight * terms["smooth"]
    return terms
