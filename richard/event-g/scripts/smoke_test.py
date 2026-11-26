"""Five checks that Model G's interface works end to end, on fake data.

Run from the repo root:  python scripts/smoke_test.py
Exit code 0 = all checks passed.
"""
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vera.config import GConfig, CUE_HEADS, ENABLE  # noqa: E402
from vera.model_g import ModelG  # noqa: E402
from vera.losses import event_loss  # noqa: E402

B, T = 4, 50          # 50 frames = 4 s at 12.5 Hz
CUT = 30              # causality check edits frames CUT.. onward
PARAM_BUDGET = 20_000_000


def fake_batch(cfg: GConfig, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    H = cfg.n_cue_heads
    x = torch.randn(B, T, cfg.d_speech, generator=g)
    m_h = torch.randn(B, T, cfg.n_motion, generator=g)
    b_m = torch.rand(B, H, cfg.d_cap, generator=g)
    b_m[..., 0] = 1.0                                  # all heads enabled by default
    targets = {
        "enable": (torch.rand(B, T, H, generator=g) < 0.3).float(),
        "role": torch.randint(0, cfg.n_roles, (B, T, H), generator=g),
        "params": torch.rand(B, T, H, cfg.n_params, generator=g),
        "onset": torch.rand(B, T, H, generator=g),
        "duration": torch.rand(B, T, H, generator=g),
    }
    return x, m_h, b_m, targets


def check_shape(model, cfg):
    x, m_h, b_m, _ = fake_batch(cfg)
    out = model(x, m_h, b_m)
    want = (B, T, cfg.n_cue_heads, cfg.d_out)
    assert tuple(out.shape) == want, f"got {tuple(out.shape)}, want {want}"
    assert torch.isfinite(out).all(), "output has NaN/inf"
    return f"output shape [B, T, 6, 11] = {list(out.shape)}"


def check_capability_mask(model, cfg):
    x, m_h, b_m, _ = fake_batch(cfg)
    gaze = CUE_HEADS.index("gaze")
    b_m[:, gaze, 0] = 0.0                              # robot has fixed eyes
    p_fire = torch.sigmoid(model(x, m_h, b_m)[..., gaze, ENABLE])
    assert (p_fire == 0).all(), f"gaze fired with max prob {p_fire.max().item():.3g}"
    return "disabled head (gaze) never fires"


def check_causal(model, cfg):
    x, m_h, b_m, _ = fake_batch(cfg)
    out_a = model(x, m_h, b_m)
    x2, m2 = x.clone(), m_h.clone()
    x2[:, CUT:] = torch.randn_like(x2[:, CUT:])
    m2[:, CUT:] = torch.randn_like(m2[:, CUT:])
    out_b = model(x2, m2, b_m)
    diff = (out_a[:, :CUT] - out_b[:, :CUT]).abs().max().item()
    assert diff < 1e-4, f"frames 0-{CUT - 1} changed by {diff:.3g} when future frames changed"
    return f"causal: future frames don't change past outputs (max diff {diff:.1e})"


def check_learns(model, cfg, steps: int = 200):
    x, m_h, b_m, targets = fake_batch(cfg, seed=1)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = None
    for _ in range(steps):
        loss = event_loss(model(x, m_h, b_m), targets, b_m)["total"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
    model.eval()
    drop = 1 - loss.item() / first
    assert drop > 0.9, f"loss only fell {drop:.0%} ({first:.3f} -> {loss.item():.3f})"
    return f"loss drops {drop:.0%} overfitting one batch ({first:.3f} -> {loss.item():.3f})"


def check_size(model, cfg):
    n = sum(p.numel() for p in model.parameters())
    assert n < PARAM_BUDGET, f"{n / 1e6:.1f}M params is over the {PARAM_BUDGET / 1e6:.0f}M budget"
    return f"params: {n / 1e6:.1f}M (budget {PARAM_BUDGET / 1e6:.0f}M)"


def main() -> int:
    torch.manual_seed(0)
    torch.set_num_threads(min(8, os.cpu_count() or 1))  # shared machine: don't grab every core
    cfg = GConfig()
    model = ModelG(cfg).eval()
    checks = [check_shape, check_capability_mask, check_causal, check_learns, check_size]
    failed = 0
    for check in checks:
        try:
            with torch.set_grad_enabled(check is check_learns):
                print(f"[PASS] {check(model, cfg)}")
        except NotImplementedError:
            print(f"[FAIL] {check.__name__}: ModelG.forward() is not implemented yet (see TODOs)")
            failed += 1
        except AssertionError as e:
            print(f"[FAIL] {check.__name__}: {e}")
            failed += 1
        except Exception:
            print(f"[FAIL] {check.__name__}: crashed")
            traceback.print_exc()
            failed += 1
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
