"""Full-duplex motion tower (§4.2-4.3, Fig. 2b).

Per block [P Fig. 2b order]: dyadic causal self-attention (RoPE over flat token index)
-> cross-attention to PersonaPlex layer l (time-aligned speech-motion RoPE, Eq. 2-4)
-> SwiGLU MLP. Pre-norm RMSNorm [A], cross-attention output projection h_c*d_c -> d_m [A],
SwiGLU hidden = 2*int(ff_scale*d)//3 [A, Moshi gating.py], separate LM head [A].
Attention is written out explicitly (no fused kernels) so that masked positions contribute
exact zeros: changing future inputs leaves earlier outputs bit-identical in fp32.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope(x: torch.Tensor, pos: torch.Tensor, max_period: float) -> torch.Tensor:
    """RoFormer rotation of adjacent pairs [P cites 32]. x [B, H, T, D], pos [T] (or [B, T])."""
    d = x.shape[-1]
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, d, 2, dtype=torch.float32, device=x.device) / d)
    angle = pos.to(torch.float32)[..., None] * freqs                      # [..., T, D/2]
    if angle.dim() == 2:
        angle = angle[None, None]
    else:
        angle = angle[:, None]
    cos, sin = angle.cos(), angle.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


def attend(q, k, v, allowed, need_weights: bool = False):
    """Masked softmax attention; allowed [Tq, Tk] boolean. Masked keys get exactly zero weight.

    Uses the fused SDPA kernel (memory-light for training); the explicit path is kept for
    inspecting attention maps (Fig. 5)."""
    if not need_weights:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed), None
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~allowed, float("-inf"))
    weights = torch.nan_to_num(torch.softmax(scores, dim=-1))
    return weights @ v, weights


class SelfAttention(nn.Module):
    def __init__(self, cfg: C.TowerConfig):
        super().__init__()
        self.h, self.dh = cfg.heads, cfg.d_model // cfg.heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.max_period, self.window = cfg.rope_max_period, cfg.window_tokens
        self.partner_visible = True        # False = Table 2 'w/o Partner' ablation

    def forward(self, x, pos, cache=None):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = rope(q, pos, self.max_period), rope(k, pos, self.max_period)
        if cache is not None:
            if cache.get("k") is not None:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
                kpos = torch.cat([cache["pos"], pos])
            else:
                kpos = pos
            if int(kpos[0]) <= int(pos[-1]) - self.window:  # sliding window of window_tokens
                keep = kpos > pos[-1] - self.window
                k, v, kpos = k[:, :, keep], v[:, :, keep], kpos[keep]
            cache.update(k=k, v=v, pos=kpos)
        else:
            kpos = pos
        allowed = (kpos[None, :] <= pos[:, None]) & (kpos[None, :] > pos[:, None] - self.window)
        if not self.partner_visible:
            # [A] w/o Partner: agent ([B]) positions never attend partner ([A]) positions, in any layer,
            # so no partner-motion information reaches the agent's predictions (Delta-User = 0 by design).
            agent_q = (pos % C.L_STEP) >= C.K + 1
            agent_k = (kpos % C.L_STEP) >= C.K + 1
            allowed = allowed & ~(agent_q[:, None] & ~agent_k[None, :])
        y, _ = attend(q, k, v, allowed)
        return self.out(y.transpose(1, 2).reshape(b, t, -1))


class CrossAttention(nn.Module):
    """Queries from motion tokens, keys/values from frozen speech states H_l [P §4.2]."""

    def __init__(self, cfg: C.TowerConfig):
        super().__init__()
        self.h, self.dc = cfg.cross_heads, cfg.cross_head_dim
        inner = cfg.cross_heads * cfg.cross_head_dim
        self.wq = nn.Linear(cfg.d_model, inner, bias=False)      # W_q in R^{d_m x h_c d_c}
        self.wk = nn.Linear(cfg.speech_dim, inner, bias=False)   # W_k in R^{d_s x h_c d_c}
        self.wv = nn.Linear(cfg.speech_dim, inner, bias=False)   # W_v in R^{d_s x h_c d_c}
        self.out = nn.Linear(inner, cfg.d_model, bias=False)     # [A] back to d_m
        self.max_period = cfg.rope_max_period
        self.window_frames = cfg.speech_window_frames or cfg.window_tokens // C.L_STEP
        self.use_rope = True
        self.last_weights = None

    def project(self, speech, s_pos):
        """speech [B, S, d_s], s_pos [S] -> rotated keys and values [B, h, S, d_c]."""
        b, s, _ = speech.shape
        k = self.wk(speech).view(b, s, self.h, self.dc).transpose(1, 2)
        v = self.wv(speech).view(b, s, self.h, self.dc).transpose(1, 2)
        if self.use_rope:
            k = rope(k, s_pos, self.max_period)                  # Eq. 3: k~_s = RoPE(K[s], s)
        return k, v

    def forward(self, x, q_frames, kv, s_pos, keep_weights=False):
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.h, self.dc).transpose(1, 2)
        if self.use_rope:
            q = rope(q, q_frames, self.max_period)               # Eq. 3: q~_t = RoPE(q_t, q_pos(t))
        k, v = kv
        # Eq. 4 causal mask s <= q_pos(t), limited to the speech window [P §4.2]
        allowed = (s_pos[None, :] <= q_frames[:, None]) & (s_pos[None, :] > q_frames[:, None] - self.window_frames)
        y, w = attend(q, k, v, allowed, keep_weights)
        if keep_weights:
            self.last_weights = w.detach()
        return self.out(y.transpose(1, 2).reshape(b, t, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: C.TowerConfig):
        super().__init__()
        hidden = (2 * int(cfg.ff_scale * cfg.d_model)) // 3
        self.linear_in = nn.Linear(cfg.d_model, 2 * hidden, bias=False)
        self.linear_out = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        gate, value = self.linear_in(x).chunk(2, dim=-1)
        return self.linear_out(F.silu(gate) * value)


class Block(nn.Module):
    def __init__(self, cfg: C.TowerConfig):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = (RMSNorm(cfg.d_model, cfg.norm_eps) for _ in range(3))
        self.self_attn, self.cross_attn, self.mlp = SelfAttention(cfg), CrossAttention(cfg), SwiGLU(cfg)
        self.use_self_attn = self.use_cross_attn = True

    def forward(self, x, pos, q_frames, kv, s_pos, cache=None, keep_weights=False):
        if self.use_self_attn:
            x = x + self.self_attn(self.norm1(x), pos, cache)
        if self.use_cross_attn:
            x = x + self.cross_attn(self.norm2(x), q_frames, kv, s_pos, keep_weights)
        return x + self.mlp(self.norm3(x))


class MotionTower(nn.Module):
    def __init__(self, cfg: C.TowerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(C.VOCAB, cfg.d_model)        # shared E in R^{V x d_m}, V = V_mot + 2
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.head = nn.Linear(cfg.d_model, C.VOCAB, bias=False)
        self.va_head = nn.Linear(cfg.d_model, 1)               # linear voice-activity head [P §4.3]
        self.apply(self._init)

    @staticmethod
    def _init(module):   # [A] GPT-style N(0, 0.02)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def ablate(self, name: str) -> "MotionTower":
        """Table 2 ablations: no_self_attn, no_partner, no_cross_attn, no_cross_rope.

        no_partner should be trained with supervise_speakers="B" (only the agent is predicted)."""
        if name not in ("no_self_attn", "no_partner", "no_cross_attn", "no_cross_rope"):
            raise ValueError(name)
        for block in self.blocks:
            if name == "no_partner":
                block.self_attn.partner_visible = False
            elif name == "no_self_attn":
                block.use_self_attn = False
            elif name == "no_cross_attn":
                block.use_cross_attn = False
            elif name == "no_cross_rope":
                block.cross_attn.use_rope = False
        return self

    def project_speech(self, speech: list[torch.Tensor], s_pos: torch.Tensor):
        """speech: L tensors [B, S, d_s] (one per PersonaPlex layer, paired one-to-one)."""
        assert len(speech) == len(self.blocks), "motion blocks pair one-to-one with speech layers"
        return [block.cross_attn.project(h, s_pos) for block, h in zip(self.blocks, speech)]

    def forward(self, tokens, pos, q_frames, speech_kv, s_pos, caches=None, keep_weights=False):
        """speech_kv: per-layer projected (k, v) from project_speech, or raw per-layer states [B, S, d_s]
        (projected here, so DDP sees the projection inside forward)."""
        if isinstance(speech_kv[0], torch.Tensor):
            speech_kv = self.project_speech(speech_kv, s_pos)
        x = self.embed(tokens)
        for i, block in enumerate(self.blocks):
            x = block(x, pos, q_frames, speech_kv[i], s_pos, None if caches is None else caches[i], keep_weights)
        x = self.norm(x)
        return self.head(x), self.va_head(x).squeeze(-1)


def masked_logits(logits: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
    """Band mask: out-of-band logits -> -inf before softmax [P §4.3]. band [..., ] part index or -1."""
    vocab = torch.arange(C.VOCAB, device=logits.device)
    lo = (band.clamp(min=0) * C.CODEBOOK_SIZE)[..., None]
    allowed = (vocab >= lo) & (vocab < lo + C.CODEBOOK_SIZE)
    return logits.masked_fill(~allowed, float("-inf"))


def loss_fn(logits, va_logits, labels, band, va_labels, beta: float):
    """L = L_CE (band-masked, supervised code positions) + beta * L_VA [P §4.3]."""
    supervised = labels >= 0
    masked = masked_logits(logits[supervised], band[supervised])
    ce = F.cross_entropy(masked, labels[supervised])
    valid = va_labels >= 0
    va = F.binary_cross_entropy_with_logits(va_logits[valid], va_labels[valid].float())
    accuracy = (masked.argmax(-1) == labels[supervised]).float().mean()
    return ce + beta * va, {"ce": float(ce.detach()), "va": float(va.detach()), "token_acc": float(accuracy)}
