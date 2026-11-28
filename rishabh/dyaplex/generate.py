"""Streaming sampler (§4.4, Eq. 5) and teacher-forced training windows (§4.3, App. D.2).

Agent-only mode: observed partner codes fill the [A] slots, the model samples only [B]
codes. Both-speaker mode: the model samples [A] and [B] codes. Speaker tags are inserted
deterministically. Sampling: band mask, temperature tau, optional top-k (X-1: §4.4 says
K_top = 200, D.2 says no truncation; both supported). Draws use one seeded uniform per
token so cached, uncached and chunked runs are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
import torch

from . import config as C
from . import layout
from .tower import MotionTower, loss_fn, masked_logits


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class Example:
    a_codes: np.ndarray      # [F, 22] global ids
    b_codes: np.ndarray
    speech: np.ndarray       # [L, F, d_s] H_l for conversation frames (agent = B)
    vad_a: np.ndarray        # [F]
    vad_b: np.ndarray
    name: str = ""


def window_batch(examples: list[Example], starts: list[tuple[int, int]], frames: int, cfg: C.TowerConfig, dev):
    tokens, labels, bands, vas, speech = [], [], [], [], []
    for i, f0 in starts:
        ex = examples[i]
        seq = layout.interleave(ex.a_codes[f0:f0 + frames], ex.b_codes[f0:f0 + frames])
        lab, band = layout.targets(seq, cfg.supervise, cfg.supervise_speakers)
        tokens.append(seq[:-1])
        labels.append(lab)
        bands.append(band)
        vas.append(layout.va_labels(len(seq) - 1, ex.vad_a, ex.vad_b, f0))
        speech.append(ex.speech[:, f0:f0 + frames])
    t = lambda x: torch.tensor(np.stack(x), device=dev)
    length = len(tokens[0])
    pos = torch.arange(length, device=dev)
    sp = torch.tensor(np.stack(speech, axis=1), device=dev, dtype=torch.float32)   # [L, B, F, d]
    return t(tokens), t(labels), t(bands), t(vas), pos, pos // C.L_STEP, [sp[l] for l in range(sp.shape[0])], \
        torch.arange(frames, device=dev)


def all_windows(examples: list[Example], frames: int, stride: int) -> list[tuple[int, int]]:
    return [(i, f0) for i, ex in enumerate(examples) for f0 in range(0, len(ex.a_codes) - frames + 1, stride)]


def train_tower(model: MotionTower, examples: list[Example], cfg: C.TowerConfig, tcfg: C.TowerTrainConfig,
                dev, seed: int = 0, log=print, eval_every: int = 200, target_acc: float | None = None):
    """AdamW(lr 3e-4, betas .9/.95, wd 0.1), clip 1.0 [P D.2]; schedule/warmup [U] -> constant."""
    frames = layout.window_frames(cfg.window_tokens)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, betas=tcfg.betas, weight_decay=tcfg.weight_decay)
    windows = all_windows(examples, frames, 1)
    history, began = [], time.time()
    for it in range(1, tcfg.iterations + 1):
        picks = [windows[k] for k in rng.integers(0, len(windows), tcfg.batch_size)]
        tokens, labels, bands, vas, pos, qf, speech, s_pos = window_batch(examples, picks, frames, cfg, dev)
        kv = model.project_speech(speech, s_pos)
        logits, va = model(tokens, pos, qf, kv, s_pos)
        loss, stats = loss_fn(logits, va, labels, bands, vas, cfg.beta_va)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()
        if it % eval_every == 0 or it == tcfg.iterations:
            acc = evaluate(model, examples, cfg, dev)
            history.append({"iter": it, "loss": float(loss), **stats, "eval_token_acc": acc,
                            "seconds": round(time.time() - began, 1)})
            log(f"  tower it {it}: loss {float(loss):.3f} ce {stats['ce']:.3f} acc(train batch) "
                f"{stats['token_acc']:.3f} teacher-forced acc {acc:.3f}")
            model.train()
            if target_acc is not None and acc >= target_acc:
                break
    model.eval()
    return history


@torch.no_grad()
def evaluate(model: MotionTower, examples: list[Example], cfg: C.TowerConfig, dev) -> float:
    """Teacher-forced next-token accuracy over every non-overlapping window."""
    model.eval()
    frames = layout.window_frames(cfg.window_tokens)
    windows = all_windows(examples, frames, frames)
    correct = total = 0
    for k in range(0, len(windows), 8):
        tokens, labels, bands, vas, pos, qf, speech, s_pos = window_batch(examples, windows[k:k + 8], frames, cfg, dev)
        logits, _ = model(tokens, pos, qf, model.project_speech(speech, s_pos), s_pos)
        sup = labels >= 0
        pred = masked_logits(logits[sup], bands[sup]).argmax(-1)
        correct += int((pred == labels[sup]).sum())
        total += int(sup.sum())
    return correct / max(total, 1)


# --------------------------------------------------------------------------- #
# Streaming generation
# --------------------------------------------------------------------------- #
@dataclass
class Session:
    """Per-conversation state: flat tokens, KV caches, sampler RNG, frame counter."""
    model: MotionTower
    speech: list[torch.Tensor]             # L x [1, F_avail, d_s] (grows in live use)
    sample: C.SampleConfig
    seed: int = 0
    use_cache: bool = True
    tokens: list[int] = field(default_factory=list)
    frame: int = 0
    log: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.rng = torch.Generator().manual_seed(self.seed)
        self.caches = [dict() for _ in self.model.blocks] if self.use_cache else None
        self.fed = 0                     # tokens already pushed through the cached model
        self.window = self.model.cfg.window_tokens
        self.timing = {"tower_s": 0.0, "tokens_sampled": 0}
        self.device = next(self.model.parameters()).device
        self.speech = [h.to(self.device) for h in self.speech]
        self._held: list[int] = []

    def _speech_kv(self):
        frames = self.frame + 1
        s_pos = torch.arange(frames, device=self.device)
        return self.model.project_speech([h[:, :frames] for h in self.speech], s_pos), s_pos

    @torch.no_grad()
    def _push(self, new: list[int]) -> torch.Tensor:
        """Append tokens (plus any held token) and return logits at the last appended position."""
        new = self._held + new
        self._held = []
        start = time.perf_counter()
        self.tokens.extend(new)
        kv, s_pos = self._speech_kv()
        if self.use_cache:
            pos = torch.arange(self.fed, len(self.tokens), device=self.device)
            x = torch.tensor([self.tokens[self.fed:]], device=self.device)
            self.fed = len(self.tokens)
        else:
            lo = max(0, len(self.tokens) - self.window)
            pos = torch.arange(lo, len(self.tokens), device=self.device)
            x = torch.tensor([self.tokens[lo:]], device=self.device)
        logits, _ = self.model(x, pos, pos // C.L_STEP, kv, s_pos, self.caches)
        out = logits[0, -1].float().cpu()
        self.timing["tower_s"] += time.perf_counter() - start
        return out

    @torch.no_grad()
    def prefill(self, a_codes: np.ndarray, b_codes: np.ndarray) -> None:
        """Teacher-force whole frames (e.g. to fill the context before timing steady-state frames)."""
        for a, b in zip(a_codes, b_codes):
            self._push([C.TAG_A] + [int(c) for c in a] + [C.TAG_B] + [int(c) for c in b])
            self.frame += 1

    def _sample(self, logits: torch.Tensor, slot: int) -> int:
        band = torch.tensor(layout.SLOT_PART[slot])
        masked = masked_logits(logits[None], band[None])[0] / self.sample.temperature
        if self.sample.top_k is not None:
            kth = torch.topk(masked, self.sample.top_k).values[-1]
            masked = masked.masked_fill(masked < kth, float("-inf"))
        probs = torch.softmax(masked.double(), dim=-1)
        u = torch.rand((), generator=self.rng, dtype=torch.float64)
        idx = int(torch.searchsorted(torch.cumsum(probs, 0), u * probs.sum()).clamp(max=C.VOCAB - 1))
        self.log.append({"frame": self.frame, "slot": slot, "token": idx, "logits": logits.detach().clone()})
        self.timing["tokens_sampled"] += 1
        return idx

    @torch.no_grad()
    def step_frame(self, partner_codes: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """One 80 ms frame. partner_codes [22] (agent-only mode) or None (both-speaker mode).

        The frame's last sampled code is held and fed together with the next frame's [A] tag.
        """
        if partner_codes is not None:
            a = [int(c) for c in partner_codes]
            logits = self._push([C.TAG_A] + a + [C.TAG_B])          # deterministic tags + observed partner
        else:
            a = []
            logits = self._push([C.TAG_A])
            for slot in range(C.K):
                tok = self._sample(logits, slot)
                a.append(tok)
                logits = self._push([tok] + ([C.TAG_B] if slot == C.K - 1 else []))
        b = []
        for slot in range(C.K):
            tok = self._sample(logits, slot)
            b.append(tok)
            if slot < C.K - 1:
                logits = self._push([tok])
            else:
                self._held = [tok]
        self.frame += 1
        return np.array(a), np.array(b)


def generate(model: MotionTower, speech: np.ndarray, frames: int, sample: C.SampleConfig, seed: int,
             partner: np.ndarray | None = None, use_cache: bool = True):
    """speech [L, F, d_s]; partner [F, 22] (agent-only) or None (both-speaker). Returns A, B codes.

    Runs single-threaded: per-token steps are tiny and thread hand-off dominates (5x slower at 4 threads)."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _generate(model, speech, frames, sample, seed, partner, use_cache)
    finally:
        torch.set_num_threads(threads)


def _generate(model, speech, frames, sample, seed, partner, use_cache):
    session = Session(model, [torch.tensor(speech[l][None], dtype=torch.float32) for l in range(speech.shape[0])],
                      sample, seed, use_cache)
    a_out, b_out = [], []
    for f in range(frames):
        a, b = session.step_frame(None if partner is None else partner[f])
        a_out.append(a)
        b_out.append(b)
    return np.array(a_out), np.array(b_out), session
