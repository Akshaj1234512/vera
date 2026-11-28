"""Motion-tower training data with online PersonaPlex states (§4.1-4.3, App. D.2).

Caching all 32 layers for the corpus would take ~40 PB (review X-3), so states are computed on
the fly: per micro-batch, one (pair, role order, context chunk) is drawn, the frozen speech tower
runs once over [prompt + chunk] (only up to the farthest window end; it is causal), and
`micro_batch` motion windows of window_tokens/46 frames are cut from that chunk.
Labels are masked where the owning speaker's frame is invalid (body: smplh:is_valid; face codes:
zero-face frames) [P C.1, C.2].
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from . import config as C
from . import data as D
from . import layout
from . import speech as S
from . import streams as ST
from .prepare import load_meta, load_valid, split_pairs


@dataclass
class Item:
    pair: str
    user: str      # partner A
    agent: str     # agent B
    frames: int    # usable 12.5 Hz frames
    chunks: list   # [(c0, c1)] context chunks


class SpeechTower:
    """Frozen PersonaPlex wrapper: [prompt + chunk] -> per-layer states for the chunk's frames."""

    def __init__(self, lm, speech_cfg: C.SpeechConfig, text_prompt: list[int]):
        self.lm, self.cfg, self.text_prompt = lm, speech_cfg, text_prompt
        self.device = next(lm.parameters()).device

    def prompt_frames(self, voice: np.ndarray) -> int:
        return voice.shape[1] + 2 * self.cfg.silence_frames + len(self.text_prompt)

    @torch.no_grad()
    def states(self, agent: dict, user: dict, c0: int, c1: int) -> torch.Tensor:
        """[L, c1 - c0, d] for conversation frames c0..c1-1 of this role order."""
        st = S.build_streams(agent["audio"][:, c0:c1].astype(np.int64), user["audio"][:, c0:c1].astype(np.int64),
                             agent["text"][c0:c1].astype(np.int64), agent["voice"].astype(np.int64), self.text_prompt,
                             self.cfg)
        tokens = torch.tensor(st.tokens[None], device=self.device)
        hidden = S.teacher_forced_fast(self.lm, tokens)
        start = st.prompt_frames + self.cfg.state_offset
        return hidden[:, start:start + (c1 - c0)]


class PairData:
    def __init__(self, prepared: Path, split: str, tokens_dir: Path, speech: SpeechTower, window_frames: int,
                 limit: int | None = None):
        self.prepared, self.tokens_dir, self.speech, self.window = prepared, tokens_dir, speech, window_frames
        self.items: list[Item] = []
        context = int(speech.lm.context)
        for key, a, b in split_pairs(prepared, split)[:limit]:
            for user, agent in ((a, b), (b, a)):                    # both role orders [A]
                frames = min(self._tokens(a).shape[0], self._tokens(b).shape[0],
                             self._streams(a)["audio"].shape[1], self._streams(b)["audio"].shape[1])
                chunk = context - speech.prompt_frames(self._streams(agent)["voice"])
                chunks = [(c0, min(frames, c0 + chunk)) for c0 in range(0, frames, chunk)]
                chunks = [c for c in chunks if c[1] - c[0] >= window_frames]
                if chunks:
                    self.items.append(Item(key, user, agent, frames, chunks))
        if not self.items:
            raise ValueError(f"no {split} pairs with >= {window_frames} frames")
        self.weights = np.array([sum(c1 - c0 - window_frames + 1 for c0, c1 in it.chunks) for it in self.items], float)
        self.weights /= self.weights.sum()

    @lru_cache(maxsize=4096)
    def _tokens(self, fid: str) -> np.ndarray:
        return np.load(self.tokens_dir / f"{fid}.npy").astype(np.int64)

    @lru_cache(maxsize=4096)
    def _streams(self, fid: str) -> dict:
        return ST.load(self.prepared, fid)

    @lru_cache(maxsize=4096)
    def _frame_info(self, fid: str) -> dict:
        meta = load_meta(self.prepared, fid)
        valid = load_valid(self.prepared, fid)
        t = len(self._tokens(fid))
        body = valid["body"][: 2 * t].reshape(t, 2).all(1)          # token frame valid iff both 25 fps frames are
        face = valid["face"][: 2 * t].reshape(t, 2).all(1)
        return {"vad": D.vad_labels([tuple(v) for v in meta["vad"]], t), "body": body, "face": face}

    def sample(self, rng, count: int) -> dict:
        it = self.items[rng.choice(len(self.items), p=self.weights)]
        lengths = np.array([c1 - c0 - self.window + 1 for c0, c1 in it.chunks], float)
        c0, c1 = it.chunks[rng.choice(len(it.chunks), p=lengths / lengths.sum())]
        starts = np.sort(rng.integers(c0, c1 - self.window + 1, size=count))
        return self.windows(it, c0, [int(s) for s in starts])

    def windows(self, it: Item, c0: int, starts: list[int]) -> dict:
        end = max(starts) + self.window
        hidden = self.speech.states(self._streams(it.agent), self._streams(it.user), c0, end)
        speech = torch.stack([hidden[:, s - c0: s - c0 + self.window] for s in starts], dim=1)   # [L, B, W, d]
        info_a, info_b = self._frame_info(it.user), self._frame_info(it.agent)
        return {"item": it, "starts": starts, "speech": speech,
                "a": np.stack([self._tokens(it.user)[s:s + self.window] for s in starts]),
                "b": np.stack([self._tokens(it.agent)[s:s + self.window] for s in starts]),
                "info_a": [{k: v[s:s + self.window] for k, v in info_a.items()} for s in starts],
                "info_b": [{k: v[s:s + self.window] for k, v in info_b.items()} for s in starts]}

    def fixed_windows(self, count: int, seed: int = 0) -> list[tuple[Item, int, list[int]]]:
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(count):
            it = self.items[rng.choice(len(self.items), p=self.weights)]
            c0, c1 = it.chunks[rng.integers(len(it.chunks))]
            out.append((it, c0, [int(rng.integers(c0, c1 - self.window + 1))]))
        return out


def mask_invalid(labels: np.ndarray, info_a: dict, info_b: dict) -> np.ndarray:
    """-100 for targets whose owning speaker's frame is invalid (body codes: body validity; face codes:
    face validity)."""
    length = len(labels) + 1
    speaker, is_code, slot = layout.token_role(length)
    frame = layout.q_pos(length)
    sp, code, sl, fr = speaker[1:], is_code[1:], slot[1:], frame[1:]
    body_ok = np.where(sp == 0, info_a["body"][fr], info_b["body"][fr])
    face_ok = np.where(sp == 0, info_a["face"][fr], info_b["face"][fr])
    ok = np.where(sl >= C.BODY_CODES, face_ok, body_ok)
    return np.where(code & ~ok, -100, labels)


def make_batch(sample: dict, cfg: C.TowerConfig, dev) -> tuple:
    tokens, labels, bands, vas = [], [], [], []
    for a, b, ia, ib in zip(sample["a"], sample["b"], sample["info_a"], sample["info_b"]):
        seq = layout.interleave(a, b)
        lab, band = layout.targets(seq, cfg.supervise, cfg.supervise_speakers)
        tokens.append(seq[:-1])
        labels.append(mask_invalid(lab, ia, ib))
        bands.append(band)
        vas.append(layout.va_labels(len(seq) - 1, ia["vad"], ib["vad"]))
    t = lambda x: torch.tensor(np.stack(x), device=dev)
    length = len(tokens[0])
    pos = torch.arange(length, device=dev)
    speech = sample["speech"].to(dev)
    frames = speech.shape[2]
    return t(tokens), t(labels), t(bands), t(vas), pos, pos // C.L_STEP, \
        [speech[l].float() for l in range(speech.shape[0])], torch.arange(frames, device=dev)
