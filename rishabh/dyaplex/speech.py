"""Frozen PersonaPlex speech tower: per-layer residual-stream taps (§3.1, §4.1, App. D.1).

Uses the pinned NVIDIA code (third_party/personaplex @ 3428dfd) unmodified:
  * LMModel / LMGen from moshi/models/lm.py, config from moshi/models/loaders.py
  * forward hooks on ``lm.transformer.layers[i]`` capture the post-block residual stream
    (before ``out_norm``) of all 32 layers [P §3.1]
Time base. LMGen.step(frame f) writes frame f at cache position f + delay and runs the
transformer on position f-1 (position 0 is the initial token; lm.py prepare_step_input).
``forward_train(tok[:, :, 1:])`` builds the identical input sequence, so

    H[s] := hidden state produced by the LMGen.step call that received frame s   (s >= 1)

is available causally at frame s in streaming and equals teacher-forced position s-1.
``SpeechConfig.state_offset`` shifts this mapping [A] (review: ±1 frame ambiguity).

Prompt [P D.1; I lm.py _step_voice_prompt/_step_audio_silence/_step_text_prompt, offline.py]:
  voice clip frames (agent audio = voice codes, text = 3, user = SINE) | 6 silence frames
  (agent = SILENCE, text = 3, user = SINE) | text prompt tokens (agent = SILENCE, user = SINE)
  | 6 silence frames | conversation (agent = B audio, user = A audio, text = inner monologue)
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

from . import config as C
from .vendor import use_personaplex

use_personaplex()
from moshi.models import loaders  # noqa: E402
from moshi.models.lm import SILENCE_TOKENS, SINE_TOKENS, LMGen, LMModel  # noqa: E402

TEXT_PAD, TEXT_EPAD = 3, 0      # [I loaders.py existing_text_padding_id=3; lm.py end_of_text_padding_id=0]
ZERO_TEXT = 3                   # [I lm.py LMGen.zero_text_code]


def build_lm(cfg: C.SpeechConfig, seed: int = 0) -> LMModel:
    """PersonaPlex LMModel with the published wiring; sizes from ``cfg`` (random weights)."""
    kwargs = dict(loaders._lm_kwargs)
    kwargs.update(dim=cfg.dim, num_heads=cfg.num_heads, num_layers=cfg.num_layers, context=cfg.context,
                  depformer_dim=cfg.depformer_dim, depformer_num_heads=cfg.depformer_num_heads,
                  depformer_num_layers=cfg.depformer_num_layers, text_card=cfg.text_card,
                  depformer_dim_feedforward=int(cfg.hidden_scale * cfg.depformer_dim))
    torch.manual_seed(seed)
    lm = LMModel(device="cpu", dtype=torch.float32, **kwargs)
    return lm.eval()


def load_real_lm(device: str = "cuda"):
    """GPU host only: the released checkpoint (gated; accept the license on Hugging Face first)."""
    from huggingface_hub import hf_hub_download
    from .vendor import PERSONAPLEX_CHECKPOINT
    repo, revision = PERSONAPLEX_CHECKPOINT
    weight = hf_hub_download(repo, loaders.MOSHI_NAME, revision=revision)
    mimi = loaders.get_mimi(hf_hub_download(repo, loaders.MIMI_NAME, revision=revision), device)
    lm = loaders.get_moshi_lm(weight, device=device)
    return lm.eval(), mimi


class LayerTap:
    """Forward hooks on every main-transformer block; records [T, d] per layer per call."""

    def __init__(self, lm: LMModel, keep_dtype: bool = False):
        self.layers = list(lm.transformer.layers)
        self.calls: list[list[torch.Tensor]] = []
        self.handles = []
        self.keep_dtype = keep_dtype

    def __enter__(self):
        self.calls = []
        current: list[torch.Tensor] = []

        def make(index):
            def hook(_module, _inputs, output):
                if index == 0:
                    current.clear()
                out = output.detach()[0]
                current.append(out.clone() if self.keep_dtype else out.float().clone())
                if index == len(self.layers) - 1:
                    self.calls.append(list(current))
            return hook

        self.handles = [layer.register_forward_hook(make(i)) for i, layer in enumerate(self.layers)]
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def stacked(self) -> torch.Tensor:
        """[L, T_total, d] concatenating calls in time order."""
        return torch.stack([torch.cat([call[l] for call in self.calls], dim=0) for l in range(len(self.layers))])


@dataclass
class Streams:
    """Per-frame tokens for the 17 PersonaPlex streams: [17, N] = text, agent 8, user 8."""
    tokens: np.ndarray
    prompt_frames: int


def build_streams(agent_codes: np.ndarray, user_codes: np.ndarray, text: np.ndarray, voice_codes: np.ndarray,
                  text_prompt_tokens: list[int], cfg: C.SpeechConfig) -> Streams:
    """agent/user codes [8, F]; text [F]; voice_codes [8, P_v]."""
    silence = np.asarray(SILENCE_TOKENS)[:, None]
    sine = np.asarray(SINE_TOKENS)[:, None]
    cols = []

    def frame(text_id, agent, user):
        cols.append(np.concatenate([[text_id], agent.reshape(-1), user.reshape(-1)]))

    for f in range(voice_codes.shape[1]):
        frame(ZERO_TEXT, voice_codes[:, f], sine)
    for _ in range(cfg.silence_frames):
        frame(ZERO_TEXT, silence, sine)
    for tok in text_prompt_tokens:
        frame(tok, silence, sine)
    for _ in range(cfg.silence_frames):
        frame(ZERO_TEXT, silence, sine)
    prompt = len(cols)
    for f in range(agent_codes.shape[1]):
        frame(int(text[f]), agent_codes[:, f], user_codes[:, f])
    return Streams(np.stack(cols, axis=1).astype(np.int64), prompt)


@torch.no_grad()
def teacher_forced_states(lm: LMModel, streams: Streams) -> torch.Tensor:
    """One forward_train() pass [P D.1]. Returns [L, N, d] with H[:, s] for stream frame s
    (H[:, 0] = 0: frame 0 is consumed as the initial step)."""
    tok = torch.tensor(streams.tokens[None])
    with LayerTap(lm) as tap:
        lm.forward_train(tok[:, :, 1:])
    hidden = tap.stacked()                                   # [L, N-1, d] positions 0..N-2
    zero = torch.zeros(hidden.shape[0], 1, hidden.shape[2])
    return torch.cat([zero, hidden], dim=1)


@torch.no_grad()
def teacher_forced_fast(lm: LMModel, tokens: torch.Tensor) -> torch.Tensor:
    """Same states as teacher_forced_states without running the depth transformer.

    tokens: [1, 17, N] long on the LM's device. Builds exactly forward_train's input
    (initial token + delayed codes, lm.py:531) and runs only embeddings + main transformer.
    Returns [L, N, d] in the LM's dtype (H[:, 0] = 0)."""
    from moshi.models.lm import _delay_sequence
    codes = tokens[:, :, 1:]
    initial = lm._get_initial_token().expand(codes.shape[0], -1, -1)
    delayed = torch.cat([initial, _delay_sequence(lm.delays, codes, initial)], dim=2)
    with LayerTap(lm, keep_dtype=True) as tap:
        lm.forward_codes(delayed[:, :, :-1])
    hidden = torch.stack([tap.calls[0][l] for l in range(len(tap.layers))])        # [L, N-1, d]
    return torch.cat([torch.zeros_like(hidden[:, :1]), hidden], dim=1)


@torch.no_grad()
def streaming_states(lm: LMModel, streams: Streams, seed: int = 0) -> torch.Tensor:
    """LMGen.step with every stream forced (the live loop with known inputs). Returns [L, N, d]."""
    torch.manual_seed(seed)
    gen = LMGen(lm, device="cpu")
    tok = torch.tensor(streams.tokens[None])
    out = []
    with gen.streaming(1), LayerTap(lm) as tap:
        for f in range(tok.shape[2]):
            before = len(tap.calls)
            gen.step(input_tokens=tok[:, 9:17, f:f + 1], moshi_tokens=tok[:, 1:9, f:f + 1],
                     text_token=tok[:, 0, f])
            if len(tap.calls) == before:          # frame 0: initial step, no transformer call
                out.append(torch.zeros(len(tap.layers), lm.dim))
            else:
                out.append(torch.stack([layer[-1] for layer in tap.calls[-1]]))
    return torch.stack(out, dim=1)


@torch.no_grad()
def sampled_tokens(lm: LMModel, user: np.ndarray, frames: int, seed: int, tap: bool) -> np.ndarray:
    """Free-running generation of agent text+audio from user audio; used to prove the taps
    do not change PersonaPlex's outputs."""
    torch.manual_seed(seed)
    gen = LMGen(lm, device="cpu")
    outs = []
    ctx = LayerTap(lm) if tap else _null()
    with gen.streaming(1), ctx:
        for f in range(frames):
            out = gen.step(input_tokens=torch.tensor(user[None, :, f:f + 1]))
            if out is not None:
                outs.append(out[0, :, 0].numpy())
    return np.stack(outs) if outs else np.zeros((0, 17))


@contextmanager
def _null():
    yield None


def conversation_states(states: torch.Tensor, streams: Streams, offset: int = 0) -> torch.Tensor:
    """Slice H for conversation frames j = 0..F-1 -> [L, F, d] (stream frame = prompt + j + offset)."""
    start = streams.prompt_frames + offset
    return states[:, start:start + streams.tokens.shape[1] - streams.prompt_frames]


# --------------------------------------------------------------------------- #
# Local stand-ins (no Mimi weights / tokenizer on this machine)
# --------------------------------------------------------------------------- #
def standin_audio_codes(audio: np.ndarray, rate: int, frames: int, card: int = 2048) -> np.ndarray:
    """STAND-IN for Mimi [P §3.1]: 8 log band energies per 80 ms frame, quantized to `card` bins.

    Keeps the Mimi interface ([8, F] ints at 12.5 Hz) and is speech-correlated, so the motion
    tower can learn something from it; it is not Mimi and carries no PersonaPlex semantics.
    """
    hop = int(round(rate / C.TOKEN_HZ))
    need = frames * hop
    audio = np.pad(audio[:need], (0, max(0, need - len(audio))))
    spec = np.abs(np.fft.rfft(audio.reshape(frames, hop) * np.hanning(hop), axis=1))
    edges = np.geomspace(60, rate / 2, 9)
    freqs = np.fft.rfftfreq(hop, 1 / rate)
    bands = np.stack([spec[:, (freqs >= lo) & (freqs < hi)].mean(1) for lo, hi in zip(edges[:-1], edges[1:])])
    log = np.log(bands + 1e-6)
    return np.clip(((log + 12.0) / 14.0 * card).astype(np.int64), 0, card - 1)


def inner_monologue(words: list[tuple[float, str]], tokenize, frames: int) -> np.ndarray:
    """Dense text stream [P D.1; I Moshi format]: a word's subword tokens start at the word's
    onset frame; other frames hold PAD, and the frame before a word's first token holds EPAD [A
    exact EPAD placement]. ``words``: (onset seconds, word)."""
    text = np.full(frames, TEXT_PAD, dtype=np.int64)
    cursor = 0
    for onset, word in sorted(words):
        start = max(int(round(onset * C.TOKEN_HZ)), cursor)
        ids = tokenize(" " + word)
        if start > 0 and text[start - 1] == TEXT_PAD:
            text[start - 1] = TEXT_EPAD
        for k, tok in enumerate(ids):
            if start + k < frames:
                text[start + k] = tok
        cursor = start + len(ids)
    return text
