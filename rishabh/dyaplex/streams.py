"""PersonaPlex input streams per participant (§4.1, App. D.1), computed once.

    python -m dyaplex build-streams --prepared P --mode real --device cuda    # Mimi + SentencePiece
    python -m dyaplex build-streams --prepared P --mode tiny                  # local stand-ins

Writes <P>/streams/<file_id>.npz:
  audio  int16 [8, F]   Mimi codes of the participant's 24 kHz audio at 12.5 Hz
  text   int32 [F]      dense inner monologue: subwords at word onsets, PAD/EPAD elsewhere [P D.1]
  voice  int16 [8, V]   ~10 s voice prompt (best own-speech window, -24 LUFS) [P D.1; A selection]
and <P>/streams/prompt.json (text-prompt token ids, mode, pins). Storage is ~200 bytes per second.
At training time the 17 streams for a role order are: text(agent), audio(agent), audio(partner).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import config as C
from . import speech as S
from .prepare import load_meta, split_pairs
from .vendor import PERSONAPLEX_CHECKPOINT, PINS


class RealFrontend:
    """Mimi + tokenizer from the pinned PersonaPlex checkpoint (no 7B LM needed here)."""

    def __init__(self, device: str = "cuda"):
        import sentencepiece
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders
        from moshi.models.lm import _iterate_audio, encode_from_sphn, load_audio, normalize_audio
        from moshi.offline import wrap_with_system_tags
        repo, revision = PERSONAPLEX_CHECKPOINT
        self.mimi = loaders.get_mimi(hf_hub_download(repo, loaders.MIMI_NAME, revision=revision), device)
        self.tok = sentencepiece.SentencePieceProcessor(
            model_file=hf_hub_download(repo, loaders.TEXT_TOKENIZER_NAME, revision=revision))
        self._iterate, self._encode, self._load, self._normalize = _iterate_audio, encode_from_sphn, load_audio, normalize_audio
        self.text_prompt = [int(t) for t in self.tok.encode(wrap_with_system_tags(C.REFERENCE["speech"].text_prompt))]
        self.sr = int(self.mimi.sample_rate)
        self.frame = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.mode = "real"

    @torch.no_grad()
    def codes(self, wav: str, start_s: float = 0.0, seconds: float | None = None, normalize: bool = False) -> np.ndarray:
        audio = self._load(wav, self.sr)                       # (1, T) at 24 kHz, as offline.py
        lo = int(start_s * self.sr)
        audio = audio[:, lo: None if seconds is None else lo + int(seconds * self.sr)]
        if normalize:
            audio = self._normalize(audio[0], self.sr, -24.0)[None]
        self.mimi.reset_streaming()
        with self.mimi.streaming(1):
            frames = [c[0, :8].cpu().numpy() for c in self._encode(self.mimi, self._iterate(audio, self.frame, pad=True))]
        return np.concatenate(frames, axis=1).astype(np.int16)

    def tokenize(self, text: str) -> list[int]:
        return [int(t) for t in self.tok.encode(text)]


class TinyFrontend:
    """Stand-ins used by the local smoke tests (band-energy codes, hashed text ids)."""

    def __init__(self, seed: int = 0):
        self.cfg = C.TINY["speech"]
        rng = np.random.default_rng(seed)
        self.text_prompt = [int(t) for t in rng.integers(4, self.cfg.text_card, 12)]
        self.mode = "tiny-standin"

    def codes(self, wav: str, start_s: float = 0.0, seconds: float | None = None, normalize: bool = False) -> np.ndarray:
        from .data import read_wav
        audio, rate = read_wav(Path(wav))
        audio = audio[int(start_s * rate): None if seconds is None else int((start_s + seconds) * rate)]
        return S.standin_audio_codes(audio, rate, int(len(audio) / rate * C.TOKEN_HZ), self.cfg.card).astype(np.int16)

    def tokenize(self, text: str) -> list[int]:
        return [4 + (sum(map(ord, text)) * 31 + i) % (self.cfg.text_card - 4) for i in range(max(1, len(text) // 4))]


def build(prepared: Path, mode: str = "real", device: str = "cuda", shard: str = "0/1", log=print) -> dict:
    """shard 'i/n': this process handles every n-th participant starting at i (run n in parallel)."""
    front = RealFrontend(device) if mode == "real" else TinyFrontend()
    out = prepared / "streams"
    out.mkdir(exist_ok=True)
    ids = sorted({pid for split in ("train", "val", "test") for _, a, b in split_pairs(prepared, split) for pid in (a, b)})
    i, n = map(int, shard.split("/"))
    ids = ids[i::n]
    done = 0
    for k, fid in enumerate(ids):
        target = out / f"{fid}.npz"
        if target.exists():
            continue
        meta = load_meta(prepared, fid)
        audio = front.codes(meta["wav"])
        text = S.inner_monologue([tuple(w) for w in meta["words"]], front.tokenize, audio.shape[1])
        voice = front.codes(meta["wav"], meta["voice_clip_start_s"], C.REFERENCE["speech"].voice_prompt_s, normalize=True)
        np.savez(target, audio=audio, text=text.astype(np.int32), voice=voice)
        done += 1
        if (k + 1) % 500 == 0:
            log(f"  streams {k + 1}/{len(ids)}")
    info = {"mode": front.mode, "text_prompt": front.text_prompt, "participants": len(ids), "built": done,
            "pins": {k: v[1] for k, v in PINS.items()}, "checkpoint": PERSONAPLEX_CHECKPOINT}
    (out / "prompt.json").write_text(json.dumps(info, indent=2) + "\n")
    return {k: v for k, v in info.items() if k != "text_prompt"}


def load(prepared: Path, file_id: str) -> dict:
    with np.load(prepared / "streams" / f"{file_id}.npz") as z:
        return {k: z[k] for k in z.files}


def prompt(prepared: Path) -> dict:
    return json.loads((prepared / "streams" / "prompt.json").read_text())
