"""Precompute PersonaPlex per-layer speech states for Seamless pairs (§4.1, App. D.1).

    python -m dyaplex extract-states --real ...   # GPU host: PersonaPlex-7B + Mimi + SentencePiece
    python -m dyaplex extract-states --tiny ...   # local: same code path with the e2e stand-ins

For each pair and each role order (agent B, partner A) [A both orders]:
  agent audio  -> Mimi codes (streams 1-8)      user audio -> Mimi codes (streams 9-16)
  agent words  -> dense inner monologue (stream 0): subwords at word onsets, PAD/EPAD [P D.1]
  voice prompt -> the agent's own ~10 s clip with the most agent speech and the least partner
                  speech [A selection; P D.1 says only "clean ~10 s sample"], normalized to -24 LUFS
                  exactly as LMGen.load_voice_prompt does, Mimi-tokenized
  text prompt  -> "<system> You enjoy having a good conversation. <system>" [P D.1]
Clips longer than the 3000-frame context minus the prompt are split into consecutive
chunks, each with the prompt prepended [A]; states are written per conversation frame.
Output per role order: <pair>__<agent>.npy float16 [L, F, d] + .json (time base, pins, chunks).
Storage: 32 x 4096 x 2 bytes = 256 KiB per frame (~11.5 GiB per hour of conversation per role order).
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch

from . import config as C
from . import data as D
from . import speech as S
from .vendor import PERSONAPLEX_CHECKPOINT, PINS

BASE = D.BASE


def fetch_transcript(folder: Path, label: str, split: str) -> Path:
    from speech_ai.sources import fetch
    target = folder / "transcript.jsonl"
    if not target.is_file():
        fetch(f"{BASE}/{label}/{split}/metadata/transcript/{folder.name}.jsonl", target)
    return target


def fetch_audio_48k(folder: Path, label: str, split: str) -> Path:
    """Original 48 kHz WAV (the local copies are 16 kHz); ~10 MB per participant-minute."""
    from speech_ai.sources import fetch
    target = folder / "audio-48k.wav"
    if not target.is_file():
        fetch(f"{BASE}/{label}/{split}/audio/{folder.name}.wav", target)
    return target


def words(transcript: Path) -> list[tuple[float, str]]:
    out = []
    for line in transcript.read_text().splitlines():
        if line.strip():
            for w in json.loads(line).get("words", []):
                if "start" in w and w.get("word", "").strip():
                    out.append((float(w["start"]), w["word"].strip()))
    return out


def choose_voice_clip(vad_agent, vad_partner, duration_s: float, clip_s: float = 10.0, step_s: float = 0.5) -> float:
    """Start time of the ~10 s window with the most agent speech minus partner speech [A]."""
    grid = np.arange(0, max(duration_s - clip_s, 0) + 1e-9, step_s)
    hz = 10.0
    agent = D.vad_labels(vad_agent, int(duration_s * hz) + 1, hz)
    partner = D.vad_labels(vad_partner, int(duration_s * hz) + 1, hz)
    scores = [agent[int(s * hz): int((s + clip_s) * hz)].mean() - partner[int(s * hz): int((s + clip_s) * hz)].mean()
              for s in grid]
    return float(grid[int(np.argmax(scores))]) if len(grid) else 0.0


class RealTower:
    """PersonaPlex-7B, Mimi and the SentencePiece tokenizer at the pinned checkpoint (CUDA host)."""

    def __init__(self, device: str = "cuda"):
        import sentencepiece
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders
        from moshi.models.lm import _iterate_audio, encode_from_sphn, load_audio, normalize_audio
        from moshi.offline import wrap_with_system_tags
        repo, revision = PERSONAPLEX_CHECKPOINT
        self.lm, self.mimi = S.load_real_lm(device)
        self.tok = sentencepiece.SentencePieceProcessor(
            model_file=hf_hub_download(repo, loaders.TEXT_TOKENIZER_NAME, revision=revision))
        self._iterate, self._encode, self._load, self._normalize = _iterate_audio, encode_from_sphn, load_audio, normalize_audio
        self.text_prompt = self.tok.encode(wrap_with_system_tags(C.REFERENCE["speech"].text_prompt))
        self.sr = int(self.mimi.sample_rate)
        self.frame = int(self.mimi.sample_rate / self.mimi.frame_rate)

    @torch.no_grad()
    def codes(self, wav: Path, start_s: float = 0.0, seconds: float | None = None, normalize: bool = False) -> np.ndarray:
        audio = self._load(str(wav), self.sr)
        lo = int(start_s * self.sr)
        audio = audio[:, lo: None if seconds is None else lo + int(seconds * self.sr)]
        if normalize:
            audio = self._normalize(audio[0], self.sr, -24.0)[None]   # as LMGen.load_voice_prompt
        self.mimi.reset_streaming()
        with self.mimi.streaming(1):
            frames = [c[0, :8].cpu().numpy() for c in self._encode(self.mimi, self._iterate(audio, self.frame, pad=True))]
        return np.concatenate(frames, axis=1)

    def tokenize(self, text: str) -> list[int]:
        return self.tok.encode(text)


class TinyTower:
    """Local stand-ins with the same interface (see e2e.py / DEVIATIONS.md)."""

    def __init__(self, seed: int = 0):
        self.cfg = C.TINY["speech"]
        self.lm = S.build_lm(self.cfg, seed)
        rng = np.random.default_rng(seed)
        self.text_prompt = [int(t) for t in rng.integers(4, self.cfg.text_card, 12)]
        self.sr = 16000

    def codes(self, wav: Path, start_s: float = 0.0, seconds: float | None = None, normalize: bool = False) -> np.ndarray:
        audio, rate = D.read_wav(wav)
        audio = audio[int(start_s * rate): None if seconds is None else int((start_s + seconds) * rate)]
        return S.standin_audio_codes(audio, rate, int(len(audio) / rate * C.TOKEN_HZ), self.cfg.card)

    def tokenize(self, text: str) -> list[int]:
        return [4 + (sum(map(ord, text)) * 31 + i) % (self.cfg.text_card - 4) for i in range(max(1, len(text) // 4))]


def extract_pair(tower, folder_a: Path, folder_b: Path, wav: dict, out: Path, context: int,
                 silence_frames: int, log=print) -> list[dict]:
    """States for both role orders of one pair."""
    people = {f.name: f for f in (folder_a, folder_b)}
    vad = {name: [tuple(map(float, (json.loads(l)["start"], json.loads(l)["end"])))
                  for l in (f / "vad.jsonl").read_text().splitlines() if l.strip()] if (f / "vad.jsonl").exists() else []
           for name, f in people.items()}
    codes = {name: tower.codes(wav[name]) for name in people}
    frames = min(c.shape[1] for c in codes.values())
    results = []
    for user, agent in ((folder_a.name, folder_b.name), (folder_b.name, folder_a.name)):
        began = time.time()
        transcript = people[agent] / "transcript.jsonl"
        text = S.inner_monologue(words(transcript) if transcript.exists() else [], tower.tokenize, frames)
        duration = frames / C.TOKEN_HZ
        start = choose_voice_clip(vad[agent], vad[user], duration)
        voice = tower.codes(wav[agent], start, 10.0, normalize=True)
        cfg = C.REFERENCE["speech"]
        prompt_len = voice.shape[1] + 2 * silence_frames + len(tower.text_prompt)
        chunk = context - prompt_len
        if chunk <= 0:
            raise ValueError("prompt longer than the context")
        parts, layout = [], []
        for c0 in range(0, frames, chunk):
            c1 = min(frames, c0 + chunk)
            st = S.build_streams(codes[agent][:, c0:c1], codes[user][:, c0:c1], text[c0:c1], voice,
                                 tower.text_prompt, cfg)
            states = S.teacher_forced_states(tower.lm, st)
            parts.append(S.conversation_states(states, st, cfg.state_offset).half().cpu().numpy())
            layout.append({"frames": [c0, c1], "prompt_frames": st.prompt_frames})
        states = np.concatenate(parts, axis=1)
        name = f"{folder_a.name.rsplit('_', 1)[0]}__{agent}"
        np.save(out / f"{name}.npy", states)
        meta = {"pair": folder_a.name.rsplit("_", 1)[0], "agent": agent, "partner": user,
                "shape": list(states.shape), "dtype": "float16", "token_hz": C.TOKEN_HZ,
                "time_base": "states[:, j] = H for conversation frame j (stream frame prompt + j, see speech.py)",
                "voice_clip_start_s": start, "chunks": layout, "text_tokens_nonpad": int((text != S.TEXT_PAD).sum()),
                "pins": {k: v[1] for k, v in PINS.items()}, "checkpoint": PERSONAPLEX_CHECKPOINT,
                "seconds": round(time.time() - began, 2)}
        (out / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n")
        log(f"  {name}: {states.shape} in {meta['seconds']} s ({len(layout)} chunk(s))")
        results.append(meta)
    return results


def run(pairs_dir: Path, out: Path, n_pairs: int, real: bool, label: str = "naturalistic", split: str = "dev",
        device: str = "cuda", log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    tower = RealTower(device) if real else TinyTower()
    context = int(tower.lm.context)
    rows = []
    for key, a, b in D.complete_pairs(pairs_dir)[:n_pairs]:
        fa, fb = pairs_dir / a, pairs_dir / b
        for f in (fa, fb):
            fetch_transcript(f, label, split)
        wav = {f.name: (fetch_audio_48k(f, label, split) if real else f / "audio-16k.wav") for f in (fa, fb)}
        rows += extract_pair(tower, fa, fb, wav, out, context, C.REFERENCE["speech"].silence_frames, log)
    frames = sum(r["shape"][1] for r in rows)
    per_frame = rows[0]["shape"][0] * rows[0]["shape"][2] * 2 if rows else 0
    summary = {"mode": "real" if real else "tiny-standin", "items": len(rows), "frames": frames,
               "bytes": frames * per_frame,
               "reference_bytes_per_hour_per_role": int(32 * 4096 * 2 * C.TOKEN_HZ * 3600)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
