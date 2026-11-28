"""Trained DyaPlex -> Samhita's rig (agent-only streaming, §4.4) for one recorded pair.

    python -m dyaplex infer --prepared P --tokens P/tokens --codecs runs/codecs \
        --tower runs/tower-face/tower_030000.pt --pair <pair key> --agent <file_id> --start-s 30 --seconds 20 \
        --out runs/demo

Replays a recorded conversation causally: PersonaPlex states from both speakers' audio, the partner's
observed motion tokens in the [A] slots, the agent's tokens sampled frame by frame (τ = 1.0, top-k 200),
decoded by the causal part decoders, mapped to the 21 human channels and driven onto the 27-actuator
rig with the existing speech_ai replay (video with the agent's audio).
Use a face-variant checkpoint (--variant face): the body-only model never trains the face codes.
Live use (microphone + camera partner motion) needs PersonaPlex's streaming loop (LMGen) and an online
body/face tracker for the partner; Session.step_frame is the per-frame entry point for that loop.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch

from . import config as C
from . import data as D
from . import rig
from . import streams as ST
from .codec import decode_all
from .codec_train import load_codecs
from .generate import Session
from .prepare import load_meta, load_norm
from .tower import MotionTower
from .tower_data import PairData, SpeechTower
from .train_tower import speech_lm


def run(prepared: Path, tokens: Path, codecs_dir: Path, tower_ckpt: Path, pair: str, agent: str, out: Path,
        start_s: float = 0.0, seconds: float = 20.0, config: str = "reference", top_k: int | None = 200,
        seed: int = 0, device: str = "auto", video: bool = True) -> dict:
    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(tower_ckpt, map_location="cpu", weights_only=False)
    tcfg = C.TowerConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["tower"].items()})
    model = MotionTower(tcfg)
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    scfg = (C.REFERENCE if config == "reference" else C.TINY)["speech"]
    speech = SpeechTower(speech_lm(config, dev), scfg, ST.prompt(prepared)["text_prompt"])
    data = PairData.__new__(PairData)                       # use the cached loaders without building items
    data.prepared, data.tokens_dir, data.speech, data.window = prepared, tokens, speech, 1
    ids = sorted(p.stem for p in tokens.glob(f"{pair}_*.npy"))
    if agent not in ids or len(ids) != 2:
        raise SystemExit(f"{pair}: expected two tokenized participants including {agent}, found {ids}")
    user = next(i for i in ids if i != agent)
    f0, frames = int(start_s * C.TOKEN_HZ), int(seconds * C.TOKEN_HZ)
    t0 = time.time()
    states = speech.states(data._streams(agent), data._streams(user), 0, f0 + frames)[:, f0:]
    speech_s = time.time() - t0
    sp = states.float().cpu().numpy()
    partner = data._tokens(user)[f0:f0 + frames]
    session = Session(model, [torch.tensor(sp[l][None]) for l in range(sp.shape[0])],
                      C.SampleConfig(temperature=1.0, top_k=top_k), seed)
    t0 = time.time()
    b = np.stack([session.step_frame(partner[f])[1] for f in range(len(partner))])
    tower_s = time.time() - t0
    norm = load_norm(prepared)
    parts = decode_all(load_codecs(codecs_dir, "cpu"), b)
    human, _ = rig.human_channels(parts, norm)
    teacher = rig.write_teacher(out / "teacher_dyaplex.npz", human, source=rig.SOURCE_TRAINED)
    partner_wav = rig.crop_wav(Path(load_meta(prepared, user)["wav"]), out / "partner.wav", start_s, seconds)
    agent_wav = rig.crop_wav(Path(load_meta(prepared, agent)["wav"]), out / "agent.wav", start_s, seconds)
    sim = rig.drive_rig(teacher, partner_wav, agent_wav, out / "rig", video,
                        title=f"DyaPlex {ck.get('variant')} step {ck.get('step')} -> projected retarget")
    report = {"pair": pair, "agent": agent, "partner": user, "start_s": start_s, "seconds": seconds,
              "tower": str(tower_ckpt), "variant": ck.get("variant"), "speech_s": speech_s,
              "tower_ms_per_frame": 1000 * tower_s / max(len(partner), 1), "video": str(out / "rig" / "rig.mp4")}
    (out / "infer.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
