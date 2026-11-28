"""Table 2 evaluation protocol (§5 'Metrics', App. E.1).

    python -m dyaplex evaluate --prepared P --tokens P/tokens --codecs runs/codecs \
        --tower runs/tower-body/tower_030000.pt --smplx /path/SMPLX_NEUTRAL.npz --out runs/eval-body.json
    python -m dyaplex evaluate ... --predictions runs/a2p-preds      # score a baseline's outputs

Protocol [P §5]: teacher-forced: ground-truth PersonaPlex states (both speakers' real audio) and the
partner's ground-truth motion tokens are given, only the agent's tokens are sampled (τ = 1.0, top-k per
--top-k). First 20 s of every test pair, 25 fps, root translation zero, 22 SMPL-X body joints.
Ground-truth joints come from the Seamless SMPL-H parameters (not codec reconstructions).
Δ-User: generate again with each pair's partner motion replaced by another pair's (derangement)
and score (GT partner, shuffled-condition agent) against (GT partner, GT agent) [A, see metrics.py].
Rows: GT, GT (Random) (agent = another pair's GT agent), model. Roles: --roles first (A = first id,
B = second) [A] or both.
Baseline predictions: <dir>/<pair>__<agent>.npz with pose6d [T, 22, 6] (or axis_angle [T, 22, 3]) at
25 fps, optional *_shuffled arrays for Δ-User; same metric code.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch

from . import config as C
from . import data as D
from . import metrics as M
from . import streams as ST
from .codec import decode_all
from .codec_train import load_codecs
from .fk import Skeleton, forward
from .generate import Session
from .prepare import load_meta, load_norm, load_parts, split_pairs
from .rotations import axis_angle_to_6d
from .tower import MotionTower
from .tower_data import PairData, SpeechTower
from .train_tower import speech_lm

SMPL_PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
# SMOKE TESTS ONLY: rough adult rest skeleton (metres) with the SMPL body tree. Not SMPL-X.
_SYNTHETIC_REST = np.array([
    [0, 0, 0], [0.06, -0.09, 0], [-0.06, -0.09, 0], [0, 0.11, -0.02], [0.11, -0.47, 0], [-0.11, -0.47, 0],
    [0, 0.25, 0], [0.09, -0.87, -0.04], [-0.09, -0.87, -0.04], [0, 0.31, 0.02], [0.12, -0.93, 0.08],
    [-0.12, -0.93, 0.08], [0, 0.52, -0.02], [0.08, 0.43, 0], [-0.08, 0.43, 0], [0, 0.6, 0.03],
    [0.19, 0.46, -0.01], [-0.19, 0.46, -0.01], [0.45, 0.43, -0.03], [-0.45, 0.43, -0.03], [0.71, 0.44, -0.03],
    [-0.71, 0.44, -0.03]])


def skeleton(smplx: Path | None, synthetic: bool) -> Skeleton:
    if smplx is not None:
        return Skeleton.from_smplx_npz(smplx)
    if synthetic:
        return Skeleton(_SYNTHETIC_REST.astype(float), SMPL_PARENTS, "synthetic-smoke-only")
    raise SystemExit("--smplx SMPLX_NEUTRAL.npz is required (register at https://smpl-x.is.tue.mpg.de)")


def gt_pose6d(prepared: Path, fid: str, frames25: int) -> np.ndarray:
    parts = {p: np.asarray(v[:frames25], dtype=np.float32) for p, v in load_parts(prepared, fid).items()}
    return D.assemble(parts)["pose6d"]


def agent_audio(meta: dict, seconds: float, sr: int = 24000) -> np.ndarray:
    from scipy.signal import resample_poly
    audio, rate = D.read_wav(Path(meta["wav"]))
    audio = audio[: int(seconds * rate)]
    return resample_poly(audio, sr, rate).astype(np.float32) if rate != sr else audio


def run(prepared: Path, tokens: Path, out: Path, split: str = "test330", codecs_dir: Path | None = None,
        tower_ckpt: Path | None = None, smplx: Path | None = None, synthetic_skeleton: bool = False,
        config: str = "reference", seconds: float = 20.0, top_k: int | None = 200, roles: str = "first",
        predictions: Path | None = None, seed: int = 0, limit: int | None = None, device: str = "auto",
        log=print) -> dict:
    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
    sk = skeleton(smplx, synthetic_skeleton)
    pairs = split_pairs(prepared, split)[:limit]
    orders = [(a, b) for _, a, b in pairs] + ([(b, a) for _, a, b in pairs] if roles == "both" else [])
    keys = [a.rsplit("_", 1)[0] for a, _ in orders]
    frames = int(seconds * C.TOKEN_HZ)
    f25 = 2 * frames
    perm = M.derangement(len(orders), seed)
    gt_a = [forward(gt_pose6d(prepared, a, f25), sk) for a, _ in orders]
    gt_b = [forward(gt_pose6d(prepared, b, f25), sk) for _, b in orders]
    audio_b = [agent_audio(load_meta(prepared, b), seconds) for _, b in orders]
    mean_vel = M.mean_speed(gt_b, C.MOTION_FPS)
    rows = {"GT": {"Diversity": _div(gt_b), "BeatAlign": _beat(gt_b, audio_b, mean_vel)}}
    random_b = [gt_b[perm[i]][: len(gt_b[i])] for i in range(len(orders))]
    rows["GT (Random)"] = M.summarize(random_b, gt_a, gt_b, audio_b, 24000, C.MOTION_FPS, None, mean_vel, seed)
    info = {"split": split, "pairs": len(pairs), "role_orders": len(orders), "seconds": seconds,
            "skeleton": sk.source, "top_k": top_k, "roles": roles}
    if predictions is not None:
        gen, gen_shuf = [], []
        for (a, b), key in zip(orders, keys):
            with np.load(predictions / f"{key}__{b}.npz") as z:
                pose = z["pose6d"] if "pose6d" in z.files else axis_angle_to_6d(z["axis_angle"])
                gen.append(forward(pose[:f25], sk))
                if "pose6d_shuffled" in z.files or "axis_angle_shuffled" in z.files:
                    ps = z["pose6d_shuffled"] if "pose6d_shuffled" in z.files else axis_angle_to_6d(z["axis_angle_shuffled"])
                    gen_shuf.append(forward(ps[:f25], sk))
        rows[str(predictions.name)] = M.summarize(gen, gt_a, gt_b, audio_b, 24000, C.MOTION_FPS,
                                                  gen_shuf or None, mean_vel, seed)
    else:
        ck = torch.load(tower_ckpt, map_location="cpu", weights_only=False)
        tcfg = C.TowerConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["tower"].items()})
        model = MotionTower(tcfg)
        if ck.get("ablation", "none") != "none":
            model.ablate(ck["ablation"])
        model.load_state_dict(ck["model"])
        model.to(dev).eval()
        codecs = load_codecs(codecs_dir, "cpu")
        norm = load_norm(prepared)
        scfg = (C.REFERENCE if config == "reference" else C.TINY)["speech"]
        speech = SpeechTower(speech_lm(config, dev), scfg, ST.prompt(prepared)["text_prompt"])
        data = PairData(prepared, split, tokens, speech, 1, limit)
        sample = C.SampleConfig(temperature=1.0, top_k=top_k)
        gen, gen_shuf = [], []
        began = time.time()
        for i, ((a, b), key) in enumerate(zip(orders, keys)):
            states = speech.states(data._streams(b), data._streams(a), 0, frames)          # [L, F, d]
            sp = states.float().cpu().numpy()
            partner = data._tokens(a)[:frames]
            shuffled = data._tokens(orders[perm[i]][0])[:frames]
            n = min(len(partner), len(shuffled), sp.shape[1])
            for target, part_codes in ((gen, partner[:n]), (gen_shuf, shuffled[:n])):
                session = Session(model, [torch.tensor(sp[l][None, :n]) for l in range(sp.shape[0])], sample, seed)
                b_codes = np.stack([session.step_frame(part_codes[f])[1] for f in range(n)])
                parts = {p: D.denormalize(v, norm[p]) for p, v in decode_all(codecs, b_codes).items()}
                target.append(forward(D.assemble(parts)["pose6d"], sk))
            log(f"  {i + 1}/{len(orders)} {key} ({time.time() - began:.0f} s)")
        rows[f"Ours ({ck.get('variant')}, {ck.get('ablation', 'none')})"] = M.summarize(
            gen, gt_a, gt_b, audio_b, 24000, C.MOTION_FPS, gen_shuf, mean_vel, seed)
        info["tower"] = str(tower_ckpt)
    result = {"info": info, "rows": rows,
              "paper_table2": {"GT": {"Diversity": 0.633, "BeatAlign": 0.049},
                               "GT (Random)": {"FGD_x1e3": 13, "Diversity": 0.683, "BeatAlign": 0.050, "P-FD_x1e3": 33},
                               "Ours (body-only)": {"FGD_x1e3": 5.6, "Diversity": 0.611, "BeatAlign": 0.059,
                                                    "P-FD_x1e3": 7.3, "Delta-User_%": 31}}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float) + "\n")
    return result


def _div(clips):
    frames = np.concatenate(clips)
    return M.diversity(frames) if len(frames) > 10_000 else float("nan")


def _beat(clips, audio, mean_vel):
    return float(np.nanmean([M.beat_align(c, a, 24000, C.MOTION_FPS, mean_vel) for c, a in zip(clips, audio)]))
