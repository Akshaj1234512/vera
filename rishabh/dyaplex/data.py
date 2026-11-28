"""Seamless Interaction pairs -> DyaPlex motion features (App. C, D.3).

Pipeline per participant (all steps tagged [P]/[I]/[A] as in config.py):
  1. load SMPL-H (global_orient, body_pose, hand poses, translation, is_valid) at 30 Hz
  2. frame validity: smplh:is_valid [P C.1]; invalid frames are kept on the timeline,
     filled by slerp from valid neighbours and masked out of statistics, codec windows
     and losses [A] (deleting them would desynchronize motion from audio; review X-gap)
  3. 180 deg rotation about x applied to the root and translation [P D.3]
  4. Savitzky-Golay smoothing of translation [P D.3; window/order A]; rescaling factor [A] = 1
  5. slerp / linear resample 30 -> 25 fps [P D.3; method A]
  6. SMPL-H -> SMPL-X joint order: body 0-21, hands -> 25-54, jaw 22 from the face track,
     eyes 23-24 identity [A]
  7. 6D rotation per joint [P D.3] and GestureLSM translation velocity (x,z diff, y absolute)
     [I GestureLSM motion_rep_transfer.py:168-172]
  8. face (56-D = 6D jaw + 50 expression) [P C.2]. SPECTRE/FLAME is not available here, so
     this test uses a STAND-IN built from Seamless FAU values (see ``face_standin``).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import wave

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

from . import config as C
from .rotations import axis_angle_to_6d, resample_linear, resample_rotations, rotation_6d_to_axis_angle, x_rotation

BASE = "https://dl.fbaipublicfiles.com/seamless_interaction"
SMPLH_EXTRAS = ("global_orient", "left_hand_pose", "right_hand_pose", "translation")
FACE_SOURCE = "standin_seamless_fau_not_spectre_flame"
SAVGOL_WINDOW, SAVGOL_ORDER = 9, 2     # [A] at 30 fps
JAW_MAX_RAD = 0.30                     # [A] stand-in scale for JawDrop -> jaw rotation


# --------------------------------------------------------------------------- #
# Pairing, fetching and clip-level filter
# --------------------------------------------------------------------------- #
def complete_pairs(folder: Path) -> list[tuple[str, str, str]]:
    """(interaction key, participant A, participant B) where both participants exist locally.

    Role assignment A=first, B=second in sorted order [A]; the paper says only "without loss of
    generality" (§4). Training uses both orders (review X-6).
    """
    by_key: dict[str, list[str]] = {}
    for path in sorted(folder.iterdir()):
        if path.is_dir() and (path / "smplh_body_pose.npy").is_file():
            by_key.setdefault(path.name.rsplit("_", 1)[0], []).append(path.name)
    return [(key, *ids) for key, ids in sorted(by_key.items()) if len(ids) == 2]


def fetch_extras(folder: Path, label: str, split: str) -> list[str]:
    """Fetch the SMPL-H arrays the G/R downloader skipped (~2 MB per participant)."""
    from speech_ai.sources import fetch
    fetched = []
    for name in SMPLH_EXTRAS:
        target = folder / f"smplh_{name}.npy"
        if not target.is_file():
            fetch(f"{BASE}/{label}/{split}/smplh/{name}/{folder.name}.npy", target)
            fetched.append(target.name)
    return fetched


def probe_video(label: str, split: str, file_id: str) -> dict:
    """Clip-level aspect-ratio / integrity check [P C.1] from the remote MP4 header only."""
    url = f"{BASE}/{label}/{split}/video/{file_id}.mp4"
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=width,height:stream_side_data=rotation:stream_tags=rotate", "-of", "json", url]
    try:
        out = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60).stdout)
        stream = out["streams"][0]
        width, height = int(stream.get("width", 0)), int(stream.get("height", 0))
        rotation = stream.get("tags", {}).get("rotate") or next(
            (s.get("rotation") for s in stream.get("side_data_list", []) if "rotation" in s), 0)
    except Exception as error:  # unreadable counts as corrupt [P C.1]
        return {"file_id": file_id, "status": "unreadable", "error": str(error)[:200], "keep": False}
    if width == 0 or height == 0:
        status = "unreadable"
    elif width == 640 and height == 480:
        status = "rotated"                       # [P C.1] 640x480 listed as rotated
    elif width > height:
        status = "landscape"                     # [P C.1] 3840x2160, 1920x1080
    else:
        status = "portrait_or_square"
    return {"file_id": file_id, "width": width, "height": height, "rotation": rotation, "status": status,
            "keep": status == "portrait_or_square"}


# --------------------------------------------------------------------------- #
# Participant loading and preprocessing
# --------------------------------------------------------------------------- #
def _movement(arrays: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    """movement_v4 if at least half its frames are valid, else movement v1 (same rule as speech_ai.seamless)."""
    found = [(arrays[f"{v}:FAUValue"].astype(np.float64), arrays[f"{v}:is_valid"].reshape(-1) > 0)
             for v in ("movement_v4", "movement") if f"{v}:FAUValue" in arrays and f"{v}:is_valid" in arrays]
    if not found:
        raise FileNotFoundError(f"{name}: no movement FAU values")
    for fau, valid in found:
        if valid.mean() >= 0.5:
            return fau, valid
    return max(found, key=lambda item: item[1].mean())



def _fill_invalid(aa: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Slerp every joint across invalid frames from the nearest valid ones [A]."""
    if valid.all():
        return aa
    times = np.arange(len(aa), dtype=np.float64)
    keys = times[valid]
    query = np.clip(times, keys[0], keys[-1])
    out = np.empty_like(aa)
    for j in range(aa.shape[1]):
        out[:, j] = Slerp(keys, Rotation.from_rotvec(aa[valid, j]))(query).as_rotvec()
    return out


def _robust(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    base = np.percentile(x[valid], 50, axis=0)
    span = np.percentile(x[valid], 99, axis=0) - base
    return np.clip((x - base) / np.where(span > 1e-6, span, 1.0), -1, 1)


def face_standin(fau: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """56-D face track in the paper's layout (6D jaw + 50 expression) built from Seamless FAUs.

    STAND-IN for SPECTRE->FLAME [P C.2], which needs the FLAME model and SPECTRE weights.
    Jaw: rotation about x by JAW_MAX_RAD * JawDrop (normalized). Expression: the 24 normalized
    FAU values in slots 0-23, zeros in 24-49. Frames with failed face tracking are all-zero,
    the same failure signature the paper describes for SPECTRE [P C.2].
    """
    from speech_ai.human import SEAMLESS_AUS
    if valid.sum() < 2:                      # tracking failed everywhere -> all-zero face [P C.2]
        return np.zeros((len(fau), 6 + C.EXPRESSION_DIM))
    norm = _robust(fau, valid)
    jaw_angle = JAW_MAX_RAD * np.clip(norm[:, SEAMLESS_AUS.index("JawDrop")], 0, 1)
    jaw6d = axis_angle_to_6d(np.column_stack([jaw_angle, np.zeros_like(jaw_angle), np.zeros_like(jaw_angle)]))
    expression = np.zeros((len(fau), C.EXPRESSION_DIM))
    expression[:, :fau.shape[1]] = norm
    face = np.column_stack([jaw6d, expression])
    face[~valid] = 0.0
    return face


def jaw_axis_angle_from_face(face: np.ndarray) -> np.ndarray:
    from .rotations import rotation_6d_to_axis_angle
    return rotation_6d_to_axis_angle(face[:, :6])


@dataclass
class Participant:
    file_id: str
    pose6d: np.ndarray        # [T25, 55, 6] SMPL-X joint order
    trans: np.ndarray         # [T25, 3] after the 180 deg x rotation
    trans_v: np.ndarray       # [T25, 3] GestureLSM velocity layout
    face: np.ndarray          # [T25, 56]
    valid: np.ndarray         # [T25] body validity (smplh:is_valid)
    face_valid: np.ndarray    # [T25] non-zero face frames
    audio: Path | None        # participant WAV (48 kHz official, 16 kHz local copies)
    vad: list[tuple[float, float]]
    body_pose_aa: np.ndarray  # [T25, 21, 3] local body rotations (for head channels)
    stats: dict = field(default_factory=dict)

    @property
    def frames(self) -> int:
        return len(self.valid)


def face_from_spectre(path: Path, frames30: int) -> tuple[np.ndarray, np.ndarray]:
    """dyaplex face-extract output -> 56-D face at the source frame rate [P C.2].

    Keeps the 6D jaw (from SPECTRE's axis-angle jaw) and the 50 FLAME expression coefficients,
    drops the head rotation (redundant with SMPL-H joint 0). Failed frames stay all-zero [P C.2]."""
    with np.load(path, allow_pickle=False) as z:
        jaw, exp, ok = z["jaw_aa"].astype(np.float64), z["exp"].astype(np.float64), z["valid"].astype(bool)
    n = min(frames30, len(jaw))
    face = np.zeros((frames30, 6 + C.EXPRESSION_DIM))
    face[:n, :6] = axis_angle_to_6d(jaw[:n])
    face[:n, 6:] = exp[:n, :C.EXPRESSION_DIM]
    valid = np.zeros(frames30, bool)
    valid[:n] = ok[:n] & (np.abs(face[:n]).sum(1) > 0)
    face[~valid] = 0.0
    return face, valid


def load_record(rec, face_mode: str = "auto") -> Participant:
    """Seamless participant (sources.Record) -> 25 fps DyaPlex features.

    face_mode: "spectre" (needs rec.face), "standin" (Seamless FAUs, see face_standin), or "auto"
    (SPECTRE when available, else stand-in). The source is recorded in stats["face_source"]."""
    a = rec.arrays
    need = ("smplh:body_pose", "smplh:global_orient", "smplh:left_hand_pose", "smplh:right_hand_pose",
            "smplh:translation", "smplh:is_valid")
    missing = [k for k in need if k not in a]
    if missing:
        raise FileNotFoundError(f"{rec.file_id}: missing {missing}")
    body = a["smplh:body_pose"].reshape(-1, 21, 3).astype(np.float64)
    root = a["smplh:global_orient"].reshape(-1, 1, 3).astype(np.float64)
    left = a["smplh:left_hand_pose"].reshape(-1, 15, 3).astype(np.float64)
    right = a["smplh:right_hand_pose"].reshape(-1, 15, 3).astype(np.float64)
    trans = a["smplh:translation"].reshape(-1, 3).astype(np.float64)
    valid = a["smplh:is_valid"].reshape(-1) > 0
    use_spectre = face_mode == "spectre" or (face_mode == "auto" and rec.face is not None)
    if use_spectre and rec.face is None:
        raise FileNotFoundError(f"{rec.file_id}: no SPECTRE face file")
    fau, fau_valid = (None, None) if use_spectre else _movement(a, rec.file_id)
    lengths = [len(body), len(root), len(left), len(right), len(trans), len(valid)]
    if fau is not None:
        lengths += [len(fau), len(fau_valid)]
    frames = min(lengths)
    body, root, left, right, trans = body[:frames], root[:frames], left[:frames], right[:frames], trans[:frames]
    finite = np.isfinite(np.concatenate([body.reshape(frames, -1), root.reshape(frames, -1), left.reshape(frames, -1),
                                         right.reshape(frames, -1), trans], axis=1)).all(1)
    valid = valid[:frames] & finite
    if valid.sum() < 8:                                   # [P C.1] fewer than 8 valid frames -> drop
        raise ValueError(f"{rec.file_id}: {int(valid.sum())} valid frames (< 8)")
    smplh = np.concatenate([root, body, left, right], axis=1)             # [T, 52, 3]
    smplh = _fill_invalid(np.nan_to_num(smplh), valid)
    # 180 deg about x maps the recovered pose into the SMPL-X world frame [P D.3]
    flip = Rotation.from_matrix(x_rotation(np.pi))
    smplh[:, 0] = (flip * Rotation.from_rotvec(smplh[:, 0])).as_rotvec()
    trans = np.nan_to_num(trans)
    if (~valid).any():
        good, bad = np.flatnonzero(valid), np.flatnonzero(~valid)
        for k in range(3):
            trans[bad, k] = np.interp(bad, good, trans[good, k])
    trans = trans @ x_rotation(np.pi).T
    trans = savgol_filter(trans, SAVGOL_WINDOW, SAVGOL_ORDER, axis=0)   # [P D.3; params A]
    # Resample 30 -> 25 fps [P D.3]
    smplh25 = resample_rotations(smplh, C.SEAMLESS_FPS, C.MOTION_FPS)
    trans25 = resample_linear(trans, C.SEAMLESS_FPS, C.MOTION_FPS)
    t25 = len(smplh25)
    src = np.arange(t25) / C.MOTION_FPS * C.SEAMLESS_FPS
    lo, hi = np.floor(src).astype(int), np.minimum(np.ceil(src).astype(int), frames - 1)
    valid25 = valid[lo] & valid[hi]
    if use_spectre:
        face30, face_ok = face_from_spectre(rec.face, frames)
        face_valid25 = face_ok[lo] & face_ok[hi]
        # interpolate jaw in rotation space, expression linearly
        jaw25 = resample_rotations(rotation_6d_to_axis_angle(face30[:, :6])[:, None], C.SEAMLESS_FPS,
                                   C.MOTION_FPS)[:t25, 0]
        exp25 = resample_linear(face30[:, 6:], C.SEAMLESS_FPS, C.MOTION_FPS)[:t25]
        face25 = np.column_stack([axis_angle_to_6d(jaw25), exp25])
        face25[~face_valid25] = 0.0
        face_source = "spectre_flame"
    else:
        face_ok = fau_valid[:frames]
        face_valid25 = face_ok[lo] & face_ok[hi]
        # build the face at 25 fps from resampled FAUs (resampling 6D linearly would break orthonormality)
        fau25 = resample_linear(fau[:frames], C.SEAMLESS_FPS, C.MOTION_FPS)[:t25]
        face25 = face_standin(fau25, face_valid25)
        face_source = FACE_SOURCE
    # SMPL-H -> SMPL-X joint order [A mapping, review §3.1]
    aa = np.zeros((t25, C.SMPLX_JOINTS, 3))
    aa[:, :22] = smplh25[:, :22]
    aa[:, C.JAW_JOINT] = jaw_axis_angle_from_face(face25)
    aa[:, 25:40] = smplh25[:, 22:37]
    aa[:, 40:55] = smplh25[:, 37:52]
    pose6d = axis_angle_to_6d(aa)
    pose6d[:, C.JAW_JOINT] = face25[:, :6]
    pose6d[:, C.JAW_JOINT][~face_valid25] = axis_angle_to_6d(np.zeros(3))
    trans_v = np.zeros_like(trans25)                  # [I GestureLSM motion_rep_transfer.py:168-172]
    trans_v[1:, 0] = trans25[1:, 0] - trans25[:-1, 0]
    trans_v[0, 0] = trans_v[1, 0]
    trans_v[1:, 2] = trans25[1:, 2] - trans25[:-1, 2]
    trans_v[0, 2] = trans_v[1, 2]
    trans_v[:, 1] = trans25[:, 1]
    return Participant(rec.file_id, pose6d.astype(np.float32), trans25.astype(np.float32), trans_v.astype(np.float32),
                       face25.astype(np.float32), valid25, face_valid25, rec.wav, list(rec.vad),
                       aa[:, 1:22].astype(np.float32),
                       {"frames_30": int(frames), "frames_25": int(t25), "valid_30": float(valid.mean()),
                        "zero_face_fraction": float(1 - face_valid25.mean()), "face_source": face_source})


def load_participant(folder: Path, face_mode: str = "auto") -> Participant:
    """Per-participant folder (FeatureLayout) -> Participant. Kept for the e2e test."""
    from .sources import FeatureLayout
    folder = Path(folder)
    return load_record(FeatureLayout(folder.parent).load(folder.name), face_mode)



# --------------------------------------------------------------------------- #
# Part features, normalization, assembly
# --------------------------------------------------------------------------- #
def _joint_cols(joints) -> np.ndarray:
    return np.array([j * 6 + k for j in joints for k in range(6)])


def part_features(p: Participant) -> dict[str, np.ndarray]:
    flat = p.pose6d.reshape(p.frames, -1)          # 330 = 55 x 6, GestureLSM layout
    return {
        "upper": flat[:, _joint_cols(C.UPPER_JOINTS)],
        "hands": flat[:, _joint_cols(C.HAND_JOINTS)],
        "lower_trans": np.column_stack([flat[:, _joint_cols(C.LOWER_JOINTS)], p.trans_v]),
        "face": p.face,
    }


def part_valid(p: Participant) -> dict[str, np.ndarray]:
    return {"upper": p.valid, "hands": p.valid, "lower_trans": p.valid, "face": p.face_valid}


def face_pair_ok(a: Participant, b: Participant) -> bool:
    """Face-codec training filter: exclude pairs with >= 20% zero-face frames in either speaker [P C.2]."""
    return a.stats["zero_face_fraction"] < 0.2 and b.stats["zero_face_fraction"] < 0.2


def fit_normalization(participants: list[Participant]) -> dict[str, dict[str, np.ndarray]]:
    """Mean/std per part on valid training frames [A: refit on Seamless; GestureLSM uses BEATX stats]."""
    stats = {}
    for part in C.PARTS:
        rows = np.concatenate([part_features(p)[part][part_valid(p)[part]] for p in participants])
        mean, std = rows.mean(0), rows.std(0)
        stats[part] = {"mean": mean.astype(np.float32), "std": np.where(std < 1e-6, 1.0, std).astype(np.float32)}
    return stats


def normalize(x: np.ndarray, stats: dict) -> np.ndarray:
    return ((x - stats["mean"]) / stats["std"]).astype(np.float32)


def denormalize(x: np.ndarray, stats: dict) -> np.ndarray:
    return (x * stats["std"] + stats["mean"]).astype(np.float32)


def assemble(parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Decoded part features -> canonical SMPL-X pose [P D.3 'Output assembly'].

    Returns 55-joint 6D pose (eyes identity [A]), FLAME-slot expression and translation
    integrated from velocity (x,z cumulative, y absolute) [I GestureLSM velocity2position].
    """
    frames = len(parts["upper"])
    pose = np.tile(axis_angle_to_6d(np.zeros(3)), (frames, C.SMPLX_JOINTS, 1)).astype(np.float32)
    pose[:, list(C.UPPER_JOINTS)] = parts["upper"].reshape(frames, -1, 6)
    pose[:, list(C.HAND_JOINTS)] = parts["hands"].reshape(frames, -1, 6)
    pose[:, list(C.LOWER_JOINTS)] = parts["lower_trans"][:, :54].reshape(frames, -1, 6)
    pose[:, C.JAW_JOINT] = parts["face"][:, :6]
    tv = parts["lower_trans"][:, 54:57]
    trans = np.column_stack([np.cumsum(tv[:, 0]), tv[:, 1], np.cumsum(tv[:, 2])])
    return {"pose6d": pose, "expression": parts["face"][:, 6:].astype(np.float32), "trans": trans.astype(np.float32)}


# --------------------------------------------------------------------------- #
# Audio and VAD on the 12.5 Hz token grid
# --------------------------------------------------------------------------- #
def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        rate, frames, width = handle.getframerate(), handle.getnframes(), handle.getsampwidth()
        raw = handle.readframes(frames)
        channels = handle.getnchannels()
    audio = np.frombuffer(raw, dtype=np.int16 if width == 2 else np.int32).astype(np.float32)
    audio = audio.reshape(-1, channels).mean(1) / (32768.0 if width == 2 else 2 ** 31)
    return audio, rate


def vad_labels(vad: list[tuple[float, float]], frames: int, hz: float = C.TOKEN_HZ) -> np.ndarray:
    """Binary speaking state per token frame: frame centre inside a VAD interval [A pooling]."""
    centres = (np.arange(frames) + 0.5) / hz
    out = np.zeros(frames, dtype=np.int64)
    for start, end in vad:
        out[(centres >= start) & (centres < end)] = 1
    return out


def speech_envelope(audio: np.ndarray, rate: int, hz: float) -> np.ndarray:
    hop = int(round(rate / hz))
    frames = len(audio) // hop
    return np.sqrt((audio[: frames * hop].reshape(frames, hop) ** 2).mean(1) + 1e-12)


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)
