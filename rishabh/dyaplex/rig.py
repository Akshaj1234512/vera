"""Decoded DyaPlex agent motion -> Samhita's rig, through the existing speech_ai path.

decoded parts (25 fps) -> denormalize -> SMPL-X assembly [P D.3] ->
  head:  neck(12) x head(15) local rotation -> head_yaw/pitch/roll  (speech_ai.human.head_angles)
  jaw:   6D jaw (joint 22) -> opening angle -> jaw_drop channel
  face:  expression slots -> 16 action-unit channels. With real FLAME this is the ridge map
         fitted by ``gr-fit-map``; with this test's FAU stand-in the slots ARE normalized FAUs,
         so the map is the identity selection below.
  gaze:  DyaPlex has no gaze output -> 0 (eyes stay neutral)
-> teacher.npz (21 human channels) -> speech_ai gr-prepare -> gr-replay --policy projected
   (fixed retarget onto the 27 rig actuators, MuJoCo, video)
"""

from __future__ import annotations

from pathlib import Path
import wave

import numpy as np

from . import config as C
from .data import JAW_MAX_RAD, assemble, denormalize
from .rotations import rotation_6d_to_axis_angle

SOURCE = "dyaplex_tiny_e2e_smoke_untrained_scale_not_a_dyaplex_checkpoint"
SOURCE_TRAINED = "dyaplex_reproduction_checkpoint"


def human_channels(parts: dict[str, np.ndarray], stats: dict) -> tuple[np.ndarray, dict]:
    from speech_ai.human import AUS, HEAD_SCALE_DEG, HUMAN_CHANNELS, SEAMLESS_AUS, _AU_SOURCE, head_angles
    raw = {p: denormalize(parts[p], stats[p]) for p in C.PARTS}
    pose = assemble(raw)
    aa = rotation_6d_to_axis_angle(pose["pose6d"])                       # [T, 55, 3]
    angles = head_angles(aa[:, 1:22])                                    # body_pose rows = joints 1..21
    head = np.clip((angles - np.median(angles, axis=0)) / HEAD_SCALE_DEG, -1, 1)
    jaw_angle = aa[:, C.JAW_JOINT, 0]                                    # opening about x
    jaw = np.clip(jaw_angle / JAW_MAX_RAD, 0, 1)
    expr = pose["expression"]
    aus = np.column_stack([np.clip(expr[:, SEAMLESS_AUS.index(_AU_SOURCE[name])], -1, 1) for name in AUS])
    aus[:, AUS.index("jaw_drop")] = jaw
    human = np.column_stack([head, np.zeros((len(head), 2)), aus]).astype(np.float32)
    assert human.shape[1] == len(HUMAN_CHANNELS)
    return human, {"jaw_angle_rad": jaw_angle, "head_deg": angles}


def write_teacher(path: Path, human: np.ndarray, start_s: float = 0.0, source: str = SOURCE) -> Path:
    from speech_ai.human import HUMAN_CHANNELS
    times = start_s + np.arange(len(human)) / C.MOTION_FPS + 1e-6
    np.savez_compressed(path, time_s=times, channels=np.array(HUMAN_CHANNELS), human=human,
                        human_valid=np.ones(len(human), np.float32), source=np.array(source))
    return path


def crop_wav(src: Path, dst: Path, start_s: float, seconds: float) -> Path:
    with wave.open(str(src), "rb") as r:
        rate, width, channels = r.getframerate(), r.getsampwidth(), r.getnchannels()
        r.setpos(int(start_s * rate))
        data = r.readframes(int(seconds * rate))
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(data)
    return dst


def drive_rig(teacher: Path, partner_wav: Path, agent_wav: Path, out: Path, video: bool = True,
              title: str = "DyaPlex tiny e2e (untrained-scale) -> projected retarget") -> dict:
    """gr-prepare + gr-replay --policy projected --profile full on the MuJoCo rig."""
    from speech_ai.gr_data import prepare_stream
    from speech_ai.gr_runtime import replay
    out.mkdir(parents=True, exist_ok=True)
    stream = out / "stream.npz"
    prepared = prepare_stream(partner_wav, agent_wav, False, teacher, None, stream)
    report = replay(stream, None, None, "full", out, None, "projected",
                    out / "rig.mp4" if video else None, agent_wav if video else None,
                    title, "mujoco")
    return {"prepare": prepared, "replay": report, "stream": str(stream)}
