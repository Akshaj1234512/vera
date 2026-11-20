"""Human facial-motion space (G's input M_h) and its fixed retarget onto the rig.

M_h has 21 channels per frame, all normalized to about [-1, 1]:

* head_yaw / head_pitch / head_roll: torso-relative head rotation (SMPL-H neck x head
  joints), degrees / 30. Positive = turn to own left / look up / left side up.
* gaze_yaw / gaze_pitch: eye direction relative to the head. Positive = look left / up.
* 16 facial action units (AU), neutral-subtracted and scaled by the participant's 99th
  percentile, so 0 = that person's neutral face and 1 = a strong expression.

This is the representation a DyaPlex export must be mapped into: DyaPlex emits SMPL-X
pose (neck/head joints) plus FLAME jaw and 50 expression coefficients, so its head
channels map directly and its FLAME coefficients map to the AU channels with
``gr-fit-map``. DyaPlex emits no gaze; gaze channels are then marked invalid.

The retarget ``RETARGET`` (human channel -> robot channel, physical units) is the
"projected DyaPlex" baseline: a hand-written semantic mapping, not learned.
"""

import numpy as np

from .embodiment import CHANNELS, SCALE

HEAD_SCALE_DEG = 30.0
AUS = ("inner_brow_raiser", "outer_brow_raiser", "brow_lowerer", "upper_lid_raiser", "cheek_raiser",
       "lid_tightener", "nose_wrinkler", "upper_lip_raiser", "lip_corner_puller", "lip_corner_depressor",
       "lower_lip_depressor", "chin_raiser", "lip_puckerer", "lip_stretcher", "jaw_drop", "eyes_closed")
HUMAN_CHANNELS = ("head_yaw", "head_pitch", "head_roll", "gaze_yaw", "gaze_pitch") + AUS

# Seamless Interaction FAUValue column order (README of facebookresearch/seamless_interaction),
# checked against video frames: LipCornerPuller peaks are smiles, JawDrop peaks are open mouths.
SEAMLESS_AUS = ("InnerBrowRaiser", "OuterBrowRaiser", "BrowLowerer", "UpperLidRaiser", "CheekRaiser",
                "LidTightener", "NoseWrinkler", "UpperLipRaiser", "LipCornerPuller", "CheekPuffer", "Dimpler",
                "LipCornerDepressor", "LowerLipDepressor", "ChinRaiser", "LipPuckerer", "LipStretcher",
                "LipFunneler", "LipTightener", "LipPressor", "LipsParts", "JawDrop", "LipSuck", "JawSideways",
                "EyesClosed")
# Kept AUs; dropped: CheekPuffer, Dimpler, LipFunneler, LipTightener, LipPressor, LipSuck, JawSideways
# (near zero in conversation) and LipsParts (an exact duplicate of LowerLipDepressor in the release).
_AU_SOURCE = dict(zip(AUS, ("InnerBrowRaiser", "OuterBrowRaiser", "BrowLowerer", "UpperLidRaiser", "CheekRaiser",
                            "LidTightener", "NoseWrinkler", "UpperLipRaiser", "LipCornerPuller",
                            "LipCornerDepressor", "LowerLipDepressor", "ChinRaiser", "LipPuckerer",
                            "LipStretcher", "JawDrop", "EyesClosed")))

# Robot channel <- sum(weight * human channel), in robot physical units (rad or m).
_MAP = {
    "neck_yaw": {"head_yaw": np.radians(HEAD_SCALE_DEG)},
    "neck_pitch": {"head_pitch": np.radians(HEAD_SCALE_DEG)},
    "neck_roll": {"head_roll": np.radians(HEAD_SCALE_DEG)},
    "left_eye_yaw": {"gaze_yaw": 0.30}, "right_eye_yaw": {"gaze_yaw": 0.30},
    "left_eye_pitch": {"gaze_pitch": 0.25}, "right_eye_pitch": {"gaze_pitch": 0.25},
    "left_brow_raise": {"inner_brow_raiser": 0.005, "outer_brow_raiser": 0.005, "brow_lowerer": -0.004},
    "right_brow_raise": {"inner_brow_raiser": 0.005, "outer_brow_raiser": 0.005, "brow_lowerer": -0.004},
    "left_eyelid_close": {"eyes_closed": -0.012, "lid_tightener": -0.003, "upper_lid_raiser": 0.001},
    "right_eyelid_close": {"eyes_closed": -0.012, "lid_tightener": -0.003, "upper_lid_raiser": 0.001},
    "jaw_open": {"jaw_drop": 0.32, "lower_lip_depressor": 0.06},
    "mouth_left_corner": {"lip_corner_puller": 0.007, "lip_corner_depressor": -0.006},
    "mouth_right_corner": {"lip_corner_puller": 0.007, "lip_corner_depressor": -0.006},
    "upper_lip_raise": {"upper_lip_raiser": 0.005},
    "upper_lip_left": {"upper_lip_raiser": 0.004, "lip_corner_puller": 0.001},
    "upper_lip_right": {"upper_lip_raiser": 0.004, "lip_corner_puller": 0.001},
    "lower_lip_drop": {"lower_lip_depressor": -0.006},
    "lower_lip_left": {"lower_lip_depressor": -0.005},
    "lower_lip_right": {"lower_lip_depressor": -0.005},
    "left_cheek_raise": {"cheek_raiser": 0.006, "lip_corner_puller": 0.002},
    "right_cheek_raise": {"cheek_raiser": 0.006, "lip_corner_puller": 0.002},
    "left_nostril_flare": {"nose_wrinkler": 0.004}, "right_nostril_flare": {"nose_wrinkler": 0.004},
    "left_nose_wrinkle": {"nose_wrinkler": 0.005}, "right_nose_wrinkle": {"nose_wrinkler": 0.005},
    "chin_protrude": {"chin_raiser": 0.005, "lip_puckerer": 0.004},
}
RETARGET = np.zeros((len(HUMAN_CHANNELS), len(CHANNELS)), dtype=np.float32)
for _robot, _terms in _MAP.items():
    for _human, _weight in _terms.items():
        RETARGET[HUMAN_CHANNELS.index(_human), CHANNELS.index(_robot)] = _weight
# The same map between normalized spaces: robot_norm = human_norm @ RETARGET_NORM.
RETARGET_NORM = (RETARGET / SCALE[None, :]).astype(np.float32)


def retarget(human: np.ndarray) -> np.ndarray:
    """Normalized human channels [..., 21] -> robot physical units [..., 27] (unconstrained)."""
    return human @ RETARGET


# --------------------------------------------------------------------------- #
# Seamless Interaction -> M_h
# --------------------------------------------------------------------------- #
def head_angles(body_pose: np.ndarray) -> np.ndarray:
    """SMPL-H neck x head local rotation -> [yaw, pitch, roll] degrees in the human convention.

    SMPL frame: x = subject's left, y = up, z = forward. Joint 12 (neck) and 15 (head) are
    body_pose rows 11 and 14. Intrinsic YXZ Euler gives yaw (left +), pitch (+ = chin
    down in SMPL, so negated to make + = look up) and roll (left side up +).
    """
    from scipy.spatial.transform import Rotation

    local = Rotation.from_rotvec(body_pose[:, 11]) * Rotation.from_rotvec(body_pose[:, 14])
    yaw, pitch, roll = local.as_euler("YXZ", degrees=True).T
    return np.column_stack([yaw, -pitch, roll])


def _robust(x: np.ndarray, valid: np.ndarray, low_pct: float = 50, high_pct: float = 99) -> np.ndarray:
    base = np.percentile(x[valid], low_pct, axis=0)
    span = np.percentile(x[valid], high_pct, axis=0) - base
    return np.clip((x - base) / np.where(span > 1e-6, span, 1.0), -1, 1)


def seamless_to_human(fau: np.ndarray, gaze: np.ndarray, movement_valid: np.ndarray,
                      body_pose: np.ndarray, smplh_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-participant normalized M_h at 30 Hz from Seamless movement_v4 and SMPL-H arrays.

    Normalization statistics use the participant's whole session (a per-person neutral
    calibration). Head angles are centered on the session median (resting posture).
    """
    frames = min(len(fau), len(gaze), len(body_pose), len(movement_valid), len(smplh_valid))
    fau, gaze, body_pose = fau[:frames], gaze[:frames], body_pose[:frames]
    valid = (movement_valid[:frames].reshape(-1) > 0) & (smplh_valid[:frames].reshape(-1) > 0)
    valid &= np.isfinite(fau).all(1) & np.isfinite(gaze).all(1) & np.isfinite(body_pose.reshape(frames, -1)).all(1)
    if valid.sum() < 30:
        raise ValueError("Fewer than 30 valid frames")
    angles = head_angles(np.nan_to_num(body_pose))
    rest = np.median(angles[valid], axis=0)
    head = np.clip((angles - rest) / HEAD_SCALE_DEG, -1, 1)
    # gaze_encodings: column 1 decreases when the eyes look down (checked on video); column 0's
    # left/right sign could not be verified at the released video resolution.
    g = gaze - np.median(gaze[valid], axis=0)
    g = np.clip(g / (2.5 * np.std(g[valid], axis=0) + 1e-6), -1, 1)
    columns = [SEAMLESS_AUS.index(_AU_SOURCE[name]) for name in AUS]
    aus = _robust(fau[:, columns], valid)
    human = np.column_stack([head, g[:, 0], g[:, 1], aus]).astype(np.float32)
    human[~valid] = 0
    stats = {"head_rest_deg": rest.round(3).tolist(), "valid_fraction": float(valid.mean()),
             "gaze_yaw_sign": "unverified", "frames": int(frames)}
    return human, valid, stats


# --------------------------------------------------------------------------- #
# Labels: intended meaning from the human, and a SYNTHETIC observer for robot motion
# --------------------------------------------------------------------------- #
LABELS = ("no_signal", "neutral_ack", "enthusiastic_ack", "unclear")


def _largest_drop(x: np.ndarray) -> float:
    """Largest decrease from an earlier sample to a later one (nod depth if x is pitch)."""
    return float(np.max(np.maximum.accumulate(x) - x)) if len(x) else 0.0


def human_intent(human_chunk: np.ndarray) -> str:
    """What the human's own 500 ms of motion conveyed (rule, used as the intended label z*)."""
    h = dict(zip(HUMAN_CHANNELS, human_chunk.T))
    nod = _largest_drop(h["head_pitch"] * HEAD_SCALE_DEG)
    smile = float(h["lip_corner_puller"].max())
    brow = float(((h["inner_brow_raiser"] + h["outer_brow_raiser"]) / 2).max())
    if nod >= 3.0:
        return "enthusiastic_ack" if (smile >= 0.5 or brow >= 0.5 or nod >= 8.0) else "neutral_ack"
    if smile >= 0.6 and brow >= 0.4:
        return "enthusiastic_ack"
    return "no_signal"


def observer_probabilities(robot_chunk: np.ndarray, start: np.ndarray) -> np.ndarray:
    """SYNTHETIC observer: P(label) for realized robot motion [10, 27] (physical units).

    Encodes assumptions, not measurements: a nod reads as acknowledgment; a look-down
    with lowered lids or an eyebrow flash reads as a weaker acknowledgment; smiles,
    brow raises and large nods read as enthusiastic; jerky motion reads as unclear.
    Replace with observer votes from gr-votes for any claim about people.
    """
    q = np.vstack([start[None], robot_chunk])
    c = {name: q[:, i] for i, name in enumerate(CHANNELS)}
    nod = _largest_drop(np.degrees(c["neck_pitch"]))
    look_down = _largest_drop(np.degrees((c["left_eye_pitch"] + c["right_eye_pitch"]) / 2))
    lids = _largest_drop((c["left_eyelid_close"] + c["right_eyelid_close"]) / 2) / 0.006
    brow = float(((c["left_brow_raise"] + c["right_brow_raise"]) / 2 - (q[0, 7] + q[0, 8]) / 2).max()) / 0.005
    smile = float(((c["mouth_left_corner"] + c["mouth_right_corner"]) / 2).max()) / 0.005
    eye_ack = 0.5 * min(look_down / 12.0, 1.5) + 0.5 * min(lids, 1.5)
    ack = max(nod / 3.0, 0.75 * eye_ack, 0.6 * brow)
    # Mean normalized second difference. Retargeted human motion sits below 0.12 (90th pct,
    # Seamless dev); only motion rougher than real faces counts toward "unclear".
    jitter = max(0.0, float(np.abs(np.diff(q / SCALE, n=2, axis=0)).mean()) - 0.12) / 0.15
    enthusiasm = 0.5 * max(smile, 0) + 0.5 * max(brow, 0) + 0.3 * nod / 8.0
    signal = 1 / (1 + np.exp(-5 * (ack - 0.7)))
    unclear = min(0.85, 0.05 + 0.6 * min(jitter, 1.0) + 0.2 * np.exp(-((ack - 0.6) / 0.2) ** 2))
    enthusiastic = 1 / (1 + np.exp(-5 * (enthusiasm - 0.6)))
    p = np.array([(1 - signal), signal * (1 - enthusiastic), signal * enthusiastic, 0.0])
    p = (1 - unclear) * p
    p[3] = unclear
    return p / p.sum()
