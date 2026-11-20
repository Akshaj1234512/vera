"""The robot G drives: Samhita Konduri's skinned MuJoCo head (``samhita_sim/robotic_head_rig.xml``).

Every position actuator in the rig is one G output channel (27 channels). Units are
native joint units: radians for the hinge joints (neck, eyes, jaw) and metres for
the slide joints that deform the face skin. Each channel is normalized by
``SCALE`` (the larger magnitude of its range) before it reaches G or R.

The rig file is used unmodified. One load-time adjustment is applied here and
documented because it changes behavior: gravity compensation (MuJoCo ``gravcomp``)
on the head, eyes and 20 skin-bone bodies, standing in for the holding torque of
real servos.

* Without it the neck cannot hold the head: a -0.2 rad pitch command falls to the
  -0.45 rad joint limit and roll falls to its 0.35 rad limit (4 kg head, center of
  mass 12 cm above the pivot, neck kp 5-8). With it every neck axis settles on its
  command (90% rise 0.24-0.34 s).
* Without it gravity (mg/kp ~= 2.8 mm) keeps upward face sliders from reaching their
  commands (brows reached 67%, mouth corners 60%, nose wrinkle 22%).

Ranges, gains and damping are Samhita's.
"""

from pathlib import Path
import xml.etree.ElementTree as ElementTree

import numpy as np

RIG_PATH = Path(__file__).resolve().parents[2] / "samhita_sim" / "robotic_head_rig.xml"

# Joint names in G's channel order, grouped so the meters and docs read naturally.
CHANNELS = (
    "neck_yaw", "neck_pitch", "neck_roll",
    "left_eye_yaw", "right_eye_yaw", "left_eye_pitch", "right_eye_pitch",
    "left_brow_raise", "right_brow_raise", "left_eyelid_close", "right_eyelid_close",
    "jaw_open", "mouth_left_corner", "mouth_right_corner",
    "upper_lip_raise", "upper_lip_left", "upper_lip_right",
    "lower_lip_drop", "lower_lip_left", "lower_lip_right",
    "left_cheek_raise", "right_cheek_raise",
    "left_nostril_flare", "right_nostril_flare", "left_nose_wrinkle", "right_nose_wrinkle",
    "chin_protrude",
)
GROUPS = {
    "neck": ("neck_yaw", "neck_pitch", "neck_roll"),
    "eyes": ("left_eye_yaw", "right_eye_yaw", "left_eye_pitch", "right_eye_pitch"),
    "brows": ("left_brow_raise", "right_brow_raise"),
    "lids": ("left_eyelid_close", "right_eyelid_close"),
    "jaw": ("jaw_open",),
    "mouth": ("mouth_left_corner", "mouth_right_corner", "upper_lip_raise", "upper_lip_left", "upper_lip_right",
              "lower_lip_drop", "lower_lip_left", "lower_lip_right", "chin_protrude"),
    "cheeks/nose": ("left_cheek_raise", "right_cheek_raise", "left_nostril_flare", "right_nostril_flare",
                    "left_nose_wrinkle", "right_nose_wrinkle"),
}
# Simulator speed limits (units per second) used in the capability descriptor. Neck
# and jaw follow typical human peak speeds; eyes allow saccades; skin sliders are
# limited so a full-range move takes about 0.15 s.
SPEED = {"neck": 2.0, "eyes": 6.0, "jaw": 3.0, "slide": 0.08}

PROFILES = {
    "full": (),
    "fixed-neck-pitch": ("neck_pitch",),
    "fixed-neck": GROUPS["neck"],
    "fixed-eyes": GROUPS["eyes"],
}


def _parse_rig(path: Path = RIG_PATH):
    root = ElementTree.parse(path).getroot()
    joints = {j.get("name"): j for j in root.iter("joint")}
    defaults = {d.get("class"): d for d in root.iter("default") if d.get("class")}
    actuators, ranges, units = {}, {}, {}
    for act in root.find("actuator"):
        joint = act.get("joint")
        ctrl = act.get("ctrlrange") or defaults[act.get("class")].find("position").get("ctrlrange")
        actuators[joint] = act.get("name")
        ranges[joint] = tuple(float(x) for x in ctrl.split())
        units[joint] = "m" if joints[joint].get("type") == "slide" else "rad"
    missing = set(CHANNELS) - set(actuators)
    if missing:
        raise ValueError(f"{path}: rig lacks actuators for {sorted(missing)}")
    return actuators, ranges, units


ACTUATORS, RANGES, UNITS = _parse_rig()


def _speed(name: str) -> float:
    if UNITS[name] == "m":
        return SPEED["slide"]
    for group in ("neck", "eyes", "jaw"):
        if name in GROUPS[group]:
            return SPEED[group]
    raise KeyError(name)


SIM_LIMITS = np.array([[1, RANGES[c][0], RANGES[c][1], _speed(c)] for c in CHANNELS], dtype=np.float32)
SCALE = np.maximum(np.abs(SIM_LIMITS[:, 1]), np.abs(SIM_LIMITS[:, 2])).astype(np.float32)


def compile_rig(path: Path = RIG_PATH):
    """Compile the rig with gravity compensation on the head, eyes and skin bones."""
    import mujoco

    spec = mujoco.MjSpec.from_file(str(path))
    for body in spec.bodies:
        if body.name.endswith("_bone") or body.name in ("head", "left_eye", "right_eye"):
            body.gravcomp = 1
    return spec.compile()


def calibration_json() -> dict:
    """The simulator calibration in the ``--calibration`` JSON shape."""
    return {name: SIM_LIMITS[i].tolist() for i, name in enumerate(CHANNELS)}
