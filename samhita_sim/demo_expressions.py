"""Interactive MuJoCo visualization for the robotic head rig (27 actuated DOFs in the supplied XML).

Run from the directory containing robotic_head_rig.xml:
    mjpython demo_expressions.py                 # interactive sliders; neutral by default
    mjpython demo_expressions.py --demo          # play the expression sequence, then allow manual control
    mjpython demo_expressions.py --demo --loop   # repeat the expression sequence
    mjpython demo_expressions.py --model path/to/robotic_head_rig.xml

The MuJoCo viewer's Control panel exposes one control for every actuator in
the loaded model. The supplied robotic_head_rig.xml defines 27 joints and 27
position actuators. This script discovers them from the XML rather than
hard-coding a DOF count, so it also adapts if the rig changes.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer


MODEL_PATH = Path(__file__).resolve().with_name("robotic_head_rig.xml")

# These are the expression commands from the original demo. An entry is used
# only when that actuator exists in the loaded model; unmatched names are
# reported instead of crashing the entire visualization.
EXPRESSIONS = {
    "neutral": {},
    "smile": {
        "mouth_left_corner_act": 0.005,
        "mouth_right_corner_act": 0.005,
        "left_cheek_raise_act": 0.004,
        "right_cheek_raise_act": 0.004,
    },
    "frown": {
        "mouth_left_corner_act": -0.004,
        "mouth_right_corner_act": -0.004,
        "left_brow_raise_act": -0.002,
        "right_brow_raise_act": -0.002,
    },
    "surprised": {
        "jaw_open_act": 0.22,
        "left_brow_raise_act": 0.008,
        "right_brow_raise_act": 0.008,
        "upper_lip_raise_act": 0.002,
    },
    "squint": {
        "left_eyelid_close_act": -0.008,
        "right_eyelid_close_act": -0.008,
        "left_cheek_raise_act": 0.003,
        "right_cheek_raise_act": 0.003,
    },
    "disgust": {
        "left_nose_wrinkle_act": 0.004,
        "right_nose_wrinkle_act": 0.004,
        "left_brow_raise_act": -0.002,
        "right_brow_raise_act": -0.002,
    },
}


def actuator_name(model: mujoco.MjModel, actuator_id: int) -> str:
    return (
        mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
        )
        or f"actuator_{actuator_id}"
    )


def joint_qpos_for_actuator(model: mujoco.MjModel, actuator_id: int) -> int | None:
    """Return the qpos address for a joint-transmission actuator, if applicable."""
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    if joint_id < 0 or joint_id >= model.njnt:
        return None
    return int(model.jnt_qposadr[joint_id])


def neutral_targets(model: mujoco.MjModel, data: mujoco.MjData):
    """Build actuator targets from the model's initial joint positions."""
    targets = data.ctrl.copy()
    for aid in range(model.nu):
        qadr = joint_qpos_for_actuator(model, aid)
        if qadr is not None:
            value = float(data.qpos[qadr])
            if model.actuator_ctrllimited[aid]:
                lo, hi = model.actuator_ctrlrange[aid]
                value = min(max(value, float(lo)), float(hi))
            targets[aid] = value
    return targets


def set_neutral_targets(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Reset once, then hold every joint at its initial pose."""
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = neutral_targets(model, data)
    mujoco.mj_forward(model, data)


def set_expression(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    name: str,
    actuator_ids: dict[str, int],
    neutral_ctrl,
) -> None:
    """Apply one expression, returning unspecified actuators to neutral."""
    data.ctrl[:] = neutral_ctrl
    missing = []
    for actuator, value in EXPRESSIONS[name].items():
        aid = actuator_ids.get(actuator)
        if aid is None:
            missing.append(actuator)
            continue
        value = float(value)
        if model.actuator_ctrllimited[aid]:
            lo, hi = model.actuator_ctrlrange[aid]
            value = min(max(value, float(lo)), float(hi))
        data.ctrl[aid] = value
    if missing:
        print(f"  Skipped actuators not present in this model: {', '.join(missing)}")


def run_interactive(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Run a passive viewer and synchronize it continuously for manual control.

    Using launch_passive avoids the launch() convenience wrapper's internal
    viewer startup path. The explicit sync is important: it transfers changes
    made in the viewer's Control panel back to the simulation data.
    """
    set_neutral_targets(model, data)
    print("\nInteractive mode — no expression sequence is playing.")
    print("Use the viewer's Control panel to move the actuators.")
    print("Every actuator in the loaded XML is exposed there.")
    print("Use the viewer's Reset button / Backspace to restore the initial state.")
    print("Close the viewer window to quit.\n")

    # On macOS, launch_passive should be started with `mjpython` so the GUI
    # event loop runs on the main thread.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            with viewer.lock():
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


def run_demo(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_ids: dict[str, int],
    loop: bool,
    interval: float,
) -> None:
    """Play expressions in the passive viewer, then leave it available for control."""
    set_neutral_targets(model, data)
    neutral_ctrl = data.ctrl.copy()
    names = list(EXPRESSIONS)
    index = 0
    next_change = time.perf_counter()
    demo_finished = False

    print("\nExpression demo is running. Close the viewer to quit.")
    print("After the sequence, use the Control panel to adjust all actuators.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            now = time.perf_counter()
            if not demo_finished and now >= next_change:
                name = names[index]
                set_expression(model, data, name, actuator_ids, neutral_ctrl)
                print(f"Expression: {name}")
                index += 1
                if index >= len(names):
                    if loop:
                        index = 0
                    else:
                        demo_finished = True
                        # Return to neutral once; after this, viewer controls
                        # are left alone and can change any actuator.
                        data.ctrl[:] = neutral_ctrl
                        print("Sequence complete — manual actuator control is enabled.")
                next_change = now + interval

            with viewer.lock():
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="MuJoCo XML model (default: robotic_head_rig.xml beside this script)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="play the built-in expression sequence before manual control",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="repeat the expression sequence (only with --demo)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.5,
        help="seconds to display each expression (default: 2.5)",
    )
    args = parser.parse_args()

    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        parser.error(
            f"Model XML not found: {model_path}\n"
            "Put robotic_head_rig.xml beside this script or pass --model."
        )
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    actuator_ids = {
        actuator_name(model, aid): aid for aid in range(model.nu)
    }

    print(
        f"Loaded {model_path.name}: {model.njnt} joints, "
        f"{model.nu} actuators, {model.nq} qpos values."
    )
    if model.nu != model.njnt:
        print(
            "Note: the number of actuators differs from the number of joints. "
            "Only actuated joints receive sliders in the Control panel."
        )
    print("Actuated joints (viewer Control panel):")
    for aid in range(model.nu):
        act_name = actuator_name(model, aid)
        joint_id = int(model.actuator_trnid[aid, 0])
        if 0 <= joint_id < model.njnt:
            joint_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            ) or f"joint_{joint_id}"
            joint_type = int(model.jnt_type[joint_id])
            # MuJoCo hinge joint values are radians; slide joint values are metres.
            unit = "rad" if joint_type == int(mujoco.mjtJoint.mjJNT_HINGE) else (
                "m" if joint_type == int(mujoco.mjtJoint.mjJNT_SLIDE) else "native units"
            )
        else:
            joint_name, unit = "non-joint transmission", "native units"
        if model.actuator_ctrllimited[aid]:
            lo, hi = model.actuator_ctrlrange[aid]
            bounds = f"[{lo:.4g}, {hi:.4g}] {unit}"
        else:
            bounds = f"(unlimited; {unit})"
        print(f"  {aid:2d}. {joint_name:<24} <- {act_name:<28} {bounds}")

    if args.demo:
        run_demo(model, data, actuator_ids, args.loop, args.interval)
    else:
        run_interactive(model, data)


if __name__ == "__main__":
    main()
