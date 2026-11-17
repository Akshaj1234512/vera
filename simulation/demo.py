"""VERA humanoid head - interactive control, optional scripted demo, validation.

Usage (from ~/VERA/simulation):
    ../.venv/bin/python   demo.py                 # DEFAULT: viewer, head holds still at neutral;
                                                  #   move joints with the Control-panel sliders
    ../.venv/bin/mjpython demo.py --demo          # opt-in scripted animation, then slider control
    ../.venv/bin/mjpython demo.py --demo --loop   # repeat the scripted animation
    ../.venv/bin/python   demo.py --headless      # no graphics: validate model + choreography

Nothing moves automatically unless --demo is given. On macOS the scripted demo
uses the passive viewer and therefore must be started with `mjpython`.

Extending: add the joint + <position> actuator in head.xml, then add an entry
to JOINTS below. Choreography is built with the `Choreography` helper.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "head.xml"

# Logical joint name -> actuator name in head.xml. Add future DOF here
# (e.g. "eye_yaw": "eye_yaw", "brow_l": "brow_l").
JOINTS: dict[str, str] = {
    "neck_yaw": "neck_yaw",      # + = turn to robot's left
    "neck_pitch": "neck_pitch",  # + = nod down (chin toward chest)
    "jaw": "jaw",                # + = open
    "eyelid_l": "eyelid_l",      # + = close
    "eyelid_r": "eyelid_r",      # + = close
    "eye_yaw_l": "eye_yaw_l",    # + = look to robot's left
    "eye_pitch_l": "eye_pitch_l",  # + = look down
    "eye_yaw_r": "eye_yaw_r",
    "eye_pitch_r": "eye_pitch_r",
    "brow_l": "brow_l",          # + = raise (rad about forehead centre; 0.032 ~ 3 mm)
    "brow_r": "brow_r",
}
EYELIDS = ("eyelid_l", "eyelid_r")
LID_CLOSED = 0.78   # rad; lid margin passes the lower lid (joint limit 0.8)
LID_OPEN = 0.0


# --------------------------------------------------------------------------- #
# Controller
# --------------------------------------------------------------------------- #
class HeadController:
    """Thin wrapper mapping logical joint names to position-actuator targets."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, joints: dict[str, str] = JOINTS):
        self.model, self.data = model, data
        self.act_id = {}
        self.jnt_qadr = {}
        for name, act in joints.items():
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act)
            if aid < 0:
                raise KeyError(f"actuator '{act}' not found in model")
            self.act_id[name] = aid
            self.jnt_qadr[name] = model.jnt_qposadr[model.actuator_trnid[aid, 0]]

    @property
    def names(self) -> list[str]:
        return list(self.act_id)

    def ctrl_range(self, name: str) -> tuple[float, float]:
        lo, hi = self.model.actuator_ctrlrange[self.act_id[name]]
        return float(lo), float(hi)

    def set(self, name: str, value: float) -> None:
        lo, hi = self.ctrl_range(name)
        self.data.ctrl[self.act_id[name]] = float(np.clip(value, lo, hi))

    def target(self, name: str) -> float:
        return float(self.data.ctrl[self.act_id[name]])

    def position(self, name: str) -> float:
        return float(self.data.qpos[self.jnt_qadr[name]])

    def set_many(self, targets: dict[str, float]) -> None:
        for k, v in targets.items():
            self.set(k, v)

    def neutral(self) -> None:
        for name in self.names:
            self.set(name, 0.0)


# --------------------------------------------------------------------------- #
# Choreography: per-joint min-jerk segments
# --------------------------------------------------------------------------- #
def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s**3 * (10 - 15 * s + 6 * s * s)


@dataclass
class Segment:
    t0: float
    duration: float
    start: float
    end: float


@dataclass
class Choreography:
    """Build a timeline of smooth joint moves.

    Each joint has its own track, so moves on different joints may overlap
    (e.g. talking while turning). `cursor` is where the next sequential move
    starts; `move(..., at=t)` places a move at an absolute time instead.
    """

    joints: list[str]
    cursor: float = 0.0
    tracks: dict[str, list[Segment]] = field(default_factory=dict)
    _last: dict[str, float] = field(default_factory=dict)
    labels: list[tuple[float, str]] = field(default_factory=list)

    def __post_init__(self):
        for j in self.joints:
            self.tracks[j] = []
            self._last[j] = 0.0

    def label(self, text: str) -> "Choreography":
        self.labels.append((self.cursor, text))
        return self

    def move(self, duration: float, at: float | None = None, advance: bool = True, **targets: float) -> "Choreography":
        t0 = self.cursor if at is None else at
        for j, v in targets.items():
            start = self._value(j, t0)          # continue smoothly from wherever the track is at t0
            self.tracks[j] = [sg for sg in self.tracks[j] if sg.t0 < t0]
            self.tracks[j].append(Segment(t0, duration, start, v))
            self._last[j] = v
        if advance and at is None:
            self.cursor += duration
        return self

    def hold(self, duration: float) -> "Choreography":
        self.cursor += duration
        return self

    def blink(self, lids=EYELIDS, close=0.09, closed_hold=0.05, open_=0.16, depth=LID_CLOSED) -> "Choreography":
        self.move(close, **{l: depth for l in lids})
        self.hold(closed_hold)
        self.move(open_, **{l: LID_OPEN for l in lids})
        return self

    def to_neutral(self, duration: float) -> "Choreography":
        return self.move(duration, **{j: 0.0 for j in self.joints})

    @property
    def duration(self) -> float:
        ends = [s.t0 + s.duration for tr in self.tracks.values() for s in tr]
        return max([self.cursor, *ends])

    def _value(self, joint: str, t: float) -> float:
        val = 0.0
        for s in self.tracks[joint]:  # segments are kept in time order per joint
            if t < s.t0:
                break
            val = s.start + (s.end - s.start) * min_jerk((t - s.t0) / s.duration)
        return val

    def sample(self, t: float) -> dict[str, float]:
        return {j: self._value(j, t) for j in self.tracks}


def build_demo(joints: list[str]) -> Choreography:
    c = Choreography(joints)
    c.label("neutral").hold(1.0)

    c.label("head rotation (yaw)")
    c.move(1.2, neck_yaw=0.9).hold(0.4)
    c.move(2.0, neck_yaw=-0.9).hold(0.4)
    c.move(1.2, neck_yaw=0.0).hold(0.3)

    c.label("nodding (pitch)")
    for _ in range(2):
        c.move(0.7, neck_pitch=0.45).move(0.9, neck_pitch=-0.35)
    c.move(0.7, neck_pitch=0.0).hold(0.3)

    c.label("jaw open / close")
    c.move(0.7, jaw=0.42).hold(0.5).move(0.6, jaw=0.0).hold(0.3)
    c.label("talking")
    for a in (0.18, 0.08, 0.25, 0.05, 0.2, 0.12, 0.22):
        c.move(0.18, jaw=a)
    c.move(0.2, jaw=0.0).hold(0.4)

    c.label("synchronized blinking")
    for _ in range(3):
        c.blink().hold(0.45)

    c.label("independent eyelids: wink left, wink right")
    c.blink(lids=("eyelid_l",), closed_hold=0.3).hold(0.5)
    c.blink(lids=("eyelid_r",), closed_hold=0.3).hold(0.5)
    c.label("half-lidded (squint)")
    c.move(0.4, eyelid_l=0.4, eyelid_r=0.4).hold(0.8).move(0.3, eyelid_l=0.0, eyelid_r=0.0).hold(0.4)

    c.label("gaze: look left, right, up, down (conjugate)")
    for yaw, pitch in ((0.45, 0.0), (-0.45, 0.0), (0.0, -0.35), (0.0, 0.35), (0.0, 0.0)):
        c.move(0.25, eye_yaw_l=yaw, eye_yaw_r=yaw, eye_pitch_l=pitch, eye_pitch_r=pitch).hold(0.5)
    c.label("independent eyes: converge (cross), then diverge")
    c.move(0.35, eye_yaw_l=-0.25, eye_yaw_r=0.25).hold(0.6)
    c.move(0.35, eye_yaw_l=0.2, eye_yaw_r=-0.2).hold(0.5).move(0.3, eye_yaw_l=0.0, eye_yaw_r=0.0)
    c.label("eyebrows: raise, lower, asymmetric")
    c.move(0.4, brow_l=0.032, brow_r=0.032).hold(0.5).move(0.4, brow_l=-0.016, brow_r=-0.016).hold(0.5)
    c.move(0.4, brow_l=0.032, brow_r=-0.008).hold(0.6).move(0.4, brow_l=0.0, brow_r=0.0).hold(0.3)

    c.label("combined: look around while talking and blinking")
    t = c.cursor
    c.move(1.5, at=t, neck_yaw=0.6, neck_pitch=-0.2)
    c.move(0.25, at=t - 0.1, eye_yaw_l=0.3, eye_yaw_r=0.3)           # eyes lead the head
    c.move(1.5, at=t + 1.7, neck_yaw=-0.5, neck_pitch=0.25)
    c.move(0.25, at=t + 1.6, eye_yaw_l=-0.3, eye_yaw_r=-0.3)
    c.move(0.6, at=t + 3.3, eye_yaw_l=0.0, eye_yaw_r=0.0)
    tt = t
    for a in (0.2, 0.05, 0.28, 0.1, 0.22, 0.0, 0.18, 0.06, 0.25, 0.0):
        c.move(0.16, at=tt, jaw=a)
        tt += 0.32
    c.cursor = t + 1.2
    c.blink()
    c.cursor = t + 3.2
    c.blink()

    c.label("return to neutral")
    c.to_neutral(1.5).hold(1.0)
    return c


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
EYE_SURFACES = ("sclera", "iris", "cornea")
VISIBLE_GROUPS = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)  # skip hidden collision proxies


def eye_covered(model, data, side: str) -> bool:
    """Cast a 3x3 grid of rays over the iris (from in front of the eye toward
    its centre). The eye counts as covered when no ray reaches the eyeball."""
    eye = data.body(f"eye_{side}")
    R = eye.xmat.reshape(3, 3)
    x, y, z = R[:, 0], R[:, 1], R[:, 2]
    geomid = np.array([-1], dtype=np.int32)
    for dy in (-0.004, 0.0003, 0.004):
        for dz in (-0.004, 0.0003, 0.004):
            start = eye.xpos + 0.05 * x + dy * y + dz * z
            mujoco.mj_ray(model, data, start, -x, VISIBLE_GROUPS, 1, -1, geomid)
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geomid[0])) or ""
            if name.startswith(EYE_SURFACES):
                return False
    return True


def robot_contacts(model, data) -> list[tuple[str, str]]:
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        if floor in (c.geom1, c.geom2):
            continue
        out.append((model.geom(c.geom1).name, model.geom(c.geom2).name))
    return out


def check_stationary(model, seconds: float = 10.0) -> float:
    """Simulate from the neutral keyframe with no commands; return max |q|."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    worst = 0.0
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        worst = max(worst, float(np.abs(data.qpos).max()))
    return worst


def collision_relevant_joints(model) -> list[int]:
    """Joints whose motion moves at least one collision geom (contype/conaffinity > 0)."""
    moves = set()
    for g in range(model.ngeom):
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        b = model.geom_bodyid[g]
        while b > 0:
            for j in range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b]):
                moves.add(j)
            b = model.body_parentid[b]
    return sorted(moves)


def check_limit_corners(model) -> tuple[int, int]:
    """Sweep {min, 0, max} of every collision-relevant joint, with all remaining
    joints held at their minimum and at their maximum. Returns (#configs, #bad)."""
    data = mujoco.MjData(model)
    rel = collision_relevant_joints(model)
    other = [j for j in range(model.njnt) if j not in rel]
    bad = n = 0
    for others_at in (0, 1):
        for idx in np.ndindex(*(3,) * len(rel)):
            data.qpos[:] = 0
            for j, i in zip(rel, idx):
                data.qpos[model.jnt_qposadr[j]] = (model.jnt_range[j][0], 0.0, model.jnt_range[j][1])[i]
            for j in other:
                data.qpos[model.jnt_qposadr[j]] = model.jnt_range[j][others_at]
            mujoco.mj_forward(model, data)
            n += 1
            bad += bool(robot_contacts(model, data))
    return n, bad


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #
def run_headless(model, data, ctl: HeadController, show: Choreography) -> bool:
    mujoco.mj_resetDataKeyframe(model, data, 0)
    n = int(np.ceil(show.duration / model.opt.timestep))
    max_err = {j: 0.0 for j in ctl.names}
    max_q = {j: -np.inf for j in ctl.names}
    contacts, covered_while_closed, closed_samples = set(), {s: 0 for s in "lr"}, {s: 0 for s in "lr"}
    for _ in range(n):
        tgt = show.sample(data.time)
        ctl.set_many(tgt)
        mujoco.mj_step(model, data)
        for j in ctl.names:
            max_err[j] = max(max_err[j], abs(ctl.position(j) - tgt[j]))
            max_q[j] = max(max_q[j], ctl.position(j))
        contacts.update(robot_contacts(model, data))
        for s in "lr":
            if ctl.position(f"eyelid_{s}") > LID_CLOSED - 0.04:
                closed_samples[s] += 1
                covered_while_closed[s] += eye_covered(model, data, s)
    # settle at neutral
    ctl.neutral()
    for _ in range(500):
        mujoco.mj_step(model, data)

    ok = True
    print(f"Simulated {data.time:.2f} s ({n} steps @ {model.opt.timestep*1e3:.0f} ms)")
    print(f"{'joint':<12}{'max |err| (rad)':>18}{'peak q (rad)':>15}{'final q (rad)':>15}")
    for j in ctl.names:
        fq = ctl.position(j)
        print(f"{j:<12}{max_err[j]:>18.4f}{max_q[j]:>15.3f}{fq:>15.4f}")
        ok &= abs(fq) < 1e-2
    finite = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
    print(f"state finite: {finite}")
    print(f"self-contacts during demo: {sorted(contacts) or 'none'}")
    for s in "lr":
        frac = covered_while_closed[s] / max(closed_samples[s], 1)
        print(f"eye_{s}: covered by lid in {frac*100:.0f}% of {closed_samples[s]} closed-lid samples")
        ok &= frac > 0.99 and closed_samples[s] > 0
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    print(f"eyes covered at neutral (should be False): {eye_covered(model, data, 'l')}, {eye_covered(model, data, 'r')}")
    still = check_stationary(model)
    print(f"stationary without input (10 s at neutral): max |q| = {still:.2e} rad")
    ok &= still < 1e-3
    n_cfg, bad = check_limit_corners(model)
    names = [model.joint(j).name for j in collision_relevant_joints(model)]
    print(f"joint-limit sweep over {names} (+ all other joints at min/max): {n_cfg} configs, {bad} with self-contact")
    ok &= finite and not contacts and bad == 0
    print("VALIDATION", "PASSED" if ok else "FAILED")
    return ok


def run_viewer(model, data, ctl: HeadController, show: Choreography, loop: bool) -> None:
    import mujoco.viewer

    mujoco.mj_resetDataKeyframe(model, data, 0)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "three_quarter")
        labels = list(show.labels)
        demo_running, t_demo0 = True, data.time
        print(f"Demo: {show.duration:.1f} s. Close the window to quit.")
        wall_start, sim_start = time.perf_counter(), data.time
        while viewer.is_running():
            wall0 = time.perf_counter()
            with viewer.lock():
                if demo_running:
                    t = data.time - t_demo0
                    while labels and labels[0][0] <= t:
                        print(f"  [{t:5.1f}s] {labels.pop(0)[1]}")
                    if t <= show.duration:
                        ctl.set_many(show.sample(t))
                    elif loop:
                        t_demo0, labels = data.time, list(show.labels)
                    else:
                        ctl.neutral()
                        demo_running = False
                        print("Demo finished; all joints at neutral.\n"
                              "Use the viewer's Control panel (right side) to move each actuator.\n"
                              "Press Tab / Esc in the viewer to free the camera.")
                # advance physics to match wall-clock time (real-time pacing)
                target = sim_start + (time.perf_counter() - wall_start)
                while data.time < target:
                    mujoco.mj_step(model, data)
            viewer.sync()
            dt = (1 / 60) - (time.perf_counter() - wall0)
            if dt > 0:
                time.sleep(dt)


def run_interactive(model, data) -> None:
    """Viewer only. The head holds its neutral pose (position servos at 0)
    until an actuator slider is moved in the viewer's Control panel."""
    import mujoco.viewer

    mujoco.mj_resetDataKeyframe(model, data, 0)
    print("Interactive mode - the head is stationary at neutral.\n"
          "Open the Control panel (right side) and drag the actuator sliders:\n"
          "  neck_yaw [-1.4, 1.4]  neck_pitch [-0.5, 0.5]  jaw [-0.02, 0.45] (~26 deg)\n"
          "  eyelid_l / eyelid_r [-0.15, 0.8]   (rad; + = left / nod down / open / close)\n"
          "  eye_yaw_l/_r [-0.6, 0.6]  eye_pitch_l/_r [-0.45, 0.45]  (rad; + = look left / down)\n"
          "  brow_l / brow_r [-0.016, 0.032]  (rad about forehead centre; + = raise, 0.032 ~ 3 mm)\n"
          "Press Backspace in the viewer to reset to neutral.")
    mujoco.viewer.launch(model, data)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--demo", action="store_true", help="play the scripted animation (use mjpython on macOS)")
    g.add_argument("--headless", action="store_true", help="validate model + choreography without graphics")
    g.add_argument("--interactive", action="store_true", help="(default) viewer with slider control only")
    p.add_argument("--loop", action="store_true", help="with --demo: repeat the animation")
    p.add_argument("--model", default=str(MODEL_PATH))
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    print(f"Loaded {Path(args.model).name}: {model.nbody - 1} bodies, {model.njnt} joints, {model.nu} actuators")
    ctl = HeadController(model, data)
    show = build_demo(ctl.names)

    if args.headless:
        raise SystemExit(0 if run_headless(model, data, ctl, show) else 1)
    if args.demo:
        run_viewer(model, data, ctl, show, args.loop)
    else:
        run_interactive(model, data)


if __name__ == "__main__":
    main()
