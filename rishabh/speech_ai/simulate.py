"""Replay speech cues on the existing MuJoCo head and log measured motion."""

import csv
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

from .cues import PROFILES

JOINT_RATES = {"jaw": 2.0, "neck_pitch": 1.5, "eye_yaw_l": 2.3, "eye_yaw_r": 2.3}
CUE_TARGETS = {
    "nod": (("neck_pitch",), "pitch", 0.30),
    "gaze": (("eye_yaw_l", "eye_yaw_r"), "yaw", 0.50),
}
FPS = 25


def overlay(pixels: np.ndarray, timeline: dict, now: float, frame_index: int, event: dict | None) -> bytes:
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import get_data_path

    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    # Keep high-contrast status strips away from the face.
    draw.rectangle((0, 0, width, 54), fill="#111a27")
    draw.rectangle((0, height - 39, width, height), fill="#111a27")
    font_path = Path(get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    title = ImageFont.truetype(str(font_path), size=19)
    labels = ImageFont.truetype(str(font_path), size=13)
    draw.text((20, 7), "VERA   /   SPEECH CUE REPLAY", font=title, fill="#f8fafc")
    draw.text((21, 33), timeline["source"] + "  /  " + timeline["profile"], font=labels, fill="#a5b7ce")
    state = timeline["speech_state"][frame_index] if frame_index >= 0 else "silence"
    line = f"{now:05.2f} s   /   {state}"
    if event is not None:
        line += f"   /   {event['role']} · {event['cue']}"
    draw.text((21, height - 28), line, font=labels, fill="#f8fafc")
    duration = timeline["time_s"][-1]
    progress = 20 + round((width - 40) * min(now / duration, 1))
    draw.line((20, height - 7, width - 20, height - 7), fill="#38475a", width=2)
    draw.line((20, height - 7, progress, height - 7), fill="#8edfd3", width=2)
    return image.tobytes()


def replay(timeline_path: Path, output: Path, video: Path | None, audio: Path | None) -> dict:
    import mujoco

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        from simulation.demo import Choreography, HeadController, MODEL_PATH, robot_contacts
    finally:
        sys.path.pop(0)

    timeline = json.loads(timeline_path.read_text())
    timestamps = np.asarray(timeline["time_s"], dtype=float)
    aperture = np.asarray(timeline["jaw_aperture"], dtype=float)
    if (timeline["version"] != 1 or not len(timestamps) or timestamps.shape != aperture.shape
            or not np.isfinite(timestamps).all() or not np.isfinite(aperture).all()
            or np.any(timestamps <= 0) or np.any(np.diff(timestamps) <= 0)
            or np.any((aperture < 0) | (aperture > 1))):
        raise ValueError("Expected an ordered cue timeline with normalized jaw aperture")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    controller = HeadController(model, data)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    allowed = PROFILES[timeline["profile"]]
    score = Choreography([joint for joint in JOINT_RATES if joint != "jaw"])
    for event in timeline["events"]:
        cue = event["cue"]
        onset, duration = float(event["onset_s"]), float(event["duration_s"])
        if not np.isfinite([onset, duration]).all() or onset < 0 or duration <= 0:
            raise ValueError(f"Invalid cue timing: {event}")
        if cue == "settle":
            score.move(duration, at=onset, **dict.fromkeys(score.joints, 0.0))
            continue
        if (cue == "nod" and not allowed.nod) or (cue == "gaze" and not allowed.eyes):
            raise ValueError(f"{cue} is unavailable in {timeline['profile']}")
        joints, key, maximum = CUE_TARGETS[cue]
        level = float(event["target"][key])
        if not np.isfinite(level) or abs(level) > 1:
            raise ValueError(f"Cue target must be normalized: {event}")
        outward = 0.38 * duration
        home = 0.45 * duration
        score.move(outward, at=onset, **dict.fromkeys(joints, maximum * level))
        score.move(home, at=onset + duration - home, **dict.fromkeys(joints, 0.0))

    output.mkdir(parents=True, exist_ok=True)
    traces = output / "joints.csv"
    joints = list(JOINT_RATES)
    controls = np.array([controller.ctrl_range(joint) for joint in joints]).T
    rates = np.array(list(JOINT_RATES.values()))
    dt = model.opt.timestep
    current = np.array([controller.target(joint) for joint in joints])
    extrema = np.zeros(len(joints))
    peak_rates = np.zeros(len(joints))
    peak_error = np.zeros(len(joints))
    contact_steps = 0
    finite = True
    duration = float(timestamps[-1])

    with TemporaryDirectory(dir=output, prefix="video-") as scratch, traces.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time_s", *[f"{joint}_target_rad" for joint in joints],
                         *[f"{joint}_measured_rad" for joint in joints]])
        frame_count = 0
        encoded = Path(scratch) / "silent.mp4"
        renderer, process = None, None
        try:
            if video is not None:
                width, height = 640, 384
                renderer = mujoco.Renderer(model, height=height, width=width)
                process = subprocess.Popen([
                    "ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{width}x{height}", "-r", str(FPS), "-i", "pipe:0", "-c:v", "libx264",
                    "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(encoded),
                ], stdin=subprocess.PIPE)

            period = 1 / FPS
            for _ in range(int(np.ceil(duration / dt))):
                now = float(data.time)
                frame_index = int(np.searchsorted(timestamps, now + 1e-9, side="right")) - 1
                targets = score.sample(now)
                targets["jaw"] = 0.25 * aperture[frame_index] if frame_index >= 0 else 0.0
                wanted = np.clip([targets[joint] for joint in joints], *controls)
                change = np.clip(wanted - current, -rates * dt, rates * dt)
                current += change
                controller.set_many(dict(zip(joints, current)))
                actual = np.asarray([controller.position(joint) for joint in joints])
                extrema = np.maximum(extrema, np.abs(actual))
                peak_rates = np.maximum(peak_rates, np.abs(change / dt))
                peak_error = np.maximum(peak_error, np.abs(actual - current))
                finite &= bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
                contact_steps += bool(robot_contacts(model, data))

                if now + 1e-8 >= frame_count * period:
                    writer.writerow([f"{now:.4f}", *np.round(current, 6), *np.round(actual, 6)])
                    if process is not None:
                        active = next((event for event in reversed(timeline["events"])
                                       if event["onset_s"] <= now < event["onset_s"] + event["duration_s"]), None)
                        renderer.update_scene(data, camera="three_quarter")
                        process.stdin.write(overlay(renderer.render(), timeline, now, frame_index, active))
                    frame_count += 1
                mujoco.mj_step(model, data)
        finally:
            if renderer is not None:
                renderer.close()
            if process is not None:
                process.stdin.close()
                process.wait()
        if process is not None:
            if process.returncode != 0:
                raise RuntimeError("ffmpeg could not encode MuJoCo frames")
            video.parent.mkdir(parents=True, exist_ok=True)
            if audio is None:
                encoded.replace(video)
            else:
                subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(encoded), "-i", str(audio),
                                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                                "-movflags", "+faststart", "-shortest", str(video)], check=True)

    report = {
        "duration_s": duration, "model": MODEL_PATH.name, "profile": timeline["profile"],
        "finite_state": finite, "self_contact_steps": contact_steps,
        "peak_abs_position_rad": dict(zip(joints, np.round(extrema, 6))),
        "peak_command_rad_s": dict(zip(joints, np.round(peak_rates, 6))),
        "peak_tracking_error_rad": dict(zip(joints, np.round(peak_error, 6))),
    }
    (output / "simulation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
