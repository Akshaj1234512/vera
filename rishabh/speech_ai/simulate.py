"""MuJoCo adapter for ``RobotInterface``: VERA's simulated head as the G/R embodiment."""

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

from .gr_model import ACTUATORS, CHANNELS
from .gr_runtime import RobotInterface

FPS = 25
WIDTH, HEIGHT = 640, 384


def _simulation():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        from simulation.demo import HeadController, MODEL_PATH, robot_contacts
    finally:
        sys.path.pop(0)
    return HeadController, MODEL_PATH, robot_contacts


class MujocoRobot(RobotInterface):
    """Position actuators with per-channel speed limits; measured state from qpos/qvel."""

    def __init__(self, limits: np.ndarray, video: Path | None = None, audio: Path | None = None,
                 title: str = "", output: Path | None = None):
        import mujoco

        self.mujoco = mujoco
        HeadController, self.model_path, self.robot_contacts = _simulation()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.controller = HeadController(self.model, self.data)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.limits = limits
        self.groups = [ACTUATORS[name] for name in CHANNELS]
        self.joints = [joint for group in self.groups for joint in group]
        for group, (_, low, high, _) in zip(self.groups, limits):
            for joint in group:
                lower, upper = self.controller.ctrl_range(joint)
                if low < lower - 1e-7 or high > upper + 1e-7:
                    raise ValueError(f"{joint}: calibration extends beyond the simulator limits")
        self.rates = np.array([limits[i, 3] for i, group in enumerate(self.groups) for _ in group])
        self.qpos = [self.controller.jnt_qadr[joint] for joint in self.joints]
        self.qvel = [self.model.jnt_dofadr[self.model.actuator_trnid[self.controller.act_id[joint], 0]]
                     for joint in self.joints]
        self.current = np.array([self.controller.target(joint) for joint in self.joints])
        self.wanted = self.current.copy()
        self.dt = self.model.opt.timestep
        self.contact_steps = 0
        self.finite = True
        self.peak_rate = np.zeros(len(self.joints))
        self.video, self.audio, self.title = video, audio, title
        self.caption = ""
        self.frame_count = 0
        self.renderer = self.process = self.scratch = None
        if video is not None:
            if output is not None:
                output.mkdir(parents=True, exist_ok=True)
            self.scratch = TemporaryDirectory(dir=output, prefix="video-")
            self.encoded = Path(self.scratch.name) / "silent.mp4"
            self.renderer = mujoco.Renderer(self.model, height=HEIGHT, width=WIDTH)
            self.process = subprocess.Popen([
                "ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "pipe:0", "-c:v", "libx264",
                "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.encoded),
            ], stdin=subprocess.PIPE)

    def _channels(self, values: np.ndarray) -> np.ndarray:
        result, k = [], 0
        for group in self.groups:
            result.append(np.mean(values[k:k + len(group)]))
            k += len(group)
        return np.asarray(result, dtype=np.float32)

    def state(self) -> np.ndarray:
        return np.stack([self._channels(self.data.qpos[self.qpos]), self._channels(self.data.qvel[self.qvel])])

    def command(self, target_rad: np.ndarray) -> None:
        self.wanted = np.array([target_rad[i] for i, group in enumerate(self.groups) for _ in group])

    def advance(self, seconds: float) -> None:
        end = self.data.time + seconds - 1e-9
        while self.data.time < end:
            if self.process is not None and self.data.time + 1e-8 >= self.frame_count / FPS:
                self._render()
            change = np.clip(self.wanted - self.current, -self.rates * self.dt, self.rates * self.dt)
            self.current += change
            self.peak_rate = np.maximum(self.peak_rate, np.abs(change / self.dt))
            self.controller.set_many(dict(zip(self.joints, self.current)))
            self.mujoco.mj_step(self.model, self.data)
            self.finite &= bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
            self.contact_steps += bool(self.robot_contacts(self.model, self.data))

    def annotate(self, index: int, session) -> None:
        if self.process is None:
            return
        chunk = session.chunks[-1]
        line = f"{self.data.time:05.2f} s   /   chunk {chunk['start_s']:.1f} s"
        if chunk["fallback"]:
            line += f"   /   {chunk['fallback'].replace('_', ' ')}"
        scored = [c for c in session.chunks if c.get("r_probabilities_on_realized_motion")]
        if scored:
            probabilities = scored[-1]["r_probabilities_on_realized_motion"]
            best = max(probabilities, key=probabilities.get)
            line += f"   /   R(last chunk): {best.replace('_', ' ')} {probabilities[best]:.2f}"
        self.caption = line

    def _render(self) -> None:
        from PIL import Image, ImageDraw, ImageFont
        from matplotlib import get_data_path

        self.renderer.update_scene(self.data, camera="three_quarter")
        image = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(image)
        font_path = Path(get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
        draw.rectangle((0, 0, WIDTH, 54), fill="#111a27")
        draw.rectangle((0, HEIGHT - 39, WIDTH, HEIGHT), fill="#111a27")
        draw.text((20, 7), "VERA   /   G/R SPEECH MOTION", font=ImageFont.truetype(str(font_path), 19), fill="#f8fafc")
        small = ImageFont.truetype(str(font_path), 13)
        draw.text((21, 33), self.title, font=small, fill="#a5b7ce")
        draw.text((21, HEIGHT - 28), self.caption, font=small, fill="#f8fafc")
        self.process.stdin.write(image.tobytes())
        self.frame_count += 1

    def close(self) -> dict:
        if self.renderer is not None:
            self.renderer.close()
        if self.process is not None:
            self.process.stdin.close()
            self.process.wait()
            if self.process.returncode != 0:
                raise RuntimeError("ffmpeg could not encode MuJoCo frames")
            self.video.parent.mkdir(parents=True, exist_ok=True)
            if self.audio is None:
                self.encoded.replace(self.video)
            else:
                subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(self.encoded),
                                "-i", str(self.audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                                "-c:a", "aac", "-movflags", "+faststart", "-shortest", str(self.video)], check=True)
            self.scratch.cleanup()
            self.process = None
        return {"model": self.model_path.name, "duration_s": round(float(self.data.time), 4),
                "finite_state": self.finite, "self_contact_steps": int(self.contact_steps),
                "peak_command_rad_s": dict(zip(self.joints, np.round(self.peak_rate, 5).tolist()))}
