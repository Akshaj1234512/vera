"""MuJoCo adapter for ``RobotInterface``: Samhita's skinned head rig as the G/R embodiment."""

from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from .embodiment import CHANNELS, GROUPS, SCALE, compile_rig
from .gr_runtime import RobotInterface

FPS = 25
VIEW_W, VIEW_H = 480, 480
METER_W = 240
# Close enough to see brows and eyes, wide enough to see the neck turn and nod.
CAMERA = {"lookat": (0.06, 0.0, 0.47), "distance": 0.42, "azimuth": 162.0, "elevation": -6.0}
METERS = (("neck yaw", ("neck_yaw",)), ("neck pitch", ("neck_pitch",)), ("neck roll", ("neck_roll",)),
          ("eye yaw", ("left_eye_yaw", "right_eye_yaw")), ("eye pitch", ("left_eye_pitch", "right_eye_pitch")),
          ("brows", GROUPS["brows"]), ("eyelids", GROUPS["lids"]), ("jaw", ("jaw_open",)),
          ("mouth corners", ("mouth_left_corner", "mouth_right_corner")),
          ("upper lip", ("upper_lip_raise",)), ("lower lip", ("lower_lip_drop",)),
          ("cheeks", ("left_cheek_raise", "right_cheek_raise")), ("nose", ("left_nose_wrinkle", "right_nose_wrinkle")),
          ("chin", ("chin_protrude",)))


class MujocoRobot(RobotInterface):
    """Position actuators with per-channel speed limits; measured state from qpos/qvel."""

    def __init__(self, limits: np.ndarray, video: Path | None = None, audio: Path | None = None,
                 title: str = "", output: Path | None = None, rating_video: Path | None = None):
        """``video`` is labeled (title, meters, R); ``rating_video`` has no text, meters or overlay.

        Observers only ever see ``rating_video``: same camera and frames, but nothing that could
        reveal the condition (the meters show which joints are locked) or a prediction.
        """
        import mujoco

        self.mujoco = mujoco
        self.model = compile_rig()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        self.limits = limits
        joints = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, c) for c in CHANNELS]
        self.qpos = np.array([self.model.jnt_qposadr[j] for j in joints])
        self.qvel = np.array([self.model.jnt_dofadr[j] for j in joints])
        self.actuators = np.array([next(a for a in range(self.model.nu) if self.model.actuator_trnid[a, 0] == j)
                                   for j in joints])
        low, high = self.model.actuator_ctrlrange[self.actuators].T
        if np.any(limits[:, 1] < low - 1e-7) or np.any(limits[:, 2] > high + 1e-7):
            raise ValueError("Calibration extends beyond the rig's actuator ranges")
        self.rates = limits[:, 3].astype(float)
        self.current = np.zeros(len(CHANNELS))
        self.wanted = np.zeros(len(CHANNELS))
        self.dt = self.model.opt.timestep
        self.finite = True
        self.peak_rate = np.zeros(len(CHANNELS))
        self.audio, self.title = audio, title
        self.caption = ""
        self.frame_count = 0
        self.outputs = {kind: path for kind, path in (("labeled", video), ("rating", rating_video)) if path}
        self.pipes, self.renderer, self.scratch = {}, None, None
        if self.outputs:
            if output is not None:
                output.mkdir(parents=True, exist_ok=True)
            self.scratch = TemporaryDirectory(dir=output, prefix="video-")
            self.renderer = mujoco.Renderer(self.model, height=VIEW_H, width=VIEW_W)
            self.camera = mujoco.MjvCamera()
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.camera.lookat[:] = CAMERA["lookat"]
            for key in ("distance", "azimuth", "elevation"):
                setattr(self.camera, key, CAMERA[key])
            for kind in self.outputs:
                width = VIEW_W + (METER_W if kind == "labeled" else 0)
                self.pipes[kind] = subprocess.Popen([
                    "ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{width}x{VIEW_H}", "-r", str(FPS), "-i", "pipe:0", "-c:v", "libx264",
                    "-preset", "fast", "-pix_fmt", "yuv420p", str(Path(self.scratch.name) / f"{kind}.mp4"),
                ], stdin=subprocess.PIPE)

    def state(self) -> np.ndarray:
        return np.stack([self.data.qpos[self.qpos], self.data.qvel[self.qvel]]).astype(np.float32)

    def command(self, target_q: np.ndarray) -> None:
        self.wanted = np.clip(np.asarray(target_q, dtype=float), self.limits[:, 1], self.limits[:, 2])

    def advance(self, seconds: float) -> None:
        end = self.data.time + seconds - 1e-9
        while self.data.time < end:
            if self.pipes and self.data.time + 1e-8 >= self.frame_count / FPS:
                self._render()
            change = np.clip(self.wanted - self.current, -self.rates * self.dt, self.rates * self.dt)
            self.current += change
            self.peak_rate = np.maximum(self.peak_rate, np.abs(change / self.dt))
            self.data.ctrl[self.actuators] = self.current
            self.mujoco.mj_step(self.model, self.data)
            self.finite &= bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())

    def annotate(self, index: int, session) -> None:
        if "labeled" not in self.pipes:
            return
        chunk = session.chunks[-1]
        line = f"{self.data.time:05.2f} s"
        if chunk["fallback"]:
            line += f"  /  {chunk['fallback'].replace('_', ' ')}"
        scored = [c for c in session.chunks if c.get("r_probabilities_on_realized_motion")]
        if scored:
            probabilities = scored[-1]["r_probabilities_on_realized_motion"]
            best = max(probabilities, key=probabilities.get)
            line += f"  /  R: {best.replace('_', ' ')} {probabilities[best]:.2f}"
        self.caption = line

    def _meters(self, draw, font, x0):
        q = self.data.qpos[self.qpos]
        draw.rectangle((x0, 0, x0 + METER_W, VIEW_H), fill="#0e1520")
        draw.text((x0 + 12, 10), "measured joints (of range)", font=font, fill="#a5b7ce")
        top, row = 34, (VIEW_H - 44) / len(METERS)
        for k, (label, names) in enumerate(METERS):
            idx = [CHANNELS.index(n) for n in names]
            value = float(np.mean(q[idx] / SCALE[idx]))
            locked = not self.limits[idx, 0].any()
            y = top + k * row
            draw.text((x0 + 12, y), label, font=font, fill="#5b6b80" if locked else "#e2e8f0")
            bx, bw = x0 + 112, METER_W - 124
            mid = bx + bw / 2
            draw.rectangle((bx, y + 3, bx + bw, y + 13), fill="#1f2a3a")
            if locked:
                draw.text((bx + 4, y), "locked", font=font, fill="#f59e0b")
            else:
                end = mid + np.clip(value, -1, 1) * bw / 2
                draw.rectangle((min(mid, end), y + 3, max(mid, end), y + 13), fill="#5eead4")
            draw.line((mid, y + 1, mid, y + 15), fill="#64748b")

    def _render(self) -> None:
        self.renderer.update_scene(self.data, camera=self.camera)
        pixels = self.renderer.render()
        self.frame_count += 1
        if "rating" in self.pipes:
            self.pipes["rating"].stdin.write(np.ascontiguousarray(pixels).tobytes())
        if "labeled" not in self.pipes:
            return
        from PIL import Image, ImageDraw, ImageFont
        from matplotlib import get_data_path

        canvas = Image.new("RGB", (VIEW_W + METER_W, VIEW_H))
        canvas.paste(Image.fromarray(pixels), (0, 0))
        draw = ImageDraw.Draw(canvas)
        font_path = Path(get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
        small = ImageFont.truetype(str(font_path), 12)
        draw.rectangle((0, 0, VIEW_W, 30), fill="#111a27")
        draw.text((10, 8), self.title, font=ImageFont.truetype(str(font_path), 14), fill="#f8fafc")
        draw.rectangle((0, VIEW_H - 26, VIEW_W, VIEW_H), fill="#111a27")
        draw.text((10, VIEW_H - 20), self.caption, font=small, fill="#f8fafc")
        self._meters(draw, small, VIEW_W)
        self.pipes["labeled"].stdin.write(canvas.tobytes())

    def close(self) -> dict:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        for kind, pipe in self.pipes.items():
            pipe.stdin.close()
            pipe.wait()
            if pipe.returncode != 0:
                raise RuntimeError("ffmpeg could not encode MuJoCo frames")
            encoded, target = Path(self.scratch.name) / f"{kind}.mp4", self.outputs[kind]
            target.parent.mkdir(parents=True, exist_ok=True)
            audio = ["-i", str(self.audio)] if self.audio else []
            maps = ["-map", "0:v:0"] + (["-map", "1:a:0", "-c:a", "aac", "-shortest"] if self.audio else [])
            # -map_metadata -1 drops container tags so files carry no condition hints.
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(encoded), *audio, *maps,
                            "-c:v", "copy", "-map_metadata", "-1", "-movflags", "+faststart", str(target)], check=True)
        if self.scratch is not None:
            self.scratch.cleanup()
            self.scratch = None
        self.pipes = {}
        return {"model": "samhita_sim/robotic_head_rig.xml (+gravcomp on skin bones)",
                "duration_s": round(float(self.data.time), 4), "finite_state": self.finite,
                "self_contact_steps": 0,  # rig geoms have contype=0; contacts cannot occur
                "peak_command_per_s": dict(zip(CHANNELS, np.round(self.peak_rate, 5).tolist()))}
