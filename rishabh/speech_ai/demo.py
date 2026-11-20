"""Side-by-side demo: a real human's motion, projected-DyaPlex retargeting and G on Samhita's rig.

Everything shown is reproducible from the command line (``gr-demo``):

* top-left: the Seamless participant whose measured motion is the teacher M_h (face crop of
  the source video, same time window)
* the rig driven by the per-frame retarget ("projected DyaPlex") and by G, each with the full
  face and with neck pitch locked, all from the same stream and conversation audio.
"""

import json
from pathlib import Path
import subprocess

import numpy as np

from .audio import FRAME_S

PANEL_W, PANEL_H = 720, 480


def excerpt_stream(stream: Path, start_s: float, duration_s: float, out: Path) -> None:
    with np.load(stream, allow_pickle=False) as data:
        first, count = round(start_s / FRAME_S), round(duration_s / FRAME_S)
        arrays = {k: data[k] for k in data.files}
    sliced = {k: (v[first:first + count] if k in ("human", "human_valid", "context") else v)
              for k, v in arrays.items()}
    sliced["time_s"] = np.round(np.arange(1, count + 1) * FRAME_S, 6)
    np.savez_compressed(out, **sliced)


def _card(lines: list[tuple[str, int, str]], size=(PANEL_W, PANEL_H), path: Path = None) -> Path:
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import get_data_path

    font = Path(get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    image = Image.new("RGB", size, "#0b1220")
    draw = ImageDraw.Draw(image)
    y = 24
    for text, px, color in lines:
        draw.text((24, y), text, font=ImageFont.truetype(str(font), px), fill=color)
        y += px + 10
    image.save(path)
    return path


def human_panel(video: Path, keypoints: Path, start_s: float, duration_s: float, label: list[str], out: Path) -> Path:
    """Static face crop around the participant's median nose position in the window."""
    points = np.load(keypoints, allow_pickle=False)
    frames = slice(round(start_s * 30), round((start_s + duration_s) * 30))
    nose = points[frames, 0, :2]
    nose = nose[np.isfinite(nose).all(1) & (nose > 0).all(1)]
    x, y = np.median(nose, axis=0)
    side = 620
    card = _card([(t, 13 if i else 16, "#f8fafc" if i == 0 else "#a5b7ce") for i, t in enumerate(label)],
                 (PANEL_W - PANEL_H, PANEL_H), out.with_suffix(".png"))
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{start_s:.3f}", "-t", f"{duration_s:.3f}",
                    "-i", str(video), "-loop", "1", "-i", str(card), "-filter_complex",
                    f"[0:v]crop={side}:{side}:{max(0, int(x - side / 2))}:{max(0, int(y - side * 0.45))},"
                    f"scale={PANEL_H}:{PANEL_H},fps=25[f];[f][1:v]hstack=inputs=2:shortest=1[v]",
                    "-map", "[v]", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def compose(stream: Path, partner_wav: Path, agent_wav: Path, video: Path, keypoints: Path, start_s: float,
            duration_s: float, g: Path, r: Path | None, out: Path, label: list[str],
            locked: str = "fixed-neck-pitch") -> dict:
    from .gr_runtime import replay

    out.mkdir(parents=True, exist_ok=True)
    excerpt_stream(stream, start_s, duration_s, out / "stream.npz")
    audio = out / "conversation.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{start_s:.3f}", "-t", f"{duration_s:.3f}",
                    "-i", str(partner_wav), "-ss", f"{start_s:.3f}", "-t", f"{duration_s:.3f}", "-i", str(agent_wav),
                    "-filter_complex", "amix=inputs=2:normalize=0", str(audio)], check=True)
    conditions = [("projected", "full", "Projected DyaPlex (retarget) / full face"),
                  ("g", "full", "G / full face"),
                  ("projected", locked, f"Projected DyaPlex (retarget) / {locked}"),
                  ("g", locked, f"G / {locked}")]
    reports = {}
    for policy, profile, title in conditions:
        name = f"{policy}-{profile}"
        report = replay(out / "stream.npz", g if policy == "g" else None, r, profile, out / name, policy=policy,
                        video=out / name / "labeled.mp4", audio=audio, title=title, session_id=name,
                        group="demo")
        reports[name] = {"metrics": report["metrics"], "policy": report["policy"], "profile": report["profile"]}
    human = human_panel(video, keypoints, start_s, duration_s, label, out / "human.mp4")
    info = _card([("How to read this", 18, "#f8fafc"),
                  ("Top-left: the real person (Seamless Interaction,", 14, "#a5b7ce"),
                  ("held-out test split). Their measured head, gaze and", 14, "#a5b7ce"),
                  ("facial action units are the teacher input M_h.", 14, "#a5b7ce"),
                  ("Projected DyaPlex: fixed retarget, frame by frame.", 14, "#a5b7ce"),
                  ("G: learned generator, plans each next 500 ms.", 14, "#a5b7ce"),
                  ("Right of each face: measured joint positions.", 14, "#a5b7ce"),
                  ("", 8, "#a5b7ce"),
                  ("Caveats", 18, "#fbbf24"),
                  ("DyaPlex itself is unreleased; nothing here is", 14, "#fbbf24"),
                  ("DyaPlex output. R and G's meaning loss use", 14, "#fbbf24"),
                  ("SYNTHETIC rule-based votes, not human raters.", 14, "#fbbf24"),
                  ("This is a pipeline demo, not a human result.", 14, "#fbbf24")],
                 path=out / "info.png")
    grid = out / "demo.mp4"
    inputs = [human, out / "projected-full/labeled.mp4", out / "g-full/labeled.mp4", info,
              out / f"projected-{locked}/labeled.mp4", out / f"g-{locked}/labeled.mp4"]
    args = []
    for path in inputs:
        args += (["-loop", "1", "-i", str(path)] if path.suffix == ".png" else ["-i", str(path)])
    args += ["-i", str(audio)]
    layout = "0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args, "-filter_complex",
                    "".join(f"[{k}:v]scale={PANEL_W}:{PANEL_H},setsar=1,fps=25[p{k}];" for k in range(6))
                    + "".join(f"[p{k}]" for k in range(6)) + f"xstack=inputs=6:layout={layout}[v]",
                    "-map", "[v]", "-map", "6:a", "-t", f"{duration_s:.3f}", "-c:v", "libx264", "-crf", "21",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(grid)], check=True)
    summary = {"stream": str(stream), "start_s": start_s, "duration_s": duration_s, "g": str(g),
               "r": str(r) if r else None, "conditions": reports, "video": str(grid)}
    (out / "demo.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
