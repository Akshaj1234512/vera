"""Extract uncalibrated listener motion and propose clips for human annotation."""

import csv
import json
import os
from pathlib import Path

import numpy as np

from .sources import fetch

FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")


def face_track(video: Path, landmark_model: Path, side: str, sample_hz: float) -> np.ndarray:
    import cv2
    import mediapipe as mp

    capture = cv2.VideoCapture(str(video))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not capture.isOpened() or fps <= 0 or not np.isfinite(fps):
            raise ValueError(f"Could not read frame rate for {video}")
        stride = max(1, round(fps / sample_hz))
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(landmark_model)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1,
            output_facial_transformation_matrixes=True, output_face_blendshapes=True,
        )
        rows = []
        with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as detector:
            frame_idx = 0
            while capture.grab():
                if frame_idx % stride == 0:
                    ok, frame = capture.retrieve()
                    if not ok:
                        break
                    mid = frame.shape[1] // 2
                    if side != "full":
                        frame = frame[:, mid:] if side == "right" else frame[:, :mid]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    time_s = frame_idx / fps
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = detector.detect_for_video(image, round(time_s * 1000))
                    pitch = yaw = jaw = np.nan
                    if result.facial_transformation_matrixes:
                        matrix = result.facial_transformation_matrixes[0]
                        pitch = np.degrees(np.arctan2(matrix[2, 1], matrix[2, 2]))
                        yaw = np.degrees(np.arctan2(-matrix[2, 0], np.hypot(matrix[2, 1], matrix[2, 2])))
                    if result.face_blendshapes:
                        jaw = next((c.score for c in result.face_blendshapes[0]
                                    if c.category_name == "jawOpen"), np.nan)
                    rows.append((time_s, pitch, yaw, jaw))
                frame_idx += 1
        if not rows:
            raise ValueError(f"No decodable frames in {video}")
        return np.asarray(rows)
    finally:
        capture.release()


def pitch_excursions(track: np.ndarray, minimum_deg: float) -> list[dict]:
    """Heuristic offline pitch excursions that return toward the starting pose."""
    seconds, pitch, yaw, _ = track.T
    if len(seconds) < 3:
        return []
    frames = max(1, round(0.45 / np.median(np.diff(seconds))))
    if len(pitch) <= 2 * frames:
        return []
    # Use a short median to reject single-frame landmark jumps and retain missing frames.
    smooth = pitch.copy()
    for i in range(1, len(pitch) - 1):
        window = pitch[i - 1:i + 2]
        if np.isfinite(window).all():
            smooth[i] = np.median(window)
    possible = []
    for i in range(frames, len(smooth) - frames):
        before, after = i - frames, i + frames
        window = smooth[before:after + 1]
        yaw_window = yaw[before:after + 1]
        if not np.isfinite(window).all() or not np.isfinite(yaw_window).all() or np.ptp(yaw_window) > 12:
            continue
        signed = smooth[i] - window[[0, -1]]
        if signed[0] * signed[1] <= 0:
            continue
        delta = float(np.min(np.abs(signed)))
        if delta >= minimum_deg and smooth[i] in (window.min(), window.max()):
            possible.append((i, delta))
    # Select more prominent peaks first when their windows cover the same movement.
    peaks = []
    for i, delta in sorted(possible, key=lambda row: -row[1]):
        if all(abs(seconds[i] - item["peak_s"]) >= 0.7 for item in peaks):
            peaks.append({"peak_s": round(float(seconds[i]), 3), "pitch_excursion_deg": round(delta, 2)})
    return sorted(peaks, key=lambda item: item["peak_s"])


def analyze_manifest(manifest: Path, model: Path, output: Path, side: str, sample_hz: float,
                     minimum_deg: float, export_proxy: bool) -> Path:
    if sample_hz <= 0 or minimum_deg <= 0:
        raise ValueError("Use positive sample rate and pitch prominence")
    if not model.is_file():
        model.parent.mkdir(parents=True, exist_ok=True)
        fetch(FACE_MODEL_URL, model)
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads(manifest.read_text())
    results, proxy = [], []
    for row in rows:
        clip = row["id"]
        if Path(clip).name != clip or clip in (".", ".."):
            raise ValueError("Clip id must be a filename component")
        video = manifest.parent / row["video"]
        track = face_track(video, model, side, sample_hz)
        candidates = pitch_excursions(track, minimum_deg)
        csv_path = output / f"{clip}-pose.csv"
        with csv_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["time_s", "head_pitch_deg", "head_yaw_deg", "jaw_open"])
            for values in track:
                writer.writerow([f"{x:.4f}" if np.isfinite(x) else "" for x in values])
        coverage = float(np.isfinite(track[:, 1]).mean())
        results.append({"id": clip, "trajectory": csv_path.name, "frames": len(track),
                        "tracked_fraction": round(coverage, 4), "candidates": candidates})
        if export_proxy and coverage >= 0.8:
            updated = dict(row)
            for field in ("partner", "agent", "video", "embeddings"):
                if updated.get(field):
                    updated[field] = os.path.relpath(manifest.parent / row[field], output)
            updated["origin"] = row["origin"] + "-pose-proxy"
            updated["ack_s"] = [candidate["peak_s"] for candidate in candidates]
            proxy.append(updated)
        print(f"{clip} | {coverage:.0%} tracked | {len(candidates)} pitch excursions", flush=True)
    destination = output / "pose-review.json"
    destination.write_text(json.dumps({
        "source": str(manifest), "crop": side, "minimum_pitch_deg": minimum_deg,
        "model": str(model), "label_type": "unreviewed pitch excursions", "clips": results,
    }, indent=2) + "\n")
    if export_proxy:
        (output / "pose-proxy-manifest.json").write_text(json.dumps(proxy, indent=2) + "\n")
    return destination
