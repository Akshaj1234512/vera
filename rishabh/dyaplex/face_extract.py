"""SPECTRE face tracks for the face-enabled variant (App. C.2). Runs inside SPECTRE's environment.

    cd third_party/spectre && python <repo>/rishabh/dyaplex/face_extract.py \
        --videos <seamless root> --out <face dir> --device cuda [--shard i/n]

Setup (SPECTRE @ 0b339a8, filby89/spectre): follow its README (quick_install.sh: FLAME 2020 model and
SPECTRE weights, which need FLAME registration at https://flame.is.tue.mpg.de; face tracker from
Visual_Speech_Recognition_for_Multiple_Languages). This script reuses SPECTRE's demo.py pipeline
(frame extraction, landmark tracking, 224x224 crops at scale 1.6, chunks of 50 frames with 2-frame
overlap) and writes per video <out>/<file_id>.npz:
    jaw_aa [T, 3]   jaw rotation (axis-angle; dyaplex converts to 6D [P C.2])
    exp    [T, 50]  FLAME expression coefficients
    head_aa [T, 3]  head rotation (kept for reference; DyaPlex discards it [P C.2])
    valid  [T]      False where no face was tracked on that frame -> all-zero face [P C.2]
                    (SPECTRE's demo interpolates landmarks over such frames; the paper zeroes them)
NOT RUN in this repository (needs the FLAME license, SPECTRE weights and a GPU). Verify on one clip
against SPECTRE's own demo.py output before running the corpus.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


def run_video(spectre, video: Path, device: str, chunk: int = 50) -> dict:
    import cv2
    import torch
    from skimage.transform import warp
    from datasets.data_utils import landmarks_interpolate      # SPECTRE
    from demo import crop_face, extract_frames                 # SPECTRE demo.py helpers

    image_paths, landmarks, _, _ = extract_frames(str(video), detect_landmarks=True)
    raw = list(landmarks)
    tracked = np.array([l is not None for l in raw])
    frames = len(image_paths)
    out = {"jaw_aa": np.zeros((frames, 3), np.float32), "exp": np.zeros((frames, 50), np.float32),
           "head_aa": np.zeros((frames, 3), np.float32), "valid": tracked.copy()}
    if not tracked.any():
        out["valid"][:] = False
        return out
    landmarks = landmarks_interpolate(raw)
    # demo.py: pad 2 frames on each side, chunks of `chunk` frames overlapping by 4
    image_paths = np.array([image_paths[0]] * 2 + list(image_paths) + [image_paths[-1]] * 2)
    landmarks = np.array([landmarks[0]] * 2 + list(landmarks) + [landmarks[-1]] * 2)
    indices = list(range(len(image_paths)))
    chunks = [indices[i: i + chunk] for i in range(0, len(indices), chunk - 4)]
    if len(chunks[-1]) < 5:
        chunks[-2] = sorted(set(chunks[-2] + chunks[-1]))
        chunks = chunks[:-1]
    exps, poses = [], []
    with torch.no_grad():
        for k, idx in enumerate(chunks):
            images = []
            for j in idx:
                frame = cv2.cvtColor(cv2.imread(str(image_paths[j])), cv2.COLOR_BGR2RGB)   # as demo.py; warp -> [0, 1]
                tform = crop_face(frame, landmarks[j], scale=1.6)
                images.append(warp(frame, tform.inverse, output_shape=(224, 224)).transpose(2, 0, 1))
            batch = torch.from_numpy(np.array(images)).float().to(device)
            code, init_exp, init_jaw = spectre.encode(batch)
            code["exp"] = code["exp"] + init_exp
            code["pose"][..., 3:] = code["pose"][..., 3:] + init_jaw
            keep = slice(None) if len(chunks) == 1 else slice(None, -2) if k == 0 else \
                slice(2, None) if k == len(chunks) - 1 else slice(2, -2)
            exps.append(code["exp"][keep].cpu().numpy())
            poses.append(code["pose"][keep].cpu().numpy())
    exp = np.concatenate(exps)[2:-2][:frames]
    pose = np.concatenate(poses)[2:-2][:frames]
    out["exp"][: len(exp)] = exp[:, :50]
    out["head_aa"][: len(pose)] = pose[:, :3]
    out["jaw_aa"][: len(pose)] = pose[:, 3:6]
    out["exp"][~out["valid"]] = 0
    out["jaw_aa"][~out["valid"]] = 0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", type=Path, required=True, help="Seamless root (searched for *.mp4)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard", default="0/1", help="i/n: process every n-th video starting at i")
    args = parser.parse_args()
    sys.path.insert(0, str(Path.cwd()))                         # run from the SPECTRE checkout
    from config import cfg as spectre_cfg
    from src.spectre import SPECTRE
    spectre_cfg.pretrained_modelpath = "pretrained/spectre_model.tar"
    spectre_cfg.model.use_tex = False
    spectre = SPECTRE(spectre_cfg, args.device).eval()
    shard, shards = map(int, args.shard.split("/"))
    videos = sorted(args.videos.rglob("*.mp4"))[shard::shards]
    args.out.mkdir(parents=True, exist_ok=True)
    for k, video in enumerate(videos):
        target = args.out / f"{video.stem}.npz"
        if target.exists():
            continue
        try:
            np.savez_compressed(target, **run_video(spectre, video, args.device))
        except Exception as error:                              # unreadable video -> all-zero face
            print(f"{video.stem}: {error}", flush=True)
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(videos)}", flush=True)


if __name__ == "__main__":
    main()
