"""Baseline data export (App. E.2-E.3): the same prepared Seamless body data in each baseline's format.

    python -m dyaplex export-baselines --prepared P --out runs/baseline-data

Per participant of every kept pair, at 25 fps (<out>/<split>/<file_id>.npz):
  dualtalk_69  [T, 69]   SMPL-X body pose 63 + global orientation 3 + translation 3, axis-angle [P E.2]
  a2p_132      [T, 132]  22 joints x 6D rotation [P E.3] (translation handled at FK: zero, mean subtracted)
  trans        [T, 3]    translation (A2P subtracts each clip's mean) [P E.3]
  valid        [T]       body validity
and <out>/<split>/pairs.json with each pair's participants and audio paths (both speakers; DualTalk uses
wav2vec2 16 kHz, A2P its frozen vq-wav2vec). Adapting the two codebases is described in BASELINES.md;
their predictions are scored with `dyaplex evaluate --predictions`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import data as D
from .prepare import load_meta, load_parts, load_valid, split_pairs
from .rotations import rotation_6d_to_axis_angle


def run(prepared: Path, out: Path, splits=("train", "val", "test330")) -> dict:
    counts = {}
    for split in splits:
        target = out / split
        target.mkdir(parents=True, exist_ok=True)
        rows = []
        for key, a, b in split_pairs(prepared, split):
            for fid in (a, b):
                parts = {p: np.asarray(v, dtype=np.float32) for p, v in load_parts(prepared, fid).items()}
                pose = D.assemble(parts)
                aa = rotation_6d_to_axis_angle(pose["pose6d"][:, :22])               # [T, 22, 3]
                trans = pose["trans"]
                np.savez_compressed(target / f"{fid}.npz",
                                    dualtalk_69=np.concatenate([aa[:, 1:22].reshape(len(aa), -1), aa[:, 0], trans], 1)
                                    .astype(np.float32),
                                    a2p_132=pose["pose6d"][:, :22].reshape(len(aa), -1).astype(np.float32),
                                    trans=trans, valid=load_valid(prepared, fid)["body"][: len(aa)])
            rows.append({"pair": key, "a": a, "b": b, "wav_a": load_meta(prepared, a)["wav"],
                         "wav_b": load_meta(prepared, b)["wav"]})
        (target / "pairs.json").write_text(json.dumps(rows, indent=1) + "\n")
        counts[split] = len(rows)
    return counts
