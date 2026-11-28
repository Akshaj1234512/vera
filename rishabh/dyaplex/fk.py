"""Neutral SMPL-X forward kinematics for joint positions (App. D.3, §5 evaluation).

Evaluation uses the 22 SMPL-X body joints with root translation set to zero [P §5]. Joint
positions only need the rest skeleton (J_regressor @ v_template, neutral shape) and the
kinematic tree, so no skinning is done here. The SMPL-X model files require registration at
https://smpl-x.is.tue.mpg.de; pass the path to SMPLX_NEUTRAL.npz.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .rotations import rotation_6d_to_matrix

BODY_JOINTS = 22


@dataclass
class Skeleton:
    rest: np.ndarray       # [J, 3] rest joint positions
    parents: np.ndarray    # [J] parent index, -1 for the root
    source: str

    @classmethod
    def from_smplx_npz(cls, path: Path) -> "Skeleton":
        model = np.load(path, allow_pickle=True)
        rest = np.asarray(model["J_regressor"]) @ np.asarray(model["v_template"])   # neutral betas = 0
        parents = np.asarray(model["kintree_table"])[0].astype(np.int64)
        parents[0] = -1
        return cls(rest[:55], parents[:55], str(path))


def forward(pose6d: np.ndarray, skeleton: Skeleton, joints: int = BODY_JOINTS) -> np.ndarray:
    """pose6d [T, >=joints, 6] local rotations (joint 0 = global orientation) -> positions [T, joints, 3]
    with zero root translation [P §5]."""
    rot = rotation_6d_to_matrix(pose6d[:, :joints])               # [T, J, 3, 3]
    frames = len(pose6d)
    world_r = np.empty_like(rot)
    pos = np.zeros((frames, joints, 3))
    for j in range(joints):
        p = skeleton.parents[j]
        if p < 0:
            world_r[:, j] = rot[:, j]
            pos[:, j] = 0.0
        else:
            world_r[:, j] = world_r[:, p] @ rot[:, j]
            offset = skeleton.rest[j] - skeleton.rest[p]
            pos[:, j] = pos[:, p] + world_r[:, p] @ offset
    return pos
