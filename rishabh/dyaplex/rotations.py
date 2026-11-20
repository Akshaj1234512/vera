"""Rotation conversions in the convention GestureLSM uses.

6D representation [Zhou et al. 2019] as implemented by pytorch3d and vendored in
GestureLSM ``utils/rotation_conversions.py:535``: the first two ROWS of the matrix.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    shape = aa.shape[:-1]
    return Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(*shape, 3, 3)


def matrix_to_axis_angle(m: np.ndarray) -> np.ndarray:
    shape = m.shape[:-2]
    return Rotation.from_matrix(m.reshape(-1, 3, 3)).as_rotvec().reshape(*shape, 3)


def matrix_to_6d(m: np.ndarray) -> np.ndarray:
    return m[..., :2, :].reshape(*m.shape[:-2], 6).copy()


def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """Gram-Schmidt as in pytorch3d. Degenerate rows (e.g. the all-zero face vectors of failed
    tracking frames, App. C.2) map to the identity rotation."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    n2 = np.linalg.norm(b2, axis=-1, keepdims=True)
    b2 = b2 / np.maximum(n2, 1e-8)
    b3 = np.cross(b1, b2)
    out = np.stack([b1, b2, b3], axis=-2)
    bad = (np.linalg.norm(a1, axis=-1) < 1e-6) | (n2[..., 0] < 1e-6)
    if np.any(bad):
        out[bad] = np.eye(3)
    return out


def axis_angle_to_6d(aa: np.ndarray) -> np.ndarray:
    return matrix_to_6d(axis_angle_to_matrix(aa))


def rotation_6d_to_axis_angle(d6: np.ndarray) -> np.ndarray:
    return matrix_to_axis_angle(rotation_6d_to_matrix(d6))


def resample_rotations(aa: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Slerp each joint of an axis-angle track [T, J, 3] from src_fps to dst_fps. [A] slerp."""
    frames, joints = aa.shape[:2]
    src_t = np.arange(frames) / src_fps
    dst_t = np.arange(int(np.floor((frames - 1) / src_fps * dst_fps)) + 1) / dst_fps
    out = np.empty((len(dst_t), joints, 3))
    for j in range(joints):
        out[:, j] = Slerp(src_t, Rotation.from_rotvec(aa[:, j])) (dst_t).as_rotvec()
    return out


def resample_linear(x: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    frames = x.shape[0]
    src_t = np.arange(frames) / src_fps
    dst_t = np.arange(int(np.floor((frames - 1) / src_fps * dst_fps)) + 1) / dst_fps
    flat = x.reshape(frames, -1)
    out = np.column_stack([np.interp(dst_t, src_t, flat[:, k]) for k in range(flat.shape[1])])
    return out.reshape(len(dst_t), *x.shape[1:])


def x_rotation(angle: float) -> np.ndarray:
    return Rotation.from_euler("x", angle).as_matrix()
