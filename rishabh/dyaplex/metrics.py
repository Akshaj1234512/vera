"""Paper metrics (§5, App. E.1): FGD, Diversity, BeatAlign, P-FD, Delta-User.

Sources:
  Frechet distance   [I ViBES 7821e5e multimodal_tokenizers/metrics/utils.py calculate_frechet_distance_np]
  Diversity          [I audio2photoreal eaadca7 utils/eval.py calculate_diversity] (mean of the returned distances)
  BeatAlign          [I ViBES utils/metric.py alignment(0.3, 7, ...) + co_speech.py]: per-joint speed
                     (central differences) normalized by a per-joint mean speed, minima with
                     argrelextrema(order=7) above threshold 0.3, librosa onsets (hop 512),
                     Gaussian hit rate, audio-anchored (for each onset, nearest motion beat),
                     averaged over the 13 upper-body joints; frames trimmed at both ends.
  Paper deltas       FD on raw joints (no feature extractor) [P E.1]; 25 fps; audio at Mimi's 24 kHz.
  Assumptions [A]    mean speed refit on Seamless GT (ViBES uses a BEAT2 file); trim = 2 s
                     (ViBES: 60 frames at 30 fps); argrelextrema order kept at 7 at 25 fps.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import linalg
from scipy.signal import argrelextrema

UPPER_BODY = (3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)


# --------------------------------------------------------------------------- #
# Frechet distances
# --------------------------------------------------------------------------- #
def frechet_distance(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    """Frechet distance between Gaussian fits of rows of x and y (ViBES calculate_frechet_distance_np)."""
    mu1, mu2 = x.mean(0), y.mean(0)
    s1, s2 = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    diff = mu1 - mu2
    covmean = linalg.sqrtm(s1.dot(s2))           # scipy >= 1.16 dropped disp=False; same result
    if not np.isfinite(covmean).all():
        offset = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + offset).dot(s2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def fgd(gen: np.ndarray, gt: np.ndarray) -> float:
    """FGD on raw 22x3 = 66-D joint positions per frame, pooled over the test set [P E.1]."""
    return frechet_distance(gen.reshape(len(gen), -1), gt.reshape(len(gt), -1))


def paired_fd(gt_a: np.ndarray, gt_b: np.ndarray, gen_b: np.ndarray) -> float:
    """P-FD on 132-D [A-joints, B-joints]: FD((GT_A, GT_B), (GT_A, Gen_B)) [P E.1]."""
    real = np.concatenate([gt_a.reshape(len(gt_a), -1), gt_b.reshape(len(gt_b), -1)], axis=1)
    fake = np.concatenate([gt_a.reshape(len(gt_a), -1), gen_b.reshape(len(gen_b), -1)], axis=1)
    return frechet_distance(fake, real)


def delta_user(pfd_shuffled: float, pfd_matched: float) -> float:
    """(P-FD_shuffled - P-FD_matched) / P-FD_matched, as a percentage [P §5]."""
    return 100.0 * (pfd_shuffled - pfd_matched) / pfd_matched


def derangement(n: int, seed: int) -> np.ndarray:
    """Shuffled partner assignment with no pair keeping its own partner [A shuffle protocol]."""
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(n)
        if n < 2 or not (perm == np.arange(n)).any():
            return perm


# --------------------------------------------------------------------------- #
# Diversity
# --------------------------------------------------------------------------- #
def diversity(frames: np.ndarray, times: int = 10_000, seed: int = 0) -> float:
    """Mean L2 between random pairs of pooled agent-pose frames [I A2P calculate_diversity]."""
    flat = frames.reshape(len(frames), -1)
    if len(flat) <= times:
        raise ValueError(f"need more than {times} frames, got {len(flat)}")
    rng = np.random.default_rng(seed)
    first = rng.choice(len(flat), times, replace=False)
    second = rng.choice(len(flat), times, replace=False)
    return float(np.linalg.norm(flat[first] - flat[second], axis=1).mean())


# --------------------------------------------------------------------------- #
# BeatAlign
# --------------------------------------------------------------------------- #
def joint_speed(joints: np.ndarray, fps: float) -> np.ndarray:
    """[T, J, 3] -> [T, J] speed with forward / central / backward differences (ViBES load_pose)."""
    dt = 1.0 / fps
    vel = np.empty_like(joints)
    vel[0] = (joints[1] - joints[0]) / dt
    vel[1:-1] = (joints[2:] - joints[:-2]) / (2 * dt)
    vel[-1] = (joints[-1] - joints[-2]) / dt
    return np.linalg.norm(vel, axis=2)


def mean_speed(clips: list[np.ndarray], fps: float) -> np.ndarray:
    """Per-joint mean speed used to normalize velocities (ViBES 'mmae') [A refit on Seamless GT]."""
    return np.concatenate([joint_speed(c, fps) for c in clips]).mean(0)


def audio_onsets(audio: np.ndarray, sr: int) -> np.ndarray:
    import librosa
    return librosa.onset.onset_detect(y=audio.astype(np.float32), sr=sr, hop_length=512, units="time")


def motion_beats(joints: np.ndarray, fps: float, mean_vel: np.ndarray, start: int, end: int,
                 order: int = 7, threshold: float = 0.3) -> list[np.ndarray]:
    speed = joint_speed(joints, fps) / np.where(mean_vel > 0, mean_vel, 1.0)   # root speed is 0 with zero translation
    beats = []
    for j in range(speed.shape[1]):
        minima = argrelextrema(speed[start:end, j], np.less, order=order)[0]
        moving = set(np.where(speed[:, j] > threshold)[0].tolist())     # ViBES indexes the mask on the full clip
        beats.append(np.array([m for m in minima if m in moving]))
    return beats


def gahr(motion_t: np.ndarray, audio_t: np.ndarray, sigma: float) -> float:
    """Audio-anchored Gaussian hit rate (ViBES GAHR(a=pose, b=audio))."""
    if len(audio_t) == 0:
        return float("nan")
    total = 0.0
    for b in audio_t:
        nearest = np.min(np.abs(motion_t - b)) if len(motion_t) else np.inf
        total += math.exp(-(nearest ** 2) / (2 * sigma ** 2))
    return total / len(audio_t)


def beat_align(joints: np.ndarray, audio: np.ndarray, sr: int, fps: float, mean_vel: np.ndarray,
               sigma: float = 0.3, order: int = 7, threshold: float = 0.3, trim_s: float = 2.0,
               upper=UPPER_BODY) -> float:
    """BeatAlign for one clip; joints [T, 22, 3] at `fps`, audio of the agent at `sr`."""
    trim = int(round(trim_s * fps))
    a_trim = int(round(trim_s * sr))
    onsets = audio_onsets(audio[a_trim: len(audio) - a_trim], sr)
    beats = motion_beats(joints, fps, mean_vel, trim, len(joints) - trim, order, threshold)
    scores = [gahr(beats[j] / fps, onsets, sigma) for j in upper]
    return float(np.nanmean(scores)) if np.isfinite(scores).any() else float("nan")


def summarize(gen_b: list[np.ndarray], gt_a: list[np.ndarray], gt_b: list[np.ndarray],
              audio_b: list[np.ndarray], sr: int, fps: float, gen_b_shuffled: list[np.ndarray] | None,
              mean_vel: np.ndarray, seed: int = 0) -> dict:
    """Table 2 row: FGD and P-FD reported x1e3 as in the paper (assumed units: metres)."""
    cat = lambda xs: np.concatenate(xs)
    out = {
        "FGD_x1e3": 1e3 * fgd(cat(gen_b), cat(gt_b)),
        "Diversity": diversity(cat(gen_b), seed=seed) if sum(map(len, gen_b)) > 10_000 else float("nan"),
        "BeatAlign": float(np.nanmean([beat_align(g, a, sr, fps, mean_vel) for g, a in zip(gen_b, audio_b)])),
        "P-FD_x1e3": 1e3 * paired_fd(cat(gt_a), cat(gt_b), cat(gen_b)),
    }
    if gen_b_shuffled is not None:
        shuffled = 1e3 * paired_fd(cat(gt_a), cat(gt_b), cat(gen_b_shuffled))
        out["P-FD_shuffled_x1e3"] = shuffled
        out["Delta-User_%"] = delta_user(shuffled, out["P-FD_x1e3"])
    return out
