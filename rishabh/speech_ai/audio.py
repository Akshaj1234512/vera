"""Causal audio-encoder features on the 20 Hz G/R clock.

Each frame summarizes only the samples that ended by that frame's timestamp, so
features at a chunk boundary never include audio from the chunk being planned.
"""

from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np

FRAME_S = 0.05
FEATURE_NAMES = ("partner_db", "partner_pitch", "partner_voiced", "partner_active",
                 "agent_db", "agent_pitch", "agent_voiced", "agent_active")


@dataclass
class Features:
    time_s: np.ndarray
    values: np.ndarray


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as file:
        channels = file.getnchannels()
        rate = file.getframerate()
        width = file.getsampwidth()
        raw = file.readframes(file.getnframes())
    if width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
    elif width == 3:
        bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        signed = bytes_[:, 0] | bytes_[:, 1] << 8 | bytes_[:, 2] << 16
        samples = ((signed ^ 0x800000) - 0x800000).astype(np.float32) / 8388608
    elif width in (2, 4):
        samples = np.frombuffer(raw, dtype=f"<i{width}").astype(np.float32) / (2 ** (width * 8 - 1))
    else:
        raise ValueError(f"{path}: unsupported PCM sample width {width}")
    if not len(samples):
        raise ValueError(f"{path}: empty WAV")
    return samples.reshape(-1, channels), rate


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    """Write a mono PCM16 WAV from samples scaled to [-1, 1]."""
    with wave.open(str(path), "wb") as file:
        file.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        file.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


def pitch(frame: np.ndarray, rate: int) -> tuple[float, float]:
    stride = max(1, rate // 6000)
    reduced_rate = rate / stride
    frame = frame[::stride]
    frame = frame - np.mean(frame)
    power = np.dot(frame, frame)
    if power < 1e-8:
        return 0.0, 0.0
    autocorr = np.correlate(frame, frame, mode="full")[len(frame) - 1:] / power
    start, stop = int(reduced_rate / 420), min(len(frame) - 1, int(reduced_rate / 65))
    if stop <= start:
        return 0.0, 0.0
    # Prefer the first strong peak to avoid choosing a multiple of the pitch period.
    lags = np.arange(max(1, start), stop)
    peaks = lags[(autocorr[lags] > autocorr[lags - 1]) & (autocorr[lags] >= autocorr[lags + 1])]
    if not len(peaks):
        return 0.0, 0.0
    strength = float(autocorr[peaks].max())
    if strength < 0.3:
        return 0.0, 0.0
    lag = peaks[autocorr[peaks] >= 0.9 * strength][0]
    return float(reduced_rate / lag), float(autocorr[lag])


def track_features(samples: np.ndarray, rate: int, frames: int, threshold_db: float) -> np.ndarray:
    values = np.zeros((frames, 4), dtype=np.float32)
    active = False
    for index in range(frames):
        start = round(index * FRAME_S * rate)
        end = round((index + 1) * FRAME_S * rate)
        frame = samples[start:end]
        power = np.dot(frame, frame) / (end - start) if len(frame) else 0.0
        db = float(10 * np.log10(max(power, 1e-8)))
        active = db >= (threshold_db - 4 if active else threshold_db)
        hz, confidence = pitch(frame, rate) if active else (0.0, 0.0)
        values[index] = (db, hz, confidence, active)
    return values


def extract(partner: tuple[np.ndarray, int], agent: tuple[np.ndarray, int] | None = None,
            threshold_db: float = -40) -> Features:
    """Frame i covers [i * 50 ms, (i + 1) * 50 ms) and is stamped at its end."""
    partner_samples, partner_rate = partner
    recordings = [partner] if agent is None else [partner, agent]
    duration = max(len(samples) / rate for samples, rate in recordings)
    frames = int(np.ceil(duration / FRAME_S - 1e-9))
    if frames == 0:
        raise ValueError("Need at least one audio frame")
    time_s = np.round(np.arange(1, frames + 1) * FRAME_S, 6)
    partner_values = track_features(partner_samples, partner_rate, frames, threshold_db)
    if agent is None:
        agent_values = np.zeros_like(partner_values)
        agent_values[:, 0] = -80
    else:
        agent_values = track_features(*agent, frames, threshold_db)
    return Features(time_s, np.concatenate([partner_values, agent_values], axis=1))


def normalized(values: np.ndarray) -> np.ndarray:
    """Map dBFS to [0, 1] above -80 dB and pitch to a fraction of 420 Hz."""
    result = values.astype(np.float32).copy()
    result[:, [0, 4]] = np.clip((result[:, [0, 4]] + 80) / 80, 0, 1)
    result[:, [1, 5]] /= 420
    return result


def load_tracks(partner_path: Path, agent_path: Path | None = None,
                stereo: bool = False) -> tuple[tuple[np.ndarray, int], tuple[np.ndarray, int] | None]:
    samples, rate = read_wav(partner_path)
    if stereo:
        if samples.shape[1] != 2 or agent_path is not None:
            raise ValueError("Stereo input needs exactly two channels: partner left, agent right")
        return (samples[:, 0], rate), (samples[:, 1], rate)
    partner = (samples.mean(axis=1), rate)
    if agent_path is None:
        return partner, None
    samples, rate = read_wav(agent_path)
    return partner, (samples.mean(axis=1), rate)
