"""Extract aligned features using only samples available at each frame end."""

from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np

FRAME_S = 0.05
HISTORY_FRAMES = 16
PAUSE_FRAMES = 3
MIN_SPEECH_FRAMES = 7
FEATURE_NAMES = ("partner_db", "partner_pitch", "partner_voiced", "partner_active",
                 "agent_db", "agent_pitch", "agent_voiced", "agent_active")


@dataclass
class Features:
    time_s: np.ndarray
    values: np.ndarray
    candidates: np.ndarray
    speech_end_s: np.ndarray


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
    partner_samples, partner_rate = partner
    recordings = [partner] if agent is None else [partner, agent]
    duration = max(len(samples) / rate for samples, rate in recordings)
    frames = int(np.ceil(duration / FRAME_S))
    if frames == 0:
        raise ValueError("Need at least one audio frame")
    time_s = np.round(np.arange(1, frames + 1) * FRAME_S, 6)
    partner_values = track_features(partner_samples, partner_rate, frames, threshold_db)
    if agent is None:
        agent_values = np.zeros_like(partner_values)
        agent_values[:, 0] = -80
    else:
        agent_values = track_features(*agent, frames, threshold_db)

    candidates, starts = [], []
    speech_frames = silence_frames = 0
    for index, active in enumerate(partner_values[:, 3]):
        if active:
            if silence_frames >= PAUSE_FRAMES:
                speech_frames = 0
            speech_frames += 1
            silence_frames = 0
            continue
        silence_frames += 1
        if silence_frames == PAUSE_FRAMES and speech_frames >= MIN_SPEECH_FRAMES:
            candidates.append(index)
            starts.append(round(time_s[index] - PAUSE_FRAMES * FRAME_S, 6))

    return Features(time_s, np.concatenate([partner_values, agent_values], axis=1),
                    np.asarray(candidates, dtype=np.int64), np.asarray(starts))


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


def context_windows(features: Features) -> np.ndarray:
    # Include the current frame and left-pad the start of a conversation as silence.
    values = features.values.copy()
    values[:, [0, 4]] = (values[:, [0, 4]] + 80) / 80
    values[:, [1, 5]] /= 420
    padded = np.pad(values, ((HISTORY_FRAMES - 1, 0), (0, 0)))
    offsets = np.arange(HISTORY_FRAMES)
    return padded[features.candidates[:, None] + offsets]


def aligned_embeddings(path: Path, cue_times: np.ndarray) -> np.ndarray:
    """Align pre-extracted causal hidden states. Reserve a column for availability."""
    with np.load(path, allow_pickle=False) as data:
        timestamps = np.asarray(data["time_s"], dtype=float)
        values = np.asarray(data["features"], dtype=np.float32)
    if (timestamps.ndim != 1 or values.ndim != 2 or len(timestamps) != len(values)
            or values.shape[1] == 0 or not np.isfinite(timestamps).all()
            or not np.isfinite(values).all() or np.any(timestamps < 0)
            or np.any(np.diff(timestamps) <= 0)):
        raise ValueError(f"{path}: expected sorted unique timestamps and finite [frames, dim] embeddings")
    aligned = np.zeros((len(cue_times), values.shape[1] + 1), dtype=np.float32)
    previous = np.searchsorted(timestamps, cue_times, side="right") - 1
    valid = previous >= 0
    valid[valid] = cue_times[valid] - timestamps[previous[valid]] <= 0.25
    aligned[valid, :-1] = values[previous[valid]]
    aligned[valid, -1] = 1
    return aligned
