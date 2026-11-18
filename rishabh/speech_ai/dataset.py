"""Load fully annotated recordings and hold out complete speaker/session groups."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .audio import Features, extract, load_tracks


@dataclass
class Clip:
    id: str
    group: str
    origin: str
    labels: np.ndarray
    features: Features
    embedding_path: Path | None


def load_manifest(manifest: Path, vad_db: float) -> list[Clip]:
    rows = json.loads(manifest.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must be a nonempty list of annotated clips")
    clips = []
    seen = set()
    for row in rows:
        clip_id = row["id"]
        if clip_id in seen:
            raise ValueError(f"Duplicate clip id: {clip_id}")
        seen.add(clip_id)
        if not row["group"] or not row["origin"]:
            raise ValueError(f"{clip_id}: group and origin are required")
        if row.get("ack_s") is None:
            raise ValueError(f"{clip_id}: complete annotation first; an empty list means reviewed with no events")
        partner = manifest.parent / row["partner"]
        agent = manifest.parent / row["agent"] if row.get("agent") else None
        features = extract(*load_tracks(partner, agent, row.get("stereo", False)), threshold_db=vad_db)
        labels = np.asarray(row["ack_s"], dtype=float)
        if (labels.ndim != 1 or not np.isfinite(labels).all() or np.any(labels < 0)
                or np.any(labels > features.time_s[-1]) or len(np.unique(labels)) != len(labels)):
            raise ValueError(f"{clip_id}: ack_s must contain unique times inside the clip")
        path = manifest.parent / row["embeddings"] if row.get("embeddings") else None
        clips.append(Clip(clip_id, row["group"], row["origin"], np.sort(labels), features, path))
    return clips


def split_groups(clips: list[Clip], seed: int) -> dict[str, list[Clip]]:
    groups = sorted({clip.group for clip in clips})
    if len(groups) < 3:
        raise ValueError("At least three disjoint speaker/session groups are required")
    shuffled = np.random.default_rng(seed).permutation(groups).tolist()
    count = max(1, round(0.2 * len(groups)))
    validation = set(shuffled[:count])
    test = set(shuffled[count:2 * count])
    split = {"train": [], "validation": [], "test": []}
    for clip in clips:
        destination = "validation" if clip.group in validation else "test" if clip.group in test else "train"
        split[destination].append(clip)
    return split


def match_events(predictions: np.ndarray, labels: np.ndarray, tolerance: float) -> list[tuple[int, int]]:
    """Match chronological events one-to-one within a symmetric tolerance."""
    matches = []
    p = y = 0
    while p < len(predictions) and y < len(labels):
        difference = predictions[p] - labels[y]
        if abs(difference) <= tolerance + 1e-9:
            matches.append((p, y))
            p += 1
            y += 1
        elif difference < 0:
            p += 1
        else:
            y += 1
    return matches
