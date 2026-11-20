"""Real human conversational motion from Meta's Seamless Interaction dataset (CC-BY-NC 4.0).

DyaPlex is trained on this dataset and its code and weights are not released, so
Seamless participants' measured motion is the closest available stand-in for DyaPlex
output: the same people, conversations and kind of signal, measured rather than
generated. Each participant becomes a teacher NPZ in the M_h format (see human.py)
whose ``source`` says exactly that. Nothing here is DyaPlex output.

Files come from https://dl.fbaipublicfiles.com/seamless_interaction/<label>/<split>/...
exactly as the dataset's own downloader builds them (src/seamless_interaction/fs.py).
"""

import csv
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

import numpy as np

from .human import HUMAN_CHANNELS, seamless_to_human
from .sources import fetch

BASE = "https://dl.fbaipublicfiles.com/seamless_interaction"
FILELIST = "https://raw.githubusercontent.com/facebookresearch/seamless_interaction/main/assets/filelist.csv"
FEATURES = ("movement_v4/FAUValue", "movement_v4/gaze_encodings", "movement_v4/is_valid",
            "movement/FAUValue", "movement/gaze_encodings", "movement/is_valid",
            "smplh/body_pose", "smplh/is_valid")
MIN_VALID = 0.5
SOURCE = "seamless_interaction_measured_human_motion_not_dyaplex"


def select(filelist: Path, label: str, split: str, count: int, seed: int, include=()) -> list[tuple[str, list[str]]]:
    """Interactions where both participants have movement features, one per session."""
    rows = [r for r in csv.DictReader(filelist.open()) if r["label"] == label and r["split"] == split]
    by_interaction: dict[str, list[str]] = {}
    moving = {r["file_id"] for r in rows if r["has_imitator_movement"] == "1"}
    for r in rows:
        by_interaction.setdefault(r["file_id"].rsplit("_", 1)[0], []).append(r["file_id"])
    pairs = {k: sorted(v) for k, v in by_interaction.items() if len(v) == 2 and set(v) <= moving}
    chosen, sessions = [], set()
    for key in include:
        if key not in pairs:
            raise ValueError(f"{key} lacks movement features for both participants")
        chosen.append(key)
        sessions.add(key.rsplit("_", 1)[0])
    for key in np.random.default_rng(seed).permutation(sorted(pairs)).tolist():
        if len(chosen) >= count:
            break
        if key.rsplit("_", 1)[0] not in sessions:  # one interaction per session keeps groups distinct
            chosen.append(key)
            sessions.add(key.rsplit("_", 1)[0])
    return [(key, pairs[key]) for key in chosen]


def download(label: str, split: str, file_id: str, out: Path) -> Path:
    """Fetch features, VAD and audio for one participant; keep a 16 kHz mono WAV."""
    folder = out / file_id
    folder.mkdir(parents=True, exist_ok=True)
    for feature in FEATURES:
        try:
            fetch(f"{BASE}/{label}/{split}/{feature}/{file_id}.npy", folder / f"{feature.replace('/', '_')}.npy")
        except subprocess.CalledProcessError:
            if feature.startswith("smplh"):
                raise
            # One movement version may be absent; convert() uses whichever is complete.
    try:  # VAD is kept for reference only; speech activity comes from the audio itself.
        fetch(f"{BASE}/{label}/{split}/metadata/vad/{file_id}.jsonl", folder / "vad.jsonl")
    except (subprocess.CalledProcessError, ValueError):
        pass
    wav = folder / "audio-16k.wav"
    if not wav.is_file():
        original = folder / "audio-48k.wav"
        fetch(f"{BASE}/{label}/{split}/audio/{file_id}.wav", original)
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(original), "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_s16le", str(wav)], check=True)
        original.unlink()
    return folder


def convert(folder: Path) -> dict:
    """Write teacher.npz (30 Hz M_h) for one downloaded participant."""
    load = lambda name: np.load(folder / f"{name}.npy", allow_pickle=False)
    complete = lambda v: all((folder / f"{v}_{n}.npy").is_file() for n in ("FAUValue", "gaze_encodings", "is_valid"))
    # movement_v4 is still being rolled out; fall back to v1 where v4 is missing or marked invalid.
    version = ("movement_v4" if complete("movement_v4") and (load("movement_v4_is_valid") > 0).mean() >= MIN_VALID
               else "movement")
    if not complete(version):
        raise ValueError("no complete movement feature set")
    human, valid, stats = seamless_to_human(load(f"{version}_FAUValue"), load(f"{version}_gaze_encodings"),
                                            load(f"{version}_is_valid"), load("smplh_body_pose"),
                                            load("smplh_is_valid"))
    if valid.mean() < MIN_VALID:
        raise ValueError(f"only {valid.mean():.0%} of frames have valid face and body motion")
    stats["movement_version"] = version
    # Seamless frame k is video time k / 30 s.
    times = np.arange(len(human)) / 30.0 + 1e-6
    np.savez_compressed(folder / "teacher.npz", time_s=times, channels=np.array(HUMAN_CHANNELS), human=human,
                        human_valid=valid.astype(np.float32), source=np.array(SOURCE))
    (folder / "teacher.json").write_text(json.dumps({"source": SOURCE, **stats}, indent=2) + "\n")
    return stats


def prepare(label: str, split: str, count: int, out: Path, seed: int = 5, include=()) -> Path:
    """Download, convert and align ``count`` interactions; returns an index of streams."""
    from .gr_data import prepare_stream

    out.mkdir(parents=True, exist_ok=True)
    filelist = out / "filelist.csv"
    fetch(FILELIST, filelist)
    chosen = select(filelist, label, split, count, seed, include)
    jobs = [fid for _, ids in chosen for fid in ids]
    def safe(fid):
        try:
            return download(label, split, fid, out)
        except (subprocess.CalledProcessError, ValueError) as error:
            print(f"skip {fid}: download failed ({error})")
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        folders = dict(zip(jobs, pool.map(safe, jobs)))
    streams = []
    for interaction, (a, b) in chosen:
        for agent, partner in ((a, b), (b, a)):
            if folders[agent] is None or folders[partner] is None:
                continue
            try:
                stats = convert(folders[agent])
            except ValueError as error:
                print(f"skip {agent}: {error}")
                continue
            stream = folders[agent] / "stream.npz"
            report = prepare_stream(folders[partner] / "audio-16k.wav", folders[agent] / "audio-16k.wav", False,
                                    folders[agent] / "teacher.npz", None, stream)
            streams.append({"stream": str(stream.relative_to(out)), "agent": agent, "partner": partner,
                            "interaction": interaction, "session": interaction.rsplit("_", 1)[0],
                            "participants": sorted([agent.rsplit("_", 1)[1], partner.rsplit("_", 1)[1]]),
                            "teacher_coverage": report["teacher"]["coverage"], **stats})
    index = out / "streams.json"
    index.write_text(json.dumps({"label": label, "split": split, "source": SOURCE,
                                 "license": "CC-BY-NC 4.0 (Seamless Interaction)", "streams": streams}, indent=2) + "\n")
    return index
