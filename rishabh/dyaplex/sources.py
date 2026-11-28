"""Where a participant's Seamless files live. Two layouts are supported:

OfficialLayout  what facebookresearch/seamless_interaction downloads (S3 or HF):
                <root>/<label>/<split>/<batch>/<archive>/<file_id>.{npz,json,wav,mp4}
                npz keys "<group>:<feature>" (e.g. smplh:body_pose); json has metadata:transcript /
                metadata:vad [I seamless_interaction fs.py gather_file_id_data_from_s3]
FeatureLayout   per-participant folders written by speech_ai.seamless / dyaplex fetchers:
                <root>/<file_id>/{smplh_body_pose.npy, ..., vad.jsonl, transcript.jsonl, audio-*.wav}

Optional SPECTRE output (dyaplex face-extract) is <face_dir>/<file_id>.npz in either layout.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

KEYS = ("smplh:body_pose", "smplh:global_orient", "smplh:left_hand_pose", "smplh:right_hand_pose",
        "smplh:translation", "smplh:is_valid", "movement_v4:FAUValue", "movement_v4:is_valid",
        "movement:FAUValue", "movement:is_valid")


@dataclass
class Record:
    file_id: str
    label: str
    split: str
    arrays: dict
    vad: list
    words: list
    wav: Path | None
    mp4: Path | None
    face: Path | None


def _words(lines) -> list[tuple[float, str]]:
    out = []
    for line in lines:
        for w in (line.get("words", []) if isinstance(line, dict) else []):
            if "start" in w and str(w.get("word", "")).strip():
                out.append((float(w["start"]), str(w["word"]).strip()))
    return out


def _vad(lines) -> list[tuple[float, float]]:
    return [(float(l["start"]), float(l["end"])) for l in lines if isinstance(l, dict) and "start" in l]


def _jsonl(path: Path) -> list:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.is_file() else []


class OfficialLayout:
    def __init__(self, root: Path, face_dir: Path | None = None):
        self.root, self.face_dir = Path(root), face_dir
        self.index = {p.stem: p for p in self.root.glob("*/*/*/*/*.npz")}

    def ids(self) -> list[str]:
        return sorted(self.index)

    def meta(self, file_id: str) -> tuple[str, str]:
        rel = self.index[file_id].relative_to(self.root).parts
        return rel[0], rel[1]

    def load(self, file_id: str) -> Record:
        npz = self.index[file_id]
        with np.load(npz, allow_pickle=False) as z:
            arrays = {k: z[k] for k in KEYS if k in z.files}
        side = json.loads(npz.with_suffix(".json").read_text()) if npz.with_suffix(".json").is_file() else {}
        label, split = self.meta(file_id)
        wav, mp4 = npz.with_suffix(".wav"), npz.with_suffix(".mp4")
        face = self.face_dir / f"{file_id}.npz" if self.face_dir else None
        return Record(file_id, label, split, arrays, _vad(side.get("metadata:vad", [])),
                      _words(side.get("metadata:transcript", [])), wav if wav.is_file() else None,
                      mp4 if mp4.is_file() else None, face if face and face.is_file() else None)


class FeatureLayout:
    def __init__(self, root: Path, label: str = "naturalistic", split: str = "dev", face_dir: Path | None = None):
        self.root, self.label, self.split, self.face_dir = Path(root), label, split, face_dir

    def ids(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if (p / "smplh_body_pose.npy").is_file())

    def meta(self, file_id: str) -> tuple[str, str]:
        return self.label, self.split

    def load(self, file_id: str) -> Record:
        folder = self.root / file_id
        arrays = {}
        for key in KEYS:
            path = folder / f"{key.replace(':', '_')}.npy"
            if path.is_file():
                arrays[key] = np.load(path, allow_pickle=False)
        wav = next((folder / n for n in ("audio-48k.wav", "audio-16k.wav") if (folder / n).is_file()), None)
        mp4 = next(iter(folder.glob("*.mp4")), None)
        face = (self.face_dir / f"{file_id}.npz") if self.face_dir else folder / "face_spectre.npz"
        return Record(file_id, self.label, self.split, arrays, _vad(_jsonl(folder / "vad.jsonl")),
                      _words(_jsonl(folder / "transcript.jsonl")), wav, mp4, face if face.is_file() else None)


def open_layout(root: Path, layout: str = "auto", face_dir: Path | None = None, label: str = "naturalistic",
                split: str = "dev"):
    root = Path(root)
    if layout == "official" or (layout == "auto" and any(root.glob("*/*/*/*/*.npz"))):
        return OfficialLayout(root, face_dir)
    return FeatureLayout(root, label, split, face_dir)


def pairs(layout) -> list[tuple[str, str, str]]:
    """(interaction key, participant A, participant B) for interactions with exactly two participants.
    Roles: A = first id, B = second [A]; training uses both orders."""
    groups: dict[str, list[str]] = {}
    for fid in layout.ids():
        groups.setdefault(fid.rsplit("_", 1)[0], []).append(fid)
    return [(k, *sorted(v)) for k, v in sorted(groups.items()) if len(v) == 2]


def make_official_fixture(src: Path, out: Path, n_pairs: int = 2, label: str = "naturalistic",
                          split: str = "dev") -> dict:
    """SMOKE TESTS: rewrite local per-feature folders as the official <label>/<split>/<batch>/<archive>/
    <file_id>.{npz,json,wav} layout (npz keys '<group>:<feature>', json metadata:vad / metadata:transcript)."""
    import shutil
    local = FeatureLayout(src, label, split)
    target = Path(out) / label / split / "0000" / "0000"
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for _, a, b in pairs(local)[:n_pairs]:
        for fid in (a, b):
            folder = Path(src) / fid
            arrays = {key: np.load(folder / f"{key.replace(':', '_')}.npy", allow_pickle=False)
                      for key in KEYS if (folder / f"{key.replace(':', '_')}.npy").is_file()}
            np.savez(target / f"{fid}.npz", **arrays)
            side = {"id": fid, "metadata:vad": _jsonl(folder / "vad.jsonl"),
                    "metadata:transcript": _jsonl(folder / "transcript.jsonl")}
            (target / f"{fid}.json").write_text(json.dumps(side))
            wav = next((folder / n for n in ("audio-48k.wav", "audio-16k.wav") if (folder / n).is_file()), None)
            if wav:
                shutil.copy(wav, target / f"{fid}.wav")
            written.append(fid)
    return {"root": str(out), "participants": written}
