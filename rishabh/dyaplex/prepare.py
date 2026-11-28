"""Seamless Interaction -> prepared DyaPlex data (§5 'Datasets', App. C, D.3).

    python -m dyaplex prepare --root <seamless download> --out <prepared> --workers 32

Output
  <out>/parts/<part>/<file_id>.npy   float16 [T25, D_part] raw (unnormalized) part features
  <out>/meta/<file_id>.json          frames, validity, VAD, transcript words, audio/video paths, stats
  <out>/valid/<file_id>.npz          body / face validity masks at 25 fps
  <out>/manifest.csv                 one row per pair: every filter decision and why
  <out>/norm.npz                     per-part mean / std over valid frames of the train split
  <out>/splits.json                  train / val / test / test330 pair keys
Storage: 371 float16 per frame -> ~67 MB per participant-hour (~460 GB for the full corpus).

Filters [P C.1, C.2, D.1, §5]
  clip   either participant's video landscape / rotated (640x480) / unreadable -> drop pair
  frame  smplh:is_valid == 0 masked; fewer than 8 valid frames -> drop pair
  voice  no ~10 s voice clip for either participant -> drop pair (paper: ~1.4%) [A rule below]
  face   >= 20% zero-face frames in either participant -> excluded from face-codec training only
  audio  test subset only: both recordings present, readable, non-silent and with speech [A rule]
Splits: train = official train, val = official dev, test = official test (+ audio check);
test330 = 330 test pairs in sha1(key) order [A: the paper's 330 ids are unpublished].
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from . import config as C
from . import data as D
from .sources import open_layout, pairs

VOICE_CLIP_S = 10.0
VOICE_MIN_OWN_SPEECH = 0.5        # [A] best 10 s window must be >= 50% the participant's own speech
AUDIO_MIN_RMS_DBFS = -50.0        # [A] test audio check
AUDIO_MIN_SPEECH = 0.05           # [A] at least 5% VAD speech
TEST_SUBSET = 330


def probe_local_video(path: Path | None) -> dict:
    if path is None:
        return {"status": "missing", "keep": None}
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
           "-of", "json", str(path)]
    try:
        stream = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60).stdout)["streams"][0]
        w, h = int(stream.get("width", 0)), int(stream.get("height", 0))
    except Exception:
        return {"status": "unreadable", "keep": False}
    if w == 0 or h == 0:
        return {"status": "unreadable", "keep": False}
    if (w, h) == (640, 480):
        return {"status": "rotated", "width": w, "height": h, "keep": False}
    if w > h:
        return {"status": "landscape", "width": w, "height": h, "keep": False}
    return {"status": "portrait_or_square", "width": w, "height": h, "keep": True}


def best_voice_window(vad: list, duration_s: float) -> tuple[float, float]:
    """(start, own-speech fraction) of the best 10 s window [A]."""
    hz = 10.0
    lab = D.vad_labels(vad, int(duration_s * hz) + 1, hz).astype(float)
    win = int(VOICE_CLIP_S * hz)
    if len(lab) < win:
        return 0.0, float(lab.mean()) if len(lab) else 0.0
    sums = np.convolve(lab, np.ones(win), "valid") / win
    k = int(np.argmax(sums))
    return k / hz, float(sums[k])


def audio_check(wav: Path | None, vad: list, duration_s: float) -> dict:
    if wav is None:
        return {"ok": False, "reason": "missing"}
    try:
        audio, rate = D.read_wav(wav)
    except Exception as error:
        return {"ok": False, "reason": f"unreadable: {str(error)[:80]}"}
    rms = 20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-12)
    speech = float(D.vad_labels(vad, int(duration_s * 10) + 1, 10.0).mean())
    ok = rms > AUDIO_MIN_RMS_DBFS and speech >= AUDIO_MIN_SPEECH and abs(len(audio) / rate - duration_s) < 2.0
    return {"ok": bool(ok), "rms_dbfs": float(rms), "speech": speech, "audio_s": len(audio) / rate}


def process_participant(root: str, layout_kind: str, face_dir: str | None, face_mode: str, out: str,
                        file_id: str, video_check: str) -> dict:
    layout = open_layout(Path(root), layout_kind, Path(face_dir) if face_dir else None)
    out = Path(out)
    rec = layout.load(file_id)
    row = {"file_id": file_id, "label": rec.label, "split": rec.split}
    if video_check == "local":
        row["video"] = probe_local_video(rec.mp4)
    elif video_check == "remote":
        row["video"] = D.probe_video(rec.label, rec.split, file_id)
    else:
        row["video"] = {"status": "skipped", "keep": None}
    try:
        p = D.load_record(rec, face_mode)
    except Exception as error:
        row.update(ok=False, reason=str(error)[:200])
        return row
    feats = D.part_features(p)
    for part, x in feats.items():
        (out / "parts" / part).mkdir(parents=True, exist_ok=True)
        np.save(out / "parts" / part / f"{file_id}.npy", x.astype(np.float16))
    (out / "valid").mkdir(exist_ok=True)
    np.savez_compressed(out / "valid" / f"{file_id}.npz", body=p.valid, face=p.face_valid)
    duration = p.frames / C.MOTION_FPS
    voice_start, voice_own = best_voice_window(rec.vad, duration)
    meta = {"file_id": file_id, "label": rec.label, "split": rec.split, "frames25": p.frames,
            "tokens": p.frames // 2, "wav": str(Path(rec.wav).resolve()) if rec.wav else None,
            "mp4": str(Path(rec.mp4).resolve()) if rec.mp4 else None,
            "vad": rec.vad, "words": rec.words, "voice_clip_start_s": voice_start, "voice_own_speech": voice_own,
            "stats": p.stats, "audio": audio_check(rec.wav, rec.vad, duration)}
    (out / "meta").mkdir(exist_ok=True)
    (out / "meta" / f"{file_id}.json").write_text(json.dumps(meta) + "\n")
    sums = {}
    for part, x in feats.items():
        ok = D.part_valid(p)[part]
        xv = x[ok].astype(np.float64)
        sums[part] = (xv.sum(0), (xv ** 2).sum(0), int(ok.sum()))
    row.update(ok=True, frames25=p.frames, valid=float(p.valid.mean()), zero_face=p.stats["zero_face_fraction"],
               voice_own=voice_own, audio_ok=meta["audio"]["ok"], face_source=p.stats["face_source"], sums=sums)
    return row


def run(root: Path, out: Path, layout_kind: str = "auto", face_dir: Path | None = None, face_mode: str = "auto",
        workers: int = 8, video_check: str = "local", limit: int | None = None, remap_splits: bool = False,
        log=print) -> dict:
    """remap_splits: SMOKE TESTS ONLY - assign pairs to train/val/test 60/20/20 by sha1(key)."""
    out.mkdir(parents=True, exist_ok=True)
    layout = open_layout(root, layout_kind, face_dir)
    all_pairs = pairs(layout)[:limit] if limit else pairs(layout)
    ids = sorted({pid for _, a, b in all_pairs for pid in (a, b)})
    log(f"{len(all_pairs)} pairs / {len(ids)} participants under {root}")
    rows = {}
    args = (str(root), layout_kind, str(face_dir) if face_dir else None, face_mode, str(out))
    if workers <= 1:
        for k, fid in enumerate(ids):
            rows[fid] = process_participant(*args, fid, video_check)
    else:
        with ProcessPoolExecutor(workers) as pool:
            futures = {pool.submit(process_participant, *args, fid, video_check): fid for fid in ids}
            for k, fut in enumerate(as_completed(futures)):
                fid = futures[fut]
                try:
                    rows[fid] = fut.result()
                except Exception as error:
                    rows[fid] = {"file_id": fid, "ok": False, "reason": f"worker: {str(error)[:200]}"}
                if (k + 1) % 500 == 0:
                    log(f"  {k + 1}/{len(ids)} participants")
    manifest, splits = [], {"train": [], "val": [], "test": []}
    acc = {"train": {}}
    for key, a, b in all_pairs:
        ra, rb = rows[a], rows[b]
        reasons = []
        for r in (ra, rb):
            if not r.get("ok"):
                reasons.append(f"{r['file_id'][-5:]}: {r.get('reason')}")
            if r.get("video", {}).get("keep") is False:
                reasons.append(f"{r['file_id'][-5:]}: video {r['video']['status']}")
            if r.get("ok") and r.get("voice_own", 0) < VOICE_MIN_OWN_SPEECH:
                reasons.append(f"{r['file_id'][-5:]}: no clean voice clip")
        split = {"train": "train", "dev": "val", "test": "test"}.get(ra.get("split", ""), ra.get("split", ""))
        if remap_splits:
            h = int(hashlib.sha1(key.encode()).hexdigest(), 16) % 10
            split = "train" if h < 6 else "val" if h < 8 else "test"
        kept = not reasons
        audio_ok = bool(ra.get("audio_ok")) and bool(rb.get("audio_ok"))
        face_ok = kept and ra["zero_face"] < 0.2 and rb["zero_face"] < 0.2
        manifest.append({"pair": key, "a": a, "b": b, "label": ra.get("label"), "split": split, "kept": kept,
                         "reasons": "; ".join(reasons), "face_codec_ok": face_ok, "audio_ok": audio_ok,
                         "frames25": min(ra.get("frames25", 0), rb.get("frames25", 0)),
                         "video_a": ra.get("video", {}).get("status"), "video_b": rb.get("video", {}).get("status")})
        if kept and split in splits and (split != "test" or audio_ok):
            splits[split].append(key)
            if split == "train":
                for r in (ra, rb):
                    for part, (s1, s2, n) in r["sums"].items():
                        if part == "face" and not face_ok:
                            continue
                        a1, a2, an = acc["train"].get(part, (0.0, 0.0, 0))
                        acc["train"][part] = (a1 + s1, a2 + s2, an + n)
    splits["test330"] = sorted(splits["test"], key=lambda k: hashlib.sha1(k.encode()).hexdigest())[:TEST_SUBSET]
    D.write_manifest(out / "manifest.csv", manifest)
    (out / "splits.json").write_text(json.dumps(splits, indent=1) + "\n")
    norm = {}
    for part, (s1, s2, n) in acc["train"].items():
        mean = s1 / max(n, 1)
        std = np.sqrt(np.maximum(s2 / max(n, 1) - mean ** 2, 0))
        norm[f"{part}_mean"] = mean.astype(np.float32)
        norm[f"{part}_std"] = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    if norm:
        np.savez(out / "norm.npz", **norm)
    kept_hours = sum(m["frames25"] for m in manifest if m["kept"] and m["split"] == "train") / C.MOTION_FPS / 3600
    summary = {"pairs": len(all_pairs), "kept": sum(m["kept"] for m in manifest),
               "train_pairs": len(splits["train"]), "train_hours": kept_hours, "val_pairs": len(splits["val"]),
               "test_pairs": len(splits["test"]), "test330": len(splits["test330"]),
               "dropped_by_reason": _reason_counts(manifest), "face_codec_pairs": sum(m["face_codec_ok"] for m in manifest),
               "paper": {"train_pairs": 57947, "train_hours": 3435, "test_pairs": 330},
               "splits_remapped_for_smoke_test": remap_splits}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _reason_counts(manifest: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for m in manifest:
        for r in filter(None, m["reasons"].split("; ")):
            kind = r.split(": ", 1)[1] if ": " in r else r
            kind = "video " + kind.split(" ")[1] if kind.startswith("video") else kind[:60]
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def load_norm(prepared: Path) -> dict:
    with np.load(prepared / "norm.npz") as z:
        return {p: {"mean": z[f"{p}_mean"], "std": z[f"{p}_std"]} for p in C.PARTS if f"{p}_mean" in z.files}


def load_meta(prepared: Path, file_id: str) -> dict:
    return json.loads((prepared / "meta" / f"{file_id}.json").read_text())


def load_parts(prepared: Path, file_id: str, mmap: bool = True) -> dict[str, np.ndarray]:
    return {p: np.load(prepared / "parts" / p / f"{file_id}.npy", mmap_mode="r" if mmap else None) for p in C.PARTS}


def load_valid(prepared: Path, file_id: str) -> dict[str, np.ndarray]:
    with np.load(prepared / "valid" / f"{file_id}.npz") as z:
        return {"body": z["body"], "face": z["face"]}


def split_pairs(prepared: Path, split: str) -> list[tuple[str, str, str]]:
    keys = set(json.loads((prepared / "splits.json").read_text())[split])
    rows = list(csv.DictReader((prepared / "manifest.csv").open()))
    return [(r["pair"], r["a"], r["b"]) for r in rows if r["pair"] in keys]
