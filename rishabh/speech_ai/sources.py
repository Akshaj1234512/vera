"""Fetch a small, traceable set of published audio and video samples."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from .audio import read_wav

PERSONAPLEX_PAGE = "https://research.nvidia.com/labs/adlr/personaplex/"
PERSONAPLEX_AUDIO = "https://research.nvidia.com/labs/adlr/files/personaplex/"
VICO_PAGE = "https://project.mhzhou.com/vico/"
SAMPLES = {
    "personaplex": {
        "assistant.mp3": PERSONAPLEX_AUDIO + "assistant.mp3",
        "backchannel.mp3": PERSONAPLEX_AUDIO + "backchannel.mp3",
    },
    "vico": {
        f"{attitude}.mp4": VICO_PAGE + f"assets/vico/demo_video_{attitude}.mp4"
        for attitude in ("positive", "neutral", "negative")
    },
}


def transcode(source: Path, destination: Path, *, channels: int | None = None, rate: int | None = None) -> None:
    options = []
    if channels is not None:
        options.extend(["-ac", str(channels)])
    if rate is not None:
        options.extend(["-ar", str(rate)])
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
                    "-map", "0:a:0", *options, "-c:a", "pcm_s16le", str(destination)], check=True)


def fetch(url: str, output: Path) -> None:
    if output.is_file() and output.stat().st_size:
        return
    with TemporaryDirectory(dir=output.parent) as scratch:
        downloaded = Path(scratch) / output.name
        with urlopen(url, timeout=30) as response, downloaded.open("wb") as file:
            shutil.copyfileobj(response, file)
        if not downloaded.stat().st_size:
            raise ValueError(f"Empty response for {url}")
        downloaded.replace(output)


def prepare_samples(collection: str, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    rows, pending = [], []
    for name, url in SAMPLES[collection].items():
        source = output / name
        fetch(url, source)
        wav = source.with_suffix(".wav")
        transcode(source, wav, rate=24000)
        samples, rate = read_wav(wav)
        if collection == "personaplex" and samples.shape[1] != 2:
            raise ValueError(f"{wav} should have partner audio on the left and agent audio on the right")
        with source.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        rows.append({
            "publisher": PERSONAPLEX_PAGE if collection == "personaplex" else VICO_PAGE,
            "source_url": url, "file": name, "wav": wav.name,
            "sha256": digest,
            "seconds": round(len(samples) / rate, 4), "channels": samples.shape[1],
        })
        if collection == "vico":
            pending.append({
                "id": source.stem, "group": "vico-preview-unverified-identities", "origin": "vico-preview",
                "partner": wav.name, "video": source.name, "ack_s": None,
            })
    if pending:
        (output / "annotation-template.json").write_text(json.dumps(pending, indent=2) + "\n")
    metadata = output / "sources.json"
    metadata.write_text(json.dumps(rows, indent=2) + "\n")
    return metadata
