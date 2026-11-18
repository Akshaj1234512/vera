"""Fetch NVIDIA's published PersonaPlex conversation audio and transcode audio with FFmpeg."""

import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from .audio import read_wav

PERSONAPLEX_PAGE = "https://research.nvidia.com/labs/adlr/personaplex/"
PERSONAPLEX_AUDIO = "https://research.nvidia.com/labs/adlr/files/personaplex/"
SAMPLES = {"assistant.mp3": PERSONAPLEX_AUDIO + "assistant.mp3",
           "backchannel.mp3": PERSONAPLEX_AUDIO + "backchannel.mp3"}


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
        subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location",
                        "--connect-timeout", "15", "--max-time", "180", "--retry", "2",
                        "--output", str(downloaded), url], check=True)
        if not downloaded.stat().st_size:
            raise ValueError(f"Empty response for {url}")
        downloaded.replace(output)


def prepare_personaplex(output: Path) -> Path:
    """Stereo WAVs with the human partner on the left and PersonaPlex on the right."""
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, url in SAMPLES.items():
        source = output / name
        fetch(url, source)
        wav = source.with_suffix(".wav")
        transcode(source, wav, rate=24000)
        samples, rate = read_wav(wav)
        if samples.shape[1] != 2:
            raise ValueError(f"{wav} should have partner audio on the left and agent audio on the right")
        with source.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        rows.append({"publisher": PERSONAPLEX_PAGE, "source_url": url, "file": name, "wav": wav.name,
                     "sha256": digest, "seconds": round(len(samples) / rate, 4), "channels": samples.shape[1]})
    metadata = output / "sources.json"
    metadata.write_text(json.dumps(rows, indent=2) + "\n")
    return metadata
