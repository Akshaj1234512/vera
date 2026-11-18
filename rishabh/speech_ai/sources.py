"""Fetch a small, traceable set of published audio and video samples."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

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
        subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location",
                        "--connect-timeout", "15", "--max-time", "180", "--retry", "2",
                        "--output", str(downloaded), url], check=True)
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
    annotation = output / "annotation-template.json"
    if pending and not annotation.exists():
        annotation.write_text(json.dumps(pending, indent=2) + "\n")
    metadata = output / "sources.json"
    metadata.write_text(json.dumps(rows, indent=2) + "\n")
    return metadata


SEAMLESS_PAGE = "https://huggingface.co/datasets/facebook/seamless-interaction"
SEAMLESS_BASE = "https://dl.fbaipublicfiles.com/seamless_interaction/naturalistic/dev"
# Five complete dyads from Meta's filelist, with no participant repeated here.
SEAMLESS_PAIRS = (
    ("V00_S0080_I00000135", "P0116", "P0115"),
    ("V00_S0152_I00000135", "P0217", "P0216"),
    ("V00_S0255_I00000125", "P0350", "P0351"),
    ("V00_S0274_I00000125", "P0371", "P0370"),
    ("V00_S0347_I00000126", "P0466", "P0467"),
)


def prepare_seamless(output: Path) -> Path:
    """Download a fixed set of paired tracks and the corresponding listener videos."""
    output.mkdir(parents=True, exist_ok=True)
    def download_pair(pair: tuple[str, str, str]) -> dict:
        interaction, speaker, listener = pair
        speaker_id, listener_id = f"{interaction}_{speaker}", f"{interaction}_{listener}"
        paths = {
            "partner": f"{speaker_id}.wav", "agent": f"{listener_id}.wav", "video": f"{listener_id}.mp4",
        }
        sources = {}
        for name, filename in paths.items():
            modality = "video" if name == "video" else "audio"
            url = f"{SEAMLESS_BASE}/{modality}/{filename}"
            fetch(url, output / filename)
            sources[name] = url
        for role in ("partner", "agent"):
            original = output / paths[role]
            pcm = original.with_name(original.stem + "-pcm.wav")
            if not pcm.is_file():
                transcode(original, pcm, channels=1, rate=24000)
            paths[role] = pcm.name
        return {**paths, "id": interaction, "group": interaction.rsplit("_", 1)[0],
                "origin": "seamless-interaction", "publisher": SEAMLESS_PAGE,
                "sources": sources, "ack_s": None}

    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest = list(pool.map(download_pair, SEAMLESS_PAIRS))
    path = output / "annotation-template.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
