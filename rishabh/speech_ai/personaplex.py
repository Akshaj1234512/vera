"""Keep the official inference environment independent of the local cue models."""

import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

from .sources import transcode

MODEL_REPO = "nvidia/personaplex-7b-v1"
DEFAULT_PROMPT = (
    "You are a conversational research assistant. Listen to the user and answer briefly. "
    "If the user corrects a detail, use the corrected detail in your next reply."
)


def run(input_wav: Path, output: Path, python: str, device: str, voice: str, prompt: str,
        seed: int, cpu_offload: bool, capture: bool) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    bridge = Path(__file__).with_name("personaplex_offline.py").resolve()
    with TemporaryDirectory(dir=output, prefix="personaplex-") as temporary:
        scratch = Path(temporary)
        partner, raw_agent, agent = (scratch / name for name in ("partner.wav", "agent-raw.wav", "agent.wav"))
        transcript = scratch / "tokens.json"
        trace, metadata = scratch / "states.npz", scratch / "run.json"
        transcode(input_wav, partner, channels=1, rate=24000)
        command = [python, str(bridge), str(metadata), str(trace) if capture else "-",
                   "--input-wav", str(partner), "--output-wav", str(raw_agent), "--output-text",
                   str(transcript), "--voice-prompt", voice, "--text-prompt", prompt,
                   "--hf-repo", MODEL_REPO, "--seed", str(seed), "--device", device]
        if cpu_offload:
            command.append("--cpu-offload")
        start = perf_counter()
        subprocess.run(command, check=True)
        wall_s = perf_counter() - start
        transcode(raw_agent, agent, channels=1, rate=24000)
        provenance = json.loads(metadata.read_text())
        provenance.update({
            "model_repo": MODEL_REPO, "input": str(input_wav.resolve()),
            "voice_prompt": voice, "text_prompt": prompt, "seed": seed, "device": device,
            "cpu_offload": cpu_offload, "captured_final_speech_states": capture,
            "run_wall_s_including_model_load": round(wall_s, 3),
        })
        metadata.write_text(json.dumps(provenance, indent=2) + "\n")
        files = [partner, agent, transcript, metadata]
        if capture:
            files.append(trace)
        for file in files:
            file.replace(output / file.name)
    return output / "run.json"
