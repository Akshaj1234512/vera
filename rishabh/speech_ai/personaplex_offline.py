"""Run NVIDIA's offline entry point and optionally collect final speech states.

This small bridge also runs in a separate Python environment where NVIDIA Moshi
is installed. The collector accepts only frames from the input WAV, after
warming up and prefilling the voice and system prompts.
"""

from contextlib import contextmanager
import json
from pathlib import Path
import sys

import numpy as np


@contextmanager
def capture_states(offline, destination: Path):
    original_load = offline.lm_load_audio
    original_step = offline.LMGen.step
    original_output = offline.LMGen.process_transformer_output
    ready = capturing = False
    frame_count = 0
    current = None
    times, states = [], []

    def load_audio(*args, **kwargs):
        nonlocal ready
        audio = original_load(*args, **kwargs)
        ready = True
        return audio

    def capture_step(gen, *args, **kwargs):
        nonlocal capturing, current, frame_count
        if not ready:
            return original_step(gen, *args, **kwargs)
        current = None
        capturing = True
        frame_count += 1
        try:
            return original_step(gen, *args, **kwargs)
        finally:
            capturing = False
            if current is not None:
                times.append(round(frame_count * gen._frame_size / gen._sample_rate, 6))
                states.append(current)

    def capture_output(gen, transformer_out, *args, **kwargs):
        nonlocal current
        if capturing:
            # Clone the vector before the next step reuses its CUDA graph buffers.
            current = transformer_out[0, -1].detach().float().cpu().numpy().copy()
        return original_output(gen, transformer_out, *args, **kwargs)

    offline.lm_load_audio = load_audio
    offline.LMGen.step = capture_step
    offline.LMGen.process_transformer_output = capture_output
    try:
        yield
    finally:
        offline.lm_load_audio = original_load
        offline.LMGen.step = original_step
        offline.LMGen.process_transformer_output = original_output
    if not states:
        raise RuntimeError("No PersonaPlex speech states captured. Check the NVIDIA Moshi version.")
    np.savez_compressed(destination, time_s=np.array(times), features=np.stack(states))


def main() -> None:
    from importlib.metadata import PackageNotFoundError, distribution
    from moshi import offline

    metadata_path, trace_path = sys.argv[1:3]
    sys.argv[1:3] = []
    try:
        package = distribution("moshi")
        source = json.loads(package.read_text("direct_url.json") or "{}")
        git_info = source.get("vcs_info", {})
        provenance = {"moshi_version": package.version, "source_revision": git_info.get("commit_id")}
    except PackageNotFoundError:
        provenance = {"moshi_version": None, "source_revision": None}

    if trace_path == "-":
        offline.main()
    else:
        with capture_states(offline, Path(trace_path)):
            offline.main()
    Path(metadata_path).write_text(json.dumps(provenance) + "\n")


if __name__ == "__main__":
    main()
