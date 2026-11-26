"""Capture PersonaPlex 7B states for every Seamless stream in a streams.json, across several GPUs.

PersonaPlex hears each stream's partner audio (the person the agent is listening to), matching the
DyaPlex setup. One states.npz per stream lands in OUT/<split>/<agent>/, ready for
`gr-prepare --embeddings`. Finished streams are skipped, so the script can be rerun after a crash.

usage: python pp_extract.py STREAMS_JSON OUT_DIR --gpus 2,3,4 [--limit N]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

# Rishabh's code (provides `python -m speech_ai personaplex`) and the PersonaPlex Python env on Lexi.
CODE = Path(os.environ.get("RISHABH_CODE", "/data/jiacheng_liang/vera-team/rishabh"))
PY = os.environ.get("PERSONAPLEX_PYTHON", "/data/jiacheng_liang/envs/personaplex/bin/python")

ap = argparse.ArgumentParser()
ap.add_argument("streams", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--gpus", required=True)
ap.add_argument("--limit", type=int)
args = ap.parse_args()

index = json.loads(args.streams.read_text())
root = args.streams.parent
jobs = index["streams"][: args.limit]
out_root = args.out.resolve() / index["split"]  # absolute: the subprocess runs from CODE
gpus = Queue()
for g in args.gpus.split(","):
    gpus.put(g)

env = dict(os.environ, OMP_NUM_THREADS="8", HF_HOME="/data/jiacheng_liang/.cache/huggingface",
           TMPDIR="/data/jiacheng_liang/tmp")


def run(job):
    partner_wav = (root / job["stream"]).parent.parent / job["partner"] / "audio-16k.wav"
    out = out_root / job["agent"]
    if (out / "states.npz").is_file():
        return job["agent"], "skip", 0.0
    gpu = gpus.get()
    try:
        out.mkdir(parents=True, exist_ok=True)
        start = time.time()
        with open(out / "log.txt", "w") as log:
            rc = subprocess.run([PY, "-m", "speech_ai", "personaplex", str(partner_wav), "--python", PY,
                                 "--capture-states", "--out", str(out)],
                                cwd=CODE, env=dict(env, CUDA_VISIBLE_DEVICES=gpu),
                                stdout=log, stderr=subprocess.STDOUT).returncode
        (out / "source.json").write_text(json.dumps({**job, "partner_wav": str(partner_wav),
                                                     "streams_json": str(args.streams)}, indent=2) + "\n")
        return job["agent"], "ok" if rc == 0 else f"FAILED rc={rc} (see {out}/log.txt)", time.time() - start
    finally:
        gpus.put(gpu)


print(f"{len(jobs)} streams on GPUs {args.gpus} -> {out_root}", flush=True)
with ThreadPoolExecutor(max_workers=gpus.qsize()) as pool:
    for n, (agent, status, secs) in enumerate(pool.map(run, jobs), 1):
        print(f"[{n}/{len(jobs)}] {agent}: {status} ({secs:.0f} s)", flush=True)
print("PP_DONE", flush=True)
sys.exit(0)
