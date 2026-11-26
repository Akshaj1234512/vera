#!/bin/bash
# Both PersonaPlex runs in results.md, on Lexi:
#   run A: raw 4096-dim states added to x_t for G and R (Rishabh's gr-train, unchanged)
#   run B: gr_pp.py, PCA-64 states for G only, R audio-only; both Gs scored under the same R
# Same settings as rishabh/README.md (meaning-weight 3, refine-epochs 20, dataset seeds 3/4).
# Reads Rishabh's data read-only; writes only under $W.
# Run on Lexi:  CUDA_VISIBLE_DEVICES=<free gpu> bash gr_pp_retrain.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)  # before any cd
W=/data/jiacheng_liang/vera-team/gr-personaplex
SRC=/data/rthakur/vera/data/seamless-motion          # Rishabh's Seamless streams (read-only)
PP=/data/jiacheng_liang/vera-team/personaplex-states  # states.npz per agent
CODE=$W/code/rishabh                                  # vera-gr-model branch checkout
PY=/data/jiacheng_liang/envs/gpu/bin/python
export TMPDIR=/data/jiacheng_liang/tmp OMP_NUM_THREADS=8
mkdir -p "$W"

# 0. Code: Rishabh's vera-gr-model branch, unmodified.
if [ ! -d "$W/code" ]; then
  git clone -q --branch vera-gr-model --single-branch https://github.com/Akshaj1234512/vera.git "$W/code"
fi
$PY -c "import mujoco, matplotlib" 2>/dev/null || $PY -m pip install -q "mujoco>=3.1" matplotlib pillow
cd "$CODE"

# 1. Rebuild every stream with PersonaPlex states (8 audio + 4096 states + 1 availability = 4105 context dims).
for split in dev test; do
  $PY - "$SRC/naturalistic-$split" "$PP/$split" "$W/streams/$split" <<'EOF'
import json, sys, subprocess
from pathlib import Path
src, pp, out = map(Path, sys.argv[1:])
index = json.loads((src / "streams.json").read_text())
for s in index["streams"]:
    agent, partner = s["agent"], s["partner"]
    target = out / agent / "stream.npz"
    if not target.is_file():
        subprocess.run([sys.executable, "-m", "speech_ai", "gr-prepare", str(src / partner / "audio-16k.wav"),
                        "--agent", str(src / agent / "audio-16k.wav"), "--teacher", str(src / agent / "teacher.npz"),
                        "--embeddings", str(pp / agent / "states.npz"), "--out", str(target)], check=True,
                       stdout=subprocess.DEVNULL)
    s["stream"] = f"{agent}/stream.npz"
(out / "streams.json").write_text(json.dumps(index, indent=2) + "\n")
print(out, len(index["streams"]), "streams")
EOF
done

# 2. Trials (same seeds as Rishabh: dev 3, test 4), 3. train, 4. held-out score.
$PY -m speech_ai gr-dataset "$W/streams/dev/streams.json" --out "$W/data/dev"
$PY -m speech_ai gr-dataset "$W/streams/test/streams.json" --seed 4 --out "$W/data/test"
$PY -m speech_ai gr-train "$W/data/dev/manifest.json" --meaning-weight 3 --refine-epochs 20 --out "$W/runs/selected"
$PY -m speech_ai gr-score "$W/data/test/manifest.json" --g "$W/runs/selected/g.pt" --r "$W/runs/selected/r.pt" \
  --out "$W/runs/selected/heldout-test-split.json"
echo "Run A done. Compare with audio-only: /data/rthakur/vera/runs/gr-real/selected/heldout-test-split.json"

# Run B: speech for G only, then score both Gs under Rishabh's R and under run B's own R.
export RISHABH_CODE=$CODE
R=/data/rthakur/vera/runs/gr-real/selected
$PY "$HERE/gr_pp.py" train "$W/data/dev/manifest.json" --components 64 --out "$W/runs/pp-g-k64"
for judge in "$R/r.pt:rishabh-r" "$W/runs/pp-g-k64/r.pt:own-r"; do
  $PY "$HERE/gr_pp.py" score "$W/data/test/manifest.json" --judge "${judge%%:*}" --audio-g "$R/g.pt" \
    --pp-g "$W/runs/pp-g-k64/g.pt" --out "$W/runs/pp-g-k64/heldout-under-${judge##*:}.json"
done
echo GR_PP_DONE
