#!/usr/bin/env bash
# Runs every DyaPlex training/eval stage end to end at toy scale on CPU, using local Seamless pairs.
# Proves the plumbing only. Usage (from rishabh/):  bash scripts/dyaplex_smoke_pipeline.sh [workdir]
set -euo pipefail
PY=${PY:-.venv/bin/python}
W=${1:-/tmp/dyaplex-smoke}
SRC=${SRC:-data/seamless-motion/naturalistic-dev}
export OMP_NUM_THREADS=1
rm -rf "$W"; mkdir -p "$W"
echo "== official-layout reader"
$PY -m dyaplex make-official-fixture --src "$SRC" --out "$W/official" --pairs 2
$PY -m dyaplex prepare --root "$W/official" --layout official --out "$W/prepared-official" --video-check skip --workers 1
echo "== prepare (10 local pairs; splits remapped for the smoke test)"
$PY -m dyaplex prepare --root "$SRC" --layout feature --out "$W/prepared" --video-check skip --workers 4 --limit 10 --remap-splits
echo "== codecs (4 parts, 40 iterations each) + resume + tokenize"
for part in upper hands lower_trans face; do
  $PY -m dyaplex train-codec --prepared "$W/prepared" --part $part --out "$W/codecs" --iters 40 --batch 16 --eval-every 20 --device cpu
done
$PY -m dyaplex train-codec --prepared "$W/prepared" --part upper --out "$W/codecs" --iters 60 --batch 16 --eval-every 20 --device cpu
$PY -m dyaplex tokenize --prepared "$W/prepared" --codecs "$W/codecs" --out "$W/prepared/tokens" --device cpu
echo "== speech streams (stand-in Mimi / tokenizer)"
$PY -m dyaplex build-streams --prepared "$W/prepared" --mode tiny
echo "== tower: 2-process DDP, face variant, resume, two ablations"
TR="$(dirname "$PY")/torchrun --nnodes 1 --nproc_per_node 2 --rdzv-backend static --master-addr 127.0.0.1 --node-rank 0"
$TR --master-port 29541 -m dyaplex.train_tower --prepared "$W/prepared" --tokens "$W/prepared/tokens" --out "$W/tower" \
  --config tiny --variant face --iterations 4 --batch-size 8 --micro-batch 2 --log-every 1 --val-every 2 --val-windows 4 --ckpt-every 2
$TR --master-port 29542 -m dyaplex.train_tower --prepared "$W/prepared" --tokens "$W/prepared/tokens" --out "$W/tower" \
  --config tiny --variant face --iterations 6 --batch-size 8 --micro-batch 2 --log-every 1 --val-every 6 --val-windows 4 --ckpt-every 6
for ab in no_partner no_cross_rope; do
  $PY -m dyaplex train-tower --prepared "$W/prepared" --tokens "$W/prepared/tokens" --out "$W/tower-$ab" --config tiny \
    --variant body --ablation $ab --iterations 2 --batch-size 2 --micro-batch 2 --val-every 2 --val-windows 2 --ckpt-every 2
done
echo "== evaluate (synthetic skeleton: smoke only) + baseline export + baseline scoring"
$PY -m dyaplex evaluate --prepared "$W/prepared" --tokens "$W/prepared/tokens" --codecs "$W/codecs" \
  --tower "$W/tower/tower_000006.pt" --synthetic-skeleton --config tiny --split train --limit 2 --seconds 6 --out "$W/eval.json" --device cpu
$PY -m dyaplex export-baselines --prepared "$W/prepared" --out "$W/baselines" --splits train
$PY - "$W" <<'PYEOF'
import json, sys, numpy as np
from pathlib import Path
w = Path(sys.argv[1]); rows = json.loads((w / "baselines/train/pairs.json").read_text())
out = w / "gt-as-baseline"; out.mkdir(exist_ok=True)
for r in rows[:2]:
    z = np.load(w / "baselines/train" / f"{r['b']}.npz")
    np.savez(out / f"{r['pair']}__{r['b']}.npz", pose6d=z["a2p_132"].reshape(len(z["a2p_132"]), 22, 6))
PYEOF
$PY -m dyaplex evaluate --prepared "$W/prepared" --tokens "$W/prepared/tokens" --predictions "$W/gt-as-baseline" \
  --synthetic-skeleton --split train --limit 2 --seconds 6 --out "$W/eval-gt-as-baseline.json"
echo "== infer -> Samhita's rig"
$PY -m dyaplex infer --prepared "$W/prepared" --tokens "$W/prepared/tokens" --codecs "$W/codecs" --tower "$W/tower/tower_000006.pt" \
  --pair V00_S0152_I00000377 --agent V00_S0152_I00000377_P0217 --start-s 30 --seconds 4 --config tiny --out "$W/infer" --device cpu --no-video
echo "SMOKE PIPELINE OK: $W"
