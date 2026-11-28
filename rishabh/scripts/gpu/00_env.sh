#!/usr/bin/env bash
# CUDA host environment for DyaPlex (run from rishabh/). Python 3.10-3.12.
# PersonaPlex pins torch<2.5 / numpy<2.2, so the training env follows its pins.
set -euo pipefail
PYTHON=${PYTHON:-python3.11}
CUDA_INDEX=${CUDA_INDEX:-https://download.pytorch.org/whl/cu121}
$PYTHON -m venv .venv-gpu
. .venv-gpu/bin/activate
pip install --upgrade pip
pip install "torch==2.4.1" --index-url "$CUDA_INDEX"
pip install -r requirements-gpu.txt
python -m dyaplex fetch-upstream                     # PersonaPlex @3428dfd, GestureLSM @f52ac2f -> third_party/
pip install -e third_party/personaplex/moshi --no-deps
python - <<'PY'
import torch, numpy
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count(), "numpy", numpy.__version__)
from dyaplex import speech, codec  # imports PersonaPlex and GestureLSM
print("dyaplex imports ok")
PY
echo "Next: huggingface-cli login (account that accepted nvidia/personaplex-7b-v1), then dyaplex/GPU_RUNBOOK.md"
