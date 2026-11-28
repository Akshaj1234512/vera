# DyaPlex on CUDA: runbook

Every stage below has been run end to end at toy scale on CPU (`bash scripts/dyaplex_smoke_pipeline.sh`).
None has run on a GPU or at paper scale. Commands run from `rishabh/`; `$D` is the Seamless download, `$P` the prepared-data directory.
Paper references are to arXiv 2606.03874v1. Assumptions and stand-ins are in [DEVIATIONS.md](DEVIATIONS.md).

## 0. Accounts, licenses, hardware

| Need | For | Where |
|---|---|---|
| Hugging Face login + accepted license for `nvidia/personaplex-7b-v1` (rev `fdaf409`) | speech tower, Mimi, tokenizer | huggingface.co/nvidia/personaplex-7b-v1 |
| Seamless Interaction (CC-BY-NC 4.0, non-commercial) | all data | github.com/facebookresearch/seamless_interaction |
| SMPL-X registration → `SMPLX_NEUTRAL.npz` | evaluation (FK) | smpl-x.is.tue.mpg.de |
| FLAME registration + SPECTRE weights | face variant only | flame.is.tue.mpg.de, github.com/filby89/spectre |
| GPUs | codecs: 4 × 1 GPU; tower: paper used 64 × H100 (80 GB) | — |
| Disk | download ~27 TB (mostly video); prepared features ~460 GB; streams ~5 GB; tokens ~14 GB | — |

## 1. Environment

```sh
bash scripts/gpu/00_env.sh          # venv with torch 2.4.1+cu121, PersonaPlex @3428dfd (editable), GestureLSM @f52ac2f
huggingface-cli login               # account that accepted the PersonaPlex license
```

PersonaPlex pins `torch<2.5` and `numpy<2.2`; this code runs with those versions.

## 2. Data

```sh
# Official downloader (whole corpus; see its README for batch-by-batch download)
python -m seamless_interaction.download ...            # or scripts/download_s3.py / download_hf.py
# -> $D/<label>/<split>/<batch>/<archive>/<file_id>.{npz,json,wav,mp4}
```

Face variant only: SPECTRE on every video, sharded over GPUs. Run it inside SPECTRE's own environment, from a SPECTRE checkout at `0b339a8`:

```sh
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python <repo>/rishabh/dyaplex/face_extract.py --videos $D --out $D/../spectre --shard $i/8 &
done; wait
```

Check one clip against SPECTRE's `demo.py` before the full run. `face_extract.py` has never been executed.

Prepare: filters, 25 fps features, normalization and splits. This step is CPU-bound and parallel.

```sh
python -m dyaplex prepare --root $D --out $P --workers 64 --video-check local [--face-dir $D/../spectre]
cat $P/summary.json      # compare with the paper: 57,947 train pairs / 3,435 h; 330 test pairs
```

## 3. Causal part codecs (one GPU each, GestureLSM recipe, 300k iterations)

```sh
k=0; for part in upper hands lower_trans face; do
  CUDA_VISIBLE_DEVICES=$k python -m dyaplex train-codec --prepared $P --part $part --out runs/codecs & k=$((k+1))
done; wait                                   # resumable; keeps the lowest held-out (val) reconstruction
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python -m dyaplex tokenize --prepared $P --codecs runs/codecs --out $P/tokens --shard $i/4 & done; wait
```

## 4. PersonaPlex input streams (Mimi codes, inner monologue, voice prompts)

```sh
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python -m dyaplex build-streams --prepared $P --mode real --shard $i/8 &
done; wait                                   # every shard writes the same prompt.json (text-prompt token ids)
```

## 5. Motion tower (online PersonaPlex states, DDP)

Caching every layer's states for the corpus would take about 40 PB, so each rank runs a frozen bf16 copy of PersonaPlex. Per micro-batch it computes states for one context chunk and cuts training windows from it.

```sh
# 8 nodes x 8 H100: effective batch 512 = 64 ranks x micro 8 x accum 1
torchrun --nnodes 8 --nproc_per_node 8 --rdzv-backend c10d --rdzv-endpoint $MASTER:29500 \
  -m dyaplex.train_tower --prepared $P --tokens $P/tokens --out runs/tower-body --variant body
torchrun ... -m dyaplex.train_tower ... --out runs/tower-face --variant face          # face demos / rig
for ab in no_self_attn no_partner no_cross_attn no_cross_rope; do                     # Table 2 ablations
  torchrun ... -m dyaplex.train_tower ... --out runs/tower-$ab --variant body --ablation $ab
done
```

- **Fewer GPUs:** keep `--batch-size 512`. Gradient accumulation is derived automatically, e.g. 8 GPUs × micro 8 gives accum 8.
- **Memory:** the tower is 0.67B parameters (~11 GB with AdamW states) plus the 7B speech tower in bf16 (~15 GB). If micro 8 × 4,096 tokens runs out of memory, add `--grad-checkpoint` or use `--micro-batch 4`.
- **Resume:** rerunning the same command resumes from `last.pt`. Checkpoints go to `tower_XXXXXX.pt` every 500 steps; logs to `log.jsonl` (train CE / VA / accuracy, validation every 500 steps).
- **Compute estimate (±3×):** about 3e20 FLOPs including the online speech tower. That's about 200–250 H100-hours, roughly 4 h on 64 GPUs at 40% MFU. The paper does not report wall time.

## 6. Evaluation (Table 2) and baselines

```sh
python -m dyaplex evaluate --prepared $P --tokens $P/tokens --codecs runs/codecs \
  --tower runs/tower-body/tower_030000.pt --smplx SMPLX_NEUTRAL.npz --out runs/eval-body.json          # top-k 200 (§4.4)
python -m dyaplex evaluate ... --top-k 0 --out runs/eval-body-notopk.json                               # D.2: no top-k (X-1)
python -m dyaplex export-baselines --prepared $P --out runs/baseline-data                              # see BASELINES.md
python -m dyaplex evaluate --prepared $P --tokens $P/tokens --predictions runs/a2p-preds --smplx SMPLX_NEUTRAL.npz --out runs/eval-a2p.json
```

The paper's 330 test pair ids are unpublished. `test330` is a deterministic stand-in (sha1 order of the audio-checked test pairs), so numbers are not directly comparable to Table 2. BeatAlign follows ViBES's definition and gives about 0.6 on ground truth; the paper's GT is 0.049. See "Open questions" in DEVIATIONS.md.

## 7. Runtime and the rig

```sh
python -m dyaplex bench --device cuda --out runs/bench-h100.json            # motion tower at 4096 / 1024 tokens
python -m dyaplex infer --prepared $P --tokens $P/tokens --codecs runs/codecs --tower runs/tower-face/tower_030000.pt \
  --pair <pair> --agent <file_id> --start-s 30 --seconds 20 --out runs/demo   # needs mujoco + speech_ai for the rig video
```

## Order and rough wall time (estimates)

| Stage | Hardware | Estimate |
|---|---|---|
| Download | network | days (~27 TB) |
| SPECTRE (face only) | 8 GPUs | ~1,000 GPU-h total, ~5 days |
| prepare | 64 CPU cores | ~0.5–1 day |
| 4 codecs | 4 GPUs in parallel | ~0.5–1 day (300k iterations each) |
| tokenize + build-streams | 8 GPUs | ~0.5 day |
| tower (each of 6 runs) | 64 H100 | ~4 h each |
| evaluate | 1 GPU | ~1 h per model |
