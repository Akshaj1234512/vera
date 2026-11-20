# DyaPlex: smallest end-to-end test

**Result:** all 26 criteria pass (`report.json`, `passed: true`) on the face-enabled tiny configuration, branch `dyaplex-rishabh`, 2026-11-20.
This is a plumbing, causality and integration test. **It says nothing about motion quality**: the codecs trained for 600 iterations on 8 pairs, and the tower was overfit on 2 pairs.

## Commands

```sh
cd rishabh
.venv/bin/pip install einops tqdm sphn sentencepiece safetensors
.venv/bin/python -m dyaplex fetch-upstream
.venv/bin/python -m unittest discover -s tests          # 41 tests, OK
.venv/bin/python -m dyaplex e2e --out runs/dyaplex-e2e  # pass 1 (body-only objective at the time)
.venv/bin/python -m dyaplex e2e --out runs/dyaplex-e2e --resume --tower-iterations 800   # pass 2, pass 3
```

## Passes

| Pass | Tower objective | Tower iters (total) | Teacher-forced acc | Jaw-speech r (rig) | Outcome |
|---|---|---|---|---|---|
| 1 (`report-pass1.json`) | body-only, 18 codes | 800 | 0.840 | 0.033 (demo agent had 58% failed face tracking) | overfit criterion **failed** |
| 2 (`report-pass2-bodyonly.json`) | body-only, 18 codes | 1,300 | 0.971 | 0.037 (reference GT 0.441) | all passed, but jaw untrained (see below) |
| 3 (`report.json`) | face-enabled, 22 codes | 1,600 | 0.974 | **0.310** (reference GT 0.441) | all passed |

Pass 2 exposed a test-design error. Under the paper's body-only objective the 4 face codes are never supervised, so the rig's face came from an untrained head. The sim test now uses the face-enabled variant (the paper's demo model). The reference benchmark config stays body-only.

## Criteria (pass 3)

| Group | Check | Result |
|---|---|---|
| data | pairing (10 real pairs, naturalistic/dev) | 10/10 |
| data | clip-level aspect/integrity filter (remote ffprobe) | 20/20 portrait or square |
| data | 30→25 fps resample sync | max 0.0067 s (< 1 frame) |
| data | 6D round trip on real poses / on matrices | 1.2e-7 / 5.6e-16 |
| data | token cache `(T, 22)` uint16, in band | all 20 participants |
| codec | held-out reconstruction falls (upper/hands/lower/face) | 0.42→0.17, 0.33→0.18, 0.72→0.23, 0.17→0.08 |
| codec | encoder: perturb future → past codes and latents | identical (diff 0.0) |
| codec | decoder causal | prefix decode identical |
| codec | batched == incremental encoding | identical, all parts |
| speech | teacher-forced `forward_train` == streaming `LMGen.step`, 32 layers, 399 frames | rel 3.7e-7 |
| speech | taps do not change PersonaPlex outputs (58 sampled frames) | identical |
| tower | 46 tokens/frame, Eq. 2 boundaries | ok (22-frame window = 1,012 tokens) |
| tower | future speech + motion changed → past logits | bit-identical, untrained and trained |
| tower | overfit teacher-forced accuracy > 0.95 | 0.974 |
| streaming | KV cache == recompute (20 frames) | same samples, logits ≤ 1.3e-5 |
| streaming | future inputs changed → first 10 frames unchanged | identical |
| streaming | agent-only (partner copied) and both-speaker modes in band | ok |
| sim | 27 rig commands finite and in actuator range (400 × 50 ms) | ok |
| sim | rig `jaw_open` vs agent speech envelope | r = 0.31 (stand-in GT jaw r = 0.44) |

Video: `rig/rig.mp4` (20 s, agent P0217→ partner P0216, pair S0152, 30–50 s).

## Timings (tiny sizes; CPU contended by unrelated jobs during the run)

Per 80 ms frame: speech tower (tiny, streaming) 63 ms; motion tower (tiny, one thread, KV cache) 520 ms; four decoders 1.4 ms; MuJoCo replay and render 156 ms.
These are **not** comparable to the paper's A6000 numbers (30 / 173 / 0.8 ms). Sizes differ, and other processes were saturating the CPU.

## What is verified vs what still needs real weights or training

See `scope` in `report.json` and `../../dyaplex/DEVIATIONS.md`. Still open:
- real PersonaPlex-7B states (gated checkpoint, CUDA GPU)
- Mimi tokens and the tokenized inner monologue
- SPECTRE/FLAME faces and the FLAME→AU calibration (`gr-fit-map`)
- SMPL-X FK and the paper's metrics
- paper-scale codec and tower training
