# DyaPlex reproduction (`dyaplex/`)

A reproduction of **DyaPlex: Full-Duplex Speech-Motion Model for Dyadic Interaction** ([arXiv 2606.03874v1](https://arxiv.org/abs/2606.03874)), driving Samhita's expressive face rig (`../samhita_sim`) through the existing `speech_ai` replay path.
NVIDIA has not released DyaPlex code or weights. Everything here is built from the paper plus pinned upstream code.

| Module | Paper | What it is |
|---|---|---|
| `config.py` | all | `REFERENCE` (paper scale) and `TINY` (smoke test) configs; every value tagged paper / inherited / assumption |
| `data.py` | §5, App. C, D.3 | Seamless pairs → validity masks, 180° x flip, Savitzky–Golay, 30 → 25 fps, SMPL-X order, 6D, part features, normalization, VAD |
| `codec.py` | §3.2, D.3 | Four causal part RVQ-VAEs built from GestureLSM's own `Encoder`/`Decoder`/`ResidualVQ` (`causal=True`, `down_t=1`), token bands |
| `speech.py` | §3.1, §4.1, D.1 | PersonaPlex `LMModel`/`LMGen` with hooks on all 32 blocks; hybrid prompt layout; teacher-forced and streaming states |
| `layout.py` | §4.2–4.3, Eq. 1–2 | 46-token dyadic interleaving, q_pos, supervised targets, band masks, VA labels |
| `tower.py` | §4.2–4.3, Eq. 3–4 | Motion tower (self-attention → time-aligned cross-attention → SwiGLU), band-masked CE + β·VA loss, Table 2 ablation switches |
| `generate.py` | §4.4, Eq. 5, D.2 | Training loop (AdamW 3e-4, betas 0.9/0.95, wd 0.1, clip 1.0) and the streaming sampler (agent-only / both-speaker, KV cache) |
| `rig.py` | — | Decoded agent motion → 21 human channels → `gr-prepare` → `gr-replay --policy projected` on the 27-actuator rig |
| `e2e.py` | review §7 | The smallest end-to-end test and its pass criteria |

Assumptions, stand-ins and paper inconsistencies are listed in [DEVIATIONS.md](DEVIATIONS.md).

## Commands (run from `rishabh/`)

```sh
.venv/bin/pip install einops tqdm sphn sentencepiece safetensors   # PersonaPlex imports
.venv/bin/python -m dyaplex fetch-upstream                            # PersonaPlex @3428dfd, GestureLSM @f52ac2f -> third_party/
.venv/bin/python -m unittest tests.test_dyaplex -v                   # fast checks
.venv/bin/python -m dyaplex e2e --out runs/dyaplex-e2e               # smallest end-to-end test (~CPU/MPS)
.venv/bin/python -m dyaplex e2e --out runs/dyaplex-e2e --resume --tower-iterations 800   # reuse codecs, continue the tower
.venv/bin/python -m dyaplex config                                    # reference vs tiny configuration
```

The e2e test needs the Seamless pairs already under `data/seamless-motion/naturalistic-dev`, which `gr-seamless` downloads. It fetches the missing SMPL-H arrays itself (about 2 MB per participant).
