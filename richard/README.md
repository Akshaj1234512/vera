# Richard (Jiacheng Liang): speech AI models

My work on the VERA speech-model side. Everything lives in this folder; nothing outside it is changed.

| Folder | What | Status |
| --- | --- | --- |
| [`event-g/`](event-g/) | Event-based Model G prototype: a causal Transformer with one capability token per cue head and role/timing outputs. It's a design alternative to `rishabh/`'s trajectory G. | Synthetic data only; passes 5 smoke checks (shape, disabled head never fires, causality, overfits one batch, 3.4M params) |
| [`personaplex/`](personaplex/) | PersonaPlex 7B speech states for the Seamless streams, and an experiment adding them to Rishabh's G | States captured for all 54 streams (182 min); the G result is inconclusive under synthetic votes. See [`results.md`](personaplex/results.md) |

## Data on Lexi (`/data/jiacheng_liang/`)

| Path | Contents |
| --- | --- |
| `vera-team/personaplex-states/{dev,test}/<agent>/states.npz` | PersonaPlex states: 4096-dim at 12.5 Hz, one file per Seamless stream (54), ready for `gr-prepare --embeddings` |
| `vera-team/gr-personaplex/runs/` | Checkpoints and held-out scores for the PersonaPlex runs |
| `datasets/learning2listen/` | Learning to Listen, 6 tar files, 53 GB (not unpacked) |
| `datasets/beat2/` | BEAT2 English, 16 GB |
| `datasets/README.md` | Dataset sheet: what each set gives G and R, size, license, access |
| `envs/gpu`, `envs/personaplex` | Python 3.12 envs: torch 2.6 cu124, and torch 2.4.1 cu124 + `moshi-personaplex` |

## Running on Lexi

Pick a free GPU first (`nvidia-smi`) and set `CUDA_VISIBLE_DEVICES`. Keep caches and temp files on `/data`, since the home quota is 5 GB:

```sh
export HF_HOME=/data/jiacheng_liang/.cache/huggingface TMPDIR=/data/jiacheng_liang/tmp
```
