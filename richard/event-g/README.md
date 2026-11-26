# Event-based Model G (prototype)

Model G for VERA: a small causal Transformer that turns conversation features and
human facial motion into cue events a specific robot face can perform.

## Inputs and outputs

| Name | Shape | Meaning |
| --- | --- | --- |
| `x` | `[B, T, 512]` | speech features per frame (stand-in for PersonaPlex), 12.5 Hz |
| `m_h` | `[B, T, 8]` | human facial motion channels (brows, jaw, neck, ...) |
| `b_m` | `[B, 6, 5]` | per cue head: enabled, min, max, max speed, latency |
| output | `[B, T, 6, 11]` | per frame, per cue head: see below |

Cue heads, in order: gaze, nod, jaw, lids, brow_left, brow_right.

The 11 output values per head:

| Index | Meaning |
| --- | --- |
| 0 | enable logit (forced to about -10000 when `b_m` says the head is disabled) |
| 1-4 | role logits: attending, acknowledging, articulating, yielding |
| 5-8 | cue parameters |
| 9 | onset (s) |
| 10 | duration (s) |

The model is causal: the output at frame `t` only depends on inputs at frames `<= t`.

## Layout

```
vera/config.py      all dimensions
vera/model_g.py     the model
vera/losses.py      event loss + smoothness loss
scripts/smoke_test.py   5 end-to-end checks on fake data
```

## Run

```bash
cd richard/event-g
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python scripts/smoke_test.py
```

The checks run on CPU in under a minute. To use a GPU on this shared machine, pick one
free card first (`nvidia-smi`), e.g. `CUDA_VISIBLE_DEVICES=3`.

## Relation to `rishabh/`

`rishabh/speech_ai/gr_model.py` is the main G (joint trajectories, per Speech.pdf). This is a separate
prototype exploring two ideas: one capability token per cue head that frames attend to, and an event
head that predicts role and timing per cue. Real PersonaPlex states are 4096-dim at 12.5 Hz, so
`d_speech` would need to be 4096 (or projected) to use them.

## Not yet included

Real PersonaPlex features, real datasets, Model R (the interpretation model), GPU training.
