# PersonaPlex states as speech input to G

Experiment on top of `vera-gr-model` (Rishabh's G/R on Seamless). All code is in this folder and uses Rishabh's code unchanged:

- `pp_extract.py`: PersonaPlex state capture.
- `gr_pp_retrain.sh`: runs A and B end to end.
- `gr_pp.py`: the G-only model and same-judge scoring.

## Data

PersonaPlex 7B (`nvidia/personaplex-7b-v1`) states were captured with `python -m speech_ai personaplex <partner audio-16k.wav> --capture-states`. There is one run per Seamless stream: PersonaPlex hears the partner, the person the agent is listening to.
- **Size:** 54 streams (46 naturalistic/dev, 8 naturalistic/test), 181.9 min.
- **Shape:** 4096-dim, one frame every 0.08 s, full coverage, all finite.
- **Location on Lexi:** `/data/jiacheng_liang/vera-team/personaplex-states/{dev,test}/<agent>/states.npz`.

Streams were rebuilt with `gr-prepare --embeddings`, giving a context of 8 audio + 4096 states + 1 availability. Dataset seeds and training settings are the same as `runs/gr-real/selected`: meaning weight 3, 20 refine epochs, seed 17.

## Runs (held-out naturalistic/test, macro P(intent) over 12 profile × intent cells)

| Run | G context | R context | R majority agreement | G macro, scored by its own R | G macro, scored by Rishabh's R |
| --- | --- | --- | ---: | ---: | ---: |
| Rishabh's selected (baseline) | 8 audio | 8 audio | 77.2% | 0.461 | 0.461 |
| A: raw states in G and R | 4105 | 4105 | 58.7% | 0.388 | n/a (context mismatch) |
| B: `gr_pp`, G only, PCA 64 | 8 + 64 + 1 | 8 audio | 76.3% | 0.455 (baseline G: 0.416) | 0.431 |

Neck locked, enthusiastic acknowledgment, P(intent):

| Judge | Baseline G | Run B G |
| --- | ---: | ---: |
| Rishabh's R | 0.51–0.55 | 0.45 |
| Run B's R | 0.43–0.47 | 0.47–0.48 |

## Reading

- **Run A:** R overfits to the 4096 speech dims. The synthetic votes depend only on motion, so speech carries no real signal for R. Even the "hold still" baseline reads as an enthusiastic acknowledgment with probability 0.27, against 0.07 for audio-only R.
- **Run B:** keeping R on audio fixes R. G with speech is a wash, though. It loses under Rishabh's R and wins under its own R. The gap between the two judges is as large as the gap between the two Gs.
- **Validation:** run B's validation macro is 0.419, against 0.428 for the baseline. This is a single seed.
- **Limit of these labels:** the rule-based observer scores motion only. Speech can help here only by forecasting the human's next 500 ms of motion; it gets no credit for reacting appropriately to what was said. Human ratings are needed to test that.

## Reproduce (on Lexi)

```sh
# 1. PersonaPlex states for every stream (about an hour on 4 GPUs)
python richard/personaplex/pp_extract.py /data/rthakur/vera/data/seamless-motion/naturalistic-dev/streams.json \
  /data/jiacheng_liang/vera-team/personaplex-states --gpus 3,4,5,6     # and the same for naturalistic-test
# 2. Rebuild the streams, then run A and run B (clones vera-gr-model; trains on CPU)
CUDA_VISIBLE_DEVICES=7 bash richard/personaplex/gr_pp_retrain.sh
```

As a check, `gr_pp score` with Rishabh's G and R reproduces his published held-out macro (0.461) exactly.

Results on Lexi: `/data/jiacheng_liang/vera-team/gr-personaplex/runs/`.
