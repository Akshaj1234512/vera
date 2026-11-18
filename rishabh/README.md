# VERA Speech AI: G/R speech-motion

When the robot lacks a facial joint, can a model substitute another gesture while preserving what a human observer understands? Two models answer this (spec: `Speech.pdf`, Akshaj Gupta):

- **G (expression generator)** predicts the next 500 ms of robot motion, M_r = G(M_h, x_t, b_m, q_t). Its inputs are the last 500 ms of DyaPlex facial motion M_h (10 × 8), conversational features x_t (causal audio encoder plus PersonaPlex), the capability descriptor b_m, and the measured robot state q_t. G is the model that is deployed.
- **R (interpretation model)** predicts P(z | M_r, b_m, x) from observer votes. R is used to train and evaluate G. It never drives the robot.

Run everything from this folder.

```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

## Workflow

| Step | Command | Output |
| --- | --- | --- |
| Capture PersonaPlex states | `personaplex partner.wav --python <env>/bin/python --capture-states --out runs/pp` | `states.npz` (`time_s`, `features`) |
| Fit the DyaPlex map (decoded features only) | `gr-fit-map pairs.json --out calibration.npz` | ridge map, held-out MAE/R² |
| Align inputs on the 20 Hz clock | `gr-prepare partner.wav --agent robot.wav --teacher dyaplex.npz [--teacher-map calibration.npz] [--embeddings states.npz] --out stream.npz` | `stream.npz` + `stream.json` coverage report |
| Run the chunk loop on a robot | `gr-live …` (arriving inputs) or `gr-replay stream.npz …` (prepared stream) | `realized-motion.npz`, `interpretations.json` |
| Package 500 ms chunks | `gr-trial stream.npz realized-motion.npz --all --out trials/s01` | trial NPZs + `trials.json` |
| Build the manifest | `gr-manifest trials/*/trials.json --intents intents.csv --out manifest.json` | unrated manifest |
| Observer study | `gr-rating-packet manifest.json --out packet` → raters fill the sheet → `gr-votes manifest.json --key packet/key.json sheets/*.csv --out rated.json` | vote counts per trial |
| Train | `gr-train rated.json --out runs/gr-real` (`--imitation-only` to bootstrap from teleop, `--residual` for M_direct + ΔM) | `g.pt`, `r.pt`, `evaluation.json`, `g-imitation.pt` |
| Evaluate substitution | `gr-evaluate rated.json --r runs/gr-real/r.pt --sessions runs/... --out report.json` | per-condition intended-vote share and unclear rate |

Every command is `python -m speech_ai <command>`. Use `--policy projected` on `gr-replay` or `gr-live` to run the projected-DyaPlex baseline through the same recording path, and `--profile fixed-neck-pitch` or `--calibration calib.json` for a restricted embodiment.

## Synthetic check and demo video

```sh
P=.venv/bin/python
$P -m speech_ai samples                                    # NVIDIA's published PersonaPlex conversations
$P -m speech_ai gr-fixture --out data/gr-v2/fixtures       # SYNTHETIC trials and votes
$P -m speech_ai gr-train data/gr-v2/fixtures/manifest.json --out runs/gr-v2/train
$P -m speech_ai gr-teacher-script data/personaplex/assistant.wav --stereo --out data/gr-v2/demo/teacher.npz
$P -m speech_ai gr-prepare data/personaplex/assistant.wav --stereo --teacher data/gr-v2/demo/teacher.npz \
  --out data/gr-v2/demo/stream.npz
for run in g:full g:fixed-neck-pitch projected:fixed-neck-pitch; do
  policy=${run%%:*}; profile=${run#*:}
  $P -m speech_ai gr-replay data/gr-v2/demo/stream.npz --policy $policy --profile $profile \
    --g runs/gr-v2/train/g.pt --r runs/gr-v2/train/r.pt --out runs/gr-v2/demo-$policy-$profile \
    --video runs/gr-v2/demo-$policy-$profile/replay.mp4 --audio data/personaplex/assistant.wav
done
ffmpeg -i runs/gr-v2/demo-projected-fixed-neck-pitch/replay.mp4 -i runs/gr-v2/demo-g-fixed-neck-pitch/replay.mp4 \
  -i runs/gr-v2/demo-g-full/replay.mp4 -filter_complex '[0:v][1:v][2:v]hstack=inputs=3[v]' \
  -map '[v]' -map 0:a:0 -c:v libx264 -c:a copy -shortest runs/gr-v2/vera-gr-speech-demo.mp4
```

`runs/gr-v2/vera-gr-speech-demo.mp4` shows three panels side by side with the conversation audio: the projected baseline with neck pitch locked, G with neck pitch locked, and G with the full face. The teacher motion is **scripted from the audio and is not DyaPlex**. G and R were trained on synthetic votes. Treat the video as an integration demo, not evidence.

[Design-to-code mapping, data contract, protocols, results and open gaps](EXPERIMENTS.md)
