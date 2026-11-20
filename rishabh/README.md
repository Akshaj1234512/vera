# VERA Speech AI: G/R speech-motion

When the robot lacks a facial joint, can a model substitute another gesture while preserving what a human observer understands? Two models answer this (spec: `Speech.pdf`, Akshaj Gupta):

- **G (expression generator)** predicts the next 500 ms of robot motion, M_r = G(M_h, x_t, b_m, q_t).
  - Inputs: 500 ms of human facial motion M_h (head, gaze and 16 facial action units), conversational features x_t (both speakers' audio, plus PersonaPlex states when available), the capability descriptor b_m, and the measured robot state q_t.
  - Output: every joint of Samhita's rig (`../samhita_sim`), 27 channels covering the neck, eyes, brows, lids, jaw, lips, cheeks, nose and chin.
  - G is the model that is deployed.
- **R (interpretation model)** predicts P(meaning | robot motion, b_m, x) from observer votes. It is used to train and evaluate G and never drives the robot.

**Current model:** [`runs/gr-real/selected/`](runs/gr-real/selected/) (`g.pt`, `r.pt`, `evaluation.json`, `heldout-test-split.json`).

**Training data:**
- **Motion and audio:** real, from 46 Seamless Interaction participants (155 min, the dataset DyaPlex is trained on).
- **Votes:** synthetic, from a rule-based observer. No human ratings exist yet.

**Demo:** [`runs/gr-real/demo/demo.mp4`](runs/gr-real/demo/demo.mp4). It shows a held-out human next to the projected-DyaPlex retarget and G on the rig, with the full face and with neck pitch locked.

Details, results and caveats are in [EXPERIMENTS.md](EXPERIMENTS.md).

```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

| Step | Command |
| --- | --- |
| Real human motion + audio → teacher streams | `gr-seamless --split dev --count 30 --out data/seamless-motion/naturalistic-dev` |
| Trials (votes: synthetic rule observer) | `gr-dataset data/seamless-motion/naturalistic-dev/streams.json --out data/gr-real/dev` |
| Train R, then refine G through frozen R | `gr-train data/gr-real/dev/manifest.json --meaning-weight 3 --refine-epochs 20 --out runs/gr-real/selected` |
| Score frozen models on held-out data | `gr-score data/gr-real/test/manifest.json --g … --r … --out …` |
| Drive the rig from a stream | `gr-replay stream.npz --g g.pt [--policy projected] --profile fixed-neck-pitch --video out.mp4 --out runs/x` |
| Demo grid | `gr-demo stream.npz --partner-wav … --agent-wav … --source-video … --keypoints … --start-s 66 --g … --out …` |
| Live inputs | `gr-live --listen HOST:PORT --wait-for-producers N …` (producers: DyaPlex or teacher, `personaplex --publish`, `AudioFrontend`; dry run with `gr-publish`) |
| PersonaPlex states (GPU host) | `personaplex partner.wav --python <env>/bin/python --capture-states --out runs/pp` |
| DyaPlex FLAME → human channels | `gr-fit-map pairs.json --out calibration.npz`, then `gr-prepare --teacher-map` |
| Observer study | `gr-trial --all` → `gr-manifest` → `gr-rating-packet` → raters → `gr-votes` → `gr-evaluate` |

Run every command as `python -m speech_ai <command>` from this folder. Profiles: `full`, `fixed-neck-pitch`, `fixed-neck`, `fixed-eyes`, or a custom `--calibration` JSON (see `examples/`).
