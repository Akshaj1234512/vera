# VERA Speech AI

G generates 500 ms of robot facial motion from DyaPlex motion, PersonaPlex features, and the robot’s capabilities and current state. R predicts what observers interpret from the resulting motion. R also guides G during training.

Run the synthetic verification from this folder.

```sh
python -m pip install -r requirements.txt
python -m speech_ai gr-fixture
python -m speech_ai gr-train data/gr-synthetic/manifest.json
```

[G/R inputs, MuJoCo replay, and experiment results](EXPERIMENTS.md)
