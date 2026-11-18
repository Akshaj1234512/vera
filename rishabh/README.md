# VERA Speech AI

Turns paired speech audio into timestamped nod, gaze, jaw, and yield cues. Run the same audio with different actuator capabilities and replay the result on the existing VERA simulator.

From this folder:

```sh
python -m pip install -r requirements.txt
python -m speech_ai demo
```

Open `runs/demo/linked-eyes/timeline.png` for the first replay. The workbench also compares rules, linear, and GRU cue selectors on held-out speaker groups.

[Simulation replay, evaluation results, and run commands](EXPERIMENTS.md)
