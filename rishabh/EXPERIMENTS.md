# Speech AI experiments

## Speech cue replay in MuJoCo

The same published PersonaPlex audio plays under two MuJoCo heads. At **1.80 s** the left head nods and the right head casts a brief gaze cue with nodding disabled. At **9.40 s** the two audio streams overlap. Agent audio ceases at **9.70 s** and the visible yield cue starts when the detector confirms the silence at **9.85 s**. Both simulations had finite state and zero self-contact steps. The gaze cue is a candidate substitute for an observer study. [NVIDIA audio examples](https://research.nvidia.com/labs/adlr/personaplex/#examples)

From `rishabh/`, install `mujoco` in addition to the base requirements. Downloaded tracks and generated outputs live under `data/` and `runs/`. The public audio tools use curl and FFmpeg. Rendering MP4 requires a working graphics context. Omit the video and audio arguments for CSV and JSON logs only.

```sh
python -m speech_ai samples personaplex
python -m speech_ai replay data/personaplex/assistant.wav --stereo --plot \
  --source "Published PersonaPlex interruption audio" --out runs/assistant
python -m speech_ai replay data/personaplex/assistant.wav --stereo --plot \
  --source "Published PersonaPlex interruption audio" --profile nod-busy --out runs/assistant-gaze
python -m speech_ai simulate runs/assistant/timeline.json --out runs/assistant/sim \
  --video runs/assistant/replay.mp4 --audio data/personaplex/assistant.wav
python -m speech_ai simulate runs/assistant-gaze/timeline.json --out runs/assistant-gaze/sim \
  --video runs/assistant-gaze/replay.mp4
ffmpeg -i runs/assistant/replay.mp4 -i runs/assistant-gaze/replay.mp4 \
  -filter_complex '[0:v][1:v]hstack=inputs=2[v]' -map '[v]' -map 0:a:0 \
  -c:v libx264 -c:a copy -shortest runs/vera-speech-cue-demo.mp4
```

The separate `backchannel.wav` example produced four outgoing-audio cessations after overlap. The floor check counted zero as yield cues because the agent had not been speaking independently just before the overlap. The plot shades overlap regions for review.

## Cue model results

Run `python -m speech_ai synthetic-data` and then `python -m speech_ai train data/synthetic/manifest.json`. The fixture encodes a statistical relationship between phrase-ending pitch and acknowledgment. The 72 clips cover 24 synthetic groups. The table uses the 15 held-out clips, a 300 ms tolerance, and thresholds chosen on validation groups.

| Selector | Event F1 | False cues/min | Matched / expected |
| --- | ---: | ---: | ---: |
| Pause rules | 0.667 | 11.67 | 35 / 35 |
| Linear | 0.853 | 1.33 | 29 / 35 |
| GRU | 0.944 | 1.00 | 34 / 35 |

Five Seamless Interaction pairs and three ViCo previews were also processed. Across the Seamless listener videos, tracking coverage was 93–100% and the offline pose heuristic proposed 57 pitch excursions. These are **unreviewed pose proxies**. After choosing the energy threshold on the training groups, the pause gate could reach 4/28 candidate labels in training and 4/21 in validation within 400 ms of the pitch peak. Regular 10 Hz proposals during recent partner activity raised coverage to 13/28 and 12/21. Both reached 0/8 in the held-out pair, and learned held-out F1 was zero. Review synchronized clips and label communicative acknowledgments before comparing model accuracy. [Seamless Interaction](https://huggingface.co/datasets/facebook/seamless-interaction) · [ViCo](https://project.mhzhou.com/vico/) · [Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)

```sh
python -m pip install mediapipe opencv-contrib-python
python -m speech_ai samples seamless
python -m speech_ai pose data/seamless/annotation-template.json --out runs/seamless-pose --export-proxy
python -m speech_ai audit runs/seamless-pose/pose-proxy-manifest.json --vad-db -58 --tolerance 0.4
python -m speech_ai audit runs/seamless-pose/pose-proxy-manifest.json --vad-db -58 --tolerance 0.4 \
  --candidates cadence --out runs/seamless-pose/cadence-coverage.json
python -m speech_ai train runs/seamless-pose/pose-proxy-manifest.json --out runs/seamless-cadence \
  --vad-db -58 --tolerance 0.4 --candidates cadence --epochs 12
```

`samples seamless` fetches about 485 MB before PCM conversion. For ViCo, use `samples vico` and pass `--side right` to `pose` for the listener half of each preview. To train with reviewed labels, copy the annotation template, set `ack_s` to acknowledgment onset times in seconds, and group any overlapping participants or sessions under the same identifier. An empty list means the clip was reviewed and contains no event.

## PersonaPlex weights and causal features

The bridge uses the official offline runner in a separate Python 3.12 environment. Install the pinned NVIDIA implementation there and accept the model license on Hugging Face. The full checkpoint needs a host with more memory than the 16 GB development Mac. The simulation replay uses NVIDIA’s released output. [PersonaPlex implementation](https://github.com/NVIDIA/personaplex) · [Weights](https://huggingface.co/nvidia/personaplex-7b-v1)

```sh
python3.12 -m venv .venv/personaplex
.venv/personaplex/bin/python -m pip install \
  'moshi-personaplex @ git+https://github.com/NVIDIA/personaplex.git@16419a2ef7b98611f58c9df07477d00e5ad61d5f#subdirectory=moshi'
python -m speech_ai personaplex partner.wav --python .venv/personaplex/bin/python \
  --capture-states --out runs/personaplex-local
```

The run stores aligned WAVs, text tokens, model settings, and optional final speech states. Each state is stamped on the input frame clock after model warm-up and prompting. Set the manifest’s `embeddings` field to the resulting NPZ and add `--embeddings` to training to compare the same models with and without the extra feature channel. A tiny randomly initialized upstream LM tests the state hook, and the repository tests exercise the alignment and checkpoint round trip. The full 7B run and human observer evaluation are still pending. [DyaPlex feature extraction](https://arxiv.org/html/2606.03874v1#S4.SS1)
