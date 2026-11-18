# Speech AI experiments

## G/R facial motion

**G** is a two-layer causal transformer. It takes ten 20 Hz frames of human motion, the latest PersonaPlex features, the robot’s capability descriptor, and its current joint position and velocity. It predicts the next ten frames of robot motion across eight channels. The capability descriptor masks locked joints and bounds position and slew. The optional `--residual` mode learns a correction to the projected DyaPlex trajectory.

**R** takes realized robot motion, capabilities, and the same conversational features. It learns a probability distribution over interpretations from observer vote counts. Training then freezes R and optimizes G for intended-interpretation likelihood and temporal smoothness. When teleoperation demonstrations are available, G also learns to reproduce them. The observer receives the conversational context and motion without the intended label.

### Run with synthetic fixtures

From `rishabh/`, run the README commands. Install `mujoco` for replay. The two commands below apply G to the same input stream with two capability settings. G uses measured simulator state at each chunk boundary. R scores the positions the simulator actually reached, which are also saved for annotation.

```sh
for profile in full fixed-neck-pitch; do
  python -m speech_ai gr-replay data/gr-synthetic/replay.npz \
    --g runs/gr-training/g.pt --r runs/gr-training/r.pt \
    --profile "$profile" --out "runs/gr-$profile" \
    --video "runs/gr-$profile/replay.mp4"
done
```

The synthetic run checks the pipeline using programmatically generated motions and ratings. The evaluation JSON separates agreement with held-out votes on recorded clips from R’s score on newly generated trajectories. R’s score on G’s output is a proxy for the objective. Human judgments are needed to measure whether the substitution actually preserves meaning.

On the held-out synthetic trials, refinement raised R’s mean probability of the intended interpretation from 0.886 to 0.934. The same replay yielded the same predicted interpretations across the two capability settings. The locked head kept neck pitch at zero and used gaze and brow motion. Both replays finished without simulated self-contact.

### Prepare real trials

Use synchronized DyaPlex motion, PersonaPlex state exports, and realized robot motion on the same recording clock. The eight channels in both motion arrays are `neck_yaw`, `neck_pitch`, `jaw`, `eye_yaw`, `eyelid_l`, `eyelid_r`, `brow_l`, and `brow_r`. The simulator maps `eye_yaw` to both eyes.

1. Run `gr-prepare partner.wav --teacher dyaplex.npz --embeddings states.npz --out stream.npz` through `python -m speech_ai`. The teacher NPZ accepts `time_s`, `channels`, and normalized `human` with shape `[frames, 8]`. For a decoded DyaPlex feature export with `time_s` and `features`, supply `--teacher-map calibration.npz` with fitted `weights` of shape `[features, 8]` and `bias` of shape `[8]`. PersonaPlex states come from `personaplex --capture-states`. Alignment uses only features observed by the start of the generated chunk.
2. Record robot motion and have observers rate what each chunk communicates in context. Include recognizable, ambiguous, and wrong-for-context examples. Use `python -m speech_ai gr-trial stream.npz realized-motion.npz --start-s 1.0 --out trial.npz` to pair a recorded chunk with its inputs. Recordings use the 20 Hz `time_s` clock, `channels`, `robot_rad [frames, 8]`, `chunk_start_s`, and `initial_state_rad [chunks, 2, 8]` for position and velocity. The first half-second of replay gathers history.
3. Create a manifest with an ordered `labels` list and `trials` list. Each row has `id`, `group` for a complete interaction or participant split, `origin`, `file` relative to the manifest, and `profile`. A row can include its intended label in `intent`, observed vote counts such as `"votes": {"neutral_ack": 7, "unclear": 2}`, and `"demonstration": true` for an approved teleoperation trajectory. For calibrated hardware, supply a `capabilities` object keyed by channel with `[enabled, min_rad, max_rad, max_rad_per_s]` values. The three synthetic label names are examples for the initial study.
4. Run `gr-train` on the reviewed manifest. If only teleoperation examples are ready, use `--imitation-only` to obtain `g-imitation.pt` and replay it without an R checkpoint to collect ratings. The combined run writes `g.pt`, `r.pt`, and `evaluation.json`. It also saves `g-imitation.pt` when it pretrains on demonstrations.

The commands above train G and R with synthetic inputs. The real DyaPlex channel calibration, paired recordings, and observer votes still need to be collected for evaluation.

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
