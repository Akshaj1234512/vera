# G/R speech-motion: design, data contract and evidence

Specification: `Speech.pdf` (5 handwritten pages). Every number below is labeled **synthetic** or **simulator replay**. No human-observer, real-DyaPlex, real-PersonaPlex-state or hardware result exists yet. The [open gaps](#open-gaps-exact-inputs-and-commands) section lists what is still needed.

## PDF → implementation map

| Page | Requirement | Implementation | Verified by | Status |
| --- | --- | --- | --- | --- |
| 1 | Substitute a gesture when a joint is missing while preserving observer meaning | Capability-conditioned G; R trained on votes; `gr-evaluate` intended-vote share per condition | `trajectory_test_by_profile` in `evaluation.json` | Synthetic only. Human test pending. |
| 1 | Two models: G deployed, R helps train and evaluate | `gr_model.Generator`, `gr_model.Observer`. `gr-replay` and `gr-live` run with or without R; R only scores. | `MujocoTests`, `SessionTests` | Implemented |
| 1 | M_h ∈ ℝ^{10×8} from 500 ms of DyaPlex output | `STEPS=10` at 20 Hz × 8 `CHANNELS`, plus a validity column for missing frames | `test_g_shapes_and_capability_bounds…` | Implemented. Real DyaPlex export pending. |
| 1–2 | b_m ∈ ℝ^K: active DoF, ranges, calibration | `capability()`: per channel [enabled, min rad, max rad, max rad/s] (K = 32). Profiles plus `--calibration` JSON. | `test_motion_obeys_enabled_joints…` | Implemented. Hardware calibration pending. |
| 2 | x_t ∈ ℝ^d from encoders + PersonaPlex | 8 causal audio-encoder features (`audio.FEATURE_NAMES`) + PersonaPlex state + availability bit | `test_stream_alignment_is_causal…` | Implemented. Real 7B states pending. |
| 2 | M_r = G_θ(M_h, x_t, b_m, q_t), q_t = current robot state | `Session.plan` reads measured [position, velocity] at each chunk boundary | `test_measured_state_feeds_back_into_each_chunk` | Implemented (sim and kinematic adapters) |
| 2 | Small causal transformer | 2-layer, width 48, causal mask over the 10 history frames | — | Implemented |
| 2 | Option: M_r = M_direct + ΔM_θ | `--residual`: projected DyaPlex + 0.5 × learned Δ, then capability constraint | `test_residual_is_projected_dyaplex_plus_learned_adjustment` | Implemented, evaluated as its own condition |
| 3 | DyaPlex expressions can exceed what the robot can do | `constrain_motion`: mask disabled joints, clamp range, clamp per-step speed from q_t | `test_motion_obeys…`, `motion_metrics` | Implemented |
| 3 | Deploy G in real conversations and observe results | `RobotInterface`, `LiveInputs` (arrival-aware), `gr-live`, session recorder | `test_live_inputs_match_offline_alignment` | Simulator rehearsal only. No robot interface in repo. |
| 3 | Collect wrong-for-context (nod at t = 4.2 s where a brow raise fits) and unrecognizable gestures | [Trial protocol](#paired-trial-protocol). Fixture `case` field covers clear, ambiguous and wrong_intent. | Fixture manifest | Protocol written. Collection pending. |
| 3 | Dataset of (robot motion, embodiment, context, observer interpretation) | `gr-trial --all` → `gr-manifest` → `gr-rating-packet` → `gr-votes` | `test_blinded_packet_votes_and_condition_report` | Tooling implemented. Votes pending. |
| 4 | R_φ(z \| M_r, b_m, x) predicts human interpretation | `Observer` on realized motion, capabilities and context | `observer_test` in `evaluation.json` | Synthetic only |
| 4 | e.g. fast nod → P(enthusiastic) .78, P(neutral) .17, P(unclear) .05 | Labels `unclear`, `neutral_ack`, `enthusiastic_ack`; soft targets are vote proportions. The PDF numbers are illustrative. | — | Implemented |
| 4 | Cross-entropy training | Soft-target `F.cross_entropy(logits, vote distribution)` | `observer_test.soft_cross_entropy` | Implemented |
| 4 | Option: teleop demos of what the robot should do | `"demonstration": true` rows; `gr-train --imitation-only` → `g-imitation.pt` | `imitation-only` run | Implemented. Teleop data pending. |
| 5 | L = −log R_φ(z* \| G_θ(M_h, x, b_m), b_m, x) | `refinement_loss` meaning term, R frozen | `test_meaning_loss_flows_through_frozen_r…` | Implemented |
| 5 | L = λ L_demo + λ L_meaning + λ L_temporal | `--demo-weight`, `--meaning-weight`, `--smooth-weight` | `test_temporal_loss_penalizes_abrupt_chunk_boundaries` | Implemented |

## Model G

**Inputs at a chunk boundary t** (multiples of 500 ms on the recording clock):

- **M_h**: teacher frames stamped in (t − 500 ms, t], normalized to [−1, 1] per channel, plus a validity bit. Frame k is stamped at (k + 1) × 50 ms and summarizes the interval ending there.
- **x_t**: the context row at the last frame before t. Audio features are partner and agent dBFS, pitch, voicing and activity, normalized as in `audio.normalized`. The PersonaPlex state (width d) follows, plus an availability bit.
- **b_m**: 8 × [enabled ∈ {0, 1}, min rad, max rad, max rad/s]. G sees positions divided by `SCALE_RAD`. Disabled channels are all-zero rows.
- **q_t**: measured [position rad, velocity rad/s] per channel from the robot, divided by `SCALE_RAD`. `eye_yaw` is the mean of both eyes.

**Output.** `Generator.forward` produces 10 × 8 targets, one per 50 ms frame. `constrain_motion` starts from the measured position clipped into calibrated range. At each step it limits motion to `max rad/s × 50 ms`, clamps to [min, max], and multiplies by `enabled`. Disabled joints therefore stay exactly at zero, and every command is reachable under the calibration.

**Commands vs. reached motion.** Target k is sent to `RobotInterface.command` at the start of frame k and should be reached at its end. The adapter enforces its own actuator rate limits as well. `realized-motion.npz` stores both `commanded_rad` and the measured `robot_rad`. `motion_metrics` reports per-channel tracking error, whether commands stayed inside position and speed limits, the largest measured excursion on disabled channels, and frames where measured motion left the limits.

In the demo replays (simulator), commands always stayed within limits. Mean tracking error was ≤ 0.009 rad on the jaw and ≤ 0.002 rad elsewhere. Disabled neck pitch moved at most 1.3e-5 rad, which is physics settling.

**Residual variant.** M_r = clamp_b(M_direct + 0.5 ΔM_θ). M_direct is the most recent 500 ms of projected DyaPlex motion, replayed causally. The capability constraint applies after the addition.

**Calibration units.** Radians and rad/s in the robot's joint frame, with the neutral pose at 0 (min ≤ 0 ≤ max). Simulator values are `gr_model.SIM_LIMITS`. A hardware calibration uses the same JSON shape as `examples/calibration.example.json`.

## Real-data contract

All clocks are seconds on one recording clock, the start of the partner audio. Use `--teacher-offset-s` and `--states-offset-s` to shift device clocks onto it, measured from a sync event such as a clap or tone recorded by every device. Each NPZ may carry `arrival_s`, the time a sample became available to the robot computer on the same clock. If it is absent, arrival equals the timestamp.

| Source | File keys | Units / rules |
| --- | --- | --- |
| DyaPlex motion | `time_s [n]`, `channels` (exact order above), `human [n, 8]` in [−1, 1], optional `human_valid [n]`, optional `arrival_s` | 1 = channel's robot-scale extreme (`SCALE_RAD`): neck ±0.25 rad, jaw 0.30, eye yaw 0.40, eyelids 0.55 (closed), brows 0.025 |
| DyaPlex decoded features | `time_s`, `features [n, f]` + `--teacher-map calibration.npz` (`weights [f, 8]`, `bias [8]`, optional `feature_mean`, `feature_scale`) | Fit with `gr-fit-map` on paired reference clips, never by hand |
| PersonaPlex states | `time_s`, `features [n, d]` from `personaplex --capture-states` | Stamped at the end of each 80 ms input frame after prompts |
| Audio | partner WAV (+ `--agent` WAV or `--stereo`) | Defines the 20 Hz grid |
| Robot telemetry (written by `gr-live`/`gr-replay`) | `time_s`, `channels`, `robot_rad`, `commanded_rad`, `initial_state_rad [chunks, 2, 8]`, `chunk_start_s`, `inference_ms`, `session` (JSON) | `session` holds session id, participant group, context id, profile, capabilities, policy, G hash |

**Alignment rule** (identical offline in `gr-prepare` and online in `LiveInputs`; `test_live_inputs_match_offline_alignment` checks this):

1. A teacher sample fills history frame k only if its timestamp is ≤ k's time, it is at most 150 ms old, and it arrived by the boundary where G consumes it.
2. PersonaPlex states may be at most 250 ms old and must have arrived by the boundary.
3. Stale, invalid or missing samples become zeros with validity 0. `stream.json` reports coverage and median age.

**Cold start and dropouts.** The first chunk has no history, so it eases to neutral (`fallback: cold_start`). A chunk whose 500 ms history contains no valid teacher frame does the same (`no_teacher_motion`). Partial gaps go to G as zeroed frames with validity 0. `gr-trial --all` skips cold-start chunks.

**Validating a DyaPlex map.** List clips in `pairs.json` (example: `examples/teacher-map-pairs.example.json`). Each clip pairs a decoded export with reference eight-channel motion on its own clock, such as motion capture of the same face retargeted by the robot team. Group clips by speaker or session. `gr-fit-map` interpolates the reference onto the feature clock, standardizes features, picks a ridge penalty on validation groups, and reports per-channel MAE and R² on unseen test groups (`calibration.json`).

## Real-conversation loop (page 3)

```
mic / partner audio ─┐
DyaPlex (teacher)  ──┼─► LiveInputs.push(name, time_s, arrival_s, values)
PersonaPlex states ──┘          │ at every 500 ms boundary t
                                ▼
          RobotInterface.state() ─► q_t ─► G (or projected baseline) ─► 10 targets
                                                     │ 20 Hz
                                     RobotInterface.command / advance
                                                     │
                     measured robot_rad + commanded_rad + inputs G saw ─► session recording
                                                     │ after each chunk
                                         R scores realized motion (evaluation only)
```

The repo has **no physical-robot interface**. The only embodiment code is `../simulation/head.xml` and its MuJoCo demo. The integration points are therefore `RobotInterface` (`gr_runtime.py`), implemented by `MujocoRobot` (`simulate.py`) and `KinematicRobot` (tests).

A hardware adapter must implement `state()` (encoder position in rad and velocity in rad/s, per channel), `command(target_rad[8])`, and `advance(seconds)` (wall-clock wait to the next 50 ms tick). It must also supply measured calibration JSON. `gr-live` currently replays recorded files through `FileFeed`, using their `arrival_s`, to rehearse latency. A live deployment pushes samples from the DyaPlex, PersonaPlex and microphone processes instead. **No live or hardware run has happened.**

## Data collection plan

**Bootstrap (solves "G needs data, R needs G's clips").**

1. If approved teleoperation trajectories exist, package them as `"demonstration": true` rows and run `gr-train manifest.json --imitation-only --out runs/gr-bootstrap`. Then drive the robot with `g-imitation.pt` and no R.
2. If not, collect the first clips with `--policy projected` (constrained DyaPlex motion) plus controlled gestures, i.e. scripted teacher tracks: `gr-teacher-script` or hand-authored NPZs in the teacher format.
3. Collect observer ratings on those clips before fitting R. Teleoperation is optional.

### Paired trial protocol

- **Pairing.** For each conversation, record the same context (same partner audio and the same DyaPlex and PersonaPlex inputs) under at least `full` and one restricted profile such as `fixed-neck-pitch`. `gr-replay` and `gr-live` with the same `--context-id` produce paired chunks. Give every session of a participant the same `--group`.
- **Coverage.** Include each of the following:
  - clearly understood gestures (nods matching the experimenter intent)
  - ambiguous or unrecognizable motion (low-amplitude or jerky nods, nods truncated by a locked joint)
  - recognizable gestures that convey the wrong intent in context
- **Nod vs. brow example (page 3).** In conversation C1 at t = 4.2 s, the partner says something surprising and a brow raise fits, but the robot nods. Keep this as a trial: write `intent` for that moment in `intents.csv` (an added label such as `surprise_interest` if the study needs it) and let observers vote on the nod. R learns that a nod there is read as something else.
- **Per chunk recorded:** conversation context (`context`), teacher motion (`human`, `human_valid`), PersonaPlex features (context columns), commanded targets, measured motion, embodiment calibration and measured initial state. `stream.npz` and `realized-motion.npz` hold all of these.

### Rater protocol (pages 3–4)

1. Run `gr-rating-packet manifest.json --out packet --context-s 3`. Each clip shows 3 s of conversation (audio and video) before the 500 ms chunk plus 0.5 s after it. Codes are random, and clip order is shuffled. Raters receive `clips/`, `labels.txt` and `rating-sheet.csv` only. `key.json` (trial id, condition, profile) stays with the experimenter.
2. Raters never see the intended label, the model or policy, or the profile name.
3. Instruction: "After the highlighted moment, what did the robot communicate to the speaker?" Choose exactly one label: `neutral_ack` (calm "I hear you"), `enthusiastic_ack` (eager agreement or interest), or `unclear` (no clear message or not a gesture). Notes are optional.
4. Collect at least 5 independent raters per clip, each rating each clip at most once. Raters must not discuss clips. Use different raters for training data and for the held-out evaluation.
5. Merge with `gr-votes manifest.json --key packet/key.json sheets/*.csv --out rated.json`. Intent stays in `intent`, written by the experimenter from the conversation design or teleop instruction. Votes go in `votes`.

Example inputs: `examples/manifest.example.json`, `examples/intents.example.csv`, `examples/rating-responses.example.csv`.

**Manifest rows** have these fields:
- `id` (unique)
- `group` (participant/session; the split unit)
- `origin`
- `file` (relative)
- `profile` or `capabilities`
- optional `pair` (paired variants; must share `group`)
- `condition`
- `intent`
- `votes`
- `demonstration`

## Training (pages 4–5)

`gr-train` checks the manifest, then splits whole groups 60/20/20 (pairs cannot span groups, so paired variants share a split). It raises if any label lacks training votes. The stages:

1. Imitation pretraining on demonstrations, if present (`g-imitation.pt`).
2. R fit on realized motion, capabilities and context against vote distributions, with soft cross-entropy and early stopping on validation groups.
3. R frozen; G refined with λ_demo · L_demo + λ_meaning · (−log R(z* | G(…))) + λ_temporal · L_temporal. L_temporal penalizes velocity changes, including the step across the chunk boundary, which is anchored to the measured velocity.

`evaluation.json` records the manifest hash, whether data is synthetic, the split groups with rated-trial and vote counts, the model variant, loss weights, profile calibrations, R's held-out soft cross-entropy and majority agreement, and per-profile R scores for projected DyaPlex vs. refined G. A G checkpoint runs without R (`gr-replay --g g.pt` with no `--r`).

## Testing the substitution (step 7)

On held-out contexts, run the same streams through three conditions:
- `--policy projected --profile fixed-neck-pitch` (baseline)
- `--g g.pt --profile fixed-neck-pitch` (substitution)
- `--g g.pt --profile full` (reference)
- optionally, the residual G as a fourth condition

Then:
1. Package everything with `gr-trial --all --condition <name>`, merge with `gr-manifest`, and cut one packet across all conditions so the clips and protocol are identical.
2. Run `gr-evaluate rated.json --r r.pt --sessions <run dirs> --out report.json`.

The report includes:
- for each condition: trials, ratings, raters, intended-interpretation vote share, unclear rate, majority = intent
- paired per-context comparisons (which substitutions worked or failed)
- R's mean P(intent) on the recorded motion
- per-session joint-limit compliance, tracking error, smoothness at and inside boundaries, inference timing

`evidence` is `human_observer_votes` only when the manifest is not marked synthetic. **Acceptance check:** with neck pitch locked, refined G's intended-vote share beats projected DyaPlex on the same contexts, without a higher unclear rate, and approaches the full-capability reference.

## Results so far (synthetic and simulator only)

**Synthetic fixtures** (`gr-fixture`, seed 31: 16 groups, 864 trials, 1126 test-split votes; programmatic votes, not people). On the held-out groups, R's soft cross-entropy was 0.604 and its majority-vote agreement 97.5%. R's mean P(intended label) on generated trajectories, by policy and profile:

| Profile | Projected DyaPlex | G refined (direct) | G refined (residual) |
| --- | ---: | ---: | ---: |
| full | 0.540 | 0.856 | 0.840 |
| fixed-neck-pitch | 0.343 | 0.838 | 0.858 |
| fixed-eyes | 0.605 | 0.858 | 0.864 |

Across all held-out trials, imitation-only G reached 0.773, and its demonstration MAE was 0.013 rad. These scores show the training and evaluation path works. They are R proxies on synthetic data, not evidence that people recover the meaning.

**Simulator demo** (`runs/gr-v2/vera-gr-speech-demo.mp4`, 26 s). Audio is NVIDIA's published PersonaPlex `assistant` conversation; the teacher is the **scripted** `gr-teacher-script` track with 8 gestures. On the 8 gesture chunks, R's mean P(intent) on the motion MuJoCo actually reached was:
- 0.25 for projected DyaPlex with neck pitch locked
- 0.59 for G with neck pitch locked
- 0.62 for G with the full face

All replays and the `gr-live` rehearsal (50 ms simulated audio latency) kept commands inside position and speed limits, had zero self-contact and finite state, and averaged 0.4–0.9 ms G inference per chunk on CPU. The demo runs, packaged trials (153 chunks), manifest and blinded rating packet are under `runs/gr-v2` and `data/gr-v2`. **No one has rated them.**

## PersonaPlex states

The bridge runs NVIDIA's offline runner in its own Python 3.12 environment ([implementation](https://github.com/NVIDIA/personaplex), [weights](https://huggingface.co/nvidia/personaplex-7b-v1), license acceptance required). The full 7B model needs a CUDA GPU or a host with well over 16 GB memory, so it was not run on the development Mac. Record the observed `feature_width` and `median_interval_s` from `stream.json` after the first real run.

```sh
python3.12 -m venv .venv-personaplex
.venv-personaplex/bin/pip install 'moshi-personaplex @ git+https://github.com/NVIDIA/personaplex.git@16419a2ef7b98611f58c9df07477d00e5ad61d5f#subdirectory=moshi'
python -m speech_ai personaplex partner.wav --python .venv-personaplex/bin/python --capture-states --out runs/real-personaplex
python -m speech_ai gr-prepare partner.wav --agent runs/real-personaplex/agent.wav \
  --teacher dyaplex.npz --embeddings runs/real-personaplex/states.npz --out stream.npz
```

G and R must be trained on streams with the same context width (8 + d + 1 with states, 8 without).

## Open gaps: exact inputs and commands

| Needed | Consumed by |
| --- | --- |
| Real DyaPlex output for recorded conversations in the teacher format above (or decoded features + paired reference motion for ≥ 3 speaker groups) | `gr-fit-map pairs.json --out calibration.npz`, then `gr-prepare … --teacher [--teacher-map]` |
| A GPU host with PersonaPlex 7B and the license accepted | `personaplex … --capture-states`, then `gr-prepare --embeddings` |
| A robot `RobotInterface` adapter plus a measured calibration JSON for the physical head | `gr-live … --calibration calib.json` (add the adapter to `run_robot`) |
| Clock sync event and per-device offsets; arrival logging on the robot computer | `--teacher-offset-s`, `--states-offset-s`, `arrival_s` |
| Paired real-conversation sessions (full + restricted) with participant groups | `gr-live`/`gr-replay` → `gr-trial --all` → `gr-manifest --intents` |
| ≥ 5 independent observers per clip (separate pools for training and held-out evaluation) | `gr-rating-packet` → `gr-votes` → `gr-train` / `gr-evaluate` |
| Optional approved teleop demonstrations | `gr-train --imitation-only` |

## Local artifacts outside the G/R deliverable

These are gitignored and left untouched.
- Earlier cue, pose and Seamless/ViCo outputs:
  - `data/seamless` (563 MB), `data/vico`, `data/synthetic`, `data/models`
  - `runs/demo`, `runs/final-smoke`, `runs/learned-replay`, `runs/training`
  - `runs/personaplex-assistant`, `runs/personaplex-assistant-nod-busy`, `runs/personaplex-backchannel`
  - `runs/seamless-cadence`, `runs/seamless-pose`, `runs/seamless-weak`, `runs/vico-pose`
  - `runs/vera-speech-cue-demo.mp4`
- Earlier G/R runs from the previous code version:
  - `data/gr-synthetic`
  - `runs/gr-full`, `runs/gr-fixed-neck-pitch`, `runs/gr-replay-fixed`, `runs/gr-residual`
  - `runs/gr-training`, `runs/gr-imitation-*`
  - `runs/vera-gr-replay.mp4`

The cue detector, cue-specific linear/GRU training and pose code they came from were removed from `speech_ai/`. They remain in git history before this change.
