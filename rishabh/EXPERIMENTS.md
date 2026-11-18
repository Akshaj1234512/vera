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
| 3 | Deploy G in real conversations and observe results | `gr-live --listen` (TCP producers, arrival stamped on receipt, wall-clock pacing), `NetworkRobot` bridge with feedback watchdog, session recorder | `LiveLinkTests`, `test_live_inputs_match_offline_alignment` | Live path runs end to end with MuJoCo and a fake bridge. No DyaPlex producer, PersonaPlex host or physical robot yet. |
| 3 | Collect wrong-for-context (nod at t = 4.2 s where a brow raise fits) and unrecognizable gestures | [Trial protocol](#paired-trial-protocol). Fixture `case` field covers clear, ambiguous and wrong_intent. | Fixture manifest | Protocol written. Collection pending. |
| 3 | Dataset of (robot motion, embodiment, context, observer interpretation) | `gr-trial --all` → `gr-manifest` → `gr-rating-packet` (blinded, text-free clips) → `gr-votes` → evidence-tiered `gr-evaluate` | `ObserverStudyTests`, `test_rating_render_has_no_labels` | Tooling implemented. Votes pending. |
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
microphone ─► AudioFrontend ─┐  JSON lines over TCP, stamped on the session clock
DyaPlex process ─────────────┼─► gr-live --listen  (LiveServer stamps arrival on receipt)
PersonaPlex --publish ───────┘          │ every 500 ms boundary t, wall-clock paced
                                        ▼
   robot bridge state ─► q_t ─► G (or projected baseline) ─► 10 targets ─► robot bridge (20 Hz)
                                        │
        commanded + measured motion + exactly what G saw ─► stream.npz + realized-motion.npz
                                        │ after each chunk
                            R scores realized motion (evaluation only, never drives)
```

**Live mode.** `gr-live --listen HOST:PORT` runs G on inputs that arrive while the session runs.
- **Clock.** The session clock is the robot computer's monotonic clock. Each producer receives the session start time on connect, so producers on the same machine stamp samples with `Publisher.session_time()`. Producers on other machines pass a measured offset (`--teacher-offset-s`, `--states-offset-s`). `--wait-for-producers N` holds the clock until N producers have connected, so every source starts aligned.
- **Rejections.** The server stamps arrival when a line is received. It rejects and counts any sample stamped after its own arrival (a clock-offset error), any out-of-order sample, and any sample with the wrong width (`input_stats` in `interpretations.json`).
- **Pacing.** The loop runs on the wall clock. Each chunk records how late its boundary was (`boundary_late_ms`).

The message schema is in the `gr_link.py` docstring:
- `teacher`: 8 normalized channels
- `teacher_features`: decoded DyaPlex features, projected with `--teacher-map`
- `states`: PersonaPlex state
- `audio`: 8 raw features from `audio.AudioFrontend`, which reproduces `extract()` exactly

**Producers.**
- **PersonaPlex.** `python -m speech_ai personaplex … --publish HOST:PORT` patches the same state hook as `--capture-states` and streams each state as it is produced.
- **Recorded files (dry runs).** `gr-publish HOST:PORT partner.wav --stereo --teacher … [--embeddings …] [--latency-s …]` streams recorded inputs in real time.
- **DyaPlex.** No DyaPlex producer exists, because no DyaPlex code or export is in hand. It must call `Publisher.send("teacher", t, values8)`, or send `teacher_features` (see blockers).

**Robot.** The robot is reached through `--robot network --robot-address HOST:PORT`, a JSON-lines bridge:
1. The bridge sends `hello` with `embodiment`.
2. It streams `state` (position rad, velocity rad/s for the 8 channels) at ≥ 20 Hz.
3. G sends a `target` every 50 ms, and `hold` on stop.

`NetworkRobot` refuses state older than 100 ms. If feedback is lost, the loop sends `hold`, cuts the session at the last complete frame and records `aborted`. A network robot also requires `--calibration`. Sessions are labeled `live_physical_robot` only when the bridge declares `embodiment: physical`.

`gr-replay --robot network` plays one prepared stream on the bridged robot at wall-clock rate. This is how full and restricted sessions get identical conversational inputs.

The VERA proposal places a vendor servo driver behind this adapter, but the vendor, protocol and head do not exist yet, so the bridge itself is not written. Instead:
- `tests/test_gr.py::FakeBridge` exercises the protocol, including feedback loss.
- MuJoCo (`simulate.MujocoRobot`) and the kinematic adapter run in-process.

`gr-live` without `--listen` still rehearses from recorded files (`live_rehearsal_from_files`).

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

1. Run `gr-rating-packet manifest.json --out packet --context-s 3`.
   - Each clip shows 3 s of conversation (audio and video) before the 500 ms chunk plus 0.5 s after it. Clips are cut only from the **text-free rating render** (`rating-source.mp4`), never from the labeled presentation video, which shows the policy, profile and R overlay. Container metadata is stripped.
   - For a physical robot, attach a camera recording with `gr-trial --all --rating-video camera.mp4`. It must use the session clock and contain no on-screen text.
   - The command refuses to run if any trial lacks a rating video.
   - Codes are random, and clip order is shuffled. Raters receive `packet/` (`clips/`, `labels.txt`, `rating-sheet.csv`) only. The key (trial id, condition, profile) is written outside it, to `packet-key.json`.
2. Raters never see the intended label, the model or policy, the profile name or R's predictions. They see only the robot's motion and the conversation.
3. Instruction: "After the highlighted moment, what did the robot communicate to the speaker?" Choose exactly one label: `neutral_ack` (calm "I hear you"), `enthusiastic_ack` (eager agreement or interest), or `unclear` (no clear message or not a gesture). Notes are optional.
4. Collect at least 5 independent raters per clip, each rating each clip at most once. Raters must not discuss clips. Use different raters for training data and for the held-out evaluation.
5. Merge with `gr-votes manifest.json --key packet-key.json sheets/*.csv --out rated.json`. Intent stays in `intent`, written by the experimenter from the conversation design or teleop instruction. Votes go in `votes`, marked `vote_source: observer_rating_sheets`. Trials that already hold votes from any other source are refused.

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

**Evidence tiers.** Every condition is reported per tier, and tiers are never pooled:
- `unrated`
- `synthetic_votes`: fixtures, hand-entered counts, or a manifest marked synthetic
- `human_votes_on_simulated_motion`
- `human_votes_on_robot_motion`

`human_observer_evidence` is true only for votes that came through rating sheets. `human_evidence_on_physical_robot` additionally requires motion recorded from a bridge that declared `embodiment: physical`. Each condition also states its `teacher` (scripted/synthetic vs. unverified export). R scores are tagged `model_proxy_not_human`. `gr-train` records the same tiers in `evaluation.json` and prints a notice when no human votes exist.

**Acceptance check:** with neck pitch locked, refined G's intended-vote share beats projected DyaPlex on the same contexts, without a higher unclear rate, and approaches the full-capability reference.

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

All replays and the `gr-live` rehearsal (50 ms simulated audio latency) kept commands inside position and speed limits, had zero self-contact and finite state, and averaged 0.4–0.9 ms G inference per chunk on CPU.

**Live-path dry run** (`runs/gr-v3/live-listen-fixed`; MuJoCo; same audio and scripted teacher). `gr-live --listen --wait-for-producers 1` was fed by `gr-publish` in real time with 40 ms delivery delay:
- 521/521 teacher frames were accepted and 0 rejected. No chunk lacked teacher motion after cold start.
- Boundaries were at most 10.1 ms late. The worst single frame was 82 ms late, while MuJoCo was rendering video.
- G inference averaged 0.52 ms per chunk.
- R's mean P(intent) on the 8 gesture chunks was 0.38, against 0.59 for the zero-latency replay. Delivery delay shifts which frames fall in each 500 ms window, and that should be measured with people.

Paired packets for projected@fixed-neck, live G@fixed-neck and G@full (153 clips) are under `runs/gr-v3/packet`, with the key in `runs/gr-v3/packet-key.json`. `gr-evaluate` reports all 153 as `unrated`, with `human_observer_evidence: false`. **No one has rated them.**

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

## What is blocked, and on what

Synthetic and simulator results above stay labeled as such. No real-data, hardware or human result exists. Each blocker, with what unblocks it and the command that takes over:

| Gap | Blocked on | Then run |
| --- | --- | --- |
| Real DyaPlex motion | No DyaPlex code, export or output format in the repo or from the team. The format cannot be invented. | The DyaPlex process sends `teacher` (or `teacher_features` plus `gr-fit-map` on ≥ 3 speaker groups of paired reference motion) to `gr-live --listen`. For offline use: `gr-prepare --teacher`. |
| Real PersonaPlex states | PersonaPlex 7B needs a CUDA GPU or a host with far more than 16 GB memory, plus NVIDIA license acceptance. The live server-mode hook is unverified: `--publish` patches the offline runner's state hook. | `personaplex partner.wav --python <env> --capture-states --publish HOST:PORT`. Record `feature_width` and timing in `stream.json`. |
| Physical robot | The head is not built. No vendor servo driver or protocol has been chosen. | Hardware team writes the JSON-lines bridge (`gr_link.py` docstring) and measures `calibration.json`. Then run `gr-live --listen … --robot network --robot-address … --calibration …`. |
| Clock sync across machines | Needs the real devices. | Record a shared sync event and set `--teacher-offset-s` and `--states-offset-s`. Check that `input_stats` shows no clock rejections. |
| Paired real conversations | Needs the robot, participants and an approved protocol. | `gr-live` (or `gr-replay --robot network` on a recorded stream) once per profile with the same `--context-id` and `--group`, then `gr-trial --all --rating-video camera.mp4` and `gr-manifest --intents`. |
| Human ratings | Needs recruited observers: ≥ 5 per clip, with separate pools for training R and for held-out evaluation. | `gr-rating-packet` → raters fill `rating-sheet.csv` → `gr-votes` → `gr-train` / `gr-evaluate`. |
| Does G preserve meaning better than projected DyaPlex? | All of the above. | `gr-evaluate rated.json --r r.pt --sessions …`. Compare `human_votes_on_robot_motion` rows for G vs. projected under the same restricted profile, using the paired comparisons. |
| Optional teleop demonstrations | A teleoperation rig. | `gr-train --imitation-only` |

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
