# G/R speech-motion: design, data and evidence

Specification: `Speech.pdf` (Akshaj's notes, 5 pages). Robot: Samhita's rig, `samhita_sim/robotic_head_rig.xml`.

Every result below is labeled with its evidence type:
- **real motion:** measured human motion.
- **synthetic votes:** the rule-based observer.
- **simulator:** MuJoCo.

No human-observer, DyaPlex-output, PersonaPlex-7B-state or hardware result exists yet ([what is blocked](#what-is-blocked-and-on-what)).

## Answers to Akshaj's questions (Nov 18)

**Is the simulation running the deployed G model? Where is it, and what data trained it?**
- **Yes.** Each "G" panel is G running in the closed loop: a chunk is planned from the measured robot state every 500 ms, and R only scores the result.
- **Model:** `runs/gr-real/selected/g.pt`, with R in `r.pt` and `evaluation.json` alongside.
- **Training data:** real conversational motion and audio of 46 Seamless Interaction participants (24 sessions, 155 min, naturalistic/dev). The labels and the meaning loss use synthetic votes from a rule-based observer, because no human ratings exist yet.
- **Earlier demo:** `runs/gr-v2/vera-gr-speech-demo.mp4` was trained on purely synthetic fixtures with a scripted teacher, on the old 8-joint head. That's superseded.

**Use Samhita's head.**
- **Done:** all 27 of her actuators are G's output channels (`speech_ai/embodiment.py`).
- **Two load-time fixes were needed, without editing her file;** details are in [the rig section](#robot-samhitas-rig).

**No eye or eyebrow movement?**
- **Old head:** brows had a 0.025 rad range and the eyes moved only in the substitution condition, so neither was visible.
- **Now:** eyes follow measured gaze, and brows, lids, mouth, cheeks, nose and chin follow measured action units. In the 40 s demo the eyes move 9–21° and the brows 4–6 mm.
- **Limits:** the rig's skin shows a brow raise only subtly, and closed lids cannot cover the eyeballs. Each demo panel has live joint meters so this motion is checkable.

**The "DyaPlex" panel had the neck locked and only the jaw moved.**
- **That panel was never DyaPlex.** It was a scripted teacher (nods, blinks, speech jaw) replayed directly, shown with neck pitch locked, which left only the jaw.
- **DyaPlex has no released code or weights** ([paper](https://arxiv.org/abs/2606.03874), [project page](https://research.nvidia.com/labs/amri/projects/DyaPlex)), so nobody outside NVIDIA can run it.
- **Replacement:** the teacher is now real measured human motion from DyaPlex's own training data. The "projected DyaPlex" baseline retargets every frame onto all 27 joints.
- **The neck moves:** in the demo the baseline moves neck yaw/pitch/roll 25° / 16° / 22°, and G moves 12° / 5° / 7°.

## PDF → implementation map

| Page | Requirement | Implementation | Status |
| --- | --- | --- | --- |
| 1 | Substitute a gesture when joints are missing while preserving observer meaning | Capability-conditioned G; R on votes; profiles `full`, `fixed-neck-pitch`, `fixed-neck`, `fixed-eyes` | Works under synthetic votes on real motion; human test pending |
| 1 | G deployed; R helps train and evaluate | `gr-replay`, `gr-live` and `gr-demo` drive the rig with G; R only scores | Implemented |
| 1 | M_h: 500 ms of DyaPlex-style facial motion | 10 frames × 21 human channels: head yaw/pitch/roll, gaze yaw/pitch, 16 action units (`human.py`) | Real human motion; DyaPlex export pending |
| 1–2 | b_m ∈ ℝ^K: active DoF, ranges, calibration | 27 × [enabled, min, max, speed] from the rig XML (K = 108); `--calibration` JSON | Implemented |
| 2 | x_t from encoders + PersonaPlex | 8 causal audio features from both speakers (+ PersonaPlex state when available) | Audio real; PersonaPlex 7B pending |
| 2 | M_r = G(M_h, x_t, b_m, q_t), q_t = robot state | Measured joint position/velocity at each 500 ms boundary | Implemented |
| 2 | Small causal transformer | 2 layers, width 64, causal over the 10 history frames | Implemented |
| 2 | Option M_r = M_direct + ΔM | `--residual`: retarget of the latest frame + learned Δ, then constrained | Trained and evaluated as its own condition |
| 3 | DyaPlex exceeds what the robot can do | `constrain_motion`: lock disabled joints, clamp range and per-step speed | Implemented |
| 3 | Real conversations; collect wrong-intent and unrecognizable cases | Live loop, paired-trial and rater protocols below | Tooling done; collection pending |
| 4 | R(z \| M_r, b_m, x) from observer interpretations | `Observer` + soft cross-entropy; labels `no_signal`, `neutral_ack`, `enthusiastic_ack`, `unclear` | Synthetic votes only |
| 4 | Option: teleop demonstrations | `demonstration: true` rows; `--imitation-only` | Retargeted human motion serves as demonstrations under `full` |
| 5 | L = −log R(z* \| G(…)) + λ_demo L_demo + λ_temporal L_temporal | `gr_train.refinement_loss` with frozen R | Implemented; λ chosen on validation groups |

## Robot: Samhita's rig

`embodiment.py` parses the rig XML: 27 position actuators (3 neck hinges, 4 eye hinges, the jaw hinge, 19 skin-bone slide joints in metres). Each actuator is one G channel, normalized by its range. Speed limits: neck 2 rad/s, eyes 6 rad/s, jaw 3 rad/s, skin sliders 0.08 m/s.

Load-time fix, applied without editing her file: MuJoCo gravity compensation on the head, the eyes and the 20 skin-bone bodies (standing in for servo holding torque). Measured without it:
- **The head topples.** A −0.2 rad pitch command falls to the −0.45 rad joint limit; roll falls to its 0.35 rad limit. The head is 4 kg with its center of mass 12 cm above the pivot, against a neck kp of 5–8.
- **Upward face sliders sag** by mg/kp ≈ 2.8 mm. Brows reached 67% of their command, mouth corners 60%, nose wrinkle 22%.

With it, all 27 actuators settle within 0.11% of range, and the neck reaches a 0.2 rad step in 0.24–0.34 s (90% rise). Both cases are covered by tests.

Findings to share with Samhita (rig, not G):
- **Eyelids:** closed lids (−12 mm) lower the skin but cannot cover the eyeballs, which sit in front of the skin. Blinks are therefore invisible.
- **Brows:** a full raise changes about 360 px in a face close-up; a brow lower changes about 9 px.
- **Skin:** there's a visible seam artifact across the right cheek.
- **Signs:** checked by rendering. `neck_pitch` + looks up, `eye_pitch` + looks up, `neck_yaw`/`eye_yaw` + turn to the robot's left, `neck_roll` + raises the left side, eyelid − closes, `lower_lip_drop` − drops.

## DyaPlex and PersonaPlex (from their papers and code)

**DyaPlex** ([arXiv 2606.03874](https://arxiv.org/abs/2606.03874), NVIDIA, June 2026). No code or weights are released.
- **Output:** SMPL-X body pose plus a FLAME face (6D jaw rotation and 50 expression coefficients) at 25 fps. Head rotation is carried by the body's neck and head joints. There is no eye gaze.
- **Architecture:** a frozen PersonaPlex speech tower; the motion tower cross-attends to the post-block hidden states of all 32 PersonaPlex layers.
- **Tokenizer:** RVQ-VAE motion tokens at 12.5 Hz.
- **Training data:** Seamless Interaction.

**PersonaPlex** ([NVIDIA/personaplex](https://github.com/NVIDIA/personaplex) at `16419a2`), a Moshi-based model at 12.5 Hz with hidden width 4096.
- **Our capture:** `--capture-states` / `--publish` patch `LMGen.step` and `process_transformer_output` and record the final, out-normed transformer output, one 4096-d vector per 80 ms frame. That's the last layer only; DyaPlex consumes all 32.
- **Server path:** `moshi.server` calls the same `LMGen.step`, but our readiness gate (after the prompts) is written for the offline runner, so server-mode capture still needs a hook and a run on a GPU host.

**Consequence for M_h.** DyaPlex output must be mapped into the 21-channel human space:
- Its neck and head joints give head yaw/pitch/roll directly.
- Its FLAME coefficients map to the action-unit channels with `gr-fit-map`, using paired clips, for example action units measured on renders of the same FLAME sequence.
- It has no gaze, so the gaze channels are marked invalid.

## Real human motion: Seamless Interaction

Because DyaPlex is unreleased, the teacher is measured motion from the dataset DyaPlex was trained on ([Seamless Interaction](https://github.com/facebookresearch/seamless_interaction), CC-BY-NC 4.0). Download it with `gr-seamless`; it fetches exactly the URLs the dataset's own downloader builds.
- **Head (3 channels):** SMPL-H neck × head local rotation, 30 Hz. Signs were verified on the source video: −16° looks down, +27° yaw turns to the person's left, and so on. Centered on the person's median posture and scaled by 30°.
- **Gaze (2 channels):** `gaze_encodings` scaled by 2.5 σ per person. Low values of column 1 match looking down on video. Column 0's left/right sign couldn't be verified at the released resolution, so it's marked `unverified`.
- **Action units (16 of 24):** 50th → 99th percentile per person, so 0 is that person's neutral face. Seven AUs that stay near zero in conversation are dropped, as is `LipsParts`, which is byte-identical to `LowerLipDepressor` in the release. The column order was checked against video: the LipCornerPuller peak is a smile and the JawDrop peak is an open mouth.
- **Feature versions:** `movement_v4` where it's marked valid, otherwise the v1 features (v4 is still being rolled out). Participants with less than 50% valid frames are skipped.
- **Splits:** naturalistic/dev gives 46 participant streams from 24 sessions (155 min) for training and model selection. naturalistic/test gives 8 streams from 4 sessions (27 min), used only for the final score and the demo.

## The "projected DyaPlex" baseline

`human.RETARGET` is a fixed, hand-written semantic map from the 21 human channels to the 27 joints:
- head → neck at 1:1 angle
- gaze → both eyes
- brow raisers/lowerer → brows
- eyes-closed and lid-tightener → eyelids
- jaw drop → jaw
- lip-corner puller/depressor → mouth corners
- upper-lip raiser → upper lips
- lower-lip depressor → lower lips
- cheek raiser → cheeks
- nose wrinkler → nose
- chin raiser and pucker → chin

At run time the baseline streams: at every 50 ms frame it commands the retarget of the newest teacher frame, within the capability descriptor. A locked joint simply drops out; there is no substitution. This is what driving the robot directly from DyaPlex would look like.

## Training data and the synthetic observer

`gr-dataset` turns each 500 ms chunk of each stream into trials:
- **M_h:** the human's previous 500 ms.
- **x_t:** both speakers' audio features at the boundary.
- **z\*:** the intent of the human's next 500 ms, by rule. A head drop of ≥ 3° is an acknowledgment. A smile, a brow raise or a nod ≥ 8° makes it enthusiastic. Otherwise it's `no_signal`.
- **q_t:** the robot state from following the retarget.
- **Candidates:** for each of 4 profiles, the retargeted next 500 ms (a demonstration under `full`), plus one variant: a look-down with lowered lids, an eyebrow flash, damped motion, or jerky motion.

**Votes** (5–9 per trial) come from `human.observer_probabilities`, a rule I wrote. It reads:
- nods as acknowledgment
- a look-down with lowered lids, or an eyebrow flash, as a weaker acknowledgment
- smiles, brow raises and big nods as enthusiastic
- motion rougher than real faces as unclear (calibrated on the real motion's 90th percentile)

This rule encodes the substitution hypothesis, so G succeeding under it is a pipeline result, not evidence. The manifest is `synthetic: true`, every row is `vote_source: synthetic_rule_observer`, and `gr-evaluate` labels all of it `synthetic_votes`.

The dev dataset has 51,888 trials in 13 person-disjoint groups. People who share any session are grouped together, so no person spans splits. Chunks by intent: 3,828 no-signal (30% subsample), 1,214 neutral and 1,444 enthusiastic acknowledgments.

## Training, selection and held-out results

`gr-train` works in three stages:
1. **Imitation** pretrains G on the `full` demonstrations.
2. **R** is fit on the votes with soft cross-entropy.
3. **G refinement:** with R frozen, G is refined with λ_demo L_demo + λ_meaning (−log R(z\*|G)) + λ_temporal L_temporal.

The sweep over λ_meaning was selected on the **validation groups**: macro P(intent) over every profile × intent cell, which weighs false gestures as much as missed acknowledgments.

| G variant | λ_meaning (refine epochs) | validation macro P(intent) | test smoothness |
| --- | --- | ---: | ---: |
| direct | 0.04 (8) | 0.380 | 0.005 |
| direct | 0.3 (20) | 0.395 | 0.018 |
| direct | 1 (20) | 0.412 | 0.041 |
| **direct (selected)** | **3 (20)** | **0.428** | 0.086 |
| direct | 10 (20) | 0.392 | 0.188 |
| direct | 30 (20) | 0.410 | 0.277 |
| residual | 0.3 / 1 / 3 (20) | 0.403 / 0.408 / 0.406 | 0.015 / 0.042 / 0.092 |

The selected checkpoints are in `runs/gr-real/selected/`; every sweep run is kept under `runs/gr-real/sweep-*`.

**Held-out score** on naturalistic/test, never used for training or selection: `gr-score`, written to `runs/gr-real/selected/heldout-test-split.json`. Evidence type: real motion with synthetic votes. On these held-out votes, R's soft cross-entropy is 0.924 and its majority agreement 77.2%.

The table gives R's mean P(intended meaning) per profile × intent. "Retarget UB" is the human's own next 500 ms retargeted with zero latency, an upper bound on the streaming baseline. "Hold" repeats the last frame for the chunk.

| profile / intent | chunks | retarget UB | hold | G (selected) | G residual λ=1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full / no_signal | 759 | 0.532 | 0.765 | 0.653 | 0.655 |
| full / neutral_ack | 272 | 0.573 | 0.130 | 0.244 | 0.254 |
| full / enthusiastic_ack | 196 | 0.508 | 0.071 | 0.468 | 0.384 |
| fixed-neck-pitch / no_signal | 759 | 0.665 | 0.872 | 0.631 | 0.593 |
| fixed-neck-pitch / neutral_ack | 272 | 0.238 | 0.057 | 0.252 | 0.287 |
| fixed-neck-pitch / enthusiastic_ack | 196 | 0.163 | 0.019 | **0.514** | 0.417 |
| fixed-neck / neutral_ack | 272 | 0.236 | 0.054 | 0.264 | 0.285 |
| fixed-neck / enthusiastic_ack | 196 | 0.165 | 0.019 | **0.552** | 0.409 |
| fixed-eyes / neutral_ack | 272 | 0.550 | 0.125 | 0.254 | 0.240 |
| fixed-eyes / enthusiastic_ack | 196 | 0.521 | 0.077 | 0.439 | 0.388 |
| **macro (12 cells)** | | 0.451 | 0.320 | **0.461** | 0.431 |

What this does and does not show:
- **With the neck locked**, retargeting loses acknowledgments: enthusiastic drops from 0.51 to 0.16. G recovers them (0.51–0.55) using eyes, lids and brows. That's the substitution the notes describe, under the assumed observer.
- **With the full face**, the zero-latency retarget conveys acknowledgments better than G, because it replays the human's actual future while G must forecast 500 ms ahead.
- **G makes more false gestures** on no-signal chunks than holding still does.
- **G is less expressive than the retarget.** In the demo window: neck yaw 12° vs 25°, jaw 1.6° vs 13.6°, eyelids 2.7 vs 11.9 mm (5–95% ranges). That's regression to the mean from forecasting with a squared-error loss, and it's worst for speech articulators, which can't be predicted 500 ms ahead. Akshaj's expectation that DyaPlex is more expressive than our model holds.
- **Recommendation for the next design step:** let fast articulators (jaw, lips) follow the streaming retarget at frame rate, and let G plan only the slower social channels (neck, gaze, brows, lids). Alternatively, give G a stochastic or adversarial objective instead of pure MSE imitation.

## Demo

[`runs/gr-real/demo/demo.mp4`](runs/gr-real/demo/demo.mp4) is a 40 s clip with both speakers' audio, rendered by `gr-demo`:
- **Top left:** held-out participant P1007 (session S0908, 66–106 s), whose measured motion is M_h.
- **Top row, robot panels:** projected DyaPlex and G with the full face.
- **Bottom row:** both with neck pitch locked.
- **Each panel:** measured joint meters; locked joints are labeled.

All four replays kept commands within position and speed limits, with finite physics. G's mean inference time was 0.4–0.5 ms per chunk. With neck pitch locked, the locked joint drifted at most 0.006 rad from coupled neck dynamics.

Measured 5–95% motion ranges in the demo:

| joint group | projected full | G full | projected, pitch locked | G, pitch locked |
| --- | ---: | ---: | ---: | ---: |
| neck yaw / pitch / roll | 25° / 16° / 22° | 12° / 5° / 7° | 25° / 0.4° / 22° | 13° / 0.3° / 7° |
| eye yaw / pitch | 21° / 18° | 17° / 9° | 21° / 18° | 21° / 10° |
| brows / eyelids | 6.2 / 11.9 mm | 4.4 / 2.7 mm | 6.2 / 11.8 mm | 4.5 / 2.6 mm |
| jaw / mouth corners | 13.6° / 8.1 mm | 1.6° / 4.3 mm | 13.6° / 8.1 mm | 2.7° / 4.2 mm |

Reproduce the whole pipeline from `rishabh/`:

```sh
P=.venv/bin/python
$P -m speech_ai gr-seamless --split dev --count 30 --include V00_S0080_I00000135 --out data/seamless-motion/naturalistic-dev
$P -m speech_ai gr-seamless --split test --count 4 --seed 11 --out data/seamless-motion/naturalistic-test
$P -m speech_ai gr-dataset data/seamless-motion/naturalistic-dev/streams.json --out data/gr-real/dev
$P -m speech_ai gr-dataset data/seamless-motion/naturalistic-test/streams.json --seed 4 --out data/gr-real/test
$P -m speech_ai gr-train data/gr-real/dev/manifest.json --meaning-weight 3 --refine-epochs 20 --out runs/gr-real/selected
$P -m speech_ai gr-score data/gr-real/test/manifest.json --g runs/gr-real/selected/g.pt --r runs/gr-real/selected/r.pt \
  --out runs/gr-real/selected/heldout-test-split.json
```

Then download P1007's `video` and `boxes_and_keypoints/keypoints` files from the naturalistic/test split and run:

```sh
T=data/seamless-motion/naturalistic-test
$P -m speech_ai gr-demo $T/V00_S0908_I00000372_P1007/stream.npz \
  --partner-wav $T/V00_S0908_I00000372_P1009/audio-16k.wav --agent-wav $T/V00_S0908_I00000372_P1007/audio-16k.wav \
  --source-video $T/V00_S0908_I00000372_P1007/video.mp4 --keypoints $T/V00_S0908_I00000372_P1007/keypoints.npy \
  --start-s 66 --duration-s 40 --g runs/gr-real/selected/g.pt --r runs/gr-real/selected/r.pt --out runs/gr-real/demo
```

## Real-data contract

- **Common clock:** all times are seconds on one recording clock. Device clocks are shifted onto it with `--teacher-offset-s` / `--states-offset-s`.
- **`arrival_s`:** an NPZ may include it, meaning when each sample reached the robot computer.
- **Rejection:** a sample stamped after its own arrival is rejected as a clock error.

| Source | Keys | Notes |
| --- | --- | --- |
| Teacher (DyaPlex or human) | `time_s`, `channels` = `human.HUMAN_CHANNELS`, `human [n, 21]` in [−1, 1], optional `human_valid`, `arrival_s`, `source` | `source` names the provenance; Seamless files say `seamless_interaction_measured_human_motion_not_dyaplex` |
| Decoded DyaPlex features | `time_s`, `features [n, f]` + `--teacher-map` (`weights [f, 21]`, `bias [21]`) | Fit with `gr-fit-map` on held-out clips |
| PersonaPlex states | `time_s`, `features [n, 4096]` | `personaplex --capture-states` |
| Robot recording | `time_s`, `channels` (27 rig joints), `robot_q`, `commanded_q`, `initial_state_q`, `chunk_start_s`, `inference_ms`, `session` | Native units: rad for hinges, m for slides |

**Alignment rule.** A teacher frame fills history slot k only if it's stamped no later than k's time, is at most 150 ms old, and had arrived by the boundary where G uses it. PersonaPlex states may be at most 250 ms old. The rule is identical offline (`gr-prepare`) and live (`LiveInputs`), which is tested. At cold start, or with no valid teacher frame in the last 500 ms, the robot eases to neutral.

## Live loop (page 3)

```
microphone ─► AudioFrontend ─┐  JSON lines over TCP, stamped on the session clock
DyaPlex/teacher process ─────┼─► gr-live --listen  (arrival stamped on receipt)
PersonaPlex --publish ───────┘          │ every 500 ms, wall-clock paced
   robot state ─► q_t ─► G ─► 10 targets ─► robot (20 Hz)   [baseline: retarget every frame]
        commanded + measured + exactly what G saw ─► stream.npz + realized-motion.npz ─► R scores
```

**`gr-live --listen HOST:PORT`** waits for N producers with `--wait-for-producers`, then starts the session clock. Producers receive the start time in a handshake. The server rejects out-of-order, wrong-width or clock-violating samples and counts each reason.

**Producer message types** (schema in `gr_link.py`):
- `teacher`: 21 channels
- `teacher_features`: needs `--teacher-map`
- `states`: PersonaPlex states
- `audio`: 8 raw features from `AudioFrontend`

**Robots:**
- **MuJoCo:** the rig itself.
- **Kinematic:** for tests.
- **`--robot network`:** a JSON-lines bridge.
  - The bridge sends `hello` with its `embodiment`, then streams `state` (27 positions and velocities) at ≥ 20 Hz.
  - G sends `target` every 50 ms, and `hold` on stop.
  - State older than 100 ms sends `hold` and aborts the session.
  - Sessions count as `live_physical_robot` only when the bridge declares `physical`.

**Replaying on a bridged robot:** `gr-replay --robot network` plays a recorded stream at wall-clock rate, which is how paired full/restricted trials share identical inputs.

**Dry runs:** `gr-publish` streams recorded files into a live session in real time.

## Observer study (pages 3–4)

**Paired trials.**
- Record each conversation context under `full` and at least one restricted profile, with the same `--context-id`. Use one `--group` per participant.
- Include clear acknowledgments, ambiguous or unrecognizable motion, and recognizable gestures with the wrong intent. Page 3's example is a nod at t = 4.2 s where a brow raise fits.
- Write the experimenter's intent in `intents.csv`. Add labels such as `surprise_interest` if the study needs them.

**Rater protocol.**
1. **Packet:** `gr-rating-packet manifest.json --out packet` cuts clips from the text-free `rating-source.mp4`: 3 s of context, the 500 ms chunk, then 0.5 s. Metadata is stripped and codes are random.
   - The key is written outside the packet.
   - The command refuses to run if a trial has no rating video.
   - For a physical robot, attach a camera recording with `gr-trial --rating-video`.
2. **Blinding:** raters see only motion and conversation, never the intent, policy, profile, meters or R.
3. **Question:** "After the highlighted moment, what did the robot communicate to the speaker?" Answers: `no_signal`, `neutral_ack`, `enthusiastic_ack` or `unclear`.
4. **Raters:** at least 5 independent raters per clip, with separate rater pools for training R and for held-out evaluation.
5. **Merge:** `gr-votes manifest.json --key packet-key.json sheets/*.csv --out rated.json`.

**Evidence tiers** in `gr-evaluate`, never pooled:
- `unrated`
- `synthetic_votes`
- `human_votes_on_simulated_motion`
- `human_votes_on_robot_motion`

`gr-train` records the tier of its training votes and prints a notice when none are human.

**Acceptance check:** with neck pitch locked, G's human intended-vote share beats the projected baseline on the same contexts without a higher unclear rate.

## PersonaPlex states

```sh
python3.12 -m venv .venv-personaplex
.venv-personaplex/bin/pip install 'moshi-personaplex @ git+https://github.com/NVIDIA/personaplex.git@16419a2ef7b98611f58c9df07477d00e5ad61d5f#subdirectory=moshi'
python -m speech_ai personaplex partner.wav --python .venv-personaplex/bin/python --capture-states --out runs/real-personaplex
```

This needs a CUDA GPU, or a host with well over 16 GB of memory, plus license acceptance. G and R must be trained on streams of the same context width: 8 with audio only, 8 + 4096 + 1 with states.

## What is blocked, and on what

| Gap | Blocked on | Then run |
| --- | --- | --- |
| DyaPlex output | No public code or weights | Map SMPL-X neck/head + FLAME to the 21 channels (`gr-fit-map`); `gr-prepare --teacher` or publish `teacher` to `gr-live --listen` |
| PersonaPlex 7B states | GPU host + license; server-mode hook untested | `personaplex … --capture-states` / `--publish`; retrain with `--embeddings` |
| Human ratings for R | Recruited observers (≥ 5 per clip, separate train/eval pools) | `gr-rating-packet` → `gr-votes` → `gr-train` / `gr-evaluate` |
| Physical robot | Head not built; no servo vendor or protocol | Hardware bridge per `gr_link.py` + measured `--calibration`; `gr-live --robot network` |
| Paired real conversations | Robot, participants, approved protocol | `gr-live` / `gr-replay --robot network` per profile → `gr-trial --all --rating-video` |
| Gaze yaw sign | Video too low-resolution to verify | Check on a higher-resolution participant or with an eye tracker |

## Earlier artifacts

These came from the earlier 8-joint `simulation/head.xml` code. They're kept in git for reference, and the current code doesn't read them:
- `runs/gr-v2`, `runs/gr-v3` and `runs/gr-verify` (scripted teacher, synthetic fixtures)
- `data/gr-v2`, `data/gr-v3`, `data/gr-verify` and `data/gr-synthetic`

The pre-G/R cue and pose outputs (`runs/demo`, `runs/training`, `runs/personaplex-*`, …) are from code removed earlier.
