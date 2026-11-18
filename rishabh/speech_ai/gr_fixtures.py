"""Synthetic stand-ins that exercise the full G/R path. None of this is human evidence.

* ``make_fixtures`` writes a manifest of 500 ms trials with programmatic votes.
* ``listener_teacher`` writes a *scripted* teacher track in the DyaPlex motion
  contract from real conversation audio, for simulator demos only. It is not a
  DyaPlex export and must never be reported as one.
"""

import json
from pathlib import Path

import numpy as np
import torch

from .audio import FRAME_S, extract, load_tracks, normalized
from .gr_model import CHANNELS, PROFILES, SCALE_RAD, STEPS, capability, constrain_motion, normalized_limits

LABELS = ("unclear", "neutral_ack", "enthusiastic_ack")
LEVELS = (0.0, 0.44, 0.78)
ORIGIN = "synthetic_gr_fixture"
TEACHER_ORIGIN = "synthetic_listener_script_not_dyaplex"
NECK, JAW, EYE, LID_L, LID_R, BROW_L, BROW_R = 1, 2, 3, 4, 5, 6, 7


def pulse(length: int, shift: int = 0, quick: bool = False) -> np.ndarray:
    phase = np.clip((np.arange(1, length + 1) - shift) / length, 0, 1)
    if quick:
        phase = phase ** 0.72
    return np.sin(np.pi * phase) ** 2


def audio_context(rng, speaking: bool, extra: int) -> np.ndarray:
    """A plausible normalized audio-feature vector (see audio.FEATURE_NAMES)."""
    talker = [rng.uniform(0.6, 0.8), rng.uniform(0.3, 0.6), rng.uniform(0.6, 0.95), 1.0]
    quiet = [rng.uniform(0.0, 0.1), 0.0, 0.0, 0.0]
    values = quiet + talker if speaking else talker + quiet
    if extra:
        values += list(rng.normal(0, 0.5, extra)) + [1.0]
    return np.asarray(values, dtype=np.float32)


def make_fixtures(out: Path, groups: int, seed: int, state_dim: int = 0) -> Path:
    if groups < 5:
        raise ValueError("Use at least five fixture groups")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []

    def teacher(intensity, shift, jaw, blink):
        human = np.zeros((STEPS, len(CHANNELS)), dtype=np.float32)
        shape = pulse(STEPS, shift)
        human[:, NECK] = intensity * shape
        human[:, BROW_L:] = intensity * 0.1 * shape[:, None]
        human[:, JAW] = jaw
        human[:, LID_L:LID_R + 1] = blink[:, None]
        return human

    def movement(label, profile, amplitude, jaw, blink, state):
        motion = np.zeros((STEPS, len(CHANNELS)), dtype=np.float32)
        shape = np.roll(pulse(STEPS, quick=label == 2), 1)
        shape[0] = 0
        if profile == "fixed-neck-pitch":
            motion[:, EYE] = amplitude * shape * 0.9
            motion[:, BROW_L:] = (amplitude * (0.52 if label == 2 else 0.22) * shape)[:, None]
        else:
            motion[:, NECK] = amplitude * shape
            motion[:, BROW_L:] = (amplitude * (0.36 if label == 2 else 0.07) * shape)[:, None]
        motion[:, JAW] = jaw
        motion[:, LID_L:LID_R + 1] = blink[:, None]
        decay = np.exp(-np.arange(1, STEPS + 1) / 3)[:, None]
        motion += state[0] / SCALE_RAD * decay
        limits = torch.from_numpy(normalized_limits(capability(profile))[None])
        realized = constrain_motion(torch.from_numpy(motion[None]), limits,
                                    torch.from_numpy((state[0] / SCALE_RAD)[None]))
        return realized[0].numpy() * SCALE_RAD

    def votes_for(observed, clarity, raters):
        probabilities = np.full(len(LABELS), (1 - clarity) / (len(LABELS) - 1))
        probabilities[observed] = clarity
        counts = rng.multinomial(raters, probabilities)
        return {LABELS[k]: int(n) for k, n in enumerate(counts) if n}

    for group in range(groups):
        scale = rng.uniform(0.88, 1.08)
        for take in range(2):
            levels = [0, rng.uniform(0.34, 0.49) * scale, rng.uniform(0.70, 0.82) * scale]
            for intent, amplitude in enumerate(levels):
                speaking = rng.random() < 0.3
                shift = int(rng.integers(-2, 3))
                jaw = (np.convolve(rng.uniform(0, 0.7, STEPS + 2), np.ones(3) / 3, "valid")
                       if speaking else np.zeros(STEPS)).astype(np.float32)
                blink = np.zeros(STEPS, dtype=np.float32)
                if rng.random() < 0.3:
                    at = int(rng.integers(1, STEPS - 3))
                    blink[at:at + 3] = (0.5, 1.0, 0.5)
                human = teacher(amplitude, shift, jaw, blink)
                context = audio_context(rng, speaking, state_dim)
                context_id = f"{group:02d}-{take}-{intent}"
                for profile in PROFILES:
                    limits = capability(profile)
                    state = np.zeros((2, len(CHANNELS)), dtype=np.float32)
                    if rng.random() < 0.5:
                        state[0] = rng.uniform(0, 0.25, len(CHANNELS)) * limits[:, 2] * limits[:, 0]
                    for observed in range(len(LABELS)):
                        trial_id = f"{context_id}-{profile}-{observed}"
                        name = f"{trial_id}.npz"
                        robot = movement(observed, profile, levels[observed], jaw, blink, state)
                        np.savez_compressed(out / name, human=human, human_valid=np.ones(STEPS),
                                            context=context, state_rad=state, robot_rad=robot)
                        case = ("clear" if observed == intent else "ambiguous" if observed == 0
                                else "wrong_intent")
                        clarity = 0.9 if observed else 0.7
                        rows.append({"id": trial_id, "group": f"synthetic-{group:02d}", "file": name,
                                     "origin": ORIGIN, "profile": profile, "pair": context_id,
                                     "case": case, "intent": LABELS[intent],
                                     "votes": votes_for(observed, clarity, int(rng.integers(5, 10))),
                                     "demonstration": observed == intent})
    path = out / "manifest.json"
    path.write_text(json.dumps({"labels": LABELS, "synthetic": True, "trials": rows}, indent=2) + "\n")

    human, context = [], []
    for k, label in enumerate((0, 1, 0, 2, 0, 1, 0, 2, 0)):
        human.append(teacher(LEVELS[label], 0, np.zeros(STEPS), np.zeros(STEPS)))
        feature = audio_context(np.random.default_rng(k), False, state_dim)
        context.append(np.repeat(feature[None], STEPS, axis=0))
    human = np.concatenate(human)
    np.savez_compressed(out / "replay.npz", time_s=np.arange(1, len(human) + 1) * FRAME_S,
                        human=human, human_valid=np.ones(len(human)), context=np.concatenate(context))
    return path


def listener_teacher(partner: Path, agent: Path | None, stereo: bool, out: Path, seed: int = 3) -> dict:
    """Scripted listener/speaker motion from audio, written in the teacher NPZ contract.

    Rules: nod after each partner phrase that ends while the agent is quiet (larger,
    faster nod when the phrase ended with rising pitch); jaw follows the agent's
    smoothed loudness; periodic blinks. Stand-in until real DyaPlex exports exist.
    """
    audio = extract(*load_tracks(partner, agent, stereo))
    values = normalized(audio.values)
    frames = len(audio.time_s)
    rng = np.random.default_rng(seed)
    human = np.zeros((frames, len(CHANNELS)), dtype=np.float32)
    partner_active, agent_active = values[:, 3] > 0, values[:, 7] > 0
    events = []
    run = 0
    for index in range(1, frames):
        run = run + 1 if partner_active[index - 1] else 0
        ended = partner_active[index - 1] and not partner_active[index]
        if ended and run >= 8 and not agent_active[index] and (not events or index - events[-1][0] >= 20):
            phrase = values[max(0, index - run):index, 1]
            voiced = phrase[phrase > 0]
            rising = len(voiced) >= 6 and voiced[-3:].mean() > voiced[:-3].mean() * 1.08
            events.append((index, LABELS[2] if rising else LABELS[1]))
    # Backchannel nods during long partner turns, and beat gestures when the agent
    # starts a phrase (larger when it opens with raised pitch).
    silence = 0
    pitch_median = np.median(values[agent_active, 5][values[agent_active, 5] > 0]) if agent_active.any() else 0
    for index in range(frames):
        busy = [i for i, _ in events if abs(i - index) < 30]
        if partner_active[index] and index >= 30 and partner_active[index - 30:index].all() and not busy:
            events.append((index, LABELS[1]))
        elif agent_active[index] and silence >= 2 and not busy:
            opening = values[index:index + 6, 5]
            voiced = opening[opening > 0]
            lively = len(voiced) >= 2 and voiced.mean() > pitch_median * 1.1
            events.append((index, LABELS[2] if lively else LABELS[1]))
        silence = 0 if agent_active[index] else silence + 1
    events.sort()
    for index, label in events:
        amplitude = LEVELS[LABELS.index(label)]
        shape = pulse(STEPS)
        end = min(frames, index + STEPS)
        human[index:end, NECK] = amplitude * shape[:end - index]
        human[index:end, BROW_L:] = amplitude * 0.1 * shape[:end - index, None]
    loud = np.where(agent_active, values[:, 4], 0)
    jaw = np.convolve(loud, np.ones(3) / 3, "same")
    human[:, JAW] = np.clip((jaw - 0.35) * 2.0, 0, 0.7)
    blink = int(rng.integers(40, 80))
    while blink + 3 < frames:
        human[blink:blink + 3, LID_L:LID_R + 1] = np.array([0.5, 1.0, 0.5])[:, None]
        blink += int(rng.integers(50, 90))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, time_s=audio.time_s, channels=np.array(CHANNELS), human=human,
                        human_valid=np.ones(frames), source=np.array(TEACHER_ORIGIN))
    report = {"origin": TEACHER_ORIGIN, "frames": frames,
              "gestures": [{"time_s": round(float(audio.time_s[i]), 2), "intent": label} for i, label in events]}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report
