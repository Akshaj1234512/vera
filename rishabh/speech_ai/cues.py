"""Choose feasible cue events from causal pause and speech-state estimates."""

from dataclasses import asdict, dataclass

import numpy as np

from .audio import FRAME_S, Features

REFRACTORY_S = 1.3
ACK_DURATION_S = 0.65
RULE_THRESHOLD = 0.5


@dataclass(frozen=True)
class Capabilities:
    nod: bool
    eyes: bool


PROFILES = {
    "linked-eyes": Capabilities(nod=True, eyes=True),
    "fixed-eyes": Capabilities(nod=True, eyes=False),
    "nod-busy": Capabilities(nod=False, eyes=True),
    "no-social-motion": Capabilities(nod=False, eyes=False),
}


@dataclass
class Cue:
    role: str
    cue: str
    target: dict[str, float]
    onset_s: float
    duration_s: float
    reason: str


def quiet_agent(features: Features, index: int) -> bool:
    return not features.values[max(0, index - 4):index + 1, 7].any()


def rule_scores(features: Features) -> np.ndarray:
    return np.asarray([
        float(quiet_agent(features, index))
        for index in features.candidates
    ])


def selected_candidates(features: Features, scores: np.ndarray, threshold: float) -> np.ndarray:
    chosen = []
    previous = -np.inf
    for candidate, score in zip(features.candidates, scores, strict=True):
        current = features.time_s[candidate]
        if score >= threshold and current - previous >= REFRACTORY_S and quiet_agent(features, candidate):
            chosen.append(candidate)
            previous = current
    return np.asarray(chosen, dtype=np.int64)


def speech_state(features: Features) -> np.ndarray:
    return np.asarray(("silence", "listening", "speaking", "overlap"))[
        features.values[:, 3].astype(int) + 2 * features.values[:, 7].astype(int)
    ]


def speech_cessations(features: Features) -> list[dict]:
    """Find sustained outgoing-audio cessations following detected overlap."""
    partner, agent = features.values[:, [3, 7]].T.astype(bool)
    stable_silence = 3
    last_overlap = -np.inf
    overlap_onset = None
    agent_had_floor = False
    quiet_frames = 0
    cessations = []
    for index, (partner_active, agent_active) in enumerate(zip(partner, agent)):
        now = features.time_s[index]
        if agent_active:
            quiet_frames = 0
            if partner_active:
                if now - last_overlap > 0.25 or overlap_onset is None:
                    overlap_onset = round(float(now - FRAME_S), 6)
                    before = slice(max(0, index - 12), index)
                    agent_had_floor = bool(np.count_nonzero(agent[before] & ~partner[before]) >= 4)
                last_overlap = now
            continue
        quiet_frames += 1
        if quiet_frames != stable_silence:
            continue
        cessation = round(float(now - stable_silence * FRAME_S), 6)
        if overlap_onset is not None and cessation - last_overlap <= 0.35:
            cessations.append({
                "overlap_onset_s": overlap_onset,
                "agent_had_floor": agent_had_floor,
                "agent_cessation_s": cessation,
                "detected_at_s": float(now),
                "delta_s": round(cessation - overlap_onset, 6),
            })
            overlap_onset = None
    return cessations


def jaw_aperture(features: Features) -> np.ndarray:
    audio = features.values[:, 4]
    target = np.clip((audio + 45) / 36, 0, 1) * features.values[:, 7]
    aperture = np.zeros_like(target)
    last = 0.0
    for index, value in enumerate(target):
        coefficient = 0.75 if value > last else 0.6
        last += coefficient * (value - last)
        aperture[index] = last
    return aperture


def build_timeline(features: Features, scores: np.ndarray, threshold: float, profile: str,
                   source: str, policy: str) -> dict:
    capabilities = PROFILES[profile]
    states = speech_state(features)
    chosen = set(selected_candidates(features, scores, threshold).tolist())
    events, decisions = [], []
    for index, speech_end, score in zip(features.candidates, features.speech_end_s, scores, strict=True):
        now = float(features.time_s[index])
        if index not in chosen:
            reason = "agent recently speaking" if not quiet_agent(features, index) else (
                "below decision threshold" if score < threshold else "acknowledgment cooldown")
            cue = None
        elif capabilities.nod:
            cue, reason = "nod", "neck pitch available"
            events.append(Cue("acknowledge", cue, {"pitch": 0.45}, now, ACK_DURATION_S, reason))
        elif capabilities.eyes:
            cue, reason = "gaze", "nod unavailable, trying a linked-eye cue"
            events.append(Cue("acknowledge", cue, {"yaw": 0.42}, now, 0.5, reason))
        else:
            cue, reason = None, "no acknowledgment motion available"
        decisions.append({"time_s": now, "partner_speech_end_s": float(speech_end),
                          "score": float(score), "cue": cue, "reason": reason})

    cessations = speech_cessations(features)
    for item in cessations:
        if item["agent_had_floor"]:
            events.append(Cue("yield", "settle", {}, item["detected_at_s"], 0.2,
                              "agent held the floor before overlap and outgoing audio then ceased"))
    events.sort(key=lambda event: event.onset_s)
    partner_db, agent_db = features.values[:, [0, 4]].T
    return {
        "version": 1,
        "source": source,
        "policy": policy,
        "profile": profile,
        "frame_s": FRAME_S,
        "time_s": features.time_s.tolist(),
        "feature_db": {"partner": partner_db.tolist(), "agent": agent_db.tolist()},
        "speech_state": states.tolist(),
        "jaw_aperture": jaw_aperture(features).tolist(),
        "events": [asdict(event) for event in events],
        "ack_decisions": decisions,
        "overlap_cessations": cessations,
    }
