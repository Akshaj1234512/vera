import unittest

import numpy as np

from speech_ai.audio import extract
from speech_ai.cues import build_timeline, rule_scores
from speech_ai.fixtures import RATE, add_voice


class CueTests(unittest.TestCase):
    @staticmethod
    def timeline(partner, agent=None, profile="linked-eyes"):
        inputs = (partner, RATE), (agent, RATE) if agent is not None else None
        features = extract(*inputs)
        return build_timeline(features, rule_scores(features), 0.5, profile, "test", "rules")

    def test_pause_detection_has_no_lookahead(self):
        audio = np.zeros(3 * RATE, dtype=np.float32)
        add_voice(audio, 0.4, 1.1, 170)
        before = self.timeline(audio[:2 * RATE])
        add_voice(audio, 2.0, 2.6, 200)
        after = self.timeline(audio)
        self.assertEqual(before["events"], after["events"][:1])
        self.assertEqual(before["events"][0]["onset_s"], 1.25)

    def test_replay_respects_capability_profiles(self):
        audio = np.zeros(2 * RATE, dtype=np.float32)
        add_voice(audio, 0.4, 1.1, 170)
        for profile, expected in [("linked-eyes", "nod"), ("fixed-eyes", "nod"),
                                  ("nod-busy", "gaze"), ("no-social-motion", None)]:
            with self.subTest(profile=profile):
                timeline = self.timeline(audio, profile=profile)
                self.assertEqual(timeline["ack_decisions"][0]["cue"], expected)

    def test_pauses_during_agent_speech_are_suppressed(self):
        partner, agent = np.zeros((2, 2 * RATE), dtype=np.float32)
        add_voice(partner, 0.3, 1.1, 170)
        add_voice(agent, 0.8, 1.55, 180)
        timeline = self.timeline(partner, agent)
        self.assertIsNone(timeline["ack_decisions"][0]["cue"])
        self.assertGreater(max(timeline["jaw_aperture"]), 0.4)

    def test_a_backchannel_end_is_not_a_yield(self):
        partner, agent = np.zeros((2, 3 * RATE), dtype=np.float32)
        add_voice(partner, 0.3, 2.2, 170)
        add_voice(agent, 0.8, 1.2, 180)
        timeline = self.timeline(partner, agent)
        self.assertEqual(len(timeline["overlap_cessations"]), 1)
        self.assertFalse(timeline["overlap_cessations"][0]["agent_had_floor"])
        self.assertNotIn("yield", [event["role"] for event in timeline["events"]])

    def test_a_yield_uses_the_detection_time(self):
        partner, agent = np.zeros((2, 3 * RATE), dtype=np.float32)
        add_voice(agent, 0.2, 1.45, 180)
        add_voice(partner, 1.1, 2.2, 170)
        timeline = self.timeline(partner, agent)
        stop = timeline["overlap_cessations"][0]
        event = next(event for event in timeline["events"] if event["role"] == "yield")
        self.assertTrue(stop["agent_had_floor"])
        self.assertEqual(event["onset_s"], stop["detected_at_s"])
        self.assertGreater(event["onset_s"], stop["agent_cessation_s"])


if __name__ == "__main__":
    unittest.main()
