import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from speech_ai.audio import aligned_embeddings, extract, load_tracks
from speech_ai.dataset import match_events
from speech_ai.fixtures import make_corpus
from speech_ai.personaplex_offline import capture_states
from speech_ai.pose import pitch_excursions
from speech_ai.training import checkpoint_scores, train


class AlignmentTests(unittest.TestCase):
    def test_speech_state_lookup_never_reads_a_future_frame(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "frames.npz"
            np.savez(path, time_s=[0.08, 0.24], features=[[1.0], [2.0]])
            aligned = aligned_embeddings(path, np.array([0.05, 0.10, 0.239, 0.24, 0.60]))
        np.testing.assert_array_equal(aligned, [[0, 0], [1, 1], [1, 1], [2, 1], [0, 0]])

    def test_event_matching_counts_each_annotation_once(self):
        matches = match_events(np.array([0.9, 1.0, 1.1, 3]), np.array([1.0, 2.9]), 0.1)
        self.assertEqual(matches, [(0, 0), (3, 1)])

    def test_pose_excursion_requires_a_return_and_stable_yaw(self):
        seconds = np.arange(0, 2.5, 0.1)
        pitch = 12 * np.exp(-((seconds - 1.2) / 0.18) ** 2)
        track = np.column_stack([seconds, pitch, np.zeros_like(pitch), np.zeros_like(pitch)])
        self.assertEqual(len(pitch_excursions(track, 5)), 1)
        track[:, 2] = seconds * 40
        self.assertEqual(pitch_excursions(track, 5), [])

    def test_personaplex_capture_skips_pre_input_state(self):
        class Generator:
            _frame_size = 1920
            _sample_rate = 24000

            def process_transformer_output(self, state):
                return state.sum().item()

            def step(self, state):
                return self.process_transformer_output(state)

        offline = SimpleNamespace(LMGen=Generator, lm_load_audio=lambda: "audio")
        original = Generator.step
        generator = Generator()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "states.npz"
            with capture_states(offline, path):
                generator.step(torch.zeros(1, 1, 2))
                offline.lm_load_audio()
                self.assertEqual(generator.step(torch.tensor([[[1.0, 2.0]]])), 3)
                generator.step(torch.tensor([[[3.0, 4.0]]]))
            with np.load(path) as result:
                np.testing.assert_array_equal(result["time_s"], [0.08, 0.16])
                np.testing.assert_array_equal(result["features"], [[1, 2], [3, 4]])
        self.assertIs(Generator.step, original)


class TrainingTests(unittest.TestCase):
    def test_group_splits_and_embedding_checkpoint_round_trip(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_corpus(root / "data", groups=6, seed=12)
            rows = json.loads(manifest.read_text())
            states_path = manifest.parent / "states.npz"
            times = np.arange(1, 151) * 0.08
            np.savez(states_path, time_s=times, features=np.column_stack([times, np.sin(times)]))
            for row in rows:
                row["embeddings"] = states_path.name
            manifest.write_text(json.dumps(rows))
            output = root / "run"
            report = train(manifest, output, epochs=1, seed=17, tolerance=0.3,
                           vad_db=-40, use_embeddings=True, mode="cadence")
            self.assertEqual(len(report["models"]), 4)
            groups = [set(split["groups"]) for split in report["splits"].values()]
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            features = extract(*load_tracks(manifest.parent / rows[0]["partner"],
                                            manifest.parent / rows[0]["agent"]))
            selected, scores, threshold = checkpoint_scores(
                output / "gru-cadence-embeddings.pt", features, states_path, -40)
            self.assertGreater(len(selected.candidates), len(features.candidates))
            self.assertEqual(scores.shape, selected.candidates.shape)
            self.assertTrue(np.all((scores >= 0) & (scores <= 1)))
            self.assertTrue(0 < threshold < 1)
