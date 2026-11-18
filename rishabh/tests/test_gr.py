import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from speech_ai.audio import write_wav
from speech_ai.gr_data import load_stream, prepare_stream, window_at
from speech_ai.gr_fixtures import make_fixtures
from speech_ai.gr_model import CHANNELS, STEPS, capability, constrain_motion, normalized_limits
from speech_ai.gr_train import load_models, train


class GRTests(unittest.TestCase):
    def test_motion_obeys_enabled_joints_range_and_slew(self):
        limits = torch.from_numpy(normalized_limits(capability("fixed-neck-pitch"))[None])
        position = torch.zeros(1, len(CHANNELS))
        position[0, 3] = -0.25
        requested = torch.full((1, STEPS, len(CHANNELS)), 2.0)
        result = constrain_motion(requested, limits, position)
        self.assertTrue((result[:, :, 1] == 0).all())
        self.assertTrue((result <= limits[:, None, :, 2]).all())
        actual_step = torch.diff(torch.cat([position[:, None, :], result], dim=1), dim=1)
        self.assertTrue((actual_step.abs() <= 0.05 * limits[:, None, :, 3] + 1e-6).all())

    def test_stream_alignment_is_causal_and_masks_stale_teacher_motion(self):
        with TemporaryDirectory() as scratch:
            path = Path(scratch)
            write_wav(path / "audio.wav", np.zeros(4800), 16000)
            np.savez(path / "teacher.npz", time_s=[0.06, 0.14], features=[[0.2], [0.7]])
            weights = np.zeros((1, len(CHANNELS)))
            weights[0, 1] = 1
            np.savez(path / "map.npz", weights=weights, bias=np.zeros(len(CHANNELS)))
            np.savez(path / "context.npz", time_s=[0.08, 0.15], features=[[2], [5]])
            prepare_stream(path / "audio.wav", None, False, path / "teacher.npz",
                           path / "context.npz", path / "stream.npz", path / "map.npz")
            _, human, context = load_stream(path / "stream.npz")
            np.testing.assert_allclose(human[:, 1], [0, 0.2, 0.7, 0.7, 0.7, 0])
            np.testing.assert_array_equal(human[:, -1], [0, 1, 1, 1, 1, 0])
            np.testing.assert_array_equal(context[:, 0], [0, 2, 5, 5, 5, 5])
            history, features = window_at(human, context, 2)
            np.testing.assert_allclose(history[-2:, 1], [0, 0.2])
            np.testing.assert_array_equal(features, [2, 1])

    def test_r_guided_training_without_demonstrations_saves_both_models(self):
        with TemporaryDirectory() as scratch:
            path = Path(scratch)
            manifest = make_fixtures(path / "data", 5, 31)
            spec = json.loads(manifest.read_text())
            for trial in spec["trials"]:
                trial["demonstration"] = False
            manifest.write_text(json.dumps(spec))
            report = train(manifest, path / "trained", 17, 1, 1, 0.12, 0.15, residual=True)
            self.assertNotIn("g_imitation", report["trajectory_test"])
            self.assertIsNone(report["trajectory_test"]["g_refined"]["demonstration_mae_rad"])
            self.assertGreater(report["observer_test"]["trials"], 0)
            g, r, metadata = load_models(path / "trained/g.pt", path / "trained/r.pt")
            self.assertTrue(g.residual)
            self.assertEqual(len(metadata["labels"]), 3)
            self.assertIsNotNone(r)
