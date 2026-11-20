"""Fast checks of the DyaPlex building blocks (no data, no training).

Run: .venv/bin/python -m unittest tests.test_dyaplex -v   (from rishabh/; needs third_party/)
"""

import unittest

import numpy as np
import torch

from dyaplex import config as C
from dyaplex import layout
from dyaplex.rotations import axis_angle_to_6d, axis_angle_to_matrix, rotation_6d_to_matrix


def random_codes(frames, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([rng.integers(layout.BAND_LO[layout.SLOT_PART], layout.BAND_HI[layout.SLOT_PART])
                     for _ in range(frames)])


class Layout(unittest.TestCase):
    def test_constants_match_paper(self):
        self.assertEqual(C.K, 22)
        self.assertEqual(C.L_STEP, 46)
        self.assertEqual(C.VOCAB, 4098)
        self.assertEqual([C.PART_DIM[p] for p in C.PARTS], [78, 180, 57, 56])
        self.assertEqual(len(C.UPPER_JOINTS) * 6, 78)
        self.assertEqual(len(C.HAND_JOINTS) * 6, 180)
        self.assertEqual(len(C.LOWER_JOINTS) * 6 + 3, 57)
        self.assertEqual(layout.window_frames(4096), 89)          # 89 frames ~ 7.1 s [P §4.2]
        self.assertEqual(layout.window_frames(1024), 22)          # ~1.8 s [P §5 runtime]

    def test_interleave_round_trip_and_qpos(self):
        a, b = random_codes(5, 1), random_codes(5, 2)
        seq = layout.interleave(a, b)
        self.assertEqual(len(seq), 5 * 46)
        self.assertTrue((seq[::46] == C.TAG_A).all() and (seq[23::46] == C.TAG_B).all())
        a2, b2 = layout.deinterleave(seq)
        self.assertTrue(np.array_equal(a, a2) and np.array_equal(b, b2))
        q = layout.q_pos(len(seq))
        self.assertEqual(q[45], 0)
        self.assertEqual(q[46], 1)

    def test_targets_supervise_body_only(self):
        seq = layout.interleave(random_codes(3), random_codes(3, 4))
        labels, band = layout.targets(seq, "body", "AB")
        self.assertEqual(int((labels >= 0).sum()), 3 * 2 * 18)
        labels_all, _ = layout.targets(seq, "all", "AB")
        self.assertEqual(int((labels_all >= 0).sum()), 3 * 2 * 22)
        labels_b, _ = layout.targets(seq, "body", "B")
        self.assertEqual(int((labels_b >= 0).sum()), 3 * 18)
        sup = labels >= 0
        lo = layout.BAND_LO[band[sup]]
        self.assertTrue(((labels[sup] >= lo) & (labels[sup] < lo + 1024)).all())

    def test_band_check(self):
        codes = random_codes(4)
        self.assertTrue(layout.in_band(codes))
        codes[0, 0] = 2000                       # upper slot holding a hands-band id
        self.assertFalse(layout.in_band(codes))


class Rotations(unittest.TestCase):
    def test_6d_round_trip(self):
        aa = np.random.default_rng(0).normal(size=(500, 3))
        self.assertLess(np.abs(rotation_6d_to_matrix(axis_angle_to_6d(aa)) - axis_angle_to_matrix(aa)).max(), 1e-9)


class TowerCausality(unittest.TestCase):
    def test_future_inputs_do_not_change_past_logits(self):
        from dyaplex.tower import MotionTower
        cfg = C.TINY["tower"]
        torch.manual_seed(0)
        model = MotionTower(cfg).eval()
        frames = 6
        seq = torch.tensor(layout.interleave(random_codes(frames), random_codes(frames, 3))[None, :-1])
        pos = torch.arange(seq.shape[1])
        qf = pos // C.L_STEP
        s_pos = torch.arange(frames)
        speech = [torch.randn(1, frames, cfg.speech_dim) for _ in range(cfg.layers)]
        with torch.no_grad():
            base, _ = model(seq, pos, qf, model.project_speech(speech, s_pos), s_pos)
            speech2 = [h.clone() for h in speech]
            for h in speech2:
                h[:, 3:] = torch.randn_like(h[:, 3:])
            seq2 = seq.clone()
            seq2[0, qf >= 3] = torch.where(seq2[0, qf >= 3] < 4096, (seq2[0, qf >= 3] + 1) % 4096, seq2[0, qf >= 3])
            alt, _ = model(seq2, pos, qf, model.project_speech(speech2, s_pos), s_pos)
        self.assertTrue(torch.equal(base[0, qf < 3], alt[0, qf < 3]))
        self.assertFalse(torch.equal(base[0, qf >= 3], alt[0, qf >= 3]))


class CodecCausality(unittest.TestCase):
    def test_encoder_has_no_lookahead(self):
        from dyaplex.codec import PartCodec
        torch.manual_seed(0)
        model = PartCodec("upper", C.REFERENCE["codec"])
        model.train()
        model(torch.randn(4, 64, 78))           # initialize EMA codebooks
        model.eval()
        x = torch.randn(1, 64, 78)
        y = x.clone()
        y[:, 32:] = torch.randn_like(y[:, 32:])
        self.assertTrue(torch.equal(model.latent(x)[..., :16], model.latent(y)[..., :16]))
        self.assertEqual(tuple(model.encode(x).shape), (1, 32, 6))


if __name__ == "__main__":
    unittest.main()
