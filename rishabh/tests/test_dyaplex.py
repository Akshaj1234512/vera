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


class ForwardKinematics(unittest.TestCase):
    def test_identity_and_root_rotation(self):
        from dyaplex.fk import Skeleton, forward
        parents = np.array([-1, 0, 1, 2])
        rest = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [1, 2, 0]], dtype=float)
        sk = Skeleton(rest, parents, "synthetic")
        ident = np.tile(axis_angle_to_6d(np.zeros(3)), (1, 4, 1))
        self.assertTrue(np.allclose(forward(ident, sk, 4)[0], rest))
        rot = ident.copy()
        rot[0, 0] = axis_angle_to_6d(np.array([0, 0, np.pi / 2]))     # root 90 deg about z
        pos = forward(rot, sk, 4)[0]
        self.assertTrue(np.allclose(pos[1], [-1, 0, 0], atol=1e-9))
        self.assertTrue(np.allclose(pos[3], [-2, 1, 0], atol=1e-9))


class Metrics(unittest.TestCase):
    def test_frechet_properties(self):
        from dyaplex.metrics import frechet_distance, paired_fd, delta_user, derangement
        rng = np.random.default_rng(0)
        x = rng.normal(size=(4000, 6))
        self.assertLess(abs(frechet_distance(x, x)), 1e-6)
        shifted = frechet_distance(x, x + 1.0)
        self.assertAlmostEqual(shifted, 6.0, delta=0.05)               # ||mu1-mu2||^2 = 6
        a, b = rng.normal(size=(4000, 66)), rng.normal(size=(4000, 66))
        self.assertLess(abs(paired_fd(a, b, b)), 1e-6)
        self.assertAlmostEqual(delta_user(13.0, 10.0), 30.0)
        perm = derangement(10, 1)
        self.assertFalse((perm == np.arange(10)).any())

    def test_diversity_and_beat_align(self):
        from dyaplex.metrics import beat_align, diversity, gahr, mean_speed
        rng = np.random.default_rng(1)
        self.assertGreater(diversity(rng.normal(size=(20000, 66))), 0)
        self.assertAlmostEqual(gahr(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 0.3), 1.0)
        fps, sr, seconds = 25.0, 24000, 12
        t = np.arange(int(fps * seconds)) / fps
        joints = np.zeros((len(t), 22, 3))
        joints[..., 0] = np.sin(2 * np.pi * 0.5 * t)[:, None]           # every joint swings at 0.5 Hz
        audio = 0.001 * rng.standard_normal(sr * seconds).astype(np.float32)
        for k in range(1, seconds * 2):                                 # clicks every 0.5 s
            audio[int(k * 0.5 * sr): int(k * 0.5 * sr) + 200] += 0.8
        score = beat_align(joints, audio, sr, fps, mean_speed([joints], fps))
        self.assertTrue(0.0 <= score <= 1.0)


class Ablations(unittest.TestCase):
    def _logits(self, model, a, b, speech, frames):
        seq = torch.tensor(layout.interleave(a, b)[None, :-1])
        pos = torch.arange(seq.shape[1])
        s_pos = torch.arange(frames)
        with torch.no_grad():
            out, _ = model(seq, pos, pos // C.L_STEP, model.project_speech(speech, s_pos), s_pos)
        speaker, _, _ = layout.token_role(seq.shape[1])
        return out[0, torch.tensor(speaker == 1)]

    def test_no_partner_blocks_partner_information(self):
        from dyaplex.tower import MotionTower
        cfg = C.TINY["tower"]
        frames = 4
        speech = [torch.randn(1, frames, cfg.speech_dim) for _ in range(cfg.layers)]
        a1, a2, b = random_codes(frames, 1), random_codes(frames, 2), random_codes(frames, 3)
        torch.manual_seed(0)
        full = MotionTower(cfg).eval()
        self.assertFalse(torch.equal(self._logits(full, a1, b, speech, frames), self._logits(full, a2, b, speech, frames)))
        torch.manual_seed(0)
        blind = MotionTower(cfg).ablate("no_partner").eval()
        self.assertTrue(torch.equal(self._logits(blind, a1, b, speech, frames), self._logits(blind, a2, b, speech, frames)))

    def test_ablation_names(self):
        from dyaplex.tower import MotionTower
        with self.assertRaises(ValueError):
            MotionTower(C.TINY["tower"]).ablate("nope")


class TrainingData(unittest.TestCase):
    def test_fast_teacher_forced_states_match_forward_train(self):
        from dyaplex import speech as S
        lm = S.build_lm(C.TINY["speech"])
        rng = np.random.default_rng(0)
        st = S.build_streams(rng.integers(0, 2048, (8, 12)), rng.integers(0, 2048, (8, 12)), np.full(12, 3),
                             rng.integers(0, 2048, (8, 5)), [5, 6], C.TINY["speech"])
        slow = S.teacher_forced_states(lm, st)
        fast = S.teacher_forced_fast(lm, torch.tensor(st.tokens[None]))
        self.assertTrue(torch.equal(slow, fast))

    def test_invalid_frames_are_not_supervised(self):
        from dyaplex.tower_data import mask_invalid
        frames = 3
        seq = layout.interleave(random_codes(frames), random_codes(frames, 5))
        labels, _ = layout.targets(seq, "all", "AB")
        ok = np.ones(frames, bool)
        bad_body = ok.copy()
        bad_body[1] = False
        info_a = {"body": ok, "face": ok, "vad": np.zeros(frames, int)}
        info_b = {"body": bad_body, "face": ok, "vad": np.zeros(frames, int)}
        masked = mask_invalid(labels, info_a, info_b)
        # B's 18 body codes in frame 1 are dropped; its 4 face codes and all of A's stay
        self.assertEqual(int((labels >= 0).sum()) - int((masked >= 0).sum()), C.BODY_CODES)
