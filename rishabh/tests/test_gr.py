import csv
import importlib.util
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from speech_ai.audio import FRAME_S, write_wav
from speech_ai.gr_data import TimedSource, align, fit_teacher_map, load_manifest, load_stream, pack_recording, \
    prepare_stream, split_groups, window_at
from speech_ai.gr_eval import evaluate, merge_votes, rating_packet
from speech_ai.gr_fixtures import LABELS, make_fixtures
from speech_ai.gr_model import CHANNELS, SCALE_RAD, STEPS, Generator, Observer, capability, constrain_motion, \
    normalized_limits
from speech_ai.gr_runtime import (FileFeed, KinematicRobot, LiveInputs, ProjectedPolicy, Session, StreamInputs,
                                  motion_metrics, run_session, write_session)
from speech_ai.gr_train import batch, load_models, smoothness, train

NECK = CHANNELS.index("neck_pitch")


def limits_tensor(profile="full"):
    return torch.from_numpy(normalized_limits(capability(profile))[None])


class ModelGTests(unittest.TestCase):
    def test_motion_obeys_enabled_joints_range_and_slew(self):
        limits = limits_tensor("fixed-neck-pitch")
        position = torch.zeros(1, len(CHANNELS))
        position[0, 3] = -0.25
        requested = torch.full((1, STEPS, len(CHANNELS)), 2.0)
        result = constrain_motion(requested, limits, position)
        self.assertTrue((result[:, :, NECK] == 0).all())
        self.assertTrue((result <= limits[:, None, :, 2]).all())
        actual_step = torch.diff(torch.cat([position[:, None, :], result], dim=1), dim=1)
        self.assertTrue((actual_step.abs() <= FRAME_S * limits[:, None, :, 3] + 1e-6).all())

    def test_g_shapes_and_capability_bounds_for_direct_and_residual(self):
        torch.manual_seed(0)
        for residual in (False, True):
            g = Generator(context_dim=10, residual=residual)
            with torch.no_grad():
                g.output.weight.normal_(0, 1.0)  # force large raw targets
            human = torch.cat([torch.rand(4, STEPS, 8) * 2 - 1, torch.ones(4, STEPS, 1)], dim=2)
            state = torch.zeros(4, 2, 8)
            state[:, 0, 0] = 0.5
            limits = limits_tensor("fixed-neck-pitch").expand(4, -1, -1)
            out = g(human, torch.randn(4, 10), limits, state)
            self.assertEqual(out.shape, (4, STEPS, 8))  # M_r in R^{10x8}
            self.assertTrue((out[:, :, NECK] == 0).all())
            self.assertTrue((out >= limits[:, None, :, 1] - 1e-6).all() and (out <= limits[:, None, :, 2] + 1e-6).all())
            steps = torch.diff(torch.cat([state[:, :1], out], dim=1), dim=1)
            self.assertTrue((steps.abs() <= FRAME_S * limits[:, None, :, 3] + 1e-6).all())

    def test_residual_is_projected_dyaplex_plus_learned_adjustment(self):
        g = Generator(context_dim=3, residual=True)
        with torch.no_grad():
            g.output.weight.zero_()
            g.output.bias.zero_()
        human = torch.cat([torch.full((1, STEPS, 8), 0.3), torch.ones(1, STEPS, 1)], dim=2)
        state = torch.zeros(1, 2, 8)
        limits = limits_tensor()
        expected = constrain_motion(human[:, :, :-1], limits, state[:, 0])
        torch.testing.assert_close(g(human, torch.zeros(1, 3), limits, state), expected)


class AlignmentTests(unittest.TestCase):
    def test_stream_alignment_is_causal_and_masks_stale_teacher_motion(self):
        with TemporaryDirectory() as scratch:
            path = Path(scratch)
            write_wav(path / "audio.wav", np.zeros(4800), 16000)
            np.savez(path / "teacher.npz", time_s=[0.06, 0.14], features=[[0.2], [0.7]])
            weights = np.zeros((1, len(CHANNELS)))
            weights[0, NECK] = 1
            np.savez(path / "map.npz", weights=weights, bias=np.zeros(len(CHANNELS)))
            np.savez(path / "states.npz", time_s=[0.08, 0.15], features=[[2], [5]])
            report = prepare_stream(path / "audio.wav", None, False, path / "teacher.npz",
                                    path / "states.npz", path / "stream.npz", path / "map.npz")
            self.assertEqual(report["context_dim"], 8 + 2)
            self.assertEqual(report["personaplex"]["feature_width"], 1)
            _, human, context = load_stream(path / "stream.npz")
            np.testing.assert_allclose(human[:, NECK], [0, 0.2, 0.7, 0.7, 0.7, 0])
            np.testing.assert_array_equal(human[:, -1], [0, 1, 1, 1, 1, 0])
            np.testing.assert_array_equal(context[:, 8], [0, 2, 5, 5, 5, 5])
            np.testing.assert_array_equal(context[:, 9], [0, 1, 1, 1, 1, 1])
            history, features = window_at(human, context, 2)
            np.testing.assert_allclose(history[-2:, NECK], [0, 0.2])
            np.testing.assert_array_equal(features[8:], [2, 1])

    def test_arrival_time_and_clock_offset(self):
        source = TimedSource(np.array([0.10, 0.20]), np.array([0.12, 0.60]), np.array([[1.0], [2.0]]),
                             np.array([True, True]))
        query = np.array([0.10, 0.20, 0.25])
        values, observed, _ = align(source, query, 0.15, deadline=0.5)
        # Sample 2 arrives at 0.60, after the 0.5 s boundary, so frame 0.20 still sees sample 1.
        np.testing.assert_array_equal(values[:, 0], [1, 1, 1])
        values, observed, _ = align(source, query, 0.15, deadline=0.6)
        np.testing.assert_array_equal(values[:, 0], [1, 2, 2])
        values, observed, _ = align(source, query, 0.15)  # deadline = frame time
        np.testing.assert_array_equal(observed, [False, True, True])
        with TemporaryDirectory() as scratch:
            path = Path(scratch)
            write_wav(path / "audio.wav", np.zeros(16000), 16000)
            times = np.arange(1, 21) * FRAME_S
            human = np.zeros((20, 8))
            human[:, NECK] = np.linspace(0, 0.9, 20)
            np.savez(path / "t.npz", time_s=times + 1.0, channels=np.array(CHANNELS), human=human)
            prepare_stream(path / "audio.wav", None, False, path / "t.npz", None, path / "s.npz",
                           teacher_offset_s=-1.0)
            _, aligned, context = load_stream(path / "s.npz")
            np.testing.assert_allclose(aligned[:, NECK], human[:, NECK], atol=1e-6)
            self.assertEqual(context.shape[1], 8)

    def test_live_inputs_match_offline_alignment(self):
        with TemporaryDirectory() as scratch:
            path = Path(scratch)
            rng = np.random.default_rng(1)
            write_wav(path / "audio.wav", rng.normal(0, 0.1, 32000), 16000)
            times = np.arange(1, 41) * FRAME_S - 0.01
            arrival = times + rng.uniform(0, 0.2, 40)
            arrival = np.maximum.accumulate(arrival)
            human = rng.uniform(-1, 1, (40, 8))
            np.savez(path / "t.npz", time_s=times, arrival_s=arrival, channels=np.array(CHANNELS), human=human)
            prepare_stream(path / "audio.wav", None, False, path / "t.npz", None, path / "s.npz")
            _, offline, offline_context = load_stream(path / "s.npz")
            from speech_ai.audio import extract, load_tracks
            from speech_ai.gr_data import load_teacher
            features = extract(*load_tracks(path / "audio.wav"))
            inputs = LiveInputs(len(features.time_s), None)
            feed = FileFeed(inputs, {"teacher": load_teacher(path / "t.npz", None),
                                     "audio": TimedSource(features.time_s, features.time_s, features.values,
                                                          np.ones(len(features.time_s), bool))})
            for index in range(0, len(features.time_s) + 1, STEPS):
                feed.deliver(index * FRAME_S)
                inputs.window(index)
            np.testing.assert_allclose(inputs.seen_human, offline[:, :-1], atol=1e-6)
            np.testing.assert_array_equal(inputs.seen_valid, offline[:, -1])
            boundaries = np.arange(STEPS, len(features.time_s) + 1, STEPS) - 1
            np.testing.assert_allclose(inputs.seen_context[boundaries], offline_context[boundaries], atol=1e-6)

    def test_teacher_map_fit_is_checked_on_held_out_clips(self):
        with TemporaryDirectory() as scratch:
            path = Path(scratch)
            rng = np.random.default_rng(4)
            truth = rng.normal(0, 0.1, (6, 8))
            rows = []
            for clip in range(10):
                times = np.arange(1, 101) * 0.04
                kernel = np.hanning(15) / np.hanning(15).sum()
                x = np.column_stack([np.convolve(rng.normal(0, 4, 114), kernel, 'valid') for _ in range(6)])
                np.savez(path / f"f{clip}.npz", time_s=times, features=x)
                ref_times = np.arange(1, 81) * 0.05
                y = np.column_stack([np.interp(ref_times, times, (x @ truth)[:, k]) for k in range(8)])
                np.savez(path / f"r{clip}.npz", time_s=ref_times, channels=np.array(CHANNELS), human=np.clip(y, -1, 1))
                rows.append({"id": f"c{clip}", "group": f"g{clip}", "features": f"f{clip}.npz",
                             "reference": f"r{clip}.npz"})
            (path / "pairs.json").write_text(json.dumps(rows))
            report = fit_teacher_map(path / "pairs.json", path / "calibration.npz")
            self.assertFalse(set(report["splits"]["test"]) & set(report["splits"]["train"]))
            self.assertGreater(min(report["test"]["r2"].values()), 0.9)
            with np.load(path / "calibration.npz") as calibration:
                self.assertEqual(calibration["weights"].shape, (6, 8))


class SessionTests(unittest.TestCase):
    def stream(self, path: Path, frames=60, context_dim=8, gap=None):
        human = np.zeros((frames, 8), dtype=np.float32)
        human[12:22, NECK] = 0.6 * np.sin(np.linspace(0, np.pi, 10)) ** 2
        valid = np.ones(frames)
        if gap is not None:
            valid[gap] = 0
        np.savez(path, time_s=np.arange(1, frames + 1) * FRAME_S, human=human, human_valid=valid,
                 context=np.zeros((frames, context_dim)))
        return path

    def test_projected_baseline_session_records_packable_chunks(self):
        with TemporaryDirectory() as scratch:
            path = Path(scratch)
            stream = self.stream(path / "stream.npz", gap=slice(30, 40))
            limits = capability("fixed-neck-pitch")
            session = Session(ProjectedPolicy(), StreamInputs(stream), 60, limits)
            run_session(session, KinematicRobot(limits))
            report = write_session(session, path / "run", {"session_id": "s1", "participant_group": "p1",
                                                          "profile": "fixed-neck-pitch", "policy": "projected"})
            metrics = report["metrics"]
            self.assertTrue(metrics["commanded_within_position_limits"])
            self.assertTrue(metrics["commanded_within_speed_limits"])
            self.assertEqual(metrics["disabled_channels_max_abs_measured_rad"], 0)
            self.assertEqual(metrics["fallback_chunks"], {"cold_start": 1, "no_teacher_motion": 1})
            fragment = json.loads(pack_recording(stream, path / "run/realized-motion.npz", path / "trials").read_text())
            self.assertEqual(len(fragment["trials"]), 5)  # cold-start chunk skipped
            row = fragment["trials"][0]
            self.assertEqual((row["group"], row["profile"], row["condition"]), ("p1", "fixed-neck-pitch", "projected"))
            with np.load(path / "trials" / row["file"]) as trial:
                self.assertEqual(trial["robot_rad"].shape, (STEPS, 8))
                self.assertEqual(trial["state_rad"].shape, (2, 8))

    def test_measured_state_feeds_back_into_each_chunk(self):
        with TemporaryDirectory() as scratch:
            stream = self.stream(Path(scratch) / "stream.npz")
            limits = capability("full")
            session = Session(ProjectedPolicy(), StreamInputs(stream), 60, limits)
            robot = KinematicRobot(limits, lag=0.3)
            run_session(session, robot)
            chunk = session.chunks[3]  # starts at 1.5 s
            index = round(chunk["start_s"] / FRAME_S)
            np.testing.assert_allclose(chunk["initial_state_rad"][0], session.measured[index - 1], atol=1e-6)


class TrainingTests(unittest.TestCase):
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
            self.assertTrue(report["data"]["synthetic"])
            self.assertEqual(report["loss_weights"], {"demo": 1.0, "meaning": 0.12, "temporal": 0.15})
            groups = [set(split["groups"]) for split in report["splits"].values()]
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            g, r, metadata = load_models(path / "trained/g.pt", path / "trained/r.pt")
            self.assertTrue(g.residual)
            self.assertEqual(len(metadata["labels"]), 3)
            self.assertIsNotNone(r)

    def test_paired_variants_cannot_span_groups(self):
        with TemporaryDirectory() as scratch:
            manifest = make_fixtures(Path(scratch), 5, 3)
            spec = json.loads(manifest.read_text())
            spec["trials"][0]["group"] = "elsewhere"
            manifest.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "must share one group"):
                load_manifest(manifest)

    def test_each_label_must_have_training_votes(self):
        with TemporaryDirectory() as scratch:
            manifest = make_fixtures(Path(scratch) / "d", 5, 3)
            spec = json.loads(manifest.read_text())
            for trial in spec["trials"]:
                trial["votes"].pop("enthusiastic_ack", None)
                if not trial["votes"]:
                    trial["votes"] = {"unclear": 1}
            manifest.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "cover every interpretation label"):
                train(manifest, Path(scratch) / "o", 1, 1, 1, 0.1, 0.1)

    def test_meaning_loss_flows_through_frozen_r_and_raises_intended_probability(self):
        torch.manual_seed(2)
        with TemporaryDirectory() as scratch:
            labels, trials = load_manifest(make_fixtures(Path(scratch), 5, 3))
        data = batch([t for t in trials if t.intent > 0][:32])
        r = Observer(len(trials[0].context), len(labels))
        optimizer = torch.optim.Adam(r.parameters(), 0.01)
        everything = batch(trials)
        for _ in range(150):
            optimizer.zero_grad()
            F.cross_entropy(r(everything["robot"], everything["context"], everything["limits"]),
                            everything["votes"]).backward()
            optimizer.step()
        r.eval().requires_grad_(False)
        r.zero_grad(set_to_none=True)
        g = Generator(len(trials[0].context))
        optimizer = torch.optim.Adam(g.parameters(), 0.003)

        def intended():
            motion = g(data["human"], data["context"], data["limits"], data["state"])
            return r(motion, data["context"], data["limits"]).softmax(1).gather(1, data["intent"][:, None]).mean()

        with torch.no_grad():
            before = float(intended())
        for step in range(40):
            optimizer.zero_grad()
            motion = g(data["human"], data["context"], data["limits"], data["state"])
            loss = F.cross_entropy(r(motion, data["context"], data["limits"]), data["intent"])
            loss.backward()
            if step == 0:
                self.assertTrue(all(p.grad is None for p in r.parameters()))
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in g.parameters() if p.grad is not None), 0)
            optimizer.step()
        with torch.no_grad():
            self.assertGreater(float(intended()), before + 0.05)

    def test_temporal_loss_penalizes_abrupt_chunk_boundaries(self):
        state = torch.zeros(1, 2, 8)
        state[0, 1, NECK] = 0.5  # moving at the boundary, normalized units per second
        continued = (torch.arange(1, STEPS + 1) * 0.5 * FRAME_S)[None, :, None].repeat(1, 1, 8)
        reversed_ = -continued
        data = {"state": state}
        self.assertLess(float(smoothness(continued, data)), float(smoothness(reversed_, data)))


class ObserverStudyTests(unittest.TestCase):
    def manifest(self, path: Path, video: Path | None, synthetic: bool):
        manifest = make_fixtures(path / "d", 5, 9)
        spec = json.loads(manifest.read_text())
        spec["synthetic"] = synthetic
        spec["trials"] = spec["trials"][:18]
        for row in spec["trials"]:
            row.pop("votes")
            row["condition"] = "projected" if row["profile"] == "fixed-neck-pitch" else "g"
            row["start_s"] = 1.0
            row["rating_video"] = str(video) if video else None
            row["origin"], row["motion_source"] = "lab_session", "simulator_replay"
            row["teacher_source"] = "external_export_unverified"
        manifest.write_text(json.dumps(spec))
        return manifest

    def video(self, path: Path) -> Path:
        out = path / "rating-source.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10",
                        "-f", "lavfi", "-i", "sine=frequency=300", "-t", "3", "-metadata", "title=g fixed-neck-pitch",
                        "-pix_fmt", "yuv420p", "-shortest", str(out)], check=True)
        return out

    def test_packet_rejects_trials_without_rating_video(self):
        with TemporaryDirectory() as scratch:
            manifest = self.manifest(Path(scratch), None, False)
            with self.assertRaisesRegex(ValueError, "18 trial\\(s\\) have no rating video"):
                rating_packet(manifest, Path(scratch) / "packet")
            self.assertFalse((Path(scratch) / "packet").exists())

    def test_blinded_packet_votes_and_evidence_tiers(self):
        with TemporaryDirectory() as scratch:
            path = Path(scratch)
            manifest = self.manifest(path, self.video(path), False)
            packet = rating_packet(manifest, path / "packet", seed=1)
            self.assertEqual(packet["clips"], 18)
            key_path = Path(packet["key"])
            self.assertFalse(key_path.resolve().is_relative_to((path / "packet").resolve()))
            self.assertEqual(sorted(p.name for p in (path / "packet").iterdir()), ["clips", "labels.txt", "rating-sheet.csv"])
            sheet = (path / "packet/rating-sheet.csv").read_text()
            for hidden in ("intent", "projected", "fixed-neck", "condition"):
                self.assertNotIn(hidden, sheet)
            clip = next((path / "packet/clips").iterdir())
            tags = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", str(clip)],
                                  capture_output=True, text=True, check=True).stdout
            self.assertNotIn("fixed-neck-pitch", tags)
            key = json.loads(key_path.read_text())["clips"]
            with (path / "responses.csv").open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["clip_code", "rater_id", "label"])
                for code, info in key.items():
                    observed = int(info["id"].rsplit("-", 1)[1])
                    for rater in ("a", "b", "c"):
                        writer.writerow([code, rater, LABELS[observed]])
            merged = merge_votes(manifest, key_path, [path / "responses.csv"], path / "rated.json")
            self.assertEqual(merged, {"trials_with_votes": 18, "raters": 3, "ratings": 54})
            report = evaluate(path / "rated.json", path / "report.json")
            self.assertTrue(report["human_observer_evidence"])
            self.assertFalse(report["human_evidence_on_physical_robot"])
            self.assertEqual(report["evidence_tiers_present"], ["human_votes_on_simulated_motion"])
            self.assertEqual(report["raters"], 3)
            for condition in report["conditions"].values():
                self.assertAlmostEqual(condition["intended_vote_share"], 1 / 3)
                self.assertEqual(condition["teacher"], ["unverified"])

    def test_synthetic_and_unrated_results_are_never_human_evidence(self):
        with TemporaryDirectory() as scratch:
            path = Path(scratch)
            fixture = make_fixtures(path / "f", 5, 2)  # programmatic votes, not rating sheets
            report = evaluate(fixture, path / "r1.json")
            self.assertFalse(report["human_observer_evidence"])
            self.assertEqual(report["evidence_tiers_present"], ["synthetic_votes"])
            self.assertEqual(report["raters"], 0)
            spec = json.loads(fixture.read_text())
            spec["synthetic"] = False  # even if someone drops the flag
            fixture.write_text(json.dumps(spec))
            self.assertFalse(evaluate(fixture, path / "r2.json")["human_observer_evidence"])
            unrated = self.manifest(path / "u", None, False)
            report = evaluate(unrated, path / "r3.json")
            self.assertEqual(report["evidence_tiers_present"], ["unrated"])
            self.assertEqual(sum(report["unrated_trials_by_condition"].values()), 18)
            with self.assertRaisesRegex(ValueError, "did not come from rating sheets"):
                key = path / "k.json"
                trial = spec["trials"][0]["id"]
                key.write_text(json.dumps({"clips": {"c1": {"id": trial}}}))
                (path / "s.csv").write_text("clip_code,rater_id,label\nc1,x,unclear\n")
                merge_votes(fixture, key, [path / "s.csv"], path / "bad.json")


class PersonaPlexTests(unittest.TestCase):
    def test_personaplex_capture_skips_pre_input_state(self):
        from types import SimpleNamespace

        from speech_ai.personaplex_offline import capture_states

        class Gen:
            _frame_size = 1920
            _sample_rate = 24000

            def process_transformer_output(self, state):
                return state.sum().item()

            def step(self, state):
                return self.process_transformer_output(state)

        offline = SimpleNamespace(LMGen=Gen, lm_load_audio=lambda: "audio")
        original = Gen.step
        generator = Gen()
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
        self.assertIs(Gen.step, original)


class FakeBridge:
    """Stands in for a robot driver process: kinematic joints behind the bridge protocol."""

    def __init__(self, limits, embodiment="simulated", stop_after_s=None):
        import socket
        import threading

        self.robot = KinematicRobot(limits, lag=0.5)
        self.embodiment, self.stop_after_s = embodiment, stop_after_s
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.address = "127.0.0.1:%d" % self.listener.getsockname()[1]
        self.targets, self.holds = 0, 0
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        import threading
        import time

        connection, _ = self.listener.accept()
        connection.sendall(json.dumps({"type": "hello", "embodiment": self.embodiment, "name": "fake"}).encode() + b"\n")
        started = time.monotonic()

        def read():
            for line in connection.makefile("rb"):
                message = json.loads(line)
                if message["type"] == "target":
                    self.robot.command(np.array(message["position"]))
                    self.targets += 1
                elif message["type"] == "hold":
                    self.holds += 1

        threading.Thread(target=read, daemon=True).start()
        try:
            while self.stop_after_s is None or time.monotonic() - started < self.stop_after_s:
                self.robot.advance(0.01)
                state = self.robot.state()
                connection.sendall(json.dumps({"type": "state", "position": state[0].tolist(),
                                               "velocity": state[1].tolist()}).encode() + b"\n")
                time.sleep(0.01)
            time.sleep(1.0)  # silent: feedback lost
        except OSError:
            pass


class LiveLinkTests(unittest.TestCase):
    def producer(self, address, duration_s, bad_clock=False):
        import threading
        import time

        from speech_ai.gr_link import Publisher

        def run():
            publisher = Publisher(*address)
            if bad_clock:
                publisher.send("teacher", publisher.session_time() + 5.0, np.zeros(8))
            k = 0
            while publisher.session_time() < duration_s:
                now = publisher.session_time()
                frame = np.zeros(8)
                frame[NECK] = 0.6 * np.sin(np.pi * ((k % 10) / 10)) ** 2
                publisher.send("teacher", now, frame)
                publisher.send("audio", now, [-25, 180, 0.8, 1, -80, 0, 0, 0])
                k += 1
                time.sleep(0.05)
            publisher.close()

        threading.Thread(target=run, daemon=True).start()

    def test_audio_frontend_streams_the_offline_features(self):
        from speech_ai.audio import AudioFrontend, extract

        rng = np.random.default_rng(3)
        rate = 16000
        partner = np.sin(2 * np.pi * 180 * np.arange(rate) / rate) * (np.arange(rate) > 4000) * 0.3
        agent = rng.normal(0, 0.05, rate)
        offline = extract((partner, rate), (agent, rate))
        frontend = AudioFrontend(rate)
        for k in range(len(offline.time_s)):
            a, b = round(k * FRAME_S * rate), round((k + 1) * FRAME_S * rate)
            np.testing.assert_allclose(frontend.frame(partner[a:b], agent[a:b]), offline.values[k], atol=1e-5)

    def test_listen_mode_uses_only_arrived_inputs_and_rejects_clock_errors(self):
        from speech_ai.gr_link import serve

        with TemporaryDirectory() as scratch:
            out = Path(scratch) / "live"
            report = serve("127.0.0.1:0", 1.5, None, None, "fixed-neck-pitch", out, policy="projected",
                           robot="kinematic", session_id="t1", group="p1", wait_for_producers=1,
                           ready=lambda address: self.producer(address, 1.6, bad_clock=True))
            self.assertEqual(report["metrics"]["fallback_chunks"]["no_teacher_motion"], 0)
            self.assertEqual(report["origin"], "live_simulated_robot")
            stats = report["input_stats"]
            self.assertGreater(stats["teacher"]["received"], 20)
            self.assertEqual(stats["teacher"]["rejected"], 1)
            self.assertIn("timestamp after arrival", next(iter(stats["teacher"]["reasons"])))
            self.assertTrue(report["metrics"]["commanded_within_speed_limits"])
            self.assertEqual(report["metrics"]["disabled_channels_max_abs_measured_rad"], 0)
            self.assertLess(report["metrics"]["boundary_late_ms"]["max"], 40)
            self.assertGreater(min(c["teacher_frames_valid"] for c in report["chunks"][1:]), 5)
            _, human, context = load_stream(out / "stream.npz")
            self.assertGreater(human[:, NECK].max(), 0.3)
            boundaries = np.arange(STEPS, len(context), STEPS) - 1  # rows G reads at planned boundaries
            self.assertTrue((context[boundaries, 3] == 1).all())  # partner active, from the audio frames
            fragment = json.loads(pack_recording(out / "stream.npz", out / "realized-motion.npz", out / "trials").read_text())
            self.assertEqual(fragment["trials"][0]["motion_source"], "live_simulated_robot")

    def test_network_robot_feeds_measured_state_back(self):
        from speech_ai.gr_link import serve

        limits = capability("full")
        bridge = FakeBridge(limits)
        with TemporaryDirectory() as scratch:
            calibration = Path(scratch) / "calib.json"
            calibration.write_text(json.dumps(dict(zip(CHANNELS, limits.tolist()))))
            report = serve("127.0.0.1:0", 1.5, None, None, "full", Path(scratch) / "net", calibration, "projected",
                           "network", bridge.address, ready=lambda address: self.producer(address, 1.6))
        self.assertIsNone(report["aborted"])
        self.assertEqual(report["origin"], "live_simulated_robot")
        self.assertGreaterEqual(bridge.targets, 29)
        self.assertGreater(max(c["initial_state_rad"][0][NECK] for c in report["chunks"]), 0.01)

    def test_lost_feedback_holds_the_robot_and_cuts_the_session(self):
        import time

        from speech_ai.gr_link import serve

        limits = capability("full")
        bridge = FakeBridge(limits, embodiment="physical", stop_after_s=0.8)
        with TemporaryDirectory() as scratch:
            calibration = Path(scratch) / "calib.json"
            calibration.write_text(json.dumps(dict(zip(CHANNELS, limits.tolist()))))
            out = Path(scratch) / "net"
            report = serve("127.0.0.1:0", 2.0, None, None, "full", out, calibration, "projected", "network",
                           bridge.address)
            time.sleep(0.1)
            self.assertIsNotNone(report["aborted"])
            self.assertLess(report["aborted"]["at_s"], 1.2)
            self.assertGreaterEqual(bridge.holds, 1)
            self.assertEqual(report["origin"], "live_physical_robot")
            with np.load(out / "realized-motion.npz") as data:
                self.assertEqual(len(data["robot_rad"]), report["aborted"]["at_frame"])

    def test_personaplex_states_stream_into_the_live_inputs(self):
        from types import SimpleNamespace

        from speech_ai.gr_link import LiveServer, SessionClock
        from speech_ai.personaplex_offline import StatePublisher, capture_states

        class Gen:
            _frame_size = 1920
            _sample_rate = 24000

            def process_transformer_output(self, state):
                return state.sum().item()

            def step(self, state):
                return self.process_transformer_output(state)

        inputs = LiveInputs(40, 2)
        with LiveServer(inputs, SessionClock()) as server:
            publisher = StatePublisher("%s:%d" % server.address)
            offline = SimpleNamespace(LMGen=Gen, lm_load_audio=lambda: "audio")
            with capture_states(offline, None, publisher):
                offline.lm_load_audio()
                for k in range(3):
                    Gen().step(torch.tensor([[[float(k), 1.0]]]))
            import time
            publisher.close()
            time.sleep(0.2)
        self.assertEqual(server.stats["states"]["received"], 3)
        self.assertEqual([row[2].tolist() for row in inputs.buffers["states"]], [[0, 1], [1, 1], [2, 1]])


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "mujoco not installed")
class MujocoTests(unittest.TestCase):
    def test_mujoco_adapter_tracks_commands_within_limits(self):
        from speech_ai.simulate import MujocoRobot

        with TemporaryDirectory() as scratch:
            stream = SessionTests.stream(None, Path(scratch) / "stream.npz")
            limits = capability("full")
            session = Session(ProjectedPolicy(), StreamInputs(stream), 60, limits)
            robot = MujocoRobot(limits)
            run_session(session, robot)
            report = robot.close()
            self.assertTrue(report["finite_state"])
            self.assertEqual(report["self_contact_steps"], 0)
            metrics = motion_metrics(session)
            self.assertEqual(metrics["measured_outside_limits_frames"], 0)
            self.assertGreater(session.measured[:, NECK].max(), 0.05)
            self.assertLess(metrics["mean_abs_tracking_error_rad"]["neck_pitch"], 0.05)

    def test_rating_render_has_no_labels(self):
        from speech_ai.simulate import MujocoRobot

        with TemporaryDirectory() as scratch:
            path = Path(scratch)
            stream = SessionTests.stream(None, path / "stream.npz", frames=30)
            limits = capability("fixed-neck-pitch")
            session = Session(ProjectedPolicy(), StreamInputs(stream), 30, limits)
            robot = MujocoRobot(limits, video=path / "labeled.mp4", rating_video=path / "rating.mp4",
                                title="SECRET-CONDITION", output=path)
            run_session(session, robot, on_frame=robot.annotate)
            robot.close()
            frames = {}
            for name in ("labeled", "rating"):
                raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", "0.5", "-i", str(path / f"{name}.mp4"),
                                      "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                     capture_output=True, check=True).stdout
                frames[name] = np.frombuffer(raw, np.uint8).reshape(384, 640, 3)
            banner = np.array([0x11, 0x1a, 0x27])
            self.assertLess(np.abs(frames["labeled"][5:50, 625].astype(int) - banner).max(), 12)
            self.assertGreater(np.abs(frames["rating"][5:50, 625].astype(int) - banner).max(), 30)


if __name__ == "__main__":
    unittest.main()
