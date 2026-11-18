"""Live process boundary: producers -> robot computer -> robot bridge.

Everything travels as newline-delimited JSON over TCP so the DyaPlex, PersonaPlex,
microphone and robot-driver processes can be written in any language.

Producer -> G (``LiveServer``), one object per line::

    {"source": "teacher",          "time_s": t, "values": [8 normalized channels], "valid": true}
    {"source": "teacher_features", "time_s": t, "values": [f decoded DyaPlex features]}   # needs --teacher-map
    {"source": "states",           "time_s": t, "values": [d PersonaPlex state]}
    {"source": "audio",            "time_s": t, "values": [8 raw audio features]}          # see audio.AudioFrontend

``time_s`` is the sample's timestamp on the producer's clock (end of the frame it
summarizes); per-source offsets map it to the session clock. Arrival is stamped
here, on receipt, with the session clock. A sample stamped after its own arrival
means the clocks are out of sync; it is rejected and counted, never used.

G <-> robot bridge (``NetworkRobot``)::

    bridge -> G: {"type": "hello", "embodiment": "physical" | "simulated", "name": "..."}  (once)
    bridge -> G: {"type": "state", "position": [8 rad], "velocity": [8 rad/s]}              (>= 20 Hz)
    G -> bridge: {"type": "target", "frame": k, "position": [8 rad]}                        (20 Hz)
    G -> bridge: {"type": "hold"}                                                           (stop / feedback lost)

The bridge owns the vendor servo driver, maps the eight channels to actuators and
enforces firmware limits; G enforces the calibrated capability descriptor.
"""

import json
import socket
import socketserver
import threading
import time

import numpy as np

from .gr_model import CHANNELS
from .gr_runtime import LiveInputs, RobotFeedbackLost, RobotInterface


class SessionClock:
    """Monotonic seconds since the session started; the common recording clock."""

    def __init__(self):
        self.restart()

    def restart(self) -> None:
        self.start = time.monotonic()
        self.start_unix = time.time()

    def __call__(self) -> float:
        return time.monotonic() - self.start


class Pacer:
    """Holds the 20 Hz loop to wall-clock time and records how late each wait ended."""

    def __init__(self, clock: SessionClock):
        self.clock = clock
        self.late_ms = []

    def wait(self, t: float) -> float:
        remaining = t - self.clock()
        if remaining > 0:
            time.sleep(remaining)
        late = max(0.0, (self.clock() - t) * 1000)
        self.late_ms.append(late)
        return late


class LiveServer:
    def __init__(self, inputs: LiveInputs, clock: SessionClock, host: str = "127.0.0.1", port: int = 0,
                 offsets: dict[str, float] | None = None, teacher_map=None, started: bool = True):
        self.inputs, self.clock = inputs, clock
        self.connected = 0
        self.started = threading.Event()
        if started:
            self.started.set()
        self.offsets = offsets or {}
        self.teacher_map = teacher_map
        self.stats = {name: {"received": 0, "rejected": 0, "max_delay_s": 0.0, "reasons": {}}
                      for name in ("teacher", "states", "audio")}
        server = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                server.connected += 1
                server.started.wait()
                # Tell the producer when the session clock started so it can stamp samples on it.
                hello = {"type": "session", "start_unix": server.clock.start_unix, "now_s": server.clock()}
                self.wfile.write(json.dumps(hello).encode() + b"\n")
                for line in self.rfile:
                    if line.strip():
                        server.receive(line, server.clock())

        socketserver.ThreadingTCPServer.allow_reuse_address = True
        self.server = socketserver.ThreadingTCPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.address = self.server.server_address
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    def receive(self, line: bytes, arrival: float) -> None:
        name = "unknown"
        try:
            message = json.loads(line)
            name = "teacher" if message["source"] == "teacher_features" else message["source"]
            if name not in self.stats:
                raise ValueError("unknown source")
            values = np.asarray(message["values"], dtype=np.float32)
            if message["source"] == "teacher_features":
                if self.teacher_map is None:
                    raise ValueError("teacher_features need --teacher-map")
                values = self.teacher_map(values[None])[0]
            stamp = float(message["time_s"]) + self.offsets.get(name, 0.0)
            if stamp > arrival + 1e-3:
                raise ValueError("timestamp after arrival; check the clock offset")
            self.inputs.push(name, stamp, max(arrival, stamp), values, bool(message.get("valid", True)))
        except (KeyError, ValueError, TypeError) as error:
            stats = self.stats.setdefault(name, {"received": 0, "rejected": 0, "max_delay_s": 0.0, "reasons": {}})
            stats["rejected"] += 1
            reason = str(error)[:80]
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
            return
        stats = self.stats[name]
        stats["received"] += 1
        stats["max_delay_s"] = max(stats["max_delay_s"], arrival - stamp)


class Publisher:
    """Client used by producer processes (DyaPlex, PersonaPlex hook, microphone frontend)."""

    def __init__(self, host: str, port: int, timeout: float = 5.0):
        self.socket = socket.create_connection((host, port), timeout=timeout)
        self.lock = threading.Lock()
        hello = json.loads(self.socket.makefile("rb").readline())
        # Producers on the same machine can stamp with session_time(); others add a measured offset.
        self.session_start_unix = hello["start_unix"]

    def session_time(self) -> float:
        return time.time() - self.session_start_unix

    def send(self, source: str, time_s: float, values, valid: bool = True) -> bool:
        """Returns False once the session has closed the connection."""
        line = json.dumps({"source": source, "time_s": float(time_s),
                           "values": np.asarray(values, dtype=float).tolist(), "valid": bool(valid)})
        try:
            with self.lock:
                self.socket.sendall(line.encode() + b"\n")
        except (BrokenPipeError, ConnectionResetError):
            return False
        return True

    def close(self) -> None:
        self.socket.close()


def parse_address(text: str) -> tuple[str, int]:
    host, _, port = text.rpartition(":")
    return host or "127.0.0.1", int(port)


class NetworkRobot(RobotInterface):
    """Robot reached through a bridge process; measured state feeds every chunk boundary."""

    def __init__(self, address: tuple[str, int], stale_s: float = 0.1, connect_timeout: float = 5.0):
        self.socket = socket.create_connection(address, timeout=connect_timeout)
        self.socket.settimeout(None)
        self.stale_s = stale_s
        self.lock = threading.Lock()
        self.latest = None
        self.received_at = -np.inf
        self.hello = {}
        self.frame = 0
        self.states_received = 0
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        if not self.ready.wait(connect_timeout):
            raise RobotFeedbackLost("The robot bridge sent no state after connecting")

    def _read(self) -> None:
        with self.socket.makefile("rb") as stream:
            for line in stream:
                message = json.loads(line)
                if message.get("type") == "hello":
                    self.hello = message
                elif message.get("type") == "state":
                    state = np.array([message["position"], message["velocity"]], dtype=np.float32)
                    if state.shape != (2, len(CHANNELS)) or not np.isfinite(state).all():
                        continue
                    with self.lock:
                        self.latest, self.received_at = state, time.monotonic()
                        self.states_received += 1
                    self.ready.set()

    @property
    def embodiment(self) -> str:
        return self.hello.get("embodiment", "unknown")

    def state(self) -> np.ndarray:
        with self.lock:
            age = time.monotonic() - self.received_at
            if self.latest is None or age > self.stale_s:
                raise RobotFeedbackLost(f"Robot state is {age * 1000:.0f} ms old")
            return self.latest.copy()

    def _send(self, message: dict) -> None:
        self.socket.sendall(json.dumps(message).encode() + b"\n")

    def command(self, target_rad: np.ndarray) -> None:
        self._send({"type": "target", "frame": self.frame, "position": np.asarray(target_rad, dtype=float).tolist()})
        self.frame += 1

    def advance(self, seconds: float) -> None:
        """Hardware moves on its own; the pacer holds the loop to wall-clock time."""

    def hold(self) -> None:
        try:
            self._send({"type": "hold"})
        except OSError:
            pass

    def close(self) -> dict:
        self.hold()
        self.socket.close()
        return {"bridge": self.hello, "states_received": self.states_received}


def serve(listen: str, duration_s: float, generator, observer, profile: str, output, calibration=None,
          policy: str = "g", robot: str = "mujoco", robot_address: str | None = None, state_dim: int | None = None,
          teacher_map=None, offsets: dict | None = None, session_id: str | None = None, group: str | None = None,
          context_id: str | None = None, video=None, conversation_audio=None, title: str | None = None,
          ready=None, wait_for_producers: int = 0, wait_timeout_s: float = 300.0) -> dict:
    """Run G live: inputs arrive over TCP, the loop runs on the wall clock, the robot reports state."""
    from pathlib import Path

    import torch

    from .gr_data import project_features
    from .gr_model import capability
    from .gr_runtime import KinematicRobot, Session, make_policy, run_session, write_session

    torch.set_num_threads(1)
    output = Path(output)
    if robot == "network" and calibration is None:
        raise ValueError("A physical or bridged robot needs its measured --calibration JSON")
    limits = capability(profile, json.loads(Path(calibration).read_text()) if calibration else None)
    chosen, judge, labels, context_dim = make_policy(generator, observer, policy)
    if context_dim is not None:
        derived = context_dim - 8 - 1 if context_dim > 8 else None
        if state_dim not in (None, derived):
            raise ValueError(f"G was trained with PersonaPlex width {derived}, not {state_dim}")
        state_dim = derived
    frames = int(round(duration_s / 0.05))
    inputs = LiveInputs(frames, state_dim)
    project = (lambda values: project_features(values, Path(teacher_map))) if teacher_map else None
    # Connect the robot first so setup time does not eat into the first chunk.
    if robot == "network":
        adapter = NetworkRobot(parse_address(robot_address))
    elif robot == "kinematic":
        adapter = KinematicRobot(limits)
    else:
        from .simulate import MujocoRobot
        adapter = MujocoRobot(limits, video=video, audio=conversation_audio, title=title or f"live {profile}",
                              output=output, rating_video=output / "rating-source.mp4" if video else None)
    session = Session(chosen, inputs, frames, limits, judge, labels)
    clock = SessionClock()
    host, port = parse_address(listen)
    with LiveServer(inputs, clock, host, port, offsets, project, started=not wait_for_producers) as server:
        if ready is not None:
            ready(server.address)
        if wait_for_producers:
            deadline = time.monotonic() + wait_timeout_s
            while server.connected < wait_for_producers:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Only {server.connected} of {wait_for_producers} producers connected")
                time.sleep(0.01)
            clock.restart()  # every waiting producer receives this start time
            server.started.set()
        pacer = Pacer(clock)
        try:
            run_session(session, adapter, on_frame=getattr(adapter, "annotate", None), pacer=pacer)
        finally:
            robot_report = adapter.close()
        stats = server.stats
    output.mkdir(parents=True, exist_ok=True)
    inputs.save_stream(output / "stream.npz", session.frames)
    embodiment = getattr(adapter, "embodiment", "simulated")
    origin = {"physical": "live_physical_robot", "simulated": "live_simulated_robot"}.get(
        embodiment, "live_network_unknown_embodiment")
    meta = {"session_id": session_id or output.name, "participant_group": group, "context_id": context_id,
            "origin": origin, "profile": profile if calibration is None else "custom", "policy": chosen.name,
            "g_sha256": chosen.digest, "robot": robot, "robot_report": robot_report, "session_start_unix": clock.start_unix,
            "input_stats": stats, "teacher_source": "live_stream_unverified",
            "rating_video": str(output / "rating-source.mp4") if video else None,
            "frame_late_ms_max": float(max(pacer.late_ms)) if pacer.late_ms else None}
    return write_session(session, output, meta)


def publish_files(address: str, partner, agent=None, stereo: bool = False, teacher=None, embeddings=None,
                  latency_s: float = 0.0) -> dict:
    """Dry-run producer: play recorded inputs into a live session at real-time rate.

    Each sample is sent when the session clock reaches its timestamp plus
    ``latency_s``, stamped on the session clock, exactly as a live producer would.
    """
    from .audio import AudioFrontend, load_tracks
    from .gr_data import load_states, load_teacher

    publisher = Publisher(*parse_address(address))
    (partner_samples, rate), agent_track = load_tracks(partner, agent, stereo)
    frontend = AudioFrontend(rate)
    frames = int(np.ceil(len(partner_samples) / rate / 0.05 - 1e-9))
    events = []
    for k in range(frames):
        a, b = round(k * 0.05 * rate), round((k + 1) * 0.05 * rate)
        events.append(((k + 1) * 0.05, "audio", (partner_samples[a:b], agent_track[0][a:b] if agent_track else None)))
    if teacher is not None:
        source = load_teacher(teacher, None)
        events += [(t, "teacher", (v, ok)) for t, v, ok in zip(source.time_s, source.values, source.valid)]
    if embeddings is not None:
        states = load_states(embeddings)
        events += [(t, "states", (v, True)) for t, v in zip(states.time_s, states.values)]
    events.sort(key=lambda event: (event[0], event[1]))
    start = publisher.session_time()
    sent = {"audio": 0, "teacher": 0, "states": 0}
    open_ = True
    for stamp, name, payload in events:
        due = start + stamp + latency_s
        delay = due - publisher.session_time()
        if delay > 0:
            time.sleep(delay)
        if name == "audio":
            open_ = publisher.send("audio", start + stamp, frontend.frame(*payload))
        else:
            open_ = publisher.send(name, start + stamp, payload[0], payload[1])
        if not open_:
            break
        sent[name] += 1
    publisher.close()
    return {"sent": sent, "session_offset_s": round(start, 4), "session_closed_early": not open_}
