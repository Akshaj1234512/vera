"""Smallest end-to-end DyaPlex test (review §7): real Seamless pairs -> causal codecs ->
PersonaPlex per-layer taps -> motion tower -> streaming sampler -> Samhita's rig.

Writes runs/dyaplex-e2e/report.json with one entry per pass criterion. What this run does and
does not verify is written into the report itself (``scope``).
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch

from . import config as C
from . import data as D
from . import layout, vendor
from .codec import decode_all, device, encode_all, load_codec, save_state, train_codec
from .rotations import axis_angle_to_6d, rotation_6d_to_matrix, matrix_to_6d, axis_angle_to_matrix

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "data" / "seamless-motion" / "naturalistic-dev"
LABEL, SPLIT = "naturalistic", "dev"

SCOPE = {
    "verifies": [
        "data plumbing on real Seamless pairs: pairing, validity masks, 30->25 fps resampling, part features",
        "causality, band layout and round trips of the four GestureLSM-derived causal RVQ-VAEs (reference architecture)",
        "PersonaPlex per-layer taps on NVIDIA's own LMModel/LMGen code: teacher-forced == streaming, taps do not "
        "change outputs, frame/time mapping",
        "motion tower wiring: 46-token interleaving, Eq. 2-4 masks, band-masked loss, causality, ability to fit",
        "streaming sampler: KV cache == recompute, prefix stability under changed future inputs, both modes",
        "decoded agent motion drives all 27 actuators of Samhita's rig through the existing speech_ai replay path",
    ],
    "does_not_verify": [
        "anything about motion quality: codecs and tower are trained for minutes on 10 / 2 pairs",
        "real PersonaPlex-7B states (needs the gated checkpoint and a CUDA GPU); the tiny LM here has the "
        "published wiring (32 layers, 17 streams, delays, context 3000) but random weights",
        "Mimi audio tokens and the SentencePiece inner-monologue text: replaced by band-energy stand-in tokens and PAD",
        "SPECTRE/FLAME faces: the 56-D face track is a stand-in built from Seamless FAU values",
        "SMPL-X forward kinematics and the paper's metrics (need the SMPL-X model files)",
        "paper-scale sizes: d_s=64 instead of 4096, d_m=128 instead of 1024, window 1024 instead of 4096 tokens",
    ],
}


def _log(msg: str) -> None:
    print(msg, flush=True)


class Report:
    def __init__(self, out: Path):
        self.out = out
        self.data = {"scope": SCOPE, "criteria": {}, "info": {}, "timing": {}}

    def check(self, group: str, name: str, passed: bool, **details) -> bool:
        self.data["criteria"].setdefault(group, {})[name] = {"pass": bool(passed), **_jsonable(details)}
        _log(f"[{'PASS' if passed else 'FAIL'}] {group}: {name} {json.dumps(_jsonable(details))[:240]}")
        return passed

    def save(self) -> None:
        crit = self.data["criteria"]
        self.data["passed"] = all(c["pass"] for g in crit.values() for c in g.values())
        (self.out / "report.json").write_text(json.dumps(_jsonable(self.data), indent=2) + "\n")


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, Path):
        return str(x)
    return x


# --------------------------------------------------------------------------- #
def run(out: Path, n_pairs: int = 10, n_overfit: int = 2, window_s: tuple[float, float] = (30.0, 50.0),
        cfg: dict = C.TINY, seed: int = 0, video: bool = True, resume: bool = False,
        tower_iterations: int | None = None) -> dict:
    """resume=True reuses codec_*.pt (+ .json) and continues training tower.pt from the same out dir."""
    out.mkdir(parents=True, exist_ok=True)
    rep = Report(out)
    rep.data["config"] = C.describe(cfg)
    rep.data["pins"] = vendor.verify()
    rep.data["info"]["personaplex_checkpoint_for_real_run"] = vendor.PERSONAPLEX_CHECKPOINT
    torch.manual_seed(seed)
    began = time.time()

    # ---------------- 1. data ---------------- #
    _log("== 1. data")
    pairs = D.complete_pairs(DEV)[:n_pairs]
    for key, a, b in pairs:
        for pid in (a, b):
            D.fetch_extras(DEV / pid, LABEL, SPLIT)
    probes = [D.probe_video(LABEL, SPLIT, pid) for _, a, b in pairs for pid in (a, b)]
    kept = {p["file_id"] for p in probes if p["keep"]}
    rep.data["info"]["video_probe"] = probes
    rep.check("data", "pairing", all(a.rsplit("_", 1)[0] == b.rsplit("_", 1)[0] == key and a != b for key, a, b in pairs),
              pairs=[[key, a, b] for key, a, b in pairs])
    rep.check("data", "clip_filter_ran", len(probes) == 2 * len(pairs),
              kept_pairs=sum(a in kept and b in kept for _, a, b in pairs), probed=len(probes),
              statuses=sorted({p["status"] for p in probes}))
    people = {pid: D.load_participant(DEV / pid) for _, a, b in pairs for pid in (a, b)}
    rep.data["info"]["participants"] = {pid: p.stats for pid, p in people.items()}
    # resample timing: 25 fps frame k sits at k/25 s on the same clock as the 16 kHz audio
    sync = []
    for pid, p in people.items():
        audio_s = D.read_wav(p.audio)[0].shape[0] / 16000.0
        motion_s = p.frames / C.MOTION_FPS
        src_s = (p.stats["frames_30"] - 1) / C.SEAMLESS_FPS
        sync.append({"id": pid, "resample_end_error_s": abs((p.frames - 1) / C.MOTION_FPS - src_s),
                     "audio_minus_motion_s": audio_s - motion_s})
    rep.check("data", "resample_sync_within_one_frame",
              max(s["resample_end_error_s"] for s in sync) < 1 / C.MOTION_FPS,
              max_resample_error_s=max(s["resample_end_error_s"] for s in sync),
              audio_minus_motion_s=[round(s["audio_minus_motion_s"], 3) for s in sync])
    # 6D round trip on real poses
    sample = np.concatenate([p.pose6d[:500] for p in people.values()])
    rt = matrix_to_6d(rotation_6d_to_matrix(sample))
    rep.check("data", "6d_round_trip", float(np.abs(rt - sample).max()) < 1e-5, max_abs=float(np.abs(rt - sample).max()))
    aa = np.random.default_rng(0).normal(size=(1000, 3))
    m1 = axis_angle_to_matrix(aa)
    m2 = rotation_6d_to_matrix(axis_angle_to_6d(aa))
    rep.check("data", "6d_matrix_round_trip", float(np.abs(m1 - m2).max()) < 1e-6, max_abs=float(np.abs(m1 - m2).max()))

    codec_pairs, held_pairs = pairs[: n_pairs - 2], pairs[n_pairs - 2:]
    train_ids = [pid for _, a, b in codec_pairs for pid in (a, b)]
    held_ids = [pid for _, a, b in held_pairs for pid in (a, b)]
    stats = D.fit_normalization([people[i] for i in train_ids])
    feats = {pid: {part: D.normalize(x, stats[part]) for part, x in D.part_features(p).items()}
             for pid, p in people.items()}
    face_ok = {key: D.face_pair_ok(people[a], people[b]) for key, a, b in pairs}
    rep.data["info"]["face_pair_filter"] = face_ok

    # ---------------- 2. codecs ---------------- #
    _log("== 2. codecs")
    codecs, codec_info = {}, {}
    t0 = time.time()
    for part in C.PARTS:
        def keep(pid):
            if part != "face":
                return True
            key = pid.rsplit("_", 1)[0]
            return face_ok[key]
        tr = [i for i in train_ids if keep(i)]
        saved = out / f"codec_{part}.pt"
        if resume and saved.exists() and saved.with_suffix(".json").exists():
            model = load_codec(part, cfg["codec"], torch.load(saved, weights_only=True))
            info = json.loads(saved.with_suffix(".json").read_text())
            _log(f"  codec {part}: reused {saved.name}")
        else:
            model, info = train_codec(part, cfg["codec"], [feats[i][part] for i in tr],
                                      [D.part_valid(people[i])[part] for i in tr],
                                      [feats[i][part] for i in held_ids],
                                      [D.part_valid(people[i])[part] for i in held_ids], seed=seed, log=_log)
            torch.save(save_state(model), saved)
            saved.with_suffix(".json").write_text(json.dumps(_jsonable(info)) + "\n")
        codecs[part], codec_info[part] = model, info
    rep.data["timing"]["codec_training_s"] = time.time() - t0
    rep.data["info"]["codecs"] = {p: {"best_held_recon": i["best_held_recon"], "history": i["history"]}
                                  for p, i in codec_info.items()}
    for part, info in codec_info.items():
        first, best = info["history"][0]["held_recon"], info["best_held_recon"]
        rep.check("codec", f"{part}_held_recon_falls", best < first, first=first, best=best)
    # causality: perturb frames after 2(i+1) -> codes <= i unchanged; batched == incremental
    leak, incremental = {}, {}
    for part, model in codecs.items():
        x = torch.tensor(feats[held_ids[0]][part][None, 200:328])
        full = model.encode(x)[0]
        lat_full = model.latent(x)
        y = x.clone()
        cut = 64
        y[:, cut:] += torch.randn_like(y[:, cut:]) * 3
        pert = model.encode(y)[0]
        lat_pert = model.latent(y)
        leak[part] = {"codes_equal": bool(torch.equal(full[: cut // 2], pert[: cut // 2])),
                      "latent_max_diff": float((lat_full[..., : cut // 2] - lat_pert[..., : cut // 2]).abs().max())}
        inc = torch.stack([model.encode(x[:, : 2 * (i + 1)])[0, -1] for i in range(full.shape[0])])
        incremental[part] = {"codes_equal": bool(torch.equal(inc, full))}
        dec_full = model.decode(full[None])[0]
        dec_prefix = model.decode(full[None, : cut // 2])[0]
        leak[part]["decoder_prefix_max_diff"] = float((dec_full[:cut] - dec_prefix).abs().max())
    rep.check("codec", "encoder_no_future_leak", all(v["codes_equal"] and v["latent_max_diff"] == 0 for v in leak.values()),
              **leak)
    rep.check("codec", "decoder_causal", all(v["decoder_prefix_max_diff"] < 1e-5 for v in leak.values()),
              **{p: v["decoder_prefix_max_diff"] for p, v in leak.items()})
    rep.check("codec", "batched_equals_incremental", all(v["codes_equal"] for v in incremental.values()), **incremental)

    # token cache [P D.3]: (T, 22) uint16 per participant
    cache_dir = out / "tokens"
    cache_dir.mkdir(exist_ok=True)
    tokens = {}
    for pid in people:
        codes = encode_all(codecs, feats[pid])
        tokens[pid] = codes
        np.save(cache_dir / f"{pid}.npy", codes.astype(np.uint16))
    shapes_ok = all(v.shape[1] == C.K and v.shape[0] == people[k].frames // 2 for k, v in tokens.items())
    rep.check("data", "token_cache_T_by_22_uint16_in_band",
              shapes_ok and all(layout.in_band(v) for v in tokens.values()) and int(max(v.max() for v in tokens.values())) < 2 ** 16,
              shapes={k: list(v.shape) for k, v in tokens.items()})

    # ---------------- 3. speech tower ---------------- #
    _log("== 3. PersonaPlex taps")
    from . import speech as S
    scfg = cfg["speech"]
    lm = S.build_lm(scfg, seed)
    f0, f1 = int(window_s[0] * C.TOKEN_HZ), int(window_s[1] * C.TOKEN_HZ)
    frames = f1 - f0
    rng = np.random.default_rng(seed)
    text_prompt = [int(t) for t in rng.integers(4, scfg.text_card, 12)]   # placeholder ids (tokenizer is gated)
    audio = {pid: D.read_wav(p.audio)[0] for pid, p in people.items()}

    def codes_for(pid, start, count):
        return S.standin_audio_codes(audio[pid][int(start / C.TOKEN_HZ * 16000):], 16000, count, scfg.card)

    def streams_for(agent, user):
        voice = codes_for(agent, 0, int(scfg.voice_prompt_s * C.TOKEN_HZ))
        return S.build_streams(codes_for(agent, f0, frames), codes_for(user, f0, frames),
                               np.full(frames, S.TEXT_PAD), voice, text_prompt, scfg)

    over = pairs[:n_overfit]
    key0, a0, b0 = over[0]
    st = streams_for(b0, a0)
    t0 = time.time()
    tf = S.teacher_forced_states(lm, st)
    t_tf = time.time() - t0
    t0 = time.time()
    sm = S.streaming_states(lm, st)
    t_stream = time.time() - t0
    diff = (tf - sm).abs()
    rel = float(diff.max() / tf.abs().max())
    rep.check("speech", "teacher_forced_equals_streaming_all_32_layers", rel < 1e-5 and tf.shape[0] == 32,
              layers=tf.shape[0], frames=tf.shape[1], max_abs=float(diff.max()), rel=rel,
              per_layer_max=[float(v) for v in diff.amax(dim=(1, 2))])
    user = st.tokens[9:17]
    plain = S.sampled_tokens(lm, user, 60, seed=11, tap=False)
    tapped = S.sampled_tokens(lm, user, 60, seed=11, tap=True)
    rep.check("speech", "taps_do_not_change_outputs", plain.shape == tapped.shape and np.array_equal(plain, tapped),
              frames_compared=len(plain))
    rep.data["info"]["speech_time_base"] = {
        "definition": "H[s] = output of the LMGen.step call that received stream frame s; equals forward_train "
                      "position s-1; conversation frame j -> stream frame prompt_frames + j (+ state_offset)",
        "prompt_frames": st.prompt_frames, "state_offset": scfg.state_offset,
        "prompt_layout": {"voice": int(scfg.voice_prompt_s * C.TOKEN_HZ), "silence": scfg.silence_frames,
                          "text": len(text_prompt), "silence_2": scfg.silence_frames}}
    rep.data["timing"]["speech_teacher_forced_s"] = t_tf
    rep.data["timing"]["speech_streaming_ms_per_frame"] = 1000 * t_stream / st.tokens.shape[1]

    from .generate import Example
    examples, roles = [], []
    for key, a, b in over:
        vad = {pid: D.vad_labels(people[pid].vad, f1)[f0:f1] for pid in (a, b)}
        for user_id, agent_id in ((a, b), (b, a)):         # both role orders [A]
            sts = streams_for(agent_id, user_id)
            states = S.conversation_states(S.teacher_forced_states(lm, sts), sts, scfg.state_offset).numpy()
            examples.append(Example(tokens[user_id][f0:f1], tokens[agent_id][f0:f1], states,
                                    vad[user_id], vad[agent_id], f"{key}:{user_id[-5:]}->{agent_id[-5:]}"))
            roles.append((user_id, agent_id))

    # ---------------- 4. motion tower ---------------- #
    _log("== 4. motion tower")
    from .generate import evaluate, generate, train_tower, window_batch
    from .tower import MotionTower
    tcfg = cfg["tower"]
    model = MotionTower(tcfg).eval()
    rep.data["info"]["tower_params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
    wf = layout.window_frames(tcfg.window_tokens)
    toks, labels, bands, vas, pos, qf, speech, s_pos = window_batch(examples, [(0, 0)], wf, tcfg, "cpu")
    rep.check("tower", "46_tokens_per_frame_and_eq2", C.L_STEP == 46 and int(qf[45]) == 0 and int(qf[46]) == 1
              and toks.shape[1] == wf * 46 - 1, window_frames=wf, tokens=int(toks.shape[1] + 1))

    def causality(m):
        kv = m.project_speech(speech, s_pos)
        base, _ = m(toks, pos, qf, kv, s_pos)
        f = wf // 2
        sp2 = [h.clone() for h in speech]
        for h in sp2:
            h[:, f + 1:] += torch.randn_like(h[:, f + 1:])
        t2 = toks.clone()
        later = qf > f
        t2[0, later] = torch.where(t2[0, later] < C.V_MOT, (t2[0, later] + 7) % C.V_MOT, t2[0, later])
        alt, _ = m(t2, pos, qf, m.project_speech(sp2, s_pos), s_pos)
        keep = qf <= f
        return bool(torch.equal(base[0, keep], alt[0, keep])), float((base[0, keep] - alt[0, keep]).abs().max()), \
            float((base[0, ~keep] - alt[0, ~keep]).abs().max())

    ok, same_max, later_max = causality(model)
    rep.check("tower", "no_future_speech_or_motion_leak_untrained", ok, max_diff_past=same_max,
              max_diff_future_changed=later_max)
    t0 = time.time()
    prior = []
    if resume and (out / "tower.pt").exists():
        model.load_state_dict(torch.load(out / "tower.pt", weights_only=True))
        prior = json.loads((out / "tower_history.json").read_text()) if (out / "tower_history.json").exists() else []
        _log(f"  tower: resumed tower.pt after {prior[-1]['iter'] if prior else '?'} iterations")
    from dataclasses import replace as _replace
    train_cfg = cfg["tower_train"] if tower_iterations is None else _replace(cfg["tower_train"], iterations=tower_iterations)
    history = train_tower(model, examples, tcfg, train_cfg, device(), seed=seed + len(prior), log=_log,
                          eval_every=100, target_acc=0.97)
    done = prior[-1]["iter"] if prior else 0
    history = prior + [{**h, "iter": h["iter"] + done} for h in history]
    (out / "tower_history.json").write_text(json.dumps(_jsonable(history)) + "\n")
    model.cpu().eval()
    rep.data["timing"]["tower_training_s"] = time.time() - t0
    rep.data["info"]["tower_history"] = history
    acc = evaluate(model, examples, tcfg, "cpu")
    rep.check("tower", "overfit_teacher_forced_token_acc_gt_0.95", acc > 0.95, token_acc=acc,
              examples=[e.name for e in examples], iterations=history[-1]["iter"])
    ok, same_max, later_max = causality(model)
    rep.check("tower", "no_future_speech_or_motion_leak_trained", ok, max_diff_past=same_max,
              max_diff_future_changed=later_max)
    torch.save(model.state_dict(), out / "tower.pt")

    # ---------------- 5. streaming ---------------- #
    _log("== 5. streaming")
    ex = examples[0]
    sample = cfg["sample"]
    nf = min(wf, 20)
    a1, b1, s1 = generate(model, ex.speech, nf, sample, seed=5, partner=ex.a_codes, use_cache=True)
    a2, b2, s2 = generate(model, ex.speech, nf, sample, seed=5, partner=ex.a_codes, use_cache=False)
    logit_diff = max(float((x["logits"] - y["logits"]).abs().max()) for x, y in zip(s1.log, s2.log))
    rep.check("streaming", "kv_cache_equals_recompute", np.array_equal(b1, b2) and logit_diff < 1e-4,
              frames=nf, max_logit_diff=logit_diff)
    sp_changed = ex.speech.copy()
    sp_changed[:, nf // 2:] += np.random.default_rng(3).standard_normal(sp_changed[:, nf // 2:].shape).astype(np.float32)
    partner_changed = ex.a_codes.copy()
    partner_changed[nf // 2:] = examples[1].a_codes[nf // 2: len(partner_changed)]
    a3, b3, s3 = generate(model, sp_changed, nf, sample, seed=5, partner=partner_changed, use_cache=True)
    a4, b4, _ = generate(model, ex.speech, nf // 2, sample, seed=5, partner=ex.a_codes, use_cache=True)
    rep.check("streaming", "past_outputs_unchanged_when_future_inputs_change",
              np.array_equal(b1[: nf // 2], b3[: nf // 2]) and np.array_equal(b1[: nf // 2], b4),
              compared_frames=nf // 2, later_frames_differ=bool(not np.array_equal(b1[nf // 2:], b3[nf // 2:])))
    ab, bb, _ = generate(model, ex.speech, 6, sample, seed=9, partner=None, use_cache=True)
    rep.check("streaming", "both_speaker_mode_in_band", layout.in_band(ab) and layout.in_band(bb) and ab.shape == (6, 22))
    rep.check("streaming", "agent_only_mode_in_band_and_partner_copied", layout.in_band(b1) and np.array_equal(a1, ex.a_codes[:nf]))

    # ---------------- 6. sim ---------------- #
    _log("== 6. Samhita's rig")
    # Demo agent: face tracked in >= 90% of the window, then the highest ground-truth jaw-speech
    # correlation (the stand-in jaw is all-zero where face tracking failed, which makes r meaningless).
    candidates = []
    for k, (user_id, agent_id) in enumerate(roles):
        agent_audio = D.read_wav(people[agent_id].audio)[0]
        env25_full = D.speech_envelope(agent_audio, 16000, 25.0)[2 * f0:2 * f1]
        gt_jaw_full = D.jaw_axis_angle_from_face(people[agent_id].face)[2 * f0:2 * f1, 0]
        m = min(len(env25_full), len(gt_jaw_full))
        face_cov = float(people[agent_id].face_valid[2 * f0:2 * f1].mean())
        r_ref = float(np.corrcoef(env25_full[:m], gt_jaw_full[:m])[0, 1]) if np.std(gt_jaw_full[:m]) > 0 else float("nan")
        candidates.append({"example": examples[k].name, "face_valid": face_cov, "gt_jaw_r": r_ref})
    rep.data["info"]["sim_agent_candidates"] = candidates
    eligible = [k for k, c in enumerate(candidates) if c["face_valid"] >= 0.9 and np.isfinite(c["gt_jaw_r"])]
    pick = max(eligible or range(len(candidates)), key=lambda k: candidates[k]["gt_jaw_r"])
    ex = examples[pick]
    rep.data["info"]["sim_agent"] = candidates[pick]
    t0 = time.time()
    agent_a, agent_b, sess = generate(model, ex.speech, frames, sample, seed=21, partner=ex.a_codes, use_cache=True)
    gen_s = time.time() - t0
    rep.check("streaming", "full_window_generation_in_band", layout.in_band(agent_b), frames=frames)
    t0 = time.time()
    decoded = decode_all(codecs, agent_b)
    dec_s = time.time() - t0
    from . import rig
    from speech_ai.embodiment import CHANNELS, compile_rig
    human, extra = rig.human_channels(decoded, stats)
    teacher = rig.write_teacher(out / "teacher_dyaplex.npz", human)
    user_id, agent_id = roles[pick]
    partner_wav = rig.crop_wav(people[user_id].audio, out / "partner.wav", window_s[0], window_s[1] - window_s[0])
    agent_wav = rig.crop_wav(people[agent_id].audio, out / "agent.wav", window_s[0], window_s[1] - window_s[0])
    t0 = time.time()
    sim = rig.drive_rig(teacher, partner_wav, agent_wav, out / "rig", video=video)
    sim_s = time.time() - t0
    realized = np.load(out / "rig" / "realized-motion.npz")
    cmd, meas = realized["commanded_q"], realized["robot_q"]
    rig_model = compile_rig()
    import mujoco
    lo_hi = []
    for name in CHANNELS:
        j = mujoco.mj_name2id(rig_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        a = next(a for a in range(rig_model.nu) if rig_model.actuator_trnid[a, 0] == j)
        lo_hi.append(rig_model.actuator_ctrlrange[a])
    lo_hi = np.array(lo_hi)
    in_range = bool(((cmd >= lo_hi[:, 0] - 1e-9) & (cmd <= lo_hi[:, 1] + 1e-9)).all())
    rep.check("sim", "27_commands_finite_and_in_range", cmd.shape[1] == 27 and np.isfinite(cmd).all()
              and np.isfinite(meas).all() and in_range, frames=cmd.shape[0], channels=cmd.shape[1])
    moving = {name: float(np.ptp(meas[:, i])) for i, name in enumerate(CHANNELS)}
    rep.data["info"]["rig_range_of_motion"] = moving
    env = D.speech_envelope(D.read_wav(agent_wav)[0], 16000, 20.0)
    jaw = meas[:, list(CHANNELS).index("jaw_open")]
    n = min(len(env), len(jaw))
    r = float(np.corrcoef(env[:n], jaw[:n])[0, 1])
    gt_parts = {p: feats[agent_id][p][2 * f0: 2 * f1] for p in C.PARTS}
    gt_human, _ = rig.human_channels(gt_parts, stats)
    gt_jaw = gt_human[:, 5 + list(__import__("speech_ai.human", fromlist=["AUS"]).AUS).index("jaw_drop")]
    env25 = D.speech_envelope(D.read_wav(agent_wav)[0], 16000, 25.0)
    m = min(len(env25), len(gt_jaw))
    r_gt = float(np.corrcoef(env25[:m], gt_jaw[:m])[0, 1])
    rep.check("sim", "jaw_open_correlates_with_agent_speech", r > 0 and r_gt > 0, pearson_r=r,
              reference_ground_truth_jaw_r=r_gt, video=str(out / "rig" / "rig.mp4") if video else None)
    per_frame = {
        "speech_tower_tiny_ms": rep.data["timing"]["speech_streaming_ms_per_frame"],
        "motion_tower_tiny_ms": 1000 * sess.timing["tower_s"] / frames,
        "decoders_ms": 1000 * dec_s / frames,
        "sim_replay_ms": 1000 * sim_s / frames,
    }
    rep.data["timing"]["per_80ms_frame"] = per_frame
    rep.data["timing"]["generation_s"] = gen_s
    rep.data["timing"]["total_s"] = time.time() - began
    rep.save()
    _log(f"report: {out / 'report.json'}  passed={rep.data['passed']}")
    return rep.data
