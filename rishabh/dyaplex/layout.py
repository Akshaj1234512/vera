"""Dyadic token interleaving, positions and masks (§4.2-4.3, Eq. 1-4).

One frame = [A] a_1..a_K [B] b_1..b_K  (L_step = 2(K+1) = 46 tokens) [P Eq. 1].
Code slot order within a speaker: upper(6) hands(6) lower_trans(6) face(4); body codes are
slots 0-17 and face the last 4 [P Fig. 5 panel labels]; upper/hands/lower order [A from Fig. 2a].
"""

from __future__ import annotations

import numpy as np

from . import config as C

SLOT_PART = np.array([C.PARTS.index(part) for part, _ in C.code_slots()])   # [K] part index per slot
BAND_LO = np.array([C.PART_BAND[p] for p in C.PARTS])
BAND_HI = BAND_LO + C.CODEBOOK_SIZE


def interleave(a_codes: np.ndarray, b_codes: np.ndarray) -> np.ndarray:
    """[F, K] global codes for A and B -> flat [F * 46] token ids."""
    frames = len(a_codes)
    out = np.empty((frames, C.L_STEP), dtype=np.int64)
    out[:, 0] = C.TAG_A
    out[:, 1:C.K + 1] = a_codes
    out[:, C.K + 1] = C.TAG_B
    out[:, C.K + 2:] = b_codes
    return out.reshape(-1)


def deinterleave(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frames = len(tokens) // C.L_STEP
    grid = tokens[: frames * C.L_STEP].reshape(frames, C.L_STEP)
    return grid[:, 1:C.K + 1].copy(), grid[:, C.K + 2:].copy()


def q_pos(length: int, start: int = 0) -> np.ndarray:
    """Frame index of every flat token [P Eq. 2]: floor(t / L_step)."""
    return (np.arange(start, start + length) // C.L_STEP).astype(np.int64)


def token_role(length: int, start: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per flat position: speaker (0=A, 1=B), is_code, slot index (-1 for tags)."""
    within = np.arange(start, start + length) % C.L_STEP
    speaker = (within >= C.K + 1).astype(np.int64)
    local = np.where(speaker == 0, within - 1, within - (C.K + 2))
    is_code = local >= 0
    return speaker, is_code, np.where(is_code, local, -1)


def targets(tokens: np.ndarray, supervise: str = "body", speakers: str = "AB"):
    """Next-token labels over inputs tokens[:-1] -> tokens[1:] [P §4.3].

    Returns (labels, band) for each input position t: labels = tokens[t+1] where t+1 is a
    supervised code position, else -100; band = part index of the target slot (-1 if none).
    """
    length = len(tokens)
    speaker, is_code, slot = token_role(length)
    nxt_speaker, nxt_code, nxt_slot = speaker[1:], is_code[1:], slot[1:]
    limit = C.BODY_CODES if supervise == "body" else C.K
    allowed = np.isin(nxt_speaker, [ "AB".index(s) for s in speakers])
    supervised = nxt_code & (nxt_slot < limit) & allowed
    labels = np.where(supervised, tokens[1:], -100)
    band = np.where(nxt_code, SLOT_PART[np.maximum(nxt_slot, 0)], -1)
    return labels.astype(np.int64), band.astype(np.int64)


def va_labels(length: int, vad_a: np.ndarray, vad_b: np.ndarray, start_frame: int = 0) -> np.ndarray:
    """Speaking state at each code position t in V [P §4.3]: VAD of the speaker who owns the code
    at that frame [A]; -1 at tag positions."""
    speaker, is_code, _ = token_role(length)
    frame = q_pos(length) + start_frame
    lab = np.where(speaker == 0, vad_a[np.minimum(frame, len(vad_a) - 1)], vad_b[np.minimum(frame, len(vad_b) - 1)])
    return np.where(is_code, lab, -1).astype(np.int64)


def band_mask_for(band: int, vocab: int = C.VOCAB) -> np.ndarray:
    """Allowed logits for a target in part `band` [P §4.3 band mask]; tags never allowed."""
    mask = np.zeros(vocab, dtype=bool)
    mask[BAND_LO[band]:BAND_HI[band]] = True
    return mask


def in_band(codes: np.ndarray) -> bool:
    """codes [F, K] -> every slot inside its part band."""
    lo = BAND_LO[SLOT_PART][None, :]
    hi = BAND_HI[SLOT_PART][None, :]
    return bool(((codes >= lo) & (codes < hi)).all())


def window_frames(window_tokens: int) -> int:
    return window_tokens // C.L_STEP
