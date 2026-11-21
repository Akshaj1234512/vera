"""Fit Samhita's scanned human head onto VERA's rig (skin + the parts that hug it).

    ../.venv/bin/python tools/build_scan_face.py         (from simulation/; a few seconds)

Input  (read-only): ../samhita_sim/human_head_skin.skn   (800k-vertex closed head scan)
Output (assets/):   scan_face.skn       face/head skin bound to bones head, jaw, brow_l, brow_r
                    eyelid_upper_scan.obj   hinged upper lid fitted to the scan's eye openings
                    brow_l_scan.obj / brow_r_scan.obj   brow strips cut from the scan forehead (VERA's brow outline)
                    scan_mouth.skn          the scan's own inner lips + oral cavity, as a second (dark) skin
                    neck_upper_scan.obj     neck skin from the scan's neck cut down into the bellows

Everything else in head.xml (eyeballs, teeth, gums, tongue, palate, neck sleeve, torso) still
comes from tools/build_head_assets.py and is used unchanged. Run that script first if you ever
regenerate everything: it does NOT touch the files written here.

Pipeline (all in VERA's HEAD frame: +x forward, +y robot's left, +z up):
  1. place the scan: rigid translation that puts the scan's lip parting line (stomion) on VERA's
     MOUTH_PT. At this placement the scan's lip groove lies on VERA's jaw seam (-23.5 vs -24.0 deg
     about the TMJ) and its eyes sit at VERA's eye height, so VERA's hinge geometry is reused as is.
  2. decimate 800k -> ~150k vertices (vertex clustering on a 1 mm grid; numpy only).
  3. trim the scan's neck/shoulders with an oblique plane under the jaw (mechanical neck below).
  4. remove the inside of the scan's eye-slit "tunnels" where VERA's eyeball and moving lid live.
  5. mouth: the scan already has separate lips (a ~0.2 mm parting) and an oral cavity behind them.
     Lip sides are labelled at full resolution (grown from the clearly-upper / clearly-lower face until
     they meet inside the cavity and at the mouth corners), and the decimation never merges facing
     walls, so the lips part without any cutting. Hidden cavity / inner-lip faces become a
     second, dark skin.
  6. skin weights: head / jaw (VERA's jaw-seam + cheek-band scheme, lips by label), brow_l / brow_r
     (Samhita's brow-bone weights, moved onto VERA's brow bodies).
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_head_assets as bha  # noqa: E402  (VERA constants + eyelid generator; main() is not run)
import meshlib as ml  # noqa: E402

SIM = Path(__file__).resolve().parent.parent
SRC = SIM.parent / "samhita_sim" / "human_head_skin.skn"
OUT = SIM / "assets"

# ---------------------------------------------------------------- placement (measured on the scan)
SCAN_BODY = np.array([0.0, 0.0, 0.34])          # Samhita's `head` body pos: skn vertices are in her world frame
SCAN_STOMION = np.array([0.0841, 0.0, 0.0785])  # lip parting line on the midline, in her head-body frame
OFFSET = bha.MOUTH_PT - SCAN_STOMION - SCAN_BODY  # scan world -> VERA head frame (pure translation)

# eyes: centre of the scan's palpebral fissure (opening y 0.0197..0.0372, z 0.0090..0.0160), deep enough
# that the eyeball clears the scan's shallow lateral orbit (cornea apex ~4 mm behind the lid skin)
EYE_L = np.array([0.0624, 0.0285, 0.0125])      # -> head.xml body eye_l pos
EYE_R = EYE_L * [1, -1, 1]
R = bha.EYE_RADIUS
LID_R_IN, LID_R_OUT = R + 0.0014, R + 0.0026     # moving upper lid shell
# scan's upper lid margin seen from the eye centre: |side angle| (deg) -> elevation (deg) the lid's leading
# edge needs to sit on it (measured on both corners, higher side + 1 deg)
MARGIN_PSI = [0.0, 4.1, 8.2, 12.4, 16.6, 20.9, 25.4, 30.0, 34.8, 45.0]
MARGIN_TH = [13.4, 13.4, 13.5, 11.1, 11.3, 11.6, 10.1, 7.6, 3.0, 3.0]
CAVITY = R + 0.0030                               # scan tunnel walls removed inside this radius

# jaw: VERA's TMJ; the scan's lip groove angle about it, sampled across the mouth (|y| in m -> rad)
TMJ = bha.TMJ
GROOVE_Y = np.array([0.0, 0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.022, 0.024])
GROOVE_TH = np.radians([-23.6, -23.4, -23.6, -24.1, -24.6, -25.2, -25.9, -26.6, -26.9])
MOUTH_SLIT = (0.014, 0.025)    # |y|: lips fully part inside / slit tapers closed at the corners
MENTON = np.array([0.075, -0.092])   # (x, z) chin bottom   } lower border of the mandible: skin below
GONION = np.array([-0.004, -0.068])  # (x, z) jaw angle     } and behind it stays on the skull
CHEEK_BAND = bha.CHEEK_BAND
JAW_BACK = bha.JAW_BACK

# neck trim: oblique plane, just under the throat at the front, under the occiput at the back
NECK_FRONT = np.array([0.035, -0.103])   # (x, z)
NECK_SLOPE = 0.22                          # dz/d(-x): the cut rises toward the back
# head.xml: head body pos="0.0135 0 0.1" on neck_pitch_link (v2 had 0.03): the scan's neck is centred over
# the neck axes, and the neck_sleeve mesh is scaled 1.05 x 1.05 x 0.70 so its rim sits just below the cut.
NECK_PITCH_IN_HEAD = np.array([-0.0135, 0.0, -0.10])   # neck_pitch_link origin in the head frame
SLEEVE_TOP = 0.064 * 0.70 - 0.05 + NECK_PITCH_IN_HEAD[2]  # bellows rim (head frame z)

DECIMATE_CELL = 0.001
BROWS = {"brow_l": (bha.BROW_L, "left_brow_bone"), "brow_r": (bha.BROW_R, "right_brow_bone")}

sm = bha._smooth01


# ---------------------------------------------------------------- io
def read_skn(path):
    b = Path(path).read_bytes()
    nv, nt, nf, nb = np.frombuffer(b, np.int32, 4)
    o = 16
    V = np.frombuffer(b, np.float32, 3 * nv, o).reshape(-1, 3).astype(float); o += 12 * nv
    o += 8 * nt
    F = np.frombuffer(b, np.int32, 3 * nf, o).reshape(-1, 3).copy(); o += 12 * nf
    bones = {}
    for _ in range(nb):
        name = b[o:o + 40].split(b"\0")[0].decode(); o += 40 + 28
        n = int(np.frombuffer(b, np.int32, 1, o)[0]); o += 4
        ids = np.frombuffer(b, np.int32, n, o); o += 4 * n
        w = np.frombuffer(b, np.float32, n, o).astype(float); o += 4 * n
        full = np.zeros(nv)
        full[ids] = w
        bones[name] = full
    return V, F, bones


def read_obj(path):
    V, F = [], []
    for line in open(path):
        if line.startswith("v "):
            V.append([float(t) for t in line.split()[1:4]])
        elif line.startswith("f "):
            F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.array(V), np.array(F)


def save_obj(name, V, F, note):
    ml.write_obj(OUT / f"{name}.obj", V, ml.vertex_normals(V, F), None, F, header=f"{name}: {note}")
    print(f"  {name:<22} {len(V):>7} verts {len(F):>7} faces")


# ---------------------------------------------------------------- mesh helpers
def decimate(V, F, A, h):
    """Vertex clustering: one vertex per occupied h-cube (mean position / attributes).

    Thin pockets in the scan (the 0.2 mm lip parting, the narrow oral cavity) put two facing walls in
    the same cube; merging them would fuse the lips and tear holes in the cavity. So each cube is
    split by surface orientation: vertices facing away from the cube's first vertex form their own
    cluster."""
    cell = np.unique(np.floor(V / h).astype(np.int64), axis=0, return_inverse=True)[1].ravel()
    N = ml.vertex_normals(V, F)
    first = np.full(cell.max() + 1, len(V))
    np.minimum.at(first, cell, np.arange(len(V)))
    back = (N * N[first[cell]]).sum(1) < 0
    key = np.column_stack([cell, back])
    print(f"  {int(np.unique(cell[back]).size)} cubes hold two facing walls (kept apart)")
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = inv.max() + 1
    cnt = np.bincount(inv, minlength=n)[:, None]
    mean = lambda X: np.stack([np.bincount(inv, X[:, k], n) for k in range(X.shape[1])], 1) / cnt
    F = inv[F]
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    _, first = np.unique(np.sort(F, 1), axis=0, return_index=True)
    V, F = mean(V), F[np.sort(first)]
    # clustering folds a few triangles over their neighbours (they render as dark specks): drop them
    for _ in range(3):
        fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-18)
        agree = np.einsum("ij,ij->i", fn, ml.vertex_normals(V, F)[F].mean(1))
        if not (agree < 0.2).any():
            break
        F = F[agree >= 0.2]
    A = mean(A)
    V, F, A, n = fill_small_holes(V, F, A)
    print(f"  filled {n} pinholes left by clustering")
    return V, F, A


def fill_small_holes(V, F, A, max_edges=120, max_perimeter=0.05):
    """Close the small boundary loops clustering leaves (each would show as a dark speck) with a fan
    around the loop centroid. Done before any intentional cut: the scan is watertight, so every loop
    here is a clustering artefact (the largest sit where the lip labels meet inside the mouth corners)."""
    be = boundary_edges(F)
    out = {}
    for a, b in be:
        out.setdefault(int(a), []).append(int(b))
    newV, newA, newF = [], [], []
    for start in list(out):                     # walk loops edge by edge (junction vertices allowed)
        while out.get(start):
            loop, v = [], start
            while out.get(v) and len(loop) <= max_edges:
                loop.append(v)
                v = out[v].pop()
            if v != start or len(loop) < 3 or len(loop) > max_edges:
                continue
            L = np.array(loop)
            if np.linalg.norm(V[np.roll(L, -1)] - V[L], axis=1).sum() > max_perimeter:
                continue
            c = len(V) + len(newV)
            newV.append(V[L].mean(0))
            newA.append(A[L].mean(0))
            newF += [(int(b), int(a), c) for a, b in zip(L, np.roll(L, -1))]   # reverse of the boundary direction
    if not newV:
        return V, F, A, 0
    return np.vstack([V, newV]), np.vstack([F, newF]), np.vstack([A, newA]), len(newV)


def as_mesh(V, F, A):
    return V, ml.vertex_normals(V, F), A, F   # meshlib tuple; attributes ride in the UV slot (clip interpolates them)


def boundary_edges(F):
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    _, inv, cnt = np.unique(np.sort(e, 1), axis=0, return_inverse=True, return_counts=True)
    return e[cnt[inv.ravel()] == 1]       # directed as in their (only) face


def front_depth(V, cell=0.0005):
    """Height field of the front-most surface: x_max over (y, z) cells."""
    key = np.floor(V[:, 1:] / cell).astype(np.int64)
    d = {}
    for (a, b), x in zip(map(tuple, key), V[:, 0]):
        if x > d.get((a, b), -1.0):
            d[(a, b)] = x

    def lookup(P):
        k = np.floor(P[:, 1:] / cell).astype(np.int64)
        out = np.full(len(P), -1.0)
        for i, (a, b) in enumerate(map(tuple, k)):       # 3x3 neighbourhood max, robust to empty cells
            out[i] = max(d.get((a + da, b + db), -1.0) for da in (-1, 0, 1) for db in (-1, 0, 1))
        return out
    return lookup


# ---------------------------------------------------------------- fields
def neck_cut(V):            # <= 0 above the cut (kept)
    zc = NECK_FRONT[1] + NECK_SLOPE * (NECK_FRONT[0] - V[:, 0])
    return zc - V[:, 2]


def eye_cavity(mesh, e):
    """Clip field (<= 0 keeps) removing the scan's eye-slit tunnel where VERA's eyeball and moving
    lid live. The scan's eye openings are narrow pockets running ~35 mm into the head; inside the
    lid cavity their walls (and lid rims) face the eye, while real lid/cheek skin faces away from it.
    Anything inside the eyeball itself is removed as well."""
    V, N = mesh[0], mesh[1]
    d = np.linalg.norm(V - e, axis=1)
    facing = (N * (V - e)).sum(1) / np.maximum(d, 1e-9)
    vals = -np.minimum(np.maximum(d - CAVITY, facing - 0.5), d - (R - 0.0005))
    return lambda X: vals


def groove_theta(V):
    return np.interp(np.abs(V[:, 1]), GROOVE_Y, GROOVE_TH)


def polar(V):
    dx, dz = V[:, 0] - TMJ[0], V[:, 2] - TMJ[2]
    return np.arctan2(dz, dx), np.hypot(dx, dz)


def seam_side(V):           # > 0 above the lip groove surface (signed distance-like, metres)
    th, rho = polar(V)
    return rho * np.sin(th - groove_theta(V))


def mouth_factor(V):
    lo, hi = MOUTH_SLIT
    return sm((hi - np.abs(V[:, 1])) / (hi - lo)) * sm((V[:, 0] - 0.06) / 0.01)


def mandible_mask(V):
    """1 on the skin over the body of the mandible, 0 on the neck below it and behind the ramus.
    The jaw is a pure hinge at the TMJ, so the ramus swings backward as the mouth opens (a real
    condyle slides forward instead); fading the weight out over 25 mm in front of the ramus lets
    that skin stretch rather than fold into the neck."""
    p = V[:, [0, 2]]

    def above(a, b, toward):  # signed distance from line a-b, positive on the side of `toward`
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]])
        n = n if (toward - a) @ n > 0 else -n
        return (p - a) @ n
    inner = np.array([0.04, -0.04])
    return sm((above(MENTON, GONION, inner) + 0.004) / 0.010) * sm(above(GONION, TMJ[[0, 2]], inner) / 0.025)


def jaw_weight(V, lower):
    """VERA's scheme: lower lip/chin 100 % jaw, upper face 100 % skull, cheeks blended."""
    th, _ = polar(V)
    w = sm((CHEEK_BAND / 2 - (th - groove_theta(V))) / CHEEK_BAND)
    w = w * sm((th - (JAW_BACK - 0.2)) / 0.4) * mandible_mask(V)
    m = mouth_factor(V)
    return w + (1 - w) * m if lower else w * (1 - m)


# ---------------------------------------------------------------- build steps
def in_mouth_box(V):
    return (np.abs(V[:, 1]) < 0.04) & (V[:, 2] > -0.085) & (V[:, 2] < -0.02) & (V[:, 0] > 0.04)


def lip_labels(V, F):
    """1 = lower-lip / jaw side, 0 = upper, for the full-resolution scan.

    Far from the lip line (|seam_side| > 6 mm) the side is geometric. Near it the scan's lips are two
    separate surfaces 0.2 mm apart (both on the same side of any smooth cut surface), so labels are
    grown over mesh edges from the clearly-labelled vertices; the two fronts meet deep inside the
    oral cavity and at the mouth corners. Outside the mouth box everything is 0 (no clustering split)."""
    lab = np.zeros(len(V), int)
    box = in_mouth_box(V)
    ss = seam_side(V)
    known = box & (np.abs(ss) > 0.006)
    lab[known] = ss[known] < 0
    todo = box & ~known
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    e = e[box[e[:, 0]] & box[e[:, 1]]]
    done = ~todo
    while True:
        step = e[done[e[:, 0]] & ~done[e[:, 1]]]
        if not len(step):
            break
        dst, first = np.unique(step[:, 1], return_index=True)
        lab[dst] = lab[step[first, 0]]
        done[dst] = True
    cross = e[(lab[e[:, 0]] != lab[e[:, 1]]) & (V[e[:, 0], 0] > 0.085)]
    print(f"  lips labelled: {int((box & (lab == 1)).sum())} lower / {int((box & (lab == 0)).sum())} upper "
          f"vertices in the mouth box; {len(cross) // 2} label-crossing edges in front of the teeth")
    return lab


def eyelid_fit(skin_V):
    """VERA's eyelid generator sized for the scan's (narrower) eye opening; reports any lid
    vertices that come within 0.2 mm of the skin outside the opening, open and closed."""
    depth = front_depth(skin_V[(skin_V[:, 0] > 0.04) & (np.abs(skin_V[:, 2] - EYE_L[2]) < 0.03)])

    def pokes(V):
        n = []
        for q in (0.0, 0.78):
            c, s = math.cos(q), math.sin(q)
            Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])   # hinge about +y, + = close (lid moves down)
            P = V @ Ry.T + EYE_L
            dp = depth(P)
            n.append(int(((P[:, 0] > dp - 2e-4) & (dp > EYE_L[0] + 0.008)).sum()))  # skin in front, lid not behind it
        return n
    # half-width: the scan's opening is +-8.75 mm about the eye centre; cover it with 4 mm to spare so the
    # lid's ends stay hidden in the corners
    psi = math.degrees(math.asin((0.00875 + 0.004) / LID_R_IN))
    lid = bha.eyelid(*bha.UPPER_LID, LID_R_IN, LID_R_OUT, psi_max=psi)
    # VERA's lid edge sits at a constant elevation (a straight line across the eye), but the scan's lid
    # margin is an almond that drops toward both corners; a straight edge cuts across the corners when
    # half closed. Bend the leading edge to follow the measured margin (+1 deg so it hides when open).
    V = lid[0]
    r = np.linalg.norm(V, axis=1)
    sp = np.clip(V[:, 1] / r, -1, 1)
    side, th = np.arcsin(sp), np.arctan2(V[:, 2], V[:, 0])
    edge = np.radians(np.interp(np.degrees(np.abs(side)), MARGIN_PSI, np.minimum.accumulate(MARGIN_TH)))
    th0, th1 = np.radians(bha.UPPER_LID)
    th = edge + (th - th0) * (th1 - edge) / (th1 - th0)
    cp = np.cos(side)
    lid = (r[:, None] * np.column_stack([cp * np.cos(th), sp, cp * np.sin(th)]),) + lid[1:]
    # the scan's orbit is shallow at the canthi: taper the lid's ends 0.9 mm toward the eyeball (still
    # 0.5 mm clear of it) so the longer ends stay under the corner skin
    V = lid[0]
    r = np.linalg.norm(V, axis=1)
    side = np.degrees(np.arctan2(np.abs(V[:, 1]), np.hypot(V[:, 0], V[:, 2])))
    V = V * ((r - 0.0009 * sm((side - 20.0) / (psi - 20.0))) / r)[:, None]   # gentle: no visible facet
    lid = (V,) + lid[1:]
    n = pokes(lid[0])
    print(f"  eyelid half-width {psi:.0f} deg: {n[0]} / {n[1]} vertices may touch the skin (open / closed)")
    return lid[0], lid[3]


def scan_brow(name, centre, sgn, mesh):
    """Eyebrow strip cut straight from the scan surface with VERA's brow outline (brow_field),
    thickened like VERA's (0.6 mm proud, 1.2 mm embedded so lowering never shows a gap)."""
    V, N, _, F = mesh
    patch = ml.clip((V, N, np.zeros((len(V), 2)), F), bha.brow_field(sgn))
    Vb, Nb, _, Fb = ml.thicken(patch, 0.0006, -0.0012, origin=centre)
    ml.write_obj(OUT / f"{name}_scan.obj", Vb, Nb, None, Fb, header=f"{name}_scan: eyebrow on the scan forehead (brow-body frame)")
    print(f"  {name + '_scan':<22} {len(Vb):>7} verts {len(Fb):>7} faces")


def neck_skin(V, F):
    """Skin-coloured neck from the scan's cut edge down into the bellows (neck_pitch_link frame)."""
    be = boundary_edges(F)
    ring = np.unique(be[np.abs(neck_cut(V[be[:, 0]])) < 1e-7])
    J = V[ring]
    c = J[:, :2].mean(0)
    ang = np.arctan2(J[:, 1] - c[1], J[:, 0] - c[0])
    rad = np.hypot(J[:, 1] - c[1], J[:, 0] - c[0])
    o = np.argsort(ang)
    K = 128
    phi = np.linspace(-np.pi, np.pi, K, endpoint=False)
    r_top = np.interp(phi, ang[o], rad[o], period=2 * np.pi)          # meets the skin's cut edge exactly
    top = np.stack([c[0] + r_top * np.cos(phi), c[1] + r_top * np.sin(phi), np.zeros(K)], 1)
    top[:, 2] = NECK_FRONT[1] + NECK_SLOPE * (NECK_FRONT[0] - top[:, 0])
    # straight down to the bellows rim (the rim is ~1 mm wider than the scan's neck), then tapered to 80 % inside the bellows so the
    # hidden part never pokes through the sleeve while nodding
    inward = np.c_[np.cos(phi), np.sin(phi), np.zeros(K)]
    axis = np.array([NECK_PITCH_IN_HEAD[0], 0.0])
    rings = [top + [0, 0, 0.006] - 0.0020 * inward, top + [0, 0, 0.002] - 0.0008 * inward]   # tuck inside the skin
    for t in np.linspace(0, 1, 12):
        rings.append(np.c_[top[:, :2] - 0.0004 * sm(t) * inward[:, :2], top[:, 2] + (SLEEVE_TOP - 0.002 - top[:, 2]) * t])
    for z, k in ((SLEEVE_TOP - 0.012, 0.85), (SLEEVE_TOP - 0.030, 0.80)):
        rings.append(np.c_[axis + (rings[-1][:, :2] - axis) * k, np.full(K, z)])
    P = np.concatenate(rings) - NECK_PITCH_IN_HEAD
    n = len(rings)
    i, j = np.meshgrid(np.arange(n - 1), np.arange(K), indexing="ij")
    v00, v01 = i * K + j, i * K + (j + 1) % K
    v10, v11 = v00 + K, v01 + K
    Fn = np.concatenate([np.stack([v00, v01, v11], -1).reshape(-1, 3), np.stack([v00, v11, v10], -1).reshape(-1, 3)])
    fn = np.cross(P[Fn[:, 1]] - P[Fn[:, 0]], P[Fn[:, 2]] - P[Fn[:, 0]])
    radial = P[Fn[:, 0]] - P.mean(0)
    radial[:, 2] = 0
    if (fn * radial).sum() < 0:
        Fn = Fn[:, ::-1]
    save_obj("neck_upper_scan", P, Fn, "neck skin: scan neck cut -> bellows sleeve (neck_pitch_link frame)")


def write_skin(name, V, F, bones):
    """Write the used part of (V, F) as a skin; bones: (body, bindpos, per-vertex weights over V)."""
    used, inv = np.unique(F, return_inverse=True)
    ml.write_skn(OUT / f"{name}.skn", V[used], np.zeros((0, 2)), inv.reshape(-1, 3),
                 [(nm, pos, [1, 0, 0, 0], np.nonzero(w[used] > 0)[0], w[used][w[used] > 0])
                  for nm, pos, w in bones if (w[used] > 0).any()])     # MuJoCo rejects empty bones
    print(f"  {name + ' (skin)':<22} {len(used):>7} verts {len(F):>7} faces")


def oral_field(V, F, behind=0.0025):
    """Per-vertex field, < 0 on the scan's inner lips and oral cavity: between the mouth corners,
    near the lip line and > 2.5 mm behind the lips (so the closed lips' parting line stays
    skin-coloured). The depth is smoothed over the mesh so the cut between the two skins is a
    smooth curve rather than a staircase of 1 mm triangles."""
    region = np.maximum.reduce([np.abs(V[:, 1]) - 0.027, np.abs(seam_side(V)) - 0.014, 0.05 - V[:, 0]])
    box = in_mouth_box(V)
    h = np.zeros(len(V))
    depth = front_depth(V[box & (V[:, 0] > 0.06)])
    h[box] = depth(V[box]) - V[box, 0]
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    deg = np.bincount(e[:, 0], minlength=len(V)).astype(float)
    for _ in range(4):                                   # Laplacian smoothing of the hidden depth
        h = np.where(box, 0.5 * h + 0.5 * np.bincount(e[:, 0], h[e[:, 1]], len(V)) / np.maximum(deg, 1), h)
    return np.maximum(behind - h, region)


def split(mesh, f):
    """Cut a mesh along the zero set of per-vertex values f, keeping both sides and sharing the new
    vertices on the cut (so the two skins meet without cracks). Returns (V, N, A, F_neg, F_pos)."""
    V, N, A, F = mesh
    neg = f <= 0
    nin = neg[F].sum(1)
    out = {True: [F[nin == 3]], False: [F[nin == 0]]}
    newV, newN, newA, cache = [], [], [], {}

    def edge_pt(a, b):
        key = (a, b) if a < b else (b, a)
        if key not in cache:
            a, b = key
            t = f[a] / (f[a] - f[b])
            newV.append(V[a] + t * (V[b] - V[a]))
            newN.append(N[a] + t * (N[b] - N[a]))
            newA.append(A[a] + t * (A[b] - A[a]))
            cache[key] = len(V) + len(newV) - 1
        return cache[key]
    for tri in F[(nin == 1) | (nin == 2)]:
        lone = bool(neg[tri].sum() == 1)                 # the odd vertex is on the negative side
        k = int(np.argmax(neg[tri])) if lone else int(np.argmin(neg[tri]))
        a, b, c = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
        ab, ac = edge_pt(a, b), edge_pt(a, c)
        out[lone].append(np.array([[a, ab, ac]]))
        out[not lone].append(np.array([[ab, b, c], [ab, c, ac]]))
    stack = lambda X, new: np.vstack([X, new]) if new else X
    return stack(V, newV), stack(N, newN), stack(A, newA), np.concatenate(out[True]), np.concatenate(out[False])


def main():
    t0 = time.time()
    print(f"reading {SRC.relative_to(SIM.parent)} ...")
    V, F, bones = read_skn(SRC)
    V = V + OFFSET
    A = np.stack([bones[BROWS["brow_l"][1]], bones[BROWS["brow_r"][1]]], 1)
    print(f"  {len(V)} verts {len(F)} faces; placed with offset {np.round(OFFSET, 4)}")

    print("labelling lips (full resolution) ...")
    lab = lip_labels(V, F)
    A = np.column_stack([A, lab])
    V, F, A = decimate(V, F, A, DECIMATE_CELL)
    print(f"decimated ({DECIMATE_CELL*1e3:.1f} mm cells): {len(V)} verts {len(F)} faces")

    mesh = ml.clip(as_mesh(V, F, A), neck_cut)
    for e in (EYE_L, EYE_R):
        mesh = ml.clip(mesh, eye_cavity(mesh, e))
    full_V = mesh[0]
    brow_mesh = mesh

    print("skinning ...")
    V, _, A, F_oral, F_face = split(mesh, oral_field(mesh[0], mesh[3]))
    lower = np.where(in_mouth_box(V), A[:, 2] > 0.5, seam_side(V) < 0)
    W = np.where(lower, jaw_weight(V, True), jaw_weight(V, False)).clip(0, 1)
    A = A[:, :2]
    W = np.where(W < 2e-3, 0.0, np.where(W > 1 - 2e-3, 1.0, W))
    A = np.clip(A / A.max(0), 0, 1) * (1 - W)[:, None]   # Samhita's brow weights, peak -> 1 (skin under the brow moves with it)
    A[A < 2e-3] = 0
    H = np.clip(1 - W - A.sum(1), 0, 1)
    bones_out = [("head", [0, 0, 0], H), ("jaw", TMJ, W),
                 ("brow_l", bha.BROW_L, A[:, 0]), ("brow_r", bha.BROW_R, A[:, 1])]
    F = np.vstack([F_face, F_oral])
    # MuJoCo culls skin back faces, so any clustering-flipped triangle in the thin oral cavity wall
    # would be a window to the outside: give the hidden mouth surfaces a reversed copy (double-sided).
    # The copies get their own vertices: MuJoCo averages face normals per skin vertex, and opposite
    # faces sharing vertices would cancel them out.
    deep = oral_field(V, F_face, behind=0.0005)[F_face].mean(1) < 0

    def backside(Fs):
        used, inv = np.unique(Fs, return_inverse=True)
        return used, len(V) + inv.reshape(-1, 3)[:, ::-1]
    used_f, back_f = backside(F_face[deep])
    used_o, back_o = backside(F_oral)
    back_o = back_o + len(used_f)
    extra = np.concatenate([used_f, used_o])
    V = np.vstack([V, V[extra]])
    bones_out = [(nm, pos, np.concatenate([w, w[extra]])) for nm, pos, w in bones_out]
    F_face = np.vstack([F_face, back_f])
    F_oral = np.vstack([F_oral, back_o])
    print(f"  double-sided: {int(deep.sum())} hidden face-skin + {len(back_o)} oral triangles")
    write_skin("scan_face", V, F_face, bones_out)
    print(f"    (skull-only {int((H == 1).sum())}, jaw-only {int((W == 1).sum())}, brow-weighted {int((A.sum(1) > 0).sum())})")
    write_skin("scan_mouth", V, F_oral, bones_out)
    lo, hi = V.min(0), V.max(0)
    print(f"  bounds x {lo[0]:.3f}..{hi[0]:.3f}  y {lo[1]:.3f}..{hi[1]:.3f}  z {lo[2]:.3f}..{hi[2]:.3f}")

    print("eyelids ...")
    Ve, Fe = eyelid_fit(full_V)
    save_obj("eyelid_upper_scan", Ve, Fe, "upper lid, open pose (eye frame)")
    d = np.linalg.norm(full_V - EYE_L, axis=1)
    print(f"  closest scan skin to the eye centre: {d.min()*1e3:.1f} mm (eyeball {R*1e3:.0f}, lid {LID_R_OUT*1e3:.1f})")

    print("eyebrows ...")
    for (name, (centre, _)), sgn in zip(BROWS.items(), (1, -1)):
        scan_brow(name, centre, sgn, brow_mesh)

    print("neck ...")
    neck_skin(V, F)

    print("mouth-part clearance (closed jaw) ...")
    depth = front_depth(full_V[(full_V[:, 0] > 0.03) & (full_V[:, 2] < -0.02) & (full_V[:, 2] > -0.11)])
    for nm in ("teeth_upper", "teeth_lower", "gum_upper", "gum_lower", "palate", "mouth_floor", "tongue"):
        P, _ = read_obj(OUT / f"{nm}.obj")
        gap = depth(P) - P[:, 0]
        print(f"  {nm:<12} min clearance to the skin in front: {gap.min()*1e3:5.1f} mm")
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
