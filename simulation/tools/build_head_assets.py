"""Build every mesh and texture used by head.xml.

    ../.venv/bin/python tools/build_head_assets.py          (from simulation/)

The face is sculpted as a signed distance field (skull, mid-face, cheekbones,
brow ridge, nose, philtrum, lips, chin, jawline, eye sockets), polygonised by
radial ray-casting, then split along a plane through the jaw hinge into a
rigid `head` part and a `jaw` part.

All coordinates are in the HEAD body frame (see head.xml):
    +x forward, +y robot's left, +z up; origin ~ centre of the head at eye level.
The constants below MUST match the body positions in head.xml.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import meshlib as ml  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "assets"

# ---------------------------------------------------------------- constants
EYE_RADIUS = 0.012
EYE_L = np.array([0.0705, 0.032, 0.012])          # head.xml: body eye_l pos (recessed)
EYE_R = EYE_L * [1, -1, 1]                        # head.xml: body eye_r pos
# Palpebral fissure (eye opening), in the eye's y-z plane. Asymmetric almond:
# medial canthus slightly lower and wider (caruncle), lateral canthus ~4 deg higher,
# upper-lid peak medial of centre, lower-lid low point lateral of centre.
FISSURE_W_MED, FISSURE_W_LAT = 0.0150, 0.0129
CANTHUS_Z_MED, CANTHUS_Z_LAT = -0.0012, 0.0010
FISSURE_UP, FISSURE_LO = 0.0080, 0.0048
LID_MOUND = EYE_RADIUS + 0.0045     # skin lids follow the eyeball curvature
APERTURE = EYE_RADIUS + 0.0032      # cavity the moving lid slides in
TMJ = np.array([0.008, 0.0, -0.012])              # head.xml: body jaw pos (condyle, ~14 mm in front of ear)
MOUTH_PT = np.array([0.0945, 0.0, -0.0505])       # front of the lip parting line (stomion)
BROW_L = np.array([0.086, 0.033, 0.031])          # head.xml: body brow_l pos
BROW_R = BROW_L * [1, -1, 1]
JAW_BACK = math.radians(-115)                     # rear boundary of jaw region
SEAM = math.atan2(MOUTH_PT[2] - TMJ[2], MOUTH_PT[0] - TMJ[0])  # jaw split plane angle
# Face skinning: one continuous skin over the face. Chin, lower lip and jawline are
# bound 100 % to the jaw bone, the upper face 100 % to the skull, and the cheeks blend
# so they stretch when the mouth opens. The white rear shell is a rigid skull mesh.
CHEEK_BAND = 0.30                                 # rad (about the TMJ) of blended cheek skin
MOUTH_SLIT = (0.012, 0.025)                       # |y|: lips fully part inside / slit tapers closed at the corners
RAY_CENTER = np.array([0.035, 0.0, -0.012])
EAR_L = np.array([-0.010, 0.0755, -0.004])       # auditory module centre (module geoms are in head frame)
EAR_POCKET_R, EAR_MODULE_R = 0.0205, 0.0195       # 1 mm clearance ring around the module

# eyelid coverage (degrees of elevation about the lid hinge, +x = 0)
UPPER_LID = (16.0, 105.0)   # relaxed open pose (lid rests ~1.5 mm over the iris); closed at 0.78 rad

SKIN = np.array([0.80, 0.68, 0.63])   # neutral matte silicone
SHELL = np.array([0.80, 0.81, 0.83])


# ---------------------------------------------------------------- head SDF
def fissure(p, e):
    """Implicit field of the eye opening: <= 0 inside the almond (y-z plane)."""
    q = p - e
    sgn = np.sign(e[1])
    lat = sgn * q[..., 1]                                   # + toward the lateral canthus
    w = np.where(lat >= 0, FISSURE_W_LAT, FISSURE_W_MED)
    u = lat / w
    uc = np.clip(u, -1, 1)
    zc = CANTHUS_Z_MED + (CANTHUS_Z_LAT - CANTHUS_Z_MED) * (uc + 1) / 2
    up = FISSURE_UP * (1 - uc**2) ** 0.55 * (1 - 0.18 * uc)
    lo = FISSURE_LO * (1 - uc**2) ** 0.75 * (1 + 0.15 * uc)
    z = q[..., 2]
    return np.maximum.reduce([z - (zc + up), (zc - lo) - z, (np.abs(u) - 1) * w])


def eye_opening_field(e):
    """Clip field (keep <= 0) that removes the skin in front of the eye inside
    the almond-shaped fissure; the cut edge becomes the eyelid margin."""
    return lambda V: -np.maximum(fissure(V, e), e[0] - V[:, 0])


def smooth_curve(knots):
    """Catmull-Rom spline through (t, value) knots (t ascending) -> f(t)."""
    k = np.asarray(knots, float)
    P = np.vstack([2 * k[0] - k[1], k, 2 * k[-1] - k[-2]])
    ts, vs = [], []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for u in np.linspace(0, 1, 40, endpoint=False):
            q = 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u**2
                       + (-p0 + 3 * p1 - 3 * p2 + p3) * u**3)
            ts.append(q[0]); vs.append(q[1])
    ts.append(k[-1, 0]); vs.append(k[-1, 1])
    ts, vs = np.array(ts), np.array(vs)
    return lambda t: np.interp(t, ts, vs)


# Lower-face design curves (head frame, metres), indexed by height z.
PROFILE = smooth_curve([            # front-most skin x at y = 0 (lips & nose added on top)
    (-0.100, 0.060), (-0.094, 0.074), (-0.088, 0.0845), (-0.080, 0.0895), (-0.072, 0.0905),
    (-0.063, 0.0880), (-0.056, 0.0885), (-0.0505, 0.0905), (-0.042, 0.0912), (-0.030, 0.0895),
    (-0.018, 0.0870), (-0.005, 0.0830), (0.005, 0.0790), (0.015, 0.0740)])
HALF_WIDTH = smooth_curve([         # front-view outline half-width (jawline)
    (-0.100, 0.006), (-0.096, 0.016), (-0.090, 0.028), (-0.080, 0.040), (-0.070, 0.050),
    (-0.058, 0.056), (-0.040, 0.061), (-0.020, 0.065), (0.005, 0.067), (0.015, 0.067)])
CURVATURE = smooth_curve([          # horizontal wrap: x recedes by K * y^2
    (-0.100, 26.0), (-0.085, 20.0), (-0.070, 15.5), (-0.0505, 11.5), (-0.030, 8.8), (-0.010, 7.8), (0.015, 7.5)])
UNDERSIDE = smooth_curve([          # submandibular line: bottom z as a function of x
    (-0.020, -0.052), (-0.005, -0.064), (0.020, -0.082), (0.050, -0.094), (0.080, -0.100)])


def lower_face(p):
    """Mandible + lower mid-face block from the design curves (smooth intersections)."""
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    f_profile = x - (PROFILE(z) - CURVATURE(z) * y**2)
    f_width = np.abs(y) - HALF_WIDTH(z)
    f_under = UNDERSIDE(x) - z
    f_back = -0.014 - x
    f_top = z - 0.014
    f = ml.smax(f_profile, f_width, 0.010)        # rounded cheeks / jaw sides
    f = ml.smax(f, f_under, 0.008)                # rounded jawline & gonial angle
    f = ml.smax(f, f_back, 0.010)
    return ml.smax(f, f_top, 0.020)


def lip(p, center, radii, k_wrap):
    """Lip ellipsoid bent to follow the horizontal curvature of the face."""
    q = p.copy()
    q[..., 0] = p[..., 0] + k_wrap * p[..., 1] ** 2
    return ml.ellipsoid(q, center, radii)


def head_sdf(p):
    # --- skull & upper mid-face
    d = ml.ellipsoid(p, [-0.008, 0, 0.024], [0.097, 0.078, 0.089])            # cranium
    d = ml.smin(d, ml.ellipsoid(p, [0.030, 0, -0.008], [0.056, 0.062, 0.038]), 0.02)  # upper mid-face
    d = ml.smin(d, ml.ellipsoid(p, [0.082, 0, 0.032], [0.012, 0.050, 0.010]), 0.016)  # brow ridge
    for s in (1, -1):
        d = ml.smin(d, ml.ellipsoid(p, [0.059, s * 0.047, -0.007], [0.019, 0.018, 0.013]), 0.020)  # cheekbone
        d = ml.smin(d, ml.ellipsoid(p, [-0.004, s * 0.064, -0.012], [0.03, 0.012, 0.028]), 0.02)   # temple
        d = ml.smin(d, ml.ellipsoid(p, [0.070, s * 0.034, -0.009], [0.010, 0.014, 0.006]), 0.008)  # infra-orbital

    # --- mandible / lower face from profile curves
    d = ml.smin(d, lower_face(p), 0.014)

    # --- nose
    q = p * [1.0, 1.25, 1.0]
    nose = ml.round_cone(q, [0.0855, 0, 0.018], [0.0985, 0, -0.016], 0.0045, 0.0062) / 1.25
    nose = ml.smin(nose, ml.ellipsoid(p, [0.0975, 0, -0.0185], [0.0072, 0.0075, 0.0068]), 0.005)
    for s in (1, -1):
        nose = ml.smin(nose, ml.ellipsoid(p, [0.0895, s * 0.0102, -0.0235], [0.0062, 0.0052, 0.0048]), 0.004)
    d = ml.smin(d, nose, 0.007)

    # --- lips: thin vermilion that follows the dental arch; they meet at the seam (no gap)
    mz = MOUTH_PT[2]
    km = float(CURVATURE(mz))
    d = ml.smin(d, lip(p, [0.0914, 0, mz + 0.0025], [0.0040, 0.0225, 0.0027], km), 0.0045)  # upper lip
    d = ml.smin(d, lip(p, [0.0902, 0, mz - 0.0033], [0.0043, 0.0205, 0.0034], km), 0.0045)  # lower lip

    # --- cylindrical pockets machined into the skull shell for the auditory modules
    for s in (1, -1):
        c = EAR_L * [1, s, 1]
        d = ml.ssub(d, ml.rounded_cylinder(p, c, 1, EAR_POCKET_R, 0.012, 0.0015), 0.0010)

    # --- skin lids draped over the globes (openings cut exactly from the mesh)
    for e in (EYE_L, EYE_R):
        d = ml.smin(d, ml.sphere(p, e, LID_MOUND), 0.006)
    return d


def check_inside(name, V, margin=0.0021):
    """Report whether an internal part stays inside the skin (shell is 2 mm thick)."""
    worst = float(head_sdf(V).max())
    ok = worst < -margin
    print(f"  clearance {name:<14} max sdf {worst*1e3:+6.2f} mm  {'OK' if ok else 'POKES THROUGH SKIN'}")
    return ok


# ---------------------------------------------------------------- regions
def hinge_side(angle):
    """Signed distance to the plane through the jaw hinge axis at `angle`;
    positive for points whose hinge angle lies in (angle, angle + 180 deg)."""
    n = np.array([-math.sin(angle), 0.0, math.cos(angle)])
    return lambda V: (V - TMJ) @ n


def neg(f):
    return lambda V: -f(V)


def _smooth01(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def mouth_factor(V):
    """1 between the lips (where the slit opens), 0 beyond the mouth corners."""
    lo, hi = MOUTH_SLIT
    return _smooth01((hi - np.abs(V[:, 1])) / (hi - lo)) * _smooth01((V[:, 0] - 0.06) / 0.01)


def jaw_weight(V, lower):
    """Skinning weight of the jaw bone (0 = skull, 1 = jaw).

    Smooth across the cheeks (band of CHEEK_BAND rad centred on the lip line),
    hard at the lips: the upper lip stays on the skull, the lower lip follows the jaw.
    """
    th = np.arctan2(V[:, 2] - TMJ[2], V[:, 0] - TMJ[0])
    s = th - SEAM
    w = _smooth01((CHEEK_BAND / 2 - s) / CHEEK_BAND)
    w = w * _smooth01((th - (JAW_BACK - 0.2)) / 0.4)          # fade out behind the ramus
    w = w * _smooth01((face_mask(V) + 0.6) / 8.0)              # zero before the rigid rear shell
    m = mouth_factor(V)
    return w + (1 - w) * m if lower else w * (1 - m)


def split_face(full):
    """Cut the lip slit, then separate the rigid rear shell from the skinned face.

    Returns (shell_mesh, (face_mesh, jaw_weights)).
    """
    upper = ml.clip(full, neg(hinge_side(SEAM)))
    lower = ml.clip(full, hinge_side(SEAM))
    # part the lips along the slit (tiny offset so thicken() does not weld them)
    V = lower[0].copy()
    on_cut = np.abs(hinge_side(SEAM)(V)) < 1e-9
    V[on_cut & (mouth_factor(V) > 0.0)] += 3e-6 * np.array([math.sin(SEAM), 0, -math.cos(SEAM)])
    lower = (V,) + lower[1:]
    shell_f = lambda X: face_mask(X) + 0.6                    # <= 0 on the rear shell
    shells, faces, ws = [], [], []
    for half, is_lower in ((upper, False), (lower, True)):
        shells.append(ml.clip(half, shell_f))
        f = ml.clip(half, neg(shell_f))
        for e in (EYE_L, EYE_R):
            f = ml.clip(f, eye_opening_field(e))
        faces.append(f)
        ws.append(np.clip(jaw_weight(f[0], is_lower), 0, 1))
    *face, w = ml.weld(ml.merge(*faces), np.concatenate(ws))   # re-join the cut except along the lip slit
    return ml.weld(ml.merge(*shells)), (tuple(face), w)


def brow_field(sgn):
    def f(V):
        x, y, z = V[:, 0], V[:, 1], V[:, 2]
        t = (sgn * y - 0.012) / 0.043
        tc = np.clip(t, 0, 1)
        zc = 0.0285 + 0.0062 * np.sin(np.pi * tc**0.85) - 0.003 * tc
        h0 = 0.0016 * (1 - tc) + 0.0006 * tc
        hh = h0 * np.sqrt(np.clip(1 - np.abs(2 * t - 1) ** 5, 0, 1))
        return np.maximum.reduce([np.abs(z - zc) - hh, 0.05 - x, -sgn * y])
    return f


def face_mask(p):
    """1 inside the skin 'face mask', 0 on the rear cranial shell."""
    phi = np.degrees(np.arctan2(np.abs(p[..., 1]), p[..., 0] + 0.005))
    z = p[..., 2]
    zz = np.clip(z / 0.072, 0, 1)
    phimax = 74.0 * np.sqrt(np.maximum(1 - zz**2.2, 0)) + 18.0 * np.clip(-z / 0.03, 0, 1)  # skin wraps the jaw ramus
    phimax = np.where(z > 0.072, -60.0, phimax)
    return phimax - phi  # >0 inside face (degrees)


# ---------------------------------------------------------------- texture
def paint(p, n):
    rgb = np.broadcast_to(SKIN, p.shape).copy()
    rgb += 0.018 * ml.value_noise3(p, 260, seed=1)[..., None]
    rgb += 0.010 * ml.value_noise3(p, 900, seed=2)[..., None] * [1, 0.9, 0.9]
    x, y, z = p[..., 0], p[..., 1], p[..., 2]

    def blend(base, color, w):
        return base * (1 - w[..., None]) + np.asarray(color) * w[..., None]

    def gauss(c, s):
        return np.exp(-(((p - c) / s) ** 2).sum(-1))

    for s in (1, -1):  # cheeks, periorbital shading, nostrils
        rgb = blend(rgb, [0.80, 0.58, 0.54], 0.20 * gauss([0.07, s * 0.045, -0.022], [0.03, 0.02, 0.018]))
        rgb = blend(rgb, [0.70, 0.55, 0.50], 0.15 * gauss(EYE_L * [1, s, 1] + [0.004, 0, 0.002], [0.012, 0.02, 0.013]))
        under = np.clip(-n[..., 2] * 2.0 - 0.8, 0, 1)
        rgb = blend(rgb, [0.32, 0.20, 0.19], 0.9 * under * gauss([0.0930, s * 0.0062, -0.0265], [0.0030, 0.0026, 0.0025]))
    # lips (vermilion)
    zm = MOUTH_PT[2]
    bow = 1 + 0.18 * np.exp(-((np.abs(y) - 0.0045) / 0.0035) ** 2) - 0.22 * np.exp(-(y / 0.0022) ** 2)
    up_h = 0.0050 * np.sqrt(np.clip(1 - (y / 0.0222) ** 2, 0, 1)) * bow
    lo_h = 0.0062 * np.sqrt(np.clip(1 - (y / 0.0205) ** 2, 0, 1))
    up = np.clip(((z - zm) >= -0.0006) * ((up_h - (z - zm)) / 0.0010), 0, 1) * np.clip(up_h / 0.0008, 0, 1)
    lo = np.clip(((zm - z) >= -0.0006) * ((lo_h - (zm - z)) / 0.0013), 0, 1) * np.clip(lo_h / 0.0008, 0, 1)
    lip_w = np.maximum(up, lo) * (x > 0.078)
    rgb = blend(rgb, [0.70, 0.50, 0.49], 0.70 * lip_w)
    part = np.exp(-((z - zm) / 0.00045) ** 2) * np.clip(1 - (y / 0.0235) ** 2, 0, 1) * (x > 0.08)
    rgb = blend(rgb, [0.42, 0.27, 0.27], 0.55 * part)

    # rear shell + seam line
    fm = face_mask(p)
    shell = np.clip(-fm / 0.6, 0, 1)
    rgb = blend(rgb, SHELL + 0.01 * ml.value_noise3(p, 400, 3)[..., None], shell)
    rgb = blend(rgb, [0.22, 0.22, 0.24], np.exp(-((fm + 0.6) / 0.45) ** 2))
    # ambient-occlusion style darkening in creases (from normal facing down/in)
    ao = np.clip(-n[..., 2], 0, 1) * 0.06
    return np.clip(rgb * (1 - ao[..., None]), 0, 1)


def iris_texture(size=512):
    yy, xx = np.mgrid[0:size, 0:size]
    u = (xx + 0.5) / size * 2 - 1
    v = (yy + 0.5) / size * 2 - 1
    r = np.sqrt(u**2 + v**2)
    ang = np.arctan2(v, u)
    rng = np.random.default_rng(7)
    fib = np.zeros_like(r)
    for _ in range(40):
        k = rng.integers(20, 160)
        fib += np.sin(k * ang + rng.uniform(0, 6.28) + rng.uniform(1, 4) * r) * rng.uniform(0.3, 1) / 40
    R_IRIS, R_PUP = 0.86, 0.30
    base = np.array([0.30, 0.42, 0.50])          # grey-blue
    inner = np.array([0.42, 0.36, 0.22])         # warm collarette
    t = np.clip((r - R_PUP) / (R_IRIS - R_PUP), 0, 1)
    col = base * (1 - np.exp(-t * 3))[..., None] + inner * np.exp(-t * 3)[..., None]
    col = col * (1 + 0.55 * fib[..., None])
    col *= (1 - 0.55 * np.clip((r - 0.70) / 0.16, 0, 1) ** 2)[..., None]   # limbal ring
    sclera = np.array([0.96, 0.95, 0.93])
    col = np.where((r > R_IRIS)[..., None], sclera, col)
    edge = np.clip((r - R_IRIS) / 0.04, 0, 1)[..., None]
    col = col * (1 - edge) + sclera * edge
    col = np.where((r < R_PUP)[..., None], 0.02, col)
    pe = np.clip((r - R_PUP) / 0.02, 0, 1)[..., None]
    col = 0.02 * (1 - pe) + col * pe
    return col[::-1]


# ---------------------------------------------------------------- helpers
def save(name, mesh, note=""):
    V, N, UV, F = mesh
    ml.write_obj(OUT / f"{name}.obj", V, N, UV, F, header=f"{name}: {note}")
    print(f"  {name:<20} {len(V):>7} verts {len(F):>7} faces")


def eyelid(th0, th1, r_in, r_out, psi_max=80.0, n_th=48, n_psi=56):
    """Thick spherical-shell eyelid around the eye centre (hinge axis = y)."""
    th = np.radians(np.linspace(th0, th1, n_th + 1))
    ps = np.radians(np.linspace(-psi_max, psi_max, n_psi + 1))
    T, Pp = np.meshgrid(th, ps, indexing="ij")
    d = np.stack([np.cos(Pp) * np.cos(T), np.sin(Pp), np.cos(Pp) * np.sin(T)], -1)

    class G:  # mimic RadialGrid for shell_patch
        wrap = False
    g = G()
    g.P = d * r_out
    g.N = d
    g.UV = np.zeros(d.shape[:2] + (2,))
    mask = np.ones((n_th, n_psi), bool)
    # shell_patch expects (elev, az) ordering with outward = cross(d/daz, d/del);
    # here axis0 = theta (elevation-like), axis1 = psi.
    V, N, UV, F = ml.shell_patch(g, mask, 0.0, r_in - r_out)
    # orientation check: flip if outer faces point inward
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    if (fn * V[F[:, 0]]).sum(1).sum() < 0:
        F = F[:, ::-1]
    return V, N, None, F


def ring_sections(S, fn):
    return np.stack([fn(i / (S - 1)) for i in range(S)])


# ---------------------------------------------------------------- build
def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    print(f"jaw seam plane angle: {math.degrees(SEAM):.1f} deg about the TMJ")

    print("ray-casting head surface ...")
    g = ml.RadialGrid(head_sdf, RAY_CENTER, n_az=400, n_el=280, r_max=0.16)
    print(f"  grid {g.P.shape[1]}x{g.P.shape[0]}  ({time.time()-t0:.1f}s)")
    shell, (face, w_jaw) = split_face(ml.grid_mesh(g))
    # all cut edges (jaw panel seams, eyelid margins) use a dark texel -> panel line / lash line
    W_t, H_t = 2048, 1024
    edge_uv = (0.5, 1.5 / H_t)
    save("head_shell", ml.thicken(shell, 0.0, -0.002, wall_uv=edge_uv), "rigid rear skull shell")
    Vs, Ns, UVs, Fs, Ws = ml.thicken(face, 0.0, -0.002, wall_uv=edge_uv, attr=w_jaw, inner_uv=edge_uv)
    Ws = np.where(Ws < 2e-3, 0.0, np.where(Ws > 1 - 2e-3, 1.0, Ws))
    ih, ij = np.nonzero(Ws < 1)[0], np.nonzero(Ws > 0)[0]
    UVs = np.column_stack([UVs[:, 0], 1.0 - UVs[:, 1]])   # skins use image-row texcoords (no OBJ v-flip)
    ml.write_skn(OUT / "face_skin.skn", Vs, UVs, Fs, [
        ("head", [0, 0, 0], [1, 0, 0, 0], ih, 1 - Ws[ih]),
        ("jaw", TMJ, [1, 0, 0, 0], ij, Ws[ij])])
    print(f"  {'face_skin (skin)':<20} {len(Vs):>7} verts {len(Fs):>7} faces  "
          f"(skull-only {int((Ws == 0).sum())}, jaw-only {int((Ws == 1).sum())}, blended {int(((Ws > 0) & (Ws < 1)).sum())})")

    print("painting face texture ...")
    u = (np.arange(W_t) + 0.5) / W_t
    v = (np.arange(H_t) + 0.5) / H_t
    U, Vv = np.meshgrid(u, v)
    P, Nn = g.sample(U, Vv)
    tex = paint(P, Nn)
    tex[:4, :] = [0.24, 0.12, 0.12]     # reserved strip (never visible): lip/lid edges + inner skin (oral cavity)
    ml.write_png(OUT / "face_skin.png", tex[::-1])   # image row 0 = v = 1 (OBJ convention)

    print("eyebrows ...")
    for name, c, sgn in (("brow_l", BROW_L, 1), ("brow_r", BROW_R, -1)):
        rel = c - RAY_CENTER
        az_c = math.atan2(rel[1], rel[0])
        el_c = math.atan2(rel[2], math.hypot(rel[0], rel[1]))
        gb = ml.RadialGrid(head_sdf, RAY_CENTER, 200, 100, 0.16,
                           az_range=(az_c - 0.55, az_c + 0.55), el_range=(el_c - 0.22, el_c + 0.22), wrap=False)
        brow = ml.clip(ml.grid_mesh(gb), brow_field(sgn))
        save(name, ml.thicken(brow, 0.0006, -0.0012, origin=c), "eyebrow (embedded 1.2 mm so lowering never shows a gap)")

    print("eyes ...")
    r = EYE_RADIUS
    save("eyelid_upper", eyelid(*UPPER_LID, r + 0.0014, APERTURE - 0.0003), "upper lid, open pose")
    alpha = math.radians(30)
    save("iris", ml.spherical_cap(r + 0.00015, alpha, uv_scale=(r + 0.00015) * math.sin(alpha)), "iris disc")
    # cornea: 8 mm radius sphere bulging 1 mm beyond the eyeball
    rc, xc = 0.008, r + 0.001 - 0.008
    meet_x = (r**2 - rc**2 + xc**2) / (2 * xc)          # plane where cornea meets eyeball
    ac = math.acos((meet_x - xc) / rc) + math.radians(2)  # small overlap into the sclera
    Vc, Nc, UVc, Fc = ml.spherical_cap(rc, ac, n_r=20, n_t=72)
    save("cornea", (Vc + [xc, 0, 0], Nc, UVc, Fc), "transparent cornea")
    ml.write_png(OUT / "iris.png", iris_texture())
    # conjunctiva / socket lining behind the globe: fills the canthi (eye-frame coords)
    lining = lambda p: ml.ellipsoid(p, [-0.006, 0, 0], [0.0115, 0.0152, 0.0125])
    save("socket_lining", ml.closed_radial_mesh(lining, [-0.006, 0, 0], 72, 48, 0.03), "socket lining")
    for name, sgn in (("caruncle_l", 1), ("caruncle_r", -1)):   # pink tissue at the medial canthus
        c = np.array([0.0032, -sgn * 0.0131, CANTHUS_Z_MED])
        car = lambda p, c=c: ml.ellipsoid(p, c, [0.0022, 0.0020, 0.0017])
        save(name, ml.closed_radial_mesh(car, c, 32, 20, 0.01), "caruncle")

    print("mouth ...")
    def arch(front_x, half_w, depth, z_top, z_bot, thick, n_teeth=0, scallop=0.0):
        S, K = 64, 28
        ys = np.linspace(-half_w, half_w, S)
        k = depth / half_w**2
        path = np.stack([front_x - k * ys**2, ys, np.full(S, 0.5 * (z_top + z_bot))], 1)
        tang = np.stack([-2 * k * ys, np.ones(S), np.zeros(S)], 1)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True)
        side = np.stack([tang[:, 1], -tang[:, 0], np.zeros(S)], 1)    # outward (forward)
        up = np.tile([0, 0, 1.0], (S, 1))
        hz = 0.5 * (z_top - z_bot)

        def sec(f):
            cs = ml.superellipse(K, thick / 2, hz, 4.0)
            if n_teeth:
                notch = np.abs(np.sin(np.pi * n_teeth * f)) ** 0.35
                lower = cs[:, 1] < 0
                cs[lower, 1] *= (1 - scallop) + scallop * notch
            return cs
        return ml.sweep(path, side, up, ring_sections(S, sec))

    MZ = MOUTH_PT[2]
    save("teeth_upper", arch(0.0873, 0.029, 0.030, MZ + 0.0075, MZ - 0.0012, 0.0034, 12, 0.18), "upper teeth")
    save("teeth_lower", arch(0.0843, 0.027, 0.029, MZ - 0.0016, MZ - 0.0085, 0.0032, 12, 0.18), "lower teeth (head frame)")
    save("gum_upper", arch(0.0866, 0.030, 0.019, MZ + 0.0135, MZ + 0.0060, 0.0042), "upper gum")
    save("gum_lower", arch(0.0835, 0.028, 0.018, MZ - 0.0050, MZ - 0.0120, 0.0040), "lower gum (head frame)")
    save("palate", ml.closed_radial_mesh(lambda p: ml.ellipsoid(p, [0.058, 0, MZ + 0.0040], [0.024, 0.022, 0.009]),
                                         [0.058, 0, MZ + 0.0040], 48, 32, 0.05), "mouth roof")
    save("mouth_floor", ml.closed_radial_mesh(lambda p: ml.ellipsoid(p, [0.055, 0, MZ - 0.0140], [0.024, 0.022, 0.009]),
                                              [0.055, 0, MZ - 0.0140], 48, 32, 0.05), "mouth floor (head frame)")

    def tongue_sdf(p):
        d = ml.ellipsoid(p, [0.058, 0, MZ - 0.0080], [0.022, 0.0170, 0.0052])
        return ml.ssub(d, ml.ellipsoid(p, [0.064, 0, MZ - 0.0026], [0.018, 0.0018, 0.0012]), 0.002)
    save("tongue", ml.closed_radial_mesh(tongue_sdf, [0.058, 0, MZ - 0.0080], 64, 40, 0.05), "tongue (head frame)")

    print("clearance of internal mouth parts (closed jaw) ...")
    for nm in ("teeth_upper", "teeth_lower", "gum_upper", "gum_lower", "palate", "mouth_floor", "tongue"):
        V = np.array([list(map(float, l.split()[1:4])) for l in open(OUT / f"{nm}.obj") if l.startswith("v ")])
        check_inside(nm, V)

    print("neck, torso, ear modules ...")
    def loft(z0, z1, fn, S=40, K=64):
        zs = np.linspace(z0, z1, S)
        path = np.stack([np.zeros(S), np.zeros(S), zs], 1)
        side = np.tile([1.0, 0, 0], (S, 1))
        up = np.tile([0, 1.0, 0], (S, 1))
        return ml.sweep(path, side, up, ring_sections(S, fn))

    # upper neck (neck_pitch_link frame): skin, slightly tapered, sternocleidomastoid fullness
    save("neck_upper", loft(-0.030, 0.105, lambda f: ml.superellipse(64, 0.043 - 0.003 * f, 0.047 - 0.004 * f, 2.2)),
         "upper neck skin")
    # lower neck sleeve (neck_yaw_link frame)
    def bellows(f):
        rib = 1 + 0.035 * np.cos(2 * np.pi * 7 * f) ** 2
        taper = 1.06 - 0.10 * f
        return ml.superellipse(64, 0.045 * rib * taper, 0.049 * rib * taper, 2.3)
    save("neck_sleeve", loft(0.0, 0.064, bellows, S=120), "neck yaw bellows sleeve")

    def bust_sec(f):
        z = f
        a = 0.105 - 0.03 * max(z - 0.6, 0) / 0.4 - 0.045 * max(z - 0.85, 0) / 0.15
        b = 0.20 * (1 - 0.15 * max(z - 0.55, 0) / 0.45) - 0.12 * max(z - 0.80, 0) / 0.20
        return ml.superellipse(96, a, b, 2.6 if z < 0.8 else 2.2)
    save("torso_shell", loft(0.215, 0.385, bust_sec, S=48, K=96), "upper torso shell")

    for name, sgn in (("ear_module_l", 1), ("ear_module_r", -1)):
        c = EAR_L * [1, sgn, 1]
        face_y = c[1] + sgn * 0.0045

        def ear_sdf(p, c=c, face_y=face_y):
            d = ml.rounded_cylinder(p, c, 1, EAR_MODULE_R, 0.0055, 0.0025)
            rad = np.sqrt((p[..., 0] - c[0]) ** 2 + (p[..., 2] - c[2]) ** 2)
            for r in (0.0065, 0.0105, 0.0145):                       # microphone grille rings
                d = ml.ssub(d, np.sqrt((rad - r) ** 2 + (p[..., 1] - face_y) ** 2) - 0.0007, 0.0004)
            return ml.ssub(d, np.maximum(rad - 0.0028, sgn * (face_y - p[..., 1]) - 0.0012), 0.0004)  # centre port
        save(name, ml.closed_radial_mesh(ear_sdf, c, 160, 96, 0.04), "auditory module")

    print(f"done in {time.time()-t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
