"""Small numpy-only toolkit for procedurally generating the head meshes.

* Signed-distance-field (SDF) primitives and smooth boolean operators, used to
  sculpt organic shapes (skull, face, nose, lips, chin ...).
* `radial_surface`: polygonises a star-shaped SDF by ray-casting from a centre
  point over a latitude/longitude grid. The grid doubles as a UV atlas, so a
  texture can be painted directly from 3-D surface positions.
* `shell_patch`: cuts a region out of that grid and gives it thickness, so
  every exported part is a closed, watertight mesh (needed when the jaw opens
  and the inside of a part would otherwise be visible).
* `sweep`: extrudes a varying cross-section along a path (neck, torso, teeth).
* OBJ / PNG writers (no third-party dependencies).
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# SDF primitives (p: (..., 3) array). Approximate distances are fine: the
# mesher relies on the sign and bisection, not on exact distances.
# --------------------------------------------------------------------------- #


def _v(a):
    return np.asarray(a, dtype=float)


def length(v):
    return np.linalg.norm(v, axis=-1)


def sphere(p, c, r):
    return length(p - _v(c)) - r


def ellipsoid(p, c, r):
    r = _v(r)
    q = (p - _v(c)) / r
    k0 = length(q)
    k1 = length(q / r)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-12)


def capsule(p, a, b, r):
    a, b = _v(a), _v(b)
    pa, ba = p - a, b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0.0, 1.0)
    return length(pa - h[..., None] * ba) - r


def round_cone(p, a, b, r1, r2):
    a, b = _v(a), _v(b)
    pa, ba = p - a, b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0.0, 1.0)
    return length(pa - h[..., None] * ba) - (r1 + (r2 - r1) * h)


def rounded_cylinder(p, c, axis, radius, half_height, rounding):
    """Cylinder with rounded rims; axis is 0/1/2 (x/y/z)."""
    q = p - _v(c)
    h = np.abs(q[..., axis])
    others = [i for i in range(3) if i != axis]
    rad = np.sqrt(q[..., others[0]] ** 2 + q[..., others[1]] ** 2)
    dx = rad - (radius - rounding)
    dy = h - (half_height - rounding)
    outside = np.sqrt(np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2)
    inside = np.minimum(np.maximum(dx, dy), 0.0)
    return outside + inside - rounding


def smin(a, b, k):
    """Polynomial smooth union."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def ssub(a, b, k):
    """Smooth subtraction a \\ b."""
    return smax(a, -b, k)


def sdf_normals(sdf, pts, eps=1e-4):
    e = np.eye(3) * eps
    g = np.stack([sdf(pts + e[i]) - sdf(pts - e[i]) for i in range(3)], axis=-1)
    return g / np.maximum(length(g)[..., None], 1e-12)


# --------------------------------------------------------------------------- #
# Radial (star-shaped) polygonisation
# --------------------------------------------------------------------------- #


def directions(az, el):
    A, E = np.meshgrid(az, el)  # (n_el, n_az)
    return np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], axis=-1)


def raycast(sdf, center, dirs, r_max, n_samples=360, n_bisect=22, chunk=1500):
    """Outermost surface hit along each ray (cast inward from r_max)."""
    center = _v(center)
    flat = dirs.reshape(-1, 3)
    t = np.linspace(r_max, 0.0, n_samples)
    hit_r = np.zeros(len(flat))
    for s in range(0, len(flat), chunk):
        d = flat[s : s + chunk]
        pts = center + t[:, None, None] * d[None]
        vals = sdf(pts.reshape(-1, 3)).reshape(n_samples, len(d))
        inside = vals < 0
        if not inside.any(axis=0).all():
            raise ValueError("ray missed the surface; increase r_max or check the centre")
        k = np.argmax(inside, axis=0)
        if (k == 0).any():
            raise ValueError("surface extends beyond r_max")
        lo, hi = t[k], t[k - 1]  # lo inside, hi outside
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            m_in = sdf(center + mid[:, None] * d) < 0
            lo = np.where(m_in, mid, lo)
            hi = np.where(m_in, hi, mid)
        hit_r[s : s + chunk] = 0.5 * (lo + hi)
    return center + hit_r.reshape(dirs.shape[:-1])[..., None] * dirs


class RadialGrid:
    """Surface sampled on a lat/long grid. Index [i, j] = (elevation, azimuth)."""

    def __init__(self, sdf, center, n_az, n_el, r_max, az_range=(-np.pi, np.pi),
                 el_range=(-np.pi / 2 + 0.004, np.pi / 2 - 0.004), wrap=True):
        self.az = np.linspace(*az_range, n_az + 1)
        self.el = np.linspace(*el_range, n_el)
        self.wrap = wrap
        self.center = _v(center)
        self.P = raycast(sdf, center, directions(self.az, self.el), r_max)
        self.N = sdf_normals(sdf, self.P)
        H, W = self.P.shape[:2]
        jj, ii = np.meshgrid(np.arange(W), np.arange(H))
        self.UV = np.stack([jj / (W - 1), ii / (H - 1)], axis=-1)

    @property
    def cell_centers(self):
        P = self.P
        return 0.25 * (P[:-1, :-1] + P[:-1, 1:] + P[1:, :-1] + P[1:, 1:])

    def sample(self, u, v):
        """Bilinear interpolation of surface position/normal at UV coords."""
        H, W = self.P.shape[:2]
        x = np.clip(u * (W - 1), 0, W - 1.0001)
        y = np.clip(v * (H - 1), 0, H - 1.0001)
        j0, i0 = x.astype(int), y.astype(int)
        fx, fy = (x - j0)[..., None], (y - i0)[..., None]

        def lerp(A):
            return ((A[i0, j0] * (1 - fx) + A[i0, j0 + 1] * fx) * (1 - fy)
                    + (A[i0 + 1, j0] * (1 - fx) + A[i0 + 1, j0 + 1] * fx) * fy)

        n = lerp(self.N)
        return lerp(self.P), n / np.maximum(length(n)[..., None], 1e-12)


def shell_patch(grid: RadialGrid, mask, off_out=0.0, off_in=-0.002, origin=(0, 0, 0)):
    """Closed thick-shell mesh of the grid cells selected by `mask` (H-1, W-1).

    Offsets are along the surface normal. Returns (V, N, UV, F) with vertices
    expressed relative to `origin`.
    """
    P, Nrm, UV = grid.P - _v(origin), grid.N, grid.UV
    H, W = P.shape[:2]
    mask = np.asarray(mask, bool)
    vid = np.arange(H * W).reshape(H, W)
    n_grid = H * W

    V = [(P + Nrm * off_out).reshape(-1, 3), (P + Nrm * off_in).reshape(-1, 3)]
    NN = [Nrm.reshape(-1, 3), -Nrm.reshape(-1, 3)]
    T = [UV.reshape(-1, 2), UV.reshape(-1, 2)]

    ci, cj = np.nonzero(mask)
    v00, v01 = vid[ci, cj], vid[ci, cj + 1]
    v10, v11 = vid[ci + 1, cj], vid[ci + 1, cj + 1]
    outer = np.concatenate([np.stack([v00, v01, v11], 1), np.stack([v00, v11, v10], 1)])
    inner = outer[:, ::-1] + n_grid
    faces = [outer, inner]

    # boundary walls -------------------------------------------------------
    nc = W - 1

    def neighbour(di, dj):
        ni, nj = ci + di, cj + dj
        if grid.wrap:
            nj = nj % nc
            valid = (ni >= 0) & (ni < H - 1)
        else:
            valid = (ni >= 0) & (ni < H - 1) & (nj >= 0) & (nj < nc)
        res = np.zeros(len(ci), bool)
        res[valid] = mask[ni[valid], np.clip(nj[valid], 0, nc - 1)]
        return res

    # CCW loop seen from outside: v00 -> v01 -> v11 -> v10
    edges = [((-1, 0), v00, v01), ((0, 1), v01, v11), ((1, 0), v11, v10), ((0, -1), v10, v00)]
    Vout, Vin = V[0], V[1]
    base = 2 * n_grid
    for (di, dj), a, b in edges:
        sel = ~neighbour(di, dj)
        a, b = a[sel], b[sel]
        if not len(a):
            continue
        ao, bo, ai, bi = Vout[a], Vout[b], Vin[a], Vin[b]
        quad = np.stack([ao, ai, bi, bo], 1)  # (n,4,3)
        nrm = np.cross(ai - ao, bi - ao)
        nrm /= np.maximum(length(nrm)[:, None], 1e-15)
        V.append(quad.reshape(-1, 3))
        NN.append(np.repeat(nrm, 4, axis=0))
        T.append(np.repeat(UV.reshape(-1, 2)[a], 4, axis=0))
        k = base + 4 * np.arange(len(a))
        faces.append(np.concatenate([np.stack([k, k + 1, k + 2], 1), np.stack([k, k + 2, k + 3], 1)]))
        base += 4 * len(a)

    return compact(np.concatenate(V), np.concatenate(NN), np.concatenate(T), np.concatenate(faces))


def closed_radial_mesh(sdf, center, n_az, n_el, r_max, origin=(0, 0, 0)):
    """Watertight star-shaped mesh (with pole caps) of an SDF."""
    g = RadialGrid(sdf, center, n_az, n_el, r_max)
    P, N = g.P - _v(origin), g.N
    H, W = P.shape[:2]
    vid = np.arange(H * W).reshape(H, W)
    i, j = np.meshgrid(np.arange(H - 1), np.arange(W - 1), indexing="ij")
    v00, v01, v10, v11 = vid[i, j], vid[i, j + 1], vid[i + 1, j], vid[i + 1, j + 1]
    F = [np.stack([v00, v01, v11], -1).reshape(-1, 3), np.stack([v00, v11, v10], -1).reshape(-1, 3)]
    poles = raycast(sdf, center, np.array([[0, 0, -1.0], [0, 0, 1.0]]), r_max)
    pn = sdf_normals(sdf, poles)
    sp, np_ = H * W, H * W + 1
    js = np.arange(W - 1)
    F.append(np.stack([np.full(W - 1, sp), vid[0, js + 1], vid[0, js]], 1))
    F.append(np.stack([np.full(W - 1, np_), vid[H - 1, js], vid[H - 1, js + 1]], 1))
    V = np.concatenate([P.reshape(-1, 3), poles - _v(origin)])
    NN = np.concatenate([N.reshape(-1, 3), pn])
    UV = np.concatenate([g.UV.reshape(-1, 2), [[0.5, 0.0], [0.5, 1.0]]])
    return compact(V, NN, UV, np.concatenate(F))


def compact(V, N, UV, F):
    used, inv = np.unique(F, return_inverse=True)
    return V[used], N[used], UV[used], inv.reshape(F.shape)


# --------------------------------------------------------------------------- #
# Sweeps (extrusions along a path)
# --------------------------------------------------------------------------- #


def superellipse(n_pts, a, b, expo=2.0):
    t = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    return np.stack([a * np.sign(c) * np.abs(c) ** (2 / expo), b * np.sign(s) * np.abs(s) ** (2 / expo)], -1)


def sweep(path, side, up, sections, cap=True):
    """Extrude closed cross-sections along a path.

    path: (S,3) points; side/up: (S,3) unit vectors spanning each section plane;
    sections: (S,K,2) section coordinates (side, up), CCW when viewed looking
    back along the path direction. Returns (V, N, UV, F) with smooth normals.
    """
    S, K = sections.shape[:2]
    V = path[:, None] + sections[..., :1] * side[:, None] + sections[..., 1:] * up[:, None]
    V = V.reshape(-1, 3)
    vid = np.arange(S * K).reshape(S, K)
    s, k = np.meshgrid(np.arange(S - 1), np.arange(K), indexing="ij")
    a, b = vid[s, k], vid[s, (k + 1) % K]
    c, d = vid[s + 1, (k + 1) % K], vid[s + 1, k]
    F = [np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)]
    UV = np.stack(np.meshgrid(np.linspace(0, 1, K), np.linspace(0, 1, S)), -1).reshape(-1, 2)[:, ::-1]
    N = vertex_normals(V, np.concatenate(F))
    Vs, Ns, UVs = [V], [N], [UV]
    if cap:
        base = len(V)
        for end, flip in ((0, True), (S - 1, False)):
            ring = V.reshape(S, K, 3)[end]
            ctr = ring.mean(0)
            nrm = np.cross(ring[1] - ctr, ring[0] - ctr)
            nrm = nrm / np.linalg.norm(nrm) * (1 if not flip else -1)
            # orient: outward along path direction at the end
            tdir = path[-1] - path[-2] if end == S - 1 else path[0] - path[1]
            if nrm @ tdir < 0:
                nrm = -nrm
            Vs.append(np.vstack([ctr, ring]))
            Ns.append(np.repeat(nrm[None], K + 1, 0))
            UVs.append(np.zeros((K + 1, 2)))
            ks = np.arange(K)
            tri = np.stack([np.full(K, base), base + 1 + ks, base + 1 + (ks + 1) % K], 1)
            e1, e2 = Vs[-1][1] - ctr, Vs[-1][2] - ctr
            if np.cross(e1, e2) @ nrm < 0:
                tri = tri[:, [0, 2, 1]]
            F.append(tri)
            base += K + 1
    return np.concatenate(Vs), np.concatenate(Ns), np.concatenate(UVs), np.concatenate(F)


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for i in range(3):
        np.add.at(N, F[:, i], fn)
    return N / np.maximum(length(N)[:, None], 1e-15)


def spherical_cap(radius, alpha_max, n_r=24, n_t=72, uv_scale=None):
    """Spherical cap around +x (single surface). UV = planar y/z projection."""
    al = np.linspace(0, alpha_max, n_r + 1)[1:]
    th = np.linspace(0, 2 * np.pi, n_t, endpoint=False)
    A, T = np.meshgrid(al, th, indexing="ij")
    d = np.stack([np.cos(A), np.sin(A) * np.cos(T), np.sin(A) * np.sin(T)], -1).reshape(-1, 3)
    d = np.vstack([[1.0, 0, 0], d])
    V = radius * d
    s = uv_scale or radius * np.sin(alpha_max)
    UV = np.stack([0.5 + 0.5 * V[:, 1] / s, 0.5 + 0.5 * V[:, 2] / s], -1)
    vid = (1 + np.arange(n_r * n_t)).reshape(n_r, n_t)
    F = [np.stack([np.zeros(n_t, int), vid[0], np.roll(vid[0], -1)], 1)]
    i, j = np.meshgrid(np.arange(n_r - 1), np.arange(n_t), indexing="ij")
    a, b = vid[i, j], vid[i, (j + 1) % n_t]
    c, dd = vid[i + 1, (j + 1) % n_t], vid[i + 1, j]
    F += [np.stack([a, dd, c], -1).reshape(-1, 3), np.stack([a, c, b], -1).reshape(-1, 3)]
    F = np.concatenate(F)
    # make outward (+radial) facing
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    if (fn * V[F[:, 0]]).sum(1).mean() < 0:
        F = F[:, ::-1]
    return V, d, UV, F


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #


def write_obj(path: Path, V, N=None, UV=None, F=None, header=""):
    with open(path, "w") as f:
        f.write(f"# {header}\n# generated by tools/build_head_assets.py\n")
        f.write("".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in V))
        if UV is not None:
            f.write("".join(f"vt {u:.5f} {v:.5f}\n" for u, v in UV))
        if N is not None:
            f.write("".join(f"vn {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in N))
        F1 = F + 1
        if UV is not None and N is not None:
            f.write("".join(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n" for a, b, c in F1))
        elif N is not None:
            f.write("".join(f"f {a}//{a} {b}//{b} {c}//{c}\n" for a, b, c in F1))
        else:
            f.write("".join(f"f {a} {b} {c}\n" for a, b, c in F1))


def write_png(path: Path, img):
    img = np.ascontiguousarray(np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
                               if img.dtype != np.uint8 else img)
    h, w, c = img.shape
    raw = b"".join(b"\x00" + img[i].tobytes() for i in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    color_type = {3: 2, 4: 6}[c]
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def value_noise3(p, freq, seed=0):
    """Cheap smooth pseudo-noise in [-1, 1] from summed random sinusoids."""
    rng = np.random.default_rng(seed)
    out = np.zeros(p.shape[:-1])
    for _ in range(6):
        k = rng.normal(size=3)
        k = k / np.linalg.norm(k) * freq * rng.uniform(0.7, 1.4)
        out += np.sin(p @ k + rng.uniform(0, 2 * np.pi))
    return out / 6 * 1.8


# --------------------------------------------------------------------------- #
# General triangle-mesh operations: exact clipping by scalar fields + thickening
# --------------------------------------------------------------------------- #


def grid_mesh(grid: "RadialGrid"):
    """Full single-sided triangle mesh of a RadialGrid surface (outward faces)."""
    P, N, UV = grid.P, grid.N, grid.UV
    H, W = P.shape[:2]
    vid = np.arange(H * W).reshape(H, W)
    i, j = np.meshgrid(np.arange(H - 1), np.arange(W - 1), indexing="ij")
    v00, v01, v10, v11 = vid[i, j], vid[i, j + 1], vid[i + 1, j], vid[i + 1, j + 1]
    F = np.concatenate([np.stack([v00, v01, v11], -1).reshape(-1, 3), np.stack([v00, v11, v10], -1).reshape(-1, 3)])
    return P.reshape(-1, 3), N.reshape(-1, 3), UV.reshape(-1, 2), F


def clip(mesh, field):
    """Keep the part of a triangle mesh where field(V) <= 0.

    Triangles crossing the zero level set are cut exactly (linear interpolation
    of position, normal and UV along edges), giving smooth boundaries instead of
    stair-stepped cell boundaries.
    """
    V, N, UV, F = mesh
    f = field(V)
    inside = f <= 0
    nin = inside[F].sum(1)
    keep = [F[nin == 3]]
    cross = F[(nin == 1) | (nin == 2)]
    newV, newN, newUV = [], [], []
    cache: dict[tuple[int, int], int] = {}
    base = len(V)

    def edge_pt(a, b):
        key = (a, b) if a < b else (b, a)
        if key not in cache:
            t = f[a] / (f[a] - f[b])
            newV.append(V[a] + t * (V[b] - V[a]))
            n = N[a] + t * (N[b] - N[a])
            newN.append(n / max(np.linalg.norm(n), 1e-12))
            newUV.append(UV[a] + t * (UV[b] - UV[a]))
            cache[key] = base + len(newV) - 1
        return cache[key]

    tris = []
    for tri in cross:
        ins = inside[tri]
        # rotate so the "odd one out" is first
        k = int(np.argmax(ins)) if ins.sum() == 1 else int(np.argmin(ins))
        a, b, c = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
        ab, ac = edge_pt(a, b), edge_pt(a, c)
        if ins.sum() == 1:          # a inside
            tris.append((a, ab, ac))
        else:                       # a outside, b and c inside
            tris.append((ab, b, c))
            tris.append((ab, c, ac))
    if tris:
        keep.append(np.array(tris))
    if newV:
        V, N, UV = np.vstack([V, newV]), np.vstack([N, newN]), np.vstack([UV, newUV])
    return compact(V, N, UV, np.concatenate(keep))


def merge(*meshes):
    Vs, Ns, UVs, Fs, off = [], [], [], [], 0
    for V, N, UV, F in meshes:
        if len(F) == 0:
            continue
        Vs.append(V); Ns.append(N); UVs.append(UV); Fs.append(F + off)
        off += len(V)
    return np.concatenate(Vs), np.concatenate(Ns), np.concatenate(UVs), np.concatenate(Fs)


def _weld_ids(V, tol=1e-7):
    key = np.round(V / tol).astype(np.int64)
    _, ids = np.unique(key, axis=0, return_inverse=True)
    return ids.reshape(-1)


def thicken(mesh, off_out=0.0, off_in=-0.002, origin=(0, 0, 0), wall_uv=None, attr=None, inner_uv=None):
    """Turn an open surface into a closed shell: offset copies + boundary walls.

    Boundary edges are detected by vertex position, so duplicated seam vertices
    (e.g. the UV seam of a lat/long grid) do not produce walls.
    `attr` (optional, one float per vertex, e.g. a skinning weight) is carried
    through to every generated vertex and returned as a 5th element.
    """
    V, N, UV, F = mesh
    V = V - _v(origin)
    n = len(V)
    Vo, Vi = V + N * off_out, V + N * off_in
    ids = _weld_ids(V)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    ek = np.sort(ids[e], axis=1)
    _, inv, cnt = np.unique(ek, axis=0, return_inverse=True, return_counts=True)
    bnd = e[cnt[inv.reshape(-1)] == 1]          # directed boundary edges (a->b, CCW from outside)
    a, b = bnd[:, 0], bnd[:, 1]
    ao, bo, ai, bi = Vo[a], Vo[b], Vi[a], Vi[b]
    quad = np.stack([ao, ai, bi, bo], 1)
    wn = np.cross(ai - ao, bi - ao)
    wn /= np.maximum(length(wn)[:, None], 1e-15)
    k = 2 * n + 4 * np.arange(len(a))
    Fw = np.concatenate([np.stack([k, k + 1, k + 2], 1), np.stack([k, k + 2, k + 3], 1)])
    Vall = np.concatenate([Vo, Vi, quad.reshape(-1, 3)])
    Nall = np.concatenate([N, -N, np.repeat(wn, 4, 0)])
    wuv = np.repeat(UV[a], 4, 0) if wall_uv is None else np.tile(np.asarray(wall_uv, float), (4 * len(a), 1))
    iuv = UV if inner_uv is None else np.tile(np.asarray(inner_uv, float), (n, 1))
    UVall = np.concatenate([UV, iuv, wuv])
    Fall = np.concatenate([F, F[:, ::-1] + n, Fw])
    if attr is None:
        return compact(Vall, Nall, UVall, Fall)
    attr = np.asarray(attr, float)
    Aall = np.concatenate([attr, attr, np.stack([attr[a], attr[a], attr[b], attr[b]], 1).reshape(-1)])
    V2, N2, UV2, F2 = compact(Vall, Nall, np.column_stack([UVall, Aall]), Fall)
    return V2, N2, UV2[:, :2], F2, UV2[:, 2]


def plane_field(point, normal):
    point, normal = _v(point), _v(normal)
    return lambda V: (V - point) @ normal


def write_skn(path: Path, V, UV, F, bones):
    """MuJoCo binary skin (.skn). bones: list of (body_name, bindpos, bindquat, vertid, vertweight)."""
    V = np.asarray(V, np.float32); UV = np.asarray(UV, np.float32); F = np.asarray(F, np.int32)
    out = [np.array([len(V), len(UV), len(F), len(bones)], np.int32).tobytes(),
           V.tobytes(), UV.tobytes(), F.tobytes()]
    for name, pos, quat, vid, w in bones:
        out.append(name.encode().ljust(40, b"\0"))
        out.append(np.asarray(pos, np.float32).tobytes() + np.asarray(quat, np.float32).tobytes())
        out.append(np.array([len(vid)], np.int32).tobytes())
        out.append(np.asarray(vid, np.int32).tobytes() + np.asarray(w, np.float32).tobytes())
    Path(path).write_bytes(b"".join(out))


def weld(mesh, attr=None, tol=1e-8):
    """Merge vertices that share position AND uv (and attr): removes duplicate
    vertices left along clip cuts so smooth shading is continuous across them."""
    V, N, UV, F = mesh
    key = [np.round(V / tol).astype(np.int64), np.round(UV / 1e-6).astype(np.int64)]
    if attr is not None:
        key.append(np.round(np.asarray(attr) / 1e-6).astype(np.int64)[:, None])
    _, first, inv = np.unique(np.hstack(key), axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    Fw = inv[F]
    Fw = Fw[(Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 0] != Fw[:, 2])]
    out = (V[first], N[first], UV[first], Fw)
    return out if attr is None else out + (np.asarray(attr)[first],)
