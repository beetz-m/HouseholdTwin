"""Procedural texture helpers. Noise is evaluated at 3D surface points (not in UV space), so patterns keep their
real-world size and do not stretch at the poles of the UV map."""

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


class ValueNoise3:
    """Lattice value noise with quintic interpolation; fbm() sums octaves. Coordinates in metres / cell size."""

    def __init__(self, seed, n=256):
        rng = np.random.default_rng(seed)
        self.v = rng.random(n ** 3).astype(np.float32).reshape(n, n, n)
        self.n = n

    def __call__(self, P):
        P = np.asarray(P, np.float64)
        i = np.floor(P).astype(np.int64)
        f = P - i
        u = _fade(f)
        n = self.n
        out = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = ((u[..., 0] if dx else 1 - u[..., 0]) * (u[..., 1] if dy else 1 - u[..., 1])
                         * (u[..., 2] if dz else 1 - u[..., 2]))
                    out = out + w * self.v[(i[..., 0] + dx) % n, (i[..., 1] + dy) % n, (i[..., 2] + dz) % n]
        return out * 2 - 1

    def fbm(self, P, scale, octaves=4, gain=0.5):
        P = np.asarray(P) / scale
        a, s, tot, norm = 1.0, 1.0, 0.0, 0.0
        for k in range(octaves):
            tot = tot + a * self(P * s + 17.3 * k)
            norm += a
            a *= gain
            s *= 2.0
        return tot / norm


def worley(P, seeds, k=1):
    """Distance to the k-th nearest seed point (cellular noise)."""
    tree = cKDTree(seeds)
    d, _ = tree.query(np.asarray(P).reshape(-1, 3), k=k)
    d = d if k == 1 else d[:, k - 1]
    return d.reshape(np.shape(P)[:-1])


def spots(P, centres, radii, sharp=2.0, k=6):
    """Union of soft discs around centres (k nearest considered); returns coverage in [0, 1]."""
    tree = cKDTree(centres)
    flat = np.asarray(P).reshape(-1, 3)
    d, idx = tree.query(flat, k=min(k, len(centres)))
    if d.ndim == 1:
        d, idx = d[:, None], idx[:, None]
    cov = 1 - np.prod(1 - np.exp(-(d / radii[idx]) ** (2 * sharp)), axis=1)
    return cov.reshape(np.shape(P)[:-1])


def sample_surface(v, f, n, rng):
    """n random points uniformly on a triangle mesh (area-weighted)."""
    a = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    tri = rng.choice(len(f), n, p=a / a.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    return (v[f[tri, 0]] * (1 - s)[:, None] + v[f[tri, 1]] * (s * (1 - r2))[:, None]
            + v[f[tri, 2]] * (s * r2)[:, None])


def save_rgb(path, rgb, gamma=True):
    a = np.clip(rgb, 0, 1)
    if gamma:
        a = a ** (1 / 2.2)
    Image.fromarray((a * 255 + 0.5).astype(np.uint8)).save(path)


def save_gray(path, g):
    Image.fromarray((np.clip(g, 0, 1) * 255 + 0.5).astype(np.uint8), "L").save(path)


def surface_grid(fn_point, w, h):
    """Evaluate fn_point(u, v) -> (h, w, 3) 3D points for a UV texel grid (texel centres)."""
    u = (np.arange(w) + 0.5) / w
    v = 1 - (np.arange(h) + 0.5) / h
    U, V = np.meshgrid(u, v, indexing="xy")
    return fn_point(U, V)


def lerp(a, b, t):
    t = np.asarray(t)[..., None] if np.ndim(t) and np.ndim(a) else t
    return np.asarray(a) * (1 - t) + np.asarray(b) * t


def lch_to_srgb(L, C, h_deg):
    """CIE L*C*h (D65) -> sRGB in [0, 1] (for colours measured with a chromameter, e.g. Ashton 2006)."""
    h = np.radians(h_deg)
    a, b = C * np.cos(h), C * np.sin(h)
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200
    d = 6 / 29

    def finv(t):
        return np.where(t > d, t ** 3, 3 * d * d * (t - 4 / 29))
    X, Y, Z = 0.95047 * finv(fx), 1.0 * finv(fy), 1.08883 * finv(fz)
    M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    rgb = M @ np.array([X, Y, Z])
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.clip(rgb, 0, None) ** (1 / 2.4) - 0.055)
    return np.clip(rgb, 0, 1)
