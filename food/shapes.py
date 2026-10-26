"""Shape tools for food bodies that are not plain bodies of revolution (wave 1: roll, banana, carrot).

  Bend        exact bend map of a straight body onto a circular arc (banana finger, a curved carrot): a straight
              model in (x, y, z) is bent about an axis parallel to y; unbend() inverts it exactly, so the same
              point-in-body test and tissue labeller work on the bent object, and meshes are bent vertex by vertex.
  DepthField  depth below the outer surface (Euclidean distance transform on a voxel grid, trilinear lookup) for
              bodies defined by an implicit inside test (the bread roll): crust thickness, crust/crumb transition.
  StarUV      texture parametrisation of a star-shaped body: direction from a centre -> (u, v), with the surface point
              found by bisection on the inside test (the roll's crust texture).
  implicit_mesh  closed mesh of {inside(P)} by marching cubes (shares the label definitions, as partgraph's "label").
"""

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator


class Bend:
    """Bend the straight frame (axis +z) onto an arc of radius R in the x-z plane, centre of curvature at x = +R.

    Straight point (x, y, z) -> bent (R - (R - x) cos(z/R), y, (R - x) sin(z/R)); the axis (x = 0) keeps its arc
    length, the concave side is +x.  Exact inverse: phi = atan2(Z, R - X), x = R - hypot(R - X, Z), z = R phi.
    """

    def __init__(self, R):
        self.R = float(R)

    def bend(self, P):
        P = np.asarray(P, float)
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        a = z / self.R
        r = self.R - x
        return np.stack([self.R - r * np.cos(a), y, r * np.sin(a)], -1)

    def unbend(self, P):
        P = np.asarray(P, float)
        X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
        dx = self.R - X
        return np.stack([self.R - np.hypot(dx, Z), Y, self.R * np.arctan2(Z, dx)], -1)

    def bend_mesh(self, m):
        """Bend vertex by vertex.  Meshes must have short edges along the axis (a long flat triangle stays a chord of
        the arc): build bent parts from lofts / dense surfaces, not from planar slices."""
        out = m.copy()
        out.vertices = self.bend(np.asarray(m.vertices))
        return out


class DepthField:
    """Depth below the surface of {inside(P)} (metres; 0 outside), from a distance transform on a voxel grid."""

    def __init__(self, inside, lo, hi, h=0.25e-3, pad=3):
        lo, hi = np.asarray(lo) - pad * h, np.asarray(hi) + pad * h
        self.ax = [np.arange(lo[i], hi[i] + h, h) for i in range(3)]
        X, Y = np.meshgrid(self.ax[0], self.ax[1], indexing="ij")
        mask = np.zeros([len(a) for a in self.ax], bool)
        for k, z in enumerate(self.ax[2]):
            P = np.stack([X, Y, np.full_like(X, z)], -1).reshape(-1, 3)
            mask[:, :, k] = inside(P).reshape(X.shape)
        d = ndimage.distance_transform_edt(mask, sampling=h).astype(np.float32)
        self.mask, self.h = mask, h
        self._f = RegularGridInterpolator(self.ax, d, bounds_error=False, fill_value=0.0)

    def __call__(self, P):
        P = np.asarray(P, float)
        return self._f(P.reshape(-1, 3)).reshape(P.shape[:-1])


class StarUV:
    """u = azimuth / 2 pi, v = 1 - polar / pi of the direction from `centre`; surface by bisection on inside()."""

    def __init__(self, inside, centre, r_max):
        self.inside, self.c, self.r_max = inside, np.asarray(centre, float), r_max

    def dirs(self, U, V):
        th, pol = 2 * np.pi * np.asarray(U), np.pi * (1 - np.asarray(V))
        return np.stack([np.sin(pol) * np.cos(th), np.sin(pol) * np.sin(th), np.cos(pol)], -1)

    def point_uv(self, U, V, it=30):
        d = self.dirs(U, V).reshape(-1, 3)
        lo, hi = np.zeros(len(d)), np.full(len(d), self.r_max)
        for _ in range(it):
            mid = (lo + hi) / 2
            ok = self.inside(self.c + d * mid[:, None])
            lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
        return (self.c + d * lo[:, None]).reshape(np.shape(U) + (3,))

    def uv_of(self, P):
        d = np.asarray(P, float) - self.c
        r = np.linalg.norm(d, axis=-1) + 1e-15
        pol = np.arccos(np.clip(d[..., 2] / r, -1, 1))
        return np.stack([np.mod(np.arctan2(d[..., 1], d[..., 0]), 2 * np.pi) / (2 * np.pi), 1 - pol / np.pi], -1)


def implicit_mesh(inside, lo, hi, h=0.3e-3, sigma=0.8, target_faces=80000):
    """Closed mesh of the region inside(P) (marching cubes of the smoothed indicator, as partgraph.mc_mesh)."""
    from .partgraph import mc_mesh
    lo, hi = np.asarray(lo) - 4 * h, np.asarray(hi) + 4 * h
    ax = [np.arange(lo[i], hi[i] + h, h) for i in range(3)]
    X, Y = np.meshgrid(ax[0], ax[1], indexing="ij")
    mask = np.zeros([len(a) for a in ax], bool)
    for k, z in enumerate(ax[2]):
        P = np.stack([X, Y, np.full_like(X, z)], -1).reshape(-1, 3)
        mask[:, :, k] = inside(P).reshape(X.shape)
    return mc_mesh(mask, np.array([a[0] for a in ax]), h, sigma, target_faces)


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def loft(rings):
    """Closed mesh through a stack of closed polygons (each (n, 3), same n, consistent orientation), with fan caps at
    the polygon centroids of the first and last ring."""
    from .meshlib import tm
    R = np.asarray(rings, float)
    m, n = R.shape[:2]
    V = np.vstack([R.reshape(-1, 3), R[0].mean(0), R[-1].mean(0)])
    f = []
    for i in range(m - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            f += [(a, b, a + n), (b, b + n, a + n)]
    s, e = m * n, m * n + 1
    f += [(s, (j + 1) % n, j) for j in range(n)]
    last = (m - 1) * n
    f += [(e, last + j, last + (j + 1) % n) for j in range(n)]
    return tm(V, np.array(f))


# ------------------------------------------------------------------------------------------------ revolved regions
def revolve_polygon(poly, n_theta=96, theta0=0.0, theta1=2 * np.pi, axis_tol=1e-7):
    """Closed mesh of a 2D region (polygon in the (rho, z) half-plane, rho >= 0, closed, any orientation) revolved
    about z over [theta0, theta1].  Vertices on the axis become single pole vertices; a partial sweep is closed by
    two planar side faces.  The same polygon (matplotlib Path) labels points, so mesh and label field agree."""
    from .geometry import cap_polygon_mesh
    from .meshlib import tm
    P = np.asarray(poly, float)
    keep = np.linalg.norm(P - np.roll(P, 1, axis=0), axis=1) > 1e-9     # drop repeated points (incl. closing one)
    P = P[keep]
    P[P[:, 0] < axis_tol, 0] = 0.0
    area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    if area < 0:                                         # counter-clockwise in (rho, z)
        P = P[::-1]
    full = abs((theta1 - theta0) - 2 * np.pi) < 1e-9
    nt = n_theta if full else n_theta + 1
    th = theta0 + (theta1 - theta0) * np.arange(nt) / (n_theta if full else n_theta)
    on_axis = P[:, 0] < axis_tol
    V, idx = [], []
    for (r, z), ax in zip(P, on_axis):
        if ax:
            idx.append([len(V)] * nt)
            V.append([0.0, 0.0, z])
        else:
            idx.append(list(range(len(V), len(V) + nt)))
            V += [[r * np.cos(t), r * np.sin(t), z] for t in th]
    F = []
    n = len(P)
    m = nt if full else nt - 1
    for i in range(n):
        a, b = idx[i], idx[(i + 1) % n]
        for j in range(m):
            j1 = (j + 1) % nt
            q = [a[j], a[j1], b[j1], b[j]]
            for tri in ((q[0], q[1], q[2]), (q[0], q[2], q[3])):
                if len(set(tri)) == 3:
                    F.append(tri)
    if not full:                                         # planar side faces at theta0 and theta1
        for t, k, flip in ((theta0, 0, False), (theta1, nt - 1, True)):
            v2, f2, _ = cap_polygon_mesh(P, lambda q: q)
            amap = {}
            for vi, (r, z) in enumerate(v2):
                d = np.linalg.norm(P - [r, z], axis=1)
                pi = int(np.argmin(d))
                amap[vi] = idx[pi][k] if d[pi] < 1e-12 else None
            extra = {}
            for vi, ref in amap.items():
                if ref is None:
                    extra[vi] = len(V)
                    V.append([v2[vi][0] * np.cos(t), v2[vi][0] * np.sin(t), v2[vi][1]])
            for f in f2:
                tri = [amap[v] if amap[v] is not None else extra[v] for v in f]
                F.append(tuple(tri[::-1]) if flip else tuple(tri))
    m_ = tm(np.array(V), np.array(F))
    return m_


def region_contains(poly, rho, z):
    from matplotlib.path import Path
    pts = np.stack([np.ravel(rho), np.ravel(z)], 1)
    return Path(np.asarray(poly)).contains_points(pts).reshape(np.shape(rho))


def offset_curve(rho, z, d):
    """Offset an open meridian curve (top -> bottom, rho >= 0) inwards by d (along the 2D normal)."""
    p = np.stack([rho, z], 1)
    t = np.gradient(p, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    q = p - d * nrm
    q[:, 0] = np.clip(q[:, 0], 0, None)
    return q[:, 0], q[:, 1]


def angle_in(th, a0, a1):
    """theta inside the sector [a0, a1] (radians, a1 > a0, width < 2 pi), wrapping."""
    return np.mod(np.asarray(th) - a0, 2 * np.pi) <= (a1 - a0)
