"""Geometry kernel for the food twins: bodies of revolution with azimuthal modulation, tissue labelling, cuts.

Every food object is described in its own frame (metres, +z = the object's long axis, origin at the geometric centre)
by a *meridian*: a polyline (rho, z) from the top pole to the bottom pole, rho >= 0, and an optional radial
modulation m(theta, z) (lobes, lopsidedness).  The outer surface is

    S(theta, t) = (m(theta, z(t)) * rho(t) * cos(theta), m(theta, z(t)) * rho(t) * sin(theta), z(t)).

Because m depends on z, not on t, a point p is inside the body iff (rho_p / m(theta_p, z_p), z_p) lies inside the 2D
meridian region -- an exact, vectorised point-in-body test.  That one test labels tissues (each tissue is a region
defined in the same normalised coordinates), integrates mass properties, and paints cut faces, so geometry,
inertia and cross-sections come from one model and cannot disagree.
"""

import numpy as np
from matplotlib.path import Path
from scipy.interpolate import PchipInterpolator


# ------------------------------------------------------------------------------------------------ meridians
def resample(rho, z, n):
    """Resample a polyline to n points uniform in arc length."""
    p = np.stack([rho, z], 1)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    u = np.linspace(0, s[-1], n)
    return np.interp(u, s, rho), np.interp(u, s, z)


def smooth_meridian(ctrl, n=400):
    """PCHIP spline through control points (rho, z) ordered top pole -> bottom pole, parametrised by arc length."""
    c = np.asarray(ctrl, float)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    t = np.linspace(0, s[-1], 4 * n)
    rho = PchipInterpolator(s, c[:, 0])(t)
    z = PchipInterpolator(s, c[:, 1])(t)
    rho[0] = rho[-1] = 0.0
    return resample(np.clip(rho, 0, None), z, n)


class Meridian:
    """2D region {(rho, z)} bounded by the meridian and the axis; fast vectorised containment."""

    def __init__(self, rho, z):
        self.rho, self.z = np.asarray(rho, float), np.asarray(z, float)
        # closed polygon: meridian then back along the axis (rho = 0) -- mirrored copy makes it robust at rho ~ 0
        poly = np.concatenate([np.stack([self.rho, self.z], 1), np.stack([-self.rho[::-1], self.z[::-1]], 1)])
        self.path = Path(poly)

    def contains(self, rho, z):
        pts = np.stack([np.ravel(rho), np.ravel(z)], 1)
        return self.path.contains_points(pts).reshape(np.shape(rho))

    def offset(self, d):
        """Inward offset by d (metres) along the 2D normal; endpoints stay on the axis."""
        p = np.stack([self.rho, self.z], 1)
        tng = np.gradient(p, axis=0)
        tng /= np.linalg.norm(tng, axis=1, keepdims=True) + 1e-12
        nrm = np.stack([-tng[:, 1], tng[:, 0]], 1)        # meridian runs top->bottom on rho>0: (-dz, drho) points out
        q = p - d * nrm
        q[:, 0] = np.clip(q[:, 0], 0, None)
        return Meridian(q[:, 0], q[:, 1])

    @property
    def zmin(self):
        return float(self.z.min())

    @property
    def zmax(self):
        return float(self.z.max())

    def radius_at(self, z):
        """Outer radius at height z (max rho over meridian crossings)."""
        out = np.zeros_like(np.asarray(z, float))
        for i in range(len(self.z) - 1):
            z0, z1, r0, r1 = self.z[i], self.z[i + 1], self.rho[i], self.rho[i + 1]
            lo, hi = min(z0, z1), max(z0, z1)
            m = (z >= lo) & (z <= hi) & (hi > lo)
            r = r0 + (r1 - r0) * (z - z0) / np.where(z1 != z0, z1 - z0, 1)
            out = np.where(m, np.maximum(out, r), out)
        return out


# ------------------------------------------------------------------------------------------------ surfaces
def revolve(mer, n_theta=256, modulation=None, displacement=None, theta0=0.0, theta_span=2 * np.pi):
    """Mesh S(theta, t). Returns verts (N,3), faces (M,3), uv (N,2) with u = theta/2pi, v = arc fraction.

    The poles are single vertices (no slivers), the seam is duplicated for UVs.
    displacement(theta, t_frac, verts, normals) -> (N,) metres along the outward normal (pebbled skin, etc.).
    """
    closed = abs(theta_span - 2 * np.pi) < 1e-9
    nt = n_theta + 1                                     # duplicate seam (or keep both cut edges)
    th = theta0 + np.linspace(0, theta_span, nt)
    rho, z = mer.rho[1:-1], mer.z[1:-1]
    nv = len(rho)
    T, R = np.meshgrid(th, rho, indexing="xy")           # (nv, nt)
    Z = np.repeat(z[:, None], nt, 1)
    M = modulation(T, Z) if modulation else 1.0
    X, Y = M * R * np.cos(T), M * R * np.sin(T)
    verts = np.stack([X, Y, Z], -1).reshape(-1, 3)
    tfrac = np.linspace(0, 1, len(mer.rho))[1:-1]
    uv = np.stack([np.tile(th / (2 * np.pi), nv), np.repeat(1 - tfrac, nt)], 1)   # absolute theta: halves share the texture
    top = np.array([[0, 0, mer.z[0]]])
    bot = np.array([[0, 0, mer.z[-1]]])
    i_top, i_bot = len(verts), len(verts) + 1
    verts = np.concatenate([verts, top, bot])
    uv = np.concatenate([uv, [[0.5, 1.0], [0.5, 0.0]]])
    f = []
    for i in range(nv - 1):
        a = i * nt + np.arange(nt - 1)
        b = a + nt
        f.append(np.stack([a, b, a + 1], 1))
        f.append(np.stack([a + 1, b, b + 1], 1))
    a = np.arange(nt - 1)
    f.append(np.stack([np.full(nt - 1, i_top), a, a + 1], 1))
    a = (nv - 1) * nt + np.arange(nt - 1)
    f.append(np.stack([a, np.full(nt - 1, i_bot), a + 1], 1))
    faces = np.concatenate(f).astype(np.int64)
    faces = faces[:, [0, 2, 1]]                          # outward winding (right-handed, theta CCW about +z)
    if displacement is not None:
        n = vertex_normals(verts, faces)
        if np.mean(np.einsum("ij,ij->i", n, verts - verts.mean(0))) < 0:   # displace along the OUTWARD normal
            n = -n
        d = displacement(np.concatenate([T.ravel(), [0, 0]]),
                         np.concatenate([np.tile(tfrac[:, None], (1, nt)).ravel(), [0, 1]]), verts, n)
        d[-2:] = 0
        verts = verts + n * d[:, None]
        if closed:                                       # seam copies must stay identical (watertight after welding)
            grid = verts[:-2].reshape(nv, nt, 3)
            grid[:, -1] = grid[:, 0]
            verts[:-2] = grid.reshape(-1, 3)
    return verts, faces, uv, closed


def vertex_normals(v, f):
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-15)


def ellipsoid(c, r, n=48, rot=None):
    """Closed UV-sphere mesh scaled to semi-axes r, rotated by rot (3x3), centred at c."""
    u = np.linspace(0, np.pi, n // 2 + 1)
    v = np.linspace(0, 2 * np.pi, n + 1)
    U, V = np.meshgrid(u, v, indexing="ij")
    p = np.stack([np.sin(U) * np.cos(V), np.sin(U) * np.sin(V), np.cos(U)], -1).reshape(-1, 3) * np.asarray(r)
    if rot is not None:
        p = p @ np.asarray(rot).T
    p = p + np.asarray(c)
    f = []
    W = n + 1
    for i in range(n // 2):
        for j in range(n):
            a, b = i * W + j, (i + 1) * W + j
            if i > 0:
                f.append((a, a + 1, b))
            if i < n // 2 - 1:
                f.append((a + 1, b + 1, b))
    uv = np.stack([V.ravel() / (2 * np.pi), 1 - U.ravel() / np.pi], 1)
    return p, np.array(f, np.int64), uv


# ------------------------------------------------------------------------------------------------ labelling
class Body:
    """A food body: outer meridian + modulation + an ordered tissue labeller.

    label(points) -> int array (0 = outside, k = index into self.tissues + 1).  Subclasses supply `tissue_of(P, inside)`
    working on normalised cylindrical coordinates (rho_n = rho / m, theta, z).
    """

    tissues = []          # list of dicts: name, density (kg/m^3), colour (for schematics)

    def __init__(self, mer, modulation=None):
        self.mer = mer
        self.mod = modulation

    def norm_coords(self, P):
        P = np.asarray(P, float)
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        th = np.arctan2(y, x)
        rho = np.hypot(x, y)
        m = self.mod(th, z) if self.mod else 1.0
        return rho / m, th, z

    def inside(self, P):
        rn, th, z = self.norm_coords(P)
        return self.mer.contains(rn, z)

    def label(self, P):
        rn, th, z = self.norm_coords(P)
        ins = self.mer.contains(rn, z)
        out = np.zeros(np.shape(rn), np.int16)
        if ins.any():
            out[ins] = self.tissue_of(rn[ins], th[ins], z[ins], np.asarray(P)[ins])
        return out

    def tissue_of(self, rn, th, z, P):          # pragma: no cover - abstract
        raise NotImplementedError

    def bbox(self):
        r = self.mer.rho.max() * (1.15 if self.mod else 1.0)
        return np.array([-r, -r, self.mer.zmin]), np.array([r, r, self.mer.zmax])

    # ---------------------------------------------------------------- mass properties by voxel integration
    def mass_properties(self, h=0.4e-3, densities=None, clip=None):
        """Mass, centre of mass, inertia tensor about the COM, per-tissue volume/mass. Voxel midpoint rule."""
        lo, hi = self.bbox()
        ax = [np.arange(lo[i] + h / 2, hi[i], h) for i in range(3)]
        dens = np.array([0.0] + [densities.get(t["name"], t["density"]) if densities else t["density"]
                                 for t in self.tissues])
        mass, mom1, mom2 = 0.0, np.zeros(3), np.zeros((3, 3))
        vol = np.zeros(len(dens))
        dv = h ** 3
        for z in ax[2]:                                   # slab by slab keeps memory flat
            X, Y = np.meshgrid(ax[0], ax[1], indexing="ij")
            P = np.stack([X, Y, np.full_like(X, z)], -1).reshape(-1, 3)
            lab = self.label(P)
            if clip is not None:
                lab = np.where(clip(P), lab, 0)
            np.add.at(vol, lab, dv)
            w = dens[lab] * dv
            sel = w > 0
            if not sel.any():
                continue
            p, w = P[sel], w[sel]
            mass += w.sum()
            mom1 += (p * w[:, None]).sum(0)
            mom2 += np.einsum("n,ni,nj->ij", w, p, p)
        com = mom1 / mass
        second = mom2 - mass * np.outer(com, com)
        inertia = np.trace(second) * np.eye(3) - second
        per = {t["name"]: {"volume_cm3": vol[i + 1] * 1e6, "mass_g": vol[i + 1] * dens[i + 1] * 1e3}
               for i, t in enumerate(self.tissues)}
        return {"mass_kg": mass, "com_m": com, "inertia_kgm2": inertia, "volume_cm3": vol[1:].sum() * 1e6,
                "tissues": per}

    # ---------------------------------------------------------------- planar sections
    def section(self, origin, e1, e2, extent, res=1024):
        """Label image of the plane origin + a*e1 + b*e2, a,b in [-extent, extent]. Returns (labels, P grid)."""
        a = np.linspace(-extent, extent, res)
        A, B = np.meshgrid(a, a[::-1], indexing="xy")
        P = np.asarray(origin) + A[..., None] * np.asarray(e1) + B[..., None] * np.asarray(e2)
        return self.label(P.reshape(-1, 3)).reshape(res, res), P


# ------------------------------------------------------------------------------------------------ mesh ops
def cap_polygon_mesh(outline_2d, z_plane_to_3d):
    """Triangulate a 2D outline (closed, (N,2)) and lift it to 3D with the given function."""
    import trimesh
    from shapely.geometry import Polygon
    poly = outline_2d if hasattr(outline_2d, "exterior") else Polygon(outline_2d).buffer(0)
    v2, f = trimesh.creation.triangulate_polygon(poly, engine="earcut")
    return z_plane_to_3d(v2), f.astype(np.int64), v2
