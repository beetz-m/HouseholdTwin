"""Closed analytic meshes for anatomical parts (all return watertight trimesh.Trimesh in object coordinates).

Every part of the food twins is built from these primitives, and the tissue label field of each object uses the same
definitions, so geometry, masses and cut sections agree.
"""

import numpy as np
import trimesh

from .geometry import Meridian, revolve


def tm(v, f):
    m = trimesh.Trimesh(np.asarray(v, float), np.asarray(f, np.int64), process=True)
    m.merge_vertices(digits_vertex=10)
    m.remove_unreferenced_vertices()
    if m.volume < 0:
        m.invert()
    return m


def surface_of_revolution(mer, n_theta=160, modulation=None, displacement=None):
    v, f, _, _ = revolve(mer, n_theta, modulation=modulation, displacement=displacement)
    return tm(v, f)


def hollow(outer, *inners):
    """Closed region between an outer closed surface and disjoint inner closed surfaces (inner ones reversed)."""
    parts = [outer.copy()]
    for i in inners:
        if i is None:
            continue
        j = i.copy()
        j.invert()
        parts.append(j)
    return trimesh.util.concatenate(parts)


def uv_sphere(n_u=48, n_v=96):
    u = np.linspace(0, np.pi, n_u + 1)
    v = np.linspace(0, 2 * np.pi, n_v, endpoint=False)
    U, V = np.meshgrid(u[1:-1], v, indexing="ij")
    d = np.stack([np.sin(U) * np.cos(V), np.sin(U) * np.sin(V), np.cos(U)], -1).reshape(-1, 3)
    d = np.vstack([[0, 0, 1], d, [0, 0, -1]])
    f = []
    nr = n_u - 1
    for j in range(n_v):
        f.append((0, 1 + j, 1 + (j + 1) % n_v))
    for i in range(nr - 1):
        for j in range(n_v):
            a, b = 1 + i * n_v + j, 1 + i * n_v + (j + 1) % n_v
            c, e = a + n_v, b + n_v
            f += [(a, c, b), (b, c, e)]
    last = len(d) - 1
    base = 1 + (nr - 1) * n_v
    for j in range(n_v):
        f.append((base + j, last, base + (j + 1) % n_v))
    return d, np.array(f, np.int64)


def radial_surface(centre, r_of_dir, n_u=64, n_v=128):
    """Star-shaped closed surface: centre + r(d) d over unit directions d."""
    d, f = uv_sphere(n_u, n_v)
    r = r_of_dir(d)
    return tm(np.asarray(centre) + d * r[:, None], f)


def ellipsoid(centre, axes, rot=None, n_u=40, n_v=80, taper=None):
    """Ellipsoid with semi-axes (a, b, c) along the rotated local x, y, z; taper(zn) scales x, y (zn in [-1, 1])."""
    d, f = uv_sphere(n_u, n_v)
    p = d * np.asarray(axes, float)
    if taper is not None:
        p[:, :2] *= taper(d[:, 2])[:, None]
    if rot is not None:
        p = p @ np.asarray(rot).T
    return tm(p + np.asarray(centre), f)


def half_ellipsoid(centre, axes, normal, gap=0.0, rot=None, n=64, taper=None):
    """The part of an ellipsoid on the +normal side of the plane through its centre (offset by gap/2), closed by the
    planar face -- e.g. one cotyledon."""
    e = ellipsoid(centre, axes, rot, n // 2 + 16, n, taper)
    nrm = np.asarray(normal, float) / np.linalg.norm(normal)
    return e.slice_plane(np.asarray(centre) + nrm * gap / 2, nrm, cap=True)


def frame_along(path):
    t = np.gradient(path, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-15
    ref = np.array([1.0, 0, 0]) if abs(t[0, 0]) < 0.9 else np.array([0, 1.0, 0])
    n = np.cross(t[0], ref)
    n /= np.linalg.norm(n)
    N, B = [n], [np.cross(t[0], n)]
    for i in range(1, len(path)):                       # parallel transport (no twist)
        n = N[-1] - t[i] * (N[-1] @ t[i])
        n /= np.linalg.norm(n) + 1e-15
        N.append(n)
        B.append(np.cross(t[i], n))
    return t, np.array(N), np.array(B)


def tube(path, radius, n=12, caps=True):
    """Closed tube swept along a polyline; radius scalar or per-point array."""
    path = np.asarray(path, float)
    r = np.broadcast_to(np.asarray(radius, float), (len(path),))
    _, N, B = frame_along(path)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.cos(th)[None, :, None] * N[:, None, :] + np.sin(th)[None, :, None] * B[:, None, :]
    v = (path[:, None, :] + r[:, None, None] * ring).reshape(-1, 3)
    f = []
    for i in range(len(path) - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            f += [(a, b, a + n), (b, b + n, a + n)]
    if caps:
        s, e = len(v), len(v) + 1
        v = np.vstack([v, path[0], path[-1]])
        f += [(s, (j + 1) % n, j) for j in range(n)]
        last = (len(path) - 1) * n
        f += [(e, last + j, last + (j + 1) % n) for j in range(n)]
    return tm(v, np.array(f))


def union_meshes(meshes):
    meshes = [m for m in meshes if m is not None]
    return trimesh.util.concatenate(meshes) if meshes else None


def meridian_region(rho, z):
    """Meridian polyline (top pole -> bottom pole) to a Meridian; resampled for uniform spacing."""
    from .geometry import resample
    r, zz = resample(np.clip(rho, 0, None), z, max(200, len(rho)))
    r[0] = r[-1] = 0.0
    return Meridian(r, zz)


def pappus_volume(mer):
    """Volume of the body of revolution of a meridian (exact for the polyline): V = 2 pi A rho_c."""
    x, y = np.asarray(mer.rho), np.asarray(mer.z)
    xs, ys = np.append(x, x[0]), np.append(y, y[0])
    cr = xs[:-1] * ys[1:] - xs[1:] * ys[:-1]
    A = cr.sum() / 2
    cx = ((xs[:-1] + xs[1:]) * cr).sum() / (6 * A)
    return abs(2 * np.pi * A * cx)


def scaled_rho(mer, s):
    return Meridian(mer.rho * s, mer.z.copy())


def points_on(mesh, n, rng, weight=None):
    """n surface points (area-weighted, optionally re-weighted by weight(p) >= 0) and their normals."""
    p, fi = trimesh.sample.sample_surface(mesh, n * (3 if weight else 1), seed=int(rng.integers(1 << 31)))
    nrm = mesh.face_normals[fi]
    if weight is not None:
        w = weight(p)
        keep = rng.random(len(p)) < w / w.max()
        p, nrm = p[keep][:n], nrm[keep][:n]
    return p, nrm
