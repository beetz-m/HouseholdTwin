"""Part graph: the partonomy of a food object as geometry, physics and SDT correspondence.

Every anatomical part is a node of a tree (Part).  Its geometry comes from one of
  ("label", [tissues])   marching cubes of the tissue label field (volumes: yolk, flesh, seed, ...)
  ("instances", tissue)  as "label" but split into connected components (each seed its own part)
  ("mesh", fn)           an analytic closed mesh (thin layers: shell, membranes, skin, seed coat, carpel walls)
  None                   a grouping node (e.g. Albumen groups its layers)
and every part becomes one USD prim under /Food/Parts/... carrying its SDT correspondence:
  sdt:represents   IRI of the physical entity the prim stands for (node != referent, as in the lab twin)
  sdt:class        ontology class of that entity
  sdt:observability, food:volume_cm3, food:mass_g, food:density_kgm3, semantics labels (class, part)
Cuts slice every part mesh by the plane (caps closed), so a half is again a complete part graph.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


@dataclass
class Part:
    key: str
    path: str                      # relative USD path below /Food/Parts, e.g. "Contents/Yolk/Latebra"
    cls: str                       # ontology class (food: namespace, see docs/food_ontology)
    geom: tuple = None             # ("label", [...]) | ("instances", name) | ("mesh", fn) | None
    material: str = None
    observability: str = "hidden"  # visible | tactile | weighed | inferred | hidden
    doc: str = ""
    density: float = None          # kg/m3 for analytic meshes (label parts use the tissue table)
    interior: bool = True          # inside the closed outer surface (not seen on the whole object)
    attrs: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------------------------ meshes
def label_volume(body, h, pad=4):
    lo, hi = body.bbox()
    lo, hi = lo - pad * h, hi + pad * h
    ax = [np.arange(lo[i], hi[i] + h, h) for i in range(3)]
    lab = np.zeros([len(a) for a in ax], np.int16)
    X, Y = np.meshgrid(ax[0], ax[1], indexing="ij")
    for k, z in enumerate(ax[2]):
        P = np.stack([X, Y, np.full_like(X, z)], -1).reshape(-1, 3)
        lab[:, :, k] = body.label(P).reshape(X.shape)
    return lab, np.array([a[0] for a in ax]), h


def mc_mesh(mask, origin, h, sigma=0.8, target_faces=60000, smooth=0):
    import trimesh
    from skimage import measure
    if mask.sum() < 8:
        return None
    nz = np.nonzero(mask)                                 # crop to the mask's bounding box (+ margin): speed
    lo = [max(int(a.min()) - 4, 0) for a in nz]
    hi = [int(a.max()) + 5 for a in nz]
    mask = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    origin = np.asarray(origin, float) + np.array(lo) * h
    m = ndimage.gaussian_filter(mask.astype(np.float32), sigma)
    v, f, _, _ = measure.marching_cubes(m, 0.5)
    mesh = trimesh.Trimesh(v * h + origin, f[:, ::-1], process=True)
    if len(mesh.faces) > target_faces:                   # quadric decimation can break manifoldness: verify
        for tf in (target_faces, 2 * target_faces, 4 * target_faces):
            if tf >= len(mesh.faces):
                break
            try:
                d = mesh.simplify_quadric_decimation(face_count=tf)
            except ImportError:                          # no fast_simplification (Isaac env): keep the full mesh
                break
            if d.is_watertight and d.is_winding_consistent:
                mesh = d
                break
    if smooth:                                            # Taubin: removes marching-cubes terracing, keeps volume
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=int(smooth))
    trimesh.repair.fix_normals(mesh)
    return mesh


def closed(v, f):
    import trimesh
    """Weld duplicated seam/pole vertices of a parametric surface into a watertight trimesh."""
    m = trimesh.Trimesh(np.asarray(v), np.asarray(f), process=True)
    m.merge_vertices(digits_vertex=9)
    m.remove_unreferenced_vertices()
    return m


def shell(outer, inner):
    """Closed thin layer between two nested closed surfaces (inner reversed)."""
    import trimesh
    o = outer.copy()
    i = inner.copy()
    i.invert()
    return trimesh.util.concatenate([o, i])


# ------------------------------------------------------------------------------------------------ graph
class PartGraph:
    def __init__(self, kind, body, parts, tissues):
        self.kind, self.body, self.parts = kind, body, parts
        self.tissues = {t["name"]: (i + 1, t) for i, t in enumerate(tissues)}
        self.meshes = {}            # path -> trimesh (instances expanded)
        self.info = {}              # path -> dict(part, volume, mass, ...)

    def build(self, h=0.3e-3, sigma=0.8, smooth=0):
        self.points = {}
        if any(p.geom and p.geom[0] in ("label", "instances") for p in self.parts):
            lab, org, h = label_volume(self.body, h)
            self.lab, self.org, self.h = lab, org, h
        for p in self.parts:
            if p.geom is None:
                self.info[p.path] = {"part": p, "group": True}
                continue
            kind = p.geom[0]
            if kind == "label":
                ids = [self.tissues[n][0] for n in p.geom[1]]
                mask = np.isin(lab, ids)
                m = mc_mesh(mask, org, h, sigma, smooth=smooth)
                vol = mask.sum() * h ** 3
                rho = np.mean([self.tissues[n][1]["density"] for n in p.geom[1]])
                self._add(p.path, p, m, vol, rho)
            elif kind == "instances":
                tid = self.tissues[p.geom[1]][0]
                comp, n = ndimage.label(lab == tid)
                sizes = ndimage.sum(np.ones_like(comp), comp, range(1, n + 1))
                order = np.argsort(-sizes)
                k = 0
                rho = self.tissues[p.geom[1]][1]["density"]
                for j in order:
                    if sizes[j] < 20:
                        continue
                    k += 1
                    mask = comp == j + 1
                    m = mc_mesh(mask, org, h, 0.6)
                    self._add(f"{p.path}_{k:02d}", p, m, mask.sum() * h ** 3, rho, instance=k)
            elif kind == "points":
                self.points[p.path] = p.geom[1]()
                self.info[p.path] = {"part": p, "points": len(self.points[p.path][0])}
            elif kind == "mesh":
                m = p.geom[1]()
                self._add(p.path, p, m, abs(m.volume) if m is not None else 0.0, p.density or 0.0)
        return self

    def _add(self, path, part, mesh, vol, rho, instance=None):
        self.meshes[path] = mesh
        self.info[path] = {"part": part, "volume_cm3": vol * 1e6, "density": rho, "mass_g": vol * rho * 1e3,
                           "instance": instance, "faces": 0 if mesh is None else len(mesh.faces)}

    def mass_g(self):
        return sum(i.get("mass_g", 0.0) for i in self.info.values())

    # ------------------------------------------------------------ cuts
    def cut(self, origin, normal):
        """Slice every part by the plane. Returns two dicts path -> (trimesh, cap_face_mask) for the +normal side
        and the -normal side.  A plane through the axis passes through the pole vertices of surfaces of revolution;
        the plane is nudged by micrometres until every slice is a closed volume."""
        n = np.asarray(normal, float) / np.linalg.norm(normal)
        out = ({}, {})
        self.cut_failures = []
        for path, m in self.meshes.items():
            if m is None:
                continue
            res = self._split_manifold(m, np.asarray(origin, float), n)
            if res is not None:
                for side in (0, 1):
                    if res[side] is not None:
                        out[side][path] = res[side]
                continue
            for eps in (1.7e-6, -2.9e-6, 4.3e-6, -6.1e-6, 9.7e-6, 3.1e-5, -7.3e-5, 1.6e-4, -3.1e-4):
                o = np.asarray(origin, float) + eps * n
                res, ok = self._slice(m, o, n)
                if ok:
                    break
            if not ok:
                self.cut_failures.append(path)
            for side in (0, 1):
                if res[side] is not None:
                    out[side][path] = res[side]
        return out

    @staticmethod
    def _split_manifold(m, o, n):
        """Exact plane split with manifold3d (closed volumes, nested shells as cavities). None if the mesh is not a
        valid manifold; the caller then falls back to trimesh slicing with nudged planes."""
        try:
            import manifold3d as M3
            import trimesh
        except ImportError:
            return None
        if not m.is_watertight:
            return None
        man = M3.Manifold(M3.Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                                  tri_verts=np.asarray(m.faces, np.uint32)))
        if man.status() != M3.Error.NoError or man.is_empty():
            return None
        pos, neg = man.split_by_plane(tuple(float(x) for x in n), float(np.asarray(o) @ n))
        res = []
        for part, sgn in ((pos, 1.0), (neg, -1.0)):
            if part.is_empty():
                res.append(None)
                continue
            mm = part.to_mesh()
            s = trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3].astype(float), np.asarray(mm.tri_verts),
                                process=False)               # manifold output is exact; merging would break it
            c = s.triangles_center
            cap = (np.abs((c - o) @ n) < 2e-7) & (np.abs(s.face_normals @ (-sgn * n)) > 0.999)
            res.append((s, cap))
        return res

    @staticmethod
    def _slice(m, o, n):
        d = (m.vertices - o) @ n
        res, ok = [None, None], True
        for side, sgn in ((0, 1.0), (1, -1.0)):
            if (sgn * d).max() <= 0:
                continue
            if (sgn * d).min() >= 0:
                res[side] = (m.copy(), np.zeros(len(m.faces), bool))
                continue
            try:
                s = m.slice_plane(o, sgn * n, cap=True)
            except Exception:
                s = None
            if s is None or len(s.faces) == 0 or (m.is_volume and not s.is_volume):
                ok = False
                if s is None or len(s.faces) == 0:
                    continue
            c = s.triangles_center
            cap = (np.abs((c - o) @ n) < 1e-7) & (np.abs(s.face_normals @ (-sgn * n)) > 0.999)
            res[side] = (s, cap)
        return res, ok
