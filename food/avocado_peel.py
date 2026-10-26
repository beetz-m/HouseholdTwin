"""Peeling outcomes of a halved avocado (protocol: halve, remove the seed, peel each half) as USD states of the twin.

A peeled half is derived from the half's part graph (assets/food/<instance>/half_b.usdc, the seed cavity half):
  - the exocarp (cuticle .. stone-cell layer) is removed except where *skin patches* stay attached: each patch is
    the set of exocarp faces inside an ellipse on the surface (jagged edge, like a real torn remnant);
  - *gouges* (flesh torn out with the skin) are ellipsoids subtracted from the mesocarp bands and vascular strands
    (manifold3d booleans); their new faces form a separate GougeSurface prim per band, labelled
    ``avocado_flesh_damaged`` so that the semantic segmentation gives the ground truth of missing flesh.
Ground truth (truth layer + JSON): remaining skin area, gouge surface and its projection, lost flesh volume.
Outcome sizes and counts are ILLUSTRATIVE (no measured peeling-defect statistics are held).
"""

import json
import math
import os
from dataclasses import asdict, dataclass, field

import manifold3d as m3
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdSemantics, Vt

EXOCARP = ["Cuticle", "Epidermis", "Hypodermis", "SkinParenchyma", "StoneCellLayer"]
FLESH = ["OuterDarkGreenBand", "MiddlePaleGreenBand", "InnerYellowBand"]


@dataclass
class Defect:
    centre: list           # m, on the peeled flesh surface
    normal: list           # outward surface normal
    radii: list            # m: two tangent semi-axes and the depth semi-axis
    kind: str              # "skin_patch" or "gouge"


@dataclass
class PeelOutcome:
    instance: str
    half: str
    seed: int
    category: str          # clean / skin_patches / gouges / mixed
    defects: list = field(default_factory=list)


# %% surface sampling
def mesh_arrays(prim):
    m = UsdGeom.Mesh(prim)
    P = np.asarray(m.GetPointsAttr().Get(), float)
    F = np.asarray(m.GetFaceVertexIndicesAttr().Get(), np.int64).reshape(-1, 3)
    return P, F


def cut_faces(prim):
    s = [g for g in UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Mesh(prim)) if g.GetPrim().GetName() == "CutFace"]
    return set(np.asarray(s[0].GetIndicesAttr().Get(), np.int64).tolist()) if s else set()


def outer_surface(stage):
    """Triangles of the peeled flesh surface: faces of the outer flesh band that face outwards (away from the
    seed cavity) and are not on the cut plane. Returns centroids, unit normals, areas."""
    prim = stage.GetPrimAtPath("/Food/Parts/Mesocarp/OuterDarkGreenBand/Geom")
    P, F = mesh_arrays(prim)
    cut = cut_faces(prim)
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    n = n / np.maximum(2 * area[:, None], 1e-15)
    cen = (a + b + c) / 3
    radial = cen - np.array([0.0, 0.0, cen[:, 2].mean()]) * np.array([0, 0, 1.0])
    radial[:, 2] *= 0.3                                       # mostly sideways on the elongated fruit
    out = (np.einsum("ij,ij->i", n, radial) > 0) & ~np.isin(np.arange(len(F)), list(cut))
    # keep the outer sheet only: the band also has an inner sheet facing the middle band
    rr = np.linalg.norm(cen[:, :2], axis=1)
    out &= rr > np.percentile(rr[out], 35) if out.any() else out
    return cen[out], n[out], area[out]


def sample_outcome(instance, half, seed, category, stage):
    """Draw a peeling outcome of the given category on this half's outer surface."""
    rng = np.random.default_rng(seed)
    cen, nrm, area = outer_surface(stage)
    w = area / area.sum()
    defects = []
    n_patch = {"clean": 0, "skin_patches": rng.integers(1, 4), "gouges": 0, "mixed": rng.integers(1, 3)}[category]
    n_gouge = {"clean": 0, "skin_patches": 0, "gouges": rng.integers(1, 4), "mixed": rng.integers(1, 3)}[category]
    for kind, n in (("skin_patch", n_patch), ("gouge", n_gouge)):
        for _ in range(int(n)):
            for _try in range(50):
                i = rng.choice(len(cen), p=w)
                if all(np.linalg.norm(cen[i] - np.array(d.centre)) > 1.6 * max(d.radii[:2]) + 0.006 for d in defects):
                    break
            if kind == "skin_patch":                          # remnant 4..14 mm across
                r = [rng.uniform(2e-3, 7e-3), rng.uniform(2e-3, 7e-3), 4e-3]
            else:                                             # gouge 6..20 mm across, 1.5..5 mm deep
                r = [rng.uniform(3e-3, 10e-3), rng.uniform(3e-3, 10e-3), rng.uniform(1.5e-3, 5e-3)]
            defects.append(Defect(cen[i].tolist(), nrm[i].tolist(), r, kind))
    return PeelOutcome(instance, half, int(seed), category, defects)


# %% geometry
def frame(normal):
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    t = np.cross(n, [0, 0, 1.0])
    if np.linalg.norm(t) < 1e-6:
        t = np.cross(n, [1.0, 0, 0])
    t /= np.linalg.norm(t)
    return np.c_[t, np.cross(n, t), n]                        # columns: tangent u, tangent v, normal


def inside_patch(X, d):
    """Points inside the patch's elliptic cylinder (tangent ellipse, any depth within 4 mm of the surface)."""
    R = frame(d.normal)
    q = (X - np.asarray(d.centre)) @ R
    return (q[:, 0] / d.radii[0]) ** 2 + (q[:, 1] / d.radii[1]) ** 2 <= 1.0


def ellipsoid(d, segments=48):
    R = frame(d.normal)
    M = np.eye(4)
    M[:3, :3] = R @ np.diag(d.radii)
    M[:3, 3] = d.centre
    return m3.Manifold.sphere(1.0, segments).transform(M[:3, :]).as_original()


def to_manifold(P, F):
    mesh = m3.Mesh(vert_properties=np.asarray(P, np.float32), tri_verts=np.asarray(F, np.uint32),
                   face_id=np.arange(len(F), dtype=np.uint32))
    mesh.merge()
    return m3.Manifold(mesh)


# %% USD state
def subset_faces(prim, keep):
    """Keep only the faces ``keep`` (bool per face) of a triangle mesh prim, with its face-varying st and the
    CutFace subset."""
    m = UsdGeom.Mesh(prim)
    P, F = mesh_arrays(prim)
    idx = np.nonzero(keep)[0]
    used = np.unique(F[idx].ravel())
    remap = -np.ones(len(P), np.int64)
    remap[used] = np.arange(len(used))
    m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(P[used].astype(np.float32)))
    m.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(idx)))
    m.GetFaceVertexIndicesAttr().Set(Vt.IntArray(remap[F[idx]].ravel().tolist()))
    pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
    if pv and pv.GetInterpolation() == UsdGeom.Tokens.faceVarying:
        st = np.asarray(pv.Get(), np.float32).reshape(-1, 3, 2)[idx].reshape(-1, 2)
        pv.Set(Vt.Vec2fArray.FromNumpy(st))
    na = m.GetNormalsAttr()                                   # per-corner normals follow the kept faces (a count
    if na.HasAuthoredValue():                                 # mismatch makes the renderer drop the material)
        nv = np.asarray(na.Get(), np.float32)
        if m.GetNormalsInterpolation() == UsdGeom.Tokens.faceVarying and len(nv) == 3 * len(F):
            na.Set(Vt.Vec3fArray.FromNumpy(nv.reshape(-1, 3, 3)[idx].reshape(-1, 3)))
        elif len(nv) == len(P):
            na.Set(Vt.Vec3fArray.FromNumpy(nv[used]))
        else:
            prim.RemoveProperty("normals")
    m.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([P[used].min(0), P[used].max(0)], np.float32)))
    old = {int(k): i for i, k in enumerate(idx)}
    for s in UsdGeom.Subset.GetAllGeomSubsets(m):
        keep_ids = [old[i] for i in s.GetIndicesAttr().Get() if int(i) in old]
        if keep_ids:
            s.GetIndicesAttr().Set(Vt.IntArray(keep_ids))
        else:                           # an empty materialBind subset makes RTX drop the mesh's own material
            prim.GetStage().RemovePrim(s.GetPath())
    return idx


def carried_uv(P0, F0, uv0, P, F):
    """Face-varying st of the new triangles F (vertices P), each interpolated barycentrically (weights clamped to the
    triangle) in its source triangle of the original mesh (P0, F0, face-varying uv0): the original triangle with the
    nearest centroid (manifold's face ids are not reliable for coplanar or re-triangulated faces). Exact for
    unchanged triangles, close for re-triangulated ones; the cut face is mapped separately (carve)."""
    from scipy.spatial import cKDTree
    src = cKDTree(P0[F0].mean(1)).query(P[F].mean(1))[1]
    a, b, c = (P0[F0[src, k]] for k in range(3))
    e1, e2 = b - a, c - a
    d00, d01, d11 = (e1 * e1).sum(1), (e1 * e2).sum(1), (e2 * e2).sum(1)
    den = np.where(np.abs(d00 * d11 - d01 ** 2) < 1e-30, 1e-30, d00 * d11 - d01 ** 2)
    U = uv0.reshape(-1, 3, 2)[src]
    out = np.zeros((len(F), 3, 2))
    for k in range(3):
        w = P[F[:, k]] - a
        d20, d21 = (w * e1).sum(1), (w * e2).sum(1)
        v = np.clip((d11 * d20 - d01 * d21) / den, 0, 1)
        u = np.clip((d00 * d21 - d01 * d20) / den, 0, 1)
        s_ = np.maximum(u + v, 1.0)
        v, u = v / s_, u / s_
        out[:, k] = (1 - v - u)[:, None] * U[:, 0] + v[:, None] * U[:, 1] + u[:, None] * U[:, 2]
    return out.reshape(-1, 2)


def set_mesh(prim, P, F, cut_ids, st=None):
    """Replace the mesh of ``prim`` (new topology): face-varying st (``st``, or zeros), smooth vertex normals with
    hard face normals on the cut face, extent, CutFace subset."""
    m = UsdGeom.Mesh(prim)
    P, F = np.asarray(P, float), np.asarray(F, np.int64)
    m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(P.astype(np.float32)))
    m.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(F)))
    m.GetFaceVertexIndicesAttr().Set(Vt.IntArray(F.ravel().tolist()))
    pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
    if pv:
        pv.Set(Vt.Vec2fArray.FromNumpy(np.asarray(st if st is not None else np.zeros((len(F) * 3, 2)), np.float32)))
    fn = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])   # area-weighted
    vn = np.zeros_like(P)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-15)
    cn = vn[F].copy()
    ids = np.asarray(sorted(cut_ids), np.int64)
    if len(ids):
        cn[ids] = (fn[ids] / np.maximum(np.linalg.norm(fn[ids], axis=1, keepdims=True), 1e-15))[:, None, :]
    m.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(cn.reshape(-1, 3).astype(np.float32)))
    m.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    m.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([P.min(0), P.max(0)], np.float32)))
    for s in UsdGeom.Subset.GetAllGeomSubsets(m):
        if len(ids):
            s.GetIndicesAttr().Set(Vt.IntArray(ids.tolist()))
        else:                           # an empty materialBind subset makes RTX drop the mesh's own material
            prim.GetStage().RemovePrim(s.GetPath())


def tri_area(P, F):
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def absolutize_textures(st, src_dir):
    """Texture asset paths made absolute (a derived state is written elsewhere than the asset it copies)."""
    from pxr import UsdShade
    for prim in st.Traverse():
        if prim.IsA(UsdShade.Shader):
            for inp in UsdShade.Shader(prim).GetInputs():
                v = inp.Get()
                if isinstance(v, Sdf.AssetPath) and v.path and not os.path.isabs(v.path):
                    inp.Set(Sdf.AssetPath(os.path.normpath(os.path.join(src_dir, v.path))))


def flesh_paths(st, with_skin=False, with_seed=False):
    """Mesh prims of the parts a carving acts on (flesh bands, strands, endocarp; optionally skin and seed)."""
    paths = [f"/Food/Parts/Mesocarp/{b}/Geom" for b in FLESH]
    vs = st.GetPrimAtPath("/Food/Parts/Mesocarp/VascularStrands")
    if vs:
        paths += [f"/Food/Parts/Mesocarp/VascularStrands/{p.GetName()}/Geom" for p in vs.GetChildren()]
    paths.append("/Food/Parts/Endocarp/Geom")
    if with_skin:
        paths += [f"/Food/Parts/Exocarp/{n}/Geom" for n in EXOCARP] + ["/Food/Parts/StemButton/Geom"]
    if with_seed:
        paths += [str(p.GetPath()) for p in st.Traverse() if p.IsA(UsdGeom.Mesh) and "/Seed/" in str(p.GetPath())]
    return [p for p in paths if st.GetPrimAtPath(p) and st.GetPrimAtPath(p).IsActive()]


def carve(st, cutter, paths, label="avocado_flesh_damaged", sdt_class="food:TornFleshSurface",
          surface_name="GougeSurface"):
    """Subtract the manifold ``cutter`` from the given part meshes. The faces the cutter creates become a separate
    prim per part (``surface_name``, semantic label ``label``) so that they can be told apart in the segmentation.
    Returns the removed volume (mL) and the new surface area per part (mm2)."""
    from pxr import UsdShade
    out = dict(lost_ml=0.0, new_area_mm2={})
    for path in paths:
        prim = st.GetPrimAtPath(path)
        P, F = mesh_arrays(prim)
        cut = cut_faces(prim)
        body = to_manifold(P, F).as_original()
        bid = body.original_id()
        if body.is_empty():
            continue
        res = body - cutter
        vol0, vol1 = body.volume(), res.volume()
        if vol0 - vol1 < 1e-12:
            continue
        out["lost_ml"] += (vol0 - vol1) * 1e6
        mesh = res.to_mesh()
        V = np.asarray(mesh.vert_properties)[:, :3]
        T = np.asarray(mesh.tri_verts, np.int64)
        orig = np.zeros(len(T), np.int64)
        for k, oid in enumerate(mesh.run_original_id):
            orig[mesh.run_index[k] // 3:mesh.run_index[k + 1] // 3] = oid
        own = orig == bid
        To = T[own]
        pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        uv0 = np.asarray(pv.Get(), float) if pv and pv.GetInterpolation() == UsdGeom.Tokens.faceVarying else None
        uv = carried_uv(P, F, uv0, V, To) if uv0 is not None and len(uv0) == 3 * len(F) else None
        new_cut = []
        if cut:      # the cut face, found geometrically (face ids of coplanar triangles are not reliable): triangles
            ci = np.asarray(sorted(cut), np.int64)             # in the plane of the original cut face
            Q = P[F[ci]].reshape(-1, 3)
            q0 = Q.mean(0)
            nrm = np.linalg.svd(Q - q0)[2][2]
            Fn = np.cross(V[To[:, 1]] - V[To[:, 0]], V[To[:, 2]] - V[To[:, 0]])
            Fn /= np.maximum(np.linalg.norm(Fn, axis=1, keepdims=True), 1e-15)
            on = (np.abs((V[To] - q0) @ nrm).max(1) < 5e-5) & (np.abs(Fn @ nrm) > 0.99)
            new_cut = np.nonzero(on)[0].tolist()
            if uv is not None and on.any():                    # section texture: planar, so affine in position
                A = np.linalg.lstsq(np.c_[Q, np.ones(len(Q))], uv0.reshape(-1, 3, 2)[ci].reshape(-1, 2),
                                    rcond=None)[0]
                C = V[To[on]].reshape(-1, 3)
                uv = uv.reshape(-1, 3, 2)
                uv[on] = (np.c_[C, np.ones(len(C))] @ A).reshape(-1, 3, 2)
                uv = uv.reshape(-1, 2)
        set_mesh(prim, V, To, new_cut, uv)
        if (~own).any():
            Fg = T[~own]
            used = np.unique(Fg.ravel())
            rm = -np.ones(len(V), np.int64)
            rm[used] = np.arange(len(used))
            gp = UsdGeom.Mesh.Define(st, prim.GetParent().GetPath().AppendChild(surface_name))
            gp.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(V[used].astype(np.float32)))
            gp.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(Fg)))
            gp.CreateFaceVertexIndicesAttr(Vt.IntArray(rm[Fg].ravel().tolist()))
            gp.CreateSubdivisionSchemeAttr("none")
            mat = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            if mat:
                UsdShade.MaterialBindingAPI.Apply(gp.GetPrim()).Bind(mat)
            UsdSemantics.LabelsAPI.Apply(gp.GetPrim(), "class").CreateLabelsAttr([label])
            gp.GetPrim().CreateAttribute("sdt:class", Sdf.ValueTypeNames.String, custom=True).Set(sdt_class)
            out["new_area_mm2"][prim.GetParent().GetName()] = float(tri_area(V, Fg).sum() * 1e6)
    return out


def write_peeled(src_dir, outcome, out_path):
    """Write the peeled half as a USD state next to the half it derives from; returns the ground-truth record."""
    src = os.path.join(src_dir, f"{outcome.half}.usdc")
    layer = Sdf.Layer.CreateNew(out_path)
    layer.TransferContent(Sdf.Layer.FindOrOpen(src))
    st = Usd.Stage.Open(layer)
    from pxr import UsdShade
    absolutize_textures(st, src_dir)
    patches = [d for d in outcome.defects if d.kind == "skin_patch"]
    gouges = [d for d in outcome.defects if d.kind == "gouge"]
    # remnant skin: an untextured skin material in the twin's skin colour for this instance's ripeness (the trimmed
    # cuticle lost its texture lookup in the renderer; the pebbled geometry stays)
    from food.avocado import skin_colour
    stage_r = json.load(open(os.path.join(src_dir, "instance.json")))["hidden_params"]["ripeness_stage"]
    patch_mat = None
    if patches:
        Sdf.CopySpec(layer, "/Food/Looks/SkinSection", layer, "/Food/Looks/SkinPatch")
        sh = UsdShade.Shader(st.GetPrimAtPath("/Food/Looks/SkinPatch/Shader"))
        sh.CreateInput("diffuse_reflection_color", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*map(float, skin_colour(stage_r))))
        sh.CreateInput("specular_reflection_roughness", Sdf.ValueTypeNames.Float).Set(0.7)    # satin, not mirror:
        sh.CreateInput("specular_reflection_weight", Sdf.ValueTypeNames.Float).Set(0.25)      # small flat patches
        # facing the key light otherwise render as a pale highlight
        patch_mat = UsdShade.Material(st.GetPrimAtPath("/Food/Looks/SkinPatch"))
    gt = dict(category=outcome.category, skin_area_mm2=0.0, gouge_surface_mm2=0.0, lost_volume_ml=0.0,
              n_skin_patches=len(patches), n_gouges=len(gouges))
    # exocarp: only the patches stay
    for name in EXOCARP:
        prim = st.GetPrimAtPath(f"/Food/Parts/Exocarp/{name}/Geom")
        P, F = mesh_arrays(prim)
        cen = P[F].mean(1)
        keep = np.zeros(len(F), bool)
        for d in patches:
            keep |= inside_patch(cen, d)
        keep &= ~np.isin(np.arange(len(F)), list(cut_faces(prim)))
        if not keep.any():
            st.GetPrimAtPath(f"/Food/Parts/Exocarp/{name}").SetActive(False)
            continue
        idx = subset_faces(prim, keep)
        if patch_mat is not None:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(patch_mat)
        if name == "Cuticle":                                 # outer sheet of the cuticle = visible skin
            gt["skin_area_mm2"] = float(tri_area(P, F[idx]).sum() / 2 * 1e6)
    for sub in ("StemButton",):
        st.GetPrimAtPath(f"/Food/Parts/{sub}").SetActive(False)   # comes off with the skin
    # flesh: gouges subtracted
    if gouges:
        cutter = None
        for d in gouges:
            e = ellipsoid(d)
            cutter = e if cutter is None else cutter + e
        res = carve(st, cutter, flesh_paths(st))
        gt["lost_volume_ml"] += res["lost_ml"]
        gt["gouge_surface_mm2"] += res["new_area_mm2"].get("OuterDarkGreenBand", 0.0)
    root = st.GetPrimAtPath("/Food")
    for k, v in (("food:state", "peeled half"), ("food:skin", "patches" if patches else "off"),
                 ("food:peel_category", outcome.category)):
        root.CreateAttribute(k, Sdf.ValueTypeNames.String, custom=True).Set(v)
    for k, v in gt.items():
        if isinstance(v, (int, float)):
            root.CreateAttribute(f"truth:peel_{k}", Sdf.ValueTypeNames.Double, custom=True).Set(float(v))
    layer.Save()
    rec = dict(outcome=dict(asdict(outcome), defects=[asdict(d) for d in outcome.defects]), ground_truth=gt,
               usd=os.path.basename(out_path))
    json.dump(rec, open(out_path.replace(".usda", ".json").replace(".usdc", ".json"), "w"), indent=1)
    return rec


def build_set(instance_dir, out_dir, n_per_category=6, half="half_b", seed0=100):
    """Peeling outcomes of every category for one avocado instance (plus the ideal peel used as the expectation)."""
    os.makedirs(out_dir, exist_ok=True)
    st = Usd.Stage.Open(os.path.join(instance_dir, f"{half}.usdc"))
    inst = os.path.basename(instance_dir.rstrip("/"))
    recs = [write_peeled(instance_dir, PeelOutcome(inst, half, 0, "ideal"), os.path.join(out_dir, "ideal.usda"))]
    k = 0
    for cat in ("clean", "skin_patches", "gouges", "mixed"):
        for j in range(n_per_category):
            oc = sample_outcome(inst, half, seed0 + k, cat, st)
            recs.append(write_peeled(instance_dir, oc, os.path.join(out_dir, f"peel_{k:03d}_{cat}.usda")))
            k += 1
    return recs
