"""Write a part graph (whole or one side of a cut) as a USD asset with SDT correspondence, plus SDT/truth layers.

  <state>.usda   /Food                      rigid body (mass, COM, inertia summed over the parts), food:* state attrs
                   /Parts/<partonomy>       Xform per part: sdt:represents, sdt:class, sdt:partOf, semantics labels,
                       /Geom                mesh; GeomSubset CutFace (section material) on cut states
                   /Collision               convex hull, physics material
                   /Looks                   OmniSurface materials (per tissue) + Section (planar section texture)
  sdt.usda       sublayers whole.usda; overs with the robot's beliefs (sdt:belief:*, sdt:epistemicStatus)
  truth.usda     overs only: truth:* hidden instance parameters (simulator ground truth, never read by the robot)
"""

import os

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdSemantics, UsdShade, Vt

from .usd_asset import convex_hull, new_stage, omni_surface, sdt_attributes

S, D = Sdf.ValueTypeNames.String, Sdf.ValueTypeNames.Double


def _prim_name(s):
    return "".join(c if c.isalnum() or c == "_" else "_" for c in s)


def ensure_xforms(stage, path):
    parts = path.strip("/").split("/")
    for i in range(1, len(parts) + 1):
        p = "/" + "/".join(parts[:i])
        if not stage.GetPrimAtPath(p):
            UsdGeom.Xform.Define(stage, p)
    return stage.GetPrimAtPath(path)


def write_part_mesh(stage, path, mesh, cap_mask, material, cap_material, uv_fn=None, cap_uv_fn=None):
    m = UsdGeom.Mesh.Define(stage, path)
    v = np.asarray(mesh.vertices, np.float32)
    f = np.asarray(mesh.faces, np.int32)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.ravel()))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    vn = np.asarray(mesh.vertex_normals, np.float32)
    if cap_mask is not None and cap_mask.any():          # hard edge along the cut: face-varying normals
        corner_n = vn[f].copy()
        corner_n[cap_mask] = np.asarray(mesh.face_normals, np.float32)[cap_mask][:, None, :]
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(corner_n.reshape(-1, 3)))
        m.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    else:
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(vn))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if uv_fn is not None or (cap_mask is not None and cap_mask.any() and cap_uv_fn is not None):
        corners = v[f].reshape(-1, 3).astype(float)
        uv = np.zeros((len(corners), 2), np.float32)
        if uv_fn is not None:
            uv[:] = uv_fn(corners)
            uvf = uv.reshape(-1, 3, 2)                           # seam fix: a face spanning u 0..1 wraps
            span = uvf[..., 0].max(1) - uvf[..., 0].min(1)
            bad = span > 0.5
            uvf[bad, :, 0] = np.where(uvf[bad, :, 0] < 0.5, uvf[bad, :, 0] + 1, uvf[bad, :, 0])
            uv = uvf.reshape(-1, 2)
        if cap_mask is not None and cap_mask.any() and cap_uv_fn is not None:
            cm = np.repeat(cap_mask, 3)
            uv[cm] = cap_uv_fn(corners[cm])
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                   UsdGeom.Tokens.faceVarying)
        pv.Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
    lo, hi = v.min(0), v.max(0)
    m.CreateExtentAttr([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material)
    if cap_mask is not None and cap_mask.any():
        sub = UsdGeom.Subset.CreateGeomSubset(m, "CutFace", UsdGeom.Tokens.face,
                                              Vt.IntArray.FromNumpy(np.nonzero(cap_mask)[0].astype(np.int32)),
                                              "materialBind")
        if cap_material is not None:
            UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(cap_material)
    return m


def label(prim, cls_label, part_label):
    api = UsdSemantics.LabelsAPI.Apply(prim, "class")
    api.CreateLabelsAttr([cls_label])
    api2 = UsdSemantics.LabelsAPI.Apply(prim, "part")
    api2.CreateLabelsAttr([part_label])


def mass_props(pieces, info):
    """Sum mass properties of part meshes (density per part)."""
    M, c1, J = 0.0, np.zeros(3), np.zeros((3, 3))
    for path, mesh in pieces.items():
        rho = info[path]["density"]
        if mesh is None or rho <= 0 or not mesh.is_volume:
            continue
        mesh = mesh.copy()
        mesh.density = rho
        mp = mesh.mass_properties
        m, c = mp["mass"], np.asarray(mp["center_mass"])
        M += m
        c1 += m * c
        J += np.asarray(mp["inertia"]) + m * ((c @ c) * np.eye(3) - np.outer(c, c))   # about the origin
    com = c1 / M
    J_com = J - M * ((com @ com) * np.eye(3) - np.outer(com, com))
    return {"mass_kg": M, "com_m": com, "inertia_kgm2": J_com}


def write_state(path, pg, iri_base, state, pieces, materials_fn, root_attrs, friction, restitution,
                uv_fns=None, cap_uv_fn=None, physics=True, hide_interior=False, points=None, voids=(),
                subsets=None, point_clip=None, collision_fn=None, bodies=None):
    """pieces: part path -> (trimesh, cap_mask) for this state (whole: cap masks all False).
    points: part path -> (positions, widths) for point-set parts (pores, lenticels); point_clip (normal, side name)
    keeps the points on the half's side.  voids (air cells, locules) stay SDT nodes but are not rendered/collided."""
    stage, x = new_stage(path)
    L = materials_fn(stage)
    UsdGeom.Xform.Define(stage, "/Food/Parts")
    root = x.GetPrim()
    root.CreateAttribute("sdt:represents", S, custom=True).Set(iri_base)
    root.CreateAttribute("sdt:class", S, custom=True).Set(root_attrs.get("aicor_class", ""))
    label(root, pg.kind, pg.kind)
    keys = {p.path: p.key for p in pg.parts}
    present = set(pieces) | set(points or {})
    for p in pg.parts:                                     # grouping nodes and parts present in this state
        if p.geom is None:
            if not any(q.startswith(p.path + "/") for q in present):
                continue
        elif p.path not in present:
            continue
        prim = ensure_xforms(stage, "/Food/Parts/" + p.path)
        _node(prim, p, iri_base, pg.kind, keys)
    meshes = {}
    for rel, (mesh, cap) in pieces.items():
        info = pg.info[rel]
        p = info["part"]
        prim = ensure_xforms(stage, "/Food/Parts/" + rel)
        vol = abs(mesh.volume) if mesh.is_volume else info["volume_cm3"] * 1e-6
        prim.CreateAttribute("food:volume_cm3", D, custom=True).Set(float(vol * 1e6))
        prim.CreateAttribute("food:density_kgm3", D, custom=True).Set(float(info["density"]))
        prim.CreateAttribute("food:mass_g", D, custom=True).Set(float(vol * info["density"] * 1e3))
        g = write_part_mesh(stage, f"/Food/Parts/{rel}/Geom", mesh, cap, L.get(p.material), L.get("Section"),
                            uv_fn=(uv_fns or {}).get(p.key), cap_uv_fn=cap_uv_fn)
        if (hide_interior and p.interior) or rel in voids:
            g.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        sub_fn = (subsets or {}).get(p.key)
        if sub_fn is not None:
            reg = sub_fn(np.asarray(mesh.triangles_center))
            outward = np.einsum("ij,ij->i", mesh.face_normals, mesh.triangles_center) > 0
            for name in sorted(set(reg)):
                idx = np.nonzero((reg == name) & outward & ~(cap if cap is not None else False))[0]
                if len(idx):
                    sb = UsdGeom.Subset.CreateGeomSubset(g, name, UsdGeom.Tokens.face,
                                                         Vt.IntArray.FromNumpy(idx.astype(np.int32)), "surfaceRegion")
                    sb.GetPrim().CreateAttribute("sdt:represents", S, custom=True).Set(
                        f"{iri_base}/{p.key}/region/{name}")
                    sb.GetPrim().CreateAttribute("sdt:class", S, custom=True).Set(f"food:SurfaceRegion_{name}")
        if rel not in voids:
            meshes[rel] = mesh
    for rel, (pos, wid) in (points or {}).items():
        p = pg.info[rel]["part"]
        pos, wid = np.asarray(pos), np.asarray(wid)
        if point_clip is not None:
            n, side = point_clip[:2]
            o = np.asarray(point_clip[2]) if len(point_clip) > 2 else np.zeros(3)
            pos_side = side in ("half_a", "pos") or (len(point_clip) > 3 and point_clip[3])
            keep = ((pos - o) @ np.asarray(n) >= 0) if pos_side else ((pos - o) @ np.asarray(n) < 0)
            pos, wid = pos[keep], wid[keep]
        prim = ensure_xforms(stage, "/Food/Parts/" + rel)
        pts = UsdGeom.Points.Define(stage, f"/Food/Parts/{rel}/Geom")
        pts.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pos.astype(np.float32)))
        pts.CreateWidthsAttr(Vt.FloatArray.FromNumpy(wid.astype(np.float32)))
        pts.CreatePurposeAttr(UsdGeom.Tokens.guide)       # microscopic features: SDT nodes, not render geometry
        prim.CreateAttribute("food:count", Sdf.ValueTypeNames.Int, custom=True).Set(int(len(pos)))
    if physics and bodies:
        # several rigid bodies (e.g. the slices of a loaf): each group prim gets its own mass and convex hull
        mps = []
        for rel, parts in bodies.items():
            ms = {k: m for k, m in meshes.items() if k in parts}
            if not ms:
                continue
            gp = ensure_xforms(stage, "/Food/Parts/" + rel)
            mpb = mass_props(ms, pg.info)
            _rigid(stage, gp, mpb, np.vstack([m.vertices for m in ms.values()]), friction, restitution,
                   "/Food/Parts/" + rel)
            mps.append(mpb)
        M = sum(m["mass_kg"] for m in mps)
        mp = {"mass_kg": M, "com_m": sum(m["mass_kg"] * m["com_m"] for m in mps) / M, "inertia_kgm2": None}
        root_attrs = dict(root_attrs, mass_g=M * 1e3, rigid_bodies=len(mps))
    elif physics:
        mp = mass_props(meshes, pg.info)
        UsdPhysics.RigidBodyAPI.Apply(root)
        ma = UsdPhysics.MassAPI.Apply(root)
        ma.CreateMassAttr(float(mp["mass_kg"]))
        ma.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, mp["com_m"])))
        w, V = np.linalg.eigh(mp["inertia_kgm2"])
        if np.linalg.det(V) < 0:
            V[:, 0] *= -1
        q = Gf.Matrix3d(*V.T.ravel().tolist()).ExtractRotation().GetQuat()
        ma.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, w)))
        ma.CreatePrincipalAxesAttr(Gf.Quatf(float(q.GetReal()), *map(float, q.GetImaginary())))
        allv = np.vstack([m.vertices for m in meshes.values()])
        from .usd_asset import write_mesh
        # one convex hull, or several (collision_fn: all vertices -> list of point sets) for curved bodies (banana)
        groups = collision_fn(allv) if collision_fn is not None else [allv]
        if len(groups) > 1:                                 # a typed parent: PhysX must see the hulls under the body
            UsdGeom.Xform.Define(stage, "/Food/Collision")
        cols = []
        for k, gv in enumerate(groups):
            hv, hf = convex_hull(gv)
            col = write_mesh(stage, "/Food/Collision" + (f"/Hull_{k:02d}" if len(groups) > 1 else ""), hv, hf,
                             purpose=UsdGeom.Tokens.guide, smooth=False)
            UsdPhysics.CollisionAPI.Apply(col.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(col.GetPrim()).CreateApproximationAttr("convexHull")
            cols.append(col)
        pm = UsdShade.Material.Define(stage, "/Food/Looks/PhysicsMaterial")
        api = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
        api.CreateStaticFrictionAttr(float(friction))
        api.CreateDynamicFrictionAttr(float(friction) * 0.85)
        api.CreateRestitutionAttr(float(restitution))
        for col in cols:
            UsdShade.MaterialBindingAPI.Apply(col.GetPrim()).Bind(pm, UsdShade.Tokens.weakerThanDescendants, "physics")
        root_attrs = dict(root_attrs, mass_g=mp["mass_kg"] * 1e3)
    else:
        mp = None
    sdt_attributes(root, dict(root_attrs, state=state))
    stage.GetRootLayer().Save()
    return mp


def _rigid(stage, prim, mp, verts, friction, restitution, path):
    """Rigid body on `prim` (group of parts) with mass properties and a convex-hull collider below it."""
    from .usd_asset import write_mesh
    UsdPhysics.RigidBodyAPI.Apply(prim)
    ma = UsdPhysics.MassAPI.Apply(prim)
    ma.CreateMassAttr(float(mp["mass_kg"]))
    ma.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, mp["com_m"])))
    w, V = np.linalg.eigh(mp["inertia_kgm2"])
    if np.linalg.det(V) < 0:
        V[:, 0] *= -1
    q = Gf.Matrix3d(*V.T.ravel().tolist()).ExtractRotation().GetQuat()
    ma.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, np.maximum(w, 1e-12))))
    ma.CreatePrincipalAxesAttr(Gf.Quatf(float(q.GetReal()), *map(float, q.GetImaginary())))
    hv, hf = convex_hull(verts)
    col = write_mesh(stage, path + "/Collision", hv, hf, purpose=UsdGeom.Tokens.guide, smooth=False)
    UsdPhysics.CollisionAPI.Apply(col.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(col.GetPrim()).CreateApproximationAttr("convexHull")
    pm = UsdShade.Material.Define(stage, "/Food/Looks/PhysicsMaterial")
    api = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
    api.CreateStaticFrictionAttr(float(friction))
    api.CreateDynamicFrictionAttr(float(friction) * 0.85)
    api.CreateRestitutionAttr(float(restitution))
    UsdShade.MaterialBindingAPI.Apply(col.GetPrim()).Bind(pm, UsdShade.Tokens.weakerThanDescendants, "physics")


def _node(prim, p, obj_iri, kind, keys):
    prim.CreateAttribute("sdt:represents", S, custom=True).Set(f"{obj_iri}/{p.key}")
    prim.CreateAttribute("sdt:class", S, custom=True).Set(p.cls)
    parent = p.path.rsplit("/", 1)[0] if "/" in p.path else None
    prim.CreateAttribute("sdt:partOf", S, custom=True).Set(f"{obj_iri}/{keys[parent]}" if parent else obj_iri)
    prim.CreateAttribute("sdt:observability", S, custom=True).Set(p.observability)
    prim.CreateAttribute("sdt:usdPrimPath", S, custom=True).Set(str(prim.GetPath()))
    if p.doc:
        prim.SetDocumentation(p.doc)
    for k, v in p.attrs.items():
        prim.CreateAttribute("food:" + k, S if isinstance(v, str) else D, custom=True).Set(v)
    label(prim, f"{kind}_{p.key}", p.key)


def write_sdt_layer(path, whole_rel, pg, belief):
    """Overs on the whole-object asset with the robot's beliefs: belief = {prim_rel_or_'': {attr: value}}."""
    if os.path.exists(path):
        os.remove(path)
    st = Usd.Stage.CreateNew(path)
    st.GetRootLayer().subLayerPaths = [whole_rel]
    st.GetRootLayer().comment = "SDT layer: the robot's beliefs about this instance (from priors and observations)."
    for rel, attrs in belief.items():
        prim = st.OverridePrim("/Food" + ("/Parts/" + rel if rel else ""))
        for k, v in attrs.items():
            if isinstance(v, str):
                prim.CreateAttribute(k, S, custom=True).Set(v)
            else:
                prim.CreateAttribute(k, D, custom=True).Set(float(v))
    st.GetRootLayer().Save()


def write_truth_layer(path, truth):
    if os.path.exists(path):
        os.remove(path)
    st = Usd.Stage.CreateNew(path)
    st.GetRootLayer().comment = ("Simulator ground truth (hidden instance parameters): overs only, never part of the "
                                 "SDT; used by the dynamics models and audits.")
    for rel, attrs in truth.items():
        prim = st.OverridePrim("/Food" + ("/Parts/" + rel if rel else ""))
        for k, v in attrs.items():
            if isinstance(v, str):
                prim.CreateAttribute(k, S, custom=True).Set(v)
            else:
                prim.CreateAttribute(k, D, custom=True).Set(float(v))
    st.GetRootLayer().Save()
