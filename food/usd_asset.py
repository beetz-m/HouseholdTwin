"""USD writer for food instances: render meshes + MDL materials + PhysX rigid body (mass, COM, inertia from the
tissue model) + convex-hull collider + SDT attributes.

Layout of one asset (defaultPrim /Food):
  /Food                      Xform, RigidBodyAPI, MassAPI, food:* attributes (kind, seed, state, discrete attributes)
    /Visual/<part>           render meshes (skin, stem, cap = cut face, seed, ...)
    /Collision               convex hull (purpose guide), MeshCollisionAPI convexHull, physics material
    /Looks/<name>            OmniSurface materials; textures in ./textures/
Materials are local to the asset (each instance has its own textures).
"""

import os

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from .geometry import vertex_normals

F, C3, A, B = Sdf.ValueTypeNames.Float, Sdf.ValueTypeNames.Color3f, Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.Bool


def new_stage(path, root="/Food"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    x = UsdGeom.Xform.Define(stage, root)
    stage.SetDefaultPrim(x.GetPrim())
    UsdGeom.Scope.Define(stage, root + "/Looks")
    UsdGeom.Xform.Define(stage, root + "/Visual")
    return stage, x


def write_mesh(stage, path, v, f, uv=None, material=None, purpose=None, smooth=True):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(v, np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(f, np.int32).ravel()))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    if smooth:
        n = vertex_normals(np.asarray(v, float), np.asarray(f))
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(n.astype(np.float32)))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if uv is not None:
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray.FromNumpy(np.asarray(uv, np.float32)))
    lo, hi = np.min(v, 0), np.max(v, 0)
    m.CreateExtentAttr([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])
    if purpose:
        m.CreatePurposeAttr(purpose)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material)
    return m


def omni_surface(stage, path, **kw):
    """OmniSurface material. kw: name -> value; *_image values are texture paths relative to the asset."""
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.SetSourceAsset(Sdf.AssetPath("OmniSurface.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniSurface", "mdl")
    out = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    for o in (mat.CreateSurfaceOutput("mdl"), mat.CreateVolumeOutput("mdl"), mat.CreateDisplacementOutput("mdl")):
        o.ConnectToSource(out)
    for k, val in kw.items():
        if k.endswith("_image"):
            a = sh.CreateInput(k, A)
            a.Set(Sdf.AssetPath(val))
            if "roughness" in k or "weight" in k or "normal" in k:
                a.GetAttr().SetMetadata("colorSpace", "raw")
        elif isinstance(val, bool):
            sh.CreateInput(k, B).Set(val)
        elif isinstance(val, (tuple, list)):
            sh.CreateInput(k, C3).Set(Gf.Vec3f(*val))
        else:
            sh.CreateInput(k, F).Set(float(val))
    return mat


def omni_glass(stage, path, color=(1, 1, 1), ior=1.5, frosting=0.0, depth=None):
    """OmniGlass (refractive, absorbing) material; depth = absorption distance (m) for the colour."""
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    out = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    for o in (mat.CreateSurfaceOutput("mdl"), mat.CreateVolumeOutput("mdl"), mat.CreateDisplacementOutput("mdl")):
        o.ConnectToSource(out)
    sh.CreateInput("glass_color", C3).Set(Gf.Vec3f(*color))
    sh.CreateInput("glass_ior", F).Set(float(ior))
    sh.CreateInput("frosting_roughness", F).Set(float(frosting))
    if depth is not None:
        sh.CreateInput("depth", F).Set(float(depth))
    sh.CreateInput("thin_walled", B).Set(False)
    return mat


def rigid_body(stage, root_path, mp, hull_v, hull_f, friction, restitution=0.1, kinematic=False):
    prim = stage.GetPrimAtPath(root_path)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    if kinematic:
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
    m = UsdPhysics.MassAPI.Apply(prim)
    m.CreateMassAttr(float(mp["mass_kg"]))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, mp["com_m"])))
    w, V = np.linalg.eigh(mp["inertia_kgm2"])
    if np.linalg.det(V) < 0:
        V[:, 0] *= -1
    q = Gf.Matrix3d(*V.T.ravel().tolist()).ExtractRotation().GetQuat()   # rows of V.T = principal axes
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, w)))
    m.CreatePrincipalAxesAttr(Gf.Quatf(float(q.GetReal()), *map(float, q.GetImaginary())))
    col = write_mesh(stage, root_path + "/Collision", hull_v, hull_f, purpose=UsdGeom.Tokens.guide, smooth=False)
    UsdPhysics.CollisionAPI.Apply(col.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(col.GetPrim()).CreateApproximationAttr("convexHull")
    pm = UsdShade.Material.Define(stage, root_path + "/Looks/PhysicsMaterial")
    api = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
    api.CreateStaticFrictionAttr(float(friction))
    api.CreateDynamicFrictionAttr(float(friction) * 0.85)
    api.CreateRestitutionAttr(float(restitution))
    UsdShade.MaterialBindingAPI.Apply(col.GetPrim()).Bind(pm, UsdShade.Tokens.weakerThanDescendants, "physics")


def convex_hull(v):
    import trimesh
    h = trimesh.Trimesh(vertices=np.asarray(v)).convex_hull
    p = np.asarray(h.vertices)
    if len(p) > 120:                                   # PhysX hull limit: farthest-point subsample, then re-hull
        sel = [int(np.argmax(p[:, 2]))]
        d = np.linalg.norm(p - p[sel[0]], axis=1)
        for _ in range(119):
            sel.append(int(np.argmax(d)))
            d = np.minimum(d, np.linalg.norm(p - p[sel[-1]], axis=1))
        h = trimesh.Trimesh(vertices=p[sel]).convex_hull
    return np.asarray(h.vertices), np.asarray(h.faces)


def sdt_attributes(prim, attrs):
    """Discrete SDT attributes as custom food:* attributes (string / float / int / bool)."""
    for k, v in attrs.items():
        name = "food:" + k
        if isinstance(v, bool):
            prim.CreateAttribute(name, Sdf.ValueTypeNames.Bool, custom=True).Set(v)
        elif isinstance(v, (int, np.integer)):
            prim.CreateAttribute(name, Sdf.ValueTypeNames.Int, custom=True).Set(int(v))
        elif isinstance(v, (float, np.floating)):
            prim.CreateAttribute(name, Sdf.ValueTypeNames.Double, custom=True).Set(float(v))
        else:
            prim.CreateAttribute(name, Sdf.ValueTypeNames.String, custom=True).Set(str(v))
