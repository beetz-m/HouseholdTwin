"""Opening a cracked egg over a bowl in Isaac Sim: the shell halves (kinematic, from the twin's crack geometry) swing
apart about the bottom of the crack like two thumbs prying, the albumen is a PBD particle fluid seeded from the
egg's own albumen tissues, the yolk is a rigid body during the fall (its settled dome shape comes from the yolk
index, see food/egg_open.py).  Writes per-frame stills and test_reports/egg_open_sim.json.

Run (GPU job alone): OMNI_KIT_ACCEPT_EULA=YES ~/Documents/LabAutomationTwin/env_isaacsim/bin/python \
    scripts/sim_egg_open.py [--frames] [--quick]
"""

import json
import math
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FRAMES = "--frames" in sys.argv
QUICK = "--quick" in sys.argv
HZ = 600
EGG = os.path.join(ROOT, "assets", "food", "egg_1")
OUT = os.path.join(ROOT, "renders", "food", "egg_open_frames")
RES = (1280, 800) if QUICK else (1600, 1000)

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "PathTracing" if FRAMES else "RaytracedLighting",
                     "width": RES[0], "height": RES[1]})

import carb  # noqa: E402
import omni.kit.app  # noqa: E402

# particle isosurfaces are not Fabric-compatible (omni.physx.demos ParticlePostProcessingDemo): use USD updates
omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.physx.fabric", False)
_cs = carb.settings.get_settings()
_cs.set("/physics/updateToUsd", True)                  # kinematic shell poses must reach the renderer
_cs.set("/physics/updateParticlesToUsd", True)
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.experimental.prims import RigidPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from omni.physx.scripts import particleUtils, physicsUtils  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt  # noqa: E402

from food import meshlib as ML  # noqa: E402
from food.geometry import Meridian  # noqa: E402

inp = np.load(os.path.join(EGG, "open_sim_inputs.npz"))
omni.usd.get_context().new_stage()
app.update()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
scene = UsdPhysics.Scene.Define(stage, "/World/Physics")
scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr(9.81)
px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
px.CreateTimeStepsPerSecondAttr(HZ)
px.CreateEnableGPUDynamicsAttr(True)
px.CreateBroadphaseTypeAttr("GPU")


def mat(path, mdl="OmniSurface", **kw):
    m = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.SetSourceAsset(Sdf.AssetPath(f"{mdl}.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier(mdl, "mdl")
    o = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    m.CreateSurfaceOutput("mdl").ConnectToSource(o)
    m.CreateVolumeOutput("mdl").ConnectToSource(o)
    for k, v in kw.items():
        if isinstance(v, bool):
            sh.CreateInput(k, Sdf.ValueTypeNames.Bool).Set(v)
        elif isinstance(v, tuple):
            sh.CreateInput(k, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*v))
        else:
            sh.CreateInput(k, Sdf.ValueTypeNames.Float).Set(float(v))
    return m


def mesh_prim(path, m, material=None, collide=False):
    g = UsdGeom.Mesh.Define(stage, path)
    g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(m.vertices, np.float32)))
    g.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(m.faces), 3, np.int32)))
    g.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(m.faces, np.int32).ravel()))
    g.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(m.vertex_normals, np.float32)))
    g.SetNormalsInterpolation("vertex")
    g.CreateSubdivisionSchemeAttr("none")
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(g.GetPrim()).CreateApproximationAttr("none")
    return g


# ---------------------------------------------------------------- set: table + glass bowl
wood = mat("/World/Looks/Wood", diffuse_reflection_color=(0.55, 0.40, 0.26), specular_reflection_roughness=0.45)
glass = mat("/World/Looks/BowlGlass", "OmniGlass", glass_color=(0.97, 0.99, 0.99), glass_ior=1.52)
tab = UsdGeom.Cube.Define(stage, "/World/Table")
tab.CreateSizeAttr(1.0)
UsdGeom.Xformable(tab).AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.015))
UsdGeom.Xformable(tab).AddScaleOp().Set(Gf.Vec3f(1.2, 1.0, 0.03))
UsdPhysics.CollisionAPI.Apply(tab.GetPrim())
UsdShade.MaterialBindingAPI.Apply(tab.GetPrim()).Bind(wood)
# bowl: thin-walled revolve (outer profile up, inner profile down): inner radius 70 mm at the rim, depth 55 mm
ro = np.array([0.0, 0.035, 0.055, 0.068, 0.075, 0.076])
zo = np.array([0.0, 0.0, 0.010, 0.030, 0.055, 0.060])
ri = np.array([0.0723, 0.066, 0.051, 0.031, 0.0])
zi = np.array([0.060, 0.034, 0.016, 0.0055, 0.0055])
rho = np.concatenate([[0.0], ro[1:][::-1], ri[1:]])
zz = np.concatenate([[0.0], zo[1:][::-1], zi[1:]])
order_r = np.concatenate([[0.0], ri[::-1][:-1], ro[::-1]])
# meridian must run top pole -> bottom pole: inner bottom centre -> inner wall up -> rim -> outer wall down -> bottom
mer_r = np.concatenate([ri[::-1], ro[::-1]])
mer_z = np.concatenate([zi[::-1], zo[::-1]])
bowl = ML.surface_of_revolution(Meridian(mer_r, mer_z), 128)
mesh_prim("/World/Bowl", bowl, glass, collide=True)

# ---------------------------------------------------------------- egg: halves, albumen particles, yolk
EGG_Z = 0.115                                   # egg centre height above the table (above the bowl)
R0 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)     # egg z (long axis) -> world +x (blunt end at +x)


def egg_to_world(P):
    return np.asarray(P) @ R0.T + np.array([0, 0, EGG_Z])


frag_paths = {}
for side in ("Pointed", "Blunt"):
    p = f"/World/Egg/{side}"
    x = UsdGeom.Xform.Define(stage, p)
    x.GetPrim().GetReferences().AddReference(os.path.join(EGG, "shell_fragments_inplace.usdc"))
    UsdPhysics.RigidBodyAPI.Apply(x.GetPrim()).CreateKinematicEnabledAttr(True)
    frag_ops = globals().setdefault("frag_ops", {})
    frag_ops[side] = x.AddTransformOp()
    frag_paths[side] = p
for _ in range(3):
    app.update()
for side, p in frag_paths.items():                     # keep only this side's fragment; colliders on its meshes
    other = "Blunt" if side == "Pointed" else "Pointed"
    stage.GetPrimAtPath(f"{p}/ShellFragment_{other}").SetActive(False)
    for prim in stage.GetPrimAtPath(f"{p}/ShellFragment_{side}").GetChildren():
        g = prim
        if g.IsA(UsdGeom.Mesh) and g.GetName() in ("InnerShellMembrane", "Cuticle"):
            UsdPhysics.CollisionAPI.Apply(g)
            UsdPhysics.MeshCollisionAPI.Apply(g).CreateApproximationAttr("none")

# particle system (albumen)
spacing = float(inp["spacing"])
fluid_rest = 0.5 * spacing
solid_rest = 0.6 * spacing
ps_path = Sdf.Path("/World/Albumen")
particleUtils.add_physx_particle_system(stage, ps_path, contact_offset=solid_rest + 0.6e-3, rest_offset=solid_rest,
                                        particle_contact_offset=fluid_rest / 0.6, solid_rest_offset=solid_rest,
                                        fluid_rest_offset=fluid_rest, solver_position_iterations=8,
                                        simulation_owner="/World/Physics", max_neighborhood=96, max_velocity=3.0)
mat_path = "/World/Looks/AlbumenPBD"
particleUtils.add_pbd_particle_material(stage, mat_path, viscosity=0.0012, cohesion=0.3, surface_tension=0.05,
                                        adhesion=0.01, friction=0.2, damping=0.0, density=1035.0)
physicsUtils.add_physics_material_to_prim(stage, stage.GetPrimAtPath(ps_path), mat_path)
particleUtils.add_physx_particle_smoothing(stage, ps_path, strength=0.8)
alb_look = mat("/World/Looks/AlbumenRender", "OmniGlass", glass_color=(0.98, 0.99, 0.94), glass_ior=1.36,
               frosting_roughness=0.04)
UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(ps_path)).Bind(alb_look)
s_ = carb.settings.get_settings()
s_.set("/rtx/translucency/maxRefractionBounces", 12)
pos = egg_to_world(inp["albumen"])
pts_path = Sdf.Path("/World/AlbumenParticles")
particleUtils.add_physx_particleset_points(stage, pts_path, Vt.Vec3fArray.FromNumpy(pos.astype(np.float32)),
                                          Vt.Vec3fArray.FromNumpy(np.zeros_like(pos, np.float32)),
                                          Vt.FloatArray.FromNumpy(np.full(len(pos), 2 * fluid_rest, np.float32)),
                                          ps_path, self_collision=True, fluid=True, particle_group=0,
                                          particle_mass=1035.0 * spacing ** 3, density=1035.0)
UsdGeom.Imageable(stage.GetPrimAtPath(pts_path)).CreatePurposeAttr(UsdGeom.Tokens.guide)   # rendered as a surface
surf = UsdGeom.Mesh.Define(stage, "/World/AlbumenSurface")
UsdShade.MaterialBindingAPI.Apply(surf.GetPrim()).Bind(alb_look)
UsdGeom.PrimvarsAPI(surf).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)


def update_surface(P, h=0.8e-3, sigma=1.1):
    """Liquid surface from particles: splat onto a grid, blur, marching cubes at a density level (as an isosurface)."""
    from scipy import ndimage
    from skimage import measure
    P = P[(np.abs(P[:, 0]) < 0.12) & (np.abs(P[:, 1]) < 0.12) & (P[:, 2] > -0.005) & (P[:, 2] < 0.2)]
    if len(P) == 0:
        return
    lo = P.min(0) - 4 * h
    idx = np.floor((P - lo) / h).astype(int)
    shape = idx.max(0) + 5
    grid = np.zeros(shape, np.float32)
    np.add.at(grid, tuple(idx.T), 1.0)
    grid = ndimage.gaussian_filter(grid, sigma)
    level = 0.5 * (h / spacing) ** 3 * 1.0
    try:
        v, f, _, _ = measure.marching_cubes(grid, level)
    except (ValueError, RuntimeError):
        return
    v = v * h + lo
    surf.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
    surf.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
    surf.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(f[:, ::-1].astype(np.int32).ravel()))
    surf.CreateSubdivisionSchemeAttr("none")


# yolk: rigid sphere during the fall
yc = egg_to_world(inp["yolk_c"][None])[0]
ry = float(inp["yolk_r"])
yolk_look = mat("/World/Looks/Yolk", diffuse_reflection_color=(0.99, 0.62, 0.10), specular_reflection_roughness=0.08,
                specular_reflection_weight=0.8, subsurface_weight=0.28, subsurface_scale=0.003,
                subsurface_scattering_color=(0.98, 0.52, 0.04))
ys = UsdGeom.Sphere.Define(stage, "/World/Yolk")
ys.CreateRadiusAttr(ry)
UsdGeom.Xformable(ys).AddTranslateOp().Set(Gf.Vec3d(*map(float, yc)))
UsdPhysics.CollisionAPI.Apply(ys.GetPrim())
UsdPhysics.RigidBodyAPI.Apply(ys.GetPrim())
UsdPhysics.MassAPI.Apply(ys.GetPrim()).CreateMassAttr(float(inp["yolk_mass_kg"]))
PhysxSchema.PhysxRigidBodyAPI.Apply(ys.GetPrim()).CreateSleepThresholdAttr(0.0)
UsdShade.MaterialBindingAPI.Apply(ys.GetPrim()).Bind(yolk_look)
ymat = UsdShade.Material.Define(stage, "/World/Looks/YolkPhys")
api = UsdPhysics.MaterialAPI.Apply(ymat.GetPrim())
api.CreateStaticFrictionAttr(0.6)
api.CreateDynamicFrictionAttr(0.5)
api.CreateRestitutionAttr(0.0)
UsdShade.MaterialBindingAPI.Apply(ys.GetPrim()).Bind(ymat, UsdShade.Tokens.weakerThanDescendants, "physics")

# lights + camera for frames
dome = UsdLux.DomeLight.Define(stage, "/World/Light/Dome")
dome.CreateIntensityAttr(900)
key = UsdLux.RectLight.Define(stage, "/World/Light/Key")
key.CreateWidthAttr(0.8)
key.CreateHeightAttr(0.6)
key.CreateIntensityAttr(9000)
kx = UsdGeom.Xformable(key)
kx.AddTranslateOp().Set(Gf.Vec3d(-0.6, -0.5, 0.7))
kx.AddRotateXYZOp().Set(Gf.Vec3f(55, 0, -50))

# ---------------------------------------------------------------- run
app_utils.play(commit=True)
SimulationManager.step(steps=2)
yolk = RigidPrim("/World/Yolk")
zc = float(inp["crack_z"])
rmax = float(inp["rmax"])
H = np.array([zc, 0.0, EGG_Z + rmax])                # hinge: top of the crack (crack faces the bowl), axis y


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def quat(Rm):
    w = math.sqrt(max(0.0, 1 + Rm[0, 0] + Rm[1, 1] + Rm[2, 2])) / 2
    x = (Rm[2, 1] - Rm[1, 2]) / (4 * w)
    y = (Rm[0, 2] - Rm[2, 0]) / (4 * w)
    z = (Rm[1, 0] - Rm[0, 1]) / (4 * w)
    return [w, x, y, z]


def set_open(phi, sep):
    """Kinematic targets through the USD transforms (PhysX reads them; the renderer sees the same pose)."""
    for sgn, side in ((1, "Pointed"), (-1, "Blunt")):  # halves swing apart, openings turning down
        Ra = rot_y(sgn * phi)
        Rm = Ra @ R0
        t = H + Ra @ (np.array([0, 0, EGG_Z]) - H) + np.array([-sgn * sep, 0, 0])
        M = Gf.Matrix4d(1.0)
        M.SetRotateOnly(Gf.Matrix3d(*Rm.T.ravel().tolist()))
        M.SetTranslateOnly(Gf.Vec3d(*map(float, t)))
        frag_ops[side].Set(M)


set_open(0.0, 0.0)
# hold the closed egg so the contents settle inside the shell
pp = particleUtils  # noqa: F841
T_OPEN, T_TOTAL = 0.6, 4.0
cam_shots = {0.0: None, 0.25: None, 0.45: None, 0.7: None, 1.2: None, 2.2: None, 4.0: None}
if FRAMES:
    import omni.replicator.core as rep
    from PIL import Image
    s = carb.settings.get_settings()
    s.set("/rtx/rendermode", "PathTracing")
    s.set("/rtx/pathtracing/spp", 8 if QUICK else 16)
    s.set("/rtx/pathtracing/totalSpp", (8 if QUICK else 16) * (12 if QUICK else 32))
    s.set("/rtx/pathtracing/maxBounces", 12)
    cam = UsdGeom.Camera.Define(stage, "/World/Cam")
    cam.CreateHorizontalApertureAttr(36.0)
    cam.CreateVerticalApertureAttr(36.0 * RES[1] / RES[0])
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50))
    cam.CreateFocalLengthAttr(45.0)
    UsdGeom.Xformable(cam).AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(0.0, -0.46, 0.27), Gf.Vec3d(0.0, 0.0, 0.085), Gf.Vec3d(0, 0, 1)).GetInverse())
    rp = rep.create.render_product(str(cam.GetPath()), RES)
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(rp)
    os.makedirs(OUT, exist_ok=True)


dome = None


def settle_yolk():
    """After landing, the yolk relaxes from a sphere into the sessile dome of its yolk index (food/egg_open.py)."""
    global dome
    yi = float(inp["yolk_index"])
    V = 4 / 3 * np.pi * ry ** 3
    zs = np.linspace(0, 1, 400)
    fill = np.trapezoid(((1 - zs ** 2.5) ** (1 / 2.5)) ** 2, zs)
    a = (V / (2 * np.pi * yi * fill)) ** (1 / 3)
    hy = 2 * a * yi
    yp = yolk.get_world_poses()[0].numpy()[0]
    zr = np.linspace(hy, 0, 80)
    rr = a * np.clip(1 - (zr / hy) ** 2.5, 0, 1) ** 0.4
    m = ML.surface_of_revolution(Meridian(np.append(rr, 0.0), np.append(zr, 0.0)), 96)
    m.apply_translation([yp[0], yp[1], yp[2] - ry])
    dome = mesh_prim("/World/YolkDome", m, yolk_look)
    UsdGeom.Imageable(stage.GetPrimAtPath("/World/Yolk")).MakeInvisible()


def frame(tag):
    if not FRAMES:
        return
    update_surface(np.asarray(pts.GetPointsAttr().Get()))
    for _ in range(2):
        app.update()
    rep.orchestrator.step(rt_subframes=12 if QUICK else 32, delta_time=0.0)
    rep.orchestrator.step(rt_subframes=1, delta_time=0.0)
    Image.fromarray(np.asarray(rgb.get_data())[..., :3].astype(np.uint8)).save(os.path.join(OUT, f"t_{tag}.png"))
    print("[frame]", tag, flush=True)


t, n = 0.0, int(T_TOTAL * HZ)
log = []
pts = UsdGeom.Points(stage.GetPrimAtPath(pts_path))
next_shot = sorted(cam_shots)
for k in range(n + 1):
    t = k / HZ
    if next_shot and t >= next_shot[0] - 1e-9:
        frame(f"{next_shot.pop(0):.2f}")
    if k == n:
        break
    if dome is None and t >= 1.8:
        settle_yolk()
    a = min(1.0, max(0.0, t / T_OPEN))
    e = a * a * (3 - 2 * a)
    set_open(math.radians(100) * e, 0.010 * e)
    SimulationManager.step(steps=1)
    if k % int(0.05 * HZ) == 0:
        yp = yolk.get_world_poses()[0].numpy()[0]
        log.append({"t": round(t, 3), "yolk": np.round(yp, 4).tolist()})
P = np.asarray(pts.GetPointsAttr().Get())
inside = (np.hypot(P[:, 0], P[:, 1]) < 0.072) & (P[:, 2] < 0.06) & (P[:, 2] > 0.0)
fell = P[:, 2] < 0.0
yp = yolk.get_world_poses()[0].numpy()[0]
rep_ = {"particles": int(len(P)), "fraction_in_bowl": float(inside.mean()), "fraction_below_table": float(fell.mean()),
        "fraction_still_high": float((P[:, 2] > 0.06).mean()),
        "yolk_final": np.round(yp, 4).tolist(), "yolk_in_bowl": bool(np.hypot(yp[0], yp[1]) < 0.07 and yp[2] < 0.06),
        "albumen_z_mm": [float(np.percentile(P[:, 2], q) * 1e3) for q in (5, 50, 95)], "log": log}
json.dump(rep_, open(os.path.join(ROOT, "test_reports", "egg_open_sim.json"), "w"), indent=1)
print("[result]", {k: v for k, v in rep_.items() if k != "log"}, flush=True)
app.close()
