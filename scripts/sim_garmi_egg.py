"""GARMI cracks an egg into a bowl in the EASE kitchen (Isaac Sim, PhysX).

  1 PickUpEgg     arm_0 approaches the pointed end horizontally, closes at 3 N (inside the H8 corridor), lifts
  2 CrackEgg      it brings the egg's equator down onto the bowl rim at ~0.25 m/s; PhysX gives the contact and the
                  egg's normal speed, food/egg_crack.strike decides the outcome on the hidden egg (retry +15 % if no
                  crack, failure branch if shattered)
  3 OpenUp        the egg is replaced by its two crack fragments (kinematic, held by the hands) with the albumen as
                  PBD particles and the yolk as a rigid body inside; arm_1 grasps the blunt end from the far side,
                  both arms move over the bowl centre and swing the halves apart about the top of the crack
Writes test_reports/garmi_egg.json and (with --frames) renders/apartment/garmi_egg/t_*.png.
Run (GPU job alone): OMNI_KIT_ACCEPT_EULA=YES ~/Documents/LabAutomationTwin/env_isaacsim/bin/python \
    scripts/sim_garmi_egg.py [--frames] [--quick]
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
EGG_SEED = 1
EGG_DIR = os.path.join(ROOT, "assets", "food", f"egg_{EGG_SEED}")
OUT = os.path.join(ROOT, "renders", "apartment", "garmi_egg")
RES = (1280, 800) if QUICK else (1600, 1000)
HZ = 240

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "PathTracing" if FRAMES else "RaytracedLighting",
                     "width": RES[0], "height": RES[1]})
import carb  # noqa: E402
import omni.kit.app  # noqa: E402

_cs = carb.settings.get_settings()
if os.environ.get("GT_FABRIC", "0") == "0":
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.physx.fabric", False)
    _cs.set("/physics/updateToUsd", True)
    _cs.set("/physics/updateParticlesToUsd", True)

import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.experimental.prims import RigidPrim  # noqa: E402
from omni.physx import get_physx_simulation_interface  # noqa: E402
from omni.physx.scripts import particleUtils, physicsUtils  # noqa: E402
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt  # noqa: E402

from food import egg_crack as EC  # noqa: E402
from food import meshlib as ML  # noqa: E402
from food.dynamics import EggTruth  # noqa: E402
from food.egg import Egg  # noqa: E402
from food.geometry import Meridian  # noqa: E402
from food.sdt import EggBelief  # noqa: E402
from household import garmi_rig as GR  # noqa: E402
from household.garmi_rig import GarmiRig, mat2q, qmat  # noqa: E402

GR.HZ, GR.DT = HZ, 1.0 / HZ
omni.usd.get_context().new_stage()
app.update()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
sc = UsdPhysics.Scene.Define(stage, "/World/Physics")
sc.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
sc.CreateGravityMagnitudeAttr(9.81)
px = PhysxSchema.PhysxSceneAPI.Apply(sc.GetPrim())
px.CreateTimeStepsPerSecondAttr(HZ)
px.CreateEnableGPUDynamicsAttr(os.environ.get("GT_GPU", "1") == "1")
px.CreateBroadphaseTypeAttr("GPU" if os.environ.get("GT_GPU", "1") == "1" else "MBP")
EVENTS = []


def ev(kind, **d):
    EVENTS.append({"t": round(rig.t if "rig" in globals() else 0.0, 3), "event": kind, **d})
    print("[event]", kind, d, flush=True)


def mat(path, mdl="OmniSurface", **kw):
    m = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.SetSourceAsset(Sdf.AssetPath(f"{mdl}.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier(mdl, "mdl")
    o = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    m.CreateSurfaceOutput("mdl").ConnectToSource(o)
    m.CreateVolumeOutput("mdl").ConnectToSource(o)
    for k, v in kw.items():
        if isinstance(v, str):
            sh.CreateInput(k, Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(v))
        elif isinstance(v, tuple):
            sh.CreateInput(k, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*v))
        else:
            sh.CreateInput(k, Sdf.ValueTypeNames.Float).Set(float(v))
    return m


def collider_box(path, size, pos, visible=False, material=None):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    UsdGeom.Xformable(c).AddTranslateOp().Set(Gf.Vec3d(*pos))
    UsdGeom.Xformable(c).AddScaleOp().Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    if not visible:
        UsdGeom.Imageable(c).MakeInvisible()
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(material)
    return c


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


# ---------------------------------------------------------------- kitchen (visual) + colliders
FL = lambda k: os.environ.get(k, "1") == "1"  # noqa: E731
ap = UsdGeom.Xform.Define(stage, "/World/Apartment") if FL("GT_APT") else None
if ap is None:
    ap = UsdGeom.Xform.Define(stage, "/World/NoApartment")
if FL("GT_APT"):
    ap.GetPrim().GetReferences().AddReference(os.path.join(ROOT, "assets/apartments/ease/apartment/apartment/apartment.usda"))
    ap.GetPrim().GetVariantSets().GetVariantSet("Physics").SetVariantSelection("none")
ISLAND = 0.946
# the island slab is added after GARMI has raised its lift: at lift 0 its forearms sit at worktop height
# no floor collider: GARMI's chassis reaches 26 mm below z = 0; its wheels inside a floor box blew up the articulation
# (the base is anchored to the world anyway)
oak = mat("/World/Looks/Oak", diffuse_reflection_color_image=os.path.expanduser(
    "~/Documents/LabAutomationTwin/assets/materials/textures/oak_albedo.png"), specular_reflection_roughness=0.45)
BOARD_TOP = ISLAND + 0.022
if FL("GT_BOARD"):
    collider_box("/World/Board", (0.24, 0.26, 0.022), (2.74, 2.33, ISLAND + 0.011), visible=True, material=oak)
# glass bowl on the island: rim radius 72 mm at 60 mm; centre chosen so the rim crosses under the egg's equator
egg = Egg(EGG_SEED)
L_egg = float(egg.mer.zmax - egg.mer.zmin)
R_eq = float(egg.mer.rho.max())
STRIKE_C = np.array([2.74, 2.60])                     # egg centre (x, y) over the rim at the strike
BOWL_C = np.array([STRIKE_C[0], STRIKE_C[1] + 0.0723])
ri = np.array([0.0723, 0.066, 0.051, 0.031, 0.0])
zi = np.array([0.060, 0.034, 0.016, 0.0055, 0.0055])
ro = np.array([0.0, 0.035, 0.055, 0.068, 0.075, 0.076])
zo = np.array([0.0, 0.0, 0.010, 0.030, 0.055, 0.060])
bowl = ML.surface_of_revolution(Meridian(np.concatenate([ri[::-1], ro[::-1]]), np.concatenate([zi[::-1], zo[::-1]])),
                                128)
bowl.apply_translation([BOWL_C[0], BOWL_C[1], ISLAND])
glass = mat("/World/Looks/BowlGlass", "OmniGlass", glass_color=(0.97, 0.99, 0.99), glass_ior=1.52)
mesh_prim("/World/Bowl", bowl, glass, collide=FL("GT_BOWL"))
RIM_TOP = ISLAND + 0.060

# ---------------------------------------------------------------- the egg (intact: rigid twin asset)
EGG_Q = Gf.Quatf(math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0, 0)   # egg +z (blunt) -> world +y
egg_pos = np.array([2.74, 2.30, BOARD_TOP + R_eq + 0.0005])
ex = UsdGeom.Xform.Define(stage, "/World/Egg")
ex.GetPrim().GetReferences().AddReference(os.path.join(EGG_DIR, "whole.usdc"))
ex.AddTranslateOp().Set(Gf.Vec3d(*egg_pos))
ex.AddOrientOp().Set(EGG_Q)
PhysxSchema.PhysxContactReportAPI.Apply(ex.GetPrim()).CreateThresholdAttr().Set(0.0)
PhysxSchema.PhysxRigidBodyAPI.Apply(ex.GetPrim()).CreateSleepThresholdAttr(0.0)

# ---------------------------------------------------------------- GARMI (fixed base)
LIFT0 = 0.12
g = UsdGeom.Xform.Define(stage, "/World/GARMI")
g.GetPrim().GetReferences().AddReference(os.path.join(ROOT, "assets/apartments/garmi_robot/garmi_clean/garmi_clean.usda"))
g.AddTranslateOp().Set(Gf.Vec3d(3.78, 2.55, 0.0))   # shoulders (z 0.82 at lift 0) clear of the island
g.AddRotateZOp().Set(180.0)
g.GetPrim().GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
app.update()
art_root = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)
            and str(p.GetPath()).startswith("/World/GARMI")][0]
for p in stage.Traverse():
    if str(p.GetPath()).startswith("/World/GARMI") and p.HasAPI(UsdPhysics.RigidBodyAPI):
        m = UsdPhysics.MassAPI.Apply(p)
        if not m.GetMassAttr().Get() or m.GetMassAttr().Get() <= 0:
            m.CreateMassAttr(0.2)
            m.CreateDiagonalInertiaAttr(Gf.Vec3f(2e-4, 2e-4, 2e-4))   # imported placeholder inertia was 1 kg m2
    if p.GetName() in ("lift_0_lower_joint", "lift_0_upper_joint"):   # imported 2000 N/m sags ~15 cm under the arms
        d = UsdPhysics.DriveAPI.Apply(p, "linear")
        d.CreateStiffnessAttr(2.0e5)
        d.CreateDampingAttr(2.0e4)
        d.CreateMaxForceAttr(1.0e4)
    if os.environ.get("GT_FINGERS", "1") == "1" and p.GetName() in (
            "arm_0_gripper_fr3_finger_joint1", "arm_0_gripper_fr3_finger_joint2",
            "arm_1_gripper_fr3_finger_joint1", "arm_1_gripper_fr3_finger_joint2"):
        d = UsdPhysics.DriveAPI.Apply(p, "linear")
        d.CreateTypeAttr("force")
        d.CreateStiffnessAttr(3000.0)
        d.CreateDampingAttr(60.0)
        d.CreateMaxForceAttr(20.0)

pad = UsdShade.Material.Define(stage, "/World/Looks/PadPhys")
api = UsdPhysics.MaterialAPI.Apply(pad.GetPrim())
api.CreateStaticFrictionAttr(1.0)
api.CreateDynamicFrictionAttr(0.9)
for p in stage.Traverse():
    if FL("GT_PAD") and "finger" in str(p.GetPath()) and p.HasAPI(UsdPhysics.CollisionAPI):
        UsdShade.MaterialBindingAPI.Apply(p).Bind(pad, UsdShade.Tokens.weakerThanDescendants, "physics")
fj = UsdPhysics.FixedJoint.Define(stage, "/World/GarmiAnchor")
fj.CreateBody1Rel().SetTargets([art_root.GetPath()])
p_root = UsdGeom.Xformable(art_root).ComputeLocalToWorldTransform(0)
fj.CreateLocalPos0Attr().Set(Gf.Vec3f(*p_root.ExtractTranslation()))
r_ = p_root.ExtractRotationQuat()
if FL("GT_ROT"):
    fj.CreateLocalRot0Attr().Set(Gf.Quatf(float(r_.GetReal()), *map(float, r_.GetImaginary())))
else:
    fj.CreateLocalRot0Attr().Set(Gf.Quatf(0, 0, 0, 1))
print("[diag] anchor rot", r_, flush=True)

# lights + camera
dome = UsdLux.DomeLight.Define(stage, "/World/Light/Sky")
dome.CreateIntensityAttr(1500)
dome.CreateColorAttr(Gf.Vec3f(0.85, 0.9, 1.0))
for i, (x, y) in enumerate(((1.5, 1.6), (2.8, 2.5), (2.8, 3.4))):
    Lp = UsdLux.RectLight.Define(stage, f"/World/Light/Panel{i}")
    Lp.CreateWidthAttr(0.9)
    Lp.CreateHeightAttr(0.9)
    Lp.CreateIntensityAttr(6000)
    UsdGeom.Xformable(Lp).AddTranslateOp().Set(Gf.Vec3d(x, y, 2.95))

# ---------------------------------------------------------------- run
app_utils.play(commit=True)
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402

SimulationManager.step(steps=5)
rig = GarmiRig(str(art_root.GetPath()))
rig.start()
egg_rp = RigidPrim("/World/Egg")
truth = EggTruth(egg, egg.V_outer * 1087)
rng = np.random.default_rng(3)
belief = EggBelief(egg.V_outer * 1087, np.random.default_rng(13))

# frames
shots = []
if FRAMES:
    import omni.replicator.core as rep
    from PIL import Image
    _cs.set("/rtx/rendermode", "PathTracing")
    _cs.set("/rtx/pathtracing/spp", 8 if QUICK else 16)
    _cs.set("/rtx/pathtracing/totalSpp", (8 if QUICK else 16) * (16 if QUICK else 40))
    _cs.set("/rtx/pathtracing/maxBounces", 10)
    _cs.set("/rtx/pathtracing/optixDenoiser/enabled", True)
    cam = UsdGeom.Camera.Define(stage, "/World/Cam")
    cam.CreateHorizontalApertureAttr(36.0)
    cam.CreateVerticalApertureAttr(36.0 * RES[1] / RES[0])
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50))
    cam.CreateFocalLengthAttr(28.0)
    cop = UsdGeom.Xformable(cam).AddTransformOp()
    rp_ = rep.create.render_product(str(cam.GetPath()), RES)
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(rp_)
    os.makedirs(OUT, exist_ok=True)


def frame(tag, eye=(2.05, 1.75, 1.45), target=(2.78, 2.55, 1.0), focal=28.0):
    if not FRAMES:
        return
    if surf_update is not None:
        surf_update()
    cop.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse())
    cam.GetFocalLengthAttr().Set(float(focal))
    for _ in range(2):
        app.update()
    rep.orchestrator.step(rt_subframes=16 if QUICK else 40, delta_time=0.0)
    rep.orchestrator.step(rt_subframes=1, delta_time=0.0)
    path = os.path.join(OUT, f"t_{len(shots):02d}_{tag}.png")
    Image.fromarray(np.asarray(rgb.get_data())[..., :3].astype(np.uint8)).save(path)
    shots.append(path)
    print("[frame]", tag, flush=True)


surf_update = None
# orientations: arm_0 approaches along +y (fingers along x); arm_1 approaches along -y
R0 = np.column_stack([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).astype(float)
R1 = np.column_stack([[0, 0, -1], [1, 0, 0], [0, -1, 0]]).astype(float)
Q0, Q1 = mat2q(R0), mat2q(R1)
rig.move_joints(rig.lift, [LIFT0, LIFT0], seconds=2.0)      # raise the arms above worktop height
rig.step(int(0.3 * HZ))
collider_box("/World/Colliders/Island", (0.905, 3.14, 0.04), (2.7445, 2.696, ISLAND - 0.02))
rig.step(5)
ev("lift_raised", lift_m=[round(v, 3) for v in rig.q()[rig.lift]], tcp0=np.round(rig.tcp("arm_0")[0], 3).tolist())
for a in ("arm_0", "arm_1"):
    rig.posture[a] = rig.q()[rig.idx[a]["arm"]].copy()
print("[diag] start egg", egg_rp.get_world_poses()[0].numpy()[0], "tcp0", rig.tcp("arm_0")[0], flush=True)
for _k in range(6):
    rig.step(20)
    print("[diag] t", round(rig.t, 2), "tcp0", np.round(rig.tcp("arm_0")[0], 3), "q", np.round(rig.q()[rig.idx["arm_0"]["arm"]], 2), flush=True)
rig.set_grip_force("arm_0", 20.0, stage)
rig.set_grip_force("arm_1", 20.0, stage)
rig.grip("arm_0", 0.075)
rig.grip("arm_1", 0.075)
rig.step(int(0.3 * HZ))
frame("start")

# ---- 1 PickUpEgg: grasp the pointed end, 15 mm in from the tip, from the -y side
e_p = egg_rp.get_world_poses()[0].numpy()[0].astype(float)
print("[diag] egg", e_p, "tcp0", rig.tcp("arm_0")[0], "tcp1", rig.tcp("arm_1")[0], flush=True)
tip_y = e_p[1] - L_egg / 2
g0 = np.array([e_p[0], tip_y + 0.015, e_p[2]])
ok = rig.move_to("arm_0", g0 + [0, -0.08, 0.06], Q0, speed=0.25)
ok = rig.move_to("arm_0", g0 + [0, -0.06, 0.0], Q0, speed=0.15)
ok = rig.move_to("arm_0", g0, Q0, speed=0.06, tol_mm=1.5)
ev("pregrasp_reached", tcp_err_mm=ok[1])
F_grip = 3.0
adm, pr = belief.admissible(F_grip, math.radians(150), a_max=2.0)
ev("grip_check_H8", F_N=F_grip, admissible=bool(adm), **{k: round(v, 3) for k, v in pr.items()})
rig.set_grip_force("arm_0", F_grip, stage)
rig.grip("arm_0", 0.0, speed=0.04)
rig.step(int(0.3 * HZ))
ev("grasped", opening_mm=round(rig.finger_opening("arm_0") * 1e3, 1))
frame("grasped", eye=(2.25, 1.95, 1.2), target=(2.74, 2.3, 0.99), focal=45)
lift_to = g0 + [0, 0, 0.12]
rig.move_to("arm_0", lift_to, Q0, speed=0.15, tol_mm=2.0)
e_p2 = egg_rp.get_world_poses()[0].numpy()[0].astype(float)
ev("lifted", egg_rise_mm=round((e_p2[2] - e_p[2]) * 1e3, 1))

# ---- 2 CrackEgg on the bowl rim
rig.set_grip_force("arm_0", 8.0, stage)                 # firmer for the strike (still < F_crack / 3)
rig.step(int(0.2 * HZ))
tcp, q_tcp, _ = rig.tcp("arm_0")
e_p = egg_rp.get_world_poses()[0].numpy()[0].astype(float)
off = e_p - tcp                                           # egg centre relative to the TCP (world)
above = np.array([STRIKE_C[0], STRIKE_C[1], RIM_TOP + R_eq + 0.035]) - off
rig.move_to("arm_0", above, Q0, speed=0.2, tol_mm=2.0)
frame("over_rim", eye=(2.25, 2.2, 1.25), target=(2.74, 2.6, 1.02), focal=45)
hit = {"v": None, "F": 0.0}


def contact_watch(_dt):
    heads, data = get_physx_simulation_interface().get_contact_report()
    for h in heads:
        a = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
        b = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
        if ("/World/Egg" in (a, b)) and any(s.startswith("/World/Bowl") for s in (a, b)):
            imp = sum(np.linalg.norm(np.array(data[k].impulse)) for k in
                      range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data))
            if hit["v"] is None:
                hit["v"] = float(-egg_rp.get_velocities()[0].numpy()[0][2])
            hit["F"] = max(hit["F"], imp * HZ)


rig.on_step.append(contact_watch)
result = None
v_strike = 0.25
for attempt in range(3):
    hit.update(v=None, F=0.0)
    start = rig.tcp("arm_0")[0]
    goal = start - np.array([0, 0, 0.055])
    n = int(0.055 / v_strike * HZ)
    for i in range(1, n + 1):                             # straight strike stroke, then stop at contact
        rig.ik_step("arm_0", start + (goal - start) * i / n, Q0, gain=1.0, max_dq=0.05)
        rig.step()
        if hit["v"] is not None:
            break
    rig.step(int(0.05 * HZ))
    v_meas = hit["v"] if hit["v"] is not None else 0.0
    r = EC.strike(truth, max(v_meas, 0.0), {"kind": "rim", "surface": "bowl_rim_ceramic", "edge_radius": 2.0e-3},
                  rng)
    ev("strike", attempt=attempt + 1, v_cmd=v_strike, v_contact=round(v_meas, 3), physx_peak_N=round(hit["F"], 1),
       model_F=round(r.F_peak, 1), F_crack=round(r.F_crack, 1), outcome=r.state, shards=r.shards_inside)
    rig.move_to("arm_0", above, Q0, speed=0.15, tol_mm=3.0)
    if r.state in ("check", "cracked_open", "shattered"):
        result = r
        break
    v_strike *= 1.15
rig.on_step.remove(contact_watch)
frame("cracked", eye=(2.25, 2.2, 1.25), target=(2.74, 2.6, 1.02), focal=45)

# ---- 3 OpenUp: swap to fragments + contents, second hand, swing apart over the bowl
inp = np.load(os.path.join(EGG_DIR, "open_sim_inputs.npz"))
report = {"egg": EGG_SEED, "events": EVENTS}
if result is not None and result.state != "shattered":
    pe, qe = (v.numpy()[0].astype(float) for v in egg_rp.get_world_poses())
    Re = qmat(qe)
    # hide and park the rigid egg; fragments take its pose
    UsdGeom.Imageable(stage.GetPrimAtPath("/World/Egg")).MakeInvisible()
    egg_rp.set_world_poses(positions=[[0.0, 0.0, -3.0]])
    UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath("/World/Egg")).GetKinematicEnabledAttr().Set(True) \
        if UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath("/World/Egg")).GetKinematicEnabledAttr() else \
        UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath("/World/Egg")).CreateKinematicEnabledAttr(True)
    frag_ops = {}
    for side in ("Pointed", "Blunt"):
        fpath = f"/World/Frag{side}"
        x = UsdGeom.Xform.Define(stage, fpath)
        x.GetPrim().GetReferences().AddReference(os.path.join(EGG_DIR, "shell_fragments_inplace.usdc"))
        UsdPhysics.RigidBodyAPI.Apply(x.GetPrim()).CreateKinematicEnabledAttr(True)
        frag_ops[side] = x.AddTransformOp()
    app.update()
    for side in ("Pointed", "Blunt"):
        other = "Blunt" if side == "Pointed" else "Pointed"
        stage.GetPrimAtPath(f"/World/Frag{side}/ShellFragment_{other}").SetActive(False)
        for prim in stage.GetPrimAtPath(f"/World/Frag{side}/ShellFragment_{side}").GetChildren():
            if prim.IsA(UsdGeom.Mesh) and prim.GetName() in ("InnerShellMembrane", "Cuticle"):
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")

    def set_frag(side, R, p):
        M = Gf.Matrix4d(1.0)
        M.SetRotateOnly(Gf.Matrix3d(*R.T.ravel().tolist()))
        M.SetTranslateOnly(Gf.Vec3d(*map(float, p)))
        frag_ops[side].Set(M)

    set_frag("Pointed", Re, pe)
    set_frag("Blunt", Re, pe)
    # contents: albumen particles + rigid yolk in the egg's current pose
    spacing = float(inp["spacing"])
    fluid_rest, solid_rest = 0.5 * spacing, 0.6 * spacing
    ps = Sdf.Path("/World/Albumen")
    particleUtils.add_physx_particle_system(stage, ps, contact_offset=solid_rest + 0.6e-3, rest_offset=solid_rest,
                                            particle_contact_offset=fluid_rest / 0.6, solid_rest_offset=solid_rest,
                                            fluid_rest_offset=fluid_rest, solver_position_iterations=8,
                                            simulation_owner="/World/Physics", max_neighborhood=96, max_velocity=3.0)
    particleUtils.add_pbd_particle_material(stage, "/World/Looks/AlbumenPBD", viscosity=0.0012, cohesion=0.3,
                                            surface_tension=0.05, adhesion=0.01, friction=0.2, density=1035.0)
    physicsUtils.add_physics_material_to_prim(stage, stage.GetPrimAtPath(ps), "/World/Looks/AlbumenPBD")
    pos = inp["albumen"] @ Re.T + pe
    pts_path = Sdf.Path("/World/AlbumenParticles")
    particleUtils.add_physx_particleset_points(stage, pts_path, Vt.Vec3fArray.FromNumpy(pos.astype(np.float32)),
                                              Vt.Vec3fArray.FromNumpy(np.zeros_like(pos, np.float32)),
                                              Vt.FloatArray.FromNumpy(np.full(len(pos), 2 * fluid_rest, np.float32)),
                                              ps, self_collision=True, fluid=True, particle_group=0,
                                              particle_mass=1035.0 * spacing ** 3, density=1035.0)
    UsdGeom.Imageable(stage.GetPrimAtPath(pts_path)).CreatePurposeAttr(UsdGeom.Tokens.guide)
    alb_look = mat("/World/Looks/AlbumenRender", "OmniGlass", glass_color=(0.98, 0.99, 0.94), glass_ior=1.36)
    surf = UsdGeom.Mesh.Define(stage, "/World/AlbumenSurface")
    UsdShade.MaterialBindingAPI.Apply(surf.GetPrim()).Bind(alb_look)
    UsdGeom.PrimvarsAPI(surf).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
    pts = UsdGeom.Points(stage.GetPrimAtPath(pts_path))

    def _surf(h=0.8e-3, sigma=1.1):
        from scipy import ndimage
        from skimage import measure
        P = np.asarray(pts.GetPointsAttr().Get())
        P = P[(np.abs(P[:, 0] - 2.74) < 0.15) & (np.abs(P[:, 1] - 2.6) < 0.2) & (P[:, 2] > 0.9) & (P[:, 2] < 1.3)]
        if len(P) < 10:
            return
        lo = P.min(0) - 4 * h
        idx = np.floor((P - lo) / h).astype(int)
        grid = np.zeros(idx.max(0) + 5, np.float32)
        np.add.at(grid, tuple(idx.T), 1.0)
        grid = ndimage.gaussian_filter(grid, sigma)
        try:
            v, f, _, _ = measure.marching_cubes(grid, 0.5 * (h / spacing) ** 3)
        except (ValueError, RuntimeError):
            return
        surf.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy((v * h + lo).astype(np.float32)))
        surf.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
        surf.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(f[:, ::-1].astype(np.int32).ravel()))
        surf.CreateSubdivisionSchemeAttr("none")

    surf_update = _surf
    ry = float(inp["yolk_r"])
    yc = Re @ inp["yolk_c"] + pe
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
    for _ in range(3):
        app.update()
    ev("swapped_to_fragments", particles=int(len(pos)), state=result.state)
    # fragments follow the hands: pointed fragment rigidly attached to arm_0's TCP
    tcp0, qt0, _ = rig.tcp("arm_0")
    Rt0 = qmat(qt0)
    rel_R0 = Rt0.T @ Re
    rel_p0 = Rt0.T @ (pe - tcp0)
    blunt_world = (Re.copy(), pe.copy())
    att = {"Blunt": None}

    def frag_follow(_dt):
        p_, q_, _ = rig.tcp("arm_0")
        R_ = qmat(q_)
        set_frag("Pointed", R_ @ rel_R0, p_ + R_ @ rel_p0)
        if att["Blunt"] is None:                          # still joined: moves with the pointed half
            set_frag("Blunt", R_ @ rel_R0, p_ + R_ @ rel_p0)
        else:
            p1, q1, _ = rig.tcp("arm_1")
            R1_ = qmat(q1)
            set_frag("Blunt", R1_ @ att["Blunt"][0], p1 + R1_ @ att["Blunt"][1])

    rig.on_step.append(frag_follow)
    rig.step(int(0.3 * HZ))
    # arm_1 grasps the blunt end from the +y side (15 mm in from the blunt tip)
    p_, q_, _ = rig.tcp("arm_0")
    egg_c = p_ + qmat(q_) @ rel_p0
    egg_axis = (qmat(q_) @ rel_R0)[:, 2]                 # egg +z (blunt) in world
    g1 = egg_c + egg_axis * (L_egg / 2 - 0.015)
    rig.move_to("arm_1", g1 + [0, 0.08, 0.06], Q1, speed=0.25, tol_mm=3.0)
    rig.move_to("arm_1", g1 + [0, 0.05, 0.0], Q1, speed=0.12, tol_mm=2.0)
    rig.move_to("arm_1", g1, Q1, speed=0.05, tol_mm=2.0)
    rig.set_grip_force("arm_1", 3.0, stage)
    rig.grip("arm_1", 0.0, speed=0.04)
    rig.step(int(0.2 * HZ))
    p1, q1, _ = rig.tcp("arm_1")
    R1_ = qmat(q1)
    Rb = qmat(q_) @ rel_R0
    att["Blunt"] = (R1_.T @ Rb, R1_.T @ (egg_c - p1))
    ev("second_hand_grasped", opening_mm=round(rig.finger_opening("arm_1") * 1e3, 1))
    frame("both_hands", eye=(2.25, 2.2, 1.25), target=(2.74, 2.6, 1.05), focal=40)
    # both hands carry the egg over the bowl centre and up a little
    shift = np.array([BOWL_C[0], BOWL_C[1], RIM_TOP + 0.10]) - egg_c
    p0s, p1s = rig.tcp("arm_0")[0], rig.tcp("arm_1")[0]
    n = int(np.linalg.norm(shift) / 0.08 * HZ)
    for i in range(1, n + 1):
        a = i / n
        rig.ik_step("arm_0", p0s + shift * a, Q0)
        rig.ik_step("arm_1", p1s + shift * a, Q1)
        rig.step()
    for _ in range(int(0.6 * HZ)):
        rig.ik_step("arm_0", p0s + shift, Q0)
        rig.ik_step("arm_1", p1s + shift, Q1)
        rig.step()
    # swing apart about the top of the crack (axis x): pointed half by -phi, blunt half by +phi
    egg_c = egg_c + shift
    crack_y = egg_c[1] + egg_axis[1] * float(inp["crack_z"])
    H = np.array([egg_c[0], crack_y, egg_c[2] + R_eq])
    P0, P1 = rig.tcp("arm_0")[0], rig.tcp("arm_1")[0]

    def rx(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    T_OPEN = 0.9
    n = int(T_OPEN * HZ)
    t_shots = {int(n * 0.35): "opening", n: "opened"}
    for i in range(1, n + 1):
        a = i / n
        e = a * a * (3 - 2 * a)
        phi = math.radians(95) * e
        Ra, Rb_ = rx(-phi), rx(phi)
        rig.ik_step("arm_0", H + Ra @ (P0 - H) + [0, -0.012 * e, 0], mat2q(Ra @ R0), gain=0.6, max_dq=0.03)
        rig.ik_step("arm_1", H + Rb_ @ (P1 - H) + [0, 0.012 * e, 0], mat2q(Rb_ @ R1), gain=0.6, max_dq=0.03)
        rig.step()
        if i in t_shots:
            frame(t_shots[i], eye=(2.2, 2.25, 1.3), target=(2.74, 2.68, 1.03), focal=38)
    for k in range(int(2.0 * HZ)):
        rig.ik_step("arm_0", H + rx(-math.radians(95)) @ (P0 - H) + [0, -0.012, 0], mat2q(rx(-math.radians(95)) @ R0))
        rig.ik_step("arm_1", H + rx(math.radians(95)) @ (P1 - H) + [0, 0.012, 0], mat2q(rx(math.radians(95)) @ R1))
        rig.step()
    P = np.asarray(pts.GetPointsAttr().Get())
    in_bowl = (np.hypot(P[:, 0] - BOWL_C[0], P[:, 1] - BOWL_C[1]) < 0.072) & (P[:, 2] < RIM_TOP) & (P[:, 2] > ISLAND)
    yp = RigidPrim("/World/Yolk").get_world_poses()[0].numpy()[0]
    yolk_in = bool(np.hypot(yp[0] - BOWL_C[0], yp[1] - BOWL_C[1]) < 0.07 and yp[2] < RIM_TOP)
    opened, yolk_state, shards_in = EC.open_up(truth, rng, age_days=egg.p["age_days"])
    ev("opened", albumen_in_bowl=round(float(in_bowl.mean()), 4), yolk_in_bowl=yolk_in, yolk=yolk_state,
       shards_in_bowl=shards_in)
    frame("in_bowl", eye=(2.3, 2.3, 1.35), target=(2.74, 2.67, 0.98), focal=40)
    report.update({"albumen_in_bowl": float(in_bowl.mean()), "yolk_in_bowl": yolk_in, "yolk": yolk_state,
                   "shards_in_bowl": shards_in, "crack": result.state})
else:
    report.update({"crack": None if result is None else result.state})
json.dump(report, open(os.path.join(ROOT, "test_reports", "garmi_egg.json"), "w"), indent=1, default=str)
print("[result]", {k: v for k, v in report.items() if k != "events"}, flush=True)
sys.stdout.flush()
os._exit(0)
