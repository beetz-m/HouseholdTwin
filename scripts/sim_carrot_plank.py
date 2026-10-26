"""Why cooks plank a carrot first: knife-on-carrot rolling stability in Isaac Sim (rigid bodies).

Lanes: the whole carrot (carrot_0/whole.usdc: multi-hull collider along the bent root) and the planked half
(carrot_0/half_a.usdc lying on its flat cut face), each pressed by a knife blade across the root (edge inclined 5 deg),
either straight down ('press') or while the blade is drawn back and forth along its length ('stroke', +-20 mm at
0.1 m/s, steel-on-carrot friction 0.3 ILLUSTRATIVE).  A frictionless press acts through the axis of a round root and
cannot roll it; the slicing stroke drags the top of the root sideways through blade friction -- that is what rolls a
round carrot, and what a plank resists.  The blade is a dynamic box on a vertical prismatic joint whose drive
force ramps to the blade shear force for this root (CarrotTruth.cut: 41.1 N x (D/28.6)^2, jah18).  A rigid blade
cannot cut, so the bench answers the stability question only: how far does the carrot roll or slide before the
blade force reaches the cutting force?  (carrot_cuts: 'cut a plank first so it does not roll', TYPICAL advice.)

Writes test_reports/carrot_plank.json.  Run (GPU job alone):
  OMNI_KIT_ACCEPT_EULA=YES ~/Documents/LabAutomationTwin/env_isaacsim/bin/python scripts/sim_carrot_plank.py
"""

import json
import math
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HZ = 240
MODES = ["press", "stroke"]                            # straight press | press while drawing the blade +-20 mm
TILTS_DEG = [5.0]

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.experimental.prims import RigidPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

from food.carrot import Carrot  # noqa: E402
from food.carrot_models import CarrotTruth  # noqa: E402

A = os.path.join(ROOT, "assets", "food", "carrot_0")
omni.usd.get_context().new_stage()
app.update()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
scene = UsdPhysics.Scene.Define(stage, "/World/Physics")
scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr(9.81)
PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim()).CreateTimeStepsPerSecondAttr(HZ)

car = Carrot(0)
F_cut = CarrotTruth(car, 0.115).cut(0.0)["F_needed"]
rec = json.load(open(os.path.join(A, "instance.json")))
board_mat = UsdShade.Material.Define(stage, "/World/Looks/Board")
api = UsdPhysics.MaterialAPI.Apply(board_mat.GetPrim())
api.CreateStaticFrictionAttr(0.5)                      # wet carrot on a wooden board (ILLUSTRATIVE)
api.CreateDynamicFrictionAttr(0.4)
steel = UsdShade.Material.Define(stage, "/World/Looks/Steel")
api = UsdPhysics.MaterialAPI.Apply(steel.GetPrim())
api.CreateStaticFrictionAttr(0.3)
api.CreateDynamicFrictionAttr(0.25)
PhysxSchema.PhysxMaterialAPI.Apply(steel.GetPrim()).CreateFrictionCombineModeAttr("min")   # steel sets the pair


def box(path, size, pos, dynamic=False, mass=0.2, kinematic=False, rot=None, material=None):
    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp().Set(Gf.Vec3d(*pos))
    if rot is not None:
        x.AddRotateXYZOp().Set(Gf.Vec3f(*rot))
    c = UsdGeom.Cube.Define(stage, path + "/Shape")
    c.CreateSizeAttr(1.0)
    UsdGeom.Xformable(c).AddScaleOp().Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
    if dynamic or kinematic:
        UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
        if kinematic:
            UsdPhysics.RigidBodyAPI(x.GetPrim()).CreateKinematicEnabledAttr(True)
        else:
            UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(mass)
        PhysxSchema.PhysxRigidBodyAPI.Apply(x.GetPrim()).CreateSleepThresholdAttr(0.0)
    return x


bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
lanes = []
li = 0
for variant, asset in (("whole", "whole.usdc"), ("planked", "half_a.usdc")):
    for mode in MODES:
        tilt = TILTS_DEG[0]
        y0 = li * 0.12
        box(f"/World/L{li}/Board", (0.40, 0.10, 0.02), (0, y0, -0.01), material=board_mat)
        root = f"/World/L{li}/Carrot"
        x = UsdGeom.Xform.Define(stage, root)
        x.GetPrim().GetReferences().AddReference(os.path.join(A, asset))
        t = x.AddTranslateOp()
        # lay the root along x: object +z (tip -> crown) onto world +x; planked: then turn the cut face (-n) down.
        # Ops compose M = T Rx Ry: a vector is rotated by Ry first, then by Rx.
        if variant == "planked":
            n = np.asarray(rec["cuts"]["half_a"]["normal"], float)      # half_a keeps the +n side: cap faces -n
            x.AddRotateXOp(opSuffix="flat").Set(-90.0 if n[1] < 0 else 90.0)
        x.AddRotateXYZOp().Set(Gf.Vec3f(0, 90, 0))
        t.Set(Gf.Vec3d(0, y0, 0))
        for _ in range(2):
            app.update()
        bbc.Clear()
        bb = bbc.ComputeWorldBound(x.GetPrim()).ComputeAlignedRange()
        lo_, hi_ = np.array(bb.GetMin()), np.array(bb.GetMax())
        t.Set(Gf.Vec3d(-(lo_[0] + hi_[0]) / 2, 2 * y0 - (lo_[1] + hi_[1]) / 2, -lo_[2] + 0.0005))   # bound incl. y0
        col = stage.GetPrimAtPath(root + "/Collision")
        if col:
            for q in Usd.PrimRange(col):
                if q.HasAPI(UsdPhysics.CollisionAPI):
                    UsdShade.MaterialBindingAPI.Apply(q).Bind(board_mat, UsdShade.Tokens.weakerThanDescendants,
                                                              "physics")
        top_z = float(hi_[2] - lo_[2])
        # knife at 1/3 of the length from the crown, edge 1 cm above the top, offset across
        kx = 0.18 * (hi_[0] - lo_[0])
        if os.environ.get("PLANK_NOKNIFE"):
            lanes.append({"variant": variant, "mode": mode, "drive": None, "carrot": root, "knife": root})
            li += 1
            continue
        knife = box(f"/World/L{li}/Knife", (0.002, 0.07, 0.05), (kx, y0, top_z + 0.01 + 0.025),
                    dynamic=True, mass=0.15, rot=(tilt, 0, 0), material=steel)
        # D6 mount from the world: z = force-capped press, y = position-driven stroke along the blade, rest locked
        j = UsdPhysics.Joint.Define(stage, f"/World/L{li}/KMount")
        j.CreateBody1Rel().SetTargets([knife.GetPath()])
        j.CreateLocalPos0Attr().Set(Gf.Vec3f(float(kx), float(y0), float(top_z + 0.035)))
        j.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot0Attr().Set(Gf.Quatf(math.cos(math.radians(tilt) / 2), math.sin(math.radians(tilt) / 2), 0, 0))
        for dof in ("transX", "rotX", "rotY", "rotZ"):
            lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), dof)
            lim.CreateLowAttr(1.0)
            lim.CreateHighAttr(-1.0)                       # low > high: locked
        drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "transZ")
        drv.CreateTypeAttr("force")
        drv.CreateTargetPositionAttr(-0.2)
        drv.CreateStiffnessAttr(2.0e4)
        drv.CreateDampingAttr(200.0)
        drv.CreateMaxForceAttr(0.5)
        sdrv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "transY")
        sdrv.CreateTypeAttr("force")
        sdrv.CreateTargetPositionAttr(0.0)
        sdrv.CreateStiffnessAttr(3.0e3)
        sdrv.CreateDampingAttr(60.0)
        lanes.append({"variant": variant, "mode": mode, "stroke": sdrv, "y0": y0, "kx": kx, "drive": drv, "carrot": root, "knife": f"/World/L{li}/Knife"})
        li += 1

app_utils.play(commit=True)
carrots = RigidPrim([l["carrot"] for l in lanes])
if os.environ.get("PLANK_DEBUG"):
    for k in range(6):
        print("[dbg0] carrot z", np.round(carrots.get_world_poses()[0].numpy()[:, 2] * 1e3, 1).tolist(), flush=True)
        SimulationManager.step(steps=int(0.05 * HZ))
SimulationManager.step(steps=int(1.0 * HZ))
p0, q0 = (a.numpy().copy() for a in carrots.get_world_poses())
cap_obj = -np.asarray(rec["cuts"]["half_a"]["normal"], float)


def qrot(q, v):
    w, x, y, z = q
    u = np.array([x, y, z])
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


for i, l in enumerate(lanes):
    if l["variant"] == "planked":
        print("[pose] planked cap normal in world", np.round(qrot(q0[i], cap_obj), 3).tolist(), "z",
              round(p0[i][2] * 1e3, 1), flush=True)


def rot_angle(q, qr):
    w = abs(float(np.dot(q, qr)))
    return math.degrees(2 * math.acos(min(1.0, w)))


knives = RigidPrim([l["knife"] for l in lanes])
if os.environ.get("PLANK_DEBUG"):
    for k in range(5):
        for l in lanes:
            l["drive"].GetMaxForceAttr().Set(20.0)
        print("[dbg] knife z", np.round(knives.get_world_poses()[0].numpy()[:, 2] * 1e3, 1).tolist(),
              "carrot z", np.round(carrots.get_world_poses()[0].numpy()[:, 2] * 1e3, 1).tolist(), flush=True)
        SimulationManager.step(steps=int(0.25 * HZ))
log = {i: [] for i in range(len(lanes))}
t_sim = 0.0
for F in np.linspace(1.0, F_cut, 25):
    for l in lanes:
        l["drive"].GetMaxForceAttr().Set(float(F))
    for _ in range(int(0.15 * HZ)):                      # stroke: blade drawn along its length (y), +-20 mm
        t_sim += 1.0 / HZ
        for l in lanes:
            if l["mode"] == "stroke":
                l["stroke"].GetTargetPositionAttr().Set(float(0.020 * math.sin(2 * math.pi * t_sim / 0.8)))
        SimulationManager.step(steps=1)
    p, q = (a.numpy() for a in carrots.get_world_poses())
    for i in range(len(lanes)):
        log[i].append({"F": float(F), "roll_deg": rot_angle(q[i], q0[i]),
                       "slide_mm": float(np.linalg.norm((p[i] - p0[i])[:2]) * 1e3)})
res = []
for i, l in enumerate(lanes):
    c = log[i]
    first = next((r["F"] for r in c if r["roll_deg"] > 10 or r["slide_mm"] > 5), None)
    res.append({"variant": l["variant"], "mode": l["mode"], "roll_deg_at_Fcut": c[-1]["roll_deg"],
                "slide_mm_at_Fcut": c[-1]["slide_mm"], "F_at_escape_N": first, "log": c})
    print("[result]", l["variant"], l["mode"], ": roll", round(c[-1]["roll_deg"], 1), "deg, slide",
          round(c[-1]["slide_mm"], 1), "mm; escapes at", first, "N (F_cut", round(F_cut, 1), "N)", flush=True)
os.makedirs(os.path.join(ROOT, "test_reports"), exist_ok=True)
json.dump({"carrot": "carrot_0", "F_cut_N": F_cut,
           "escape_criterion": "roll > 10 deg or slide > 5 mm before F_cut", "lanes": res},
          open(os.path.join(ROOT, "test_reports", "carrot_plank.json"), "w"), indent=1)
app.close()
