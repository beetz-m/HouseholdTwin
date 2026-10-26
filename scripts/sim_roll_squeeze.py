"""Staling squeeze test in Isaac Sim: PhysX FEM volume-deformable bread rolls (roll_0 outer shape) squeezed across
their width between a fixed pad and a force-limited moving pad (30 x 30 mm gripper pads at mid-height).  One lane per
crumb modulus of the staling curve (lagrain_2012: ~20 / 70 / 125 / 157 kPa at 5 h / 2 d / 4 d / 7 d), Poisson 0.103
(lagrain_2012), bulk density from the twin.  The crust is NOT a separate layer here (a fresh crisp crust adds
stiffness and crackles; GAP: no crust modulus held) -- this lane set calibrates the crumb part of the robot's squeeze
observation (food/roll_models.RollBelief.observe_squeeze).

Writes test_reports/roll_squeeze.json.  Run (GPU job alone):
  OMNI_KIT_ACCEPT_EULA=YES ~/Documents/LabAutomationTwin/env_isaacsim/bin/python scripts/sim_roll_squeeze.py
"""

import json
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HZ = int(os.environ.get("SQ_HZ", 240))
IT = int(os.environ.get("SQ_IT", 60))
AGES_H = [5.0, 48.0, 96.0, 168.0]
FORCES = [0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0]
HOLD = 0.8
PAD = 0.030

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.experimental.prims import RigidPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from omni.physx.scripts import deformableUtils, physicsUtils  # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, Vt  # noqa: E402

from food.roll import Roll  # noqa: E402
from food.roll_models import crumb_E_kPa  # noqa: E402
from food.shapes import implicit_mesh  # noqa: E402

omni.usd.get_context().new_stage()
app.update()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
scene = UsdPhysics.Scene.Define(stage, "/World/Physics")
scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr(9.81)
px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
px.CreateTimeStepsPerSecondAttr(HZ)
px.CreateEnableGPUDynamicsAttr(True)
px.CreateBroadphaseTypeAttr("GPU")

roll = Roll(0)
lo, hi = roll.bbox()
surf = implicit_mesh(roll.inside, lo, hi, h=1.2e-3, sigma=0.9, target_faces=6000)
W = float(2 * roll.b)
H = float(roll.H)
mass_kg = 0.048
vol = float(surf.volume)
rho = mass_kg / vol


def box(path, size, pos, dynamic=False, mass=0.2, kinematic=False):
    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp().Set(Gf.Vec3d(*pos))
    c = UsdGeom.Cube.Define(stage, path + "/Shape")
    c.CreateSizeAttr(1.0)
    UsdGeom.Xformable(c).AddScaleOp().Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    if dynamic or kinematic:
        UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
        if kinematic:
            UsdPhysics.RigidBodyAPI(x.GetPrim()).CreateKinematicEnabledAttr(True)
        else:
            UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(mass)
        PhysxSchema.PhysxRigidBodyAPI.Apply(x.GetPrim()).CreateSleepThresholdAttr(0.0)
    return x


lanes = []
for i, age in enumerate(AGES_H):
    x0 = i * 0.30
    E = float(crumb_E_kPa(age)) * 1e3
    box(f"/World/L{i}/Table", (0.2, 0.3, 0.02), (x0, 0, -0.01))
    root = f"/World/L{i}/Roll"
    xf = UsdGeom.Xform.Define(stage, root)
    xf.AddTranslateOp().Set(Gf.Vec3d(x0, 0, 0.0015))
    m = UsdGeom.Mesh.Define(stage, root + "/mesh")
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(surf.vertices, np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(surf.faces), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(surf.faces, np.int32).ravel()))
    deformableUtils.create_auto_volume_deformable_hierarchy(
        stage, root_prim_path=root, simulation_tetmesh_path=root + "/simMesh", collision_tetmesh_path=root + "/collMesh",
        cooking_src_mesh_path=root + "/mesh", simulation_hex_mesh_enabled=True,
        cooking_src_simplification_enabled=False, set_visibility_with_guide_purpose=True)
    prim = stage.GetPrimAtPath(root)
    prim.GetAttribute("physxDeformableBody:resolution").Set(20)
    prim.ApplyAPI("PhysxBaseDeformableBodyAPI")
    prim.GetAttribute("physxDeformableBody:selfCollision").Set(False)
    prim.GetAttribute("physxDeformableBody:solverPositionIterationCount").Set(IT)
    mp = f"/World/L{i}/Mat"
    deformableUtils.add_deformable_material(stage, mp, density=rho, static_friction=0.8, dynamic_friction=0.6,
                                            youngs_modulus=E, poissons_ratio=0.103)
    stage.GetPrimAtPath(mp).ApplyAPI("PhysxDeformableMaterialAPI")
    stage.GetPrimAtPath(mp).GetAttribute("physxDeformableMaterial:elasticityDamping").Set(0.01)
    physicsUtils.add_physics_material_to_prim(stage, prim, mp)
    coll = PhysxSchema.PhysxCollisionAPI.Apply(stage.GetPrimAtPath(root + "/collMesh"))
    coll.GetContactOffsetAttr().Set(0.0008)
    coll.CreateRestOffsetAttr().Set(0.0)
    zc = 0.45 * H
    # fixed pad at -y, moving pad at +y on a prismatic joint with a force-limited drive
    box(f"/World/L{i}/FixedPad", (PAD, 0.01, PAD), (x0, -W / 2 - 0.0058, zc))
    anchor = box(f"/World/L{i}/Anchor", (0.01, 0.01, 0.01), (x0, W / 2 + 0.06, zc), kinematic=True)
    pad = box(f"/World/L{i}/Pad", (PAD, 0.01, PAD), (x0, W / 2 + 0.0058, zc), dynamic=True, mass=0.03)
    j = UsdPhysics.PrismaticJoint.Define(stage, f"/World/L{i}/Slide")
    j.CreateAxisAttr("Y")
    j.CreateBody0Rel().SetTargets([anchor.GetPath()])
    j.CreateBody1Rel().SetTargets([pad.GetPath()])
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(0, -(0.06 - 0.0058), 0))
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
    drv.CreateTypeAttr("force")
    drv.CreateTargetPositionAttr(-0.06)
    drv.CreateStiffnessAttr(5.0e4)
    drv.CreateDampingAttr(40.0)
    drv.CreateMaxForceAttr(0.02)
    lanes.append({"age_h": age, "E_kPa": E / 1e3, "drive": drv, "pad": f"/World/L{i}/Pad"})

app_utils.play(commit=True)
SimulationManager.step(steps=int(1.0 * HZ))
pads = RigidPrim([l["pad"] for l in lanes])
y0 = pads.get_world_poses()[0].numpy()[:, 1].copy()
curves = {i: [] for i in range(len(lanes))}
for F in FORCES:
    for l in lanes:
        l["drive"].GetMaxForceAttr().Set(float(F))
    SimulationManager.step(steps=int(HOLD * HZ))
    y = pads.get_world_poses()[0].numpy()[:, 1]
    for i in range(len(lanes)):
        curves[i].append({"F": F, "d_mm": float((y0[i] - y[i]) * 1e3)})
    print("[F]", F, np.round((y0 - y) * 1e3, 3).tolist(), flush=True)
res = []
for i, l in enumerate(lanes):
    c = curves[i]
    d1 = next(p["d_mm"] for p in c if p["F"] == 1.0)
    res.append({"age_h": l["age_h"], "E_kPa": l["E_kPa"], "curve": c, "k_secant_1N_N_per_mm": 1.0 / max(d1, 1e-6),
                "strain_at_3N": c[-1]["d_mm"] / (W * 1e3)})
os.makedirs(os.path.join(ROOT, "test_reports"), exist_ok=True)
json.dump({"roll": "roll_0 outer shape (implicit mesh, 1.2 mm), uniform crumb modulus; crust not separate (GAP)",
           "pad_mm": PAD * 1e3, "W_mm": W * 1e3, "H_mm": H * 1e3, "density_kgm3": rho, "poisson": 0.103,
           "hz": HZ, "iterations": IT, "lanes": res},
          open(os.path.join(ROOT, "test_reports", "roll_squeeze.json"), "w"), indent=1)
for r_ in res:
    print("[result]", r_["age_h"], "h", round(r_["E_kPa"], 1), "kPa  k(1N) =", round(r_["k_secant_1N_N_per_mm"], 3),
          "N/mm  strain@3N", round(r_["strain_at_3N"], 3), flush=True)
app.close()
