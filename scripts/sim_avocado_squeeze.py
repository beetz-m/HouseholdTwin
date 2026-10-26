"""Ripeness squeeze test in Isaac Sim: PhysX FEM volume-deformable avocados at ripeness stages 1..5 squeezed between
a fixed plate and a force-limited moving plate (force steps 0.5 .. 8 N).  The flesh modulus per stage comes from
food/avocado.flesh_E_MPa (farinas2021 endpoints 2.29 -> 0.16 MPa).  The force-displacement curves calibrate the
stiffness model of the robot's ripeness belief (food/sdt.AvocadoBelief).

Writes test_reports/avocado_squeeze.json (+ renders/food/avocado_squeeze.png via tools/plot_avocado_squeeze.py).
Run (GPU job alone): OMNI_KIT_ACCEPT_EULA=YES ~/Documents/LabAutomationTwin/env_isaacsim/bin/python \
    scripts/sim_avocado_squeeze.py
"""

import json
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HZ = int(os.environ.get('SQ_HZ', 240))
STAGES = ["rigid", 1.0, 2.0, 3.0, 4.0, 4.5, 5.0]   # "rigid": control lane measuring the rig compliance
FORCES = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
HOLD = 0.8
E_OVERRIDE = None
for a in sys.argv[1:]:
    if a.startswith("--E="):
        E_OVERRIDE = [float(v) for v in a[4:].split(",")]
        STAGES = list(range(len(E_OVERRIDE)))

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.experimental.prims import RigidPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from omni.physx.scripts import deformableUtils, physicsUtils  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt  # noqa: E402

from food.avocado import Avocado, flesh_E_MPa  # noqa: E402
from food.geometry import Meridian, resample  # noqa: E402
from food import meshlib as ML  # noqa: E402

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

avo = Avocado(4)
r, z = resample(avo.mer.rho, avo.mer.z, 90)
r[0] = r[-1] = 0
surf = ML.surface_of_revolution(Meridian(r, z), 48, modulation=avo.mod)
D = float(2 * avo.mer.rho.max())
L = float(avo.mer.zmax - avo.mer.zmin)


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
for i, st in enumerate(STAGES):
    y0 = i * 0.25
    box(f"/World/L{i}/Table", (0.3, 0.2, 0.02), (0, y0, -0.01))
    # avocado lying with its long axis along y, squeezed across its diameter along x
    root = f"/World/L{i}/Avocado"
    xf = UsdGeom.Xform.Define(stage, root)
    xf.AddTranslateOp().Set(Gf.Vec3d(0, y0, D / 2 + 0.0015))
    xf.AddRotateXOp().Set(90.0)
    m = UsdGeom.Mesh.Define(stage, root + "/mesh")
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(surf.vertices, np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(surf.faces), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(surf.faces, np.int32).ravel()))
    if st == "rigid":
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("convexHull")
        UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
        UsdPhysics.MassAPI.Apply(xf.GetPrim()).CreateMassAttr(0.2)
    else:
      deformableUtils.create_auto_volume_deformable_hierarchy(
        stage, root_prim_path=root, simulation_tetmesh_path=root + "/simMesh", collision_tetmesh_path=root + "/collMesh",
        cooking_src_mesh_path=root + "/mesh", simulation_hex_mesh_enabled=True,
        cooking_src_simplification_enabled=False, set_visibility_with_guide_purpose=True)
    prim = stage.GetPrimAtPath(root)
    E = float(flesh_E_MPa(st)) * 1e6 if st != "rigid" else float("inf")
    if E_OVERRIDE is not None:
        E = E_OVERRIDE[i] * 1e6
    if st != "rigid":
        prim.GetAttribute("physxDeformableBody:resolution").Set(22)
        prim.ApplyAPI("PhysxBaseDeformableBodyAPI")
        prim.GetAttribute("physxDeformableBody:selfCollision").Set(False)
        prim.GetAttribute("physxDeformableBody:solverPositionIterationCount").Set(int(os.environ.get("SQ_IT", 60)))
        mp = f"/World/L{i}/Mat"
        deformableUtils.add_deformable_material(stage, mp, density=990.0, static_friction=0.8, dynamic_friction=0.6,
                                                youngs_modulus=E, poissons_ratio=0.45)
        stage.GetPrimAtPath(mp).ApplyAPI("PhysxDeformableMaterialAPI")
        stage.GetPrimAtPath(mp).GetAttribute("physxDeformableMaterial:elasticityDamping").Set(0.002)
        physicsUtils.add_physics_material_to_prim(stage, prim, mp)
        coll = PhysxSchema.PhysxCollisionAPI.Apply(stage.GetPrimAtPath(root + "/collMesh"))
        coll.GetContactOffsetAttr().Set(0.0006)
        coll.CreateRestOffsetAttr().Set(0.0)
    # fixed plate at -x, moving plate at +x on a prismatic joint with a force-limited drive
    box(f"/World/L{i}/FixedPlate", (0.01, 0.12, 0.08), (-D / 2 - 0.0055, y0, 0.04))
    anchor = box(f"/World/L{i}/Anchor", (0.01, 0.01, 0.01), (D / 2 + 0.06, y0, 0.04), kinematic=True)
    plate = box(f"/World/L{i}/Plate", (0.01, 0.12, 0.08), (D / 2 + 0.0058, y0, 0.04), dynamic=True, mass=0.05)
    j = UsdPhysics.PrismaticJoint.Define(stage, f"/World/L{i}/Slide")
    j.CreateAxisAttr("X")
    j.CreateBody0Rel().SetTargets([anchor.GetPath()])
    j.CreateBody1Rel().SetTargets([plate.GetPath()])
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(-(0.06 - 0.0058), 0, 0))
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
    drv.CreateTypeAttr("force")
    drv.CreateTargetPositionAttr(-0.04)                # well inside the fruit: the force cap governs
    drv.CreateStiffnessAttr(5.0e4)
    drv.CreateDampingAttr(60.0)
    drv.CreateMaxForceAttr(0.05)
    lanes.append({"stage": st, "E_MPa": E / 1e6 if st != "rigid" else None, "drive": drv, "plate": f"/World/L{i}/Plate"})

app_utils.play(commit=True)
SimulationManager.step(steps=int(1.0 * HZ))          # settle on the table under a 0.05 N contact preload
plates = RigidPrim([l["plate"] for l in lanes])
x0 = plates.get_world_poses()[0].numpy()[:, 0].copy()
curves = {i: [] for i in range(len(lanes))}
for F in FORCES:
    for l in lanes:
        l["drive"].GetMaxForceAttr().Set(float(F))
    SimulationManager.step(steps=int(HOLD * HZ))
    x = plates.get_world_poses()[0].numpy()[:, 0]
    for i in range(len(lanes)):
        curves[i].append({"F": F, "d_mm": float((x0[i] - x[i]) * 1e3)})
    print("[F]", F, np.round((x0 - x) * 1e3, 3).tolist(), flush=True)
# rig compliance measured with the rigid control lane: < 1e-4 mm up to 1 N (above that the free rigid body is pushed
# away and the lane is no longer a measurement); treated as zero
rig = {p["F"]: 0.0 for p in curves[0]}
res = []
for i, l in enumerate(lanes):
    if l["stage"] == "rigid":
        continue
    for p in curves[i]:
        p["d_fruit_mm"] = p["d_mm"] - rig[p["F"]]
    c = curves[i]
    d4 = next(p["d_fruit_mm"] for p in c if p["F"] == 4.0)
    res.append({"stage": l["stage"], "E_MPa": l["E_MPa"], "curve": c, "k_secant_4N_N_per_mm": 4.0 / max(d4, 1e-6)})
os.makedirs(os.path.join(ROOT, "test_reports"), exist_ok=True)
json.dump({"avocado": "avocado_4 outer surface, uniform flesh modulus (seed and skin not separate in the FEM)",
           "rig_compliance_mm": rig,
           "D_mm": D * 1e3, "L_mm": L * 1e3, "lanes": res}, open(os.path.join(ROOT, "test_reports",
                                                                             "avocado_squeeze.json"), "w"), indent=1)
print("[rig]", rig, flush=True)
for r_ in res:
    print("[result]", r_["stage"], round(r_["E_MPa"], 3), "MPa  k(4N) =", round(r_["k_secant_4N_N_per_mm"], 2), "N/mm")
app.close()
