"""Tomato plate-firmness cross-check in Isaac Sim: PhysX FEM tomatoes (tomato_15 outer shape, homogeneous) squeezed
between 100 mm plates (the ince16 set-up: plates, up to 20 N).  Lanes take the whole-fruit Hertz modulus of ANOTHER
study (tomato_mech: 0.58 MPa small red .. 1.97 MPa small green, Poisson 0.3) and the bench compares the FEM plate
stiffness with (a) Hertz theory for two sphere-plate contacts in series and (b) the plate firmness measured by
ince16 (initial 15.1 / 6.0 / 4.5 N/mm green / pink / red) -- an independent consistency check of the register.

Writes test_reports/tomato_squeeze.json.  Run (GPU job alone):
  OMNI_KIT_ACCEPT_EULA=YES ~/Documents/LabAutomationTwin/env_isaacsim/bin/python scripts/sim_tomato_squeeze.py
"""

import json
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HZ = int(os.environ.get("SQ_HZ", 240))
IT = int(os.environ.get("SQ_IT", 60))
E_MPA = [0.58, 1.0, 1.4, 1.97]
FORCES = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0]
HOLD = 0.6
PAD = 0.100

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.experimental.prims import RigidPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from omni.physx.scripts import deformableUtils, physicsUtils  # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, Vt  # noqa: E402

from food import meshlib as ML  # noqa: E402
from food.geometry import Meridian, resample  # noqa: E402
from food.tomato import Tomato  # noqa: E402

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

tom = Tomato(15)
r_, z_ = resample(*tom.curve(0.0), 90)
r_[0] = r_[-1] = 0
surf = ML.surface_of_revolution(Meridian(r_, z_), 48, modulation=tom.mod)
W = float(2 * tom.a)
YMAX = float(np.abs(np.asarray(surf.vertices)[:, 1]).max())     # ribbed outline: pads clear of the real extent
H = float(2 * tom.c)
mass_kg = 0.072
rho = mass_kg / float(surf.volume)
NU = 0.3


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
for i, Em in enumerate(E_MPA):
    x0 = i * 0.30
    E = Em * 1e6
    box(f"/World/L{i}/Table", (0.2, 0.3, 0.02), (x0, 0, -0.01))
    root = f"/World/L{i}/Roll"
    xf = UsdGeom.Xform.Define(stage, root)
    xf.AddTranslateOp().Set(Gf.Vec3d(x0, 0, float(tom.c) + 0.0015))
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
                                            youngs_modulus=E, poissons_ratio=NU)
    stage.GetPrimAtPath(mp).ApplyAPI("PhysxDeformableMaterialAPI")
    stage.GetPrimAtPath(mp).GetAttribute("physxDeformableMaterial:elasticityDamping").Set(0.01)
    physicsUtils.add_physics_material_to_prim(stage, prim, mp)
    coll = PhysxSchema.PhysxCollisionAPI.Apply(stage.GetPrimAtPath(root + "/collMesh"))
    coll.GetContactOffsetAttr().Set(0.0008)
    coll.CreateRestOffsetAttr().Set(0.0)
    zc = float(tom.c) + 0.0015
    # fixed pad at -y, moving pad at +y on a prismatic joint with a force-limited drive
    box(f"/World/L{i}/FixedPad", (PAD, 0.01, 1.2 * float(tom.c)), (x0, -YMAX - 0.0058, zc))
    anchor = box(f"/World/L{i}/Anchor", (0.01, 0.01, 0.01), (x0, YMAX + 0.06, zc), kinematic=True)
    pad = box(f"/World/L{i}/Pad", (PAD, 0.01, 1.2 * float(tom.c)), (x0, YMAX + 0.0058, zc), dynamic=True, mass=0.05)
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
    lanes.append({"E_MPa": Em, "drive": drv, "pad": f"/World/L{i}/Pad"})

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
R = float(tom.a)


def hertz_two_plates(F, E, nu=NU):
    """Total plate approach of a sphere of radius R between two flat plates (two Hertz contacts in series)."""
    Es = E / (1 - nu ** 2)
    d1 = (3 * F / (4 * Es * np.sqrt(R))) ** (2 / 3)
    return 2 * d1


res = []
for i, l in enumerate(lanes):
    c = curves[i]
    d = {p["F"]: p["d_mm"] for p in c}
    k_fem = (20.0 - 4.0) / max(d[20.0] - d[4.0], 1e-6)            # N/mm, secant 4 -> 20 N (contact closed)
    k_h = (20.0 - 4.0) / ((hertz_two_plates(20.0, l["E_MPa"] * 1e6) - hertz_two_plates(4.0, l["E_MPa"] * 1e6)) * 1e3)
    res.append({"E_MPa": l["E_MPa"], "curve": c, "k_fem_4_20N": k_fem, "k_hertz_4_20N": k_h})
    print("[result]", l["E_MPa"], "MPa  k_FEM =", round(k_fem, 2), "N/mm  k_Hertz =", round(k_h, 2), flush=True)
os.makedirs(os.path.join(ROOT, "test_reports"), exist_ok=True)
json.dump({"tomato": "tomato_15 outer shape, homogeneous FEM", "D_mm": W * 1e3, "H_mm": H * 1e3, "poisson": NU,
           "hz": HZ, "iterations": IT, "measured_ince16": {"initial_N_per_mm": {"green": 15.1, "pink": 6.0, "red": 4.5},
                                                          "average_N_per_mm": {"green": 13.2, "pink": 4.7, "red": 3.1}},
           "lanes": res}, open(os.path.join(ROOT, "test_reports", "tomato_squeeze.json"), "w"), indent=1)
app.close()
