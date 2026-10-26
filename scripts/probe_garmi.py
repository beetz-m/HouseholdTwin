"""Probe: GARMI in the EASE kitchen, fixed base; print dofs, hand TCP poses, and reachability of the island board."""
import os, sys
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from isaacsim import SimulationApp  # noqa: E402
app = SimulationApp({"headless": True})
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics  # noqa: E402
from household.garmi_rig import GarmiRig  # noqa: E402

omni.usd.get_context().new_stage(); app.update()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
sc = UsdPhysics.Scene.Define(stage, "/World/Physics")
sc.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1)); sc.CreateGravityMagnitudeAttr(9.81)
PhysxSchema.PhysxSceneAPI.Apply(sc.GetPrim()).CreateTimeStepsPerSecondAttr(240)
g = UsdGeom.Xform.Define(stage, "/World/GARMI")
g.GetPrim().GetReferences().AddReference(os.path.join(ROOT, "assets/apartments/garmi_robot/garmi_clean/garmi_clean.usda"))
g.AddTranslateOp().Set(Gf.Vec3d(3.62, 2.55, 0.0)); g.AddRotateZOp().Set(180.0)
g.GetPrim().GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
app.update()
root = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)][0]
nfix = 0
for p in stage.Traverse():                      # links imported without mass (covers, sensors): give them 0.2 kg
    if p.HasAPI(UsdPhysics.RigidBodyAPI):
        m = UsdPhysics.MassAPI.Apply(p)
        if not m.GetMassAttr().Get() or m.GetMassAttr().Get() <= 0:
            m.CreateMassAttr(0.2); nfix += 1
print("massless links fixed", nfix)
print("art root", root.GetPath())
fj = UsdPhysics.FixedJoint.Define(stage, "/World/GarmiAnchor")
fj.CreateBody1Rel().SetTargets([root.GetPath()])
p_root = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(0).ExtractTranslation()
fj.CreateLocalPos0Attr().Set(Gf.Vec3f(*p_root))
fj.CreateLocalRot0Attr().Set(Gf.Quatf(0, 0, 0, 1))
app_utils.play(commit=True)
SimulationManager.step(steps=10)
rig = GarmiRig(str(root.GetPath())); rig.start()
print("dofs", rig.names)
q = rig.q(); print("q", np.round(np.degrees(q), 1))
for a in ("arm_0", "arm_1"):
    p, qq, _ = rig.tcp(a); print(a, "tcp", np.round(p, 3), "quat", np.round(qq, 3))
SimulationManager.step(steps=240)
for a in ("arm_0", "arm_1"):
    p, qq, _ = rig.tcp(a); print(a, "tcp after 1 s", np.round(p, 3))
print("lift", rig.lim[0][rig.lift], rig.lim[1][rig.lift])
sys.stdout.flush()
os._exit(0)
