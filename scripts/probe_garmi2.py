"""Bisection probe: the stable GARMI probe plus the task scene's elements, each behind a flag (P_<NAME>=1)."""
import os, sys, math
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
F = lambda k: os.environ.get("P_" + k, "0") == "1"  # noqa: E731
from isaacsim import SimulationApp  # noqa: E402
app = SimulationApp({"headless": True})
import carb  # noqa: E402
import omni.kit.app  # noqa: E402
if F("NOFABRIC"):
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.physx.fabric", False)
    carb.settings.get_settings().set("/physics/updateToUsd", True)
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
px = PhysxSchema.PhysxSceneAPI.Apply(sc.GetPrim()); px.CreateTimeStepsPerSecondAttr(240)
if F("GPU"):
    px.CreateEnableGPUDynamicsAttr(True); px.CreateBroadphaseTypeAttr("GPU")
if F("EGG"):
    ex = UsdGeom.Xform.Define(stage, "/World/Egg")
    ex.GetPrim().GetReferences().AddReference(os.path.join(ROOT, "assets/food/egg_1/whole.usdc"))
    ex.AddTranslateOp().Set(Gf.Vec3d(2.74, 2.30, 1.2))
if F("ISLAND"):
    c = UsdGeom.Cube.Define(stage, "/World/Island"); c.CreateSizeAttr(1.0)
    UsdGeom.Xformable(c).AddTranslateOp().Set(Gf.Vec3d(2.7445, 2.696, 0.926)); UsdGeom.Xformable(c).AddScaleOp().Set(Gf.Vec3f(0.905, 3.14, 0.04))
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
g = UsdGeom.Xform.Define(stage, "/World/GARMI")
g.GetPrim().GetReferences().AddReference(os.path.join(ROOT, "assets/apartments/garmi_robot/garmi_clean/garmi_clean.usda"))
g.AddTranslateOp().Set(Gf.Vec3d(float(os.environ.get("P_X", "3.62")), 2.55, 0.0)); g.AddRotateZOp().Set(180.0)
g.GetPrim().GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
app.update()
root = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)][0]
for p in stage.Traverse():
    if p.HasAPI(UsdPhysics.RigidBodyAPI) and str(p.GetPath()).startswith("/World/GARMI"):
        m = UsdPhysics.MassAPI.Apply(p)
        if not m.GetMassAttr().Get() or m.GetMassAttr().Get() <= 0:
            m.CreateMassAttr(0.2)
            if F("INERTIA"):
                m.CreateDiagonalInertiaAttr(Gf.Vec3f(2e-4, 2e-4, 2e-4))
fj = UsdPhysics.FixedJoint.Define(stage, "/World/GarmiAnchor")
fj.CreateBody1Rel().SetTargets([root.GetPath()])
p_root = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(0).ExtractTranslation()
fj.CreateLocalPos0Attr().Set(Gf.Vec3f(*p_root)); fj.CreateLocalRot0Attr().Set(Gf.Quatf(0, 0, 0, 1))
app_utils.play(commit=True)
SimulationManager.step(steps=10)
rig = GarmiRig(str(root.GetPath()), log=lambda *a: None); rig.start()
SimulationManager.step(steps=120)
from isaacsim.core.experimental.prims import RigidPrim
for path in rig.art.link_paths[0]:
    pp = RigidPrim(path).get_world_poses()[0].numpy()[0]
    if pp[0] < 3.3 and pp[2] < 1.1:
        print("[link]", path.split("/")[-1], np.round(pp, 3), flush=True)
print("[probe2]", {k: F(k) for k in ("NOFABRIC", "GPU", "EGG", "ISLAND", "INERTIA")}, "tcp0", np.round(rig.tcp("arm_0")[0], 3), flush=True)
sys.stdout.flush(); os._exit(0)
