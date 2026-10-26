"""EASE (IAI) apartment kitchen with GARMI and the food twins, path-traced.

The apartment and GARMI are the USD imports of the CRAM viz bundles (scripts/import_apartments.py); the food assets
are the part-graph twins.  The board stands on the free stretch of the kitchen island between cooktop and sink.
Run (GPU job alone): OMNI_KIT_ACCEPT_EULA=YES ~/Documents/LabAutomationTwin/env_isaacsim/bin/python \
    scripts/render_apartment_kitchen.py [--quick]
"""

import math
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
QUICK = "--quick" in sys.argv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, "assets")
OUT = os.path.join(ROOT, "renders", "apartment")
LAB_TEX = os.path.expanduser("~/Documents/LabAutomationTwin/assets/materials/textures")
RES = (1280, 800) if QUICK else (1920, 1200)
ISLAND_TOP = 0.946

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "PathTracing", "width": RES[0], "height": RES[1]})
import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade  # noqa: E402

s = carb.settings.get_settings()
s.set("/rtx/rendermode", "PathTracing")
SPP, SUB = (8, 24) if QUICK else (16, 64)
s.set("/rtx/pathtracing/spp", SPP)
s.set("/rtx/pathtracing/totalSpp", SPP * SUB)
s.set("/rtx/pathtracing/maxBounces", 8)
s.set("/rtx/pathtracing/optixDenoiser/enabled", True)
omni.usd.get_context().new_stage()
app.update()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)


def ref(path, asset, pos=(0, 0, 0), yaw=0.0, rot=None, prim_path=None):
    x = UsdGeom.Xform.Define(stage, path)
    if prim_path:
        x.GetPrim().GetReferences().AddReference(asset, prim_path)
    else:
        x.GetPrim().GetReferences().AddReference(asset)
    t = x.AddTranslateOp()
    t.Set(Gf.Vec3d(*pos))
    x.AddRotateZOp().Set(yaw)
    if rot is not None:
        x.AddRotateXYZOp().Set(Gf.Vec3f(*rot))
    return x, t


bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])


def drop(x, t, z_surface):
    """Rest an object on a surface by its bounding box."""
    for _ in range(2):
        app.update()
    bbc.Clear()
    lo = bbc.ComputeWorldBound(x.GetPrim()).ComputeAlignedRange().GetMin()
    p = t.Get()
    t.Set(Gf.Vec3d(p[0], p[1], p[2] + z_surface - lo[2] + 0.0003))


def mat(path, **kw):
    m = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.SetSourceAsset(Sdf.AssetPath("OmniSurface.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniSurface", "mdl")
    o = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    m.CreateSurfaceOutput("mdl").ConnectToSource(o)
    for k, v in kw.items():
        if isinstance(v, str):
            sh.CreateInput(k, Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(v))
        elif isinstance(v, tuple):
            sh.CreateInput(k, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*v))
        else:
            sh.CreateInput(k, Sdf.ValueTypeNames.Float).Set(float(v))
    return m


ref("/World/Apartment", os.path.join(A, "apartments", "ease", "apartment", "apartment", "apartment.usda"))
ref("/World/GARMI", os.path.join(A, "apartments", "garmi_robot", "garmi_clean", "garmi_clean.usda"),
    pos=(3.95, 2.55, 0.0), yaw=180.0)
# cutting board on the island (free stretch between cooktop y < 1.9 and sink y > 3.15)
oak = mat("/World/Looks/Oak", diffuse_reflection_color_image=os.path.join(LAB_TEX, "oak_albedo.png"),
          specular_reflection_roughness=0.45)
board = UsdGeom.Cube.Define(stage, "/World/Board")
board.CreateSizeAttr(1.0)
UsdGeom.Xformable(board).AddTranslateOp().Set(Gf.Vec3d(2.78, 2.50, ISLAND_TOP + 0.011))
UsdGeom.Xformable(board).AddScaleOp().Set(Gf.Vec3f(0.30, 0.42, 0.022))
UsdShade.MaterialBindingAPI.Apply(board.GetPrim()).Bind(oak)
top = ISLAND_TOP + 0.022
F = os.path.join(A, "food")
items = [("Apple", "apple_3/whole.usdc", (2.74, 2.36), 20.0, None),
         ("AppleHalf", "apple_3/half_b.usdc", (2.85, 2.40), 110.0, None),
         ("Avocado", "avocado_4/whole.usdc", (2.70, 2.58), -30.0, (90, 0, 0)),
         ("AvocadoHalf", "avocado_4/half_a.usdc", (2.86, 2.62), 60.0, None),
         ("Egg1", "egg_1/whole.usdc", (2.72, 2.70), 70.0, (90, 0, 0)),
         ("Egg2", "egg_0/whole.usdc", (2.78, 2.76), 20.0, (90, 0, 0)),
         ("Egg3", "egg_2/whole.usdc", (2.85, 2.73), -40.0, (90, 0, 0))]
for name, asset, xy, yaw, rot in items:
    x, t = ref(f"/World/Food/{name}", os.path.join(F, asset), pos=(xy[0], xy[1], 0.0), yaw=yaw, rot=rot)
    drop(x, t, top)
# lights: daylight dome through the windows + ceiling panels over the kitchen and the table
dome = UsdLux.DomeLight.Define(stage, "/World/Light/Sky")
dome.CreateIntensityAttr(1500)
dome.CreateColorAttr(Gf.Vec3f(0.85, 0.9, 1.0))
for i, (x, y) in enumerate(((1.5, 1.6), (1.5, 3.2), (2.8, 2.5), (5.0, 4.0))):
    L = UsdLux.RectLight.Define(stage, f"/World/Light/Panel{i}")
    L.CreateWidthAttr(0.9)
    L.CreateHeightAttr(0.9)
    L.CreateIntensityAttr(6000)
    L.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
    UsdGeom.Xformable(L).AddTranslateOp().Set(Gf.Vec3d(x, y, 2.95))
for _ in range(30):
    app.update()
cam = UsdGeom.Camera.Define(stage, "/World/Cam")
cam.CreateHorizontalApertureAttr(36.0)
cam.CreateVerticalApertureAttr(36.0 * RES[1] / RES[0])
cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
op = UsdGeom.Xformable(cam).AddTransformOp()
rp = rep.create.render_product(str(cam.GetPath()), RES)
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach(rp)
os.makedirs(OUT, exist_ok=True)


def shot(name, eye, target, focal):
    op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse())
    cam.GetFocalLengthAttr().Set(float(focal))
    for _ in range(4):
        app.update()
    rep.orchestrator.step(rt_subframes=SUB, delta_time=0.0)
    rep.orchestrator.step(rt_subframes=1, delta_time=0.0)
    Image.fromarray(np.asarray(rgb.get_data())[..., :3].astype(np.uint8)).save(os.path.join(OUT, name + ".png"))
    print("[render]", name, flush=True)


shot("ease_kitchen_overview", (5.6, 0.6, 2.3), (2.6, 2.6, 0.9), 18)
shot("ease_kitchen_island", (3.55, 1.85, 1.55), (2.78, 2.55, 0.97), 35)
shot("ease_island_food", (3.25, 2.15, 1.25), (2.78, 2.55, 0.97), 60)
app.close()
