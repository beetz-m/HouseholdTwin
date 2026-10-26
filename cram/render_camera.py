"""The PR2 head camera's view of the avocado on the kitchen counter (cram/avocado_pr2_demo.py): the instance's whole
fruit lying on its side as in the CRAM world (fruit frame: x = cut normal, y = long axis, z up), seen from the robot.

Run (one GPU job at a time):
  OMNI_KIT_ACCEPT_EULA=YES ~/Documents/LabAutomationTwin/env_isaacsim/bin/python cram/render_camera.py [assets/food/avocado_4]
Writes cram/camera/<instance>_counter.png.
"""

import math
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
QUICK = "--quick" in sys.argv
STEPS = {a.split("=", 1)[1] for a in sys.argv if a.startswith("--step=")} or {"scored", "halves", "pitted"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INST = os.path.abspath([a for a in sys.argv[1:] if not a.startswith("--")][0] if len([a for a in sys.argv[1:] if not a.startswith("--")]) else os.path.join(ROOT, "assets", "food", "avocado_4"))
A = os.path.join(ROOT, "assets", "food")
OUT = os.path.join(ROOT, "renders", "food")
LAB_TEX = os.path.expanduser("~/Documents/LabAutomationTwin/assets/materials/textures")
RES = (1280, 800)

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
SPP, SUB = (8, 16) if QUICK else (16, 48)
s.set("/rtx/pathtracing/spp", SPP)
s.set("/rtx/pathtracing/totalSpp", SPP * SUB)
s.set("/rtx/pathtracing/maxBounces", 10)
s.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 10)
s.set("/rtx/pathtracing/optixDenoiser/enabled", True)

omni.usd.get_context().new_stage()
app.update()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)


def mat(path, mdl="OmniSurface", **kw):
    m = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.SetSourceAsset(Sdf.AssetPath(f"{mdl}.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier(mdl, "mdl")
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


def box(path, size, pos, material):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    x = UsdGeom.Xformable(c)
    x.AddTranslateOp().Set(Gf.Vec3d(*pos))
    x.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(material)
    return c


# ---------------------------------------------------------------- set: quartz worktop, oak board, tiled wall
quartz = mat("/World/Looks/Quartz", diffuse_reflection_color=(0.80, 0.80, 0.78), specular_reflection_roughness=0.18,
             specular_reflection_weight=0.6)
oak = mat("/World/Looks/Oak", diffuse_reflection_color_image=os.path.join(LAB_TEX, "oak_albedo.png"),
          specular_reflection_roughness=0.42, specular_reflection_weight=0.4)
wall = mat("/World/Looks/Tile", diffuse_reflection_color=(0.88, 0.88, 0.86), specular_reflection_roughness=0.12,
           specular_reflection_weight=0.7)
box("/World/Set/Worktop", (1.2, 1.6, 0.04), (0.1, 0.0, -0.02), quartz)
box("/World/Set/Wall", (0.02, 1.6, 0.8), (0.55, 0.0, 0.38), wall)
dome = UsdLux.DomeLight.Define(stage, "/World/Light/Dome")
dome.CreateIntensityAttr(380)
dome.CreateColorAttr(Gf.Vec3f(0.92, 0.95, 1.0))
key = UsdLux.RectLight.Define(stage, "/World/Light/Window")
key.CreateWidthAttr(0.9)
key.CreateHeightAttr(0.7)
key.CreateIntensityAttr(9000)
key.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
kx = UsdGeom.Xformable(key)
kx.AddTranslateOp().Set(Gf.Vec3d(-0.9, -0.5, 0.75))
kx.AddRotateXYZOp().Set(Gf.Vec3f(55, 0, -58))
fill = UsdLux.SphereLight.Define(stage, "/World/Light/Fill")
fill.CreateRadiusAttr(0.15)
fill.CreateIntensityAttr(2500)
UsdGeom.Xformable(fill).AddTranslateOp().Set(Gf.Vec3d(0.7, -0.6, 0.5))

# ---------------------------------------------------------------- the avocado, lying as in the CRAM world
import json  # noqa: E402

rec = json.load(open(os.path.join(INST, "instance.json")))
az = math.radians(rec["cuts"]["half_a"]["plane_azimuth_deg"])
n = np.array([math.sin(az), -math.cos(az), 0.0])
R = np.vstack([n, [0.0, 0.0, 1.0], np.cross(n, [0.0, 0.0, 1.0])])     # asset -> fruit (= world here)
p = UsdGeom.Xform.Define(stage, "/World/Food/Avocado")
p.GetPrim().GetReferences().AddReference(os.path.join(INST, "whole.usdc"))
t = p.AddTranslateOp()
M = Gf.Matrix3d(*R.T.ravel().tolist())
p.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(M.GetTranspose().ExtractRotation().GetQuat()))
bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
r = bbc.ComputeWorldBound(p.GetPrim()).ComputeAlignedRange()
t.Set(Gf.Vec3d(-r.GetMidpoint()[0], -r.GetMidpoint()[1], -r.GetMin()[2] + 0.0002))
for _ in range(10):
    app.update()

# ---------------------------------------------------------------- head camera: 0.55 m in front, 0.40 m above the counter
cam = UsdGeom.Camera.Define(stage, "/World/Cam")
cam.CreateHorizontalApertureAttr(36.0)
cam.CreateVerticalApertureAttr(36.0 * RES[1] / RES[0])
cam.CreateClippingRangeAttr(Gf.Vec2f(0.005, 50))
cam.GetFocalLengthAttr().Set(120.0)
eye, target = Gf.Vec3d(-0.55, 0.0, 0.40), Gf.Vec3d(0.0, 0.0, 0.03)
UsdGeom.Xformable(cam).AddTransformOp().Set(Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse())
rp = rep.create.render_product(str(cam.GetPath()), RES)
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach(rp)
for _ in range(4):
    app.update()
rep.orchestrator.step(rt_subframes=SUB, delta_time=0.0)
rep.orchestrator.step(rt_subframes=1, delta_time=0.0)
os.makedirs(os.path.join(ROOT, "cram", "camera"), exist_ok=True)
out = os.path.join(ROOT, "cram", "camera", os.path.basename(INST) + "_counter.png")
Image.fromarray(np.asarray(rgb.get_data())[..., :3].astype(np.uint8)).save(out)
print("[render]", out, flush=True)
app.close()
