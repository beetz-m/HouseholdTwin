"""Path-traced stills of the opened egg: contents on a white plate, shell fragments beside it, a whole egg.
Run: OMNI_KIT_ACCEPT_EULA=YES ~/Documents/LabAutomationTwin/env_isaacsim/bin/python scripts/render_egg_open.py [--quick]"""
import math
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
QUICK = "--quick" in sys.argv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EGG = os.path.join(ROOT, "assets", "food", "egg_1")
OUT = os.path.join(ROOT, "renders", "food")
LAB_TEX = os.path.expanduser("~/Documents/LabAutomationTwin/assets/materials/textures")
RES = (1280, 800) if QUICK else (1920, 1200)
from isaacsim import SimulationApp  # noqa: E402
app = SimulationApp({"headless": True, "renderer": "PathTracing", "width": RES[0], "height": RES[1]})
import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade  # noqa: E402
sys.path.insert(0, ROOT)
from food import meshlib as ML  # noqa: E402
from food.geometry import Meridian  # noqa: E402

s = carb.settings.get_settings()
s.set("/rtx/rendermode", "PathTracing")
SPP, SUB = (8, 24) if QUICK else (16, 64)
s.set("/rtx/pathtracing/spp", SPP)
s.set("/rtx/pathtracing/totalSpp", SPP * SUB)
s.set("/rtx/pathtracing/maxBounces", 12)
s.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 16)
s.set("/rtx/pathtracing/optixDenoiser/enabled", True)
omni.usd.get_context().new_stage()
app.update()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)


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


def mesh_prim(path, m, material):
    g = UsdGeom.Mesh.Define(stage, path)
    g.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in m.vertices])
    g.CreateFaceVertexCountsAttr([3] * len(m.faces))
    g.CreateFaceVertexIndicesAttr([int(i) for i in np.asarray(m.faces).ravel()])
    g.CreateNormalsAttr([Gf.Vec3f(*map(float, n)) for n in m.vertex_normals])
    g.SetNormalsInterpolation("vertex")
    g.CreateSubdivisionSchemeAttr("none")
    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(material)
    return g


wood = mat("/World/Looks/Oak", diffuse_reflection_color_image=os.path.join(LAB_TEX, "oak_albedo.png"),
           specular_reflection_roughness=0.45)
porcelain = mat("/World/Looks/Porcelain", diffuse_reflection_color=(0.93, 0.93, 0.92), specular_reflection_roughness=0.08,
                specular_reflection_weight=0.9, coat_weight=0.6, coat_roughness=0.03)
wall = mat("/World/Looks/Wall", diffuse_reflection_color=(0.80, 0.82, 0.80), specular_reflection_roughness=0.5)
tab = UsdGeom.Cube.Define(stage, "/World/Table")
tab.CreateSizeAttr(1.0)
UsdGeom.Xformable(tab).AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.015))
UsdGeom.Xformable(tab).AddScaleOp().Set(Gf.Vec3f(1.4, 1.0, 0.03))
UsdShade.MaterialBindingAPI.Apply(tab.GetPrim()).Bind(wood)
w = UsdGeom.Cube.Define(stage, "/World/Wall")
w.CreateSizeAttr(1.0)
UsdGeom.Xformable(w).AddTranslateOp().Set(Gf.Vec3d(0, 0.5, 0.4))
UsdGeom.Xformable(w).AddScaleOp().Set(Gf.Vec3f(1.4, 0.02, 0.8))
UsdShade.MaterialBindingAPI.Apply(w.GetPrim()).Bind(wall)
# plate: shallow porcelain dish (profile revolved), flat well of radius 85 mm, rim to 125 mm
r = np.array([0.0, 0.085, 0.095, 0.115, 0.125, 0.126, 0.116, 0.094, 0.083, 0.0])
z = np.array([0.006, 0.006, 0.008, 0.016, 0.019, 0.017, 0.014, 0.003, 0.0, 0.0])
plate = ML.surface_of_revolution(Meridian(r, z), 160)
mesh_prim("/World/Plate", plate, porcelain)
c = UsdGeom.Xform.Define(stage, "/World/Contents")
c.GetPrim().GetReferences().AddReference(os.path.join(EGG, "opened_contents.usdc"))
c.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0059))   # sits on (0.1 mm into) the plate: no air gap
f = UsdGeom.Xform.Define(stage, "/World/Fragments")
f.GetPrim().GetReferences().AddReference(os.path.join(EGG, "shell_fragments.usdc"))
f.AddTranslateOp().Set(Gf.Vec3d(0.17, -0.02, 0.0))
f.AddRotateZOp().Set(-25.0)
e = UsdGeom.Xform.Define(stage, "/World/WholeEgg")
e.GetPrim().GetReferences().AddReference(os.path.join(EGG, "whole.usdc"))
e.AddTranslateOp().Set(Gf.Vec3d(-0.17, 0.06, 0.022))
e.AddRotateXYZOp().Set(Gf.Vec3f(90, 0, 35))
dome = UsdLux.DomeLight.Define(stage, "/World/Light/Dome")
dome.CreateIntensityAttr(900)
dome.CreateColorAttr(Gf.Vec3f(0.95, 0.96, 1.0))
key = UsdLux.RectLight.Define(stage, "/World/Light/Window")
key.CreateWidthAttr(0.9)
key.CreateHeightAttr(0.7)
key.CreateIntensityAttr(9500)
kx = UsdGeom.Xformable(key)
kx.AddTranslateOp().Set(Gf.Vec3d(-0.7, -0.5, 0.7))
kx.AddRotateXYZOp().Set(Gf.Vec3f(55, 0, -55))
cam = UsdGeom.Camera.Define(stage, "/World/Cam")
cam.CreateHorizontalApertureAttr(36.0)
cam.CreateVerticalApertureAttr(36.0 * RES[1] / RES[0])
cam.CreateClippingRangeAttr(Gf.Vec2f(0.005, 50))
op = UsdGeom.Xformable(cam).AddTransformOp()
rp = rep.create.render_product(str(cam.GetPath()), RES)
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach(rp)
for _ in range(20):
    app.update()


def shot(name, eye, target, focal):
    op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse())
    cam.GetFocalLengthAttr().Set(float(focal))
    for _ in range(4):
        app.update()
    rep.orchestrator.step(rt_subframes=SUB, delta_time=0.0)
    rep.orchestrator.step(rt_subframes=1, delta_time=0.0)
    Image.fromarray(np.asarray(rgb.get_data())[..., :3].astype(np.uint8)).save(os.path.join(OUT, name + ".png"))
    print("[render]", name, flush=True)


shot("egg_opened_scene", (0.0, -0.55, 0.38), (0.0, 0.0, 0.0), 45)
shot("egg_opened_closeup", (0.0, -0.20, 0.17), (0.0, 0.0, 0.01), 70)
shot("egg_shell_fragments", (0.17, -0.20, 0.17), (0.17, -0.02, 0.01), 85)
app.close()
