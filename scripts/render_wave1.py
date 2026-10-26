"""Path-traced stills of the wave-1 food twins (bread roll, banana, carrot) on the kitchen worktop.

Run (one GPU job at a time):
  OMNI_KIT_ACCEPT_EULA=YES ~/Documents/LabAutomationTwin/env_isaacsim/bin/python scripts/render_wave1.py [--quick]
Writes renders/food/wave1_*.png.  Objects rest on the board by their bounding boxes (not simulated); halves are
turned cut face up from the cut plane recorded in each instance.json.
"""

import math
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
QUICK = "--quick" in sys.argv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, "assets", "food")
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
box("/World/Set/Worktop", (1.6, 0.9, 0.04), (0, 0.05, -0.02 - 0.022), quartz)
box("/World/Set/Board", (0.62, 0.38, 0.022), (0, 0, -0.011), oak)
box("/World/Set/Wall", (1.6, 0.02, 0.8), (0, 0.48, 0.38), wall)
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

# ---------------------------------------------------------------- objects
import json  # noqa: E402

bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])


def quat_cols(cx, cy, cz):
    """Orientation whose rotation maps object x, y, z onto the given world directions."""
    M = Gf.Matrix3d(*np.array([cx, cy, cz], float).T.ravel().tolist())
    return Gf.Quatf(M.ExtractRotation().GetQuat())


def quat_vec(a, b):
    return Gf.Quatf(Gf.Rotation(Gf.Vec3d(*map(float, a)), Gf.Vec3d(*map(float, b))).GetQuat())


def place(name, asset, xy, q=None, yaw=0.0):
    p = UsdGeom.Xform.Define(stage, f"/World/Food/{name}")
    p.GetPrim().GetReferences().AddReference(asset)
    t = p.AddTranslateOp()
    p.AddRotateZOp().Set(yaw)
    if q is not None:
        p.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(q)
    t.Set(Gf.Vec3d(xy[0], xy[1], 0))
    bbc.Clear()
    r = bbc.ComputeWorldBound(p.GetPrim()).ComputeAlignedRange()
    lo, mid = r.GetMin(), r.GetMidpoint()
    t.Set(Gf.Vec3d(2 * xy[0] - mid[0], 2 * xy[1] - mid[1], -lo[2] + 0.0002))
    return p


def cap_out(asset):
    d, name = os.path.split(asset)
    c = json.load(open(os.path.join(d, "instance.json")))["cuts"][name.split(".")[0]]
    n = np.asarray(c["normal"], float)
    return -n if c["side"] == "pos" else n


def cap_up(name, asset, xy, yaw=0.0):
    return place(name, asset, xy, quat_vec(cap_out(asset), (0, 0, 1)), yaw)


def first(kind):
    return sorted(d for d in os.listdir(A) if d.startswith(kind + "_"))[0]


ROLL, BAN, CAR = (os.path.join(A, first(k)) for k in ("roll", "banana", "carrot"))
LYING = quat_cols((0, 1, 0), (0, 0, 1), (1, 0, 0))            # bent long axis (z) along world x, bend plane flat
place("Roll", os.path.join(ROLL, "whole.usdc"), (-0.20, 0.08), yaw=15)
cap_up("RollBottom", os.path.join(ROLL, "bottom_half.usdc"), (-0.215, -0.075), yaw=-10)
cap_up("RollTop", os.path.join(ROLL, "top_half.usdc"), (-0.095, -0.06), yaw=35)
place("Banana", os.path.join(BAN, "whole.usdc"), (0.02, 0.10), LYING, yaw=-8)
cap_up("BananaHalf", os.path.join(BAN, "half_a.usdc"), (0.03, -0.035), yaw=-5)
place("Carrot", os.path.join(CAR, "whole.usdc"), (0.20, 0.06), LYING, yaw=-80)
cap_up("CarrotHalf", os.path.join(CAR, "half_b.usdc"), (0.255, 0.0), yaw=-95)
place("CarrotCrown", os.path.join(CAR, "crown_piece.usdc"), (0.12, -0.11), LYING, yaw=20)
for _ in range(20):
    app.update()
# ---------------------------------------------------------------- camera
cam = UsdGeom.Camera.Define(stage, "/World/Cam")
cam.CreateHorizontalApertureAttr(36.0)
cam.CreateVerticalApertureAttr(36.0 * RES[1] / RES[0])
cam.CreateClippingRangeAttr(Gf.Vec2f(0.005, 50))
op = UsdGeom.Xformable(cam).AddTransformOp()
rp = rep.create.render_product(str(cam.GetPath()), RES)
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach(rp)
os.makedirs(OUT, exist_ok=True)


def shot(name, eye, target, focal):
    op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse())
    cam.GetFocalLengthAttr().Set(float(focal))
    for _ in range(4):                                            # let the camera move reach the renderer
        app.update()
    rep.orchestrator.step(rt_subframes=SUB, delta_time=0.0)
    rep.orchestrator.step(rt_subframes=1, delta_time=0.0)       # annotator returns the previous step's frame
    Image.fromarray(np.asarray(rgb.get_data())[..., :3].astype(np.uint8)).save(os.path.join(OUT, "wave1_" + name + ".png"))
    print("[render]", name, flush=True)


def pos(name):
    bbc.Clear()
    r = bbc.ComputeWorldBound(stage.GetPrimAtPath(f"/World/Food/{name}")).ComputeAlignedRange()
    return np.array(r.GetMidpoint())


def frame(names, elev_deg=40, az_deg=-90, focal=60, margin=1.35, tag=None):
    bbc.Clear()
    rng = Gf.Range3d()
    for n in names:
        rng.UnionWith(bbc.ComputeWorldBound(stage.GetPrimAtPath(f"/World/Food/{n}")).ComputeAlignedRange())
    c = np.array(rng.GetMidpoint())
    r = np.linalg.norm(np.array(rng.GetSize())) / 2
    half_fov = math.atan(36.0 * RES[1] / RES[0] / 2 / focal)
    d = margin * r / math.tan(half_fov)
    el, az = math.radians(elev_deg), math.radians(az_deg)
    eye = c + d * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    shot(tag, eye, c, focal)


shot("hero", (0.02, -0.62, 0.40), (0.01, 0.0, 0.02), 45)
shot("top", (0.0, -0.02, 0.75), (0.0, 0.0, 0.0), 40)
frame(["Roll", "RollBottom", "RollTop"], 35, -100, 70, 0.8, tag="roll")
frame(["RollBottom", "RollTop"], 55, -90, 85, 0.75, tag="roll_sliced")
frame(["Banana"], 25, -100, 80, 0.75, tag="banana")
frame(["BananaHalf"], 60, -90, 70, 0.7, tag="banana_half")
frame(["Carrot", "CarrotHalf"], 40, -60, 60, 0.75, tag="carrot")
frame(["CarrotCrown"], 25, -150, 100, 0.8, tag="carrot_crown_cut")
app.close()
