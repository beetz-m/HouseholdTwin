"""Path-traced stills of the wave-2 and wave-3 food twins on the same kitchen worktop as wave 1: per item the whole
item beside a cut half (cut face up), and one group shot per wave. Generic: items are taken from assets/food.

Run (one GPU job at a time):
  OMNI_KIT_ACCEPT_EULA=YES ~/Documents/LabAutomationTwin/env_isaacsim/bin/python scripts/render_items.py [--quick] [kind ...]
Writes renders/food/item_<kind>.png and renders/food/wave2_group.png, wave3_group.png. Objects rest on the board by
their bounding boxes (not simulated); halves are turned cut face up from the cut plane recorded in instance.json.
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
    state = name.split(".")[0]
    c = json.load(open(os.path.join(d, "instance.json")))["cuts"][state]
    if "normal" not in c:                                     # older builder: vertical plane given by its azimuth
        az = math.radians(c["plane_azimuth_deg"])         # plane spanned by (cos az, sin az, 0) and the axis z;
        n = np.array([math.sin(az), -math.cos(az), 0.0])  # half_a on the +n side (cut face looks along -n)
        return -n if state == "half_a" else n
    n = np.asarray(c["normal"], float)
    return -n if c["side"] == "pos" else n


def cap_up(name, asset, xy, yaw=0.0):
    return place(name, asset, xy, quat_vec(cap_out(asset), (0, 0, 1)), yaw)


def first(kind):
    if os.path.isdir(os.path.join(A, kind)):                  # an instance named directly
        return kind
    return sorted(d for d in os.listdir(A) if d.startswith(kind + "_"))[0]


WAVES = {"wave2": ["loaf", "onion", "orange", "potato", "tomato"],
         "wave3": ["lemon", "cucumber", "croissant", "baguette", "pepper"]}
ONLY = [a for a in sys.argv[1:] if not a.startswith("--")]
INSTANCE = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--instance=")), None)
if INSTANCE:                                                  # one named instance, e.g. a photo-based avocado
    WAVES = {"single": [INSTANCE]}
    ONLY = [INSTANCE]
LYING = quat_cols((0, 1, 0), (0, 0, 1), (1, 0, 0))            # long axis (z) along world x


def extents(asset):
    st_ = Usd.Stage.Open(asset)
    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]).ComputeWorldBound(
        st_.GetPseudoRoot()).ComputeAlignedRange()
    return np.array(r.GetSize())


def lying(asset):
    """items modelled with their long axis along z (banana, carrot, cucumber) are laid on the board."""
    s_ = extents(asset)
    return s_[2] > 1.6 * max(s_[0], s_[1])


def is_long(asset):
    s_ = np.sort(extents(asset))
    return s_[2] > 1.6 * s_[1]


CUT_PREF = {"baguette": "bottom_half", "croissant": "bottom_half", "loaf": "half_a"}   # crumb face up


def cut_state(d, kind=None):
    cuts = json.load(open(os.path.join(d, "instance.json")))["cuts"]
    pref = CUT_PREF.get(kind)
    return pref if pref in cuts else ("half_a" if "half_a" in cuts else sorted(cuts)[0])


placed = []


def pos_size(name):
    bbc.Clear()
    r = bbc.ComputeWorldBound(stage.GetPrimAtPath(f"/World/Food/{name}")).ComputeAlignedRange()
    return np.array(r.GetSize())


def put(name, *a, **k):
    placed.append(name)
    return (cap_up if k.pop("cap", False) else place)(name, *a, **k)


def put_x(name, asset, xy, q=None, yaw=0.0, cap=False):
    """place, and turn by 90 deg about z if the item's long side ends up along y (across the camera axis is x)."""
    put(name, asset, xy, q, yaw=yaw, cap=cap) if not cap else put(name, asset, xy, yaw=yaw, cap=True)
    sz = pos_size(name)
    if sz[1] > 1.2 * sz[0]:
        stage.RemovePrim(f"/World/Food/{name}")
        placed.remove(name)
        put(name, asset, xy, q, yaw=yaw + 90, cap=cap) if not cap else put(name, asset, xy, yaw=yaw + 90, cap=True)


def clear():
    for n in placed:
        stage.RemovePrim(f"/World/Food/{n}")
    placed.clear()
    for _ in range(4):
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
    Image.fromarray(np.asarray(rgb.get_data())[..., :3].astype(np.uint8)).save(os.path.join(OUT, name + ".png"))
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


for wave, kinds in WAVES.items():
    kinds = [k for k in kinds if (not ONLY or k in ONLY) and any(d == k or d.startswith(k + "_") for d in os.listdir(A))]
    for kind in kinds:                                            # per item: whole + cut half
        d = os.path.join(A, first(kind))
        w = os.path.join(d, "whole.usdc")
        long_ = is_long(w)
        put_x("Whole", w, (0.0, 0.07) if long_ else (-0.09, 0.03), LYING if lying(w) else None,
              yaw=-6 if long_ else -10)
        put_x("Half", os.path.join(d, cut_state(d, kind) + ".usdc"), (0.0, -0.08) if long_ else (0.10, -0.02),
              yaw=0 if long_ else 15, cap=True)
        for _ in range(10):
            app.update()
        frame(["Whole", "Half"], 38, -95, 70, 0.85, tag=f"item_{kind}")
        clear()
    if not ONLY and kinds:                                        # group shot: all wholes of the wave in a row
        n, x = len(kinds), 0.0
        for i, kind in enumerate(kinds):                          # in a row, spaced by their measured widths
            w = os.path.join(A, first(kind), "whole.usdc")
            q = LYING if lying(w) else None
            put_x(f"W{i}", w, (0.0, 0.0), q, yaw=-15)
            wx = pos_size(f"W{i}")[0]
            stage.RemovePrim(f"/World/Food/W{i}")
            placed.remove(f"W{i}")
            put_x(f"W{i}", w, (x + wx / 2, 0.0), q, yaw=-15)
            x += wx + 0.03
        for i in range(n):                                        # centre the row
            op_ = UsdGeom.Xformable(stage.GetPrimAtPath(f"/World/Food/W{i}")).GetOrderedXformOps()[0]
            v = op_.Get()
            op_.Set(Gf.Vec3d(v[0] - (x - 0.03) / 2, v[1], v[2]))
        for _ in range(10):
            app.update()
        frame([f"W{i}" for i in range(n)], 32, -92, 50, 0.8, tag=f"{wave}_group")
        clear()
app.close()
