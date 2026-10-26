"""Path-traced stills of an avocado halving / pitting run of the emulator (food/avocado_emulator.py): the scored fruit,
the two halves cut face up, and the pitted half with the seed and the other half.

Run (one GPU job at a time):
  OMNI_KIT_ACCEPT_EULA=YES ~/Documents/LabAutomationTwin/env_isaacsim/bin/python scripts/render_process_states.py RUN_DIR [--quick] [--step=scored|halves|pitted]
Writes RUN_DIR/render_<step>.png. On an 8 GB GPU shared with another Isaac session, render one step per process
(--step): the renderer does not return the memory of a removed scene, and the third scene runs out of it.
"""

import math
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
QUICK = "--quick" in sys.argv
STEPS = {a.split("=", 1)[1] for a in sys.argv if a.startswith("--step=")} or {"scored", "halves", "pitted"}
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


RUN = [a for a in sys.argv[1:] if not a.startswith("--")][0]
P = json.load(open(os.path.join(RUN, "process.json")))
az = math.radians(P["plane"]["azimuth_deg"])
N_CUT = np.array([math.sin(az), -math.cos(az), 0.0])          # half_a lies on the +n side: its cut face looks along -n


def face_up(name, path, xy, half, yaw=0.0):
    n = -N_CUT if half == "half_a" else N_CUT
    p = place(name, path, xy, quat_vec(n, (0, 0, 1)), yaw)
    placed.append(name)
    return p


placed = []


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
    Image.fromarray(np.asarray(rgb.get_data())[..., :3].astype(np.uint8)).save(os.path.join(RUN, name + ".png"))
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


def f(n):
    """the state file, binary (.usdc) or text (.usda)."""
    base = os.path.join(RUN, os.path.splitext(n)[0])
    return base + ".usdc" if os.path.exists(base + ".usdc") else base + ".usda"
OUT = RUN
if "scored" in STEPS and os.path.exists(f("scored.usda")):
    B = np.c_[N_CUT, [0.0, 0.0, 1.0], np.cross(N_CUT, [0.0, 0.0, 1.0])]   # object: cut normal, long axis, third
    Wd = np.c_[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]       # world: normal -> y, axis -> x
    Mw = Wd @ B.T
    place("Scored", f("scored.usda"), (0.0, 0.0), quat_cols(Mw[:, 0], Mw[:, 1], Mw[:, 2]), yaw=-10)
    placed.append("Scored")
    for _ in range(10):
        app.update()
    frame(["Scored"], 55, -95, 85, 0.8, tag="render_scored")
    clear()
if "halves" in STEPS and os.path.exists(f("half_a.usda")) and os.path.exists(f("half_b.usda")):
    face_up("HalfA", f("half_a.usda"), (-0.07, 0.0), "half_a", yaw=0)
    face_up("HalfB", f("half_b.usda"), (0.07, 0.0), "half_b", yaw=180)
    for _ in range(10):
        app.update()
    frame(["HalfA", "HalfB"], 50, -90, 70, 0.8, tag="render_halves")
    clear()
if "pitted" in STEPS and os.path.exists(f("half_a_pitted.usda")):
    face_up("Pitted", f("half_a_pitted.usda"), (-0.08, 0.0), "half_a", yaw=0)
    face_up("HalfB", f("half_b.usda"), (0.07, 0.02), "half_b", yaw=180)
    place("Seed", f("seed.usda"), (0.0, -0.07), None, 30)
    placed.append("Seed")
    for _ in range(10):
        app.update()
    frame(["Pitted", "HalfB", "Seed"], 50, -90, 65, 0.8, tag="render_pitted")
    frame(["Pitted"], 70, -90, 90, 0.75, tag="render_cavity")
    clear()
app.close()
