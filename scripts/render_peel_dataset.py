"""Synthetic test set for the avocado peeling monitor: every peeled-half state in <set>/ (food/avocado_peel.py) is
placed cut face down on the worktop and rendered from a fixed oblique top view: RGB, depth (distance to the image
plane), semantic segmentation (UsdSemantics class labels: avocado_cuticle .. = skin remaining, avocado_flesh_damaged =
torn flesh) and the camera. The ideal peel is rendered as the twin's expectation at a perturbed pose (2-4 mm,
2-4 deg, ILLUSTRATIVE registration error) for every sample, so the monitor has to register it itself.

Run (one GPU job at a time):
  OMNI_KIT_ACCEPT_EULA=YES ~/Documents/LabAutomationTwin/env_isaacsim/bin/python scripts/render_peel_dataset.py SET_DIR [--quick]
Writes SET_DIR/render/<sample>.npz.
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
RES = (960, 600) if QUICK else (1280, 800)

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


SET = [a for a in sys.argv[1:] if not a.startswith("--")][0]
OUTR = os.path.join(SET, "render")
os.makedirs(OUTR, exist_ok=True)
SRC = json.load(open(os.path.join(SET, "set.json")))
inst_dir = SRC["instance_dir"]


def cut_down(name, asset, xy, yaw=0.0, dxy=(0.0, 0.0)):
    """the half lying on its cut face (cut-face normal pointing down)."""
    d = os.path.dirname(inst_dir.rstrip("/")) if False else inst_dir
    n = cap_out(os.path.join(inst_dir, SRC["half"] + ".usdc"))
    p = place(name, asset, xy, quat_vec(n, (0, 0, -1)), yaw)
    if dxy != (0.0, 0.0):
        op = UsdGeom.Xformable(p).GetOrderedXformOps()[0]
        v = op.Get()
        op.Set(Gf.Vec3d(v[0] + dxy[0], v[1] + dxy[1], v[2]))
    return p


cam = UsdGeom.Camera.Define(stage, "/World/Cam")
cam.CreateHorizontalApertureAttr(36.0)
cam.CreateVerticalApertureAttr(36.0 * RES[1] / RES[0])
cam.CreateClippingRangeAttr(Gf.Vec2f(0.005, 50))
cam.GetFocalLengthAttr().Set(50.0)
cop = UsdGeom.Xformable(cam).AddTransformOp()
eye, target = np.array([0.0, -0.17, 0.30]), np.array([0.0, 0.0, 0.02])
Mcam = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
cop.Set(Mcam)
rp = rep.create.render_product(str(cam.GetPath()), RES)
ann = {k: rep.AnnotatorRegistry.get_annotator(k, **kw) for k, kw in
       (("rgb", {}), ("distance_to_image_plane", {}), ("semantic_segmentation", {"init_params": {"colorize": False}}))}
for a_ in ann.values():
    a_.attach(rp)


def grab():
    for _ in range(4):
        app.update()
    rep.orchestrator.step(rt_subframes=SUB, delta_time=0.0)
    rep.orchestrator.step(rt_subframes=1, delta_time=0.0)
    seg = ann["semantic_segmentation"].get_data()
    return (np.asarray(ann["rgb"].get_data())[..., :3].astype(np.uint8),
            np.asarray(ann["distance_to_image_plane"].get_data(), np.float32),
            np.asarray(seg["data"]), seg["info"]["idToLabels"])


intr = dict(width=RES[0], height=RES[1], focal_mm=50.0, h_aperture_mm=36.0, eye=eye.tolist(), target=target.tolist(),
            cam_to_world=np.array(Mcam).T.tolist())
rng = np.random.default_rng(11)
for rec in SRC["samples"]:
    name = os.path.splitext(rec["usd"])[0]
    if name == "ideal":
        continue
    cut_down("Half", os.path.join(SET, rec["usd"]), (0.0, 0.0), yaw=20.0)
    rgb, depth, seg, labels = grab()
    stage.RemovePrim("/World/Food/Half")
    th = rng.uniform(0, 2 * np.pi)
    err_t = rng.uniform(2e-3, 4e-3) * np.array([np.cos(th), np.sin(th)])
    err_yaw = rng.uniform(2, 4) * rng.choice([-1, 1])
    cut_down("Expect", os.path.join(SET, "ideal.usda"), (0.0, 0.0), yaw=20.0 + err_yaw, dxy=tuple(err_t))
    _, depth_exp, seg_exp, labels_exp = grab()
    stage.RemovePrim("/World/Food/Expect")
    np.savez_compressed(os.path.join(OUTR, name + ".npz"), rgb=rgb, depth=depth, seg=seg,
                        labels=json.dumps(labels), depth_expected=depth_exp, seg_expected=seg_exp,
                        labels_expected=json.dumps(labels_exp), intrinsics=json.dumps(intr),
                        registration_error=json.dumps(dict(dxy_m=err_t.tolist(), dyaw_deg=float(err_yaw))))
    print("[render]", name, flush=True)
app.close()
