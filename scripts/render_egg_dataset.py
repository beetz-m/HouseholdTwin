"""Synthetic images for the egg perception layer (L4), path traced in Isaac Sim:

  --bowl SEED   every sample of assets/food/egg_<seed>/bowl_set/ (scripts/build_egg_bowl_set.py): the opened contents
                on a white plate, seen from above (slightly oblique): RGB, depth (distance to the image plane),
                semantic segmentation (egg_yolk, egg_yolk_broken, egg_shell_shard, albumen classes) and the camera
                -> bowl_set/render/<sample>.npz (for scripts/eval_egg_bowl_monitor.py)
  --photos      the built egg instances (assets/food/egg_0..7/whole.usdc) lying on their side on an oak board, side
                view, RGB -> renders/egg/photos/egg_<k>.png + truth.json (for the photo -> node test)

Run (one GPU job at a time, through the shared lock):
  flock <gpu.lock> env OMNI_KIT_ACCEPT_EULA=YES ~/Documents/LabAutomationTwin/env_isaacsim/bin/python \
      scripts/render_egg_dataset.py --bowl 3 [--photos] [--quick]
"""

import json
import math
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
QUICK = "--quick" in sys.argv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, "assets", "food")
LAB_TEX = os.path.expanduser("~/Documents/LabAutomationTwin/assets/materials/textures")
RES = (960, 600) if QUICK else (1280, 800)
BOWL = int(sys.argv[sys.argv.index("--bowl") + 1]) if "--bowl" in sys.argv else None
PHOTOS = "--photos" in sys.argv

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


quartz = mat("/World/Looks/Quartz", diffuse_reflection_color=(0.80, 0.80, 0.78), specular_reflection_roughness=0.18,
             specular_reflection_weight=0.6)
oak = mat("/World/Looks/Oak", diffuse_reflection_color_image=os.path.join(LAB_TEX, "oak_albedo.png"),
          specular_reflection_roughness=0.42, specular_reflection_weight=0.4)
wall = mat("/World/Looks/Tile", diffuse_reflection_color=(0.88, 0.88, 0.86), specular_reflection_roughness=0.12,
           specular_reflection_weight=0.7)
porcelain = mat("/World/Looks/Plate", diffuse_reflection_color=(0.95, 0.95, 0.94), specular_reflection_roughness=0.08,
                specular_reflection_weight=0.8)
box("/World/Set/Worktop", (1.6, 0.9, 0.04), (0, 0.05, -0.02 - 0.022), quartz)
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

cam = UsdGeom.Camera.Define(stage, "/World/Cam")
cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 50.0))         # default near clip 1 m hides close-ups
cam.GetHorizontalApertureAttr().Set(36.0)
cam.GetVerticalApertureAttr().Set(36.0 * RES[1] / RES[0])
cam.GetFocalLengthAttr().Set(50.0)
cop = UsdGeom.Xformable(cam).AddTransformOp()
rp = rep.create.render_product(str(cam.GetPath()), RES)
ann = {k: rep.AnnotatorRegistry.get_annotator(k, **kw) for k, kw in
       (("rgb", {}), ("distance_to_image_plane", {}), ("semantic_segmentation", {"init_params": {"colorize": False}}))}
for a_ in ann.values():
    a_.attach(rp)


def look(eye, target):
    M = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
    cop.Set(M)
    return M


def grab():
    for _ in range(4):
        app.update()
    rep.orchestrator.step(rt_subframes=SUB, delta_time=0.0)
    rep.orchestrator.step(rt_subframes=1, delta_time=0.0)
    seg = ann["semantic_segmentation"].get_data()
    return (np.asarray(ann["rgb"].get_data())[..., :3].astype(np.uint8),
            np.asarray(ann["distance_to_image_plane"].get_data(), np.float32),
            np.asarray(seg["data"]), seg["info"]["idToLabels"])


def place(path, asset, translate, rot_y=0.0, yaw=0.0):
    x = UsdGeom.Xform.Define(stage, path)
    x.GetPrim().GetReferences().AddReference(asset)
    xf = UsdGeom.Xformable(x)
    xf.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xf.AddRotateZOp().Set(float(yaw))
    xf.AddRotateYOp().Set(float(rot_y))
    return x


if BOWL is not None:
    base = os.path.join(A, f"egg_{BOWL}", "bowl_set")
    S = json.load(open(os.path.join(base, "set.json")))
    outd = os.path.join(base, "render")
    os.makedirs(outd, exist_ok=True)
    plate = UsdGeom.Cylinder.Define(stage, "/World/Set/Plate")
    plate.CreateRadiusAttr(0.12)
    plate.CreateHeightAttr(0.012)
    plate.CreateAxisAttr("Z")
    UsdGeom.Xformable(plate).AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.006))
    UsdShade.MaterialBindingAPI.Apply(plate.GetPrim()).Bind(porcelain)
    eye, target = np.array([0.0, -0.06, 0.30]), np.array([0.0, 0.0, 0.0])
    M = look(eye, target)
    intr = dict(width=RES[0], height=RES[1], focal_mm=50.0, h_aperture_mm=36.0, eye=eye.tolist(),
                target=target.tolist(), cam_to_world=np.array(M).T.tolist())
    for rec in S["samples"]:
        place("/World/Food/Contents", os.path.join(base, rec["usd"]), (0, 0, 0), yaw=25.0)
        rgb, depth, seg, labels = grab()
        stage.RemovePrim("/World/Food/Contents")
        np.savez_compressed(os.path.join(outd, rec["sample"] + ".npz"), rgb=rgb, depth=depth, seg=seg,
                            labels=json.dumps(labels), intrinsics=json.dumps(intr), truth=json.dumps(rec))
        Image.fromarray(rgb).save(os.path.join(outd, rec["sample"] + ".png"))
        print("[render] bowl", rec["sample"], flush=True)
    stage.RemovePrim("/World/Set/Plate")

if PHOTOS:
    sys.path.insert(0, ROOT)
    from food import params as PR  # noqa: E402
    outd = os.path.join(ROOT, "renders", "egg", "photos")
    os.makedirs(outd, exist_ok=True)
    box("/World/Set/Board", (0.62, 0.38, 0.022), (0, 0, -0.011), oak)
    look((0.0, -0.32, 0.05), (0.0, 0.0, 0.022))
    truth = {}
    for k in range(8):
        asset = os.path.join(A, f"egg_{k}", "whole.usdc")
        if not os.path.exists(asset):
            continue
        p = PR.sample("egg", k)
        B = p["shape_index_B_over_L"] * p["length_L"] * 1e-3
        place("/World/Food/Egg", asset, (0, 0, B / 2), rot_y=90.0 if k % 2 == 0 else -90.0, yaw=0.0)
        rgb, _, _, _ = grab()
        stage.RemovePrim("/World/Food/Egg")
        Image.fromarray(rgb).save(os.path.join(outd, f"egg_{k}.png"))
        truth[f"egg_{k}"] = dict(length_mm=p["length_L"], shape_index=p["shape_index_B_over_L"],
                                 asymmetry_w_mm=p["asymmetry_w"], blunt_end=("+x" if k % 2 == 0 else "-x"))
        print("[render] photo", k, flush=True)
    json.dump(truth, open(os.path.join(outd, "truth.json"), "w"), indent=1)
app.close()
