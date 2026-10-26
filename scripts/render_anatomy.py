"""Exploded anatomical plates of the food twins: the cut state's part prims are displaced in nesting order (outside ->
inside) by USD overs on the referenced asset, path-traced, then labelled with prim path, class and mass.

Run (alone on the GPU): OMNI_KIT_ACCEPT_EULA=YES ~/Documents/LabAutomationTwin/env_isaacsim/bin/python \
    scripts/render_anatomy.py [--quick]
Writes renders/food/anatomy_<kind>.png (raw) and anatomy_<kind>_labelled.png.
"""

import json
import math
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
QUICK = "--quick" in sys.argv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, "assets", "food")
OUT = os.path.join(ROOT, "renders", "food")
RES = (1600, 800) if QUICK else (2400, 1200)

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "PathTracing", "width": RES[0], "height": RES[1]})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade  # noqa: E402

s = carb.settings.get_settings()
s.set("/rtx/rendermode", "PathTracing")
SPP, SUB = (8, 16) if QUICK else (16, 48)
s.set("/rtx/pathtracing/spp", SPP)
s.set("/rtx/pathtracing/totalSpp", SPP * SUB)
s.set("/rtx/pathtracing/maxBounces", 10)
s.set("/rtx/pathtracing/optixDenoiser/enabled", True)

# groups of part paths (relative to /Food/Parts), outside -> inside; label shown on the plate
PLATES = {
    "egg": ("cutaway", [
        ("Shell: cuticle, palisade, mammillary", ["Shell/Cuticle", "Shell/PalisadeLayer", "Shell/MammillaryLayer"]),
        ("Shell membranes (outer, inner)", ["ShellMembranes/OuterShellMembrane", "ShellMembranes/InnerShellMembrane"]),
        ("Outer thin albumen (23 %)", ["Albumen/OuterThinAlbumen"]),
        ("Thick albumen (57 %)", ["Albumen/ThickAlbumen"]),
        ("Inner thin albumen (17 %)", ["Albumen/InnerThinAlbumen"]),
        ("Chalaziferous layer (3 %) + chalazae", ["Albumen/ChalaziferousLayer", "Albumen/Chalazae/BluntEnd",
                                                  "Albumen/Chalazae/PointedEnd"]),
        ("Yolk: vitelline membrane, yolk body", ["Yolk/VitellineMembrane", "Yolk/YolkBody"]),
        ("Latebra, neck, nucleus of Pander, germinal disc", ["Yolk/Latebra", "Yolk/LatebraNeck",
                                                             "Yolk/NucleusOfPander", "Yolk/GerminalDisc"]),
    ]),
    "apple": ("half_a", [
        ("Skin: cuticle, epidermis, hypodermis", ["Skin/Cuticle", "Skin/Epidermis", "Skin/Hypodermis"]),
        ("Cortex (hypanthium flesh)", ["Cortex"]),
        ("Vascular system: 5 sepal + 5 petal bundles", ["VascularSystem"]),
        ("Core: pith (carpellary tissue)", ["Core/Pith"]),
        ("Carpels: endocarp, locules, seeds", ["Core/Carpel_1", "Core/Carpel_2", "Core/Carpel_3", "Core/Carpel_4",
                                               "Core/Carpel_5"]),
        ("Calyx: sepals, calyx tube, stamens; stem", ["Calyx", "Stem"]),
    ]),
    "avocado": ("half_a", [
        ("Exocarp: cuticle .. stone-cell layer", ["Exocarp/Cuticle", "Exocarp/Epidermis", "Exocarp/Hypodermis",
                                                 "Exocarp/SkinParenchyma", "Exocarp/StoneCellLayer", "StemButton"]),
        ("Mesocarp: outer dark-green band", ["Mesocarp/OuterDarkGreenBand"]),
        ("Mesocarp: middle pale-green band", ["Mesocarp/MiddlePaleGreenBand"]),
        ("Mesocarp: inner yellow band + 6 strands", ["Mesocarp/InnerYellowBand", "Mesocarp/VascularStrands"]),
        ("Endocarp", ["Endocarp"]),
        ("Seed coat: outer + inner testa", ["Seed/SeedCoat"]),
        ("Cotyledons + embryonic axis", ["Seed/Cotyledon_1", "Seed/Cotyledon_2", "Seed/EmbryonicAxis"]),
    ]),
    "roll": ("half_a", [
        ("Upper crust", ["Crust/UpperCrust"]),
        ("Score crust (Ausbund)", ["Crust/ScoreCrust"]),
        ("Bottom crust", ["Crust/BottomCrust"]),
        ("Crust-crumb transition zone", ["Crumb/TransitionZone"]),
        ("Crumb core (gas-cell foam)", ["Crumb/CrumbCore"]),
    ]),
    "banana": ("half_a", [
        ("Peel: epidermis + chlorenchyma", ["Peel/Epidermis", "Peel/Chlorenchyma"]),
        ("Peel parenchyma (latex vessels)", ["Peel/Parenchyma"]),
        ("Inner aerenchyma", ["Peel/InnerAerenchyma"]),
        ("Stalk, crown cushion, blossom tip", ["Stalk", "CrownCushion", "BlossomTip"]),
        ("Pulp: 3 carpel segments", ["Pulp/Segment_1", "Pulp/Segment_2", "Pulp/Segment_3"]),
        ("Central axis + phloem strings", ["Pulp/CentralAxis", "Pulp/PhloemStrings"]),
    ]),
    "carrot": ("half_a", [
        ("Periderm (skin)", ["Periderm"]),
        ("Secondary phloem ('cortex')", ["SecondaryPhloem"]),
        ("Vascular cambium", ["VascularCambium"]),
        ("Secondary + primary xylem ('core')", ["SecondaryXylem", "PrimaryXylem"]),
        ("Crown: petiole stubs", ["Crown"]),
    ]),
    "onion": ("half_a", [
        ("Dry scales (tunics) + neck", ["DryScales", "Neck"]),
        ("Fleshy scales 1-2", ["FleshyScales/Scale_01", "FleshyScales/Scale_02"]),
        ("Fleshy scales 3-4", ["FleshyScales/Scale_03", "FleshyScales/Scale_04"]),
        ("Fleshy scales 5-8", ["FleshyScales/Scale_05", "FleshyScales/Scale_06", "FleshyScales/Scale_07",
                               "FleshyScales/Scale_08"]),
        ("Shoot apex", ["ShootApex"]),
        ("Basal plate + roots", ["BasalPlate", "Roots"]),
    ]),
    "orange": ("half_a", [
        ("Flavedo + stem button", ["Peel/Flavedo", "StemButton"]),
        ("Albedo", ["Peel/Albedo"]),
        ("Segment membranes (endocarp)", [f"Segments/Segment_{k:02d}/Membrane" for k in range(1, 13)]),
        ("Juice vesicles", [f"Segments/Segment_{k:02d}/Vesicles" for k in range(1, 13)]),
        ("Core + seeds", ["Core"] + [f"Segments/Segment_{k:02d}/Seed_{j}" for k in range(1, 13) for j in range(1, 8)]),
    ]),
    "tomato": ("half_a", [
        ("Skin", ["Skin"]),
        ("Outer pericarp wall", ["Pericarp/OuterWall"]),
        ("Septa + columella", ["Pericarp/Septa", "Pericarp/Columella"]),
        ("Placenta", [f"Locules/Locule_{k}/Placenta" for k in range(1, 6)]),
        ("Locular gel", [f"Locules/Locule_{k}/Gel" for k in range(1, 6)]),
        ("Seeds", [f"Locules/Locule_{k}/Seeds" for k in range(1, 6)]),
        ("Calyx, stem scar, pedicel", ["Calyx", "StemScar", "Pedicel"]),
    ]),
    "potato": ("half_a", [
        ("Periderm, eyes, stolon scar", ["Periderm", "Eyes", "StolonScar"]),
        ("Cortex", ["Cortex"]),
        ("Vascular ring", ["VascularRing"]),
        ("Perimedullary zone", ["PerimedullaryZone"]),
        ("Pith (medulla)", ["Pith"]),
    ]),
    "loaf": ("half_a", [
        ("Top crust (lidded tin)", ["Crust/TopCrust"]),
        ("Tin crust", ["Crust/TinCrust"]),
        ("Crumb", ["Crumb"]),
    ]),
    # wave 3 (20 Oct 2026)
    "lemon": ("half_a", [
        ("Flavedo, oil glands, stem button", ["Peel/Flavedo", "Peel/OilGlands", "StemButton"]),
        ("Albedo", ["Peel/Albedo"]),
        ("Segment membranes", [f"Segments/Segment_{k:02d}/Membrane" for k in range(1, 13)]),
        ("Juice vesicles", [f"Segments/Segment_{k:02d}/Vesicles" for k in range(1, 13)]),
        ("Core + seeds", ["Core"] + [f"Segments/Segment_{k:02d}/Seed_{j}" for k in range(1, 13) for j in range(1, 10)]),
    ]),
    "cucumber": ("half_a", [
        ("Skin (exocarp)", ["Skin"]),
        ("Flesh (mesocarp)", ["Flesh"]),
        ("Placentae", ["SeedCavity/Placenta_1", "SeedCavity/Placenta_2", "SeedCavity/Placenta_3"]),
        ("Core", ["SeedCavity/Core"]),
        ("Seeds", ["SeedCavity/Seeds"]),
    ]),
    "croissant": ("half_a", [
        ("Crust", ["Crust"]),
        ("Laminated crumb", ["LaminatedCrumb"]),
        ("Flakes", ["Flakes"]),
    ]),
    "baguette": ("half_a", [
        ("Upper + score crust", ["Crust/UpperCrust", "Crust/ScoreCrust"]),
        ("Bottom crust", ["Crust/BottomCrust"]),
        ("Crumb transition zone", ["Crumb/TransitionZone"]),
        ("Crumb core + gas cells", ["Crumb/CrumbCore", "Crumb/GasCells"]),
        ("Dusting", ["Dusting"]),
    ]),
    "pepper": ("half_a", [
        ("Skin", ["Pericarp/Skin"]),
        ("Wall (pericarp)", ["Pericarp/Wall"]),
        ("Ribs", ["Ribs/Rib_1", "Ribs/Rib_2", "Ribs/Rib_3"]),
        ("Placenta + seeds", ["Placenta", "Seeds"]),
        ("Stem (peduncle + calyx)", ["Stem/Peduncle", "Stem/Calyx"]),
    ]),
}
ONLY = [a for a in sys.argv[1:] if not a.startswith("--")]
if ONLY:
    PLATES = {k: v for k, v in PLATES.items() if k in ONLY}

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
        sh.CreateInput(k, Sdf.ValueTypeNames.Color3f if isinstance(v, tuple) else Sdf.ValueTypeNames.Float).Set(
            Gf.Vec3f(*v) if isinstance(v, tuple) else float(v))
    return m


backdrop = mat("/World/Looks/Backdrop", diffuse_reflection_color=(0.86, 0.86, 0.85), specular_reflection_roughness=0.6)
floor = UsdGeom.Cube.Define(stage, "/World/Floor")
floor.CreateSizeAttr(1.0)
UsdGeom.Xformable(floor).AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.06))
UsdGeom.Xformable(floor).AddScaleOp().Set(Gf.Vec3f(3, 3, 0.01))
UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(backdrop)
wall = UsdGeom.Cube.Define(stage, "/World/Wall")
wall.CreateSizeAttr(1.0)
UsdGeom.Xformable(wall).AddTranslateOp().Set(Gf.Vec3d(0, 0.35, 0.4))
UsdGeom.Xformable(wall).AddScaleOp().Set(Gf.Vec3f(3, 0.01, 1.0))
UsdShade.MaterialBindingAPI.Apply(wall.GetPrim()).Bind(backdrop)
dome = UsdLux.DomeLight.Define(stage, "/World/Light/Dome")
dome.CreateIntensityAttr(500)
key = UsdLux.RectLight.Define(stage, "/World/Light/Key")
key.CreateWidthAttr(1.2)
key.CreateHeightAttr(0.8)
key.CreateIntensityAttr(7000)
kx = UsdGeom.Xformable(key)
kx.AddTranslateOp().Set(Gf.Vec3d(-0.4, -0.9, 0.6))
kx.AddRotateXYZOp().Set(Gf.Vec3f(60, 0, -20))
cam = UsdGeom.Camera.Define(stage, "/World/Cam")
cam.CreateHorizontalApertureAttr(36.0)
cam.CreateVerticalApertureAttr(36.0 * RES[1] / RES[0])
cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50))
cop = UsdGeom.Xformable(cam).AddTransformOp()
rp = rep.create.render_product(str(cam.GetPath()), RES)
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach(rp)
os.makedirs(OUT, exist_ok=True)
bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])


def first(kind):
    return sorted(d for d in os.listdir(A) if d.startswith(kind + "_"))[0]


def project(p, eye, target, focal):
    fwd = np.asarray(target) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    v = np.asarray(p) - eye
    x, y, z = v @ right, v @ up, v @ fwd
    W, H = RES
    fx = focal / 36.0 * W
    return W / 2 + fx * x / z, H / 2 - fx * y / z


for kind, (state, groups) in PLATES.items():
    d = os.path.join(A, first(kind))
    rec = json.load(open(os.path.join(d, "instance.json")))
    if "normal" in rec["cuts"][state]:                    # spec-built assets record the plane normal
        n = np.asarray(rec["cuts"][state]["normal"], float)
        if rec["cuts"][state]["side"] == "neg":
            n = -n
        state_side = "half_a"
    else:
        th = math.radians(rec["cuts"][state]["plane_azimuth_deg"])
        n = np.array([math.sin(th), -math.cos(th), 0.0])
        state_side = state
    cap_n = -n if state_side == "half_a" else n
    root = f"/World/{kind}"
    if stage.GetPrimAtPath(root):
        stage.RemovePrim(root)
    x = UsdGeom.Xform.Define(stage, root)
    x.GetPrim().GetReferences().AddReference(os.path.join(d, f"{state}.usdc"))
    # orient: cut face towards -y (camera), long axis up; explode along +x
    az = math.degrees(math.atan2(cap_n[1], cap_n[0]))
    x.AddRotateZOp().Set(-90.0 - az)
    for _ in range(3):
        app.update()
    gap = {"egg": 0.072, "apple": 0.10, "avocado": 0.095, "roll": 0.115, "banana": 0.085, "carrot": 0.07,
           "onion": 0.085, "orange": 0.095, "tomato": 0.075, "potato": 0.075, "loaf": 0.28,
           "lemon": 0.085, "cucumber": 0.07, "croissant": 0.12, "baguette": 0.10, "pepper": 0.11}[kind]
    labels = []
    for gi, (lab, paths) in enumerate(groups):
        dx = gap * (gi - (len(groups) - 1) / 2)
        mass = 0.0
        for p in paths:
            prim = stage.GetPrimAtPath(f"{root}/Parts/{p}")
            if not prim:
                continue
            xf = UsdGeom.Xformable(prim)
            # translate in the referenced frame: world +x mapped back through the root rotation
            ang = math.radians(-(-90.0 - az))
            xf.AddTranslateOp(opSuffix="explode").Set(Gf.Vec3d(dx * math.cos(ang), dx * math.sin(ang), 0))
            for q in Usd.PrimRange(prim):
                a = q.GetAttribute("food:mass_g")
                if a and a.Get() is not None:
                    mass += a.Get()
        labels.append((lab, paths, dx, mass))
    for _ in range(5):
        app.update()
    bbc.Clear()
    rng_ = bbc.ComputeWorldBound(x.GetPrim()).ComputeAlignedRange()
    lo, hi = np.array(rng_.GetMin()), np.array(rng_.GetMax())
    zc = (lo[2] + hi[2]) / 2
    xf_ = UsdGeom.Xformable(x)
    tz = xf_.AddTranslateOp(opSuffix="lift")
    tz.Set(Gf.Vec3d(0, 0, -lo[2] - 0.055))
    zc += -lo[2] - 0.055
    width = (hi[0] - lo[0]) + 0.8 * (hi[1] - lo[1])     # depth term: parts reaching towards the camera look larger
    focal = 70.0
    height = hi[2] - lo[2]
    dist = max(width * 1.18 / (36.0 / focal), height * 1.6 / (36.0 * RES[1] / RES[0] / focal))
    dist *= {"croissant": 1.35}.get(kind, 1.0)          # its crust shell curves towards the camera
    eye = np.array([0.0, -dist, zc + 0.12 * dist])
    target = np.array([0.0, 0.0, zc])
    cop.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse())
    cam.GetFocalLengthAttr().Set(focal)
    for _ in range(4):
        app.update()
    rep.orchestrator.step(rt_subframes=SUB, delta_time=0.0)
    rep.orchestrator.step(rt_subframes=1, delta_time=0.0)
    img = Image.fromarray(np.asarray(rgb.get_data())[..., :3].astype(np.uint8))
    raw = os.path.join(OUT, f"anatomy_{kind}.png")
    img.save(raw)
    # labels
    im = img.copy()
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", int(RES[1] * 0.017))
        fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(RES[1] * 0.02))
    except OSError:
        font = fb = ImageFont.load_default()
    for i, (lab, paths, dx, mass) in enumerate(labels):
        u, v = project([dx, 0, lo[2] - 0.055 + (-lo[2] - 0.055) * 0 + 0.0], eye, target, focal)
        u, v_top = project([dx, 0, zc - (hi[2] - lo[2]) / 2], eye, target, focal)
        y0 = RES[1] * (0.86 if i % 2 == 0 else 0.93)
        dr.line([(u, v_top + 6), (u, y0 - 4)], fill=(90, 90, 90), width=2)
        words = lab
        tw = dr.textlength(words, font=font)
        dr.text((u - tw / 2, y0), words, fill=(25, 25, 25), font=font)
        sub = f"{mass:.2f} g" if mass < 10 else f"{mass:.1f} g"
        tw2 = dr.textlength(sub, font=font)
        dr.text((u - tw2 / 2, y0 + RES[1] * 0.022), sub, fill=(90, 90, 90), font=font)
    title = {"egg": "Hen egg - anatomical cutaway (USD part graph, egg_%s)",
             "apple": "Apple - longitudinal half (USD part graph, apple_%s)",
             "avocado": "Hass avocado - half with seed (USD part graph, avocado_%s)",
             "roll": "Wheat roll (Weizenbroetchen) - cross-section through the score (USD part graph, roll_%s)",
             "banana": "Cavendish banana - lengthwise half (USD part graph, banana_%s)",
             "carrot": "Nantes carrot - lengthwise half (USD part graph, carrot_%s)",
             "onion": "Yellow onion - halved through the root (USD part graph, onion_%s)",
             "orange": "Sweet orange - lengthwise half (USD part graph, orange_%s)",
             "tomato": "Truss tomato - lengthwise half through two locules (USD part graph, tomato_%s)",
             "potato": "Potato tuber - lengthwise half (USD part graph, potato_%s)",
             "loaf": "Sandwich loaf (Toastbrot) - lengthwise half (USD part graph, loaf_%s)",
             "lemon": "Lemon - lengthwise half (USD part graph, lemon_%s)",
             "cucumber": "Greenhouse cucumber - lengthwise half (USD part graph, cucumber_%s)",
             "croissant": "Butter croissant - cut through the middle (USD part graph, croissant_%s)",
             "baguette": "Baguette - cross-section half (USD part graph, baguette_%s)",
             "pepper": "Bell pepper - lengthwise half (USD part graph, pepper_%s)"}[kind] % first(kind).split("_")[1]
    dr.text((RES[0] * 0.02, RES[1] * 0.025), title, fill=(20, 20, 20), font=fb)
    dr.text((RES[0] * 0.02, RES[1] * 0.065), "Every part is a prim under /Food/Parts carrying sdt:represents, sdt:class, "
            "sdt:partOf, food:mass_g; exploded outside -> inside", fill=(70, 70, 70), font=font)
    im.save(os.path.join(OUT, f"anatomy_{kind}_labelled.png"))
    print("[anatomy]", kind, [(l[0], round(l[3], 2)) for l in labels], flush=True)
    stage.RemovePrim(root)
    for _ in range(3):
        app.update()
app.close()
