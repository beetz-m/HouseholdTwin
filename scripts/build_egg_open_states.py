"""Cracked/opened egg states as USD assets with SDT correspondence (same entity IRIs as the whole egg):

  assets/food/egg_<s>/shell_fragments.usdc   the shell split along a jagged crack into a blunt-end and a pointed-end
                                             fragment, each with its layers (cuticle, palisade, mammillary) and the
                                             torn membranes still attached; fragments lie open side up
  assets/food/egg_<s>/opened_contents.usdc   contents released onto a flat surface: yolk dome (yolk index), thick
                                             albumen mound (Haugh unit), thin albumen ring, chalazae, germinal disc
Run: ~/Documents/LabAutomationTwin/env_cad/bin/python scripts/build_egg_open_states.py [seed]
     [--out DIR] [--hu HU --yi YI] [--shards N] [--yolk-broken]      (emulator states, food/egg_emulator.py: the
     freshness clock's Haugh unit and yolk index, shell fragments in the bowl, a broken yolk)
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pxr import Gf, Sdf, UsdGeom, UsdPhysics  # noqa: E402

from food import egg_open as EO  # noqa: E402
from food.egg import TISSUES, Egg, yolk_fan_rgb  # noqa: E402
from food.partgraph import PartGraph  # noqa: E402
from food.usd_asset import new_stage, omni_glass, omni_surface, sdt_attributes, write_mesh  # noqa: E402
from food.usd_partgraph import ensure_xforms, label, write_part_mesh  # noqa: E402

S = Sdf.ValueTypeNames.String
D = Sdf.ValueTypeNames.Double
import argparse  # noqa: E402
_ap = argparse.ArgumentParser()
_ap.add_argument("seed", nargs="?", type=int, default=1)
_ap.add_argument("--out")
_ap.add_argument("--hu", type=float)
_ap.add_argument("--yi", type=float)
_ap.add_argument("--shards", type=int, default=0)
_ap.add_argument("--yolk-broken", action="store_true")
ARGS = _ap.parse_args()
seed = ARGS.seed
if ARGS.hu is not None:
    EO.OVERRIDE.update(haugh_unit=ARGS.hu, yolk_index=ARGS.yi if not ARGS.yolk_broken else 0.12)
elif ARGS.yolk_broken:
    EO.OVERRIDE.update(haugh_unit=70.0, yolk_index=0.12)
egg = Egg(seed)
pg = PartGraph("egg", egg, egg.parts(), TISSUES).build()
d = ARGS.out or os.path.join(ROOT, "assets", "food", f"egg_{seed}")
os.makedirs(d, exist_ok=True)
iri = f"https://aicor.knowledge/household/food/egg_{seed}"
rng = np.random.default_rng(seed + 900)
mass_g = pg.mass_g()


def lin(c):
    """sRGB -> linear: OmniSurface constant colours are linear (only textures are sRGB-decoded); passing sRGB values
    straight washes them out (yolk rendered at saturation 0.47 instead of ~0.8)."""
    c = np.asarray(c, float)
    return tuple(float(v) for v in np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4))


yolk_rgb = lin(yolk_fan_rgb(egg.p["yolk_colour_fan"]))
shell_rgb = lin(egg.shell_base_rgb())


def node(prim, key, cls, part_of, obs, doc="", extra=None):
    prim.CreateAttribute("sdt:represents", S, custom=True).Set(f"{iri}/{key}")
    prim.CreateAttribute("sdt:class", S, custom=True).Set(cls)
    prim.CreateAttribute("sdt:partOf", S, custom=True).Set(part_of)
    prim.CreateAttribute("sdt:observability", S, custom=True).Set(obs)
    prim.CreateAttribute("sdt:usdPrimPath", S, custom=True).Set(str(prim.GetPath()))
    if doc:
        prim.SetDocumentation(doc)
    for k, v in (extra or {}).items():
        prim.CreateAttribute("food:" + k, S if isinstance(v, str) else D, custom=True).Set(v)
    label(prim, f"egg_{key}", key)


# ---------------------------------------------------------------- shell fragments
zc, z0 = EO.crack_curve(egg, rng)
solid = EO.below_solid(zc)
layers = ["Shell/Cuticle", "Shell/PalisadeLayer", "Shell/MammillaryLayer", "ShellMembranes/OuterShellMembrane",
          "ShellMembranes/InnerShellMembrane"]
frag = {"Pointed": {}, "Blunt": {}}
for p in layers:
    lo, hi = EO.split(pg.meshes[p], solid)
    frag["Pointed"][p], frag["Blunt"][p] = lo, hi
stage, x = new_stage(os.path.join(d, "shell_fragments.usdc"))
L = {"Shell": omni_surface(stage, "/Food/Looks/Shell", diffuse_reflection_color_image="./textures/skin.png",
                           specular_reflection_roughness=0.48, specular_reflection_weight=0.35),
     "ShellSection": omni_surface(stage, "/Food/Looks/ShellSection", diffuse_reflection_color=shell_rgb,
                                  specular_reflection_roughness=0.6),
     "Membrane": omni_surface(stage, "/Food/Looks/Membrane", diffuse_reflection_color=(0.94, 0.93, 0.89),
                              specular_reflection_roughness=0.35, subsurface_weight=0.3, subsurface_scale=0.0005,
                              subsurface_scattering_color=(0.97, 0.95, 0.9))}
mat_of = {"Shell/Cuticle": "Shell", "Shell/PalisadeLayer": "ShellSection", "Shell/MammillaryLayer": "ShellSection",
          "ShellMembranes/OuterShellMembrane": "Membrane", "ShellMembranes/InnerShellMembrane": "Membrane"}
x.GetPrim().CreateAttribute("sdt:represents", S, custom=True).Set(iri + "/shell")
sdt_attributes(x.GetPrim(), {"kind": "egg", "instance_seed": seed, "state": "shell_fragments",
                             "shell_condition": "opened", "crack_z_mm": z0 * 1e3})
rec = {"crack_z_mm": z0 * 1e3, "fragments": {}}
for side, parts in frag.items():
    fp = f"/Food/Parts/ShellFragment_{side}"
    ensure_xforms(stage, fp)
    node(stage.GetPrimAtPath(fp), f"shell_fragment_{side.lower()}", "food:EggshellFragment", iri + "/shell",
         "visible", f"{side}-end shell fragment after opening along the crack (with the torn membranes attached).")
    # lay the fragment open side up next to the origin: pointed half stays, blunt half flips over
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(fp))
    if side == "Blunt":
        xf.AddTranslateOp().Set(Gf.Vec3d(0.07, 0.0, float(egg.mer.zmax)))
        xf.AddRotateXOp().Set(180.0)
    else:
        xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -float(egg.mer.zmin)))
    m_tot = 0.0
    for p, mesh in parts.items():
        info = pg.info[p]
        key = info["part"].key
        pp = f"{fp}/{p.split('/')[-1]}"
        ensure_xforms(stage, pp)
        node(stage.GetPrimAtPath(pp), f"{key}_{side.lower()}", info["part"].cls, iri + f"/shell_fragment_{side.lower()}",
             "visible", f"{side}-end piece of the {key.replace('_', ' ')}")
        uv = egg.uv_of if p == "Shell/Cuticle" else None
        write_part_mesh(stage, pp + "/Geom", mesh, np.zeros(len(mesh.faces), bool), L[mat_of[p]], None, uv_fn=uv)
        m = abs(mesh.volume) * info["density"] * 1e3
        stage.GetPrimAtPath(pp).CreateAttribute("food:mass_g", D, custom=True).Set(float(m))
        m_tot += m
    rec["fragments"][side] = round(m_tot, 3)
stage.GetRootLayer().Save()

# ---------------------------------------------------------------- opened contents
cont, q = EO.opened_contents(egg, pg.info, mass_g, rng, yolk_broken=ARGS.yolk_broken)
stage, x = new_stage(os.path.join(d, "opened_contents.usdc"))
L = {"AlbumenThin": omni_glass(stage, "/Food/Looks/AlbumenThin", color=(0.99, 0.995, 0.96), ior=1.36, frosting=0.0,
                               depth=0.3),
     "AlbumenThick": omni_glass(stage, "/Food/Looks/AlbumenThick", color=(0.98, 0.99, 0.93), ior=1.36,
                                frosting=0.05, depth=0.15),
     "Yolk": omni_surface(stage, "/Food/Looks/Yolk", diffuse_reflection_color=yolk_rgb, specular_reflection_roughness=0.07,
                          specular_reflection_weight=0.8, subsurface_weight=0.28, subsurface_scale=0.003,
                          subsurface_scattering_color=(0.98, 0.52, 0.04)),
     "GerminalDisc": omni_surface(stage, "/Food/Looks/GerminalDisc", diffuse_reflection_color=(0.99, 0.93, 0.80),
                                  specular_reflection_roughness=0.2),
     "Chalaza": omni_glass(stage, "/Food/Looks/Chalaza", color=(0.93, 0.93, 0.88), ior=1.38, frosting=0.35,
                           depth=0.004)}
meta = {"Albumen/ThinAlbumenPool": ("thin_albumen_pool", "food:ThinAlbumen", "AlbumenThin",
                                    "Outer + inner thin albumen, spread as a ring round the thick albumen."),
        "Albumen/ThickAlbumen": ("thick_albumen", "food:ThickAlbumen", "AlbumenThick",
                                 f"Thick albumen mound: height {q['thick_height_mm']:.1f} mm -> Haugh unit "
                                 f"{q['haugh_unit']:.0f} ({q['usda_albumen_grade']})."),
        "Yolk/YolkBody": ("yolk", "food:Yolk", "Yolk", f"Yolk dome, yolk index {q['yolk_index']:.2f}."),
        "Yolk/GerminalDisc": ("germinal_disc", "food:GerminalDisc", "GerminalDisc", "Faces up once released."),
        "Albumen/Chalazae/BluntEnd": ("chalaza_blunt", "food:Chalaza", "Chalaza", ""),
        "Albumen/Chalazae/PointedEnd": ("chalaza_pointed", "food:Chalaza", "Chalaza", "")}
x.GetPrim().CreateAttribute("sdt:represents", S, custom=True).Set(iri + "/contents")
sdt_attributes(x.GetPrim(), {"kind": "egg", "instance_seed": seed, "state": "opened_contents", "contents": "released",
                             "yolk": "whole", **{k: (float(v) if not isinstance(v, str) else v) for k, v in q.items()}})
for p, mesh in cont.items():
    key, cls, mat, doc = meta[p]
    pp = "/Food/Parts/" + p
    ensure_xforms(stage, pp)
    node(stage.GetPrimAtPath(pp), key, cls, iri + "/contents", "visible", doc)
    g = write_part_mesh(stage, pp + "/Geom", mesh, np.zeros(len(mesh.faces), bool), L[mat], None)
    if mat in ("AlbumenThin", "AlbumenThick", "Chalaza"):     # clear liquid: no hard shadow (RTX would need caustics)
        UsdGeom.PrimvarsAPI(g).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
    stage.GetPrimAtPath(pp).CreateAttribute("food:volume_cm3", D, custom=True).Set(float(abs(mesh.volume) * 1e6))
if ARGS.yolk_broken:                                     # broken yolk: flat yolk pool, labelled as damage
    yp = stage.GetPrimAtPath("/Food/Parts/Yolk/YolkBody")
    yp.CreateAttribute("sdt:class", S, custom=True).Set("food:YolkBroken")
    yp.SetDocumentation(f"Broken yolk: membrane torn, yolk run out as a flat pool ({q['yolk_diameter_mm']:.0f} mm, "
                        f"{q['yolk_height_mm']:.1f} mm high).")
    label(yp, "egg_yolk_broken", "yolk_broken")
srng = np.random.default_rng(seed + 77)
if ARGS.shards:
    L["ShardShell"] = omni_surface(stage, "/Food/Looks/ShardShell", diffuse_reflection_color=shell_rgb,
                                   specular_reflection_roughness=0.5)
import trimesh  # noqa: E402
_surf = trimesh.util.concatenate([cont[p] for p in ("Albumen/ThinAlbumenPool", "Albumen/ThickAlbumen", "Yolk/YolkBody")])
_ray = trimesh.ray.ray_triangle.RayMeshIntersector(_surf)
placed = []
r_lo, r_hi = q["yolk_diameter_mm"] * 0.5e-3 + 3e-3, q["R_thick_mm"] * 1e-3 * 0.95
for k in range(ARGS.shards):                              # shell fragments that fell into the bowl (monitor ground truth):
    a_ = srng.uniform(3.0e-3, 7e-3)                       # curved shell pieces (shell radius of curvature ~ B/2) with
    nv = srng.integers(5, 8)                              # a jagged outline, lying ON the contents outside the yolk
    th = np.sort(srng.uniform(0, 2 * np.pi, nv))          # (they float on the albumen; fished out with a shell half)
    rr = a_ / 2 * srng.uniform(0.6, 1.0, nv)
    poly = np.stack([rr * np.cos(th), rr * np.sin(th)], 1)
    t_s = egg.t_s
    sh = trimesh.creation.extrude_polygon(trimesh.path.polygons.Polygon(poly), t_s)
    Rc = egg.p["shape_index_B_over_L"] * egg.p["length_L"] * 0.5e-3
    v = sh.vertices.copy()
    v[:, 2] += (1 if srng.random() < 0.5 else -1) * (v[:, 0] ** 2 + v[:, 1] ** 2) / (2 * Rc)   # cup up or dome up
    sh = trimesh.Trimesh(v, sh.faces, process=False)
    for _ in range(200):
        ang, r_ = srng.uniform(0, 2 * np.pi), srng.uniform(r_lo, max(r_lo + 1e-3, r_hi))
        xy = np.array([r_ * np.cos(ang), r_ * np.sin(ang)])
        if all(np.linalg.norm(xy - o) > 9e-3 for o in placed):
            break
    placed.append(xy)
    hit, _, _ = _ray.intersects_location([[xy[0], xy[1], 0.05]], [[0, 0, -1]])
    ztop = float(hit[:, 2].max()) if len(hit) else 0.0
    sh.apply_transform(trimesh.transformations.rotation_matrix(srng.uniform(0, np.pi), [0, 0, 1]))
    sh.apply_translation([xy[0], xy[1], ztop - float(sh.vertices[:, 2].min()) - 0.1e-3])   # settles 0.1 mm in
    pp = f"/Food/Parts/ShellShards/Shard_{k}"
    ensure_xforms(stage, pp)
    node(stage.GetPrimAtPath(pp), f"shell_shard_{k}", "food:EggshellFragment", iri + "/shell", "visible",
         "A shell fragment in the bowl (from the strike / opening; egg_crack.open_up).")
    label(stage.GetPrimAtPath(pp), "egg_shell_shard", "shell_shard")
    write_part_mesh(stage, pp + "/Geom", sh, np.zeros(len(sh.faces), bool), L["ShardShell"], None)
stage.GetRootLayer().Save()
rec["contents"] = {k: (float(v) if not isinstance(v, str) else v) for k, v in q.items()}
rec["shards_in_bowl"], rec["yolk_broken"] = ARGS.shards, bool(ARGS.yolk_broken)
json.dump(rec, open(os.path.join(d, "open_states.json"), "w"), indent=1)
print(json.dumps(rec, indent=1))

# ---------------------------------------------------------------- simulation inputs: fragments in the egg frame + fill
from food.egg import TI  # noqa: E402

stage, x = new_stage(os.path.join(d, "shell_fragments_inplace.usdc"))
Ls = {"Shell": omni_surface(stage, "/Food/Looks/Shell", diffuse_reflection_color_image="./textures/skin.png",
                            specular_reflection_roughness=0.48, specular_reflection_weight=0.35),
      "ShellSection": omni_surface(stage, "/Food/Looks/ShellSection", diffuse_reflection_color=shell_rgb,
                                   specular_reflection_roughness=0.6),
      "Membrane": omni_surface(stage, "/Food/Looks/Membrane", diffuse_reflection_color=(0.94, 0.93, 0.89),
                               specular_reflection_roughness=0.35)}
for side, parts in frag.items():
    fp = f"/Food/ShellFragment_{side}"
    ensure_xforms(stage, fp)
    for p, mesh in parts.items():
        uv = egg.uv_of if p == "Shell/Cuticle" else None
        write_part_mesh(stage, f"{fp}/{p.split('/')[-1]}", mesh, np.zeros(len(mesh.faces), bool), Ls[mat_of[p]], None,
                        uv_fn=uv)
stage.GetRootLayer().Save()
h = 1.5e-3
lo_, hi_ = egg.bbox()
ax = [np.arange(lo_[i] + h / 2, hi_[i], h) for i in range(3)]
G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
lab = egg.label(G)
alb = np.isin(lab, [TI[n] for n in ("outer_thin_albumen", "thick_albumen", "inner_thin_albumen",
                                    "chalaziferous_layer", "chalaza")])
rn_, th_, z_ = egg.norm_coords(G)
alb &= egg.m_alb.offset(1.2e-3).contains(rn_, z_)                       # clear of the membrane (no overlap at start)
alb &= np.linalg.norm(G - egg.c_yolk, axis=1) > egg.r_yolk + 1.0e-3     # clear of the rigid yolk
np.savez(os.path.join(d, "open_sim_inputs.npz"), albumen=G[alb], spacing=h, crack_z=z0, yolk_c=egg.c_yolk,
         yolk_r=egg.r_yolk, zmin=egg.mer.zmin, zmax=egg.mer.zmax, rmax=egg.mer.rho.max(),
         yolk_index=q["yolk_index"], yolk_mass_kg=sum(pg.info[p]["mass_g"] for p in pg.info if p.startswith("Yolk/")
                                                      and "mass_g" in pg.info[p]) * 1e-3)
print("albumen particles", int(alb.sum()), "at", h * 1e3, "mm")
