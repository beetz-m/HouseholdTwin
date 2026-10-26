"""Build the food twins' USD assets from their part graphs (one prim per anatomical part, SDT correspondence on every
prim), with cut states, an SDT belief layer and a simulator-truth layer.

Run: ~/Documents/LabAutomationTwin/env_cad/bin/python scripts/build_food_assets.py [egg apple avocado] [--seeds egg:3,...]

assets/food/<kind>_<seed>/
  whole.usdc            part graph, rigid body (mass/COM/inertia summed over parts), interior parts invisible
  <cut>.usdc            egg: cutaway; apple: half_a, half_b (through the axis, both halves open a carpel);
                        avocado: half_a (keeps the whole seed), half_b (seed cavity) -- the knife goes round the seed
  sdt.usda              robot beliefs (priors from the parameter register) over whole.usdc
  truth.usda            hidden instance parameters (overs; never read by the robot)
  textures/             skin.png, seed.png, section_<cut>.png
  instance.json         parts, masses, register statuses, hidden parameters
"""

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from food import params as PR  # noqa: E402
from food.apple import TISSUES as APPLE_T, Apple  # noqa: E402
from food.avocado import TISSUES as AVO_T, Avocado  # noqa: E402
from food.egg import TISSUES as EGG_T, Egg, yolk_fan_rgb  # noqa: E402
from food.banana import TISSUES as BANANA_T, Banana  # noqa: E402
from food.carrot import TISSUES as CARROT_T, Carrot  # noqa: E402
from food.partgraph import PartGraph  # noqa: E402
from food.roll import TISSUES as ROLL_T, Roll  # noqa: E402
from food.loaf import Loaf  # noqa: E402
from food.onion import Onion  # noqa: E402
from food.orange import TISSUES as ORANGE_T, Orange  # noqa: E402
from food.potato import TISSUES as POTATO_T, Potato  # noqa: E402
from food.tomato import TISSUES as TOMATO_T, Tomato  # noqa: E402
from food.lemon import Lemon  # noqa: E402
from food.cucumber import TISSUES as CUCUMBER_T, Cucumber  # noqa: E402
from food.croissant import TISSUES as CROISSANT_T, Croissant  # noqa: E402
from food.baguette import Baguette  # noqa: E402
from food.pepper import TISSUES as PEPPER_T, Pepper  # noqa: E402
from food.texture import save_rgb, surface_grid  # noqa: E402
from food.usd_asset import omni_surface  # noqa: E402
from food.usd_partgraph import write_sdt_layer, write_state, write_truth_layer  # noqa: E402

OUT = os.path.join(ROOT, "assets", "food")
TEX_W, TEX_H, SEC_RES = 2048, 1024, 2048
KINDS = {"egg": (Egg, EGG_T), "apple": (Apple, APPLE_T), "avocado": (Avocado, AVO_T),
         "roll": (Roll, ROLL_T), "banana": (Banana, BANANA_T), "carrot": (Carrot, CARROT_T),
         "loaf": (Loaf, None), "tomato": (Tomato, TOMATO_T), "potato": (Potato, POTATO_T), "onion": (Onion, None),
         "orange": (Orange, ORANGE_T),
         # wave 3 (built 20 Oct 2026): lemon reuses the orange's tissues, the baguette the roll's
         "lemon": (Lemon, ORANGE_T), "cucumber": (Cucumber, CUCUMBER_T), "croissant": (Croissant, CROISSANT_T),
         "baguette": (Baguette, ROLL_T), "pepper": (Pepper, PEPPER_T)}


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


# ------------------------------------------------------------------------------------------------ materials
def material_table(kind, body):
    """Material key -> OmniSurface inputs (sRGB colours). 'Section' (cut faces) is added per state."""
    tex = lambda n: f"./textures/{n}"  # noqa: E731
    if kind == "egg":
        sh = tuple(float(c) for c in body.shell_base_rgb())
        yolk = tuple(float(c) for c in yolk_fan_rgb(body.p["yolk_colour_fan"]))
        return {
            "Shell": dict(diffuse_reflection_color_image=tex("skin.png"), specular_reflection_roughness=0.48,
                          specular_reflection_weight=0.35, subsurface_weight=0.22, subsurface_scale=0.0015,
                          subsurface_scattering_color=(0.95, 0.78, 0.60)),
            "ShellSection": dict(diffuse_reflection_color=sh, specular_reflection_roughness=0.6),
            "Membrane": dict(diffuse_reflection_color=(0.95, 0.94, 0.90), specular_reflection_roughness=0.35),
            "AlbumenThin": dict(diffuse_reflection_color=(0.82, 0.85, 0.76), specular_reflection_roughness=0.08,
                                specular_reflection_weight=0.8),
            "AlbumenThick": dict(diffuse_reflection_color=(0.88, 0.90, 0.81), specular_reflection_roughness=0.08,
                                 specular_reflection_weight=0.8),
            "Chalaza": dict(diffuse_reflection_color=(0.96, 0.95, 0.89), specular_reflection_roughness=0.2),
            "Vitelline": dict(diffuse_reflection_color=yolk, specular_reflection_roughness=0.06,
                              specular_reflection_weight=0.9),
            "Yolk": dict(diffuse_reflection_color=yolk, specular_reflection_roughness=0.15, subsurface_weight=0.4,
                         subsurface_scale=0.003, subsurface_scattering_color=yolk),
            "WhiteYolk": dict(diffuse_reflection_color=(0.99, 0.82, 0.48), specular_reflection_roughness=0.3),
            "GerminalDisc": dict(diffuse_reflection_color=(0.99, 0.93, 0.80), specular_reflection_roughness=0.3),
            "AirCell": dict(diffuse_reflection_color=(0.6, 0.6, 0.6)),
        }
    if kind == "apple":
        return {
            "Skin": dict(diffuse_reflection_color_image=tex("skin.png"), specular_reflection_roughness=0.32,
                         specular_reflection_weight=0.5, coat_weight=0.35, coat_roughness=0.12, subsurface_weight=0.12,
                         subsurface_scale=0.002, subsurface_scattering_color=(0.95, 0.55, 0.35)),
            "SkinSection": dict(diffuse_reflection_color=(0.80, 0.30, 0.18), specular_reflection_roughness=0.4),
            "Stem": dict(diffuse_reflection_color=(0.30, 0.20, 0.10), specular_reflection_roughness=0.7),
            "Sepal": dict(diffuse_reflection_color=(0.17, 0.11, 0.05), specular_reflection_roughness=0.75),
            "CalyxTube": dict(diffuse_reflection_color=(0.55, 0.42, 0.25), specular_reflection_roughness=0.6),
            "Flesh": dict(diffuse_reflection_color=(0.90, 0.84, 0.60), specular_reflection_roughness=0.22,
                          subsurface_weight=0.15, subsurface_scale=0.002, subsurface_scattering_color=(0.92, 0.86, 0.62)),
            "Pith": dict(diffuse_reflection_color=(0.87, 0.85, 0.58), specular_reflection_roughness=0.22),
            "Bundle": dict(diffuse_reflection_color=(0.82, 0.83, 0.56), specular_reflection_roughness=0.3),
            "Endocarp": dict(diffuse_reflection_color=(0.94, 0.91, 0.79), specular_reflection_roughness=0.18,
                             specular_reflection_weight=0.7),
            "Locule": dict(diffuse_reflection_color=(0.86, 0.81, 0.64)),
            "Seed": dict(diffuse_reflection_color=(0.26, 0.12, 0.05), specular_reflection_roughness=0.25,
                         specular_reflection_weight=0.7),
        }
    return {
        "Skin": dict(diffuse_reflection_color_image=tex("skin.png"), specular_reflection_roughness=0.5,
                     specular_reflection_weight=0.45),
        "SkinSection": dict(diffuse_reflection_color=(0.25, 0.33, 0.13), specular_reflection_roughness=0.5),
        "StemButton": dict(diffuse_reflection_color=(0.32, 0.24, 0.12), specular_reflection_roughness=0.7),
        "FleshOuter": dict(diffuse_reflection_color=(0.62, 0.70, 0.30), specular_reflection_roughness=0.25,
                           subsurface_weight=0.25, subsurface_scale=0.003, subsurface_scattering_color=(0.7, 0.8, 0.4)),
        "FleshMiddle": dict(diffuse_reflection_color=(0.80, 0.82, 0.48), specular_reflection_roughness=0.25,
                            subsurface_weight=0.25, subsurface_scale=0.003, subsurface_scattering_color=(0.85, 0.85, 0.5)),
        "FleshInner": dict(diffuse_reflection_color=(0.88, 0.85, 0.52), specular_reflection_roughness=0.22,
                           subsurface_weight=0.25, subsurface_scale=0.003, subsurface_scattering_color=(0.9, 0.88, 0.55)),
        "Strand": dict(diffuse_reflection_color=(0.74, 0.74, 0.40), specular_reflection_roughness=0.3),
        "Endocarp": dict(diffuse_reflection_color=(0.88, 0.84, 0.55), specular_reflection_roughness=0.22,
                         specular_reflection_weight=0.6),
        "SeedCoat": dict(diffuse_reflection_color_image=tex("seed.png"), specular_reflection_roughness=0.38,
                         specular_reflection_weight=0.55),
        "SeedCoatInner": dict(diffuse_reflection_color=(0.55, 0.36, 0.20), specular_reflection_roughness=0.5),
        "Cotyledon": dict(diffuse_reflection_color=(0.90, 0.82, 0.62), specular_reflection_roughness=0.35),
        "Embryo": dict(diffuse_reflection_color=(0.80, 0.70, 0.45), specular_reflection_roughness=0.4),
    }


def materials_fn(kind, body, section_tex=None):
    table = material_table(kind, body)

    def fn(stage):
        L = {k: omni_surface(stage, f"/Food/Looks/{k}", **v) for k, v in table.items()}
        if section_tex:
            L["Section"] = omni_surface(stage, "/Food/Looks/Section", diffuse_reflection_color_image=section_tex,
                                        specular_reflection_roughness=0.2, specular_reflection_weight=0.6,
                                        subsurface_weight=0.15, subsurface_scale=0.002,
                                        subsurface_scattering_color=(0.95, 0.9, 0.7))
        return L
    return fn


# ------------------------------------------------------------------------------------------------ SDT priors / truth
def prior_beliefs(kind, body, mp):
    """The robot's starting beliefs: register priors (distribution summaries) + visible/weighed facts."""
    def q(spec, n=4000, seed=1):
        rng = np.random.default_rng(seed)
        x = np.array([PR.draw(spec, rng) for _ in range(n)])
        return float(x.mean()), float(np.percentile(x, 5)), float(np.percentile(x, 95))
    R = PR.REGISTERS[kind]
    B = {"": {"sdt:epistemicStatus": "prior + observation", "sdt:belief:mass_g": mp["mass_kg"] * 1e3,
              "sdt:belief:mass_g_source": "weighed"}}
    if kind == "egg":
        m, lo, hi = q(R["F_break_equatorial"][1])
        B[""].update({"sdt:belief:shell_condition": "intact 0.97 | microcracked 0.03 (prior)",
                      "sdt:belief:contents": "in_shell 1.0"})
        B["Shell"] = {"sdt:belief:F_break_equatorial_N_mean": m, "sdt:belief:F_break_equatorial_N_p05": lo,
                      "sdt:belief:F_break_equatorial_N_p95": hi, "sdt:epistemicStatus": "prior (register: SOURCED)"}
        B["AirCell"] = {"sdt:belief:height_mm": "unknown until candled (age and storage unknown)",
                        "sdt:epistemicStatus": "unobserved"}
    elif kind == "apple":
        m, lo, hi = q(R["flesh_E"][1])
        B[""].update({"sdt:belief:bruise": "none 0.9 | minor 0.1 (prior; bruises show after hours)",
                      "sdt:belief:integrity": "whole 1.0"})
        B["Cortex"] = {"sdt:belief:flesh_E_MPa_mean": m, "sdt:belief:flesh_E_MPa_p05": lo,
                       "sdt:belief:flesh_E_MPa_p95": hi, "sdt:epistemicStatus": "prior"}
        B["Core"] = {"sdt:belief:seed_count": "mean 7.0 (5 carpels x 0-2 seeds), unobserved until cut",
                     "sdt:epistemicStatus": "unobserved"}
    else:
        B[""].update({"sdt:belief:ripeness_stage_mean": 3.0, "sdt:belief:ripeness_stage_sd": 1.15,
                      "sdt:belief:ripeness_stage_source": "flat prior over the 1..5 chart; squeeze to update",
                      "sdt:belief:integrity": "whole 1.0"})
        B["Mesocarp"] = {"sdt:belief:bruised": "unknown (no squeeze history)", "sdt:epistemicStatus": "unobserved"}
        B["Seed"] = {"sdt:epistemicStatus": "unobserved (inside)"}
    return B


def truth(kind, body):
    p = {k: v for k, v in body.p.items() if not k.startswith("_")}
    T = {"": {f"truth:{k}": v for k, v in p.items() if isinstance(v, (int, float, str))}}
    if kind == "egg":
        T["Shell"] = {"truth:F_break_equatorial_N": body.break_force(np.pi / 2),
                      "truth:F_break_axial_N": body.break_force(0.0),
                      "truth:contact_stiffness_equator_N_per_mm": body.contact_stiffness(np.pi / 2) / 1e3}
        T["AirCell"] = {"truth:height_mm": body.p["air_cell_height"]}
    if kind == "avocado":
        T[""]["truth:penetrometer_force_N"] = body.firmness()
        T[""]["truth:flesh_E_MPa"] = body.flesh_E()
    return T


def subsets_fn(kind, body):
    """Named surface regions of the outer skin (GeomSubsets on the cuticle mesh) -- apple for now."""
    return {"cuticle": body.skin_regions} if kind == "apple" else {}


# ------------------------------------------------------------------------------------------------ build
def plane_uv(origin, e1, e2, ext):
    o, e1, e2 = map(np.asarray, (origin, e1, e2))
    return lambda P: np.stack([((P - o) @ e1 + ext) / (2 * ext), ((P - o) @ e2 + ext) / (2 * ext)], 1)


def build_spec(kind, seed, body, tissues, t0):
    """Generic build for food classes that declare their asset in body.asset_spec() (wave 1 onwards):
    materials, textures, root attributes, cuts (any plane, two named sides each), beliefs and truth."""
    spec = body.asset_spec()
    d = os.path.join(OUT, f"{kind}_{seed}")
    tex = os.path.join(d, "textures")
    os.makedirs(tex, exist_ok=True)
    pg = PartGraph(kind, body, body.parts(), getattr(body, "tissues", tissues)).build(**spec.get("build", {}))
    iri = f"https://aicor.knowledge/household/food/{kind}_{seed}"
    for name, fn_point, albedo, W, H in spec.get("textures", []):
        P = surface_grid(fn_point, W, H)
        save_rgb(os.path.join(tex, name), albedo(P.reshape(-1, 3), seed + 3).reshape(H, W, 3), gamma=False)

    def mats(section_tex=None):
        table = {k: {kk: (f"./textures/{vv[4:]}" if isinstance(vv, str) and vv.startswith("tex:") else vv)
                     for kk, vv in v.items()} for k, v in spec["materials"].items()}

        def fn(stage):
            L = {k: omni_surface(stage, f"/Food/Looks/{k}", **v) for k, v in table.items()}
            if section_tex:
                L["Section"] = omni_surface(stage, "/Food/Looks/Section", diffuse_reflection_color_image=section_tex,
                                            **spec.get("section_material", {"specular_reflection_roughness": 0.5}))
            return L
        return fn
    voids = {path for path, i in pg.info.items() if 0 < i.get("density", 1000) < 10}
    pieces = {p: (m, np.zeros(len(m.faces), bool)) for p, m in pg.meshes.items() if m is not None}
    root = dict({"kind": kind, "instance_seed": seed, "aicor_class": f"food:{type(body).__name__}"}, **spec["root"])
    common = dict(uv_fns=spec.get("uv_fns", {}), points=pg.points, voids=voids, subsets=spec.get("subsets", {}),
                  collision_fn=spec.get("collision_fn"))
    mp = write_state(os.path.join(d, "whole.usdc"), pg, iri, "whole", pieces, mats(), root, spec["friction"],
                     spec["restitution"], hide_interior=True, **common)
    rec = {"kind": kind, "seed": seed, "iri": iri, "hidden_params": body.p,
           "register_status": dict(PR.status_counts(kind)),
           "whole": {"mass_g": mp["mass_kg"] * 1e3, "com_mm": mp["com_m"] * 1e3},
           "parts": {k: {kk: vv for kk, vv in v.items() if kk != "part"} for k, v in pg.info.items()},
           "cuts": {}, "cut_failures": {}}
    for cut in spec.get("cuts", []):
        o, e1, e2, ext = (np.asarray(cut["origin"], float), np.asarray(cut["e1"], float),
                          np.asarray(cut["e2"], float), cut["ext"])
        normal = np.cross(e1, e2)
        lab, Pg = body.section(o, e1, e2, ext, cut.get("res", SEC_RES))
        sec = f"section_{cut['name']}.png"
        save_rgb(os.path.join(tex, sec), body.section_rgb(lab, Pg, seed), gamma=False)
        keep = cut.get("keep_whole", set())
        sub = PartGraph.__new__(PartGraph)
        sub.meshes = {p: m for p, m in pg.meshes.items() if m is not None and p not in keep}
        side = dict(zip(("pos", "neg"), sub.cut(o, normal)))
        if sub.cut_failures:
            print(f"  WARNING {kind}_{seed} {cut['name']}: slices not closed: {sub.cut_failures}", flush=True)
        rec["cut_failures"][cut["name"]] = sub.cut_failures
        for p in keep:
            side[cut.get("keep_side", "pos")][p] = (pg.meshes[p], np.zeros(len(pg.meshes[p].faces), bool))
        for sname, (which, over) in cut["states"].items():
            pcs = side[which]
            mph = write_state(os.path.join(d, f"{sname}.usdc"), pg, iri + f"/{sname}", sname, pcs,
                              mats(f"./textures/{sec}"), dict(root, **over), spec["friction"], spec["restitution"],
                              cap_uv_fn=plane_uv(o, e1, e2, ext), point_clip=(normal, which, o), **common)
            rec["cuts"][sname] = {"cut": cut["name"], "origin_mm": o * 1e3, "normal": normal, "side": which,
                                  "parts": len(pcs), "mass_g": None if mph is None else mph["mass_kg"] * 1e3}
    for sl in spec.get("slicings", []):                   # parallel cuts -> one rigid body per slice (kitchen slicing)
        n_ = np.asarray(sl["normal"], float)
        n_ /= np.linalg.norm(n_)
        offs = sorted(sl["offsets"])
        e1, e2 = np.asarray(sl["e1"], float), np.asarray(sl["e2"], float)
        o_sec = n_ * sl.get("section_at", 0.0)
        lab, Pg = body.section(o_sec, e1, e2, sl["ext"], sl.get("res", SEC_RES))
        sec = f"section_{sl['name']}.png"
        save_rgb(os.path.join(tex, sec), body.section_rgb(lab, Pg, seed), gamma=False)
        from food.partgraph import Part as _Part
        nsl = len(offs) + 1
        grp, pc = sl["group"], sl["piece"]
        parts2 = [_Part(grp.lower(), grp, "food:SliceGroup", None, observability="visible",
                        doc=f"{nsl} slices cut along {n_.round(3).tolist()}", attrs={"count": nsl})]
        info2, pieces2, bodies = {}, {}, {}
        for k in range(nsl):
            sp = f"{grp}/{pc}_{k + 1:02d}"
            end = k in (0, nsl - 1)
            parts2.append(_Part(f"{pc.lower()}_{k + 1:02d}", sp, sl.get("end_class" if end else "piece_class",
                                                                          "food:Slice"), None, observability="visible",
                                attrs={"index": k + 1, "end_piece": "yes" if end else "no"}))
            bodies[sp] = set()
        for q in pg.parts:                                 # every original part, below each slice
            for k in range(nsl):
                sp = f"{grp}/{pc}_{k + 1:02d}"
                parts2.append(_Part(f"{pc.lower()}_{k + 1:02d}_{q.key}", f"{sp}/{q.path}", q.cls,
                                    None if q.geom is None or q.geom[0] == "points" else q.geom, q.material,
                                    q.observability, q.doc, q.density, q.interior, dict(q.attrs)))
        for path, m in pg.meshes.items():
            if m is None or path in voids:
                continue
            rest = m
            for k, off in enumerate(offs + [None]):
                if off is None:
                    piece = rest
                else:
                    res = PartGraph._split_manifold(rest, n_ * off, n_) if rest is not None else None
                    if res is None:
                        piece, rest = None, None
                        if rest is None:
                            pass
                    else:
                        piece = res[1][0] if res[1] is not None else None
                        rest = res[0][0] if res[0] is not None else None
                if piece is None or len(piece.faces) == 0:
                    continue
                c = piece.triangles_center @ n_
                on = np.zeros(len(piece.faces), bool)
                for off2 in offs:
                    on |= np.abs(c - off2) < 2e-7
                cap = on & (np.abs(piece.face_normals @ n_) > 0.999)
                sp = f"{grp}/{pc}_{k + 1:02d}"
                pieces2[f"{sp}/{path}"] = (piece, cap)
                info2[f"{sp}/{path}"] = dict(pg.info[path], part=[x for x in parts2 if x.path == f"{sp}/{path}"][0])
                bodies[sp].add(f"{sp}/{path}")
        pg2 = PartGraph.__new__(PartGraph)
        pg2.kind, pg2.parts, pg2.info = pg.kind, parts2, info2
        bodies = {k: v for k, v in bodies.items() if v}
        mps = write_state(os.path.join(d, f"{sl['name']}.usdc"), pg2, iri + f"/{sl['name']}", sl["name"], pieces2,
                          mats(f"./textures/{sec}"), dict(root, **sl.get("root", {})), spec["friction"],
                          spec["restitution"], cap_uv_fn=plane_uv(o_sec, e1, e2, sl["ext"]), bodies=bodies,
                          uv_fns={f"{pc.lower()}_{k + 1:02d}_{key}": fn for key, fn in spec.get("uv_fns", {}).items()
                                  for k in range(nsl)}, collision_fn=None, subsets={})
        rec["cuts"][sl["name"]] = {"cut": sl["name"], "slices": len(bodies), "side": "all",
                                   "mass_g": mps["mass_kg"] * 1e3, "offsets_mm": [o * 1e3 for o in offs]}
    for ex in spec.get("extra_states", []):               # same pieces, several rigid bodies (e.g. a sliced loaf)
        mpx = write_state(os.path.join(d, f"{ex['name']}.usdc"), pg, iri + f"/{ex['name']}", ex["name"], pieces,
                          mats(), dict(root, **ex.get("root", {})), spec["friction"], spec["restitution"],
                          hide_interior=ex.get("hide_interior", False), bodies=ex["bodies"], **common)
        rec["cuts"][ex["name"]] = {"cut": "extra_state", "bodies": len(ex["bodies"]), "side": "all",
                                   "mass_g": mpx["mass_kg"] * 1e3}
    write_sdt_layer(os.path.join(d, "sdt.usda"), "./whole.usdc", pg, spec["beliefs"](mp))
    write_truth_layer(os.path.join(d, "truth.usda"), spec["truth"]())
    rec.update(spec.get("record", {}))
    with open(os.path.join(d, "instance.json"), "w") as fh:
        json.dump(jsonable(rec), fh, indent=1)
    print(f"[{kind}_{seed}] {len(pg.meshes)} part meshes, {len(pg.points)} point sets, mass {mp['mass_kg'] * 1e3:.1f} g, "
          f"states {list(rec['cuts'])} ({time.time() - t0:.0f} s) -> {d}", flush=True)
    return d


def build(kind, seed, body=None, name=None, belief_update=None, truth_extra=None, observation=None):
    """Build one instance. Optional: a prepared ``body`` (e.g. with parameters estimated from a photo), the instance
    ``name`` (default <kind>_<seed>), ``belief_update`` merged into the SDT belief layer per prim ({"": {...}}),
    ``truth_extra`` added to the truth layer's root, and an ``observation`` record stored in instance.json."""
    t0 = time.time()
    cls, tissues = KINDS[kind]
    body = body if body is not None else cls(seed)
    if hasattr(body, "asset_spec"):
        return build_spec(kind, seed, body, tissues, t0)
    inst_name = name or f"{kind}_{seed}"
    d = os.path.join(OUT, inst_name)
    tex = os.path.join(d, "textures")
    os.makedirs(tex, exist_ok=True)
    pg = PartGraph(kind, body, body.parts(), tissues).build()
    iri = f"https://aicor.knowledge/household/food/{inst_name}"
    friction = body.p.get("friction_shell_rubber", body.p.get("friction_skin_rubber", 0.6))
    rest = {"egg": 0.05, "apple": 0.25, "avocado": 0.1}[kind]
    albedo = body.shell_albedo if kind == "egg" else body.skin_albedo
    P = surface_grid(body.point_uv, TEX_W, TEX_H)
    save_rgb(os.path.join(tex, "skin.png"), albedo(P.reshape(-1, 3), seed + 3).reshape(TEX_H, TEX_W, 3), gamma=False)
    uv_fns = {"cuticle": body.uv_of}
    if kind == "avocado":
        P = surface_grid(body.seed_point_uv, 1024, 512)
        save_rgb(os.path.join(tex, "seed.png"), body.seed_albedo(P.reshape(-1, 3), seed).reshape(512, 1024, 3),
                 gamma=False)
        uv_fns["outer_testa"] = body.seed_uv_of
    voids = {path for path, i in pg.info.items() if 0 < i.get("density", 1000) < 10}
    pieces = {p: (m, np.zeros(len(m.faces), bool)) for p, m in pg.meshes.items() if m is not None}
    root = {"kind": kind, "instance_seed": seed, "aicor_class": f"food:{cls.__name__}"}
    if kind == "egg":
        g, eu, us = body.grade(pg.mass_g())
        root.update({"shell_condition": "intact", "contents": "in_shell", "weight_grade": g, "eu_quality": eu,
                     "usda_air_cell_grade": us, "shell_colour": body.p["colour"]})
    elif kind == "apple":
        root.update({"integrity": "whole", "bruise": "none", "peel": "on", "core": "in"})
    else:
        root.update({"integrity": "whole", "seed": "in", "skin": "on"})
    mp = write_state(os.path.join(d, "whole.usdc"), pg, iri, "whole", pieces, materials_fn(kind, body), root,
                     friction, rest, uv_fns=uv_fns, hide_interior=True, points=pg.points, voids=voids,
                     subsets=subsets_fn(kind, body))
    rec = {"kind": kind, "seed": seed, "iri": iri, "hidden_params": body.p,
           "register_status": dict(PR.status_counts(kind)),
           "whole": {"mass_g": mp["mass_kg"] * 1e3, "com_mm": mp["com_m"] * 1e3},
           "parts": {k: {kk: vv for kk, vv in v.items() if kk != "part"} for k, v in pg.info.items()}}
    ext = float(max(body.mer.rho.max() * 1.12, (body.mer.zmax - body.mer.zmin) / 2 * 1.06))
    if kind == "egg":
        cuts = {"cutaway": (0.0, "keep_neg")}             # plane x-z contains the germinal disc (+x)
    elif kind == "apple":
        best = None
        for tk in body.loc_theta:                         # halving plane 18 deg off a carpel: both halves open one
            for off in (-np.radians(18), np.radians(18)):
                e_ = np.array([np.cos(tk + off), np.sin(tk + off), 0.0])
                lab_, _ = body.section([0, 0, 0], e_, [0, 0, 1], ext, 256)
                score = int((lab_ == 9).sum()) * 4 + int((lab_ == 8).sum())
                if best is None or score > best[0]:
                    best = (score, tk + off)
        cuts = {"half": (best[1], "both")}
    else:
        cuts = {"half": (body.split_angle + np.pi / 2, "both")}
    rec["cuts"] = {}
    for name, (thc, mode) in cuts.items():
        e1, e2 = np.array([np.cos(thc), np.sin(thc), 0.0]), np.array([0, 0, 1.0])
        normal = np.cross(e1, e2)
        lab, Pg = body.section([0, 0, 0], e1, e2, ext, SEC_RES)
        sec = f"section_{name}.png"
        save_rgb(os.path.join(tex, sec), body.section_rgb(lab, Pg, seed), gamma=False)
        keep_whole = set()
        if kind == "avocado":                             # the knife goes round the seed: it stays in half_a
            keep_whole = {p for p in pg.meshes if p.startswith("Seed/") or
                          (p == "Endocarp" and body.p["testa_adheres_to"] == "seed")}
        sub = PartGraph.__new__(PartGraph)
        sub.meshes = {p: m for p, m in pg.meshes.items() if m is not None and p not in keep_whole}
        side_pos, side_neg = sub.cut(np.zeros(3), normal)
        if sub.cut_failures:
            print(f"  WARNING {kind}_{seed} {name}: slices not closed: {sub.cut_failures}", flush=True)
        rec.setdefault("cut_failures", {})[name] = sub.cut_failures
        for p in keep_whole:
            side_pos[p] = (pg.meshes[p], np.zeros(len(pg.meshes[p].faces), bool))
        states = [("cutaway", side_neg)] if mode == "keep_neg" else [("half_a", side_pos), ("half_b", side_neg)]
        for sname, pcs in states:
            r = dict(root)
            if kind == "egg":
                r["note"] = "anatomical cutaway (illustration): raw contents shown as if fixed in place"
            else:
                r["integrity"] = "halved"
            if kind == "avocado":
                r["seed"] = "in" if sname == "half_a" else "out"
            pts = {} if kind == "egg" else {k: v for k, v in pg.points.items()}
            mph = write_state(os.path.join(d, f"{sname}.usdc"), pg, iri + f"/{sname}", sname, pcs,
                              materials_fn(kind, body, f"./textures/{sec}"), r, friction, rest, uv_fns=uv_fns,
                              cap_uv_fn=plane_uv([0, 0, 0], e1, e2, ext), physics=(kind != "egg"), points=pts,
                              voids=voids, subsets=subsets_fn(kind, body), point_clip=(normal, sname))
            rec["cuts"][sname] = {"plane_azimuth_deg": float(np.degrees(thc)), "parts": len(pcs),
                                  "mass_g": None if mph is None else mph["mass_kg"] * 1e3}
    B = prior_beliefs(kind, body, mp)
    for prim, kv in (belief_update or {}).items():
        B.setdefault(prim, {}).update(kv)
    write_sdt_layer(os.path.join(d, "sdt.usda"), "./whole.usdc", pg, B)
    T = truth(kind, body)
    T[""].update(truth_extra or {})
    write_truth_layer(os.path.join(d, "truth.usda"), T)
    if observation is not None:
        rec["observation"] = observation
    with open(os.path.join(d, "instance.json"), "w") as fh:
        json.dump(jsonable(rec), fh, indent=1)
    print(f"[{inst_name}] {len(pg.meshes)} part meshes, {len(pg.points)} point sets, mass {mp['mass_kg'] * 1e3:.1f} g, "
          f"cuts {list(rec['cuts'])} ({time.time() - t0:.0f} s) -> {d}", flush=True)
    return d


if __name__ == "__main__":
    seeds = {"egg": 1, "apple": 3, "avocado": 4, "roll": 0, "banana": 0, "carrot": 0,
             "loaf": 0, "tomato": 15, "potato": 0, "onion": 0, "orange": 0,
             "lemon": 0, "cucumber": 0, "croissant": 0, "baguette": 0, "pepper": 0}
    for a in sys.argv[1:]:
        if a.startswith("--seeds"):
            for kv in a.split("=", 1)[1].split(","):
                k, s = kv.split(":")
                seeds[k] = int(s)
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    for kind, s in seeds.items():
        if not only or kind in only:
            build(kind, s)
