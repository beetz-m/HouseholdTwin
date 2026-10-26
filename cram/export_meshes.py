"""Light STL meshes of an avocado instance and its process states, for the CRAM world (semantic_digital_twin / CRAMERA).

Each piece (whole fruit, the two halves, the pitted half, the seed) becomes a few coloured part meshes: skin, the three
flesh bands, the endocarp, the seed coat; decimated to a few thousand triangles each, in the asset frame (metres).
The CRAM side (cram/avocado_world.py) builds one body per part, fixed to the piece's first part.

Run (needs pxr): ~/Documents/LabAutomationTwin/env_cad/bin/python cram/export_meshes.py [assets/food/avocado_4] [ripe]
Writes cram/meshes/<instance>/<piece>/<part>.stl and manifest.json.
"""

import json
import math
import os
import sys

import fast_simplification
import numpy as np
import trimesh
from pxr import Usd, UsdGeom, UsdShade

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from food.avocado import skin_colour  # noqa: E402

inst = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "assets", "food", "avocado_4"))
scenario = sys.argv[2] if len(sys.argv) > 2 else "ripe"
name = os.path.basename(inst)
out = os.path.join(ROOT, "cram", "meshes", name)
rec = json.load(open(os.path.join(inst, "instance.json")))
stage = rec["hidden_params"]["ripeness_stage"]
run = os.path.join(inst, "process", scenario)

PARTS = {                                   # part -> prim subtrees (all active meshes below them)
    "skin": ["/Food/Parts/Exocarp/Cuticle", "/Food/Parts/StemButton"],
    "flesh_outer": ["/Food/Parts/Mesocarp/OuterDarkGreenBand"],
    "flesh_middle": ["/Food/Parts/Mesocarp/MiddlePaleGreenBand"],
    "flesh_inner": ["/Food/Parts/Mesocarp/InnerYellowBand"],
    "endocarp": ["/Food/Parts/Endocarp"],
    "seed_coat": ["/Food/Parts/Seed/SeedCoat/OuterTesta"],
}
PIECES = {                                  # piece -> (USD file, parts)
    "whole": (os.path.join(inst, "whole.usdc"), ["skin"]),
    "half_a": (os.path.join(run, "half_a.usdc"),
               ["skin", "flesh_outer", "flesh_middle", "flesh_inner", "endocarp", "seed_coat"]),
    "half_b": (os.path.join(run, "half_b.usdc"), ["skin", "flesh_outer", "flesh_middle", "flesh_inner", "endocarp"]),
    "half_a_pitted": (os.path.join(run, "half_a_pitted.usdc"),
                      ["skin", "flesh_outer", "flesh_middle", "flesh_inner", "endocarp", "seed_coat"]),
    "seed": (os.path.join(run, "seed.usdc"), ["seed_coat"]),
    "half_b_clean": (os.path.join(inst, "half_b.usdc"),                  # a clean twist: no torn flesh
                     ["skin", "flesh_outer", "flesh_middle", "flesh_inner", "endocarp"]),
    "half_a_pitted_clean": (os.path.join(run, "half_a_pitted.usdc"),      # seed-coat sheets peeled off
                            ["skin", "flesh_outer", "flesh_middle", "flesh_inner", "endocarp"]),
}
MAX_FACES = {"skin": 9000, "seed_coat": 5000}
FIXED_RGB = {"seed_coat": [0.42, 0.22, 0.12]}   # textured in USD (testa image); its mean colour here


def colour(prim):
    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if mat:
        for c in Usd.PrimRange(mat.GetPrim()):
            if c.IsA(UsdShade.Shader):
                v = UsdShade.Shader(c).GetInput("diffuse_reflection_color")
                if v and v.Get() is not None:
                    return [float(x) for x in v.Get()]
    return None


def gather(st, roots):
    V, F, col, n = [], [], None, 0
    for r in roots:
        top = st.GetPrimAtPath(r)
        if not top or not top.IsActive():
            continue
        for p in Usd.PrimRange(top):
            if not p.IsA(UsdGeom.Mesh) or not p.IsActive():
                continue
            m = UsdGeom.Mesh(p)
            P = np.asarray(m.GetPointsAttr().Get(), float)
            f = np.asarray(m.GetFaceVertexIndicesAttr().Get(), np.int64).reshape(-1, 3)
            M = np.asarray(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            P = P @ M[:3, :3] + M[3, :3]
            V.append(P)
            F.append(f + n)
            n += len(P)
            col = col or colour(p)
    if not V:
        return None, None, None
    return np.vstack(V), np.vstack(F), col


manifest = dict(instance=name, scenario=scenario, stage=stage,
                cut_plane_azimuth_deg=rec["cuts"]["half_a"]["plane_azimuth_deg"], pieces={})
az = math.radians(manifest["cut_plane_azimuth_deg"])
manifest["cut_normal"] = [math.sin(az), -math.cos(az), 0.0]       # half_a lies on the +n side
for piece, (path, parts) in PIECES.items():
    st = Usd.Stage.Open(path)
    os.makedirs(os.path.join(out, piece), exist_ok=True)
    rows = []
    for part in parts:
        V, F, col = gather(st, PARTS[part])
        if V is None:
            continue
        if part == "skin":
            col = [float(x) for x in skin_colour(stage)]
        col = FIXED_RGB.get(part, col)
        target = MAX_FACES.get(part, 4000)
        if len(F) > target:
            V, F = fast_simplification.simplify(V.astype(np.float32), F.astype(np.int32), 1 - target / len(F))
        mesh = trimesh.Trimesh(V, F, process=True)
        f = os.path.join(out, piece, f"{part}.stl")
        mesh.export(f)
        rows.append(dict(part=part, file=os.path.relpath(f, out), rgb=col or [0.5, 0.5, 0.5], faces=len(mesh.faces)))
    lo = np.min([trimesh.load(os.path.join(out, r["file"])).bounds[0] for r in rows], 0)
    hi = np.max([trimesh.load(os.path.join(out, r["file"])).bounds[1] for r in rows], 0)
    manifest["pieces"][piece] = dict(parts=rows, bounds=[lo.tolist(), hi.tolist()])
    print(piece, [(r["part"], r["faces"]) for r in rows])
# the lemon the robot uses against browning (wave-3 twin lemon_0): its peel, in its own asset frame
lemon = os.path.join(ROOT, "assets", "food", "lemon_0", "whole.usdc")
if os.path.exists(lemon):
    st = Usd.Stage.Open(lemon)
    V, F, _ = gather(st, ["/Food/Parts/Peel/Flavedo", "/Food/Parts/StemButton"])
    if len(F) > 6000:
        V, F = fast_simplification.simplify(V.astype(np.float32), F.astype(np.int32), 1 - 6000 / len(F))
    os.makedirs(os.path.join(out, "lemon"), exist_ok=True)
    mesh = trimesh.Trimesh(V, F, process=True)
    mesh.export(os.path.join(out, "lemon", "peel.stl"))
    manifest["pieces"]["lemon"] = dict(parts=[dict(part="peel", file="lemon/peel.stl", rgb=[0.93, 0.80, 0.18],
                                                   faces=len(mesh.faces))],
                                       bounds=[mesh.bounds[0].tolist(), mesh.bounds[1].tolist()], own_frame=True)
    print("lemon", len(mesh.faces), np.round(mesh.extents, 3))
json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)
print("wrote", out)
