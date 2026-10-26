"""Avocado SDT node from a photo: estimate the instance's parameters and beliefs from one image (food/avocado_vision.py),
write an analysis sheet and the observation record, and build the USD part graph with the photo-based belief layer.

Run: ~/Documents/LabAutomationTwin/env_cad/bin/python scripts/avocado_from_image.py IMAGE [--length-mm 95]
         [--name avocado_photo_x] [--bbox x,y,w,h] [--no-build]
     ... --watch DIR    process every new image dropped into DIR (polling; Ctrl-C to stop)
Writes renders/vision/<name>_analysis.png, assets/food/<name>/observation.json and (unless --no-build)
assets/food/<name>/{whole,half_a,half_b}.usdc, sdt.usda (beliefs from the photo), truth.usda, instance.json.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from food import avocado_vision as AV  # noqa: E402
from food.avocado import Avocado, skin_colour  # noqa: E402

OUT_V = os.path.join(ROOT, "renders", "vision")
ASSETS = os.path.join(ROOT, "assets", "food")


def analysis_sheet(obs, dbg, name, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rgb = dbg["bgr"][..., ::-1]
    sh = dbg["shape"]
    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1], height_ratios=[1, 1])
    a0 = fig.add_subplot(gs[:, 0])
    a0.imshow(rgb)
    cnt = dbg["contour"][:, 0, :]
    a0.plot(cnt[:, 0], cnt[:, 1], color="#00e5ff", lw=1.2)
    c, ax = sh["centre"], sh["axis"]
    p0, p1 = c + ax * sh["t_min"], c + ax * sh["t_max"]
    a0.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#ffd400", lw=1.0, ls="--")
    a0.annotate("stem end", p1, color="#ffd400", fontsize=8, xytext=(8, -8), textcoords="offset points")
    if dbg["stem_xy"] is not None:
        a0.plot(*dbg["stem_xy"], "o", mfc="none", mec="#ff3b30", ms=12, mew=1.5)
    a0.set_title(f"{name}: segmentation, axis, stem button", fontsize=9)
    a0.axis("off")
    a1 = fig.add_subplot(gs[0, 1])
    s = np.linspace(0, 1, len(sh["width_px"]))
    a1.plot(s, sh["width_px"] / sh["width_px"].max(), color="k", lw=1.2, label="photo silhouette")
    sm, ym = AV.model_profile(obs.estimates["neck"].value, len(s))
    a1.plot(sm, ym, color="#1f77b4", lw=1.2, ls="--", label=f"twin meridian, neck {obs.estimates['neck'].value:.2f}")
    a1.set_xlabel("blossom end -> stem end", fontsize=8)
    a1.set_ylabel("width / max", fontsize=8)
    a1.legend(fontsize=7)
    a1.set_title(f"profile (L/D {obs.estimates['aspect_L_over_D'].value:.2f})", fontsize=9)
    a2 = fig.add_subplot(gs[1, 1])
    sg, post = dbg["ripeness"]
    a2.fill_between(sg, post / post.max(), color="#6b8e23", alpha=0.35)
    for st in range(1, 6):
        a2.axvspan(st - 0.5, st + 0.5, ymax=0.06, color=np.clip(skin_colour(st) / 0.6, 0, 1))
    r = obs.estimates["ripeness_stage"]
    a2.axvline(r.value, color="k", lw=1)
    a2.set_xlim(0.8, 5.2)
    a2.set_xlabel("ripeness stage (Avocados Australia 1..5)", fontsize=8)
    a2.set_title(f"ripeness posterior: {r.value:.2f} +- {r.sd:.2f}", fontsize=9)
    a3 = fig.add_subplot(gs[:, 2])
    a3.axis("off")
    rows = []
    for k, e in obs.estimates.items():
        v = f"{e.value}" + (f" +- {e.sd:g}" if isinstance(e.sd, (int, float)) and e.sd else "")
        rows.append(f"{k}: {v} {e.unit if e.unit not in ('-', '1') else ''}   [{e.status}]")
    txt = "\n".join(rows) + "\n\nflags:\n" + "\n".join("- " + f for f in obs.flags)
    a3.text(0, 1, txt, va="top", fontsize=7.2, family="monospace", wrap=True)
    a3.set_title("estimates (status)", fontsize=9, loc="left")
    fig.tight_layout()
    os.makedirs(OUT_V, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def process(image, length_mm=None, name=None, bbox=None, build_asset=True):
    obs, dbg = AV.estimate(image, length_mm=length_mm, bbox=bbox)
    name = name or f"avocado_photo_{obs.image_sha1[:8]}"
    sheet = os.path.join(OUT_V, f"{name}_analysis.png")
    analysis_sheet(obs, dbg, name, sheet)
    d = os.path.join(ASSETS, name)
    os.makedirs(d, exist_ok=True)
    rec = obs.as_dict()
    rec["instance_overrides"] = AV.instance_overrides(obs)
    json.dump(rec, open(os.path.join(d, "observation.json"), "w"), indent=1)
    print(f"[{name}] estimates:")
    for k, e in obs.estimates.items():
        print(f"   {k:20s} {str(e.value):>28s} {'' if e.sd is None else '+- ' + format(e.sd, 'g'):>10s} {e.unit:6s} "
              f"{e.status}")
    for f in obs.flags:
        print("   flag:", f)
    print(f"   analysis sheet: {sheet}")
    # probabilistic node: register prior -> Bayesian update with the photo's observations (food/avocado_belief.py)
    from food import avocado_belief as AB
    import avocado_observe as AO
    belief = AB.AvocadoBelief.prior(4000)
    for o in AO.photo_observations(image, 0.0)[0]:
        belief.observe(o, np.random.default_rng(0))
    if length_mm is not None:
        belief.observe(AB.Observation("length_mm", dict(value=length_mm, sd=1.0), 0.0, "user measurement"),
                       np.random.default_rng(0))
    S = belief.summary()
    for k in ("length_L", "diameter_D", "neck", "ripeness_stage"):     # geometry and state: posterior means
        rec["instance_overrides"][k] = float(S[k]["mean"])
    print(f"   posterior: stage {S['ripeness_stage']['mean']:.2f} +- {S['ripeness_stage']['sd']:.2f}, "
          f"L {S['length_L']['mean']:.1f} +- {S['length_L']['sd']:.1f} mm, mass {S['mass_g']['mean']:.0f} +- "
          f"{S['mass_g']['sd']:.0f} g")
    if build_asset:
        import build_food_assets as BF
        seed = int(obs.image_sha1[:6], 16) % 100000          # reproducible draw for what the photo cannot see
        body = Avocado(seed, overrides=rec["instance_overrides"])
        BF.build("avocado", seed, body=body, name=name, belief_update=AV.belief_update(obs, name),
                 truth_extra={"truth:source": "instance parameters set from photo estimates (" + obs.image_sha1 +
                              "); the real fruit's hidden values are unknown, interior drawn from the register"},
                 observation=rec)
        AO.write_node(d, belief)                              # sdt_belief.usda + belief/belief.npz
    return name, obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--length-mm", type=float, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--bbox", default=None, help="x,y,w,h of the fruit if the automatic segmentation fails")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--watch", default=None, help="directory to watch for new images")
    a = ap.parse_args()
    bbox = [int(v) for v in a.bbox.split(",")] if a.bbox else None
    if a.watch:
        seen = set()
        print(f"watching {a.watch} for images (Ctrl-C to stop)")
        while True:
            for f in sorted(os.listdir(a.watch)):
                p = os.path.join(a.watch, f)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) and p not in seen:
                    seen.add(p)
                    process(p, build_asset=not a.no_build)
            time.sleep(2)
    process(a.image, a.length_mm, a.name, bbox, not a.no_build)


if __name__ == "__main__":
    main()
