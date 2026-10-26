"""Evaluate photo -> node (food/egg_vision.py) on the twin's egg photos (scripts/render_egg_dataset.py --photos):
silhouette IoU against the projected true outline (the egg's cuticle mesh through the render camera), shape index,
asymmetry w/L, shell colour, crack line (all eggs are intact: any crack is a false positive).

    ~/Documents/LabAutomationTwin/env_cad/bin/python scripts/eval_egg_photos.py
Writes renders/egg/photos/eval/summary.json and panels.png. Renders only: no real photo has been evaluated.
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pxr import Gf, Usd, UsdGeom  # noqa: E402

from food import egg_vision as EV  # noqa: E402
from food import params as PR  # noqa: E402
from food.egg import Egg  # noqa: E402

PH = os.path.join(ROOT, "renders", "egg", "photos")
W, H = 1280, 800
F_PX = 50.0 / 36.0 * W                                    # render_egg_dataset.py: 50 mm lens, 36 mm aperture
EYE, TARGET = (0.0, -0.32, 0.05), (0.0, 0.0, 0.022)


def true_silhouette(k):
    """The egg's cuticle mesh placed as in render_egg_dataset.py, projected through the render camera."""
    st = Usd.Stage.Open(os.path.join(ROOT, "assets", "food", f"egg_{k}", "whole.usdc"))
    P = []
    for p in st.Traverse():
        if p.IsA(UsdGeom.Mesh) and "Cuticle" in str(p.GetPath()):
            X = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(p)).T
            v = np.array(UsdGeom.Mesh(p).GetPointsAttr().Get())
            P.append((X @ np.c_[v, np.ones(len(v))].T).T[:, :3])
    P = np.vstack(P)
    p = PR.sample("egg", k)
    B = p["shape_index_B_over_L"] * p["length_L"] * 1e-3
    a = np.deg2rad(90.0 if k % 2 == 0 else -90.0)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Pw = P @ Ry.T + [0, 0, B / 2]
    M = np.array(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*EYE), Gf.Vec3d(*TARGET), Gf.Vec3d(0, 0, 1))).T
    C = (M @ np.c_[Pw, np.ones(len(Pw))].T).T[:, :3]
    uv = np.c_[W / 2 + F_PX * C[:, 0] / -C[:, 2], H / 2 - F_PX * C[:, 1] / -C[:, 2]]
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [cv2.convexHull(uv.astype(np.float32)).astype(np.int32)], 1)
    return m.astype(bool)


def main():
    truth = json.load(open(os.path.join(PH, "truth.json")))
    out = os.path.join(PH, "eval")
    os.makedirs(out, exist_ok=True)
    rows, tiles = [], []
    for name, t in truth.items():
        k = int(name.split("_")[1])
        path = os.path.join(PH, name + ".png")
        est = EV.estimate(path).estimates
        bgr = cv2.imread(path)
        m, _ = EV.segment_egg(bgr)
        gt = true_silhouette(k)
        o = EV.outline(gt)
        si_sil = o["max_width_px"] / o["length_px"]
        rows.append(dict(photo=name, iou=float((m & gt).sum() / (m | gt).sum()),
                         si_true=t["shape_index"], si_true_silhouette=si_sil, si=est["shape_index"].value,
                         w_over_L_true=t["asymmetry_w_mm"] / t["length_mm"], w_over_L=est["asymmetry_w_over_L"].value,
                         colour_true=Egg(k).p["colour"], colour=est["shell_colour"].value,
                         crack_seen=est["crack_line"].value))
        v = bgr.copy()
        for mm, col in ((gt, (0, 200, 0)), (m, (0, 0, 255))):
            cs = cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            cv2.drawContours(v, cs, -1, col, 2)
        cv2.putText(v, f"{name} SI {rows[-1]['si']:.3f} (true {t['shape_index']:.3f}) IoU {rows[-1]['iou']:.3f}",
                    (350, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
        tiles.append(v[200:600, 340:940])
    e_si = np.array([r["si"] - r["si_true"] for r in rows])
    e_w = np.array([r["w_over_L"] - r["w_over_L_true"] for r in rows])
    summ = dict(photos=len(rows), iou_mean=float(np.mean([r["iou"] for r in rows])),
                iou_min=float(np.min([r["iou"] for r in rows])),
                si_mae=float(np.abs(e_si).mean()), si_rms=float(np.sqrt((e_si ** 2).mean())), si_bias=float(e_si.mean()),
                w_over_L_mae=float(np.abs(e_w).mean()), w_over_L_bias=float(e_w.mean()),
                colour_correct=f"{sum(r['colour'] == r['colour_true'] for r in rows)}/{len(rows)}",
                crack_false_positives=f"{sum(bool(r['crack_seen']) for r in rows)}/{len(rows)}", rows=rows,
                note="twin renders only (side view, oak board, white wall); no real photo evaluated")
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1, default=float)
    cv2.imwrite(os.path.join(out, "panels.png"), np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])[::2, ::2])
    for r in rows:
        print({k_: (round(v_, 4) if isinstance(v_, float) else v_) for k_, v_ in r.items()})
    print({k_: v_ for k_, v_ in summ.items() if k_ != "rows"})


if __name__ == "__main__":
    main()
