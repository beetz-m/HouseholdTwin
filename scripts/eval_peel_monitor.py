"""Evaluate the avocado peeling monitor (food/peel_monitor.py) on a rendered synthetic set (scripts/render_peel_dataset.py):
per sample, the monitor's maps against the semantic-segmentation ground truth (skin remaining = exocarp labels,
flesh missing = avocado_flesh_damaged), the measured areas against the ground truth, and the verdict against the
true outcome (from the peeling model: all defects, also those the single view cannot see). RGB-D and RGB-only.

Run: ~/Documents/LabAutomationTwin/env_cad/bin/python scripts/eval_peel_monitor.py assets/food/avocado_4/peel_set
Writes <set>/eval/summary.json, <set>/eval/panels.png, <set>/eval/scatter.png and per-sample overlays.
"""

import json
import math
import os
import sys

import numpy as np
from PIL import Image
from pxr import Usd, UsdGeom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from food import avocado_peel as AP  # noqa: E402
from food import peel_monitor as PM  # noqa: E402

SKIN_LABELS = ("avocado_cuticle", "avocado_epidermis", "avocado_hypodermis", "avocado_skin_parenchyma",
               "avocado_stone_cell_layer")


def label_masks(seg, labels):
    lab = {int(k): v.get("class", "") for k, v in labels.items()}
    skin_ids = [k for k, v in lab.items() if any(s in v for s in SKIN_LABELS)]
    dmg_ids = [k for k, v in lab.items() if "avocado_flesh_damaged" in v]
    obj_ids = [k for k, v in lab.items() if "avocado" in v]
    return np.isin(seg, skin_ids), np.isin(seg, dmg_ids), np.isin(seg, obj_ids)


def flesh_volume_ml(path):
    st = Usd.Stage.Open(path)
    vol = 0.0
    for p in st.Traverse():
        if p.IsA(UsdGeom.Mesh) and "/Mesocarp/" in str(p.GetPath()) and p.IsActive():
            P, F = AP.mesh_arrays(p)
            a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
            vol += abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
    return vol * 1e6


def true_verdict(gt, defects, flesh_ml, cr):
    patches = [math.pi * d["radii"][0] * d["radii"][1] * 1e6 for d in defects if d["kind"] == "skin_patch"]
    depth = max([d["radii"][2] * 1e3 for d in defects if d["kind"] == "gouge"], default=0.0)
    reasons = []
    if gt["skin_area_mm2"] / OUTER_MM2 > cr.max_skin_fraction:
        reasons.append("skin")
    if patches and max(patches) > cr.max_skin_patch_mm2:
        reasons.append("patch")
    if gt["lost_volume_ml"] / flesh_ml > cr.max_lost_fraction:
        reasons.append("lost")
    if depth > cr.max_gouge_depth_mm:
        reasons.append("depth")
    return ("success" if not reasons else "rework"), reasons


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def ratio(hit, ref):
    """precision / recall; undefined (nan) without predictions / without ground truth, so that clean samples do not
    enter the means as 0."""
    n = ref.sum()
    return float(hit.sum() / n) if n else float("nan")


def main():
    S = sys.argv[1]
    meta = json.load(open(os.path.join(S, "set.json")))
    out = os.path.join(S, "eval")
    os.makedirs(out, exist_ok=True)
    global OUTER_MM2
    half = Usd.Stage.Open(os.path.join(meta["instance_dir"], meta["half"] + ".usdc"))
    OUTER_MM2 = float(AP.outer_surface(half)[2].sum() * 1e6)
    flesh_ml = flesh_volume_ml(os.path.join(S, "ideal.usda"))
    cr = PM.Criteria()
    rows, panels = [], []
    for rec in meta["samples"]:
        name = os.path.splitext(rec["usd"])[0]
        f = os.path.join(S, "render", name + ".npz")
        if name == "ideal" or not os.path.exists(f):
            continue
        d = np.load(f)
        intr = json.loads(str(d["intrinsics"]))
        skin_gt, dmg_gt, obj_gt = label_masks(d["seg"], json.loads(str(d["labels"])))
        gt = rec["ground_truth"]
        tv, treasons = true_verdict(gt, rec["outcome"]["defects"], flesh_ml, cr)
        row = dict(sample=name, category=gt["category"], true_verdict=tv, true_reasons=treasons,
                   gt_skin_mm2=gt["skin_area_mm2"], gt_lost_ml=gt["lost_volume_ml"],
                   gt_gouge_surface_mm2=gt["gouge_surface_mm2"],
                   registration_error=json.loads(str(d["registration_error"])))
        for mode in ("rgbd", "rgb"):
            cls, rep, ex = PM.judge(d["rgb"], d["depth"], intr, d["depth_expected"], flesh_ml, OUTER_MM2, cr,
                                    rgb_only=(mode == "rgb"))
            J, JD = ex["judged"], ex["judged_depth"]
            A, _, _ = PM.pixel_areas(PM.back_project(np.where(np.isfinite(d["depth"]), d["depth"], 1.0), intr),
                                      d["depth"], intr)
            A = A * 1e6
            row[mode] = dict(report=PM.report_dict(rep),
                             iou_skin=iou(cls == PM.SKIN, skin_gt & J), iou_missing=iou(cls == PM.MISSING, dmg_gt & JD),
                             vis_gt_skin_mm2=float(A[skin_gt & J].sum()), vis_gt_missing_mm2=float(A[dmg_gt & JD].sum()),
                             precision_skin=ratio((cls == PM.SKIN) & skin_gt, cls == PM.SKIN),
                             recall_skin=ratio((cls == PM.SKIN) & skin_gt & J, skin_gt & J),
                             precision_missing=ratio((cls == PM.MISSING) & dmg_gt, cls == PM.MISSING),
                             recall_missing=ratio((cls == PM.MISSING) & dmg_gt & JD, dmg_gt & JD))
            if mode == "rgbd":
                ov = PM.overlay(d["rgb"], cls)
                Image.fromarray(ov).save(os.path.join(out, f"{name}_overlay.png"))
                gtv = d["rgb"].copy()
                gtv[skin_gt] = (0.55 * gtv[skin_gt] + 0.45 * np.array(PM.CLASS_RGB[PM.SKIN])).astype(np.uint8)
                gtv[dmg_gt] = (0.55 * gtv[dmg_gt] + 0.45 * np.array(PM.CLASS_RGB[PM.MISSING])).astype(np.uint8)
                ys, xs = np.nonzero(obj_gt)
                crop = (max(ys.min() - 40, 0), ys.max() + 40, max(xs.min() - 40, 0), xs.max() + 40)
                panels.append((name, d["rgb"], gtv, ov, rep, tv, crop))
        rows.append(row)
        r = row["rgbd"]
        print(f"  {name:24s} true {tv:8s} | RGB-D {r['report']['verdict']:10s} score {r['report']['score']:5.1f} "
              f"skin {r['report']['skin_area_mm2']:6.1f}/{r['vis_gt_skin_mm2']:6.1f} mm2 IoU {r['iou_skin']:.2f} | "
              f"missing {r['report']['flesh_missing_area_mm2']:6.1f}/{r['vis_gt_missing_mm2']:6.1f} mm2 "
              f"IoU {r['iou_missing']:.2f} | RGB {row['rgb']['report']['verdict']}", flush=True)
    summary = dict(n=len(rows), outer_surface_mm2=OUTER_MM2, flesh_volume_ml=flesh_ml, criteria=cr.__dict__)
    for mode in ("rgbd", "rgb"):
        dec = [r for r in rows if r[mode]["report"]["verdict"] != "not judged"]
        acc = np.mean([(r[mode]["report"]["verdict"] == r["true_verdict"]) for r in dec]) if dec else float("nan")
        tp = sum(r[mode]["report"]["verdict"] == "rework" and r["true_verdict"] == "rework" for r in dec)
        fn = sum(r[mode]["report"]["verdict"] == "success" and r["true_verdict"] == "rework" for r in dec)
        fp = sum(r[mode]["report"]["verdict"] == "rework" and r["true_verdict"] == "success" for r in dec)
        tn = sum(r[mode]["report"]["verdict"] == "success" and r["true_verdict"] == "success" for r in dec)
        nanm = lambda k: float(np.nanmean([r[mode][k] for r in rows if not math.isnan(r[mode][k])]))  # noqa: E731
        summary[mode] = dict(verdict_accuracy=float(acc), confusion=dict(rework_found=tp, rework_missed=fn,
                                                                        false_alarm=fp, success_confirmed=tn),
                             not_judged=len(rows) - len(dec),
                             mean_iou_skin=nanm("iou_skin"), mean_iou_missing=nanm("iou_missing"),
                             mean_recall_skin=nanm("recall_skin"), mean_recall_missing=nanm("recall_missing"),
                             mean_precision_skin=nanm("precision_skin"),
                             mean_precision_missing=nanm("precision_missing"),
                             skin_area_mae_mm2=float(np.mean([abs(r[mode]["report"]["skin_area_mm2"] -
                                                                  r[mode]["vis_gt_skin_mm2"]) for r in rows])),
                             missing_area_mae_mm2=float(np.mean([abs(r[mode]["report"]["flesh_missing_area_mm2"] -
                                                                     r[mode]["vis_gt_missing_mm2"]) for r in rows])))
    json.dump(dict(summary=summary, samples=rows), open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pick = [p for p in panels if p[0].endswith(("clean",))][:1] + [p for p in panels if "skin_patches" in p[0]][:2] + \
        [p for p in panels if "gouges" in p[0]][:2] + [p for p in panels if "mixed" in p[0]][:2]
    fig, axs = plt.subplots(max(len(pick), 1), 3, figsize=(11, 2.6 * max(len(pick), 1)), squeeze=False)
    for i, (name, rgb, gtv, ov, rep, tv, (y0, y1, x0, x1)) in enumerate(pick):
        for j, (img, t) in enumerate(((rgb, f"{name}"), (gtv, f"ground truth ({tv})"),
                                      (ov, f"monitor: {rep.verdict}, score {rep.score:.0f}"))):
            axs[i, j].imshow(img[y0:y1, x0:x1])
            axs[i, j].set_title(t, fontsize=8)
            axs[i, j].axis("off")
    fig.suptitle("Peel monitor (RGB-D): red = skin remaining, orange = flesh missing, green = peeled ok", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "panels.png"), dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for mode, mk in (("rgbd", "o"), ("rgb", "x")):
        ax[0].scatter([r[mode]["vis_gt_skin_mm2"] for r in rows], [r[mode]["report"]["skin_area_mm2"] for r in rows],
                      marker=mk, label=mode.upper())
        ax[1].scatter([r[mode]["vis_gt_missing_mm2"] for r in rows],
                      [r[mode]["report"]["flesh_missing_area_mm2"] for r in rows], marker=mk, label=mode.upper())
    for a_, t in zip(ax, ("skin remaining (mm2, visible)", "flesh missing (mm2, visible)")):
        lim = max(a_.get_xlim()[1], a_.get_ylim()[1])
        a_.plot([0, lim], [0, lim], "k--", lw=0.7)
        a_.set_xlabel("ground truth")
        a_.set_ylabel("monitor")
        a_.set_title(t, fontsize=9)
        a_.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "scatter.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
