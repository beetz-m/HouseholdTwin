"""Evaluate the egg bowl monitor (food/egg_bowl_monitor.py) on the rendered bowl set (scripts/render_egg_dataset.py):
shard count, shard IoU against the semantic segmentation (egg_shell_shard), broken-yolk accuracy, yolk index and
thick-albumen height against the set's ground truth (and the Haugh unit), verdict accuracy.

    ~/Documents/LabAutomationTwin/env_cad/bin/python scripts/eval_egg_bowl_monitor.py [seed]
Writes assets/food/egg_<s>/bowl_set/eval/summary.json and panels.png.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from food import egg_bowl_monitor as BM  # noqa: E402


def seg_mask(seg, labels, needle):
    ids = [int(k) for k, v in labels.items() if needle in json.dumps(v)]
    return np.isin(seg, ids)


def main(seed=3):
    global SHELL
    from food.egg import Egg
    SHELL = Egg(seed).p["colour"]
    base = os.path.join(ROOT, "assets", "food", f"egg_{seed}", "bowl_set")
    rd = os.path.join(base, "render")
    out = os.path.join(base, "eval")
    os.makedirs(out, exist_ok=True)
    rows, panels = [], []
    for f in sorted(os.listdir(rd)):
        if not f.endswith(".npz"):
            continue
        d = np.load(os.path.join(rd, f))
        truth = json.loads(str(d["truth"]))
        intr = json.loads(str(d["intrinsics"]))
        labels = json.loads(str(d["labels"]))
        rep = BM.judge(d["rgb"], d["depth"], intr, mass_g=truth["mass_g"], shell=SHELL)   # shell colour: OBSERVED (photo)
        gt_sh = seg_mask(d["seg"], labels, "egg_shell_shard")
        pr_sh = rep.masks.get("shards", np.zeros_like(gt_sh))
        inter, union = (gt_sh & pr_sh).sum(), (gt_sh | pr_sh).sum()
        t_verdict = "yolk broken" if truth["yolk_broken"] else ("fish out shards" if truth["shards"] else "ok")
        rows.append(dict(sample=truth["sample"], shards_true=truth["shards"], shards_found=rep.shards,
                         shard_iou=float(inter / union) if union else None, yolk_broken_true=truth["yolk_broken"],
                         yolk_broken_found=rep.yolk_broken, yi_true=truth["yolk_index"], yi_found=rep.yolk_index,
                         h_true=truth["thick_height_mm"], h_found=rep.albumen_height_mm, hu_true=truth["haugh_unit"],
                         hu_found=rep.haugh_unit, verdict_true=t_verdict, verdict=rep.verdict))
        panels.append((truth["sample"], BM.overlay(d["rgb"], rep), rep.verdict))
    ok = [r for r in rows if r["verdict"] == r["verdict_true"]]
    whole = [r for r in rows if not r["yolk_broken_true"]]
    summ = dict(samples=len(rows), verdict_accuracy=f"{len(ok)}/{len(rows)}",
                shard_count_exact=sum(r["shards_found"] == r["shards_true"] for r in rows),
                yolk_broken_accuracy=sum(r["yolk_broken_found"] == r["yolk_broken_true"] for r in rows),
                yolk_index_mae=float(np.nanmean([abs(r["yi_found"] - r["yi_true"]) for r in whole])) if whole else None,
                albumen_height_observed=f"{sum(np.isfinite(r['h_found']) for r in rows)}/{len(rows)}",
                albumen_height_mae_mm=float(np.nanmean([abs(r["h_found"] - r["h_true"]) for r in rows]))
                if any(np.isfinite(r["h_found"]) for r in rows) else None,
                haugh_unit_mae=float(np.nanmean([abs(r["hu_found"] - r["hu_true"]) for r in rows]))
                if any(np.isfinite(r["hu_found"]) for r in rows) else None, shell=SHELL, rows=rows,
                note="thresholds calibrated on the twin's renders of one instance only (ILLUSTRATIVE)")
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1, default=float)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(panels)
    cols = 4
    fig, ax = plt.subplots((n + cols - 1) // cols, cols, figsize=(16, 3.2 * ((n + cols - 1) // cols)))
    for a in np.ravel(ax):
        a.axis("off")
    for a, (name, img, v) in zip(np.ravel(ax), panels):
        a.imshow(img[img.shape[0] // 6: -img.shape[0] // 6, img.shape[1] // 5: -img.shape[1] // 5])
        a.set_title(f"{name}: {v}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "panels.png"), dpi=90)
    for r in rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print({k: v for k, v in summ.items() if k != "rows"})


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
