"""Run the avocado peeling monitor (food/peel_monitor.py) on one view and write the result into the twin.

Input: an .npz with rgb (H,W,3 uint8), depth (H,W float32, distance to the image plane in m), depth_expected (the twin's
ideal peel of this instance from the same camera at the believed pose), intrinsics (JSON: width, height, focal_mm,
h_aperture_mm, cam_to_world) - as written by scripts/render_peel_dataset.py, or assembled from a real RGB-D camera.

Run: ~/Documents/LabAutomationTwin/env_cad/bin/python scripts/peel_monitor_run.py VIEW.npz --instance assets/food/avocado_4
         [--state assets/food/avocado_4/peel_set/peel_006_skin_patches.usda] [--rgb-only] [--out DIR]
Writes <out>/<view>_overlay.png, <out>/<view>_report.json and, with --state, <out>/<view>_sdt.usda (belief layer
over the peeled-half state: verdict, score, measures, reasons).
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from pxr import Usd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from food import avocado_peel as AP  # noqa: E402
from food import peel_monitor as PM  # noqa: E402
from food.usd_partgraph import write_sdt_layer  # noqa: E402
from eval_peel_monitor import flesh_volume_ml  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("view")
ap.add_argument("--instance", required=True)
ap.add_argument("--half", default="half_b")
ap.add_argument("--state", default=None)
ap.add_argument("--rgb-only", action="store_true")
ap.add_argument("--out", default=None)
a = ap.parse_args()
d = np.load(a.view)
intr = json.loads(str(d["intrinsics"]))
half = Usd.Stage.Open(os.path.join(a.instance, a.half + ".usdc"))
outer = float(AP.outer_surface(half)[2].sum() * 1e6)
ideal = os.path.join(a.instance, "peel_set", "ideal.usda")
flesh = flesh_volume_ml(ideal if os.path.exists(ideal) else os.path.join(a.instance, a.half + ".usdc"))
cls, rep, _ = PM.judge(d["rgb"], d["depth"], intr, d["depth_expected"], flesh, outer, rgb_only=a.rgb_only)
out = a.out or os.path.dirname(os.path.abspath(a.view))
os.makedirs(out, exist_ok=True)
name = os.path.splitext(os.path.basename(a.view))[0] + ("_rgb" if a.rgb_only else "")
Image.fromarray(PM.overlay(d["rgb"], cls)).save(os.path.join(out, f"{name}_overlay.png"))
json.dump(PM.report_dict(rep), open(os.path.join(out, f"{name}_report.json"), "w"), indent=1)
if a.state:
    write_sdt_layer(os.path.join(out, f"{name}_sdt.usda"), os.path.relpath(os.path.abspath(a.state), out), None,
                    PM.sdt_update(rep))
print(f"{name}: {rep.verdict} (score {rep.score:.0f}); skin {rep.skin_area_mm2:.0f} mm2 in {rep.n_skin_patches} "
      f"patch(es), flesh missing {rep.flesh_missing_area_mm2:.0f} mm2 / {rep.lost_volume_ml:.2f} mL, deepest "
      f"{rep.deepest_gouge_mm:.1f} mm, coverage {rep.coverage:.0%}; " + ("; ".join(rep.reasons) or "no defects"))
