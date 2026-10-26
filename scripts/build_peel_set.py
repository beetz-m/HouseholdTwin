"""Build a set of peeling outcomes (food/avocado_peel.py) for one avocado instance: the ideal peel and n outcomes per
category (clean, skin_patches, gouges, mixed) as USD states + ground-truth JSON, listed in <out>/set.json.

Run: ~/Documents/LabAutomationTwin/env_cad/bin/python scripts/build_peel_set.py assets/food/avocado_4 [--n 6] [--out DIR]
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from food import avocado_peel as AP  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("instance_dir")
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--half", default="half_b")
ap.add_argument("--out", default=None)
a = ap.parse_args()
inst = os.path.abspath(a.instance_dir)
out = a.out or os.path.join(inst, "peel_set")
recs = AP.build_set(inst, out, a.n, a.half)
json.dump(dict(instance_dir=inst, half=a.half, samples=recs), open(os.path.join(out, "set.json"), "w"), indent=1)
for r in recs:
    g = r["ground_truth"]
    print(f"  {r['usd']:28s} skin {g['skin_area_mm2']:6.1f} mm2  gouge surface {g['gouge_surface_mm2']:6.1f} mm2  "
          f"lost {g['lost_volume_ml']:.2f} mL")
