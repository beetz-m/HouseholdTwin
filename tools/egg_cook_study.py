"""Boiling study with the egg heat model: doneness timelines vs the published tables, and temperature sections.
Run: ~/Documents/LabAutomationTwin/env_cad/bin/python tools/egg_cook_study.py -> renders/food/egg_cook_study.png"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from food.egg import Egg  # noqa: E402
from food.egg_cook import EggHeat, cold_start, hot_start  # noqa: E402

egg = Egg(4)
H = EggHeat(egg)
Y = H.grp == "yolk"
T = np.arange(3.0, 16.01, 0.5)
cases = {"hot start, room-temp egg (20 C)": (20.0, hot_start(98.0)),
         "hot start, fridge egg (4 C)": (4.0, hot_start(98.0)),
         "cold start, off the heat after the boil": (4.0, cold_start(4.0))}
curves = {}
for name, (T0, water) in cases.items():
    rows = []
    for tm in T:
        _, d = H.run(T0, water, tm * 60)
        rows.append({"t_min": float(tm), **{k: d[k] for k in ("yolk", "f65", "T_centre_max", "white_set_fraction")}})
    curves[name] = rows
fig = plt.figure(figsize=(18, 9))
gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.05])
ax = fig.add_subplot(gs[0, :3])
cl = {"hot start, room-temp egg (20 C)": "#e65100", "hot start, fridge egg (4 C)": "#1565c0",
      "cold start, off the heat after the boil": "#6a1b9a"}
for name, rows in curves.items():
    ax.plot([r["t_min"] for r in rows], [r["f65"] for r in rows], lw=2.2, color=cl[name], label=name)
for y0, y1, lab in ((0, 0.15, "runny"), (0.15, 0.45, "jammy"), (0.45, 0.85, "creamy"), (0.85, 1.02, "firm")):
    ax.axhspan(y0, y1, color="#000", alpha=0.035 if lab in ("jammy", "firm") else 0.0)
    ax.text(16.1, (y0 + y1) / 2, lab, va="center", fontsize=9, color="#444")
ref_room = [(4.5, "runny"), (6, "jammy"), (7, "jammy"), (8.5, "creamy"), (10, "firm"), (11, "firm")]
ref_fridge = [(6, "runny"), (8.5, "creamy"), (11, "firm")]
mid = {"runny": 0.07, "jammy": 0.3, "creamy": 0.65, "firm": 0.93}
ax.plot([t for t, _ in ref_room], [mid[c] for _, c in ref_room], "o", ms=9, mfc="none", mec="#e65100", mew=2,
        label="alphafoodie table (medium eggs, hot start)")
ax.plot([t for t, _ in ref_fridge], [mid[c] for _, c in ref_fridge], "s", ms=9, mfc="none", mec="#1565c0", mew=2,
        label="Lopez-Alt, fridge eggs")
ax.set_xlabel("time in the pot [min]")
ax.set_ylabel("yolk fraction that reached 65 C")
ax.set_xlim(3, 16.6)
ax.set_ylim(0, 1.02)
ax.legend(fontsize=8.5, loc="upper left")
ax.set_title(f"a  Doneness vs time (egg_4, {egg.V_outer * 1087 * 1e3:.0f} g); markers: cooking sources", loc="left")
ax = fig.add_subplot(gs[0, 3:])
for name, rows in curves.items():
    ax.plot([r["t_min"] for r in rows], [r["T_centre_max"] for r in rows], lw=2, color=cl[name])
ax.axhline(71, color="#b71c1c", ls="--", lw=1)
ax.text(3.2, 72, "USDA 71 C (160 F)", color="#b71c1c", fontsize=8.5)
ax.set_xlabel("time [min]")
ax.set_ylabel("max yolk-centre temperature [C]")
ax.set_title("b  Yolk centre (the slowest point)", loc="left")
for i, tm in enumerate((4.5, 6.0, 8.5, 11.0, 13.0)):
    _, d = H.run(20.0, hot_start(98.0), tm * 60)
    ax = fig.add_subplot(gs[1, i])
    F = np.where(H.inside, H.T, np.nan)
    full = np.concatenate([F[::-1], F], 0)
    r = np.concatenate([-H.r[::-1], H.r]) * 1e3
    im = ax.pcolormesh(r, H.z * 1e3, full.T, cmap="inferno", vmin=20, vmax=98, shading="auto")
    yolk = np.concatenate([Y[::-1], Y], 0).astype(float)
    ax.contour(r, H.z * 1e3, yolk.T, levels=[0.5], colors="w", linewidths=0.8)
    setm = np.where(Y, H.Tmax >= 65, False)
    ax.contour(r, H.z * 1e3, np.concatenate([setm[::-1], setm], 0).astype(float).T, levels=[0.5], colors="#00e5ff",
               linewidths=1.0, linestyles="--")
    ax.set_aspect("equal")
    ax.set_title(f"{tm:g} min: {d['yolk']} (f65 {d['f65']:.2f})", fontsize=10)
    ax.set_xticks([])
    if i == 0:
        ax.set_ylabel("z [mm] (blunt end / air cell up)")
    else:
        ax.set_yticks([])
fig.colorbar(im, ax=fig.axes[-5:], shrink=0.8, label="temperature [C]")
fig.suptitle("Boiling an egg: axisymmetric heat conduction through the twin's own tissue map (air cell, yolk, albumen); "
             "white contour = yolk, cyan dashed = yolk already set (>= 65 C)", fontsize=12)
out = os.path.join(ROOT, "renders", "food", "egg_cook_study.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
json.dump(curves, open(os.path.join(ROOT, "test_reports", "egg_cook_study.json"), "w"), indent=1)
for name, rows in curves.items():
    print(name, [(r["t_min"], r["yolk"]) for r in rows if r["t_min"] in (4.5, 6, 7, 8.5, 10, 11, 13)])
print(out)
