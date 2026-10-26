"""Which surface, how hard? Monte-Carlo study of the crack-and-open model over the egg population of the register.

For each strike tool and strike speed: outcome distribution (no crack / check / cracked open / shattered) and the
probability that a fried-egg task succeeds (yolk whole, no shell in the pan) -- directly, and with the recovery
humans use (crack into a cup first, fish shards out with a half shell, divert a broken-yolk egg to scrambling).
The robot's corridor [v-, v+] is the Morgenstern crack window CC2 made numeric.
Run: ~/Documents/LabAutomationTwin/env_cad/bin/python tools/egg_crack_study.py  -> renders/food/egg_crack_study.png
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from food import egg_crack as EC  # noqa: E402
from food.dynamics import EggTruth  # noqa: E402
from food.egg import Egg  # noqa: E402

N = 400
TOOLS = {"flat counter (granite)": {"kind": "flat", "surface": "granite"},
         "thin bowl rim (r 1.5 mm)": {"kind": "rim", "surface": "bowl_rim_ceramic", "edge_radius": 1.5e-3},
         "thick bowl rim (r 4 mm)": {"kind": "rim", "surface": "bowl_rim_ceramic", "edge_radius": 4.0e-3},
         "another egg": {"kind": "egg"}}
V = np.round(np.linspace(0.05, 1.5, 30), 3)
rng = np.random.default_rng(7)
eggs = [Egg(5000 + i) for i in range(N)]
masses = [e.V_outer * 1087 for e in eggs]

res = {}
for tname, tool in TOOLS.items():
    rows = []
    for v in V:
        c = {"intact": 0, "microcracked": 0, "check": 0, "cracked_open": 0, "shattered": 0}
        ok_direct = ok_recover = 0
        for i, e in enumerate(eggs):
            t = EggTruth(e, masses[i])
            other = EggTruth(eggs[(i + 1) % N], masses[(i + 1) % N]) if tool["kind"] == "egg" else None
            r = EC.strike(t, float(v), tool, rng, other=other)
            c[r.state] += 1
            target = t if r.which == "this" else other
            if r.state in ("check", "cracked_open"):
                opened, yolk, shards = EC.open_up(target, rng, age_days=e.p["age_days"])
                ok_direct += int(opened and yolk == "whole" and shards == 0)
                left = EC.fish_out(shards, rng)
                ok_recover += int(opened and yolk == "whole" and left == 0)
        rows.append({"v": float(v), **{k: c[k] / N for k in c}, "p_success": ok_direct / N,
                     "p_success_recovery": ok_recover / N})
    res[tname] = rows

fig, axes = plt.subplots(2, 4, figsize=(19, 8.4), sharex=True)
cols = {"intact": "#9e9e9e", "microcracked": "#f9a825", "check": "#7cb342", "cracked_open": "#2e7d32",
        "shattered": "#b71c1c"}
summary = {}
for j, (tname, rows) in enumerate(res.items()):
    ax = axes[0, j]
    v = [r["v"] for r in rows]
    base = np.zeros(len(v))
    for k in ("intact", "microcracked", "check", "cracked_open", "shattered"):
        y = np.array([r[k] for r in rows])
        ax.fill_between(v, base, base + y, color=cols[k], alpha=0.9, label=k.replace("_", " "), lw=0)
        base += y
    ax.set_title(tname, fontsize=11)
    ax.set_ylim(0, 1)
    if j == 0:
        ax.set_ylabel("outcome share")
        ax.legend(fontsize=8, loc="center right")
    ax = axes[1, j]
    ps = np.array([r["p_success"] for r in rows])
    pr = np.array([r["p_success_recovery"] for r in rows])
    ax.plot(v, ps, color="#1565c0", lw=2, label="direct (yolk whole, no shell)")
    ax.plot(v, pr, color="#2e7d32", lw=2, ls="--", label="with half-shell fish-out")
    i = int(np.argmax(pr))
    ok = [vv for vv, p in zip(v, pr) if p >= 0.9 * pr[i]]
    ax.axvspan(min(ok), max(ok), color="#c8e6c9", alpha=0.6)
    ax.annotate(f"best {v[i]:.2f} m/s: {pr[i]:.0%}", (v[i], pr[i]), xytext=(0.6, 0.12), textcoords="axes fraction",
                fontsize=9, arrowprops=dict(arrowstyle="-", color="#555"))
    summary[tname] = {"v_best": v[i], "p_best_recovery": float(pr[i]), "p_best_direct": float(ps[int(np.argmax(ps))]),
                      "window_90pct": [min(ok), max(ok)]}
    ax.set_ylim(0, 1)
    ax.set_xlabel("strike speed [m/s]")
    if j == 0:
        ax.set_ylabel("P(fried-egg ready)")
        ax.legend(fontsize=8, loc="upper right")
fig.suptitle(f"Cracking an egg: outcomes over {N} register-sampled eggs (top) and task success (bottom); green band "
             "= speeds within 90 % of the best -- the crack window [v-, v+]. Strike model ILLUSTRATIVE (see register)",
             fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = os.path.join(ROOT, "renders", "food", "egg_crack_study.png")
fig.savefig(out, dpi=130)
json.dump({"summary": summary, "curves": res}, open(os.path.join(ROOT, "test_reports", "egg_crack_study.json"), "w"),
          indent=1)
for k, s in summary.items():
    print(f"{k:28s} best v {s['v_best']:.2f} m/s  P direct {s['p_best_direct']:.2f}  P with recovery "
          f"{s['p_best_recovery']:.2f}  window {s['window_90pct']}")
print(out)
