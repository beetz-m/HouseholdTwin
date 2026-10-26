"""Avocado squeeze study figure: FEM force-displacement per ripeness stage, stiffness vs stage, a belief update from
one squeeze of a hidden stage-4.2 fruit, and the safe squeeze force against bruising.
Run: ~/Documents/LabAutomationTwin/env_cad/bin/python tools/plot_avocado_squeeze.py -> renders/food/avocado_squeeze.png"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from food import params as PR  # noqa: E402
from food.avocado import penetrometer_force  # noqa: E402
from food.sdt import AvocadoBelief, squeeze_table_model  # noqa: E402

P = os.path.join(ROOT, "test_reports", "avocado_squeeze.json")
d = json.load(open(P))
lanes = sorted(d["lanes"], key=lambda l: l["stage"])
kmod = squeeze_table_model(P)
fig, ax = plt.subplots(1, 4, figsize=(21, 5.4))
cm = plt.get_cmap("viridis")
for i, l in enumerate(lanes):
    c = l["curve"]
    ax[0].plot([p["d_fruit_mm"] for p in c], [p["F"] for p in c], "o-", color=cm(i / (len(lanes) - 1)),
               label=f"stage {l['stage']:g} (E {l['E_MPa']:.2f} MPa)")
ax[0].set_xlabel("plate travel = fruit compression [mm]")
ax[0].set_ylabel("squeeze force [N]")
ax[0].set_title("a  FEM squeeze (PhysX volume deformable)", loc="left")
ax[0].legend(fontsize=8)
st = np.linspace(1, 5, 50)
for F, ls in ((2.0, ":"), (4.0, "-"), (8.0, "--")):
    ax[1].plot(st, kmod(st, F) / 1e3, ls, color="#2e7d32", label=f"at {F:g} N")
ax[1].set_xlabel("ripeness stage (Avocados Australia 1 hard .. 5 ripe)")
ax[1].set_ylabel("secant stiffness [N/mm]")
ax[1].set_title("b  Stiffness model for the belief", loc="left")
ax[1].legend(fontsize=8)
# belief update: hidden stage 4.2, one squeeze at 4 N with 8 % measurement noise
rng = np.random.default_rng(4)
true = 4.2
k_obs = kmod(true, 4.0) * (1 + rng.normal(0, 0.08))
B = AvocadoBelief(rng, stiffness_model=lambda s, F: kmod(s, F), sigma_rel=0.10)
prior = B.stage.copy()
B.observe_stiffness(k_obs, 4.0)
s1 = B.summary()
B2 = AvocadoBelief(np.random.default_rng(5), stiffness_model=lambda s, F: kmod(s, F), sigma_rel=0.10)
B2.observe_stiffness(k_obs, 4.0)
B2.observe_stiffness(kmod(true, 2.0) * (1 + rng.normal(0, 0.08)), 2.0)
s2 = B2.summary()
bins = np.linspace(1, 5, 33)
ax[2].hist(prior, bins, weights=np.full(len(prior), 1 / len(prior)), color="#bdbdbd", label="prior (flat)")
ax[2].hist(B.stage, bins, weights=B.w, color="#43a047", alpha=0.8, label=f"after 1 squeeze: {s1['mean']:.2f} +- {s1['sd']:.2f}")
ax[2].hist(B2.stage, bins, weights=B2.w, histtype="step", lw=2, color="#1b5e20",
           label=f"after 2 squeezes: {s2['mean']:.2f} +- {s2['sd']:.2f}")
ax[2].axvline(true, color="#b71c1c", ls="--")
ax[2].text(true + 0.05, ax[2].get_ylim()[1] * 0.9, "hidden 4.2", color="#b71c1c", fontsize=9)
ax[2].set_xlabel("ripeness stage")
ax[2].set_title(f"c  Belief: P(ready, stage >= 4) = {s2['p_ready']:.2f}", loc="left")
ax[2].legend(fontsize=8)
# safe squeeze force vs bruising (contact pressure vs ratio x penetrometer pressure)
ratio = PR.AVOCADO["bruise_force_ratio"][1]
A_pen = np.pi * (11.1e-3 / 2) ** 2
R = d["D_mm"] / 2 * 1e-3
Fsafe = []
for l in lanes:
    c = l["curve"]
    lim = ratio * penetrometer_force(l["stage"]) / A_pen
    ok = 0.0
    for p in c:
        delta = max(p["d_fruit_mm"], 1e-6) * 1e-3 / 2
        area = np.pi * R * delta                       # Hertz contact area of a flat plate on a sphere
        if p["F"] / area < lim:
            ok = p["F"]
    Fsafe.append(ok)
ax[3].bar([f"{l['stage']:g}" for l in lanes], Fsafe, color=[cm(i / (len(lanes) - 1)) for i in range(len(lanes))])
ax[3].axhline(10, color="#b71c1c", ls="--")
ax[3].text(0, 10.2, "10 N thumb press bruises firm-ripe fruit (hia2018)", color="#b71c1c", fontsize=8.5)
ax[3].set_ylim(0, 11)
ax[3].set_xlabel("ripeness stage")
ax[3].set_ylabel("largest plate force without bruising [N] (tested up to 8 N)")
ax[3].set_title("d  Safe squeeze force", loc="left")
fig.suptitle("Avocado ripeness by squeezing: Isaac FEM (1000 Hz, 120 iterations) -> stiffness model -> robot belief; "
             "flesh modulus 2.29 -> 0.16 MPa (farinas2021), uniform body (seed/skin not separate)", fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = os.path.join(ROOT, "renders", "food", "avocado_squeeze.png")
fig.savefig(out, dpi=130)
print(out, "k_obs", round(k_obs / 1e3, 2), "N/mm", s1, s2, "Fsafe", Fsafe)
