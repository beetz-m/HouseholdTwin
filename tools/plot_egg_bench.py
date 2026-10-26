"""Figure for the Isaac egg bench (test_reports/egg_bench.json) -> renders/food/egg_bench.png."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(ROOT, "test_reports", "egg_bench.json")))
COL = {"intact": "#2e7d32", "microcracked": "#f9a825", "cracked": "#e65100", "shattered": "#b71c1c"}
fig = plt.figure(figsize=(17, 6.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.9, 1.15])
# (a) grip sweep
ax = fig.add_subplot(gs[0])
G = [g for g in R["grasp"] if g["exp"] == "G"]
F = np.array([g["F_cmd"] for g in G])
ps = np.array([g["belief"]["slip"] for g in G])
pr = np.array([g["belief"]["rupture"] for g in G])
ax.semilogx(F, ps, "o-", color="#1565c0", label="belief P(slip) at 2 m/s$^2$")
ax.semilogx(F, pr, "s-", color="#c62828", label="belief P(rupture)")
adm = [g["F_cmd"] for g in G if g["H8_admissible"]]
ax.axvspan(min(adm) * 0.92, max(adm) * 1.08, color="#c8e6c9", alpha=0.6, label="H8 admissible (eps 0.05, gamma 0.8)")
for g in G:
    out = "slipped" if g["slipped"] else g["shell"]
    c = "#1565c0" if g["slipped"] else COL[g["shell"]]
    ax.plot(g["F_cmd"], 1.08, "v", color=c, ms=11)
    ax.text(g["F_cmd"], 1.14, out, rotation=60, fontsize=8, ha="left", va="bottom")
ax.axvline(G[0]["F_crack_truth"], color="#e65100", ls="--", lw=1)
ax.text(G[0]["F_crack_truth"] * 1.03, 0.55, f"hidden crack force\n{G[0]['F_crack_truth']:.0f} N (egg_{R['egg']})",
        fontsize=8, color="#e65100")
ax.set_ylim(-0.03, 1.45)
ax.set_xlabel("commanded pad force per finger [N] (measured = commanded within 0.2 N)")
ax.set_ylabel("probability")
ax.set_title("a  Grip sweep: belief (prior) vs simulated outcome (markers)", loc="left", fontsize=11)
ax.legend(fontsize=8, loc="center left")
# (b) instances at 30 N
ax = fig.add_subplot(gs[1])
V = [g for g in R["grasp"] if g["exp"] == "V"]
for k, g in enumerate(V):
    ax.barh(k, g["F_crack_truth"], color=COL[g["shell"]])
    ax.text(g["F_crack_truth"] + 0.6, k, g["shell"], va="center", fontsize=8)
ax.axvline(30, color="k", lw=1.2)
ax.text(30.4, len(V) - 0.4, "grip 30 N", fontsize=8)
ax.set_yticks(range(len(V)), [f"egg_{g['egg']}" for g in V])
ax.set_xlabel("hidden crack force at the equator [N]")
ax.set_title(f"b  8 egg instances at 30 N (prior P(rupture) {V[0]['belief']['rupture']:.2f})", loc="left",
             fontsize=11)
# (c) drops
ax = fig.add_subplot(gs[2])
S = ["granite", "oak_board", "silicone_mat", "towel"]
H = sorted({d["h_cm"] for d in R["drop"]})
for d in R["drop"]:
    i, j = H.index(d["h_cm"]), S.index(d["surface"])
    ax.add_patch(plt.Rectangle((i - 0.45, j - 0.4), 0.9, 0.8, color=COL[d["shell"]], alpha=0.85))
    ax.text(i, j + 0.12, d["shell"], ha="center", fontsize=7.5, color="w", weight="bold")
    ax.text(i, j - 0.18, f"{d['F_peak']:.0f} N, v {d['v_n']:.2f}", ha="center", fontsize=7, color="w")
for j, s in enumerate(S):
    hc = np.median([d["h_crit_cm_at_this_polar"] for d in R["drop"] if d["surface"] == s])
    ax.text(len(H) - 0.4, j, f"h_crit~{hc:.1f} cm", va="center", fontsize=8)
ax.set_xlim(-0.6, len(H) + 0.6)
ax.set_ylim(-0.6, len(S) - 0.4)
ax.set_xticks(range(len(H)), [f"{h:g}" for h in H])
ax.set_yticks(range(len(S)), S)
ax.set_xlabel("release height [cm] (PhysX impact speed v [m/s] -> peak force)")
ax.set_title("c  Drops (random orientation) onto kitchen surfaces", loc="left", fontsize=11)
fig.suptitle("Egg manipulation bench in Isaac Sim (PhysX 240 Hz contacts) + hidden crack model + robot belief "
             "(AICOR H8); impact model calibrated to Cohen et al. 2025 drop data", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = os.path.join(ROOT, "renders", "food", "egg_bench.png")
fig.savefig(out, dpi=140)
print(out)
