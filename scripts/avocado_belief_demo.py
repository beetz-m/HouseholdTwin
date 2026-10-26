"""Validation of the probabilistic avocado node (food/avocado_belief.py): a known instance (hidden truth from its
truth layer) ripens over four days while the filter sees simulated, noisy observations - a photo (shape, skin
colour, stem button), a weighing, a second photo two days later, a gripper squeeze, and finally the cut (seed
length, bruise check). Posteriors after every step are compared with the truth.

Run: ~/Documents/LabAutomationTwin/env_cad/bin/python scripts/avocado_belief_demo.py [assets/food/avocado_4]
Writes <instance>/belief/{belief.npz, sdt_belief.usda, timeline.json, timeline.png, posteriors.png}.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from food import avocado_belief as AB  # noqa: E402

inst = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "assets", "food", "avocado_4"))
H = json.load(open(os.path.join(inst, "instance.json")))
truth = H["hidden_params"]
mass_true = H["whole"]["mass_g"]
out = os.path.join(inst, "belief")
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(3)
b = AB.AvocadoBelief.prior(4000, seed=1)
stage0, rate = truth["ripeness_stage"], truth["ripening_rate"]


def stage_at(h):
    return min(5.0, stage0 + rate * h / 24.0)


def colour_obs(stage):
    t = (stage - 1) / 4
    return dict(L=40 - 13 * t + rng.normal(0, 4), C=max(0.5, 15 - 12 * t + rng.normal(0, 2)),
                h=115.0 if stage < 2.5 else 40.0)


TRACK = ["ripeness_stage", "mass_g", "length_L", "diameter_D", "flesh_E_MPa", "seed_length", "ripening_rate"]
timeline = [dict(step="prior", time_h=0.0, summary={k: b.summary()[k] for k in TRACK})]


def step(label, obs=None, hours=0.0):
    if hours:
        b.predict(hours, rng)
    if obs is not None:
        b.observe(obs, rng)
    S = b.summary()
    timeline.append(dict(step=label, time_h=b.time_h, summary={k: S[k] for k in TRACK}, ess=b.ess()))
    print(f"  {label:34s} t={b.time_h:5.0f} h  stage {S['ripeness_stage']['mean']:.2f}+-{S['ripeness_stage']['sd']:.2f} "
          f"(true {stage_at(b.time_h):.2f})  mass {S['mass_g']['mean']:.0f}+-{S['mass_g']['sd']:.0f} g (true "
          f"{mass_true:.0f})  L {S['length_L']['mean']:.1f}+-{S['length_L']['sd']:.1f} (true {truth['length_L']:.1f})"
          f"  ESS {b.ess():.0f}")


O = AB.Observation
step("photo: shape", O("photo_shape", dict(aspect=truth["length_L"] / truth["diameter_D"] + rng.normal(0, 0.03),
                                           aspect_sd=0.05, neck=truth["neck"] + rng.normal(0, 0.03), neck_sd=0.05),
                       0.0, "simulated photo"))
step("photo: skin colour", O("skin_colour", colour_obs(stage_at(0)), 0.0, "simulated photo"))
step("photo: stem button", O("stem_button", dict(present=True), 0.0, "simulated photo"))
step("scale: mass", O("mass_g", dict(value=mass_true + rng.normal(0, 2), sd=2.0), 0.0, "simulated kitchen scale"))
step("photo after 48 h: skin colour", O("skin_colour", colour_obs(stage_at(48)), 48.0, "simulated photo"), hours=48)
k_true = float(np.interp(stage_at(72), AB.SQUEEZE_STAGE, AB.SQUEEZE_K))
step("squeeze at 72 h (4 N)", O("squeeze", dict(k=k_true * (1 + rng.normal(0, 0.1)), force_N=4.0), 72.0,
                                "simulated gripper squeeze"), hours=24)
step("cut at 96 h: seed length", O("seed_length_mm", dict(value=truth["seed_length"] + rng.normal(0, 1.0)), 96.0,
                                   "simulated measurement after halving"), hours=24)
b.save(os.path.join(out, "belief.npz"))
AB.write_belief_layer(os.path.join(out, "sdt_belief.usda"), "../whole.usdc", b, "./belief.npz")
truth_track = dict(ripeness_stage=[stage_at(t["time_h"]) for t in timeline], mass_g=mass_true,
                   length_L=truth["length_L"], diameter_D=truth["diameter_D"],
                   flesh_E_MPa=[float(AB.flesh_E_MPa(stage_at(t["time_h"]))) for t in timeline],
                   seed_length=truth["seed_length"], ripening_rate=rate)
json.dump(dict(timeline=timeline, truth=truth_track), open(os.path.join(out, "timeline.json"), "w"), indent=1)
# figures
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, axs = plt.subplots(2, 3, figsize=(13, 6.5))
labels = [t["step"] for t in timeline]
xs = np.arange(len(timeline))
for ax, k in zip(axs.ravel(), ["ripeness_stage", "mass_g", "length_L", "flesh_E_MPa", "seed_length", "ripening_rate"]):
    m = [t["summary"][k]["mean"] for t in timeline]
    lo = [t["summary"][k]["p05"] for t in timeline]
    hi = [t["summary"][k]["p95"] for t in timeline]
    ax.fill_between(xs, lo, hi, color="#6b8e23", alpha=0.3, label="5-95 %")
    ax.plot(xs, m, color="#2f4f2f", marker="o", ms=3, label="posterior mean")
    tv = truth_track[k]
    ax.plot(xs, tv if isinstance(tv, list) else [tv] * len(xs), "r--", lw=1, label="truth")
    ax.set_title(f"{k} ({AB.UNITS[k]})", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=6)
axs[0, 0].legend(fontsize=7)
fig.suptitle(f"Bayesian filtering of the avocado node ({os.path.basename(inst)}): prior -> observations over 4 days",
             fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(out, "timeline.png"), dpi=130)
plt.close(fig)
prior = AB.AvocadoBelief.prior(4000, seed=1).summary()
post = b.summary()
fig, axs = plt.subplots(2, 4, figsize=(14, 5.5))
for ax, k in zip(axs.ravel(), ["ripeness_stage", "mass_g", "length_L", "diameter_D", "flesh_E_MPa", "seed_length",
                               "squeeze_k_N_per_mm", "bruised"]):
    for S, c, lab in ((prior, "#999999", "prior"), (post, "#2f6f2f", "posterior")):
        e = np.array(S[k]["hist_edges"])
        ax.stairs(S[k]["hist_p"], e, fill=True, alpha=0.45, color=c, label=lab)
    ax.set_title(f"{k} ({AB.UNITS[k]})", fontsize=9)
axs[0, 0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(out, "posteriors.png"), dpi=130)
plt.close(fig)
print("wrote", out)
