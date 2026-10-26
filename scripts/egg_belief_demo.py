"""Validation of the egg's probabilistic SDT node (food/egg_belief.py) on an instance with known truth.

The hidden egg is a register draw (the instance's seed) aged at its storage temperature by food/egg_lifetime.py; the
robot observes it only through the measurement models (noise from the register's sensor_* entries, sampled here):
  day 0  fetched from the fridge: storage known, best-before on the carton, weighed, candled, float test
  day 7  (kept in the fridge) float test again; picked up at the twin's best grip force (H8 admissibility checked)
  day 7  struck on a bowl rim at the speed the belief recommends; opened: thick-albumen height and yolk index
After every step: where the truth falls in the belief (inside the 90 % band?) for the decision variables.

    ~/Documents/LabAutomationTwin/env_cad/bin/python scripts/egg_belief_demo.py [seed]
Writes assets/food/egg_<seed>/sdt_belief.usda, belief/belief.npz, belief/timeline.png, test_reports/egg_belief.json.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from food import egg_crack as EC  # noqa: E402
from food import egg_lifetime as EL  # noqa: E402
from food import params as PR  # noqa: E402
from food.dynamics import EggTruth  # noqa: E402
from food.egg import Egg  # noqa: E402
from food.egg_belief import EggBelief, Observation, write_belief_layer  # noqa: E402

E = PR.EGG
TRACK = ["age_days", "mass_g", "air_cell_mm", "haugh_unit", "yolk_index", "albumen_pH", "F_crack_equator_N",
         "float_tilt_deg"]
RIM = {"kind": "rim", "surface": "bowl_rim_ceramic", "edge_radius": 2e-3}


def truth_values(clock, egg_p):
    s = clock.summary()
    sh = egg_p["shell_thickness"]
    Fc = egg_p["F_break_equatorial"] * (sh / 0.35) ** 2
    return dict(age_days=s["age_d"], mass_g=s["mass_g"], air_cell_mm=s["air_cell_mm"], haugh_unit=s["haugh_unit"],
                yolk_index=s["yolk_index"], albumen_pH=s["albumen_pH"], F_crack_equator_N=Fc,
                float_tilt_deg=s["float_tilt_deg"])


def snapshot(b, truth, step):
    S = b.summary(posture=True)
    row = dict(step=step, time_h=b.time_h, ess=b.ess())
    for k in TRACK:
        s = S[k]
        t = truth[k]
        row[k] = dict(mean=s["mean"], sd=s["sd"], p05=s["p05"], p95=s["p95"], truth=t,
                      inside=bool(s["p05"] - 1e-9 <= t <= s["p95"] + 1e-9))
    return row


def main(seed=3):
    rng = np.random.default_rng(1000 + seed)
    p = PR.sample("egg", seed)
    p["storage_temp_C"] = 6.0                               # this egg comes from the fridge (the scenario)
    egg = Egg(seed)
    clock = EL.EggClock.of(p, lay=False)                    # aged from lay to its register age at 6 C
    b = EggBelief.prior(n=4000, seed=seed + 50)
    rows = []
    sd = lambda k: float(E[k][1])                           # noqa: E731
    tv = truth_values(clock, p)
    rows.append(snapshot(b, tv, "prior"))
    # day 0
    days_left = int(round(E["best_before_days"][1] - tv["age_days"]))
    obs = [Observation("storage", {"where": "fridge"}, 0.0, "fetched from the fridge"),
           Observation("carton_bbd", {"days_left": days_left}, 0.0, "best-before on the carton"),
           Observation("mass_g", {"value": tv["mass_g"] + rng.normal(0, sd("sensor_scale_sd_g"))}, 0.0, "scale"),
           Observation("candling", {"air_cell_mm": tv["air_cell_mm"] + rng.normal(0, sd("sensor_candling_air_sd_mm")),
                                    "crack_seen": False}, 0.0, "candling lamp"),
           Observation("float_test", {"tilt_deg": tv["float_tilt_deg"] + rng.normal(0, sd("sensor_float_tilt_sd_deg"))},
                       0.0, "float test in water")]
    for o in obs:
        b.observe(o, rng)
        rows.append(snapshot(b, tv, o.kind))
    # day 7 in the fridge
    b.predict(7 * 24.0, rng, storage="fridge")
    clock.advance(7 * 24.0, 6.0)
    tv = truth_values(clock, p)
    rows.append(snapshot(b, tv, "predict 7 d fridge"))
    b.observe(Observation("float_test", {"tilt_deg": tv["float_tilt_deg"] + rng.normal(0, sd("sensor_float_tilt_sd_deg"))},
                          b.time_h, "float test in water"), rng)
    rows.append(snapshot(b, tv, "float_test"))
    # pick up at the best admissible grip
    cor = b.corridor(np.pi / 2, a_max=2.0)
    F = (cor[0] + cor[1]) / 2 if cor else 3.0
    adm, pr = b.admissible(F)
    m = tv["mass_g"] * 1e-3
    slipped = F < m * (9.81 + 2.0) / (2 * p["friction_shell_rubber"])
    cracked = F >= tv["F_crack_equator_N"]
    b.observe(Observation("grip", {"F_N": F, "slipped": bool(slipped), "cracked": bool(cracked)}, b.time_h,
                          "picked up at the equator"), rng)
    rows.append(snapshot(b, tv, f"grip {F:.1f} N"))
    # strike on the bowl rim at the speed with the best P(cracked open) and P(shatter) < 2 %
    vs = np.linspace(0.05, 1.2, 47)
    best = None
    for v in vs:
        q = b.strike_outcome_probs(v, RIM)
        pq = (b.weights[:, None] * q).sum(0)
        if pq[3] < 0.02 and (best is None or pq[2] > best[1][2]):
            best = (v, pq)
    v = float(best[0])
    et = EggTruth(egg, m)
    et.egg.p.update({k: p[k] for k in ("shell_thickness", "F_break_equatorial", "F_break_axial", "shell_E")})
    res = EC.strike(et, v, RIM, rng)
    outcome = {"intact": "intact", "microcracked": "intact", "check": "check", "cracked_open": "cracked_open",
               "shattered": "shattered"}[res.state]
    b.observe(Observation("strike", {"v": v, "tool": RIM, "outcome": outcome}, b.time_h,
                          f"strike on the bowl rim at {v:.2f} m/s"), rng)
    rows.append(snapshot(b, tv, f"strike {v:.2f} m/s -> {outcome}"))
    strike_rec = dict(v=v, predicted=dict(zip(EggBelief.STRIKE_STATES, map(float, best[1]))), outcome=outcome,
                      ratio_truth=res.ratio)
    if outcome in ("check", "cracked_open"):
        b.observe(Observation("opened", {"albumen_height_mm": float(clock.state.albumen_height_mm[0]) +
                                         rng.normal(0, sd("sensor_albumen_height_sd_mm")),
                                         "yolk_index": tv["yolk_index"] + rng.normal(0, sd("sensor_yolk_index_sd"))},
                              b.time_h, "opened into the bowl"), rng)
        rows.append(snapshot(b, tv, "opened"))
    # write the node
    node = os.path.join(ROOT, "assets", "food", f"egg_{seed}")
    built = os.path.exists(os.path.join(node, "whole.usdc"))
    figdir = os.path.join(node, "belief") if built else os.path.join(ROOT, "renders", "egg", "belief")
    os.makedirs(figdir, exist_ok=True)
    if built:                                           # a built node: write the belief layer and the particles
        b.save(os.path.join(node, "belief", "belief.npz"))
        write_belief_layer(os.path.join(node, "sdt_belief.usda"), "whole.usdc", b, "belief/belief.npz")
    inside = {k: sum(r[k]["inside"] for r in rows) for k in TRACK}
    rep = dict(seed=seed, truth_params={k: p[k] for k in ("age_days", "storage_temp_C", "ageing_rate_factor",
                                                          "F_break_equatorial", "shell_thickness")},
               grip=dict(F=F, corridor=cor, admissible=adm, **pr), strike=strike_rec, rows=rows,
               inside_count=inside, steps=len(rows))
    os.makedirs(os.path.join(ROOT, "test_reports"), exist_ok=True)
    json.dump(rep, open(os.path.join(ROOT, "test_reports", f"egg_belief_{seed}.json"), "w"), indent=1, default=float)
    figure(rows, os.path.join(figdir, "timeline.png" if built else f"timeline_{seed}.png"), seed)
    for r in rows:
        print(f"{r['step']:28s} ess {r['ess']:6.0f} | " + " | ".join(
            f"{k} {r[k]['mean']:.2f}±{r[k]['sd']:.2f} (t {r[k]['truth']:.2f}{'' if r[k]['inside'] else ' OUT'})"
            for k in ("age_days", "air_cell_mm", "haugh_unit", "F_crack_equator_N")))
    print("inside the 90 % band:", inside, "of", len(rows))
    print("grip:", rep["grip"])
    print("strike:", strike_rec)


def figure(rows, path, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = ["age_days", "mass_g", "air_cell_mm", "haugh_unit", "albumen_pH", "F_crack_equator_N"]
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    x = np.arange(len(rows))
    for a, k in zip(ax.flat, ks):
        m = np.array([r[k]["mean"] for r in rows])
        lo = np.array([r[k]["p05"] for r in rows])
        hi = np.array([r[k]["p95"] for r in rows])
        t = np.array([r[k]["truth"] for r in rows])
        a.fill_between(x, lo, hi, alpha=0.25, label="belief 90 %")
        a.plot(x, m, "o-", label="belief mean")
        a.plot(x, t, "k--", label="hidden truth")
        a.set_title(k)
        a.set_xticks(x)
        a.set_xticklabels([r["step"][:16] for r in rows], rotation=60, ha="right", fontsize=7)
    ax.flat[0].legend(fontsize=7)
    fig.suptitle(f"egg_{seed}: belief vs hidden truth (noise ILLUSTRATIVE; ageing DERIVED from pan2026)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
