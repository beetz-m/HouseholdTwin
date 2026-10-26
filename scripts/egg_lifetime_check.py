"""Check of the egg lifetime model (food/egg_lifetime.py) against its sources, and the L7 gate figure.

  1. the clock at 4 C and 25 C reproduces the pan2026 table (factor 1, the table's own fresh values);
  2. the air cell from the mass loss against Grashorn's anchors (gr: < 4 mm after 28 d at 6 C, > 6 mm after 25 d at
     22 C) -- a cross-check of two independent sources through the twin's own shell geometry;
  3. the float posture over age at fridge and counter temperature for the register's mean egg;
  4. a storage history (fridge 10 d -> counter 5 d -> fridge) and the best day to boil and peel.

    ~/Documents/LabAutomationTwin/env_cad/bin/python scripts/egg_lifetime_check.py
Writes test_reports/egg_lifetime.json and renders/egg/lifetime.png.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from food import egg_lifetime as EL  # noqa: E402
from food import params as PR  # noqa: E402

E = PR.EGG
MEAN = dict(length_L=57.0, shape_index_B_over_L=0.765, asymmetry_w=2.5, lay_SG=1.087, hu_fresh=81.3, yi_fresh=0.43,
            ph_fresh=7.9, ageing_rate_factor=1.0)


def table_fit():
    out = {}
    for T in (EL.T_LO, EL.T_HI):
        tab = EL.TAB[T]
        p = dict(MEAN, hu_fresh=tab["haugh_unit"][0], yi_fresh=tab["yolk_index"][0], ph_fresh=tab["albumen_pH"][0])
        c = EL.EggClock.of(p)
        c.state.T = T
        c.state.fresh["albumen_height_mm"] = np.array([tab["albumen_height_mm"][0]])
        rows = []
        for k, d in enumerate(EL.DAYS):
            if k:
                c.state.advance(EL.DAYS[k] - EL.DAYS[k - 1], T)
            s = c.state
            rows.append(dict(day=float(d), wl=float(s.weight_loss_pct[0]), h=float(s.albumen_height_mm[0]),
                             yi=float(s.yolk_index[0]), ph=float(s.albumen_pH[0])))
        err = {q: float(np.max(np.abs(np.array([r[k] for r in rows]) - np.array(tab[q]))))
               for k, q in (("wl", "weight_loss_pct"), ("h", "albumen_height_mm"), ("yi", "yolk_index"),
                            ("ph", "albumen_pH"))}
        out[f"{T:.0f}C"] = dict(max_abs_error=err, rows=rows)
    return out


def air_cell_anchors():
    res = {}
    for T, days, claim in ((6.0, 28, "< 4 mm"), (22.0, 25, "> 6 mm")):
        c = EL.EggClock.of(MEAN)
        c.advance(days * 24.0, T)
        res[f"{T:.0f}C_{days}d"] = dict(model_mm=round(c.summary()["air_cell_mm"], 2), gr=claim)
    return res


def float_timeline():
    out = {}
    for storage in ("fridge", "counter"):
        c = EL.EggClock.of(MEAN)
        rows = []
        for d in range(0, 61, 2):
            s = c.summary()
            rows.append(dict(day=d, tilt=s["float_tilt_deg"], cls=s["float"], SG=round(s["SG"], 4),
                             air=round(s["air_cell_mm"], 2), hu=round(s["haugh_unit"], 1), grade=s["usda_grade"]))
            c.advance(48.0, storage)
        out[storage] = rows
    return out


def history():
    c = EL.EggClock.of(MEAN)
    rows = []
    for hours, st in ((240, "fridge"), (120, "counter"), (240, "fridge")):
        c.advance(hours, st)
        rows.append(dict(after=f"{hours / 24:.0f} d {st}", **{k: (round(v, 3) if isinstance(v, float) else v)
                                                                for k, v in c.summary().items()}))
    best = {use: EL.best_day(MEAN, storage="fridge", use=use)[0] for use in ("boil_and_peel", "fry")}
    return rows, best


def figure(fit, flt, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))
    qs = [("wl", "weight_loss_pct", "weight loss %"), ("h", "albumen_height_mm", "thick albumen height mm"),
          ("yi", "yolk_index", "yolk index"), ("ph", "albumen_pH", "albumen pH")]
    for k, (key, q, lab) in enumerate(qs):
        a = ax.flat[k]
        for T, col in ((EL.T_LO, "tab:blue"), (EL.T_HI, "tab:red")):
            a.plot(EL.DAYS, EL.TAB[T][q], "o", color=col, label=f"pan2026 {T:.0f} C")
            rows = fit[f"{T:.0f}C"]["rows"]
            a.plot([r["day"] for r in rows], [r[key] for r in rows], "-", color=col, alpha=0.6, label=f"twin {T:.0f} C")
        for T, col in ((12.0, "tab:green"),):
            c = EL.EggClock.of(MEAN)
            xs, ys = [], []
            for d in np.arange(0, 31, 1.0):
                xs.append(d)
                ys.append(float(getattr(c.state, q)[0]))
                c.state.advance(1.0, T)
            a.plot(xs, ys, "--", color=col, label="twin 12 C (blend)")
        a.set_xlabel("days since lay")
        a.set_title(lab)
        if k == 0:
            a.legend(fontsize=7)
    for st, col in (("fridge", "tab:blue"), ("counter", "tab:red")):
        rows = flt[st]
        ax.flat[4].plot([r["day"] for r in rows], [r["air"] for r in rows], color=col, label=st)
        ax.flat[5].plot([r["day"] for r in rows], [r["tilt"] for r in rows], color=col, label=st)
    ax.flat[4].axhline(3.2, ls=":", c="k")
    ax.flat[4].axhline(4.8, ls=":", c="k")
    ax.flat[4].axhline(6.0, ls="--", c="gray")
    ax.flat[4].plot([28], [4.0], "v", c="tab:blue", label="gr: < 4 mm (28 d, 6 C)")
    ax.flat[4].plot([25], [6.0], "^", c="tab:red", label="gr: > 6 mm (25 d, 22 C)")
    ax.flat[4].set_title("air cell depth mm (USDA AA/A, EU A lines)")
    ax.flat[4].legend(fontsize=7)
    ax.flat[5].set_title("float test: resting tilt deg (90 = on its end)")
    ax.flat[5].legend(fontsize=7)
    for a in ax.flat[4:]:
        a.set_xlabel("days since lay")
    fig.suptitle("Egg lifetime model vs sources (DERIVED from pan2026; air cell and float test from the twin's geometry)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)


def main():
    fit = table_fit()
    anchors = air_cell_anchors()
    flt = float_timeline()
    hist, best = history()
    rep = dict(table_fit={k: v["max_abs_error"] for k, v in fit.items()}, air_cell_anchors=anchors,
               float_timeline=flt, history=hist, best_day=best, q10_weight_loss=EL.Q10)
    os.makedirs(os.path.join(ROOT, "test_reports"), exist_ok=True)
    json.dump(rep, open(os.path.join(ROOT, "test_reports", "egg_lifetime.json"), "w"), indent=1, default=str)
    figure(fit, flt, os.path.join(ROOT, "renders", "egg", "lifetime.png"))
    print("table fit (max abs error):", json.dumps(rep["table_fit"], indent=None))
    print("air cell anchors:", anchors)
    for st in ("fridge", "counter"):
        print(st, [(r["day"], r["cls"], round(r["tilt"], 0) if r["tilt"] == r["tilt"] else None, r["grade"])
                   for r in flt[st][::3]])
    for r in hist:
        print(r)
    print("best day:", best, "Q10 (weight loss):", round(EL.Q10, 2))


if __name__ == "__main__":
    main()
