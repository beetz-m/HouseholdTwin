"""Held-out evaluation of the avocado microtheory (seeds 1000-1029, 30 avocados per cell; development used seeds
< 1000). Every policy runs on the same avocados and plant seeds within a cell, so comparisons are paired.

    ~/Documents/LabAutomationTwin/env_cad/bin/python -m theory.run_microtheory      (from ~/Documents/HouseholdTwin)

Writes theory/results/microtheory.json: Task A and B tables (classes x policies, ablations, misspecified plants),
the Task A corridor sweep, the Task B band sweep, paired McNemar tests, verification verdicts (q = 0.8) and the
belief policy's calibration; and the plant traces of three reference episodes (theory/results/traces.json).
"""

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from theory import avocado_bws as W
from theory import avocado_policies as P

SEEDS = list(range(1000, 1030))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _job(args):
    task, cls, policy, seed, mis = args
    m = W.Misspec(**mis)
    plant = W.Plant(W.instance(seed, cls), misspec=m, seed=seed)
    rng = np.random.default_rng(seed + 31)
    if task == "A":
        ok, ready, forces = P.task_A(plant, policy, rng)
        return dict(task=task, cls=cls, policy=policy, seed=seed, mis=m.name, ok=bool(ok), ready=bool(ready),
                    forces=forces, label=W.failure_label(plant, "A", ready), stage=plant.stage)
    ok, day, label = P.task_B(plant, policy, rng)
    return dict(task=task, cls=cls, policy=policy, seed=seed, mis=m.name, ok=bool(ok), day=day, label=label,
                predicted=getattr(P.task_B, "predicted", None),
                stage0=W.instance(seed, cls)["ripeness_stage"], stage=plant.stage)


def _sweep_A(args):
    cls, F, seed = args
    plant = W.Plant(W.instance(seed, cls), seed=seed)
    k_true = plant.stiffness_true()
    k = plant.squeeze(F)
    return dict(cls=cls, F=F, seed=seed, bruised="Bruised(Mesocarp)" in plant.facts,
                informative=abs(k - k_true) / k_true <= 0.12)


def _sweep_B(args):
    cls, day, seed = args
    plant = W.Plant(W.instance(seed, cls), seed=seed)
    rng = np.random.default_rng(seed + 31)
    ok, _, label = P.task_B(plant, "cut now" if day == 0 else f"wait {day} days", rng)
    return dict(cls=cls, day=day, seed=seed, ok=bool(ok), label=label)


def analytic_corridor(cls, forces, n=3000):
    """Model marginals over the register population of a ripeness class: P(bruise | F) and P(informative | F)."""
    from food.avocado import penetrometer_force
    from food import params as PR
    lo, hi = W.CLASSES[cls]
    st = np.random.default_rng(5).uniform(lo, hi, n)
    thr = PR.AVOCADO["bruise_force_ratio"][1] * np.array([float(penetrometer_force(s)) for s in st])
    pb = [float(np.mean(F > thr)) for F in forces]
    pi = [float(math.erf(0.12 / (0.4 / F) / math.sqrt(2))) for F in forces]
    return pb, pi


def summarise(rows, key):
    out = {}
    for r in rows:
        k = tuple(r[x] for x in key)
        out.setdefault(k, []).append(r)
    res = []
    for k, rs in out.items():
        n, s = len(rs), sum(r["ok"] for r in rs)
        lo, hi = W.wilson(s, n)
        labels = {}
        for r in rs:
            if not r["ok"]:
                labels[r["label"]] = labels.get(r["label"], 0) + 1
        res.append(dict(zip(key, k), n=n, ok=s, lo=lo, hi=hi, verdict=W.verdict(s, n), failures=labels,
                        mean_day=float(np.mean([r["day"] for r in rs])) if "day" in rs[0] else None))
    return res


def paired(rows, cell_key, a, b):
    """Exact McNemar between policies a and b, per cell and pooled."""
    by = {}
    for r in rows:
        by.setdefault((tuple(r[x] for x in cell_key), r["policy"], r["seed"]), r["ok"])
    cells = sorted({tuple(r[x] for x in cell_key) for r in rows})
    out = []
    xa_all, xb_all = [], []
    for c in cells:
        xa = [by.get((c, a, s)) for s in SEEDS]
        xb = [by.get((c, b, s)) for s in SEEDS]
        if None in xa or None in xb:
            continue
        xa_all += xa
        xb_all += xb
        n10, n01, p = W.mcnemar(xa, xb)
        out.append(dict(cell=list(c), a=a, b=b, a_only=n10, b_only=n01, p=p))
    n10, n01, p = W.mcnemar(xa_all, xb_all)
    out.append(dict(cell=["pooled"], a=a, b=b, a_only=n10, b_only=n01, p=p))
    return out


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for cls in W.CLASSES:
        for pol in P.POLICIES_A + P.ABLATIONS_A:
            jobs += [("A", cls, pol, s, {}) for s in SEEDS]
        for pol in P.POLICIES_B + P.ABLATIONS_B:
            jobs += [("B", cls, pol, s, {}) for s in SEEDS]
    misA = [dict(bruise_ratio=0.6), dict(colour_L_bias=-10.0)]
    misB = [dict(ripening_rate=0.6), dict(ripening_rate=1.6), dict(colour_L_bias=-10.0), dict(adhesion=2.0)]
    for mis in misA:
        for cls in ("breaking", "ripe"):
            for pol in ("closed loop", "open-loop prior plan", "photo only", "firm thumb 10 N"):
                jobs += [("A", cls, pol, s, mis) for s in SEEDS]
    for mis in misB:
        for cls in ("unripe", "breaking"):
            for pol in ("belief band", "photo-only band", "cut now", "wait 2 days"):
                jobs += [("B", cls, pol, s, mis) for s in SEEDS]
    for mis in (dict(colour_L_bias=10.0), dict(ripening_rate=1.6)):           # riper than it looks / than modelled
        for pol in ("belief band", "photo-only band", "cut now", "wait 2 days"):
            jobs += [("B", "ripe", pol, s, mis) for s in SEEDS]
    forces = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.33, 4.0, 5.0, 7.0, 10.0]
    sweepA = [(cls, F, s) for cls in W.CLASSES for F in forces for s in range(1000, 1024)]
    sweepB = [(cls, d, s) for cls in W.CLASSES for d in range(6) for s in SEEDS]
    with ProcessPoolExecutor(14) as ex:
        rows = list(ex.map(_job, jobs, chunksize=8))
        sa = list(ex.map(_sweep_A, sweepA, chunksize=16))
        sb = list(ex.map(_sweep_B, sweepB, chunksize=8))
    A = [r for r in rows if r["task"] == "A"]
    B = [r for r in rows if r["task"] == "B"]
    res = dict(seeds=[SEEDS[0], SEEDS[-1]], n_per_cell=len(SEEDS), seconds=None,
               A=summarise(A, ("cls", "mis", "policy")), B=summarise(B, ("cls", "mis", "policy")),
               mcnemar_A=paired([r for r in A if r["mis"] == "well specified"], ("cls",), "closed loop", "firm thumb 10 N")
               + paired([r for r in A if r["mis"] == "well specified"], ("cls",), "closed loop", "open-loop prior plan")
               + paired([r for r in A if r["mis"] == "well specified"], ("cls",), "closed loop", "photo only")
               + paired([r for r in A if r["mis"] == "well specified"], ("cls",), "closed loop",
                        "bruise not owned by ripeness"),
               mcnemar_B=[x for b in ("cut now", "wait 2 days", "wait 4 days", "photo-only band", "ready / not ready",
                                      "no testa side", "no repair", "population ripening rate")
                          for x in paired([r for r in B if r["mis"] == "well specified"], ("cls",), "belief band", b)]
               + [x for b in ("photo-only band", "cut now", "wait 2 days")
                  for x in paired([r for r in B if r["mis"] != "well specified"], ("mis", "cls"), "belief band", b)],
               calibration_B=calibration([r for r in B if r["policy"] == "belief band"]),
               corridor=dict(forces=forces, simulated=summarise_sweep(sa),
                             analytic={c: dict(zip(("p_bruise", "p_inform"), analytic_corridor(c, forces)))
                                       for c in W.CLASSES}),
               band=summarise([dict(r, policy=f"day {r['day']}") for r in sb], ("cls", "policy")),
               episodes=rows)
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(OUT, "microtheory.json"), "w"), indent=1, default=str)
    traces = {}
    for cls, pol in (("unripe", "belief band"), ("unripe", "cut now"), ("ripe", "firm thumb 10 N")):
        plant = W.Plant(W.instance(1000, cls), seed=1000)
        rng = np.random.default_rng(1031)
        (P.task_A if pol.startswith("firm") else P.task_B)(plant, pol, rng)
        traces[f"{cls} / {pol}"] = plant.trace
    json.dump(traces, open(os.path.join(OUT, "traces.json"), "w"), indent=1, default=str)
    print("done", res["seconds"], "s;", len(rows), "episodes +", len(sa) + len(sb), "sweep episodes")


def calibration(rows):
    """The belief policy's predicted P(success) at its cut decision against the outcome (Brier score, bins)."""
    rs = [r for r in rows if r.get("predicted") is not None]
    if not rs:
        return {}
    p = np.array([r["predicted"] for r in rs])
    y = np.array([r["ok"] for r in rs], float)
    bins = []
    for lo, hi in ((0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01)):
        sel = (p >= lo) & (p < hi)
        if sel.any():
            bins.append(dict(range=[lo, min(hi, 1.0)], n=int(sel.sum()), predicted=float(p[sel].mean()),
                             observed=float(y[sel].mean())))
    return dict(n=len(rs), brier=float(np.mean((p - y) ** 2)), predicted_mean=float(p.mean()),
                observed_mean=float(y.mean()), bins=bins)


def summarise_sweep(sa):
    out = {}
    for r in sa:
        d = out.setdefault(r["cls"], {}).setdefault(r["F"], [0, 0, 0])
        d[0] += 1
        d[1] += r["bruised"]
        d[2] += r["informative"]
    return {c: {str(F): dict(n=v[0], p_bruise=v[1] / v[0], p_inform=v[2] / v[0]) for F, v in d.items()}
            for c, d in out.items()}


if __name__ == "__main__":
    main()
