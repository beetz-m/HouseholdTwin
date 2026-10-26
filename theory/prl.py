"""Programmable reinforcement learning inside the theory: the RL problem is generated from a theory version T_k --
its state features from the concepts in Sigma (only perceivable distinctions become features), its action space from
the designators and their parameters, its reward from the claim's KPIs, and its hard constraints from the forbidden
effects (a shield masks every action whose forbidden effect the theory predicts). The learned policy is a proposal
component; the commitment gate keeps the authority (theory/package.py).

Learner: batched every-visit Monte-Carlo control with epsilon-greedy exploration over tabular action values, trained
on research-SDT episodes (development seeds 2000-9999, ripeness classes mixed), evaluated greedily on the held-out
avocados (seeds 1000-1029 per class) against the hand-coded theory executive and fixed policies.

    ~/Documents/LabAutomationTwin/env_cad/bin/python -m theory.prl      (from ~/Documents/HouseholdTwin)
Writes theory/results/prl.json.
"""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from theory import avocado_bws as W
from theory import avocado_policies as P

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
HELD = list(range(1000, 1030))


@dataclass
class RLProgram:
    """An RL problem generated from a theory version."""
    name: str
    concepts: tuple                        # the theory's Sigma (names)
    kpis: dict                             # reward terms: name -> weight
    forbidden: dict                        # effect -> 'shield' (hard) or a soft penalty weight
    actions: dict = field(default_factory=dict)

    @classmethod
    def from_theory(cls, name, concepts, forbidden):
        acts = {"day": ["cut", "wait"], "removal": ["spoon", "knife_twist"]}
        if "SeedCoatSide" in concepts:
            acts["peel"] = [0, 1, 2]
        kpis = {"requirement": 1.0, "eating_quality": 0.2, "gouge": -0.3, "torn": -0.4, "seed_coat_left": -0.3,
                "day": -0.02}
        return cls(name, tuple(concepts), kpis, dict(forbidden), acts)

    def features(self, decision, b, ctx):
        f = [decision]
        if decision == "day":
            f.append(ctx["day"])
            if "RipenessStage" in self.concepts:
                m = float(np.sum(b.weights * b.particles["ripeness_stage"]))
                sd = float(np.sqrt(np.sum(b.weights * (b.particles["ripeness_stage"] - m) ** 2)))
                f += [int(np.digitize(m, [2.25, 2.75, 3.1, 3.4, 3.75, 4.25])), int(sd > 0.45)]
        elif decision == "removal" and "TornFlesh" in self.concepts:
            f.append(ctx.get("torn", False))
        elif decision == "peel":
            f.append(ctx.get("testa_seen", False))
        return tuple(f)

    def allowed(self, decision):
        acts = self.actions[decision]
        if decision == "removal" and self.forbidden.get("hazard") == "shield":
            return ["spoon"]                                # the theory predicts the hazard of the knife strike
        return acts

    def reward(self, plant, day):
        o = plant.outcome()
        f = o["facts"]
        r = self.kpis["requirement"] * W.G_B(plant) if "UnsafeToolUse(KnifeStrike)" not in f else \
            self.kpis["requirement"] * W.G_B_without_safety(plant)
        r += self.kpis["eating_quality"] * o["eating_quality"] + self.kpis["day"] * day
        r += self.kpis["gouge"] * ("Gouge(Cavity)" in f) + self.kpis["torn"] * ("TornFlesh(HalfB)" in f)
        r += self.kpis["seed_coat_left"] * ("TestaIn(Cavity)" in f)
        pen = self.forbidden.get("hazard")
        if isinstance(pen, (int, float)) and "UnsafeToolUse(KnifeStrike)" in f:
            r -= pen
        return float(r)


def episode(prog, Q, eps, seed, cls, misspec=None, greedy=False):
    """One research-SDT episode under the program's policy; returns (visited (state, action) list, reward, info)."""
    plant = W.Plant(W.instance(seed, cls), misspec=misspec, seed=seed)
    rng = np.random.default_rng(seed + 31)
    b = W.prior()
    W.observe_photo(b, plant.photo(), rng)
    visited = []

    def choose(decision, ctx):
        s = prog.features(decision, b, ctx)
        acts = prog.allowed(decision)
        if not greedy and rng.random() < eps:
            a = acts[rng.integers(len(acts))]
        else:
            q = Q.get(s, {})
            a = max(acts, key=lambda x: (q.get(str(x), (0.0, 0))[0] if q.get(str(x), (0, 0))[1] else -1e9 + 0.0,
                                         -acts.index(x)))
            if not any(q.get(str(x), (0, 0))[1] for x in acts):
                a = acts[0] if decision != "day" else "wait" if ctx["day"] < 4 else "cut"
        visited.append((s, str(a)))
        return a

    day = 0
    while True:
        if "TactileRipeness" in prog.concepts:
            for F in (P.F_INFORM, 2.0, 1.5, 1.0):
                if W.p_bruise(b, F) <= 0.01:
                    W.observe_squeeze(b, plant.squeeze(F), F, rng)
                    break
        if day >= 5 or choose("day", dict(day=day)) == "cut":
            break
        plant.wait(24.0)
        b.predict(24.0, rng)
        day += 1
    out = plant.cut()
    if out.outcome == "partly_scored":
        plant.cut()
    tw = plant.twist()
    if not tw.ok:
        plant.cut()
        tw = plant.twist()
    if tw.ok:
        rm_method = choose("removal", dict(torn=tw.outcome == "halved_torn"))
        rm = plant.remove_seed(rm_method)
        if rm.ok and "peel" in prog.actions:
            n = int(choose("peel", dict(testa_seen=bool(rm.measured.get("testa_visible_in_cavity")))))
            for _ in range(n):
                if plant.remnant > 0.05:
                    plant.peel(1)
    r = prog.reward(plant, day)
    return visited, r, dict(ok=bool(W.G_B(plant)), label=W.failure_label(plant), day=day,
                            hazard="UnsafeToolUse(KnifeStrike)" in plant.facts)


def _ep(args):
    prog, Q, eps, seed, cls, greedy = args
    return episode(prog, Q, eps, seed, cls, greedy=greedy)


def train(prog, rounds=8, per_round=700, workers=14):
    Q = {}
    curve = []
    rng = np.random.default_rng(0)
    seed = 2000
    for k in range(rounds):
        eps = max(0.05, 0.4 * (1 - k / (rounds - 1)))
        jobs = []
        for _ in range(per_round):
            jobs.append((prog, Q, eps, seed, rng.choice(list(W.CLASSES)), False))
            seed += 1
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_ep, jobs, chunksize=20))
        for visited, r, _ in res:
            for s, a in visited:
                q = Q.setdefault(s, {})
                m, n = q.get(a, (0.0, 0))
                q[a] = (m + (r - m) / (n + 1), n + 1)
        curve.append(dict(round=k, eps=eps, mean_reward=float(np.mean([r for _, r, _ in res])),
                          success=float(np.mean([i["ok"] for _, _, i in res]))))
    return Q, curve


def evaluate(prog, Q, workers=14):
    jobs = [(prog, Q, 0.0, s, cls, True) for cls in W.CLASSES for s in HELD]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_ep, jobs, chunksize=10))
    out = {}
    for (prog_, Q_, _, s, cls, _), (_, r, info) in zip(jobs, res):
        out.setdefault(cls, []).append(dict(seed=s, reward=r, **info))
    return out


def policy_table(prog, Q):
    rows = []
    for s, q in sorted(Q.items(), key=lambda kv: str(kv[0])):
        best = max(q.items(), key=lambda kv: kv[1][0])
        rows.append(dict(state=list(map(str, s)), best=best[0], values={a: round(v[0], 3) for a, v in q.items()},
                         visits=sum(v[1] for v in q.values())))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    T1 = ("Whole", "Scored", "HalvesApart", "SeedLocation", "RipenessStage", "TornFlesh")
    T3 = T1 + ("SeedCoatSide", "TactileRipeness")
    progs = [RLProgram.from_theory("RL from T1 (no seed-coat concept), shielded", T1, {"hazard": "shield"}),
             RLProgram.from_theory("RL from T3, shielded", T3, {"hazard": "shield"}),
             RLProgram.from_theory("RL from T3, hazard as a soft penalty 0.05", T3, {"hazard": 0.05})]
    res = dict(programs=[], seconds=None)
    for prog in progs:
        Q, curve = train(prog)
        ev = evaluate(prog, Q)
        res["programs"].append(dict(name=prog.name, concepts=list(prog.concepts), actions=prog.actions,
                                    kpis=prog.kpis, forbidden=prog.forbidden, curve=curve,
                                    eval={c: dict(ok=sum(e["ok"] for e in v), n=len(v),
                                                  hazard=sum(e["hazard"] for e in v),
                                                  mean_day=float(np.mean([e["day"] for e in v])),
                                                  failures={l: sum(1 for e in v if e["label"] == l and not e["ok"])
                                                            for l in {e["label"] for e in v}},
                                                  per_seed=[e["ok"] for e in v])
                                          for c, v in ev.items()},
                                    policy=policy_table(prog, Q)))
        print(prog.name, {c: (v["ok"], v["hazard"]) for c, v in res["programs"][-1]["eval"].items()},
              round(time.time() - t0), "s")
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(OUT, "prl.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
