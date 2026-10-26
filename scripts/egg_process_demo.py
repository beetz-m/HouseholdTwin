"""Egg designators on the emulator with the belief in the loop (L5 + L6 gate of the egg SDT stack).

Scenarios on one instance (seed s), each from the same hidden egg (fresh emulator per scenario):
  rim      CheckFreshness -> PickUpEgg at the belief's grip -> CrackEgg on a bowl rim at the belief's strike speed
           (P(cracked open) max with P(shatter) < 2 %), retries +15 % -> open -> fish out shards
  flat     the same cracked on a flat counter (fewer shards, more checks)
  boil     the egg kept 10 more days on the counter, then CheckFreshness -> BoilEgg (the twin's time for a
           creamy yolk from 22 C, egg_tasks.best_boil_time) -> peel
Every outcome's measured part goes into the belief; the log shows prediction vs outcome.

    ~/Documents/LabAutomationTwin/env_cad/bin/python scripts/egg_process_demo.py [seed]
Writes assets/food/egg_<s>/process/<scenario>/ (USD states, log.json) and test_reports/egg_process_<s>.json.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from food import egg_tasks as T  # noqa: E402
from food.egg_belief import EggBelief, Observation  # noqa: E402
from food.egg_emulator import EggEmulator  # noqa: E402


def best_strike(b, tool, p_shatter=0.02):
    best = None
    for v in np.linspace(0.05, 1.2, 47):
        pq = (b.weights[:, None] * b.strike_outcome_probs(v, tool)).sum(0)
        if pq[3] < p_shatter and (best is None or pq[2] > best[1][2]):
            best = (float(v), pq)
    return best


def scenario(seed, name, rng):
    out_dir = os.path.join(ROOT, "assets", "food", f"egg_{seed}", "process", name)
    emu = EggEmulator(seed, out_dir, seed=hash(name) % 1000, storage="fridge")
    b = EggBelief.prior(n=4000, seed=seed + 7, storage="fridge")
    rec = dict(scenario=name, steps=[])

    def fold(obs):
        for o in obs:
            b.observe(o, rng)

    if name == "boil":
        emu.wait(240, "counter")
        b.predict(240, rng, storage="counter")
    ok, outs, obs = T.run("CheckFreshness", emu, float_test=name != "rim")
    fold(obs)
    S = b.summary(posture=True)
    rec["steps"].append(dict(designator="CheckFreshness", outcomes=[o.outcome for o in outs],
                             belief={k: (round(S[k]["mean"], 2), round(S[k]["sd"], 2))
                                     for k in ("age_days", "air_cell_mm", "haugh_unit", "p_clean_peel")},
                             truth={k: round(v, 2) for k, v in emu.fresh().items()
                                    if k in ("age_d", "air_cell_mm", "haugh_unit", "p_clean_peel")}))
    if name in ("rim", "flat"):
        cor = b.corridor()
        F = (cor[0] + cor[1]) / 2 if cor else 3.0
        ok, outs, obs = T.run("PickUpEgg", emu, grip_force_N=F)
        fold(obs)
        rec["steps"].append(dict(designator="PickUpEgg", F=F, corridor=cor, outcome=outs[0].outcome))
        tool = T.RIM if name == "rim" else T.FLAT
        v, pq = best_strike(b, tool)
        ok, outs, obs = T.run("CrackEgg", emu, surface="bowl rim" if name == "rim" else "flat counter",
                              strike_speed=v)
        fold(obs)
        rec["steps"].append(dict(designator="CrackEgg", v=v, predicted=dict(zip(EggBelief.STRIKE_STATES,
                                                                                np.round(pq, 3).tolist())),
                                 outcomes=[(o.action, o.outcome, o.measured.get("v")) for o in outs], success=ok,
                                 shards_left=emu.s.shards_in_bowl, yolk=emu.s.yolk))
    else:
        pc = b.summary()["p_clean_peel"]
        t_min = T.best_boil_time(emu.egg, "creamy", egg_temp_C=22.0)     # the twin's recommendation (egg_cook)
        ok, outs, obs = T.run("BoilEgg", emu, time_min=t_min, egg_temp_C=22.0, target="creamy")
        fold(obs)
        rec["steps"].append(dict(designator="BoilEgg", time_min=t_min, predicted_p_clean_peel=round(pc["mean"], 3),
                                 outcomes=[o.outcome for o in outs], success=ok))
    S = b.summary(posture=True)
    f = emu.fresh()
    rec["final_belief"] = {k: (round(S[k]["mean"], 3), round(S[k]["sd"], 3)) for k in
                           ("age_days", "haugh_unit", "yolk_index", "albumen_pH", "F_crack_equator_N")}
    rec["truth"] = dict(age_days=f["age_d"], haugh_unit=f["haugh_unit"], yolk_index=f["yolk_index"],
                        albumen_pH=f["albumen_pH"], F_crack_equator_N=emu.truth.crack_force(np.pi / 2))
    rec["pieces"] = emu.s.pieces
    emu.save_log()
    return rec


def main(seed=3):
    rng = np.random.default_rng(seed)
    recs = [scenario(seed, n, rng) for n in ("rim", "flat", "boil")]
    json.dump(recs, open(os.path.join(ROOT, "test_reports", f"egg_process_{seed}.json"), "w"), indent=1, default=float)
    for r in recs:
        print("==", r["scenario"])
        for s in r["steps"]:
            print("  ", {k: v for k, v in s.items() if k != "belief"}, "belief" if "belief" in s else "",
                  s.get("belief", ""))
        print("   final belief", r["final_belief"])
        print("   truth       ", {k: round(v, 3) for k, v in r["truth"].items()})


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
