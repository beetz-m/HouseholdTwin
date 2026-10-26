"""Cucumber: freshness, chilling, slicing, seeding, salting (register params.CUCUMBER).

  firmness      flesh puncture 8.7 N fresh (sakata11), -49 % over 9 d (jah16): exponential decay, rate scaled by
                storage (wrapped / unwrapped weight-loss ratio, cuc_misc; ILLUSTRATIVE mapping)
  chilling      below 10 C injury after 1-3 d (ucd_cuc): dose = sum of days below 10 C weighted by (10 - T)/3
  rounds        5 mm slices (SliceIt!); seeding: halve and scoop the cavity (its mass from the part graph)
  salting       30 min (wikibooks); water drawn out: GAP -> ILLUSTRATIVE 15 % of the slice mass
"""

import numpy as np


class CucumberTruth:
    def __init__(self, cu, mass_kg):
        p = cu.p
        self.c, self.m = cu, mass_kg
        self.age_d, self.storage = float(p["age_d"]), p["storage"]
        self.chill = 0.0
        if self.storage.startswith("fridge"):
            self.chill = self.age_d * (10 - 4) / 3.0
        self.cut, self.seeded, self.salted = "whole", False, False

    def firmness_N(self):
        k = np.log(1 / 0.51) / 9 * (1.0 if "unwrapped" not in self.storage else 1.6)
        return 8.7 * float(np.exp(-k * self.age_d))

    def discrete(self):
        F = self.firmness_N()
        return {"freshness": "crisp" if F > 6.5 else ("soft" if F > 4.5 else "limp"),
                "chilling_injury": "yes" if self.chill > 3.0 else "no", "cut_state": self.cut,
                "seeded": "yes" if self.seeded else "no"}

    def slice_rounds(self, thickness_mm=5.0):
        n = int(self.c.L * 1e3 * 0.9 / thickness_mm)
        self.cut = f"{n} rounds"
        return {"rounds": n}

    def seed_out(self):
        self.seeded = True
        return {"removed_frac": self.c.cav_frac ** 2}

    def salt(self, minutes=30):
        self.salted = minutes >= 20
        return {"water_out_frac": 0.15 if self.salted else 0.05}


def cucumber_beliefs(cu, mp):
    return {"": {"sdt:epistemicStatus": "prior + observation", "sdt:belief:mass_g": mp["mass_kg"] * 1e3,
                 "sdt:belief:mass_g_source": "weighed", "sdt:belief:freshness": "crisp 0.8 | soft 0.2 (prior)"}}


def cucumber_truth(cu):
    t = CucumberTruth(cu, 0.35)
    T = {"": {f"truth:{k}": v for k, v in cu.p.items() if not k.startswith("_") and isinstance(v, (int, float, str))}}
    T[""].update({f"truth:{k}": v for k, v in t.discrete().items()})
    T[""]["truth:firmness_N"] = t.firmness_N()
    return T
