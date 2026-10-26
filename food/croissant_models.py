"""Croissant: staling, reheating, handling (register params.CROISSANT).
  firmness   TPA 5.16 N at 1 h -> 19.2 N at 24 h (ste25): log-time interpolation
  reheating  from frozen 180 C 8 min (croissant_misc); refresh 160-180 C 3-5 min (TYPICAL) partly resets staling
  handling   fragile and flaky: a gentle pinch below ~1 N or a lift from below (inference from the TPA firmness, not
             measured); flakes shed in proportion to crispness (ILLUSTRATIVE)
"""

import numpy as np


def firmness_N(age_h):
    t = np.clip(np.asarray(age_h, float), 1.0, 72.0)
    return float(np.exp(np.interp(np.log(t), [0.0, np.log(24.0)], np.log([5.16, 19.2]))))


class CroissantTruth:
    def __init__(self, c, mass_kg):
        self.c, self.m = c, mass_kg
        self.age_h = float(c.p["age_h"])
        self.cut, self.warm = "whole", False

    def discrete(self):
        F = firmness_N(self.age_h)
        return {"freshness": "fresh" if F < 8 else ("day_old" if F < 22 else "stale"), "cut_state": self.cut,
                "temperature": "warm" if self.warm else "room"}

    def reheat(self, T_C=170.0, minutes=4.0):
        if T_C >= 150 and minutes >= 3:
            self.age_h = max(1.0, 0.3 * self.age_h)
            self.warm = True
        return self.discrete()

    def grip_limit_N(self):
        return 0.15 * firmness_N(self.age_h)

    def flakes_shed(self, action="pick"):
        crisp = 1.0 if self.age_h < 6 else 0.4
        return {"pick": 3, "split": 25, "tear": 40}[action] * crisp


def croissant_beliefs(c, mp):
    return {"": {"sdt:epistemicStatus": "prior + observation", "sdt:belief:mass_g": mp["mass_kg"] * 1e3,
                 "sdt:belief:mass_g_source": "weighed", "sdt:belief:freshness": "fresh 0.5 | day_old 0.5 (prior)"}}


def croissant_truth(c):
    t = CroissantTruth(c, 0.057)
    T = {"": {f"truth:{k}": v for k, v in c.p.items() if not k.startswith("_") and isinstance(v, (int, float, str))}}
    T[""].update({f"truth:{k}": v for k, v in t.discrete().items()})
    T[""]["truth:firmness_N"] = firmness_N(t.age_h)
    return T
