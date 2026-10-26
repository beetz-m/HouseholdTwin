"""Lemon: juicing, zesting, wedges (register params.LEMON).  Juice by hand = mass x juice_fraction x HAND_EFF
(orange_models; DERIVED); check against 'juice of 1 lemon' 48 g (USDA).  Zest = flavedo only, ~1 mm (jen22_lemon)."""

import numpy as np

from .orange_models import HAND_EFF


class LemonTruth:
    def __init__(self, lemon, mass_kg):
        self.l, self.m = lemon, mass_kg
        self.juice_ml, self.zested, self.pieces = 0.0, 0.0, 1

    def juice(self):
        ml = self.m * 1e3 * self.l.p["juice_fraction"] / 100 * HAND_EFF / 1.03
        self.juice_ml = ml
        return {"juice_ml": ml}

    def zest(self, depth_mm=1.0):
        """Zest mass from the instance's flavedo; deeper than the flavedo takes bitter albedo."""
        A = 4 * np.pi * ((self.l.a ** 1.6 * self.l.a ** 1.6 + 2 * self.l.a ** 1.6 * self.l.c ** 1.6) / 3) ** (1 / 1.6)
        g = A * min(depth_mm, self.l.p["flavedo_thickness"]) * 1e-3 * self.l.p["peel_density"] * 1e3
        return {"zest_g": g, "bitter": depth_mm > self.l.p["flavedo_thickness"]}

    def wedges(self, n=8):
        self.pieces = n
        return {"wedges": n, "wedge_g": self.m * 1e3 / n}

    def discrete(self):
        return {"juice": "none" if self.juice_ml == 0 else "squeezed",
                "cut_state": "whole" if self.pieces == 1 else f"{self.pieces} pieces"}


def lemon_beliefs(l, mp):
    return {"": {"sdt:epistemicStatus": "prior + observation", "sdt:belief:mass_g": mp["mass_kg"] * 1e3,
                 "sdt:belief:mass_g_source": "weighed",
                 "sdt:belief:juice_ml": mp["mass_kg"] * 1e3 * 0.375 * HAND_EFF / 1.03,
                 "sdt:belief:seeds": "1-8 (unobserved)"}}


def lemon_truth(l):
    t = LemonTruth(l, 0.15)
    T = {"": {f"truth:{k}": v for k, v in l.p.items() if not k.startswith("_") and isinstance(v, (int, float, str))}}
    T[""]["truth:juice_ml_by_hand"] = t.juice()["juice_ml"]
    return T
