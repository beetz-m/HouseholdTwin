"""Orange: juicing, peeling, oil spray, segmenting (register params.ORANGE).

  juice        lab juice = mass x juice_fraction (ucr_citrus 45-54 % navel); hand reaming recovers HAND_EFF of it --
               DERIVED from 60-80 mL by hand (orange_juice) for a ~200 g navel at ~50 % -> 0.6-0.8; cold fruit
               yields less (orange_juice: use room temperature; factor ILLUSTRATIVE)
  bitterness   navel juice turns bitter within a few hours (edis891)
  peeling      navel medium-easy, Valencia hard with strips breaking; cold fruit harder (ritenour13); strip force
               per cultivar ILLUSTRATIVE (register peel_strip_force)
  oil spray    bending the peel bursts glands (30-140 kPa, smith18): spray when the bend radius drops below ~4 mm
"""

import numpy as np

from . import params as PR

HAND_EFF = 0.70


class OrangeTruth:
    def __init__(self, orange, mass_kg):
        self.o, self.m = orange, mass_kg
        self.peel_on, self.segments_free = 1.0, 0
        self.juice_ml, self.juice_age_h = 0.0, None
        self.T_C = 20.0

    def juice(self, rolled=True):
        frac = self.o.p["juice_fraction"] / 100
        cold = np.interp(self.T_C, [5, 20], [0.85, 1.0])
        ml = self.m * 1e3 * frac * HAND_EFF * cold / 1.045
        self.juice_ml, self.juice_age_h = ml, 0.0
        return {"juice_ml": ml}

    def strip_force(self):
        strip = 4.0 if self.o.p["cultivar"] == "navel" else 8.0
        return strip * float(np.interp(self.T_C, [5, 22], [1.4, 1.0]))

    def peel(self, F_pull):
        strip = self.strip_force()
        tear = strip * (1.8 if self.o.p["cultivar"] == "navel" else 1.2)
        if F_pull < strip:
            return {"ok": False, "why": "peel holds"}
        broke = F_pull > tear
        self.peel_on = max(0.0, self.peel_on - (0.12 if broke else 0.35))
        return {"ok": True, "strip_broke": bool(broke), "peel_on": self.peel_on}

    def bend_peel(self, radius_mm):
        return {"oil_spray": radius_mm < 4.0, "jet_m_s": 8.5 if radius_mm < 4.0 else 0.0}

    def discrete(self):
        bitter = self.juice_age_h is not None and self.juice_age_h > PR.ORANGE["bitter_juice_h"][1] and \
            self.o.p["cultivar"] == "navel"
        return {"peel_state": "on" if self.peel_on > 0.95 else ("off" if self.peel_on < 0.05 else "partly"),
                "juice": "none" if self.juice_ml == 0 else ("bitter" if bitter else "fresh"),
                "segments": int(self.o.n_seg)}


def orange_beliefs(o, mp):
    return {"": {"sdt:epistemicStatus": "prior + observation", "sdt:belief:mass_g": mp["mass_kg"] * 1e3,
                 "sdt:belief:mass_g_source": "weighed",
                 "sdt:belief:cultivar": "navel if a navel is visible at the stylar end, else Valencia",
                 "sdt:belief:juice_ml": mp["mass_kg"] * 1e3 * 0.495 * HAND_EFF / 1.045,
                 "sdt:belief:segments": "9-12 (unobserved)"}}


def orange_truth(o):
    t = OrangeTruth(o, 0.19)
    T = {"": {f"truth:{k}": v for k, v in o.p.items() if not k.startswith("_") and isinstance(v, (int, float, str))}}
    T[""].update({f"truth:{k}": v for k, v in t.discrete().items()})
    T[""]["truth:juice_ml_by_hand"] = t.juice()["juice_ml"]
    return T
