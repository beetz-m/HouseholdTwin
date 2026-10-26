"""Bell pepper: shrivel, chilling, coring, cutting, charring (register params.PEPPER).
  water loss  2.9-4.8 g/d at 17 C / 55 % RH (alt20) -> shrivel from 6-8 % loss; gloss and roughness cues
  chilling    below 7 C (pepper_misc)
  coring      cut round the stem, pull out stem + placenta + seeds as one plug; loose seeds scatter (ILLUSTRATIVE 10 %)
  charring    skin blisters and slips off; times GAP (TYPICAL: under a flame until blackened, sweat covered)
"""


class PepperTruth:
    def __init__(self, pe, mass_kg):
        self.pe, self.m = pe, mass_kg
        self.loss_pct = 3.8 / (mass_kg * 1e3) * 100 * float(pe.p["age_d"]) * 0.4   # room-shelf share (ILLUSTRATIVE)
        self.cored, self.skin, self.cut = False, "on", "whole"

    def discrete(self):
        L = self.loss_pct
        return {"shrivel": "none" if L < 6 else ("moderate" if L < 12 else "severe"),
                "gloss": "high" if L < 6 else ("medium" if L < 12 else "low"),
                "cored": "yes" if self.cored else "no", "skin": self.skin, "cut_state": self.cut}

    def core(self):
        self.cored = True
        return {"seeds_scattered": int(0.1 * self.pe.p["seeds"])}

    def char(self, blackened=True):
        if blackened:
            self.skin = "loose"
        return self.skin

    def strips(self, width_mm=10.0):
        import numpy as np
        circ = np.pi * self.pe.R * 2e3
        self.cut = f"{int(circ / width_mm)} strips"
        return {"strips": int(circ / width_mm)}


def pepper_beliefs(pe, mp):
    return {"": {"sdt:epistemicStatus": "prior + observation", "sdt:belief:mass_g": mp["mass_kg"] * 1e3,
                 "sdt:belief:mass_g_source": "weighed", "sdt:belief:colour": pe.p["colour"] + " (visible)",
                 "sdt:belief:seeds": "40-140 (unobserved)"}}


def pepper_truth(pe):
    t = PepperTruth(pe, 0.23)
    T = {"": {f"truth:{k}": v for k, v in pe.p.items() if not k.startswith("_") and isinstance(v, (int, float, str))}}
    T[""].update({f"truth:{k}": v for k, v in t.discrete().items()})
    return T
