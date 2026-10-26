"""Onion: tear gas after cutting, dicing, sauteing, storage (register params.ONION).

  lachrymatory factor   released where cells are cut: peak at ~1 min, gone by ~10 min (lokke12) -> dose(t) =
                        A * cut_area * (t/tau) exp(1 - t/tau), tau = 1 min, truncated at 10 min; blunt or fast knives
                        release up to ~40x / ~4x more droplets (wu25) -> knife factor
  dicing                ATK method: n_h horizontal + n_v vertical cuts on a half, then across -> dice ~ spacing;
                        layers hold together while the root end stays on
  sauteing              translucent 5-10, golden 15-20, caramelised 30-45 min (onion_misc, TYPICAL) at medium heat;
                        a heat factor scales the clock (ILLUSTRATIVE)
  storage               rest 4-6 weeks, then sprouting at room temperature; 0 C keeps 6-9 months (hb66_onion)
"""

import numpy as np


def lf_dose(cut_area_cm2, minutes, knife="sharp", tau=1.0):
    k = {"sharp": 1.0, "blunt": 40.0, "fast_sharp": 4.0}[knife]
    t = np.linspace(0, min(minutes, 10.0), 200)
    rate = (t / tau) * np.exp(1 - t / tau)
    return float(k * cut_area_cm2 * np.trapezoid(rate, t))


def saute_state(minutes, heat=1.0):
    t = minutes * heat
    return "raw" if t < 3 else ("translucent" if t < 12 else ("golden" if t < 25 else
                                                              ("caramelised" if t < 50 else "burnt")))


class OnionTruth:
    def __init__(self, onion, mass_kg):
        self.o, self.m = onion, mass_kg
        self.age_d = float(onion.p["age_d"])
        self.peeled = self.trimmed = False
        self.cut_area_cm2, self.pieces = 0.0, 1

    def discrete(self):
        return {"peel_state": "peeled" if self.peeled else "dry_skin_on",
                "trim_state": "trimmed" if self.trimmed else "untrimmed",
                "sprouting": "yes" if self.age_d > 42 else "no",
                "cut_state": "whole" if self.pieces == 1 else f"{self.pieces} pieces"}

    def dice(self, spacing_mm=6.0, n_horizontal=2):
        """Halve, then horizontal + vertical cuts towards the root, then across (onion_howto)."""
        D, H = self.o.D * 1e3, self.o.H * 1e3
        n_v = int(D / 2 / spacing_mm)
        n_x = int(D / spacing_mm)
        cut_area = 2 * (np.pi * D * H / 4) / 100.0 * (1 + n_horizontal + n_v + n_x) / 2   # rough cm2 (ILLUSTRATIVE)
        self.cut_area_cm2 += cut_area
        self.pieces = 2 * (n_horizontal + 1) * (n_v + 1) * (n_x + 1) // 2
        self.trimmed = True
        return {"dice_mm": spacing_mm, "pieces_est": self.pieces, "cut_area_cm2": cut_area}


def onion_beliefs(o, mp):
    return {"": {"sdt:epistemicStatus": "prior + observation", "sdt:belief:mass_g": mp["mass_kg"] * 1e3,
                 "sdt:belief:mass_g_source": "weighed", "sdt:belief:fleshy_scales": "7-8 (unobserved until cut)",
                 "sdt:belief:sprouting": "no 0.9 | yes 0.1 (prior)"}}


def onion_truth(o):
    t = OnionTruth(o, 0.15)
    T = {"": {f"truth:{k}": v for k, v in o.p.items() if not k.startswith("_") and isinstance(v, (int, float, str))}}
    T[""].update({f"truth:{k}": v for k, v in t.discrete().items()})
    T[""]["truth:scales"] = len(o.scale_s)
    return T
