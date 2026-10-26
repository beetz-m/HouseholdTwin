"""Tomato: ripening, handling, cutting and blanching models; the robot's belief (register params.TOMATO).

Hidden state: ripeness stage s (USDA 1..6), temperature history (chilling, volatile loss), damage, cut state.
  firmness k(s)          plate firmness, initial slope: G/P/R 15.1 / 6.0 / 4.5 N/mm at s = 1 / 4 / 6 (ince16, SV)
  skin rupture F_r(s)    2 mm probe: 6.92 / 3.99 / 2.74 N (ince16, SV)
  grip limits            safe strain 1.5 %, visible damage from 3 % of the diameter (tomato_mech, SV)
  volatiles              -65 % after 8 d at 5 C (zhang16): first-order loss below 12.5 C, k = ln(1/0.35) / 8 d at 5 C
                         scaled linearly to 0 at 12.5 C (scaling ILLUSTRATIVE)
  cutting                a blade breaks the skin when its local pressure beats the 2 mm-probe rupture pressure; a dull
                         smooth blade presses the fruit first (squash if the plate strain passes 3 %); serrated teeth
                         act as small probes (tomato_howto: 'serrated or very sharp knife')
  gel release            gel of the opened locules runs out: fraction per cut face ILLUSTRATIVE
  blanching              skin loosens after 30-45 s at 100 C (tomato_howto); > 60 s cooks the outer wall
"""

import numpy as np

from . import params as PR

S_PTS = [1.0, 4.0, 6.0]


def plate_k(s):
    return float(np.interp(s, S_PTS, [15.1, 6.0, 4.5]))


def skin_rupture_N(s):
    return float(np.interp(s, S_PTS, [6.92, 3.99, 2.74]))


def stage_class(s):
    return ["green", "breaker", "turning", "pink", "light_red", "red"][int(np.clip(round(s), 1, 6)) - 1]


class TomatoTruth:
    def __init__(self, tomato, mass_kg):
        p = tomato.p
        self.t, self.m = tomato, mass_kg
        self.s = float(p["ripeness_stage"])
        self.rate = float(p["stage_rate_20C"])
        self.volatiles = 1.0
        self.bruised = False
        self.cut_faces, self.gel_left = 0, 1.0
        self.skin = "on"
        self.events = []

    def advance(self, dt_d, T_C=20.0):
        thr = PR.TOMATO["chilling_threshold"][1]
        if T_C >= thr:
            self.s = min(6.5, self.s + self.rate * dt_d * np.clip((T_C - thr) / (20 - thr), 0, 1.4))
        else:
            k = np.log(1 / 0.35) / 8.0 * np.clip((thr - T_C) / (thr - 5.0), 0, 2)
            self.volatiles *= np.exp(-k * dt_d)

    def discrete(self):
        return {"ripeness_class": stage_class(self.s), "bruised": "yes" if self.bruised else "no",
                "flavour": "full" if self.volatiles > 0.8 else ("reduced" if self.volatiles > 0.5 else "flat"),
                "cut_state": "whole" if self.cut_faces == 0 else f"cut ({self.cut_faces} faces)", "skin": self.skin}

    def grasp_corridor(self, mu=None, a_max=2.0):
        mu = mu or self.t.p["friction_skin_rubber"]
        F_slip = self.m * (9.81 + a_max) / (2 * mu)
        D_mm = self.t.p["diameter"]
        F_safe = plate_k(self.s) * 0.015 * D_mm
        F_damage = plate_k(self.s) * 0.03 * D_mm
        return F_slip, F_safe, F_damage

    def squeeze(self, F):
        _, F_safe, F_dmg = self.grasp_corridor()
        if F > F_dmg:
            self.bruised = True
        return {"deflection_mm": F / plate_k(self.s), "bruised": self.bruised}

    def cut(self, knife="serrated", F_down=4.0):
        """One cut through the fruit.  Serrated: teeth (~2 mm contact each) break the skin at the 2 mm-probe force;
        smooth sharp: ~1.5 x that; smooth dull: ~5 x (ILLUSTRATIVE factors) -- the fruit is pressed meanwhile."""
        need = skin_rupture_N(self.s) * {"serrated": 1.0, "sharp": 1.5, "dull": 5.0}[knife]
        strain = need / plate_k(self.s) / self.t.p["diameter"]
        squashed = strain > PR.TOMATO["damage_strain"][1]
        ok = F_down >= need
        if ok:
            self.cut_faces += 2
            spill = 0.25 * self.gel_left                     # gel running out of the opened locules (ILLUSTRATIVE)
            self.gel_left -= spill
        else:
            spill = 0.0
        self.events.append(("cut", knife, F_down, ok, squashed))
        return {"ok": bool(ok), "F_needed": need, "squashed": bool(squashed and ok), "gel_out_frac": spill}

    def blanch(self, seconds, scored=True):
        if seconds < 20:
            res = "skin still tight"
        elif seconds <= 60:
            res = "skin slips off" if scored else "skin loosened (score an X to start)"
            self.skin = "loose"
        else:
            res = "skin slips off, outer wall cooked soft"
            self.skin = "loose"
        return res


class TomatoBelief:
    """Particles over the stage; observations: skin hue (camera) and squeeze stiffness."""

    def __init__(self, rng, n=3000):
        self.s = rng.uniform(1, 6, n)
        self.w = np.full(n, 1.0 / n)

    def observe_hue(self, h, sd=4.0):
        pred = np.interp(self.s, np.arange(1, 7), PR.TOMATO["skin_hue_by_stage"][1])
        self._u(np.exp(-0.5 * ((h - pred) / sd) ** 2))

    def observe_stiffness(self, k, sigma_rel=0.15):
        pred = np.array([plate_k(x) for x in self.s])
        self._u(np.exp(-0.5 * (np.log(k / pred) / sigma_rel) ** 2))

    def _u(self, lik):
        self.w = self.w * lik
        self.w /= self.w.sum()

    def summary(self):
        m = float(self.w @ self.s)
        return {"stage_mean": m, "stage_sd": float(np.sqrt(self.w @ (self.s - m) ** 2)),
                "p_ripe": float(self.w @ (self.s >= 5))}


def tomato_beliefs(t, mp):
    return {"": {"sdt:epistemicStatus": "prior + observation", "sdt:belief:mass_g": mp["mass_kg"] * 1e3,
                 "sdt:belief:mass_g_source": "weighed", "sdt:belief:ripeness_stage_mean": 3.5,
                 "sdt:belief:ripeness_source": "flat prior over USDA 1..6; observe the skin hue",
                 "sdt:belief:locules": "2-4 (unobserved until cut)"},
            "Locules": {"sdt:epistemicStatus": "unobserved (inside)"}}


def tomato_truth(t):
    tt = TomatoTruth(t, 0.1)
    T = {"": {f"truth:{k}": v for k, v in t.p.items() if not k.startswith("_") and isinstance(v, (int, float, str))}}
    T[""].update({f"truth:{k}": v for k, v in tt.discrete().items()})
    T[""].update({"truth:plate_k_N_per_mm": plate_k(tt.s), "truth:skin_rupture_N": skin_rupture_N(tt.s)})
    return T
