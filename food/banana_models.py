"""Banana: ripening dynamics, action outcome models and the robot's belief (register entries in params.BANANA).

Hidden state: continuous ripeness stage s (1..7 colour scale), temperature history (chilling), damage.
  ds/dt = r20 * Q10^((T - 20)/10) at T >= 13 C; below 13 C ripening stalls and chilling injury accumulates (wflo)
  pulp E(s)            ciro05 1.75 -> 1.09 MPa (compression), mapped onto stages 2 -> 7 (mapping ILLUSTRATIVE)
  pulp firmness(s)     jaiswal12 38 -> 14 N (2 mm probe, 10 days), same mapping
  peel strength(s)     khanal23 53 -> 2.1 N peel fracture, log-linear over stages 1 -> 7 (SOURCED-VERIFY)
  finger drop(s)       ~49 N on day 1, > 2x lower by day 3 (imsabai, mbeguie09)
  Brix(s)              chiw23 4.5 -> 23.7 Bx over 10 days (stages 1 -> 7 at ~0.6 stage/day)
Actions: separate a finger from the hand, snap the neck, peel (strip by strip), cut rounds, mash, bruise on impact,
browning of cut pulp.
"""

import numpy as np

from . import params as PR

S7 = np.arange(1, 8, dtype=float)


def pulp_E_MPa(s):
    return float(np.interp(s, [2.0, 7.0], [1.75, 1.09]))


def pulp_firmness_N(s):
    return float(np.interp(s, [1.0, 7.0], [38.0, 14.0]))


def peel_fracture_N(s):
    return float(np.exp(np.interp(s, [1.0, 7.0], np.log([53.0, 2.1]))))


def finger_drop_N(s):
    return float(np.exp(np.interp(s, [1.0, 2.5, 7.0], np.log([49.0, 22.0, 8.0]))))   # tail beyond day 3 ILLUSTRATIVE


def brix(s):
    return float(np.interp(s, [1.0, 2.8, 3.9, 5.1, 6.3], [4.5, 6.8, 8.0, 16.7, 23.7]))


def peel_hue(s):
    return float(np.interp(s, S7, PR.BANANA["peel_hue_by_stage"][1]))


class BananaTruth:
    """t in hours."""

    def __init__(self, banana, mass_kg):
        p = banana.p
        self.b, self.m = banana, mass_kg
        self.s = float(p["ripeness_stage"])
        self.r20 = float(p["stage_rate_20C"])
        self.chill_h = 0.0
        self.peel_on = 1.0                              # fraction of peel still attached
        self.strips = 0
        self.neck = "intact"
        self.bruise_mJ = 0.0
        self.cut_exposed_h = None
        self.events = []

    def advance(self, dt_h, T_C=20.0):
        if T_C >= PR.BANANA["chilling_threshold"][1]:
            self.s = min(7.5, self.s + self.r20 * PR.BANANA["q10_ripening"][1] ** ((T_C - 20) / 10) * dt_h / 24)
        else:
            self.chill_h += dt_h * (13.0 - T_C) / 3.0      # degree-hours below 13 C (dose scale ILLUSTRATIVE)
        if self.cut_exposed_h is not None:
            self.cut_exposed_h += dt_h

    def discrete(self):
        st = int(np.clip(round(self.s), 1, 7))
        return {"ripeness_stage": st, "edible": "yes" if 4.5 <= self.s <= 7.3 else ("unripe" if self.s < 4.5
                                                                                  else "overripe"),
                "peel_state": "on" if self.peel_on > 0.95 else ("off" if self.peel_on < 0.05 else "partly"),
                "neck": self.neck, "chilling_injury": "yes" if self.chill_h > 8 else "no",
                "bruised": "yes" if self.bruise_mJ > self.bruise_threshold() else "no",
                "pulp_browning": self.browning_class()}

    def bruise_threshold(self):
        """bugaud14: no visible bruise up to 200 mJ (green Grande Naine); susceptibility rises with ripening --
        threshold falling to a quarter at stage 7 (ILLUSTRATIVE)."""
        return PR.BANANA["bruise_threshold"][1] * np.interp(self.s, [1, 7], [1.0, 0.25])

    # ---------------------------------------------------------------- actions
    def separate_from_hand(self, F_pull):
        F = finger_drop_N(self.s)
        ok = F_pull >= F
        self.events.append(("separate", F_pull, F, ok))
        return {"ok": bool(ok), "F_needed": F, "failure": None if ok else "pull harder or twist at the pedicel"}

    def snap_neck(self, F_tip, lever=0.017):
        """Bend the stalk until the peel at the neck fractures.  The needed tip force scales with the peel fracture
        force (khanal23 53 -> 2.1 N); green fruit is 'too stiff' (kim22) -- above ~25 N we call it a failure."""
        F = 0.5 * peel_fracture_N(self.s) + 2.0
        if F_tip >= F:
            self.neck = "snapped"
            return {"ok": True, "F_needed": F}
        return {"ok": False, "F_needed": F, "failure": "neck too stiff: peel from the tip or cut the neck"}

    def peel_bonds(self):
        """(adhesion, seam, tear) strengths in N: register peel_bond_ratios x peel_fracture_N(stage)."""
        a, sm, t = self.b.p["peel_bond_ratios"]
        F = peel_fracture_N(self.s)
        return a * F, sm * F, t * F

    def peel_strip(self, F_pull, from_end="stem"):
        """Pull one strip down the finger.  The peel front needs adhesion + two seams (sources: see peel_bonds); the
        strip tears off short when the pull exceeds its own strength (scripts/sim_banana_peel.py shows the mechanism
        with breakable joints)."""
        if from_end == "stem" and self.neck != "snapped":
            return {"ok": False, "failure": "snap the neck first"}
        adh, seam, tear = self.peel_bonds()
        need = adh + 2 * seam
        if F_pull < need:
            return {"ok": False, "F_needed": need, "failure": "strip does not move"}
        n = int(self.b.p["ridges"])
        torn = F_pull > tear
        self.strips += 1
        self.peel_on = max(0.0, 1 - self.strips / max(3, n - 1)) if not torn else self.peel_on - 0.5 / n
        if self.peel_on <= 0.05:
            self.cut_exposed_h = 0.0
        return {"ok": not torn, "torn_short": bool(torn), "peel_on": self.peel_on, "F_needed": need}

    def impact(self, E_mJ):
        self.bruise_mJ = max(self.bruise_mJ, E_mJ)
        return self.bruise_mJ > self.bruise_threshold()

    def cut_rounds(self, thickness_mm=None, F_blade=None):
        t = thickness_mm or PR.BANANA["rounds_thickness"][1]
        L = self.b.L * 1e3 * 0.9
        self.cut_exposed_h = 0.0
        return {"pieces": int(L // t), "thickness_mm": t, "squash": pulp_E_MPa(self.s) < 1.2 and t < 4}

    def browning_class(self, lemon=False):
        """Cut pulp: untreated shelf life < 6 h (SOURCED-VERIFY); lemon juice slows it (factor ILLUSTRATIVE)."""
        if self.cut_exposed_h is None:
            return "none"
        h = self.cut_exposed_h / (4.0 if lemon else 1.0)
        return "fresh" if h < 1 else ("slight" if h < 6 else "brown")


# ------------------------------------------------------------------------------------------------ belief
class BananaBelief:
    """Particles over the stage.  Observations: peel hue from the camera (ringer21 hue per stage, noise sd 3 deg),
    squeeze firmness (pulp E through the peel; sigma_rel 0.15)."""

    def __init__(self, rng, n=3000):
        self.rng = rng
        self.s = rng.uniform(1, 7, n)
        self.w = np.full(n, 1.0 / n)

    def observe_hue(self, h_deg, sd=3.0):
        pred = np.interp(self.s, S7, PR.BANANA["peel_hue_by_stage"][1])
        self._update(np.exp(-0.5 * ((h_deg - pred) / sd) ** 2))

    def observe_firmness(self, E_obs_MPa, sigma_rel=0.15):
        pred = np.interp(self.s, [2.0, 7.0], [1.75, 1.09])
        self._update(np.exp(-0.5 * (np.log(E_obs_MPa / pred) / sigma_rel) ** 2))

    def _update(self, lik):
        self.w = self.w * lik
        self.w /= self.w.sum()

    def summary(self):
        m = float(self.w @ self.s)
        sd = float(np.sqrt(self.w @ (self.s - m) ** 2))
        return {"stage_mean": m, "stage_sd": sd, "p_edible": float(self.w @ ((self.s >= 4.5) & (self.s <= 7.3)))}


def banana_beliefs(banana, mp):
    return {"": {"sdt:epistemicStatus": "prior + observation", "sdt:belief:mass_g": mp["mass_kg"] * 1e3,
                 "sdt:belief:mass_g_source": "weighed",
                 "sdt:belief:ripeness_stage_mean": 4.0, "sdt:belief:ripeness_stage_sd": 1.7,
                 "sdt:belief:ripeness_source": "flat prior over the 1..7 chart; observe the peel hue to update",
                 "sdt:belief:peel_state": "on 1.0"},
            "Pulp": {"sdt:epistemicStatus": "unobserved (inside the peel)"},
            "Peel": {"sdt:belief:bruised": "unknown", "sdt:epistemicStatus": "visible"}}


def banana_truth(banana):
    t = BananaTruth(banana, 0.12)
    T = {"": {f"truth:{k}": v for k, v in banana.p.items() if not k.startswith("_")
              and isinstance(v, (int, float, str))}}
    T[""].update({f"truth:{k}": v for k, v in t.discrete().items()})
    T[""].update({"truth:pulp_E_MPa": pulp_E_MPa(t.s), "truth:brix": brix(t.s), "truth:peel_hue_deg": peel_hue(t.s),
                  "truth:finger_drop_N": finger_drop_N(t.s), "truth:peel_fracture_N": peel_fracture_N(t.s)})
    return T
