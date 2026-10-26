"""Carrot: storage, cooking and cutting models, the robot's belief (register entries in params.CARROT).

Hidden state: age since purchase, storage, water loss (relative water content RWC), peel coverage, cooking dose.
  water loss         rate by storage (register water_loss_rate, ILLUSTRATIVE from 'unacceptable after ~3 d')
  stiffness          E = E0 / 1.32^(t / 7 d) in the fridge bag (vob24); turgor firmness ~0 below RWC 0.95
                     (herp99_calbo00) -> 'limp' (bends instead of snapping)
  cooking            2nd-order fractional conversion (peng14): C = 1 / (1 + k t), F = Finf + (F0 - Finf) C,
                     k(T) = 0.035 / min * exp(-Ea/R (1/T - 1/353.15 K)), Ea 138.9 kJ/mol; the piece's centre
                     temperature follows a first-eigenvalue conduction lag (diffusivity TYPICAL)
  doneness classes   crisp-tender F/F0 <= 0.40 (CALIBRATED: 6 mm rounds reach it at 4.8 min, carrot_boil 4-5 min),
                     tender <= 0.30 (CALIBRATED: 6.1 min, carrot_boil '+1-2 min soft'), soft <= 0.12, mushy below
                     (ILLUSTRATIVE); check: whole 31 mm root crisp-tender at 16 min vs 'whole 10-15 min' (+1 min)
Actions: snap (three-point bend, jah18 48.6 N), cut (blade shear 41.1 N whole root, jah18), peel strokes (2 mm deep,
con22), top and tail, white blush on cut / peeled faces (sim09_cz95).
"""

import numpy as np

from . import params as PR

R_GAS = 8.314
K80, EA, T80 = 0.035, 138.9e3, 353.15
FINF = 0.02
ALPHA = 1.4e-7                                         # m2/s carrot thermal diffusivity (TYPICAL, ~1.3-1.5e-7)
LOSS_RATE = {"fridge_bag": 0.3, "fridge_open": 1.0, "room_open": 2.0}   # %/day (ILLUSTRATIVE)


def k_soft(T_C):
    T = np.asarray(T_C, float) + 273.15
    return K80 * np.exp(-EA / R_GAS * (1 / T - 1 / T80))


def doneness_class(f):
    if f > 0.70:
        return "raw_crisp"
    if f > 0.40:
        return "par_cooked"
    if f > 0.30:
        return "crisp_tender"
    if f > 0.12:
        return "tender"
    if f > 0.05:
        return "soft"
    return "mushy"


def boil(shape, size_mm, minutes, T_water=100.0, T0=20.0, dt_s=1.0, conversion0=1.0):
    """Remaining firmness fraction F/F0 after boiling a piece.  shape: 'slab' (rounds, size = thickness) or
    'cylinder' (whole / batons, size = diameter).  Softening driven by the centre temperature (first-eigenvalue lag:
    slab tau = (h/2)^2 / (alpha (pi/2)^2), cylinder tau = r^2 / (alpha 2.405^2))."""
    half = size_mm * 5e-4
    lam = (np.pi / 2) ** 2 if shape == "slab" else 2.405 ** 2
    tau = half ** 2 / (ALPHA * lam)
    inv_c = 1.0 / conversion0                           # 1/C - 1 = integral of k dt (2nd-order fractional conversion)
    t = 0.0
    while t < minutes * 60:
        Tc = T_water - (T_water - T0) * np.exp(-t / tau)
        inv_c += float(k_soft(Tc)) * dt_s / 60.0
        t += dt_s
    C = 1.0 / inv_c
    return FINF + (1 - FINF) * C, tau


def time_to(shape, size_mm, target_f, T_water=100.0, T0=20.0):
    lo, hi = 0.0, 60.0
    for _ in range(30):
        mid = (lo + hi) / 2
        f, _ = boil(shape, size_mm, mid, T_water, T0, dt_s=2.0)
        lo, hi = (mid, hi) if f > target_f else (lo, mid)
    return hi


class CarrotTruth:
    """t in days for storage, minutes for cooking."""

    def __init__(self, carrot, mass_kg):
        p = carrot.p
        self.c, self.m0 = carrot, mass_kg
        self.age_d, self.storage = float(p["age_d"]), p["storage"]
        self.loss_pct = LOSS_RATE[self.storage] * self.age_d
        self.peeled_frac, self.trimmed, self.f_cook = 0.0, set(), 1.0
        self.cut_faces_age_h = None
        self.events = []

    def advance_days(self, d, storage=None):
        self.storage = storage or self.storage
        self.age_d += d
        self.loss_pct += LOSS_RATE[self.storage] * d
        if self.cut_faces_age_h is not None:
            self.cut_faces_age_h += 24 * d

    @property
    def rwc(self):
        return 1 - self.loss_pct / 100 / 0.88             # water is ~88 % of fresh mass (fdc, jah18)

    @property
    def E_MPa(self):
        p = self.c.p
        E0 = 0.5 * (p["E_phloem"] + p["E_xylem"])
        E = E0 / PR.CARROT["E_decay_7d"][1] ** (self.age_d / 7)
        return E * float(np.clip((self.rwc - 0.90) / 0.05, 0.15, 1.0))   # turgor collapse below RWC 0.95

    def discrete(self):
        limp = self.rwc < PR.CARROT["limp_rwc"][1]
        return {"firmness_class": "limp" if limp else "crisp",
                "peel_state": "peeled" if self.peeled_frac > 0.95 else ("partly" if self.peeled_frac > 0.05
                                                                          else "unpeeled"),
                "trim_state": "+".join(sorted(self.trimmed)) or "untrimmed",
                "cooked_state": doneness_class(self.f_cook), "surface_blush": self.blush()}

    # ---------------------------------------------------------------- actions
    def snap(self, F):
        """Three-point bend: 48.6 N at 21.9 mm deflection for a fresh root of ~28.6 mm (jah18); scales with D^3 and
        E.  A limp root bends without snapping."""
        D = self.c.D * 1e3
        F_snap = PR.CARROT["bend_Fmax"][1] * (D / 28.6) ** 3 * self.E_MPa / 4.7
        if self.discrete()["firmness_class"] == "limp":
            return {"ok": False, "F_snap": F_snap, "failure": "limp carrot bends instead of snapping: cut it"}
        return {"ok": bool(F >= F_snap), "F_snap": F_snap}

    def cut(self, F_blade, serrated=False, where="middle"):
        """Straight-blade shear 41.1 N through a whole root (jah18), scaled by the local diameter squared; luo_2026:
        chopping with a serrated knife fails on carrots, sawing works."""
        D = self.c.D * 1e3 * (0.6 if where == "tip" else 1.0)
        F_need = PR.CARROT["blade_shear"][1] * (D / 28.6) ** 2 * (self.E_MPa / 4.7) ** 0.5
        ok = F_blade >= F_need and not serrated
        if ok:
            self.cut_faces_age_h = 0.0
        return {"ok": bool(ok), "F_needed": F_need,
                "failure": None if ok else ("chopping with a serrated knife fails: saw" if serrated else
                                            "press harder or rock the knife")}

    def top_and_tail(self):
        self.trimmed |= {"top", "tail"}
        self.cut_faces_age_h = 0.0
        return self.discrete()["trim_state"]

    def peel_stroke(self, strip_width_mm=17.0):
        """One peeler stroke removes a ~2 mm deep strip (con22) of strip_width over half the length; the carrot is
        turned between strokes (carrot_cuts).  Coverage per stroke = width / circumference / 2 halves."""
        circ = np.pi * self.c.D * 1e3
        self.peeled_frac = min(1.0, self.peeled_frac + strip_width_mm / circ / 2)
        self.cut_faces_age_h = 0.0 if self.cut_faces_age_h is None else self.cut_faces_age_h
        return self.peeled_frac

    def boil(self, shape, size_mm, minutes, T_water=100.0):
        C0 = (self.f_cook - FINF) / (1 - FINF)
        f, tau = boil(shape, size_mm, minutes, T_water, conversion0=max(C0, 1e-6))
        self.f_cook = f
        return {"F_over_F0": f, "class": doneness_class(f), "tau_s": tau}

    def blush(self):
        h = self.cut_faces_age_h
        if h is None:
            return "none"
        if h < PR.CARROT["blush_reversible_h"][1]:
            return "none"
        return "reversible" if h < PR.CARROT["blush_irreversible_h"][1] else "irreversible"


# ------------------------------------------------------------------------------------------------ belief
class CarrotBelief:
    """Particles over (age_d, storage) -> RWC.  Observations: bend test with the gripper (stiffness through E and
    limpness), weight against the fresh-size prior, white blush on cut faces."""

    def __init__(self, rng, n=3000):
        self.rng = rng
        self.age = rng.uniform(1, 21, n)
        self.storage = rng.choice(list(LOSS_RATE), n, p=[0.6, 0.25, 0.15])
        self.w = np.full(n, 1.0 / n)

    def _rwc(self):
        loss = np.array([LOSS_RATE[s] for s in self.storage]) * self.age
        return 1 - loss / 100 / 0.88

    def observe_bend(self, limp, p_err=0.05):
        pred = self._rwc() < PR.CARROT["limp_rwc"][1]
        self._update(np.where(pred == limp, 1 - p_err, p_err))

    def _update(self, lik):
        self.w = self.w * lik
        self.w /= self.w.sum()

    def summary(self):
        r = self._rwc()
        return {"age_d_mean": float(self.w @ self.age), "p_limp": float(self.w @ (r < 0.95)),
                "rwc_mean": float(self.w @ r)}


def carrot_beliefs(carrot, mp):
    b = CarrotBelief(np.random.default_rng(1)).summary()
    return {"": {"sdt:epistemicStatus": "prior + observation", "sdt:belief:mass_g": mp["mass_kg"] * 1e3,
                 "sdt:belief:mass_g_source": "weighed", "sdt:belief:p_limp": b["p_limp"],
                 "sdt:belief:age_d_mean": b["age_d_mean"], "sdt:belief:cooked_state": "raw 1.0"},
            "SecondaryXylem": {"sdt:belief:radius_fraction": "0.33 +- 0.04 (prior)",
                               "sdt:epistemicStatus": "unobserved until cut"}}


def carrot_truth(carrot):
    t = CarrotTruth(carrot, 0.115)
    T = {"": {f"truth:{k}": v for k, v in carrot.p.items() if not k.startswith("_")
              and isinstance(v, (int, float, str))}}
    T[""].update({f"truth:{k}": v for k, v in t.discrete().items()})
    T[""].update({"truth:rwc": t.rwc, "truth:E_MPa": t.E_MPa, "truth:water_loss_pct": t.loss_pct})
    return T
