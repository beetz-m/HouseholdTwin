"""Potato: boiling, greening, sweetening, peeling and cut-surface models; the robot's belief (params.POTATO).

  boiling      cold start: water heats 20 -> 100 C in T_HEAT minutes (ILLUSTRATIVE pot), the piece's centre follows
               with a first-eigenvalue conduction lag (alpha 0.15e-6 m2/s, kk18); first-order softening at the centre,
               k(T) = k100 exp(-Ea/R (1/T - 1/373 K)) with k100 0.28-0.40 /min, Ea 133 kJ/mol (potato_cook);
               'done' (skewer passes) at F/F0 <= F_DONE -- CALIBRATED: 2 cm cubes from cold done at 17.5 min
               (gritzer 15-20); checks: whole 55 mm tuber (German guides 20-25 min), large 75 mm (up to 60 min)
  greening     under light: unacceptable after ~1.5 d at 20 C, from day 2 at 15 C, day 4 at 10 C, none <= 5 C
               (potato_storage) -> rate interpolated in temperature
  sweetening   cold-induced below 8 C (potato_storage)
  peeling      ~1.1 mm removed (potato_peel); the twin computes the mass share from the geometry (source: ~9 %)
"""

import numpy as np

from . import params as PR

R_GAS = 8.314
ALPHA = 0.15e-6
T_HEAT = 8.0                       # min, cold water 20 -> 100 C (ILLUSTRATIVE pot)
F_DONE = None                      # set by calibrate() on import


def k_soft(T_C, k100=0.34, Ea=133e3):
    T = np.asarray(T_C, float) + 273.15
    return k100 * np.exp(-Ea / R_GAS * (1 / T - 1 / 373.15))


def boil(size_mm, minutes, shape="sphere", cold_start=True, k100=0.34, dt_s=2.0, T0=15.0):
    """Centre softening F/F0 of a piece boiled for `minutes` (sphere: whole tuber / cube ~ sphere of equal half-size)."""
    half = size_mm * 5e-4
    lam = np.pi ** 2 if shape == "sphere" else 3 * (np.pi / 2) ** 2
    tau = half ** 2 / (ALPHA * lam) / 60.0          # min
    Tc, lnF, t = T0, 0.0, 0.0
    while t < minutes:
        Tw = 20 + 80 * min(t / T_HEAT, 1.0) if cold_start else 100.0
        Tc += (Tw - Tc) * (dt_s / 60.0) / tau
        lnF -= float(k_soft(Tc, k100)) * dt_s / 60.0
        t += dt_s / 60.0
    return float(np.exp(lnF)), tau


def time_to_done(size_mm, shape="sphere", cold_start=True, k100=0.34, f_done=None):
    f_done = f_done or F_DONE
    lo, hi = 0.0, 120.0
    for _ in range(26):
        mid = (lo + hi) / 2
        f, _ = boil(size_mm, mid, shape, cold_start, k100, dt_s=4.0)
        lo, hi = (mid, hi) if f > f_done else (lo, mid)
    return hi


def calibrate(target_min=17.5, size_mm=20.0):
    """F_DONE such that 2 cm cubes from cold are done at the target time."""
    f, _ = boil(size_mm, target_min, "cube", True, dt_s=4.0)
    return f


F_DONE = calibrate()


def greening_rate(T_C):
    """Fraction of the unacceptable-greening dose per day under retail light."""
    return float(np.interp(T_C, [5.0, 10.0, 15.0, 20.0], [0.0, 1 / 4.0, 1 / 2.0, 1 / 1.5]))


class PotatoTruth:
    def __init__(self, potato, mass_kg):
        p = potato.p
        self.pt, self.m = potato, mass_kg
        self.green = greening_rate(20.0) * p["light_exposure_h"] / 24.0
        self.sweet = False
        self.f_cook, self.peeled, self.cut_exposed_h = 1.0, False, None
        self.k100 = float(p["k100"])

    def store(self, days, T_C, light=False):
        if light:
            self.green += greening_rate(T_C) * days
        if T_C < 8.0 and days > 3:
            self.sweet = True

    def discrete(self):
        return {"greening": "none" if self.green < 0.3 else ("slight" if self.green < 1.0 else "unacceptable"),
                "sweetened": "yes" if self.sweet else "no", "peel_state": "peeled" if self.peeled else "unpeeled",
                "cooked_state": "raw" if self.f_cook > 0.9 else ("done" if self.f_cook <= F_DONE else "par_cooked")}

    def boil(self, size_mm, minutes, shape="sphere"):
        f, tau = boil(size_mm, minutes, shape, True, self.k100)
        self.f_cook = min(self.f_cook, f)
        return {"F_over_F0_centre": f, "done": f <= F_DONE, "tau_min": tau}

    def peel(self, depth_mm=None):
        self.peeled = True
        return peel_loss_fraction(self.pt, depth_mm or PR.POTATO["peel_depth"][1])


def peel_loss_fraction(potato, depth_mm):
    """Mass share within depth_mm of the surface (from the instance's meridian; source ~9 %)."""
    from .geometry import Meridian  # noqa: F401
    from .meshlib import pappus_volume
    whole = pappus_volume(potato.mer)
    inner = pappus_volume(potato.mer.offset(depth_mm * 1e-3))
    return 1 - inner / whole


class PotatoBelief:
    """Cooking type from density: the robot weighs the tuber and estimates its volume from vision (sd 5 %);
    density / specific gravity -> dry matter -> mealy vs firm (higher SG = more mealy, potato_mech / hest56)."""

    def __init__(self, rng, n=3000):
        self.sg = rng.uniform(1.050, 1.100, n)
        self.w = np.full(n, 1.0 / n)

    def observe_density(self, rho_obs, sd_rel=0.05):
        self.w *= np.exp(-0.5 * ((rho_obs / 1000 - self.sg) / (sd_rel * self.sg)) ** 2)
        self.w /= self.w.sum()

    def summary(self):
        return {"sg_mean": float(self.w @ self.sg), "p_mealy": float(self.w @ (self.sg > 1.085))}


def potato_beliefs(pt, mp):
    return {"": {"sdt:epistemicStatus": "prior + observation", "sdt:belief:mass_g": mp["mass_kg"] * 1e3,
                 "sdt:belief:mass_g_source": "weighed", "sdt:belief:cooking_type": "unknown (label or density)",
                 "sdt:belief:greening": "none 0.85 | slight 0.15 (prior)"}}


def potato_truth(pt):
    t = PotatoTruth(pt, 0.13)
    T = {"": {f"truth:{k}": v for k, v in pt.p.items() if not k.startswith("_") and isinstance(v, (int, float, str))}}
    T[""].update({f"truth:{k}": v for k, v in t.discrete().items()})
    T[""]["truth:peel_loss_fraction_1p1mm"] = float(peel_loss_fraction(pt, 1.1))
    return T
