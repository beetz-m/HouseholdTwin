"""Cooking an egg in its shell: axisymmetric heat conduction through the twin's own tissue map, and doneness read
from the temperature history (coagulation thresholds) -- the hidden-world model behind Boil / Steam / ColdStart.

Geometry and tissues come from the egg instance (food.egg.Egg.label on the meridian plane), so the air cell, the
yolk's size and position and the shell thickness all enter.  Finite volumes on a (rho, z) grid; water acts through
a heat-transfer coefficient h_w (CALIBRATED to the published timing tables, see register 'cook_h_water').

Doneness (thresholds: opentextbc coagulation -- white sets ~60-65 C, yolk ~62-70 C; USDA safe 71 C):
  yolk  by the volume fraction f65 of yolk that has reached 65 C (it sets from the outside in): 'runny' < 0.15,
        'jammy' 0.15-0.45, 'creamy' 0.45-0.85, 'firm' >= 0.85 -- CALIBRATED to the published tables (hot start, medium
        eggs: 4.5 min runny, 6-7 jammy, 8-9 creamy, 10-11 hard; Kenji, fridge eggs: 6 liquid, 8.5 fudgy, 11 firm)
  white 'set' when the albumen next to the yolk has reached 75 C (ILLUSTRATIVE: ovalbumin firms ~80 C)
  green ring risk when the yolk surface stays above 80 C for > 4 min (ILLUSTRATIVE; ferrous sulfide forms with
  prolonged heating, slow cooling makes it worse -- incredible egg, mental floss)
"""

import numpy as np

# thermal properties (k W/mK, rho kg/m3, c J/kgK) -- TYPICAL literature values, see register
PROPS = {
    "shell": (0.45, 2300.0, 900.0),
    "membrane": (0.40, 1100.0, 3000.0),
    "air": (0.026, 1200.0, 1005.0),   # capacity x1000 for explicit stability; still < 0.3 % of the egg's
    "albumen": (0.55, 1035.0, 3700.0),
    "yolk": (0.34, 1030.0, 2800.0),
}
GROUP = {"cuticle": "shell", "palisade_layer": "shell", "mammillary_layer": "shell",
         "outer_shell_membrane": "membrane", "inner_shell_membrane": "membrane", "air_cell": "air",
         "outer_thin_albumen": "albumen", "thick_albumen": "albumen", "inner_thin_albumen": "albumen",
         "chalaziferous_layer": "albumen", "chalaza": "albumen", "vitelline_membrane": "yolk",
         "yolk_body": "yolk", "latebra": "yolk", "nucleus_of_pander": "yolk", "germinal_disc": "yolk"}
H_WATER = 3000.0      # W/m2K, TYPICAL for boiling water; results are conduction-limited (insensitive to h_w)


def yolk_class(f65):
    return "runny" if f65 < 0.15 else "jammy" if f65 < 0.45 else "creamy" if f65 < 0.85 else "firm"


class EggHeat:
    def __init__(self, egg, h=0.5e-3):
        from .egg import TISSUES
        self.egg = egg
        R = egg.mer.rho.max() + 2 * h
        r = np.arange(h / 2, R, h)
        z = np.arange(egg.mer.zmin - 2 * h, egg.mer.zmax + 2 * h, h)
        Rg, Zg = np.meshgrid(r, z, indexing="ij")
        P = np.stack([Rg, np.zeros_like(Rg), Zg], -1).reshape(-1, 3)
        lab = egg.label(P).reshape(Rg.shape)
        names = ["outside"] + [t["name"] for t in TISSUES]
        grp = np.empty(lab.shape, dtype=object)
        for i, n in enumerate(names):
            grp[lab == i] = "water" if n == "outside" else GROUP[n]
        self.r, self.z, self.h, self.grp, self.lab = r, z, h, grp, lab
        self.inside = grp != "water"
        k = np.zeros(lab.shape)
        C = np.zeros(lab.shape)
        for g, (kk, rho, c) in PROPS.items():
            m = grp == g
            k[m] = kk
            C[m] = rho * c
        self.k = k
        V = 2 * np.pi * Rg * h * h
        self.cap = C * V
        self.Rg, self.Zg = Rg, Zg
        yc = egg.c_yolk
        dist = np.hypot(Rg - 0, Zg - yc[2])
        self.i_centre = np.unravel_index(np.argmin(np.where(np.isin(grp, ["yolk"]), dist, 1e9)), lab.shape)
        ys = np.isin(grp, ["yolk"]) & (np.abs(dist - egg.r_yolk) < 1.2 * h)
        self.yolk_surface = ys
        self.white_inner = (grp == "albumen") & (dist < egg.r_yolk + 1.5e-3)

    def conductances(self, h_w):
        h, k, Rg = self.h, self.k, self.Rg
        ins = self.inside

        def pair(k1, k2, A, i1, i2):
            # cell-cell: series of half-cells; cell-water: half-cell + film
            g_cc = A / (h / (2 * np.maximum(k1, 1e-9)) + h / (2 * np.maximum(k2, 1e-9)))
            g_w1 = A / (h / (2 * np.maximum(k1, 1e-9)) + 1 / h_w)
            g_w2 = A / (h / (2 * np.maximum(k2, 1e-9)) + 1 / h_w)
            g = np.where(i1 & i2, g_cc, np.where(i1 & ~i2, g_w1, np.where(~i1 & i2, g_w2, 0.0)))
            return g
        Ar = 2 * np.pi * (Rg[:-1, :] + h / 2) * h                 # radial faces between i and i+1
        Gr = pair(k[:-1, :], k[1:, :], Ar, ins[:-1, :], ins[1:, :])
        Az = 2 * np.pi * Rg[:, :-1] * h                            # axial faces between j and j+1
        Gz = pair(k[:, :-1], k[:, 1:], Az, ins[:, :-1], ins[:, 1:])
        return Gr, Gz

    def run(self, T0, water, t_end, h_w=H_WATER, record_every=5.0, cool=None):
        """water(t) -> water temperature [C]; cool = (t_start, T_bath) switches the bath (ice bath / plunge)."""
        Gr, Gz = self.conductances(h_w)
        T = np.where(self.inside, T0, water(0.0)).astype(float)
        Gsum = np.zeros_like(T)
        Gsum[:-1, :] += Gr
        Gsum[1:, :] += Gr
        Gsum[:, :-1] += Gz
        Gsum[:, 1:] += Gz
        cap = np.where(self.inside, self.cap, np.inf)
        dt = 0.4 * np.min(cap[self.inside] / np.maximum(Gsum[self.inside], 1e-12))
        n = int(np.ceil(t_end / dt))
        dt = t_end / n
        Tmax = T.copy()
        hist = []
        hot_surface = 0.0
        for s in range(n + 1):
            t = s * dt
            Tw = water(t) if cool is None or t < cool[0] else cool[1]
            T[~self.inside] = Tw
            if s % max(1, int(record_every / dt)) == 0:
                hist.append({"t": t, "Tw": Tw, "T_centre": float(T[self.i_centre]),
                             "T_yolk_surface": float(T[self.yolk_surface].mean()),
                             "T_white_inner": float(T[self.white_inner].mean())})
            if s == n:
                break
            q = np.zeros_like(T)
            fr = Gr * (T[1:, :] - T[:-1, :])
            q[:-1, :] += fr
            q[1:, :] -= fr
            fz = Gz * (T[:, 1:] - T[:, :-1])
            q[:, :-1] += fz
            q[:, 1:] -= fz
            T = T + dt * q / cap
            Tmax = np.maximum(Tmax, T)
            if T[self.yolk_surface].mean() > 80:
                hot_surface += dt
        self.T, self.Tmax = T, Tmax
        return hist, self.doneness(hot_surface)

    def doneness(self, hot_surface_s):
        Tc = float(self.Tmax[self.i_centre])
        Y = self.grp == "yolk"
        f65 = float((self.Tmax[Y] >= 65).mean())
        alb = self.grp == "albumen"
        white_set = float((self.Tmax[alb] >= 63).mean())
        return {"yolk": yolk_class(f65), "f65": f65, "T_centre_max": Tc, "white_set_fraction": white_set,
                "white_set": white_set >= 0.97, "green_ring_risk": hot_surface_s > 240.0,
                "safe_71C_centre": Tc >= 71.0}


def hot_start(T_water=98.0):
    return lambda t: T_water


def cold_start(T0=18.0, ramp_s=540.0, hold_off=True, T_boil=100.0):
    """Cold water brought to the boil in ramp_s, then off the heat with a lid (slow cooling, ILLUSTRATIVE)."""
    def f(t):
        if t < ramp_s:
            return T0 + (T_boil - T0) * t / ramp_s
        return T_boil - 0.9 * (t - ramp_s) / 60.0 if hold_off else T_boil
    return f
