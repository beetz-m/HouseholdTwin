"""Lifetime of a hen egg: ageing at its storage temperature (moisture and CO2 loss through the shell), what the ageing
does to the egg's quality and to what the robot can observe (air cell, specific gravity, the float test), and the
freshness rules a plan needs (grades, best-before, peelability after boiling).

Register entries (food/params.py EGG, 'freshness and ageing'): ageing_table (SOURCED, pan2026: weight loss, yolk
index, thick-albumen height, Haugh unit and albumen pH at 4 C and 25 C over 30 days), ageing_law and
air_cell_from_mass_loss and float_test_model (DERIVED), hu_fresh / yi_fresh / ph_fresh / lay_SG (SOURCED-VERIFY means,
assumed spread), ageing_rate_factor (ILLUSTRATIVE), air_cell_grades_usda (SOURCED), best_before_days and
peel_ph_threshold (SOURCED-VERIFY), peel_ph_width (ILLUSTRATIVE).

Ageing law: for each quantity q in {weight loss, thick-albumen height, yolk index, albumen pH} the change since lay
  follows the pan2026 curve of the storage temperature (between 4 and 25 C the two curves blended linearly in T;
  outside, the nearer curve with time scaled by the weight-loss Q10); past day 30 the last segment's slope continues.
  The egg's ageing_rate_factor scales its time. After a temperature change each quantity continues from the
  equivalent age on the new curve (earliest age with the same change), so histories compose. The Haugh unit is
  computed from the thick-albumen height and the current mass (usda_egg formula), as the industry does.
Air cell: the lens at lay (1.5 mm deep) plus the lost water volume, as a cap at the blunt end of the egg's own
  Hügelschäffer shell (food/egg.py); the cross-check against Grashorn's anchors is in scripts/egg_lifetime_check.py.
Float test: in water the net weight W - B acts at c_eff = (m c_m - rho_w V c_b) / (m - rho_w V); the air cell pulls
  c_m to the pointed end, and as m - rho_w V shrinks with age, c_eff runs out along the axis: the egg rests at the tilt
  that minimises the height of c_eff above the bottom -- lies flat, tilts, stands on its pointed end, floats (SG < 1).

All functions are vectorised over particles (numpy arrays of the register variables), so the belief
(food/egg_belief.py) and the emulator share them with the truth.
"""

from dataclasses import dataclass, field

import numpy as np

from . import params as PR

E = PR.EGG
TAB = E["ageing_table"][1]
DAYS = np.array(TAB["days"], float)
QUANT = ("weight_loss_pct", "albumen_height_mm", "yolk_index", "albumen_pH")
T_LO, T_HI = 4.0, 25.0
RHO_W = 1000.0                       # kg/m3, water (lost moisture; float test)
AIR0_MM = 1.5                        # air-cell depth at lay (food/egg.py)
LIMITS = {"weight_loss_pct": (0.0, 25.0), "albumen_height_mm": (0.8, 12.0), "yolk_index": (0.12, 0.50),
          "albumen_pH": (7.0, 9.7)}  # pH ceiling ~9.7 (stadelman_ph)


# %% ageing
def _curve(T, q):
    return np.array(TAB[T][q], float)


def _delta(T, q, t):
    """Change since lay of quantity q after t days at constant T (pan2026 curves; between 4 and 25 C the two curves
    are blended linearly in T; outside, the nearer curve with its time scaled by the weight-loss Q10)."""
    t = np.asarray(t, float)
    c4, c25 = _curve(T_LO, q), _curve(T_HI, q)
    d4, d25 = c4 - c4[0], c25 - c25[0]

    def ev(d, tt):
        s_end = (d[-1] - d[-2]) / (DAYS[-1] - DAYS[-2])
        return np.where(tt <= DAYS[-1], np.interp(tt, DAYS, d), d[-1] + s_end * (tt - DAYS[-1]))
    if T < T_LO:
        return ev(d4, t * Q10 ** ((T - T_LO) / 10.0))
    if T > T_HI:
        return ev(d25, t * Q10 ** ((T - T_HI) / 10.0))
    u = (T - T_LO) / (T_HI - T_LO)
    return (1 - u) * ev(d4, t) + u * ev(d25, t)


def _wl_q10():
    w4, w25 = TAB[T_LO]["weight_loss_pct"][-1], TAB[T_HI]["weight_loss_pct"][-1]
    return float(np.clip((w25 / w4) ** (10.0 / (T_HI - T_LO)), 1.0, 4.0))


Q10 = _wl_q10()


def _equivalent_age(T, q, dq, t_max=120.0):
    """Earliest age on the T curve whose change since lay reaches dq (remap after a temperature change)."""
    tt = np.linspace(0.0, t_max, 961)
    d = _delta(T, q, tt)
    sgn = 1.0 if d[-1] >= 0 else -1.0
    env = np.maximum.accumulate(sgn * d)
    x = sgn * np.asarray(dq, float)
    k = np.searchsorted(env, x, side="left")
    return tt[np.clip(k, 0, len(tt) - 1)]


@dataclass
class EggState:
    """Freshness state of one egg or of a particle set (arrays): the age, and per quantity the equivalent age on the
    curve of the current storage temperature (so that histories compose: fridge -> counter -> fridge)."""
    age_d: np.ndarray
    fresh: dict
    tau: dict
    T: float = 4.0

    @classmethod
    def at_lay(cls, hu_fresh, yi_fresh, ph_fresh, mass0_g, T=4.0):
        hu, yi, ph, m0 = (np.atleast_1d(np.asarray(v, float)) for v in (hu_fresh, yi_fresh, ph_fresh, mass0_g))
        fresh = {"weight_loss_pct": np.zeros_like(hu), "albumen_height_mm": thick_height_for_hu(hu, m0),
                 "yolk_index": yi.copy(), "albumen_pH": ph.copy()}
        return cls(np.zeros_like(hu), fresh, {q: np.zeros_like(hu) for q in QUANT}, float(T))

    def copy(self):
        return EggState(self.age_d.copy(), {k: v.copy() for k, v in self.fresh.items()},
                        {k: v.copy() for k, v in self.tau.items()}, self.T)

    def value(self, q):
        lo, hi = LIMITS[q]
        return np.clip(self.fresh[q] + _delta(self.T, q, self.tau[q]), lo, hi)

    weight_loss_pct = property(lambda self: self.value("weight_loss_pct"))
    albumen_height_mm = property(lambda self: self.value("albumen_height_mm"))
    yolk_index = property(lambda self: self.value("yolk_index"))
    albumen_pH = property(lambda self: self.value("albumen_pH"))

    def advance(self, days, T_C, factor=1.0):
        """Age by `days` at temperature T_C; the egg's rate factor (scalar or per-particle array) scales time."""
        T_C = float(T_C)
        if abs(T_C - self.T) > 1e-9:                     # remap every quantity onto the new temperature's curve
            for q in QUANT:
                dq = _delta(self.T, q, self.tau[q])
                self.tau[q] = _equivalent_age(T_C, q, dq)
            self.T = T_C
        for q in QUANT:
            self.tau[q] = self.tau[q] + days * np.asarray(factor, float)
        self.age_d = self.age_d + days
        return self


def haugh_unit(h_mm, w_g):
    """usda_egg: HU = 100 log10(h - 1.7 w^0.37 + 7.6)."""
    return 100 * np.log10(np.maximum(np.asarray(h_mm) - 1.7 * np.asarray(w_g) ** 0.37 + 7.6, 1e-3))


def thick_height_for_hu(hu, w_g):
    return 10 ** (np.asarray(hu) / 100) + 1.7 * np.asarray(w_g) ** 0.37 - 7.6


# %% geometry of the shell (vectorised Hügelschäffer, food/egg.py; frame: z up = blunt end, origin mid-length)
def meridian(L_mm, SI, w_mm, n=160):
    """rho(z) of the outer shell for arrays of length L (mm), shape index B/L and asymmetry w (mm). Returns z (n,)
    as fractions of L (from -L/2 at the pointed pole to +L/2 at the blunt pole) and rho (N, n) in metres."""
    L = np.atleast_1d(np.asarray(L_mm, float))[:, None] * 1e-3
    B = np.atleast_1d(np.asarray(SI, float))[:, None] * L
    w = np.atleast_1d(np.asarray(w_mm, float))[:, None] * 1e-3
    s = np.linspace(-0.5, 0.5, n)[None, :]
    x = -s * L                                            # z = -x: the blunt end (x < 0) is up
    y = B / 2 * np.sqrt(np.clip((L ** 2 - 4 * x ** 2) / (L ** 2 + 8 * w * x + 4 * w ** 2), 0, None))
    return s[0], y, L[:, 0]


def shape_props(L_mm, SI, w_mm, n=160):
    """Per egg: outer volume V (m3), centroid z_b (m), and the cap-volume table of the blunt end V_cap(depth)."""
    s, rho, L = meridian(L_mm, SI, w_mm, n)
    z = s[None, :] * L[:, None]
    dz = np.gradient(z, axis=1)
    A = np.pi * rho ** 2
    V = (A * dz).sum(1)
    zb = (A * z * dz).sum(1) / V
    # cap volumes from the blunt pole downwards (depth grid in metres)
    capV = np.cumsum((A * dz)[:, ::-1], axis=1)
    depth = (L[:, None] / 2 - z[:, ::-1])
    return dict(V=V, zb=zb, depth=depth, capV=capV, L=L, z=z, rho=rho)


def air_cell_depth_mm(props, V_air_m3):
    """Depth of a blunt-end cap of volume V_air (the air cell as a flat-bottomed cap; DERIVED approximation)."""
    out = np.empty(len(props["V"]))
    for i in range(len(out)):
        out[i] = np.interp(V_air_m3[i], props["capV"][i], props["depth"][i]) * 1e3
    return out


def cap_volume(props, depth_mm):
    out = np.empty(len(props["V"]))
    for i in range(len(out)):
        out[i] = np.interp(depth_mm[i] * 1e-3, props["depth"][i], props["capV"][i])
    return out


def cap_centroid_z(props, depth_mm):
    """z of the centroid of the blunt cap of the given depth (m)."""
    out = np.empty(len(props["V"]))
    for i in range(len(out)):
        zt = props["L"][i] / 2
        zz = np.linspace(zt - depth_mm[i] * 1e-3, zt, 40)
        rr = np.interp(zz, props["z"][i], props["rho"][i])
        a = np.pi * rr ** 2
        out[i] = (a * zz).sum() / max(a.sum(), 1e-18)
    return out


@dataclass
class Observables:
    mass_g: np.ndarray
    SG: np.ndarray
    air_cell_mm: np.ndarray
    haugh_unit: np.ndarray
    float_tilt_deg: np.ndarray        # 0 = lying flat, 90 = standing on the pointed end; NaN when it floats
    floats: np.ndarray


def observables(props, lay_SG, state):
    """What ageing does to the observable egg: mass, specific gravity, air cell, Haugh unit and the float posture."""
    V = props["V"]
    m0 = V * np.asarray(lay_SG) * 1000.0                  # kg at lay
    V_air0 = cap_volume(props, np.full(len(V), AIR0_MM))
    lost = m0 * state.weight_loss_pct / 100.0
    m = m0 - lost
    V_air = V_air0 + lost / RHO_W
    depth = air_cell_depth_mm(props, V_air)
    SG = m / (V * RHO_W)
    hu = haugh_unit(state.albumen_height_mm, m * 1000)
    tilt, floats = float_posture(props, m, V_air, depth)
    return Observables(m * 1000, SG, depth, hu, tilt, floats)


def float_posture(props, m_kg, V_air, depth_mm, n_tilt=91):
    """Resting tilt in water (DERIVED statics, register float_test_model). The contents are taken as homogeneous
    (yolk and albumen densities within 0.5 %), so the centre of mass is the centroid minus the air cell."""
    V, zb = props["V"], props["zb"]
    za = cap_centroid_z(props, depth_mm)
    rho0 = (m_kg / np.maximum(V - V_air, 1e-12))
    net = m_kg - RHO_W * V                                 # kg; > 0 sinks
    floats = net <= 0
    # c_eff on the axis, offset from the centroid (negative = towards the pointed end)
    d = -(rho0 * V_air * (za - zb)) / np.where(floats, 1.0, net)
    tilt = np.full(len(V), np.nan)
    th = np.radians(np.linspace(-90, 90, n_tilt))         # th > 0: the blunt end down (the down vector has +z)
    for i in np.nonzero(~floats)[0]:
        z, r = props["z"][i] - zb[i], props["rho"][i]
        # body points in its own meridian plane (both sides), relative to the centroid
        P = np.concatenate([np.stack([z, r], 1), np.stack([z, -r], 1)])
        # height of c_eff above the bottom for each tilt: support of the body in the 'down' direction
        down = np.stack([np.sin(th), -np.cos(th)], 1)      # in (axis, radial) coordinates; th>0: pointed end down
        ce = np.array([d[i], 0.0])
        h = (((P[None, :, :] - ce[None, None, :]) * down[:, None, :]).sum(2)).max(1)
        tilt[i] = -np.degrees(th[int(np.argmin(h))])      # reported: > 0 = pointed end down (blunt end up)
    return tilt, floats


def float_class(tilt_deg, floats):
    t = np.asarray(tilt_deg)
    out = np.where(floats, "floats", np.where(np.abs(t) < 20, "lies flat", np.where(np.abs(t) < 60, "tilts",
                                                                                     "stands on its end")))
    return out


# %% freshness rules
def usda_grade(air_mm, hu):
    """Interior grade from the air cell (usda_std) and the Haugh unit (usda_egg: AA >= 72, A >= 60)."""
    a = np.where(air_mm <= 3.2, 2, np.where(air_mm <= 4.8, 1, 0))
    h = np.where(hu >= 72, 2, np.where(hu >= 60, 1, 0))
    g = np.minimum(a, h)
    return np.array(["B", "A", "AA"])[g]


def eu_class(air_mm):
    """eu589_2008: Class A air space <= 6 mm ('extra' <= 4 mm)."""
    return np.where(air_mm <= 4.0, "A extra", np.where(air_mm <= 6.0, "A", "below A"))


def past_best_before(age_d):
    return np.asarray(age_d) > E["best_before_days"][1]


def p_clean_peel(ph):
    """P(the shell comes off a hard-boiled egg cleanly) as a logistic in albumen pH (peel_ph: easy at 8.7-8.9)."""
    return 1.0 / (1.0 + np.exp(-(np.asarray(ph) - E["peel_ph_threshold"][1]) / E["peel_ph_width"][1]))


def yolk_index_ok_for_frying(yi):
    """A domed yolk for fried / poached eggs (ILLUSTRATIVE threshold on the yolk index)."""
    return np.asarray(yi) >= 0.36


# %% one instance (truth side)
@dataclass
class EggClock:
    """Freshness clock of one egg instance (register draw), driven by its storage history."""
    p: dict
    props: dict
    state: EggState
    history: list = field(default_factory=list)

    @classmethod
    def of(cls, p, lay=True):
        props = shape_props(p["length_L"], p["shape_index_B_over_L"], p["asymmetry_w"])
        m0 = props["V"] * p.get("lay_SG", 1.087) * 1e6
        st = EggState.at_lay(p.get("hu_fresh", 81.3), p.get("yi_fresh", 0.43), p.get("ph_fresh", 7.9), m0)
        c = cls(p, props, st)
        if not lay:                                       # the register's age and storage temperature so far
            c.advance(p["age_days"] * 24.0, "fridge" if p["storage_temp_C"] < 10 else "counter")
        return c

    def advance(self, hours, storage="fridge"):
        T = {"fridge": 6.0, "counter": 22.0, "warm": 25.0, "cellar": 12.0}.get(storage, storage)   # register storage_temp_C
        self.state.advance(hours / 24.0, float(T), self.p.get("ageing_rate_factor", 1.0))
        self.history.append((hours, storage))
        return self

    def observe(self):
        o = observables(self.props, np.atleast_1d(self.p.get("lay_SG", 1.087)), self.state)
        return {k: (float(v[0]) if np.ndim(v) else v) for k, v in o.__dict__.items()}

    def summary(self):
        o = self.observe()
        s = self.state
        return dict(age_d=float(s.age_d[0]), mass_g=o["mass_g"], SG=o["SG"], air_cell_mm=o["air_cell_mm"],
                    haugh_unit=o["haugh_unit"], yolk_index=float(s.yolk_index[0]), albumen_pH=float(s.albumen_pH[0]),
                    float=str(float_class(np.atleast_1d(o["float_tilt_deg"]), np.atleast_1d(o["floats"]))[0]),
                    float_tilt_deg=o["float_tilt_deg"], usda_grade=str(usda_grade(o["air_cell_mm"], o["haugh_unit"])),
                    eu_class=str(eu_class(o["air_cell_mm"])), past_best_before=bool(past_best_before(s.age_d)[0]),
                    p_clean_peel=float(p_clean_peel(s.albumen_pH)[0]))


def best_day(p, storage="fridge", use="boil_and_peel", horizon_days=35, step_h=24.0):
    """Day (from lay) that maximises the use's quality within best-before: boil_and_peel -> P(clean peel) x (not past
    best-before); fry -> yolk dome and HU (freshest is best)."""
    c = EggClock.of(p)
    best, out = None, []
    for k in range(int(horizon_days * 24 / step_h) + 1):
        s = c.summary()
        if use == "boil_and_peel":
            q = s["p_clean_peel"] * (not s["past_best_before"])
        else:
            q = (s["haugh_unit"] / 90.0) * float(yolk_index_ok_for_frying(s["yolk_index"])) * (not s["past_best_before"])
        out.append((s["age_d"], q))
        if best is None or q > best[1] + 1e-9:
            best = (s["age_d"], q)
        c.advance(step_h, storage)
    return best, out
