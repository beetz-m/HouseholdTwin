"""Probabilistic SDT node of a hen egg: a particle filter over the egg's hidden parameters and its freshness state.

State: one particle = one complete candidate egg -- the register's random parameters (shape, shell thickness and
modulus, break forces, shell friction, yolk fraction, specific gravity at lay, fresh Haugh unit / yolk index / pH,
ageing rate factor, age since lay, storage so far) plus a micro-crack flag and the freshness state of
food/egg_lifetime.py (weight loss, thick-albumen height, yolk index, albumen pH as equivalent ages per quantity).
Derived per particle (so dependencies carry information): mass, specific gravity, air-cell depth, Haugh unit, float
posture, direction-dependent crack force, Reissner contact stiffness, the slip force of a grip.
Prior: the register (food/params.py EGG) -- the population of retail hen eggs. Supersedes the grip-only
food/sdt.EggBelief, whose H8 queries (P(slip), P(rupture), corridor, best grip) are methods here.
Prediction: ageing at the storage temperature (food/egg_lifetime.py; each particle at its own rate factor).
Measurement models (noise in the register's sensor_* entries, ILLUSTRATIVE unless stated):
  carton_bbd      best-before printed on the carton -> age (laid 28 d before, register best_before_days; sd 1 d)
  storage         where the egg has been kept (fridge / counter): conditions the history
  mass_g          kitchen scale (sd 0.3 g) + 1 % volume-model error
  length_mm       calliper (sd 0.3 mm); photo_size: length and shape index from a photo with a scale reference;
                  photo_shape: shape index only; photo_crack: a dark line seen (P 0.2 | micro-crack, 0.02 | intact)
  candling        air-cell depth (sd 0.6 mm) and a crack seen / not seen (P(seen | micro-crack) 0.6, false 0.02)
  float_test      resting tilt in water (sd 8 deg) or floating (food/egg_lifetime.float_posture)
  grip            held at force F without slip / without a crack (soft constraints, force reading sd 10 %)
  strike          strike outcome at speed v on a tool (food/egg_crack.py physics per particle): intact / check /
                  cracked open / shattered
  opened          thick-albumen height (sd 0.4 mm) and yolk index (sd 0.02) of the opened egg
  peel            a hard-boiled egg peeled clean / torn (food/egg_lifetime.p_clean_peel)

    b = EggBelief.prior(n=4000, seed=0)
    b.observe(Observation("carton_bbd", {"days_left": 18}, 0.0, "carton"), rng)
    b.predict(hours=72, rng=rng, storage="fridge")
"""

import json
import math
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from . import egg_crack as EC
from . import egg_lifetime as EL
from . import params as PR
from .dynamics import SURFACES, series

E = PR.EGG
RANDOM = ["length_L", "shape_index_B_over_L", "asymmetry_w", "shell_thickness", "shell_E", "F_break_equatorial",
          "F_break_axial", "friction_shell_rubber", "yolk_mass_fraction", "lay_SG", "hu_fresh", "yi_fresh",
          "ph_fresh", "ageing_rate_factor", "age_days"]
UNITS = {"length_L": "mm", "shape_index_B_over_L": "1", "asymmetry_w": "mm", "shell_thickness": "mm", "shell_E": "GPa",
         "F_break_equatorial": "N", "F_break_axial": "N", "friction_shell_rubber": "1", "yolk_mass_fraction": "1",
         "lay_SG": "1", "hu_fresh": "HU", "yi_fresh": "1", "ph_fresh": "1", "ageing_rate_factor": "1",
         "age_days": "d", "mass_g": "g", "SG": "1", "air_cell_mm": "mm", "haugh_unit": "HU", "yolk_index": "1",
         "albumen_pH": "1", "weight_loss_pct": "%", "float_tilt_deg": "deg", "F_crack_equator_N": "N",
         "F_crack_pole_N": "N", "microcracked": "1", "p_clean_peel": "1", "fridge": "1"}
MICRO_PRIOR = 0.03                      # ILLUSTRATIVE (food/sdt.EggBelief): eggs arriving micro-cracked
NU = E["shell_poisson"][1]


@dataclass
class Observation:
    kind: str
    value: dict
    time_h: float
    source: str
    note: str = ""


def _sd(key):
    return float(E[key][1])


@dataclass
class EggBelief:
    n: int
    particles: dict
    weights: np.ndarray
    state: EL.EggState
    props: dict
    time_h: float = 0.0
    history: list = field(default_factory=list)
    log_evidence: float = 0.0
    _cache: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- construction
    @classmethod
    def prior(cls, n=4000, seed=0, storage=None):
        """Particles from the register (the population of retail eggs); each particle aged from lay to its age at its
        storage temperature (fridge 6 C / counter 22 C in the register; `storage` fixes it when known)."""
        draws = [PR.sample("egg", seed * 100000 + i) for i in range(n)]
        P = {k: np.array([d[k] for d in draws], float) for k in RANDOM}
        P["fridge0"] = np.array([d["storage_temp_C"] < 10 for d in draws]) if storage is None else \
            np.full(n, storage == "fridge")              # where it was kept before the robot got it
        P["fridge"] = P["fridge0"].copy()                 # where it is kept now
        rng = np.random.default_rng(seed + 17)
        P["microcracked"] = rng.random(n) < MICRO_PRIOR
        b = cls(n, P, np.full(n, 1.0 / n), None, None)
        b._rebuild_state()
        return b

    def _rebuild_state(self):
        """Geometry tables and the freshness state from the particles' parameters (after a resample with jitter)."""
        P = self.particles
        P["fridge"] = P["fridge0"].copy()
        self.props = EL.shape_props(P["length_L"], P["shape_index_B_over_L"], P["asymmetry_w"])
        m0 = self.props["V"] * P["lay_SG"] * 1e6
        st = EL.EggState.at_lay(P["hu_fresh"], P["yi_fresh"], P["ph_fresh"], m0, T=4.0)
        for fridge, T in ((True, 6.0), (False, 22.0)):
            sel = P["fridge"] == fridge
            if not sel.any():
                continue
            for q in EL.QUANT:                           # time on the curve = age x the particle's rate factor
                st.tau[q][sel] = P["age_days"][sel] * P["ageing_rate_factor"][sel]
            st.age_d[sel] = P["age_days"][sel]
        st.T = None                                      # mixed: per-particle temperatures (see _T)
        self.state = st
        self._cache = {}

    def _T(self):
        return np.where(self.particles["fridge"], 6.0, 22.0)

    # ---------------------------------------------------------------- derived quantities
    def _values(self, q):
        """Current value of a freshness quantity per particle (each particle on its own storage curve)."""
        out = np.empty(self.n)
        for fridge, T in ((True, 6.0), (False, 22.0)):
            sel = self.particles["fridge"] == fridge
            if sel.any():
                lo, hi = EL.LIMITS[q]
                out[sel] = np.clip(self.state.fresh[q][sel] + EL._delta(T, q, self.state.tau[q][sel]), lo, hi)
        return out

    def derived(self, posture=False):
        key = ("derived", posture)
        if key in self._cache:
            return self._cache[key]
        P = self.particles
        wl = self._values("weight_loss_pct")
        V = self.props["V"]
        m0 = V * P["lay_SG"] * 1000.0
        V_air0 = EL.cap_volume(self.props, np.full(self.n, EL.AIR0_MM))
        lost = m0 * wl / 100.0
        m = m0 - lost
        V_air = V_air0 + lost / EL.RHO_W
        air = EL.air_cell_depth_mm(self.props, V_air)
        h = self._values("albumen_height_mm")
        D = dict(mass_g=m * 1000, SG=m / (V * EL.RHO_W), air_cell_mm=air, weight_loss_pct=wl,
                 haugh_unit=EL.haugh_unit(h, m * 1000), yolk_index=self._values("yolk_index"),
                 albumen_pH=self._values("albumen_pH"), F_crack_equator_N=self.F_crack(np.pi / 2),
                 F_crack_pole_N=self.F_crack(0.0))
        D["p_clean_peel"] = EL.p_clean_peel(D["albumen_pH"])
        if posture:
            tilt, floats = EL.float_posture(self.props, m, V_air, air)
            D["float_tilt_deg"], D["floats"] = tilt, floats
        self._cache[key] = D
        return D

    def F_crack(self, polar):
        """Direction-dependent quasi-static crack force per particle (food/egg.py break_force; micro-crack x 0.6)."""
        P = self.particles
        s = np.sin(polar) ** 2
        F = (s * P["F_break_equatorial"] + (1 - s) * P["F_break_axial"]) * (P["shell_thickness"] / 0.35) ** 2
        return F * np.where(P["microcracked"], 1 - E["microcrack_strength_loss"][1], 1.0)

    def local_radius(self, polar):
        """Mean principal radius of curvature of the shell at a polar angle (0 = blunt pole), per particle (m)."""
        z, rho = self.props["z"], self.props["rho"]
        n = z.shape[1]
        i = int(np.clip((1 - polar / np.pi) * (n - 3), 1, n - 3))   # meridian runs pointed -> blunt
        p0 = np.stack([rho[:, i - 1], z[:, i - 1]], 1)
        p1 = np.stack([rho[:, i], z[:, i]], 1)
        p2 = np.stack([rho[:, i + 1], z[:, i + 1]], 1)
        a, b, c = (np.linalg.norm(p1 - p0, axis=1), np.linalg.norm(p2 - p1, axis=1), np.linalg.norm(p2 - p0, axis=1))
        u_, v_ = p1 - p0, p2 - p0
        area2 = np.abs(u_[:, 0] * v_[:, 1] - u_[:, 1] * v_[:, 0])
        R1 = a * b * c / (2 * area2 + 1e-18)
        R2 = p1[:, 0] / np.maximum(np.abs((p2 - p0)[:, 1] / c), 1e-6)
        return np.minimum(2 / (1 / R1 + 1 / R2), 0.2)

    def contact_stiffness(self, polar):
        P = self.particles
        t = P["shell_thickness"] * 1e-3
        return 4 * P["shell_E"] * 1e9 * t ** 2 / (self.local_radius(polar) * math.sqrt(3 * (1 - NU ** 2)))

    def F_slip(self, a_max=2.0, n_pads=2):
        """Grip force per pad needed to hold the egg against gravity + acceleration (register F_slip)."""
        D = self.derived()
        return D["mass_g"] * 1e-3 * (9.81 + a_max) / (n_pads * self.particles["friction_shell_rubber"])

    # ---------------------------------------------------------------- H8 queries (food/sdt.EggBelief, generalised)
    def probs(self, F_grip, polar=np.pi / 2, a_max=2.0):
        w = self.weights
        return dict(p_slip=float(np.sum(w * (F_grip < self.F_slip(a_max)))),
                    p_rupture=float(np.sum(w * (F_grip >= self.F_crack(polar)))))

    def admissible(self, F_grip, polar=np.pi / 2, a_max=2.0, eps_s=0.05, eps_r=0.05):
        p = self.probs(F_grip, polar, a_max)
        return p["p_slip"] < eps_s and p["p_rupture"] < eps_r, p

    def corridor(self, polar=np.pi / 2, a_max=2.0, eps_s=0.05, eps_r=0.05, F_range=(0.2, 40.0)):
        Fs = np.linspace(*F_range, 400)
        ok = [F for F in Fs if self.admissible(F, polar, a_max, eps_s, eps_r)[0]]
        return (float(min(ok)), float(max(ok))) if ok else None

    # ---------------------------------------------------------------- strike outcome model (food/egg_crack.py)
    def strike_ratio(self, v, tool, polar=np.pi / 2):
        """F_peak / F_crack per particle for a strike at speed v on a tool ({kind, surface, edge_radius})."""
        P = self.particles
        D = self.derived()
        m_eff = D["mass_g"] * 1e-3 * E["impact_effective_mass_fraction"][1] + EC.P["coupled_mass"]
        k_s = self.contact_stiffness(polar)
        k_surf = SURFACES[tool.get("surface", "bowl_rim_ceramic")][0]
        F = v * np.sqrt(series(k_s, k_surf) * m_eff)
        Fc = self.F_crack(polar) * E["impact_rate_factor"][1]
        if tool["kind"] == "rim":
            Fc = Fc * np.sqrt(max(tool.get("edge_radius", 2e-3), 0.2e-3) / 5e-3)
        return F / Fc

    @staticmethod
    def strike_probs(x, kind):
        """P(intact, check, cracked_open, shattered) for ratio x (the categorical version of egg_crack.strike; a
        micro-crack counts as 'intact' to an observer)."""
        band = E["microcrack_band"][1]
        tear, shat = EC.P["tear_ratio"][kind], EC.P["shatter_ratio"][kind]
        x = np.asarray(x, float)
        cracked = x >= 1.0
        p_sh = (x >= shat).astype(float)
        if kind == "rim":
            p_open_given_crack = np.where(x >= tear, 1.0, np.clip(0.6 + 0.4 * (x - 1), 0, 1))
        else:
            p_open_given_crack = (x >= tear).astype(float)
        p_open = cracked * (1 - p_sh) * p_open_given_crack
        p_check = cracked * (1 - p_sh) * (1 - p_open_given_crack)
        p_int = 1.0 - cracked
        _ = band
        return np.stack([p_int, p_check, p_open, p_sh], 1)

    STRIKE_STATES = ("intact", "check", "cracked_open", "shattered")

    def strike_outcome_probs(self, v, tool, polar=np.pi / 2, force_sd=0.10):
        """Belief-weighted outcome probabilities of a strike; force uncertainty smooths the thresholds."""
        x = self.strike_ratio(v, tool, polar)
        acc = 0
        for z in (-1.0, 0.0, 1.0):                         # 3-point quadrature over the force reading error
            acc = acc + (0.25 if z else 0.5) * self.strike_probs(x * (1 + force_sd * z), tool["kind"])
        return acc

    # ---------------------------------------------------------------- bookkeeping
    def bounds(self, k):
        spec = E[k][1]
        if isinstance(spec, tuple) and spec[0] == "normal":
            return spec[3], spec[4]
        if isinstance(spec, tuple) and spec[0] == "uniform":
            return spec[1], spec[2]
        return -np.inf, np.inf

    def ess(self):
        return float(1.0 / np.sum(self.weights ** 2))

    def resample(self, rng, a=0.9):
        """Systematic resampling with Liu-West kernel shrinkage on the continuous parameters: each copy is pulled
        towards the weighted mean by a and jittered with sd sqrt(1 - a^2) x the weighted sd, which keeps the
        posterior's mean and variance while restoring diversity (a small fixed jitter collapsed the dimensions the
        observations do not inform, e.g. the crack force after a best-before reading). Values are clipped to the
        register bounds; the freshness state is rebuilt from lay and the seen storage history (_replay)."""
        u = (rng.random() + np.arange(self.n)) / self.n
        idx = np.minimum(np.searchsorted(np.cumsum(self.weights), u), self.n - 1)
        h = math.sqrt(1 - a * a)
        for k in RANDOM:
            x = self.particles[k]
            mean = float(np.sum(self.weights * x))
            sd = math.sqrt(max(float(np.sum(self.weights * (x - mean) ** 2)), 0.0))
            lo, hi = self.bounds(k)
            if k == "age_days":
                lo, hi = 0.0, 120.0
            if k == "ageing_rate_factor":
                lo, hi = 0.3, 3.0
            self.particles[k] = np.clip(a * x[idx] + (1 - a) * mean + rng.normal(0, h * sd, self.n), lo, hi)
        for k in ("fridge", "fridge0", "microcracked"):
            self.particles[k] = self.particles[k][idx]
        st = self.state
        self.state = EL.EggState(st.age_d[idx], {q: v[idx] for q, v in st.fresh.items()},
                                 {q: v[idx] for q, v in st.tau.items()}, None)
        # keep the freshness state consistent with the jittered parameters: rebuild from lay, then replay storage
        self._replay()
        self.weights = np.full(self.n, 1.0 / self.n)

    def _replay(self):
        """Rebuild the freshness state from lay: the part of the age before the belief started at the particle's
        prior storage, then the storage history the belief has seen (predict calls)."""
        P = self.particles
        pre = P["age_days"] - self.time_h / 24.0
        seen = [h for h in self.history if h.get("kind") == "predict"]
        saved = P["age_days"].copy()
        P["age_days"] = np.maximum(pre, 0.0)
        self._rebuild_state()
        P["age_days"] = saved
        for h in seen:
            self._age_states(h["value"]["hours"], h["value"]["storage"])
        self._cache = {}

    def _apply(self, lik, obs, rng):
        lik = np.clip(lik, 1e-300, None)
        w = self.weights * lik
        z = w.sum()
        self.log_evidence += math.log(max(z, 1e-300))
        self.weights = w / z
        rec = asdict(obs)
        rec["ess_after"] = self.ess()
        self.history.append(rec)
        self._cache = {}
        if self.ess() < self.n / 2:
            self.resample(rng)

    # ---------------------------------------------------------------- prediction
    def _age_states(self, hours, storage):
        T = {"fridge": 6.0, "counter": 22.0}.get(storage, storage)
        P = self.particles
        d = hours / 24.0
        for fridge, T0 in ((True, 6.0), (False, 22.0)):
            sel = P["fridge"] == fridge
            if not sel.any() or abs(T0 - float(T)) < 1e-9:  # same curve: no remap (a remap would jump back on plateaus)
                continue
            for q in EL.QUANT:                            # remap from the particle's curve onto the storage curve
                dq = EL._delta(T0, q, self.state.tau[q][sel])
                self.state.tau[q][sel] = EL._equivalent_age(float(T), q, dq)
        for q in EL.QUANT:
            self.state.tau[q] = self.state.tau[q] + d * P["ageing_rate_factor"]
        self.state.age_d = self.state.age_d + d
        P["fridge"] = np.full(self.n, float(T) < 10.0)
        if abs(float(T) - (6.0 if float(T) < 10 else 22.0)) > 1e-9:     # other temperatures: map to the nearest
            pass                                                         # register class (ILLUSTRATIVE)

    def predict(self, hours, rng=None, storage="fridge"):
        """Ageing over `hours` in storage (fridge 6 C / counter 22 C); every particle at its own rate factor."""
        self._age_states(hours, storage)
        self.particles["age_days"] = self.particles["age_days"] + hours / 24.0
        self.time_h += hours
        self._cache = {}
        self.history.append(dict(kind="predict", value=dict(hours=hours, storage=storage), time_h=self.time_h,
                                 source="ageing model (food/egg_lifetime.py, pan2026)", note="",
                                 ess_after=self.ess()))

    # ---------------------------------------------------------------- measurement models
    def observe(self, obs, rng):
        P, v, k = self.particles, obs.value, obs.kind
        if k == "carton_bbd":                             # best-before printed: laid <= 28 d before it (sd 1 d)
            age_pred = E["best_before_days"][1] - v["days_left"]
            lik = np.exp(-0.5 * ((P["age_days"] - age_pred) / v.get("sd", 1.0)) ** 2)
        elif k == "storage":                              # where it has been kept (the robot fetched it from there)
            lik = np.where(P["fridge0"] == (v["where"] == "fridge"), 1.0, 1e-6)
        elif k == "mass_g":
            sd = math.hypot(v.get("sd", _sd("sensor_scale_sd_g")), 0.01 * v["value"])
            lik = np.exp(-0.5 * ((self.derived()["mass_g"] - v["value"]) / sd) ** 2)
        elif k == "length_mm":
            lik = np.exp(-0.5 * ((P["length_L"] - v["value"]) / v.get("sd", _sd("sensor_calliper_sd_mm"))) ** 2)
        elif k == "photo_size":                           # length and shape index with a scale reference
            lik = np.exp(-0.5 * ((P["length_L"] - v["length_mm"]) / (_sd("sensor_photo_size_rel_sd") * v["length_mm"]))
                         ** 2 - 0.5 * ((P["shape_index_B_over_L"] - v["shape_index"]) / v.get("si_sd", _sd("sensor_photo_si_sd"))) ** 2)
        elif k == "photo_shape":                          # shape index only (no scale reference in the photo)
            lik = np.exp(-0.5 * ((P["shape_index_B_over_L"] - v["shape_index"]) / v.get("si_sd", _sd("sensor_photo_si_sd"))) ** 2)
        elif k == "photo_crack":                          # a dark line on the shell in a photo (ILLUSTRATIVE rates)
            lik = np.where(P["microcracked"], 0.2 if v["seen"] else 0.8, 0.02 if v["seen"] else 0.98)
        elif k == "candling":
            D = self.derived()
            lik = np.ones(self.n)
            if v.get("air_cell_mm") is not None:
                lik *= np.exp(-0.5 * ((D["air_cell_mm"] - v["air_cell_mm"]) /
                                      v.get("sd", _sd("sensor_candling_air_sd_mm"))) ** 2)
            if v.get("crack_seen") is not None:
                p_det = _sd("sensor_candling_crack_detect")
                lik *= np.where(P["microcracked"], p_det if v["crack_seen"] else 1 - p_det,
                                0.02 if v["crack_seen"] else 0.98)
        elif k == "float_test":
            D = self.derived(posture=True)
            if v.get("floats"):
                lik = np.where(D["floats"], 1.0, 1e-4)
            else:
                t = np.nan_to_num(D["float_tilt_deg"], nan=90.0)
                lik = np.where(D["floats"], 1e-4, np.exp(-0.5 * ((t - v["tilt_deg"]) /
                                                                  v.get("sd", _sd("sensor_float_tilt_sd_deg"))) ** 2))
        elif k == "grip":                                 # held at force F (per pad) through an acceleration
            F, polar, a = v["F_N"], v.get("polar", np.pi / 2), v.get("a_max", 2.0)
            s = _sd("sensor_force_rel_sd") * F
            ph = lambda x: 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))  # noqa: E731
            lik = np.ones(self.n)
            if v.get("slipped") is not None:
                p_hold = ph((F - self.F_slip(a)) / s)
                lik *= p_hold if not v["slipped"] else 1 - p_hold
            if v.get("cracked") is not None:
                p_cr = ph((F - self.F_crack(polar)) / s)
                lik *= p_cr if v["cracked"] else 1 - p_cr
        elif k == "strike":
            pr = self.strike_outcome_probs(v["v"], v["tool"], v.get("polar", np.pi / 2))
            lik = pr[:, self.STRIKE_STATES.index(v["outcome"])]
            if v["outcome"] != "intact":
                P["microcracked"] = np.zeros(self.n, bool)   # the shell is cracked now: the flag no longer applies
        elif k == "opened":
            D = self.derived()
            lik = np.ones(self.n)
            if v.get("albumen_height_mm") is not None:
                lik *= np.exp(-0.5 * ((self._values("albumen_height_mm") - v["albumen_height_mm"]) /
                                      v.get("h_sd", _sd("sensor_albumen_height_sd_mm"))) ** 2)
            if v.get("yolk_index") is not None:
                lik *= np.exp(-0.5 * ((D["yolk_index"] - v["yolk_index"]) /
                                      v.get("yi_sd", _sd("sensor_yolk_index_sd"))) ** 2)
        elif k == "peel":
            pc = self.derived()["p_clean_peel"]
            lik = pc if v["clean"] else 1 - pc
        else:
            raise ValueError(f"unknown observation kind {k}")
        self._apply(lik, obs, rng)

    # ---------------------------------------------------------------- summaries
    def summary(self, bins=24, posture=False):
        D = self.derived(posture=posture)
        allv = {**{k: self.particles[k] for k in RANDOM}, "microcracked": self.particles["microcracked"].astype(float),
                "fridge": self.particles["fridge"].astype(float),
                **{k: v for k, v in D.items() if k != "floats"}}
        out = {}
        for k, x in allv.items():
            x = np.nan_to_num(np.asarray(x, float), nan=90.0)
            o = np.argsort(x)
            cw = np.cumsum(self.weights[o])
            q = lambda p: float(x[o][min(np.searchsorted(cw, p), self.n - 1)])  # noqa: E731
            mean = float(np.sum(self.weights * x))
            sd = float(math.sqrt(max(np.sum(self.weights * (x - mean) ** 2), 0.0)))
            lo, hi = float(x.min()), float(x.max())
            edges = np.linspace(lo, hi if hi > lo else lo + 1e-9, bins + 1)
            hist, _ = np.histogram(x, edges, weights=self.weights)
            out[k] = dict(unit=UNITS.get(k, "1"), mean=mean, sd=sd, p05=q(0.05), p50=q(0.5), p95=q(0.95),
                          support=[lo, hi], hist_edges=edges.tolist(), hist_p=hist.tolist())
        return out

    def p(self, cond):
        """Belief probability of a boolean particle condition."""
        return float(np.sum(self.weights * np.asarray(cond, float)))

    def save(self, path):
        np.savez_compressed(path, weights=self.weights, time_h=self.time_h, history=json.dumps(self.history,
                                                                                               default=str),
                            log_evidence=self.log_evidence, **{f"p_{k}": v for k, v in self.particles.items()})

    @classmethod
    def load(cls, path):
        d = np.load(path)
        P = {k[2:]: d[k] for k in d.files if k.startswith("p_")}
        for k in ("fridge", "fridge0", "microcracked"):
            P[k] = P[k].astype(bool)
        b = cls(len(d["weights"]), P, d["weights"], None, None, float(d["time_h"]), json.loads(str(d["history"])),
                float(d["log_evidence"]))
        b._replay()
        return b

    def sdt_attributes(self, sidecar):
        S = self.summary(posture=True)
        A = {"sdt:belief:representation": f"particle filter, {self.n} particles (food/egg_belief.py)",
             "sdt:belief:particles": sidecar, "sdt:belief:time_h": self.time_h, "sdt:belief:ess": self.ess(),
             "sdt:belief:observations": json.dumps([f"{h['time_h']:.0f} h {h['kind']} {h.get('source', '')}"
                                                    for h in self.history]),
             "sdt:epistemicStatus": "posterior over register priors and observations"}
        for k, s in S.items():
            for f in ("mean", "sd", "p05", "p50", "p95"):
                A[f"sdt:belief:{k}:{f}"] = s[f]
            A[f"sdt:belief:{k}:unit"] = s["unit"]
            A[f"sdt:belief:{k}:support"] = list(map(float, s["support"]))
            A[f"sdt:belief:{k}:hist_edges"] = list(map(float, s["hist_edges"]))
            A[f"sdt:belief:{k}:hist_p"] = list(map(float, s["hist_p"]))
        return A


def write_belief_layer(path, whole_rel, belief, sidecar):
    """SDT belief layer over an egg asset: scalars as doubles, distributions as double arrays."""
    from pxr import Sdf, Usd, Vt
    if os.path.exists(path):
        os.remove(path)
    st = Usd.Stage.CreateNew(path)
    st.GetRootLayer().subLayerPaths = [whole_rel]
    st.GetRootLayer().comment = ("SDT layer: the robot's probabilistic belief about this egg (particle filter over "
                                 "register priors and the ageing model, updated by observations).")
    prim = st.OverridePrim("/Food")
    for k, v in belief.sdt_attributes(sidecar).items():
        if isinstance(v, str):
            prim.CreateAttribute(k, Sdf.ValueTypeNames.String, custom=True).Set(v)
        elif isinstance(v, list):
            prim.CreateAttribute(k, Sdf.ValueTypeNames.DoubleArray, custom=True).Set(Vt.DoubleArray(v))
        else:
            prim.CreateAttribute(k, Sdf.ValueTypeNames.Double, custom=True).Set(float(v))
    st.GetRootLayer().Save()
