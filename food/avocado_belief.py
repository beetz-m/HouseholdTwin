"""Probabilistic SDT node of an avocado: a particle filter over the fruit's hidden parameters.

State: one particle = one complete candidate avocado (the register's random parameters: size, neck, skin, seed,
ripeness stage, ripening rate, friction, stem button) plus a bruise flag; derived quantities (mass, aspect ratio,
penetrometer force, flesh modulus, squeeze stiffness, seed mass) are computed per particle, so their dependencies
are kept (weighing also informs size, a squeeze also informs ripeness and colour).
Prior: the parameter register (food/params.py AVOCADO), i.e. the population of Hass fruit.
Prediction (Bayesian filtering in time): ripeness advances at the particle's own ripening rate (process noise
ILLUSTRATIVE); a squeeze harder than the particle's bruise threshold sets its bruise flag.
Update: each observation multiplies the weights by its likelihood (measurement models below, noise levels stated
with each); systematic resampling with a small regularising jitter when the effective sample size drops.
The distributions are written into the SDT belief layer (per variable: mean, sd, 5/50/95 % quantiles, a histogram
and the support) and the particle set into a sidecar file.
"""

import json
import math
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from . import params as PR
from .avocado import flesh_E_MPa, penetrometer_force, skin_colour  # noqa: F401  (skin_colour: colour model doc)

RANDOM = ["length_L", "diameter_D", "neck", "skin_thickness", "pebble_scale", "seed_length", "seed_width_ratio",
          "seed_offset", "ripeness_stage", "ripening_rate", "friction_skin_rubber", "stem_button_length"]
UNITS = {"length_L": "mm", "diameter_D": "mm", "neck": "1", "skin_thickness": "mm", "pebble_scale": "mm",
         "seed_length": "mm", "seed_width_ratio": "1", "seed_offset": "mm", "ripeness_stage": "1..5",
         "ripening_rate": "stage/d", "friction_skin_rubber": "1", "stem_button_length": "mm", "mass_g": "g",
         "aspect_L_over_D": "1", "penetrometer_N": "N", "flesh_E_MPa": "MPa", "squeeze_k_N_per_mm": "N/mm",
         "seed_mass_g": "g", "flesh_mass_g": "g", "bruised": "P", "testa_adheres_to_flesh": "P"}
SQUEEZE_STAGE = np.array([1.0, 2.0, 3.0, 4.0, 4.5, 5.0])
SQUEEZE_K = np.array([7.0, 5.4, 3.3, 2.4, 2.1, 1.8])       # register squeeze_stiffness_model (DERIVED, FEM table)


@dataclass
class Observation:
    kind: str                 # length_mm, mass_g, photo_shape, skin_colour, squeeze, stem_button, seed_length_mm, peel
    value: dict
    time_h: float
    source: str
    note: str = ""


# %% derived quantities per particle
def body_volume_m3(L, D, neck, n=160):
    """Volume of the twin's body of revolution (food/avocado.py meridian) for arrays of L, D (m) and neck."""
    s = np.linspace(0.0, 1.0, n)
    x = (s - 0.5)[None, :] * L[:, None]
    w = 0.13 * L[:, None]
    y = D[:, None] / 2 * np.sqrt(np.clip((L[:, None] ** 2 - 4 * x ** 2) / (L[:, None] ** 2 + 8 * w * x + 4 * w ** 2),
                                         0, None))
    g = np.clip((s - 0.45) / 0.45, 0, 1)[None, :]
    y = y * (1 - neck[:, None] * g ** 2 * (3 - 2 * g))
    return np.trapezoid(np.pi * y ** 2, x, axis=1)


def derived(P):
    """Derived quantities of every particle (dict of arrays)."""
    reg = PR.AVOCADO
    L, D = P["length_L"] * 1e-3, P["diameter_D"] * 1e-3
    V = body_volume_m3(L, D, P["neck"])
    sl = P["seed_length"] * 1e-3
    sw = P["seed_width_ratio"] * sl
    Vs = 4 / 3 * np.pi * (sl / 2) * (sw / 2) ** 2
    rho_f, rho_s = reg["flesh_density"][1], reg["seed_density"][1]
    m = ((V - Vs) * rho_f + Vs * rho_s) * 1.021              # calibrated on the built twins (denser skin; DERIVED)
    stage = P["ripeness_stage"]
    return {"mass_g": m * 1e3, "aspect_L_over_D": P["length_L"] / P["diameter_D"],
            "penetrometer_N": penetrometer_force(stage), "flesh_E_MPa": flesh_E_MPa(stage),
            "squeeze_k_N_per_mm": np.interp(stage, SQUEEZE_STAGE, SQUEEZE_K), "seed_mass_g": Vs * rho_s * 1e3,
            "flesh_mass_g": (V - Vs) * rho_f * 1e3, "bruised": P["bruised"].astype(float),
            "testa_adheres_to_flesh": P["testa_flesh"].astype(float) if "testa_flesh" in P else np.zeros(len(L))}


def seed_depths_mm(P):
    """Flesh depths a blade feels round the seed (side, top, bottom; mm) for every particle, from the twin's meridian
    and seed placement (seed centre at -0.10 L on the axis; seed coat + endocarp ~0.5 mm)."""
    L, D = P["length_L"], P["diameter_D"]
    zc = -0.10 * L
    sl, sw = P["seed_length"], P["seed_width_ratio"] * P["seed_length"]
    w = 0.13 * L
    y = D / 2 * np.sqrt(np.clip((L ** 2 - 4 * zc ** 2) / (L ** 2 + 8 * w * zc + 4 * w ** 2), 0, None))
    g = 0.5
    return dict(side=y - sw / 2 - g, top=L / 2 - (zc + sl / 2) - g, bottom=(zc - sl / 2 + g) + L / 2)


def seed_half_interface(P):
    """Half the seed surface (m2) and the lever arm (m) per particle (as food/avocado_process.seed_interface)."""
    a, b = P["seed_length"] / 2e3, P["seed_width_ratio"] * P["seed_length"] / 2e3
    area = 4 * np.pi * ((2 * (a * b) ** 1.6075 + (b * b) ** 1.6075) / 3) ** (1 / 1.6075)
    return area / 2, 0.6 * b


# %% the filter
@dataclass
class AvocadoBelief:
    n: int
    particles: dict
    weights: np.ndarray
    time_h: float = 0.0
    history: list = field(default_factory=list)
    log_evidence: float = 0.0

    @classmethod
    def prior(cls, n=4000, seed=0):
        """Particles drawn from the register's distributions (the population of Hass fruit)."""
        draws = [PR.sample("avocado", seed * 100000 + i) for i in range(n)]
        P = {k: np.array([d[k] for d in draws], float) for k in RANDOM}
        P["bruised"] = np.zeros(n, bool)
        P["testa_flesh"] = np.array([d["testa_adheres_to"] == "flesh" for d in draws])   # register choice_str
        return cls(n, P, np.full(n, 1.0 / n))

    # ---------------------------------------------------------------- bookkeeping
    def bounds(self, k):
        spec = PR.AVOCADO[k][1]
        if spec[0] == "normal":
            return spec[3], spec[4]
        if spec[0] == "uniform":
            return spec[1], spec[2]
        return -np.inf, np.inf

    def ess(self):
        return float(1.0 / np.sum(self.weights ** 2))

    def resample(self, rng, jitter=0.05):
        """Systematic resampling; regularising jitter (jitter x the weighted sd of each variable), clipped to the
        register bounds, so the particle set does not collapse onto a few values."""
        u = (rng.random() + np.arange(self.n)) / self.n
        idx = np.minimum(np.searchsorted(np.cumsum(self.weights), u), self.n - 1)
        for k in RANDOM:
            v = self.particles[k][idx]
            sd = math.sqrt(np.cov(self.particles[k], aweights=self.weights)) if self.n > 1 else 0.0
            lo, hi = self.bounds(k)
            self.particles[k] = np.clip(v + rng.normal(0, jitter * sd, self.n), lo, hi)
        self.particles["bruised"] = self.particles["bruised"][idx]
        self.particles["testa_flesh"] = self.particles["testa_flesh"][idx]
        self.weights = np.full(self.n, 1.0 / self.n)

    def _apply(self, lik, obs, rng):
        lik = np.clip(lik, 1e-300, None)
        w = self.weights * lik
        z = w.sum()
        self.log_evidence += math.log(max(z, 1e-300))
        self.weights = w / z
        rec = asdict(obs)
        rec["ess_after"] = self.ess()
        self.history.append(rec)
        if self.ess() < self.n / 2:
            self.resample(rng)

    # ---------------------------------------------------------------- prediction
    def predict(self, hours, rng, process_sd_per_day=0.05):
        """Ripening over ``hours``: stage += rate x days + noise (bounded at 5). Process noise ILLUSTRATIVE."""
        d = hours / 24.0
        P = self.particles
        P["ripeness_stage"] = np.clip(P["ripeness_stage"] + P["ripening_rate"] * d +
                                      rng.normal(0, process_sd_per_day * math.sqrt(d), self.n), 1.0, 5.0)
        self.time_h += hours
        self.history.append(dict(kind="predict", value=dict(hours=hours), time_h=self.time_h,
                                 source="ripening model (register ripening_rate)", note="", ess_after=self.ess()))

    # ---------------------------------------------------------------- measurement models
    def observe(self, obs, rng):
        """Bayesian update with one observation (likelihood per particle)."""
        P, Dv, v = self.particles, derived(self.particles), obs.value
        k = obs.kind
        if k == "length_mm":                                  # calliper, sd 1 mm (ILLUSTRATIVE)
            lik = np.exp(-0.5 * ((P["length_L"] - v["value"]) / v.get("sd", 1.0)) ** 2)
        elif k == "mass_g":                                   # kitchen scale sd 2 g + surrogate model error 3 %
            sd = math.hypot(v.get("sd", 2.0), 0.03 * v["value"])
            lik = np.exp(-0.5 * ((Dv["mass_g"] - v["value"]) / sd) ** 2)
        elif k == "photo_shape":                              # silhouette aspect ratio and neck from a photo
            lik = np.exp(-0.5 * ((Dv["aspect_L_over_D"] - v["aspect"]) / v.get("aspect_sd", 0.05)) ** 2)
            if v.get("neck") is not None:
                lik *= np.exp(-0.5 * ((P["neck"] - v["neck"]) / v.get("neck_sd", 0.05)) ** 2)
        elif k == "skin_colour":                              # CIELAB of the skin vs the twin's colour chart
            t = (P["ripeness_stage"] - 1) / 4
            Lp, Cp = 40 - 13 * t, 15 - 12 * t
            ll = -0.5 * (((v["L"] - Lp) / v.get("sd_L", 9.0)) ** 2 + ((v["C"] - Cp) / v.get("sd_C", 4.0)) ** 2)
            green = np.clip((3.0 - P["ripeness_stage"]) / 2.0, 0, 1)
            is_green = 90.0 <= v["h"] <= 150.0
            p_hue = green * 0.9 + (1 - green) * 0.1 if is_green else green * 0.1 + (1 - green) * 0.9
            lik = np.exp(ll + min(1.0, v["C"] / 8.0) * np.log(p_hue))
        elif k == "squeeze":                                  # gripper squeeze: stiffness at 4 N, model sd 20 %
            pred = Dv["squeeze_k_N_per_mm"]
            sd = math.hypot(v.get("sd", 0.1 * v["k"]), 0.2 * v["k"])
            lik = np.exp(-0.5 * ((pred - v["k"]) / sd) ** 2)
            f = v.get("force_N", 4.0)                         # transition: a hard squeeze bruises soft fruit
            thr = PR.AVOCADO["bruise_force_ratio"][1] * Dv["penetrometer_N"]
            P["bruised"] = P["bruised"] | (f > thr)
        elif k == "stem_button":
            seen = v["present"]
            lik = np.where(P["stem_button_length"] > 1.0, 0.9 if seen else 0.1, 0.3 if seen else 0.7)
        elif k == "seed_length_mm":                           # after halving: seed length measured, sd 1.5 mm
            lik = np.exp(-0.5 * ((P["seed_length"] - v["value"]) / v.get("sd", 1.5)) ** 2)
        elif k == "seed_depths":                              # blade stop depths round the seed (cut), sd 1.5 mm
            pr = seed_depths_mm(P)
            sd = v.get("sd", 1.5)
            lik = np.ones(self.n)
            for kk in ("side", "top", "bottom"):
                if v.get(kk) is not None:
                    lik *= np.exp(-0.5 * ((pr[kk] - v[kk]) / sd) ** 2)
        elif k == "twist":                                    # twist apart: clean / torn and the torque at separation
            from . import avocado_process as PRC
            st_ = P["ripeness_stage"]
            pc = np.array([PRC.p_clean_twist(x) for x in st_])
            lik = pc if v["clean"] else 1 - pc
            if v.get("T_Nm") is not None and v.get("full_cut", True) and v["clean"]:
                A, r = seed_half_interface(P)
                Tp = np.array([PRC.seed_adhesion(x) for x in st_]) * A * r
                lik = lik * np.exp(-0.5 * ((Tp - v["T_Nm"]) / (0.3 * v["T_Nm"])) ** 2)
        elif k == "seed_lift_force":                          # spoon lever force, model error 30 %
            from . import avocado_process as PRC
            A, _ = seed_half_interface(P)
            Fp = 0.3 * np.array([PRC.seed_adhesion(x) for x in P["ripeness_stage"]]) * A
            lik = np.exp(-0.5 * ((Fp - v["F_N"]) / (0.3 * v["F_N"])) ** 2)
        elif k == "seed_twist_torque":                        # knife-twist torque releasing the seed, 30 % error
            from . import avocado_process as PRC
            A, r = seed_half_interface(P)
            Tp = np.array([PRC.seed_adhesion(x) for x in P["ripeness_stage"]]) * A * r
            lik = np.exp(-0.5 * ((Tp - v["T_Nm"]) / (0.3 * v["T_Nm"])) ** 2)
        elif k == "testa_seen":                               # seed coat left in the cavity after pitting
            lik = np.where(P["testa_flesh"], 0.95 if v["seen"] else 0.05, 0.03 if v["seen"] else 0.97)
        elif k == "bruise_seen":                              # flesh inspected after cutting
            p_det = v.get("detection_p", 0.9)
            lik = np.where(P["bruised"], p_det if v["seen"] else 1 - p_det, 0.05 if v["seen"] else 0.95)
        else:
            raise ValueError(f"unknown observation kind {k}")
        self._apply(lik, obs, rng)

    # ---------------------------------------------------------------- summaries
    def summary(self, bins=24):
        """Per variable: weighted mean, sd, quantiles, histogram and support."""
        out = {}
        D = derived(self.particles)
        allv = {**{k: self.particles[k] for k in RANDOM}, **D}
        order = np.argsort
        for k, x in allv.items():
            x = np.asarray(x, float)
            o = order(x)
            cw = np.cumsum(self.weights[o])
            q = lambda p: float(x[o][min(np.searchsorted(cw, p), self.n - 1)])  # noqa: E731
            mean = float(np.sum(self.weights * x))
            sd = float(math.sqrt(max(np.sum(self.weights * (x - mean) ** 2), 0.0)))
            lo, hi = float(x.min()), float(x.max())
            edges = np.linspace(lo, hi if hi > lo else lo + 1e-9, bins + 1)
            hist, _ = np.histogram(x, edges, weights=self.weights)
            out[k] = dict(unit=UNITS[k], mean=mean, sd=sd, p05=q(0.05), p50=q(0.5), p95=q(0.95), support=[lo, hi],
                          hist_edges=edges.tolist(), hist_p=hist.tolist())
        return out

    def save(self, path):
        np.savez_compressed(path, weights=self.weights, time_h=self.time_h, history=json.dumps(self.history),
                            log_evidence=self.log_evidence, **{f"p_{k}": v for k, v in self.particles.items()})

    @classmethod
    def load(cls, path):
        d = np.load(path)
        P = {k[2:]: d[k] for k in d.files if k.startswith("p_")}
        P["bruised"] = P["bruised"].astype(bool)
        if "testa_flesh" in P:
            P["testa_flesh"] = P["testa_flesh"].astype(bool)
        b = cls(len(d["weights"]), P, d["weights"], float(d["time_h"]), json.loads(str(d["history"])),
                float(d["log_evidence"]))
        return b

    def sdt_attributes(self, sidecar):
        """Belief-layer attributes (root prim) carrying the full distributions."""
        S = self.summary()
        A = {"sdt:belief:representation": f"particle filter, {self.n} particles (food/avocado_belief.py)",
             "sdt:belief:particles": sidecar, "sdt:belief:time_h": self.time_h,
             "sdt:belief:ess": self.ess(), "sdt:belief:observations": json.dumps(
                 [f"{h['time_h']:.0f} h {h['kind']} {h.get('source', '')}" for h in self.history]),
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
    """SDT belief layer over an avocado asset: scalars as doubles, distributions as double arrays."""
    from pxr import Sdf, Usd, Vt
    if os.path.exists(path):
        os.remove(path)
    st = Usd.Stage.CreateNew(path)
    st.GetRootLayer().subLayerPaths = [whole_rel]
    st.GetRootLayer().comment = ("SDT layer: the robot's probabilistic belief about this avocado (particle filter "
                                 "over register priors, updated by observations).")
    prim = st.OverridePrim("/Food")
    for k, v in belief.sdt_attributes(sidecar).items():
        if isinstance(v, str):
            prim.CreateAttribute(k, Sdf.ValueTypeNames.String, custom=True).Set(v)
        elif isinstance(v, list):
            prim.CreateAttribute(k, Sdf.ValueTypeNames.DoubleArray, custom=True).Set(Vt.DoubleArray(v))
        else:
            prim.CreateAttribute(k, Sdf.ValueTypeNames.Double, custom=True).Set(float(v))
    st.GetRootLayer().Save()
