"""The robot's side of the food twins: beliefs over the hidden attributes, updated only by observations.

Mirrors the lab's culture SDT: the truth (food/dynamics.py) is never read here.  Priors are the parameter register's
distributions; the posteriors are particle sets.  The beliefs answer the questions the AICOR axioms ask but do not
quantify -- H8 admissibility Prob(slip) < eps_s, Prob(rupture) < eps_r, Prob(stable) > gamma -- and the ones a
household plan needs ("is this avocado ready to eat today?").
"""

import numpy as np

from . import params as PR
from .avocado import flesh_E_MPa, penetrometer_force

G = 9.81


def _draw(spec, rng, n):
    return np.array([PR.draw(spec, rng) for _ in range(n)])


class EggBelief:
    """Belief over one egg's crack strength and grip friction, given its observed mass and shape."""

    def __init__(self, mass_kg, rng, n=4000, stiffness_fn=None):
        R = PR.EGG
        self.m = mass_kg
        self.rng = rng
        self.F_eq = _draw(R["F_break_equatorial"][1], rng, n)
        self.F_ax = _draw(R["F_break_axial"][1], rng, n)
        self.t = _draw(R["shell_thickness"][1], rng, n)
        self.mu = _draw(R["friction_shell_rubber"][1], rng, n)
        self.micro = rng.random(n) < 0.03                    # ILLUSTRATIVE prior: 3 % arrive micro-cracked
        self.w = np.full(n, 1.0 / n)
        self.log = []

    def F_crack(self, polar):
        s = np.sin(polar) ** 2
        F = (s * self.F_eq + (1 - s) * self.F_ax) * (self.t / 0.35) ** 2
        return np.where(self.micro, F * (1 - PR.EGG["microcrack_strength_loss"][1]), F)

    def F_slip(self, a_max=2.0):
        """Normal force per pad needed to hold the egg in a two-pad grip under vertical acceleration a_max."""
        return self.m * (G + a_max) / (2 * self.mu)

    def probs(self, F_grip, polar, a_max=2.0):
        w = self.w
        rupture = float(np.sum(w * (F_grip >= self.F_crack(polar))))
        slip = float(np.sum(w * (F_grip < self.F_slip(a_max))))
        return {"slip": slip, "rupture": rupture, "stable": max(0.0, 1 - slip - rupture)}

    def admissible(self, F_grip, polar, a_max=2.0, eps_s=0.05, eps_r=0.05, gamma=0.80):
        """AICOR H8: a grasp is admissible iff Prob(slip) < eps_s, Prob(rupture) < eps_r, Prob(stable) > gamma."""
        p = self.probs(F_grip, polar, a_max)
        return p["slip"] < eps_s and p["rupture"] < eps_r and p["stable"] > gamma, p

    def best_grip(self, polar, a_max=2.0, F_range=(0.2, 40.0)):
        Fs = np.linspace(*F_range, 400)
        P = [self.probs(F, polar, a_max)["stable"] for F in Fs]
        i = int(np.argmax(P))
        return float(Fs[i]), float(P[i])

    def corridor(self, polar, a_max=2.0, eps_s=0.05, eps_r=0.05):
        """Grip forces admissible under H8 with the default thresholds: [lo, hi] or None."""
        Fs = np.linspace(0.05, 60, 1200)
        ok = [F for F in Fs if self.admissible(F, polar, a_max, eps_s, eps_r)[0]]
        return (min(ok), max(ok)) if ok else None

    # ------------------------------------------------------------ observations
    def observe_held(self, F_grip, polar):
        """The egg survived a grip of F_grip (no crack seen): particles that would have cracked lose weight."""
        lik = np.where(F_grip >= self.F_crack(polar), 0.02, 1.0)
        self._update(lik, ("held", F_grip))

    def observe_no_slip(self, F_grip, a):
        lik = np.where(F_grip >= self.m * (G + a) / (2 * self.mu), 1.0, 0.05)
        self._update(lik, ("no_slip", F_grip, a))

    def _update(self, lik, what):
        self.w = self.w * lik
        self.w /= self.w.sum()
        ess = 1.0 / np.sum(self.w ** 2)
        self.log.append((what, float(ess)))
        if ess < 0.3 * len(self.w):
            idx = self.rng.choice(len(self.w), len(self.w), p=self.w)
            for k in ("F_eq", "F_ax", "t", "mu", "micro"):
                setattr(self, k, getattr(self, k)[idx])
            self.w = np.full(len(self.w), 1.0 / len(self.w))


class AvocadoBelief:
    """Belief over ripeness stage from squeeze stiffness and (weakly) skin colour."""

    def __init__(self, rng, n=3000, stiffness_model=None, sigma_rel=0.12):
        self.rng = rng
        self.stage = rng.uniform(1.0, 5.0, n)               # flat prior over the chart's scale
        self.w = np.full(n, 1.0 / n)
        self.k_model = stiffness_model or hertz_squeeze_stiffness
        self.sigma = sigma_rel
        self.log = []

    def observe_stiffness(self, k_obs, F_at):
        k_pred = self.k_model(self.stage, F_at)
        lik = np.exp(-0.5 * (np.log(k_obs / k_pred) / self.sigma) ** 2)
        self._update(lik, ("stiffness", k_obs, F_at))

    def observe_colour(self, darkness):
        """darkness in [0, 1] (green -> purple-black). Weak: the chart warns colour is unreliable for Hass."""
        pred = np.clip((self.stage - 1) / 4, 0, 1)
        lik = np.exp(-0.5 * ((darkness - pred) / 0.35) ** 2)
        self._update(lik, ("colour", darkness))

    def _update(self, lik, what):
        self.w = self.w * lik + 1e-300
        self.w /= self.w.sum()
        self.log.append((what, self.summary()))

    def summary(self):
        m = float(np.sum(self.w * self.stage))
        sd = float(np.sqrt(np.sum(self.w * (self.stage - m) ** 2)))
        classes = {k: float(np.sum(self.w * (np.clip(np.round(self.stage), 1, 5) == k))) for k in range(1, 6)}
        return {"mean": m, "sd": sd, "classes": classes, "p_ready": float(np.sum(self.w * (self.stage >= 4.0)))}

    def safe_squeeze_force(self, contact_area, p_bruise=0.05, ratio=0.5):
        """Largest pad force whose bruise probability stays below p_bruise (bruise model of dynamics.AvocadoTruth)."""
        lim = ratio * penetrometer_force(self.stage) / (np.pi * (11.1e-3 / 2) ** 2) * contact_area
        order = np.argsort(lim)
        cw = np.cumsum(self.w[order])
        return float(lim[order][np.searchsorted(cw, p_bruise)])


def hertz_squeeze_stiffness(stage, F, R=0.033, nu=0.45):
    """Analytic two-pad squeeze of a sphere of radius R, flat rigid pads: per contact F = 4/3 E* sqrt(R) d^1.5,
    two contacts in series: total closing 2 d.  Returns the secant stiffness F / (2 d) at force F [N/m].
    A first model only -- replaced by the Isaac FEM calibration table when it exists."""
    E = flesh_E_MPa(np.asarray(stage)) * 1e6
    Es = E / (1 - nu ** 2)
    d = (3 * F / (4 * Es * np.sqrt(R))) ** (2 / 3)
    return F / (2 * d)


class AppleBelief:
    """Expected bruise after events the robot knows about (drop heights, grip forces), from the register priors."""

    def __init__(self, mass_kg, volume_cm3, rng, n=3000):
        R = PR.APPLE
        self.m, self.V = mass_kg, volume_cm3
        self.E = _draw(R["flesh_E"][1], rng, n) * 1e6 * np.exp(-_draw(R["storage_days"][1], rng, n) / 180)
        self.th = _draw(R["bruise_energy_threshold"][1], rng, n)
        self.tau = _draw(R["flesh_shear_strength"][1], rng, n) * 1e6
        self.bruise = np.zeros(n)

    def after_drop(self, h, e=0.25):
        KE = self.m * G * h
        absorbed = (1 - e ** 2) * KE
        self.bruise = self.bruise + np.maximum(0, absorbed - self.th) * 0.01 * self.V / 0.41
        return self.summary()

    def p_bruise_static(self, F, R=0.037, nu=0.3):
        Es = self.E / (1 - nu ** 2)
        p0 = (6 * F * Es ** 2 / (np.pi ** 3 * R ** 2)) ** (1 / 3)
        return float(np.mean(0.31 * p0 > self.tau))

    def summary(self):
        return {"bruise_cm3_mean": float(self.bruise.mean()), "p_visible": float(np.mean(self.bruise > 0.2)),
                "p_severe": float(np.mean(self.bruise >= 1.0))}


def squeeze_table_model(path):
    """Stiffness model from the Isaac FEM squeeze sweep (test_reports/avocado_squeeze.json): secant stiffness
    F / d_fruit at force F, interpolated in log k over stage and linearly over force."""
    import json
    d = json.load(open(path))
    lanes = sorted(d["lanes"], key=lambda l: l["stage"])
    st = np.array([l["stage"] for l in lanes], float)
    Fs = np.array([p["F"] for p in lanes[0]["curve"]], float)
    K = np.array([[p["F"] / max(p["d_fruit_mm"], 1e-6) * 1e3 for p in l["curve"]] for l in lanes])   # N/m

    def k(stage, F):
        stage = np.clip(np.asarray(stage, float), st[0], st[-1])
        kF = np.array([np.interp(F, Fs, K[i]) for i in range(len(st))])
        return np.exp(np.interp(stage, st, np.log(kF)))
    return k
