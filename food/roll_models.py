"""Bread roll: hidden-state dynamics, action outcome models and the robot's belief (register entries in params.ROLL).

Hidden state of a roll: age since baking (h), storage (paper bag, plastic bag, open), reheat history.  Everything the
robot cares about follows from it:
  crust water w(t)          CALIBRATED power law through hirte_2012 (9.0 % @ 0.5 h, 13.9 % @ 24 h; 40 % RH, 22 C)
  crispness c(w)            logistic around the critical 10.1 % (hirte_2012: half the crispness lost there); width
                            ILLUSTRATIVE
  crumb modulus E(t)        log-time interpolation of lagrain_2012 (20 -> 157 kPa over 5 h .. 7 d; pan bread)
  crumb firmness F(t)       Avrami, cardone_2020 (loaf, n = 1)
  freshness class           from c and E (thresholds ILLUSTRATIVE, stated below)
Actions: reheat ('aufbacken'), toast cut faces, saw through (crust break, crumb squash, strokes, crumbs), grasp.
"""

import numpy as np

from . import params as PR

E_PTS = np.array([[5.0, 20.0], [48.0, 70.0], [96.0, 125.0], [168.0, 157.0]])   # h, kPa (lagrain_2012)
W_A, W_B = 9.73, 0.1123                                # crust water % = W_A t_h^W_B (CALIBRATED, see register)
W_CRIT, W_WIDTH = 10.1, 0.6                            # %, logistic width ILLUSTRATIVE
FRESH_C, DAYOLD_E, STALE_E = 0.5, 60.0, 110.0          # class thresholds (ILLUSTRATIVE)
STORAGE_FACTOR = {"open": 1.0, "paper_bag": 1.0, "plastic_bag": PR.ROLL["plastic_bag_factor"][1]}


def crumb_E_kPa(age_h):
    t = np.clip(np.asarray(age_h, float), E_PTS[0, 0], E_PTS[-1, 0] * 3)
    return np.interp(np.log(t), np.log(E_PTS[:, 0]), E_PTS[:, 1])


def crust_water(t_h, storage="paper_bag"):
    te = np.maximum(np.asarray(t_h, float) * STORAGE_FACTOR[storage], 0.02)
    return W_A * te ** W_B


def crispness(w):
    return 1.0 / (1.0 + np.exp((np.asarray(w, float) - W_CRIT) / W_WIDTH))


def firmness_N(age_h):
    """Avrami (cardone_2020, loaf TPA, n = 1, k per day)."""
    return 66.01 - 59.53 * np.exp(-0.186 * np.asarray(age_h, float) / 24.0)


def freshness_class(c, E):
    if c >= FRESH_C and E < DAYOLD_E:
        return "fresh"
    if E < DAYOLD_E:
        return "soft_crust"            # crumb still fresh, crust leathery (bag, a few hours)
    if E < STALE_E:
        return "day_old"               # 'vom Vortag'
    return "stale"


class RollTruth:
    """Simulator-side state of one roll.  t in hours."""

    def __init__(self, roll, mass_kg):
        p = roll.p
        self.roll, self.m = roll, mass_kg
        self.age_h = float(p["age_h"])                  # time since baking
        self.storage = p["storage"]
        self.crust_clock = self.age_h                  # crust water clock (reset by reheating)
        self.crumb_age = self.age_h                    # crumb staling clock (partly reset by reheating)
        self.toast = 0.0                               # browning dose on the cut faces (toaster-minutes)
        self.cut_state, self.events = "whole", []

    # ---------------------------------------------------------------- state
    def advance(self, dt_h, storage=None):
        if storage:
            self.storage = storage
        self.age_h += dt_h
        self.crust_clock += dt_h
        self.crumb_age += dt_h

    @property
    def w(self):
        return float(crust_water(self.crust_clock, self.storage))

    @property
    def crisp(self):
        return float(crispness(self.w))

    @property
    def E_kPa(self):
        return float(crumb_E_kPa(self.crumb_age))

    def discrete(self):
        return {"freshness": freshness_class(self.crisp, self.E_kPa),
                "crust_texture": "crisp" if self.crisp >= 0.5 else "leathery",
                "cut_state": self.cut_state, "toast_level": toast_level(self.toast)}

    # ---------------------------------------------------------------- actions
    def reheat(self, T_C=160.0, minutes=6.0):
        """'Aufbacken' (utopia_aufbacken: 150-170 C, 5-10 min): the crust dries and crisps again; heating the crumb
        above the amylopectin melting range undoes part of the staling (reset fraction ILLUSTRATIVE)."""
        dose = minutes * max(0.0, (T_C - 100.0) / 60.0)
        if dose >= 4.0:
            self.crust_clock = 0.1
            self.crumb_age = max(1.0, 0.25 * self.crumb_age)
            self.events.append(("reheat", T_C, minutes, "crisp again"))
        else:
            self.events.append(("reheat", T_C, minutes, "too short: crust still soft"))
        return self.discrete()

    def toast_faces(self, minutes, slot_power=1.0):
        """Cut faces in a toaster; levels CALIBRATED to kaur_2026 (3 min light .. 4.5 min starting to burn)."""
        self.toast += minutes * slot_power
        self.crust_clock = min(self.crust_clock, 0.1)
        return toast_level(self.toast)

    def saw(self, F_down, strokes, stroke_len=0.04, serrated=True):
        """Saw horizontally through the roll.  Returns the outcome dict.
        crust: breaks when the force on the serration points exceeds the blade-probe break force (hirte_2012: >= 3 N
          on a 1 x 10 mm blade); serration contact share ILLUSTRATIVE.
        crumb: compressive strain under the blade eps = F / (E A); > crumb_squash_strain (ILLUSTRATIVE) = squashed.
        progress: depth per stroke proportional to stroke length (ILLUSTRATIVE), ~6-10 strokes for a roll (TYPICAL).
        """
        r = self.roll
        width = 2 * r.b
        contact = width * (0.15 if serrated else 1.0)    # metres of edge pressing on the crust
        F_crust = PR.ROLL["crust_break_force"][1] * contact / 0.010
        A = width * 0.02                                 # blade-loaded crumb area (20 mm band) -- ILLUSTRATIVE
        eps = F_down / (self.E_kPa * 1e3 * A)
        squashed = eps > PR.ROLL["crumb_squash_strain"][1]
        if F_down < F_crust:
            return {"ok": False, "why": "crust not broken (saw harder or use a serrated knife)", "F_crust": F_crust,
                    "strain": eps}
        depth_per_stroke = 0.35 * stroke_len * (1.0 if serrated else 0.4)
        need = width / depth_per_stroke
        done = strokes >= need
        crumbs_g = 0.004 * (width * 1e2) * (1 + 2 * self.crisp)    # ILLUSTRATIVE: crisp crust sheds more
        if done:
            self.cut_state = "sliced_open"
        self.events.append(("saw", F_down, strokes, done, squashed))
        return {"ok": done, "strokes_needed": float(np.ceil(need)), "squashed": bool(squashed), "strain": eps,
                "F_crust": F_crust, "crumbs_g": crumbs_g}

    def grasp_corridor(self, mu=None, a_max=2.0, pad_area=4e-4):
        """Two-pad grip: F_slip = m (g + a) / (2 mu) < F < F_dent; F_dent = crumb strain 0.3 under the pads, plus the
        crust's support while crisp (factor ILLUSTRATIVE)."""
        mu = mu or self.roll.p["friction_crust_rubber"]
        F_slip = self.m * (9.81 + a_max) / (2 * mu)
        F_dent = 0.3 * self.E_kPa * 1e3 * pad_area * (1 + 3 * self.crisp)
        return F_slip, F_dent


def toast_level(dose_min):
    if dose_min < 1.5:
        return "none"
    if dose_min < 3.0:
        return "light"
    if dose_min < 4.0:
        return "golden"
    if dose_min < 4.5:
        return "dark"
    return "burnt"


# ------------------------------------------------------------------------------------------------ belief
class RollBelief:
    """Particles over (age_h, storage).  Observations: squeeze stiffness of the crumb (gripper), crackle events while
    squeezing (crisp crust; hirte_2012 measures crispness acoustically), and the bakery-bag time stamp if known."""

    def __init__(self, rng, n=3000):
        self.rng = rng
        self.age = rng.choice(PR.ROLL["age_h"][1][1:], n) * np.exp(rng.normal(0, 0.3, n))
        self.storage = rng.choice(["paper_bag", "plastic_bag", "open"], n, p=[0.6, 0.25, 0.15])
        self.w = np.full(n, 1.0 / n)

    def _pred(self):
        crisp = np.array([crispness(crust_water(a, s)) for a, s in zip(self.age, self.storage)])
        return crisp, crumb_E_kPa(self.age)

    def observe_squeeze(self, E_obs_kPa, sigma_rel=0.25):
        _, E = self._pred()
        self._update(np.exp(-0.5 * (np.log(E_obs_kPa / E) / sigma_rel) ** 2))

    def observe_stiffness(self, k_N_per_mm, sigma_rel=0.25):
        """Gripper squeeze across the width (30 mm pads, 1 -> 3 N): k = 0.29 N/mm x E / 20 kPa (register
        squeeze_k_model, from scripts/sim_roll_squeeze.py)."""
        self.observe_squeeze(20.0 * k_N_per_mm / 0.29, sigma_rel)

    def observe_crackles(self, n_events, rate_crisp=12.0):
        """Poisson count of crackle events in a gentle squeeze; rate proportional to crispness (ILLUSTRATIVE)."""
        crisp, _ = self._pred()
        lam = 0.3 + rate_crisp * crisp
        self._update(np.exp(n_events * np.log(lam) - lam))

    def observe_bag_time(self, age_h, sd_h=1.0):
        self._update(np.exp(-0.5 * ((self.age - age_h) / sd_h) ** 2))

    def _update(self, lik):
        self.w = self.w * lik
        self.w /= self.w.sum()
        if 1.0 / np.sum(self.w ** 2) < len(self.w) / 3:
            i = self.rng.choice(len(self.w), len(self.w), p=self.w)
            self.age = self.age[i] * np.exp(self.rng.normal(0, 0.05, len(i)))
            self.storage = self.storage[i]
            self.w = np.full(len(i), 1.0 / len(i))

    def p_class(self):
        crisp, E = self._pred()
        cls = np.array([freshness_class(c, e) for c, e in zip(crisp, E)])
        return {k: float(self.w[cls == k].sum()) for k in ("fresh", "soft_crust", "day_old", "stale")}

    def summary(self):
        crisp, E = self._pred()
        return {"age_h_mean": float(self.w @ self.age), "p_crisp": float(self.w @ (crisp >= 0.5)),
                "E_kPa_mean": float(self.w @ E), **{f"p_{k}": v for k, v in self.p_class().items()}}


# ------------------------------------------------------------------------------------------------ USD layers
def roll_beliefs(roll, mp):
    b = RollBelief(np.random.default_rng(1)).summary()
    return {"": {"sdt:epistemicStatus": "prior + observation", "sdt:belief:mass_g": mp["mass_kg"] * 1e3,
                 "sdt:belief:mass_g_source": "weighed",
                 "sdt:belief:freshness": " | ".join(f"{k} {b['p_' + k]:.2f}" for k in
                                                    ("fresh", "soft_crust", "day_old", "stale")),
                 "sdt:belief:age_h_mean": b["age_h_mean"], "sdt:belief:cut_state": "whole 1.0"},
            "Crust": {"sdt:belief:p_crisp": b["p_crisp"], "sdt:epistemicStatus": "prior; squeeze/crackle to update"},
            "Crumb": {"sdt:belief:E_kPa_mean": b["E_kPa_mean"], "sdt:epistemicStatus": "prior (unobserved inside)"}}


def roll_truth(roll):
    t = RollTruth(roll, 0.05)
    T = {"": {f"truth:{k}": v for k, v in roll.p.items() if not k.startswith("_") and isinstance(v, (int, float, str))}}
    T[""].update({f"truth:{k}": v for k, v in t.discrete().items()})
    T["Crust"] = {"truth:crust_water_pct": t.w, "truth:crispness": t.crisp}
    T["Crumb"] = {"truth:E_kPa": t.E_kPa, "truth:firmness_N": float(firmness_N(t.age_h))}
    return T
