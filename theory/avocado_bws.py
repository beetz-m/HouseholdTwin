"""The avocado body-world system as a research SDT: the simulated plant (hidden truth, laws, observation model), its
event-patch trace, the robot's belief, task policies and phase-indexed requirements -- the executable part of the
avocado manipulation microtheory (docs/avocado_microtheory, transferred from the egg microtheory).

Plant. One avocado instance drawn from the register (food/params.py AVOCADO) with its hidden parameters Theta
(ripeness stage and rate, sizes, seed, testa side, ...). Its laws are the twin's reduced-order models: squeeze
stiffness and bruising (food/avocado_belief.py tables, register bruise_force_ratio), the processing mechanics
(food/avocado_process.py: cut round the seed, twist apart, remove the seed, peel the seed coat) and the lifetime
model (food/avocado_lifetime.py: ripening, ripe window, decline, spoilage). Every structural change is a patch
Delta = (t, event, ops, cause) on a set of relational facts, so the trace can be read as a chronicle (theory/axioms.py).

Observation model Omega. Photo (silhouette aspect, neck, skin L* C* h, stem button) with noise and an optional
colour bias; gripper squeeze stiffness with force-dependent noise; blade stop depths, twist torque and outcome, spoon
force, cavity inspection (seed coat seen), extracted seed length. Only Omega's outputs reach the robot.

Robot. A particle belief over Theta (food/avocado_belief.py) started from a cached register prior; policies read only
observations and the belief.

Tasks (the same physical event, two intentions):
  A  assess ripeness by touch, intact: squeeze without bruising and decide 'ready' (stage >= 4) correctly;
     any bruise is a failure.
  B  halve and pit within three days: separation of the flesh is required, but only in the cut plane at the cut
     phase; torn flesh, a bruise, an unsafe tool use, an overripe / spoiled fruit or seed coat left are failures.
"""

import contextlib
import math
from dataclasses import dataclass, field

import numpy as np

from food import avocado_belief as AB
from food import avocado_process as PRC
from food import params as PR
from food.avocado import Avocado, penetrometer_force
from food.avocado_lifetime import FruitClock, temperature_factor, STORAGE_T

READY_STAGE = 4.0
CLASSES = {"unripe": (1.4, 2.6), "breaking": (2.6, 3.6), "ripe": (3.6, 4.6)}


class LightAvocado(Avocado):
    """The instance's geometry for the mechanics, without the flesh-band colour tables (1 ms instead of 1 s)."""

    def _build_radial_tables(self):
        pass


# %% plant
@dataclass
class Misspec:
    """Plant laws the robot's model does not share (all 1.0 / 0.0 = well specified)."""
    ripening_rate: float = 1.0      # the plant ripens this much faster than its drawn rate
    adhesion: float = 1.0           # seed-flesh adhesion scale (spoon force, twist)
    colour_L_bias: float = 0.0      # photo lightness bias (negative: dark skin reads riper)
    bruise_ratio: float = 1.0       # bruise threshold scale (lower: flesh bruises earlier)
    stiffness: float = 1.0          # squeeze stiffness scale

    @property
    def name(self):
        d = [f"{k} x{v:g}" if k != "colour_L_bias" else f"colour L {v:+g}" for k, v in self.__dict__.items()
             if (v != 1.0 and k != "colour_L_bias") or (k == "colour_L_bias" and v != 0.0)]
        return ", ".join(d) or "well specified"


@contextlib.contextmanager
def scaled_register(**scale):
    """Temporarily scale register constants read by the mechanics (misspecified plants)."""
    old = {}
    for k, f in scale.items():
        old[k] = PR.AVOCADO[k]
        u, v, s, src = old[k]
        PR.AVOCADO[k] = (u, v * f, s, src)
    try:
        yield
    finally:
        PR.AVOCADO.update(old)


def instance(seed, cls=None):
    """Hidden parameters of held-out avocado ``seed``; ``cls`` forces the ripeness stage into a class range."""
    hp = {k: v for k, v in PR.sample("avocado", seed).items()}
    if cls is not None:
        lo, hi = CLASSES[cls]
        hp["ripeness_stage"] = float(np.random.default_rng(seed + 7777).uniform(lo, hi))
    return hp


class Plant:
    """The simulated avocado body-world system (evaluation truth only)."""

    def __init__(self, hp, misspec=None, seed=0):
        self.hp = dict(hp)
        self.m = misspec or Misspec()
        self.rng = np.random.default_rng(seed)
        p = {k: v for k, v in hp.items() if not k.startswith("_")}
        self.avo = LightAvocado(int(seed), overrides=p)
        self.clock = FruitClock.of(p)
        self.clock = FruitClock(self.clock.stage, self.clock.rate_per_day * self.m.ripening_rate,
                                ripe_window_h=self.clock.ripe_window_h, spoil_days=self.clock.spoil_days)
        self.plane = PRC.CutPlane(0.0, 360.0)
        self.t = 0.0
        self.facts = {"Whole(Avocado)", "Intact(Mesocarp)", "SeedIn(Avocado)", "OnBoard(Avocado)"}
        self.trace = []
        self.remnant = 0.0
        self._patch("Init", add=["Ripe(Avocado)"] if self.clock.stage >= READY_STAGE else [], cause="initial state")

    # ---------------------------------------------------------------- patches
    def _patch(self, event, add=(), rem=(), cause="", attrs=None):
        add = [f for f in add if f not in self.facts]
        rem = [f for f in rem if f in self.facts]
        self.facts |= set(add)
        self.facts -= set(rem)
        self.trace.append(dict(t=self.t, event=event, add=add, rem=rem, cause=cause, attrs=attrs or {},
                               facts=sorted(self.facts), stage=self.clock.stage))

    @property
    def stage(self):
        return self.clock.stage

    # ---------------------------------------------------------------- observation model Omega
    def photo(self):
        L, D = self.hp["length_L"], self.hp["diameter_D"]
        t = (self.stage - 1) / 4
        r = self.rng
        obs = dict(aspect=L / D * (1 + r.normal(0, 0.02)), neck=self.hp["neck"] + r.normal(0, 0.03),
                   L=40 - 13 * t + r.normal(0, 4.0) + self.m.colour_L_bias, C=15 - 12 * t + r.normal(0, 2.0),
                   h=(120 if self.stage < 3.0 else 220) + r.normal(0, 10), stem=self.hp["stem_button_length"] > 1.0)
        self._patch("Photo", cause="camera", attrs=obs)
        return obs

    def stiffness_true(self):
        P = {k: np.array([float(self.hp[k])]) for k in AB.RANDOM}
        P["ripeness_stage"] = np.array([self.stage])
        P["bruised"] = np.array([False])
        P["testa_flesh"] = np.array([self.hp["testa_adheres_to"] == "flesh"])
        return float(AB.derived(P)["squeeze_k_N_per_mm"][0]) * self.m.stiffness

    def bruise_threshold(self):
        return PR.AVOCADO["bruise_force_ratio"][1] * self.m.bruise_ratio * float(penetrometer_force(self.stage))

    def squeeze(self, F):
        k = self.stiffness_true() * (1 + self.rng.normal(0, 0.1 * 4.0 / F))
        bruised = F > self.bruise_threshold()
        self._patch("Squeeze", add=["Bruised(Mesocarp)"] if bruised else [],
                    rem=["Intact(Mesocarp)"] if bruised else [],
                    cause=f"contact pressure {F:.1f} N vs bruise threshold {self.bruise_threshold():.1f} N",
                    attrs=dict(F=F, k_meas=k))
        return k

    # ---------------------------------------------------------------- time
    def wait(self, hours, storage="counter"):
        was_ripe = self.stage >= READY_STAGE
        self.clock = self.clock.advance(hours, storage)
        self.t += hours
        add = []
        if self.stage >= READY_STAGE and not was_ripe:
            add.append("Ripe(Avocado)")
        if self.clock.days_past_window > 0:
            add.append("Overripe(Avocado)")
        if self.clock.spoiled:
            add.append("Spoiled(Avocado)")
        self._patch("Wait", add=add, cause=f"{hours:.0f} h in the {storage}", attrs=dict(hours=hours, storage=storage))

    # ---------------------------------------------------------------- processing
    def _mech(self):
        return scaled_register(seed_adhesion_unripe=self.m.adhesion)

    def cut(self, F_max=40.0, coverage=360.0):
        self.plane.coverage_deg = coverage
        with self._mech():
            out = PRC.cut_round_seed(self.avo, self.plane, self.rng, self.stage, F_max)
        if out.outcome == "scored":
            self._patch("CutRoundSeed", add=["Scored(Avocado)"], rem=["PartlyScored(Avocado)"],
                        cause="blade round the seed")
        elif out.outcome == "partly_scored":
            self._patch("CutRoundSeed", add=["PartlyScored(Avocado)"], cause="force limit stalled the blade")
        else:
            self._patch("CutRoundSeed", cause="no entry")
        return out

    def twist(self, T_max=1.5):
        with self._mech():
            out = PRC.twist_apart(self.avo, self.plane, self.rng, self.stage, self.plane.coverage_deg, T_max)
        if out.ok:
            add = ["Apart(HalfA,HalfB)", "SeedIn(HalfA)"] + (["TornFlesh(HalfB)", "FleshOnSeed(Seed)"]
                                                           if out.outcome == "halved_torn" else [])
            self._patch("TwistApart", add=add, rem=["Whole(Avocado)", "Scored(Avocado)", "PartlyScored(Avocado)",
                                                    "SeedIn(Avocado)"],
                        cause="released at the seed" if out.outcome == "halved" else "flesh clings to the seed")
        else:
            self._patch("TwistApart", cause="stuck: uncut bridge or adhesion beyond the torque limit")
        return out

    def remove_seed(self, method="spoon", F_max=25.0, T_max=1.0):
        with self._mech():
            out = PRC.remove_seed(self.avo, self.rng, self.stage, method, F_max, T_max)
        if out.ok:
            self.remnant = out.truth.get("testa_remnant_fraction", 0.0)
            add = ["SeedOut(HalfA)"] + (["TestaIn(Cavity)"] if self.remnant > 0.05 else []) + \
                (["Gouge(Cavity)"] if out.truth.get("gouges") else []) + \
                (["UnsafeToolUse(KnifeStrike)"] if method == "knife_twist" else [])
            self._patch("RemoveSeed", add=add, rem=["SeedIn(HalfA)"], cause=method)
        else:
            self._patch("RemoveSeed", cause=f"{method}: seed does not lift within the limit")
        return out

    def peel(self, passes=1):
        out = PRC.peel_seed_coat(None, self.rng, self.remnant, passes)
        self.remnant = out.truth["testa_remnant_fraction"]
        self._patch("PeelSeedCoat", add=["Gouge(Cavity)"] if out.truth["gouges"] else [],
                    rem=["TestaIn(Cavity)"] if self.remnant <= 0.05 else [], cause="spoon tip")
        return out

    def outcome(self):
        """End-of-episode facts used by the requirements (hidden truth)."""
        return dict(stage=self.stage, eating_quality=self.clock.eating_quality, spoiled=self.clock.spoiled,
                    overripe=self.clock.days_past_window > 0, facts=set(self.facts))


# %% robot belief
_PRIOR = {}


def prior(n=1000, seed=11):
    """A cached register prior (copied per episode)."""
    if (n, seed) not in _PRIOR:
        _PRIOR[(n, seed)] = AB.AvocadoBelief.prior(n, seed=seed)
    b = _PRIOR[(n, seed)]
    return AB.AvocadoBelief(b.n, {k: v.copy() for k, v in b.particles.items()}, b.weights.copy())


def observe_photo(b, o, rng):
    b.observe(AB.Observation("photo_shape", dict(aspect=o["aspect"], neck=o["neck"]), b.time_h, "photo"), rng)
    b.observe(AB.Observation("skin_colour", dict(L=o["L"], C=o["C"], h=o["h"]), b.time_h, "photo"), rng)
    b.observe(AB.Observation("stem_button", dict(present=o["stem"]), b.time_h, "photo"), rng)


def observe_squeeze(b, k, F, rng):
    b.observe(AB.Observation("squeeze", dict(k=k, sd=0.1 * 4.0 / F * k, force_N=F), b.time_h, "squeeze"), rng)


def p_bruise(b, F):
    D = AB.derived(b.particles)
    thr = PR.AVOCADO["bruise_force_ratio"][1] * D["penetrometer_N"]
    return float(np.sum(b.weights * (F > thr)))


def p_stage(b, lo=-np.inf, hi=np.inf):
    s = b.particles["ripeness_stage"]
    return float(np.sum(b.weights * ((s >= lo) & (s < hi))))


# %% requirements (phase-indexed, over the plant trace)
def G_A(plant, decided_ready):
    """Task A: never bruised, and the ready / not-ready decision is right."""
    o = plant.outcome()
    return ("Bruised(Mesocarp)" not in o["facts"]) and (decided_ready == (o["stage"] >= READY_STAGE))


def G_B(plant):
    """Task B: separation only in the cut plane at the cut phase (no torn flesh), halves apart, seed out, cavity clean,
    no bruise, no unsafe tool use, the fruit edible when cut (eating quality >= 0.6) and not spoiled."""
    o = plant.outcome()
    f = o["facts"]
    return ("Apart(HalfA,HalfB)" in f and "SeedOut(HalfA)" in f and "TornFlesh(HalfB)" not in f
            and "TestaIn(Cavity)" not in f and "Bruised(Mesocarp)" not in f and "UnsafeToolUse(KnifeStrike)" not in f
            and o["eating_quality"] >= 0.6 and not o["spoiled"])


def failure_label(plant, task="B", decided_ready=None):
    o = plant.outcome()
    f = o["facts"]
    if "Bruised(Mesocarp)" in f:
        return "bruised"
    if task == "A":
        return "ok" if G_A(plant, decided_ready) else ("missed ripe" if o["stage"] >= READY_STAGE else "false ripe")
    for fact, lab in (("UnsafeToolUse(KnifeStrike)", "unsafe tool"), ("TornFlesh(HalfB)", "torn"),
                      ("TestaIn(Cavity)", "seed coat left")):
        if fact in f:
            return lab
    if "Apart(HalfA,HalfB)" not in f:
        return "not apart"
    if "SeedOut(HalfA)" not in f:
        return "seed stuck"
    if o["spoiled"]:
        return "spoiled"
    if o["eating_quality"] < 0.6:
        return "not edible" if o["stage"] < READY_STAGE else "overripe"
    return "ok"


# %% statistics
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(a, b):
    """Exact two-sided McNemar test on paired successes (lists of bool): discordant counts and p."""
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return n10, n01, min(1.0, p)


def verdict(k, n, q=0.8):
    lo, hi = wilson(k, n)
    return "verified" if lo >= q else ("refuted" if hi < q else "undecided")


def G_B_without_safety(plant):
    """G_B with the safety requirement left out (what a KPI-only reward would score)."""
    o = plant.outcome()
    f = o["facts"] - {"UnsafeToolUse(KnifeStrike)"}
    return ("Apart(HalfA,HalfB)" in f and "SeedOut(HalfA)" in f and "TornFlesh(HalfB)" not in f
            and "TestaIn(Cavity)" not in f and "Bruised(Mesocarp)" not in f
            and o["eating_quality"] >= 0.6 and not o["spoiled"])
