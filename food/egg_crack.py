"""Cracking and opening an egg: strike outcomes on a flat surface, a bowl rim or another egg, and opening over a
vessel -- the hidden-world model behind the CrackOnSurface / OpenUp actions.

Structure from the kitchen literature (register sources atk_egg, egginfo_crack, atk_egg_on_egg, wh_separate):
  flat surface  fewer shards pushed into the egg, but several cracks and the membrane often stays intact (a 'check')
                and must be torn by hand; too hard -> egg everywhere
  bowl rim      clean break round the equator, tears the membrane, but small shards that can puncture the yolk;
                a thick rim crushes, a thin rim cracks
  another egg   only the weaker egg cracks, leaving a clean dent; the membrane holds the shards
Force and stiffness come from the shell model (Reissner stiffness, crack force from the register); the outcome
bands, shard rates and puncture probabilities are ILLUSTRATIVE and listed in params.EGG (crack_*).

Outcome states (USDA vocabulary): intact, microcracked, check (cracked, membranes intact), cracked_open (membrane
torn: ready to pull apart), shattered (crushed, contents leaking, many fragments).
"""

from dataclasses import dataclass, field

import numpy as np

from .dynamics import SURFACES, series

G = 9.81
P = {   # ILLUSTRATIVE constants of the strike / opening model (mirrored in params.EGG with sources)
    "coupled_mass": 0.05,          # kg of hand/gripper mass coupled through the grasp during the first ms
    "tear_ratio": {"flat": 1.8, "rim": 1.05, "egg": 3.0},     # F / F_crack at which the membrane tears
    "shatter_ratio": {"flat": 2.6, "rim": 2.2, "egg": 2.4},
    "shards_per_ratio": {"flat": 0.15, "rim": 0.6, "egg": 0.0},  # mean small shards pushed inside, x (F/Fc)
    "puncture_per_shard": 0.06,
    "yolk_break_base": 0.02,
    "yolk_break_per_week": 0.01,    # vitelline membrane weakens with age (wiki vitelline)
    "yolk_break_per_cm_drop": 0.04,  # contents falling more than 3 cm
    "thumb_tear_shard": 0.15,       # tearing an intact membrane with thumbs pushes a fragment in
    "shard_falls_in": 0.5,
    "fish_out_success": 0.9,        # half-shell scoop (atk_egg, jamie oliver tip)
}


@dataclass
class CrackResult:
    state: str
    F_peak: float
    F_crack: float
    ratio: float
    shards_inside: int = 0
    cracks: int = 0
    which: str = "this"            # egg-on-egg: which egg cracked
    detail: dict = field(default_factory=dict)


def strike(egg_truth, v, tool, rng, polar=np.pi / 2, other=None):
    """Hand/gripper-held strike of `egg_truth` (food.dynamics.EggTruth) at normal speed v [m/s] against `tool`:
    {"kind": "flat"|"rim"|"egg", "surface": SURFACES key, "edge_radius": m}. `other` = EggTruth for egg-on-egg."""
    e = egg_truth
    kind = tool["kind"]
    m_eff = e.m * e.egg.p["impact_effective_mass_fraction"] + P["coupled_mass"]
    k_self = e.egg.contact_stiffness(polar)
    if kind == "egg":
        k_other = other.egg.contact_stiffness(np.pi / 2)
        F = v * np.sqrt(series(k_self, k_other) * m_eff)
        Fc_self = e.crack_force(polar) * e.egg.p["impact_rate_factor"]
        Fc_other = other.crack_force(np.pi / 2) * other.egg.p["impact_rate_factor"]
        target, Fc, which = (e, Fc_self, "this") if Fc_self <= Fc_other else (other, Fc_other, "other")
    else:
        k_surf = SURFACES[tool.get("surface", "granite")][0]
        F = v * np.sqrt(series(k_self, k_surf) * m_eff)
        r_edge = tool.get("edge_radius") if kind == "rim" else None
        Fc = e.crack_force(polar, r_edge) * e.egg.p["impact_rate_factor"]
        target, which = e, "this"
    x = F / Fc
    tear, shatter = P["tear_ratio"][kind], P["shatter_ratio"][kind]
    if x < e.egg.p["microcrack_band"]:
        state = "intact"
    elif x < 1.0:
        state = "microcracked" if rng.random() < (x - e.egg.p["microcrack_band"]) / (1 - e.egg.p["microcrack_band"]) \
            else "intact"
    elif x >= shatter:
        state = "shattered"
    elif x >= tear or (kind == "rim" and rng.random() < min(1.0, 0.6 + 0.4 * (x - 1))):
        state = "cracked_open"
    else:
        state = "check"
    shards = int(rng.poisson(P["shards_per_ratio"][kind] * x)) if state in ("check", "cracked_open") else 0
    if state == "shattered":
        shards = int(rng.poisson(6))
    cracks = 0 if state in ("intact", "microcracked") else (1 if kind in ("rim", "egg") else 1 + int(rng.poisson(x)))
    # map onto the truth object's shell vocabulary
    target.shell = {"intact": target.shell, "microcracked": "microcracked", "check": "cracked",
                    "cracked_open": "cracked", "shattered": "shattered"}[state]
    if state == "microcracked":
        target.strength = 1 - target.egg.p["microcrack_strength_loss"]
    target.membrane = "intact" if state in ("intact", "microcracked", "check") else "torn"
    target.shards_inside = shards
    return CrackResult(state, float(F), float(Fc), float(x), shards, cracks, which,
                       {"kind": kind, "v": v, "m_eff": m_eff})


def open_up(egg_truth, rng, crack_up=True, drop_height=0.03, age_days=7.0):
    """Pull the halves apart over a vessel (thumbs or two grippers). Returns (ok, yolk, shards_in_vessel)."""
    e = egg_truth
    if e.shell != "cracked":
        return False, e.yolk, 0
    shards = getattr(e, "shards_inside", 0)
    if getattr(e, "membrane", "torn") == "intact":           # a 'check': membrane must be torn by hand
        if rng.random() < P["thumb_tear_shard"]:
            shards += 1
        e.membrane = "torn"
    if not crack_up:
        e.contents = "spilled"
        return False, e.yolk, shards
    in_vessel = int(rng.binomial(shards, P["shard_falls_in"]))
    p_break = (P["yolk_break_base"] + P["yolk_break_per_week"] * age_days / 7 + P["puncture_per_shard"] * shards
               + P["yolk_break_per_cm_drop"] * max(0.0, drop_height * 100 - 3))
    if rng.random() < min(p_break, 1.0):
        e.yolk = "broken"
    e.shell, e.contents = "opened", "released"
    return True, e.yolk, in_vessel


def fish_out(shards, rng, attempts=3):
    """Remove shell fragments from the vessel with a half shell; returns fragments left."""
    for _ in range(attempts):
        shards -= int(rng.binomial(shards, P["fish_out_success"]))
        if shards == 0:
            break
    return shards
