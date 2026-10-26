"""Egg kitchen tasks as action designators: what people do (docs/EGG_TASKS.md, researched on cooking sites), as
parameterised actions with preconditions, effects, failure modes and recoveries, bound to the twin's models.

Each designator:
  name, goal            the task and what success means (as conditions on the egg's discrete SDT attributes)
  params                free parameters with the ranges people use (and the twin's recommended value, where a model
                        computes one: strike speed from tools/egg_crack_study.py, boil time from food/egg_cook.py)
  steps                 the action sequence (sub-actions refer to other designators or motion primitives)
  pre / effects         discrete preconditions and effects on the SDT attributes
  failures              failure -> (detection cue, recovery)
  model                 the hidden-world model that decides the outcome in the twin
  sources               register source keys

run_task(name, egg, rng, **params) executes a designator symbolically against the twin's hidden models and returns
the trace of SDT state changes -- the same structure CRAM-style plans would execute.
"""

import numpy as np

from . import egg_crack as EC
from .dynamics import EggTruth

TASKS = {
    "PickUpEgg": {
        "goal": "egg held, shell_condition unchanged",
        "params": {"grip_force_N": (1.0, 15.0, "H8 corridor from EggBelief; people: 'firm but gentle'"),
                   "contact": ("equator, two pads, spread contact", None, "no point loads (ring, fingertip edge)")},
        "steps": ["detect egg in carton/tray", "check for cracks and a stuck egg (lift gently; resists -> stuck)",
                  "approach equator", "close at <= 0.1 m/s", "force-limited grip", "lift <= 2 m/s^2"],
        "pre": ["shell_condition in {intact, microcracked}", "support in {carton, tray, bowl, board}"],
        "effects": ["support := gripper"],
        "failures": {"slip": ("egg lags the gripper / tilts", "regrasp with higher force inside the corridor"),
                     "crack": ("crack line, wet spot", "shell_condition := cracked; use now, cook thoroughly"),
                     "stuck egg": ("resistance on lift", "free gently; cracked eggs: cook thoroughly or discard")},
        "model": "PhysX contacts + EggTruth.static_load; EggBelief.admissible (AICOR H8)",
        "sources": ["usda_handling", "masterclass_crack"],
    },
    "TransportEgg": {
        "goal": "egg at target support, intact",
        "params": {"max_accel": (0.5, 2.0, "m/s^2; keeps F_slip below grip force"),
                   "target": ("carton, bowl, cup, egg tray", None, "never loose on a bare counter: eggs roll in curves")},
        "steps": ["keep grip", "move", "place: descend <= 30 mm/s at contact", "release"],
        "pre": ["support = gripper"],
        "effects": ["support := target"],
        "failures": {"drop": ("egg leaves the gripper", "drop impact model: towel ~10 cm safe, stone ~1 cm")},
        "model": "EggTruth.impact (calibrated to cohen2025)",
        "sources": ["cookscountry_storage", "cohen2025"],
    },
    "CrackEgg": {
        "goal": "shell cracked open, contents in the cup/bowl, yolk whole, no shell inside",
        "params": {"surface": ("flat counter | bowl rim | another egg", "another egg", "sources disagree: atk_egg"),
                   "strike_speed": (0.2, 0.65, "m/s; twin optimum: egg-on-egg 0.5, counter 0.45, thin rim 0.25"),
                   "into": ("small cup first", None, "catch shell and bad eggs before the batch")},
        "steps": ["hold egg (PickUpEgg)", "strike equator once", "inspect: dent + sideways crack",
                  "over the cup: thumbs (or two grippers) into the dent, pull halves apart",
                  "inspect contents: shell fragments? yolk whole? smell"],
        "pre": ["support = gripper", "cup present"],
        "effects": ["shell_condition := opened", "contents := released (in cup)"],
        "failures": {"no crack": ("no dent", "strike again, a little faster"),
                     "check (membrane intact)": ("crack but halves won't separate", "tear membrane with thumb/edge"),
                     "shattered": ("egg everywhere / many fragments", "clean up; egg to scramble if contents saved"),
                     "shell in cup": ("fragment visible", "scoop with a half shell (or wet fingertip)"),
                     "yolk broken": ("yolk spreads", "divert egg to scrambling; keep the batch clean")},
        "model": "food/egg_crack.py strike + open_up + fish_out",
        "sources": ["atk_egg", "egginfo_crack", "atk_egg_on_egg", "eggs_ca"],
    },
    "SeparateEgg": {
        "goal": "yolk in yolk bowl (whole), white in white bowl with no yolk",
        "params": {"method": ("shell-to-shell | hand | slotted spoon | bottle", "slotted spoon", "shell edge can cut "
                              "the yolk"),
                   "egg_temperature": ("cold", None, "cold yolks are firmer")},
        "steps": ["three bowls: transition, yolk, whites", "crack on a flat surface (CrackEgg into transition bowl)",
                  "hold the yolk back (spoon/shell/hand), let the white drain", "yolk -> yolk bowl",
                  "inspect white, then -> whites bowl"],
        "pre": ["egg cold (fridge)", "three clean glass/metal bowls"],
        "effects": ["contents := separated"],
        "failures": {"yolk in white": ("yellow streak or speck", "lift with a half shell; streak -> discard that "
                                      "white (fat stops whipping)")},
        "model": "EC.open_up yolk-break model (age, drop height, shards); separation success ILLUSTRATIVE",
        "sources": ["atk_crack_separate", "egginfo_separate", "foodrepublic_separate"],
    },
    "BoilEgg": {
        "goal": "yolk doneness = target (runny | jammy | creamy | firm), shell intact, peels cleanly",
        "params": {"start": ("hot (lower into simmering water) | cold", "hot", "hot start peels ~90 % vs ~50 %"),
                   "time_min": (4.5, 13.0, "food/egg_cook.py: room-temp medium egg 6 jammy, 8.5 creamy, 10+ firm; "
                                           "fridge egg +1-1.5 min"),
                   "cooling": ("ice bath 5 min | cold running water", None, "")},
        "steps": ["bring water to a rolling simmer", "lower egg with a spoon (no drop)", "time", "transfer to ice bath",
                  "peel: tap all over, roll, start at the blunt end under the membrane, under running water"],
        "pre": ["shell_condition = intact"],
        "effects": ["contents := cooked(doneness)"],
        "failures": {"crack in the pot": ("white streams out", "salt/vinegar only helps it set; lower gently next time"),
                     "overcooked": ("grey-green ring (>14 min)", "shorter time; cool fast"),
                     "hard to peel": ("white tears", "older eggs (7-10 d), hot start, peel under water")},
        "model": "food/egg_cook.py EggHeat (conduction through the tissue map; thresholds CALIBRATED)",
        "sources": ["alphafoodie_boil", "kenji_boil", "incredibleegg_boil", "milkstreet_peel"],
    },
    "ScrambleEggs": {
        "goal": "curds set but glossy, no liquid egg, not browned",
        "params": {"heat": ("low | medium | medium-high then low", "medium-high then low", "ATK dual heat"),
                   "stir": ("constant (small curds) | push edges to centre (large curds)", None, ""),
                   "salt": ("early (10-15 min before) | at the end", "early", "sources disagree")},
        "steps": ["crack eggs (CrackEgg) into bowl", "beat with fork until uniform yellow", "butter in nonstick pan, "
                  "foaming not brown", "add eggs; scrape/push curds", "lower heat; fold", "off the heat while glossy"],
        "pre": ["eggs opened in bowl"],
        "effects": ["contents := scrambled"],
        "failures": {"rubbery/weeping": ("tight curds in water", "less heat, off earlier, add cold cream")},
        "model": "not yet simulated (coagulation thresholds from opentextbc)",
        "sources": ["atk_scrambled", "incredibleegg_scrambled", "pepin_pbs"],
    },
    "FrySunnySideUp": {
        "goal": "white set and opaque incl. around the yolk, yolk whole and runny, no shell",
        "params": {"fat": ("1 tsp per egg", None, ""), "set_top": ("lid | steam (1 tsp water + lid) | baste", "lid", ""),
                   "time_min": (2.0, 4.0, "1 min open + 2-3 min lid on low")},
        "steps": ["preheat pan medium with fat", "CrackEgg into a cup", "slide from the cup at pan level",
                  "cook open ~1 min until edges white", "lid / steam / baste until the white is opaque",
                  "slide out with a thin spatula"],
        "pre": ["egg opened in a cup, yolk whole"],
        "effects": ["contents := fried(sunny side up)"],
        "failures": {"broken yolk": ("yolk spreads", "serve over hard / scramble"),
                     "raw white near yolk": ("translucent ring", "lid or baste longer")},
        "model": "not yet simulated",
        "sources": ["loveandlemons_fried", "incredibleegg_fried", "kenji_baste"],
    },
}


def run_task(name, egg, rng, mass_kg=None, **kw):
    """Execute a designator symbolically on the hidden twin; returns (success, trace). Implemented for CrackEgg,
    SeparateEgg-on-crack and BoilEgg; the others return the designator only."""
    mass_kg = mass_kg or egg.V_outer * 1087
    trace = []
    if name in ("CrackEgg", "SeparateEgg"):
        t = EggTruth(egg, mass_kg)
        surface = kw.get("surface", "another egg")
        tool = {"another egg": {"kind": "egg"}, "flat counter": {"kind": "flat", "surface": "granite"},
                "bowl rim": {"kind": "rim", "surface": "bowl_rim_ceramic", "edge_radius": 1.5e-3}}[surface]
        other = EggTruth(kw["other_egg"], kw.get("other_mass", mass_kg)) if tool["kind"] == "egg" else None
        v = kw.get("strike_speed", 0.5)
        for attempt in range(3):
            r = EC.strike(t, v, tool, rng, other=other)
            trace.append(("strike", attempt + 1, surface, v, r.state, round(r.F_peak, 1), round(r.F_crack, 1)))
            target = t if r.which == "this" else other
            if r.state in ("check", "cracked_open", "shattered"):
                break
            v *= 1.15                                       # recovery: strike again a little faster
        if r.state == "shattered":
            return False, trace + [("shattered", "clean up")]
        if r.state not in ("check", "cracked_open"):
            return False, trace + [("not cracked after 3 strikes",)]
        ok, yolk, shards = EC.open_up(target, rng, age_days=egg.p["age_days"])
        trace.append(("open_up", r.state, "yolk " + yolk, f"{shards} shard(s) in cup"))
        left = EC.fish_out(shards, rng) if shards else 0
        if shards:
            trace.append(("fish_out with half shell", f"{left} left"))
        success = ok and yolk == "whole" and left == 0
        if name == "SeparateEgg" and success:
            sep_ok = rng.random() > 0.05                   # ILLUSTRATIVE yolk break in the spoon/shell transfer
            trace.append(("separate", "clean" if sep_ok else "yolk in white -> discard white"))
            success = sep_ok
        return success, trace
    if name == "BoilEgg":
        from .egg_cook import EggHeat, hot_start
        H = EggHeat(egg)
        T0 = kw.get("egg_temp_C", 4.0)
        tm = kw.get("time_min", 8.5)
        _, d = H.run(T0, hot_start(98.0), tm * 60)
        trace.append(("boil", f"{tm} min from {T0} C", d["yolk"], f"centre max {d['T_centre_max']:.0f} C"))
        return d["yolk"] == kw.get("target", d["yolk"]), trace
    return None, [("designator only", TASKS[name]["steps"])]


def best_boil_time(egg, target, egg_temp_C=4.0, lo=3.0, hi=16.0):
    """Shortest time that reaches the target doneness (bisection on the heat model)."""
    from .egg_cook import EggHeat, hot_start
    order = ["runny", "jammy", "creamy", "firm"]
    H = EggHeat(egg)
    for _ in range(9):
        mid = (lo + hi) / 2
        _, d = H.run(egg_temp_C, hot_start(98.0), mid * 60)
        if order.index(d["yolk"]) >= order.index(target):
            hi = mid
        else:
            lo = mid
    return round(hi, 2)


# ================================================================================================ on the emulator
# The designators executed on the egg emulator (food/egg_emulator.py, like food/avocado_tasks.run): each run returns
# the outcomes of its steps and the observations they give the egg's belief (food/egg_belief.py) -- from the
# measured part only, never from the truth.
RIM = {"kind": "rim", "surface": "bowl_rim_ceramic", "edge_radius": 2e-3}
FLAT = {"kind": "flat", "surface": "granite"}


def observations(outcome, t_h=0.0):
    """Belief observations of one emulator outcome (measured part only)."""
    from .egg_belief import Observation
    m, a = outcome.measured, outcome.action
    src = f"{a} (emulator)"
    if a == "read_carton":
        return [Observation("carton_bbd", {"days_left": m["days_left"]}, t_h, src)]
    if a == "weigh":
        return [Observation("mass_g", {"value": m["mass_g"]}, t_h, src)]
    if a == "candle":
        return [Observation("candling", {"air_cell_mm": m["air_cell_mm"], "crack_seen": m["crack_seen"]}, t_h, src)]
    if a == "float_test":
        return [Observation("float_test", {"floats": m["floats"], "tilt_deg": m.get("tilt_deg")}, t_h, src)]
    if a == "grip":
        return [Observation("grip", {"F_N": m["F_N"], "slipped": m["slipped"], "cracked": m["cracked"]}, t_h, src)]
    if a == "strike":
        tool = RIM if m["tool"] == "rim" else FLAT
        return [Observation("strike", {"v": m["v"], "tool": tool, "outcome": m["outcome"]}, t_h, src)]
    if a == "open_up" and m:
        o = {"albumen_height_mm": m["albumen_height_mm"]}
        if not m.get("yolk_broken"):
            o["yolk_index"] = m["yolk_index"]
        return [Observation("opened", o, t_h, src)]
    if a == "peel":
        return [Observation("peel", {"clean": m["clean"]}, t_h, src)]
    return []


def run(name, emu, **kw):
    """Execute a designator on the emulator. Returns (success, outcomes, observations)."""
    outs = []

    def do(fn, *a, **k):
        o = fn(*a, **k)
        outs.append(o)
        return o
    if name == "CheckFreshness":                        # carton, scale, candling lamp, float test (no damage)
        for fn in (emu.read_carton, emu.weigh, emu.candle):
            do(fn)
        if kw.get("float_test", True):
            do(emu.float_test)
        ok = True
    elif name == "PickUpEgg":
        o = do(emu.grip, kw.get("grip_force_N", 4.0), kw.get("polar", np.pi / 2), kw.get("a_max", 2.0))
        ok = o.outcome == "held"
    elif name in ("CrackEgg", "SeparateEgg"):
        tool = {"bowl rim": RIM, "flat counter": FLAT}[kw.get("surface", "bowl rim")]
        v = kw.get("strike_speed", 0.2)
        o = None
        for _ in range(kw.get("max_strikes", 4)):        # recovery: strike again 15 % faster (TASKS failures)
            o = do(emu.strike, v, tool)
            if o.outcome != "intact":
                break
            v *= 1.15
        ok = False
        if o.outcome in ("check", "cracked_open"):
            op = do(emu.open_up, True, kw.get("drop_height", 0.03))
            ok = op.ok
            if emu.s.shards_in_bowl:
                ok = do(emu.fish_out).ok and ok
    elif name == "BoilEgg":
        o = do(emu.boil, kw.get("time_min", 8.5), kw.get("egg_temp_C", 6.0), kw.get("start", "hot"))
        ok = o.outcome == kw.get("target", o.outcome)
        if kw.get("peel", True):
            ok = do(emu.peel).ok and ok
    else:
        raise ValueError(f"designator {name} has no emulator binding")
    obs = [x for o in outs for x in observations(o, emu.s.time_h)]
    return ok, outs, obs


TASKS["CheckFreshness"] = {
    "goal": "belief over age, air cell and interior grade narrowed; egg still intact",
    "params": {"float_test": (True, None, "wets the shell (grip friction lower until dry)")},
    "steps": ["read the best-before on the carton (laid <= 28 d before)", "weigh", "candle: air cell depth, cracks",
              "float test in a bowl of water: lies flat / tilts / stands on its end / floats"],
    "pre": ["shell_condition = intact"], "effects": ["(observations only)"],
    "failures": {"floats": ("egg floats", "past use: discard (very old or a leaker)"),
                 "crack seen": ("bright line when candling", "use now, cook thoroughly")},
    "model": "food/egg_lifetime.py (ageing, air cell, float statics) + food/egg_belief.py measurement models",
    "sources": ["pan2026", "usda_std", "eu2023_2465", "ro"],
}
