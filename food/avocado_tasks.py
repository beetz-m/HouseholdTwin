"""Avocado halving and pitting as action designators, executed on the emulator (food/avocado_emulator.py).

Same structure as food/egg_tasks.py and food/wave1_tasks.py: goal, params (ranges people use + the twin's
recommendation), steps, pre / effects on the discrete SDT attributes (food/schema.py AVOCADO), failures -> (cue,
recovery), model, sources. coraplex mapping: CutRoundSeed = CuttingAction(technique=HALVING) with a cut-round-the-seed
tool path; TwistApart and RemoveSeed have no coraplex counterpart yet. run() executes a designator on the emulator and
returns the outcome and the observations it gives the node's belief (food/avocado_belief.py).
"""

import numpy as np

from .avocado_belief import Observation

TASKS = {
    "CutRoundSeed": {
        "item": "avocado", "goal": "integrity = scored: a lengthwise cut through skin and flesh all round the seed",
        "params": {"knife": ("chef's knife, 1.6 mm blade", None, "afm_howto"),
                   "coverage_deg": (355, 360, "the cut must close on itself, else a flesh bridge holds the halves"),
                   "F_max_N": (15, 40, "puncture at the stem end 6-15 N, slicing 5-15 N (twin, ILLUSTRATIVE)"),
                   "hold": ("fruit on a board, held from the side; turn the fruit, not the blade", None, "afm_howto")},
        "steps": ["place the fruit on its side on the board", "pierce at the stem end until the blade meets the seed",
                  "keep the edge on the seed coat", "turn the fruit once round under the blade", "withdraw"],
        "pre": ["integrity = whole", "fruit held"], "effects": ["integrity := scored"],
        "failures": {"partial_cut": ("blade stalls / gap in the cut line", "complete the cut over the gap"),
                     "no_entry": ("blade slides on the skin", "pierce at the stem button with the tip")},
        "model": "avocado_process.cut_round_seed", "sources": ["afm_howto"]},
    "TwistApart": {
        "item": "avocado", "goal": "integrity = halved; the seed stays in one half",
        "params": {"T_max_Nm": (0.2, 1.5, "ripe fruit separates at ~0.1 N m; unripe or a bridge needs more (twin)"),
                   "grip": ("one half in each hand / gripper, twist in opposite directions", None, "afm_howto")},
        "steps": ["grip both halves across the cut", "twist in opposite directions about the cut-plane normal",
                  "pull apart once the halves turn"],
        "pre": ["integrity in {scored, partly_scored}"], "effects": ["integrity := halved | halved_torn"],
        "failures": {"stuck": ("torque rises, halves do not turn", "check the cut line for a bridge; re-cut"),
                     "torn": ("flesh stays on the seed, rough cavity wall", "unripe: accept or wait 1-2 days")},
        "model": "avocado_process.twist_apart", "sources": ["afm_howto"]},
    "RemoveSeed": {
        "item": "avocado", "goal": "seed = out; cavity clean (no seed coat, no gouge)",
        "params": {"method": ("spoon | knife_twist | squeeze_out", "spoon", "afm_howto: spoon; knife strike "
                              "discouraged (farley2019); squeeze-out only when ripe"),
                   "F_max_N": (5, 25, "spoon lever (twin)"), "T_max_Nm": (0.2, 1.0, "knife twist (twin)")},
        "steps": ["hold the half that holds the seed, cut face up", "spoon: slide the bowl under the seed edge, "
                  "lever it out along the seed coat", "inspect the cavity: brown seed-coat patches? gouges?",
                  "remove remaining seed coat"],
        "pre": ["integrity in {halved, halved_torn}", "seed = in"], "effects": ["seed := out"],
        "failures": {"stuck": ("seed does not lift", "unripe: knife twist in a fixture, or wait"),
                     "testa_left": ("brown papery patches in the cavity", "peel them off with the spoon tip"),
                     "gouge": ("spoon took flesh", "lower the spoon angle; follow the seed coat")},
        "model": "avocado_process.remove_seed", "sources": ["afm_howto", "farley2019", "cs42"]},
}


def observations(outcome):
    """Belief observations from an outcome's *measured* part (never from its truth)."""
    m, t = outcome.measured, 0.0
    obs = []
    if outcome.action == "cut_round_seed" and m.get("stop_depth_mm"):
        psi = np.asarray(m["psi_deg"]) % 360
        d = np.asarray(m["stop_depth_mm"])

        def near(a, w=6.0):
            sel = np.abs((psi - a + 180) % 360 - 180) < w
            return float(np.median(d[sel])) if sel.any() else None
        side = [x for x in (near(0.0), near(180.0)) if x is not None]
        obs.append(Observation("seed_depths", dict(side=float(np.mean(side)) if side else None, top=near(90.0),
                                                   bottom=near(270.0)), t, "blade stop depths (cut round the seed)"))
    if outcome.action == "twist_apart" and outcome.ok:
        obs.append(Observation("twist", dict(clean=not m.get("flesh_on_seed_visible", False), T_Nm=m.get("T_peak_Nm"),
                                             full_cut=m.get("full_cut", True)), t, "twist torque + cavity inspection"))
    if outcome.action == "remove_seed" and outcome.ok:
        if m.get("F_peak_N") is not None:
            obs.append(Observation("seed_lift_force", dict(F_N=m["F_peak_N"]), t, "spoon lever force"))
        if m.get("T_peak_Nm") is not None:
            obs.append(Observation("seed_twist_torque", dict(T_Nm=m["T_peak_Nm"]), t, "knife twist torque"))
        if m.get("seed_length_mm") is not None:
            obs.append(Observation("seed_length_mm", dict(value=m["seed_length_mm"], sd=1.0), t,
                                   "extracted seed measured"))
        obs.append(Observation("testa_seen", dict(seen=bool(m.get("testa_visible_in_cavity"))), t,
                               "cavity inspection after pitting"))
    return obs


def run(name, emu, **kw):
    """Execute a designator on the emulator; returns (outcome, belief observations)."""
    if name == "CutRoundSeed":
        out = emu.cut_round_seed(kw.get("coverage_deg", 360.0), kw.get("F_max_N", 40.0))
    elif name == "TwistApart":
        out = emu.twist_apart(kw.get("T_max_Nm", 1.5))
    elif name == "RemoveSeed":
        out = emu.remove_seed(kw.get("method", "spoon"), kw.get("F_max_N", 25.0), kw.get("T_max_Nm", 1.0))
    else:
        raise KeyError(name)
    return out, observations(out)
