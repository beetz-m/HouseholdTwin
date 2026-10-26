"""Wave-1 kitchen tasks as action designators (roll, banana, carrot), bound to the hidden-state models.

Same structure as food/egg_tasks.py: goal, params (ranges people use + the twin's recommendation), steps, pre /
effects on the discrete SDT attributes (food/schema.py), failures -> (cue, recovery), model, sources.  run() executes a
designator symbolically against an instance's truth model and returns the trace of SDT state changes.
"""

import numpy as np

from . import banana_models as BM
from . import carrot_models as CM
from . import roll_models as RM

TASKS = {
    # ---------------------------------------------------------------- bread roll
    "PickRoll": {
        "item": "roll", "goal": "roll held, not dented",
        "params": {"grip_force_N": ("F_slip .. F_dent", None, "RollTruth.grasp_corridor; pads >= 20 x 20 mm"),
                   "approach": ("from above, across the short axis", None, "bag: open the bag mouth first")},
        "steps": ["locate roll in bag/basket", "open bag mouth", "approach across the width", "close at <= 5 cm/s",
                  "force-limited grip inside the corridor", "lift"],
        "pre": ["support in {bag, basket, board, plate}"], "effects": ["support := gripper"],
        "failures": {"dent": ("crumb stays compressed, crust cracks audible", "lower the grip force"),
                     "slip": ("roll rotates in the grip", "regrasp across the width with more pad area")},
        "model": "RollTruth.grasp_corridor", "sources": ["hirte_2012", "lagrain_2012"]},
    "SliceRollOpen": {
        "item": "roll", "goal": "cut_state = sliced_open (or hinged), crumb not squashed",
        "params": {"knife": ("serrated bread knife", None, "wikihow_slice"),
                   "F_down_N": (2.0, 12.0, "saw lightly: > crust break, below crumb squash (twin)"),
                   "stroke_mm": (30, 50, "+-20 mm strokes at 20 mm/s (luo_2026 robot set-point, baguette)"),
                   "height_frac": (0.45, 0.45, "through the middle"),
                   "hinge": ("stop 10 mm short", None, "for sandwiches (wikihow_slice)")},
        "steps": ["roll flat on board, score up", "palm flat on top (or second gripper pressing lightly)",
                  "start the blade in the middle of the long side", "saw back and forth, light pressure",
                  "halfway: stand on the uncut edge, finish downwards (tastingtable_bagel)", "open"],
        "pre": ["support = board", "cut_state = whole"], "effects": ["cut_state := sliced_open"],
        "failures": {"squashed": ("crumb compressed under the blade", "less down-force, more strokes"),
                     "crust skids": ("blade slides on the crust", "more down-force or start on the score side"),
                     "uneven halves": ("blade tilts", "keep the blade level; second gripper on top")},
        "model": "RollTruth.saw", "sources": ["wikihow_slice", "tastingtable_bagel", "luo_2026", "hirte_2012"]},
    "ReheatRoll": {
        "item": "roll", "goal": "crust_texture = crisp, temperature = warm",
        "params": {"T_C": (150, 170, "utopia_aufbacken"), "minutes": (5, 10, "dampen first if day-old")},
        "steps": ["dampen lightly (day-old)", "oven preheated", "bake 5-8 min", "rest 1 min"],
        "pre": ["cut_state = whole"], "effects": ["crust_texture := crisp", "temperature := warm"],
        "failures": {"still soft": ("no crackle on squeeze", "2 more minutes"),
                     "too dark": ("crust colour darker than golden", "lower temperature next time")},
        "model": "RollTruth.reheat", "sources": ["utopia_aufbacken", "hirte_2012"]},
    "ToastRollHalves": {
        "item": "roll", "goal": "toast_level = golden on the cut faces",
        "params": {"minutes": (2.5, 3.5, "toaster slots; kaur_2026 3 min light .. 4.5 min burning")},
        "steps": ["SliceRollOpen", "halves cut face in", "toast", "check colour"],
        "pre": ["cut_state = sliced_open"], "effects": ["toast_level := golden"],
        "failures": {"burnt": ("dark faces, smoke", "discard; shorter next time")},
        "model": "RollTruth.toast_faces", "sources": ["kaur_2026"]},
    # ---------------------------------------------------------------- banana
    "SeparateBanana": {
        "item": "banana", "goal": "one finger free, peel at the pedicel intact",
        "params": {"method": ("twist down at the pedicel", None, "no abscission zone: the peel ruptures (ketsa25)"),
                   "F_N": ("finger_drop_N(stage)", None, "~49 N green .. ~8 N ripe")},
        "steps": ["hold the hand at the crown", "grip the finger near the pedicel", "bend down and twist"],
        "pre": ["attached_to_hand = yes"], "effects": ["attached_to_hand := no"],
        "failures": {"peel tears open": ("pulp exposed at the neck", "eat/use soon"),
                     "won't separate": ("no movement", "cut the pedicel with a knife")},
        "model": "BananaTruth.separate_from_hand", "sources": ["imsabai", "mbeguie09"]},
    "PeelBanana": {
        "item": "banana", "goal": "peel_state = off, pulp unbruised, strings mostly on the peel",
        "params": {"end": ("stem (snap the neck) | tip ('monkey')", "stem if ripe, tip if the neck is too stiff",
                           "banana_howto; kim22: fresh neck too stiff"),
                   "strips": ("3-5", None, "peel splits along the ridges (khanal23)")},
        "steps": ["hold mid-finger", "snap the neck (or pinch the tip open)", "pull strip 1 to the far end",
                  "rotate, pull the remaining strips", "lift the pulp off the peel"],
        "pre": ["peel_state = on"], "effects": ["peel_state := off"],
        "failures": {"neck too stiff": ("neck bends, no break", "pinch the tip open instead"),
                     "strip tears short": ("strip breaks halfway", "regrasp at the torn edge"),
                     "pulp squashed": ("dent where held", "less grip; hold through the peel")},
        "model": "BananaTruth.snap_neck, peel_strip", "sources": ["banana_howto", "kim22", "khanal23"]},
    "SliceBananaRounds": {
        "item": "banana", "goal": "integrity = rounds of ~6 mm",
        "params": {"thickness_mm": (3, 6, "banana_howto"), "anti_browning": ("lemon juice", None, "banana_howto")},
        "steps": ["PeelBanana", "lay on board", "slice across", "toss with lemon juice if not used now"],
        "pre": ["peel_state = off"], "effects": ["integrity := rounds", "pulp_browning starts"],
        "failures": {"squashed rounds": ("oval rounds", "sharper knife; thicker rounds for overripe fruit")},
        "model": "BananaTruth.cut_rounds, browning_class", "sources": ["banana_howto"]},
    # ---------------------------------------------------------------- carrot
    "PeelCarrot": {
        "item": "carrot", "goal": "peel_state = peeled, ~2 mm removed",
        "params": {"tool": ("swivel or Y peeler", None, "carrot_cuts"),
                   "strokes": ("~ circumference / 17 mm per half", None, "derived; morpheus_2024 robot 90 % area")},
        "steps": ["hold by the crown, tip down over a bowl (or clamp)", "stroke away from the body", "rotate",
                  "repeat to the end", "turn round, peel the other half"],
        "pre": ["peel_state = unpeeled"], "effects": ["peel_state := peeled"],
        "failures": {"missed strips": ("darker skin strips (hard to see: morpheus_2024)", "feel / vibrate check"),
                     "slip": ("peeler skids off", "lower angle, firmer hold")},
        "model": "CarrotTruth.peel_stroke", "sources": ["con22", "carrot_cuts", "morpheus_2024"]},
    "CutCarrotRounds": {
        "item": "carrot", "goal": "cut_shape = rounds (5-6 mm)",
        "params": {"knife": ("chef's knife", None, "serrated chopping fails (luo_2026)"),
                   "F_N": ("41.1 (D/28.6)^2 N", None, "jah18 blade shear")},
        "steps": ["TopAndTail", "plank a flat side", "claw grip", "cut across"],
        "pre": ["support = board"], "effects": ["cut_shape := rounds"],
        "failures": {"rolls away": ("carrot rotates under the blade", "plank first"),
                     "stuck blade in the core": ("blade stops", "rock the knife")},
        "model": "CarrotTruth.cut", "sources": ["jah18", "carrot_cuts", "luo_2026"]},
    "BoilCarrot": {
        "item": "carrot", "goal": "cooked_state = crisp_tender (or tender)",
        "params": {"rounds_6mm_min": (4, 5, "carrot_boil; twin: 4.8 min"), "whole_min": (10, 15, "twin: 16 min")},
        "steps": ["water at a rolling boil", "add carrots", "time", "fork test", "drain"],
        "pre": ["cut_shape known"], "effects": ["cooked_state := crisp_tender"],
        "failures": {"mushy": ("falls apart on the fork", "shorter time; time by piece size")},
        "model": "carrot_models.boil (peng14 + conduction lag)", "sources": ["peng14", "carrot_boil"]},
}


def run(name, truth, **kw):
    """Execute a designator symbolically; returns (ok, trace of (step, outcome))."""
    tr = []
    if name == "PickRoll":
        lo, hi = truth.grasp_corridor()
        F = kw.get("F", (lo + hi) / 2)
        ok = lo < F < hi
        tr.append(("grip", {"F": F, "corridor": (lo, hi), "ok": ok}))
        return ok, tr
    if name == "SliceRollOpen":
        r = truth.saw(kw.get("F_down", 6.0), kw.get("strokes", 12))
        tr.append(("saw", r))
        return r["ok"] and not r.get("squashed"), tr
    if name == "ReheatRoll":
        tr.append(("reheat", truth.reheat(kw.get("T_C", 160), kw.get("minutes", 6))))
        return truth.crisp >= 0.5, tr
    if name == "ToastRollHalves":
        lvl = truth.toast_faces(kw.get("minutes", 3.0))
        tr.append(("toast", lvl))
        return lvl == "golden", tr
    if name == "SeparateBanana":
        r = truth.separate_from_hand(kw.get("F", 30.0))
        tr.append(("pull", r))
        return r["ok"], tr
    if name == "PeelBanana":
        r = truth.snap_neck(kw.get("F_tip", 15.0))
        tr.append(("snap_neck", r))
        if not r["ok"]:
            truth.neck = "pinched_tip"
            tr.append(("fallback", "monkey method: pinch the tip"))
        adh, seam, tear = truth.peel_bonds()
        F_pull = kw.get("F_pull", min(1.15 * (adh + 2 * seam), 0.9 * tear))   # inside the window
        for k in range(8):
            r = truth.peel_strip(F_pull, from_end="tip" if truth.neck != "snapped" else "stem")
            tr.append((f"strip_{k + 1}", r))
            if truth.peel_on <= 0.05:
                break
        return truth.peel_on <= 0.05, tr
    if name == "SliceBananaRounds":
        ok, tr = run("PeelBanana", truth)
        if not ok:
            return False, tr
        r = truth.cut_rounds(kw.get("thickness_mm"))
        tr.append(("cut", r))
        return not r["squash"], tr
    if name == "PeelCarrot":
        n = 0
        while truth.peeled_frac < 0.97 and n < 40:
            truth.peel_stroke()
            n += 1
        tr.append(("strokes", n))
        return truth.peeled_frac >= 0.97, tr
    if name == "CutCarrotRounds":
        r = truth.cut(kw.get("F", 50.0), serrated=kw.get("serrated", False))
        tr.append(("cut", r))
        return r["ok"], tr
    if name == "BoilCarrot":
        r = truth.boil(kw.get("shape", "slab"), kw.get("size_mm", 6.0), kw.get("minutes", 5.0))
        tr.append(("boil", r))
        return r["class"] in ("crisp_tender", "tender"), tr
    raise KeyError(name)


def demo(seed=0):
    """One pass of every designator on fresh instances (used by tools/wave1_tasks_doc.py)."""
    from .banana import Banana
    from .carrot import Carrot
    from .roll import Roll
    out = {}
    roll, ban, car = Roll(seed), Banana(seed), Carrot(seed)
    for name, spec in TASKS.items():
        truth = {"roll": lambda: RM.RollTruth(roll, 0.048), "banana": lambda: BM.BananaTruth(ban, 0.12),
                 "carrot": lambda: CM.CarrotTruth(car, 0.115)}[spec["item"]]()
        ok, tr = run(name, truth)
        out[name] = {"ok": ok, "trace": tr, "state": truth.discrete()}
    return out
