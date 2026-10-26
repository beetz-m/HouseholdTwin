"""Wave-2 kitchen tasks as action designators (loaf, tomato, potato, onion, orange), bound to the hidden models.
Same structure as food/wave1_tasks.py; run() executes one symbolically and returns (ok, trace)."""

from . import loaf_models as LM
from . import onion_models as OM
from . import orange_models as XM
from . import potato_models as PM
from . import tomato_models as TM

TASKS = {
    "TakeToastSlice": {
        "item": "loaf", "goal": "one slice out of the bag, untorn; bag closed again",
        "params": {"pull_N": (0.2, 2.0, "below the slice's tensile limit (nuss90, scaled)")},
        "steps": ["open clip / twist tie", "pinch the end slice at its top edge", "tilt away, lift out", "close bag"],
        "pre": ["slices_left > 0"], "effects": ["slices_left -= 1"],
        "failures": {"torn": ("slice tears at the grip", "grip lower and wider, pull slower")},
        "model": "LoafTruth.take_slice; Isaac: loaf 'slices' state (one rigid body per slice)",
        "sources": ["test25_toast", "nuss90"]},
    "ToastSlice": {
        "item": "loaf", "goal": "toast_level = golden",
        "params": {"seconds": (80, 150, "per toaster (toaster26)")},
        "steps": ["slice into the slot", "set the time", "pop / stop early when golden", "lift out by the edge"],
        "pre": ["slice held"], "effects": ["toast_level := golden"],
        "failures": {"burnt": ("dark faces, smoke", "stop button; shorter next time (acrylamide: go for gold)")},
        "model": "LoafTruth.toast", "sources": ["toaster26", "toast_misc"]},
    "SliceTomato": {
        "item": "tomato", "goal": "slices 6-13 mm, not squashed, gel kept",
        "params": {"knife": ("serrated | sharp chef's knife", "serrated", "tomato_howto"),
                   "thickness_mm": (6, 13, "tomato_howto")},
        "steps": ["core: cone cut round the stem scar", "lay on its side", "saw lightly", "transfer slices flat"],
        "pre": ["on board"], "effects": ["cut_state := slices"],
        "failures": {"squashed": ("deformed slices, juice everywhere", "serrated knife, less down-force"),
                     "gel loss": ("seeds and gel on the board", "cut through the locules last")},
        "model": "TomatoTruth.cut", "sources": ["tomato_howto", "ince16"]},
    "PeelTomato": {
        "item": "tomato", "goal": "skin off, flesh firm",
        "params": {"blanch_s": (30, 45, "tomato_howto"), "score": ("X 2.5 cm at the blossom end", None, "")},
        "steps": ["core", "score X", "boiling water 30-45 s", "ice water", "slip skin off"],
        "pre": ["skin = on"], "effects": ["skin := off"],
        "failures": {"tight skin": ("skin does not slip", "10-15 s more"),
                     "cooked outer wall": ("soft, translucent outer flesh", "shorter blanch")},
        "model": "TomatoTruth.blanch", "sources": ["tomato_howto"]},
    "BoilPotatoes": {
        "item": "potato", "goal": "done: a skewer passes without resistance",
        "params": {"pieces": ("whole | halves | 2-3 cm cubes", None, "gritzer"),
                   "start": ("cold salted water (1-2 %)", None, "gritzer")},
        "steps": ["scrub or peel", "cut to even size", "cold water + salt", "bring to a simmer", "skewer test",
                  "drain"],
        "pre": ["raw"], "effects": ["cooked_state := done"],
        "failures": {"hard centre": ("skewer stops in the centre", "simmer longer; cut smaller"),
                     "sloughing": ("outside falls apart (mealy types)", "firm type for salads; gentle simmer")},
        "model": "potato_models.boil (conduction + first-order softening, calibrated to 2 cm cubes)",
        "sources": ["gritzer", "potato_cook", "kk18"]},
    "PeelPotato": {
        "item": "potato", "goal": "peeled, eyes and green parts removed",
        "params": {"depth_mm": (1.0, 1.2, "potato_peel")},
        "steps": ["hold firmly", "peel away from the body", "gouge eyes with the peeler tip",
                  "cut away green parts generously", "keep in cold water"],
        "pre": ["unpeeled"], "effects": ["peel_state := peeled", "~10 % mass removed (twin)"],
        "failures": {"green under the skin": ("green tinge", "cut deeper / discard (glycoalkaloids)")},
        "model": "potato_models.peel_loss_fraction", "sources": ["potato_peel", "potato_storage"]},
    "DiceOnion": {
        "item": "onion", "goal": "even dice of the target size, root end kept until last",
        "params": {"spacing_mm": (3, 19, "mince .. coarse (onion_howto)"), "horizontal_cuts": (2, 3, "")},
        "steps": ["cut off the top", "halve through the root", "peel dry skin (+ tough first scale)",
                  "flat side down, horizontal cuts to the root", "vertical cuts to the root", "slice across",
                  "discard the root end"],
        "pre": ["dry_skin_on or peeled"], "effects": ["cut_state := diced"],
        "failures": {"layers fall apart": ("scales slide", "keep the root on; claw grip"),
                     "tears": ("irritation", "sharp knife (blunt: ~40x more droplets, wu25)")},
        "model": "OnionTruth.dice, onion_models.lf_dose", "sources": ["onion_howto", "lokke12", "wu25"]},
    "SauteOnion": {
        "item": "onion", "goal": "translucent | golden | caramelised as asked",
        "params": {"minutes": (5, 45, "onion_misc (typical)")},
        "steps": ["fat in the pan", "add diced onion", "stir", "lower heat for caramelising"],
        "pre": ["diced or sliced"], "effects": ["cooked_state"], "failures": {"burnt": ("black edges", "lower heat")},
        "model": "onion_models.saute_state", "sources": ["onion_misc"]},
    "JuiceOrange": {
        "item": "orange", "goal": "juice in the glass, few seeds",
        "params": {"roll_s": (10, 15, "orange_juice"), "temperature": ("room", None, "cold yields less")},
        "steps": ["roll on the counter", "halve at the equator", "ream with a twist", "strain"],
        "pre": ["whole"], "effects": ["juice := fresh"],
        "failures": {"bitter": ("navel juice after hours", "drink soon (edis891)")},
        "model": "OrangeTruth.juice", "sources": ["orange_juice", "ucr_citrus", "edis891"]},
    "PeelOrange": {
        "item": "orange", "goal": "peel off in few pieces, segments intact",
        "params": {"start": ("round the navel, thumb under the peel", None, "orange_howto")},
        "steps": ["score round the navel", "thumb under the peel", "work round in a spiral", "pull off white pith"],
        "pre": ["peel on"], "effects": ["peel_state := off"],
        "failures": {"strips break": ("short pieces (Valencia, cold fruit)", "room temperature; score quarters"),
                     "oil spray": ("mist when bending the peel", "bend gently (smith18)")},
        "model": "OrangeTruth.peel, bend_peel", "sources": ["ritenour13", "smith18", "orange_howto"]},
}


def run(name, truth, **kw):
    tr = []
    if name == "TakeToastSlice":
        r = truth.take_slice(kw.get("F", 1.0))
        tr.append(("take", r))
        return r["ok"], tr
    if name == "ToastSlice":
        lvl = truth.toast(kw.get("seconds", truth.t_golden))
        tr.append(("toast", lvl))
        return lvl == "golden", tr
    if name == "SliceTomato":
        oks = []
        for _ in range(int(kw.get("slices", 6))):
            r = truth.cut(kw.get("knife", "serrated"), kw.get("F_down", 4.0))
            tr.append(("cut", r))
            oks.append(r["ok"] and not r["squashed"])
        return all(oks), tr
    if name == "PeelTomato":
        r = truth.blanch(kw.get("seconds", 40))
        tr.append(("blanch", r))
        return truth.skin == "loose" and "cooked" not in r, tr
    if name == "BoilPotatoes":
        r = truth.boil(kw.get("size_mm", 25.0), kw.get("minutes", 20.0), kw.get("shape", "cube"))
        tr.append(("boil", r))
        return r["done"], tr
    if name == "PeelPotato":
        f = truth.peel()
        tr.append(("peel_loss", f))
        return True, tr
    if name == "DiceOnion":
        r = truth.dice(kw.get("spacing_mm", 6.0))
        tr.append(("dice", r))
        tr.append(("lf_dose", OM.lf_dose(r["cut_area_cm2"], 10, kw.get("knife", "sharp"))))
        return True, tr
    if name == "SauteOnion":
        st = OM.saute_state(kw.get("minutes", 8.0))
        tr.append(("saute", st))
        return st == kw.get("target", "translucent"), tr
    if name == "JuiceOrange":
        r = truth.juice()
        tr.append(("juice", r))
        return r["juice_ml"] > 40, tr
    if name == "PeelOrange":
        for k in range(10):
            r = truth.peel(kw.get("F", 1.1 * truth.strip_force()))
            tr.append((f"strip_{k + 1}", r))
            if truth.peel_on <= 0.05 or not r["ok"]:
                break
        return truth.peel_on <= 0.05, tr
    raise KeyError(name)


def demo(seed=0):
    from .loaf import Loaf
    from .onion import Onion
    from .orange import Orange
    from .potato import Potato
    from .tomato import Tomato
    objs = {"loaf": lambda: LM.LoafTruth(Loaf(seed), 0.5), "tomato": lambda: TM.TomatoTruth(Tomato(15), 0.072),
            "potato": lambda: PM.PotatoTruth(Potato(seed), 0.126), "onion": lambda: OM.OnionTruth(Onion(seed), 0.146),
            "orange": lambda: XM.OrangeTruth(Orange(seed), 0.19)}
    cache, out = {}, {}
    for name, spec in TASKS.items():
        if spec["item"] not in cache:
            cache[spec["item"]] = objs[spec["item"]]
        truth = cache[spec["item"]]()
        ok, tr = run(name, truth)
        out[name] = {"ok": ok, "trace": tr, "state": truth.discrete()}
    return out
