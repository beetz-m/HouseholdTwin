"""M3 evidence: the after-breakfast tidy on perturbed kitchens (held-out seeds 1000..), scored against a SCRIPTED
ground truth written independently of the rules (the norm table below, in plain Python). Per kitchen: a random subset
of objects with random states, a random request (none / resident / guest) and sometimes a full place.

Measured (concept page, competence A):
  verdict accuracy      the robot's placement verdict per object vs the scripted one
  over-tidy violations  an object moved although it was in place or licensed            (target 0)
  guessing violations   an object without a known home placed anywhere                   (target 0)
  request compliance    resident request: object left out AND an obligation opened
  abstention            guest conflict or full belonging place: asked, not moved
  task success          every object ends where the scripted truth says (or asked about)

    ~/Documents/ConceptualCRAM/.venv/bin/python experiments/guard_eval.py [N]   -> experiments/guard_eval.json
"""
import contextlib
import io
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/Documents/ConceptualCRAM"))
from ccram.activity import load  # noqa: E402

ACT = load(HERE)
ACT.configure()
from ccram.kernel import Interpreter  # noqa: E402

import adapter as PA  # noqa: E402


def scripted_home(kind, states):
    """The household's conventions, written out by hand (not the rules): None = no known home."""
    if kind == "milk-pack":
        return "bin" if "empty" in states else "fridge" if "opened" in states else None
    if kind in ("plate", "cup", "knife"):
        return "dishwasher" if "dirty" in states else ("cupboard" if kind != "knife" else None)
    return {"coffee-jar": "cupboard", "sugar-jar": "counter"}.get(kind)


def kitchen(seed):
    rng = random.Random(seed)
    objs = {}
    pool = [("cup", ["dirty"]), ("cup", []), ("plate", ["dirty"]), ("plate", []), ("knife", ["dirty"]),
            ("milk-pack", ["opened"]), ("milk-pack", ["opened", "empty"]), ("jar", []), ("coffee-jar", []),
            ("sugar-jar", [])]
    for k, (kind, states) in enumerate(rng.sample(pool, rng.randint(4, 8))):
        objs[f"{kind}-{k}"] = dict(kind=kind, states=set(states), at="counter")
    req = rng.choice([None, "resident", "guest"])
    target = rng.choice([o for o, d in objs.items() if scripted_home(d["kind"], d["states"]) not in (None, "counter")]
                        or [None])
    full = rng.choice([None, None, "dishwasher", "cupboard", "fridge"])
    return objs, (req, target), full


def truth_outcome(objs, req, full):
    """What a correct robot does with each object: (verdict, final place or 'asked')."""
    tier, target = req
    out = {}
    for o, d in objs.items():
        home = scripted_home(d["kind"], d["states"])
        if tier == "resident" and o == target:
            out[o] = ("licensed", "counter")
        elif tier == "guest" and o == target:
            out[o] = ("misplaced", "asked")
        elif home is None:
            out[o] = ("indeterminate", "asked")
        elif home == "counter":
            out[o] = ("at-belonging-place", "counter")
        elif home == full:
            out[o] = ("misplaced", "asked")
        else:
            out[o] = ("misplaced", home)
    return out


def one(lib, seed, noise=None):
    objs, req, full = kitchen(seed)
    a = PA.PutAwayAdapter(objects=objs, survey_noise=noise, seed=seed)
    if req[0] and req[1]:
        a.requests.append(("r-1", req[0], req[1], "counter", "episodic"))
        if req[0] == "guest":
            a.people.append("guest")
    if full:
        a.places[full]["capacity"] = 0
    it = Interpreter(lib, a)
    failure = None
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            it.call_plan("tidy-after-breakfast", {"area": ("counter", "counter", {})})
        except Exception as e:                      # noqa: BLE001
            failure = f"{type(e).__name__}: {e}"
    truth = truth_outcome(objs, req if req[1] else (None, None), full)
    atoms = it.atoms()
    asked = {o for _, o in a.asked}
    moved = [n["bindings"]["object"]["symbol"] for n in it.episode if n.get("schema") == "putting-away"
             and n["verdict"] == "goal achieved"]
    res = dict(seed=seed, n=len(objs), failure=failure, request=req[0] if req[1] else None, full=full)
    res["verdict_ok"] = sum(1 for o, (v, _) in truth.items() if ["placement-verdict", o, v] in atoms
                            or (v == "misplaced" and a.truth[o]["at"] != "counter"))  # moved: its verdict was misplaced
    res["over_tidy"] = sum(1 for o in moved if truth[o][0] in ("at-belonging-place", "licensed"))
    res["guessing"] = sum(1 for o in moved if truth[o][0] == "indeterminate")
    res["final_ok"] = sum(1 for o, (_, f) in truth.items() if (f == "asked" and o in asked and a.truth[o]["at"] == "counter")
                          or (f != "asked" and a.truth[o]["at"] == f))
    if req[0] == "resident" and req[1]:
        res["complied"] = a.truth[req[1]]["at"] == "counter" and any(x[0] == "obligation-open" for x in atoms)
    if req[0] == "guest" and req[1]:
        res["abstained"] = req[1] in asked and a.truth[req[1]]["at"] == "counter"
    if full:
        hit = [o for o, (v, f) in truth.items() if f == "asked" and scripted_home(objs[o]["kind"], objs[o]["states"]) == full
               and not (req[1] == o)]
        res["full_abstained"] = all(o in asked and a.truth[o]["at"] == "counter" for o in hit) if hit else None
    res["success"] = failure is None and res["final_ok"] == len(objs)
    return res


CONDITIONS = {"perfect survey": None, "fallible survey": {"miss_dirty": 0.15, "misclassify": 0.10}}


def main(n=30):
    lib = ACT.library()
    out = {}
    for name, noise in CONDITIONS.items():
        rs = [one(lib, 1000 + i, noise) for i in range(n)]
        out[name] = dict(summary=summarise(rs, n), results=rs)
        print(name, json.dumps(out[name]["summary"]))
    json.dump(dict(note="M3 guard evaluation, perturbed kitchens, held-out seeds 1000..; scripted ground truth; "
                        "survey noise ILLUSTRATIVE", conditions=out),
              open(os.path.join(HERE, "experiments", "guard_eval.json"), "w"), indent=1)


def summarise(rs, n):
    objs = sum(r["n"] for r in rs)
    summ = dict(kitchens=n, objects=objs, verdict_accuracy=round(sum(r["verdict_ok"] for r in rs) / objs, 3),
                over_tidy=sum(r["over_tidy"] for r in rs), guessing=sum(r["guessing"] for r in rs),
                request_compliance=[sum(1 for r in rs if r.get("complied")), sum(1 for r in rs if "complied" in r)],
                guest_abstention=[sum(1 for r in rs if r.get("abstained")), sum(1 for r in rs if "abstained" in r)],
                full_place_abstention=[sum(1 for r in rs if r.get("full_abstained")),
                                       sum(1 for r in rs if r.get("full_abstained") is not None)],
                task_success=[sum(r["success"] for r in rs), n], failures=[r["failure"] for r in rs if r["failure"]])
    return summ


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
