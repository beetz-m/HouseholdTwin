"""M2 evidence: crack-egg-cleanly on held-out eggs (register instances 1000-1029, emulator seed = instance), under
  belief          the belief decides surface + speed (P(shatter) <= 0.05), grip force from the H8 corridor, pre-act
                  verdict with P(success), weighing when its VOI >= 0.02, retries re-decided on the updated belief
  study           baseline: the crack study's best speed on the thin rim, +15 % per retry, 6 N, no belief
  belief-weak     belief corrupted: shells believed to break at 0.6 x their true force
  belief-strong   belief corrupted: shells believed to break at 1.6 x their true force
  cautious*       as belief*, but the first strike must also keep P(shatter) <= 0.05 if shells are 0.6 x as strong
                  as believed (robust chance constraint); after a strike outcome the plain bound applies
Outcome per egg: cracked into the cup (plan done) / shattered / crushed / retries exhausted; strikes; probes; whether
shell reached the batch-free cup (truth). Paired: the same instances in every condition.

    ~/Documents/ConceptualCRAM/.venv/bin/python experiments/strike_eval.py [N] [cond,cond]  -> experiments/strike_eval.json
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/Documents/ConceptualCRAM"))
from ccram.activity import load  # noqa: E402

ACT = load(HERE)
ACT.configure()
from ccram.kernel import Interpreter, KernelFailure  # noqa: E402

import adapter as EA  # noqa: E402

CONDITIONS = {"belief": dict(policy="belief"), "study": dict(policy="study"),
              "belief-weak": dict(policy="belief", bias={"F_break_equatorial": 0.6, "F_break_axial": 0.6}),
              "belief-strong": dict(policy="belief", bias={"F_break_equatorial": 1.6, "F_break_axial": 1.6}),
              "cautious": dict(policy="cautious"),
              "cautious-weak": dict(policy="cautious", bias={"F_break_equatorial": 0.6, "F_break_axial": 0.6}),
              "cautious-strong": dict(policy="cautious", bias={"F_break_equatorial": 1.6, "F_break_axial": 1.6})}


def one(lib, inst, cond):
    a = EA.EggAdapter(eggs={"egg-1": dict(instance=inst)}, seed=inst, policy=cond["policy"],
                      belief_bias=cond.get("bias"), out_dir=os.path.join(HERE, "episodes", "eval"))
    it = Interpreter(lib, a)
    import contextlib
    import io
    kind = None
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            it.call_plan("crack-egg-cleanly", {"egg": ("egg-1", "egg-1", {}), "cup": ("cup-1", "cup-1", {})})
        except KernelFailure as f:
            kind = f.kind
    steps = [n for n in it.episode if "schema" in n]
    strikes = [n for n in steps if n["schema"] == "cracking" and not n["verdict"].startswith("pre-empted")]
    truth = a.truth()["egg-1"]
    return dict(instance=inst, failure=kind, ok=kind is None, strikes=len(strikes),
                surfaces=[n["bindings"]["tool"]["symbol"] for n in strikes],
                speeds=[n["bindings"]["speed"]["symbol"] for n in strikes],
                p_success=[n.get("p_success") for n in strikes], probes=sum(1 for n in steps if n["schema"] == "weighing"),
                shell=truth["shell"], yolk=truth["yolk"], shards_left=truth["shards_in_vessel"])


def main(n=30, only=None):
    lib = ACT.library()
    out = os.path.join(HERE, "experiments", "strike_eval.json")
    res = json.load(open(out))["results"] if only and os.path.exists(out) else {}
    for name, cond in CONDITIONS.items():
        if only and name not in only:
            continue
        t0 = time.time()
        res[name] = [one(lib, 1000 + i, cond) for i in range(n)]
        r = res[name]
        print(f"{name:14s} ok {sum(x['ok'] for x in r):2d}/{n}  shattered {sum(x['shell'] == 'shattered' for x in r):2d}  "
              f"exhausted {sum(x['failure'] == 'goal-unachievable' for x in r):2d}  strikes {sum(x['strikes'] for x in r)}  "
              f"probes {sum(x['probes'] for x in r)}  ({time.time() - t0:.0f} s)", flush=True)
    json.dump(dict(note="M2 strike evaluation, held-out instances 1000.. (paired)", n=n, results=res),
              open(out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30, sys.argv[2].split(",") if len(sys.argv) > 2 else None)
