"""Halving and pitting an avocado on the emulator, with the node's belief updated after every step.

Scenarios on one instance (hidden truth from its instance.json), each a designator sequence (food/avocado_tasks.py):
  ripe          CutRoundSeed 360 -> TwistApart -> RemoveSeed spoon
  unripe        (stage forced to 1.8) CutRoundSeed 330 -> TwistApart (stuck) -> CutRoundSeed 360 (complete) ->
                TwistApart (torn) -> RemoveSeed spoon
  knife         CutRoundSeed 360 -> TwistApart -> RemoveSeed knife_twist
The belief starts from the register prior; the posteriors of seed length, ripeness and the seed-coat side are
compared with the truth.

Run: ~/Documents/LabAutomationTwin/env_cad/bin/python scripts/avocado_process_demo.py [assets/food/avocado_4]
Writes <instance>/process/<scenario>/ (USD states, process.json, belief.npz, sdt_belief.usda) and process/summary.json.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from food import avocado_belief as AB  # noqa: E402
from food import avocado_tasks as T  # noqa: E402
from food.avocado_emulator import AvocadoEmulator, PreconditionError  # noqa: E402

inst = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "assets", "food", "avocado_4"))
truth = json.load(open(os.path.join(inst, "instance.json")))["hidden_params"]
SCEN = {"ripe": (None, [("CutRoundSeed", dict(coverage_deg=360)), ("TwistApart", {}),
                        ("RemoveSeed", dict(method="spoon"))]),
        "unripe": (1.8, [("CutRoundSeed", dict(coverage_deg=330)), ("TwistApart", {}),
                         ("CutRoundSeed", dict(coverage_deg=360)), ("TwistApart", {}),
                         ("RemoveSeed", dict(method="spoon"))]),
        "knife": (None, [("CutRoundSeed", dict(coverage_deg=360)), ("TwistApart", {}),
                         ("RemoveSeed", dict(method="knife_twist"))])}
summary = {}
for name, (stage, steps) in SCEN.items():
    out_dir = os.path.join(inst, "process", name)
    emu = AvocadoEmulator(inst, out_dir, seed=2, stage=stage)
    b = AB.AvocadoBelief.prior(4000, seed=5)
    rng = np.random.default_rng(1)
    rows = []
    print(f"[{name}] stage {emu.stage:.2f}, testa adheres to {truth['testa_adheres_to']}")
    for task, kw in steps:
        try:
            out, obs = T.run(task, emu, **kw)
        except PreconditionError as e:
            print(f"   {task}: precondition fails - {e}")
            rows.append(dict(task=task, outcome="precondition_failed", note=str(e)))
            continue
        for o in obs:
            b.observe(o, rng)
        S = b.summary()
        rows.append(dict(task=task, params=kw, outcome=out.outcome, ok=out.ok, measured={k: v for k, v in
                                                                                         out.measured.items()
                                                                                         if not isinstance(v, list)},
                         events=out.events, observations=[o.kind for o in obs],
                         belief=dict(seed_length=[S["seed_length"]["mean"], S["seed_length"]["sd"]],
                                     ripeness_stage=[S["ripeness_stage"]["mean"], S["ripeness_stage"]["sd"]],
                                     testa_adheres_to_flesh=S["testa_adheres_to_flesh"]["mean"])))
        print(f"   {task:13s} -> {out.outcome:18s} {'; '.join(out.events)[:70]:70s} | seed {S['seed_length']['mean']:.1f}"
              f"+-{S['seed_length']['sd']:.1f} mm (true {truth['seed_length']:.1f}), stage "
              f"{S['ripeness_stage']['mean']:.2f}+-{S['ripeness_stage']['sd']:.2f} (true {emu.stage:.2f}), "
              f"P(testa->flesh) {S['testa_adheres_to_flesh']['mean']:.2f}")
    b.save(os.path.join(out_dir, "belief.npz"))
    AB.write_belief_layer(os.path.join(out_dir, "sdt_belief.usda"), os.path.join(inst, "sdt.usda"), b, "./belief.npz")
    summary[name] = dict(stage=emu.stage, steps=rows, pieces=emu.state.pieces, integrity=emu.state.integrity,
                         seed=emu.state.seed)
json.dump(dict(instance=inst, truth=dict(seed_length=truth["seed_length"], testa_adheres_to=truth["testa_adheres_to"],
                                        ripeness_stage=truth["ripeness_stage"]), scenarios=summary),
          open(os.path.join(inst, "process", "summary.json"), "w"), indent=1)
print("wrote", os.path.join(inst, "process"))
