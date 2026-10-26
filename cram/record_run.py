"""Run the scripted PR2 avocado scenario (cram/avocado_pr2_demo.py) headless and save the robot's records -- belief
variables, competence answers, imagined outcomes, execution monitoring -- and the hidden truth to
cram/results/pr2_run.json (read by report/build_report.py).

Run: source ~/ros_ws/install/setup.bash; source ~/.virtualenvs/cram-env/bin/activate
     CORAPLEX_VISUALIZATION=none python ~/Documents/HouseholdTwin/cram/record_run.py
"""

import json
import os
import sys
import time
from dataclasses import asdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("CORAPLEX_VISUALIZATION", "none")

import avocado_pr2_demo as D  # noqa: E402


def main():
    t0 = time.time()
    world, node = D.main()
    truth = D.TrueAvocado(D.INSTANCE)
    out = dict(
        when=time.strftime("%Y-%m-%d %H:%M"), seconds=round(time.time() - t0, 1),
        truth=dict(stage=truth.stage, seed_length=truth.truth["seed_length"],
                   testa_adheres_to=truth.truth["testa_adheres_to"]),
        variables=[asdict(v) for v in node.variables],
        answers=[asdict(a) for a in node.answers],
        imagined=[asdict(i) for i in node.imagined],
        monitoring=[asdict(m) for m in node.monitoring],
        history=[{k: v for k, v in h.items() if k in ("kind", "time_h", "source", "ess_after")}
                 for h in node.belief.history],
        pieces=sorted(b.name.name for b in world.bodies if b.name.name.startswith("avocado") and
                      b.name.name.endswith(("_skin", "_seed_coat"))))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", "pr2_run.json")
    json.dump(out, open(path, "w"), indent=1, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
