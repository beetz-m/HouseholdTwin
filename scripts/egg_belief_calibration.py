"""Calibration of the egg belief over many eggs: share of steps (and of final steps) where the hidden truth lies in
the belief's 90 % band, per variable, from the test_reports/egg_belief_<seed>.json files of scripts/egg_belief_demo.py.

    for s in $(seq 1000 1023); do .../python scripts/egg_belief_demo.py $s; done
    .../python scripts/egg_belief_calibration.py 1000 1023      -> test_reports/egg_belief_calibration.json
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
a, b = int(sys.argv[1]), int(sys.argv[2])
allv, final, n_files, strikes = {}, {}, 0, []
for s in range(a, b + 1):
    f = os.path.join(ROOT, "test_reports", f"egg_belief_{s}.json")
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    n_files += 1
    strikes.append((d["strike"]["outcome"], d["strike"]["predicted"]))
    for r in d["rows"][1:]:                              # the prior row excluded
        for k, v in r.items():
            if isinstance(v, dict) and "inside" in v:
                allv.setdefault(k, []).append(v["inside"])
    for k, v in d["rows"][-1].items():
        if isinstance(v, dict) and "inside" in v:
            final.setdefault(k, []).append(v["inside"])
out = dict(eggs=n_files, all_steps={k: round(sum(v) / len(v), 3) for k, v in allv.items()},
           final_step={k: round(sum(v) / len(v), 3) for k, v in final.items()},
           strike_outcomes={o: sum(1 for x, _ in strikes if x == o) for o in ("intact", "check", "cracked_open",
                                                                             "shattered")},
           strike_p_open_mean=round(sum(p["cracked_open"] for _, p in strikes) / max(len(strikes), 1), 3))
json.dump(out, open(os.path.join(ROOT, "test_reports", "egg_belief_calibration.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
