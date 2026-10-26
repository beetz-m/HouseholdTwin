"""Update an egg's probabilistic SDT node with observations (food/egg_belief.py) and rewrite its belief layer.

    ~/Documents/LabAutomationTwin/env_cad/bin/python scripts/egg_observe.py assets/food/egg_3 \
        [--storage fridge|counter] [--bbd-days-left D] [--wait HOURS --kept fridge|counter] [--mass G] [--length MM]
        [--air-cell MM] [--crack-seen yes|no] [--float-tilt DEG | --floats] [--grip-force N --slipped yes|no
        --cracked yes|no] [--strike V --tool rim|flat --outcome intact|check|cracked_open|shattered]
        [--albumen-height MM] [--yolk-index YI] [--peel clean|torn]

Creates the node's belief from the register prior if it has none (4000 particles). Writes sdt_belief.usda (over
whole.usdc) and belief/belief.npz, and prints the decision variables.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from food.egg_belief import EggBelief, Observation, write_belief_layer  # noqa: E402

TOOLS = {"rim": {"kind": "rim", "surface": "bowl_rim_ceramic", "edge_radius": 2e-3},
         "flat": {"kind": "flat", "surface": "granite"}}


def yes(s):
    return None if s is None else s.lower() in ("yes", "y", "true", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("node")
    ap.add_argument("--storage")
    ap.add_argument("--bbd-days-left", type=float)
    ap.add_argument("--wait", type=float)
    ap.add_argument("--kept", default="fridge")
    ap.add_argument("--mass", type=float)
    ap.add_argument("--length", type=float)
    ap.add_argument("--air-cell", type=float)
    ap.add_argument("--crack-seen")
    ap.add_argument("--float-tilt", type=float)
    ap.add_argument("--floats", action="store_true")
    ap.add_argument("--grip-force", type=float)
    ap.add_argument("--slipped")
    ap.add_argument("--cracked")
    ap.add_argument("--strike", type=float)
    ap.add_argument("--tool", default="rim")
    ap.add_argument("--outcome")
    ap.add_argument("--albumen-height", type=float)
    ap.add_argument("--yolk-index", type=float)
    ap.add_argument("--peel")
    a = ap.parse_args()
    rng = np.random.default_rng()
    path = os.path.join(a.node, "belief", "belief.npz")
    b = EggBelief.load(path) if os.path.exists(path) else EggBelief.prior()
    t = b.time_h
    obs = []
    if a.storage:
        obs.append(Observation("storage", {"where": a.storage}, t, "operator"))
    if a.bbd_days_left is not None:
        obs.append(Observation("carton_bbd", {"days_left": a.bbd_days_left}, t, "carton"))
    for o in obs:
        b.observe(o, rng)
    if a.wait:
        b.predict(a.wait, rng, storage=a.kept)
        t = b.time_h
    obs = []
    if a.mass is not None:
        obs.append(Observation("mass_g", {"value": a.mass}, t, "scale"))
    if a.length is not None:
        obs.append(Observation("length_mm", {"value": a.length}, t, "calliper"))
    if a.air_cell is not None or a.crack_seen is not None:
        obs.append(Observation("candling", {"air_cell_mm": a.air_cell, "crack_seen": yes(a.crack_seen)}, t,
                               "candling"))
    if a.floats or a.float_tilt is not None:
        obs.append(Observation("float_test", {"floats": a.floats, "tilt_deg": a.float_tilt}, t, "float test"))
    if a.grip_force is not None:
        obs.append(Observation("grip", {"F_N": a.grip_force, "slipped": yes(a.slipped), "cracked": yes(a.cracked)}, t,
                               "grip"))
    if a.strike is not None:
        obs.append(Observation("strike", {"v": a.strike, "tool": TOOLS[a.tool], "outcome": a.outcome}, t, "strike"))
    if a.albumen_height is not None or a.yolk_index is not None:
        obs.append(Observation("opened", {"albumen_height_mm": a.albumen_height, "yolk_index": a.yolk_index}, t,
                               "opened"))
    if a.peel:
        obs.append(Observation("peel", {"clean": a.peel == "clean"}, t, "peeled"))
    for o in obs:
        b.observe(o, rng)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    b.save(path)
    whole = os.path.join(a.node, "whole.usdc")
    if os.path.exists(whole):
        write_belief_layer(os.path.join(a.node, "sdt_belief.usda"), "whole.usdc", b, "belief/belief.npz")
    S = b.summary(posture=True)
    for k in ("age_days", "mass_g", "air_cell_mm", "haugh_unit", "yolk_index", "albumen_pH", "F_crack_equator_N",
              "float_tilt_deg", "p_clean_peel", "microcracked"):
        s = S[k]
        print(f"{k:20s} {s['mean']:8.3f} +- {s['sd']:6.3f}  [{s['p05']:.3f}, {s['p95']:.3f}] {s['unit']}")
    print(f"ESS {b.ess():.0f}, t = {b.time_h:.0f} h, {len(b.history)} records")


if __name__ == "__main__":
    main()
