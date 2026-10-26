"""Bayesian filtering on an avocado SDT node: load its particle belief, let time pass (ripening prediction), apply
one or more observations, and rewrite the node's probabilistic belief layer.

Run: ~/Documents/LabAutomationTwin/env_cad/bin/python scripts/avocado_observe.py assets/food/<node>
         [--wait HOURS] [--photo IMG] [--mass G] [--length MM] [--squeeze K_N_PER_MM [--force N]]
         [--seed-length MM] [--bruise-seen yes|no] [--init]
--init starts from the register prior. The node keeps belief/belief.npz (particles) and sdt_belief.usda (layer over
sdt.usda with every variable's mean, sd, quantiles, histogram and support, and the observation history).
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from food import avocado_belief as AB  # noqa: E402


def photo_observations(img, t):
    from food import avocado_vision as AV
    obs, _ = AV.estimate(img)
    E = obs.estimates
    src = f"photo {os.path.basename(img)} ({obs.image_sha1})"
    return [AB.Observation("photo_shape", dict(aspect=float(E["aspect_L_over_D"].value), aspect_sd=0.05,
                                               neck=float(E["neck"].value), neck_sd=0.05), t, src),
            AB.Observation("skin_colour", dict(L=float(E["skin_Lab_L"].value), C=float(E["skin_chroma"].value),
                                               h=float(E["skin_hue"].value)), t, src),
            AB.Observation("stem_button", dict(present=E["stem_button"].value == "present"), t, src)], obs


def write_node(node, b):
    os.makedirs(os.path.join(node, "belief"), exist_ok=True)
    b.save(os.path.join(node, "belief", "belief.npz"))
    base = "./sdt.usda" if os.path.exists(os.path.join(node, "sdt.usda")) else "./whole.usdc"
    AB.write_belief_layer(os.path.join(node, "sdt_belief.usda"), base, b, "./belief/belief.npz")


def report(b):
    S = b.summary()
    for k in ("ripeness_stage", "flesh_E_MPa", "mass_g", "length_L", "diameter_D", "seed_length", "bruised"):
        s = S[k]
        print(f"   {k:16s} {s['mean']:8.2f} +- {s['sd']:6.2f}  [{s['p05']:.2f} .. {s['p95']:.2f}] {s['unit']}")
    print(f"   t = {b.time_h:.0f} h, ESS {b.ess():.0f} of {b.n}, {len(b.history)} steps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("node")
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--wait", type=float, default=0.0)
    ap.add_argument("--photo")
    ap.add_argument("--mass", type=float)
    ap.add_argument("--length", type=float)
    ap.add_argument("--squeeze", type=float)
    ap.add_argument("--force", type=float, default=4.0)
    ap.add_argument("--seed-length", type=float)
    ap.add_argument("--bruise-seen", choices=["yes", "no"])
    a = ap.parse_args()
    rng = np.random.default_rng()
    path = os.path.join(a.node, "belief", "belief.npz")
    b = AB.AvocadoBelief.prior(4000) if a.init or not os.path.exists(path) else AB.AvocadoBelief.load(path)
    if a.wait:
        b.predict(a.wait, rng)
    t = b.time_h
    O = AB.Observation
    if a.photo:
        for o in photo_observations(a.photo, t)[0]:
            b.observe(o, rng)
    if a.mass is not None:
        b.observe(O("mass_g", dict(value=a.mass, sd=2.0), t, "kitchen scale"), rng)
    if a.length is not None:
        b.observe(O("length_mm", dict(value=a.length, sd=1.0), t, "calliper"), rng)
    if a.squeeze is not None:
        b.observe(O("squeeze", dict(k=a.squeeze, force_N=a.force), t, "gripper squeeze"), rng)
    if a.seed_length is not None:
        b.observe(O("seed_length_mm", dict(value=a.seed_length), t, "measured after halving"), rng)
    if a.bruise_seen:
        b.observe(O("bruise_seen", dict(seen=a.bruise_seen == "yes"), t, "flesh inspected"), rng)
    write_node(a.node, b)
    print(f"[{os.path.basename(a.node.rstrip('/'))}] belief updated:")
    report(b)


if __name__ == "__main__":
    main()
