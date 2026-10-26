"""Synthetic test set for the egg bowl monitor (food/egg_bowl_monitor.py): opened-egg states of known freshness with
shell shards and broken yolks, as USD states of the part graph (scripts/build_egg_open_states.py; shards labelled
egg_shell_shard, a broken yolk egg_yolk_broken), plus their ground truth.

Samples per instance: fresh (day 2, fridge) / middle (day 14, fridge) / old (day 28, counter) x {clean, 1 shard,
3 shards} + a broken yolk (middle) and a broken yolk with 2 shards (old) -> 11 samples.

    ~/Documents/LabAutomationTwin/env_cad/bin/python scripts/build_egg_bowl_set.py [seed]
Writes assets/food/egg_<s>/bowl_set/<sample>/opened_contents.usdc and bowl_set/set.json.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from food import egg_lifetime as EL  # noqa: E402
from food import params as PR  # noqa: E402


def main(seed=3):
    base = os.path.join(ROOT, "assets", "food", f"egg_{seed}", "bowl_set")
    os.makedirs(base, exist_ok=True)
    p = PR.sample("egg", seed)
    fresh = {}
    for tag, days, storage in (("fresh", 2, "fridge"), ("middle", 14, "fridge"), ("old", 28, "counter")):
        q = dict(p, age_days=0.0, storage_temp_C=6.0)
        c = EL.EggClock.of(q)
        c.advance(days * 24.0, storage)
        fresh[tag] = c.summary()
    samples = []
    for tag in ("fresh", "middle", "old"):
        for shards in (0, 1, 3):
            samples.append((f"{tag}_s{shards}", tag, shards, False))
    samples += [("middle_broken", "middle", 0, True), ("old_broken_s2", "old", 2, True)]
    rec = []
    for name, tag, shards, broken in samples:
        f = fresh[tag]
        d = os.path.join(base, name)
        args = [sys.executable, os.path.join(ROOT, "scripts", "build_egg_open_states.py"), str(seed), "--out", d,
                "--hu", f"{f['haugh_unit']:.3f}", "--yi", f"{f['yolk_index']:.4f}", "--shards", str(shards)]
        if broken:
            args.append("--yolk-broken")
        subprocess.run(args, check=True, capture_output=True, cwd=ROOT)
        tex = os.path.join(ROOT, "assets", "food", f"egg_{seed}", "textures")
        if os.path.isdir(tex) and not os.path.exists(os.path.join(d, "textures")):
            os.symlink(tex, os.path.join(d, "textures"))
        meta = json.load(open(os.path.join(d, "open_states.json")))
        rec.append(dict(sample=name, usd=os.path.join(name, "opened_contents.usdc"), freshness=tag,
                        shards=shards, yolk_broken=broken, haugh_unit=f["haugh_unit"],
                        yolk_index=(meta["contents"]["yolk_height_mm"] / meta["contents"]["yolk_diameter_mm"]) if broken
                        else f["yolk_index"], mass_g=f["mass_g"],
                        thick_height_mm=meta["contents"]["thick_height_mm"],
                        yolk_height_mm=meta["contents"]["yolk_height_mm"],
                        yolk_diameter_mm=meta["contents"]["yolk_diameter_mm"]))
        print("[bowl set]", name, f"HU {f['haugh_unit']:.1f} YI {rec[-1]['yolk_index']:.3f}", flush=True)
    json.dump(dict(seed=seed, samples=rec), open(os.path.join(base, "set.json"), "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
