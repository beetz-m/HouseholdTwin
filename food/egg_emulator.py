"""Emulator for handling, cracking, opening and boiling an egg: actions in, new USD states + measurements + events out.

State machine (food/schema.py EGG):
  intact --strike--> intact | microcracked (invisible) | cracked: check (membrane intact) | cracked_open | shattered
  cracked --open_up--> opened: contents in the bowl (yolk whole / broken, shell shards that fell in)
  opened --fish_out--> fewer shards;     intact --boil--> boiled (doneness) --peel--> peeled clean | torn
  any --wait--> older (food/egg_lifetime.py at the storage temperature)
Non-destructive probes: read_carton, weigh, candle, float_test (wets the shell), grip (may slip or crack).
The mechanics are the twin's models (food/dynamics.EggTruth, food/egg_crack.py, food/egg_cook.py, food/egg_lifetime.py)
on the hidden instance (register draw, aged at its storage temperature); the robot sees only `measured` (noise from the
register's sensor_* entries). Opening writes the contents as a USD state of the part graph
(scripts/build_egg_open_states.py with the clock's Haugh unit and yolk index, the shards in the bowl and a broken yolk
as labelled prims), so the bowl monitor has segmentation ground truth.

    emu = EggEmulator(3, "assets/food/egg_3/process/run0", seed=0)
    out = emu.strike(0.22, {"kind": "rim", "surface": "bowl_rim_ceramic", "edge_radius": 2e-3})
    out.measured, out.truth
"""

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field

import numpy as np

from . import egg_crack as EC
from . import egg_lifetime as EL
from . import params as PR
from .dynamics import EggTruth

E = PR.EGG
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


@dataclass
class EggOutcome:
    action: str
    ok: bool
    outcome: str
    measured: dict = field(default_factory=dict)
    truth: dict = field(default_factory=dict)
    events: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


@dataclass
class EmulatorState:
    shell: str = "intact"                 # intact / microcracked / cracked / shattered / opened
    membrane: str = "intact"
    contents: str = "in_shell"            # in_shell / released / spilled
    yolk: str = "whole"
    shards_in_bowl: int = 0
    wet: bool = False
    boiled: str = ""                      # '' / runny / jammy / creamy / firm
    peeled: str = ""                      # '' / clean / torn
    pieces: dict = field(default_factory=dict)
    time_h: float = 0.0
    log: list = field(default_factory=list)


class PreconditionError(Exception):
    pass


def _sd(k):
    return float(E[k][1])


class EggEmulator:
    def __init__(self, instance_seed, out_dir, seed=0, storage="fridge", age_days=None, write_usd=True):
        from .egg import Egg
        self.iseed = int(instance_seed)
        self.out = os.path.abspath(out_dir)
        os.makedirs(self.out, exist_ok=True)
        self.rng = np.random.default_rng(seed)
        self.p = PR.sample("egg", self.iseed)
        self.p["storage_temp_C"] = 6.0 if storage == "fridge" else 22.0
        if age_days is not None:
            self.p["age_days"] = float(age_days)
        self.egg = Egg(self.iseed)
        self.clock = EL.EggClock.of(self.p, lay=False)
        self.truth = EggTruth(self.egg, self.mass_g() * 1e-3)
        self.storage = storage
        self.write_usd = write_usd
        self.s = EmulatorState(pieces={"egg": os.path.join(ROOT, "assets", "food", f"egg_{self.iseed}", "whole.usdc")})

    # ---------------------------------------------------------------- truth helpers
    def mass_g(self):
        return self.clock.summary()["mass_g"]

    def fresh(self):
        return self.clock.summary()

    def _log(self, out):
        self.s.log.append(dict(t_h=self.s.time_h, **out.as_dict()))
        return out

    def _need(self, cond, why):
        if not cond:
            raise PreconditionError(why)

    # ---------------------------------------------------------------- time
    def wait(self, hours, storage=None):
        storage = storage or self.storage
        self.clock.advance(hours, storage)
        self.storage = storage
        self.s.time_h += hours
        self.s.wet = False
        self.truth.m = self.mass_g() * 1e-3
        f = self.fresh()
        return self._log(EggOutcome("wait", True, f"{hours:.0f} h in the {storage}", {},
                                    {k: f[k] for k in ("age_d", "air_cell_mm", "haugh_unit", "albumen_pH")}))

    # ---------------------------------------------------------------- non-destructive probes
    def read_carton(self):
        f = self.fresh()
        days_left = int(np.floor(E["best_before_days"][1] - f["age_d"]))
        return self._log(EggOutcome("read_carton", True, f"best before in {days_left} d",
                                    {"days_left": days_left}, {"age_d": f["age_d"]}))

    def weigh(self):
        self._need(self.s.shell in ("intact", "microcracked"), "only a whole egg is weighed")
        m = self.mass_g()
        return self._log(EggOutcome("weigh", True, "weighed", {"mass_g": m + self.rng.normal(0, _sd("sensor_scale_sd_g"))},
                                    {"mass_g": m}))

    def candle(self):
        self._need(self.s.shell in ("intact", "microcracked"), "candling needs a whole egg")
        f = self.fresh()
        micro = self.s.shell == "microcracked"
        seen = self.rng.random() < (_sd("sensor_candling_crack_detect") if micro else 0.02)
        return self._log(EggOutcome("candle", True, "crack seen" if seen else "no crack seen",
                                    {"air_cell_mm": f["air_cell_mm"] + self.rng.normal(0, _sd("sensor_candling_air_sd_mm")),
                                     "crack_seen": bool(seen)}, {"air_cell_mm": f["air_cell_mm"], "microcracked": micro}))

    def float_test(self):
        self._need(self.s.shell == "intact", "a cracked egg takes in water")   # micro-cracks: not visible, allowed
        f = self.fresh()
        self.s.wet = True
        if f["SG"] < 1.0:
            meas = {"floats": True}
        else:
            meas = {"floats": False, "tilt_deg": f["float_tilt_deg"] + self.rng.normal(0, _sd("sensor_float_tilt_sd_deg"))}
        return self._log(EggOutcome("float_test", True, f["float"], meas,
                                    {"tilt_deg": f["float_tilt_deg"], "SG": f["SG"], "class": f["float"]}))

    def grip(self, F_N, polar=np.pi / 2, a_max=2.0):
        """Hold at F per pad and lift with a_max. A wet shell grips worse (friction x 0.7, ILLUSTRATIVE)."""
        self._need(self.s.shell in ("intact", "microcracked", "cracked"), "nothing to grip")
        mu = self.p["friction_shell_rubber"] * (0.7 if self.s.wet else 1.0)
        F_slip = self.truth.m * (9.81 + a_max) / (2 * mu)
        before = self.truth.shell
        self.truth.static_load(self.s.time_h * 3600, F_N, polar, self.rng)
        self.s.shell = self.truth.shell
        cracked = self.truth.shell in ("cracked", "shattered") and before not in ("cracked", "shattered")
        slipped = F_N < F_slip
        meas = {"F_N": F_N * (1 + self.rng.normal(0, _sd("sensor_force_rel_sd"))), "slipped": bool(slipped),
                "cracked": bool(cracked)}
        if self.truth.shell == "shattered":
            self.s.contents = "spilled"
        out = "crushed" if self.truth.shell == "shattered" else "cracked" if cracked else "slipped" if slipped else "held"
        return self._log(EggOutcome("grip", out == "held", out, meas,
                                    {"F_slip": F_slip, "F_crack": self.truth.crack_force(polar),
                                     "shell": self.truth.shell}))

    # ---------------------------------------------------------------- transforming actions
    def strike(self, v, tool, polar=np.pi / 2):
        self._need(self.s.shell in ("intact", "microcracked", "cracked"), "the egg is already open")
        r = EC.strike(self.truth, v, tool, self.rng, polar=polar)
        self.s.shell = self.truth.shell
        self.s.membrane = getattr(self.truth, "membrane", self.s.membrane)
        seen = {"intact": "intact", "microcracked": "intact", "check": "check", "cracked_open": "cracked_open",
                "shattered": "shattered"}[r.state]
        if r.state == "shattered":
            self.s.contents = "spilled"
        return self._log(EggOutcome("strike", seen in ("check", "cracked_open"), seen,
                                    {"outcome": seen, "v": v, "tool": tool.get("kind")},
                                    {"state": r.state, "ratio": r.ratio, "F_peak": r.F_peak, "F_crack": r.F_crack,
                                     "shards_inside": r.shards_inside}))

    def open_up(self, crack_up=True, drop_height=0.03, name="opened"):
        self._need(self.s.shell == "cracked", "open_up needs a cracked shell")
        f = self.fresh()
        ok, yolk, shards = EC.open_up(self.truth, self.rng, crack_up=crack_up, drop_height=drop_height,
                                      age_days=f["age_d"])
        self.s.shell, self.s.contents, self.s.yolk = self.truth.shell, self.truth.contents, yolk
        self.s.shards_in_bowl += shards
        h = float(self.clock.state.albumen_height_mm[0])
        meas = {}
        if ok:
            meas = {"albumen_height_mm": h + self.rng.normal(0, _sd("sensor_albumen_height_sd_mm")),
                    "yolk_index": (f["yolk_index"] if yolk == "whole" else 0.12) +
                    self.rng.normal(0, _sd("sensor_yolk_index_sd")), "yolk_broken": yolk != "whole"}
            if self.write_usd:
                self.s.pieces[name] = self._write_contents(name, f, yolk != "whole", self.s.shards_in_bowl)
        return self._log(EggOutcome("open_up", ok and yolk == "whole", "released" if ok else "spilled", meas,
                                    {"yolk": yolk, "shards_in_bowl": self.s.shards_in_bowl,
                                     "haugh_unit": f["haugh_unit"], "yolk_index": f["yolk_index"]}))

    def fish_out(self, attempts=3):
        self._need(self.s.contents == "released", "nothing in the bowl")
        before = self.s.shards_in_bowl
        self.s.shards_in_bowl = EC.fish_out(before, self.rng, attempts)
        if self.write_usd and before != self.s.shards_in_bowl:
            f = self.fresh()
            self.s.pieces["opened_fished"] = self._write_contents("opened_fished", f, self.s.yolk != "whole",
                                                                  self.s.shards_in_bowl)
        return self._log(EggOutcome("fish_out", self.s.shards_in_bowl == 0, f"{self.s.shards_in_bowl} left",
                                    {}, {"before": before, "after": self.s.shards_in_bowl}))

    def boil(self, minutes, egg_temp_C=6.0, start="hot"):
        self._need(self.s.shell in ("intact", "microcracked"), "boil a whole egg")
        from .egg_cook import EggHeat, cold_start, hot_start
        H = EggHeat(self.egg)
        _, d = H.run(egg_temp_C, hot_start(98.0) if start == "hot" else cold_start(), minutes * 60)
        self.s.boiled = d["yolk"]
        leak = self.s.shell == "microcracked"
        return self._log(EggOutcome("boil", True, d["yolk"], {"white_in_water": bool(leak)},
                                    {"yolk": d["yolk"], "T_centre_max": d.get("T_centre_max")}))

    def peel(self):
        self._need(bool(self.s.boiled), "peel a boiled egg")
        p = float(EL.p_clean_peel(self.clock.state.albumen_pH)[0])
        clean = self.rng.random() < p
        self.s.peeled = "clean" if clean else "torn"
        return self._log(EggOutcome("peel", clean, self.s.peeled, {"clean": bool(clean)},
                                    {"p_clean": p, "albumen_pH": float(self.clock.state.albumen_pH[0])}))

    # ---------------------------------------------------------------- USD states
    def _write_contents(self, name, f, broken, shards):
        d = os.path.join(self.out, name)
        args = [PY, os.path.join(ROOT, "scripts", "build_egg_open_states.py"), str(self.iseed), "--out", d,
                "--hu", f"{f['haugh_unit']:.3f}", "--yi", f"{f['yolk_index']:.4f}", "--shards", str(shards)]
        if broken:
            args.append("--yolk-broken")
        subprocess.run(args, check=True, capture_output=True, cwd=ROOT)
        tex = os.path.join(ROOT, "assets", "food", f"egg_{self.iseed}", "textures")
        if os.path.isdir(tex) and not os.path.exists(os.path.join(d, "textures")):
            os.symlink(tex, os.path.join(d, "textures"))
        return os.path.join(d, "opened_contents.usdc")

    def save_log(self):
        json.dump(dict(instance=self.iseed, storage=self.storage, log=self.s.log, pieces=self.s.pieces,
                       truth_params={k: self.p[k] for k in ("age_days", "F_break_equatorial", "shell_thickness",
                                                            "ageing_rate_factor")}),
                  open(os.path.join(self.out, "log.json"), "w"), indent=1, default=float)
