"""A goal-directed avocado agent: it knows which effects its mission must bring about (enforce) and which it must
not cause (prevent), predicts both for every candidate plan over the fruit's lifetime by mental simulation, chooses
the plan that meets the goals with the required probability while keeping every risk under its bound, explains the
choice, executes it, checks each step's effects and repairs or replans when a goal is missed or a risk grows.

World knowledge used:
- the SDT node's belief (cram/avocado_sdt.py; particle filter food/avocado_belief.py),
- the processing mechanics (food/avocado_process.py: cut round the seed, twist apart, remove the seed, peel the seed
  coat),
- the lifetime model (food/avocado_lifetime.py: ripening under storage, ripe window and decline, spoilage, browning
  of cut flesh under storage treatments),
- geometry (where the holding hand may be while the blade sweeps the cut plane).

Effects (EFFECTS): goals -- halves apart, seed out, cavity clean, flesh intact, edible at service, fresh at service;
prevented -- bruise, torn flesh, gouge, seed coat left, hazardous knife strike, blade near the holding hand, spoiled,
browned. A mission (MISSIONS) names its servings (portion, hour) and the bounds: P(goal) >= min, P(prevented) <= max;
hard bounds (0) are safety rules. This module has no CRAM import; cram/avocado_kitchen.py executes its plans.
"""

import itertools
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from avocado_sdt import particle_avocado  # noqa: E402
from food import avocado_belief as AB  # noqa: E402
from food import avocado_process as PRC  # noqa: E402
from food import params as PR  # noqa: E402
from food.avocado_lifetime import TREATMENT, FruitClock, best_day  # noqa: E402

# %% effects, goals, guards
EFFECTS = {
    "halves_apart": ("goal", "the two halves are separated"),
    "seed_out": ("goal", "the seed is out of the fruit"),
    "cavity_clean": ("goal", "no seed-coat sheets left in the cavity"),
    "flesh_intact": ("goal", "no torn flesh, no gouges"),
    "edible_at_service": ("goal", "every serving mashes well and is not past its best (eating quality >= 0.6)"),
    "fresh_at_service": ("goal", "no serving's cut face is visibly brown"),
    "bruise": ("prevent", "a squeeze bruises the flesh"),
    "torn": ("prevent", "the twist tears the flesh (it clings to the seed)"),
    "gouge": ("prevent", "the spoon takes flesh"),
    "testa_left": ("prevent", "seed-coat sheets stay in the cavity"),
    "hazard": ("prevent", "a knife strike into the seed of a held half (pitting injuries, farley2019)"),
    "blade_near_hand": ("prevent", "the holding gripper within 10 mm of the blade's plane"),
    "spoiled": ("prevent", "the fruit spoils before it is eaten"),
    "browned": ("prevent", "a cut face is visibly brown when served"),
}


@dataclass
class Bound:
    effect: str
    p: float                 # goals: minimum probability; prevented effects: maximum probability
    hard: bool = False


@dataclass
class Mission:
    name: str
    label: str
    servings: list           # [(portion, hour after now)], portion: whole | half
    goals: list
    guards: list

    @property
    def first_service_h(self):
        return min(h for _, h in self.servings)


def _standard_bounds():
    goals = [Bound("halves_apart", 0.95), Bound("seed_out", 0.95), Bound("cavity_clean", 0.9),
             Bound("edible_at_service", 0.7), Bound("fresh_at_service", 0.8), Bound("flesh_intact", 0.3)]
    guards = [Bound("hazard", 0.0, hard=True), Bound("blade_near_hand", 0.0, hard=True), Bound("bruise", 0.02),
              Bound("spoiled", 0.05), Bound("browned", 0.2), Bound("testa_left", 0.1), Bound("torn", 0.6),
              Bound("gouge", 0.6)]
    return goals, guards


def mission(name):
    g, b = _standard_bounds()
    return {"guacamole_now": Mission("guacamole_now", "Guacamole now", [("whole", 0.0)], g, b),
            "half_now_half_tomorrow": Mission("half_now_half_tomorrow", "Half now, half for tomorrow's breakfast",
                                              [("half", 0.0), ("half", 15.0)], g, b),
            "guacamole_in_two_days": Mission("guacamole_in_two_days", "Guacamole in two days", [("whole", 48.0)], g, b),
            }[name]


MISSIONS = ["guacamole_now", "half_now_half_tomorrow", "guacamole_in_two_days"]


# %% plans
@dataclass(frozen=True)
class PlanOption:
    prepare_at_h: float            # when to cut (hours from now)
    keep_whole_in: str             # counter | fridge (until it is cut)
    removal: str                   # spoon | knife_twist
    peel_coat: bool                # lift seed-coat sheets left in the cavity
    kept_treatment: str            # how a half kept for later is stored (TREATMENT key), 'none' if nothing is kept

    @property
    def label(self):
        t = (f"cut now" if self.prepare_at_h == 0 else
             f"keep it whole in the {self.keep_whole_in} for {self.prepare_at_h:.0f} h, then cut")
        return (f"{t}; seed out with the {self.removal.replace('_', ' ')}"
                + ("; peel off seed-coat sheets" if self.peel_coat else "")
                + (f"; keep the other half: {self.kept_treatment}" if self.kept_treatment != "none" else ""))


@dataclass
class OptionResult:
    """What the robot predicts for one plan (shown in the query panel)."""
    plan: str
    feasible: bool
    utility: float
    violations: str
    p_goals: str
    p_risks: str
    option: PlanOption = None

    @property
    def name(self):
        return self.plan


@dataclass
class EffectAudit:
    """After execution: was a goal effect achieved / a prevented effect avoided, against what was predicted."""
    effect: str
    kind: str
    predicted: str
    observed: str
    status: str               # achieved | missed | prevented | occurred

    @property
    def name(self):
        return self.effect


@dataclass
class Decision:
    """The agent's choice and why."""
    mission: str
    chosen: str
    why: str
    risks: str
    lifetime: str

    @property
    def name(self):
        return self.mission


# %% the agent
@dataclass
class AvocadoAgent:
    node: object                         # AvocadoSDT
    mission: Mission
    n: int = 80
    seed: int = 0
    results: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    audits: list = field(default_factory=list)
    hold_choice: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- plan space
    def options(self):
        t0 = self.mission.first_service_h
        prep = [(0.0, "counter")] + ([(t0, "counter"), (t0, "fridge")] if t0 > 0 else [])
        keeps_half = any(p == "half" for p, h in self.mission.servings if h > t0) or len(self.mission.servings) > 1
        treatments = list(TREATMENT) if keeps_half else ["none"]
        return [PlanOption(t, store, rem, peel, tr) for (t, store), rem, peel, tr in
                itertools.product(prep, ["spoon", "knife_twist"], [False, True], treatments)]

    # ---------------------------------------------------------------- mental simulation
    def _mechanics(self, idx, prep_h, store, method, rng):
        """Halving and pitting of the imagined avocados idx at prep_h, kept whole in ``store`` until then; with the
        repairs the agent would make (re-cut a gap, re-cut a stuck fruit once)."""
        rows = []
        for i in idx:
            p = {k: float(self.node.belief.particles[k][i]) for k in AB.RANDOM}
            clock = FruitClock.of(p).advance(prep_h, store)
            avo = particle_avocado(self.node.belief, i)
            avo.p["ripeness_stage"] = clock.stage
            plane = PRC.CutPlane(0.0, 360.0)
            cut = PRC.cut_round_seed(avo, plane, rng, clock.stage)
            if not cut.ok:                                   # repair: complete the cut
                cut = PRC.cut_round_seed(avo, plane, rng, clock.stage)
            tw = PRC.twist_apart(avo, plane, rng, clock.stage, plane.coverage_deg) if cut.ok else None
            if tw is not None and not tw.ok:                 # repair: re-cut the bridge, twist again
                tw = PRC.twist_apart(avo, plane, rng, clock.stage, 360.0)
            rm = PRC.remove_seed(avo, rng, clock.stage, method) if tw is not None and tw.ok else None
            rows.append(dict(clock=clock, cut=cut, tw=tw, rm=rm))
        return rows

    def evaluate_all(self):
        rng = np.random.default_rng(self.seed)
        w = self.node.belief.weights
        idx = rng.choice(self.node.belief.n, self.n, p=w / w.sum())
        cache = {}
        self.results = []
        for opt in self.options():
            key = (opt.prepare_at_h, opt.keep_whole_in, opt.removal)
            if key not in cache:
                cache[key] = self._mechanics(idx, *key, rng)
            self.results.append(self._effects(opt, cache[key], np.random.default_rng(self.seed + 1)))
        self.results.sort(key=lambda r: (not r.feasible, -r.utility))
        return self.results

    def _effects(self, opt, rows, rng):
        E = {k: [] for k in EFFECTS}
        U = []
        hold_clear = self.hold_choice.get("clear", True)
        for r in rows:
            tw, rm = r["tw"], r["rm"]
            apart = tw is not None and tw.ok
            out = rm is not None and rm.ok
            torn = apart and tw.outcome == "halved_torn"
            gouge = out and bool(rm.truth.get("gouges"))
            rem = rm.truth.get("testa_remnant_fraction", 0.0) if out else 0.0
            if opt.peel_coat and rem > 0.05:
                pc = PRC.peel_seed_coat(None, rng, rem, passes=2)
                rem = pc.truth["testa_remnant_fraction"]
                gouge = gouge or bool(pc.truth["gouges"])
            testa = out and rem > 0.05
            hazard = out and opt.removal == "knife_twist"
            # servings: the half served at the cut is fresh; a kept half browns with its treatment
            ok_eat, fresh_all, spoiled_any, u = True, True, False, []
            whole = r["clock"]
            for portion, at_h in self.mission.servings:
                if at_h < opt.prepare_at_h:              # served before it is cut: impossible plan
                    ok_eat, fresh_all = False, False
                    u.append(0.0)
                    continue
                c = FruitClock(whole.stage, whole.rate_per_day, whole.hours_ripe, whole.ripe_window_h,
                               whole.spoil_days, cut=True)
                wait = at_h - opt.prepare_at_h
                if wait > 0:
                    c = c.advance(wait, "fridge" if "fridge" in opt.kept_treatment else "counter",
                                  opt.kept_treatment)
                ok_eat &= c.eating_quality >= 0.6
                fresh_all &= not c.browned
                spoiled_any |= c.spoiled or whole.spoiled
                u.append(c.eating_quality * c.freshness *
                         (1.0 - 0.35 * torn - 0.3 * gouge - 0.25 * testa) * (1.0 if out else 0.0))
            vals = dict(halves_apart=apart, seed_out=out, cavity_clean=out and not testa,
                        flesh_intact=apart and not torn and not gouge, edible_at_service=ok_eat,
                        fresh_at_service=fresh_all, bruise=False, torn=torn, gouge=gouge, testa_left=testa,
                        hazard=hazard, blade_near_hand=not hold_clear, spoiled=spoiled_any, browned=not fresh_all)
            for k, v in vals.items():
                E[k].append(bool(v))
            U.append(float(np.mean(u)))
        P = {k: float(np.mean(v)) for k, v in E.items()}
        viol = []
        for b in self.mission.goals:
            if P[b.effect] < b.p:
                viol.append(f"{b.effect} {P[b.effect]:.2f} < {b.p:.2f}")
        for b in self.mission.guards:
            if P[b.effect] > b.p:
                viol.append(f"{b.effect} {P[b.effect]:.2f} > {b.p:.2f}" + (" (safety rule)" if b.hard else ""))
        hard = any(P[b.effect] > b.p for b in self.mission.guards if b.hard)
        res = OptionResult(opt.label, not viol, float(np.mean(U)), "; ".join(viol) or "none",
                           ", ".join(f"{b.effect} {P[b.effect]:.2f}" for b in self.mission.goals),
                           ", ".join(f"{b.effect} {P[b.effect]:.2f}" for b in self.mission.guards), opt)
        res._P, res._hard = P, hard
        return res

    # ---------------------------------------------------------------- choice and explanation
    def decide(self):
        R = self.evaluate_all()
        feas = [r for r in R if r.feasible]
        if feas:
            best = max(feas, key=lambda r: r.utility)
            why = (f"meets every goal and stays within every risk bound; expected quality {best.utility:.2f} "
                   f"(best of {len(feas)} feasible plans out of {len(R)})")
        else:                                             # no plan meets everything: no safety rule may be broken,
            safe = [r for r in R if not r._hard]           # then the fewest and smallest shortfalls, then quality
            if safe:
                best = min(safe, key=lambda r: (r.violations.count(";"), -r.utility))
                why = (f"no plan meets every goal and risk bound; this one breaks no safety rule and misses the "
                       f"fewest ({best.violations}); expected quality {best.utility:.2f}")
            else:
                best = min(R, key=lambda r: (r.violations.count(";"), -r.utility))
                why = (f"every plan breaks a safety rule ({best.violations}): I will not execute any of them")
        rej = [r for r in R if r is not best and r._hard][:1]
        if rej:
            why += f". Rejected e.g. '{rej[0].plan}': {rej[0].violations}"
        risks = ", ".join(f"{e} {best._P[e]:.2f}" for e, (kind, _) in EFFECTS.items() if kind == "prevent")
        dec = Decision(self.mission.label, best.plan, why, risks, self.lifetime_summary())
        self.decisions.append(dec)
        self.chosen = best
        return best, dec

    def lifetime_summary(self, horizon_days=12, step_h=6.0):
        """Lifetime analysis of the whole fruit under the belief, for the counter and the fridge: the edible window
        (eating quality >= 0.6; median start and end over the belief) and when spoilage becomes likely (P > 5 %)."""
        P = self.node.belief.particles
        w = self.node.belief.weights
        rng = np.random.default_rng(self.seed + 2)
        idx = rng.choice(self.node.belief.n, 200, p=w / w.sum())
        hs = np.arange(0.0, horizon_days * 24 + 1e-9, step_h)
        txt = []
        for store in ("counter", "fridge"):
            starts, ends, sp = [], [], []
            for i in idx:
                c = FruitClock.of({k: float(P[k][i]) for k in AB.RANDOM})
                q = np.array([c.advance(h, store).eating_quality for h in hs])
                s_ = np.array([c.advance(h, store).spoiled for h in hs])
                good = np.nonzero(q >= 0.6)[0]
                starts.append(hs[good[0]] if len(good) else np.inf)
                ends.append(hs[good[-1]] if len(good) else -np.inf)
                sp.append(s_)
            sp = np.array(sp).mean(0)
            st, en = np.median(starts), np.median(ends)
            first_sp = hs[np.argmax(sp > 0.05)] if (sp > 0.05).any() else None
            win = (f"edible from {st / 24:.1f} d to {en / 24:.1f} d" if np.isfinite(st) and np.isfinite(en)
                   else "not edible within the horizon")
            txt.append(f"{store}: {win}" + (f", P(spoiled) > 5 % from {first_sp / 24:.1f} d" if first_sp is not None
                                             else f", no spoilage risk within {horizon_days} d"))
        return "; ".join(txt)

    # ---------------------------------------------------------------- goal watch during execution
    def reassess(self, elapsed_h, state):
        """Re-predict the goals still open from the current belief and state (after a step). ``state``: integrity,
        seed, remnant (testa), kept_treatment, cut_at_h (elapsed hours at the cut, or None). Returns (goal, P, bound,
        reason) for the goals now below their bound."""
        P = self.node.belief.particles
        w = self.node.belief.weights / self.node.belief.weights.sum()
        risks = []
        remaining = [(p_, h) for p_, h in self.mission.servings if h >= elapsed_h - 1e-9]
        if remaining:
            ok = np.ones(self.node.belief.n, bool)
            for portion, at_h in remaining:
                for i in range(self.node.belief.n):
                    if not ok[i]:
                        continue
                    c = FruitClock.of({k: float(P[k][i]) for k in AB.RANDOM}).advance(max(0.0, at_h - elapsed_h))
                    ok[i] = c.eating_quality >= 0.6
            p_ok = float(np.sum(w * ok))
            b = next(b for b in self.mission.goals if b.effect == "edible_at_service")
            if p_ok < b.p:
                risks.append(("edible_at_service", p_ok, b.p,
                              "the evidence says the fruit is less ripe than planned "
                              f"(ripeness {np.sum(w * P['ripeness_stage']):.2f}); a cut avocado does not catch up"))
        if state.get("integrity") == "halved_torn":
            risks.append(("flesh_intact", 0.0, next(b.p for b in self.mission.goals if b.effect == "flesh_intact"),
                          "the twist tore the flesh"))
        if state.get("seed") == "out" and state.get("remnant", 0) > 0.05:
            risks.append(("cavity_clean", 0.0, 0.9, "seed-coat sheets seen in the cavity: peel them off"))
        return risks

    # ---------------------------------------------------------------- execution-time effect audit
    def audit(self, observed):
        """Compare the observed effects with what the chosen plan predicted."""
        P = self.chosen._P
        for e, (kind, _) in EFFECTS.items():
            if e not in observed:
                continue
            o = bool(observed[e])
            status = (("achieved" if o else "missed") if kind == "goal" else ("occurred" if o else "prevented"))
            self.audits.append(EffectAudit(e, kind, f"P = {P[e]:.2f}", "yes" if o else "no", status))
        return self.audits


# %% geometry: where may the holding hand be while the blade sweeps the cut plane?
BLADE_HALF_THICKNESS = 0.006      # big-knife.stl, held flat
PAD_HALF_HEIGHT = 0.0075          # PR2 finger pad, along the approach axis
CLEARANCE_MIN = 0.010


def fruit_half_height(g, y):
    """Height of the fruit's upper surface above the cut plane at long-axis position y (fruit frame, from the belief
    means: an egg-shaped profile narrowing towards the neck at +y)."""
    L, D = g.length_m, g.diameter_m
    s = np.clip(2 * y / L, -1, 1)
    r = D / 2 * math.sqrt(max(0.0, 1 - s ** 2)) * (1 - 0.35 * max(0.0, s) ** 2)
    return r


def plan_hold(g):
    """Choose how the left hand holds the fruit during the cut. Pinch: the finger pads grip the flank from above,
    their lower edge at least CLEARANCE_MIN above the blade (needs a fruit tall enough there). Press: closed fingers
    press on the top (holds less firmly, far from the blade). A safe pinch is preferred, then a safe press; returns
    the candidates (with clearance) and the choice (None when no hold is safe)."""
    cands = []
    for name, frac in (("neck", 0.36), ("shoulder", 0.18), ("middle", 0.0), ("broad end", -0.2)):
        y = frac * g.length_m
        top = fruit_half_height(g, y)
        pad_low = BLADE_HALF_THICKNESS + CLEARANCE_MIN            # lowest admissible pad bottom above the plane
        tcp_h = pad_low + PAD_HALF_HEIGHT                          # tool frame height above the plane
        grip = tcp_h + PAD_HALF_HEIGHT <= top - 0.002               # the pads still touch the flank
        width = 2 * math.sqrt(max(0.0, top ** 2 - tcp_h ** 2)) if top > tcp_h else 0.0
        cands.append(dict(name=f"pinch at the {name}", kind="pinch", y=y, tcp_height=tcp_h, grip=grip, width=width,
                          top=top, clearance_mm=1e3 * (tcp_h - PAD_HALF_HEIGHT - BLADE_HALF_THICKNESS)))
        press_h = top + PAD_HALF_HEIGHT - 0.001                     # closed pads resting on the top
        cands.append(dict(name=f"press on the {name}", kind="press", y=y, tcp_height=press_h, grip=top > 0.02,
                          width=0.0, top=top, clearance_mm=1e3 * (press_h - PAD_HALF_HEIGHT - BLADE_HALF_THICKNESS)))
    ok = [c for c in cands if c["grip"] and c["clearance_mm"] >= 1e3 * CLEARANCE_MIN - 1e-6]
    pinch = [c for c in ok if c["kind"] == "pinch"]
    press = [c for c in ok if c["kind"] == "press"]
    choice = (max(pinch, key=lambda c: c["width"]) if pinch else
              max(press, key=lambda c: c["top"]) if press else None)
    return cands, choice


if __name__ == "__main__":                                      # offline check (no CRAM)
    import time
    import avocado_sdt as SDT
    node = SDT.AvocadoSDT.on_detection("avocado_1")
    obs, _ = SDT.photo_observations(os.path.join(HERE, "camera", "avocado_4_counter.png"), 0.0)
    node.perceive(obs)
    from avocado_motions import FruitGeometry
    S = node.belief.summary()
    g = FruitGeometry(np.zeros(3), S["length_L"]["mean"] * 1e-3, S["diameter_D"]["mean"] * 1e-3,
                      S["seed_length"]["mean"] * 1e-3, S["seed_width_ratio"]["mean"] * S["seed_length"]["mean"] * 1e-3)
    cands, choice = plan_hold(g)
    for c in cands:
        print(f"  hold {c['name']:9s} top {1e3 * c['top']:.1f} mm, pads {1e3 * c['tcp_height']:.1f} mm, clearance "
              f"{c['clearance_mm']:.1f} mm, grip {c['grip']}")
    print("  hold choice:", choice and choice["name"])
    for m in MISSIONS:
        ag = AvocadoAgent(node, mission(m), hold_choice=dict(clear=choice is not None))
        t = time.time()
        best, dec = ag.decide()
        print(f"\n[{m}] {time.time() - t:.0f} s; chosen: {dec.chosen}\n   why: {dec.why}\n   risks: {dec.risks}\n   "
              f"lifetime: {dec.lifetime}")
        for r in ag.results[:4]:
            print(f"   {'OK ' if r.feasible else '-- '} U {r.utility:.2f}  {r.plan[:90]} | {r.violations[:120]}")
