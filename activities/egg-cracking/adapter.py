"""Cracking eggs into a batch: the adapter between ccram's kernel and the HouseholdTwin egg (an emulator twin: no CRAM
world). Entities are symbols: egg-1, egg-2, cup-1, batch-bowl, bowl-rim, counter.

Two sides, kept apart (SDT principle):
  the twin    food/egg_emulator.EggEmulator per egg -- hidden truth (register draw, age, micro-cracks, shards)
  the robot   food/egg_belief.EggBelief per egg (particles over the egg's hidden parameters) fed ONLY with measured
              outcomes; atoms() reports what was observed; servants, predictions and probes use the belief.

  servants     strike surface + speed: maximise P(crack) under the chance bound P(shatter) <= 0.05 (belief decision,
               revisable after a probe or a failed strike); grip force: the belief's H8 corridor (p_slip, p_rupture < 5 %)
  predict()    P(success) of picking-up / cracking from the belief (the pre-act verdict's quantitative basis, D4)
  probe()      weighing before cracking when its expected gain in P(crack) (preposterior over imagined readings) >= 0.02
  r_<schema>   emulator actions; outcomes become observations (belief updates) or typed failures (Q2)
Durations are ILLUSTRATIVE (the emulator has no kinematics).
"""
from __future__ import annotations

import copy
import json
import os

import numpy as np

from ccram.adapter import Adapter
from ccram.kernel import KernelFailure

HT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))       # HouseholdTwin
TOOLS = {"bowl-rim": {"kind": "rim", "surface": "bowl_rim_ceramic", "edge_radius": 1.5e-3},
         "counter": {"kind": "flat", "surface": "granite"}}
SECONDS = {"picking-up": 2.0, "cracking": 1.5, "opening": 3.0, "inspecting": 2.0, "fishing-out": 8.0,
           "float-testing": 20.0, "weighing": 3.0, "emptying": 3.0}                  # ILLUSTRATIVE
P_SEE_SHARD = 0.95      # bowl monitor: shard count exact 11/11 on renders (docs/EGG_SDT.md L4) -> ILLUSTRATIVE for real
P_SHATTER_MAX = 0.05    # chance bound of the strike decision (shatter is unrecoverable) -- ILLUSTRATIVE
EPS_STRIKE = 0.1        # strike outcomes the belief's model does not explain (tempering) -- ILLUSTRATIVE
CAUTION = 0.6           # cautious policy: the first strike is safe even if shells are 0.6 x as strong as believed (ILLUSTRATIVE)
VOI_MIN = 0.02          # a probe is worth it when it raises the expected P(crack) by >= 0.02 -- ILLUSTRATIVE
SPEEDS = np.round(np.arange(0.05, 0.90, 0.01), 2)
N_PARTICLES = 2000
DEFAULT_EGGS = {"egg-1": dict(instance=3), "egg-2": dict(instance=5)}
STUDY_V = {"bowl-rim": 0.25, "counter": 0.45}   # test_reports/egg_crack_study.json v_best (thin rim, granite)
STRIKE_SEEN = {"intact": "intact", "check": "check", "cracked_open": "cracked_open", "shattered": "shattered"}


class EggAdapter(Adapter):
    OBSERVABLE = {"held", "shell-state", "contents-in", "yolk-whole", "yolk-broken", "inspected", "shell-free",
                  "shell-seen", "float-tested", "float-observed", "weighed", "empty", "batch-bowl"}
    MODEL_ONLY = {"spilled", "pre-grasp", "in-water", "tilted", "upright", "above", "contact"}
    EXCEPTION_KINDS = {"PreconditionError": "manipulation-failure"}
    instance_servant = "egg-carton"

    def __init__(self, eggs=None, seed=0, out_dir=None, belief_bias=None, policy="belief"):
        super().__init__(None, None)
        self.policy = policy                # "belief" | "cautious" (robust first strike) | "study" (baseline: crack-study speed on the rim, +15 % per
                                            # retry, no belief decisions, no prediction, no probes)
        from food.egg_belief import EggBelief
        from food.egg_emulator import EggEmulator
        self.out = out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "episodes", "emulator")
        self.spec = dict(eggs or DEFAULT_EGGS)
        self.emu = {s: EggEmulator(d["instance"], os.path.join(self.out, s), seed=seed + k,
                                   storage=d.get("storage", "fridge"), age_days=d.get("age_days"), write_usd=False)
                    for k, (s, d) in enumerate(self.spec.items())}
        self.belief = {s: EggBelief.prior(n=N_PARTICLES, seed=seed + 10 + k, storage=d.get("storage", "fridge"))
                       for k, (s, d) in enumerate(self.spec.items())}
        if belief_bias:
            self.corrupt_belief(belief_bias)
        self.obs = {s: dict(shell="intact", held=False, float=None, mass=None, yolk=None, where=None) for s in self.emu}
        self.cups = {"cup-1": dict(egg=None, inspected=False, seen=None)}
        self.bowl = []                      # eggs whose contents are in the batch bowl
        self.probed = set()                 # (egg, probe) done
        self.tries = {s: 0 for s in self.emu}   # failed strikes per egg (baseline policy)
        self.struck = set()                 # eggs struck at least once (their belief has seen a strike outcome)
        self.decisions = {}                 # egg -> current strike decision (surface, v, probs)
        self.speed_override, self.force_override = [], []
        self.break_yolk = set()             # scenario: eggs whose yolk breaks on opening
        self.t = 0.0
        self.tracked = list(self.emu) + list(self.cups) + ["batch-bowl"]
        self.rng = np.random.default_rng(seed + 100)

    def corrupt_belief(self, bias):
        """A misspecified belief (repair-survival test): scale particle parameters, e.g. {"F_break_equatorial": 0.6}.
        The twin's truth is unchanged -- the robot believes something wrong about it."""
        for b in self.belief.values():
            for key, f in bias.items():
                b.particles[key] = b.particles[key] * f
            b._rebuild_state()
        getattr(self, "decisions", {}).clear()

    # -- entities
    def instances(self, kw):
        if kw == "egg":                     # unopened eggs first (an attributive designator: one still usable)
            return sorted(self.emu, key=lambda s: self.obs[s]["shell"] in ("opened", "shattered"))
        return {"cup": list(self.cups), "batch-bowl": ["batch-bowl"], "bowl-rim": ["bowl-rim"],
                "flat-surface": ["counter"], "strike-surface": list(TOOLS)}.get(kw)

    def resolve_symbol(self, s):
        return s if s in self.emu or s in self.cups or s in TOOLS or s == "batch-bowl" else None

    def sym(self, v):
        return v if isinstance(v, str) else super().sym(v)

    def time(self):
        return self.t

    # -- what the robot has observed
    def atoms(self):
        a = [["batch-bowl", "batch-bowl"]]
        for s, o in self.obs.items():
            a.append(["shell-state", s, o["shell"]])
            if o["held"]:
                a.append(["held", s])
            if o["float"] is not None:
                a += [["float-tested", s], ["float-observed", s, o["float"][0], o["float"][1]]]
            if o["mass"] is not None:
                a += [["weighed", s], ["mass", s, ["mass", o["mass"]]]]
            if o["yolk"] is not None:
                a.append(["yolk-whole" if o["yolk"] == "whole" else "broken-yolk", s])
            if o["where"] is not None:
                a.append(["contents-in", s, o["where"]])
        for c, st in self.cups.items():
            if st["egg"] is None:
                a.append(["empty", c])
            if st["inspected"]:
                a += [["inspected", c], ["shell-seen", c, st["seen"]]]
                if st["seen"] == 0:
                    a.append(["shell-free", c])
        return a

    # -- the belief: decisions
    def _observe(self, egg, kind, value):
        from food.egg_belief import Observation
        b, obs = self.belief[egg], Observation(kind, value, self.t / 3600.0, "robot")
        if kind == "strike":                # tempered likelihood (robust Bayes): the belief's strike model is nearly
            pr = b.strike_outcome_probs(value["v"], value["tool"])      # deterministic; with probability EPS_STRIKE
            lik = pr[:, b.STRIKE_STATES.index(value["outcome"])]        # the outcome comes from variation it does not
            lik = (1 - EPS_STRIKE) * lik + EPS_STRIKE / 4               # model -- one no-crack no longer makes the
            if value["outcome"] != "intact":                            # next strike look certain (it shattered eggs)
                b.particles["microcracked"] = np.zeros(b.n, bool)
            b._apply(lik, obs, self.rng)
            return
        b.observe(obs, self.rng)

    @staticmethod
    def _strike_decision(b, surfaces=TOOLS, guard=None):
        """argmax P(check or cracked open) under the belief b subject to P(shatter) <= P_SHATTER_MAX -- under b, and
        also under the pessimistic belief `guard` when given (robust chance constraint: the cautious first strike)."""
        best = None
        for name in surfaces:
            tool = TOOLS[name]
            for v in SPEEDS:
                p = b.strike_outcome_probs(float(v), tool).T @ b.weights
                if p[3] > P_SHATTER_MAX:
                    continue
                if guard is not None and (guard.strike_outcome_probs(float(v), tool).T @ guard.weights)[3] > P_SHATTER_MAX:
                    continue
                if best is None or p[1] + p[2] > best[2]:
                    best = (name, float(v), float(p[1] + p[2]), p)
        if best is None:                    # no speed meets the bound: the least shattering one on the rim
            v = float(SPEEDS[0])
            p = b.strike_outcome_probs(v, TOOLS["bowl-rim"]).T @ b.weights
            best = ("bowl-rim", v, float(p[1] + p[2]), p)
        return best

    def _pessimistic(self, egg):
        """The belief with shells at CAUTION x their believed strength (the misspecification the first strike must
        survive); None after the egg's first strike (its outcome has corrected the belief)."""
        if self.policy != "cautious" or egg in self.struck:
            return None
        g = copy.deepcopy(self.belief[egg])
        for k in ("F_break_equatorial", "F_break_axial"):
            g.particles[k] = g.particles[k] * CAUTION
        g._rebuild_state()
        return g

    def _decide(self, egg, surfaces=TOOLS):
        d = self._strike_decision(self.belief[egg], surfaces, self._pessimistic(egg))
        self.decisions[egg] = d
        return d

    # -- servants
    def resolve(self, x, env, role, climb):
        if not isinstance(x, list) or not x:
            return None
        egg = env.get("_object", (None, None))[1]
        if self.policy == "study":
            if x[0] in ("a", "an") and x[1] in ("strike-surface", "bowl-rim", "flat-surface"):
                return TOOLS["bowl-rim"], "bowl-rim", dict(servant="strike-study", status="anchored")
            if x[0] == "some" and x[1] == "strike-speed":
                v = round(STUDY_V["bowl-rim"] * 1.15 ** self.tries.get(egg, 0), 3)
                return v, v, dict(servant="strike-study", status="anchored", note="v_best +15 % per retry")
            if x[0] == "some" and x[1] == "grip-force":
                return 6.0, 6.0, dict(servant="grip-corridor", status="commonsense")
        if x[0] in ("a", "an") and x[1] in ("strike-surface", "bowl-rim", "flat-surface") and egg in self.emu:
            allowed = {"strike-surface": list(TOOLS), "bowl-rim": ["bowl-rim"], "flat-surface": ["counter"]}[x[1]]
            name, v, p, probs = self._decide(egg, allowed)
            return TOOLS[name], name, dict(servant="belief-decision", status="anchored", revisable=x[1] == "strike-surface",
                                           note=f"max P(crack) {p:.2f} at {v} m/s with P(shatter) "
                                                f"{probs[3]:.3f} <= {P_SHATTER_MAX} (EggBelief, {N_PARTICLES} particles)")
        if x[0] == "some" and x[1] == "strike-speed":
            if self.speed_override:
                v = self.speed_override.pop(0)
                return v, v, dict(servant="strike-study", status="assumed", note="scenario: a timid first strike")
            if egg not in self.decisions:
                self._decide(egg)
            name, v, p, probs = self.decisions[egg]
            return v, v, dict(servant="belief-decision", status="anchored", revisable=True,
                              note=f"on {name}: P(crack) {p:.2f}, P(no crack) {probs[0]:.2f}, P(shatter) {probs[3]:.3f}")
        if x[0] == "some" and x[1] == "grip-force":
            if self.force_override:
                f = self.force_override.pop(0)
                return f, f, dict(servant="grip-corridor", status="assumed", note="scenario: an override")
            cor = self.belief[egg].corridor() if egg in self.belief else None
            if cor is None:
                return 6.0, 6.0, dict(servant="grip-corridor", status="commonsense",
                                      note="no admissible corridor in the belief: 'firm but gentle'")
            f = round(float(np.sqrt(cor[0] * cor[1])), 2)
            p = self.belief[egg].probs(f)
            return f, f, dict(servant="grip-corridor", status="anchored",
                              note=f"H8 corridor {cor[0]:.1f}-{cor[1]:.1f} N (geometric mean); "
                                   f"P(slip) {p['p_slip']:.3f}, P(rupture) {p['p_rupture']:.3f}")
        return None

    # -- prediction and probes (D4)
    def predict(self, t, R):
        egg = R.get("object", (None, None))[1]
        if egg not in self.belief or self.policy == "study":
            return None
        b = self.belief[egg]
        if t == "picking-up":
            p = b.probs(float(R["force"][0]))
            ps = 1 - p["p_slip"] - p["p_rupture"]
            fail = "egg-slipped" if p["p_slip"] > 0.5 else "egg-crushed" if p["p_rupture"] > 0.5 else None
            return dict(prediction=f"P(slip) {p['p_slip']:.3f}, P(rupture) {p['p_rupture']:.3f}", predicted_failure=fail,
                        p_success=ps)
        if t == "cracking":
            pr = b.strike_outcome_probs(float(R["speed"][0]), R["tool"][0]).T @ b.weights
            fail = "no-crack" if pr[0] > 0.5 else "egg-shattered" if pr[3] > 0.5 else None
            if fail == "no-crack" and self._pessimistic(egg) is not None:     # the cautious first strike: a likely
                return dict(prediction="cautious first strike (informative: its outcome corrects the belief); "   # no-crack
                            "P(intact, check, open, shatter) = " + ", ".join(f"{q:.2f}" for q in pr),          # is its
                            predicted_failure=None, p_success=float(pr[1] + pr[2]))                            # point
            return dict(prediction="P(intact, check, open, shatter) = " + ", ".join(f"{q:.2f}" for q in pr),
                        predicted_failure=fail, p_success=float(pr[1] + pr[2]))
        return None

    def probe(self, t, R):
        """Weigh before cracking when the preposterior expected P(crack) gain >= VOI_MIN (mass narrows the impact
        mass and the shell geometry of the particles)."""
        egg = R.get("object", (None, None))[1]
        if self.policy == "study" or t != "cracking" or egg not in self.belief or (egg, "weighing") in self.probed or self.obs[egg]["mass"]:
            return None
        if R.get("speed", (None, None, {}))[2].get("status") == "assumed":
            return None                     # an override: nothing to re-decide
        b = self.belief[egg]
        guard = self._pessimistic(egg)
        now = self._strike_decision(b, TOOLS, guard)[2]
        from food.egg_belief import Observation
        D = b.derived()
        rng = np.random.default_rng(7)
        after = []
        for i in rng.choice(b.n, 12, p=b.weights):          # imagined readings from the belief itself
            bb = copy.deepcopy(b)
            bb.observe(Observation("mass_g", {"value": float(D["mass_g"][i] + rng.normal(0, 0.3))}, 0, "imagined"), rng)
            after.append(self._strike_decision(bb, TOOLS, guard)[2])
        gain = float(np.mean(after) - now)
        self.probed.add((egg, "weighing"))
        if gain < VOI_MIN:
            return None
        return dict(designator=["an", "action", ["type", "weighing"], ["object", ["the", "egg", ["tag", egg]]]],
                    why=f"weighing {egg}: expected P(crack) {now:.3f} -> {np.mean(after):.3f} (VOI {gain:+.3f} >= {VOI_MIN})")

    def gate_failure(self, phi):
        return "batch-spoiled"

    # -- acting: the emulator's actions, one callable per realisation
    def execute(self, nodes, attach=None):
        for step in nodes:
            step()

    def _tick(self, schema):
        self.t += SECONDS.get(schema, 1.0)

    def r_picking_up(self, R):
        egg, f = R["object"][1], float(R["force"][0])

        def grip():
            self._tick("picking-up")
            out = self.emu[egg].grip(f)
            m = out.measured
            self._observe(egg, "grip", {"F_N": float(m["F_N"]), "slipped": bool(m["slipped"]), "cracked": bool(m["cracked"])})
            o = self.obs[egg]
            if out.outcome == "crushed":
                o.update(shell="shattered", held=False)
                raise KernelFailure("egg-crushed", f"{egg} crushed at {f} N")
            if out.outcome == "slipped":
                raise KernelFailure("egg-slipped", f"{egg} slipped at {f} N")
            if out.outcome == "cracked":
                o["shell"] = "cracked"          # a crack line is visible
            o["held"] = True
        return [grip]

    def r_cracking(self, R):
        egg, tool, v = R["object"][1], R["tool"][0], float(R["speed"][0])

        def strike():
            self._tick("cracking")
            out = self.emu[egg].strike(v, tool)
            o = self.obs[egg]
            seen = out.measured["outcome"]
            self._observe(egg, "strike", {"v": v, "tool": tool, "outcome": STRIKE_SEEN[seen]})
            self.struck.add(egg)
            self.decisions.pop(egg, None)       # the next strike is decided on the updated belief
            if seen == "intact":
                self.tries[egg] += 1
                raise KernelFailure("no-crack", f"{egg} at {v} m/s on {tool['kind']}: no dent")
            if seen == "shattered":
                o.update(shell="shattered", held=False)
                raise KernelFailure("egg-shattered", f"{egg} shattered at {v} m/s on {tool['kind']}")
            o["shell"] = "cracked"
        return [strike]

    def r_opening(self, R):
        egg, dest = R["object"][1], R["destination"][1]

        def open_up():
            self._tick("opening")
            self.events.append((self.t, ["opened-into", egg, dest], self.atoms()))
            out = self.emu[egg].open_up()
            o = self.obs[egg]
            o.update(shell="opened", held=False)
            if out.outcome != "released":
                raise KernelFailure("egg-shattered", f"{egg}: contents spilled")
            m = out.measured
            if egg in self.break_yolk:          # scenario disturbance: the yolk breaks on opening
                self.emu[egg].s.yolk = self.emu[egg].truth.yolk = "broken"
                m = dict(m, yolk_broken=True)
            self._observe(egg, "opened", {"albumen_height_mm": None, "yolk_index": None if m.get("yolk_broken")
                                          else float(m["yolk_index"])})
            o["where"] = dest
            o["yolk"] = "broken" if m.get("yolk_broken") else "whole"
            if dest in self.cups:
                self.cups[dest].update(egg=egg, inspected=False, seen=None)
            if o["yolk"] == "broken":
                self.interp.signal("yolk-broken", f"{egg}: the yolk spreads in {dest}")
        return [open_up]

    def _look(self, cup):
        st = self.cups[cup]
        n = self.emu[st["egg"]].s.shards_in_bowl if st["egg"] else 0
        st.update(inspected=True, seen=int(self.rng.binomial(n, P_SEE_SHARD)))
        return st["seen"]

    def r_inspecting(self, R):
        cup = R["container"][1]

        def look():
            self._tick("inspecting")
            self._look(cup)
        return [look]

    def r_fishing_out(self, R):
        cup = R["container"][1]

        def fish():
            self._tick("fishing-out")
            egg = self.cups[cup]["egg"]
            if egg:
                self.emu[egg].fish_out()
            if self._look(cup):
                raise KernelFailure("shell-left-in-contents", f"{self.cups[cup]['seen']} fragments still seen in {cup}")
        return [fish]

    def r_float_testing(self, R):
        egg = R["object"][1]

        def test():
            self._tick("float-testing")
            m = self.emu[egg].float_test().measured
            self.obs[egg]["float"] = ("floats", 90.0) if m["floats"] else ("sinks", round(float(m["tilt_deg"]), 1))
            self._observe(egg, "float_test", {"floats": True} if m["floats"] else {"floats": False,
                                                                                  "tilt_deg": float(m["tilt_deg"])})
        return [test]

    def r_weighing(self, R):
        egg = R["object"][1]

        def weigh():
            self._tick("weighing")
            g = float(self.emu[egg].weigh().measured["mass_g"])
            self.obs[egg]["mass"] = round(g / 1000, 4)
            self._observe(egg, "mass_g", {"value": g})
        return [weigh]

    def r_emptying(self, R):
        cup, bowl = R["object"][1], R["destination"][1]

        def tip():
            self._tick("emptying")
            self.events.append((self.t, ["emptied-into", cup, bowl], self.atoms()))
            egg = self.cups[cup]["egg"]
            if egg:
                self.obs[egg]["where"] = bowl
                self.bowl.append(egg)
            self.cups[cup].update(egg=None, inspected=False, seen=None)
        return [tip]

    def truth(self):
        """The emulator's hidden state (for the researcher and tests; never read by the kernel)."""
        return {s: dict(shell=e.s.shell, yolk=e.s.yolk, shards_in_vessel=e.s.shards_in_bowl,
                        age_d=round(e.fresh()["age_d"], 1), float_class=e.fresh()["float"]) for s, e in self.emu.items()}


def make_adapter(demo, world, context):
    """activity.toml [twin] adapter entry point (an emulator twin: demo/world/context are None)."""
    return EggAdapter(eggs=json.loads(os.environ["EGG_SPEC"]) if os.environ.get("EGG_SPEC") else None,
                      belief_bias=json.loads(os.environ["EGG_BELIEF_BIAS"]) if os.environ.get("EGG_BELIEF_BIAS") else None)
