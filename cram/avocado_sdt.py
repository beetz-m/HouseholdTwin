"""The robot's avocado SDT node: a belief over the avocado's hidden properties, created when the robot detects an
avocado, and the competence questions it answers by consulting the belief, the task models and mental simulation.

Belief: the particle filter of food/avocado_belief.py (4000 candidate avocados from the register prior, narrowed by
observations). Discrete state: integrity, seed, seed coat (food/schema.py AVOCADO). Mental simulation: the process
models of food/avocado_process.py (the mechanics the emulator runs) executed on avocados drawn from the belief
("imagined avocados"), so that a plan's outcomes are predicted as distributions before it is executed. Execution
monitoring: the measured outcome of a step is located in the imagined distribution of that step (expected /
surprising), then the belief is updated with the observations the step yields (food/avocado_tasks.py).

Everything the node knows or did is kept as plain records (BeliefVariable, CompetenceAnswer, ImaginedOutcome,
MonitoringRecord) so that the CRAMERA query panel can range over them. No CRAM import here: the node is usable on its
own (cram/avocado_world.py connects it to the semantic digital twin).
"""

import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from food import avocado_belief as AB  # noqa: E402
from food import avocado_process as PRC  # noqa: E402
from food import avocado_tasks as T  # noqa: E402
from food.avocado import Avocado  # noqa: E402
from food.schema import AVOCADO as SCHEMA  # noqa: E402

STAGE_WORDS = {1: "hard", 2: "firm", 3: "softening", 4: "nearly ripe", 5: "ripe, eat now"}
"""Words for the ripeness stages; the end points are the scale's (avo_au_stages: 1 hard .. 5 ripe / eat now), the
words in between are paraphrases (ILLUSTRATIVE)."""

SHOWN = ["ripeness_stage", "mass_g", "length_L", "diameter_D", "seed_length", "skin_thickness", "penetrometer_N",
         "flesh_E_MPa", "ripening_rate", "bruised", "testa_adheres_to_flesh"]
"""The belief variables offered to the query panel."""


# %% records the query panel ranges over
@dataclass
class BeliefVariable:
    """One variable of the node's belief: its distribution as mean, sd and the 90 % interval."""
    node: str
    variable: str
    mean: float
    sd: float
    p05: float
    p95: float
    unit: str

    @property
    def name(self):
        return f"{self.node}.{self.variable}"


@dataclass
class CompetenceAnswer:
    """A competence question the robot answered about the avocado, with what the answer rests on."""
    question: str
    answer: str
    evidence: str
    method: str          # belief | task model | ontology | mental simulation | execution monitoring
    time_s: float = 0.0

    @property
    def name(self):
        return self.question


@dataclass
class ImaginedOutcome:
    """The outcome distribution of one plan, imagined on avocados drawn from the belief."""
    plan: str
    wait_h: float
    n: int
    p_success: float          # halves apart and seed out
    p_clean_halves: float     # the twist releases at the seed, no torn flesh
    p_torn: float
    p_stuck: float            # a step fails within its force / torque limit
    p_seed_coat_left: float
    p_gouge: float
    hazardous: bool           # the plan contains a step flagged hazardous (knife strike into the seed)
    quality: float            # expected quality 0..1 of the pitted halves
    twist_torque_Nm_p50: float
    twist_torque_Nm_p95: float
    seed_force_N_p50: float
    seed_force_N_p95: float

    @property
    def name(self):
        return f"{self.plan}" + (f" (after {self.wait_h:.0f} h)" if self.wait_h else "")


@dataclass
class MonitoringRecord:
    """One executed step: what the imagination expected, what was measured, and how the belief moved."""
    step: str
    expected: str
    measured: str
    verdict: str              # as expected | unexpected (P < 0.2) | surprising (P < 0.05 or outside the 95 % band)
    belief_before: str
    belief_after: str

    @property
    def name(self):
        return self.step


# %% imagined avocados
class ImaginedAvocado(Avocado):
    """An avocado of the belief for mental simulation: the geometry the process models use (shape, seed, layers),
    without the radial colour tables that only the flesh-band meshes need (1 ms instead of 1 s)."""

    def _build_radial_tables(self):
        pass


def particle_avocado(belief, i, wait_h=0.0):
    """The avocado of particle ``i``, ripened by ``wait_h`` hours at its own rate."""
    P = belief.particles
    over = {k: float(P[k][i]) for k in AB.RANDOM}
    over["ripeness_stage"] = min(5.0, over["ripeness_stage"] + over["ripening_rate"] * wait_h / 24.0)
    over["testa_adheres_to"] = "flesh" if P["testa_flesh"][i] else "seed"
    return ImaginedAvocado(int(i), overrides=over)


def photo_observations(img, t):
    """Belief observations from a camera image of the avocado (food/avocado_vision.py; as scripts/avocado_observe.py)."""
    from food import avocado_vision as AV
    obs, _ = AV.estimate(img)
    E = obs.estimates
    src = f"photo {os.path.basename(img)} ({obs.image_sha1})"
    return [AB.Observation("photo_shape", dict(aspect=float(E["aspect_L_over_D"].value), aspect_sd=0.05,
                                               neck=float(E["neck"].value), neck_sd=0.05), t, src),
            AB.Observation("skin_colour", dict(L=float(E["skin_Lab_L"].value), C=float(E["skin_chroma"].value),
                                               h=float(E["skin_hue"].value)), t, src),
            AB.Observation("stem_button", dict(present=E["stem_button"].value == "present"), t, src)], obs


def summary_line(belief, keys=("ripeness_stage", "seed_length", "testa_adheres_to_flesh")):
    S = belief.summary()
    return "; ".join(f"{k} {S[k]['mean']:.2f} +- {S[k]['sd']:.2f}" for k in keys)


# %% the node
@dataclass
class AvocadoSDT:
    """The robot's SDT node of one avocado."""
    name: str
    belief: AB.AvocadoBelief
    integrity: str = "whole"
    seed: str = "in"
    seed_coat: str = "on_seed"
    current_task: str = "none"
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    answers: list = field(default_factory=list)
    imagined: list = field(default_factory=list)
    monitoring: list = field(default_factory=list)
    clock_s: float = 0.0

    @classmethod
    def on_detection(cls, name, n=4000, seed=0):
        """A new node for a detected avocado: the register prior (the population of Hass fruit)."""
        return cls(name, AB.AvocadoBelief.prior(n, seed=seed), rng=np.random.default_rng(seed))

    # %% perception
    def perceive(self, observations):
        """Fold observations (food/avocado_belief.py Observation) into the belief."""
        for o in observations:
            self.belief.observe(o, self.rng)

    @property
    def variables(self):
        S = self.belief.summary()
        return [BeliefVariable(self.name, k, S[k]["mean"], S[k]["sd"], S[k]["p05"], S[k]["p95"], S[k]["unit"])
                for k in SHOWN]

    def _answer(self, question, answer, evidence, method):
        a = CompetenceAnswer(question, answer, evidence, method, self.clock_s)
        self.answers.append(a)
        return a

    # %% competence questions: what is it?
    def what_is_it(self):
        S = self.belief.summary()
        st = S["ripeness_stage"]
        k = int(np.clip(round(st["mean"]), 1, 5))
        parts = "; ".join(f"{p}: {', '.join(c)}" for p, c in SCHEMA["parts"].items())
        return self._answer(
            "What is the object?",
            f"a Hass avocado (Avocado < Fruit < Produce < Food), {S['mass_g']['mean']:.0f} +- {S['mass_g']['sd']:.0f} "
            f"g, {S['length_L']['mean']:.0f} x {S['diameter_D']['mean']:.0f} mm, ripeness stage "
            f"{st['mean']:.1f} +- {st['sd']:.1f} ({STAGE_WORDS[k]}); parts: {parts}",
            f"belief after {len(self.belief.history)} observation steps, ESS {self.belief.ess():.0f}", "belief")

    def how_ripe(self):
        P, w = self.belief.particles, self.belief.weights
        st = P["ripeness_stage"]
        p_eat = float(np.sum(w[st >= 4.5]))
        p_hard = float(np.sum(w[st < 2.0]))
        days = np.clip((4.5 - st) / np.maximum(P["ripening_rate"], 1e-3), 0, None)
        d50 = float(np.interp(0.5, np.cumsum(w[np.argsort(days)]), np.sort(days)))
        return self._answer(
            "How ripe is it, and when can it be eaten?",
            f"P(ready to eat, stage >= 4.5) = {p_eat:.2f}; P(still hard, stage < 2) = {p_hard:.2f}; ready in about "
            f"{d50:.1f} days (median over the belief)",
            "ripeness stage and ripening rate per particle (register ripening model)", "belief")

    def where_is_the_seed(self):
        d = AB.seed_depths_mm(self.belief.particles)
        w = self.belief.weights
        lines = []
        for k, v in d.items():
            o = np.argsort(v)
            c = np.cumsum(w[o])
            lines.append(f"{k} {np.interp(0.05, c, v[o]):.0f}..{np.interp(0.95, c, v[o]):.0f} mm")
        return self._answer(
            "Where is the seed, how deep will the knife go?",
            "blade stops on the seed coat at a depth (90 % interval) of " + ", ".join(lines),
            "seed size and offset, skin thickness, fruit size per particle", "belief")

    # %% what can be done with it?
    def applicable_actions(self):
        state = dict(integrity=self.integrity, seed=self.seed)
        ok = []
        for name, t in T.TASKS.items():
            pre = " ".join(t["pre"])
            if ("integrity = whole" in pre and state["integrity"] in ("whole", "partly_scored")) or \
               ("scored" in pre and state["integrity"] in ("scored", "partly_scored")) or \
               ("halved" in pre and state["integrity"] in ("halved", "halved_torn") and state["seed"] == "in"):
                ok.append(name)
        return ok

    def what_can_be_done(self):
        acts = self.applicable_actions()
        desc = "; ".join(f"{a}: {T.TASKS[a]['goal']}" for a in acts) or "nothing from the halving / pitting tasks"
        return self._answer(
            "What can I do with it now?",
            f"state integrity={self.integrity}, seed={self.seed}: {desc}",
            "preconditions of the task models (food/avocado_tasks.py, food/schema.py)", "task model")

    def what_am_i_doing(self):
        t = T.TASKS.get(self.current_task)
        if t is None:
            return self._answer("What am I doing with it?", "nothing yet", "current task", "task model")
        fails = "; ".join(f"{k}: watch for '{c}', then {r}" for k, (c, r) in t["failures"].items())
        return self._answer(
            "What am I doing with it?",
            f"{self.current_task}: {t['goal']}; steps: {' -> '.join(t['steps'])}; possible failures: {fails}",
            f"designator {self.current_task} (food/avocado_tasks.py)", "task model")

    # %% mental simulation
    def imagine(self, options, n=150, wait_h=0.0, seed=0):
        """Imagine plans on ``n`` avocados drawn from the belief, ripened by ``wait_h`` hours. ``options`` maps a plan
        name to its seed-removal method; the halving steps (cut round the seed, twist apart) are shared. Returns one
        ImaginedOutcome per option."""
        rng = np.random.default_rng(seed)
        w = self.belief.weights
        idx = rng.choice(self.belief.n, n, p=w / w.sum())
        rows = {o: [] for o in options}
        for i in idx:
            avo = particle_avocado(self.belief, i, wait_h)
            stage = avo.p["ripeness_stage"]
            plane = PRC.CutPlane(0.0, 360.0)
            cut = PRC.cut_round_seed(avo, plane, rng, stage)
            tw = PRC.twist_apart(avo, plane, rng, stage, plane.coverage_deg) if cut.ok else None
            for o, method in options.items():
                rm = PRC.remove_seed(avo, rng, stage, method) if tw is not None and tw.ok else None
                rows[o].append((cut, tw, rm))
        out = []
        for o, method in options.items():
            R = rows[o]
            ok = np.array([rm is not None and rm.ok for _, _, rm in R])
            clean = np.array([tw is not None and tw.outcome == "halved" for _, tw, _ in R])
            torn = np.array([tw is not None and tw.outcome == "halved_torn" for _, tw, _ in R])
            stuck = np.array([(not c.ok) or (tw is not None and not tw.ok) or (rm is not None and not rm.ok)
                              for c, tw, rm in R])
            coat = np.array([rm is not None and rm.ok and rm.truth.get("testa_remnant_fraction", 0) > 0.05
                             for _, _, rm in R])
            gouge = np.array([rm is not None and rm.ok and bool(rm.truth.get("gouges")) for _, _, rm in R])
            hazard = any(rm is not None and any("hazardous" in e for e in rm.events) for _, _, rm in R)
            q = ok * (1.0 - 0.35 * torn - 0.25 * coat - 0.3 * gouge)
            T_tw = [tw.measured["T_peak_Nm"] for _, tw, _ in R if tw is not None and "T_peak_Nm" in tw.measured]
            F_rm = [rm.measured.get("F_peak_N", rm.measured.get("T_peak_Nm", np.nan)) for _, _, rm in R
                    if rm is not None]
            pct = (lambda v, p: float(np.nanpercentile(v, p)) if len(v) else float("nan"))
            io = ImaginedOutcome(o, wait_h, n, float(ok.mean()), float(clean.mean()), float(torn.mean()),
                                 float(stuck.mean()), float(coat.mean()), float(gouge.mean()), hazard,
                                 float(q.mean()), pct(T_tw, 50), pct(T_tw, 95), pct(F_rm, 50), pct(F_rm, 95))
            self.imagined.append(io)
            out.append(io)
        return out

    # %% value of information: squeeze before deciding?
    def _rollout(self, i, wait_h, method, rng):
        """Quality of the pitted halves and the ripeness stage at preparation time, for particle ``i``."""
        avo = particle_avocado(self.belief, i, wait_h)
        stage = avo.p["ripeness_stage"]
        plane = PRC.CutPlane(0.0, 360.0)
        cut = PRC.cut_round_seed(avo, plane, rng, stage)
        tw = PRC.twist_apart(avo, plane, rng, stage, plane.coverage_deg) if cut.ok else None
        rm = PRC.remove_seed(avo, rng, stage, method) if tw is not None and tw.ok else None
        if rm is None or not rm.ok:
            return 0.0, stage
        torn = tw.outcome == "halved_torn"
        coat = rm.truth.get("testa_remnant_fraction", 0) > 0.05
        gouge = bool(rm.truth.get("gouges"))
        return 1.0 - 0.35 * torn - 0.25 * coat - 0.3 * gouge, stage

    @staticmethod
    def mashable(stage):
        """How well the flesh mashes for guacamole: 0 below stage 2.5, 1 from stage 4 (ILLUSTRATIVE)."""
        return np.clip((np.asarray(stage) - 2.5) / 1.5, 0.0, 1.0)

    def squeeze_force(self, forces=(4.0, 3.0, 2.0), max_p_bruise=0.02):
        """The strongest squeeze whose bruise risk under the belief stays below ``max_p_bruise``."""
        Dv = AB.derived(self.belief.particles)
        thr = AB.PR.AVOCADO["bruise_force_ratio"][1] * Dv["penetrometer_N"]
        w = self.belief.weights
        for f in forces:
            p = float(np.sum(w * (f > thr)))
            if p < max_p_bruise:
                return f, p
        return None, p

    def value_of_squeeze(self, n=120, seed=3, wait_cost=0.05):
        """Preposterior analysis for the goal 'guacamole now, or store the avocado two more days': utility = quality
        of the pitted halves x mashability at preparation time (- a small cost of waiting). Is a squeeze before the
        decision worth it? The squeeze's force is the strongest one that will not bruise the fruit (belief); its
        reading is simulated with the belief's own measurement model (stiffness; sensor error grows as the force
        falls)."""
        rng = np.random.default_rng(seed)
        force, p_bruise = self.squeeze_force()
        w = self.belief.weights
        idx = rng.choice(self.belief.n, n, p=w / w.sum())
        options = {"guacamole now (halve, spoon out the seed)": (0.0, 0.0),
                   "store 2 days, then guacamole": (48.0, wait_cost)}
        U = {}
        for a, (wh, cost) in options.items():
            r = [self._rollout(i, wh, "spoon", rng) for i in idx]
            U[a] = np.array([q * float(self.mashable(st)) - cost for q, st in r])
        names = list(options)
        v0 = {a: float(U[a].mean()) for a in names}
        a0 = max(names, key=v0.get)
        if force is None:
            self._answer("Should I squeeze it before deciding?",
                         f"no: every squeeze force risks a bruise (P >= {p_bruise:.2f}); best without it: '{a0}'",
                         "bruise threshold 0.6 x penetrometer force (register) under the belief", "belief")
            return None, 0.0, p_bruise, a0
        k = AB.derived(self.belief.particles)["squeeze_k_N_per_mm"][idx]
        rel = 0.1 * 4.0 / force
        vals, changed = [], 0
        for j in range(n):
            y = k[j] * (1 + rng.normal(0, rel))
            sd = math.hypot(rel * y, 0.2 * y)
            lw = np.exp(-0.5 * ((k - y) / sd) ** 2)
            lw /= lw.sum()
            post = {a: float(lw @ U[a]) for a in names}
            best = max(names, key=post.get)
            vals.append(post[best])
            changed += best != a0
        voi = float(np.mean(vals)) - v0[a0]
        worth = voi > 0.005
        self._answer(
            "Should I squeeze it before deciding?",
            f"{'yes' if worth else 'no'}: without squeezing I would choose '{a0}' (expected utility {v0[a0]:.2f}; "
            + ", ".join(f"{a}: {v:.2f}" for a, v in v0.items()) + f"); a {force:.0f} N squeeze (P(bruise) "
            f"{p_bruise:.3f}) changes that choice in {changed / n:.0%} of the imagined readings and is worth "
            f"{voi:+.3f}",
            f"preposterior analysis: {n} imagined avocados x {n} imagined squeeze readings; utility = quality of the "
            "pitted halves x mashability (ILLUSTRATIVE)", "mental simulation")
        return force if worth else None, voi, p_bruise, a0

    def decide(self, wait_cost=0.05, n=150, seed=4):
        """'Guacamole now, or store it?' under the current belief."""
        rng = np.random.default_rng(seed)
        w = self.belief.weights
        idx = rng.choice(self.belief.n, n, p=w / w.sum())
        v = {}
        for a, (wh, cost) in {"guacamole now": (0.0, 0.0), "store 2 days, then guacamole": (48.0, wait_cost)}.items():
            r = [self._rollout(i, wh, "spoon", rng) for i in idx]
            v[a] = float(np.mean([q * float(self.mashable(st)) - cost for q, st in r]))
        best = max(v, key=v.get)
        self._answer("Guacamole now, or store it?",
                     f"{best} (expected utility " + ", ".join(f"{a}: {x:.2f}" for a, x in v.items()) + ")",
                     "mental simulation of halving and pitting now and after 48 h of ripening; utility = quality x "
                     "mashability (ILLUSTRATIVE)", "mental simulation")
        return best, v

    def expected_squeeze(self, force_N=4.0):
        Dv = AB.derived(self.belief.particles)
        k = Dv["squeeze_k_N_per_mm"]
        w = self.belief.weights
        o = np.argsort(k)
        c = np.cumsum(w[o])
        return float(np.sum(w * k)), float(np.interp(0.025, c, k[o])), float(np.interp(0.975, c, k[o]))

    def monitor_squeeze(self, k_meas, force_N=4.0):
        mean, lo, hi = self.expected_squeeze(force_N)
        rel = 0.1 * 4.0 / force_N
        before = summary_line(self.belief, ("ripeness_stage", "penetrometer_N"))
        self.perceive([AB.Observation("squeeze", dict(k=float(k_meas), sd=rel * float(k_meas), force_N=force_N),
                                      self.belief.time_h,
                                      "PR2 gripper squeeze")])
        after = summary_line(self.belief, ("ripeness_stage", "penetrometer_N"))
        verdict = "as expected" if lo <= k_meas <= hi else "surprising"
        rec = MonitoringRecord("SqueezeTest", f"stiffness {lo:.2f}..{hi:.2f} N/mm (95 %), mean {mean:.2f}",
                               f"stiffness {k_meas:.2f} N/mm at {force_N:.0f} N", verdict, before, after)
        self.monitoring.append(rec)
        self._answer("What did the squeeze tell me?",
                     f"{verdict}: {k_meas:.2f} N/mm (imagined {lo:.2f}..{hi:.2f}); ripeness belief {before} -> {after}",
                     "squeeze stiffness -> ripeness (register FEM squeeze table, belief measurement model)",
                     "execution monitoring")
        return rec

    def what_if_halved_now(self):
        res = self.imagine({"halve now, spoon out the seed": "spoon",
                            "halve now, knife-twist the seed out": "knife_twist"})
        later = self.imagine({"wait 2 days, halve, spoon": "spoon"}, wait_h=48.0, seed=1)
        best = max(res, key=lambda r: r.quality - (0.5 if r.hazardous else 0.0))
        txt = " | ".join(f"{r.plan}: P(success) {r.p_success:.2f}, P(clean halves) {r.p_clean_halves:.2f}, "
                         f"P(torn) {r.p_torn:.2f}, P(seed coat left) {r.p_seed_coat_left:.2f}, P(gouge) "
                         f"{r.p_gouge:.2f}, quality {r.quality:.2f}" + (", HAZARD" if r.hazardous else "")
                         for r in res + later)
        self._answer("What happens if I halve and pit it now?", txt,
                     f"mental simulation: cut round the seed, twist apart, remove the seed on {res[0].n} imagined "
                     "avocados from the belief (food/avocado_process.py)", "mental simulation")
        wait = later[0]
        self._answer(
            "Should I rather wait?",
            (f"waiting 2 days raises the expected quality from {best.quality:.2f} to {wait.quality:.2f} "
             f"(P(clean halves) {best.p_clean_halves:.2f} -> {wait.p_clean_halves:.2f})") if wait.quality > best.quality
            else f"no: waiting 2 days does not improve the expected quality ({wait.quality:.2f} vs {best.quality:.2f})",
            "the same mental simulation on the imagined avocados ripened by 48 h at their own rates",
            "mental simulation")
        return best, wait

    def how_to_do_it(self, best):
        a = self._answer(
            "How should I do it (parameters)?",
            f"{best.plan}; twist torque limit {max(0.3, 1.5 * best.twist_torque_Nm_p95):.2f} N m (imagined p95 "
            f"{best.twist_torque_Nm_p95:.2f}), seed removal limit {max(5.0, 1.5 * best.seed_force_N_p95):.1f} "
            f"(imagined p95 {best.seed_force_N_p95:.1f}); keep the blade on the seed coat",
            "limits = 1.5 x the 95th percentile of the imagined peak loads", "mental simulation")
        return a

    # %% execution monitoring
    def expect(self, task, n=150, seed=7, covered_deg=360.0):
        """Imagined distribution of the next step's measurable outcome, from the current state and belief."""
        rng = np.random.default_rng(seed)
        w = self.belief.weights
        idx = rng.choice(self.belief.n, n, p=w / w.sum())
        outs = []
        for i in idx:
            avo = particle_avocado(self.belief, i)
            stage, plane = avo.p["ripeness_stage"], PRC.CutPlane(0.0, covered_deg)
            if task == "CutRoundSeed":
                outs.append(PRC.cut_round_seed(avo, plane, rng, stage))
            elif task == "TwistApart":
                outs.append(PRC.twist_apart(avo, plane, rng, stage, covered_deg))
            else:
                outs.append(PRC.remove_seed(avo, rng, stage, "spoon"))
        return outs

    def monitor(self, task, outcome, expected):
        """Locate a step's measured outcome in its imagined distribution, then update belief and state."""
        p_out = float(np.mean([e.outcome == outcome.outcome for e in expected]))
        key = {"CutRoundSeed": "F_puncture_N", "TwistApart": "T_peak_Nm", "RemoveSeed": "F_peak_N"}[task]
        vals = np.array([e.measured[key] for e in expected if key in e.measured], float)
        m = outcome.measured.get(key)
        surprising = p_out < 0.05
        q_txt = ""
        if m is not None and len(vals):
            q = float(np.mean(vals <= m))
            surprising |= q < 0.025 or q > 0.975
            q_txt = (f"; {key} imagined {np.percentile(vals, 5):.2f}..{np.percentile(vals, 95):.2f}, measured "
                     f"{m:.2f} (quantile {q:.2f})")
        before = summary_line(self.belief)
        self.perceive(T.observations(outcome))
        after = summary_line(self.belief)
        self.integrity = {"scored": "scored", "partly_scored": "partly_scored", "halved": "halved",
                          "halved_torn": "halved_torn"}.get(outcome.outcome, self.integrity)
        if task == "RemoveSeed" and outcome.ok:
            self.seed = "out"
            self.seed_coat = "in_cavity" if outcome.measured.get("testa_visible_in_cavity") else "removed"
        verdict = "surprising" if surprising else ("unexpected" if p_out < 0.2 else "as expected")
        rec = MonitoringRecord(task, f"P({outcome.outcome}) = {p_out:.2f}" + q_txt.split(", measured")[0],
                               f"{outcome.outcome}" + (f", {key} {m:.2f}" if m is not None else "") +
                               (f"; {'; '.join(outcome.events)}" if outcome.events else ""),
                               verdict, before, after)
        self.monitoring.append(rec)
        self._answer(f"Did {task} go as expected?",
                     f"{rec.verdict}: expected {rec.expected}; measured {rec.measured}",
                     f"belief {before} -> {after}", "execution monitoring")
        return rec


def outcome_text(io):
    return (f"{io.plan}{' (after %.0f h)' % io.wait_h if io.wait_h else ''}: P(success) {io.p_success:.2f}, "
            f"P(clean) {io.p_clean_halves:.2f}, P(torn) {io.p_torn:.2f}, quality {io.quality:.2f}")


if __name__ == "__main__":                                   # quick check without CRAM
    node = AvocadoSDT.on_detection("avocado_1")
    for f in (node.what_is_it, node.how_ripe, node.where_is_the_seed, node.what_can_be_done):
        a = f()
        print(f"Q: {a.question}\nA: {a.answer}\n")
    best, wait = node.what_if_halved_now()
    for a in node.answers[-2:]:
        print(f"Q: {a.question}\nA: {a.answer}\n")
    print(math.isfinite(best.quality))
