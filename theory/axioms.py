"""Relational interval axioms of the avocado microtheory, and a replay audit against simulated episodes.

Language. holds(f, i): relation f persists over interval i; occurs(a, j): action or physical event a over interval j;
meets(i, j) orders intervals. An action axiom has a precondition over the relations holding before, a set of guarded
branches (guard over the hidden parameters Theta and the action's parameters; added and removed relations), and the
frame rule that every other relation persists. The abstraction alpha maps a plant state to its relations (the plant's
facts, food/avocado_bws.Plant.facts).

Audit (the refinement premises of the scoped correctness proposition, checked on executed steps): for every step of
every audited episode, (i) the precondition held in the abstract state before; (ii) the observed patch is one of the
axiom's branches; (iii) where a branch guard is deterministic in Theta (the squeeze's bruise guard, the knife-strike
flag), the branch taken is the one the guard selects; (iv) nothing outside the branch changed (frame). Counterexamples:
pairs of steps with the same abstract state and action whose branches differ -- the qualitative state alone does not
determine the outcome, the hidden attribute does.
"""

from dataclasses import dataclass

AXIOMS = {
    "Squeeze": dict(
        pre=lambda S: "Whole(Avocado)" in S,
        branches={"no bruise": (set(), set()), "bruise": ({"Bruised(Mesocarp)"}, {"Intact(Mesocarp)"})},
        guard=lambda att, th: "bruise" if att["F"] > th["bruise_threshold"] else "no bruise",
        text="occurs(Squeeze(F), j) & holds(Whole, i) & meets(i, j) -> holds(Bruised, j') iff F > F_bruise(stage)"),
    "Wait": dict(
        pre=lambda S: True,
        branches={"no change": (set(), set()), "ripe": ({"Ripe(Avocado)"}, set()),
                  "overripe": ({"Overripe(Avocado)"}, set()), "ripe+overripe": ({"Ripe(Avocado)", "Overripe(Avocado)"}, set()),
                  "overripe+spoiled": ({"Overripe(Avocado)", "Spoiled(Avocado)"}, set()),
                  "spoiled": ({"Spoiled(Avocado)"}, set()),
                  "ripe+overripe+spoiled": ({"Ripe(Avocado)", "Overripe(Avocado)", "Spoiled(Avocado)"}, set())},
        text="occurs(Wait(h, store), j) -> stage advances at rate x Q10(store); Ripe at stage 4; Overripe past the "
             "ripe window; Spoiled past spoil_days (lifetime model)"),
    "CutRoundSeed": dict(
        pre=lambda S: "Whole(Avocado)" in S,
        branches={"scored": ({"Scored(Avocado)"}, {"PartlyScored(Avocado)"}),
                  "scored (first)": ({"Scored(Avocado)"}, set()),
                  "partly scored": ({"PartlyScored(Avocado)"}, set()), "no entry": (set(), set())},
        text="occurs(CutRoundSeed, j) & holds(Whole, i) -> holds(Scored) if the blade's force suffices all round, "
             "else holds(PartlyScored)"),
    "TwistApart": dict(
        pre=lambda S: "Scored(Avocado)" in S or "PartlyScored(Avocado)" in S,
        branches={"clean": ({"Apart(HalfA,HalfB)", "SeedIn(HalfA)"},
                            {"Whole(Avocado)", "Scored(Avocado)", "PartlyScored(Avocado)", "SeedIn(Avocado)"}),
                  "torn": ({"Apart(HalfA,HalfB)", "SeedIn(HalfA)", "TornFlesh(HalfB)", "FleshOnSeed(Seed)"},
                           {"Whole(Avocado)", "Scored(Avocado)", "PartlyScored(Avocado)", "SeedIn(Avocado)"}),
                  "stuck": (set(), set())},
        text="occurs(TwistApart, j) & holds(Scored, i) -> Apart; TornFlesh unless the flesh releases at the seed "
             "(p_clean(stage)); Stuck if bridges + adhesion exceed the torque limit"),
    "RemoveSeed": dict(
        pre=lambda S: "Apart(HalfA,HalfB)" in S and "SeedIn(HalfA)" in S,
        branches={f"out{t}{g}{k}": ({"SeedOut(HalfA)"} | ({"TestaIn(Cavity)"} if t else set())
                                    | ({"Gouge(Cavity)"} if g else set()) | ({"UnsafeToolUse(KnifeStrike)"} if k else set()),
                                    {"SeedIn(HalfA)"})
                  for t in ("", "+testa") for g in ("", "+gouge") for k in ("", "+knife")} | {"stuck": (set(), set())},
        guard=lambda att, th: None,
        text="occurs(RemoveSeed(m), j) & holds(Apart & SeedIn(HalfA), i) -> SeedOut; TestaIn(Cavity) if the testa "
             "adheres to the flesh; Gouge possible (spoon); UnsafeToolUse iff m = knife strike"),
    "PeelSeedCoat": dict(
        pre=lambda S: "TestaIn(Cavity)" in S,
        branches={"clean": (set(), {"TestaIn(Cavity)"}), "left": (set(), set()),
                  "clean+gouge": ({"Gouge(Cavity)"}, {"TestaIn(Cavity)"}), "left+gouge": ({"Gouge(Cavity)"}, set())},
        text="occurs(PeelSeedCoat, j) & holds(TestaIn(Cavity), i) -> not TestaIn w.p. peel_coat_success"),
    "Photo": dict(pre=lambda S: True, branches={"none": (set(), set())}, text="observation only (frame)"),
}


@dataclass
class AuditResult:
    steps: int = 0
    pre_violations: int = 0
    branch_mismatches: int = 0
    guard_mismatches: int = 0
    frame_violations: int = 0
    counterexamples: int = 0


def audit_trace(trace, theta, res, seen):
    """Audit one plant trace (list of patches) against AXIOMS; ``seen`` collects (abstract state, action) -> branch
    sets for the counterexample search."""
    S = set(trace[0]["facts"])
    for st in trace[1:]:
        ax = AXIOMS[st["event"]]
        add, rem = set(st["add"]), set(st["rem"])
        res.steps += 1
        if not ax["pre"](S):
            res.pre_violations += 1
        # branches whose add / remove sets explain the observed patch (relations already present are no-ops)
        match = [b for b, (A, R) in ax["branches"].items()
                 if add == (A - S) and rem == (R & S)]
        if not match:
            res.branch_mismatches += 1
        if "guard" in ax and ax["guard"](st["attrs"], theta) is not None:
            g = ax["guard"](st["attrs"], theta)
            if g not in match:
                res.guard_mismatches += 1
        if st["event"] == "RemoveSeed" and (("UnsafeToolUse(KnifeStrike)" in add) != (st["cause"] == "knife_twist")):
            res.guard_mismatches += 1
        new = (S | add) - rem
        if set(st["facts"]) != new:
            res.frame_violations += 1
        key = (frozenset(S), st["event"], st["cause"] if st["event"] in ("RemoveSeed",) else "")
        seen.setdefault(key, set()).add(frozenset(add) | frozenset("-" + r for r in rem))
        S = set(st["facts"])
    return res


def run_audit(seeds=range(1000, 1030)):
    """Audit every step of the Task A and Task B policies on the held-out avocados of every ripeness class."""
    import numpy as np
    from theory import avocado_bws as W
    from theory import avocado_policies as P
    res, seen = AuditResult(), {}
    n_ep = 0
    for cls in W.CLASSES:
        for s in seeds:
            for pol in P.POLICIES_A:
                plant = W.Plant(W.instance(s, cls), seed=s)
                P.task_A(plant, pol, np.random.default_rng(s + 31))
                audit_trace(plant.trace, dict(bruise_threshold=plant.bruise_threshold()), res, seen)
                n_ep += 1
            for pol in P.POLICIES_B + P.ABLATIONS_B:
                plant = W.Plant(W.instance(s, cls), seed=s)
                P.task_B(plant, pol, np.random.default_rng(s + 31))
                # the bruise threshold changes with the stage over the episode: audit squeezes at their own stage
                audit_trace_stagewise(plant, res, seen)
                n_ep += 1
    ce = [(k, v) for k, v in seen.items() if len(v) > 1]
    res.counterexamples = len(ce)
    examples = []
    for (S, ev, cause), v in sorted(ce, key=lambda kv: -len(kv[1]))[:6]:
        examples.append(dict(state=sorted(S), action=ev + (f"({cause})" if cause else ""),
                             outcomes=[sorted(x) for x in v]))
    by_action = {}
    for (S, ev, cause), v in ce:
        by_action[ev] = by_action.get(ev, 0) + 1
    twist = [dict(state=sorted(S), outcomes=[sorted(x) for x in v]) for (S, ev, cause), v in ce if ev == "TwistApart"][:2]
    return dict(episodes=n_ep, **res.__dict__, examples=examples, by_action=by_action, twist_examples=twist,
                axioms={k: v["text"] for k, v in AXIOMS.items()})


def audit_trace_stagewise(plant, res, seen):
    """As audit_trace, with the bruise guard evaluated at the stage of each squeeze."""
    from food import params as PR
    from food.avocado import penetrometer_force
    S = set(plant.trace[0]["facts"])
    trace = plant.trace
    for st in trace[1:]:
        th = dict(bruise_threshold=PR.AVOCADO["bruise_force_ratio"][1] * plant.m.bruise_ratio
                  * float(penetrometer_force(st["stage"])))
        audit_trace([dict(facts=sorted(S))] + [st], th, res, seen)
        S = set(st["facts"])
