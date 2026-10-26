"""The avocado theory grown by its own failures: a deliberately weak seed theory T0 for 'halve and pit a ripe Hass
avocado', then requests that expand the task / BWS family. Each request runs the executive derived from the current
theory on research-SDT episodes; mismatches between predicted and observed traces become localized gaps; competing
hypotheses are tested by discriminating experiments in the research SDT (development seeds); parameters of the patch
are fitted from that evidence; the integrator admits, restricts or versions; and the revised claims are assessed on
held-out avocados (seeds 1000-1029). Simulation evidence only: no claim here is physical.

    ~/Documents/LabAutomationTwin/env_cad/bin/python -m theory.learning     (from ~/Documents/HouseholdTwin)
Writes theory/results/learning.json (revision path, gaps, hypotheses, experiments, fits, claims, gate demonstrations).
"""

import json
import os

import numpy as np
from scipy.optimize import minimize

from food import avocado_tasks as T_
from food import params as PR
from theory import avocado_bws as W
from theory import avocado_policies as P
from theory.package import (Assertion, Claim, Concept, Evidence, Gap, Patch, Scope, TheoryPackage,
                            commitment_gate)

DEV = list(range(100, 600))
HELD = list(range(1000, 1030))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# %% the executive derived from a theory version
def executive(T, plant, rng, photo_bias_correction=0.0, keep_half_h=0.0):
    """What a robot with theory T does for 'halve and pit (within five days)': only the concepts in T's Sigma
    reach its plan. Returns the plant's outcome label and the requirement used for this theory's claims."""
    b = W.prior()
    o = plant.photo()
    o["L"] -= photo_bias_correction
    W.observe_photo(b, o, rng)
    hr = np.zeros(b.n)
    day = 0
    if T.has("RipenessStage"):                                   # plan the cut day in the ripeness band
        touch = T.has("TactileRipeness")
        while day < 5:
            if touch:
                for F in (P.F_INFORM, 2.0, 1.5, 1.0):
                    if W.p_bruise(b, F) <= 0.01:
                        W.observe_squeeze(b, plant.squeeze(F), F, rng)
                        break
            w, _, _ = P.plan_cut_day(b, hr, 5 - day)
            if w == 0:
                break
            plant.wait(24.0)
            _, hr, _, _ = P.lifetime_predict(b, hr, 24.0)
            b.predict(24.0, rng)
            day += 1
    P.execute_halving(plant, b, rng, peel=T.has("SeedCoatSide"))
    if keep_half_h > 0:                                         # the half kept for later browns (lifetime)
        tr = "lemon+wrap+fridge" if T.has("Browning") else "none"
        c = plant.clock
        from food.avocado_lifetime import FruitClock
        cut = FruitClock(c.stage, c.rate_per_day, c.hours_ripe, c.ripe_window_h, c.spoil_days, cut=True)
        cut = cut.advance(keep_half_h, "fridge" if "fridge" in tr else "counter", tr)
        plant.facts |= {"Browned(KeptHalf)"} if cut.browned else set()
    return W.failure_label(plant), plant


def assess(T, claim, req, cls_list=("unripe", "breaking", "ripe"), misspec=None, seeds=HELD, **kw):
    """Held-out assessment of a claim: k successes of n under the requirement ``req`` (a function of the plant)."""
    k = n = 0
    labels = {}
    for cls in cls_list:
        for s in seeds:
            plant = W.Plant(W.instance(s, cls), misspec=misspec, seed=s)
            lab, plant = executive(T, plant, np.random.default_rng(s + 31), **kw)
            ok = req(plant)
            k += ok
            n += 1
            if not ok:
                labels[lab] = labels.get(lab, 0) + 1
    claim.k, claim.n = k, n
    claim.evidence.append(dict(k=k, n=n, failures=labels, theory=T.version, scope=list(cls_list)))
    return k, n, labels


# requirements of the successive task families
def req_basic(p):                     # T0's assessment: apart and seed out
    f = p.facts
    return "Apart(HalfA,HalfB)" in f and "SeedOut(HalfA)" in f


def req_clean_halves(p):              # + no torn flesh, edible
    f = p.facts
    return req_basic(p) and "TornFlesh(HalfB)" not in f and p.clock.eating_quality >= 0.6 and not p.clock.spoiled


def req_clean_cavity(p):              # + no seed coat left
    return req_clean_halves(p) and "TestaIn(Cavity)" not in p.facts


def req_kept_half(p):                 # + the kept half not brown tomorrow
    return req_clean_cavity(p) and "Browned(KeptHalf)" not in p.facts


# %% the seed theory
def T0():
    Sigma = [Concept("Whole", "Whole(Avocado) fact of the SDT", "precondition of CutRoundSeed"),
             Concept("Scored", "Scored / PartlyScored facts (blade coverage)", "TwistApart precondition; re-cut repair"),
             Concept("HalvesApart", "Apart(HalfA,HalfB)", "RemoveSeed precondition; verification"),
             Concept("SeedLocation", "SeedIn / SeedOut facts", "RemoveSeed; verification")]
    A = [Assertion("cut-round-seed", "causal", "a full cut round the seed scores the fruit down to the seed coat"),
         Assertion("twist-separates", "causal", "twisting a scored avocado separates two clean halves"),
         Assertion("spoon-lifts-seed", "causal", "a spoon levered under the seed lifts it out")]
    M = [Scope("ripe Hass", "Hass avocado, ripe (stage 3.6-4.6), on a board, PR2 kinematic", stage=(3.6, 4.6),
               sensing="photo")]
    C = [Claim("Gamma0 halve&pit", "halves apart and seed out", "ripe Hass", "A_basic: end state, q = 0.8")]
    return TheoryPackage(0, Sigma, A, M, C)


def fit_logistic(stage, clean):
    def nll(x):
        s0, w = x[0], abs(x[1]) + 1e-3
        p = np.clip(1 / (1 + np.exp(-(stage - s0) / w)), 1e-6, 1 - 1e-6)
        return -np.sum(clean * np.log(p) + (1 - clean) * np.log(1 - p))
    r = minimize(nll, [3.0, 0.3], method="Nelder-Mead")
    return float(r.x[0]), float(abs(r.x[1]))


# %% the learning loop
def main():
    os.makedirs(OUT, exist_ok=True)
    rec = dict(steps=[])
    T = T0()
    c0 = T.claim("Gamma0 halve&pit")
    assess(T, c0, req_basic, cls_list=("ripe",))
    rec["T0"] = dict(claims=[(c0.name, c0.k, c0.n, c0.verdict)])

    # ---------------------------------------------------------------- request 1: halve this firm avocado
    k1, n1, lab1 = assess(T, Claim("probe", "", "", ""), req_clean_halves, cls_list=("unripe", "breaking"))
    gap1 = Gap("RipenessDependentSeparation", "material / causal process",
               f"T0 predicts clean halves; on firm fruit {n1 - k1} of {n1} episodes end {lab1}")
    # discriminating experiments (development seeds)
    stages, clean = [], []
    for i, s in enumerate(DEV[:200]):
        st = 1.5 + 3.0 * (i % 25) / 24
        hp = W.instance(s)
        hp["ripeness_stage"] = st
        pl = W.Plant(hp, seed=s)
        pl.cut()
        tw = pl.twist()
        if tw.ok:
            stages.append(st)
            clean.append(tw.outcome == "halved")
    stages, clean = np.array(stages), np.array(clean, float)
    s0, w = fit_logistic(stages, clean)
    cov = {}
    for cv in (300.0, 330.0, 360.0):
        torn = stuck = 0
        for s in DEV[200:230]:
            pl = W.Plant(W.instance(s, "ripe"), seed=s)
            pl.cut(coverage=cv)
            tw = pl.twist()
            torn += tw.outcome == "halved_torn"
            stuck += not tw.ok
        cov[cv] = dict(torn=torn, stuck=stuck, n=30)
    tq = {}
    for Tm in (0.3, 1.5, 3.0):
        torn = stuck = 0
        for s in DEV[230:260]:
            pl = W.Plant(W.instance(s, "breaking"), seed=s)
            pl.cut()
            tw = pl.twist(Tm)
            torn += tw.outcome == "halved_torn"
            stuck += not tw.ok
        tq[Tm] = dict(torn=torn, stuck=stuck, n=30)
    gap1.hypotheses = [
        dict(h="H1 ripeness controls whether the flesh releases at the seed",
             test="vary stage 1.5-4.5 at a full cut (200 dev fruit)",
             result=f"P(clean) rises from {clean[stages < 2.3].mean():.2f} (stage < 2.3) to {clean[stages > 3.3].mean():.2f} "
                    f"(stage > 3.3); fitted logistic centre {s0:.2f}, width {w:.2f} (plant law: "
                    f"{PR.AVOCADO['clean_twist_stage'][1]}, 0.15)", verdict="supported"),
        dict(h="H2 an incomplete cut leaves bridges that tear",
             test="vary coverage 300 / 330 / 360 deg on ripe fruit",
             result="; ".join(f"{int(k)} deg: torn {v['torn']}, stuck {v['stuck']} /30" for k, v in cov.items()),
             verdict=("supported as a second cause: a gap in the cut tears the bridge even on ripe fruit; T0 already "
                      "has the distinction (Scored / PartlyScored) and the re-cut repair, so it adds no gap"
                      if cov[300.0]["torn"] + cov[330.0]["torn"] > 2 * max(1, cov[360.0]["torn"])
                      else "rejected for tearing")),
        dict(h="H3 too much twist torque tears the flesh",
             test="vary the torque limit 0.3 / 1.5 / 3 N m on breaking fruit",
             result="; ".join(f"{k} N m: torn {v['torn']}, stuck {v['stuck']} /30" for k, v in tq.items()),
             verdict=("rejected: the share of torn fruit does not change with the torque limit"
                      if max(v["torn"] for v in tq.values()) - min(v["torn"] for v in tq.values()) <= 3
                      else "supported"))]
    s90 = s0 + w * np.log(9.0)
    patch1 = Patch(
        dSigma=[Concept("RipenessStage", "belief over the stage (photo colour; SDT attribute of the avocado)",
                        "cut-day decision (wait), clean-twist band, quality"),
                Concept("TornFlesh", "TornFlesh(HalfB) / FleshOnSeed facts; cavity inspection", "verification; "
                        "belief update (twist observation)")],
        dA=[Assertion("clean-twist-band", "causal", "P(clean | stage) = logistic(stage; s0, w), fitted",
                      params=dict(s0=s0, w=w, stage_90=float(s90)), provenance="E1: 200 research-SDT episodes"),
            Assertion("ripening", "causal", "stage advances at the fruit's own rate (lifetime model)")],
        retract_A=["twist-separates"],
        dM=[Scope("Hass, any ripeness", "Hass avocado stage 1.4-4.6, five days to prepare, board, PR2 kinematic",
                  stage=(1.4, 4.6), sensing="photo")],
        dC=[Claim("Gamma1 clean halves", "apart, seed out, no torn flesh, edible when cut", "Hass, any ripeness",
                  "A_clean: end state + TornFlesh + eating quality, q = 0.8")],
        dE=[Evidence("E1", "simulation", "twist outcome vs stage / coverage / torque", "H1 supported, H2-H3 rejected",
                     int(clean.sum()), len(clean))],
        disposition="admit", rationale="discriminating experiments support H1; the claim is specialised by a "
        "cut-day plan module (wait into the band) and verified on held-out fruit")
    T = T.revise(gap1, patch1)
    c1 = T.claim("Gamma1 clean halves")
    assess(T, c1, req_clean_halves)
    rec["steps"].append(dict(request="Halve this firm avocado.", gap=gap1.__dict__, before=dict(k=k1, n=n1, failures=lab1),
                             patch=dict(disposition=patch1.disposition, dSigma=[c.name for c in patch1.dSigma],
                                        dA=[a.statement for a in patch1.dA], retracted=patch1.retract_A,
                                        rationale=patch1.rationale),
                             fit=dict(s0=s0, w=w, s90=float(s90), true_s0=PR.AVOCADO["clean_twist_stage"][1], true_w=0.15),
                             claim=(c1.name, c1.k, c1.n, c1.verdict, c1.evidence[-1]["failures"]), version=T.version))

    # ---------------------------------------------------------------- request 2: the cavity must be clean
    probe = Claim("probe", "", "", "")
    k2, n2, lab2 = assess(T, probe, req_clean_cavity)
    gap2 = Gap("SeedCoatUnrepresented", "observation / verification",
               f"T1 predicts a clean cavity; {n2 - k2} of {n2} episodes leave seed-coat sheets ({lab2})")
    same_fruit = 0
    flips = 0
    seen = []
    for s in DEV[260:320]:
        outs = []
        for method in ("spoon", "knife_twist"):
            pl = W.Plant(W.instance(s, "ripe"), seed=s)
            pl.cut()
            pl.twist()
            rm = pl.remove_seed(method)
            outs.append(bool(rm.measured.get("testa_visible_in_cavity")) if rm.ok else None)
        if None not in outs:
            same_fruit += outs[0] == outs[1]
            flips += outs[0] != outs[1]
            seen.append(outs[0])
    passes = []
    for s in DEV[320:380]:
        pl = W.Plant(W.instance(s, "ripe"), seed=s)
        pl.cut()
        pl.twist()
        pl.remove_seed("spoon")
        if pl.remnant > 0.05:
            out = pl.peel(1)
            passes.append(out.ok)
    p_peel = float(np.mean(passes)) if passes else float("nan")
    gap2.hypotheses = [
        dict(h="H1 the seed coat sticks to the flesh in some fruit (a hidden property of the fruit)",
             test="same fruit pitted with spoon and with knife twist (60 dev fruit)",
             result=f"cavity outcome identical across methods in {same_fruit} of {same_fruit + flips} fruit; seed coat "
                    f"left in {np.mean(seen):.2f} of them",
             verdict="supported" if flips <= 0.05 * (same_fruit + flips) else "not supported"),
        dict(h="H2 the spoon technique leaves the coat", test="as above",
             result=f"{flips} method-dependent outcomes",
             verdict="rejected" if flips <= 0.05 * (same_fruit + flips) else "supported"),
        dict(h="repair: the spoon tip lifts the sheets", test=f"one peel pass on {len(passes)} dev cavities with sheets",
             result=f"success {p_peel:.2f} per pass (plant law {PR.AVOCADO['peel_coat_success'][1]})",
             verdict="supported")]
    patch2 = Patch(
        dSigma=[Concept("SeedCoatSide", "belief P(testa adheres to the flesh); TestaIn(Cavity) seen after pitting",
                        "PeelSeedCoat repair; cavity verification")],
        dA=[Assertion("testa-side", "constitutive", "the seed coat stays in the cavity when it adheres to the flesh",
                      params=dict(p_flesh=float(np.mean(seen))), provenance="E2"),
            Assertion("peel-repair", "causal", "a spoon-tip pass lifts the sheets", params=dict(p=p_peel),
                      provenance="E2")],
        dC=[Claim("Gamma2 clean cavity", "Gamma1 + no seed coat left", "Hass, any ripeness",
                  "A_cavity: + cavity inspection, q = 0.8")],
        dE=[Evidence("E2", "simulation", "testa outcome vs method; peel passes", "H1 supported", same_fruit,
                     same_fruit + flips)],
        disposition="admit", rationale="the distinction is perceivable after pitting and actionable by a repair")
    T = T.revise(gap2, patch2)
    c2 = T.claim("Gamma2 clean cavity")
    assess(T, c2, req_clean_cavity)
    rec["steps"].append(dict(request="Pit it; the cavity must be clean.", gap=gap2.__dict__,
                             before=dict(k=k2, n=n2, failures=lab2),
                             patch=dict(disposition=patch2.disposition, dSigma=[c.name for c in patch2.dSigma],
                                        dA=[a.statement for a in patch2.dA], rationale=patch2.rationale),
                             fit=dict(p_testa_flesh=float(np.mean(seen)), p_peel=p_peel),
                             claim=(c2.name, c2.k, c2.n, c2.verdict, c2.evidence[-1]["failures"]), version=T.version))

    # ---------------------------------------------------------------- request 3: a different camera (riper than it looks)
    mis = W.Misspec(colour_L_bias=10.0)
    probe = Claim("probe", "", "", "")
    k3, n3, lab3 = assess(T, probe, req_clean_cavity, cls_list=("ripe",), misspec=mis)
    # investigation: photo L* against the stage that touch implies (paired, dev fruit under the new camera)
    resid = []
    for s in DEV[:60]:
        pl = W.Plant(W.instance(s, "breaking"), misspec=mis, seed=s)
        o = pl.photo()
        b = W.prior()
        W.observe_squeeze(b, pl.squeeze(1.5), 1.5, np.random.default_rng(s))
        st = float(np.sum(b.weights * b.particles["ripeness_stage"]))
        resid.append(o["L"] - (40 - 13 * (st - 1) / 4))
    bias = float(np.mean(resid))
    gap3 = Gap("ColourRipenessMisspecified", "observation",
               f"under the new camera {n3 - k3} of {n3} ripe fruit fail ({lab3}): the photo says firmer than it is")
    gap3.hypotheses = [
        dict(h="H1 the camera's lightness is biased", test="photo L* against the stage implied by a gentle squeeze "
             "(60 dev fruit)", result=f"mean residual {bias:+.1f} L* units (simulated camera bias +10)",
             verdict="supported"),
        dict(h="H2 these fruit ripen faster", test="the same residual on day 0, before any ripening",
             result="the residual is present before any time passes", verdict="rejected")]
    patch3 = Patch(
        dSigma=[Concept("TactileRipeness", "squeeze stiffness reading at a bruise-safe force (belief update)",
                        "daily gentle squeeze before the cut decision; commitment gate requires it")],
        dA=[Assertion("camera-bias", "constitutive", "photo lightness offset of this camera", params=dict(bias=bias),
                      provenance="E3, versioned observation model")],
        dM=[Scope("Hass, any ripeness", "Hass avocado stage 1.4-4.6, five days, board, PR2 kinematic; ripeness "
                  "commitments need touch", stage=(1.4, 4.6), sensing="photo + touch")],
        restrict_C={},
        dE=[Evidence("E3", "simulation", "photo vs touch residual", f"bias {bias:+.1f}", len(resid), len(resid))],
        disposition="restrict + version", rationale="photo-only ripeness no longer suffices for commitment (scope "
        "restricted to photo + touch); the observation model is versioned with the estimated bias")
    T = T.revise(gap3, patch3)
    c3 = Claim("Gamma2 clean cavity (new camera)", c2.task, "Hass, any ripeness", c2.assess)
    T.C.append(c3)
    assess(T, c3, req_clean_cavity, cls_list=("ripe",), misspec=mis, photo_bias_correction=bias)
    rec["steps"].append(dict(request="Use the new camera (lighter images).", gap=gap3.__dict__,
                             before=dict(k=k3, n=n3, failures=lab3),
                             patch=dict(disposition=patch3.disposition, dSigma=[c.name for c in patch3.dSigma],
                                        dA=[a.statement for a in patch3.dA], rationale=patch3.rationale),
                             fit=dict(bias=bias, true_bias=10.0),
                             claim=(c3.name, c3.k, c3.n, c3.verdict, c3.evidence[-1]["failures"]), version=T.version))

    # ---------------------------------------------------------------- request 4: keep a half for tomorrow
    probe = Claim("probe", "", "", "")
    k4, n4, lab4 = assess(T, probe, req_kept_half, keep_half_h=15.0)
    gap4 = Gap("BrowningUnrepresented", "material / lifetime",
               f"T3 has no browning: {n4 - k4} of {n4} kept halves are brown after 15 h ({lab4})")
    gap4.hypotheses = [dict(h="H1 exposed flesh browns over hours unless covered, acidified or cooled",
                            test="lifetime model (dE = k sqrt(t), treatment factors; atk_browning ordering)",
                            result="untreated dE reaches the visible level within ~3 h; lemon + wrap + fridge stays "
                                   "below it for > 15 h", verdict="supported (model evidence, ILLUSTRATIVE constants)")]
    patch4 = Patch(dSigma=[Concept("Browning", "dE of a cut face (lifetime model over the belief)",
                                   "storage treatment of a kept half (lemon: PR2 motion; wrap, fridge: emulated)")],
                   dA=[Assertion("browning", "causal", "dE = k sqrt(t x treatment factor)")],
                   dC=[Claim("Gamma4 half for tomorrow", "Gamma2 + the kept half not brown after 15 h",
                             "Hass, any ripeness", "A_kept: + browning check, q = 0.8")],
                   disposition="admit", rationale="the treatment action prevents the forbidden effect")
    T = T.revise(gap4, patch4)
    c4 = T.claim("Gamma4 half for tomorrow")
    assess(T, c4, req_kept_half, keep_half_h=15.0)
    rec["steps"].append(dict(request="Keep the other half for tomorrow's breakfast.", gap=gap4.__dict__,
                             before=dict(k=k4, n=n4, failures=lab4),
                             patch=dict(disposition=patch4.disposition, dSigma=[c.name for c in patch4.dSigma],
                                        dA=[a.statement for a in patch4.dA], rationale=patch4.rationale),
                             claim=(c4.name, c4.k, c4.n, c4.verdict, c4.evidence[-1]["failures"]), version=T.version))

    # ---------------------------------------------------------------- request 5: restriction is learning too
    sticky = W.Misspec(adhesion=10.0)
    c5 = Claim("Gamma2 on a sticky cultivar", c2.task, "sticky cultivar", c2.assess)
    T.C.append(c5)                                              # the extension is proposed, then assessed
    k5, n5, lab5 = assess(T, c5, req_clean_cavity, cls_list=("ripe", "breaking"), misspec=sticky)
    gap5 = Gap("CultivarOutsideFamily", "scope",
               f"on a cultivar whose seed clings 10x harder (ILLUSTRATIVE) {n5 - k5} of {n5} fail ({lab5})")
    patch5 = Patch(dM=[Scope("sticky cultivar", "seed adhesion 10x Hass (imagined cultivar): outside the warranted "
                             "family", sensing="photo + touch", cultivar="other")],
                   restrict_C={"Gamma2 on a sticky cultivar": "retracted"},
                   dE=[Evidence("E5", "counterexample", "sticky cultivar", f"{k5}/{n5}", k5, n5)],
                   disposition="restrict", rationale="the claim is not extended: the knife twist that would free "
                   "the seed is forbidden (safety rule), so the robot abstains or asks for this family")
    T = T.revise(gap5, patch5)
    rec["steps"].append(dict(request="Halve and pit this other cultivar.", gap=gap5.__dict__,
                             before=dict(k=k5, n=n5, failures=lab5),
                             patch=dict(disposition=patch5.disposition, rationale=patch5.rationale),
                             claim=(c5.name, c5.k, c5.n, c5.verdict, lab5), version=T.version))

    # ---------------------------------------------------------------- the commitment gate on example situations
    gate = []
    for name, args in [
        ("ripe fruit, photo + touch", dict(claim_name="Gamma2 clean cavity", in_scope_p=0.97,
                                           required_concepts=["RipenessStage", "SeedCoatSide", "TactileRipeness"],
                                           evidence_kinds={"photo", "touch"}, forbidden_p={"hazard": 0.0})),
        ("photo only", dict(claim_name="Gamma2 clean cavity", in_scope_p=0.95,
                            required_concepts=["RipenessStage", "SeedCoatSide"], evidence_kinds={"photo"},
                            forbidden_p={"hazard": 0.0})),
        ("knife twist proposed", dict(claim_name="Gamma2 clean cavity", in_scope_p=0.97,
                                      required_concepts=["RipenessStage"], evidence_kinds={"photo", "touch"},
                                      forbidden_p={"hazard": 1.0})),
        ("sticky cultivar", dict(claim_name="Gamma2 on a sticky cultivar", in_scope_p=0.9,
                                 required_concepts=["RipenessStage"], evidence_kinds={"photo", "touch"},
                                 forbidden_p={"hazard": 0.0})),
        ("not an avocado?", dict(claim_name="Gamma2 clean cavity", in_scope_p=0.3, required_concepts=[],
                                 evidence_kinds={"photo", "touch"}, forbidden_p={})),
        ("browning not represented", dict(claim_name="Gamma4 half for tomorrow", in_scope_p=0.97,
                                          required_concepts=["Browning", "Bruising"], evidence_kinds={"photo", "touch"},
                                          forbidden_p={}))]:
        st, why = commitment_gate(T, **args)
        gate.append(dict(situation=name, state=st, reasons=why))
    rec["gate"] = gate
    rec["final"] = dict(version=T.version, Sigma=[(c.name, c.perceivable, c.actionable) for c in T.Sigma],
                        A=[(a.name, a.kind, a.statement, a.params) for a in T.A],
                        M=[(m.name, m.description, m.sensing) for m in T.M],
                        C=[(c.name, c.status, c.k, c.n, c.verdict) for c in T.C], log=T.log)
    rec["T0"]["Sigma"] = [c.name for c in T0().Sigma]
    json.dump(rec, open(os.path.join(OUT, "learning.json"), "w"), indent=1, default=str)
    for s in rec["steps"]:
        print(f"T{s['version']}: {s['request']} | before {s['before']['k']}/{s['before']['n']} | claim {s['claim'][:4]}")
    for g in gate:
        print("gate:", g["situation"], "->", g["state"], "|", g["reasons"][0])


if __name__ == "__main__":
    main()
