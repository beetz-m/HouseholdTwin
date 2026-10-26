"""Report section: from a reasoning robot to an acting agent and a theory-building robot -- the goal-directed avocado
agent (effects to enforce and to prevent, lifetime-based decisions), the avocado manipulation microtheory, theory
construction from failures and programmable reinforcement learning. Called from build_report.py; the numbers come
from cram/avocado_agent.py (cached) and theory/results/*.json; the figures are those of
report/build_avocado_microtheory.py (run it first).
"""

import json
import os
import re
import sys


def _agent_missions(B):
    """The agent's decisions for the three missions, on the node after the head camera's photo (as in the PR2 demo)."""
    def run():
        sys.path.insert(0, B.R("cram"))
        import avocado_agent as AG
        import avocado_sdt as SDT
        node = SDT.AvocadoSDT.on_detection("avocado_1")
        img = B.R("cram", "camera", "avocado_4_counter.png")
        obs, _ = SDT.photo_observations(img, node.belief.time_h)
        node.perceive(obs)
        out = dict(belief=SDT.summary_line(node.belief), missions=[])
        for m in AG.MISSIONS:
            ag = AG.AvocadoAgent(node, AG.mission(m))
            best, dec = ag.decide()
            out["missions"].append(dict(
                name=m, label=dec.mission, chosen=dec.chosen, why=dec.why, risks=dec.risks, lifetime=dec.lifetime,
                n=len(ag.results), feasible=sum(r.feasible for r in ag.results),
                options=[dict(plan=r.plan, feasible=r.feasible, utility=r.utility, violations=r.violations,
                              P=r._P, prep=r.option.prepare_at_h, store=r.option.keep_whole_in,
                              removal=r.option.removal, peel=r.option.peel_coat, kept=r.option.kept_treatment)
                         for r in ag.results]))
        return out
    return B.cached("avocado_agent_missions", run)


def write(w_, B):
    def w(s):
        w_(re.sub(r"(?<!\\)Gamma(\d)", r"Γ\1", s))
    esc, fig, table = B.esc, B.fig, B.table
    RES = B.R("theory", "results")
    if not os.path.exists(os.path.join(RES, "prl.json")):
        return
    J = lambda n: json.load(open(os.path.join(RES, n)))  # noqa: E731
    r, L, Rl, AX = J("microtheory.json"), J("learning.json"), J("prl.json"), J("axiom_audit.json")
    ag = _agent_missions(B)
    sys.path.insert(0, B.R("cram"))
    import avocado_agent as AG
    from food.avocado_lifetime import TREATMENT
    FIGD = B.FIG

    def cell(rows, cls, pol, mis="well specified"):
        return next((x for x in rows if x["cls"] == cls and x["policy"] == pol and x["mis"] == mis), None)

    def tot(rows, pol):
        return sum(cell(rows, c, pol)["ok"] for c in ("unripe", "breaking", "ripe"))

    mA = {}
    for m in r["mcnemar_A"]:
        mA.setdefault((m["b"], tuple(m["cell"])), m)
    mB = {}
    for m in r["mcnemar_B"]:
        mB.setdefault((m["b"], tuple(m["cell"])), m)

    def pair(d, b):
        m = d[(b, ("pooled",))]
        return f"{m['a_only']}:{m['b_only']}"

    w(r"""\clearpage\section[From reasoning to acting, and from acting to theories]{From reasoning to acting, and from
acting to theories}\label{sec:avocado-theory}
Section~\ref{sec:avocado-reasoning} gave the robot a belief about one avocado and the means to ask questions about it.
This section takes three further steps. First, the robot becomes an \emph{agent} that knows which effects its task
must bring about and which it must not cause, predicts both over the fruit's lifetime, and chooses, executes and
audits plans accordingly. Second, the knowledge it acts on is written down as a \emph{microtheory} -- the egg
microtheory of the AICOR working report transferred to the avocado -- and tested like a scientific theory against
held-out simulated fruit, with ablations and misspecified plants. Third, the robot \emph{learns} that theory: it
grows from a deliberately weak seed by its own failures, and reinforcement-learning problems are generated from its
successive versions. The full microtheory is a separate document, \texttt{report/avocado\_microtheory.pdf}
(15 pages); this section summarises it. Everything is simulation evidence on partly ILLUSTRATIVE laws.
""")

    # ------------------------------------------------------------------ 1 agent
    m = {x["name"]: x for x in ag["missions"]}
    goals, guards = AG._standard_bounds()
    rowsE = [[esc(b.effect), "enforce", f"$\\ge {b.p:.2f}$", esc(AG.EFFECTS[b.effect][1])] for b in goals] + \
            [[esc(b.effect), "prevent" + (" (safety rule)" if b.hard else ""), f"$\\le {b.p:.2f}$",
              esc(AG.EFFECTS[b.effect][1])] for b in guards]
    w(r"""\subsection{An agent that enforces and prevents effects}
\texttt{cram/avocado\_agent.py} turns the node's knowledge into decisions. A \emph{mission} names what is to be
served when (guacamole now; half now and half for tomorrow's breakfast; guacamole in two days) and bounds every
effect: a goal must hold with at least its probability, a prevented effect may occur with at most its probability,
and a bound of zero is a safety rule that no utility can trade against (Table~\ref{tab:effects}). The plan space
combines when to cut (now or at the serving time), where the whole fruit waits (counter or fridge), how the seed comes
out (spoon or knife twist), whether seed-coat sheets are lifted, and how a kept half is stored (six treatments from
nothing to vacuum-packed in the fridge). For each plan the agent runs mental simulations over 80 avocados drawn from
its belief: the lifetime model (\texttt{food/avocado\_lifetime.py}: ripening at the fruit's rate with a $Q_{10}$ of
2.5 between counter and fridge, a ripe window of 24--48~h, decline, spoilage, browning of a cut face as
$\mathrm{d}E=k\sqrt{t\,f_\text{treatment}}$) advances each one to the cut, the processing mechanics halve and pit it
with the repairs the agent would make, and the servings are scored. A geometric check (\texttt{plan\_hold}) places
the holding gripper at least 10~mm from the blade's plane, pinching the fruit's top or, if that fails, pressing it
from above.
""")
    w(table(["l", "l", "l", "X"], ["Effect", "Kind", "Bound", "Meaning"], rowsE,
            "Effects and bounds of the standard missions (cram/avocado\\_agent.py).\\label{tab:effects}",
            size="scriptsize"))
    rowsM = [[esc(x["label"]), esc(x["chosen"]), f"{x['feasible']}/{x['n']}", esc(x["why"].split(". Rejected")[0])]
             for x in ag["missions"]]
    w(table(["p{0.17\\linewidth}", "p{0.3\\linewidth}", "l", "X"], ["Mission", "Chosen plan", "feasible",
                                                                     "Why"], rowsM,
            "The agent's decisions for the avocado the PR2 sees on the counter (belief after the head-camera photo: " +
            esc(ag["belief"]) + "). Every knife-twist plan is rejected by the safety rule (P(hazard) = 1).",
            size="scriptsize"))
    half = m["half_now_half_tomorrow"]
    rowsT = []
    for t in TREATMENT:
        o = next((x for x in half["options"] if x["prep"] == 0.0 and x["removal"] == "spoon" and x["peel"]
                  and x["kept"] == t), None)
        if o:
            rowsT.append([esc(t), f"{TREATMENT[t]:.2f}", f"{o['P']['browned']:.2f}", f"{o['utility']:.2f}",
                          "yes" if o["feasible"] else esc(o["violations"])])
    w(r"""\paragraph{Lifetime-based decisions} The same fruit leads to different plans for different missions. Its
lifetime under the belief: """ + esc(m["guacamole_now"]["lifetime"]) + r""". For guacamole in two days the agent keeps
the fruit whole on the counter and cuts at the serving time (expected quality higher than cutting now, because the
fruit ripens into its best window). For half now and half tomorrow the kept half is what decides: only treatments
that slow browning enough keep its cut face fresh after 15~hours (Table~\ref{tab:keep}).
""")
    w(table(["l", "l", "l", "l", "X"], ["Kept half stored as", "browning factor", "P(browned)", "quality",
                                        "Feasible / violated bound"], rowsT,
            "Half now, half tomorrow: the plans that cut now, use the spoon and lift the seed coat, by storage of the "
            "kept half.\\label{tab:keep}", size="scriptsize"))
    w(r"""\paragraph{Execution with a goal watch and an effect audit} In the CRAMERA kitchen
(\texttt{cram/avocado\_kitchen.py}) a mission is a command: the agent decides, explains the choice and the rejected
alternatives, and the PR2 executes the plan with the motions of Section~\ref{sec:pr2} (with new ones for holding,
lifting the seed coat with the spoon tip and squeezing lemon on the kept half). After every step a goal watch compares
the observed effects with the predicted ones and reassesses the remaining risk (e.g.\ a torn twist raises the
gouge risk of the spoon); after the mission an effect audit lists each goal and prevented effect as achieved,
violated or still pending (the kept half until breakfast). Advancing the kitchen's clock ripens, spoils and browns
the hidden fruit, so a decision to wait can be checked against what happens.
""")

    # ------------------------------------------------------------------ 2 microtheory
    th, ol, po = (mA[(b, ("pooled",))] for b in ("firm thumb 10 N", "open-loop prior plan", "photo only"))
    rows = [
        ["A: assess ripeness by touch, intact", "closed-loop squeeze under a 1 \\% bruise chance constraint",
         f"{tot(r['A'], 'closed loop')}/90", f"thumb 10 N {pair(mA, 'firm thumb 10 N')}; photo only "
         f"{pair(mA, 'photo only')}; open-loop prior plan {pair(mA, 'open-loop prior plan')}"],
        ["B: halve and pit within 5 days", "belief band: daily squeeze, cut on the planned day, repairs, seed-coat "
         "lifting", f"{tot(r['B'], 'belief band')}/90", f"cut now {pair(mB, 'cut now')}; wait 2 d "
         f"{pair(mB, 'wait 2 days')}; wait 4 d {pair(mB, 'wait 4 days')}; photo-only band "
         f"{pair(mB, 'photo-only band')}"],
        ["B, ablations", "remove one distinction from the robot's model", "--",
         f"no seed-coat side {pair(mB, 'no testa side')}; ready / not ready {pair(mB, 'ready / not ready')}; no repair "
         f"{pair(mB, 'no repair')}; population ripening rate {pair(mB, 'population ripening rate')}"]]
    w(r"""\subsection{An avocado manipulation microtheory}
The egg microtheory distinguishes grasping an egg intact from cracking it open: the same fracture is a failure in one
task and the goal in the other. For the avocado the event is the separation of the flesh: forbidden while the fruit
is squeezed to judge its ripeness (Task A), required in the cut plane and at the twist when it is halved and pitted
(Task B), and forbidden everywhere else (bruise, torn cavity wall, gouge). The microtheory
(\texttt{theory/}) has the egg's parts: a research SDT whose hidden parameters are drawn once per avocado and whose
events are patches over relational facts; reduced-order laws for squeezing, bruising, cutting, twisting, seed removal,
peeling, ripening and browning; a probabilistic hybrid model whose jump probabilities are marginals over the hidden
parameters; an observation model and a particle belief; and phase-indexed requirements over the patch trace. Its two
central predictions are a squeeze \emph{corridor} between an informative and a bruising force, which is empty for ripe
fruit, and a ripeness \emph{band in time} for the cut, between torn flesh and overripeness
(Figure~\ref{fig:mt}).
""")
    w(B.figrow([(os.path.join(FIGD, "avo_mt_corridor.pdf"), "Task A: squeeze corridor (model lines, simulation "
                 "markers; green: safe and informative)"),
                (os.path.join(FIGD, "avo_mt_band.pdf"), "Task B: success by day of the cut (simulated, model dashed)")],
               "The two central predictions of the avocado microtheory, checked against held-out simulated avocados "
               "(details in avocado\\_microtheory.pdf).\\label{fig:mt}", r"0.2\textheight"))
    w(table(["p{0.18\\linewidth}", "p{0.3\\linewidth}", "l", "X"], ["Task", "Policy", "success",
                                                                     "Paired against (only policy : only baseline)"],
            [[esc(a), esc(b), c, esc(d)] for a, b, c, d in rows],
            "Held-out evaluation: the same 90 avocados (30 unripe, 30 breaking, 30 ripe) for every policy; discordant "
            "pairs and exact McNemar tests in the microtheory document.", size="scriptsize"))
    wf = cell(r["A"], "ripe", "closed loop", "bruise_ratio x0.6")
    w(r"""\paragraph{What it shows} The belief regulates better than fixed policies (the thumb press bruises every ripe
fruit; cutting at once tears unripe fruit; waiting four days lets ripe fruit go past its best). Only one removed
distinction is load-bearing on these avocados -- which side the seed coat sticks to; the others make no detectable
difference, which needs a discriminating test, not a conclusion. An error that makes the flesh weaker than believed
cannot be corrected in time (the closed loop then succeeds on """ + f"{wf['ok']}/30" + r""" ripe fruit: the bruise is
the first evidence), while a camera bias is corrected by touch. A replay audit of """ + f"{AX['episodes']:,}" +
      r""" episodes (""" + f"{AX['steps']:,}" + r""" steps) finds no mismatch between the relational action axioms and the
simulator's patches; """ + f"{AX['counterexamples']}" + r""" pairs of steps from the same relational state reach different
branches -- the hidden attributes (ripeness, the seed-coat side) decide, so a proof needs them.
""")

    # ------------------------------------------------------------------ 3 theory construction
    st = L["steps"]
    rowsL = [["T0", esc("halve and pit a ripe Hass"), "--", "--",
              esc(f"{L['T0']['claims'][0][1]}/{L['T0']['claims'][0][2]} {L['T0']['claims'][0][3]}")]]
    for s in st:
        rowsL.append([f"T{s['version']}", esc(s["request"]), esc(f"{s['gap']['name']}: {s['before']['k']}/"
                                                                   f"{s['before']['n']}"),
                      esc(s["patch"]["disposition"] + ("; +" + ", ".join(s["patch"].get("dSigma", []))
                                                       if s["patch"].get("dSigma") else "")),
                      esc(f"{s['claim'][1]}/{s['claim'][2]} {s['claim'][3]}")])
    f1, f2, f3 = st[0]["fit"], st[1]["fit"], st[2]["fit"]
    w(r"""\subsection{Learning the theory from failures}
Following \emph{Manipulation as Theory Construction}, the robot's competence is a theory
$\mathcal{T}_k=\langle\Sigma_k,A_k,M_k,C_k,E_k\rangle$: concepts that are perceivable through the SDT, actionable through
designators and traceable; assertions; body-world-system families (scopes); scoped claims
$\Gamma=\mathrm{Competent}(F_\text{task},F_\text{BWS},A_\text{assess})$; and evidence (\texttt{theory/package.py}). The
seed $\mathcal{T}_0$ knows only how to halve and pit a ripe Hass avocado. Five requests expand the task or the family;
each runs the executive derived from the current theory, the failures become a localized gap, competing hypotheses are
tested by discriminating experiments in the research SDT on development avocados, a patch is fitted, and the
integrator admits, restricts or versions it; the revised claim is assessed on the held-out avocados
(\texttt{theory/learning.py}, Table~\ref{tab:revision}).
""")
    w(table(["l", "p{0.25\\linewidth}", "p{0.26\\linewidth}", "p{0.2\\linewidth}", "X"],
            ["", "Request", "Gap: theory before", "Patch", "Held-out after"], rowsL,
            "The revision path of the avocado theory (simulation evidence).\\label{tab:revision}", size="scriptsize"))
    w(r"""The patches recover the plant's laws from the experiments: the clean-release law's centre at stage """ +
      f"{f1['s0']:.2f}" + r""" (plant """ + f"{f1['true_s0']}" + r"""; its width is overestimated on a coarse grid), the share of
fruit whose seed coat sticks to the flesh """ + f"{f2['p_testa_flesh']:.2f}" + r""", the peel repair's success
""" + f"{f2['p_peel']:.2f}" + r""" per pass (plant 0.85) and the new camera's lightness bias """ + f"{f3['bias']:+.1f}" +
      r""" (plant +10). The last request -- a cultivar whose seed clings ten times harder -- is answered by
\emph{restriction}: the only remedy, the knife twist, is forbidden, so the claim is retracted for that family; the
evidence grows while the warranted scope contracts. At runtime a \emph{commitment gate} separates warranted
commitments from uncertain, unsupported, counterevidenced and out-of-scope ones:
""")
    w(table(["p{0.3\\linewidth}", "l", "X"], ["Situation", "State", "Reason"],
            [[esc(g["situation"]), esc(g["state"]), esc(g["reasons"][0])] for g in L["gate"]],
            "The commitment gate of the final theory.", size="scriptsize"))

    # ------------------------------------------------------------------ 4 programmable RL
    rowsR = [[esc(p["name"]), f"{sum(v['ok'] for v in p['eval'].values())}/90",
              str(sum(v["hazard"] for v in p["eval"].values()))] for p in Rl["programs"]]
    rowsR.append([esc("hand-coded executive of the same theory (belief band)"), f"{tot(r['B'], 'belief band')}/90", "0"])
    w(r"""\subsection{Reinforcement learning programmed by the theory}
Reinforcement learning lives inside the theory (\texttt{theory/prl.py}): the learning problem is generated from a
theory version. Its state features are the theory's perceivable concepts (believed stage and spread, day, a torn
twist, the seed coat seen), its actions the designators' parameters (cut or wait, spoon or knife twist, peel passes
only if the theory has the seed-coat concept), its reward the claim's KPIs, and its hard constraints the forbidden
effects, as a shield that masks every action whose forbidden effect the theory predicts. A tabular Monte-Carlo learner
trains on 5,600 research-SDT episodes and is evaluated greedily on the held-out avocados (Table~\ref{tab:prl}).
""")
    w(table(["X", "l", "l"], ["Program / policy", "success", "hazards"], rowsR,
            "Programmable RL on the 90 held-out avocados of Task B.\\label{tab:prl}", size="scriptsize"))
    w(fig(os.path.join(FIGD, "avo_mt_rl.pdf"), "Left: success during training. Right: the cut / wait rule learned from "
          "$\\mathcal{T}_3$ with the shield, by believed stage and day -- the ripeness band, learned.",
          r"0.2\textheight"))
    w(r"""The learner inherits the theory's distinctions and its safety rules: from $\mathcal{T}_3$ with the shield it
learns the band and matches the hand-coded executive; from $\mathcal{T}_1$, which has no seed-coat concept, no amount
of training cleans a cavity whose dirt it cannot represent; and with the hazard as a small penalty instead of a shield
it buys the forbidden knife strike in every episode. What the theory does not represent the learner cannot learn, and
what the theory forbids a reward must not be allowed to buy.

\paragraph{What this section does not show} No physical validation: the plant and the robot share the laws except
where a misspecification is injected; contact is quasi-static mechanics, not contact simulation; the photo in the
microtheory is a symbolic observation from the colour law; the learning loop's agents are scripted functions, and the
RL is tabular on a handful of features. The 30 held-out avocados per class are reused across tables (paired, not
independent).
""")
