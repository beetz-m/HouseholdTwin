"""An Avocado Manipulation Microtheory -- the egg microtheory transferred to the avocado, extended by theory
construction and programmable reinforcement learning. Built from the result files of theory/ (run them first):

    E=~/Documents/LabAutomationTwin/env_cad/bin/python
    $E -m theory.run_microtheory && $E -m theory.learning && $E -m theory.prl
    $E -c "import json; from theory import axioms as AX; json.dump(AX.run_audit(), open('theory/results/axiom_audit.json','w'))"
    $E report/build_avocado_microtheory.py          # -> report/avocado_microtheory.pdf
"""

import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import build_report as B  # noqa: E402  (LaTeX and figure helpers)
from theory import avocado_bws as W  # noqa: E402
from theory import avocado_policies as P  # noqa: E402

TEX = os.path.join(HERE, "avocado_microtheory.tex")
RES = os.path.join(ROOT, "theory", "results")
esc, table, fig, figrow, tt = B.esc, B.table, B.fig, B.figrow, B.tt


def J(name):
    return json.load(open(os.path.join(RES, name)))


def cell(rows, cls, policy, mis="well specified"):
    return next((x for x in rows if x["cls"] == cls and x["policy"] == policy and x["mis"] == mis), None)


def fmt(x):
    if x is None:
        return "--"
    s = f"{x['ok']}/{x['n']}"
    if x["ok"] < x["n"]:
        s += f" [{100 * x['lo']:.0f},{100 * x['hi']:.0f}]"
    return s


def pf(p):
    """A p value in LaTeX math."""
    if p >= 0.001:
        return f"{p:.2g}"
    m, e = f"{p:.0e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def fails(x, k=3):
    if not x or not x["failures"]:
        return ""
    return ", ".join(f"{v} {k_}" for k_, v in sorted(x["failures"].items(), key=lambda kv: -kv[1])[:k])


# %% figures
def fig_corridor(r):
    plt = B.mpl()
    F = r["corridor"]["forces"]
    f, axs = plt.subplots(1, 3, figsize=(7.4, 2.6), sharey=True)
    for ax, cls in zip(axs, ("unripe", "breaking", "ripe")):
        an = r["corridor"]["analytic"][cls]
        sim = r["corridor"]["simulated"][cls]
        pb = np.array(an["p_bruise"])
        pi = np.array(an["p_inform"])
        ok = (pb <= 0.01) & (pi >= 0.75)
        Fa = np.array(F)
        if ok.any():
            ax.axvspan(Fa[ok].min(), Fa[ok].max(), color=B.PAL[2], alpha=0.15, lw=0)
        ax.plot(F, pb, color=B.PAL[7], label="P(bruise) model")
        ax.plot(F, pi, color=B.PAL[0], label="P(informative) model")
        xs = sorted(sim, key=float)
        ax.plot([float(x) for x in xs], [sim[x]["p_bruise"] for x in xs], "s", ms=3.5, mfc="none", color=B.PAL[7],
                label="bruised, simulated")
        ax.plot([float(x) for x in xs], [sim[x]["p_inform"] for x in xs], "o", ms=3, color=B.PAL[0],
                label="informative, simulated")
        ax.set_xscale("log")
        ax.set_title(f"{cls} (stage {W.CLASSES[cls][0]}-{W.CLASSES[cls][1]})", fontsize=8)
        ax.set_xlabel("squeeze force [N]")
    axs[0].set_ylabel("probability")
    h, l = axs[0].get_legend_handles_labels()
    f.legend(h, l, fontsize=6.5, loc="lower center", ncol=4, frameon=False)
    f.tight_layout(rect=(0, 0.08, 1, 1))
    out = os.path.join(B.FIG, "avo_mt_corridor.pdf")
    f.savefig(out)
    plt.close(f)
    return out


def fig_band(r):
    plt = B.mpl()
    f, ax = plt.subplots(figsize=(7.0, 2.7))
    for i, cls in enumerate(("unripe", "breaking", "ripe")):
        xs = [x for x in r["band"] if x["cls"] == cls]
        xs.sort(key=lambda x: int(x["policy"].split()[1]))
        d = [int(x["policy"].split()[1]) for x in xs]
        ax.plot(d, [x["ok"] / x["n"] for x in xs], "o-", color=B.PAL[i], label=f"{cls}: simulated (30 each)")
        # model prediction over the class population (the robot's model with the true instance parameters)
        m = []
        for day in range(6):
            ok = []
            for s in range(1000, 1030):
                pl = W.Plant(W.instance(s, cls), seed=s)
                c = pl.clock.advance(24.0 * day)
                ok.append(float(P.p_clean(c.stage)) * (c.eating_quality >= 0.6) * (not c.spoiled))
            m.append(np.mean(ok))
        ax.plot(range(6), m, "--", color=B.PAL[i], lw=1.2, label=f"{cls}: model")
    ax.set_xlabel("day of the cut")
    ax.set_ylabel("P(success) in Task B")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=6.5, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    f.tight_layout()
    out = os.path.join(B.FIG, "avo_mt_band.pdf")
    f.savefig(out)
    plt.close(f)
    return out


def fig_belief_trace():
    """One unripe avocado under the belief-band policy: belief over the stage per day against the hidden truth."""
    plt = B.mpl()
    s, cls = 1003, "unripe"
    plant = W.Plant(W.instance(s, cls), seed=s)
    rng = np.random.default_rng(s + 31)
    b = W.prior()
    W.observe_photo(b, plant.photo(), rng)
    hr = np.zeros(b.n)
    rows = []

    def snap(label):
        m = float(np.sum(b.weights * b.particles["ripeness_stage"]))
        sd = float(np.sqrt(np.sum(b.weights * (b.particles["ripeness_stage"] - m) ** 2)))
        rows.append((plant.t / 24.0, m, sd, plant.stage, label))
    snap("photo")
    day = 0
    plan = []
    while day < 5:
        for F in (P.F_INFORM, 2.0, 1.5, 1.0):
            if W.p_bruise(b, F) <= 0.01:
                W.observe_squeeze(b, plant.squeeze(F), F, rng)
                snap(f"squeeze {F:.1f} N")
                break
        w, pw, ps = P.plan_cut_day(b, hr, 5 - day)
        plan.append((day, [round(p, 2) for p in ps]))
        if w == 0:
            break
        plant.wait(24.0)
        _, hr, _, _ = P.lifetime_predict(b, hr, 24.0)
        b.predict(24.0, rng)
        day += 1
        snap("wait 1 d")
    P.execute_halving(plant, b, rng)
    snap("after halving")
    t = [x[0] for x in rows]
    f, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.fill_between(t, [x[1] - x[2] for x in rows], [x[1] + x[2] for x in rows], color=B.PAL[0], alpha=0.18, lw=0)
    ax.plot(t, [x[1] for x in rows], "o-", color=B.PAL[0], label="belief mean $\\pm$ sd")
    ax.plot(t, [x[3] for x in rows], "--", color=B.PAL[1], label="hidden stage")
    ax.axhspan(3.3, 5.0, color=B.PAL[2], alpha=0.08, lw=0)
    ax.text(0.02, 3.4, "clean twist and edible", fontsize=7, color=B.PAL[2], transform=ax.get_yaxis_transform())
    for x in rows:
        if x[4].startswith("squeeze") or x[4] == "after halving":
            ax.annotate(x[4], (x[0], x[1]), xytext=(2, 8), textcoords="offset points", fontsize=6, color=B.INK2)
    ax.set_xlabel("days")
    ax.set_ylabel("ripeness stage")
    ax.legend(fontsize=7, loc="lower right")
    f.tight_layout()
    out = os.path.join(B.FIG, "avo_mt_trace.pdf")
    f.savefig(out)
    plt.close(f)
    return out, rows, plan, W.failure_label(plant)


def fig_revision(L):
    plt = B.mpl()
    from matplotlib.patches import FancyBboxPatch
    f, ax = plt.subplots(figsize=(7.4, 1.9))
    ax.axis("off")
    c0 = L["T0"]["claims"][0]
    steps = [("T0", "seed theory", f"{c0[1]}/{c0[2]} {c0[3]}")] + [
        (f"T{s['version']}", s["patch"]["disposition"], f"{s['claim'][1]}/{s['claim'][2]} {s['claim'][3]}")
        for s in L["steps"]]
    labels = ["", "ripeness band", "seed-coat side", "new camera:\ntouch + bias", "browning", "sticky\ncultivar"]
    for i, (n, d, c) in enumerate(steps):
        x = 0.1 + i * 1.6
        bad = "refuted" in c
        ax.add_patch(FancyBboxPatch((x, 0.1), 1.1, 0.8, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="#f7ece6" if bad else "#eef4f2", ec="#a0522d" if bad else "#2a6f62"))
        ax.text(x + 0.55, 0.66, n, ha="center", fontsize=9, weight="bold", color="#a0522d" if bad else "#1d4d44")
        ax.text(x + 0.55, 0.42, d, ha="center", fontsize=6.5)
        ax.text(x + 0.55, 0.2, c, ha="center", fontsize=6.0, color=B.INK2)
        if i:
            ax.annotate("", (x - 0.02, 0.5), (x - 0.48, 0.5), arrowprops=dict(arrowstyle="-|>", color=B.INK2, lw=0.8))
            ax.text(x - 0.25, 1.02, labels[i], fontsize=6, ha="center", va="bottom", color=B.PAL[0])
    ax.set_xlim(0, 9.7)
    ax.set_ylim(0, 1.35)
    f.tight_layout()
    out = os.path.join(B.FIG, "avo_mt_revision.pdf")
    f.savefig(out)
    plt.close(f)
    return out


def fig_rl(R):
    plt = B.mpl()
    f, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.7), gridspec_kw=dict(width_ratios=[1, 1.15]))
    for i, p in enumerate(R["programs"]):
        a1.plot([c["round"] for c in p["curve"]], [c["success"] for c in p["curve"]], "o-", color=B.PAL[i], ms=3,
                label=p["name"].replace("RL from ", "").replace(", hazard as a soft penalty 0.05", ", soft hazard 0.05"))
    a1.set_xlabel("training round (700 episodes each)")
    a1.set_ylabel("success in training")
    a1.legend(fontsize=5.8, loc="center right")
    p = R["programs"][1]
    grid = np.full((7, 5), np.nan)
    for x in p["policy"]:
        st = x["state"]
        if st[0] == "day" and len(st) >= 4 and st[3] == "0" and x["visits"] >= 20:
            grid[int(st[2]), int(st[1])] = 1.0 if x["best"] == "cut" else 0.0
    a2.imshow(grid, origin="lower", cmap="RdYlGn", vmin=-0.2, vmax=1.2, aspect="auto")
    for (i, j), v in np.ndenumerate(grid):
        if not np.isnan(v):
            a2.text(j, i, "cut" if v else "wait", ha="center", va="center", fontsize=6)
    a2.set_xlabel("day")
    a2.set_yticks(range(7), ["<2.25", "2.25-2.75", "2.75-3.1", "3.1-3.4", "3.4-3.75", "3.75-4.25", ">4.25"],
                  fontsize=6)
    a2.set_ylabel("believed stage (sd < 0.45)", fontsize=7)
    a2.set_title("learned cut / wait (T3, shielded)", fontsize=8)
    a2.grid(False)
    f.tight_layout()
    out = os.path.join(B.FIG, "avo_mt_rl.pdf")
    f.savefig(out)
    plt.close(f)
    return out


# %% the document
def main():
    t0 = time.time()
    r = J("microtheory.json")
    L = J("learning.json")
    R = J("prl.json")
    AXA = J("axiom_audit.json")
    A, Bt = r["A"], r["B"]
    f_cor, f_band, f_rev, f_rl = fig_corridor(r), fig_band(r), fig_revision(L), fig_rl(R)
    f_tr, trace_rows, trace_plan, trace_label = fig_belief_trace()
    out = []
    w = out.append
    w(r"""\documentclass[10.5pt,a4paper]{article}
\usepackage[margin=2.2cm,headheight=14pt]{geometry}
\usepackage{fontspec,unicode-math}
\setmainfont{Libertinus Serif}\setsansfont{IBM Plex Sans}\setmonofont[Scale=0.84]{IBM Plex Mono}\setmathfont{Libertinus Math}
\usepackage{microtype,graphicx,booktabs,tabularx,longtable,xcolor,fvextra,caption,fancyhdr,titlesec,enumitem,array,placeins,amsmath}
\usepackage{xurl}
\usepackage[colorlinks,linkcolor=teal!60!black,urlcolor=teal!60!black,citecolor=teal!60!black]{hyperref}
\captionsetup{font=small,labelfont=bf}
\renewcommand{\tabularxcolumn}[1]{>{\raggedright\arraybackslash}p{#1}}
\setlength{\emergencystretch}{3em}\setlist{itemsep=2pt,topsep=3pt}
\titleformat{\section}{\sffamily\Large\bfseries\color{teal!50!black}}{\thesection}{0.8em}{}
\titleformat{\subsection}{\sffamily\large\bfseries}{\thesubsection}{0.7em}{}
\titleformat{\paragraph}[runin]{\sffamily\bfseries}{}{}{}[.]
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\sffamily\footnotesize An avocado manipulation microtheory}
\fancyhead[R]{\sffamily\footnotesize\nouppercase{\leftmark}}\fancyfoot[C]{\sffamily\footnotesize\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\renewcommand{\floatpagefraction}{0.7}\renewcommand{\topfraction}{0.9}\renewcommand{\textfraction}{0.1}
\newcommand{\holds}{\mathrm{holds}}\newcommand{\occurs}{\mathrm{occurs}}\newcommand{\meets}{\mathrm{meets}}
\begin{document}
\begin{center}
{\sffamily\LARGE\bfseries An Avocado Manipulation Microtheory\par}\vspace{4pt}
{\large Relational action axioms, a probabilistic hybrid model over an executable semantic digital twin,\\
theory construction from failures, and reinforcement learning programmed by the theory\par}\vspace{6pt}
{\sffamily\small HouseholdTwin working report \quad$\cdot$\quad transferred from the egg microtheory \quad$\cdot$\quad 20 October 2026\par}
\end{center}
\begin{abstract}\noindent
An avocado must stay intact while it is squeezed to judge its ripeness, yet its flesh must be separated on purpose
when it is halved and pitted -- in the cut plane, at the cut phase, and nowhere else. The same physical event, the
separation of the flesh, is a failure in one task and the goal in another. We transfer the egg microtheory to this
fruit: the episode is a probabilistic hybrid action model over a semantic digital twin (SDT) whose events are
graph patches; reduced-order laws for squeezing, bruising, cutting round the seed, twisting the halves apart, removing
the seed, peeling the seed coat, ripening and browning drive it; a particle belief over the hidden parameters is the
robot's state; task requirements are phase-indexed formulas over the patch trace. Task A (assess ripeness by
touch, intact) has a squeeze corridor between an informative and a bruising force, empty for ripe fruit; Task B
(halve and pit within five days) has a ripeness band in time between torn flesh and overripeness. On held-out
simulated avocados (30 per cell) the closed-loop squeeze beats the 10~N thumb press 46:0 on paired fruit, and the
belief-planned cut day beats cutting at once 48:0 and fixed waits of two and four days 21:2 and 20:2; removing the
seed-coat distinction from the robot's model costs 49:0, while other removed distinctions make no detectable
difference here. A replay audit finds no axiom mismatch in """ + f"{AXA['steps']:,}" + r""" executed steps;
counterexamples show that the qualitative state alone does not fix the outcome. We then grow the theory from a
deliberately weak seed by its own failures -- five requests, localized gaps, discriminating experiments in the
research SDT, fitted patches, admit / restrict / version decisions -- and generate reinforcement-learning problems
from successive theory versions: the learned policy matches the hand-coded executive when the theory supplies the
seed-coat concept and the hazard shield (88/90), fails without the concept (38/90), and buys the forbidden knife
strike in every episode when the hazard is only a small penalty. Everything is simulation evidence on behaviourally
plausible, partly ILLUSTRATIVE laws; nothing here is validated on real avocados or a real robot.
\end{abstract}
\tableofcontents\clearpage
""")

    # ---------------------------------------------------------------- 1 introduction
    w(r"""\section{Introduction}
Judging an avocado's ripeness by squeezing it and preparing it are two tasks a kitchen robot meets every week. In
the first the robot must establish enough contact to learn how soft the flesh is without pressing hard enough to
bruise it; neither bound is visible, and both depend on the fruit's hidden ripeness. In the second the robot must
cut through skin and flesh all round the seed, separate the halves, lift the seed out and leave a clean cavity -- and
the moment it may do this depends on ripeness too: an unripe fruit tears, the flesh clinging to the seed; an
overripe one is past eating. The same physical event, the separation of the flesh, is forbidden in the first task,
required at one place and phase in the second, and forbidden everywhere else (a bruise, a torn cavity wall, a gouge).

\paragraph{From the egg to the avocado} The egg microtheory (AICOR working report, October 2026) made this
distinction explicit for grasping an egg intact and cracking it into a bowl: a relational interval axiomatization of
action consequences and phase-specific goals tied to an executable SDT, a probabilistic hybrid action model (PHAM)
over that SDT, a partial belief, candidate-action queries, reactive task programs and held-out simulation evidence
with ablations and misspecified plants. This report follows its structure section by section for the avocado and
states where the transfer is direct, where it changes, and what is new: a lifetime (ripening, spoilage, browning)
that turns the decision into one about \emph{when} to act, a hidden binary (which side the seed coat sticks to)
that is perceivable only after the fact and needs a repair, and a corridor that is often empty.

\paragraph{From one microtheory to theory construction} The companion paper \emph{Manipulation as Theory
Construction} treats competence as an evolving, scoped, evidence-bounded theory
$\mathcal{T}_k=\langle\Sigma_k,A_k,M_k,C_k,E_k\rangle$ whose claims $\Gamma=\mathrm{Competent}(F_\text{task},
F_\text{BWS},A_\text{assess})$ are maintained only while the evidence warrants them, a commitment gate that separates
plausible proposals from warranted commitments, a learning loop in which failures expose localized gaps, and
reinforcement learning that lives inside the theory. Sections~\ref{sec:learning} and \ref{sec:prl} implement these
three ideas on the avocado: the microtheory is grown from a weak seed by its own failures, and reinforcement-learning
problems are generated from the successive theory versions.

\paragraph{Hypotheses} As for the egg: \textbf{H1} belief-dependent queries and feedback regulate better than prior-only
and fixed-action baselines; \textbf{H2} removing decision-relevant distinctions from the robot's model changes what
it achieves, and additional structure is not uniformly useful; \textbf{H3} some model mismatches are compensated
through evidence produced in time, others cause irreversible loss before a correction is possible. Added here:
\textbf{H4} a theory grown from its failures by discriminating experiments recovers the plant's laws and keeps its
claims calibrated to their scope; \textbf{H5} reinforcement learning programmed by the theory inherits its
distinctions and its safety rules -- what the theory does not represent the learner cannot learn, and what the theory
forbids a reward cannot buy.

\paragraph{What this report is} One executable research SDT of a Hass avocado in simulation (the HouseholdTwin
avocado twin: USD part graph, register of """ + f"{len(B.PR.AVOCADO)}" + r""" parameters with status labels,
reduced-order mechanics, lifetime model, particle belief); code in \texttt{theory/} of the repository. The robot is
not simulated at contact level here: Task programs act on the plant through the exogenous inputs (forces, times,
tools); the PR2 motions of the CRAM kitchen (separate report) realise the same programs kinematically.
""")

    # ---------------------------------------------------------------- 2 related work (short)
    w(r"""\section{Related work}
The egg microtheory places this line of work between logical formalisations of everyday physical tasks (Hayes'
naive physics, Davis' egg-cracking problem and its logical solutions, Morgenstern's mid-sized axiomatizations),
simulation-based reasoning in KnowRob (Kunze and Beetz), probabilistic simulation as a model of physical
understanding (Battaglia, Hamrick, Tenenbaum), belief-space and chance-constrained planning, probabilistic hybrid
action models (Beetz and Grosskreutz) and semantic digital twins; we do not repeat that review. Avocado-specific
sources enter through the parameter register: the Avocados Australia ripeness chart and penetrometer bands
(\texttt{avo\_au\_stages}), Hort Innovation's finding that a 10~N thumb press bruises firm-ripe fruit
(\texttt{hia2018}), the Cummings and Schroeder anatomy (\texttt{cs42}), the knife-injury epidemiology of pitting
(\texttt{farley2019}), UC Davis postharvest storage temperatures (\texttt{ucd\_avocado}), the softening-stage storage
study of Marques et al.\ (\texttt{marques2011}), and an informal kitchen test of browning treatments
(\texttt{atk\_browning}). Cutting as controlled separation follows Atkins. Theory construction, the commitment gate,
NEEMs and reinforcement learning inside the theory follow \emph{Manipulation as Theory Construction} and the AICOR
working book.
""")

    # ---------------------------------------------------------------- 3 problem
    w(r"""\section{The problem is an interaction episode, not a force}
\paragraph{Definition 1 (Avocado manipulation episode)} An episode is a tuple
\[\mathcal{E}=\langle\text{participants},\text{roles},\mathcal{S}_{0:T},q_{0:T},\Delta_{1:K},o_{0:T},u_{0:T},
\text{outcome}\rangle:\]
the participants (robot, gripper pads, knife, spoon, avocado and its parts, board) fill roles
(squeezer, squeezed fruit, cutter, cut object, holder, container of the seed, receptacle); $\mathcal{S}_t$ is the
structured state, $q_t$ the factored mode, $\Delta_k$ the structural changes (patches), $o_t$ the robot's observations
and $u_t$ the exogenous inputs (pad force, blade, torque, spoon, waiting time). Plan and belief are outside the plant.

\paragraph{Two tasks with one physics}
\begin{itemize}
\item \textbf{Task A (assess ripeness by touch, intact).} Squeeze the fruit and decide whether it is ready
(stage $\ge 4$) without ever bruising it. Success: $\neg\mathsf{Bruised}$ and the right decision.
\item \textbf{Task B (halve and pit within five days).} Choose the day, cut round the seed, twist the halves apart,
remove the seed, leave a clean cavity. Success: halves apart, seed out, no torn flesh, no seed coat in the cavity, no
bruise, no unsafe tool use, edible when cut (eating quality $\ge 0.6$), not spoiled.
\end{itemize}

\paragraph{The squeeze corridor (Task A)} A squeeze informs only above a force, because the stiffness reading's
relative noise falls as $0.1\cdot4\,\mathrm{N}/F$, and bruises above a ripeness-dependent threshold:
\begin{equation}
F_\text{inform}(\epsilon) \;<\; F_\text{squeeze} \;<\; F_\text{bruise}(\text{stage}) = 0.6\,F_\text{pen}(\text{stage}),
\label{eq:corridor}\end{equation}
with the penetrometer force falling from 118~N at stage~1 to 4.9~N at stage~5 (\texttt{avo\_au\_stages}) and the
ratio 0.6 from the 10~N thumb press that bruises firm-ripe fruit (\texttt{hia2018}, SOURCED-VERIFY). For ripe fruit
the corridor is empty: no squeeze is both informative and safe.

\paragraph{The ripeness band (Task B)} The flesh releases cleanly from the seed only from a stage near 3
(register \texttt{clean\_twist\_stage}, ILLUSTRATIVE), mashes well only from about 3.4, and declines past a ripe
window of 24--48~h (\texttt{ripe\_window\_counter}, SOURCED-VERIFY) until it spoils. The central prediction of Task B is
therefore a \emph{band in time}, not a force interval:
\begin{equation}
P(\text{success}\mid d) = \mathbb{E}_\Theta\big[p_\text{clean}(s_d)\;\mathbf{1}\{q_\text{eat}(s_d,h_d)\ge 0.6\}\;
\mathbf{1}\{\neg\text{spoiled}_d\}\big],\qquad d^* = \min\{d: P(\cdot\mid d)\ge \max_{d'} P(\cdot\mid d') - 0.02\},
\label{eq:band}\end{equation}
where $s_d$ is the stage and $h_d$ the hours at full ripeness after $d$ days.

\paragraph{Why it is hard} (1) Ripeness is hidden; colour is a weak cue, touch is informative only where it may
bruise. (2) The outcome of the twist is decided by a hidden property at the moment of the cut, and cannot be undone.
(3) Which side the seed coat sticks to is invisible until the seed is out. (4) Time is part of the action: waiting
is the main intervention, and it can be too long as well as too short.
""")

    # ---------------------------------------------------------------- 4 partonomy
    w(r"""\section{The avocado is a layered fruit around a hidden seed}
\paragraph{Partonomy at fidelity level L1} The twin's USD part graph (HouseholdTwin \texttt{assets/food/avocado\_4})
is the canonical level: \texttt{Exocarp} (cuticle, epidermis, hypodermis, skin parenchyma, stone-cell layer), stem
button, \texttt{Mesocarp} (outer dark-green, middle pale-green, inner yellow band, six vascular strands),
\texttt{Endocarp}, \texttt{Seed} (seed coat with outer and inner testa, two cotyledons, embryonic axis). Events create
further entities: the kerf and cut faces, the two halves, torn-flesh surfaces, the cavity, seed-coat sheets left in
it, gouges, a browned cut face. Every prim under \texttt{/Food/Parts} is an SDT node with \texttt{sdt:class},
\texttt{sdt:partOf} and \texttt{sdt:observability} (visible, after cutting, tactile, hidden).
""")
    w(figrow([("renders/food/anatomy_avocado_labelled.png", "L1 partonomy (exploded anatomy plate of the twin)"),
              ("assets/food/avocado_4/process/ripe/render_halves.png", "event-created entities: halves, cut faces, "
               "torn cavity wall")],
             "The avocado's partonomy and two of the entities that events create (renders of the twin).",
             r"0.22\textheight"))
    w(table(["p{0.24\\linewidth}", "X"], ["Owner", "Attributes (state owned by exactly one node)"], [
        ["Avocado", "pose, mass, ripeness stage, ripening rate, hours at full ripeness, integrity mode (whole, scored, "
         "partly scored, halved, halved with torn flesh), storage"],
        ["Exocarp / Cuticle", "thickness, colour L* C* h, friction, peelability"],
        ["Mesocarp (bands)", "modulus $E(\\text{stage})$, penetrometer force, bruise threshold, cutting toughness, bruise "
         "mode, eating quality; on cut faces: browning dE"],
        ["Seed", "length, width ratio, offset, adhesion to the flesh"],
        ["SeedCoat", "side it adheres to (flesh / seed), remnant fraction in the cavity"],
        ["Cut (relation)", "covered sector, stop-depth profile, peak force"],
        ["Grasp / Squeeze (relation)", "pad force, pad width, stiffness reading"],
        ["Twist (relation)", "applied torque, peak torque, outcome mode"]],
        "Attribute ownership (Principle 1 of the egg microtheory): ripeness belongs to the fruit, bruising and "
        "browning to the flesh, adhesion to the seed, the testa side to the seed coat; the global integrity is a "
        "derived aggregate.", size="scriptsize"))

    # ---------------------------------------------------------------- 5 SDT and patches
    w(r"""\section{The SDT is the state space; events are patches}
As for the egg, $\mathcal{S}_t=\langle N_t,P_t,R_t,A_t,\Phi_t\rangle$ (nodes, partonomy, relationships, attributes,
relational facts) and a patch $\Delta=\langle t,\text{event},\text{ops},\text{cause}\rangle$ applies structural
operations atomically. The plant of \texttt{theory/avocado\_bws.py} keeps the relational facts and logs every patch;
three examples from simulated runs:
\begin{align*}
\Delta_\text{Squeeze} &= \{+\mathsf{Bruised}(\mathsf{Mesocarp}),\;-\mathsf{Intact}(\mathsf{Mesocarp})\}
 \;\Leftarrow\; F > 0.6\,F_\text{pen}(\text{stage}),\\
\Delta_\text{TwistTorn} &= \{+\mathsf{Apart}(\mathsf{HalfA},\mathsf{HalfB}),+\mathsf{SeedIn}(\mathsf{HalfA}),\\
 &\qquad +\mathsf{TornFlesh}(\mathsf{HalfB}),+\mathsf{FleshOnSeed}(\mathsf{Seed}),-\mathsf{Whole},-\mathsf{Scored}\}
 \;\Leftarrow\; \text{no release at the seed},\\
\Delta_\text{RemoveSeed} &= \{+\mathsf{SeedOut}(\mathsf{HalfA}),-\mathsf{SeedIn}(\mathsf{HalfA}),
 +\mathsf{TestaIn}(\mathsf{Cavity})\} \;\Leftarrow\; \text{testa adheres to the flesh}.
\end{align*}
The trace of a Task B episode is $\mathcal{S}_0\xrightarrow{\Delta_1}\mathcal{S}_1\cdots$ interleaved with the
continuous evolution of attributes (ripening over days; browning of a cut face over hours).
""")

    # ---------------------------------------------------------------- 6 physics
    reg = B.PR.AVOCADO
    keys = ["ripeness_stage", "ripening_rate", "penetrometer_force", "flesh_E", "bruise_force_ratio",
            "squeeze_stiffness_model", "skin_puncture_force", "flesh_cut_toughness_unripe", "seed_adhesion_unripe",
            "clean_twist_stage", "testa_adheres_to", "testa_remnant_fraction", "peel_coat_success", "ripening_q10",
            "ripe_window_counter", "decline_per_day_counter", "spoil_days_past_window", "browning_rate_cut",
            "browning_treatment_factors"]
    w(r"""\section{The physics is a reduced-order layer over the twin}
Every law is a standard engineering approximation or a behavioural assumption; Table~\ref{tab:params} gives its
status. \emph{Squeeze:} the stiffness at 4~N per stage comes from the twin's Isaac Sim FEM squeeze table (DERIVED;
stages 1--2 under-resolved); the reading's relative noise is $0.1\cdot 4/F$. \emph{Bruise:} the flesh bruises when the
pad force exceeds 0.6 times the penetrometer force of the stage. \emph{Cut round the seed:} the blade force is skin
resistance times thickness plus a cutting toughness that falls with $\sqrt{E(\text{stage})/E_\text{unripe}}$ times
depth; the blade stops on the seed coat; a force limit can stall it (partly scored). \emph{Twist:} the torque needed is
that of uncut flesh bridges plus the seed adhesion over half the seed; the flesh releases at the seed with
$p_\text{clean}(\text{stage})=\sigma((\text{stage}-2.8)/0.15)$, else it tears. \emph{Seed removal:} spoon lever
($0.3\times$ adhesion $\times$ area), knife twist (adhesion torque; a knife strike into a hand-held half is the pitting
injury of \texttt{farley2019}), squeeze-out (ripe only); the spoon gouges with a probability that grows as the flesh
clings; the seed coat stays in the cavity when it adheres to the flesh, a spoon-tip pass lifts it with probability
0.85. \emph{Lifetime:} the stage advances at the fruit's own rate times a Q10 temperature factor; past stage~5 the
fruit stays at its best for its ripe window, then declines and spoils; a cut face browns as $\mathrm{d}E = k\sqrt{t
\cdot f_\text{treatment}}$.
""")
    w(table(["p{0.25\\linewidth}", "p{0.32\\linewidth}", "p{0.13\\linewidth}", "X"],
            ["Quantity", "Value / prior", "Status", "Basis"],
            [[esc(k), esc(B.fmt_val(reg[k][1]) + " " + reg[k][0]), esc(reg[k][2]), esc(reg[k][3][:90])]
             for k in keys if k in reg],
            "Parameters of the avocado laws (food/params.py AVOCADO). ILLUSTRATIVE values are assumptions of the twin.\\label{tab:params}",
            size="scriptsize"))

    # ---------------------------------------------------------------- 7 PHAM
    w(r"""\section{The episode is a probabilistic hybrid model over the SDT}
\paragraph{Definition 4 (SDT probabilistic hybrid automaton)} $\mathcal{H}_\text{avo}=(Q,\mathbb{S},\Theta,U,
\text{Flow},\text{Inv},\text{Guard},\text{Reset},P)$ with factored modes $q=(\text{integrity},\text{flesh},
\text{seed},\text{cavity},\text{maturity})$: integrity $\in$ \{whole, partly scored, scored, halved, halved torn\};
flesh $\in$ \{intact, bruised\}; seed $\in$ \{in, out\}; cavity $\in$ \{clean, seed coat, gouged\}; maturity $\in$
\{unripe, ripe, overripe, spoiled\}. $\Theta$ (sizes, seed, stage, rate, testa side, window, spoil time) is sampled
once per run, so the plant is deterministic given $\Theta$ and the inputs; every jump probability is a marginal over
$\Theta$ (Table~\ref{tab:jumps}).

\paragraph{Definition 5 (Avocado PHAM)} The processes \textsc{squeeze}, \textsc{ripen}, \textsc{cut},
\textsc{twist}, \textsc{seed-removal}, \textsc{peel} and \textsc{brown} read and write the shared SDT and never call
each other: a squeeze reads the stage that \textsc{ripen} writes; the twist reads the coverage that \textsc{cut}
wrote and the stage; the cavity process reads the testa side. The plan is outside the PHAM: it sets the exogenous
inputs and synchronises on observed events (\texttt{waitfor}, \texttt{whenever}).
""")
    w(table(["l", "p{0.3\\linewidth}", "p{0.27\\linewidth}", "X"], ["Jump", "Guard (SDT query)", "Reset $\\Delta$",
                                                                     "Probability source"], [
        ["Bruise", "squeeze $F > 0.6\\,F_\\text{pen}$(stage)", "+Bruised(Mesocarp)", "$P_\\Theta$(stage $>$ s(F))"],
        ["Ripe / Overripe / Spoiled", "stage $\\ge 4$; hours at stage 5 $>$ window; past spoil time", "+Ripe; +Overripe; +Spoiled",
         "rate, window, spoil time ($\\Theta$)"],
        ["Scored / PartlyScored", "blade force $\\le F_\\text{max}$ all round / not", "+Scored / +PartlyScored",
         "skin resistance, toughness(stage)"],
        ["ReleaseAtSeed / TearFlesh", "scored, torque $\\ge$ needed", "+Apart (+TornFlesh, +FleshOnSeed)",
         "$p_\\text{clean}$(stage)"],
        ["Stuck", "torque $<$ bridges + adhesion", "none", "coverage, adhesion(stage)"],
        ["SeedOut (+TestaIn, +Gouge)", "apart, seed in, force $\\ge$ needed", "+SeedOut, +TestaIn, +Gouge",
         "testa side; $p_\\text{gouge}$(stage)"],
        ["UnsafeToolUse", "seed removal by knife strike", "+UnsafeToolUse(KnifeStrike)", "the tool choice"],
        ["CoatLifted", "TestaIn, spoon-tip pass", "$-$TestaIn", "peel\\_coat\\_success"],
        ["Browned", "dE $\\ge$ visible", "+Browned(face)", "time, treatment"]],
        "Jump schemas of the avocado PHAM. As for the egg, no transition probability is a free parameter: each is a "
        "marginal over $\\Theta$.\\label{tab:jumps}", size="scriptsize"))
    w(B.code("""plan ASSESS-RIPENESS-BY-TOUCH                       plan HALVE-AND-PIT (within 5 days)
  photo; belief update                                photo; belief update
  F0 <- largest probe with P(bruise | F0) <= 1 %      loop each day
  squeeze(F0); belief update                            gentle squeeze if P(bruise) <= 1 %; belief update
  if P(bruise | F_inform) <= 1 %:                       d* <- band plan (Eq. 2); if d* = 0: exit loop
     squeeze(F_inform); belief update                   wait 1 day; predict
  decide ready <- P(stage >= 4) > 0.5                 cut-round-seed      whenever partly scored do re-cut
                                                      twist-apart         whenever stuck do re-cut; twist
                                                      remove-seed(spoon)
                                                      while seed coat seen and passes < 2: peel-seed-coat""",
             "The task programs (theory/avocado\\_policies.py)"))

    # ---------------------------------------------------------------- 8-9 observation and belief
    w(r"""\section{The robot sees the SDT only through a noisy observation model}
The observation generator $\Omega(o_t\mid\mathcal{S}_t)$ reads the plant SDT and produces: a photo (silhouette aspect
ratio with 2~\% noise, neck taper, skin L* C* h from the stage with noise and an optional bias, stem button); a
squeeze stiffness reading (relative noise $0.4/F$); the blade's stop depths round the seed; the twist outcome and
peak torque; the spoon's lever force; the seed coat seen in the cavity; the extracted seed's length. Ground truth for
evaluation is kept separately and never reaches the belief.

\section{The robot estimates selected SDT attributes}
\paragraph{Definition 6 (Implemented belief)} A particle filter (1,000 particles from the register prior, cached;
food/avocado\_belief.py) over $\Theta$ with a bruise flag and the testa side; the likelihoods are the physics: the
FEM stiffness table for a squeeze, the colour chart for a photo, $p_\text{clean}$ for the twist, the adhesion model
for the spoon force, the testa side for the cavity inspection. Prediction is ripening at each particle's rate.
Queries: $P(\text{bruise}\mid F)$, $P(\text{stage}\ge 4)$, and the band of Eq.~\eqref{eq:band} over the lifetime.
Figure~\ref{fig:trace} follows one unripe held-out avocado through Task B.
""")
    w(fig(f_tr, f"Belief against the hidden stage for held-out avocado 1003 (unripe) under the belief-band policy: "
          f"each day a gentle squeeze, then the band plan; the cut comes when the planned success is near its best "
          f"(outcome: {esc(trace_label)}). Planned P(success) by remaining day at each decision: "
          + esc("; ".join(f"day {d}: {p}" for d, p in trace_plan[:3])) + ".\\label{fig:trace}",
          r"0.24\textheight"))

    # ---------------------------------------------------------------- 10 intentions
    w(r"""\section{Relating predicted physical evolution to intentions}
The same patch carries two meanings. The phase-indexed requirements over the plant trace (\textbf{G} always,
\textbf{F} eventually, \textbf{U} until):
\begin{align}
\mathcal{G}_A &= \mathbf{G}\,\neg\mathsf{Bruised}(\mathsf{Mesocarp})\;\wedge\;\text{decide}(\text{ready})
 \leftrightarrow \text{stage}\ge 4,\\
\mathcal{G}_B &= \neg\mathsf{Separated}\;\mathbf{U}\;\text{CutPhase}\;\wedge\;\mathbf{F}\,\mathsf{Apart}
 \wedge \mathbf{F}\,\mathsf{SeedOut}\;\wedge\;\mathbf{G}\,\neg\mathsf{TornFlesh}\wedge\mathbf{G}\,\neg\mathsf{Bruised}
 \wedge\mathbf{G}\,\neg\mathsf{UnsafeToolUse}\nonumber\\
 &\quad\wedge\;\neg\mathsf{TestaIn}(\mathsf{Cavity})(T)\;\wedge\;q_\text{eat}(t_\text{cut})\ge 0.6\;\wedge\;
 \neg\mathsf{Spoiled}(t_\text{cut}).
\end{align}
The implemented checks (\texttt{G\_A}, \texttt{G\_B}) evaluate exactly these on the patch trace; gouges lower the
quality but are not failures.
""")
    w(table(["p{0.13\\linewidth}", "p{0.18\\linewidth}", "p{0.2\\linewidth}", "p{0.18\\linewidth}", "X"],
            ["Task, phase", "Intended change", "Persistent requirement", "Relevant uncertainty", "Intervention"], [
        ["A: squeeze", "none (information)", "flesh intact", "stage, bruise threshold", "probe force, corridor; stop"],
        ["B: wait", "ripening into the band", "not overripe, not spoiled", "stage, rate", "daily squeeze, band re-plan"],
        ["B: cut", "kerf all round the seed", "no bruise from the hold", "toughness, seed size", "force limit; re-cut a gap"],
        ["B: twist", "separation in the cut plane", "no torn flesh", "stage (release), bridges", "torque limit; re-cut"],
        ["B: pit", "seed out, cavity clean", "no unsafe tool use", "testa side, adhesion", "spoon; peel passes"]],
        "How phase-specific intentions assess the same physical model (the egg's Table 8, for the avocado).",
        size="scriptsize"))

    # ---------------------------------------------------------------- 11 Task A
    rowsA = []
    for pol in P.POLICIES_A + P.ABLATIONS_A:
        rowsA.append([esc(pol)] + [esc(fmt(cell(A, c, pol))) for c in ("unripe", "breaking", "ripe")] +
                     [esc("; ".join(f"{c}: {fails(cell(A, c, pol))}" for c in ("breaking", "ripe")
                                    if fails(cell(A, c, pol))))])
    mA = {(m["b"], tuple(m["cell"])): m for m in r["mcnemar_A"]}
    th = mA[("firm thumb 10 N", ("pooled",))]
    po = mA[("photo only", ("pooled",))]
    ol = mA[("open-loop prior plan", ("pooled",))]
    w(r"""\section{Task A: the chance constraint protects ripe fruit}
The closed loop probes with the largest force in \{2, 1.5, 1\}~N whose bruise risk under the belief is at most 1~\%,
updates, and squeezes at the informative force $F_\text{inform}=3.3$~N only if the corridor is still non-empty. The
open-loop prior plan computes one force on the photo belief; the fixed policies squeeze at 4~N or with the 10~N thumb
press of \texttt{hia2018}; the ablation gives every fruit one bruise threshold (the population median) instead of a
ripeness-owned one. Figure~\ref{fig:corridor} shows the corridor of Eq.~\eqref{eq:corridor} per class: model
marginals (lines) against simulated squeezes (markers, 24 avocados per force).
""")
    w(fig(f_cor, "The squeeze corridor: P(bruise | F) and P(reading within 12 \\% of the true stiffness | F), model "
          "(lines) and simulated (markers). Green: forces that are both safe (P $\\le$ 1 \\%) and informative "
          "(P $\\ge$ 0.75); empty for ripe fruit.\\label{fig:corridor}", r"0.25\textheight"))
    w(table(["p{0.25\\linewidth}", "l", "l", "l", "X"], ["Policy", "unripe", "breaking", "ripe", "Failures"], rowsA,
            "Task A on held-out avocados (seeds 1000--1029, 30 per class), successes with 95 \\% Wilson intervals "
            "below 30/30. Every policy runs on the same avocados and plant seeds within a class.", size="scriptsize"))
    w(r"""Paired over all 90 avocados the closed loop beats the thumb press """ +
      f"{th['a_only']}:{th['b_only']} (exact McNemar $p={pf(th['p'])}$)" + r""" and photo-only judgement """ +
      f"{po['a_only']}:{po['b_only']} ($p={pf(po['p'])}$)" + r"""; against the open-loop prior plan it is """ +
      f"{ol['a_only']}:{ol['b_only']} ($p={pf(ol['p'])}$)" + r""", so feedback adds little here: as for the egg,
the chance constraint evaluated once on the (photo-updated) prior already protects most fruit, and the feedback
only sharpens decisions near the ready threshold. Ripe fruit is the hard class for every policy: the corridor is
empty, the gentle reading is noisy, and the ready threshold at stage~4 lies inside the class. The thumb press is
informative and bruises every ripe and most breaking fruit -- the shopper's mistake the source warns about.
""")

    # ---------------------------------------------------------------- 12 Task B
    rowsB = []
    for pol in P.POLICIES_B:
        rowsB.append([esc(pol)] + [esc(fmt(cell(Bt, c, pol))) + f" ({cell(Bt, c, pol)['mean_day']:.1f} d)"
                                   for c in ("unripe", "breaking", "ripe")] +
                     [esc("; ".join(f"{c}: {fails(cell(Bt, c, pol), 2)}" for c in ("unripe", "breaking", "ripe")
                                    if fails(cell(Bt, c, pol), 2)))])
    mB, mBmis = {}, {}
    for m in r["mcnemar_B"]:                    # the first pooled set is well specified, the second pooled over misspecified plants
        (mBmis if (m["b"], tuple(m["cell"])) in mB else mB)[(m["b"], tuple(m["cell"]))] = m
    w(r"""\section{Task B: halving on purpose reverses one constraint at one place and time}
The belief policy squeezes gently every day (where safe), plans the day of the cut by Eq.~\eqref{eq:band} over the
lifetime model, re-plans every day, cuts with repairs (a stalled cut is completed, a stuck fruit is re-cut and twisted
again), removes the seed with the spoon and peels the seed-coat sheets while they are seen. Figure~\ref{fig:band}
shows the band per class: success against the day of the cut for fixed-day programs (simulated) and the model's
prediction for the same avocados.
""")
    w(fig(f_band, "The ripeness band of Task B: success by day of the cut, simulated (30 avocados per point) and "
          "predicted by the model (dashed, from the instances' parameters). Unripe fruit needs days, ripe fruit goes "
          "past its best; breaking fruit has the widest band.\\label{fig:band}", r"0.25\textheight"))
    w(table(["p{0.15\\linewidth}", "l", "l", "l", "X"], ["Policy", "unripe", "breaking", "ripe", "Failures"], rowsB,
            "Task B on held-out avocados, successes (mean day of the cut). Every policy runs on the same avocados and "
            "plant seeds within a class.", size="scriptsize"))
    w(r"""Paired over the 90 avocados the belief policy beats cutting at once """ +
      f"{mB[('cut now', ('pooled',))]['a_only']}:{mB[('cut now', ('pooled',))]['b_only']} ($p={pf(mB[('cut now', ('pooled',))]['p'])}$)" +
      r""", waiting two days """ + f"{mB[('wait 2 days', ('pooled',))]['a_only']}:{mB[('wait 2 days', ('pooled',))]['b_only']} "
      f"($p={pf(mB[('wait 2 days', ('pooled',))]['p'])}$)" + r""" and waiting four days """ +
      f"{mB[('wait 4 days', ('pooled',))]['a_only']}:{mB[('wait 4 days', ('pooled',))]['b_only']} "
      f"($p={pf(mB[('wait 4 days', ('pooled',))]['p'])}$)" + r""". Each fixed day fails one population: cutting at once
tears unripe fruit, waiting four days lets ripe fruit go past its best. The photo-only band (no squeezes) is as good
as the belief band on these well-specified plants (""" +
      f"{mB[('photo-only band', ('pooled',))]['a_only']}:{mB[('photo-only band', ('pooled',))]['b_only']}" + r"""):
in simulation the photo's colour likelihood is the plant's own colour law, so the camera is informative enough; the
squeezes earn their keep only on misspecified plants (Section~\ref{sec:abl}). The knife twist removes the seed in
every episode and fails every one on the safety requirement -- a hazard the requirement forbids whatever the quality.
The belief policy's own prediction of success at its decision averaged """ +
      f"{r['calibration_B']['predicted_mean']:.3f} against {r['calibration_B']['observed_mean']:.3f} observed (Brier "
      f"{r['calibration_B']['brier']:.3f})" + r""": it is overconfident because the band prediction covers clean
separation, edibility and spoilage but not the seed coat left after two peel passes.
""")

    # ---------------------------------------------------------------- 13 ablations, misspecified plants
    rowsAb = []
    for pol in P.ABLATIONS_B:
        tot = sum(cell(Bt, c, pol)["ok"] for c in W.CLASSES)
        m = mB[(pol, ("pooled",))]
        rowsAb.append([esc(pol), f"{tot}/90", f"{m['a_only']}:{m['b_only']}", f"${pf(m['p'])}$",
                       esc("; ".join(fails(cell(Bt, c, pol), 2) for c in W.CLASSES if fails(cell(Bt, c, pol), 2)))])
    bb = sum(cell(Bt, c, "belief band")["ok"] for c in W.CLASSES)
    rowsMis = []
    for mis in sorted({x["mis"] for x in Bt} - {"well specified"}):
        for cls in ("unripe", "breaking", "ripe"):
            if cell(Bt, cls, "belief band", mis) is None:
                continue
            rowsMis.append([esc(mis), esc(cls)] + [esc(fmt(cell(Bt, cls, p, mis))) for p in
                                                    ("belief band", "photo-only band", "cut now", "wait 2 days")])
    rowsMisA = []
    for mis in sorted({x["mis"] for x in A} - {"well specified"}):
        for cls in ("breaking", "ripe"):
            rowsMisA.append([esc(mis), esc(cls)] + [esc(fmt(cell(A, cls, p, mis))) + (
                f" ({fails(cell(A, cls, p, mis), 1)})" if fails(cell(A, cls, p, mis)) else "")
                for p in ("closed loop", "photo only", "firm thumb 10 N")])
    w(r"""\section{Does the represented structure matter? Ablations and misspecified plants}\label{sec:abl}
\paragraph{Design: remove one distinction} Each ablation keeps the plant, sensors and every other part of the robot
and removes one piece of structure from the robot's model together with its consumers: \emph{no testa side} (the seed
coat is not represented, so nothing inspects or repairs the cavity); \emph{no repair} (a stalled cut or stuck twist is
not answered); \emph{ready / not ready} (ripeness is binary: cut when $P(\text{stage}\ge4)>0.5$, no band);
\emph{population ripening rate} (the fruit's own rate is replaced by the population mean).
""")
    w(table(["p{0.24\\linewidth}", "l", "l", "l", "X"], ["Removed distinction", "success", "discordant",
                                                         "p", "failures"], rowsAb,
            f"Task B ablations against the full belief policy ({bb}/90) on the same 90 held-out avocados; discordant: "
            "pairs where only the full policy / only the ablation succeeds.", size="scriptsize"))
    w(r"""Only one removed distinction is load-bearing on these avocados: without the seed-coat side the cavity is left
dirty whenever the coat sticks to the flesh (about half the fruit). The others make no detectable difference here,
which is not evidence that they are useless: no repair is needed because no held-out cut stalled or twist stuck with a
full cut; a binary ripeness does as well because the planner's daily re-sensing already lands the cut inside the band;
the population rate suffices because daily sensing corrects the drift. As for the egg's regional strength, a
distinction needs a discriminating test.

\paragraph{Design: change the plant, not the robot} Misspecified plants keep the robot's model and change a law of
the simulated physics: the fruit ripens 0.6 or 1.6 times as fast as its drawn rate; the camera reads 10 L* units darker
(riper) or lighter (less ripe); the seed sticks twice as hard; for Task A the flesh bruises at 0.6 times the modelled
threshold (weak flesh).
""")
    w(table(["l", "l", "l", "l", "l"], ["Plant change (hidden)", "class", "closed loop", "photo only", "thumb 10 N"],
            rowsMisA, "Task A on misspecified plants.", size="scriptsize"))
    w(table(["l", "l", "l", "l", "l", "l"], ["Plant change (hidden)", "class", "belief band", "photo-only band",
                                             "cut now", "wait 2 days"], rowsMis,
            "Task B on misspecified plants (30 held-out avocados per cell).", size="scriptsize"))
    wf = cell(A, "ripe", "closed loop", "bruise_ratio x0.6")
    cr = cell(A, "ripe", "closed loop", "colour L -10")
    cp = cell(A, "ripe", "photo only", "colour L -10")
    w(r"""\paragraph{The direction of an error decides whether it can be corrected (H3)} Weak flesh is an error in the
dangerous direction: the robot believes a squeeze safe that bruises, and the bruise is the first evidence -- the
closed loop succeeds on """ + f"{wf['ok']}/30" + r""" ripe fruit. A camera bias is an error in a correctable
direction for a robot that also touches: the squeezes override the photo (closed loop """ + f"{cr['ok']}/30" +
      r""" against photo-only """ + f"{cp['ok']}/30" + r""" on ripe fruit). In Task B the same asymmetry appears with
small numbers: on ripe fruit that ripens faster than modelled or looks less ripe than it is, the daily squeeze
notices the softening and the photo-only planner lets 3--4 of 30 go overripe or spoil (paired 3:0 and 4:0,
$p=0.25$ and $0.13$: not established at this sample size). A fruit that ripens slower than modelled is the other
direction: waiting is the correction, and the belief policy pays for it with later cuts. Pooled over all misspecified
Task B plants the belief band and the photo-only band are not distinguishable (""" + \
      f"{mBmis[('photo-only band', ('pooled',))]['a_only']}:{mBmis[('photo-only band', ('pooled',))]['b_only']}, "
      f"$p={pf(mBmis[('photo-only band', ('pooled',))]['p'])}$" + r"""), while both still beat cutting at once
(belief band """ + f"{mBmis[('cut now', ('pooled',))]['a_only']}:{mBmis[('cut now', ('pooled',))]['b_only']}" + r""") and
waiting two days (""" + f"{mBmis[('wait 2 days', ('pooled',))]['a_only']}:{mBmis[('wait 2 days', ('pooled',))]['b_only']}" + r""").
""")

    # ---------------------------------------------------------------- 14 axioms
    examples = AXA.get("twist_examples") or []
    w(r"""\section{Relational axioms and scoped manipulation correctness}
\paragraph{The axioms} For adjacent intervals $i,i'$ and an action interval $j$ with $\meets(i,j)$ and
$\meets(j,i')$, each action axiom has a precondition over $\holds(\cdot,i)$, guarded branches that add and remove
relations, and the frame rule that every other relation persists (theory/axioms.py):
""")
    w(B.code("\n".join(f"{k:13s} {v}" for k, v in AXA["axioms"].items()), "Action axioms (holds / occurs)"))
    w(r"""\paragraph{Definition 7 and Proposition 1, transferred} A scoped manipulation problem
$\mathcal{P}=\langle R,\tau,\Psi,L\rangle$ (roles, their types with parameter distributions, a success predicate over SDT
traces, the plant's laws), a plan meets requirement $q$ if $P_{\beta\sim\tau,\theta\sim L}[\Psi(\mathrm{trace})]\ge q$,
and the egg's conditional proposition carries over unchanged: if (i) the initial abstract relations represent each
physical initial state, (ii) the guards of the abstract branches cover exactly the simulator's successors and their
effect and frame axioms match the patches, (iii) the abstract observations preserve what the policies branch on, and
(iv) NEEM extraction and abstract goal evaluation agree, then abstract and simulated goal satisfaction agree for every
case, and a complete search of a finite policy language finds a solution iff one exists. The proof is the egg's.

\paragraph{Replay audit} Over """ + f"{AXA['episodes']:,}" + r""" held-out episodes (every Task A and Task B policy and
ablation, three classes, 30 avocados) and """ + f"{AXA['steps']:,}" + r""" executed steps, the audit finds """ +
      f"{AXA['pre_violations']} precondition violations, {AXA['branch_mismatches']} patches outside the axioms' "
      f"branches, {AXA['guard_mismatches']} deterministic-guard mismatches and {AXA['frame_violations']} frame violations"
      + r""". It checks the visited steps, not alternative actions from the visited states. As for the egg, the
qualitative relations alone do not fix the successor: """ + f"{AXA['counterexamples']}" + r""" pairs of steps share an
abstract state and an action but reach different branches (""" +
      esc(", ".join(f"{k} {v}" for k, v in (AXA.get("by_action") or {}).items())) + r"""). The seed removal from the
same relational state ends with or without seed-coat sheets, with or without a gouge -- the testa side and the stage
decide; the twist from the same scored state ends clean or torn -- the stage decides. A proof for a bounded
executive must refine these cells by the hidden attributes or use set-valued transitions.
""")
    rowsV = []
    for pol, label in (("belief band", "Task B belief band"), ("cut now", "Task B cut now"),
                       ("wait 4 days", "Task B wait 4 days")):
        for c in W.CLASSES:
            x = cell(Bt, c, pol)
            rowsV.append([esc(label), esc(c), f"{x['ok']}/30", esc(x["verdict"])])
    for pol in ("closed loop", "firm thumb 10 N"):
        for c in W.CLASSES:
            x = cell(A, c, pol)
            rowsV.append([esc("Task A " + pol), esc(c), f"{x['ok']}/30", esc(x["verdict"])])
    w(table(["X", "l", "l", "l"], ["Plan", "Scope (class)", "Result", "Verdict, q = 0.8"], rowsV,
            "Statistical verification against typed scopes: verified if the 95 \\% Wilson interval lies at or above q, "
            "refuted if it lies below, undecided otherwise.", size="scriptsize"))

    # ---------------------------------------------------------------- 15 learning theories
    st = L["steps"]
    rowsL = [["T0", esc("seed: halve & pit a ripe Hass"), "--", "--", esc("Gamma0: apart & seed out (ripe)"),
              esc(f"{L['T0']['claims'][0][1]}/{L['T0']['claims'][0][2]} {L['T0']['claims'][0][3]}")]]
    for s in st:
        rowsL.append([f"T{s['version']}", esc(s["request"]),
                      esc(f"{s['gap']['name']} ({s['gap']['layer']}): before {s['before']['k']}/{s['before']['n']}"),
                      esc(s["patch"]["disposition"] + ("; +" + ", ".join(s["patch"].get("dSigma", []))
                                                       if s["patch"].get("dSigma") else "")),
                      esc(s["claim"][0]), esc(f"{s['claim'][1]}/{s['claim'][2]} {s['claim'][3]}")])
    hyp_rows = []
    for s in st[:3]:
        for h in s["gap"]["hypotheses"]:
            hyp_rows.append([f"T{s['version']}", esc(h["h"]), esc(h["result"]), esc(h["verdict"])])
    f1 = st[0]["fit"]
    f2 = st[1]["fit"]
    f3 = st[2]["fit"]
    w(r"""\section{Learning theories: growing the avocado theory from its failures}\label{sec:learning}
\paragraph{The package} The robot's competence for this task family is a theory
$\mathcal{T}_k=\langle\Sigma_k,A_k,M_k,C_k,E_k\rangle$ (theory/package.py): concepts that must each be perceivable
through the SDT, actionable through designators and traceable; causal, constitutive and intentional assertions; BWS
families; scoped claims $\Gamma$ with their assessment regime; and evidence. A claim is maintained only while its
held-out assessment verifies it.

\paragraph{The seed theory} $\mathcal{T}_0$ knows whole, scored, halves apart and seed location; it asserts that a cut
round the seed and a twist give two clean halves and a spoon lifts the seed; its scope is ripe Hass fruit; its claim
$\Gamma_0$ (apart and seed out) is verified on ripe fruit. Then requests expand the task and the BWS family. Each runs
the executive derived from the current theory (only concepts in $\Sigma_k$ reach its plan), the mismatches become a
localized gap, competing hypotheses are tested by discriminating experiments on development avocados of the research
SDT, the patch's parameters are fitted from that evidence, and the integrator admits, restricts or versions; the
revised claim is assessed on the held-out avocados (theory/learning.py).
""")
    w(table(["l", "p{0.17\\linewidth}", "p{0.22\\linewidth}", "p{0.18\\linewidth}", "p{0.14\\linewidth}", "X"],
            ["", "Request", "Gap (layer): theory before", "Patch", "Claim", "Held-out"], rowsL,
            "The revision path of the avocado theory. Simulation evidence only.", size="scriptsize"))
    w(table(["l", "p{0.25\\linewidth}", "p{0.36\\linewidth}", "X"], ["", "Hypothesis", "Experiment result", "Verdict"],
            hyp_rows, "Hypotheses and discriminating experiments (development avocados, research SDT).",
            size="scriptsize"))
    w(fig(f_rev, "Revision path: each request (blue) exposes a gap; each patch is admitted, restricted or versioned, and the "
          "revised claim is assessed on held-out avocados. The last restricts the scope after counterevidence.",
          r"0.14\textheight"))
    w(r"""\paragraph{What was learned, and how well} The fitted clean-release law has its centre at """ +
      f"{f1['s0']:.2f}" + r""" and width """ + f"{f1['w']:.2f}" + r""" (plant: """ + f"{f1['true_s0']}, {f1['true_w']}" +
      r"""): the centre is recovered, the transition is estimated wider because the experiment's stage grid is coarse
against a width of 0.15. The share of fruit whose seed coat sticks to the flesh is estimated at """ +
      f"{f2['p_testa_flesh']:.2f}" + r""" (plant: a fair choice), the peel repair's success at """ +
      f"{f2['p_peel']:.2f}" + r""" per pass (plant 0.85), and the new camera's lightness bias at """ +
      f"{f3['bias']:+.1f}" + r""" (plant +10). Two hypotheses were tested and kept apart: an incomplete cut also tears
(through the uncut bridge) even on ripe fruit -- a second cause that $\mathcal{T}_0$ already handles by its
partly-scored state and re-cut repair -- while the torque limit does not change the share of torn fruit. In the first
version of this loop these verdicts were hard-coded and one contradicted the data; they are now computed from the
experiment counts.

\paragraph{Restriction is learning too} A cultivar whose seed clings ten times harder (an imagined family,
ILLUSTRATIVE) fails the cavity claim in """ + f"{st[4]['before']['n'] - st[4]['before']['k']}/{st[4]['before']['n']}" +
      r""" episodes, mostly with the seed stuck. The knife twist would free it but is forbidden by the safety rule, so
the claim is retracted for that family rather than extended: the evidential record grows while the warranted scope
contracts.

\paragraph{The commitment gate} At runtime the executive asks the gate before committing:
""")
    w(table(["p{0.22\\linewidth}", "l", "X"], ["Situation", "Epistemic state", "Reason"],
            [[esc(g["situation"]), esc(g["state"]), esc(g["reasons"][0])] for g in L["gate"]],
            "The commitment gate of the final theory on example situations. Only 'warranted' permits commitment; the "
            "others call for information gathering, a safer specialization, help or abstention.", size="scriptsize"))

    # ---------------------------------------------------------------- 16 programmable RL
    rowsR = []
    for p in R["programs"]:
        ev = p["eval"]
        rowsR.append([esc(p["name"])] + [f"{ev[c]['ok']}/30" for c in ("unripe", "breaking", "ripe")] +
                     [f"{sum(ev[c]['hazard'] for c in ev)}", esc(" / ".join(f"{ev[c]['mean_day']:.1f}"
                                                                          for c in ("unripe", "breaking", "ripe")))])
    rowsR.append([esc("hand-coded theory executive (belief band)")] +
                 [f"{cell(Bt, c, 'belief band')['ok']}/30" for c in ("unripe", "breaking", "ripe")] + ["0", esc(
                     " / ".join(f"{cell(Bt, c, 'belief band')['mean_day']:.1f}" for c in ("unripe", "breaking", "ripe")))])
    w(r"""\section{Programmable reinforcement learning inside the theory}\label{sec:prl}
Reinforcement learning lives inside the theory: the RL problem is \emph{generated} from a theory version
(theory/prl.py). Its state features come from the concepts in $\Sigma_k$ -- only distinctions the theory can perceive
become features (the believed stage and its spread, the day, a torn twist, the seed coat seen); its actions are the
designators' parameters (cut or wait, spoon or knife twist, 0--2 peel passes, the last only if $\Sigma_k$ has the
seed-coat concept); its reward comes from the claim's KPIs (the requirement, eating quality, penalties for gouge, torn
flesh, seed coat left and days); its hard constraints come from the forbidden effects as a shield that masks every
action whose forbidden effect the theory predicts. The learner is batched every-visit Monte-Carlo control with
$\epsilon$-greedy exploration over tabular values (8 rounds of 700 research-SDT episodes on development avocados);
evaluation is greedy on the held-out avocados. The learned policy is a proposal component; the gate keeps authority.
""")
    w(table(["p{0.34\\linewidth}", "l", "l", "l", "l", "X"], ["Program / policy", "unripe", "breaking", "ripe",
                                                              "hazards", "mean day"], rowsR,
            "Programmable RL on the held-out avocados (30 per class) against the hand-coded executive of the same "
            "theory.", size="scriptsize"))
    w(fig(f_rl, "Left: success during training for the three programs. Right: the learned cut / wait decision of the "
          "program generated from $\\mathcal{T}_3$ (shielded), by believed stage and day (cells visited at least 20 "
          "times): it learned the band -- wait below a believed stage of about 3.75, cut above.", r"0.24\textheight"))
    w(r"""\paragraph{What this shows (H5)} The program generated from $\mathcal{T}_3$ learns a cut-or-wait rule that
reproduces the band (wait while the believed stage is below about 3.75, cut above), and with the shield it matches
the hand-coded executive of the same theory on the held-out avocados. The program generated from $\mathcal{T}_1$ has
neither the seed-coat feature nor the peel action, so no amount of training cleans a cavity whose dirt it cannot
represent: its failures are the seed coat left. And when the hazard is only a soft penalty (0.05, below the gouge
cost the knife avoids), the learner buys the forbidden knife strike in every episode: a high reward cannot be allowed
to pay for a forbidden effect -- the theory's safety rule has to be a shield, not a term. Peeling is learned as 'always
two passes' because the declared KPIs give a pass no cost; a cost per pass would change that.
""")

    # ---------------------------------------------------------------- 17 adequacy and limits
    w(r"""\section{Adequacy, evidence and limits}
\paragraph{What the evidence says} H1: within the modelled scope the belief regulates better than the fixed and
prior-only baselines (Task A 46:0 against the thumb press; Task B 48:0, 21:2 and 20:2 against fixed days); feedback adds
little over the chance constraint evaluated once in Task A. H2: the seed-coat distinction is load-bearing (49:0); three
other removed distinctions make no detectable difference on these avocados. H3: errors that make the fruit weaker than
believed fail before evidence arrives; errors that a touch can reveal are corrected in time. H4: the theory grown from
its failures recovers the plant's laws (centre of the clean-release law, testa share, peel success, camera bias) and
keeps its claims verified within their scopes, contracting one. H5: RL programmed by the theory inherits its
distinctions and safety rules.

\paragraph{Four questions about fit} \emph{Parameter calibration:} the laws are behavioural; several constants are
ILLUSTRATIVE (cutting forces, adhesion, clean-release transition, spoilage, browning). \emph{Structural sufficiency:}
the ablations answer this for four distinctions within the simulator. \emph{Predictive calibration:} the Task B
prediction is overconfident (it omits the seed-coat failure). \emph{Physical adequacy:} nothing here tests it.

\paragraph{Limits} The plant and the robot share the same laws except where a misspecification is injected, which
favours a model-based robot. The 30 base avocados are reused across tables (paired, not independent); no correction
for multiple tests. The photo is a symbolic observation from the colour law, not an image; the rendered camera view
of the CRAM scenario reads the twin's dark skin as riper than the law -- a real gap between rendered and modelled
appearance. Contact is not simulated: the cut, twist and spoon are quasi-static mechanics, not contact physics. The
learning loop's agents are scripted functions with typed inputs and outputs, not autonomous scientists; human
authority over consequential revisions is not modelled. The RL is tabular on a handful of features.

\paragraph{Further tests} (1) Real avocados: penetrometer, squeeze stiffness and photos at known stages to calibrate the
colour and stiffness likelihoods and the bruise ratio. (2) Contact simulation of the cut and twist (breakable joints
carrying bridge and adhesion strengths). (3) A capable reactive baseline without belief, to separate prediction from
bundled control. (4) Information-gain planning of squeezes. (5) A second fruit family through the same learning loop.
""")
    w(r"""\section{Conclusion}
The avocado completes the egg's lesson with time: the same separation of the flesh is forbidden in one task and
required at one place and phase in another, and here the main intervention is when to act. A microtheory of parts,
patches, reduced-order laws, a belief and phase-indexed requirements predicts the squeeze corridor and the ripeness
band, regulates better than fixed policies on held-out simulated fruit, and shows which distinctions carry the
decisions. Grown from a weak seed by its own failures, the theory recovers its laws and keeps its claims scoped; and
reinforcement learning programmed by that theory learns only what the theory can represent and cannot buy what it
forbids. All of it is simulation evidence; physical validation is the next step.

\appendix
\section{Reproducing the results}
""")
    w(B.code("""cd ~/Documents/HouseholdTwin
E=~/Documents/LabAutomationTwin/env_cad/bin/python
$E -m theory.run_microtheory          # Task A / B, ablations, misspecified plants, sweeps (~80 s, 14 workers)
$E -m theory.learning                 # theory construction T0 -> T5, gate (~3 min)
$E -m theory.prl                      # programmable RL (~6 min, 14 workers)
$E -c "import json; from theory import axioms as AX; json.dump(AX.run_audit(), open('theory/results/axiom_audit.json','w'))"
$E report/build_avocado_microtheory.py"""))
    # FPICU + FrameNet
    w(r"""\section{FPICU and FrameNet for avocado manipulation}
""")
    w(table(["p{0.12\\linewidth}", "p{0.2\\linewidth}", "p{0.25\\linewidth}", "X"],
            ["", "Dark because", "Represented as", "Operational in"], [
        ["Functionality", "the seed cavity as a container, the skin as a peelable boundary, the half as a bowl",
         "partonomy (Seed in Mesocarp; Cavity created by the twist), SeedIn / SeedOut, TestaIn(Cavity)",
         "seed removal and peel repairs; Task B goal"],
        ["Physics", "ripeness, flesh modulus, adhesion, toughness, browning are not visible",
         "attributes on their owners with the laws of Section 6 and $\\Theta$", "corridor, band, jump probabilities"],
        ["Intent", "whether separation is a failure depends on task and phase", "phase-indexed requirements over SDT facts",
         "plan programs, chance constraints, outcome checks"],
        ["Causality", "why the flesh tore is not in the end state", "patches with causes (no release at the seed)",
         "monitoring verdicts, belief update from the twist, goal watch"],
        ["Utility", "which risk is worth which delay is a preference", "external constraints and objectives over the "
         "belief: $P(\\text{bruise})\\le 1\\%$, band within 2 \\% of the best day", "squeeze force, cut day"]],
        "FPICU dimensions of avocado manipulation, each with a place in the SDT and a consumer.", size="scriptsize"))
    w(table(["l", "p{0.22\\linewidth}", "X"], ["Event (patch)", "FrameNet frame", "Frame elements $\\to$ SDT nodes"], [
        ["Squeeze", "Manipulation", "Agent=Robot, Entity=Avocado, Bodypart\\_of\\_agent=Pads"],
        ["Bruise", "Damaging", "Patient=Mesocarp, Cause=Squeeze"],
        ["CutRoundSeed", "Cutting", "Agent=Robot, Item=Avocado, Instrument=Knife, Pieces=HalfA, HalfB"],
        ["TwistApart", "Separating", "Agent=Robot, Whole=Avocado, Parts=HalfA, HalfB"],
        ["RemoveSeed", "Removing", "Agent=Robot, Theme=Seed, Source=HalfA, Instrument=Spoon"],
        ["PeelSeedCoat", "Removing", "Theme=SeedCoatSheet, Source=Cavity, Instrument=Spoon"],
        ["Wait", "Waiting", "Protagonist=Robot, Expected\\_event=Ripe(Avocado)"],
        ["Spoil", "Rotting", "Patient=Avocado"]],
        "Frames for the avocado episode. Frame and element names are from FrameNet~1.7 as we recall them; unlike the "
        "egg report they are not checked against a pinned frame inventory, and no annotator was built here.",
        size="scriptsize"))
    w(r"""\section{Development record}
Faults found while building this microtheory, kept because they show how traces and evidence localized them:
(1) the first version of the learning loop hard-coded its hypothesis verdicts; the experiment showed that an
incomplete cut does tear ripe fruit through the bridge, contradicting the hard-coded 'rejected' -- verdicts are now
computed from the counts; (2) the development-seed list was shorter than the experiments' slices, so the peel-repair
experiment ran on no fruit and returned no estimate; (3) the first Task B planner waited to the end of its horizon
whenever waiting cost nothing; the band rule now takes the earliest day within two percentage points of the best; (4)
with a three-day horizon ripe fruit could not go overripe, so the band was one-sided; the horizon is five days; (5)
in the twin's renders, edited meshes lost their materials through untrimmed normals, lost texture coordinates and
empty material subsets (fixed in food/avocado\_peel.py); (6) the CRAM scenario's camera image reads the dark rendered
skin as riper than the colour law -- the same misspecification that Section~\ref{sec:abl} and the camera request of
Section~\ref{sec:learning} study.
\end{document}
""")
    open(TEX, "w").write(re.sub(r"(?<!\\)Gamma(\d)", r"Γ\1", "".join(out)))
    r_ = subprocess.run(["latexmk", "-lualatex", "-interaction=nonstopmode", "-halt-on-error", os.path.basename(TEX)],
                        cwd=HERE, capture_output=True, text=True)
    print(r_.stdout[-2500:] if r_.returncode else "ok", r_.returncode, f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
