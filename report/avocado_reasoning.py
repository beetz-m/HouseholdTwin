"""Report section: one avocado as a node a robot reasons with (photo -> probabilistic SDT node -> peeling monitor ->
halving / pitting emulator -> mental simulation and competence questions -> PR2 in the CRAM kitchen -> interactive
CRAMERA interface). Called by report/build_report.py with the builder module (its LaTeX and figure helpers); every
number is read from the result files named in the text.
"""

import json
import os
import re

import numpy as np

RIPE_RE = re.compile(r"ripeness_stage ([0-9.]+) \+- ([0-9.]+)")


def _j(B, *p):
    path = B.R(*p)
    return json.load(open(path)) if os.path.exists(path) else None


def _ripe(text):
    m = RIPE_RE.search(text or "")
    return (float(m.group(1)), float(m.group(2))) if m else None


# %% figures drawn by the report
def plot_pipeline(B):
    plt = B.mpl()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.1, 2.25, 2.1, 1.5, "perceive", "detection (CRAM)\nhead-camera photo\ngripper squeeze\nblade, twist, spoon\nforces"),
        (2.75, 2.25, 2.3, 1.5, "SDT node", "USD part graph\n4000-particle belief\ndiscrete state\n(integrity, seed, coat)"),
        (5.6, 2.25, 2.0, 1.5, "reason", "belief queries\ntask models\nmental simulation\nvalue of information"),
        (8.1, 2.25, 1.8, 1.5, "act", "PR2 motions\nin the CRAM\nkitchen"),
        (5.6, 0.2, 2.0, 1.35, "monitor", "imagined vs\nmeasured outcome\n-> verdict"),
    ]
    for x, y, w, h, title, body in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                                    fc="#eef4f2", ec="#2a6f62", lw=1.2))
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=9.5, weight="bold", color="#1d4d44")
        ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", fontsize=7.4, color="#222", linespacing=1.25)

    def arrow(a, b, txt=None, rad=0.0):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#52514e",
                                     connectionstyle=f"arc3,rad={rad}"))
        if txt:
            ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.12, txt, ha="center", fontsize=7, color="#52514e")
    arrow((2.2, 3.0), (2.75, 3.0), "Bayes")
    arrow((5.05, 3.0), (5.6, 3.0))
    arrow((7.6, 3.0), (8.1, 3.0), "plan")
    arrow((9.0, 2.25), (7.6, 0.9), "outcome", rad=-0.2)
    arrow((5.6, 0.9), (3.9, 2.25), "update", rad=-0.2)
    ax.text(5.0, 3.93, "the avocado's SDT node as the robot's belief state", ha="center", fontsize=8.5, color="#1d4d44")
    fig.tight_layout()
    out = os.path.join(B.FIG, "avocado_pipeline.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_pr2_belief(B, run, prior):
    plt = B.mpl()
    mon = run["monitoring"]
    pts = [("register prior", prior)]
    sq = [m for m in mon if m["step"] == "SqueezeTest"]
    if sq:
        pts.append(("head-camera photo", _ripe(sq[0]["belief_before"])))
    for m in mon:
        r = _ripe(m["belief_after"])
        if r:
            pts.append((f"{m['step']}\n({m['verdict']})", r))
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    x = np.arange(len(pts))
    mu = np.array([p[1][0] for p in pts])
    sd = np.array([p[1][1] for p in pts])
    ax.fill_between(x, mu - sd, mu + sd, color=B.PAL[0], alpha=0.18, lw=0, label="belief mean $\\pm$ sd")
    ax.plot(x, mu, "o-", color=B.PAL[0])
    ax.axhline(run["truth"]["stage"], color=B.PAL[1], ls="--", lw=1.5, label=f"hidden truth ({run['truth']['stage']:.2f})")
    for xi, (lab, _) in zip(x, pts):
        if "unexpected" in lab or "surprising" in lab:
            ax.annotate("not as\nexpected", (xi, mu[xi]), xytext=(xi - 0.6, mu[xi] + 0.9), fontsize=7.5,
                        color=B.PAL[7], arrowprops=dict(arrowstyle="-|>", color=B.PAL[7], lw=1))
    ax.set_xticks(x, [p[0] for p in pts], fontsize=7.2)
    ax.set_ylabel("ripeness stage (1 hard .. 5 ripe)")
    ax.set_ylim(1, 5.2)
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    out = os.path.join(B.FIG, "avocado_pr2_belief.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


# %% the section
def write(w, B):
    esc, fig, figrow, table, tt = B.esc, B.fig, B.figrow, B.table, B.tt
    obs = _j(B, "assets", "food", "avocado_photo_burlap", "observation.json")
    tl = _j(B, "assets", "food", "avocado_4", "belief", "timeline.json")
    peel = _j(B, "assets", "food", "avocado_4", "peel_set", "eval", "summary.json")
    proc = _j(B, "assets", "food", "avocado_4", "process", "summary.json")
    run = _j(B, "cram", "results", "pr2_run.json")
    from food import avocado_belief as AB
    from food import avocado_tasks as T
    prior_b = AB.AvocadoBelief.prior(4000, seed=0).summary()["ripeness_stage"]
    prior = (prior_b["mean"], prior_b["sd"])

    w(r"""\clearpage\section[An avocado as the robot's belief state]{An avocado as the robot's belief state: from perception to action}\label{sec:avocado-reasoning}
The sections before describe food items as part graphs with beliefs and truths. This section follows one item, the
Hass avocado, through the whole loop a robot needs: perceive it, keep a probabilistic belief about its hidden
properties, ask and answer questions about it, predict what an action will do, act, and check the outcome against the
prediction. Everything here is code in this repository and runs on the twin (\texttt{avocado\_4} as the hidden truth);
nothing is validated on a real fruit or a real robot.
""")
    w(fig(plot_pipeline(B), "The loop of this section. The SDT node (USD part graph, particle belief, discrete state) is "
          "the robot's belief state about the avocado; perception updates it by Bayes' rule, reasoning queries it, "
          "action outcomes are compared with what the node predicted and fed back.", r"0.26\textheight"))

    # ---------------------------------------------------------------- 1 photo
    w(r"""\subsection{From a photo to an SDT node}
\texttt{scripts/avocado\_from\_image.py} (\texttt{food/avocado\_vision.py}) estimates an avocado's node from one
photo: colour clustering picks the fruit-like blob, GrabCut refines it and its convex hull restores the shaded flank;
the white balance comes from the fruit's own specular highlights (dichromatic reflection: highlights carry the
illuminant's colour); the silhouette gives the aspect ratio and, by fitting the twin's meridian, the neck taper; the
skin's CIELAB lightness and chroma enter a ripeness likelihood (hue only as green / not green, because the register's
hue chart for ripe Hass is doubtful); skin relief and the stem button are read off as well. Every estimate is
labelled OBSERVED, ESTIMATED or PRIOR, and the observations go into the belief of the next subsection, from whose
posterior the instance is then built.
""")
    if obs:
        rows = []
        for k, e in obs["estimates"].items():
            if isinstance(e, dict) and "value" in e:
                v = e["value"]
                vs = f"{v:.3g}" if isinstance(v, float) else esc(v)
                sd = f"{e['sd']:.2g}" if isinstance(e.get("sd"), (int, float)) else "--"
                rows.append([tt(k), vs, sd, esc(e.get("unit", "")), esc(e.get("status", "")), esc(e.get("method", ""))])
        w(table(["p{0.17\\linewidth}", "r", "r", "l", "l", "X"], ["Estimate", "value", "sd", "unit", "status", "method"],
                rows, "Estimates from the burlap photo (assets/food/avocado\\_photo\\_burlap/observation.json).",
                size="scriptsize"))
    w(figrow([("renders/vision/avocado_photo_burlap_analysis.png", "analysis sheet of a real photo"),
              ("cram/camera/avocado_4_counter.png", "the PR2's head-camera view (Isaac Sim render of avocado\\_4)")],
             "Photo input. Left: the analysis of a user photo (segmentation, white balance, profile fit, skin colour). "
             "Right: the image the PR2 scenario uses as its camera view, rendered from the hidden-truth instance.",
             r"0.24\textheight"))
    if run:
        sq = [m for m in run["monitoring"] if m["step"] == "SqueezeTest"]
        ph = _ripe(sq[0]["belief_before"]) if sq else None
        if ph:
            w(r"""On the rendered camera view the shape estimates are close (aspect ratio and length within a few
per cent of the instance), but the ripeness reading is biased: the rendered skin is darker than the colour chart
expects for the true stage, and the photo moves the belief to stage """ + f"{ph[0]:.2f}~$\\pm$~{ph[1]:.2f}" +
              r""" while the truth is """ + f"{run['truth']['stage']:.2f}" + r""". Skin colour is a weak ripeness cue
(avo\_au\_stages: firmness is the better index); the likelihood is not calibrated on renders. Section~\ref{sec:pr2}
shows how the robot recovers from this.
""")

    # ---------------------------------------------------------------- 2 belief
    w(r"""\subsection{The node as a probability distribution: Bayesian filtering}
The avocado node is a particle filter (\texttt{food/avocado\_belief.py}): 4000 particles, each a complete candidate
avocado drawn from the register (length, diameter, neck, skin thickness, tubercle size, seed length, width and
offset, ripeness stage and ripening rate, skin friction, stem button, bruise flag, which side the seed coat sticks
to). Derived per particle: mass (body of revolution of the twin's meridian plus the seed, calibrated on the built
twins), aspect ratio, penetrometer force and flesh modulus (register ripeness model), squeeze stiffness (FEM table of
the previous section), seed and flesh mass. \emph{Prediction} is ripening at each particle's own rate; a squeeze
above a particle's bruise threshold flips its bruise flag. \emph{Measurement models} (noise ILLUSTRATIVE unless
stated): photo shape and skin colour, stem button, calliper, scale, gripper squeeze, the knife's stop depths round the
seed, the twist outcome and torque, the spoon's lever force, the extracted seed's length, the seed coat seen in the
cavity, a bruise seen after cutting. Resampling is systematic with a regularising jitter when the effective sample
size halves. The node is written into USD as \texttt{sdt\_belief.usda}: per variable mean, sd, 5/50/95~\% quantiles,
histogram and support as attributes of the part prims, plus the observation history and the particle file.
""")
    if tl:
        tr = tl["truth"]
        rows = []
        for i, s in enumerate(tl["timeline"]):
            S = s["summary"]
            rs = S["ripeness_stage"]
            ms = S["mass_g"]
            ts = tr["ripeness_stage"][i] if isinstance(tr["ripeness_stage"], list) else tr["ripeness_stage"]
            rows.append([esc(s["step"]), f"{s['time_h']:.0f}", f"{rs['mean']:.2f} $\\pm$ {rs['sd']:.2f}", f"{ts:.2f}",
                         f"{ms['mean']:.0f} $\\pm$ {ms['sd']:.0f}", f"{S['seed_length']['mean']:.1f} $\\pm$ "
                         f"{S['seed_length']['sd']:.1f}"])
        w(table(["X", "r", "r", "r", "r", "r"], ["Step", "t (h)", "ripeness belief", "truth", "mass (g)",
                                                  "seed length (mm)"], rows,
                f"Validation of the filter on avocado\\_4 with simulated noisy observations over four days "
                f"(scripts/avocado\\_belief\\_demo.py; truth: mass {tr['mass_g']:.0f}~g, seed "
                f"{tr['seed_length']:.1f}~mm).", size="scriptsize"))
    w(figrow([("assets/food/avocado_4/belief/timeline.png", "belief over time"),
              ("assets/food/avocado_4/belief/posteriors.png", "posteriors")],
             "The avocado node's belief while it ripens and is observed (assets/food/avocado\\_4/belief/).",
             r"0.24\textheight"))

    # ---------------------------------------------------------------- 3 peel monitor
    w(r"""\subsection{Watching the robot peel: a vision-based peeling monitor}
\texttt{food/peel\_monitor.py} judges a peeled avocado half (lying cut face down, one oblique RGB-D view) against the
twin's \emph{expectation}: the ideal peeled surface of this instance, registered to the observed depth (trimmed ICP
and a height refinement over $x$, $y$ and yaw). Every visible surface point is classed as peeled well, skin remaining
(pebbled texture, darker, or standing proud of the expected surface), flesh missing (deeper than expected, with a
slope-dependent threshold and hysteresis over the torn area) or not judged. From the classes come the measures
(peeled share, skin area and largest patch, missing area, lost volume, deepest gouge, coverage), a score from 0 to
100 and a verdict (success / rework / not judged) with reasons, written into the node. The test set is synthetic:
\texttt{food/avocado\_peel.py} turns peeling outcomes into USD states (skin patches kept on the flesh, gouges cut out
of it by mesh booleans, ground truth recorded) and Isaac Sim renders them with depth and segmentation, the
expectation at a perturbed pose (2--4~mm, 2--4$^\circ$) so that the monitor must register it itself. The thresholds
are calibrated on these renders only.
""")
    if peel:
        S = peel["summary"]

        def row(mode, name):
            m = S[mode]
            c = m["confusion"]
            return [name, f"{sum(c.values()) - c['rework_missed'] - c['false_alarm']}/{sum(c.values())}",
                    f"{c['rework_found']}/{c['rework_found'] + c['rework_missed']}", str(c["false_alarm"]),
                    f"{m['mean_iou_skin']:.2f}", f"{m['mean_precision_skin']:.2f} / {m['mean_recall_skin']:.2f}",
                    f"{m['skin_area_mae_mm2']:.0f}", f"{m['mean_iou_missing']:.2f}",
                    f"{m['mean_precision_missing']:.2f} / {m['mean_recall_missing']:.2f}",
                    f"{m['missing_area_mae_mm2']:.0f}"]
        w(table(["l", "r", "r", "r", "r", "r", "r", "r", "r", "r"],
                ["Input", "verdicts", "reworks found", "false alarms", "IoU skin", "P / R skin", "err (mm2)",
                 "IoU missing", "P / R missing", "err (mm2)"], [row("rgbd", "RGB-D"), row("rgb", "RGB only")],
                f"Peeling monitor on {S['n']} synthetic peels (6 clean, 6 with skin patches, 6 with gouges, 6 mixed; "
                "peel\\_set/eval/summary.json). IoU, precision and recall are means over the samples that contain the "
                "defect; err: mean absolute area error.", size="scriptsize"))
        w(r"""Depth carries the gouges: from colour alone the missing flesh is not found at all, while the skin patches,
being darker and pebbled, are found from colour too. While building the set, three rendering faults of edited meshes
had to be fixed before the evaluation meant anything: per-corner normals that were not trimmed with the faces, texture
coordinates lost in the mesh booleans, and material subsets that became empty -- each made the renderer drop the
mesh's material, so the skin patches came out pale.
""")
    w(fig("assets/food/avocado_4/peel_set/eval/panels.png", "Peeling monitor examples (left: render; middle: ground "
          "truth; right: the monitor's classes, red skin remaining, orange flesh missing, green peeled well).",
          r"0.55\textheight", maxw=1600))
    w(fig("assets/food/avocado_4/peel_set/eval/scatter.png", "Skin and missing areas: monitor against ground truth, "
          "RGB-D and RGB only.", r"0.22\textheight"))

    # ---------------------------------------------------------------- 4 emulator
    w(r"""\subsection{Halving and pitting: an emulator}
The emulator (\texttt{food/avocado\_emulator.py}, mechanics in \texttt{food/avocado\_process.py}) is a state machine
on the node: whole $\to$ scored / partly scored $\to$ halved / halved with torn flesh $\to$ seed out, the seed coat on
the seed or left in the cavity. Each action checks its precondition, runs quasi-static mechanics on the hidden
instance and writes the new USD states, and returns what a robot would sense (\emph{measured}) apart from the truth:
""")
    w(table(["p{0.18\\linewidth}", "X", "p{0.27\\linewidth}"], ["Action", "Mechanics (constants ILLUSTRATIVE)", "USD state"], [
        [tt("cut_round_seed"), "puncture at the stem end; force = skin resistance $\\times$ skin thickness + flesh "
         "cutting toughness(stage) $\\times$ depth; the blade stops on the seed coat (stop-depth profile); a force limit "
         "can stall the blade", "kerf carved from skin and flesh down to the seed coat; cut faces labelled"],
        [tt("twist_apart"), "torque = uncut flesh bridges (polar moment $\\times$ flesh shear strength) + seed "
         "adhesion over half the seed; ripe fruit releases at the seed, unripe tears ($p_\\text{clean}$(stage))",
         "two halves; torn flesh as holes in the cavity wall"],
        [tt("remove_seed"), "spoon lever (0.3 $\\times$ adhesion $\\times$ area), knife twist (flagged hazardous for a "
         "hand-held half, farley2019), squeeze-out (ripe only); the spoon may gouge; the seed coat stays in the cavity "
         "when it adheres to the flesh", "pitted half with seed-coat sheets and gouges; the seed"]],
        "Emulator actions.", size="scriptsize"))
    T_rows = [[tt(n), esc(t["goal"]), esc(", ".join(t["failures"]))] for n, t in T.TASKS.items()]
    w(table(["p{0.15\\linewidth}", "X", "p{0.25\\linewidth}"], ["Designator", "Goal", "Failures (cue, recovery)"],
            T_rows, "Avocado task designators (food/avocado\\_tasks.py); CutRoundSeed is coraplex's CuttingAction "
            "with the technique HALVING.", size="scriptsize"))
    if proc:
        rows = []
        for sc, d in proc["scenarios"].items():
            for s in d["steps"]:
                b = s.get("belief") or {}
                rows.append([esc(sc), tt(s["task"]), esc(s["outcome"]),
                             f"{b['ripeness_stage'][0]:.2f} $\\pm$ {b['ripeness_stage'][1]:.2f}" if b else "--",
                             f"{b['seed_length'][0]:.1f} $\\pm$ {b['seed_length'][1]:.1f}" if b else "--",
                             f"{b['testa_adheres_to_flesh']:.2f}" if b else "--"])
        t = proc["truth"]
        w(table(["l", "l", "l", "r", "r", "r"], ["Scenario", "Step", "Outcome", "ripeness belief", "seed (mm)",
                                                 "P(coat on flesh)"], rows,
                f"Three emulator runs on avocado\\_4 (process/summary.json; truth: stage {t['ripeness_stage']:.2f}, "
                f"forced to 1.8 in the unripe run; seed {t['seed_length']:.1f}~mm; seed coat adheres to the "
                f"{esc(t['testa_adheres_to'])}). The belief starts from the register prior and is updated only from the "
                "measured part of each outcome.", size="scriptsize"))
    w(figrow([("assets/food/avocado_4/process/ripe/render_scored.png", "cut round the seed"),
              ("assets/food/avocado_4/process/ripe/render_halves.png", "halves (torn twist)"),
              ("assets/food/avocado_4/process/unripe/render_pitted.png", "pitted, seed, other half")],
             "Emulator states, path-traced in Isaac Sim from the USD states the emulator writes.", r"0.17\textheight"))

    # ---------------------------------------------------------------- 5 mental simulation
    w(r"""\subsection{Competence questions and mental simulation}
With the node as its belief state, the robot answers competence questions from three kinds of knowledge
(\texttt{cram/avocado\_sdt.py}): the \emph{belief} (what is it, how ripe and when edible, where is the seed and how
deep will the blade go), the \emph{task models} (what can I do now, what am I doing and what can go wrong), and
\emph{mental simulation}: the halving and pitting mechanics run on avocados drawn from the belief. An imagined
avocado needs only the geometry the mechanics use, which takes 1~ms to build instead of 1~s for the full twin, so a
question can roll out 150 imagined avocados in seconds. Answers carry the evidence they rest on.

\paragraph{Should it squeeze first? Value of information} The goal is ``guacamole now, or store the avocado two
days''. Utility = quality of the pitted halves (penalties for torn flesh, seed coat left, gouges) $\times$ how well
the flesh mashes at preparation time (0 below stage 2.5, 1 from stage 4; ILLUSTRATIVE), waiting costs 0.05. The robot
first picks the squeeze force: the strongest of 4, 3 and 2~N whose bruise risk under the belief is below 2~\% (bruise
threshold 0.6 $\times$ penetrometer force, register). Then a preposterior analysis imagines squeeze readings for the
imagined avocados (the belief's own measurement model; sensor error grows as the force falls), updates on each and
asks whether the decision changes and by how much the expected utility rises.
""")
    if run:
        seen, rows = set(), []
        for a in run["answers"]:
            if a["question"] in seen or a["question"].startswith("Did "):
                continue
            seen.add(a["question"])
            ans = a["answer"]
            rows.append([esc(a["question"]), esc(a["method"]), esc(ans[:330] + (" ..." if len(ans) > 330 else ""))])
        w(table(["p{0.2\\linewidth}", "p{0.12\\linewidth}", "p{0.62\\linewidth}"], ["Question", "Answered from",
                                                                                        "Answer"], rows,
                f"The robot's competence answers in the PR2 scenario (cram/results/pr2\\_run.json, recorded "
                f"{esc(run['when'])}).", size="scriptsize", long=True))
        rows = [[esc(i["plan"]), f"{i['wait_h']:.0f}", str(i["n"]), f"{i['p_clean_halves']:.2f}", f"{i['p_torn']:.2f}",
                 f"{i['p_seed_coat_left']:.2f}", f"{i['p_gouge']:.2f}", "yes" if i["hazardous"] else "no",
                 f"{i['quality']:.2f}"] for i in run["imagined"]]
        w(table(["X", "r", "r", "r", "r", "r", "r", "l", "r"], ["Plan", "wait (h)", "n", "P(clean)", "P(torn)",
                                                                "P(coat left)", "P(gouge)", "hazard", "quality"], rows,
                "Imagined outcomes of the plans the robot compared (after the photo and the squeeze).",
                size="scriptsize"))

    # ---------------------------------------------------------------- 6 PR2
    w(r"""\subsection{The PR2 in the CRAM kitchen}\label{sec:pr2}
\texttt{cram/avocado\_pr2\_demo.py} puts the node into the CRAM framework (\texttt{\textasciitilde/WORK}, imported, not
changed): the apartment kitchen of the CRAM resources with the PR2, an avocado on a 4~cm cutting board on the
counter, a knife in the PR2's right gripper and a spoon on the counter. \texttt{Avocado(Fruit)} is a semantic
annotation of the semantic digital twin that carries the robot's node; the twin's meshes are exported as coloured
part bodies (skin, three flesh bands, endocarp, seed coat) for each state (\texttt{cram/export\_meshes.py}).
The PR2 drives to the counter, looks at the board and runs coraplex's \texttt{DetectAction}; the detection creates the
node from the register prior and the head-camera image is folded in. The robot then answers the questions above,
decides whether to squeeze, squeezes with the left gripper (fingers set to the believed diameter minus force over
believed stiffness; the kinematic simulation has no force control, so the reading comes from the fruit model), and
halves and pits the avocado with its own motions (\texttt{cram/avocado\_motions.py}): the left hand holds the neck,
the knife held flat in the horizontal cut plane moves its tip once round the seed outline of the belief means
(\texttt{CutRoundSeedAction}, a coraplex \texttt{CuttingAction} with its own tool path), the knife is laid down, the
left hand pinches the lower half, the right hand twists the upper half off about the cut-plane normal and puts it
down, then takes the spoon, slides it down beside the seed and under it, levers it out and drops it on the board.
Forces and outcomes come from the mechanics run on the hidden true avocado; after each step the measured outcome is
placed in the imagined distribution (as expected; unexpected if $P<0.2$; surprising if $P<0.05$ or outside the
95~\% band), the belief is updated and the world shows the new pieces.

Making the motions reachable shaped the scene: from the first stance the PR2 could not hold the knife flat close to
its body at board height (it stands 15~cm further back now); a flat knife at counter height drove the forearm into
the counter (hence the board); the cut-plane alignment pointed the gripper back at the robot until one axis was
flipped; and the spoon scoops from the far side of the seed towards the robot, lifting up and back out of the cavity.
""")
    if run:
        rows = [[tt(m["step"]), esc(m["expected"]), esc(m["measured"]), esc(m["verdict"]),
                 esc(m["belief_after"])] for m in run["monitoring"]]
        w(table(["p{0.13\\linewidth}", "p{0.22\\linewidth}", "p{0.22\\linewidth}", "l", "X"],
                ["Step", "Imagined", "Measured", "Verdict", "Belief after"], rows,
                "Execution monitoring in the PR2 scenario (cram/results/pr2\\_run.json).", size="scriptsize"))
        w(fig(plot_pr2_belief(B, run, prior), "The robot's ripeness belief through the PR2 scenario: the photo makes the "
              "avocado look riper than it is, the squeeze (a weak cue) corrects little, the plan predicts clean halves, "
              "the twist tears the flesh -- not as expected -- and the belief falls towards the truth.",
              r"0.26\textheight"))
        tr = run["truth"]
        fin = run["variables"]
        fv = {v["variable"]: v for v in fin}
        w(r"""At the end the node holds ripeness """ + f"{fv['ripeness_stage']['mean']:.2f}~$\\pm$~"
          f"{fv['ripeness_stage']['sd']:.2f}" + r""" (truth """ + f"{tr['stage']:.2f}" + r"""), seed length """ +
          f"{fv['seed_length']['mean']:.1f}~$\\pm$~{fv['seed_length']['sd']:.1f}~mm" + r""" (truth """ +
          f"{tr['seed_length']:.1f}" + r"""~mm) and $P(\text{seed coat sticks to the flesh}) = """ +
          f"{fv['testa_adheres_to_flesh']['mean']:.2f}" + r"""$ (truth: """ + esc(tr["testa_adheres_to"]) + r"""), from """ +
          f"{len(run['answers'])} answers, {len(run['imagined'])} imagined plans and {len(run['monitoring'])} monitored "
          r"""steps. The decision itself is honest about its context: the robot would store the avocado for two days;
the scenario asks for now, so it proceeds and expects torn flesh with low probability -- which is exactly where it is
wrong, because the photo misled it.
""")

    # ---------------------------------------------------------------- 7 interface
    w(r"""\subsection{An interactive kitchen in CRAMERA}
\texttt{cram/avocado\_kitchen.py} keeps the same kitchen live in CRAMERA, the CRAM viewer (3D scene, EQL query panel,
plan graph), and lets an operator drive it from the query panel. Commands and questions are ready-made queries
(buttons); typed requests are matched to them:
""")
    w(table(["p{0.2\\linewidth}", "X"], ["Kind", "Entries"], [
        ["commands (queued, executed with PR2 motions)", "Detect the avocado; Squeeze the avocado; Wait two days; Cut "
         "round the seed; Twist the halves apart; Spoon out the seed; Reset the kitchen"],
        ["questions answered at once", "What is the object? How ripe is it? Where is the seed? What can you do now? "
         "What are you doing?"],
        ["questions answered by mental simulation", "Should you squeeze it first? Guacamole now, or store it? What "
         "happens if you halve it now?"],
        ["views", "the robot's answers, its belief, what it imagined, how the execution went, the command log, the "
         "commands with their availability"],
        ["typed phrasings (examples)", "squeeze it; halve it; open the avocado; remove the seed; is it ripe; what can "
         "you do; start over"]], "The operator's controls in the CRAMERA query panel.", size="scriptsize"))
    w(r"""A command is checked against the robot's and the avocado's state and refused with the reason (``the halves
have to be apart first'', ``I have not detected an avocado yet''), otherwise queued and executed by a worker while
the viewer follows the robot. ``Wait two days'' ripens the hidden avocado and the belief; a twist after waiting
usually comes out clean. Two details of CRAMERA shaped the design: the viewer builds every preset's code to word it
in English, so a command fires only when the query is evaluated (the command sits in an attribute read by the query's
filter); and the world records every annotation it adds or removes as JSON, so the node is kept beside the
annotation, not in it. Running beside another CRAMERA session needs its own bridge port, data directory and viewer.
No screenshot of the 3D view is included: the headless browser available here has no WebGL.
""")
    w(r"""\paragraph{What this section does not show} No real avocado, no real robot. The kinematic simulation has no
contacts: knife, fingers and spoon pass through the fruit, and the mechanics supply forces and outcomes. The
mechanics' constants, the utility and the photo's colour likelihood are ILLUSTRATIVE; the peeling monitor's
thresholds are calibrated on the twin's own renders. The robot's seed-centre estimate is the twin's nominal position
(no sideways offset), and torn flesh is not shown on the seed piece.
""")
