"""Report: semantic digital twins of kitchen food items -- what we have so far.

Assembled from the repository (parameter register, built USD assets, bench reports in test_reports/, research notes in
docs/, existing renders), in the style of ~/Documents/CuxhavenSiteTwin/report/build_report.py, so it can be rebuilt as
the work moves.  CPU only: it never starts Isaac Sim; it reads existing renders and writes its own matplotlib plots and
downscaled JPEG copies of the renders to report/figures/.

    ~/Documents/LabAutomationTwin/env_cad/bin/python report/build_report.py            # report/kitchen_food_twins.pdf
    ~/Documents/LabAutomationTwin/env_cad/bin/python report/build_report.py --refresh  # re-run the cached CPU checks
"""

import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from food import params as PR  # noqa: E402

R = lambda *p: os.path.join(ROOT, *p)  # noqa: E731
TEX = os.path.join(HERE, "kitchen_food_twins.tex")
FIG = os.path.join(HERE, "figures")
CACHE = os.path.join(HERE, "cache")
REFRESH = "--refresh" in sys.argv
os.makedirs(FIG, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
CHECKS = []          # (check, result) collected while building, printed in the data-quality section

# ================================================================================================ LaTeX helpers
_ESC = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#",
        "$": r"\$", "~": r"$\sim$", "^": r"\^{}", "->": r"$\to$", "<=": r"$\le$", ">=": r"$\ge$", "+-": r"$\pm$"}
_ESC_RE = re.compile(r"->|<=|>=|\+-|[\\{}_&%#$~^]")


def esc(s):
    return _ESC_RE.sub(lambda m: _ESC[m.group(0)], str(s))


def tt(s):
    return r"\texttt{" + esc(s).replace(r"\_", r"\_\allowbreak{}") + "}"


def img(path, maxw=1800, quality=85):
    """Project-relative image -> downscaled JPEG copy in report/figures (PDF figures pass through)."""
    src = R(path)
    if not os.path.exists(src):
        return None
    if src.endswith(".pdf"):
        return src
    from PIL import Image
    name = path.replace("/", "_").rsplit(".", 1)[0] + ".jpg"
    dst = os.path.join(FIG, name)
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
        im = Image.open(src)
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", im.size, "white")
            bg.paste(im.convert("RGBA"), mask=im.convert("RGBA").split()[3])
            im = bg
        im = im.convert("RGB")
        if im.width > maxw:
            im = im.resize((maxw, round(im.height * maxw / im.width)), Image.LANCZOS)
        im.save(dst, quality=quality, optimize=True)
    return dst


def fig(path, caption, height=r"0.42\textheight", pos="htbp", maxw=1800):
    p = path if os.path.isabs(path) else img(path, maxw)
    if not p or not os.path.exists(p):
        return f"% missing figure {path}\n"
    return (f"\\begin{{figure}}[{pos}]\\centering\\includegraphics[width=\\linewidth,height={height},keepaspectratio]"
            f"{{{p}}}\\caption{{{caption}}}\\end{{figure}}\n")


def figrow(items, caption, height=r"0.2\textheight", maxw=1400):
    rows = []
    for p, s in items:
        q = p if os.path.isabs(p) else img(p, maxw)
        if q and os.path.exists(q):
            rows.append((q, s))
    if not rows:
        return ""
    w = f"{0.98 / len(rows):.3f}\\linewidth"
    cells = [f"\\begin{{minipage}}[t]{{{w}}}\\centering\\includegraphics[width=\\linewidth,height={height},"
             f"keepaspectratio]{{{p}}}\\\\[-1pt]{{\\scriptsize {sub}}}\\end{{minipage}}" for p, sub in rows]
    return "\\begin{figure}[htbp]\\centering" + "\\hfill".join(cells) + f"\\caption{{{caption}}}\\end{{figure}}\n"


def table(cols, head, rows, caption, size="footnotesize", long=False):
    pct = lambda t: re.sub(r"(?<!\\)%", r"\\%", str(t))  # noqa: E731  unescaped % would comment out the row
    head = [pct(h) for h in head]
    rows = [[pct(c) for c in r] for r in rows]
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    if long:
        cols = [c if c != "X" else "p{0.26\\linewidth}" for c in cols]
        hd = f"{' & '.join(head)} \\\\\\midrule"
        cont = re.sub(r"\\label\{[^}]*\}", "", caption)     # the label only on the first page (no duplicate)
        return (f"{{\\{size}\\begin{{longtable}}{{@{{}}{''.join(cols)}@{{}}}}\\caption{{{caption}}}\\\\\\toprule "
                f"{hd}\\endfirsthead\n\\caption[]{{{cont} (continued)}}\\\\\\toprule {hd}\\endhead\n"
                f"{body} \\\\\\bottomrule\\end{{longtable}}}}\n")
    return (f"\\begin{{table}}[htbp]\\centering\\{size}\\caption{{{caption}}}\n\\begin{{tabularx}}{{\\linewidth}}"
            f"{{@{{}}{''.join(cols)}@{{}}}}\\toprule {' & '.join(head)} \\\\\\midrule\n{body} \\\\\\bottomrule"
            f"\\end{{tabularx}}\\end{{table}}\n")


def code(text, title=None):
    t = f"\\par\\medskip\\noindent{{\\small\\sffamily\\bfseries {title}}}\\par\\nopagebreak\\vspace{{2pt}}" if title else "\\par"
    return (t + "\\begin{Verbatim}[fontsize=\\scriptsize,frame=lines,framesep=2mm,breaklines]\n" + text.rstrip()
            + "\n\\end{Verbatim}\n")


def num(x, d=1):
    return f"{x:,.{d}f}"


def pct(a, b, d=0):
    return f"{100 * a / b:.{d}f}~\\%"


# ================================================================================================ register helpers
STATUS_ORDER = ["STANDARD", "SOURCED", "CALIBRATED", "DERIVED", "SOURCED-VERIFY", "TYPICAL", "ILLUSTRATIVE"]
STATUS_MEANING = OrderedDict([
    ("SOURCED", "value or range taken from the cited source"),
    ("SOURCED-VERIFY", "value from a source whose details (species, cultivar, protocol) still need checking in the "
                       "full text"),
    ("STANDARD", "a regulation or grading standard (definitional, not a measurement)"),
    ("DERIVED", "computed from other register entries by a stated physical model"),
    ("TYPICAL", "common textbook value, no citation held in this repository yet"),
    ("CALIBRATED", "free model constant fitted to a cited observation (the fit and its residual are stated)"),
    ("ILLUSTRATIVE", "twin assumption chosen to be plausible; must be calibrated before any quantitative claim")])


def fmt_val(v):
    if isinstance(v, tuple) and v and isinstance(v[0], str):
        k = v[0]
        if k == "normal":
            return f"N({v[1]:g}, {v[2]:g}) in [{v[3]:g}, {v[4]:g}]"
        if k == "uniform":
            return f"U({v[1]:g} .. {v[2]:g})"
        if k == "int_uniform":
            return f"integer {v[1]:g} .. {v[2]:g}"
        if k == "lognormal":
            return f"lognormal, median {v[1]:g}, factor {v[2]:g}"
        if k in ("choice", "choice_str"):
            vals = list(dict.fromkeys(str(x) if k == "choice_str" else f"{x:g}" for x in v[1:]))
            return "one of " + ", ".join(vals)
    if isinstance(v, tuple):
        return "(" + ", ".join(f"{x:g}" for x in v) + ")"
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, list):
        return "[" + ", ".join(f"{x:g}" for x in v) + "]"
    return str(v)


def reg_row(kind, key):
    u, v, s, src = PR.REGISTERS[kind][key]
    unit = "" if u in ("-", "1") else f" {u}"
    return [tt(key), esc(fmt_val(v) + unit), esc(s), esc(src)]


def keytable(kind, keys, caption):
    rows = [reg_row(kind, k) for k in keys if k in PR.REGISTERS[kind]]
    return table(["p{0.19\\linewidth}", "p{0.27\\linewidth}", "p{0.12\\linewidth}", "X"],
                 ["Register key", "Value / distribution", "Status", "Source and note"], rows, caption,
                 size="scriptsize")


def src_keys(text):
    return [t for t in re.findall(r"[A-Za-z][A-Za-z0-9_]+", text) if t in PR.SOURCES]


# ================================================================================================ project data
def jload(*p):
    return json.load(open(R(*p)))


ITEMS = OrderedDict([   # kind: (name, wave, asset instance, docs)
    ("egg", ("Hen egg", "egg set", "egg_1", "docs/ANATOMY.md, docs/EGG_TASKS.md")),
    ("apple", ("Apple (Gala-like)", "egg set", "apple_3", "docs/ANATOMY.md")),
    ("avocado", ("Hass avocado", "egg set", "avocado_4", "docs/ANATOMY.md")),
    ("roll", ("Wheat roll (Weizenbrötchen)", "wave 1", "roll_0", "docs/wave1/ROLL.md")),
    ("banana", ("Cavendish banana", "wave 1", "banana_0", "docs/wave1/BANANA.md")),
    ("carrot", ("Nantes carrot", "wave 1", "carrot_0", "docs/wave1/CARROT.md")),
    ("loaf", ("Sandwich loaf (Toastbrot)", "wave 2", "loaf_0", "docs/wave2/LOAF.md")),
    ("onion", ("Yellow onion", "wave 2", "onion_0", "docs/wave2/ONION.md")),
    ("orange", ("Sweet orange", "wave 2", "orange_0", "docs/wave2/ORANGE.md")),
    ("potato", ("Potato", "wave 2", "potato_0", "docs/wave2/POTATO.md")),
    ("tomato", ("Round / truss tomato", "wave 2", "tomato_15", "docs/wave2/TOMATO.md")),
    ("lemon", ("Lemon", "wave 3", None, "docs/wave3/LEMON.md (missing)")),
    ("cucumber", ("Greenhouse cucumber", "wave 3", None, "docs/wave3/CUCUMBER.md (missing)")),
    ("croissant", ("Butter croissant", "wave 3", None, "docs/wave3/CROISSANT.md (missing)")),
    ("baguette", ("Baguette", "wave 3", None, "docs/wave3/BAGUETTE.md (missing)")),
    ("pepper", ("Bell pepper", "wave 3", None, "docs/wave3/PEPPER.md (missing)")),
])
WAVE3_CLS = {"lemon": "Lemon", "cucumber": "Cucumber", "croissant": "Croissant", "baguette": "Baguette",
             "pepper": "Pepper"}
# wave 3 was built and rendered on 20 Oct 2026: take the built instance where it exists
for _k in WAVE3_CLS:
    if os.path.exists(R("assets", "food", f"{_k}_0", "instance.json")):
        ITEMS[_k] = (ITEMS[_k][0], ITEMS[_k][1], f"{_k}_0", ITEMS[_k][3])
W3_BUILT = all(os.path.exists(R("assets", "food", f"{_k}_0", "instance.json")) for _k in WAVE3_CLS)
W2_RENDERED = all(os.path.exists(R("renders", "food", f"item_{_k}.png"))
                  for _k in ("loaf", "onion", "orange", "potato", "tomato"))


def instances():
    out = OrderedDict()
    for d in sorted(os.listdir(R("assets", "food"))):
        p = R("assets", "food", d, "instance.json")
        if os.path.exists(p):
            out[d] = json.load(open(p))
    return out


def part_attrs(inst):
    """sdt:class / sdt:observability per part path from the built USD (whole.usdc)."""
    try:
        from pxr import Usd
    except ImportError:
        return {}
    st = Usd.Stage.Open(R("assets", "food", inst, "whole.usdc"))
    out = {}
    root = st.GetPrimAtPath("/Food/Parts")
    if not root:
        return out
    for p in Usd.PrimRange(root):
        a = {x.GetName(): x.Get() for x in p.GetAttributes() if x.GetName().startswith("sdt:")}
        if "sdt:class" in a:
            rel = str(p.GetPath())[len("/Food/Parts/"):]
            out[rel] = (a.get("sdt:class"), a.get("sdt:observability"))
    return out


def collapse_tree(rows):
    """rows: [(path, cls, obs, mass or None)] -> collapsed rows with indexed siblings merged:
    [(depth, name, cls, obs, count, mass_sum)]."""
    groups = OrderedDict()
    for path, cls, obs, mass in rows:
        segs = path.split("/")
        cseg = [re.sub(r"_(\d+)$", "_k", s) for s in segs]
        key = "/".join(cseg)
        g = groups.setdefault(key, {"depth": len(segs) - 1, "name": cseg[-1], "cls": cls, "obs": obs,
                                    "members": set(), "mass": 0.0, "has_mass": False})
        g["members"].add("/".join(segs))
        if mass is not None:
            g["mass"] += mass
            g["has_mass"] = True
    return [(g["depth"], g["name"], g["cls"], g["obs"], len(g["members"]), g["mass"] if g["has_mass"] else None)
            for g in groups.values()]


def tree_table(kind, inst_name, data, caption):
    attrs = part_attrs(inst_name)
    rows = []
    for path, p in data["parts"].items():
        cls, obs = attrs.get(path, ("", ""))
        if p.get("group"):
            mass = None
        elif "points" in p:
            mass = None
        else:
            mass = p.get("mass_g")
        rows.append((path, cls or "", obs or "", mass))
    pts = {path: p.get("points") or p.get("count") for path, p in data["parts"].items() if "points" in p}
    out = []
    for depth, name, cls, obs, n, mass in collapse_tree(rows):
        label = "\\hspace*{" + f"{0.9 * depth:.1f}" + "em}" + esc(name.replace("_k", "_1.." + str(n)) if n > 1 else name)
        note = ""
        for pth, cnt in pts.items():
            if pth.split("/")[-1] == name and cnt is not None:
                note = f"{cnt:,} points"
        out.append([label, tt(cls) if cls else "", esc(obs), str(n) if n > 1 else "",
                    (f"{mass:.2f}" if mass is not None else "") or esc(note)])
    return table(["p{0.30\\linewidth}", "p{0.27\\linewidth}", "p{0.17\\linewidth}", "r", "X"],
                 ["Part (below /Food/Parts)", "sdt:class", "observability", "n", "mass (g)"], out, caption,
                 size="scriptsize", long=len(out) > 30)


def md_table(path, heading):
    """First markdown table after a heading line that contains `heading`."""
    lines = open(R(path)).read().splitlines()
    rows, on = [], False
    for ln in lines:
        if ln.startswith("#") and heading in ln:
            on = True
            continue
        if on and ln.startswith("#") and rows:
            break
        if on and ln.startswith("|"):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            rows.append(cells)
        elif on and rows and not ln.startswith("|"):
            break
    return rows


def mdclean(s):
    return esc(re.sub(r"\*\*|`", "", s))


# ================================================================================================ CPU checks
def cached(name, fn):
    p = os.path.join(CACHE, name + ".json")
    if os.path.exists(p) and not REFRESH:
        return json.load(open(p))
    t = time.time()
    out = fn()
    out = json.loads(json.dumps(out, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    out["_runtime_s"] = round(time.time() - t, 1)
    out["_date"] = time.strftime("%Y-%m-%d %H:%M")
    json.dump(out, open(p, "w"), indent=1)
    return out


def run_designator_demos():
    from food import wave1_tasks as W1, wave2_tasks as W2
    out = {}
    for mod in (W1, W2):
        for k, v in mod.demo(0).items():
            out[k] = {"ok": bool(v["ok"]) if isinstance(v, dict) else None,
                      "state": v.get("state") if isinstance(v, dict) else None,
                      "trace": str(v.get("trace"))[:300] if isinstance(v, dict) else str(v)[:300]}
    return {"demos": out}


def run_wave3_parts():
    import importlib
    out = {}
    for kind, cls in WAVE3_CLS.items():
        try:
            m = importlib.import_module("food." + kind)
            b = getattr(m, cls)(0)
            out[kind] = {"ok": True, "parts": [[p.path, p.cls, p.observability] for p in b.parts()]}
        except Exception as e:  # noqa: BLE001
            out[kind] = {"ok": False, "err": repr(e)[:300]}
    return out


def crack_repro():
    """Re-run tools/egg_crack_study.py's computation in memory with the current register (nothing written)."""
    from food import egg_crack as EC
    from food.dynamics import EggTruth
    from food.egg import Egg
    N = 400
    TOOLS = {"flat counter (granite)": {"kind": "flat", "surface": "granite"},
             "thin bowl rim (r 1.5 mm)": {"kind": "rim", "surface": "bowl_rim_ceramic", "edge_radius": 1.5e-3},
             "thick bowl rim (r 4 mm)": {"kind": "rim", "surface": "bowl_rim_ceramic", "edge_radius": 4.0e-3},
             "another egg": {"kind": "egg"}}
    V = np.round(np.linspace(0.05, 1.5, 30), 3)
    rng = np.random.default_rng(7)
    eggs = [Egg(5000 + i) for i in range(N)]
    masses = [e.V_outer * 1087 for e in eggs]
    out = {}
    for tname, tool in TOOLS.items():
        best = (0.0, 0.0)
        for v in V:
            ok = 0
            for i, e in enumerate(eggs):
                t = EggTruth(e, masses[i])
                other = EggTruth(eggs[(i + 1) % N], masses[(i + 1) % N]) if tool["kind"] == "egg" else None
                r = EC.strike(t, float(v), tool, rng, other=other)
                target = t if r.which == "this" else other
                if r.state in ("check", "cracked_open"):
                    opened, yolk, shards = EC.open_up(target, rng, age_days=e.p["age_days"])
                    left = EC.fish_out(shards, rng)
                    ok += int(opened and yolk == "whole" and left == 0)
            if ok / N > best[1]:
                best = (float(v), ok / N)
        out[tname] = best
    return {"best": out, "m_eff_fraction": PR.EGG["impact_effective_mass_fraction"][1]}


def avocado_belief():
    """The belief numbers of tools/plot_avocado_squeeze.py (same seeds), recomputed without writing the figure."""
    from food.sdt import AvocadoBelief, squeeze_table_model
    kmod = squeeze_table_model(R("test_reports", "avocado_squeeze.json"))
    rng = np.random.default_rng(4)
    true = 4.2
    k_obs = kmod(true, 4.0) * (1 + rng.normal(0, 0.08))
    B = AvocadoBelief(rng, stiffness_model=lambda s, F: kmod(s, F), sigma_rel=0.10)
    B.observe_stiffness(k_obs, 4.0)
    s1 = B.summary()
    B2 = AvocadoBelief(np.random.default_rng(5), stiffness_model=lambda s, F: kmod(s, F), sigma_rel=0.10)
    B2.observe_stiffness(k_obs, 4.0)
    B2.observe_stiffness(kmod(true, 2.0) * (1 + rng.normal(0, 0.08)), 2.0)
    return {"true": true, "k_obs_N_per_mm": float(k_obs) / 1e3, "one": s1, "two": B2.summary()}


# ================================================================================================ plots
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2})
    return plt


def plot_status():
    plt = mpl()
    kinds = list(PR.REGISTERS)
    names = ["all items"] + kinds
    data = {s: [] for s in STATUS_ORDER}
    tot = Counter()
    for k in kinds:
        tot += PR.status_counts(k)
    for k in [None] + kinds:
        c = tot if k is None else PR.status_counts(k)
        n = sum(c.values())
        for s in STATUS_ORDER:
            data[s].append(c.get(s, 0) / n)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    y = np.arange(len(names))[::-1]
    left = np.zeros(len(names))
    for i, s in enumerate(STATUS_ORDER):
        v = np.array(data[s])
        ax.barh(y, v, left=left, color=PAL[i], edgecolor="white", linewidth=1.0, height=0.72, label=s)
        left += v
    for yi, v in zip(y, data["ILLUSTRATIVE"]):
        ax.text(1.01, yi, f"{100 * v:.0f} % ILL.", va="center", fontsize=7.5, color=INK2)
    counts = [sum(tot.values())] + [len(PR.REGISTERS[k]) for k in kinds]
    ax.set_yticks(y, [f"{n} ({c})" for n, c in zip(names, counts)])
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of register entries")
    ax.xaxis.set_major_formatter(lambda x, _: f"{100 * x:.0f} %")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=4, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    out = os.path.join(FIG, "register_status.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def avo_probe():
    """(stage, E, k at 4 N) from the avocado convergence probe log (solver settings not recorded there)."""
    p = R("renders", "avocado_squeeze_conv.log")
    if not os.path.exists(p):
        return []
    avo = jload("test_reports", "avocado_squeeze.json")
    st = {round(l["E_MPa"], 2): l["stage"] for l in avo["lanes"]}
    out = []
    for m in re.finditer(r"\[result\] \d+ ([0-9.]+) MPa\s+k\(4N\) = ([0-9.]+) N/mm", open(p).read()):
        E, k = float(m.group(1)), float(m.group(2))
        s = st.get(round(E, 2))
        if s is None:
            s = min(st.items(), key=lambda it: abs(it[0] - E))[1]
        out.append((s, E, k))
    return out


def plot_squeeze(avo_hertz):
    plt = mpl()
    tom1, tom4 = jload("test_reports", "tomato_squeeze_1000hz.json"), jload("test_reports", "tomato_squeeze.json")
    avo = jload("test_reports", "avocado_squeeze.json")
    rs = jload("test_reports", "roll_squeeze_summary.json")
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
    E = [l["E_MPa"] for l in tom4["lanes"]]
    ax[0].plot(E, [l["k_hertz_4_20N"] for l in tom4["lanes"]], color=PAL[0], label="Hertz, two contacts")
    ax[0].plot(E, [l["k_fem_4_20N"] for l in tom4["lanes"]], "o-", ms=5, color=PAL[1], label="FEM 4000 Hz / 250 it")
    ax[0].plot(E, [l["k_fem_4_20N"] for l in tom1["lanes"]], "s--", ms=5, color=PAL[2], label="FEM 1000 Hz / 120 it")
    m = tom4["measured_ince16"]["initial_N_per_mm"]
    ax[0].plot([E[0]], [m["red"]], "D", ms=7, mfc="white", mec=INK, label="measured, red (ince16)")
    ax[0].plot([E[-1]], [m["green"]], "^", ms=7, mfc="white", mec=INK, label="measured, green (ince16)")
    ax[0].set_xlabel("whole-fruit modulus E [MPa] (tomato_mech)")
    ax[0].set_ylabel("plate stiffness 4-20 N [N/mm]")
    ax[0].set_title("a  Tomato: FEM vs Hertz vs measured", loc="left", fontsize=9.5)
    ax[0].legend(fontsize=7)
    st = [l["stage"] for l in avo["lanes"]]
    ax[1].plot(st, [l["k_secant_4N_N_per_mm"] for l in avo["lanes"]], "o-", ms=5, color=PAL[1],
               label="FEM (final run, test_reports)")
    ax[1].plot(st, [avo_hertz[str(s)] for s in st], color=PAL[0], label="Hertz (food/sdt.py, R 33 mm)")
    probe = avo_probe()
    if probe:
        ax[1].plot([s for s, _, _ in probe], [k for _, _, k in probe], "s", ms=6, mfc="white", mec=PAL[2],
                   label="FEM probe run (renders/avocado_squeeze_conv.log)")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("ripeness stage (1 hard .. 5 ripe)")
    ax[1].set_ylabel("secant stiffness at 4 N [N/mm]")
    ax[1].set_title("b  Avocado: FEM vs Hertz", loc="left", fontsize=9.5)
    ax[1].set_ylim(0.6, 20)
    ax[1].legend(fontsize=7, loc="lower left")
    Ek = [r["E_kPa"] for r in rs["runs"]["1000Hz_120it"]]
    ax[2].plot(Ek, [r["k_incr_1to3N"] for r in rs["runs"]["1000Hz_120it"]], "s--", ms=5, color=PAL[2],
               label="FEM 1000 Hz / 120 it")
    ax[2].plot(Ek, [r["k_incr_1to3N"] for r in rs["runs"]["2000Hz_200it"]], "o-", ms=5, color=PAL[1],
               label="FEM 2000 Hz / 200 it")
    ax[2].plot(Ek, [0.29 * e / 20 for e in Ek], color=PAL[0], label="register model k = 0.29 E / 20 kPa")
    ax[2].set_xlabel("crumb modulus E [kPa] (by roll age)")
    ax[2].set_ylabel("incremental stiffness 1-3 N [N/mm]")
    ax[2].set_title("c  Roll: convergence with solver rate", loc="left", fontsize=9.5)
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(FIG, "squeeze_validation.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_banana_carrot():
    plt = mpl()
    bp, cp = jload("test_reports", "banana_peel.json"), jload("test_reports", "carrot_plank.json")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for i, l in enumerate(bp["lanes"]):
        ax[0].plot([p["t"] for p in l["log"]], [p["F_N"] for p in l["log"]], color=PAL[i], lw=1.4,
                   label=f"stage {l['stage']:g}: {l['outcome']}")
    ax[0].set_xlabel("time [s] (finger pulls at 0.02 m/s)")
    ax[0].set_ylabel("finger force [N]")
    ax[0].set_title("a  Banana peel bench: pull force on strip 1", loc="left", fontsize=9.5)
    ax[0].legend(fontsize=7.5)
    for i, l in enumerate(cp["lanes"]):
        F = [p["F"] for p in l["log"]]
        ax[1].plot(F, [min(p["roll_deg"], 180) for p in l["log"]], "o-" if l["mode"] == "press" else "s--", ms=3.5,
                   lw=1.4, color=PAL[i], label=f"{l['variant']}, {l['mode']}")
    ax[1].axhline(10, color=INK2, lw=0.8, ls=":")
    ax[1].text(44, 6.5, "escape: roll > 10 deg", fontsize=7.5, color=INK2, ha="right")
    ax[1].set_yscale("symlog", linthresh=1)
    ax[1].set_ylim(0, 400)
    ax[1].set_xlabel("knife force [N] (up to F_cut = %.0f N)" % cp["F_cut_N"])
    ax[1].set_ylabel("carrot roll [deg]")
    ax[1].set_title("b  Carrot plank bench: does the root roll?", loc="left", fontsize=9.5)
    ax[1].legend(fontsize=7.5)
    fig.tight_layout()
    out = os.path.join(FIG, "banana_carrot_benches.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


# ================================================================================================ main
def main():
    t0 = time.time()
    reg = PR.REGISTERS
    allc = Counter()
    for k in reg:
        allc += PR.status_counts(k)
    n_reg = sum(allc.values())
    INST = instances()
    built_kinds = sorted({d["kind"] for d in INST.values()}, key=list(ITEMS).index)
    from food import egg_tasks as ET, schema as SC, wave1_tasks as W1, wave2_tasks as W2
    from food import avocado_tasks as AT
    tasks_by_item = {"egg": list(ET.TASKS)}
    for mod in (W1, W2, AT):
        for k, v in mod.TASKS.items():
            tasks_by_item.setdefault(v["item"], []).append(k)
    demos = cached("designator_demos", run_designator_demos)["demos"]
    w3 = cached("wave3_parts", run_wave3_parts)
    crk = cached("egg_crack_repro", crack_repro)
    avb = cached("avocado_belief", avocado_belief)
    from food.sdt import hertz_squeeze_stiffness
    avo = jload("test_reports", "avocado_squeeze.json")
    avo_hertz = {str(l["stage"]): float(hertz_squeeze_stiffness(l["stage"], 4.0, R=avo["D_mm"] / 2e3, nu=0.45)) / 1e3
                 for l in avo["lanes"]}
    f_status = plot_status()
    f_squeeze = plot_squeeze(avo_hertz)
    f_bc = plot_banana_carrot()

    # ---- source-key resolution
    unresolved = Counter()
    for k, r in reg.items():
        for key, (u, v, s, src) in r.items():
            if s in ("SOURCED", "SOURCED-VERIFY", "CALIBRATED") and not src_keys(src):
                unresolved[s] += 1
    n_sourced = allc["SOURCED"] + allc["SOURCED-VERIFY"] + allc["CALIBRATED"]
    used_keys = set()
    for k, r in reg.items():
        for key, (u, v, s, src) in r.items():
            used_keys.update(src_keys(src))
    n_sampled = sum(1 for r in reg.values() for (u, v, s, src) in r.values()
                    if isinstance(v, (int, float)) or (isinstance(v, tuple)))

    # ---- mass / cut checks
    mass_rows, mass_issues = [], []
    for name, d in INST.items():
        whole = d["whole"]["mass_g"]
        leaf = sum((p.get("mass_g") or 0) for p in d["parts"].values() if not p.get("group"))
        grp = OrderedDict()
        for cname, c in d["cuts"].items():
            if c.get("mass_g") is None:
                continue
            grp.setdefault(c.get("cut", "half"), []).append((cname, c["mass_g"]))
        cuts = []
        for g, mem in grp.items():
            if len(mem) >= 2:
                s = sum(m for _, m in mem)
                cuts.append(f"{'+'.join(n for n, _ in mem)}: {s / whole:.4f}")
            elif len(mem) == 1:
                cuts.append(f"{mem[0][0]}: {mem[0][1] / whole:.4f}")
        fails = sum(len(v) for v in d["cut_failures"].values())
        dev = leaf / whole - 1
        if abs(dev) > 0.005:
            mass_issues.append((name, whole, leaf, dev))
        mass_rows.append([tt(name), f"{whole:.1f}", f"{leaf:.1f}", f"{100 * dev:+.1f}",
                          esc("; ".join(cuts)) or "--", esc(", ".join(f for v in d["cut_failures"].values() for f in v))
                          or "none"])
    stale_egg = [n for n, d in INST.items() if d["kind"] == "egg"
                 and Counter(d["register_status"]) != PR.status_counts("egg")]
    truth_meff = None
    tp = R("assets", "food", "egg_1", "truth.usda")
    if os.path.exists(tp):
        m = re.search(r"truth:impact_effective_mass_fraction = ([0-9.]+)", open(tp).read())
        truth_meff = float(m.group(1)) if m else None

    out = []
    w = out.append
    # ------------------------------------------------------------------ preamble (as the site-twin report)
    w(r"""\documentclass[10.5pt,a4paper]{article}
\usepackage[margin=2.2cm,headheight=14pt]{geometry}
\usepackage{fontspec,unicode-math}
\setmainfont{Libertinus Serif}\setsansfont{IBM Plex Sans}\setmonofont[Scale=0.84]{IBM Plex Mono}\setmathfont{Libertinus Math}
\usepackage{microtype,graphicx,booktabs,tabularx,longtable,xcolor,fvextra,caption,fancyhdr,titlesec,enumitem,array,placeins}
\usepackage{xurl}
\usepackage[colorlinks,linkcolor=teal!60!black,urlcolor=teal!60!black,citecolor=teal!60!black]{hyperref}
\captionsetup{font=small,labelfont=bf}
\renewcommand{\tabularxcolumn}[1]{>{\raggedright\arraybackslash}p{#1}}
\setlength{\emergencystretch}{3em}\setlist{itemsep=2pt,topsep=3pt}
\titleformat{\section}{\sffamily\Large\bfseries\color{teal!50!black}}{\thesection}{0.8em}{}
\titleformat{\subsection}{\sffamily\large\bfseries}{\thesubsection}{0.7em}{}
\titleformat{\paragraph}[runin]{\sffamily\bfseries}{}{}{}[.]
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\sffamily\footnotesize Kitchen food twins: semantic digital twins of food for robot manipulation}
\fancyhead[R]{\sffamily\footnotesize\nouppercase{\leftmark}}\fancyfoot[C]{\sffamily\footnotesize\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\renewcommand{\floatpagefraction}{0.7}\renewcommand{\topfraction}{0.9}\renewcommand{\textfraction}{0.1}
\begin{document}
\begin{titlepage}\sffamily
""")
    w(f"\\noindent\\includegraphics[width=\\linewidth,height=0.42\\textheight,keepaspectratio]{{{img('renders/food/hero.png')}}}\\par\\vspace{{0.8cm}}\n")
    n_items = len(reg)
    w(r"""\noindent{\color{teal!50!black}\rule{\linewidth}{1.2pt}}\par\vspace{0.4cm}
{\Huge\bfseries Food that robots can reason about\par}\vspace{0.3cm}
{\Large Semantic digital twins of fruit, vegetables, eggs and bread\\ for robot manipulation in household kitchens: what we have so far\par}\vspace{0.3cm}
\noindent{\color{teal!50!black}\rule{\linewidth}{1.2pt}}\par\vspace{0.6cm}
{\large """ + f"{n_items} kitchen items for the EASE and GARMI apartments, modelled part by part" + r"""\par}\vspace{0.3cm}
{\normalsize [Authors and affiliations] \hfill 20 October 2026\par}\vfill
{\small\rmfamily This is a research twin built from published biology, food-science and cooking sources. It is not a
measured model of any particular piece of food. Every quantitative value carries a status label from the project's
parameter register (SOURCED, SOURCED-VERIFY, STANDARD, DERIVED, TYPICAL, CALIBRATED, ILLUSTRATIVE;
Section~\ref{sec:quality} and Appendix~\ref{app:register}). ILLUSTRATIVE values are plausible assumptions of the twin
and must not be quoted as facts. Bench results are simulations, not experiments.\par}
\end{titlepage}
\tableofcontents\clearpage
""")
    # ------------------------------------------------------------------ summary
    n_inst = len(INST)
    n_parts_total = sum(len(d["parts"]) for d in INST.values())
    ill = allc["ILLUSTRATIVE"]
    w(r"""\section*{Summary}\addcontentsline{toc}{section}{Summary}
\paragraph{Purpose} We want models of everyday food that a robot can plan with, act on and learn about: geometric and
physical objects in Isaac Sim whose every part is a node of a semantic digital twin (SDT), that knows what it is
(\emph{shell}, \emph{yolk}, \emph{seed}, \emph{crust}), how the robot can come to know it (visible, only after
cutting, tactile, inferred, hidden), and how it behaves when it is gripped, dropped, squeezed, cut, peeled or cooked.
The target scenes are the EASE (IAI Bremen) and GARMI apartments, which already exist as robot-simulation scenes in
our CRAM tooling. Each item has three layers: a continuous model of a single, randomly drawn \emph{instance} (its
hidden truth), the robot's \emph{beliefs} about it (priors from a parameter register, updated by observations), and a
small \emph{discrete} model (attributes, states and actions) that plans and action designators refer to.
""")
    w(r"""\paragraph{What was built} """ + f"{n_items} items have a code module and a parameter register: "
      "the egg set (egg, apple, avocado), wave~1 (roll, banana, carrot), wave~2 (loaf, onion, orange, potato, tomato) "
      "and wave~3 (lemon, cucumber, croissant, baguette, pepper). " +
      (f"All {len(built_kinds)} are built as USD assets: {n_inst} instances in " if len(built_kinds) == n_items else
       f"{len(built_kinds)} of them are built as USD assets: {n_inst} instances in ") +
      r"\texttt{assets/food} with " + f"{n_parts_total:,}" + r""" part prims in total, each with cut states, a belief layer
(\texttt{sdt.usda}) and a hidden-truth layer (\texttt{truth.usda}). The register holds """ + f"{n_reg}" +
      r""" entries and """ + f"{len(PR.SOURCES)}" + r""" sources, """ + f"{len(used_keys)}" + r""" of which the entries cite by key. There are """ +
      f"{sum(len(v) for v in tasks_by_item.values())}" + " task designators (" + ", ".join(
      f"{k} {len(v)}" for k, v in tasks_by_item.items()) + r"""), seven
Isaac Sim benches with results in \texttt{test\_reports/} (egg grasp and drop, egg opening, avocado, roll and tomato
squeeze, banana peel, carrot plank) and two CPU studies (egg cracking and boiling), and the EASE and GARMI apartments
and the GARMI robot are imported into USD.
""")
    w(r"""\paragraph{One avocado, end to end} The avocado now goes through the whole loop a robot needs
(Section~\ref{sec:avocado-reasoning}): a photo becomes an SDT node; the node is a 4000-particle belief updated by
Bayes' rule from photos, squeezes, cuts, twists and the spoon; a vision-based monitor judges peeling against the
twin's expectation; an emulator halves and pits the avocado as USD states; the robot answers competence questions
from its belief, its task models and mental simulation (including whether a squeeze is worth it); and a PR2 in the
CRAM kitchen detects, squeezes, cuts round the seed, twists the halves apart and spoons out the seed, checking every
step against what it imagined -- scripted, or driven by an operator from the CRAMERA viewer.

\paragraph{From reasoning to acting and to theories} The robot then acts as an agent that enforces the effects its
mission requires and prevents the ones it forbids, deciding over the fruit's lifetime (ripening, spoilage, browning),
and its knowledge is written down and tested as an avocado manipulation microtheory -- the egg microtheory transferred
-- which the robot also learns from its own failures and which programs its reinforcement learning
(Section~\ref{sec:avocado-theory}; the full microtheory is \texttt{report/avocado\_microtheory.pdf}).
""")
    cook = jload("test_reports", "egg_cook_study.json")
    room = {r["t_min"]: r["yolk"] for r in cook["hot start, room-temp egg (20 C)"]}
    fridge = {r["t_min"]: r["yolk"] for r in cook["hot start, fridge egg (4 C)"]}
    ref_room = [(4.5, "runny"), (6, "jammy"), (7, "jammy"), (8.5, "creamy"), (10, "firm"), (11, "firm")]
    ref_fridge = [(6, "runny"), (8.5, "creamy"), (11, "firm")]
    hit_room = sum(room[t] == c for t, c in ref_room)
    hit_fridge = sum(fridge[t] == c for t, c in ref_fridge)
    _peel = json.load(open(R("assets", "food", "avocado_4", "peel_set", "eval", "summary.json")))["summary"]["rgbd"]
    _pc = _peel["confusion"]
    PEEL_HEAD = (f"{_pc['rework_found'] + _pc['success_confirmed']}/{sum(_pc.values())} correct verdicts "
                 f"(skin IoU {_peel['mean_iou_skin']:.2f})")
    plank = {l["variant"]: l["F_at_escape_N"] for l in jload("test_reports", "carrot_plank.json")["lanes"]
             if l["mode"] == "stroke"}
    tom4 = jload("test_reports", "tomato_squeeze.json")
    tl = {l["E_MPa"]: l for l in tom4["lanes"]}
    red = tl[min(tl)]
    green = tl[max(tl)]
    w(r"""\paragraph{Headline results} (1) The egg boiling model (heat conduction through the twin's own tissue map,
doneness thresholds CALIBRATED) reproduces """ + f"{hit_room}/{len(ref_room)}" + r""" doneness classes of the
hot-start timing table it was calibrated on and """ + f"{hit_fridge}/{len(ref_fridge)}" + r""" of the fridge-egg
timings; this is a fit, not an independent test. (2) The egg crack study turns kitchen advice into numbers: over 400
register-sampled eggs the best strike on another egg is """ +
      f"{jload('test_reports', 'egg_crack_study.json')['summary']['another egg']['v_best']:.2f}" + r"""~m/s and on a flat
counter """ + f"{jload('test_reports', 'egg_crack_study.json')['summary']['flat counter (granite)']['v_best']:.2f}" +
      r"""~m/s (strike-model constants ILLUSTRATIVE). (3) For the red tomato the PhysX FEM squeeze at 4000~Hz gives
""" + f"{red['k_fem_4_20N']:.1f}" + r"""~N/mm against """ + f"{red['k_hertz_4_20N']:.2f}" + r"""~N/mm from Hertz
theory and """ + f"{tom4['measured_ince16']['initial_N_per_mm']['red']:.1f}" + r"""~N/mm measured (ince16,
SOURCED-VERIFY); stiffer lanes stay under-converged (green: """ + f"{green['k_fem_4_20N']:.1f}" + r""" vs Hertz
""" + f"{green['k_hertz_4_20N']:.1f}" + r"""~N/mm). The same solver limit appears in the avocado and roll benches.
(4) Opening a cracked egg over a bowl in Isaac Sim keeps """ +
      f"{100 * jload('test_reports', 'egg_open_sim.json')['fraction_in_bowl']:.1f}" + r"""~\% of """ +
      f"{jload('test_reports', 'egg_open_sim.json')['particles']:,}" + r""" albumen particles and the yolk in the bowl.
(5) The carrot bench shows in simulation why cooks cut a plank before slicing: with a slicing stroke the round root
rolls away at """ + f"{plank['whole']:.0f}" + r"""~N already, the planked half only at about """ +
      f"{plank['planked']:.0f}" + r"""~N. (6) The avocado's peeling monitor finds """ + PEEL_HEAD + r""" on synthetic
peels from RGB-D, and in the PR2 scenario the robot's ripeness belief, misled by the photo, is corrected by an
unexpected twist outcome towards the hidden truth (Section~\ref{sec:pr2}).
""")
    w(r"""\paragraph{What it is not, yet} """ + f"{ill} of {n_reg} register entries ({pct(ill, n_reg)}) are "
      r"""ILLUSTRATIVE and another """ + f"{allc['SOURCED-VERIFY']}" + r""" need their source checked. """ + (
      r"""Wave~3 is now built (USD assets with cut states, renders, anatomy plates) but has no task designators or
benches, and its research notes are referenced but missing. """ if W3_BUILT else r"""Wave~3 exists only as code and
register (no assets, renders, designators or benches; its research notes are referenced but missing). """) + (
      "" if W2_RENDERED else r"""The wave~2 items have anatomy plates but no beauty renders. """) + r"""Only egg,
apple and avocado have been placed in a kitchen,
and the end-to-end GARMI egg task stopped with a numerical failure of the arm controller. The anatomy plates are
waiting for the user's review, which is a gate of the build plan. The physics benches are single runs of simplified
bodies (homogeneous FEM, rigid peel segments, kinematic shells).
""")
    w(fig("renders/food/wave1_hero.png", "Wave~1 items, path-traced in Isaac Sim~6.2 (existing render): the roll "
          "whole and as top and bottom halves, the banana whole and as a lengthwise half, and the carrot whole, as a "
          "lengthwise half and as a crown piece.", r"0.32\textheight"))
    # inventory table
    rows = []
    for kind, (name, wave, inst, docs) in ITEMS.items():
        r = reg[kind]
        c = PR.status_counts(kind)
        ninst = sum(1 for d in INST.values() if d["kind"] == kind)
        states = ""
        if inst and inst in INST:
            states = ", ".join(INST[inst]["cuts"].keys())
        plate = os.path.exists(R("renders", "food", f"anatomy_{kind}_labelled.png"))
        beauty = any(os.path.exists(R("renders", "food", f)) for f in
                     {"egg": ["egg_cutaway.png"], "apple": ["apple_halves.png"], "avocado": ["avocado_halves.png"],
                      "roll": ["wave1_roll.png"], "banana": ["wave1_banana.png"],
                      "carrot": ["wave1_carrot.png"]}.get(kind, [f"item_{kind}.png"]))
        bench = {"egg": "grasp/drop, open, crack, cook", "avocado": "FEM squeeze, peel monitor, emulator, PR2 (CRAM)",
                 "roll": "FEM squeeze",
                 "banana": "peel", "carrot": "plank", "tomato": "FEM squeeze"}.get(kind, "--")
        rows.append([esc(name), esc(wave), str(len(r)), f"{100 * c.get('ILLUSTRATIVE', 0) / len(r):.0f}",
                     str(ninst) if ninst else "--", esc(states) or "--",
                     ("plate" if plate else "--") + (" + renders" if beauty else ""),
                     str(len(tasks_by_item.get(kind, []))) or "0", esc(bench)])
    w(table(["p{0.17\\linewidth}", "l", "r", "r", "r", "p{0.17\\linewidth}", "p{0.1\\linewidth}", "r", "X"],
            ["Item", "Wave", "Reg.", "\\% ILL.", "USD", "Cut / derived states", "Images", "Des.", "Bench"], rows,
            "What exists per item. Reg.: register entries; \\% ILL.: share of ILLUSTRATIVE entries; USD: built "
            "instances in assets/food; Des.: task designators. Counted from the repository by this script.",
            size="scriptsize"))
    # ------------------------------------------------------------------ 1 method
    w(r"""\clearpage\section{Method: a food item as a semantic digital twin}\label{sec:method}
\paragraph{From biology to a part tree} Each item is a Python class (\texttt{food/<item>.py}) that draws one instance
from the register (\texttt{params.sample(kind, seed)}) and builds an \emph{analytic anatomy} for it: meridian curves
of revolution, offsets and lofts (\texttt{food/geometry.py}, \texttt{food/shapes.py}) and a label field that says, for
any point in space, which tissue it belongs to (shell, thick albumen, yolk; flavedo, albedo, juice vesicles; crust,
crumb). The class's \texttt{parts()} returns the anatomical partonomy as a tree of \texttt{Part} records: a USD path,
an ontology class, an observability class, a documentation string with sources, and the geometry source (a label of
the field, an analytic mesh, or a point set for things such as pores, lenticels, gas cells and seeds). The anatomy was
researched in biology and food-science sources; the user rejected two early versions (a wrong apple core, an egg-like
avocado seed), and the anatomy plates are still awaiting review (Section~\ref{sec:next}).

\paragraph{Closed meshes and cuts} \texttt{food/partgraph.py} turns every labelled part into a closed (watertight)
mesh by marching cubes on the label field, and computes volume and mass from the part's density. Cuts are plane
splits of every part mesh (manifold3d \texttt{split\_by\_plane}, nudged off the symmetry axis), so a half is again a
complete part graph with its own mass, centre of mass and inertia. Table~\ref{tab:mass} checks mass conservation.

\paragraph{USD part graph} \texttt{food/usd\_partgraph.py} writes one USD asset per instance. Every prim under
\texttt{/Food/Parts} is an SDT node with \texttt{sdt:represents} (the entity IRI of the physical part),
\texttt{sdt:class}, \texttt{sdt:partOf} and \texttt{sdt:observability}, UsdSemantics labels for perception, and
\texttt{food:mass\_g}, \texttt{food:volume\_cm3}, \texttt{food:density\_kgm3} from its closed mesh. The whole object
is one rigid body whose mass properties are summed over the parts; interior parts are invisible until a cut state
exposes them. Two layers sit over the geometry: \texttt{sdt.usda} holds the robot's beliefs (priors from the
register, observations), and \texttt{truth.usda} holds the instance's hidden parameters, which the dynamics models and
audits read but the robot never does.
""")
    w(fig("renders/food/scenegraph_sdt.pdf", "The USD scene graphs of the first three items as SDT part graphs "
          "(existing figure from tools/anatomy\\_doc.py): prim, sdt:class and part mass; the colour of the dot is the "
          "observability class.", r"0.5\textheight"))
    # USD excerpt from the real egg_1 asset
    excerpt = []
    try:
        from pxr import Usd
        st = Usd.Stage.Open(R("assets", "food", "egg_1", "whole.usdc"))
        for path in ("/Food", "/Food/Parts/Shell", "/Food/Parts/Shell/Cuticle", "/Food/Parts/Yolk/GerminalDisc"):
            p = st.GetPrimAtPath(path)
            if not p:
                continue
            excerpt.append(f'def {p.GetTypeName()} "{p.GetName()}"   # {path}')
            for a in p.GetAttributes():
                if a.GetName().startswith(("sdt:", "food:mass", "food:volume", "food:density", "food:kind",
                                           "food:shell_condition", "semantics:labels")):
                    v = a.Get()
                    v = f"{v:.4g}" if isinstance(v, float) else repr(v) if isinstance(v, str) else str(v)
                    excerpt.append(f"    {a.GetTypeName()} {a.GetName()} = {v}")
    except Exception as e:  # noqa: BLE001
        excerpt.append(f"(could not open egg_1 with pxr: {e})")
    sdt_txt = open(R("assets", "food", "egg_1", "sdt.usda")).read().splitlines()
    tr_txt = open(R("assets", "food", "egg_1", "truth.usda")).read().splitlines()
    w(code("\n".join(excerpt), "Excerpt of assets/food/egg\\_1/whole.usdc (read with pxr by this script)"))
    w(code("\n".join(sdt_txt[8:28]), "assets/food/egg\\_1/sdt.usda (beliefs, lines 9--28)"))
    w(code("\n".join(tr_txt[:16]), "assets/food/egg\\_1/truth.usda (hidden truth, first lines)"))
    w(r"""\paragraph{The parameter register} \texttt{food/params.py} holds every number the models use, per item, as
\emph{(unit, value or distribution, status, source)}. The status rule is carried over from the lab twin: no invented
physical parameter is presented as a fact. Distributions (normal with truncation, uniform, integer, choice, lognormal)
are what the hidden-world sampler draws per instance; the robot's belief starts from the same distribution as its
prior and learns the instance's values only through observations. Of the """ + f"{n_reg}" + r""" entries, """ +
      f"{n_sampled}" + r""" carry a number or a distribution; the rest are model notes (formulas, tables, ranges as
text) that the code implements.
""")
    w(table(["l", "X"], ["Status", "Meaning (food/params.py)"],
            [[esc(k), esc(v)] for k, v in STATUS_MEANING.items()], "Status labels of the parameter register."))
    sch_rows = []
    for k, s in SC.SCHEMAS.items():
        sch_rows.append([esc(s["class"]), esc(", ".join(s["attributes"])), esc(", ".join(s["actions"]))])
    w(r"""\paragraph{Discrete schema, dynamics and beliefs} \texttt{food/schema.py} is the symbolic layer: per item, a
class, attributes with domains and observability, and actions with preconditions, effects and \emph{guards} -- the
continuous quantities, computed by the twin, that decide which effect occurs (for the egg: the grip corridor
$F_\mathrm{slip} < F < F_\mathrm{break}$ and the AICOR H8 admissibility $P(\mathrm{slip}) < \varepsilon_s$,
$P(\mathrm{rupture}) < \varepsilon_r$, $P(\mathrm{stable}) > \gamma$). The schema covers the first six items; the
wave~2 and wave~3 items expose their discrete state through \texttt{discrete()} methods of their truth classes
instead. The hidden dynamics are in \texttt{food/dynamics.py} (egg, apple, avocado) and \texttt{food/<item>\_models.py}
(the later items); the beliefs in \texttt{food/sdt.py} and the \texttt{*\_beliefs} functions. Task designators
(\texttt{food/egg\_tasks.py}, \texttt{wave1\_tasks.py}, \texttt{wave2\_tasks.py}) list goal, parameters, steps,
preconditions, effects and failure modes with cues and recoveries, each bound to the hidden model that executes it.
""")
    w(table(["p{0.11\\linewidth}", "X", "p{0.3\\linewidth}"], ["Class", "Attributes", "Actions"], sch_rows,
            "The discrete schema (food/schema.py).", size="scriptsize"))
    w(r"""\paragraph{Mass conservation} The table checks, for every built instance, that the part masses add up to the
whole and that the pieces of each cut add up to the whole (ratio 1.0000 = conserved).
""")
    w(table(["l", "r", "r", "r", "X", "p{0.17\\linewidth}"],
            ["Instance", "whole (g)", "parts (g)", "diff. \\%", "cut pieces / whole", "cut failures"],
            mass_rows, "Mass checks of the built instances (assets/food/*/instance.json).\\label{tab:mass}",
            size="scriptsize", long=True))
    if mass_issues:
        w("Part sums deviate from the whole for " + ", ".join(
            f"{tt(n)} ({l:.1f} vs {wh:.1f}~g, {100 * dv:+.1f}~\\%)" for n, wh, l, dv in mass_issues) +
          r""": these are the two breads, whose crust and crumb meshes appear to overlap or are meshed at a different
resolution from the whole; the wave~1 checklist in \texttt{docs/wave1/README.md} reports this gate as passed, which
this check does not confirm. The apple's lengthwise cut lost two empty locules (cut failures above); their mass is
negligible.
""")
        CHECKS.append(("part masses sum to the whole", f"{len(INST) - len(mass_issues)}/{len(INST)} instances "
                       "within 0.5 %; roll and loaf +1.4 %"))
    # ------------------------------------------------------------------ 2 egg
    e1 = INST["egg_1"]
    w(r"""\clearpage\section{The hen egg}\label{sec:egg}
The egg is the most developed item: anatomy, crack and cook models, a belief that computes the AICOR H8 probabilities,
seven task designators, a grasp-and-drop bench and an opening simulation in Isaac Sim. Eight instances are built
(\texttt{egg\_0}..\texttt{egg\_7}, """ + f"{min(d['whole']['mass_g'] for d in INST.values() if d['kind'] == 'egg'):.1f}"
      + "--" + f"{max(d['whole']['mass_g'] for d in INST.values() if d['kind'] == 'egg'):.1f}" + r"""~g); the
examples use \texttt{egg\_1} (""" + f"{e1['whole']['mass_g']:.1f}" + r"""~g).
""")
    w(fig("renders/food/anatomy_egg_labelled.png", "Exploded anatomy plate of egg\\_1 (existing render): outside to "
          "inside, every part is a prim of the USD part graph with its mass.", r"0.3\textheight", maxw=2400))
    w(figrow([("renders/food/egg_cutaway.png", "cutaway state"), ("renders/food/egg_opened_scene.png", "opened, fried"),
              ("renders/food/egg_shell_fragments.png", "half shell")],
             "Egg renders (existing, path-traced). The cutaway shows the yolk layers, latebra and chalazae."))
    w(tree_table("egg", "egg_1", e1, "Part tree of egg\\_1 (instance.json and whole.usdc)."))
    w(keytable("egg", ["length_L", "shape_index_B_over_L", "shell_thickness", "shell_E", "F_break_equatorial",
                       "F_break_axial", "shell_contact_stiffness", "yolk_mass_fraction", "air_cell_height",
                       "microcrack_band", "impact_effective_mass_fraction", "impact_rate_factor",
                       "crack_strike_model", "cook_yolk_thresholds"], "Key register entries of the egg."))
    eg = SC.SCHEMAS["egg"]
    w(r"""\paragraph{States and hidden dynamics} The schema's attributes are """ + esc(", ".join(eg["attributes"])) +
      r""". Shell condition follows the USDA vocabulary (intact, microcracked, cracked, shattered, opened); a
micro-crack is invisible and lowers the break force by the register's \texttt{microcrack\_strength\_loss}.
\texttt{EggTruth} (\texttt{food/dynamics.py}) cracks the shell when the effective contact force exceeds the
direction-dependent break force; impacts use $F = v\sqrt{k_\mathrm{eff}\, m_\mathrm{eff}}$ with the Reissner shell
stiffness in series with the surface. \texttt{food/egg\_crack.py} adds strike outcomes on a counter, a bowl rim or
another egg and the opening over a vessel; its constants are ILLUSTRATIVE. \texttt{food/egg\_cook.py} solves
axisymmetric heat conduction through the egg's own tissue map. The belief \texttt{EggBelief} (\texttt{food/sdt.py})
holds particles over crack force, shell thickness and friction and returns the grip corridor and H8 admissibility.
""")
    rows = [[esc(k), esc(v["goal"]), esc(v["model"])] for k, v in ET.TASKS.items()]
    w(table(["p{0.15\\linewidth}", "X", "p{0.36\\linewidth}"], ["Designator", "Goal", "Twin model"], rows,
            "Egg task designators (food/egg\\_tasks.py; rendered in docs/EGG\\_TASKS.md).", size="scriptsize"))
    # bench
    eb = jload("test_reports", "egg_bench.json")
    g = [x for x in eb["grasp"] if x["exp"] == "G"]
    v = [x for x in eb["grasp"] if x["exp"] == "V"]
    adm = [x["F_cmd"] for x in g if x["H8_admissible"]]
    first_crack = min(x["F_cmd"] for x in g if x["shell"] != "intact")
    fc = sorted(x["F_crack_truth"] for x in v)
    n_fail = sum(x["shell"] != "intact" for x in v)
    p_pred = np.mean([x["belief"]["rupture"] for x in v])
    pad_err = max(abs(x["F_pad_measured"] - x["F_cmd"]) for x in eb["grasp"])
    drops = eb["drop"]
    crack1 = [d["surface"].replace("_", " ") for d in drops if d["h_cm"] == min(x["h_cm"] for x in drops)
              and d["shell"] != "intact"]
    towel_ok = max((d["h_cm"] for d in drops if d["surface"] == "towel" and d["shell"] == "intact"), default=0)
    towel_crack = min((d["h_cm"] for d in drops if d["surface"] == "towel" and d["shell"] != "intact"), default=None)
    w(r"""\subsection{Bench: grasp and drop (Isaac Sim)}
\texttt{scripts/sim\_egg\_bench.py} grips egg\_1 with PhysX contacts at commanded pad forces and drops eggs onto four
kitchen surfaces; outcomes come from the hidden \texttt{EggTruth}, predictions from \texttt{EggBelief}. In the grip
sweep, the measured pad force equals the commanded one within """ + f"{pad_err:.2f}" + r"""~N; the egg slips at """ +
      f"{min(x['F_cmd'] for x in g):g}" + r"""~N, the H8-admissible commands run from """ +
      f"{min(adm):g} to {max(adm):g}" + r"""~N, and the shell cracks from """ + f"{first_crack:g}" +
      r"""~N (hidden crack force of this egg """ + f"{g[0]['F_crack_truth']:g}" + r"""~N). In a validation lane eight
instances were gripped at 30~N: """ + f"{n_fail}" + r""" of 8 cracked or micro-cracked, while the belief predicted a
rupture probability of about """ + f"{p_pred:.2f}" + r""". The eight hidden crack forces (median """ +
      f"{np.median(fc):.1f}" + r"""~N) sit below the register prior's mean (35~N, SOURCED); with eight eggs this is
not conclusive, but it is worth checking whether the truth sampler and the belief draw crack forces from the same
effective distribution (the truth applies thickness scaling, which the belief may not). Drops from """ +
      f"{min(x['h_cm'] for x in drops):g}" + r"""~cm already crack this egg on """ + esc(", ".join(crack1)) +
      r"""; on a towel it survives up to """ + f"{towel_ok:g}" + r"""~cm""" +
      (f" and cracks at {towel_crack:g}~cm" if towel_crack else "") + r""" (critical heights in the figure). The impact
model's effective-mass fraction is CALIBRATED to the drop statistics of Cohen et al.\ 2025 (cohen2025).
""")
    w(fig("renders/food/egg_bench.png", "Egg bench (existing figure, tools/plot\\_egg\\_bench.py): (a) grip sweep, "
          "belief vs simulated outcome; (b) eight instances at 30~N; (c) drops onto kitchen surfaces with peak force "
          "and impact speed.", r"0.3\textheight"))
    cs = jload("test_reports", "egg_crack_study.json")["summary"]
    rows = []
    for k, s in cs.items():
        rep = crk["best"].get(k, [None, None])
        rows.append([esc(k), f"{s['v_best']:.2f}", f"{s['p_best_recovery']:.2f}", f"{s['p_best_direct']:.2f}",
                     f"{s['window_90pct'][0]:.2f}--{s['window_90pct'][1]:.2f}",
                     f"{rep[0]:.2f} / {rep[1]:.2f}" if rep[0] is not None else "--"])
    same = all(abs(crk["best"][k][0] - s["v_best"]) < 1e-9 and abs(crk["best"][k][1] - s["p_best_recovery"]) < 1e-9
               for k, s in cs.items())
    CHECKS.append(("egg crack study reproducible with the current register", "yes" if same else "NO"))
    w(r"""\subsection{Study: how hard to crack an egg (CPU)}
\texttt{tools/egg\_crack\_study.py} strikes 400 register-sampled eggs at 30 speeds against four tools and counts how
often the egg ends up open in the cup with a whole yolk and no shell -- directly, and with the recovery people use
(crack into a cup first, fish shards out with a half shell). The strike-outcome bands, shard rates and puncture
probabilities are ILLUSTRATIVE (register \texttt{crack\_strike\_model}), so the optimum speeds are model answers, not
measurements. This report re-ran the computation in memory with the current register (impact effective-mass fraction
""" + f"{crk['m_eff_fraction']:g}" + r"""): the best speeds and probabilities are """ +
      ("identical to the stored study" if same else "\\textbf{different from} the stored study") + ".\n")
    w(table(["X", "r", "r", "r", "r", "r"], ["Strike surface", "best v (m/s)", "P with recovery", "P direct (best)",
                                            "90 \\% window (m/s)", "re-run v / P"], rows,
            "Egg crack study (test\\_reports/egg\\_crack\\_study.json) and this report's re-run.", size="scriptsize"))
    w(fig("renders/food/egg_crack_study.png", "Crack study (existing figure): outcome shares (top) and task success "
          "(bottom) over strike speed per tool.", r"0.3\textheight"))
    rows = []
    for t_, c_ in ref_room:
        rows.append([f"{t_:g}", "hot start, room-temperature egg", esc(c_), esc(room[t_]),
                     "yes" if room[t_] == c_ else "no"])
    for t_, c_ in ref_fridge:
        rows.append([f"{t_:g}", "hot start, fridge egg", esc(c_), esc(fridge[t_]), "yes" if fridge[t_] == c_ else "no"])
    w(r"""\subsection{Study: boiling (CPU)}
\texttt{tools/egg\_cook\_study.py} runs \texttt{EggHeat} on egg\_4 (thermal properties TYPICAL, doneness thresholds
on the yolk fraction that reached 65~\textdegree C CALIBRATED to the alphafoodie timing table and the Lopez-Alt fridge-egg
timings). The table compares the model's doneness class with the source at each source time. Because the thresholds
were fitted to these points, agreement shows a consistent fit, not a validation; no independent timing data are held.
The cold-start case (water heated to the boil in 9~min, then off the heat, ILLUSTRATIVE profile) is not compared with
a source.
""")
    w(table(["r", "X", "l", "l", "l"], ["min", "Case", "Source class", "Model class", "Match"], rows,
            "Egg boiling model vs the cooking sources it was calibrated on (test\\_reports/egg\\_cook\\_study.json).",
            size="scriptsize"))
    w(fig("renders/food/egg_cook_study.png", "Boiling study (existing figure): doneness over time with the source "
          "markers, yolk-centre temperature, and temperature sections at five times.", r"0.36\textheight"))
    eo = jload("test_reports", "egg_open_sim.json")
    w(r"""\subsection{Simulation: opening a cracked egg (Isaac Sim)}
\texttt{scripts/sim\_egg\_open.py} replaces the cracked egg by its two shell fragments (kinematic, from the crack
model), the albumen by """ + f"{eo['particles']:,}" + r""" PBD particles and the yolk by a rigid body, and swings the
halves apart over a glass bowl. After """ + f"{eo['log'][-1]['t']:.2f}" + r"""~s, """ +
      f"{100 * eo['fraction_in_bowl']:.2f}" + r"""~\% of the particles are in the bowl, none below the table, and the
yolk is in the bowl. The yolk cannot break in this simulation (rigid body); yolk breakage comes from the hidden
\texttt{open\_up} model instead. The albumen surface is reconstructed from the particles by marching cubes, because the
PhysX isosurface does not work with Fabric.
""")
    w(fig("renders/food/egg_open_filmstrip.png", "Opening simulation (existing filmstrip): halves swing apart, the "
          "albumen streams, the yolk drops whole and relaxes to its dome.", r"0.33\textheight", maxw=2400))
    # ------------------------------------------------------------------ 3 apple and avocado
    a3, v4 = INST["apple_3"], INST["avocado_4"]
    w(r"""\clearpage\section{Apple and avocado}
\subsection{Apple (Gala-like)}
One instance is built (\texttt{apple\_3}, """ + f"{a3['whole']['mass_g']:.1f}" + r"""~g) with two lengthwise halves.
The anatomy follows the pome: hypanthial flesh (cortex) round a core of five fused carpels, ten primary vascular
bundles, a calyx with five sepals, lenticels as a point set and five named surface regions of the cuticle.
""")
    w(fig("renders/food/anatomy_apple_labelled.png", "Exploded anatomy plate of apple\\_3 (existing render).",
          r"0.28\textheight", maxw=2400))
    w(figrow([("renders/food/apple_halves.png", "whole and halves"), ("renders/food/closeup_apple_cut.png",
                                                                     "cut face: core and seeds")],
             "Apple renders (existing, path-traced)."))
    w(tree_table("apple", "apple_3", a3, "Part tree of apple\\_3 (indexed siblings merged)."))
    w(keytable("apple", ["diameter_D", "flesh_density", "flesh_E", "mt_firmness", "flesh_shear_strength",
                         "bruise_energy_threshold", "bruise_volume_per_J", "cutting_force_peak", "friction_skin_rubber",
                         "core_line_radius", "browning_dE"], "Key register entries of the apple."))
    ap = SC.SCHEMAS["apple"]
    w(r"""\paragraph{States, dynamics, tasks} Schema attributes: """ + esc(", ".join(ap["attributes"])) +
      "; actions: " + esc(", ".join(ap["actions"])) + r""". \texttt{AppleTruth} bruises under a static load when the Hertz
maximum shear $0.31\,p_0$ exceeds the flesh shear strength, and under impact with a bruise volume linear in absorbed
energy above a threshold (fsct2008 anchor, SOURCED-VERIFY); \texttt{AppleBelief} tracks the expected bruise after
known drops and grips. There are \textbf{no apple task designators and no apple bench} yet; the apple's cutting force
is ILLUSTRATIVE (no source found).
""")
    w(r"""\subsection{Hass avocado}
One instance is built (\texttt{avocado\_4}, """ + f"{v4['whole']['mass_g']:.1f}" + r"""~g); the knife goes round the
seed, so \texttt{half\_a} keeps the whole seed and \texttt{half\_b} has the seed cavity. The exocarp has five layers
including the stone-cell layer that lets ripe Hass peel cleanly, the flesh three colour bands and six vascular
strands, and the seed a two-layer testa, two cotyledons and the embryonic axis.
""")
    w(fig("renders/food/anatomy_avocado_labelled.png", "Exploded anatomy plate of avocado\\_4 (existing render).",
          r"0.28\textheight", maxw=2400))
    w(figrow([("renders/food/avocado_halves.png", "whole and halves"),
              ("renders/food/closeup_avocado_seed.png", "half with seed")], "Avocado renders (existing, path-traced)."))
    w(tree_table("avocado", "avocado_4", v4, "Part tree of avocado\\_4 (indexed siblings merged)."))
    w(keytable("avocado", ["length_L", "diameter_D", "skin_thickness", "seed_length", "ripeness_stage",
                           "ripening_rate", "penetrometer_force", "flesh_E", "flesh_poisson", "bruise_force_ratio"],
               "Key register entries of the avocado."))
    av = SC.SCHEMAS["avocado"]
    w(r"""\paragraph{States, dynamics, tasks} Schema attributes: """ + esc(", ".join(av["attributes"])) +
      "; actions: " + esc(", ".join(av["actions"])) + r""". Ripeness advances with time on the Avocados Australia scale
(STANDARD); flesh bruises when contact pressure exceeds a fraction of the penetrometer pressure of the stage; halving
around the seed separates cleanly only from stage~3. \texttt{AvocadoBelief} updates the ripeness stage from squeeze
stiffness and skin darkness. The halving and pitting designators, the probabilistic node and everything the robot
does with them follow in Section~\ref{sec:avocado-reasoning}.
""")
    ratio = {l["stage"]: l["k_secant_4N_N_per_mm"] / avo_hertz[str(l["stage"])] for l in avo["lanes"]}
    soft = [r_ for s_, r_ in ratio.items() if s_ >= 3]
    hard = sorted((s_, r_) for s_, r_ in ratio.items() if s_ < 3)
    probe = avo_probe()
    p1 = [(s_, E_, k_) for s_, E_, k_ in probe if s_ == min(ratio)]
    k1 = [l["k_secant_4N_N_per_mm"] for l in avo["lanes"] if l["stage"] == min(ratio)][0]
    AVO_TEXT = (f"The FEM is stiffer than this sphere model by a nearly constant factor of {min(soft):.2f}--"
                f"{max(soft):.2f} for stages 3--5. The avocado is not a sphere, so the factor is probably a shape effect (not "
                "established; in the tomato bench the FEM lies below Hertz instead), and the check confirms the trend "
                "over ripeness, not the absolute value. The factor falls to " +
                " and ".join(f"{r_:.2f} at stage {s_:g}" for s_, r_ in reversed(hard)) + ", and " +
                (f"a separate probe run gave {p1[0][2]:.1f}~N/mm for the stage-{p1[0][0]:g} modulus against "
                 f"{k1:.1f}~N/mm in the final run (the probe's solver settings are not recorded in its log): "
                 if p1 else "") +
                "the hard lanes are under-converged, as in the tomato and roll benches.")
    rows = []
    for l in avo["lanes"]:
        kh = avo_hertz[str(l["stage"])]
        rows.append([f"{l['stage']:g}", f"{l['E_MPa']:.3f}", f"{l['k_secant_4N_N_per_mm']:.2f}", f"{kh:.2f}",
                     f"{l['k_secant_4N_N_per_mm'] / kh:.2f}"])
    w(r"""\subsection{Bench: ripeness by squeezing (Isaac Sim FEM)}
\texttt{scripts/sim\_avocado\_squeeze.py} squeezes a PhysX volume-deformable avocado (avocado\_4's outer surface,
uniform flesh, seed and skin not separate) between a fixed and a force-controlled plate at six ripeness stages; the
flesh modulus per stage maps the penetrometer force to the Farinas~2021 modulus range (endpoints SOURCED, mapping
ILLUSTRATIVE). The secant stiffness table feeds the belief's stiffness model. As a cross-check this report computes
the project's own Hertz model (\texttt{food.sdt.hertz\_squeeze\_stiffness}, two contacts on a sphere of the fruit's
diameter """ + f"{avo['D_mm']:.1f}" + r"""~mm, $\nu$~0.45) at 4~N. """ + AVO_TEXT + r""" An earlier 1000~Hz run (\texttt{avocado\_squeeze\_1000hz.json})
is invalid: its rigid control lane moved 9~mm, so all its stiffnesses were set to a placeholder.
""")
    w(table(["r", "r", "r", "r", "r"], ["stage", "E (MPa)", "k FEM (N/mm)", "k Hertz (N/mm)", "FEM / Hertz"], rows,
            "Avocado squeeze stiffness at 4~N (test\\_reports/avocado\\_squeeze.json; Hertz column DERIVED in this "
            "report).", size="scriptsize"))
    w(r"""With the stiffness table, one simulated squeeze of a hidden stage-""" + f"{avb['true']:g}" +
      r""" fruit at 4~N (8~\% noise) moves the flat prior to a stage estimate of """ +
      f"{avb['one']['mean']:.2f}~$\\pm$~{avb['one']['sd']:.2f}" + r""", and a second squeeze at 2~N to """ +
      f"{avb['two']['mean']:.2f}~$\\pm$~{avb['two']['sd']:.2f}" + r""" with $P(\text{ready, stage} \ge 4) = """ +
      f"{avb['two']['p_ready']:.2f}" + r"""$ (recomputed here with the plot script's seeds). The bruise-safe plate
force falls from 8~N (the largest tested) for hard fruit to 1~N for ripe fruit (panel d), consistent with the
Hort~Innovation warning that a 10~N thumb press bruises firm-ripe fruit (hia2018, SOURCED).
""")
    w(fig("renders/food/avocado_squeeze.png", "Avocado squeeze study (existing figure, tools/plot\\_avocado\\_squeeze"
          ".py): FEM force-compression curves, stiffness model, belief update, bruise-safe force.", r"0.25\textheight",
          maxw=2400))
    # ------------------------------------------------------------------ 4 wave 1
    calib = md_table("docs/wave1/README.md", "Calibration anchors")
    belief = md_table("docs/wave1/README.md", "Belief models scored")
    w(r"""\clearpage\section{Wave 1: bread roll, banana, carrot}
Wave~1 was built with the twelve-step pipeline of the briefing document (scope, research notes, part tree, register,
geometry, USD, hidden models, beliefs, designators, bench, renders, review). The research notes are in
\texttt{docs/wave1/}; the summary page \texttt{docs/wave1/README.md} is generated by \texttt{tools/wave1\_doc.py}.
""")
    if calib:
        w(table(["p{0.2\\linewidth}", "X", "p{0.27\\linewidth}", "p{0.18\\linewidth}"], [mdclean(c) for c in calib[0]],
                [[mdclean(c) for c in r] for r in calib[1:]],
                "Calibration anchors of the wave-1 hidden-state models (from docs/wave1/README.md).", size="scriptsize"))
    if belief:
        w(table(["p{0.1\\linewidth}", "X", "p{0.4\\linewidth}"], [mdclean(c) for c in belief[0]],
                [[mdclean(c) for c in r] for r in belief[1:]],
                "Wave-1 beliefs scored on simulated observations (from docs/wave1/README.md; 200 simulated "
                "instances each).", size="scriptsize"))

    def designator_table(kind):
        rows = []
        for name in tasks_by_item.get(kind, []):
            spec = (W1.TASKS.get(name) or W2.TASKS.get(name))
            dm = demos.get(name, {})
            rows.append([esc(name), esc(spec["goal"]), esc(spec["model"]),
                         {True: "ok", False: "failed"}.get(dm.get("ok"), "--")])
        return table(["p{0.16\\linewidth}", "X", "p{0.3\\linewidth}", "l"], ["Designator", "Goal", "Twin model",
                                                                           "Demo"], rows,
                     f"Task designators for the {esc(ITEMS[kind][0].lower())}. Demo: symbolic run on seed 0 by this "
                     "report (default parameters; a smoke test, not a validation).", size="scriptsize")

    def item_head(kind, intro):
        name, wave, inst, docs = ITEMS[kind]
        d = INST[inst]
        states = ", ".join(f"{k} ({v['mass_g']:.1f} g)" if v.get("mass_g") else k for k, v in d["cuts"].items())
        return (f"\\FloatBarrier\\subsection{{{esc(name)}}}\\label{{sec:{kind}}}\n" + intro +
                f" Built instance: {tt(inst)}, {d['whole']['mass_g']:.1f}~g; states: whole, {esc(states)}. "
                f"Research notes: {tt(docs)}.\n")

    w(item_head("roll", "A roll is a baked foam, not a plant organ: a dense browned crust round a crumb of gas cells. "
                "The twin is the oval single-score Weizenbrötchen; the score (Ausbund) is a separate crust part, the "
                "gas cells a point set that paints the cut faces."))
    w(figrow([("renders/food/anatomy_roll_labelled.png", "anatomy plate"), ("renders/food/wave1_roll_sliced.png",
                                                                            "sliced open")],
             "Roll: exploded anatomy plate and cut halves (existing renders).", r"0.2\textheight"))
    w(tree_table("roll", "roll_0", INST["roll_0"], "Part tree of roll\\_0."))
    w(keytable("roll", ["height", "crust_thickness_top", "crumb_density", "crumb_porosity", "crumb_E_kPa_by_age",
                        "crumb_poisson", "crust_water_model", "crust_water_critical", "crust_break_force",
                        "squeeze_k_model", "saw_setpoint"], "Key register entries of the roll."))
    w(r"""\paragraph{States and dynamics} Schema attributes: """ + esc(", ".join(SC.SCHEMAS["roll"]["attributes"])) +
      r""". Crust softening is water migration from the crumb (power law CALIBRATED through two hirte\_2012 points),
crumb staling a log-time interpolation of the Lagrain~2012 modulus (SOURCED-VERIFY, pan bread).""")
    w(designator_table("roll"))
    rs = jload("test_reports", "roll_squeeze_summary.json")
    rows = [[f"{a['E_kPa']:g}", f"{a['age_h']:g}", f"{a['k_incr_1to3N']:.3f}", f"{b['k_incr_1to3N']:.3f}",
             f"{c['change_pct']:+.1f}", f"{0.29 * a['E_kPa'] / 20:.2f}"]
            for a, b, c in zip(rs["runs"]["1000Hz_120it"], rs["runs"]["2000Hz_200it"], rs["convergence"])]
    w(r"""\paragraph{Bench: staling squeeze (Isaac Sim FEM)} \texttt{scripts/sim\_roll\_squeeze.py} squeezes a
crumb-only FEM roll (crust not separate: GAP) between 30~mm pads at four crumb moduli. Only the fresh lane converges
(+4.5~\% between solver settings); the stiffer lanes still rise with the solver rate and are lower bounds. The
register therefore uses the converged lane scaled linearly in $E$ (\texttt{squeeze\_k\_model}, DERIVED).
""")
    w(table(["r", "r", "r", "r", "r", "r"], ["E (kPa)", "age (h)", "k 1000 Hz", "k 2000 Hz", "change \\%",
                                            "k model (N/mm)"], rows,
            "Roll squeeze, incremental stiffness 1--3~N in N/mm (test\\_reports/roll\\_squeeze\\_summary.json).",
            size="scriptsize"))
    # banana
    w(item_head("banana", "A banana finger is a parthenocarpic berry of three carpels. The peel (exocarp and outer "
                "mesocarp) is four tissue layers with latex vessels; the pulp three carpel segments round the "
                "placental axis with ovule remnants and the 'strings'. Four finger instances are built."))
    w(figrow([("renders/food/anatomy_banana_labelled.png", "anatomy plate"),
              ("renders/food/wave1_banana_half.png", "lengthwise half")],
             "Banana: exploded anatomy plate and lengthwise half (existing renders).", r"0.2\textheight"))
    w(tree_table("banana", "banana_0", INST["banana_0"], "Part tree of banana\\_0."))
    w(keytable("banana", ["length_outer", "calibre", "peel_thickness", "ripeness_stage", "peel_hue_by_stage",
                          "pulp_E_compression", "peel_fracture", "finger_drop_force", "peel_bond_ratios",
                          "stage_rate_20C", "ridges"], "Key register entries of the banana."))
    w(r"""\paragraph{States and dynamics} Schema attributes: """ + esc(", ".join(SC.SCHEMAS["banana"]["attributes"]))
      + r""". Ripening on the seven-stage colour scale (STANDARD) at a SOURCED-VERIFY rate; peel hue per stage
SOURCED-VERIFY; the peeling bonds are ILLUSTRATIVE ratios of the peel fracture force, because no published peeling
force was found (GAP).""")
    w(designator_table("banana"))
    bp = jload("test_reports", "banana_peel.json")
    rows = [[f"{l['stage']:g}", f"{l['F_adh_N']:.1f} / {l['F_seam_N']:.1f} / {l['F_tear_N']:.1f}",
             f"{l['F_peak_N']:.1f}", f"{l['F_median_pulling_N']:.2f}",
             f"{l['strip0_segments_detached']}/{bp['segments_per_strip']}", str(l["others_detached"]), esc(l["outcome"])]
            for l in bp["lanes"]]
    w(r"""\paragraph{Bench: peeling (Isaac Sim)} \texttt{scripts/sim\_banana\_peel.py} models the peel as """ +
      f"{bp['strips']}" + r""" strips of """ + f"{bp['segments_per_strip']}" + r""" rigid segments joined by breakable
adhesion, seam and hinge joints, and pulls strip~1 from the snapped neck with a spring finger. The force level
inherits the ILLUSTRATIVE bond ratios, so only the trend is meaningful: green fruit needs higher peaks and its strip
may tear short; ripe fruit peels cleanly at a fraction of a newton. The stage-4 strip tore below its nominal strength
(joint forces of light segments are spiky), and the outcome flips between runs. The rendered filmstrip shows the strips
only faintly.
""")
    w(table(["r", "l", "r", "r", "r", "r", "X"], ["stage", "adh. / seam / tear (N)", "peak (N)", "median (N)",
                                                 "strip 1", "others", "outcome"], rows,
            "Banana peel bench (test\\_reports/banana\\_peel.json).", size="scriptsize"))
    # carrot
    w(item_head("carrot", "A carrot is a storage taproot: periderm, orange secondary phloem, vascular cambium, paler "
                "secondary xylem (the core) and the primary xylem; the crown carries petiole stubs, lenticel bands "
                "and lateral-root scars are point sets."))
    w(figrow([("renders/food/anatomy_carrot_labelled.png", "anatomy plate"),
              ("renders/food/wave1_carrot_crown_cut.png", "crown piece")],
             "Carrot: exploded anatomy plate and crown piece (existing renders).", r"0.2\textheight"))
    w(tree_table("carrot", "carrot_0", INST["carrot_0"], "Part tree of carrot\\_0."))
    w(keytable("carrot", ["length", "shoulder_diameter", "xylem_radius_fraction", "E_phloem", "poisson", "bend_Fmax",
                          "blade_shear", "limp_rwc", "softening_kinetics", "peel_depth", "boil_rounds_6mm"],
               "Key register entries of the carrot."))
    w(r"""\paragraph{States and dynamics} Schema attributes: """ + esc(", ".join(SC.SCHEMAS["carrot"]["attributes"]))
      + r""". Boiling uses the Peng~2014 softening kinetics (SOURCED) with a conduction lag and CALIBRATED doneness
thresholds; limpness follows turgor and relative water content (SOURCED-VERIFY).""")
    w(designator_table("carrot"))
    cp = jload("test_reports", "carrot_plank.json")
    rows = [[esc(l["variant"]), esc(l["mode"]), f"{l['roll_deg_at_Fcut']:.1f}", f"{l['slide_mm_at_Fcut']:.1f}",
             f"{l['F_at_escape_N']:.0f}" if l["F_at_escape_N"] is not None else "--"] for l in cp["lanes"]]
    w(r"""\paragraph{Bench: why cooks plank a carrot (Isaac Sim)} \texttt{scripts/sim\_carrot\_plank.py} presses a knife
onto carrot\_0 up to the cutting force (""" + f"{cp['F_cut_N']:.0f}" + r"""~N, the jah18 blade force scaled to this
root, SOURCED-VERIFY), straight down or with a slicing stroke, on the whole root and on a planked half. A straight
press acts through the axis and hardly moves the round root; a stroke rolls it away at once. The planked half holds
during strokes up to about 10~N, then the blade friction yaws it -- a claw-grip hand is needed. Escape criterion: """
      + esc(cp["escape_criterion"]) + ".\n")
    w(table(["l", "l", "r", "r", "r"], ["carrot", "knife", "roll at F\\textsubscript{cut} (deg)", "slide (mm)", "escapes at (N)"],
            rows, "Carrot plank bench (test\\_reports/carrot\\_plank.json).", size="scriptsize"))
    w(fig(f_bc, "Banana and carrot bench traces drawn by this report from the stored logs: (a) finger force during "
          "peeling per ripeness stage; (b) roll angle of the carrot over knife force (log scale above 1 deg).",
          r"0.25\textheight"))
    # ------------------------------------------------------------------ 5 wave 2
    w(r"""\clearpage\section{Wave 2: loaf, onion, orange, potato, tomato}
The five wave-2 items are built as USD assets with cut states and have exploded anatomy plates, research notes in
\texttt{docs/wave2/} and ten task designators (\texttt{food/wave2\_tasks.py}). """ + (
      r"""Path-traced renders of each item, whole and cut, were added on 20~October. """ if W2_RENDERED else
      r"""They have \textbf{no beauty renders} yet (only the plates). """) + r"""They have no entries in
\texttt{food/schema.py} (their discrete state comes from the truth classes' \texttt{discrete()}), and only the
tomato has a physics bench.
""")
    if os.path.exists(R("renders", "food", "wave2_group.png")):
        w(fig("renders/food/wave2_group.png", "Wave~2 items, path-traced: sandwich loaf, onion, orange, potato and "
              "tomato.", r"0.22\textheight"))
    w2 = {"loaf": ("Industrial German Toastbrot, baked in a lidded tin: thin soft crust on the tin faces and the flat "
                   "top, square cut ends (most packs have no heel slices), pre-sliced into 20 slices.",
                   ["length", "crumb_density", "slices", "heels", "crust_thickness_top", "firmness_by_day",
                    "tensile_failure", "toast_time", "fridge_staling_factor"]),
          "onion": ("Swollen leaf bases (fleshy scales) on a compressed stem, the basal plate, under dry papery "
                    "outer skins; each scale is a revolved region converging on the plate.",
                    ["diameter", "n_fleshy", "scale_thickness_outer", "n_dry", "flesh_density", "bulb_E", "tissue_E",
                     "crushing_load", "lf_window", "dice_spacing"]),
          "orange": ("A hesperidium: flavedo with oil glands, spongy albedo, segment membranes (endocarp) with juice "
                     "vesicles, three vascular bundles per segment and the central axis; navel or Valencia.",
                     ["cultivar", "diameter", "peel_thickness", "segments", "flavedo_E", "albedo_E",
                      "gland_burst_pressure", "juice_fraction", "peel_strip_force", "bitter_juice_h"]),
          "potato": ("The tuber is a stem: periderm with lenticels, eyes on a spiral (denser at the rose end), cortex, "
                     "vascular ring, perimedullary starch store and a branched pith.",
                     ["cultivar", "length", "width", "eyes", "phyllotaxis", "periderm_thickness", "density", "E",
                      "softening", "boil_times", "peel_depth"]),
          "tomato": ("Skin, outer pericarp wall, septa, columella with placenta, locules filled with gel round the "
                     "seeds; 2--4 locules; truss tomatoes keep calyx and pedicel.",
                     ["diameter", "locules", "pericarp_thickness", "plate_firmness", "skin_rupture_2mm", "skin_E",
                      "damage_strain", "skin_hue_by_stage", "chilling_threshold", "flesh_density"])}
    for kind, (intro, keys) in w2.items():
        inst = ITEMS[kind][2]
        states = {json.dumps(demos[n]["state"], sort_keys=True) for n in tasks_by_item.get(kind, []) if
                  demos.get(n, {}).get("state")}
        attrs = sorted({k for st_ in states for k in json.loads(st_)})
        if attrs:
            intro += " Discrete attributes exposed by the truth class: " + esc(", ".join(attrs)) + "."
        w(item_head(kind, intro))
        if os.path.exists(R("renders", "food", f"item_{kind}.png")):
            w(figrow([(f"renders/food/anatomy_{kind}_labelled.png", "exploded anatomy plate"),
                      (f"renders/food/item_{kind}.png", "whole and cut half on the worktop")],
                     f"{esc(inst)}: anatomy plate and path-traced render (Isaac Sim 6.2).", r"0.19\textheight",
                     maxw=2400))
        else:
            w(fig(f"renders/food/anatomy_{kind}_labelled.png", f"Exploded anatomy plate of {esc(inst)} (existing "
                  "render). No other render of this item exists.", r"0.26\textheight", maxw=2400))
        w(tree_table(kind, inst, INST[inst], f"Part tree of {esc(inst)} (indexed siblings merged)."))
        w(keytable(kind, keys, f"Key register entries of the {esc(ITEMS[kind][0].lower())}."))
        w(designator_table(kind))
        if kind == "potato":
            w(r"""\paragraph{Model checks reported in the notes} \texttt{docs/wave2/POTATO.md} reports the boiling
model (conduction lag plus first-order softening, k100 and Ea SOURCED) calibrated to 2~cm cubes done at 17.5~min
(sources 15--20) and checked on 3~cm cubes (22~min) and a large 75~mm tuber (50~min; sources up to 60); a whole
55~mm tuber takes 34~min against 20--25 in German guides, so the model runs long there. The register's 1.1~mm peel
removes 10.0~\% of the twin's mass against about 9~\% measured -- an independent check of the geometry.
""")
        if kind == "orange":
            w(r"""The hand-juicing model gives 63~mL for orange\_0 (notes; source 60--80~mL by hand, TYPICAL). Peeling
force in newtons is a GAP; the register's strip forces are ILLUSTRATIVE.
""")
        if kind == "tomato":
            rows = []
            t1 = {l["E_MPa"]: l for l in jload("test_reports", "tomato_squeeze_1000hz.json")["lanes"]}
            meas = tom4["measured_ince16"]
            lab = {min(tl): "red", max(tl): "green"}
            for E in sorted(tl):
                rows.append([f"{E:g}", f"{t1[E]['k_fem_4_20N']:.2f}", f"{tl[E]['k_fem_4_20N']:.2f}",
                             f"{tl[E]['k_hertz_4_20N']:.2f}", f"{tl[E]['k_fem_4_20N'] / tl[E]['k_hertz_4_20N']:.2f}",
                             (f"{lab[E]}: {meas['initial_N_per_mm'][lab[E]]:g} / {meas['average_N_per_mm'][lab[E]]:g}"
                              if E in lab else "--")])
            w(r"""\paragraph{Bench: plate firmness (Isaac Sim FEM)} \texttt{scripts/sim\_tomato\_squeeze.py} is an
independent consistency check of the register: lanes take the whole-fruit Hertz modulus of one study (tomato\_mech,
0.58~MPa small red .. 1.97~MPa small green) and compare the FEM plate stiffness with Hertz theory and with the plate
firmness measured by another study (ince16). Analytically the red fruit matches; the green one comes out lower than
measured. The FEM converges for the soft lane only at 4000~Hz / 250 iterations; at 1000~Hz all lanes gave about
1.8~N/mm whatever the modulus.
""")
            w(table(["r", "r", "r", "r", "r", "X"], ["E (MPa)", "FEM 1000 Hz", "FEM 4000 Hz", "Hertz", "FEM/Hertz",
                                                    "measured initial / average"], rows,
                    "Tomato plate stiffness 4--20~N in N/mm (test\\_reports/tomato\\_squeeze*.json). Measured "
                    "values ince16, SOURCED-VERIFY.", size="scriptsize"))
    w(fig(f_squeeze, "Squeeze benches compared by this report: (a) tomato FEM at two solver settings vs Hertz and "
          "the measured plate firmness; (b) avocado FEM vs the project's Hertz model and an unrecorded-settings probe "
          "run; (c) roll FEM at two solver settings vs the register's linear model. Common finding: soft lanes "
          "converge, stiff lanes need a higher solver rate than used.", r"0.25\textheight"))
    # ------------------------------------------------------------------ 6 wave 3
    rows = []
    for kind in WAVE3_CLS:
        d = w3.get(kind, {})
        n = len(d.get("parts", []))
        rows.append([esc(ITEMS[kind][0]), tt(f"food/{kind}.py"), str(len(reg[kind])),
                     f"{100 * PR.status_counts(kind).get('ILLUSTRATIVE', 0) / len(reg[kind]):.0f}",
                     ("yes, " + str(n) + " parts") if d.get("ok") else "failed: " + esc(d.get("err", "?")),
                     esc(", ".join(INST[f"{kind}_0"]["cuts"].keys())) if f"{kind}_0" in INST else "no",
                     "plate + render" if os.path.exists(R("renders", "food", f"item_{kind}.png")) else "no"])
    w(r"""\clearpage\section{Wave 3: lemon, cucumber, croissant, baguette, pepper}
The five wave-3 items exist as code (\texttt{food/<item>.py} with geometry, part tree and asset spec;
\texttt{food/<item>\_models.py} with hidden models and beliefs) and as registers. """ + (r"""On 20~October they were
added to the asset builder and built (seed~0: USD part graph, cut states, belief and truth layers), rendered on the
worktop and as exploded anatomy plates. They still have no task designators and no benches. """ if W3_BUILT else
r"""They are \textbf{not built}: the asset builder does not list them, so there are no USD assets, no cut states, no
renders or anatomy plates, no task designators and no benches. """) + r"""Their module docstrings refer to research
notes in \texttt{docs/wave3/}, which \textbf{does not exist} in the repository. The part trees below come from
instantiating each item (seed~0) on the CPU.
""")
    if os.path.exists(R("renders", "food", "wave3_group.png")):
        w(fig("renders/food/wave3_group.png", "Wave~3 items (seed~0), path-traced: lemon, cucumber, croissant, "
              "baguette and pepper.", r"0.24\textheight"))
    w(table(["p{0.16\\linewidth}", "l", "r", "r", "l", "p{0.2\\linewidth}", "X"], ["Item", "Module", "Reg.",
                                                                 "\\% ILL.", "instantiates", "USD states", "images"],
            rows,
            "Wave-3 status (checked by this report; cache in report/cache/wave3\\_parts.json).", size="scriptsize"))
    w3keys = {"lemon": ["diameter", "shape_index", "peel_thickness", "juice_fraction", "puncture_force",
                        "cutting_force", "compression_force", "nipple_height"],
              "cucumber": ["length", "diameter", "skin_thickness", "cavity_fraction", "puncture_3mm", "E_apparent",
                           "firmness_loss_9d", "chilling"],
              "croissant": ["length", "mass_check", "rolling_turns", "butter_layers", "crumb_density",
                            "crust_thickness", "firmness", "flakes"],
              "baguette": ["length", "width", "n_scores", "crust_thickness_top", "crumb_density", "crumb_porosity",
                           "cell_diameter_median", "crust_L"],
              "pepper": ["diameter", "length", "lobes", "wall_thickness", "seeds", "rib_depth", "firmness_whole",
                         "weight_loss", "red_LCh"]}
    w3desc = {"lemon": "A hesperidium like the orange (the class derives from it) with a prolate outline, the stylar "
                       "nipple at the blossom end and a short neck at the stem end; thicker, paler albedo.",
              "cucumber": "A pepo of three fused carpels: skin, firm mesocarp, a seed cavity with three placentae "
                          "carrying immature seeds round a small core; bent onto a gentle arc.",
              "croissant": "A rolled triangle of laminated dough: a glazed crust, a laminated crumb whose cut faces "
                           "show the spiral of dough sheets and voids, and loose flakes.",
              "baguette": "The roll's baked-foam model on a long pointed footprint with five to seven overlapping "
                          "diagonal scores (grignes) and a thicker crust.",
              "pepper": "A hollow berry: pericarp wall round an air cavity, 2--4 lobes, ribs (septa), a placenta "
                        "hanging from the stem end with the seeds, peduncle and calyx."}
    for kind in WAVE3_CLS:
        has_r = os.path.exists(R("renders", "food", f"item_{kind}.png"))
        w(f"\\FloatBarrier\\subsection{{{esc(ITEMS[kind][0])}}}\n{esc(w3desc[kind])}"
          + ("" if has_r else " No render exists.") + "\n")
        if has_r:
            w(figrow([(f"renders/food/anatomy_{kind}_labelled.png", "exploded anatomy plate"),
                      (f"renders/food/item_{kind}.png", "whole and cut half on the worktop")],
                     f"{esc(kind)}\\_0: anatomy plate and path-traced render (Isaac Sim 6.2).", r"0.19\textheight",
                     maxw=2400))
        d = w3.get(kind, {})
        if d.get("ok"):
            tr = collapse_tree([(p, c, o, None) for p, c, o in d["parts"]])
            rows = [["\\hspace*{" + f"{0.9 * dep:.1f}" + "em}" + esc(nm.replace("_k", "_1.." + str(n)) if n > 1 else nm),
                     tt(c), esc(o), str(n) if n > 1 else ""] for dep, nm, c, o, n, _ in tr]
            w(table(["p{0.32\\linewidth}", "p{0.3\\linewidth}", "X", "r"], ["Part (below /Food/Parts)", "class",
                                                                           "observability", "n"], rows,
                    f"Part tree of {esc(kind)} seed 0 from {tt('parts()')}" + ("." if W3_BUILT else " (not built as USD)."),
                    size="scriptsize"))
        w(keytable(kind, w3keys[kind], f"Key register entries of the {esc(ITEMS[kind][0].lower())}."))
    # ------------------------------------------------------------------ 7 kitchens
    urdfs = {"EASE (IAI) apartment": os.path.expanduser("~/ResearchSpace/foundation/software/cram/cram_viz/scenes/"
                                                        "garmi_ease_apartment/apartment.urdf"),
             "GARMI (TUM MIRMI) apartment": os.path.expanduser("~/ResearchSpace/foundation/software/cram/cram_viz/"
                                                               "scenes/garmi_garmiapartment/scene-bodies.urdf"),
             "GARMI robot (cleaned URDF)": R("assets", "apartments", "garmi_robot", "garmi_clean.urdf")}
    rows = []
    for name, p in urdfs.items():
        if os.path.exists(p):
            t = open(p).read()
            links = len(re.findall(r"<link\b", t))
            joints = re.findall(r"<joint\b[^>]*type=\"([a-z]+)\"", t)
            mov = sum(j in ("revolute", "prismatic", "continuous") for j in joints)
            rows.append([esc(name), str(links), str(len(joints)), str(mov)])
    probe = open(R("renders", "probe_garmi.log")).read() if os.path.exists(R("renders", "probe_garmi.log")) else ""
    m = re.search(r"dofs=(\d+) links=(\d+)", probe)
    ge = open(R("renders", "garmi_egg.log")).read() if os.path.exists(R("renders", "garmi_egg.log")) else ""
    nan_fail = "cannot convert float NaN" in ge
    import avocado_reasoning as AR
    AR.write(w, sys.modules[__name__])
    import avocado_theory as AT
    AT.write(w, sys.modules[__name__])
    w(r"""\clearpage\section{The kitchens: EASE and GARMI apartments}\label{sec:kitchen}
\texttt{scripts/import\_apartments.py} imports the EASE and GARMI apartments and the GARMI robot from the CRAM viz
scene bundles (read in place from \texttt{\textasciitilde/ResearchSpace}, which stays read-only) into USD with Isaac
Sim's URDF importer: doors and drawers become joints, collisions are convex decompositions. The GARMI URDF needed its
gazebo, ros2\_control and transmission blocks stripped before the importer would accept it.
""")
    w(table(["X", "r", "r", "r"], ["Scene", "links", "joints", "movable joints"], rows,
            "Imported scenes, counted in the source URDFs by this report."))
    w(r"""\paragraph{The GARMI rig} \texttt{household/garmi\_rig.py} drives the imported GARMI (two FR3 arms with Franka
hands, a two-stage lift, a pan-tilt head, a wheeled base) as one PhysX articulation with the base fixed for
manipulation""" + (f" (the probe found {m.group(1)} degrees of freedom and {m.group(2)} links)" if m else "") +
      r""": damped-least-squares Cartesian IK per arm on the PhysX Jacobian with a null-space posture bias,
force-limited grasping (finger drives in force mode, targets ramped -- an instantly set stiff target slams the
fingers shut, a lesson from the egg bench), and the Franka TCP 0.1034~m along the hand axis.

\paragraph{Food in the kitchen} \texttt{scripts/render\_apartment\_kitchen.py} places GARMI and a cutting board on the
EASE kitchen island with seven food assets: apple\_3 whole and half, avocado\_4 whole and half, and three eggs. Only
these egg-set items have been placed; none of the wave-1, wave-2 or wave-3 items is in a kitchen yet, and the GARMI
apartment has been imported but not rendered or furnished with food.
""" + (r"""\paragraph{End-to-end task: not yet working} \texttt{scripts/sim\_garmi\_egg.py} is the first
end-to-end task (GARMI picks an egg, cracks it on the bowl rim and opens it with both hands). Its last run
(\texttt{renders/garmi\_egg.log}) stopped during the approach: the arm's joint state and TCP became NaN and the
controller raised \emph{cannot convert float NaN to integer}. No GARMI task log exists yet.
""" if nan_fail else ""))
    w(fig("renders/apartment/ease_kitchen_overview.png", "The EASE apartment kitchen with GARMI (existing render, "
          "path-traced in Isaac Sim~6.2).", r"0.36\textheight"))
    w(figrow([("renders/apartment/ease_kitchen_island.png", "island with board and food"),
              ("renders/apartment/ease_island_food.png", "close-up: apple, avocado, eggs")],
             "Food twins on the EASE kitchen island (existing renders)."))
    # ------------------------------------------------------------------ 8 data quality
    rows = []
    for kind in reg:
        c = PR.status_counts(kind)
        n = len(reg[kind])
        rows.append([esc(kind)] + [str(c.get(s, 0)) for s in STATUS_ORDER] + [str(n)])
    rows.append(["\\textbf{all}"] + [f"\\textbf{{{allc.get(s, 0)}}}" for s in STATUS_ORDER] + [f"\\textbf{{{n_reg}}}"])
    w(r"""\clearpage\section{Data quality}\label{sec:quality}
""")
    w(fig(f_status, "Register entries by status, per item and overall (drawn by this report from food/params.py). "
          "Numbers in brackets: entries per item.", r"0.4\textheight"))
    w(f"Of {n_reg} register entries, {allc['SOURCED']} ({pct(allc['SOURCED'], n_reg)}) are SOURCED, "
      f"{allc['SOURCED-VERIFY']} ({pct(allc['SOURCED-VERIFY'], n_reg)}) SOURCED-VERIFY, {allc['STANDARD']} STANDARD, "
      f"{allc['DERIVED']} DERIVED, {allc['CALIBRATED']} CALIBRATED, {allc['TYPICAL']} TYPICAL and {ill} "
      f"({pct(ill, n_reg)}) ILLUSTRATIVE. The ILLUSTRATIVE values concentrate where sources end: colours "
      "(L*C*h of flesh and skin), internal dimensions (membrane, rib and core sizes), friction coefficients, the "
      "peeling and cutting forces, and the outcome constants of the strike and peel models. The sampled instance "
      "state (age, storage) is ILLUSTRATIVE by design: it is the scenario, not a property of food.\n")
    w(table(["l"] + ["r"] * (len(STATUS_ORDER) + 1), ["Item"] + [esc(s) for s in STATUS_ORDER] + ["total"], rows,
            "Register status counts per item (food/params.py).", size="scriptsize"))
    CHECKS.append(("SOURCED / SOURCED-VERIFY / CALIBRATED entries naming a source key of params.SOURCES",
                   f"{n_sourced - sum(unresolved.values())}/{n_sourced}"))
    CHECKS.append(("source keys in params.SOURCES that the register cites", f"{len(used_keys)}/{len(PR.SOURCES)}"))
    CHECKS.append(("egg assets built with the current egg register",
                   "no: " + ", ".join(stale_egg) + " were built with an earlier version" if stale_egg else "yes"))
    if truth_meff is not None:
        CHECKS.append(("egg_1 truth.usda impact_effective_mass_fraction vs register",
                       f"{truth_meff:g} in the asset vs {PR.EGG['impact_effective_mass_fraction'][1]:g} "
                       "(CALIBRATED) in the register"))
    CHECKS.append(("docs/ANATOMY.md up to date with the register",
                   "no: generated before the impact calibration (shows 0.3, ILLUSTRATIVE)"
                   if "impact_effective_mass_fraction | 1 | 0.3 | ILLUSTRATIVE" in open(R("docs", "ANATOMY.md")).read()
                   else "yes"))
    CHECKS.append(("research notes for wave 3 (docs/wave3)", "missing" if not os.path.isdir(R("docs", "wave3"))
                   else "present"))
    CHECKS.append(("wave-1 and wave-2 designator demos (seed 0)",
                   f"{sum(1 for d in demos.values() if d.get('ok'))}/{len(demos)} reach their goal"))
    w(r"""\paragraph{Consistency checks run by this report} The checks below are computed when the report is built;
they test the repository, not the food.
""")
    w(table(["X", "p{0.45\\linewidth}"], ["Check", "Result"], [[esc(a), esc(b)] for a, b in CHECKS],
            "Repository consistency checks."))
    w(r"""Further known inconsistencies: \texttt{docs/wave1/ROLL.md} gives roll\_0 as 48.1~g (the part sum) while the
asset's whole body is 47.4~g; the \texttt{food/dynamics.py} docstring still calls both impact constants ILLUSTRATIVE
although one is now CALIBRATED; the drop-surface stiffnesses and the strike-model constants live in code
(\texttt{dynamics.SURFACES}, \texttt{egg\_crack.P}) and are mirrored only as a text note in the register.
""")
    # ------------------------------------------------------------------ 9 limitations
    w(r"""\section{Limitations and next steps}\label{sec:next}
\paragraph{What is validated, and how far}
\begin{itemize}
\item \emph{Against published measurements, independently:} the tomato Hertz stiffness from one study's modulus
matches another study's measured red-fruit plate firmness (the green fruit does not); the potato peel loss (10.0 vs
about 9~\%); part and cut masses against the register's mass checks.
\item \emph{Fits:} the egg boiling thresholds, the egg impact constant, the roll crust-water curve and the carrot boiling
thresholds are CALIBRATED -- they reproduce the data they were fitted to.
\item \emph{Solver-limited:} the FEM squeezes converge only for soft lanes (ripe avocado, fresh roll, red tomato at
4000~Hz); stiff lanes are lower bounds.
\item \emph{Mechanism only:} banana peeling, carrot planking and the egg crack study show plausible trends, but their
forces or outcome constants are ILLUSTRATIVE.
\item \emph{Not validated at all:} appearance (no comparison with reference photographs yet), wave-3 behaviour, any
real robot run.
\end{itemize}
\paragraph{Gaps in the build}
\begin{itemize}
\item \textbf{Anatomy review gate.} The exploded plates were sent for the user's review; the gate is still pending for
every item, so all part trees are provisional.
""" + (r"""\item \textbf{Wave~3 incomplete:} built and rendered (20~October), but the \texttt{docs/wave3/} research
notes, task designators, schema entries and benches are missing.
""" if W3_BUILT else r"""\item \textbf{Wave~3 not built:} add lemon, cucumber, croissant, baguette and pepper to the asset builder,
write the missing \texttt{docs/wave3/} notes, plates, renders, designators.
""") + (r"""\item \textbf{Wave~2} has no schema entries; only the tomato has a bench.""" if W2_RENDERED else
       r"""\item \textbf{Wave~2 without renders} and without schema entries; only the tomato has a bench.""") + r"""
\item \textbf{Apple without designators or bench.} The avocado has designators, an emulator and the PR2 scenario
(kinematic: no contact forces; Section~\ref{sec:avocado-reasoning}).
\item \textbf{Kitchens:} only the egg set has been placed (EASE); the GARMI apartment is imported but unused; the
GARMI egg task fails with NaN in the arm controller.
\item \textbf{Bodies are simplified in physics:} homogeneous FEM bodies (no skin, seed or crust), rigid peel segments,
kinematic shell fragments, a rigid yolk.
\item \textbf{Stale artefacts:} rebuild the egg assets and \texttt{docs/ANATOMY.md} with the current register; fix the
bread part-mass overlap.
\end{itemize}
\paragraph{Next steps} (1) Pass the anatomy review; (2) rerun the stiff FEM lanes at higher solver rates (the
memory note: 4000~Hz / 250 iterations was enough for soft lanes only) or replace them by the analytic models; (3) fix
the GARMI IK failure and log the first end-to-end egg task in the EASE kitchen; (4) build wave~3 and render wave~2;
(5) for the avocado: calibrate the photo ripeness likelihood and the peeling monitor on real images, give the PR2
contact physics for the knife, the twist and the spoon (breakable joints carrying the bridge and adhesion strengths),
and estimate the seed's sideways offset; (6) verify the """ + f"{allc['SOURCED-VERIFY']}" + r""" SOURCED-VERIFY entries and calibrate the ILLUSTRATIVE ones that
the benches depend on (peel bonds, strike constants, cutting forces, friction); (7) test the avocado microtheory on
real fruit (penetrometer, squeeze stiffness and photos at known stages to calibrate the bruise ratio and the colour
and stiffness likelihoods) and take a second fruit through the theory-construction loop.
""")
    # ------------------------------------------------------------------ appendices
    w(r"""\clearpage\appendix\section{Parameter register}\label{app:register}
""")
    for kind in reg:
        rows = [reg_row(kind, k) for k in reg[kind]]
        w(table(["p{0.2\\linewidth}", "p{0.27\\linewidth}", "p{0.12\\linewidth}", "X"],
                ["Key", "Value / distribution", "Status", "Source and note"], rows,
                f"Register of the {esc(ITEMS[kind][0].lower())} (params.{kind.upper()}, {len(reg[kind])} entries).",
                size="scriptsize", long=True))

    def src_text(v):
        parts = []
        for tok in v.split(" "):
            if ("http" in tok or re.match(r"^[\w.-]+\.(org|com|gov|edu|de|cz|be|it|rs|ac|net|io)\b", tok)
                    or (tok.count("/") >= 2 and len(tok) > 25)):
                t = tok.replace("\\", "/").replace("%", "\\%").replace("#", "\\#").replace("{", "").replace("}", "")
                parts.append("\\url{" + t + "}")
            else:
                parts.append(esc(tok))
        return " ".join(parts)
    w(r"""\section{Sources}
""")
    rows = [[tt(k), src_text(v)] for k, v in PR.SOURCES.items()]
    w(table(["p{0.16\\linewidth}", "p{0.8\\linewidth}"], ["Key", "Source"], rows, "Sources held in food/params.py (SOURCES).",
            size="scriptsize", long=True))
    w(r"""\section{Files and rebuild}
All paths are relative to \texttt{\textasciitilde/Documents/HouseholdTwin}.
\begin{Verbatim}[fontsize=\scriptsize,frame=lines]
food/params.py               parameter register (status per value) and SOURCES
food/<item>.py               analytic anatomy, part tree, asset spec per item
food/<item>_models.py        hidden-state models and beliefs (wave 1-3)
food/dynamics.py, sdt.py     egg/apple/avocado truth dynamics and beliefs
food/egg_crack.py, egg_cook.py, egg_open.py   egg crack, boil and opening models
food/schema.py               discrete SDT schema (egg, apple, avocado, roll, banana, carrot)
food/partgraph.py, usd_partgraph.py           closed meshes, cuts, USD part graph
food/egg_tasks.py, wave1_tasks.py, wave2_tasks.py   task designators
household/garmi_rig.py       GARMI articulation rig (IK, force-limited grasp)
scripts/build_food_assets.py assets/food/<kind>_<seed>/ (whole + cut states, sdt.usda, truth.usda)
scripts/import_apartments.py assets/apartments/ (EASE, GARMI apartment, GARMI robot)
scripts/sim_*.py             Isaac Sim benches -> test_reports/*.json
scripts/render_*.py          path-traced renders -> renders/
tools/*.py                   CPU studies, plots and generated docs
food/avocado_vision.py, avocado_belief.py    photo -> node estimates; particle-filter belief of the avocado node
food/avocado_peel.py, peel_monitor.py        peeling outcomes as USD states; vision-based peeling monitor
food/avocado_process.py, avocado_emulator.py, avocado_tasks.py   halving / pitting mechanics, emulator, designators
scripts/avocado_from_image.py, avocado_observe.py, avocado_process_demo.py, *peel*.py   CLI and studies
cram/avocado_sdt.py          node, competence questions, mental simulation, value of information (CRAM side)
cram/avocado_world.py, avocado_motions.py   Avocado(Fruit) annotation, pieces, PR2 motions
cram/avocado_pr2_demo.py     scripted PR2 scenario; cram/record_run.py -> cram/results/pr2_run.json
cram/avocado_kitchen.py      interactive kitchen for the CRAMERA viewer
report/build_report.py       this document (CPU only; cache in report/cache, figures in report/figures)
report/avocado_reasoning.py  the section on the avocado as the robot's belief state
cram/avocado_agent.py        goal-directed agent: effects to enforce / prevent, missions, lifetime-based decisions
food/avocado_lifetime.py     ripening, ripe window, decline, spoilage, browning under storage treatments
theory/avocado_bws.py, avocado_policies.py, axioms.py   research SDT, task programs, relational axioms and audit
theory/run_microtheory.py    held-out evaluation, ablations, misspecified plants -> theory/results/
theory/package.py, learning.py, prl.py   theory packages and gate, theory construction, programmable RL
report/avocado_theory.py     the section on the agent, the microtheory, theory construction and RL
report/build_avocado_microtheory.py   the standalone microtheory (report/avocado_microtheory.pdf)

E=~/Documents/LabAutomationTwin/env_cad/bin/python
I=~/Documents/LabAutomationTwin/env_isaacsim/bin/python
$E scripts/build_food_assets.py
$E tools/egg_crack_study.py && $E tools/egg_cook_study.py && $E tools/anatomy_doc.py
OMNI_KIT_ACCEPT_EULA=YES $I scripts/sim_egg_bench.py        (GPU, one job at a time)
OMNI_KIT_ACCEPT_EULA=YES $I scripts/render_food.py
$E report/build_report.py [--refresh]
\end{Verbatim}
\end{document}
""")
    open(TEX, "w").write("".join(out))
    r = subprocess.run(["latexmk", "-lualatex", "-interaction=nonstopmode", "-halt-on-error", os.path.basename(TEX)],
                       cwd=HERE, capture_output=True, text=True)
    print(r.stdout[-2500:] if r.returncode else "ok", r.returncode, f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
