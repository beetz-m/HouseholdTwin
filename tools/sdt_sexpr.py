"""The twin as one compositional tree of symbolic names, written as Lisp s-expressions.

Every SDT node of an item (the prims below /Food/Parts of its USD part graph) becomes a `node` form named by a local
RPL tag -- `(:tag vitelline-membrane (node ...))` inside `(:tag yolk ...)` inside `(:tag egg-3 ...)`. The node path
(egg-3 yolk vitelline-membrane) is implicit: computed from the node's position, never written. References go through
the tag table -- `(tagged vitelline-membrane egg-3)` -- and stay valid when the tree is restructured (RPL: ":TAG ...
give stable references that survive transformation"); knowledge bases use these references. The node carries its grounding (USD file +
prim path, the IRI of `sdt:represents`), its class, its observability, and the robot's belief where one exists. The
same reader resolves a name back to its grounding: the node IS the referent of the symbol.

Actions are written the same way: an executed emulator run (food/egg_emulator.py log.json) becomes an `episode` node
whose parts are the action tokens, each pointing at the node names it acted on -- the action tree mirrors the world
tree. Notation: docs/SDT_NOTATION.md.

    ~/Documents/LabAutomationTwin/env_cad/bin/python tools/sdt_sexpr.py assets/food/egg_3 [assets/food/avocado_4 ...]
        -> docs/sexpr/<instance>.lisp
    ... tools/sdt_sexpr.py --episode assets/food/egg_3/process/rim/log.json  -> docs/sexpr/<instance>_<run>.lisp
    ... tools/sdt_sexpr.py --ground "germinal-disc egg-3"                  -> implicit path, USD prim, IRI
"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "sexpr")
BELIEF_KEYS = ("age_days", "mass_g", "air_cell_mm", "haugh_unit", "yolk_index", "albumen_pH", "F_crack_equator_N",
               "float_tilt_deg", "p_clean_peel", "microcracked", "ripeness_stage", "seed_length", "length_L")


# %% names
def kebab(s):
    s = s.replace("pH", "Ph").replace("_N", "N")              # albumen_pH -> albumen-ph, F_crack_equator_N -> ...-n
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", s).replace("_", "-")
    return s.lower()


def class_symbol(cls):
    """food:EggCuticle -> food:egg-cuticle (namespace kept, local name in kebab case)."""
    ns, _, local = cls.rpartition(":")
    return f"{ns}:{kebab(local)}" if ns else kebab(local)


def instance_symbol(instance_dir):
    return kebab(os.path.basename(os.path.normpath(instance_dir)))


def atom(v):
    if isinstance(v, bool):
        return "t" if v else "nil"
    if isinstance(v, (int, float)):
        return f"{v:.4g}" if isinstance(v, float) else str(v)
    s = str(v)
    if re.fullmatch(r"[a-z0-9:][a-z0-9:/.\-]*", s):
        return s
    return json.dumps(s)


# %% item tree from the USD part graph
def item_tree(instance_dir):
    from pxr import Usd
    sym = instance_symbol(instance_dir)
    rel = os.path.relpath(instance_dir, ROOT)
    st = Usd.Stage.Open(os.path.join(instance_dir, "sdt.usda"))
    root = st.GetPrimAtPath("/Food")

    def attr(p, k):
        a = p.GetAttribute(k)
        return a.Get() if a and a.HasAuthoredValue() else None

    def node(p, name, depth):
        pad = "  " * depth
        cls = attr(p, "sdt:class")
        tag = name.split("/")[-1]                          # the local tag; the path stays implicit (RPL :tag)
        lines = [f"{pad}(:tag {tag}", f"{pad} (node"]
        if cls:
            lines.append(f"{pad}  (is-a {class_symbol(cls)})")
        obs = attr(p, "sdt:observability")
        if obs:
            lines.append(f"{pad}  (observability {atom(kebab(obs.replace(' ', '-')))})")
        es = attr(p, "sdt:epistemicStatus")
        if es:
            lines.append(f"{pad}  (epistemic-status {atom(es)})")
        iri = attr(p, "sdt:represents")
        prim = str(p.GetPath())
        lines.append(f'{pad}  (grounding (usd "{rel}/whole.usdc" "{prim}")' + (f' (iri "{iri}"))' if iri else ")"))
        if p.GetPath() == root.GetPath():
            m = attr(p, "physics:mass")
            if m is not None:
                lines.append(f"{pad}  (mass-kg {float(m):.4g} :status computed :from part-graph)")
            b = belief_forms(instance_dir, depth + 1)
            if b:
                lines.append(b)
        parts = p.GetPrimAtPath("Parts") if p.GetPath() == root.GetPath() else p
        kids = [c for c in (parts.GetChildren() if parts else []) if c.GetAttribute("sdt:class")
                and c.GetName() not in ("Geom", "Visual", "Collision", "Looks")]
        if kids:
            lines.append(f"{pad}  (parts")
            for c in kids:
                lines.append(node(c, f"{name}/{kebab(c.GetName())}", depth + 2))
            lines[-1] += ")"
        lines[-1] += "))"
        return "\n".join(lines)
    return node(root, sym, 0)


def belief_forms(instance_dir, depth):
    from pxr import Usd
    path = os.path.join(instance_dir, "sdt_belief.usda")
    if not os.path.exists(path):
        return ""
    stage = Usd.Stage.Open(path)                        # keep the stage alive while its prims are read
    p = stage.GetPrimAtPath("/Food")
    pad = "  " * depth
    out = [f"{pad}(belief :representation \"particle filter\" :at-hours "
           f"{atom(float(p.GetAttribute('sdt:belief:time_h').Get() or 0.0))}"]
    for k in BELIEF_KEYS:
        a = p.GetAttribute(f"sdt:belief:{k}:mean")
        if not a or not a.HasAuthoredValue():
            continue
        g = lambda f: float(p.GetAttribute(f"sdt:belief:{k}:{f}").Get())  # noqa: E731
        unit = p.GetAttribute(f"sdt:belief:{k}:unit").Get()
        out.append(f"{pad}  ({kebab(k)} :mean {g('mean'):.4g} :sd {g('sd'):.3g} :p05 {g('p05'):.4g} "
                   f":p95 {g('p95'):.4g} :unit {atom(unit)})")
    out[-1] += ")"
    return "\n".join(out)


# %% episodes: the action tree of an emulator run
def episode_tree(log_path):
    d = json.load(open(log_path))
    inst = f"egg-{d['instance']}"
    run = kebab(os.path.basename(os.path.dirname(log_path)))
    name = f"{run}"
    ref = lambda *tags: f"(tagged {' '.join(tags)})" if len(tags) > 1 else tags[0]   # noqa: E731
    target = {"read_carton": inst, "weigh": inst, "candle": ref("air-cell", inst), "float_test": inst,
              "grip": ref("shell", inst), "strike": ref("shell", inst), "open_up": ref("shell", inst),
              "fish_out": "bowl-1", "boil": inst, "peel": ref("shell", inst), "wait": inst}
    lines = [f"(:tag {name}", " (node", "  (is-a aicor:manipulation-episode)", f"  (object {inst})",
             f'  (grounding (record "{os.path.relpath(log_path, ROOT)}"))', "  (parts"]
    counts = {}
    for e in d["log"]:
        a = e["action"]
        counts[a] = counts.get(a, 0) + 1
        tok = f"{kebab(a)}-{counts[a]}"
        meas = " ".join(f":{kebab(k)} {atom(v)}" for k, v in e["measured"].items() if not isinstance(v, (dict, list)))
        lines.append(f"    (:tag {tok} (node")
        lines.append("      (is-a cram:action-token)")
        lines.append(f"      (designator (the action (type {kebab(a)}) (object {target.get(a, inst)})))")
        lines.append(f"      (at-hours {atom(float(e['t_h']))})")
        lines.append(f"      (outcome {atom(kebab(str(e['outcome']).replace(' ', '-')))} :ok {atom(bool(e['ok']))})")
        if meas:
            lines.append(f"      (measured {meas})))")
        else:
            lines[-1] += "))"
    lines[-1] += ")))"
    return "\n".join(lines), inst, run


# %% reader and grounding
TOKEN = re.compile(r'\s*(;[^\n]*|\(|\)|"(?:\\.|[^"])*"|[^\s()";]+)')


def parse(text):
    """Minimal s-expression reader: lists, strings, numbers, symbols (as str). Comments start with ';'."""
    stack, cur = [], []
    for m in TOKEN.finditer(text):
        t = m.group(1)
        if t.startswith(";"):
            continue
        if t == "(":
            stack.append(cur)
            cur = []
        elif t == ")":
            if not stack:
                raise ValueError(f"unbalanced ')' at offset {m.start()}")
            done, cur = cur, stack.pop()
            cur.append(done)
        elif t.startswith('"'):
            cur.append(json.loads(t))
        else:
            try:
                cur.append(float(t) if re.fullmatch(r"-?\d+(\.\d*)?(e-?\d+)?", t) else t)
            except ValueError:
                cur.append(t)
    if stack:
        raise ValueError(f"{len(stack)} unclosed '('")
    return cur


class TagTable:
    """The tag table of a tree (RPL :TAG): tag -> node handle. Node paths are implicit -- computed from the node's
    position when asked -- so a reference by tag stays valid when the tree is restructured (a subtree moved,
    replaced or re-parented); only the computed path changes."""

    def __init__(self, forms):
        self.parent, self.node, self.tag_of = {}, {}, {}
        self._walk(forms, None)

    def _walk(self, forms, parent):
        for f in forms:
            if isinstance(f, list) and len(f) == 3 and f[0] == ":tag":
                h = id(f[2])
                self.node[h], self.parent[h], self.tag_of[h] = f[2], parent, f[1]
                for g in f[2][1:]:
                    if isinstance(g, list) and g and g[0] == "parts":
                        self._walk(g[1:], h)

    def path(self, h):
        """The implicit node path: tags from the root to the node."""
        out = []
        while h is not None:
            out.append(self.tag_of[h])
            h = self.parent[h]
        return out[::-1]

    def find(self, tag, within=None):
        """Handle of the node tagged `tag`, optionally inside the node tagged `within` -- a (tagged TAG WITHIN) ref."""
        hits = [h for h, t in self.tag_of.items() if t == tag and (within is None or within in self.path(h)[:-1])]
        if len(hits) != 1:
            raise KeyError(f"(tagged {tag}{' ' + within if within else ''}): {len(hits)} nodes")
        return hits[0]

    def move(self, h, new_parent):
        """Restructure: re-parent a subtree (e.g. the yolk into the bowl's contents); tags stay bound."""
        self.parent[h] = new_parent


def index(forms):
    """implicit path ('a/b/c') -> node form, for printing and tests."""
    tt = TagTable(forms)
    return {"/".join(tt.path(h)): n for h, n in tt.node.items()}


def ground(name):
    """The grounding of a reference: 'egg-3', or 'germinal-disc egg-3' meaning (tagged germinal-disc egg-3)."""
    tags = name.split()
    inst = tags[-1]
    path = os.path.join(OUT, f"{inst}.lisp")
    tt = TagTable(parse(open(path).read()))
    h = tt.find(tags[0], tags[1] if len(tags) > 1 else None)
    node = tt.node[h]
    g = next(x for x in node if isinstance(x, list) and x and x[0] == "grounding")
    usd = next(x for x in g if isinstance(x, list) and x[0] == "usd")
    iri = next((x[1] for x in g if isinstance(x, list) and x[0] == "iri"), None)
    return dict(reference=f"(tagged {name})" if len(tags) > 1 else name, implicit_path=tt.path(h),
                usd_file=usd[1], prim=usd[2], iri=iri)


def main():
    a = sys.argv[1:]
    os.makedirs(OUT, exist_ok=True)
    if a and a[0] == "--ground":
        print(json.dumps(ground(a[1]), indent=1))
        return
    if a and a[0] == "--episode":
        text, inst, run = episode_tree(a[1])
        p = os.path.join(OUT, f"{inst}_{run}.lisp")
        open(p, "w").write(f";;; generated by tools/sdt_sexpr.py -- edit the twin, not this file\n{text}\n")
        print("wrote", p)
        return
    for d in a:
        text = item_tree(d)
        p = os.path.join(OUT, f"{instance_symbol(d)}.lisp")
        open(p, "w").write(f";;; generated by tools/sdt_sexpr.py from {os.path.relpath(d, ROOT)}/sdt.usda "
                           f"-- edit the twin, not this file\n{text}\n")
        n = len(index(parse(text)))
        print("wrote", p, n, "nodes")


if __name__ == "__main__":
    main()
