"""Put things where they belong: the adapter between ccram's kernel and a kitchen-state emulator (an emulator twin;
the PR2 kitchen version can subclass it as the egg kitchen does).

The emulator holds the truth: objects (kind, state, place), places (capacity), the people present. The robot sees
what its survey and its own actions show. Belonging follows coraplex.household_norms (ResearchSpace fork), re-expressed:
  rules        (belongs-at o place relation confidence), (prohibited-at o place)       BE1 (rules/belonging.lisp)
  adapter      applicable place = most confident applicable norm whose place is not prohibited     BelongingStore
               verdict = licensed | indeterminate (no norm) | at-belonging-place | misplaced      assess_placement
               requests: resident -> licensed + obligation (rules); guest conflict -> ask            requests.resolve
  each_items   the guards: misplaced -> put away; in place -> leave; licensed -> leave (debt recorded);
               no known home -> ask (no guessing); request conflict -> ask
  r_putting_away  PL1 admissibility of the belonging place now (capacity): not admissible -> signal
               place-not-admissible (the plan asks; never "somewhere merely admissible", T3)
"""
from __future__ import annotations

from ccram import goals as G
from ccram.adapter import Adapter
from ccram.kernel import KernelFailure

PLACES = {"fridge": dict(kind="refrigerator", capacity=6), "dishwasher": dict(kind="dishwasher", capacity=12),
          "cupboard": dict(kind="cupboard", capacity=8), "bin": dict(kind="trash-container", capacity=4),
          "counter": dict(kind="countertop", capacity=99)}
AFTER_BREAKFAST = {   # object: kind, observable states, where it is (the scenario of the concept page, competence A)
    "cup-1": dict(kind="cup", states={"dirty"}, at="counter"),
    "milk-1": dict(kind="milk-pack", states={"opened"}, at="counter"),
    "plate-1": dict(kind="plate", states={"dirty"}, at="counter"),
    "knife-1": dict(kind="knife", states={"dirty"}, at="counter"),
    "jar-1": dict(kind="jar", states=set(), at="counter"),               # unknown jar: no belonging norm
    "coffee-1": dict(kind="coffee-jar", states=set(), at="counter"),
    "sugar-1": dict(kind="sugar-jar", states=set(), at="counter"),       # belongs on the counter: in place
}
SECONDS = {"surveying": 4.0, "putting-away": 12.0, "asking": 6.0}       # ILLUSTRATIVE


class PutAwayAdapter(Adapter):
    OBSERVABLE = {"at", "surveyed", "seen-in", "is-a", "dirty", "opened", "empty", "context", "asked", "flagged",
                  "request", "request-ended", "obligation-discharged", "placement-verdict", "applicable-place",
                  "misplaced", "in-place", "licensed", "no-known-home"}
    MODEL_ONLY = {"dropped", "held", "above", "released"}
    instance_servant = "survey"

    def __init__(self, objects=None, places=None, survey_noise=None, seed=0):
        super().__init__(None, None)
        import random
        self.noise = dict(survey_noise or {})   # {"miss_dirty": p, "misclassify": p}: a fallible survey (ILLUSTRATIVE)
        self.rng = random.Random(seed)
        self.truth = {o: dict(kind=d["kind"], states=set(d["states"]), at=d["at"])
                      for o, d in (objects or AFTER_BREAKFAST).items()}
        self.places = {p: dict(d) for p, d in (places or PLACES).items()}
        self.known = {}                     # what the robot has observed: object -> dict(kind, states, at)
        self.surveyed = set()
        self.requests = []                  # (r, tier, object, place, scope)
        self.ended = set()
        self.asked, self.flagged, self.discharged = [], set(), set()
        self.people = ["resident"]
        self.answers = {}                   # scripted answers to questions: object -> place (a told norm)
        self.told = {}                      # norms told by a person: object -> place
        self.t = 0.0
        self.tracked = list(self.truth)
        self.context_name = "after-breakfast"

    # -- entities
    def instances(self, kw):
        if kw == "household-object":
            return [o for o in self.known]
        if kw == "countertop":
            return ["counter"]
        if kw == "person":
            return list(self.people)
        if kw == "place":
            return list(self.places)
        return None

    def resolve_symbol(self, s):
        return s if s in self.truth or s in self.places or s in self.people else None

    def sym(self, v):
        return v if isinstance(v, str) else super().sym(v)

    def time(self):
        return self.t

    def load(self, o):
        return sum(1 for d in self.truth.values() if d["at"] == o)

    # -- what the robot has observed + the belonging verdicts
    def observed(self):
        a = [["context", self.context_name]]
        for s in self.surveyed:
            a.append(["surveyed", s])
        for o, k in self.known.items():
            a += [["is-a", o, k["kind"]], ["at", o, k["at"]]] + [[st, o] for st in sorted(k["states"])]
            if k["at"] in self.surveyed:
                a.append(["seen-in", o, k["at"]])
        a += [["request", *r] for r in self.requests] + [["request-ended", r] for r in sorted(self.ended)]
        a += [["asked", p, o] for p, o in self.asked] + [["flagged", o] for o in sorted(self.flagged)]
        a += [["obligation-discharged", o, r] for o, r in sorted(self.discharged)]
        a += [["belongs-at", o, p, "on", 1.0] for o, p in self.told.items()]     # a norm a person told the robot
        return a

    def applicable_place(self, o, closed):
        """BelongingStore.applicable_place: the most confident applicable norm whose place is not prohibited."""
        prohibited = {x[2] for x in closed if x[0] == "prohibited-at" and x[1] == o}
        cands = [(x[4], x[2], x[3]) for x in closed if x[0] == "belongs-at" and x[1] == o and x[2] not in prohibited]
        if not cands:
            return None
        c, p, rel = max(cands)
        return dict(place=p, relation=rel, confidence=c, alternatives=len(cands))

    def verdict(self, o, closed):
        """assess_placement: licensed > indeterminate (no norm) > at-belonging-place > misplaced."""
        if ["licensed", o] in closed:
            return "licensed"
        ap = self.applicable_place(o, closed)
        if ap is None:
            return "indeterminate"
        return "at-belonging-place" if self.known.get(o, {}).get("at") == ap["place"] else "misplaced"

    def atoms(self):
        obs = self.observed()
        if self.interp is None:
            return obs
        closed = self.interp.lib.rules.closure(obs)
        out = list(obs)
        for o in self.known:
            v = self.verdict(o, closed)
            out.append(["placement-verdict", o, v])
            out.append([{"licensed": "licensed", "indeterminate": "no-known-home", "at-belonging-place": "in-place",
                         "misplaced": "misplaced"}[v], o])
            ap = self.applicable_place(o, closed)
            if ap:
                out.append(["applicable-place", o, ap["place"], ap["relation"]])
        return out

    # -- servants
    def resolve(self, x, env, role, climb):
        if isinstance(x, list) and x[:2] == ["its", "destination"]:            # D7: the applicable belonging place
            o = env.get("_object", (None, None))[1]
            closed = self.interp.atoms()
            ap = self.applicable_place(o, closed)
            if ap is None:
                raise KernelFailure("no-belonging-place", f"{o}: no belonging norm applies (MP3)")
            norm = next((r.name for r in self.interp.lib.rules.rules
                         if any(G.holds(["belongs-at", o, ap["place"], ap["relation"], ap["confidence"]], [c])
                                for c in r.fire(closed))), "told by a person" if o in self.told else "?")
            admissible = [p for p in self.places if self.load(p) < self.places[p]["capacity"]]
            return ap["place"], ap["place"], dict(
                servant="belonging-store", status="recorded" if o in self.told else "commonsense",
                norm=norm, confidence=ap["confidence"], relation=ap["relation"],
                note=f"{len(admissible)} places PL1-admissible now ({', '.join(admissible)}); adequate: {ap['place']} only (T3)")
        return None

    def gate_failure(self, phi):
        return "prohibited-placement"

    # -- the guards (TD, RQ*): per object, what to do
    def each_items(self, goal, atoms):
        for o in list(self.known):
            v = self.verdict(o, atoms)
            ask = ["an", "action", ["type", "asking"], ["recipient", ["a", "person"]],
                   ["query", ["the", "household-object", ["tag", o]]]]
            conflict = next((x for x in atoms if x[0] == "request-conflict" and x[1] == o), None)
            if conflict is not None:
                yield dict(note=f"{o}: request {conflict[2]} conflicts with the standing norm and a guest may not "
                                f"overrule it (RQ3) -- abstain and ask", designator=ask if not self._asked(o) else None)
            elif v == "licensed":
                ob = next((x for x in atoms if x[0] == "obligation-open" and x[1] == o), None)
                yield dict(note=f"{o}: licensed by request {ob[3] if ob else '?'} (resident) -- left out; restoration "
                                f"obligation to {ob[2] if ob else '?'} open (RQ4)")
            elif v == "at-belonging-place":
                yield dict(note=f"{o}: in place ({self.known[o]['at']}) -- leave it (no over-tidying, TD)")
            elif v == "indeterminate":
                if o in self.flagged:
                    yield dict(note=f"{o}: no known home, already asked -- left")
                else:
                    yield dict(note=f"{o}: no belonging norm applies (MP3) -- ask, don't guess", designator=ask)
            else:
                yield dict(goal=[goal[0], o, self.applicable_place(o, atoms)["place"]],
                           designator=["an", "action", ["type", "putting-away"],
                                       ["object", ["the", "household-object", ["tag", o]]],
                                       ["destination", ["its", "destination"]]])

    def _asked(self, o):
        return any(x == o for _, x in self.asked)

    # -- acting (the emulator)
    def execute(self, nodes, attach=None):
        for step in nodes:
            step()

    def r_surveying(self, R):
        area = R["area"][1]

        def look():
            self.t += SECONDS["surveying"]
            self.surveyed.add(area)
            for o, d in self.truth.items():
                if d["at"] == area:
                    kind, states = d["kind"], set(d["states"])
                    if "dirty" in states and self.rng.random() < self.noise.get("miss_dirty", 0.0):
                        states.discard("dirty")                    # the dirt was not seen
                    if self.rng.random() < self.noise.get("misclassify", 0.0):
                        kind = {"coffee-jar": "jar", "sugar-jar": "jar", "jar": "coffee-jar", "cup": "plate",
                                "plate": "cup"}.get(kind, kind)       # a look-alike
                    self.known[o] = dict(kind=kind, states=states, at=area)
        return [look]

    def r_putting_away(self, R):
        o, p = R["object"][1], R["destination"][1]

        def put():
            self.t += SECONDS["putting-away"]
            if self.load(p) >= self.places[p]["capacity"]:                     # PL1 hard constraint fails now
                self.interp.signal("place-not-admissible", f"{p} is full: {o} not placed (T3: not 'somewhere else')")
                self.asked.append(("resident", o))                             # resumed by the plan: asked instead
                self.flagged.add(o)
                return
            due = [r for (r, tier, oo, place, scope) in self.requests if oo == o and r in self.ended]
            self.truth[o]["at"] = p
            self.known[o]["at"] = p
            for r in due:
                self.discharged.add((o, r))
        return [put]

    def r_asking(self, R):
        who, o = R["recipient"][1], R["query"][1]

        def ask():
            self.t += SECONDS["asking"]
            self.asked.append((who, o))
            self.flagged.add(o)
            if o in self.answers:                                              # a told norm (BE2: evidential)
                self.told[o] = self.answers[o]
        return [ask]


def make_adapter(demo, world, context):
    return PutAwayAdapter()
