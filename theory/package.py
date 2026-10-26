"""Theory packages for manipulation competence (after 'Manipulation as Theory Construction', IAI Bremen): a theory
T_k = <Sigma_k, A_k, M_k, C_k, E_k> -- the distinctions the robot can represent (concepts, each perceivable through
the SDT and actionable through designators), the regularities it asserts, the body-world-system (BWS) families they
cover, the scoped competence claims Gamma = Competent(F_task, F_BWS, A_assess) it maintains, and the evidence for and
against them -- plus the runtime commitment gate and the revision operators of the learning loop.
"""

import copy
from dataclasses import dataclass, field

from theory.avocado_bws import verdict, wilson


@dataclass
class Concept:
    """A distinction the robot can represent; it counts only if perceivable, actionable (or abstention-relevant)
    and traceable (the operational criterion)."""
    name: str
    perceivable: str              # how the SDT represents its referent, value, uncertainty and provenance
    actionable: str               # which designator, parameter, monitor, repair or verification it changes
    traceable: str = "SDT patch / belief record"

    @property
    def covered(self):
        return bool(self.perceivable) and bool(self.actionable) and bool(self.traceable)


@dataclass
class Assertion:
    name: str
    kind: str                     # constitutive | causal | intentional
    statement: str
    params: dict = field(default_factory=dict)
    provenance: str = ""


@dataclass
class Scope:
    """A BWS family: which bodies, objects, materials, conditions and sensing the claims cover."""
    name: str
    description: str
    stage: tuple = (1.0, 5.0)
    sensing: str = "photo + touch"
    cultivar: str = "Hass"


@dataclass
class Claim:
    """Gamma = Competent(F_task, F_BWS, A_assess)."""
    name: str
    task: str                     # F_task: intended outcome and forbidden events
    bws: str                      # F_BWS: Scope name
    assess: str                   # A_assess: observables, requirement, q
    q: float = 0.8
    status: str = "maintained"    # maintained | restricted | retracted
    k: int = 0
    n: int = 0
    evidence: list = field(default_factory=list)

    @property
    def verdict(self):
        return verdict(self.k, self.n, self.q) if self.n else "unassessed"

    @property
    def interval(self):
        return wilson(self.k, self.n) if self.n else (0.0, 1.0)


@dataclass
class Evidence:
    id: str
    kind: str                     # simulation (research SDT) | physical | formal | counterexample
    about: str
    result: str
    k: int = 0
    n: int = 0


@dataclass
class TheoryPackage:
    version: int
    Sigma: list
    A: list
    M: list
    C: list
    E: list = field(default_factory=list)
    log: list = field(default_factory=list)

    def concept(self, name):
        return next((c for c in self.Sigma if c.name == name), None)

    def scope(self, name):
        return next((m for m in self.M if m.name == name), None)

    def claim(self, name):
        return next((c for c in self.C if c.name == name), None)

    def has(self, name):
        return self.concept(name) is not None

    def revise(self, gap, patch):
        """T_{k+1} = Revise(T_k, gap, hypotheses, interventions, evidence): apply the patch's deltas."""
        T = copy.deepcopy(self)
        T.version = self.version + 1
        T.Sigma += patch.dSigma
        T.A = [a for a in T.A if a.name not in patch.retract_A] + patch.dA
        for m in patch.dM:
            T.M = [x for x in T.M if x.name != m.name] + [m]
        for c in patch.dC:
            T.C = [x for x in T.C if x.name != c.name] + [c]
        for name, status in patch.restrict_C.items():
            if T.claim(name):
                T.claim(name).status = status
        T.E += patch.dE
        T.log.append(dict(version=T.version, gap=gap.name, layer=gap.layer, disposition=patch.disposition,
                          rationale=patch.rationale))
        return T


@dataclass
class Gap:
    name: str
    layer: str                    # task | grounding | state | material | causal | geometry | control | observation |
    mismatch: str                 # monitoring | verification | scope
    hypotheses: list = field(default_factory=list)


@dataclass
class Patch:
    dSigma: list = field(default_factory=list)
    dA: list = field(default_factory=list)
    retract_A: list = field(default_factory=list)
    dM: list = field(default_factory=list)
    dC: list = field(default_factory=list)
    restrict_C: dict = field(default_factory=dict)
    dE: list = field(default_factory=list)
    disposition: str = "admit"    # admit | restrict | version | reject
    rationale: str = ""


# %% runtime commitment gate
EPISTEMIC = ("warranted", "uncertain", "unsupported", "counterevidenced", "out-of-scope")


def commitment_gate(T, claim_name, in_scope_p, required_concepts, evidence_kinds, forbidden_p, p_scope_min=0.9):
    """WarrantedCommitment(a) <=> Grounded & Applicable & Feasible & Safe & Observable & RepairableOrAbstainable &
    Verifiable & EvidenceSufficient -- reduced to the checks this implementation can make. Returns (state, reasons).

    in_scope_p: the belief's probability that the instance lies in the claim's BWS family; required_concepts: the
    concepts the claim needs (each must be in Sigma and operationally covered); evidence_kinds: what the belief rests
    on now (e.g. {'photo', 'touch'}) against what the claim's scope requires; forbidden_p: predicted probabilities of
    the forbidden effects of the candidate plan (each must be 0 for a safety rule)."""
    reasons = []
    c = T.claim(claim_name)
    if c is None:
        return "unsupported", ["no claim covers this task"]
    if c.status == "retracted":
        return "counterevidenced", [f"claim {claim_name} retracted"]
    missing = [r for r in required_concepts if not (T.has(r) and T.concept(r).covered)]
    if missing:
        return "unsupported", [f"concepts not operationally covered: {', '.join(missing)}"]
    if any(p > 0 for p in forbidden_p.values()):
        return "unsupported", [f"forbidden effect predicted: {k} P = {p:.2f}" for k, p in forbidden_p.items() if p > 0]
    if in_scope_p < 0.5:
        return "out-of-scope", [f"P(instance in {c.bws}) = {in_scope_p:.2f}"]
    sc = T.scope(c.bws)
    if sc is not None and "touch" in sc.sensing and "touch" not in evidence_kinds:
        return "uncertain", ["the scope requires tactile ripeness evidence; only " + "+".join(sorted(evidence_kinds))]
    if in_scope_p < p_scope_min:
        return "uncertain", [f"P(instance in {c.bws}) = {in_scope_p:.2f} < {p_scope_min}"]
    if c.verdict == "refuted":
        return "counterevidenced", [f"claim {claim_name} refuted ({c.k}/{c.n})"]
    if c.verdict != "verified":
        return "unsupported", [f"claim {claim_name} {c.verdict} ({c.k}/{c.n})"]
    reasons.append(f"claim {claim_name} verified ({c.k}/{c.n}), instance in scope (P = {in_scope_p:.2f})")
    return "warranted", reasons
