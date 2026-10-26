"""Mechanics of halving and pitting an avocado (protocol: cut lengthwise round the seed, twist the halves apart, remove
the seed; afm_howto) on the twin's anatomy (food/avocado.py). All models are quasi-static and their parameters are
ILLUSTRATIVE unless the register says otherwise (food/params.py AVOCADO, 'processing' entries).

Cut round the seed: the knife enters at the stem end (puncture), the edge goes down to the seed coat and the fruit is
turned once round under the blade; the cut is a plane through the long axis. Along the perimeter (angle psi round
the seed centre, in the plane) the blade engages the flesh from the skin down to the seed: depth(psi). Cutting force
F(psi) = skin resistance x skin thickness + G_flesh(stage) x depth(psi) (fracture energy per cut area). The blade
stops on the seed: the stop depths are what a robot feels, and they measure the seed.

Twist apart: the halves are held by the uncut flesh bridges (if the cut did not go fully round) and by the seed's
adhesion on the half that releases it. Ripe fruit separates at the seed-flesh interface; unripe fruit tears in the
flesh, which stays on the seed (p_clean(stage), transition at clean_twist_stage).

Remove the seed (from the half that holds it): spoon lever, knife strike and twist, or squeezing it out from the skin
side. The seed coat (testa) stays in the cavity if it adheres to the flesh (register testa_adheres_to).
"""

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from . import params as PR
from .avocado import flesh_E_MPa, penetrometer_force

E_UNRIPE = float(flesh_E_MPa(1.0))                         # MPa


@dataclass
class CutPlane:
    azimuth_deg: float = 0.0          # plane spanned by (cos az, sin az, 0) and the long axis
    coverage_deg: float = 360.0       # how far round the fruit the cut goes (from the stem end)

    def basis(self):
        az = math.radians(self.azimuth_deg)
        e1 = np.array([math.cos(az), math.sin(az), 0.0])
        return e1, np.array([0.0, 0.0, 1.0]), np.cross(e1, [0.0, 0.0, 1.0])   # in-plane u, z; normal


@dataclass
class ProcessOutcome:
    action: str
    ok: bool
    outcome: str
    measured: dict = field(default_factory=dict)     # what a robot can sense (forces, depths, torques, images)
    truth: dict = field(default_factory=dict)        # hidden ground truth of the step
    events: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


# %% material laws over ripening (ILLUSTRATIVE scalings of the register's flesh modulus)
def flesh_toughness(stage):
    """Cutting fracture energy of the flesh, J/m2."""
    return PR.AVOCADO["flesh_cut_toughness_unripe"][1] * math.sqrt(float(flesh_E_MPa(stage)) / E_UNRIPE)


def seed_adhesion(stage):
    """Shear strength of the seed-flesh interface, Pa."""
    return PR.AVOCADO["seed_adhesion_unripe"][1] * 1e3 * (float(flesh_E_MPa(stage)) / E_UNRIPE) ** 0.7


def flesh_shear_strength(stage):
    """Shear strength of the flesh, Pa (0.08 x modulus)."""
    return 0.08 * float(flesh_E_MPa(stage)) * 1e6


def p_clean_twist(stage):
    s0 = PR.AVOCADO["clean_twist_stage"][1]
    return 1.0 / (1.0 + math.exp(-(stage - s0) / 0.25))


# %% geometry of the cut plane
def plane_profile(avo, plane, n=360):
    """Skin and seed-coat radii along rays from the seed centre (projected into the cut plane), per angle psi
    (psi = 90 deg points to the stem end). Returns psi, r_skin, r_seed (m) and the in-plane seed centre."""
    e1, ez, nrm = plane.basis()
    c = avo.seed_c - nrm * float(avo.seed_c @ nrm)
    psi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    D = np.cos(psi)[:, None] * e1 + np.sin(psi)[:, None] * ez
    g = avo.t_testa_o + avo.t_testa_i + avo.t_endo

    def march(inside, r_max=0.09, it=40):
        lo, hi = np.zeros(n), np.full(n, r_max)
        for _ in range(it):
            mid = (lo + hi) / 2
            ok = inside(c + D * mid[:, None])
            lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
        return lo
    r_skin = march(lambda P: avo.inside(P))
    r_seed = march(lambda P: avo.seed_e(P, g) < 1)
    return psi, r_skin, r_seed, c


def seed_features(avo):
    """Robust seed measures a blade feels: flesh depth beside the seed at its widest, above it (to the stem end) and
    below it (to the blossom end), m."""
    psi, rs, rd, _ = plane_profile(avo, CutPlane(0.0))
    k = lambda a: int(round(math.radians(a) / (2 * np.pi) * len(psi))) % len(psi)  # noqa: E731
    side = 0.5 * ((rs - rd)[k(0)] + (rs - rd)[k(180)])
    return dict(side=float(side), top=float((rs - rd)[k(90)]), bottom=float((rs - rd)[k(270)]))


# %% the three steps
def cut_round_seed(avo, plane, rng, stage, F_max=40.0, depth_noise_mm=0.5):
    """Cut lengthwise round the seed; returns the force and stop-depth profile and the cut extent."""
    p = avo.p
    psi, rs, rd, _ = plane_profile(avo, plane)
    depth = rs - rd
    start = math.radians(90.0)                                # from the stem end round the fruit
    rel = (psi - start) % (2 * np.pi)
    cut = rel <= math.radians(plane.coverage_deg)
    F_skin = p.get("skin_cut_resistance", 2.0) * p["skin_thickness"]
    F = F_skin + flesh_toughness(stage) * depth
    F_punct = p.get("skin_puncture_force", 10.0)
    events = []
    if F_punct > F_max:
        return ProcessOutcome("cut_round_seed", False, "no_entry", dict(F_puncture_N=F_max),
                              dict(F_puncture_needed_N=F_punct), ["blade does not pierce the skin"])
    over = cut & (F > F_max)
    if over.any():                                            # the blade stalls where the force exceeds the limit
        cut &= ~over
        events.append(f"blade stalled on {np.degrees(np.ptp(psi[over])):.0f} deg of the perimeter (force limit)")
    order = np.argsort(rel)
    stop_mm = (depth * 1e3 + rng.normal(0, depth_noise_mm, len(depth)))
    measured = dict(F_puncture_N=float(F_punct), F_cut_N=F[order][cut[order]].round(2).tolist(),
                    psi_deg=np.degrees(psi[order][cut[order]]).round(1).tolist(),
                    stop_depth_mm=stop_mm[order][cut[order]].round(2).tolist(),
                    seed_contact=bool(cut.any()), work_J=float(np.sum(F[cut] * rs[cut]) * 2 * np.pi / len(psi)))
    covered = float(np.degrees(cut.sum() * 2 * np.pi / len(psi)))
    truth = dict(coverage_deg=covered, uncut_deg=360.0 - covered, depth_mm=(depth * 1e3).round(2).tolist(),
                 psi_deg=np.degrees(psi).round(1).tolist(), flesh_toughness_J_m2=flesh_toughness(stage),
                 seed=seed_features(avo))
    ok = covered >= 355.0
    return ProcessOutcome("cut_round_seed", ok, "scored" if ok else "partly_scored", measured, truth, events)


def bridge_torque_capacity(avo, plane, stage, covered_deg):
    """Torque the uncut flesh bridges carry against twisting the halves (polar moment of the uncut annulus)."""
    psi, rs, rd, _ = plane_profile(avo, plane)
    rel = (psi - math.radians(90.0)) % (2 * np.pi)
    uncut = rel > math.radians(covered_deg)
    dpsi = 2 * np.pi / len(psi)
    J = np.sum((rs[uncut] ** 3 - rd[uncut] ** 3) / 3.0) * dpsi
    return flesh_shear_strength(stage) * J


def seed_interface(avo):
    """Half the seed's surface (the contact a half releases) and its mean lever arm about the twist axis."""
    a, b = avo.seed_L / 2, avo.seed_W / 2
    pth = 1.6075
    area = 4 * math.pi * (((a * b) ** pth + (a * b) ** pth + (b * b) ** pth) / 3) ** (1 / pth)
    return area / 2, 0.6 * b


def twist_apart(avo, plane, rng, stage, covered_deg, T_max=1.5):
    """Twist the halves about the cut plane's normal until they separate (or the torque limit is reached)."""
    A, r = seed_interface(avo)
    T_bridge = bridge_torque_capacity(avo, plane, stage, covered_deg)
    pc = p_clean_twist(stage)
    clean = rng.random() < pc
    tau = seed_adhesion(stage) if clean else min(flesh_shear_strength(stage), 1.5 * seed_adhesion(stage))
    T_need = T_bridge + tau * A * r
    measured = dict(T_max_applied_Nm=T_max)
    truth = dict(T_needed_Nm=T_need, T_bridge_Nm=T_bridge, p_clean=pc, interface="seed-flesh" if clean else "flesh",
                 seed_adhesion_kPa=seed_adhesion(stage) / 1e3)
    if T_need > T_max:
        measured["T_peak_Nm"] = T_max
        return ProcessOutcome("twist_apart", False, "stuck", measured, truth,
                              ["halves do not separate within the torque limit" +
                               (" (uncut bridge)" if T_bridge > 0.3 * T_need else "")])
    measured["T_peak_Nm"] = float(T_need * (1 + rng.normal(0, 0.05)))
    chunks = []
    if T_bridge > 0:                                          # the uncut bridge tears through the flesh
        truth["torn_bridge_deg"] = 360.0 - covered_deg
    if not clean:                                             # flesh torn out of half_b, stuck to the seed
        for _ in range(int(rng.integers(1, 4))):
            chunks.append(dict(psi_deg=float(rng.uniform(0, 360)), lat_deg=float(rng.uniform(-50, 50)),
                               radii_mm=[float(rng.uniform(4, 10)), float(rng.uniform(4, 10)), float(rng.uniform(2, 5))]))
    truth["torn_chunks"] = chunks
    measured["flesh_on_seed_visible"] = bool(chunks)            # cavity inspection after the twist
    measured["full_cut"] = covered_deg >= 355.0
    torn = (not clean) or T_bridge > 0
    ev = ([] if clean else ["flesh tears and stays on the seed"]) + \
        ([f"uncut bridge ({360.0 - covered_deg:.0f} deg) torn through the flesh"] if T_bridge > 0 else [])
    return ProcessOutcome("twist_apart", True, "halved_torn" if torn else "halved", measured, truth, ev)


def remove_seed(avo, rng, stage, method="spoon", F_max=25.0, T_max=1.0):
    """Remove the seed from the half that holds it."""
    p = avo.p
    A, r = seed_interface(avo)
    tau = seed_adhesion(stage)
    pc = p_clean_twist(stage)
    testa_flesh = p.get("testa_adheres_to") == "flesh"
    remnant = float(rng.uniform(*PR.AVOCADO["testa_remnant_fraction"][1][1:])) if testa_flesh else 0.0
    truth = dict(method=method, seed_adhesion_kPa=tau / 1e3, testa_adheres_to=p.get("testa_adheres_to"),
                 testa_remnant_fraction=remnant)
    measured, events = {}, []
    gouges = []
    if method == "spoon":
        F_need = 0.3 * tau * A                                # peeling the seed out is easier than shearing it off
        measured["F_peak_N"] = float(min(F_need, F_max) * (1 + rng.normal(0, 0.08)))
        if F_need > F_max:
            return ProcessOutcome("remove_seed", False, "stuck", measured, dict(truth, F_needed_N=F_need),
                                  ["seed does not lift within the force limit"])
        if rng.random() < 0.15 + 0.6 * (1 - pc):              # the spoon takes flesh along (firm, clinging flesh)
            gouges.append(dict(psi_deg=float(rng.uniform(0, 360)), lat_deg=float(rng.uniform(-30, 30)),
                               radii_mm=[float(rng.uniform(5, 10)), float(rng.uniform(4, 8)), float(rng.uniform(2, 4))]))
            events.append("spoon took flesh from the cavity rim")
        truth["F_needed_N"] = F_need
    elif method == "knife_twist":
        F_strike = 40.0                                       # embed the heel of the blade in the seed (ILLUSTRATIVE)
        T_need = tau * A * r
        measured.update(F_strike_N=F_strike, T_peak_Nm=float(min(T_need, T_max) * (1 + rng.normal(0, 0.08))))
        events.append("knife strike into the seed: hazardous for a hand-held half (farley2019); robot fixture only")
        if T_need > T_max:
            return ProcessOutcome("remove_seed", False, "stuck", measured, dict(truth, T_needed_Nm=T_need), events)
        truth["T_needed_Nm"] = T_need
    elif method == "squeeze_out":
        F_sq = 12.0                                           # press the skin behind the seed (ILLUSTRATIVE)
        ok = rng.random() < 1.0 / (1.0 + math.exp(-(stage - 4.0) / 0.2))
        lim = p["bruise_force_ratio"] * penetrometer_force(stage)
        measured["F_squeeze_N"] = F_sq
        truth["bruised"] = F_sq > lim
        if truth["bruised"]:
            events.append("squeezing bruises the flesh behind the seed")
        if not ok:
            return ProcessOutcome("remove_seed", False, "stuck", measured, truth, events + ["seed does not pop out"])
    else:
        raise ValueError(method)
    truth["gouges"] = gouges
    measured["testa_visible_in_cavity"] = remnant > 0.05
    measured["seed_length_mm"] = float(p["seed_length"] + rng.normal(0, 1.0))   # the extracted seed, measured
    out = "pitted" if remnant <= 0.05 and not gouges else ("pitted_testa_left" if remnant > 0.05 else "pitted_gouged")
    return ProcessOutcome("remove_seed", True, out, measured, truth, events)


def peel_seed_coat(avo, rng, remnant_fraction, passes=1):
    """Lift the seed-coat sheets left in the cavity with the spoon tip (corrective step after pitting). Each pass
    lifts them all with the register's peel_coat_success (ILLUSTRATIVE), else leaves 40 % of them; a pass takes a
    little flesh with it at 5 % (ILLUSTRATIVE)."""
    p = PR.AVOCADO["peel_coat_success"][1]
    rem = float(remnant_fraction)
    gouges, events = [], []
    for _ in range(passes):
        if rem <= 0.05:
            break
        rem = 0.0 if rng.random() < p else rem * 0.4
        if rng.random() < 0.05:
            gouges.append(dict(psi_deg=float(rng.uniform(0, 360)), lat_deg=float(rng.uniform(-40, 40)),
                               radii_mm=[3.0, 3.0, 1.5]))
            events.append("the spoon tip took a little flesh")
    measured = dict(testa_visible_in_cavity=rem > 0.05)
    truth = dict(testa_remnant_fraction=rem, gouges=gouges)
    ok = rem <= 0.05
    return ProcessOutcome("peel_seed_coat", ok, "cavity_clean" if ok else "testa_left", measured, truth,
                          events + ([] if ok else ["seed-coat sheets still in the cavity"]))
