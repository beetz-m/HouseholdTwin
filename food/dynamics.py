"""Hidden-world dynamics of the food twins: what physically happens to an instance under manipulation.

The simulator (Isaac/PhysX) delivers contact facts -- quasi-static loads (grip force, contact location, pad size,
tool edge radius) and impacts (relative normal speed, surface, location).  These classes turn them into the
instance's discrete state transitions and continuous damage, using the instance's hidden parameters (food/params.py).
The robot never reads these objects; its SDT (food/sdt.py) only sees observations.

Models (statuses in the parameter register):
  egg      crack when the effective contact force exceeds the direction-dependent break force; impacts via the
           spring model F = v sqrt(k_eff m) with k_eff = shell (Reissner) in series with the surface; a near-miss
           band may leave a hidden micro-crack; above 2 x break force the shell shatters (Morgenstern's f+).
  apple    static bruise when the Hertz maximum shear 0.31 p0 exceeds the flesh shear strength; impact bruise volume
           linear in absorbed energy above a threshold (fsct2008 anchor); cut when blade force exceeds cutting force.
  avocado  ripeness advances with time; flesh bruises when contact pressure exceeds a fraction of the penetrometer
           pressure of its stage; halving around the seed separates cleanly only from stage 3.
"""

from dataclasses import dataclass, field

import numpy as np

G = 9.81
# surface contact stiffness [N/m] and restitution for drops (ILLUSTRATIVE: hard stone, wooden board, folded towel)
SURFACES = {"granite": (5e7, 0.25), "quartz_worktop": (5e7, 0.25), "steel": (8e7, 0.3), "oak_board": (2e6, 0.2),
            "silicone_mat": (2e5, 0.1), "towel": (1.5e4, 0.05), "bowl_rim_ceramic": (3e7, 0.25)}


def series(*k):
    return 1.0 / sum(1.0 / x for x in k)


@dataclass
class Event:
    t: float
    kind: str
    detail: dict = field(default_factory=dict)


# ================================================================================================ egg
class EggTruth:
    SHELL = ("intact", "microcracked", "cracked", "shattered", "opened")

    def __init__(self, egg, mass_kg):
        self.egg, self.m = egg, mass_kg
        self.shell = "intact"
        self.contents = "in_shell"
        self.yolk = "whole"
        self.strength = 1.0
        self.crack_at = None
        self.events = []

    def crack_force(self, polar, edge_radius=None):
        F = self.egg.break_force(polar) * self.strength
        if edge_radius is not None:                      # sharp tool parts concentrate the load (ILLUSTRATIVE)
            F *= np.sqrt(max(edge_radius, 0.2e-3) / 5e-3)
        return F

    def _apply(self, t, F, polar, edge_radius, cause, rng):
        if self.shell in ("cracked", "shattered", "opened"):
            return self.shell
        Fc = self.crack_force(polar, edge_radius)
        p = self.egg.p
        if F >= 2.0 * Fc:
            self.shell, self.contents = "shattered", "spilled"
            self.yolk = "broken" if rng.random() < 0.7 else "whole"
        elif F >= Fc:
            self.shell = "cracked"
        elif F >= p["microcrack_band"] * Fc and self.shell == "intact":
            if rng.random() < (F / Fc - p["microcrack_band"]) / (1 - p["microcrack_band"]):
                self.shell = "microcracked"
                self.strength = 1 - p["microcrack_strength_loss"]
        if self.shell != "intact":
            self.crack_at = self.crack_at or polar
        self.events.append(Event(t, cause, {"F": F, "F_crack": Fc, "polar_deg": np.degrees(polar),
                                            "edge_radius": edge_radius, "shell": self.shell}))
        return self.shell

    def static_load(self, t, F, polar, rng, edge_radius=None):
        """Quasi-static squeeze (grip, press). F = normal force at one contact of a two-sided grip."""
        return self._apply(t, F, polar, edge_radius, "static_load", rng)

    def impact(self, t, v_n, polar, surface, rng, edge_radius=None):
        """Impact with relative normal speed v_n: F = v sqrt(k_eff m_eff), compared with the rate-strengthened crack
        force (register: impact_effective_mass_fraction, impact_rate_factor -- both ILLUSTRATIVE, to calibrate)."""
        p = self.egg.p
        rate_factor = p["impact_rate_factor"]
        k_s = self.egg.contact_stiffness(polar)
        k_surf, _ = SURFACES[surface]
        F = v_n * np.sqrt(series(k_s, k_surf) * self.m * p["impact_effective_mass_fraction"])
        self._apply(t, F / rate_factor, polar, edge_radius, "impact:" + surface, rng)
        self.events[-1].detail["F_peak"] = F
        return F

    def open_up(self, t, crack_up, rng, drop_height=0.05):
        """Pull a cracked shell apart over a vessel. Contents delivered only from an up-facing crack (CC4)."""
        if self.shell != "cracked":
            return False
        if not crack_up:
            self.contents = "spilled"
            return False
        self.shell, self.contents = "opened", "released"
        if rng.random() < 0.02 + 0.4 * max(0.0, drop_height - 0.08):
            self.yolk = "broken"
        self.events.append(Event(t, "open_up", {"yolk": self.yolk}))
        return True

    def drop_height_to_crack(self, polar, surface):
        """Smallest drop height that cracks this egg on the surface (for reports): v = F / sqrt(k m)."""
        p = self.egg.p
        F = self.crack_force(polar) * p["impact_rate_factor"]
        v = F / np.sqrt(series(self.egg.contact_stiffness(polar), SURFACES[surface][0]) * self.m
                        * p["impact_effective_mass_fraction"])
        return v ** 2 / (2 * G)


# ================================================================================================ apple
class AppleTruth:
    def __init__(self, apple, mass_kg):
        self.a, self.m = apple, mass_kg
        self.integrity = "whole"
        self.bruises = []                       # (t, location (3,), volume cm3, cause)
        self.cut_time = None
        self.events = []
        p = apple.p
        self.E = p["flesh_E"] * 1e6 * np.exp(-p["storage_days"] / 180.0)
        self.Estar_rigid = self.E / (1 - p["flesh_poisson"] ** 2)
        self.R = apple.R
        self.V_cm3 = None

    def bruise_volume(self):
        return float(sum(b[2] for b in self.bruises))

    def static_load(self, t, F, location, rng, pad_radius=None):
        """Hertz contact on a flat (or large-radius) pad: p0 = (6 F E*^2 / (pi^3 R^2))^(1/3); bruise if 0.31 p0 > tau_f."""
        R = self.R if pad_radius is None else 1 / (1 / self.R + 1 / pad_radius)
        p0 = (6 * F * self.Estar_rigid ** 2 / (np.pi ** 3 * R ** 2)) ** (1 / 3)
        tau = 0.31 * p0
        tf = self.a.p["flesh_shear_strength"] * 1e6
        vol = 0.0
        if tau > tf:
            a = (3 * F * R / (4 * self.Estar_rigid)) ** (1 / 3)       # contact radius
            vol = (2 / 3) * np.pi * a ** 3 * 1e6 * (tau / tf - 1)     # ILLUSTRATIVE: damaged hemisphere fraction
            self.bruises.append((t, np.asarray(location), vol, "static"))
        self.events.append(Event(t, "static_load", {"F": F, "p0_MPa": p0 / 1e6, "tau_MPa": tau / 1e6,
                                                    "bruise_cm3": vol}))
        return vol

    def impact(self, t, v_n, location, surface, rng):
        k_surf, e = SURFACES[surface]
        KE = 0.5 * self.m * v_n ** 2
        absorbed = (1 - e ** 2) * KE
        # Hertz impact peak force on a rigid flat (for reports and the gripper's observation)
        d_max = (15 * self.m * v_n ** 2 / (16 * self.Estar_rigid * np.sqrt(self.R))) ** 0.4
        F_peak = 4 / 3 * self.Estar_rigid * np.sqrt(self.R) * d_max ** 1.5
        soft = min(1.0, k_surf / 5e6)                                    # soft surfaces absorb the energy instead
        absorbed *= soft
        V = self.V_cm3 or 200.0
        k_b = 0.01 * V / 0.41                                            # fsct2008 anchor: 1 % of V at 0.41 J
        vol = max(0.0, absorbed - self.a.p["bruise_energy_threshold"]) * k_b
        if vol > 0:
            self.bruises.append((t, np.asarray(location), vol, "impact:" + surface))
        self.events.append(Event(t, "impact:" + surface, {"v": v_n, "KE_J": KE, "absorbed_J": absorbed,
                                                          "F_peak": F_peak, "bruise_cm3": vol}))
        return vol

    def blade(self, t, F_blade):
        if self.integrity == "whole" and F_blade >= self.a.p["cutting_force_peak"]:
            self.integrity, self.cut_time = "halved", t
            self.events.append(Event(t, "cut", {"F": F_blade}))
            return True
        return False

    def browning(self, t, tau_s=40 * 60.0):
        """Enzymatic browning index of a cut face, 0..1 (ILLUSTRATIVE time constant)."""
        return 0.0 if self.cut_time is None else float(1 - np.exp(-(t - self.cut_time) / tau_s))


# ================================================================================================ avocado
class AvocadoTruth:
    PEN_AREA = np.pi * (11.1e-3 / 2) ** 2                 # penetrometer tip of the Avocados Australia chart

    def __init__(self, avo, mass_kg):
        self.a, self.m = avo, mass_kg
        self.stage = avo.p["ripeness_stage"]
        self.integrity = "whole"
        self.seed = "in"
        self.bruises = []
        self.events = []
        self.t = 0.0

    def advance(self, t):
        """Ripening at the instance's rate (stage per day at 20 C), capped at 5 (then overripe accumulates)."""
        dt_d = (t - self.t) / 86400.0
        self.stage = min(5.5, self.stage + self.a.p["ripening_rate"] * dt_d)
        self.t = t

    def penetrometer_pressure(self):
        from .avocado import penetrometer_force
        return penetrometer_force(self.stage) / self.PEN_AREA

    def flesh_E(self):
        from .avocado import flesh_E_MPa
        return flesh_E_MPa(self.stage) * 1e6

    def squeeze(self, t, F, contact_area, location):
        """Quasi-static squeeze by a pad pair; bruise if the mean pressure exceeds a fraction of penetration pressure."""
        self.advance(t)
        p = F / max(contact_area, 1e-6)
        lim = self.a.p["bruise_force_ratio"] * self.penetrometer_pressure()
        vol = 0.0
        if p > lim:
            vol = contact_area ** 1.5 * 1e6 * (p / lim - 1) * 0.5          # ILLUSTRATIVE
            self.bruises.append((t, np.asarray(location), vol, "squeeze"))
        self.events.append(Event(t, "squeeze", {"F": F, "p_kPa": p / 1e3, "limit_kPa": lim / 1e3, "bruise_cm3": vol}))
        return vol

    def halve(self, t, F_blade, rng):
        """Cut lengthwise around the seed and twist: separates cleanly from stage 3 (unripe flesh clings)."""
        self.advance(t)
        F_needed = 6.0 + 2.5 * (5 - min(self.stage, 5))                    # ILLUSTRATIVE skin + flesh resistance
        if F_blade < F_needed:
            self.events.append(Event(t, "halve_failed", {"F": F_blade, "needed": F_needed}))
            return "uncut"
        clean = self.stage >= 3.0 or rng.random() < 0.2
        self.integrity = "halved" if clean else "halved_torn"
        self.events.append(Event(t, "halve", {"clean": clean, "stage": self.stage}))
        return self.integrity
