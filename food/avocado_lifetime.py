"""Lifetime of an avocado: ripening under storage, the ripe window and the decline after it, spoilage, and the
browning of cut flesh under storage treatments. Used for lifetime-based decisions (when to prepare it, where to keep
it, how to keep a cut half).

Register entries (food/params.py AVOCADO, 'lifetime'): storage_temperatures (SOURCED), ripening_q10 (ILLUSTRATIVE),
ripe_window_counter (SOURCED-VERIFY), decline_per_day_counter, spoil_days_past_window, browning_rate_cut,
browning_visible_dE, browning_treatment_factors (ILLUSTRATIVE).

State of a whole fruit over time: ``stage`` (1..5, the register's ripeness scale) and ``hours_ripe`` (time spent at
stage 5). Past the ripe window the eating quality declines and the fruit eventually spoils. A cut half additionally
has ``dE``, the browning of its exposed flesh.
"""

import math
from dataclasses import dataclass, replace

import numpy as np

from . import params as PR

A = PR.AVOCADO
T_COUNTER, T_FRIDGE = 20.0, 4.0
STORAGE_T = {"counter": T_COUNTER, "fridge": T_FRIDGE}
Q10 = A["ripening_q10"][1]
BROWN_K = A["browning_rate_cut"][1]
BROWN_VISIBLE = A["browning_visible_dE"][1]
DECLINE = A["decline_per_day_counter"][1]
TREATMENT = {"none": 1.0, "lemon": 0.45, "wrap": 0.35, "wrap+fridge": 0.12, "lemon+wrap+fridge": 0.06,
             "vacuum+fridge": 0.03}
"""Browning-rate factors of the storage treatments of a cut face (register browning_treatment_factors; the fridge is
already part of the factor where named)."""


def mean_of(key):
    spec = A[key][1]
    if isinstance(spec, tuple) and spec[0] == "uniform":
        return 0.5 * (spec[1] + spec[2])
    if isinstance(spec, tuple) and spec[0] == "normal":
        return spec[1]
    return spec


def temperature_factor(T_C):
    """Rate factor of ripening, decline and browning relative to 20 C (Q10 rule)."""
    return Q10 ** ((T_C - T_COUNTER) / 10.0)


@dataclass
class FruitClock:
    """Lifetime state of one avocado (or of the flesh of one of its halves)."""
    stage: float
    rate_per_day: float
    hours_ripe: float = 0.0
    ripe_window_h: float = 36.0
    spoil_days: float = 2.0
    dE: float = 0.0              # browning of the exposed flesh (cut halves only)
    cut: bool = False

    @classmethod
    def of(cls, p):
        """From instance or particle parameters (missing lifetime keys take the register's means)."""
        return cls(float(p["ripeness_stage"]), float(p["ripening_rate"]),
                   ripe_window_h=float(p.get("ripe_window_counter", mean_of("ripe_window_counter"))),
                   spoil_days=float(p.get("spoil_days_past_window", mean_of("spoil_days_past_window"))))

    def advance(self, hours, storage="counter", treatment="none"):
        """The clock after ``hours`` in ``storage`` (counter / fridge); a cut face browns with ``treatment``."""
        f = temperature_factor(STORAGE_T[storage])
        days = hours / 24.0
        stage = self.stage + self.rate_per_day * f * days
        hr = self.hours_ripe
        if stage >= 5.0:                                       # time spent at full ripeness, at the storage rate
            t_to_ripe = max(0.0, (5.0 - self.stage) / max(self.rate_per_day * f, 1e-9)) * 24.0
            hr += max(0.0, hours - t_to_ripe) * f
            stage = 5.0
        dE = self.dE
        if self.cut:
            fb = TREATMENT[treatment] if "fridge" in treatment else TREATMENT[treatment] * f
            # dE = k sqrt(t): advance on the sqrt-time scale
            t_eq = (dE / BROWN_K) ** 2
            dE = BROWN_K * math.sqrt(t_eq + hours * fb)
        return replace(self, stage=stage, hours_ripe=hr, dE=dE)

    # %% judgements
    @property
    def days_past_window(self):
        return max(0.0, (self.hours_ripe - self.ripe_window_h) / 24.0)

    @property
    def spoiled(self):
        return self.days_past_window >= self.spoil_days

    @property
    def mashable(self):
        """How well the flesh mashes (0 below stage 2.5, 1 from stage 4; ILLUSTRATIVE)."""
        return float(np.clip((self.stage - 2.5) / 1.5, 0.0, 1.0))

    @property
    def eating_quality(self):
        """0..1: mashable and not past its best (decline per day past the ripe window), 0 when spoiled."""
        if self.spoiled:
            return 0.0
        return self.mashable * math.exp(-DECLINE * self.days_past_window)

    @property
    def browned(self):
        return self.cut and self.dE >= BROWN_VISIBLE

    @property
    def freshness(self):
        """0..1 look of the cut face: 1 fresh, 0 at twice the visible browning."""
        return float(np.clip(1.0 - self.dE / (2 * BROWN_VISIBLE), 0.0, 1.0)) if self.cut else 1.0


def best_day(clock, storage="counter", horizon_days=10, step_h=6.0):
    """Hour of the best eating quality within the horizon, and the quality curve (hours, quality, spoiled)."""
    hs = np.arange(0.0, horizon_days * 24 + 1e-9, step_h)
    q, sp = [], []
    for h in hs:
        c = clock.advance(h, storage)
        q.append(c.eating_quality)
        sp.append(c.spoiled)
    i = int(np.argmax(q))
    return float(hs[i]), hs, np.array(q), np.array(sp)
