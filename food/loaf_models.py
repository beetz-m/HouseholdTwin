"""Sandwich loaf: staling, toasting, taking slices, spreading (register params.LOAF).

  firmness      sehn20 lab white pan loaf: 1.98 / 3.62 / 4.18 / 5.64 N at day 1 / 5 / 9 / 13 (25 C, bagged),
                interpolated; open bag dries faster (ILLUSTRATIVE x1.5), fridge stales ~6x faster (toast_misc, weak)
  toasting      time in the toaster -> level; 'best' (golden) between 80 and 150 s depending on the toaster
                (toaster26); per-instance toaster speed drawn from that range; light / dark / burnt at 0.6 / 1.5 / 2
                times the golden time (ILLUSTRATIVE)
  taking slices one slice at a time from the bag; slices_left counter; a slice tears if pulled harder than its tensile
                limit scaled from nuss90 (0.31-0.39 N on a 12 mm neck -> ~2.8 N over a 100 mm slice edge,
                ILLUSTRATIVE scaling)
  spreading     cold butter tears the slice when the drag force exceeds the slice's tensile limit; butter yield
                stress by temperature ILLUSTRATIVE (GAP)
"""

import numpy as np

DAYS, FIRM = np.array([1.0, 5.0, 9.0, 13.0]), np.array([1.98, 3.62, 4.18, 5.64])


def firmness_N(age_d):
    return float(np.interp(age_d, DAYS, FIRM, left=FIRM[0], right=FIRM[-1] + 0.35 * (age_d - 13)))


def toast_level(seconds, t_golden=110.0):
    r = seconds / t_golden
    return "none" if r < 0.3 else ("light" if r < 0.75 else ("golden" if r < 1.4 else ("dark" if r < 2.0
                                                                                        else "burnt")))


class LoafTruth:
    def __init__(self, loaf, mass_kg):
        p = loaf.p
        self.l, self.m = loaf, mass_kg
        self.age_d, self.storage = float(p["age_d"]), p["storage"]
        self.slices_left = int(p["slices"])
        self.t_golden = float(np.random.default_rng(loaf.seed + 31).uniform(80, 150))

    def effective_age(self):
        f = {"bag_closed": 1.0, "bag_open": 1.5, "fridge": 6.0}[self.storage]
        return self.age_d * f

    def discrete(self):
        F = firmness_N(self.effective_age())
        return {"freshness": "fresh" if F < 3.0 else ("firm" if F < 5.0 else "stale"),
                "slices_left": self.slices_left, "mould_risk": "yes" if self.age_d > 14 else "no"}

    def take_slice(self, F_pull=1.0):
        limit = 0.35 * 100 / 12
        if self.slices_left == 0:
            return {"ok": False, "why": "bag empty"}
        if F_pull > limit:
            return {"ok": False, "why": "slice torn", "limit_N": limit}
        self.slices_left -= 1
        return {"ok": True, "slices_left": self.slices_left}

    def toast(self, seconds):
        return toast_level(seconds, self.t_golden)

    def spread(self, butter_C, area_cm2=80.0):
        """Drag of the knife through butter; yield stress kPa vs temperature (ILLUSTRATIVE: 4 C ~ 30 kPa .. 20 C ~
        3 kPa) on a ~0.3 cm2 blade contact -> drag force; tear if above the slice limit."""
        tau = float(np.interp(butter_C, [4, 10, 20], [30.0, 12.0, 3.0])) * 1e3
        drag = tau * 0.3e-4 * 3
        limit = 0.35 * 100 / 12
        return {"drag_N": drag, "torn": drag > limit, "butter_g": 0.068 * area_cm2}


def loaf_beliefs(l, mp):
    return {"": {"sdt:epistemicStatus": "prior + observation", "sdt:belief:mass_g": mp["mass_kg"] * 1e3,
                 "sdt:belief:mass_g_source": "weighed", "sdt:belief:slices_left": int(l.p["slices"]),
                 "sdt:belief:best_before": "printed on the bag (12-20 d, test25_toast)"}}


def loaf_truth(l):
    t = LoafTruth(l, 0.5)
    T = {"": {f"truth:{k}": v for k, v in l.p.items() if not k.startswith("_") and isinstance(v, (int, float, str))}}
    T[""].update({f"truth:{k}": v for k, v in t.discrete().items()})
    T[""]["truth:toaster_golden_s"] = t.t_golden
    return T
