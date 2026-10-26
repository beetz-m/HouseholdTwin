"""Baguette twin (baguette de tradition): the roll's baked-foam model (food/roll.py) on a long, pointed footprint.

Structure (docs/wave3/BAGUETTE.md for the sources): ~60 cm x 5.5 x 4 cm, ~250 g; five to seven overlapping diagonal
SCORES ('coupe parisienne', wp_grigne) open into GRIGNES with raised lips ('oreilles'); a thicker, crisp crust (1.7 mm
mean over 64 French breads, dv2012); an open, irregular crumb with large cells; pointed ends ('quignons').

Same partonomy as the roll: Crust/{UpperCrust, ScoreCrust, BottomCrust}, Crumb/{TransitionZone, CrumbCore,
GasCells}; the score crust collects all grignes (the score index of a crust point is in the label field).
"""

import numpy as np

from .roll import Roll


class Baguette(Roll):
    kind = "baguette"
    depth_h = 0.45e-3
    footprint_n = 5.0                                      # cylindrical along most of its length, pointed ends

    def __init__(self, seed=0, overrides=None):
        self._scores = None
        super().__init__(seed, overrides)

    def scores(self):
        """(x0, angle, half_length) per score: evenly spread along the loaf, overlapping ~1/3."""
        if self._scores is None:
            p = self.p
            n = int(p["n_scores"])
            ell = p["score_length_mm"] * 5e-4
            span = 2 * self.a * 0.80
            x0s = np.linspace(-span / 2 + ell, span / 2 - ell, n)
            rng = np.random.default_rng(self.seed + 41)
            ang = np.radians(p["score_axis_angle"]) * (1 + 0.1 * rng.normal(size=n))
            self._scores = [(x0 + rng.normal(0, 2e-3), a, ell) for x0, a in zip(x0s, ang)]
        return self._scores

    def _score_frame(self, x, y):
        """The score that dominates at (x, y): its along / across coordinates and length profile."""
        best = None
        for x0, a, ell in self.scores():
            c, s = np.cos(a), np.sin(a)
            u = (x - x0) * c + y * s
            w = -(x - x0) * s + y * c
            g = np.sqrt(np.clip(1 - (u / ell) ** 2, 0, 1))
            prof = g * np.exp(-(w / (self.score_w * 0.55 * np.maximum(g, 1e-3) + 1e-9)) ** 2)
            if best is None:
                best = [u, w, g, prof]
            else:
                take = prof > best[3]
                for i, v in enumerate((u, w, g, prof)):
                    best[i] = np.where(take, v, best[i])
        return best[0], best[1], best[2]

    # tube UV: u along the loaf, v round it
    def point_uv(self, U, V, it=28):
        U, V = np.asarray(U, float), np.asarray(V, float)
        x = (U - 0.5) * 2 * self.a * 0.985
        ph = 2 * np.pi * V
        d = np.stack([np.zeros_like(ph), np.cos(ph), np.sin(ph)], -1).reshape(-1, 3)
        c = np.stack([x.ravel(), np.zeros(x.size), np.full(x.size, 0.45 * self.H)], -1)
        lo, hi = np.zeros(len(d)), np.full(len(d), self.b * 1.5)
        for _ in range(it):
            mid = (lo + hi) / 2
            ok = self.inside(c + d * mid[:, None])
            lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
        return (c + d * lo[:, None]).reshape(U.shape + (3,))

    def uv_of(self, P):
        P = np.asarray(P, float)
        u = P[..., 0] / (2 * self.a * 0.985) + 0.5
        v = np.mod(np.arctan2(P[..., 2] - 0.45 * self.H, P[..., 1]), 2 * np.pi) / (2 * np.pi)
        return np.stack([np.clip(u, 0, 1), v], -1)

    def asset_spec(self):
        from .baguette_models import baguette_beliefs, baguette_truth
        spec = super().asset_spec()
        p = self.p
        spec["build"] = {"h": 0.4e-3, "sigma": 0.7, "smooth": 10}
        spec["textures"] = [("skin.png", self.point_uv, self.skin_albedo, 4096, 1024)]
        z_slice = p["slice_height_frac"] * self.H
        ext = float(self.a * 1.05)
        th = np.radians(45.0)
        n = [float(np.cos(th)), float(np.sin(th)), 0.0]
        t = p["bias_slice_mm"] * 1e-3
        offs = [float(o) for o in np.arange(-5, 6) * t]
        spec["cuts"] = [
            {"name": "across", "origin": [0, 0, 0], "e1": [0, 1, 0], "e2": [0, 0, 1], "ext": float(self.H * 1.1),
             "res": 1024, "states": {"half_a": ("pos", {"cut_state": "halved_across"}),
                                     "half_b": ("neg", {"cut_state": "halved_across"})}},
            {"name": "split", "origin": [0, 0, z_slice], "e1": [1, 0, 0], "e2": [0, 1, 0], "ext": ext,
             "states": {"top_half": ("pos", {"cut_state": "split_lengthwise"}),
                        "bottom_half": ("neg", {"cut_state": "split_lengthwise"})}},
        ]
        spec["slicings"] = [{"name": "crostini", "normal": n, "offsets": offs, "e1": [-n[1], n[0], 0.0],
                             "e2": [0, 0, 1], "ext": float(self.b * 1.6), "res": 1024, "section_at": 0.0,
                             "group": "Slices", "piece": "Slice", "piece_class": "food:BiasSlice",
                             "end_class": "food:BaguettePiece", "root": {"cut_state": "bias_sliced"}}]
        spec["beliefs"] = lambda mp: baguette_beliefs(self, mp)
        spec["truth"] = lambda: baguette_truth(self)
        spec["root"]["shape"] = "baguette"
        return spec
