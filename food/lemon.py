"""Lemon twin (Citrus x limon, Primofiori / Verna): a hesperidium like the orange (food/orange.py) with a prolate
outline, the stylar NIPPLE (mammilla) at the blossom end and a short NECK at the stem end; thicker, paler albedo
(0.39 g/cm3, ben26), ~10 segments, a few seeds, sour juice (20 % EU minimum, 30-45 % measured).
Sources: docs/wave3/LEMON.md; register params.LEMON.

Partonomy: as the orange (Peel/{Flavedo, Albedo, OilGlands}, StemButton, Core, Segments/Segment_k/{Membrane,
Vesicles, Seed_j}); no navel.
"""

import numpy as np

from . import meshlib as ML
from .orange import Orange
from .shapes import smoothstep


class Lemon(Orange):
    kind = "lemon"

    def surf_curve(self, depth):
        """Prolate body with a nipple at the bottom (stylar end) and a neck at the top, moved inwards by depth."""
        a, c = self.a - depth, self.c - depth
        t = np.linspace(0, np.pi, 240)
        rho = a * np.sin(t)
        z = c * np.cos(t)
        nip = self.p["nipple_height"] * 1e-3 * max(0.0, 1 - depth / (0.8 * self.t_peel + 1e-9)) ** 0.5
        neck = self.p["neck_height"] * 1e-3 * max(0.0, 1 - depth / (0.8 * self.t_peel + 1e-9)) ** 0.5
        wb = np.exp(-((np.pi - t) / 0.28) ** 2)
        wt = np.exp(-(t / 0.22) ** 2)
        z = z - nip * wb + neck * wt
        rho = rho * (1 - 0.15 * wb * (nip > 0))
        return rho, z

    def stem_mesh(self):
        z0 = float(self.surf_curve(0.0)[1].max()) - 1.0e-3
        return ML.tube(np.linspace([0, 0, z0], [0, 0, z0 + 3.0e-3], 6), np.linspace(2.4e-3, 1.8e-3, 6), 16)

    def bbox(self):
        r = self.a * 1.06
        z = self.surf_curve(0.0)[1]
        return np.array([-r, -r, z.min() - 1e-3]), np.array([r, r, z.max() + 4e-3])

    def uv_of(self, P):
        rn, th, z = self.norm_coords(np.asarray(P, float))
        z0 = self.surf_curve(0.0)[1]
        v = np.clip((z - z0.min()) / (z0.max() - z0.min()), 0, 1)
        return np.stack([np.mod(th, 2 * np.pi) / (2 * np.pi), v], 1)

    def asset_spec(self):
        from .lemon_models import lemon_beliefs, lemon_truth
        spec = super().asset_spec()
        spec["root"] = {"peel_state": "on", "cut_state": "whole", "juice_state": "intact", "cultivar": self.p["cultivar"]}
        spec["beliefs"] = lambda mp: lemon_beliefs(self, mp)
        spec["truth"] = lambda: lemon_truth(self)
        z = self.surf_curve(0.0)[1]
        spec["cuts"][0]["origin"] = [0, 0, float(0.5 * (z.max() + z.min()))]
        spec["cuts"][0]["states"] = {"stem_half": ("pos", {"cut_state": "halved_across"}),
                                     "blossom_half": ("neg", {"cut_state": "halved_across"})}
        for c in spec["cuts"]:
            c["ext"] = float(max(self.a, (z.max() - z.min()) / 2) * 1.15)
        return spec
