"""Sandwich loaf twin (German Toastbrot / Kastenweissbrot): tin-baked white bread, sold pre-sliced.

Structure (docs/wave2/LOAF.md for the sources): a box-shaped baked foam -- thin, soft, pale crust where the dough
touched the tin (sides, base), a browner TOP crust (domed for an open tin, flat for a lidded 'Pullman' tin), and a
fine, even crumb.  Sold as N SLICES of equal thickness with the two HEELS ('Kanten'/'Knust') at the ends; every slice
is its own part (crust rim + crumb), so a robot can take one out: the asset has a `slices` state with one rigid body
per slice.

Geometry: an implicit body (tin box with rounded edges, optional dome) like the roll (food/roll.py): crust = depth
below the surface (DepthField) thinner on the tin faces; crumb cells from a cell field (finer than the roll's);
slice index from x.  Frame: x along the loaf (slices stack along x), base at z = 0.

Partonomy (USD paths below /Food/Parts):
  Slices/Slice_kk/{Crust, Crumb}   (kk = 01 .. N; 01 and N are the heels)
  Slices/Slice_kk/Crumb/GasCells   (points, guide)
"""

import numpy as np
from scipy.spatial import cKDTree

from . import params as PR
from .geometry import Body
from .partgraph import Part
from .shapes import DepthField, StarUV, smoothstep
from .texture import ValueNoise3, lch_to_srgb, worley


class Loaf(Body):
    kind = "loaf"

    def __init__(self, seed=0, overrides=None):
        p = PR.sample("loaf", seed)
        p.update(overrides or {})
        rng = np.random.default_rng(p["_rng_state"])
        self.p, self.seed, self.rng = p, seed, rng
        self.Lx, self.Wy, self.Hz = p["length"] * 1e-3, p["width"] * 1e-3, p["tin_height"] * 1e-3
        self.dome = p["dome_height"] * 1e-3
        self.rc = p["edge_radius"] * 1e-3
        self.n = int(p["slices"])
        self.t_slice = self.Lx / self.n
        self.t_top, self.t_side = p["crust_thickness_top"] * 1e-3, p["crust_thickness_tin"] * 1e-3
        self.vn = ValueNoise3(seed + 201)
        super().__init__(None)
        # no heels (most packs, test25_toast): the ends are cut faces -> crust only from the long sides
        if int(p["heels"]) == 0:
            lo, hi = self.bbox()
            lo[0] -= 8e-3
            hi[0] += 8e-3
            self.depth = DepthField(lambda P: self.inside(P, ends=False), lo, hi, h=0.35e-3)
        else:
            self.depth = DepthField(self.inside, *self.bbox(), h=0.35e-3)
        self.star = StarUV(self.inside, [0, 0, 0.5 * self.Hz], 0.25)
        self._cells = None

    # ---------------------------------------------------------------- outer shape
    def z_top(self, x, y):
        """Top surface: tin rim height plus a dome (rounded over y, flattened towards the loaf ends)."""
        u = np.clip(np.abs(y) / (self.Wy / 2), 0, 1)
        v = np.clip(np.abs(x) / (self.Lx / 2), 0, 1)
        dome = self.dome * np.sqrt(np.clip(1 - u ** 2.2, 0, 1)) * (1 - 0.35 * v ** 6)
        P = np.stack([x, y, np.zeros_like(x)], -1)
        return self.Hz + dome * (1 + 0.04 * self.vn.fbm(P, 0.04, 2)) - 1.0e-3 * smoothstep(0.85, 1.0, u)

    def inside(self, P, ends=True):
        P = np.asarray(P, float)
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        rc = self.rc
        if not ends:                                      # an endless loaf (depth from the long sides only)
            x = np.clip(x, -self.Lx / 2 + rc, self.Lx / 2 - rc)
        # rounded box of the tin part: superellipse corners in y-z and x-z
        dx = np.clip(np.abs(x) - (self.Lx / 2 - rc), 0, None)
        dy = np.clip(np.abs(y) - (self.Wy / 2 - rc), 0, None)
        dzb = np.clip(rc - z, 0, None)
        box = (dx ** 2 + dy ** 2 + dzb ** 2 <= rc ** 2) & (z >= 0)
        if int(self.p["heels"]) == 0:                     # cut ends: square ends, no rounding
            dx = 0.0 * dx
            box = (dy ** 2 + dzb ** 2 <= rc ** 2) & (z >= 0)
        return box & (z <= self.z_top(x, y)) & (np.abs(x) <= self.Lx / 2) & (np.abs(y) <= self.Wy / 2)

    def bbox(self):
        return np.array([-self.Lx / 2 - 1e-3, -self.Wy / 2 - 1e-3, -1e-3]), \
            np.array([self.Lx / 2 + 1e-3, self.Wy / 2 + 1e-3, self.Hz + self.dome + 2e-3])

    def slice_of(self, x):
        return np.clip(((np.asarray(x) + self.Lx / 2) / self.t_slice).astype(int), 0, self.n - 1)

    # ---------------------------------------------------------------- labels
    @property
    def tissues(self):
        return [{"name": "top_crust", "density": self.p["crust_density"], "colour": lch_to_srgb(*self.p["crust_LCh"])},
                {"name": "tin_crust", "density": self.p["crust_density"] * 0.9,
                 "colour": lch_to_srgb(self.p["crust_LCh"][0] + 14, self.p["crust_LCh"][1] - 6,
                                       self.p["crust_LCh"][2] + 6)},
                {"name": "crumb", "density": self.p["crumb_density"], "colour": lch_to_srgb(*self.p["crumb_LCh"])}]

    def is_top(self, P):
        return P[:, 2] > self.Hz - 1.5e-3

    def label(self, P):
        P = np.asarray(P, float)
        ins = self.inside(P)
        out = np.zeros(P.shape[:-1], np.int16)
        if not ins.any():
            return out
        Q = P[ins]
        top = self.is_top(Q)
        crust = self.depth(Q) < np.where(top, self.t_top, self.t_side)
        lab = np.full(len(Q), 3, np.int16)
        lab[crust & top] = 1
        lab[crust & ~top] = 2
        out[ins] = lab
        return out

    def tissue_of(self, rn, th, z, P):
        return self.label(P)

    def slice_planes(self):
        """x offsets of the cuts between the slices (pre-sliced loaf)."""
        return -self.Lx / 2 + self.t_slice * np.arange(1, self.n)

    # ---------------------------------------------------------------- crumb cells (as in the roll, finer)
    def cells(self):
        if self._cells is None:
            rng = np.random.default_rng(self.seed + 11)
            lo, hi = self.bbox()
            n = int(self.p["cell_number_density"] * np.prod(hi - lo) * 1e6)
            C = lo + rng.random((n, 3)) * (hi - lo)
            C = C[self.inside(C)]
            r = self.p["cell_diameter_median"] * 5e-4 * np.exp(0.5 * rng.normal(size=len(C)))
            self._cells = (C, r)
        return self._cells

    def foam(self, P):
        C, r = self.cells()
        Cs, Ps = C.copy(), np.asarray(P, float).copy()
        Cs[:, 2] /= self.p["cell_elongation"]
        Ps[:, 2] /= self.p["cell_elongation"]
        d, i = cKDTree(Cs).query(Ps, k=6)
        pw = d ** 2 - (1.1 * r[i]) ** 2
        o = np.argsort(pw, axis=1)
        rows = np.arange(len(Ps))
        ia, ib = i[rows, o[:, 0]], i[rows, o[:, 1]]
        sep = np.linalg.norm(Cs[ia] - Cs[ib], axis=1) + 1e-12
        gap = (pw[rows, o[:, 1]] - pw[rows, o[:, 0]]) / sep
        w = 0.06e-3 * (0.6 + 0.9 * np.clip(0.5 + 0.5 * self.vn.fbm(P, 0.7e-3, 2), 0, 1))
        return np.exp(-(gap / w) ** 2), np.clip(gap / (2 * r[ia] + 1e-12), 0, 1)

    # ---------------------------------------------------------------- parts
    def parts(self):
        cells = lambda: (self.cells()[0], 2 * self.cells()[1])  # noqa: E731
        return [
            Part("crust", "Crust", "food:BreadCrust", None, observability="visible",
                 doc="Thin soft tin crust (sides, base, ends) and the browner top crust."),
            Part("top_crust", "Crust/TopCrust", "food:TopCrust", ("label", ["top_crust"]), "TopCrust", "visible",
                 "Oven-browned top (domed or flat).", interior=False,
                 attrs={"thickness_mm": self.p["crust_thickness_top"]}),
            Part("tin_crust", "Crust/TinCrust", "food:TinCrust", ("label", ["tin_crust"]), "TinCrust", "visible",
                 "Where the dough touched the tin: thin, pale, soft.", interior=False,
                 attrs={"thickness_mm": self.p["crust_thickness_tin"]}),
            Part("crumb", "Crumb", "food:BreadCrumb", ("label", ["crumb"]), "Crumb", "visible after cutting",
                 "Fine, even crumb of tin bread."),
            Part("gas_cells", "Crumb/GasCells", "food:GasCell", ("points", cells), None, "visible after cutting",
                 "Cell field of the crumb (guide points; painted on the cut faces)."),
        ]

    # ---------------------------------------------------------------- appearance
    def point_uv(self, U, V):
        return self.star.point_uv(U, V)

    def uv_of(self, P):
        return self.star.uv_of(P)

    def skin_albedo(self, P, seed):
        P = np.asarray(P, float)
        vn = ValueNoise3(seed + 211)
        z = P[..., 2]
        top = smoothstep(self.Hz - 3e-3, self.Hz + 2e-3, z)
        c_top = lch_to_srgb(*self.p["crust_LCh"])
        c_tin = lch_to_srgb(self.p["crust_LCh"][0] + 14, self.p["crust_LCh"][1] - 6, self.p["crust_LCh"][2] + 6)
        col = c_tin * (1 - top[..., None]) + c_top * top[..., None]
        col = col * (1 + 0.07 * vn.fbm(P, 5e-3, 3))[..., None]
        streak = 0.5 + 0.5 * vn.fbm(np.stack([P[..., 0] * 0.3, P[..., 1] * 3, P[..., 2] * 3], -1), 2e-3, 2)
        col = col * (1 - 0.08 * (1 - top[..., None]) * streak[..., None])
        return np.clip(col, 0, 1)

    def section_rgb(self, lab, P, seed=0):
        shape = lab.shape
        lab, P = lab.ravel(), P.reshape(-1, 3)
        rgb = np.ones((len(lab), 3)) * 0.86
        crumb = lch_to_srgb(*self.p["crumb_LCh"])
        is_crust = (lab == 1) | (lab == 2)
        is_crumb = lab == 3
        if is_crust.any():
            dep = self.depth(P[is_crust])
            t = np.clip(dep / self.t_side, 0, 1)[:, None]
            c = self.skin_albedo(P[is_crust], seed)
            rgb[is_crust] = c * (1 - t) + crumb * 0.95 * t
        if is_crumb.any():
            wall, din = self.foam(P[is_crumb])
            walls = crumb * 1.08
            hole = crumb * (1.0 - 0.28 * din ** 0.8)[:, None] * np.array([1.0, 0.975, 0.94])
            rgb[is_crumb] = walls * wall[:, None] + hole * (1 - wall[:, None])
        return np.clip(rgb, 0, 1).reshape(shape + (3,))

    # ---------------------------------------------------------------- asset declaration
    def asset_spec(self):
        from .loaf_models import loaf_beliefs, loaf_truth
        p = self.p
        crumb = tuple(float(c) for c in lch_to_srgb(*p["crumb_LCh"]))
        ext = float(max(self.Wy, self.Hz + self.dome) * 0.62)
        return {
            "build": {"h": 0.4e-3, "sigma": 0.7, "smooth": 8},
            "textures": [("skin.png", self.point_uv, self.skin_albedo, 2048, 1024)],
            "uv_fns": {"top_crust": self.uv_of, "tin_crust": self.uv_of},
            "materials": {
                "TopCrust": dict(diffuse_reflection_color_image="tex:skin.png", specular_reflection_roughness=0.55,
                                 specular_reflection_weight=0.3),
                "TinCrust": dict(diffuse_reflection_color_image="tex:skin.png", specular_reflection_roughness=0.7,
                                 specular_reflection_weight=0.2),
                "Crumb": dict(diffuse_reflection_color=crumb, specular_reflection_roughness=0.85,
                              subsurface_weight=0.1, subsurface_scale=0.002, subsurface_scattering_color=crumb),
            },
            "section_material": {"specular_reflection_roughness": 0.85, "specular_reflection_weight": 0.15},
            "root": {"freshness": "fresh", "slices_left": self.n, "bag": "closed", "toast_level": "none"},
            "friction": p["friction_bread"], "restitution": 0.1,
            "cuts": [
                {"name": "lengthwise", "origin": [0, 0, 0], "e1": [1, 0, 0], "e2": [0, 0, 1], "ext": float(self.Lx * 0.56),
                 "states": {"half_a": ("pos", {"state_note": "anatomical lengthwise cut"}),
                            "half_b": ("neg", {"state_note": "anatomical lengthwise cut"})}},
            ],
            "slicings": [{"name": "slices", "normal": [1, 0, 0], "offsets": list(map(float, self.slice_planes())),
                          "e1": [0, 1, 0], "e2": [0, 0, 1], "ext": ext, "section_at": 0.0,
                          "group": "Slices", "piece": "Slice", "piece_class": "food:BreadSlice",
                          "end_class": "food:BreadHeel", "root": {"state_note": "pre-sliced: one rigid body per slice"}}],
            "beliefs": lambda mp: loaf_beliefs(self, mp),
            "truth": lambda: loaf_truth(self),
        }
