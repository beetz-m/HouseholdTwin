"""Wheat bread roll twin (German Weizenbroetchen, the scored 'Schnittbroetchen'): a baked foam, not a plant organ.

Structure (docs/wave1/ROLL.md for the sources): a dense, browned, glassy CRUST around a CRUMB of gas cells in a
gelatinised starch / gluten matrix.  The crust has three zones that look and behave differently: the upper crust
(oven-browned, glossy from steam, fine cooling cracks), the score crust ('Ausbund': the cut edge torn open by oven
spring -- paler, rougher, thinner) and the bottom crust (baked on the hearth or tray: flat, often darker).  Under the
crust a transition zone of compressed, smaller cells; inside, the crumb proper with cells elongated in the rise
direction.  Gas cells are not meshed: they live in the crumb's cell field (CellField) which paints every cut face and
gives the crumb its porosity attributes; the crumb part's mass uses the bulk density.

Frame: footprint ellipse in x (long, along the score) and y; base at z = 0, +z up.

Partonomy (USD paths below /Food/Parts; every prim is an SDT node):
  Crust/{UpperCrust, ScoreCrust, BottomCrust}      shells of the label field (depth below the surface < t_crust)
  Crumb/{TransitionZone, CrumbCore}                crumb under the crust (smaller, flattened cells) and the core
  Crumb/GasCells (points)                          cell centres (guide points; the cell field paints sections)
  Dusting (points)                                 flour on the upper crust (variant 'dusted')
"""

import numpy as np
from scipy.spatial import cKDTree

from . import params as PR
from .geometry import Body
from .partgraph import Part
from .shapes import DepthField, StarUV, smoothstep
from .texture import ValueNoise3, lch_to_srgb, sample_surface, worley

TISSUES = [
    {"name": "upper_crust", "density": 500.0, "colour": (0.62, 0.38, 0.16)},
    {"name": "score_crust", "density": 450.0, "colour": (0.80, 0.62, 0.38)},
    {"name": "bottom_crust", "density": 520.0, "colour": (0.55, 0.36, 0.17)},
    {"name": "transition_zone", "density": 300.0, "colour": (0.93, 0.88, 0.76)},
    {"name": "crumb", "density": 230.0, "colour": (0.96, 0.92, 0.82)},
]
TI = {t["name"]: i + 1 for i, t in enumerate(TISSUES)}


class Roll(Body):
    tissues = TISSUES
    kind = "roll"

    def __init__(self, seed=0, overrides=None):
        p = PR.sample(self.kind, seed)
        p.update(overrides or {})
        rng = np.random.default_rng(p["_rng_state"])
        self.p, self.seed, self.rng = p, seed, rng
        for t in TISSUES:                                   # densities from the register (per instance)
            t["density"] = {"upper_crust": p["crust_density"], "score_crust": p["crust_density"] * 0.9,
                            "bottom_crust": p["crust_density"] * 1.05, "transition_zone": p["transition_density"],
                            "crumb": p["crumb_density"]}[t["name"]]
        self.a, self.b, self.H = p["length"] * 5e-4, p["width"] * 5e-4, p["height"] * 1e-3
        self.e = p["base_edge_height"] * 1e-3               # height where the rounded base edge meets the dome
        self.vn = ValueNoise3(seed + 101)
        self.lump = p["lumpiness"]
        self.score_len = p["score_length_frac"] * self.a
        self.score_w = p["score_width"] * 5e-4
        self.score_d = p["score_depth"] * 1e-3
        self.score_x0 = rng.normal(0, 0.03) * self.a
        self.score_angle = np.radians(rng.normal(0, 4.0))
        self.ear = rng.choice([-1.0, 1.0])                  # the side whose lip lifts ('ear')
        self.t_upper, self.t_score, self.t_bottom = (p["crust_thickness_top"] * 1e-3, p["crust_thickness_score"] * 1e-3,
                                                     p["crust_thickness_bottom"] * 1e-3)
        self.t_trans = p["transition_thickness"] * 1e-3
        super().__init__(None)
        self.depth = DepthField(self.inside, *self.bbox(), h=getattr(self, "depth_h", 0.25e-3))
        self.star = StarUV(self.inside, [0, 0, 0.42 * self.H], max(self.a, self.b) * 1.3)
        self._cells = None
        self.dust_pts = None

    # ---------------------------------------------------------------- outer shape (implicit)
    def _score_frame(self, x, y):
        c, s = np.cos(self.score_angle), np.sin(self.score_angle)
        u = (x - self.score_x0) * c + y * s                 # along the score
        w = -(x - self.score_x0) * s + y * c                # across
        g = np.sqrt(np.clip(1 - (u / self.score_len) ** 2, 0, 1))
        return u, w, g

    footprint_n = 2.0                                      # superellipse exponent along x (baguette: blunter)

    def z_top(self, x, y):
        r = (np.abs(x / self.a) ** self.footprint_n + (y / self.b) ** 2) ** 0.5
        P = np.stack([x, y, np.zeros_like(x)], -1)
        r = r * (1 + self.lump * self.vn.fbm(P, 0.03, 2))
        dome = self.e + (self.H - self.e) * np.clip(1 - np.clip(r, 0, 1) ** 2.4, 0, 1) ** 0.5
        u, w, g = self._score_frame(x, y)
        prof = np.exp(-(w / (self.score_w * 0.55 * np.maximum(g, 1e-3) + 1e-9)) ** 2) * g
        lip = np.exp(-((w - self.ear * 0.9 * self.score_w * g) / (0.35 * self.score_w + 1e-9)) ** 2) * g
        z = dome - self.score_d * prof + 0.25 * self.score_d * lip
        return np.where(r <= 1, z, -1.0), r

    def z_bot(self, r):
        q = np.clip((r - 0.86) / 0.14, 0, 1)
        return self.e * (1 - np.sqrt(np.clip(1 - q ** 2, 0, 1)))

    def inside(self, P):
        P = np.asarray(P, float)
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        zt, r = self.z_top(x, y)
        return (r <= 1) & (z >= self.z_bot(r)) & (z <= zt)

    def bbox(self):
        return np.array([-self.a * 1.08, -self.b * 1.08, -1e-4]), np.array([self.a * 1.08, self.b * 1.08,
                                                                              self.H * 1.06])

    def in_score(self, P):
        """Score region: within the opened cut's width and the upper part of the roll."""
        u, w, g = self._score_frame(P[..., 0], P[..., 1])
        return (np.abs(w) < self.score_w * (0.75 * g + 0.05)) & (g > 0.05) & (P[..., 2] > 0.55 * self.H)

    # ---------------------------------------------------------------- labels
    def label(self, P):
        P = np.asarray(P, float)
        ins = self.inside(P)
        out = np.zeros(P.shape[:-1], np.int16)
        if not ins.any():
            return out
        Q = P[ins]
        dep = self.depth(Q)
        bottom = Q[:, 2] < self.e + 1.5e-3
        score = self.in_score(Q)
        t = np.where(bottom, self.t_bottom, np.where(score, self.t_score, self.t_upper))
        lab = np.full(len(Q), TI["crumb"], np.int16)
        lab[dep < t + self.t_trans] = TI["transition_zone"]
        crust = dep < t
        lab[crust] = TI["upper_crust"]
        lab[crust & score] = TI["score_crust"]
        lab[crust & bottom & ~score] = TI["bottom_crust"]
        out[ins] = lab
        return out

    def tissue_of(self, rn, th, z, P):                      # Body API (label() is overridden)
        return self.label(P)

    # ---------------------------------------------------------------- crumb cell field
    def cells(self):
        """Gas cells: centres, radii (lognormal, register), vertical stretch; smaller and flattened near the crust."""
        if self._cells is not None:
            return self._cells
        rng = np.random.default_rng(self.seed + 7)
        lo, hi = self.bbox()
        n_try = int(self.p["cell_number_density"] * np.prod(hi - lo) * 1e6)
        C = lo + rng.random((n_try, 3)) * (hi - lo)
        ins = self.inside(C)
        C = C[ins]
        dep = self.depth(C)
        keep = dep > self.t_bottom * 0.6
        C, dep = C[keep], dep[keep]
        r = self.p["cell_diameter_median"] * 5e-4 * np.exp(self.p["cell_diameter_gsd_log"] * rng.normal(size=len(C)))
        near = 1 - smoothstep(0, self.t_upper + 2.5 * self.t_trans, dep)
        r = r * (1 - 0.55 * near)
        stretch = 1 + (self.p["cell_elongation"] - 1) * (1 - near)
        self._cells = (C, r, stretch)
        return self._cells

    def cell_hit(self, P):
        """Foam section from the cell field: (wall weight 0..1, depth into the cell 0 wall .. 1 centre).  Worley F1/F2
        on the (rise-stretched) cell centres: the bisector between neighbouring cells is a wall; cell walls are
        20-61 um (hirte_2012), i.e. hairlines at texture scale, thicker in the transition zone."""
        C, r, st = self.cells()
        Ps = np.asarray(P, float).copy()
        Cs = C.copy()
        s_ = float(np.median(st))
        Ps[:, 2] /= s_
        Cs[:, 2] /= st
        d, i = cKDTree(Cs).query(Ps, k=8)
        # power (Laguerre) diagram: big cells claim more space -> realistic spread of hole sizes
        pw = d ** 2 - (1.15 * r[i]) ** 2
        o = np.argsort(pw, axis=1)
        a, b = o[:, 0], o[:, 1]
        rows = np.arange(len(Ps))
        ia, ib = i[rows, a], i[rows, b]
        sep = np.linalg.norm(Cs[ia] - Cs[ib], axis=1) + 1e-12
        gap = (pw[rows, b] - pw[rows, a]) / sep             # 2 x distance to the power bisector (the wall)
        dep = self.depth(P)
        vn = ValueNoise3(self.seed + 43)
        w_wall = (0.07e-3 + 0.20e-3 * (1 - smoothstep(0, self.t_upper + 2.5 * self.t_trans, dep))) * \
            (0.6 + 0.9 * np.clip(0.5 + 0.5 * vn.fbm(np.asarray(P), 0.8e-3, 2), 0, 1))   # oblique cuts look wider
        wall = np.exp(-(gap / w_wall) ** 2)
        depth_in = np.clip(gap / (2 * r[ia] + 1e-12), 0, 1)
        return wall, depth_in

    # ---------------------------------------------------------------- parts
    def dusting_points(self):
        if self.p["variant"] != "dusted":
            return np.zeros((0, 3)), np.zeros(0)
        U, V = np.meshgrid(np.linspace(0, 1, 360), np.linspace(0.55, 1.0, 120))
        S = self.star.point_uv(U, V).reshape(-1, 3)
        rng = np.random.default_rng(self.seed + 9)
        vn = ValueNoise3(self.seed + 10)
        w = np.clip(vn.fbm(S, 0.012, 3) + 0.2, 0, 1)
        keep = rng.random(len(S)) < w * 0.8
        S = S[keep] + rng.normal(0, 0.15e-3, (keep.sum(), 3))
        return S, np.full(len(S), 0.12e-3)

    def parts(self):
        cells = lambda: (self.cells()[0], 2 * self.cells()[1])  # noqa: E731
        p = self.p
        return [
            Part("crust", "Crust", "food:BreadCrust", None, observability="visible",
                 doc="Dense, browned (Maillard + caramelisation), glassy when fresh: crisp; softens as it takes up "
                     "water from the crumb and the air (Roesche lost within hours)."),
            Part("upper_crust", "Crust/UpperCrust", "food:UpperCrust", ("label", ["upper_crust"]), "UpperCrust",
                 "visible", "Oven-browned top and sides, glossy from steam; fine cooling cracks.",
                 interior=False, attrs={"thickness_mm": p["crust_thickness_top"]}),
            Part("score_crust", "Crust/ScoreCrust", "food:ScoreCrust", ("label", ["score_crust"]), "ScoreCrust",
                 "visible", "'Ausbund': the score torn open by oven spring; paler, rougher, thinner crust.",
                 interior=False, attrs={"thickness_mm": p["crust_thickness_score"]}),
            Part("bottom_crust", "Crust/BottomCrust", "food:BottomCrust", ("label", ["bottom_crust"]), "BottomCrust",
                 "visible", "Baked on the hearth or tray: flat, often darker.", interior=False,
                 attrs={"thickness_mm": p["crust_thickness_bottom"]}),
            Part("crumb_group", "Crumb", "food:BreadCrumb", None, observability="visible after cutting",
                 doc="Solid foam: gas cells in a gelatinised starch / gluten matrix; soft and springy when fresh, "
                     "firms over days (starch retrogradation)."),
            Part("transition_zone", "Crumb/TransitionZone", "food:CrustCrumbTransition",
                 ("label", ["transition_zone"]), "Transition", "visible after cutting",
                 "Under the crust: compressed, smaller, flattened cells; denser than the core.",
                 attrs={"thickness_mm": p["transition_thickness"]}),
            Part("crumb", "Crumb/CrumbCore", "food:CrumbCore", ("label", ["crumb"]), "Crumb",
                 "visible after cutting", "Crumb proper: open cells, elongated in the rise direction.",
                 attrs={"porosity": p["crumb_porosity"], "cell_diameter_median_mm": p["cell_diameter_median"]}),
            Part("gas_cells", "Crumb/GasCells", "food:GasCell", ("points", cells), None, "visible after cutting",
                 "Cell centres and diameters of the crumb's cell field (guide points; sections are painted from it)."),
            Part("dusting", "Dusting", "food:FlourDusting", ("points", self.dusting_points), None, "visible",
                 "Flour on the upper crust (variant 'dusted').", interior=False),
        ]

    # ---------------------------------------------------------------- appearance
    def crust_colour(self, P):
        """Upper crust colour from the register's L*a*b*-derived LCh (browning stronger on the top)."""
        p = self.p
        hgt = np.clip(P[..., 2] / self.H, 0, 1)
        top = lch_to_srgb(p["crust_L"] - 6, p["crust_C"] + 4, p["crust_h"] - 3)
        side = lch_to_srgb(p["crust_L"] + 10, p["crust_C"] - 2, p["crust_h"] + 6)
        t = smoothstep(0.25, 0.85, hgt)[..., None]
        return side * (1 - t) + top * t

    def skin_albedo(self, P, seed):
        P = np.asarray(P, float)
        vn = ValueNoise3(seed + 31)
        col = self.crust_colour(P)
        col = col * (1 + 0.10 * vn.fbm(P, 6e-3, 3))[..., None]
        # fine cooling cracks of the glassy crust (network of thin lines)
        rng = np.random.default_rng(seed + 32)
        if not hasattr(self, "_crack_seeds"):
            U, V = rng.random(4000), 0.35 + 0.65 * rng.random(4000)
            self._crack_seeds = self.star.point_uv(U, V)
        d1, d2 = worley(P, self._crack_seeds, 1), worley(P, self._crack_seeds, 2)
        crack = np.exp(-((d2 - d1) / 0.10e-3) ** 2) * (P[..., 2] > 0.4 * self.H)
        col = col * (1 + 0.18 * crack[..., None])           # cracks show paler (thinner glaze)
        # score: paler, rough torn dough; lip edge darker (most heat)
        sc = self.in_score(P).astype(float)
        pale = lch_to_srgb(self.p["crust_L"] + 18, self.p["crust_C"] - 6, self.p["crust_h"] + 8)
        rough = 0.85 + 0.25 * (0.5 + 0.5 * vn.fbm(P + 5, 0.8e-3, 3))
        col = col * (1 - sc[..., None]) + (pale * rough[..., None]) * sc[..., None]
        u, w, g = self._score_frame(P[..., 0], P[..., 1])
        edge = np.exp(-((np.abs(w) - self.score_w * (0.75 * g + 0.05)) / 0.6e-3) ** 2) * (g > 0.05) * \
            (P[..., 2] > 0.55 * self.H)
        col = col * (1 - 0.30 * edge[..., None])
        # bottom: darker, matte, with flour / tray marks
        bot = (P[..., 2] < self.e + 0.8e-3).astype(float)
        col = col * (1 - 0.18 * bot[..., None]) + np.array([0.85, 0.80, 0.70]) * 0.12 * bot[..., None] * \
            np.clip(vn.fbm(P + 9, 2e-3, 2), 0, 1)[..., None]
        if self.p["variant"] == "dusted":
            dw = np.clip(ValueNoise3(self.seed + 10).fbm(P, 0.012, 3) + 0.2, 0, 1) * (P[..., 2] > 0.55 * self.H)
            col = col * (1 - 0.55 * dw[..., None]) + np.array([0.93, 0.91, 0.86]) * 0.55 * dw[..., None]
        return np.clip(col, 0, 1)

    def point_uv(self, U, V):
        return self.star.point_uv(U, V)

    def uv_of(self, P):
        return self.star.uv_of(P)

    def section_rgb(self, lab, P, seed=0):
        """Cut face: crust zones by colour, crumb with its cells (shadowed cell interiors, light cell walls)."""
        shape = lab.shape
        lab = lab.ravel()
        P = P.reshape(-1, 3)
        rgb = np.ones((len(lab), 3)) * np.array([0.86, 0.86, 0.85])
        vn = ValueNoise3(seed + 41)
        crumb_c = lch_to_srgb(self.p["crumb_L"], self.p["crumb_C"], self.p["crumb_h"])
        for i, t in enumerate(TISSUES):
            rgb[lab == i + 1] = t["colour"]
        cr = np.isin(lab, [TI["upper_crust"], TI["score_crust"], TI["bottom_crust"]])
        if cr.any():
            dep = self.depth(P[cr])
            outer = self.crust_colour(P[cr]) * 0.85
            inner = lch_to_srgb(self.p["crust_L"] + 22, self.p["crust_C"] - 4, self.p["crust_h"] + 8)
            t = np.clip(dep / max(self.t_upper, 1e-4), 0, 1)[:, None] ** 0.7
            rgb[cr] = outer * (1 - t) + inner * t
        cm = np.isin(lab, [TI["transition_zone"], TI["crumb"]])
        if cm.any():
            Q = P[cm]
            wall, din = self.cell_hit(Q)
            walls = crumb_c * 1.10 * (1 + 0.03 * vn.fbm(Q, 0.3e-3, 2))[:, None]
            # the cut opens each cell: its inner surface falls away from the wall into shadow (warmer, darker)
            hole = crumb_c * (1.0 - 0.30 * din ** 0.8)[:, None] * np.array([1.0, 0.975, 0.94])
            tz = (lab[cm] == TI["transition_zone"])[:, None]
            col = walls * wall[:, None] + hole * (1 - wall[:, None])
            col = np.where(tz, 0.6 * col + 0.4 * walls, col)
            rgb[cm] = col
        return np.clip(rgb, 0, 1).reshape(shape + (3,))

    # ---------------------------------------------------------------- asset declaration (scripts/build_food_assets)
    def asset_spec(self):
        from .roll_models import roll_beliefs, roll_truth
        p = self.p
        crust = tuple(float(c) for c in self.crust_colour(np.array([0, 0, self.H])))
        crumb = tuple(float(c) for c in lch_to_srgb(p["crumb_L"], p["crumb_C"], p["crumb_h"]))
        z_slice = p["slice_height_frac"] * self.H
        ext = float(max(self.a, self.H) * 1.12)
        return {
            "build": {"h": 0.3e-3, "sigma": 0.7, "smooth": 12},
            "textures": [("skin.png", self.point_uv, self.skin_albedo, 2048, 1024)],
            "uv_fns": {"upper_crust": self.uv_of, "score_crust": self.uv_of, "bottom_crust": self.uv_of},
            "materials": {
                "UpperCrust": dict(diffuse_reflection_color_image="tex:skin.png", specular_reflection_roughness=0.38,
                                   specular_reflection_weight=0.45, coat_weight=0.15, coat_roughness=0.3),
                "ScoreCrust": dict(diffuse_reflection_color_image="tex:skin.png", specular_reflection_roughness=0.7,
                                   specular_reflection_weight=0.2),
                "BottomCrust": dict(diffuse_reflection_color_image="tex:skin.png", specular_reflection_roughness=0.8,
                                    specular_reflection_weight=0.15),
                "Transition": dict(diffuse_reflection_color=crumb, specular_reflection_roughness=0.8),
                "Crumb": dict(diffuse_reflection_color=crumb, specular_reflection_roughness=0.85,
                              subsurface_weight=0.1, subsurface_scale=0.002, subsurface_scattering_color=crumb),
            },
            "section_material": {"specular_reflection_roughness": 0.85, "specular_reflection_weight": 0.15},
            "root": {"freshness": "fresh", "crust_texture": "crisp", "cut_state": "whole", "toast_level": "none",
                     "variant": p["variant"], "crust_colour_srgb": str(crust)},
            "friction": p["friction_crust_rubber"], "restitution": 0.15,
            "cuts": [
                {"name": "cross", "origin": [0, 0, 0], "e1": [0, 1, 0], "e2": [0, 0, 1], "ext": ext,
                 "states": {"half_a": ("pos", {"cut_state": "halved_across"}),
                            "half_b": ("neg", {"cut_state": "halved_across"})}},
                {"name": "sliced", "origin": [0, 0, z_slice], "e1": [1, 0, 0], "e2": [0, 1, 0], "ext": ext,
                 "states": {"top_half": ("pos", {"cut_state": "sliced_open"}),
                            "bottom_half": ("neg", {"cut_state": "sliced_open"})}},
            ],
            "beliefs": lambda mp: roll_beliefs(self, mp),
            "truth": lambda: roll_truth(self),
        }
