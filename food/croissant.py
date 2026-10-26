"""Butter croissant twin (Buttercroissant): a rolled triangle of laminated yeast dough (Plunder).

Structure (docs/wave3/CROISSANT.md for the sources): the triangle is rolled 3.5-4 turns from its base, so its
cross-section is a SPIRAL of dough layers; lamination (3 single folds -> 27 butter layers) makes each turn bake into
thin flaky sheets with large elongated voids between them (the honeycomb crumb); the outside is a thin, glossy,
deep-golden CRUST from the egg wash, shedding FLAKES; the roll shows as STEPS along the body; thin TIPS at the ends.
German law: >= 30 kg butter per 100 kg flour for Plunder, butter the only fat in a Buttercroissant (Leitsaetze).

Geometry: a straight spindle with a flat-bottomed, wider-than-tall cross-section and step ridges (modulation), bent
into a crescent (food.shapes.Bend) -- object frame: x along the chord, y towards the inside of the crescent, z up.
Crust = shell below the outer face; the laminated interior is labelled by its spiral turn (cut faces paint the
dough sheets and the voids).

Partonomy (USD paths below /Food/Parts):  Crust (glazed), LaminatedCrumb, Flakes (points)
"""

import numpy as np

from . import meshlib as ML
from . import params as PR
from .geometry import Body, Meridian, resample, smooth_meridian
from .partgraph import Part
from .shapes import Bend, smoothstep
from .texture import ValueNoise3, lch_to_srgb

TISSUES = [
    {"name": "crust", "density": 450.0, "colour": (0.70, 0.42, 0.14)},
    {"name": "crumb", "density": 300.0, "colour": (0.95, 0.86, 0.62)},
]
TI = {t["name"]: i + 1 for i, t in enumerate(TISSUES)}


def to_bent(P):
    """object (x chord, y inside, z up) -> bend frame (X bend direction, Y up, Z chord)."""
    P = np.asarray(P, float)
    return P[..., [1, 2, 0]]


def from_bent(Q):
    Q = np.asarray(Q, float)
    return Q[..., [2, 0, 1]]


class Croissant(Body):
    tissues = TISSUES
    kind = "croissant"

    def __init__(self, seed=0, overrides=None):
        p = PR.sample("croissant", seed)
        p.update(overrides or {})
        rng = np.random.default_rng(p["_rng_state"])
        self.p, self.seed, self.rng = p, seed, rng
        TISSUES[0]["density"], TISSUES[1]["density"] = p["crust_density"], p["crumb_density"]
        L, W, H = p["length"] * 1e-3, p["width"] * 1e-3, p["height"] * 1e-3
        self.L, self.W, self.H = L, W, H
        s = np.linspace(0, 1, 41)
        r = W / 2 * np.clip(np.sin(np.pi * s), 0, 1) ** 0.75 * (1 - 0.15 * np.abs(2 * s - 1) ** 2)
        r[0] = r[-1] = 0.0
        z = (s - 0.5) * L
        rho, zz = smooth_meridian(np.stack([r[::-1], z[::-1]], 1), 500)
        n_steps, a_step = int(p["steps"]), p["step_relief"]
        aspect = H / W
        self.vn = ValueNoise3(seed + 301)

        def mod(th, zc):
            # cross-section: wider than tall (aspect), flattened bottom (y < 0 in the straight frame = down)
            c, sn = np.cos(th), np.sin(th)
            ell = 1.0 / np.sqrt(c ** 2 + (sn / aspect) ** 2)
            flat = 1 - 0.45 * np.clip(-sn, 0, 1) ** 3
            u = (zc / L + 0.5)
            steps = 1 + a_step * (0.5 + 0.5 * np.cos(2 * np.pi * n_steps * np.abs(u - 0.5) * 2))
            P = np.stack([c * 0.02, sn * 0.02, zc], -1)
            return ell * flat * steps * (1 + 0.02 * self.vn.fbm(P, 0.02, 2))
        super().__init__(Meridian(rho, zz), mod)
        self.bend = Bend(p["curvature_radius"] * 1e-3)
        self.t_crust = p["crust_thickness"] * 1e-3
        self.m_in = self.mer.offset(self.t_crust)
        self.turns = p["rolling_turns"]

    # ---------------------------------------------------------------- labels (object -> bend frame -> straight)
    def straight(self, P):
        return self.bend.unbend(to_bent(P))

    def label(self, P):
        S = self.straight(P)
        rn, th, z = self.norm_coords(S)
        out = np.zeros(S.shape[:-1], np.int16)
        ins = self.mer.contains(rn, z)
        out[ins] = TI["crumb"]
        out[ins & ~self.m_in.contains(rn, z)] = TI["crust"]
        return out

    def tissue_of(self, rn, th, z, P):
        return self.label(P)

    def inside(self, P):
        return self.label(P) > 0

    def bbox(self):
        z = np.linspace(self.mer.zmin, self.mer.zmax, 16)
        pts = from_bent(self.bend.bend(np.array([[sx * self.W * 0.6, sy * self.W * 0.6, zz] for sx in (-1, 1)
                                                 for sy in (-1, 1) for zz in z])))
        return pts.min(0) - 1e-3, pts.max(0) + 1e-3

    def spiral(self, P):
        """Spiral coordinate of the rolled dough in the cross-section: sheet phase (0..1 between sheets)."""
        S = self.straight(P)
        rn, th, z = self.norm_coords(S)
        R = np.maximum(np.interp(z, self.mer.z[::-1], self.mer.rho[::-1]), 1e-4)
        q = np.clip(rn / R, 0, 1)
        phase = q * self.turns - (np.mod(th, 2 * np.pi)) / (2 * np.pi)
        return np.mod(phase, 1.0), q

    # ---------------------------------------------------------------- meshes
    def surf(self, mer, n_theta=128, n_mer=300):
        r, z = resample(mer.rho, mer.z, n_mer)
        r[0] = r[-1] = 0
        m = ML.surface_of_revolution(Meridian(r, z), n_theta, modulation=self.mod)
        m = self.bend.bend_mesh(m)
        m.vertices = from_bent(np.asarray(m.vertices))
        return m

    def flake_points(self):
        rng = np.random.default_rng(self.seed + 7)
        m = self.surf(self.mer, 64, 120)
        pts, _ = ML.points_on(m, int(self.p["flakes"]), rng)
        return pts, np.full(len(pts), 0.8e-3)

    def parts(self):
        cache = {}

        def c(k, fn):
            def g():
                if k not in cache:
                    cache[k] = fn()
                return cache[k]
            return g
        so, si = c("o", lambda: self.surf(self.mer)), c("i", lambda: self.surf(self.m_in))
        p = self.p
        return [
            Part("crust", "Crust", "food:GlazedCrust", ("mesh", lambda: ML.hollow(so(), si())), "Crust", "visible",
                 "Thin flaky crust, glossy from the egg wash; sheds flakes when handled.", TISSUES[0]["density"],
                 interior=False, attrs={"steps": int(p["steps"]), "glaze": "egg wash"}),
            Part("crumb", "LaminatedCrumb", "food:LaminatedCrumb", ("mesh", si), "Crumb", "visible after cutting",
                 "Spiral of laminated dough sheets with large elongated voids (honeycomb).", TISSUES[1]["density"],
                 attrs={"butter_layers": int(p["butter_layers"]), "rolling_turns": float(p["rolling_turns"])}),
            Part("flakes", "Flakes", "food:Flake", ("points", self.flake_points), None, "visible",
                 "Loose crust flakes (shed on handling and cutting).", interior=False),
        ]

    # ---------------------------------------------------------------- appearance
    def point_uv(self, U, V):
        k = np.linspace(0, 1, len(self.mer.rho))
        t = 1 - np.asarray(V)
        rho, z = np.interp(t, k, self.mer.rho), np.interp(t, k, self.mer.z)
        th = 2 * np.pi * np.asarray(U)
        m = self.mod(th, z)
        S = np.stack([m * rho * np.cos(th), m * rho * np.sin(th), z], -1)
        return from_bent(self.bend.bend(S))

    def uv_of(self, P):
        from scipy.spatial import cKDTree
        S = self.straight(P)
        rn, th, z = self.norm_coords(S)
        k = np.linspace(0, 1, len(self.mer.rho))
        _, i = cKDTree(np.stack([self.mer.rho, self.mer.z], 1)).query(np.stack([rn, z], 1))
        return np.stack([np.mod(th, 2 * np.pi) / (2 * np.pi), 1 - k[i]], 1)

    def skin_albedo(self, P, seed):
        P = np.asarray(P, float)
        S = self.straight(P)
        vn = ValueNoise3(seed + 311)
        base = lch_to_srgb(*self.p["crust_LCh"])
        z = S[..., 2]
        u = z / self.L + 0.5
        ridge = 0.5 + 0.5 * np.cos(2 * np.pi * int(self.p["steps"]) * np.abs(u - 0.5) * 2)
        col = base * (0.80 + 0.32 * ridge[..., None])            # ridges browner, grooves paler
        sheet = 0.5 + 0.5 * vn.fbm(np.stack([S[..., 0] * 3, S[..., 1] * 3, S[..., 2] * 0.5], -1), 0.8e-3, 3)
        col = col * (0.9 + 0.2 * sheet[..., None])
        th = np.arctan2(S[..., 1], S[..., 0])
        bottom = smoothstep(0.3, 0.9, -np.sin(th))
        col = col * (1 - 0.15 * bottom[..., None])
        tips = np.maximum(1 - smoothstep(0.0, 0.1, u), smoothstep(0.9, 1.0, u))
        col = col * (1 - 0.25 * tips[..., None])
        return np.clip(col, 0, 1)

    def section_rgb(self, lab, P, seed=0):
        shape = lab.shape
        lab, P = lab.ravel(), P.reshape(-1, 3)
        rgb = np.ones((len(lab), 3)) * 0.86
        crumb = lch_to_srgb(*self.p["crumb_LCh"])
        rgb[lab == TI["crust"]] = lch_to_srgb(*self.p["crust_LCh"]) * 0.9
        m = lab == TI["crumb"]
        if m.any():
            ph, q = self.spiral(P[m])
            vn = ValueNoise3(seed + 321)
            n = 0.5 + 0.5 * vn.fbm(P[m] * np.array([1, 1, 0.3]), 0.7e-3, 3)
            # dough sheet near phase 0 and 1, void in between whose width varies (the honeycomb)
            d = np.minimum(ph, 1 - ph)
            sheet = d < 0.10 + 0.12 * n
            voidm = ~sheet & (n > 0.35)
            col = np.where(sheet[:, None], crumb * 1.04, crumb * 0.86)
            col = np.where(voidm[:, None], crumb * (0.55 + 0.25 * (1 - n))[:, None], col)
            col[q > 0.85] = crumb * 0.97                       # outer turn: denser
            rgb[m] = col
        return np.clip(rgb, 0, 1).reshape(shape + (3,))

    # ---------------------------------------------------------------- asset declaration
    def asset_spec(self):
        from .croissant_models import croissant_beliefs, croissant_truth
        p = self.p
        lo, hi = self.bbox()
        cen = (lo + hi) / 2
        crumb = tuple(float(c) for c in lch_to_srgb(*p["crumb_LCh"]))
        ext = float(np.max(hi - lo) * 0.58)
        zc = float(cen[2])
        return {
            "build": {"h": 0.3e-3},
            "textures": [("skin.png", self.point_uv, self.skin_albedo, 2048, 1024)],
            "uv_fns": {"crust": self.uv_of},
            "materials": {
                "Crust": dict(diffuse_reflection_color_image="tex:skin.png", specular_reflection_roughness=0.25,
                              specular_reflection_weight=0.65, coat_weight=0.45, coat_roughness=0.15),
                "Crumb": dict(diffuse_reflection_color=crumb, specular_reflection_roughness=0.6,
                              subsurface_weight=0.1, subsurface_scale=0.002, subsurface_scattering_color=crumb),
            },
            "section_material": {"specular_reflection_roughness": 0.5, "specular_reflection_weight": 0.35},
            "root": {"freshness": "fresh", "cut_state": "whole", "temperature": "room", "filled": "no"},
            "friction": p["friction_crust"], "restitution": 0.1,
            "cuts": [
                {"name": "split", "origin": [0, 0, zc], "e1": [1, 0, 0], "e2": [0, 1, 0], "ext": ext,
                 "states": {"top_half": ("pos", {"cut_state": "split_open"}),
                            "bottom_half": ("neg", {"cut_state": "split_open"})}},
                {"name": "across", "origin": [float(cen[0]), 0, 0], "e1": [0, 1, 0], "e2": [0, 0, 1], "ext": ext,
                 "states": {"half_a": ("pos", {"cut_state": "halved_across"}),
                            "half_b": ("neg", {"cut_state": "halved_across"})}},
            ],
            "beliefs": lambda mp: croissant_beliefs(self, mp),
            "truth": lambda: croissant_truth(self),
        }
