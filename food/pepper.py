"""Bell pepper twin (Capsicum annuum, blocky 'Paprika'): a hollow berry.

Anatomy (docs/wave3/PEPPER.md for the sources): the PERICARP WALL (cuticle ~10 um, exocarp, thick mesocarp, endocarp;
7.5 mm red, he22) round an AIR CAVITY; 2-4 LOBES at the sunken blossom end (one per locule); thin white RIBS (septa)
run down the inner wall between the lobes; the PLACENTA hangs from the stem end with the SEEDS (38-142); the STEM
(peduncle) with the CALYX on the shoulders.  After charring the thin SKIN slips off.

Geometry: an outline of revolution (shoulders at the stem, sunken blossom end) with lobe modulation; wall = outline
minus its inward offset; ribs are angular sectors along the inner wall; the placenta is a revolved bulb under the
stem; seeds are a point set (painted on cut faces); the cavity is air (label 0).

Partonomy (USD paths below /Food/Parts):
  Pericarp/{Skin, Wall}, Ribs/Rib_k, Placenta, Seeds (points), Stem/{Peduncle, Calyx}
"""

import numpy as np
from scipy.spatial import cKDTree

from . import meshlib as ML
from . import params as PR
from .geometry import Body, Meridian, resample, smooth_meridian
from .partgraph import Part
from .shapes import angle_in, region_contains, revolve_polygon, smoothstep
from .texture import ValueNoise3, lch_to_srgb

TISSUES = [
    {"name": "skin", "density": 1000.0, "colour": (0.70, 0.08, 0.06)},
    {"name": "wall", "density": 1000.0, "colour": (0.80, 0.12, 0.08)},
    {"name": "rib", "density": 980.0, "colour": (0.95, 0.93, 0.85)},
    {"name": "placenta", "density": 980.0, "colour": (0.95, 0.93, 0.82)},
    {"name": "seed", "density": 1050.0, "colour": (0.93, 0.88, 0.70)},
]
TI = {t["name"]: i + 1 for i, t in enumerate(TISSUES)}


class Pepper(Body):
    tissues = TISSUES
    kind = "pepper"

    def __init__(self, seed=0, overrides=None):
        p = PR.sample("pepper", seed)
        p.update(overrides or {})
        rng = np.random.default_rng(p["_rng_state"])
        self.p, self.seed, self.rng = p, seed, rng
        for t in TISSUES:
            if t["name"] != "seed":
                t["density"] = p["wall_density"]
        R, H = p["diameter"] * 5e-4, p["length"] * 1e-3
        self.R, self.H = R, H
        ctrl = np.array([[0.0, H / 2 - 7e-3], [0.25 * R, H / 2 - 1e-3], [0.7 * R, H / 2 - 3e-3], [0.97 * R, H / 2 - 16e-3],
                         [1.0 * R, 0.0], [0.97 * R, -H / 2 + 16e-3], [0.75 * R, -H / 2 + 2e-3], [0.35 * R, -H / 2],
                         [0.0, -H / 2 + 7e-3]])
        rho, z = smooth_meridian(ctrl, 600)
        self.n_lobes = int(p["lobes"])
        self.phi = rng.uniform(0, 2 * np.pi)
        lob = p["lobe_depth"]

        def mod(th, zz):
            w = 0.4 + 0.6 * smoothstep(0.2 * H, -0.5 * H, zz)          # lobes strongest at the blossom end
            return 1 + lob * w * np.cos(self.n_lobes * (th - self.phi))
        super().__init__(Meridian(rho, z), mod)
        self.t_wall = p["wall_thickness"] * 1e-3
        self.t_skin = 0.25e-3
        self.m_skin = self.mer.offset(self.t_skin)
        self.m_in = self.mer.offset(self.t_wall)
        self.rib_angles = self.phi + np.pi / self.n_lobes + np.arange(self.n_lobes) * 2 * np.pi / self.n_lobes
        self._seeds = None

    # ---------------------------------------------------------------- regions
    def r_in(self, z):
        return np.interp(z, self.m_in.z[::-1], self.m_in.rho[::-1])

    def z_in_top(self):
        return float(self.m_in.z.max())

    def placenta_poly(self):
        zt = self.z_in_top() + 0.5e-3
        h = self.p["placenta_height"] * 1e-3
        r = self.p["placenta_diameter"] * 5e-4
        a = np.linspace(0, np.pi, 30)
        bulb = np.stack([r * np.sin(a) ** 0.8, zt - h * (1 - np.cos(a)) / 2], 1)
        return np.vstack([[[0.0, zt + 1.5e-3]], [[0.35 * r, zt + 1.5e-3]], bulb[1:-1], [[0.0, zt - h]]])

    def rib_poly(self):
        zt = self.z_in_top() - 4e-3
        zb = float(self.m_in.z.min()) + 12e-3
        zz = np.linspace(zt, zb, 60)
        ro = self.r_in(zz) + 0.3e-3
        depth = self.p["rib_depth"] * 1e-3 * (0.4 + 0.6 * smoothstep(zb, zt, zz))
        ri = np.maximum(ro - depth, 2e-3)
        return np.vstack([np.stack([ro, zz], 1), np.stack([ri[::-1], zz[::-1]], 1)])

    def rib_sector(self, k):
        a = self.rib_angles[k]
        w = self.p["rib_thickness"] * 1e-3 / (0.8 * self.R)
        return a - w / 2, a + w / 2

    def label(self, P):
        P = np.asarray(P, float)
        rn, th, z = self.norm_coords(P)
        out = np.zeros(P.shape[:-1], np.int16)
        ins = self.mer.contains(rn, z)
        wall = ins & ~self.m_in.contains(rn, z)
        out[wall] = TI["wall"]
        out[ins & ~self.m_skin.contains(rn, z)] = TI["skin"]
        rp = self.rib_poly()
        for k in range(self.n_lobes):
            a0, a1 = self.rib_sector(k)
            out[ins & ~wall & angle_in(th, a0, a1) & region_contains(rp, rn, z)] = TI["rib"]
        pl = ins & region_contains(self.placenta_poly(), rn, z) & ~wall
        out[pl] = TI["placenta"]
        C, r = self.seeds()
        sel = np.nonzero(ins & ~wall)
        if len(sel[0]):
            d, i = cKDTree(C).query(P[sel])
            hit = d < r[i]
            out[tuple(a[hit] for a in sel)] = TI["seed"]
        return out

    def tissue_of(self, rn, th, z, P):
        return self.label(P)

    def bbox(self):
        r = self.R * (1 + self.p["lobe_depth"]) * 1.05
        return np.array([-r, -r, -self.H / 2 - 1e-3]), np.array([r, r, self.H / 2 + self.p["stem_length"] * 1e-3 + 3e-3])

    def seeds(self):
        """Seeds on the placenta surface (flat discs ~ register seed_diameter), a few loose in the cavity."""
        if self._seeds is None:
            rng = np.random.default_rng(self.seed + 5)
            n = int(self.p["seeds"])
            pp = self.placenta_poly()
            zt, zb = pp[:, 1].max(), pp[:, 1].min()
            C = []
            for _ in range(n):
                z = rng.uniform(zb + 2e-3, zt - 2e-3)
                prof = pp[2:-1]
                r = np.interp(z, prof[::-1, 1], prof[::-1, 0]) + 1.2e-3
                a = rng.uniform(0, 2 * np.pi)
                C.append([r * np.cos(a), r * np.sin(a), z])
            self._seeds = (np.array(C), np.full(n, self.p["seed_diameter"] * 5e-4))
        return self._seeds

    # ---------------------------------------------------------------- meshes
    def modulate(self, m):
        v = np.asarray(m.vertices).copy()
        k = self.mod(np.arctan2(v[:, 1], v[:, 0]), v[:, 2])
        v[:, :2] *= k[:, None]
        m.vertices = v
        return m

    def surf(self, mer, n=144):
        r, z = resample(mer.rho, mer.z, 260)
        r[0] = r[-1] = 0
        return ML.surface_of_revolution(Meridian(r, z), n, modulation=self.mod)

    def calyx_mesh(self):
        zt = float(self.mer.z.max())
        a = np.linspace(0, np.pi, 12)
        poly = np.stack([9e-3 * np.sin(a) ** 0.5 + 0.1e-3, zt + 1.2e-3 * np.cos(a) - 0.2e-3], 1)
        return revolve_polygon(np.vstack([[[0, zt + 1.0e-3]], poly[1:-1], [[0, zt - 1.4e-3]]]), 40)

    def stem_mesh(self):
        zt = float(self.mer.z.max())
        L = self.p["stem_length"] * 1e-3
        path = np.stack([np.linspace(0, 2e-3, 8), np.zeros(8), zt + np.linspace(0.5e-3, L, 8)], 1)
        return ML.tube(path, np.linspace(3.5e-3, 3.0e-3, 8), 12)

    def parts(self):
        cache = {}

        def c(k, fn):
            def g():
                if k not in cache:
                    cache[k] = fn()
                return cache[k]
            return g
        so, ss, si = (c(k, lambda m=m: self.surf(m)) for k, m in (("o", self.mer), ("s", self.m_skin),
                                                                   ("i", self.m_in)))
        R = {t["name"]: t["density"] for t in TISSUES}
        P = [Part("pericarp", "Pericarp", "food:Pericarp", None, observability="visible",
                  doc="Fruit wall round the air cavity.", attrs={"wall_mm": self.p["wall_thickness"]}),
             Part("skin", "Pericarp/Skin", "food:PepperSkin", ("mesh", lambda: ML.hollow(so(), ss())), "Skin",
                  "visible", "Cuticle + exocarp: glossy; blisters and slips off after charring.", R["skin"],
                  interior=False),
             Part("wall", "Pericarp/Wall", "food:PepperWall", ("mesh", lambda: ML.hollow(ss(), si())), "Wall",
                  "visible after cutting", "Crisp mesocarp: the strips and rings.", R["wall"]),
             Part("ribs", "Ribs", "food:SeptumGroup", None, observability="visible after cutting",
                  doc="White septa ('veins') between the lobes; trimmed off.")]
        rp = self.rib_poly()
        for k in range(self.n_lobes):
            a0, a1 = self.rib_sector(k)
            P.append(Part(f"rib_{k + 1}", f"Ribs/Rib_{k + 1}", "food:Septum",
                          ("mesh", lambda a0=a0, a1=a1: self.modulate(revolve_polygon(rp, 4, a0, a1))), "Rib",
                          "visible after cutting", "", R["rib"]))
        P += [Part("placenta", "Placenta", "food:Placenta",
                   ("mesh", lambda: self.modulate(revolve_polygon(self.placenta_poly(), 48))), "Rib",
                   "visible after cutting", "Spongy white placenta under the stem carrying the seeds.", R["placenta"]),
              Part("seeds", "Seeds", "food:PepperSeed", ("points", lambda: (self.seeds()[0], 2 * self.seeds()[1])),
                   None, "visible after cutting", "Flat cream seeds; scatter when the core is pulled.",
                   attrs={"count": int(self.p["seeds"])}),
              Part("stem", "Stem", "food:StemGroup", None, observability="visible"),
              Part("peduncle", "Stem/Peduncle", "food:Peduncle", ("mesh", self.stem_mesh), "Stem", "visible",
                   "Cut stalk.", 900.0, interior=False),
              Part("calyx", "Stem/Calyx", "food:Calyx", ("mesh", self.calyx_mesh), "Stem", "visible",
                   "Calyx on the shoulders.", 900.0, interior=False)]
        return P

    # ---------------------------------------------------------------- appearance
    def colour(self):
        return lch_to_srgb(*self.p[f"{self.p['colour']}_LCh"])

    def point_uv(self, U, V):
        k = np.linspace(0, 1, len(self.mer.rho))
        t = 1 - np.asarray(V)
        rho, z = np.interp(t, k, self.mer.rho), np.interp(t, k, self.mer.z)
        th = 2 * np.pi * np.asarray(U)
        m = self.mod(th, z)
        return np.stack([m * rho * np.cos(th), m * rho * np.sin(th), z], -1)

    def uv_of(self, P):
        rn, th, z = self.norm_coords(np.asarray(P, float))
        k = np.linspace(0, 1, len(self.mer.rho))
        _, i = cKDTree(np.stack([self.mer.rho, self.mer.z], 1)).query(np.stack([rn, z], 1))
        return np.stack([np.mod(th, 2 * np.pi) / (2 * np.pi), 1 - k[i]], 1)

    def skin_albedo(self, P, seed):
        P = np.asarray(P, float)
        vn = ValueNoise3(seed + 401)
        col = np.broadcast_to(self.colour(), P.shape).copy()
        col = col * (1 + 0.05 * vn.fbm(P, 6e-3, 3))[..., None]
        th = np.arctan2(P[..., 1], P[..., 0])
        crease = np.exp(-(np.abs(((th[..., None] - self.rib_angles) + np.pi) % (2 * np.pi) - np.pi) / 0.08) ** 2
                        ).max(-1) * smoothstep(0.1 * self.H, -0.4 * self.H, P[..., 2])
        col = col * (1 - 0.25 * crease[..., None])
        return np.clip(col, 0, 1)

    def section_rgb(self, lab, P, seed=0):
        rgb = np.zeros(lab.shape + (3,)) + 0.86
        vn = ValueNoise3(seed + 411)
        for i, t in enumerate(TISSUES):
            rgb[lab == i + 1] = t["colour"]
        c = self.colour()
        inner = lch_to_srgb(*self.p["inner_LCh"][self.p["colour"]]) if isinstance(self.p.get("inner_LCh"), dict) \
            else np.clip(c * 1.15, 0, 1)
        m = lab == TI["wall"]
        if m.any():
            rn, th, z = self.norm_coords(P[m])
            R_o = np.interp(z, self.mer.z[::-1], self.mer.rho[::-1])
            f = np.clip((R_o - rn) / self.t_wall, 0, 1)[:, None]
            rgb[m] = (c * (1 - f) + inner * f) * (1 + 0.03 * vn.fbm(P[m], 0.5e-3, 2))[:, None]
        rgb[lab == TI["skin"]] = c * 0.85
        return np.clip(rgb, 0, 1)

    # ---------------------------------------------------------------- asset declaration
    def asset_spec(self):
        from .pepper_models import pepper_beliefs, pepper_truth
        p = self.p
        c = tuple(float(x) for x in self.colour())
        ext = float(max(self.R, self.H / 2) * 1.25)
        a = self.rib_angles[0] + np.pi / self.n_lobes                 # lengthwise plane through two lobes
        return {
            "build": {"h": 0.3e-3},
            "textures": [("skin.png", self.point_uv, self.skin_albedo, 2048, 1024)],
            "uv_fns": {"skin": self.uv_of},
            "materials": {
                "Skin": dict(diffuse_reflection_color_image="tex:skin.png", specular_reflection_roughness=0.1,
                             specular_reflection_weight=0.85, coat_weight=0.5, coat_roughness=0.06),
                "Wall": dict(diffuse_reflection_color=c, specular_reflection_roughness=0.2, subsurface_weight=0.3,
                             subsurface_scale=0.003, subsurface_scattering_color=c),
                "Rib": dict(diffuse_reflection_color=(0.95, 0.93, 0.84), specular_reflection_roughness=0.5),
                "Stem": dict(diffuse_reflection_color=(0.25, 0.35, 0.12), specular_reflection_roughness=0.55),
            },
            "section_material": {"specular_reflection_roughness": 0.2, "specular_reflection_weight": 0.6},
            "root": {"colour": p["colour"], "cut_state": "whole", "cored": "no", "skin": "on"},
            "friction": p["friction_skin_rubber"], "restitution": 0.25,
            "cuts": [
                {"name": "lengthwise", "origin": [0, 0, 0], "e1": [float(np.cos(a)), float(np.sin(a)), 0.0],
                 "e2": [0, 0, 1], "ext": ext,
                 "states": {"half_a": ("pos", {"cut_state": "halved"}), "half_b": ("neg", {"cut_state": "halved"})}},
                {"name": "across", "origin": [0, 0, -0.1 * self.H], "e1": [1, 0, 0], "e2": [0, 1, 0], "ext": ext,
                 "states": {"top_piece": ("pos", {"cut_state": "cut_across"}),
                            "bottom_piece": ("neg", {"cut_state": "cut_across"})}},
            ],
            "beliefs": lambda mp: pepper_beliefs(self, mp),
            "truth": lambda: pepper_truth(self),
        }
