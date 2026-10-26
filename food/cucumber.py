"""Greenhouse cucumber twin (Cucumis sativus, 'Schlangengurke'): a pepo of three fused carpels.

Anatomy (docs/wave3/CUCUMBER.md for the sources): the SKIN (exocarp: cuticle, epidermis, a green collenchymatous
hypodermis), the firm MESOCARP (the flesh of a slice), and the SEED CAVITY -- the soft endocarp / placental tissue of
three carpels, each with its PLACENTA carrying two rows of immature SEEDS -- round a small central CORE.  A slightly
narrower STEM END (neck) and the rounded BLOSSOM END with the flower scar.  Long greenhouse types are smooth with
faint longitudinal ribs.

Geometry: a straight body of revolution with faint ribs (modulation), bent onto an arc (food.shapes.Bend); flesh and
cavity boundaries are radii as functions of z; the three placentae are 120-degree sectors of the cavity
(food.shapes.revolve_polygon); seeds are a point set painted on cut faces.

Partonomy (USD paths below /Food/Parts):
  Skin, Flesh (mesocarp), SeedCavity/{Placenta_1..3, Core}, SeedCavity/Seeds (points), StemEnd, BlossomScar
"""

import numpy as np
from scipy.spatial import cKDTree

from . import meshlib as ML
from . import params as PR
from .geometry import Body, Meridian, smooth_meridian
from .partgraph import Part
from .shapes import Bend, angle_in, revolve_polygon, smoothstep
from .texture import ValueNoise3, lch_to_srgb

TISSUES = [
    {"name": "skin", "density": 1000.0, "colour": (0.15, 0.30, 0.10)},
    {"name": "flesh", "density": 960.0, "colour": (0.85, 0.92, 0.70)},
    {"name": "placenta", "density": 960.0, "colour": (0.80, 0.88, 0.65)},
    {"name": "core", "density": 960.0, "colour": (0.82, 0.90, 0.66)},
    {"name": "seed", "density": 1000.0, "colour": (0.92, 0.92, 0.80)},
]
TI = {t["name"]: i + 1 for i, t in enumerate(TISSUES)}


class Cucumber(Body):
    tissues = TISSUES
    kind = "cucumber"

    def __init__(self, seed=0, overrides=None):
        p = PR.sample("cucumber", seed)
        p.update(overrides or {})
        rng = np.random.default_rng(p["_rng_state"])
        self.p, self.seed, self.rng = p, seed, rng
        for t in TISSUES:
            if t["name"] != "seed":
                t["density"] = p["density"]
        L, D = p["length"] * 1e-3, p["diameter"] * 1e-3
        self.L, self.D = L, D
        s = np.array([0.0, 0.01, 0.04, 0.10, 0.5, 0.86, 0.93, 0.975, 1.0])
        r = np.array([0.0, 0.45, 0.82, 0.97, 1.0, 0.96, p["neck_ratio"], p["neck_ratio"] * 0.85, 0.0]) * D / 2
        z = (s - 0.5) * L
        rho, zz = smooth_meridian(np.stack([r[::-1], z[::-1]], 1), 700)
        self.phi = rng.uniform(0, 2 * np.pi)
        nr, rib = int(p["ribs"]), p["rib_depth"]

        def mod(th, zc):
            return 1 + rib * np.cos(nr * (th - self.phi))
        super().__init__(Meridian(rho, zz), mod)
        self.bend = Bend(p["bend_radius"] * 1e-3)
        self.t_skin = p["skin_thickness"] * 1e-3
        self.cav_frac = p["cavity_fraction"]
        self.r_core = p["core_diameter"] * 5e-4
        self.seg_phi = self.phi + np.pi / 3
        self._seeds = None

    # ---------------------------------------------------------------- radii (straight frame)
    def r_out(self, z):
        return np.interp(z, self.mer.z[::-1], self.mer.rho[::-1])

    def r_cav(self, z):
        q = np.clip((z - self.mer.zmin) / self.L, 0, 1)
        taper = smoothstep(0.0, 0.12, q) * (1 - smoothstep(0.82, 0.95, q))
        return self.cav_frac * self.r_out(z) * taper

    def cav_poly(self, inset=0.0, r_in=None):
        zz = np.linspace(self.mer.zmin, self.mer.zmax, 600)
        rc = self.r_cav(zz) - inset
        ok = rc > (r_in if r_in is not None else 0.0) + 0.6e-3
        zz, rc = zz[ok], rc[ok]
        r0 = r_in if r_in is not None else 0.0
        return np.vstack([[[r0, zz[-1]]], np.stack([rc[::-1], zz[::-1]], 1), [[r0, zz[0]]]])

    def label(self, P):
        S = self.bend.unbend(np.asarray(P, float))
        rn, th, z = self.norm_coords(S)
        out = np.zeros(S.shape[:-1], np.int16)
        ins = self.mer.contains(rn, z)
        if not ins.any():
            return out
        rho = np.hypot(S[..., 0], S[..., 1])
        lab = np.full(S.shape[:-1], TI["flesh"], np.int16)
        lab[~self.mer.offset(self.t_skin).contains(rn, z)] = TI["skin"]
        cav = rho < self.r_cav(z)
        lab[cav] = TI["placenta"]
        lab[cav & (rho < self.r_core)] = TI["core"]
        out[ins] = lab[ins]
        C, r = self.seeds()
        sel = np.nonzero(out == TI["placenta"])
        if len(sel[0]) and len(C):
            d, i = cKDTree(C).query(S[sel])
            hit = d < r[i]
            out[tuple(a[hit] for a in sel)] = TI["seed"]
        return out

    def tissue_of(self, rn, th, z, P):
        return self.label(P)

    def inside(self, P):
        return self.label(P) > 0

    def bbox(self):
        z = np.linspace(self.mer.zmin, self.mer.zmax, 12)
        pts = self.bend.bend(np.array([[sx * self.D * 0.6, sy * self.D * 0.6, zz] for sx in (-1, 1)
                                       for sy in (-1, 1) for zz in z]))
        return pts.min(0) - 1e-3, pts.max(0) + 1e-3

    def seeds(self):
        """Immature seeds: two rows per placenta, flat ellipsoids (straight frame)."""
        if self._seeds is None:
            rng = np.random.default_rng(self.seed + 5)
            n = int(self.p["seeds_per_row"])
            C, R = [], []
            for j in range(3):
                for side in (-1, 1):
                    a = self.seg_phi + j * 2 * np.pi / 3 + side * 0.32
                    for zz in np.linspace(self.mer.zmin + 0.15 * self.L, self.mer.zmax - 0.2 * self.L, n):
                        rr = 0.55 * self.r_cav(zz)
                        C.append([rr * np.cos(a), rr * np.sin(a), zz + rng.normal(0, 1e-3)])
                        R.append(self.p["seed_length"] * 5e-4)
            self._seeds = (np.array(C), np.array(R))
        return self._seeds

    # ---------------------------------------------------------------- meshes
    def surf(self, mer, n_theta=96, n_mer=420):
        from .geometry import resample
        r, z = resample(mer.rho, mer.z, n_mer)
        r[0] = r[-1] = 0
        return self.bend.bend_mesh(ML.surface_of_revolution(Meridian(r, z), n_theta, modulation=self.mod))

    def sector_mesh(self, poly, a0, a1, n=24):
        return self.bend.bend_mesh(revolve_polygon(_densify(poly), n, a0, a1))

    def parts(self):
        cache = {}

        def c(name, fn):
            def g():
                if name not in cache:
                    cache[name] = fn()
                return cache[name]
            return g
        out_s = c("o", lambda: self.surf(self.mer))
        in_s = c("i", lambda: self.surf(self.mer.offset(self.t_skin)))
        cav = c("cv", lambda: self.bend.bend_mesh(revolve_polygon(_densify(self.cav_poly()), 72)))
        R = {t["name"]: t["density"] for t in TISSUES}
        P = [Part("skin", "Skin", "food:CucumberSkin", ("mesh", lambda: ML.hollow(out_s(), in_s())), "Skin",
                  "visible", "Exocarp: cuticle, epidermis, green hypodermis.", R["skin"], interior=False,
                  attrs={"thickness_mm": self.p["skin_thickness"]}),
             Part("flesh", "Flesh", "food:Mesocarp", ("mesh", lambda: ML.hollow(in_s(), cav())), "Flesh",
                  "visible after cutting", "Firm mesocarp: the crisp ring of a slice.", R["flesh"]),
             Part("seed_cavity", "SeedCavity", "food:SeedCavity", None, observability="visible after cutting",
                  doc="Soft endocarp / placental tissue of three carpels with the seeds; scooped out with a spoon.",
                  attrs={"diameter_fraction": self.cav_frac})]
        cp = self.cav_poly(inset=0.15e-3, r_in=self.r_core + 0.15e-3)
        for k in range(3):
            a0 = self.seg_phi + k * 2 * np.pi / 3 + 0.004
            a1 = a0 + 2 * np.pi / 3 - 0.008
            P.append(Part(f"placenta_{k + 1}", f"SeedCavity/Placenta_{k + 1}", "food:Placenta",
                          ("mesh", lambda a0=a0, a1=a1: self.sector_mesh(cp, a0, a1)), "Cavity",
                          "visible after cutting", "One carpel's placenta with two seed rows.", R["placenta"]))
        P += [Part("core", "SeedCavity/Core", "food:CentralCore",
                   ("mesh", lambda: self.bend.bend_mesh(revolve_polygon(_densify(_core_poly(self)), 24))),
                   "Cavity", "visible after cutting", "", R["core"]),
              Part("seeds", "SeedCavity/Seeds", "food:CucumberSeed",
                   ("points", lambda: (self.bend.bend(self.seeds()[0]), 2 * self.seeds()[1])), None,
                   "visible after cutting", "Immature, soft seeds in two rows per placenta (painted on cut faces).")]
        return P

    # ---------------------------------------------------------------- appearance
    def point_uv(self, U, V):
        k = np.linspace(0, 1, len(self.mer.rho))
        t = 1 - np.asarray(V)
        rho, z = np.interp(t, k, self.mer.rho), np.interp(t, k, self.mer.z)
        th = 2 * np.pi * np.asarray(U)
        m = self.mod(th, z)
        return self.bend.bend(np.stack([m * rho * np.cos(th), m * rho * np.sin(th), z], -1))

    def uv_of(self, P):
        S = self.bend.unbend(np.asarray(P, float))
        rn, th, z = self.norm_coords(S)
        k = np.linspace(0, 1, len(self.mer.rho))
        _, i = cKDTree(np.stack([self.mer.rho, self.mer.z], 1)).query(np.stack([rn, z], 1))
        return np.stack([np.mod(th, 2 * np.pi) / (2 * np.pi), 1 - k[i]], 1)

    def skin_albedo(self, P, seed):
        P = np.asarray(P, float)
        S = self.bend.unbend(P)
        vn = ValueNoise3(seed + 141)
        base = lch_to_srgb(*self.p["skin_LCh"])
        th = np.arctan2(S[..., 1], S[..., 0])
        stripe = 0.5 + 0.5 * vn.fbm(np.stack([np.cos(th) * 0.02, np.sin(th) * 0.02, S[..., 2] * 0.15], -1), 0.004, 3)
        col = base * (0.85 + 0.25 * stripe[..., None])
        col = col * (1 + 0.06 * vn.fbm(P, 2e-3, 3))[..., None]
        q = (S[..., 2] - self.mer.zmin) / self.L
        end = np.maximum(1 - smoothstep(0.0, 0.02, q), smoothstep(0.98, 1.0, q))
        col = col * (1 - 0.5 * end[..., None]) + lch_to_srgb(40, 20, 100) * 0.5 * end[..., None]
        return np.clip(col, 0, 1)

    def section_rgb(self, lab, P, seed=0):
        rgb = np.zeros(lab.shape + (3,)) + 0.86
        vn = ValueNoise3(seed + 151)
        for i, t in enumerate(TISSUES):
            rgb[lab == i + 1] = t["colour"]
        fl = lch_to_srgb(*self.p["flesh_LCh"])
        m = lab == TI["flesh"]
        if m.any():
            S = self.bend.unbend(P[m])
            rho = np.hypot(S[:, 0], S[:, 1])
            q = np.clip(rho / (self.r_out(S[:, 2]) + 1e-9), 0, 1)
            rim = smoothstep(0.75, 1.0, q)                     # greener towards the skin
            rgb[m] = fl * (1 - 0.25 * rim)[:, None] + lch_to_srgb(70, 40, 115) * 0.25 * rim[:, None]
            rgb[m] *= (1 + 0.025 * vn.fbm(P[m], 0.6e-3, 2))[:, None]
        cav = lch_to_srgb(self.p["flesh_LCh"][0] - 4, self.p["flesh_LCh"][1] - 8, self.p["flesh_LCh"][2] + 5)
        m = np.isin(lab, [TI["placenta"], TI["core"]])
        rgb[m] = cav * (1 + 0.05 * vn.fbm(P[m], 0.4e-3, 3))[:, None]
        rgb[lab == TI["skin"]] = lch_to_srgb(*self.p["skin_LCh"])
        rgb[lab == TI["seed"]] = (0.93, 0.93, 0.82)
        return np.clip(rgb, 0, 1)

    # ---------------------------------------------------------------- asset declaration
    def collision_groups(self, V):
        S = self.bend.unbend(V)
        edges = np.linspace(S[:, 2].min(), S[:, 2].max() + 1e-9, 7)
        return [V[(S[:, 2] >= edges[i] - 2e-3) & (S[:, 2] <= edges[i + 1] + 2e-3)] for i in range(6)]

    def asset_spec(self):
        from .cucumber_models import cucumber_beliefs, cucumber_truth
        p = self.p
        lo, hi = self.bbox()
        cen = (lo + hi) / 2
        fl = tuple(float(c) for c in lch_to_srgb(*p["flesh_LCh"]))
        # rounds: slices across the middle part (the ends are cut off first)
        t = p["round_thickness"] * 1e-3
        zs = np.arange(-0.25 * self.L, 0.25 * self.L + 1e-9, t)
        return {
            "build": {"h": 0.35e-3},
            "textures": [("skin.png", self.point_uv, self.skin_albedo, 2048, 1024)],
            "uv_fns": {"skin": self.uv_of},
            "materials": {
                "Skin": dict(diffuse_reflection_color_image="tex:skin.png", specular_reflection_roughness=0.3,
                             specular_reflection_weight=0.55, coat_weight=0.2, coat_roughness=0.2),
                "Flesh": dict(diffuse_reflection_color=fl, specular_reflection_roughness=0.2,
                              specular_reflection_weight=0.7, subsurface_weight=0.45, subsurface_scale=0.004,
                              subsurface_scattering_color=fl),
                "Cavity": dict(diffuse_reflection_color=tuple(c * 0.97 for c in fl),
                               specular_reflection_roughness=0.12, specular_reflection_weight=0.8,
                               subsurface_weight=0.5, subsurface_scale=0.004, subsurface_scattering_color=fl),
            },
            "section_material": {"specular_reflection_roughness": 0.15, "specular_reflection_weight": 0.75,
                                 "subsurface_weight": 0.35, "subsurface_scale": 0.003,
                                 "subsurface_scattering_color": fl},
            "root": {"peel_state": "on", "cut_state": "whole", "freshness": "crisp"},
            "friction": p["friction_skin_rubber"], "restitution": 0.15,
            "collision_fn": self.collision_groups,
            "cuts": [{"name": "length", "origin": [cen[0], 0, cen[2]], "e1": [1, 0, 0], "e2": [0, 0, 1],
                      "ext": float(np.max(hi - lo) * 0.56),
                      "states": {"half_a": ("pos", {"cut_state": "halved_lengthwise"}),
                                 "half_b": ("neg", {"cut_state": "halved_lengthwise"})}}],
            "slicings": [{"name": "rounds", "normal": [0, 0, 1], "offsets": list(map(float, zs)),
                          "e1": [1, 0, 0], "e2": [0, 1, 0], "ext": float(self.D * 0.7), "res": 1024,
                          "section_at": 0.0, "group": "Rounds", "piece": "Round", "piece_class": "food:CucumberRound",
                          "end_class": "food:CucumberEnd", "root": {"cut_state": "rounds"}}],
            "beliefs": lambda mp: cucumber_beliefs(self, mp),
            "truth": lambda: cucumber_truth(self),
        }


def _core_poly(cu):
    zz = np.linspace(cu.mer.zmin, cu.mer.zmax, 600)
    rc = cu.r_cav(zz)
    ok = rc > cu.r_core + 0.8e-3
    z0, z1 = zz[ok][0], zz[ok][-1]
    return np.array([[0, z1], [cu.r_core, z1], [cu.r_core, z0], [0, z0]])


def _densify(poly, max_edge=2.0e-3):
    """Subdivide long polygon edges (the long straight sides along z) so revolved meshes bend correctly."""
    P = np.asarray(poly, float)
    out = []
    for a, b in zip(P, np.roll(P, -1, axis=0)):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / max_edge)))
        for k in range(n):
            out.append(a + (b - a) * k / n)
    return np.array(out)
