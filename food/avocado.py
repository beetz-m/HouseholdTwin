"""Hass avocado twin: anatomy after Cummings & Schroeder 1942 (cs42), Wikipedia (Avocado: a single-seeded berry),
Ashton et al. 2006 (colours), USPTO PP14239 (Hass skin, seed) -- docs/ANATOMY.md.

Frame: +z from the blossom end (bottom, wide) to the stem end (top, narrow), origin at mid-length.  Ripeness is a
continuous hidden stage on the Avocados Australia 1..5 scale; it sets skin colour, firmness and the FEM material.

Partonomy (USD paths below /Food/Parts; every prim is an SDT node):
  Exocarp/{Cuticle, Epidermis, Hypodermis, SkinParenchyma, StoneCellLayer}, Exocarp/Lenticels (points)
  StemButton                          clipped pedicel remnant (~3 mm)
  Mesocarp/{OuterDarkGreenBand, MiddlePaleGreenBand, InnerYellowBand}   three equal-thickness bands (ashton2006)
  Mesocarp/VascularStrands/Strand_1..6                                   pedicel -> around the seed -> apex (cs42)
  Endocarp                            thin layer that may stay on the seed coat
  Seed/SeedCoat/{OuterTesta, InnerTesta}, Seed/Cotyledon_1, Seed/Cotyledon_2,
  Seed/EmbryonicAxis/{Plumule, Hypocotyl, Radicle}
"""

import numpy as np
from scipy.spatial import cKDTree

from . import meshlib as ML
from . import params as PR
from .geometry import Body, Meridian, resample, revolve
from .partgraph import Part
from .texture import ValueNoise3, lch_to_srgb, sample_surface, worley

TISSUES = [
    {"name": "cuticle", "density": 1050.0, "colour": (0.10, 0.12, 0.07)},
    {"name": "epidermis", "density": 1050.0, "colour": (0.12, 0.14, 0.08)},
    {"name": "hypodermis", "density": 1050.0, "colour": (0.16, 0.20, 0.09)},
    {"name": "skin_parenchyma", "density": 1050.0, "colour": (0.22, 0.30, 0.12)},
    {"name": "stone_cell_layer", "density": 1100.0, "colour": (0.30, 0.38, 0.14)},
    {"name": "flesh_outer", "density": 990.0, "colour": (0.70, 0.74, 0.44)},
    {"name": "flesh_middle", "density": 990.0, "colour": (0.84, 0.83, 0.52)},
    {"name": "flesh_inner", "density": 990.0, "colour": (0.86, 0.82, 0.50)},
    {"name": "vascular_strand", "density": 1000.0, "colour": (0.74, 0.74, 0.40)},
    {"name": "endocarp", "density": 1050.0, "colour": (0.80, 0.70, 0.48)},
    {"name": "outer_testa", "density": 1100.0, "colour": (0.47, 0.29, 0.15)},
    {"name": "inner_testa", "density": 1100.0, "colour": (0.55, 0.36, 0.20)},
    {"name": "cotyledon", "density": 1100.0, "colour": (0.90, 0.82, 0.62)},
    {"name": "embryonic_axis", "density": 1100.0, "colour": (0.80, 0.70, 0.45)},
]
TI = {t["name"]: i + 1 for i, t in enumerate(TISSUES)}
STAGE_PEN = np.array([118.0, 46.0, 15.0, 8.3, 4.9])        # N, geometric centres of the chart's bands


def penetrometer_force(stage):
    s = np.clip(np.asarray(stage, float), 1, 5)
    return np.exp(np.interp(s, [1, 2, 3, 4, 5], np.log(STAGE_PEN)))


def flesh_E_MPa(stage):
    return 2.29 * (penetrometer_force(stage) / 118.0) ** 0.837


def skin_colour(stage):
    """Whole-fruit skin colour (ashton2006: L 40 -> 27, C 15 -> 3, h 125 -> 250); diffuse albedo assumed 0.6 x the
    chromameter reading, which includes the glossy cuticle's specular part (ILLUSTRATIVE)."""
    t = float(np.clip((float(stage) - 1) / 4, 0, 1))
    return 0.6 * lch_to_srgb(40 - 13 * t, 15 - 12 * t, 125 + 125 * t)


def flesh_band_colours(stage):
    t = float(np.clip((stage - 1) / 4, 0, 1))
    C = 43 - 8 * t
    return (lch_to_srgb(74 - 10 * t, C, 113), lch_to_srgb(83 - 8 * t, C, 105), lch_to_srgb(83 - 8 * t, C, 101))


class Avocado(Body):
    tissues = TISSUES
    kind = "avocado"
    SEED_TAPER = 0.05

    def __init__(self, seed=0, overrides=None, stage=None):
        p = PR.sample("avocado", seed)
        p.update(overrides or {})
        if stage is not None:
            p["ripeness_stage"] = float(stage)
        rng = np.random.default_rng(p["_rng_state"])
        self.p, self.seed, self.rng = p, seed, rng
        L, D = p["length_L"] * 1e-3, p["diameter_D"] * 1e-3
        x = np.linspace(-L / 2, L / 2, 2400)
        w = 0.13 * L
        y = D / 2 * np.sqrt(np.clip((L ** 2 - 4 * x ** 2) / (L ** 2 + 8 * w * x + 4 * w ** 2), 0, None))
        s = (x + L / 2) / L
        g = np.clip((s - 0.45) / 0.45, 0, 1)
        y = y * (1 - p["neck"] * g ** 2 * (3 - 2 * g))
        rho, z = resample(y[::-1], x[::-1], 600)
        self.L, self.D = L, D
        self.phi = rng.uniform(0, 2 * np.pi)
        a1 = rng.uniform(0.0, 0.03)

        def mod(th, zz):
            return 1 + a1 * np.cos(th - self.phi) * (0.5 + zz / L) + 0.015 * np.cos(2 * (th - self.phi))
        super().__init__(Meridian(rho, z), mod)
        # exocarp layers (cs42); thin-layer thicknesses ILLUSTRATIVE, total = Hass 1.6 mm
        t = p["skin_thickness"] * 1e-3
        self.t_layers = {"cuticle": 15e-6, "epidermis": 25e-6, "hypodermis": 80e-6, "stone_cell_layer": 0.20e-3}
        self.t_layers["skin_parenchyma"] = t - sum(self.t_layers.values())
        acc = 0.0
        self.m_layer = {}
        for n in ("cuticle", "epidermis", "hypodermis", "skin_parenchyma", "stone_cell_layer"):
            acc += self.t_layers[n]
            self.m_layer[n] = self.mer.offset(acc)          # inner face of each layer
        self.m_skin = self.m_layer["stone_cell_layer"]
        # seed (+ testa + endocarp), off-centre
        self.seed_L = p["seed_length"] * 1e-3
        self.seed_W = p["seed_width_ratio"] * self.seed_L
        off = p["seed_offset"] * 1e-3
        a = rng.uniform(0, 2 * np.pi)
        self.seed_c = np.array([off * np.cos(a), off * np.sin(a), -0.10 * L])
        self.t_testa_o, self.t_testa_i = 0.15e-3, 0.10e-3
        self.t_endo = PR.AVOCADO["endocarp_thickness"][1] * 1e-3
        self.split_angle = rng.uniform(0, np.pi)            # plane between the cotyledons
        self.strand_az = self.phi + np.arange(6) * np.pi / 3 + rng.normal(0, 0.12, 6)
        self.stem_len = p["stem_button_length"] * 1e-3
        self.pebble_seeds = None
        self._build_radial_tables()

    # ---------------------------------------------------------------- seed geometry
    def seed_e(self, P, grow=0.0):
        """Implicit tapered ellipsoid of the seed, grown outward by `grow` (testa/endocarp faces)."""
        d = np.asarray(P) - self.seed_c
        a, b = self.seed_L / 2 + grow, self.seed_W / 2 + grow
        zz = d[..., 2] / a
        return (d[..., 0] ** 2 + d[..., 1] ** 2) / (b * (1 - self.SEED_TAPER * zz)) ** 2 + zz ** 2

    def seed_mesh(self, grow=0.0, n=64):
        a, b = self.seed_L / 2 + grow, self.seed_W / 2 + grow
        return ML.ellipsoid(self.seed_c, (b, b, a), n_u=n, n_v=2 * n, taper=lambda zn: 1 - self.SEED_TAPER * zn)

    # ---------------------------------------------------------------- flesh bands: radial blend seed -> skin
    def _ray(self, dirs, inside, r_max=0.08, it=36):
        lo, hi = np.zeros(len(dirs)), np.full(len(dirs), r_max)
        for _ in range(it):
            mid = (lo + hi) / 2
            ok = inside(self.seed_c + dirs * mid[:, None])
            lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
        return lo

    def _build_radial_tables(self):
        d, _ = ML.uv_sphere(90, 180)
        self._dirs = d
        g_out = self.t_testa_o + self.t_testa_i + self.t_endo
        self._r_seed = self._ray(d, lambda P: self.seed_e(P, g_out) < 1)
        self._r_skin = self._ray(d, lambda P: self._inside_mer(P, self.m_skin))
        self._dtree = cKDTree(d)

    def _inside_mer(self, P, mer):
        rn, th, z = self.norm_coords(P)
        return mer.contains(rn, z)

    def band_fraction(self, P):
        """0 at the endocarp, 1 at the skin's inner face (stone-cell layer), along rays from the seed centre."""
        v = np.asarray(P) - self.seed_c
        r = np.linalg.norm(v, axis=-1)
        _, i = self._dtree.query(v / (r[..., None] + 1e-15))
        return (r - self._r_seed[i]) / (self._r_skin[i] - self._r_seed[i] + 1e-15)

    def band_mesh(self, f):
        def r_of(d):
            _, i = self._dtree.query(d)
            return self._r_seed[i] + f * (self._r_skin[i] - self._r_seed[i])
        return ML.radial_surface(self.seed_c, r_of, 90, 180)

    def strand_path(self, az, n=90):
        """One of the six strands: from the stem end, round the seed in the inner band, joining under the seed."""
        s = np.linspace(0.04, 0.97, n)
        pol = np.pi * s
        d = np.stack([np.sin(pol) * np.cos(az), np.sin(pol) * np.sin(az), np.cos(pol)], 1)
        _, i = self._dtree.query(d)
        f = 0.28 * np.sin(np.pi * s) ** 0.6
        r = self._r_seed[i] + f * (self._r_skin[i] - self._r_seed[i])
        p = self.seed_c + d * r[:, None]
        top = np.array([0, 0, self.mer.zmax - self.t_layers["cuticle"] - 1.5e-3])
        lead = np.linspace(top, p[0], 12)[:-1]
        return np.vstack([lead, p])

    def embryo_c(self):
        return self.seed_c + np.array([0, 0, self.seed_L / 2 - 4.5e-3])

    # ---------------------------------------------------------------- labels (same definitions as the meshes)
    def tissue_of(self, rn, th, z, P):
        out = np.full(len(rn), TI["flesh_middle"], np.int16)
        prev = np.ones(len(rn), bool)
        for n in ("cuticle", "epidermis", "hypodermis", "skin_parenchyma", "stone_cell_layer"):
            ins = self.m_layer[n].contains(rn, z)
            out[prev & ~ins] = TI[n]
            prev = ins
        flesh = prev
        f = self.band_fraction(P)
        out[flesh & (f > 2 / 3)] = TI["flesh_outer"]
        out[flesh & (f < 1 / 3)] = TI["flesh_inner"]
        g = self.t_testa_o + self.t_testa_i
        out[self.seed_e(P, g + self.t_endo) < 1] = TI["endocarp"]
        out[self.seed_e(P, g) < 1] = TI["outer_testa"]
        out[self.seed_e(P, self.t_testa_i) < 1] = TI["inner_testa"]
        cot = self.seed_e(P) < 1
        out[cot] = TI["cotyledon"]
        n_ = np.array([np.cos(self.split_angle), np.sin(self.split_angle), 0])
        out[cot & (np.abs((P - self.seed_c) @ n_) < 0.15e-3)] = TI["inner_testa"]
        out[cot & (np.linalg.norm(P - self.embryo_c(), axis=1) < 2.4e-3)] = TI["embryonic_axis"]
        pts = np.vstack([self.strand_path(a) for a in self.strand_az])
        near = flesh & (out != TI["endocarp"])
        if near.any():
            dd, _ = cKDTree(pts).query(P[near])
            out[np.nonzero(near)[0][dd < 0.32e-3]] = TI["vascular_strand"]
        return out

    # ---------------------------------------------------------------- meshes
    def pebble_displacement(self):
        rng = np.random.default_rng(self.seed + 5)
        cell = self.p["pebble_scale"] * 1e-3
        if self.pebble_seeds is None:
            v, f, _, _ = revolve(self.mer, 160, modulation=self.mod)
            area = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1).sum() / 2
            self.pebble_seeds = sample_surface(v, f, int(area / cell ** 2 * 1.1), rng)
        vn = ValueNoise3(self.seed + 6)

        def disp(T, tfrac, verts, normals):
            F1 = worley(verts, self.pebble_seeds, 1)
            bump = np.clip(1 - (F1 / (0.62 * cell)) ** 2, 0, 1)
            amp = 0.30e-3 * (0.7 + 0.6 * (0.5 + 0.5 * vn.fbm(verts, 8e-3, 2)))
            zfac = np.clip((self.mer.zmax - verts[:, 2]) / 8e-3, 0.3, 1)
            return amp * bump * zfac
        return disp

    def surf(self, mer, pebbled=False, n_mer=None, n_theta=None):
        n_mer, n_theta = n_mer or (260 if pebbled else 160), n_theta or (288 if pebbled else 128)
        r, z = resample(mer.rho, mer.z, n_mer)
        r[0] = r[-1] = 0
        return ML.surface_of_revolution(Meridian(r, z), n_theta, modulation=self.mod,
                                        displacement=self.pebble_displacement() if pebbled else None)

    def stem_button_mesh(self):
        z0 = self.mer.zmax - 0.5e-3
        path = np.linspace([0, 0, z0], [0.3e-3, 0, z0 + self.stem_len], 8)
        return ML.tube(path, np.linspace(2.6e-3, 2.1e-3, 8), 18)

    def cotyledon_mesh(self, k):
        n = np.array([np.cos(self.split_angle), np.sin(self.split_angle), 0]) * (1 if k == 0 else -1)
        a, b = self.seed_L / 2, self.seed_W / 2
        return ML.half_ellipsoid(self.seed_c, (b, b, a), n, gap=0.3e-3, n=112,
                                 taper=lambda zn: 1 - self.SEED_TAPER * zn)

    def lenticel_points(self):
        m = self.surf(self.mer, n_mer=120, n_theta=96)
        n = int(3.0 * m.area * 1e4)                         # ILLUSTRATIVE density; denser towards the apex (cs42)
        zmin, zmax = self.mer.zmin, self.mer.zmax
        p, _ = ML.points_on(m, n, self.rng, weight=lambda q: 1.6 - 0.6 * (q[:, 2] - zmin) / (zmax - zmin))
        return p, np.full(len(p), 0.25e-3)

    def parts(self):
        cache = {}

        def c(name, fn):
            def g():
                if name not in cache:
                    cache[name] = fn()
                return cache[name]
            return g
        order = ["outer", "cuticle", "epidermis", "hypodermis", "skin_parenchyma", "stone_cell_layer"]
        pebbled = {"outer", "cuticle", "epidermis"}   # tubercles: cuticle + epidermis over hypodermal cork
        surf = {"outer": c("outer", lambda: self.surf(self.mer, True))}
        for n in order[1:]:
            surf[n] = c(n, lambda n=n: self.surf(self.m_layer[n], n in pebbled))
        g_t = self.t_testa_o + self.t_testa_i
        endo_o = c("eo", lambda: self.seed_mesh(g_t + self.t_endo))
        testa_o, testa_i = c("to", lambda: self.seed_mesh(g_t)), c("ti", lambda: self.seed_mesh(self.t_testa_i))
        seed_s = c("s", lambda: self.seed_mesh(0.0))
        b13, b23 = c("b13", lambda: self.band_mesh(1 / 3)), c("b23", lambda: self.band_mesh(2 / 3))
        R = {t["name"]: t["density"] for t in TISSUES}
        names = {"cuticle": "Cuticle", "epidermis": "Epidermis", "hypodermis": "Hypodermis",
                 "skin_parenchyma": "SkinParenchyma", "stone_cell_layer": "StoneCellLayer"}
        doc_layer = {"cuticle": "Thin waxy cuticle (gloss).", "epidermis": "One cell layer.",
                     "hypodermis": "1-3 cell layers; cork in the lenticels makes Hass skin pebbly.",
                     "skin_parenchyma": "Parenchyma ~40 um cells.",
                     "stone_cell_layer": "Stone cells stay with the peel: the skin/flesh boundary that makes ripe "
                                         "Hass peel cleanly."}
        P = [Part("exocarp", "Exocarp", "food:Exocarp", None, observability="visible",
                  doc="Hass skin 1.6 mm (uspto_pp14239), pebbled; green -> purple-black with ripening, unreliable as a "
                      "ripeness signal (avo_au_stages, ashton2006).")]
        for i, n in enumerate(order[1:]):
            P.append(Part(n, "Exocarp/" + names[n], "food:" + names[n],
                          ("mesh", lambda i=i: ML.hollow(surf[order[i]](), surf[order[i + 1]]())),
                          "Skin" if n == "cuticle" else "SkinSection", "visible" if n == "cuticle" else "hidden",
                          doc_layer[n], R[n], interior=(n != "cuticle")))
        P += [
            Part("lenticels", "Exocarp/Lenticels", "food:Lenticel", ("points", self.lenticel_points), None, "visible",
                 "Corky white/grey spots, denser towards the apex (cs42)."),
            Part("stem_button", "StemButton", "food:PedicelRemnant", ("mesh", self.stem_button_mesh), "StemButton",
                 "visible", "Clipped ~3 mm button left at harvest; its cap is the folk ripeness test (weak).", 900.0,
                 interior=False, attrs={"length_mm": float(self.stem_len * 1e3)}),
            Part("mesocarp", "Mesocarp", "food:Mesocarp", None, observability="visible after cutting",
                 doc="Flesh: parenchyma with oil droplets and oil idioblasts (~2 % of volume); ~72 % of fruit mass."),
            Part("flesh_outer", "Mesocarp/OuterDarkGreenBand", "food:FleshBand",
                 ("mesh", lambda: ML.hollow(surf["stone_cell_layer"](), b23())), "FleshOuter", "visible after cutting",
                 "Dark-green outer third (ashton2006).", R["flesh_outer"], attrs={"band": "outer"}),
            Part("flesh_middle", "Mesocarp/MiddlePaleGreenBand", "food:FleshBand",
                 ("mesh", lambda: ML.hollow(b23(), b13())), "FleshMiddle", "visible after cutting",
                 "Pale-green middle third.", R["flesh_middle"], attrs={"band": "middle"}),
            Part("flesh_inner", "Mesocarp/InnerYellowBand", "food:FleshBand",
                 ("mesh", lambda: ML.hollow(b13(), endo_o())), "FleshInner", "visible after cutting",
                 "Yellow inner third next to the seed.", R["flesh_inner"], attrs={"band": "inner"}),
            Part("vascular_strands", "Mesocarp/VascularStrands", "food:VascularStrands", None,
                 observability="visible after cutting",
                 doc="Pedicel cylinder splits into 6 strands, joining near the apex into the seed coat (cs42); the "
                     "'stringy' fibres of soft fruit."),
        ]
        for k, az in enumerate(self.strand_az):
            P.append(Part(f"strand_{k + 1}", f"Mesocarp/VascularStrands/Strand_{k + 1}", "food:VascularStrand",
                          ("mesh", lambda az=az: ML.tube(self.strand_path(az), 0.30e-3, 8)), "Strand",
                          "visible after cutting", "", R["vascular_strand"]))
        P += [
            Part("endocarp", "Endocarp", "food:Endocarp", ("mesh", lambda: ML.hollow(endo_o(), testa_o())), "Endocarp",
                 "visible after cutting", "A few rows of flattened cells; may stay on the seed coat (cs42).",
                 R["endocarp"], attrs={"adheres_to": str(self.p["testa_adheres_to"])}),
            Part("seed", "Seed", "food:AvocadoSeed", None, observability="visible after halving",
                 doc="Single seed, 'variable, mostly ellipsoid', 28-31 g (uspto); slightly off-centre (cs42)."),
            Part("seed_coat", "Seed/SeedCoat", "food:Testa", None, observability="visible after halving",
                 doc="Two papery coats stuck together, stone cells over tannin cells; brown (cs42, ufl)."),
            Part("outer_testa", "Seed/SeedCoat/OuterTesta", "food:OuterTesta",
                 ("mesh", lambda: ML.hollow(testa_o(), testa_i())), "SeedCoat", "visible after halving", "",
                 R["outer_testa"], interior=True),
            Part("inner_testa", "Seed/SeedCoat/InnerTesta", "food:InnerTesta",
                 ("mesh", lambda: ML.hollow(testa_i(), seed_s())), "SeedCoatInner", "hidden", "", R["inner_testa"]),
            Part("cotyledon_1", "Seed/Cotyledon_1", "food:Cotyledon", ("mesh", lambda: self.cotyledon_mesh(0)),
                 "Cotyledon", "hidden", "Starch-rich; near orange-white, browns (PPO) when cut (cs42).",
                 R["cotyledon"]),
            Part("cotyledon_2", "Seed/Cotyledon_2", "food:Cotyledon", ("mesh", lambda: self.cotyledon_mesh(1)),
                 "Cotyledon", "hidden", "", R["cotyledon"]),
            Part("embryonic_axis", "Seed/EmbryonicAxis", "food:EmbryonicAxis", None, observability="hidden",
                 doc="Attached centrally between the cotyledons (cs42)."),
        ]
        ec = self.embryo_c()
        for n, dz, r in (("Plumule", 1.4e-3, 0.7e-3), ("Hypocotyl", 0.0, 0.9e-3), ("Radicle", -1.5e-3, 0.8e-3)):
            P.append(Part(n.lower(), f"Seed/EmbryonicAxis/{n}", f"food:{n}",
                          ("mesh", lambda dz=dz, r=r: ML.ellipsoid(ec + [0, 0, dz], (r, r, 0.9e-3), n_u=12, n_v=24)),
                          "Embryo", "hidden", "", R["embryonic_axis"]))
        return P

    # ---------------------------------------------------------------- appearance
    def point_uv(self, U, V):
        t = 1 - V
        k = np.linspace(0, 1, len(self.mer.rho))
        rho, z = np.interp(t, k, self.mer.rho), np.interp(t, k, self.mer.z)
        th = 2 * np.pi * U
        m = self.mod(th, z)
        return np.stack([m * rho * np.cos(th), m * rho * np.sin(th), z], -1)

    def uv_of(self, P):
        rn, th, z = self.norm_coords(P)
        k = np.linspace(0, 1, len(self.mer.rho))
        _, i = cKDTree(np.stack([self.mer.rho, self.mer.z], 1)).query(np.stack([rn, z], 1))
        return np.stack([np.mod(th, 2 * np.pi) / (2 * np.pi), 1 - k[i]], 1)

    def seed_point_uv(self, U, V):
        pol = (1 - V) * np.pi
        ph = 2 * np.pi * U
        a, b = self.seed_L / 2, self.seed_W / 2
        r = b * (1 - self.SEED_TAPER * np.cos(pol)) * np.sin(pol)
        return np.stack([r * np.cos(ph), r * np.sin(ph), a * np.cos(pol)], -1) + self.seed_c

    def seed_uv_of(self, P):
        d = np.asarray(P) - self.seed_c
        pol = np.arccos(np.clip(d[:, 2] / (self.seed_L / 2 + 1e-9), -1, 1))
        return np.stack([np.mod(np.arctan2(d[:, 1], d[:, 0]), 2 * np.pi) / (2 * np.pi), 1 - pol / np.pi], 1)

    def skin_albedo(self, P, seed):
        if self.pebble_seeds is None:
            self.pebble_displacement()
        vn = ValueNoise3(seed)
        stage = self.p["ripeness_stage"]
        patch = np.clip(vn.fbm(P, 9e-3, 3), 0, 1)
        col = skin_colour(stage) * (1 - 0.35 * patch)[..., None] + \
            skin_colour(min(stage + 1.2, 5)) * (0.35 * patch)[..., None]
        F1 = worley(P, self.pebble_seeds, 1)
        ridge = np.clip(1 - F1 / (0.62 * self.p["pebble_scale"] * 1e-3), 0, 1)
        col = col * (0.75 + 0.45 * ridge)[..., None]
        col *= 1 + 0.08 * vn.fbm(P + 4, 0.7e-3, 2)[..., None]
        sp = (vn.fbm(P + 9, 0.35e-3, 2) > 0.55).astype(float)
        col = col * (1 - 0.5 * sp[..., None]) + np.array([0.45, 0.42, 0.25]) * 0.5 * sp[..., None]
        return np.clip(col, 0, 1)

    def seed_albedo(self, P, seed):
        """Testa: thin papery coat, mottled mid/dark brown, fine darker veins, lighter dried patches (sRGB)."""
        vn = ValueNoise3(seed + 21)
        rng = np.random.default_rng(seed + 22)
        mid, dark, flake = np.array([0.47, 0.29, 0.15]), np.array([0.24, 0.13, 0.06]), np.array([0.66, 0.50, 0.33])
        m = np.clip(0.45 + 0.6 * vn.fbm(P, 8e-3, 4), 0, 1)
        col = mid * (1 - m[..., None]) + dark * m[..., None]
        sm = self.seed_mesh(0.0, 24)
        pts = sample_surface(np.asarray(sm.vertices), np.asarray(sm.faces), 900, rng)
        d1, d2 = worley(P, pts, 1), worley(P, pts, 2)
        vein = np.exp(-((d2 - d1) / 0.12e-3) ** 2)
        col = col * (1 - 0.25 * vein[..., None]) + dark * 0.6 * 0.25 * vein[..., None]
        fl = np.clip((vn.fbm(P + 7, 9e-3, 3) - 0.30) / 0.35, 0, 1) ** 1.5
        col = col * (1 - 0.45 * fl[..., None]) + flake * 0.45 * fl[..., None]
        col *= 1 + 0.06 * vn.fbm(P + 3, 0.4e-3, 2)[..., None]
        return np.clip(col, 0, 1)

    def section_rgb(self, lab, P, seed=0):
        vn = ValueNoise3(seed + 13)
        rgb = np.zeros(lab.shape + (3,))
        for i, t in enumerate(TISSUES):
            rgb[lab == i + 1] = t["colour"]
        fl = np.isin(lab, [TI["flesh_outer"], TI["flesh_middle"], TI["flesh_inner"]])
        c_out, c_mid, c_in = flesh_band_colours(self.p["ripeness_stage"])
        f = np.clip(self.band_fraction(P[fl]), 0, 1)
        w_in = np.clip(1 - f * 3, 0, 1) ** 1.2
        w_out = np.clip(f * 3 - 2, 0, 1) ** 0.8
        col = c_mid * (1 - w_in - w_out)[:, None] + c_in * w_in[:, None] + c_out * w_out[:, None]
        under = np.exp(-(1 - f) / 0.05)                      # vivid green just under the stone cells
        col = col * (1 - 0.5 * under[:, None]) + lch_to_srgb(60, 45, 118) * 0.5 * under[:, None]
        rgb[fl] = col * (1 + 0.025 * vn.fbm(P[fl], 0.6e-3, 2))[:, None]
        sk = np.isin(lab, [TI["cuticle"], TI["epidermis"], TI["hypodermis"]])
        rgb[sk] = skin_colour(self.p["ripeness_stage"])
        for n in ("outer_testa", "inner_testa"):
            m = lab == TI[n]
            rgb[m] *= (0.85 + 0.3 * (0.5 + 0.5 * vn.fbm(P[m], 0.8e-3, 2)))[:, None]
        ct = lab == TI["cotyledon"]
        rgb[ct] *= (0.96 + 0.06 * vn.fbm(P[ct], 1.2e-3, 3))[:, None]
        return np.clip(rgb, 0, 1)

    # ---------------------------------------------------------------- mechanics (hidden truth)
    def firmness(self):
        return float(penetrometer_force(self.p["ripeness_stage"]))

    def flesh_E(self):
        return float(flesh_E_MPa(self.p["ripeness_stage"]))
