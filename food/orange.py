"""Sweet orange twin (Citrus sinensis; navel or Valencia): a hesperidium.

Anatomy (docs/wave2/ORANGE.md for the sources): the PEEL is the flavedo (coloured exocarp, pitted by oil glands
whose oil sprays when the peel is bent) over the white spongy albedo (mesocarp).  Inside, SEGMENTS (carpels, about
10-12) arranged round a central CORE (columella, pith); each segment is wrapped in a thin MEMBRANE (endocarp) and
filled with JUICE VESICLES (juice sacs) growing inwards from its back wall; SEEDS sit at the inner edge near the
core (Valencia: a few; navel: usually none).  Navel oranges carry a small secondary fruit, the NAVEL, at the stylar
end.  At the stem end the STEM BUTTON with the calyx.

Geometry: shells and the core are regions of the (rho, z) half-plane revolved fully; segments are the same pulp
region revolved over angular sectors (food.shapes.revolve_polygon), membrane = segment minus its inset pulp.  The
flavedo's outer face carries the oil-gland pits as a displacement.

Partonomy (USD paths below /Food/Parts):
  Peel/{Flavedo, Albedo}, Peel/OilGlands (points), StemButton, Core, Segments/Segment_k/{Membrane, Vesicles},
  Segments/Segment_k/Seed_j, Navel (navel cultivar)
"""

import numpy as np
from scipy.spatial import cKDTree

from . import meshlib as ML
from . import params as PR
from .geometry import Body, Meridian, resample
from .partgraph import Part
from .shapes import angle_in, region_contains, revolve_polygon
from .texture import ValueNoise3, lch_to_srgb, sample_surface, worley

TISSUES = [
    {"name": "flavedo", "density": 950.0, "colour": (0.95, 0.50, 0.08)},
    {"name": "albedo", "density": 700.0, "colour": (0.96, 0.93, 0.85)},
    {"name": "core", "density": 750.0, "colour": (0.95, 0.92, 0.82)},
    {"name": "membrane", "density": 1040.0, "colour": (0.97, 0.85, 0.60)},
    {"name": "vesicles", "density": 1040.0, "colour": (0.98, 0.60, 0.12)},
    {"name": "seed", "density": 1100.0, "colour": (0.92, 0.88, 0.72)},
    {"name": "navel", "density": 900.0, "colour": (0.97, 0.75, 0.40)},
    {"name": "stem_button", "density": 900.0, "colour": (0.35, 0.40, 0.18)},
]
TI = {t["name"]: i + 1 for i, t in enumerate(TISSUES)}
N_TH = 144


def ellipse_curve(a, c, z0=0.0, n=200, r_scale=1.0):
    """Meridian of an ellipsoid of revolution (equatorial radius a, polar half-axis c), top pole -> bottom pole."""
    t = np.linspace(0, np.pi, n)
    return a * np.sin(t) * r_scale, z0 + c * np.cos(t)


class Orange(Body):
    tissues = TISSUES
    kind = "orange"

    def __init__(self, seed=0, overrides=None):
        p = PR.sample(self.kind, seed)
        p.update(overrides or {})
        rng = np.random.default_rng(p["_rng_state"])
        self.p, self.seed, self.rng = p, seed, rng
        for t in TISSUES:
            t["density"] = {"flavedo": p["peel_density"], "albedo": p["albedo_density"],
                            "vesicles": p["pulp_density"], "membrane": p["pulp_density"]}.get(t["name"], t["density"])
        p["height"] = p["diameter"] * p["shape_index"]
        if p["cultivar"] == "navel":
            p["seeds"] = 0                                  # Washington navel: essentially seedless (ucr_citrus)
        self.a, self.c = p["diameter"] * 5e-4, p["height"] * 5e-4
        self.t_flav = p["flavedo_thickness"] * 1e-3
        self.t_peel = p["peel_thickness"] * 1e-3
        self.r_core = p["core_diameter"] * 5e-4
        self.n_seg = int(p["segments"])
        self.t_mem = p["membrane_thickness"] * 1e-3
        self.gap = 0.12e-3
        self.phi = rng.uniform(0, 2 * np.pi)
        # segment widths vary a little (they are not all equal)
        w = rng.normal(1.0, 0.08, self.n_seg)
        w = np.clip(w, 0.75, 1.25)
        self.seg_edges = self.phi + np.concatenate([[0], np.cumsum(w / w.sum() * 2 * np.pi)])
        rho, z = ellipse_curve(self.a, self.c)
        lop = rng.uniform(0, 0.02)

        def mod(th, zz):
            return 1 + lop * np.cos(th - self.phi) + 0.006 * np.cos(self.n_seg * (th - self.phi))
        super().__init__(Meridian(rho, z), mod)
        self.navel = p["cultivar"] == "navel"
        self._seeds = None
        self._glands = None

    # ---------------------------------------------------------------- regions (normalised rho, z)
    def surf_curve(self, depth):
        """Ellipsoid outline moved inwards by `depth` (approximately parallel: shrink both semi-axes)."""
        return ellipse_curve(self.a - depth, self.c - depth)

    def shell_poly(self, d_out, d_in):
        ro, zo = self.surf_curve(d_out)
        ri, zi = self.surf_curve(d_in)
        return np.vstack([np.stack([ro, zo], 1), np.stack([ri[::-1], zi[::-1]], 1)])

    def solid_poly(self, depth):
        r, z = self.surf_curve(depth)
        return np.stack([r, z], 1)

    def core_r(self, z):
        cp = self.core_poly()
        prof = cp[1:-1]                                     # top rim -> bottom rim
        return np.interp(z, prof[::-1, 1], prof[::-1, 0])

    def pulp_poly(self, inset=0.0):
        """Region between the core (its tapered profile) and the inner peel face."""
        ro, zo = self.surf_curve(self.t_peel + self.gap + inset)
        keep = ro > self.core_r(zo) + 0.3e-3 + inset
        ro, zo = ro[keep], zo[keep]
        zi = np.linspace(zo[-1], zo[0], 60)
        ri = self.core_r(zi) + inset
        return np.vstack([np.stack([ro, zo], 1), np.stack([ri, zi], 1)])

    def core_poly(self):
        ri, zi = self.surf_curve(self.t_peel + self.gap)
        ztop, zbot = zi.max(), zi.min()
        rc = self.r_core
        return np.array([[0, ztop], [rc * 1.1, ztop], [rc * 0.9, 0.5 * ztop], [rc * 0.7, 0.0], [rc * 0.8, 0.5 * zbot],
                         [rc * 1.0, zbot], [0, zbot]])

    def seg_sector(self, k, inset=0.0):
        a0, a1 = self.seg_edges[k], self.seg_edges[k + 1]
        rmid = 0.5 * (self.r_core + self.a)
        da = (self.gap + inset) / rmid
        return a0 + da, a1 - da

    # ---------------------------------------------------------------- labels
    def label(self, P):
        P = np.asarray(P, float)
        rn, th, z = self.norm_coords(P)
        out = np.zeros(P.shape[:-1], np.int16)
        ins = region_contains(self.solid_poly(0.0), rn, z)
        out[ins] = TI["albedo"]
        out[ins & ~region_contains(self.solid_poly(self.t_flav), rn, z)] = TI["flavedo"]
        pulp = ins & region_contains(self.pulp_poly(), rn, z)
        vpoly = self.pulp_poly(self.t_mem)
        for k in range(self.n_seg):
            a0, a1 = self.seg_sector(k)
            seg = pulp & angle_in(th, a0, a1)
            out[seg] = TI["membrane"]
            b0, b1 = self.seg_sector(k, self.t_mem)
            out[seg & angle_in(th, b0, b1) & region_contains(vpoly, rn, z)] = TI["vesicles"]
        out[ins & region_contains(self.core_poly(), rn, z)] = TI["core"]
        if self.navel:
            out[ins & (self.navel_e(P) < 1)] = TI["navel"]
        C, ax = self.seeds()
        if len(C):
            d, i = cKDTree(C).query(P[out > 0])
            hit = d < ax[i, 0]
            idx = np.nonzero(out > 0)
            out[tuple(a[hit] for a in idx)] = TI["seed"]
        return out

    def tissue_of(self, rn, th, z, P):
        return self.label(P)

    def inside(self, P):
        return self.label(P) > 0

    def bbox(self):
        r = self.a * 1.06
        return np.array([-r, -r, -self.c - 1e-3]), np.array([r, r, self.c + 4e-3])

    # ---------------------------------------------------------------- small parts
    def navel_e(self, P):
        zc = -self.c + self.t_peel + 4e-3
        d = np.asarray(P) - [0, 0, zc]
        return (d[..., 0] ** 2 + d[..., 1] ** 2) / (6e-3) ** 2 + (d[..., 2] / 4e-3) ** 2

    def seeds(self):
        if self._seeds is None:
            n = int(self.p["seeds"])
            rng = np.random.default_rng(self.seed + 5)
            C, AX = [], []
            for k in rng.choice(self.n_seg, size=min(n, self.n_seg), replace=False) if n else []:
                a = 0.5 * (self.seg_edges[k] + self.seg_edges[k + 1])
                r = self.r_core + 3.5e-3
                C.append([r * np.cos(a), r * np.sin(a), rng.normal(0, 3e-3)])
                AX.append([self.p["seed_length"] * 5e-4, 1.6e-3, 1.0e-3])
            self._seeds = (np.array(C).reshape(-1, 3), np.array(AX).reshape(-1, 3))
        return self._seeds

    def gland_points(self):
        if self._glands is None:
            m = self.surface(0.0, n=64)
            n = int(self.p["oil_gland_density"] * m.area * 1e4)
            pts, _ = ML.points_on(m, n, np.random.default_rng(self.seed + 9))
            self._glands = pts
        return self._glands

    def surface(self, depth, n=N_TH, displaced=False):
        r, z = self.surf_curve(depth)
        r, z = resample(r, z, 220)
        r[0] = r[-1] = 0
        disp = None
        if displaced:
            G = self.gland_points()
            cell = 2.0 / np.sqrt(self.p["oil_gland_density"]) * 1e-2 / 2

            def disp(T, tf, verts, normals):
                F1 = worley(verts, G, 1)
                return -0.12e-3 * np.exp(-(F1 / (0.45 * cell)) ** 2)
        return ML.surface_of_revolution(Meridian(r, z), n, modulation=self.mod, displacement=disp)

    def modulate(self, m):
        v = np.asarray(m.vertices).copy()
        k = self.mod(np.arctan2(v[:, 1], v[:, 0]), v[:, 2])
        v[:, :2] *= k[:, None]
        m.vertices = v
        return m

    def stem_mesh(self):
        z0 = self.c - 1.0e-3
        return ML.tube(np.linspace([0, 0, z0], [0, 0, z0 + 3.5e-3], 6), np.linspace(3.0e-3, 2.2e-3, 6), 18)

    def parts(self):
        cache = {}

        def c(name, fn):
            def g():
                if name not in cache:
                    cache[name] = fn()
                return cache[name]
            return g
        outer = c("o", lambda: self.surface(0.0, displaced=True))
        flav_in = c("fi", lambda: self.surface(self.t_flav))
        alb_in = c("ai", lambda: self.surface(self.t_peel))
        R = {t["name"]: t["density"] for t in TISSUES}
        P = [Part("peel", "Peel", "food:CitrusPeel", None, observability="visible",
                  doc="Flavedo over albedo; peels off the segments; ~20-30 % of the fruit's mass."),
             Part("flavedo", "Peel/Flavedo", "food:Flavedo", ("mesh", lambda: ML.hollow(outer(), flav_in())), "Peel",
                  "visible", "Coloured exocarp with oil glands (zest).", R["flavedo"], interior=False,
                  attrs={"thickness_mm": self.p["flavedo_thickness"]}),
             Part("albedo", "Peel/Albedo", "food:Albedo", ("mesh", lambda: ML.hollow(flav_in(), alb_in())), "Albedo",
                  "visible after peeling", "White spongy mesocarp: bitter, avoided when zesting.", R["albedo"],
                  attrs={"thickness_mm": self.p["peel_thickness"] - self.p["flavedo_thickness"]}),
             Part("oil_glands", "Peel/OilGlands", "food:OilGland",
                  ("points", lambda: (self.gland_points(), np.full(len(self.gland_points()), 0.6e-3))), None,
                  "visible", "Oil cavities in the flavedo: the spray when peel is bent."),
             Part("stem_button", "StemButton", "food:Calyx", ("mesh", self.stem_mesh), "Stem", "visible",
                  "Calyx and pedicel stub at the stem end.", R["stem_button"], interior=False),
             Part("core", "Core", "food:Columella", ("mesh", lambda: self.modulate(revolve_polygon(self.core_poly(),
                                                                                                   48))),
                  "Core", "visible after cutting", "Central axis (pith) the segments are arranged round.", R["core"]),
             Part("segments", "Segments", "food:SegmentGroup", None, observability="visible after peeling",
                  doc="Carpels: each a membrane bag of juice vesicles.", attrs={"count": self.n_seg})]
        vpoly = self.pulp_poly(self.t_mem)
        ppoly = self.pulp_poly()
        C, AX = self.seeds()
        seg_of_seed = [int(np.searchsorted(np.mod(self.seg_edges[1:] - self.phi, 2 * np.pi),
                                           np.mod(np.arctan2(s[1], s[0]) - self.phi, 2 * np.pi))) for s in C]
        for k in range(self.n_seg):
            a0, a1 = self.seg_sector(k)
            b0, b1 = self.seg_sector(k, self.t_mem)
            outer_k = c(f"s{k}", lambda a0=a0, a1=a1: self.modulate(revolve_polygon(ppoly, 24, a0, a1)))
            inner_k = c(f"v{k}", lambda b0=b0, b1=b1: self.modulate(revolve_polygon(vpoly, 24, b0, b1)))
            base = f"Segments/Segment_{k + 1:02d}"
            P += [Part(f"segment_{k + 1}", base, "food:Segment", None, observability="visible after peeling",
                       attrs={"angle_deg": float(np.degrees(a1 - a0))}),
                  Part(f"membrane_{k + 1}", base + "/Membrane", "food:SegmentMembrane",
                       ("mesh", lambda o=outer_k, i=inner_k: ML.hollow(o(), i())), "Membrane",
                       "visible after peeling", "Endocarp film around the segment.", R["membrane"]),
                  Part(f"vesicles_{k + 1}", base + "/Vesicles", "food:JuiceVesicles", ("mesh", inner_k), "Pulp",
                       "visible after cutting", "Juice sacs (pulp).", R["vesicles"])]
            for j, (cc, ax) in enumerate(zip(C, AX)):
                if seg_of_seed[j] == k:
                    a = np.arctan2(cc[1], cc[0])
                    rot = np.array([[0, -np.sin(a), np.cos(a)], [0, np.cos(a), np.sin(a)], [1, 0, 0]]).T
                    P.append(Part(f"seed_{k + 1}_{j + 1}", f"{base}/Seed_{j + 1}", "food:CitrusSeed",
                                  ("mesh", lambda cc=cc, ax=ax, rot=rot: ML.ellipsoid(cc, (ax[1], ax[2], ax[0]), rot,
                                                                                      12, 24)),
                                  "Seed", "visible after cutting", "", R["seed"]))
        if self.navel:
            P.append(Part("navel", "Navel", "food:NavelSecondaryFruit",
                          ("mesh", lambda: ML.ellipsoid([0, 0, -self.c + self.t_peel + 4e-3], (6e-3, 6e-3, 4e-3),
                                                        n_u=16, n_v=32)),
                          "Navel", "visible after cutting", "Small secondary fruit at the stylar end.", R["navel"]))
        return P

    # ---------------------------------------------------------------- appearance
    def point_uv(self, U, V):
        r, z = self.surf_curve(0.0)
        k = np.linspace(0, 1, len(r))
        t = 1 - np.asarray(V)
        rr, zz = np.interp(t, k, r), np.interp(t, k, z)
        th = 2 * np.pi * np.asarray(U)
        m = self.mod(th, zz)
        return np.stack([m * rr * np.cos(th), m * rr * np.sin(th), zz], -1)

    def uv_of(self, P):
        rn, th, z = self.norm_coords(np.asarray(P, float))
        pol = np.arccos(np.clip(z / self.c, -1, 1))
        return np.stack([np.mod(th, 2 * np.pi) / (2 * np.pi), 1 - pol / np.pi], 1)

    def skin_albedo(self, P, seed):
        P = np.asarray(P, float)
        vn = ValueNoise3(seed + 81)
        col = np.broadcast_to(lch_to_srgb(*self.p["peel_LCh"]), P.shape).copy()
        col = col * (1 + 0.06 * vn.fbm(P, 6e-3, 3))[..., None]
        F1 = worley(P, self.gland_points(), 1)
        cell = 2.0 / np.sqrt(self.p["oil_gland_density"]) * 1e-2 / 2
        pit = np.exp(-(F1 / (0.45 * cell)) ** 2)
        col = col * (1 - 0.10 * pit[..., None])
        zz = P[..., 2] / self.c
        stem = np.exp(-((1 - zz) / 0.03) ** 2)
        col = col * (1 - stem[..., None]) + lch_to_srgb(45, 25, 95) * stem[..., None]
        return np.clip(col, 0, 1)

    def section_rgb(self, lab, P, seed=0):
        rgb = np.zeros(lab.shape + (3,)) + 0.86
        vn = ValueNoise3(seed + 91)
        for i, t in enumerate(TISSUES):
            rgb[lab == i + 1] = t["colour"]
        rgb[lab == TI["flavedo"]] = lch_to_srgb(*self.p["peel_LCh"])
        rgb[lab == TI["albedo"]] = lch_to_srgb(*self.p["albedo_LCh"])
        m = lab == TI["vesicles"]
        if m.any():                                          # juice sacs: spindles radiating from the back wall
            rn, th, z = self.norm_coords(P[m])
            Q = np.stack([th * rn / 0.7e-3, rn / 2.6e-3, z / 0.7e-3], -1)   # sacs ~0.7 mm wide, ~2.6 mm long radially
            lo, hi = Q.min(0) - 2, Q.max(0) + 2
            n = int(np.prod(hi - lo) * 0.9) + 10
            seeds = lo + np.random.default_rng(seed + 17).random((min(n, 400000), 3)) * (hi - lo)
            d1, d2 = worley(Q, seeds, 1), worley(Q, seeds, 2)
            wall = np.exp(-((d2 - d1) / 0.12) ** 2)
            pulp = lch_to_srgb(*self.p["pulp_LCh"])
            shade = (0.88 + 0.16 * np.clip(1 - d1, 0, 1)) * (1 - 0.18 * wall)
            rgb[m] = pulp * shade[:, None] * (1 + 0.03 * vn.fbm(P[m], 0.5e-3, 2))[:, None]
        rgb[lab == TI["membrane"]] = lch_to_srgb(*self.p["membrane_LCh"])
        return np.clip(rgb, 0, 1)

    # ---------------------------------------------------------------- asset declaration
    def asset_spec(self):
        from .orange_models import orange_beliefs, orange_truth
        p = self.p
        ext = float(max(self.a, self.c) * 1.12)
        pulp = tuple(float(c) for c in lch_to_srgb(*p["pulp_LCh"]))
        return {
            "build": {"h": 0.3e-3},
            "textures": [("skin.png", self.point_uv, self.skin_albedo, 2048, 1024)],
            "uv_fns": {"flavedo": self.uv_of},
            "materials": {
                "Peel": dict(diffuse_reflection_color_image="tex:skin.png", specular_reflection_roughness=0.35,
                             specular_reflection_weight=0.55, coat_weight=0.2, coat_roughness=0.25),
                "Albedo": dict(diffuse_reflection_color=tuple(float(c) for c in lch_to_srgb(*p["albedo_LCh"])),
                               specular_reflection_roughness=0.8),
                "Stem": dict(diffuse_reflection_color=(0.30, 0.34, 0.14), specular_reflection_roughness=0.6),
                "Core": dict(diffuse_reflection_color=(0.93, 0.90, 0.80), specular_reflection_roughness=0.6),
                "Membrane": dict(diffuse_reflection_color=tuple(float(c) for c in lch_to_srgb(*p["membrane_LCh"])),
                                 specular_reflection_roughness=0.3, specular_reflection_weight=0.6,
                                 subsurface_weight=0.3, subsurface_scale=0.002, subsurface_scattering_color=pulp),
                "Pulp": dict(diffuse_reflection_color=pulp, specular_reflection_roughness=0.15,
                             specular_reflection_weight=0.8, subsurface_weight=0.5, subsurface_scale=0.004,
                             subsurface_scattering_color=pulp),
                "Seed": dict(diffuse_reflection_color=(0.92, 0.88, 0.72), specular_reflection_roughness=0.4),
                "Navel": dict(diffuse_reflection_color=(0.95, 0.72, 0.35), specular_reflection_roughness=0.4),
            },
            "section_material": {"specular_reflection_roughness": 0.15, "specular_reflection_weight": 0.8,
                                 "subsurface_weight": 0.3, "subsurface_scale": 0.002,
                                 "subsurface_scattering_color": pulp},
            "root": {"peel_state": "on", "cut_state": "whole", "juice_state": "intact", "cultivar": p["cultivar"]},
            "friction": p["friction_peel_rubber"], "restitution": 0.3,
            "cuts": [
                {"name": "across", "origin": [0, 0, 0.002], "e1": [1, 0, 0], "e2": [0, 1, 0], "ext": ext,
                 "states": {"stem_half": ("pos", {"cut_state": "halved_across"}),
                            "navel_half": ("neg", {"cut_state": "halved_across"})}},
                {"name": "lengthwise", "origin": [0, 0, 0],
                 "e1": [float(np.cos(self.seg_edges[0] + 0.3)), float(np.sin(self.seg_edges[0] + 0.3)), 0.0],
                 "e2": [0, 0, 1], "ext": ext,
                 "states": {"half_a": ("pos", {"cut_state": "halved_lengthwise"}),
                            "half_b": ("neg", {"cut_state": "halved_lengthwise"})}},
            ],
            "beliefs": lambda mp: orange_beliefs(self, mp),
            "truth": lambda: orange_truth(self),
        }
