"""Potato tuber twin (Solanum tuberosum, table potato): a swollen underground stem.

Anatomy (docs/wave2/POTATO.md for the sources): the PERIDERM (skin: a few layers of suberised phellem cells) with
LENTICELS; EYES -- axillary buds in the axils of scale-leaf scars ('eyebrows') -- arranged in a spiral (the tuber is a
stem: phyllotaxis), crowded towards the apical ROSE end; at the other end the HEEL with the STOLON SCAR.  Inside,
the CORTEX (storage parenchyma outside the vascular ring), the VASCULAR RING (the thin line a few mm under the skin),
the PERIMEDULLARY ZONE (the main starch store) and the PITH (medulla) whose branches run out to the eyes (the
translucent star of a cut tuber).

Geometry: a body of revolution about the long axis with an elliptic cross-section, lumps and the eye pits all in the
modulation m(theta, z) -- so meshes and labels share the eye depressions; tissue boundaries are inward offsets of the
outline (food.geometry.Meridian.offset); the pith is a scaled outline with branches (label field) to every eye.

Partonomy (USD paths below /Food/Parts):
  Periderm, Periderm/Lenticels (points), Eyes/Eye_k (buds), StolonScar, Cortex, VascularRing, PerimedullaryZone,
  Pith
"""

import numpy as np
from scipy.spatial import cKDTree

from . import meshlib as ML
from . import params as PR
from .geometry import Body, Meridian, resample
from .partgraph import Part
from .shapes import smoothstep
from .texture import ValueNoise3, lch_to_srgb, spots

TISSUES = [
    {"name": "periderm", "density": 1080.0, "colour": (0.75, 0.60, 0.38)},
    {"name": "cortex", "density": 1080.0, "colour": (0.96, 0.92, 0.70)},
    {"name": "vascular_ring", "density": 1080.0, "colour": (0.90, 0.86, 0.62)},
    {"name": "perimedullary", "density": 1080.0, "colour": (0.97, 0.93, 0.72)},
    {"name": "pith", "density": 1060.0, "colour": (0.92, 0.89, 0.72)},
    {"name": "eye", "density": 1050.0, "colour": (0.55, 0.40, 0.30)},
]
TI = {t["name"]: i + 1 for i, t in enumerate(TISSUES)}


class Potato(Body):
    tissues = TISSUES
    kind = "potato"

    def __init__(self, seed=0, overrides=None):
        p = PR.sample("potato", seed)
        p.update(overrides or {})
        rng = np.random.default_rng(p["_rng_state"])
        self.p, self.seed, self.rng = p, seed, rng
        for t in TISSUES:
            t["density"] = p["density"] if t["name"] != "pith" else p["density"] - 15.0
        L, W = p["length"] * 1e-3, p["width"] * 1e-3
        self.L, self.W = L, W
        self.tw = p["thickness_ratio"]                       # thickness / width
        n = 400
        t = np.linspace(0, np.pi, n)
        rho = W / 2 * np.sin(t) ** 0.92
        z = L / 2 * np.cos(t)
        self.vn = ValueNoise3(seed + 3)
        # eyes on a spiral (stem phyllotaxis), crowded towards the rose end (+z)
        ne = int(p["eyes"])
        u = (np.arange(ne) + 0.5) / ne
        ze = L / 2 * (1 - 2 * (1 - u) ** p["eye_crowding"]) * 0.92
        self.eye_z = ze
        self.eye_th = rng.uniform(0, 2 * np.pi) + np.arange(ne) * 2 * np.pi * p["phyllotaxis"]
        self.eye_depth = p["eye_depth"] * 1e-3
        self.eye_w = p["eye_width"] * 1e-3
        e = self.tw

        def mod(th, zz):
            ell = 1.0 / np.sqrt(np.cos(th) ** 2 + (np.sin(th) / e) ** 2)
            P = np.stack([np.cos(th) * 0.02, np.sin(th) * 0.02, zz], -1)
            lump = p["lumpiness"] * self.vn.fbm(P, 0.03, 2)
            m = ell * (1 + lump)
            r_loc = W / 2 * ell
            dth = np.abs(((th[..., None] - self.eye_th) + np.pi) % (2 * np.pi) - np.pi)
            d2 = (dth * r_loc[..., None]) ** 2 + (zz[..., None] - self.eye_z) ** 2
            pit = (self.eye_depth * np.exp(-d2 / self.eye_w ** 2)).max(-1)
            return m * (1 - pit / np.maximum(r_loc, 1e-3))
        super().__init__(Meridian(rho, z), mod)
        self.t_skin = p["periderm_thickness"] * 1e-6
        self.d_ring = p["vascular_ring_depth"] * 1e-3
        self.t_ring = 0.4e-3
        self.m_skin = self.mer.offset(self.t_skin)
        self.m_ring_o = self.mer.offset(self.d_ring)
        self.m_ring_i = self.mer.offset(self.d_ring + self.t_ring)
        s = p["pith_fraction"]
        self.m_pith = Meridian(self.mer.rho * s, self.mer.z * (0.55 + 0.4 * s))

    # ---------------------------------------------------------------- labels
    def eye_points(self, depth=0.0):
        r = self.mer.radius_at(self.eye_z)
        e = self.tw
        ell = 1.0 / np.sqrt(np.cos(self.eye_th) ** 2 + (np.sin(self.eye_th) / e) ** 2)
        rr = r * ell - self.eye_depth - depth
        return np.stack([rr * np.cos(self.eye_th), rr * np.sin(self.eye_th), self.eye_z], 1)

    def label(self, P):
        P = np.asarray(P, float)
        rn, th, z = self.norm_coords(P)
        out = np.zeros(P.shape[:-1], np.int16)
        ins = self.mer.contains(rn, z)
        out[ins] = TI["perimedullary"]
        out[ins & ~self.m_ring_o.contains(rn, z)] = TI["cortex"]
        out[ins & self.m_ring_o.contains(rn, z) & ~self.m_ring_i.contains(rn, z)] = TI["vascular_ring"]
        out[ins & ~self.m_skin.contains(rn, z)] = TI["periderm"]
        pith = ins & self.m_pith.contains(rn, z)
        # pith branches: thin strands from the pith out to each eye (the star of a cut tuber)
        sel = np.nonzero(ins & ~pith & self.m_ring_i.contains(rn, z))
        if len(sel[0]):
            E = self.eye_points(self.d_ring)
            Q = P[sel]
            A = np.zeros_like(E)
            A[:, 2] = E[:, 2] * 0.8
            ab = E - A
            t = np.clip(np.einsum("nkj,kj->nk", Q[:, None, :] - A[None], ab) / np.einsum("kj,kj->k", ab, ab), 0, 1)
            d = np.linalg.norm(Q[:, None, :] - (A[None] + t[..., None] * ab[None]), axis=2).min(1)
            br = d < self.p["pith_branch_width"] * 5e-4
            pith_mask = np.zeros(out.shape, bool)
            pith_mask[tuple(a[br] for a in sel)] = True
            pith |= pith_mask
        out[pith] = TI["pith"]
        E = self.eye_points(-0.3e-3)
        sel = np.nonzero(ins)
        d, i = cKDTree(E).query(P[sel])
        hit = d < 1.0e-3
        out[tuple(a[hit] for a in sel)] = TI["eye"]
        return out

    def tissue_of(self, rn, th, z, P):
        return self.label(P)

    def bbox(self):
        r = self.W / 2 * 1.15
        return np.array([-r, -r, self.mer.zmin - 1e-3]), np.array([r, r, self.mer.zmax + 1e-3])

    # ---------------------------------------------------------------- meshes
    def surf(self, mer, n_theta=160, n_mer=260):
        r, z = resample(mer.rho, mer.z, n_mer)
        r[0] = r[-1] = 0
        return ML.surface_of_revolution(Meridian(r, z), n_theta, modulation=self.mod)

    def lenticel_points(self):
        m = self.surf(self.mer, 80, 120)
        n = int(self.p["lenticel_density"] * m.area * 1e4)
        pts, _ = ML.points_on(m, n, np.random.default_rng(self.seed + 4))
        return pts, np.full(len(pts), 0.35e-3)

    def parts(self):
        cache = {}

        def c(name, fn):
            def g():
                if name not in cache:
                    cache[name] = fn()
                return cache[name]
            return g
        S = {k: c(k, lambda m=m: self.surf(m)) for k, m in (("o", self.mer), ("s", self.m_skin),
                                                            ("ro", self.m_ring_o), ("ri", self.m_ring_i),
                                                            ("p", self.m_pith))}
        R = {t["name"]: t["density"] for t in TISSUES}
        p = self.p
        P = [Part("periderm", "Periderm", "food:Periderm", ("mesh", lambda: ML.hollow(S["o"](), S["s"]())), "Skin",
                  "visible", "Corky skin (suberised phellem); greens under light.", R["periderm"], interior=False,
                  attrs={"thickness_um": p["periderm_thickness"]}),
             Part("lenticels", "Periderm/Lenticels", "food:Lenticel", ("points", self.lenticel_points), None,
                  "visible", "Breathing pores of the skin.", interior=False),
             Part("eyes", "Eyes", "food:EyeGroup", None, observability="visible",
                  doc="Axillary buds in scale-leaf scars, on a spiral (phyllotaxis), crowded at the rose end; "
                      "removed with the tip of the peeler.", attrs={"count": len(self.eye_z),
                                                                   "phyllotaxis": p["phyllotaxis"]})]
        E = self.eye_points(-0.2e-3)
        for k, e in enumerate(E):
            P.append(Part(f"eye_{k + 1}", f"Eyes/Eye_{k + 1:02d}", "food:Eye",
                          ("mesh", lambda e=e: ML.ellipsoid(e, (1.0e-3, 1.0e-3, 1.0e-3), n_u=8, n_v=16)), "Eye",
                          "visible", "", R["eye"], interior=False,
                          attrs={"position_along_axis": float((self.eye_z[k] - self.mer.zmin) / self.L)}))
        P += [
            Part("stolon_scar", "StolonScar", "food:StolonScar",
                 ("mesh", lambda: ML.ellipsoid([0, 0, self.mer.zmin + 0.4e-3], (2.2e-3, 2.2e-3, 0.8e-3), n_u=8,
                                               n_v=16)), "Eye", "visible", "Where the tuber hung on its stolon (heel).",
                 R["eye"], interior=False),
            Part("cortex", "Cortex", "food:Cortex", ("mesh", lambda: ML.hollow(S["s"](), S["ro"]())), "Flesh",
                 "visible after peeling", "Storage parenchyma outside the vascular ring.", R["cortex"]),
            Part("vascular_ring", "VascularRing", "food:VascularRing", ("mesh", lambda: ML.hollow(S["ro"](), S["ri"]())),
                 "Ring", "visible after cutting", "The thin line a few mm under the skin.", R["vascular_ring"],
                 attrs={"depth_mm": p["vascular_ring_depth"]}),
            Part("perimedullary", "PerimedullaryZone", "food:PerimedullaryZone",
                 ("mesh", lambda: ML.hollow(S["ri"](), S["p"]())), "Flesh", "visible after cutting",
                 "Main starch store.", R["perimedullary"]),
            Part("pith", "Pith", "food:Pith", ("mesh", S["p"]), "Pith", "visible after cutting",
                 "Medulla, more watery; its branches to the eyes are in the label field (cut faces).", R["pith"]),
        ]
        return P

    # ---------------------------------------------------------------- appearance
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
        vn = ValueNoise3(seed + 121)
        col = np.broadcast_to(lch_to_srgb(*self.p["skin_LCh"]), P.shape).copy()
        col = col * (1 + 0.12 * vn.fbm(P, 3e-3, 4))[..., None]
        net = np.clip(vn.fbm(P * np.array([1, 1, 0.6]), 0.8e-3, 3) - 0.25, 0, 1)   # russet netting / scurf
        col = col * (1 - 0.25 * net[..., None])
        pts, _ = self.lenticel_points()
        col = col * (1 - 0.3 * spots(P, pts, np.full(len(pts), 0.3e-3))[..., None])
        E = self.eye_points(-0.1e-3)
        brow = spots(P, E + np.array([0, 0, 1.2e-3]), np.full(len(E), 1.8e-3))
        col = col * (1 - 0.35 * brow[..., None])
        eye = spots(P, E, np.full(len(E), 0.9e-3))
        col = col * (1 - eye[..., None]) + np.array([0.35, 0.22, 0.18]) * eye[..., None]
        dirt = np.clip(vn.fbm(P + 5, 6e-3, 3) - 0.3, 0, 1) * self.p["soil_residue"]
        col = col * (1 - dirt[..., None]) + np.array([0.33, 0.27, 0.20]) * dirt[..., None]
        return np.clip(col, 0, 1)

    def section_rgb(self, lab, P, seed=0):
        rgb = np.zeros(lab.shape + (3,)) + 0.86
        vn = ValueNoise3(seed + 131)
        flesh = lch_to_srgb(*self.p["flesh_LCh"])
        for i, t in enumerate(TISSUES):
            rgb[lab == i + 1] = t["colour"]
        for n_, f in (("cortex", 1.0), ("perimedullary", 1.0), ("vascular_ring", 0.93), ("pith", 0.97)):
            m = lab == TI[n_]
            c = flesh * f
            if n_ == "pith":
                c = flesh * np.array([0.95, 0.97, 1.02])     # more translucent, watery
            rgb[m] = np.clip(c, 0, 1) * (1 + 0.025 * vn.fbm(P[m], 0.5e-3, 2))[:, None]
        rgb[lab == TI["periderm"]] = lch_to_srgb(*self.p["skin_LCh"])
        return np.clip(rgb, 0, 1)

    # ---------------------------------------------------------------- asset declaration
    def asset_spec(self):
        from .potato_models import potato_beliefs, potato_truth
        p = self.p
        flesh = tuple(float(c) for c in lch_to_srgb(*p["flesh_LCh"]))
        ext = float(self.L * 0.58)
        return {
            "build": {"h": 0.3e-3},
            "textures": [("skin.png", self.point_uv, self.skin_albedo, 2048, 1024)],
            "uv_fns": {"periderm": self.uv_of},
            "materials": {
                "Skin": dict(diffuse_reflection_color_image="tex:skin.png", specular_reflection_roughness=0.7,
                             specular_reflection_weight=0.25),
                "Flesh": dict(diffuse_reflection_color=flesh, specular_reflection_roughness=0.3,
                              specular_reflection_weight=0.6, subsurface_weight=0.25, subsurface_scale=0.002,
                              subsurface_scattering_color=flesh),
                "Ring": dict(diffuse_reflection_color=tuple(c * 0.93 for c in flesh),
                             specular_reflection_roughness=0.3),
                "Pith": dict(diffuse_reflection_color=tuple(min(1.0, c * k) for c, k in zip(flesh, (0.95, 0.97, 1.02))),
                             specular_reflection_roughness=0.25),
                "Eye": dict(diffuse_reflection_color=(0.40, 0.27, 0.20), specular_reflection_roughness=0.8),
            },
            "section_material": {"specular_reflection_roughness": 0.25, "specular_reflection_weight": 0.6,
                                 "subsurface_weight": 0.2, "subsurface_scale": 0.002,
                                 "subsurface_scattering_color": flesh},
            "root": {"peel_state": "unpeeled", "cut_state": "whole", "cooked_state": "raw", "greening": "no",
                     "sprouting": "no", "cooking_type": p["cooking_type"]},
            "friction": p["friction_skin_rubber"], "restitution": 0.2,
            "cuts": [
                {"name": "lengthwise", "origin": [0, 0, 0], "e1": [1, 0, 0], "e2": [0, 0, 1], "ext": ext,
                 "states": {"half_a": ("pos", {"cut_state": "halved"}), "half_b": ("neg", {"cut_state": "halved"})}},
                {"name": "across", "origin": [0, 0, 0.003], "e1": [1, 0, 0], "e2": [0, 1, 0], "ext": float(self.W * 0.65),
                 "res": 1024,
                 "states": {"rose_piece": ("pos", {"cut_state": "cut_across"}),
                            "heel_piece": ("neg", {"cut_state": "cut_across"})}},
            ],
            "beliefs": lambda mp: potato_beliefs(self, mp),
            "truth": lambda: potato_truth(self),
        }
