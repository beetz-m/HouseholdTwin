"""Carrot twin (Daucus carota subsp. sativus, Nantes type, loose, leaves trimmed): a storage taproot.

Anatomy (docs/wave1/CARROT.md for the sources): under a thin PERIDERM (the 'skin', a few cell layers) lies the orange
SECONDARY PHLOEM -- botanically phloem, called the 'cortex' in the kitchen -- then the VASCULAR CAMBIUM (a thin ring
that makes phloem outwards and xylem inwards), the paler SECONDARY XYLEM ('core'), and a small primary-xylem centre.
Nantes carrots are bred for a small core.  At the top the CROWN (shoulder) carries the cut petiole stubs and turns
green where it was exposed to light; the root tapers to a thin TIP.  The surface shows horizontal LENTICEL bands
(fine rings) and lateral-root scars in vertical rows.

Frame: +z from the root tip (bottom) to the crown (top), origin at mid-length; a slight bend (food.shapes.Bend).

Partonomy (USD paths below /Food/Parts):
  Periderm, SecondaryPhloem, VascularCambium, SecondaryXylem, PrimaryXylem
  Crown/{Shoulder, PetioleStubs/Stub_k}, RootTip, LateralRootScars (points), LenticelBands (points)
"""

import numpy as np

from . import meshlib as ML
from . import params as PR
from .geometry import Body, Meridian, smooth_meridian
from .partgraph import Part
from .shapes import Bend, smoothstep
from .texture import ValueNoise3, lch_to_srgb, spots

TISSUES = [
    {"name": "periderm", "density": 1030.0, "colour": (0.80, 0.42, 0.12)},
    {"name": "phloem", "density": 1040.0, "colour": (0.95, 0.50, 0.10)},
    {"name": "cambium", "density": 1030.0, "colour": (0.96, 0.75, 0.35)},
    {"name": "xylem", "density": 1030.0, "colour": (0.95, 0.62, 0.20)},
    {"name": "primary_xylem", "density": 1030.0, "colour": (0.96, 0.78, 0.45)},
    {"name": "shoulder", "density": 1030.0, "colour": (0.55, 0.55, 0.20)},
]
TI = {t["name"]: i + 1 for i, t in enumerate(TISSUES)}


class Carrot(Body):
    tissues = TISSUES
    kind = "carrot"

    def __init__(self, seed=0, overrides=None):
        p = PR.sample("carrot", seed)
        p.update(overrides or {})
        rng = np.random.default_rng(p["_rng_state"])
        self.p, self.seed, self.rng = p, seed, rng
        L, D = p["length"] * 1e-3, p["shoulder_diameter"] * 1e-3
        self.L, self.D = L, D
        # Nantes: near-cylindrical, rounded shoulder, blunt-tapering bottom third into a thin tip
        s = np.array([0.0, 0.004, 0.02, 0.06, 0.15, 0.35, 0.65, 0.88, 0.96, 0.99, 1.0])
        tb = p["tip_bluntness"]
        r = np.array([0.0, 0.035, 0.12 + 0.2 * tb, 0.38 + 0.25 * tb, 0.70 + 0.12 * tb, 0.90, 0.98, 1.0, 0.93,
                      0.70, 0.0]) * D / 2
        z = (s - 0.5) * L
        ctrl = np.stack([r[::-1], z[::-1]], 1)
        rho, zz = smooth_meridian(ctrl, 600)
        vn = ValueNoise3(seed + 71)
        self.vn = vn
        k_band = 2 * np.pi / (p["lenticel_band_spacing"] * 1e-3)
        a_band = p["lenticel_band_relief"] * 1e-3

        def mod(th, zc):
            P = np.stack([np.cos(th) * 0.01, np.sin(th) * 0.01, zc], -1)
            lump = 0.025 * vn.fbm(P, 0.02, 2)
            ring = a_band / (D / 2) * (0.5 + 0.5 * np.cos(k_band * zc + 2.0 * vn.fbm(P, 0.01, 1))) ** 6
            return 1 + lump - ring
        super().__init__(Meridian(rho, zz), mod)
        self.bend = Bend(p["bend_radius"] * 1e-3)
        self.t_peri = p["periderm_thickness"] * 1e-6
        self.m_peri = self.mer.offset(self.t_peri)
        self.xylem_frac = p["xylem_radius_fraction"]
        self.t_camb = 0.15e-3
        self.prim_r = p["primary_xylem_diameter"] * 5e-4
        self.n_stub = int(p["petiole_stubs"])

    # ---------------------------------------------------------------- radial structure (straight frame)
    def r_out(self, z):
        return np.interp(z, self.mer.z[::-1], self.mer.rho[::-1])

    def r_xylem(self, z):
        q = np.clip((z - self.mer.zmin) / self.L, 0, 1)
        return self.xylem_frac * self.r_out(z) * (1 - 0.15 * smoothstep(0.85, 1.0, q))

    def label(self, P):
        S = self.bend.unbend(np.asarray(P, float))
        rn, th, z = self.norm_coords(S)
        out = np.zeros(S.shape[:-1], np.int16)
        ins = self.mer.contains(rn, z)
        if not ins.any():
            return out
        rho = np.hypot(S[..., 0], S[..., 1])
        lab = np.full(S.shape[:-1], TI["phloem"], np.int16)
        rx = self.r_xylem(z)
        lab[rho < rx + self.t_camb / 2] = TI["cambium"]
        lab[rho < rx - self.t_camb / 2] = TI["xylem"]
        lab[rho < self.prim_r] = TI["primary_xylem"]
        lab[~self.m_peri.contains(rn, z)] = TI["periderm"]
        q = (z - self.mer.zmin) / self.L
        green = (q > 1 - self.p["green_shoulder_length"] * 1e-3 / self.L) & (self.p["green_shoulder_length"] > 0) & \
            ~self.m_peri.offset(0.6e-3).contains(rn, z)
        lab[green] = TI["shoulder"]
        out[ins] = lab[ins]
        return out

    def tissue_of(self, rn, th, z, P):
        return self.label(P)

    def inside(self, P):
        return self.label(P) > 0

    def bbox(self):
        z = np.linspace(self.mer.zmin, self.mer.zmax + 8e-3, 12)
        pts = self.bend.bend(np.array([[sx * self.D * 0.62, sy * self.D * 0.62, zz]
                                       for sx in (-1, 1) for sy in (-1, 1) for zz in z]))
        return pts.min(0) - 1e-3, pts.max(0) + 1e-3

    # ---------------------------------------------------------------- meshes
    def surf(self, mer, n_theta=128, n_mer=320):
        from .geometry import resample
        r, zz = resample(mer.rho, mer.z, n_mer)
        r[0] = r[-1] = 0
        return self.bend.bend_mesh(ML.surface_of_revolution(Meridian(r, zz), n_theta, modulation=self.mod))

    def radial_mesh(self, rfun, n_z=240, n_th=96, z_lo=None, z_hi=None):
        z_lo = self.mer.zmin + 1.0e-3 if z_lo is None else z_lo
        z_hi = self.mer.zmax - 0.8e-3 if z_hi is None else z_hi
        zz = np.linspace(z_lo, z_hi, n_z)
        r = rfun(zz)
        ctrl = np.concatenate([[[0, z_hi]], np.stack([r[::-1], zz[::-1]], 1), [[0, z_lo]]])
        return self.bend.bend_mesh(ML.surface_of_revolution(Meridian(ctrl[:, 0], ctrl[:, 1]), n_th))

    def stub_meshes(self):
        """Cut petiole stubs on the crown: short tubes, the cut face at the trim height."""
        out = []
        zc = self.mer.zmax
        h = self.p["petiole_stub_length"] * 1e-3
        for k in range(self.n_stub):
            a = 2 * np.pi * k / self.n_stub + self.rng.normal(0, 0.2)
            rr = (0.12 + 0.18 * self.rng.random()) * self.D / 2
            base = np.array([rr * np.cos(a), rr * np.sin(a), zc - 1.5e-3])
            tip = base + np.array([0.15 * np.cos(a), 0.15 * np.sin(a), 1.0]) * h
            out.append(self.bend.bend_mesh(ML.tube(np.linspace(base, tip, 5), 1.0e-3 + 0.5e-3 * self.rng.random(),
                                                   10)))
        return out

    def scar_points(self):
        """Lateral-root scars in vertical rows (register 'lateral_root_rows')."""
        n_rows = int(self.p["lateral_root_rows"])
        a0 = self.rng.uniform(0, 2 * np.pi)
        pts = []
        for j in range(n_rows):
            a = a0 + 2 * np.pi * j / n_rows
            for zz in np.arange(self.mer.zmin + 0.12 * self.L, self.mer.zmax - 0.08 * self.L,
                                self.p["lateral_root_scar_spacing"] * 1e-3):
                zz += self.rng.normal(0, 1.5e-3)
                r = self.r_out(zz) * self.mod(np.array(a), np.array(zz))
                pts.append([r * np.cos(a), r * np.sin(a), zz])
        pts = np.array(pts)
        return self.bend.bend(pts), np.full(len(pts), 0.6e-3)

    def band_points(self):
        sp = self.p["lenticel_band_spacing"] * 1e-3
        pts = []
        for zz in np.arange(self.mer.zmin + 0.05 * self.L, self.mer.zmax - 0.03 * self.L, sp):
            for a in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                if self.rng.random() < 0.6:
                    r = self.r_out(zz) * self.mod(np.array(a), np.array(zz))
                    pts.append([r * np.cos(a), r * np.sin(a), zz])
        pts = np.array(pts)
        return self.bend.bend(pts), np.full(len(pts), 0.3e-3)

    def parts(self):
        cache = {}

        def c(name, fn):
            def g():
                if name not in cache:
                    cache[name] = fn()
                return cache[name]
            return g
        outer = c("o", lambda: self.surf(self.mer))
        inner = c("i", lambda: self.surf(self.m_peri))
        camb_o = c("co", lambda: self.radial_mesh(lambda z: self.r_xylem(z) + self.t_camb / 2))
        camb_i = c("ci", lambda: self.radial_mesh(lambda z: self.r_xylem(z) - self.t_camb / 2,
                                                   z_lo=self.mer.zmin + 1.6e-3, z_hi=self.mer.zmax - 1.4e-3))
        prim = c("pr", lambda: self.radial_mesh(lambda z: np.full_like(z, self.prim_r), n_th=24,
                                                 z_lo=self.mer.zmin + 0.06 * self.L, z_hi=self.mer.zmax - 2e-3))
        R = {t["name"]: t["density"] for t in TISSUES}
        p = self.p
        P = [
            Part("periderm", "Periderm", "food:Periderm", ("mesh", lambda: ML.hollow(outer(), inner())), "Skin",
                 "visible", "Thin protective skin (a few cork-cell layers); scrubbed or peeled off.", R["periderm"],
                 interior=False, attrs={"thickness_um": p["periderm_thickness"]}),
            Part("phloem", "SecondaryPhloem", "food:SecondaryPhloem",
                 ("mesh", lambda: ML.hollow(inner(), camb_o())), "Phloem", "visible after peeling",
                 "Orange outer tissue ('cortex' in the kitchen): most of the carotene and sugar.", R["phloem"]),
            Part("cambium", "VascularCambium", "food:VascularCambium",
                 ("mesh", lambda: ML.hollow(camb_o(), camb_i())), "Cambium", "visible after cutting",
                 "Thin pale ring between phloem and xylem: where the root splits when it is snapped or cracked.",
                 R["cambium"]),
            Part("xylem", "SecondaryXylem", "food:SecondaryXylem",
                 ("mesh", lambda: ML.hollow(camb_i(), prim())), "Xylem", "visible after cutting",
                 "Paler core with radial rays; small in Nantes carrots.", R["xylem"],
                 attrs={"radius_fraction": p["xylem_radius_fraction"]}),
            Part("primary_xylem", "PrimaryXylem", "food:PrimaryXylem", ("mesh", prim), "Xylem",
                 "visible after cutting", "Thin central strand.", R["primary_xylem"]),
            Part("crown", "Crown", "food:Crown", None, observability="visible",
                 doc="Shoulder at the top: leaf bases (trimmed), green where it grew above the soil."),
            Part("shoulder", "Crown/Shoulder", "food:Shoulder", None, observability="visible",
                 doc="The surface region of the crown (a GeomSubset of the periderm; green when sun-exposed).",
                 attrs={"green_length_mm": p["green_shoulder_length"]}),
            Part("stubs", "Crown/PetioleStubs", "food:PetioleStubs", None, observability="visible",
                 doc="Cut leaf-stalk bases."),
        ]
        for k in range(self.n_stub):
            P.append(Part(f"stub_{k + 1}", f"Crown/PetioleStubs/Stub_{k + 1}", "food:PetioleStub",
                          ("mesh", lambda k=k: c("stubs", self.stub_meshes)()[k]), "Stub", "visible", "", 600.0,
                          interior=False))
        P += [
            Part("root_tip", "RootTip", "food:RootTip", None, observability="visible",
                 doc="The tapering lower end ('tail'): trimmed off in the kitchen.", attrs={}),
            Part("lateral_root_scars", "LateralRootScars", "food:LateralRootScar", ("points", self.scar_points),
                 None, "visible", "Scars of fine lateral roots in vertical rows.", interior=False),
            Part("lenticel_bands", "LenticelBands", "food:LenticelBand", ("points", self.band_points), None,
                 "visible", "Horizontal lenticel rings across the skin.", interior=False),
        ]
        return P

    def skin_regions(self, C):
        S = self.bend.unbend(C)
        q = (S[:, 2] - self.mer.zmin) / self.L
        g = self.p["green_shoulder_length"] * 1e-3 / self.L
        return np.where(q > 1 - max(g, 0.03), "Shoulder", np.where(q < 0.15, "RootTip", "Body"))

    # ---------------------------------------------------------------- appearance
    def point_uv(self, U, V):
        k = np.linspace(0, 1, len(self.mer.rho))
        t = 1 - np.asarray(V)
        rho, z = np.interp(t, k, self.mer.rho), np.interp(t, k, self.mer.z)
        th = 2 * np.pi * np.asarray(U)
        m = self.mod(th, z)
        return self.bend.bend(np.stack([m * rho * np.cos(th), m * rho * np.sin(th), z], -1))

    def uv_of(self, P):
        from scipy.spatial import cKDTree
        S = self.bend.unbend(np.asarray(P, float))
        rn, th, z = self.norm_coords(S)
        k = np.linspace(0, 1, len(self.mer.rho))
        _, i = cKDTree(np.stack([self.mer.rho, self.mer.z], 1)).query(np.stack([rn, z], 1))
        return np.stack([np.mod(th, 2 * np.pi) / (2 * np.pi), 1 - k[i]], 1)

    def skin_albedo(self, P, seed):
        P = np.asarray(P, float)
        S = self.bend.unbend(P)
        p = self.p
        vn = ValueNoise3(seed + 81)
        col = np.broadcast_to(lch_to_srgb(*p["skin_LCh"]), P.shape).copy()
        col = col * (1 + 0.07 * vn.fbm(np.stack([S[..., 0] * 3, S[..., 1] * 3, S[..., 2] * 0.3], -1),
                                       3e-3, 3))[..., None]
        k_band = 2 * np.pi / (p["lenticel_band_spacing"] * 1e-3)
        band = (0.5 + 0.5 * np.cos(k_band * S[..., 2] + 2.0 * vn.fbm(np.stack(
            [np.cos(np.arctan2(S[..., 1], S[..., 0])) * 0.01, np.sin(np.arctan2(S[..., 1], S[..., 0])) * 0.01,
             S[..., 2]], -1), 0.01, 1))) ** 6
        brk = np.clip(vn.fbm(P + 3, 2.5e-3, 2) + 0.3, 0, 1)
        col = col * (1 - 0.30 * (band * brk)[..., None])
        sc, _ = self.scar_points()
        col = col * (1 - 0.35 * spots(P, sc, np.full(len(sc), 0.7e-3))[..., None])
        q = (S[..., 2] - self.mer.zmin) / self.L
        g = p["green_shoulder_length"] * 1e-3 / self.L
        if g > 0:
            gw = smoothstep(1 - g * 1.3, 1 - g * 0.5, q)[..., None]
            col = col * (1 - 0.7 * gw) + lch_to_srgb(45, 30, 110) * 0.7 * gw
        top = q > 0.995
        col[top] = lch_to_srgb(55, 25, 85)
        dirt = np.clip(vn.fbm(P + 11, 4e-3, 3) - 0.35, 0, 1) * p["soil_residue"]
        col = col * (1 - dirt[..., None]) + np.array([0.35, 0.28, 0.20]) * dirt[..., None]
        return np.clip(col, 0, 1)

    def section_rgb(self, lab, P, seed=0):
        rgb = np.zeros(lab.shape + (3,)) + 0.86
        vn = ValueNoise3(seed + 91)
        p = self.p
        for i, t in enumerate(TISSUES):
            rgb[lab == i + 1] = t["colour"]
        ph = lch_to_srgb(*p["phloem_LCh"])
        xy = lch_to_srgb(*p["xylem_LCh"])
        m = lab == TI["phloem"]
        rgb[m] = ph * (1 + 0.04 * vn.fbm(P[m], 0.4e-3, 2))[:, None]
        m = lab == TI["xylem"]
        if m.any():                                         # radial rays in the xylem
            S = self.bend.unbend(P[m])
            th = np.arctan2(S[:, 1], S[:, 0])
            ray = (0.5 + 0.5 * np.cos(th * 28 + 3 * vn.fbm(S, 2e-3, 2))) ** 4
            rgb[m] = xy * (1 + 0.06 * ray)[:, None]
        rgb[lab == TI["cambium"]] = lch_to_srgb(p["xylem_LCh"][0] + 8, p["xylem_LCh"][1] - 25, p["xylem_LCh"][2] + 5)
        rgb[lab == TI["periderm"]] = lch_to_srgb(*p["skin_LCh"])
        return np.clip(rgb, 0, 1)

    # ---------------------------------------------------------------- asset declaration
    def collision_groups(self, V):
        S = self.bend.unbend(V)
        edges = np.linspace(S[:, 2].min(), S[:, 2].max() + 1e-9, 5)
        return [V[(S[:, 2] >= edges[i] - 2e-3) & (S[:, 2] <= edges[i + 1] + 2e-3)] for i in range(4)]

    def asset_spec(self):
        from .carrot_models import carrot_beliefs, carrot_truth
        p = self.p
        lo, hi = self.bbox()
        cen = (lo + hi) / 2
        ext = float(np.max(hi - lo) * 0.56)
        zc = self.mer.zmin + 0.55 * self.L
        o = self.bend.bend(np.array([0.0, 0, zc]))
        tng = self.bend.bend(np.array([0.0, 0, zc + 1e-4])) - o
        tng /= np.linalg.norm(tng)
        e1 = np.array([0, 1.0, 0])
        e2 = np.cross(tng, e1)
        return {
            "build": {"h": 0.3e-3},
            "textures": [("skin.png", self.point_uv, self.skin_albedo, 2048, 1024)],
            "uv_fns": {"periderm": self.uv_of},
            "subsets": {"periderm": self.skin_regions},
            "materials": {
                "Skin": dict(diffuse_reflection_color_image="tex:skin.png", specular_reflection_roughness=0.55,
                             specular_reflection_weight=0.3, subsurface_weight=0.15, subsurface_scale=0.002,
                             subsurface_scattering_color=(1.0, 0.5, 0.15)),
                "Phloem": dict(diffuse_reflection_color=tuple(float(c) for c in lch_to_srgb(*p["phloem_LCh"])),
                               specular_reflection_roughness=0.3, subsurface_weight=0.3, subsurface_scale=0.002,
                               subsurface_scattering_color=(1.0, 0.5, 0.12)),
                "Cambium": dict(diffuse_reflection_color=(0.96, 0.78, 0.40), specular_reflection_roughness=0.3),
                "Xylem": dict(diffuse_reflection_color=tuple(float(c) for c in lch_to_srgb(*p["xylem_LCh"])),
                              specular_reflection_roughness=0.3, subsurface_weight=0.3, subsurface_scale=0.002,
                              subsurface_scattering_color=(1.0, 0.6, 0.2)),
                "Stub": dict(diffuse_reflection_color=(0.42, 0.45, 0.20), specular_reflection_roughness=0.6),
            },
            "section_material": {"specular_reflection_roughness": 0.28, "specular_reflection_weight": 0.5,
                                 "subsurface_weight": 0.25, "subsurface_scale": 0.002,
                                 "subsurface_scattering_color": (1.0, 0.55, 0.15)},
            "root": {"peel_state": "unpeeled", "trim_state": "untrimmed", "integrity": "whole",
                     "firmness_class": "crisp", "cooked_state": "raw"},
            "friction": p["friction_skin_rubber"], "restitution": 0.2,
            "collision_fn": self.collision_groups,
            "cuts": [
                {"name": "length", "origin": [cen[0], 0, cen[2]], "e1": [1, 0, 0], "e2": [0, 0, 1], "ext": ext,
                 "states": {"half_a": ("pos", {"integrity": "halved_lengthwise"}),
                            "half_b": ("neg", {"integrity": "halved_lengthwise"})}},
                {"name": "across", "origin": o, "e1": e1, "e2": e2, "ext": float(self.D * 0.7), "res": 1024,
                 "states": {"tip_piece": ("neg", {"integrity": "cut_across"}),
                            "crown_piece": ("pos", {"integrity": "cut_across"})}},
            ],
            "beliefs": lambda mp: carrot_beliefs(self, mp),
            "truth": lambda: carrot_truth(self),
        }
