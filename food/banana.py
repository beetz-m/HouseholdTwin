"""Cavendish banana twin (Musa acuminata AAA, one finger): a fleshy berry from an inferior ovary of three carpels.

Anatomy (docs/wave1/BANANA.md for the sources): the PEEL is the exocarp plus the outer mesocarp -- cuticle and
epidermis, chlorenchyma (the green, chlorophyll-bearing hypodermal layers that turn yellow), parenchyma with latex
vessels (laticifers) and vascular bundles, and a white inner aerenchyma that touches the pulp.  The PULP is the inner
mesocarp/endocarp of three fused carpels: it splits lengthwise into three segments along the carpel boundaries, round
a central axis (placenta) that carries the aborted ovules ('seeds', tiny dark specks in two rows per locule).
Phloem bundles ('strings') run lengthwise on the pulp surface and stay with peel or pulp when peeling.  The finger is
ridged (an angular cross-section that rounds out as it ripens), curved, ends in the pedicel (stalk/'neck') at the
stem end -- where it was cut from the crown of the hand -- and in the dried perianth remnant at the blossom end tip.

Geometry: the finger is modelled straight in (x, y, z) -- +z from the blossom end to the stalk -- with a ridged
cross-section, and bent onto an arc by food.shapes.Bend (concave side +x).  label() unbends points first, so meshes,
labels, masses and sections share one definition.

Partonomy (USD paths below /Food/Parts):
  Peel/{Epidermis, Chlorenchyma, Parenchyma, InnerAerenchyma}, Peel/Laticifers (points), Peel/Ridges (attributes)
  Stalk (pedicel), CrownCushion (cut remnant of the crown), BlossomTip (perianth remnant)
  Pulp/{Segment_1, Segment_2, Segment_3}, Pulp/CentralAxis, Pulp/Ovules/Ovule_k, Pulp/PhloemStrings/String_k
"""

import numpy as np

from . import meshlib as ML
from . import params as PR
from .geometry import Body, Meridian, resample, smooth_meridian
from .partgraph import Part
from .shapes import Bend, loft, smoothstep
from .texture import ValueNoise3, lch_to_srgb, sample_surface, spots

TISSUES = [
    {"name": "epidermis", "density": 1000.0, "colour": (0.85, 0.75, 0.20)},
    {"name": "chlorenchyma", "density": 1000.0, "colour": (0.80, 0.78, 0.30)},
    {"name": "peel_parenchyma", "density": 950.0, "colour": (0.93, 0.92, 0.80)},
    {"name": "aerenchyma", "density": 700.0, "colour": (0.97, 0.96, 0.90)},
    {"name": "pulp", "density": 1050.0, "colour": (0.96, 0.90, 0.68)},
    {"name": "central_axis", "density": 1050.0, "colour": (0.92, 0.84, 0.60)},
    {"name": "ovule", "density": 1100.0, "colour": (0.25, 0.18, 0.12)},
    {"name": "phloem_string", "density": 1050.0, "colour": (0.88, 0.80, 0.55)},
    {"name": "stalk", "density": 900.0, "colour": (0.50, 0.50, 0.25)},
    {"name": "blossom_tip", "density": 800.0, "colour": (0.20, 0.15, 0.10)},
]
TI = {t["name"]: i + 1 for i, t in enumerate(TISSUES)}


def peel_colour_stage(stage):
    """Peel colour along the 1..7 ripening scale as LCh (register 'peel_hue_by_stage'); interpolated."""
    S = np.array([1, 2, 3, 4, 5, 6, 7], float)
    L = np.array(PR.BANANA["peel_L_by_stage"][1])
    C = np.array(PR.BANANA["peel_C_by_stage"][1])
    h = np.array(PR.BANANA["peel_hue_by_stage"][1])
    s = float(np.clip(stage, 1, 7))
    return lch_to_srgb(np.interp(s, S, L), np.interp(s, S, C), np.interp(s, S, h))


class Banana(Body):
    tissues = TISSUES
    kind = "banana"

    def __init__(self, seed=0, overrides=None, stage=None):
        p = PR.sample("banana", seed)
        p.update(overrides or {})
        if stage is not None:
            p["ripeness_stage"] = float(stage)
        rng = np.random.default_rng(p["_rng_state"])
        self.p, self.seed, self.rng = p, seed, rng
        for t in TISSUES:                                   # register densities (per instance)
            if t["name"] in ("epidermis", "chlorenchyma", "peel_parenchyma"):
                t["density"] = p["peel_density"]
            elif t["name"] in ("pulp", "central_axis", "phloem_string"):
                t["density"] = p["pulp_density"]
        D = p["calibre"] * 1e-3                             # max diameter (calibre, across the ridges)
        # register length = convex face incl. pedicel (eu1333_2011); axis length of the finger proper (D016: DERIVED)
        L = (p["length_outer"] - p["stalk_length"]) * 1e-3 / (1 + D / 2 / (p["curvature_radius"] * 1e-3))
        self.L, self.D = L, D
        # meridian (straight frame), stalk end at +L/2: blossom tip -> full -> neck -> stalk base
        s = np.array([0.0, 0.015, 0.05, 0.12, 0.22, 0.5, 0.70, 0.82, 0.92, 1.0])
        r = np.array([0.0, 0.10, 0.30, 0.72, 0.93, 1.0, 0.96, 0.80, 0.52, 0.40]) * D / 2
        z = (s - 0.5) * L
        ctrl = np.stack([r[::-1], z[::-1]], 1)
        ctrl[0, 0] = 0.0
        rho, zz = smooth_meridian(ctrl, 500)
        self.n_ridge = int(p["ridges"])
        self.phi = rng.uniform(0, 2 * np.pi / self.n_ridge)
        st = float(p["ripeness_stage"])
        self.ang = p["angularity"] * (1 - 0.5 * smoothstep(3, 7, st))   # ridges soften as it ripens

        k = self.n_ridge
        aa = np.linspace(-np.pi / k, np.pi / k, 2001)
        self._pbar = float(np.mean(np.cos(np.pi / k) / np.cos(aa)))

        def mod(th, zc):
            a = ((th - self.phi) % (2 * np.pi / k)) - np.pi / k
            poly = np.cos(np.pi / k) / np.cos(a)            # polygon, inradius cos(pi/k), circumradius 1 (ridges)
            q = np.clip((zc + L / 2) / L, 0, 1)
            fade = smoothstep(0.02, 0.15, q) * (1 - 0.4 * smoothstep(0.8, 0.98, q))
            return 1 + self.ang * fade * (poly - self._pbar)
        super().__init__(Meridian(rho, zz), mod)
        self.bend = Bend(p["curvature_radius"] * 1e-3)
        self.t_peel = p["peel_thickness"] * 1e-3 * (1 - p["peel_thinning"] * smoothstep(1, 7, st))
        fr = p["peel_layer_fractions"]
        self.t_epi, self.t_chl = 0.04e-3, fr[0] * self.t_peel
        self.t_aer = fr[2] * self.t_peel
        self.axis_r = p["central_axis_diameter"] * 5e-4
        self.seg_phi = self.phi + np.pi / self.n_ridge + rng.uniform(0, 2 * np.pi / 3)
        self.stalk_len, self.stalk_w = p["stalk_length"] * 1e-3, p["stalk_width"] * 1e-3
        self._ovules = None

    # ---------------------------------------------------------------- straight-frame helpers
    def r_outer(self, th, z):
        rn = np.interp(z, self.mer.z[::-1], self.mer.rho[::-1])
        return rn * self.mod(th, z)

    def peel_t(self, z):
        q = np.clip((z + self.L / 2) / self.L, 0, 1)
        return self.t_peel * (1 + 0.6 * smoothstep(0.75, 1.0, q) + 0.5 * (1 - smoothstep(0.0, 0.12, q)))

    def r_pulp(self, z):
        """Pulp is round in cross-section (the peel absorbs the ridges)."""
        rn = np.interp(z, self.mer.z[::-1], self.mer.rho[::-1])
        return np.clip(rn * (1 - 0.5 * self.ang) - self.peel_t(z), 0, None)

    # ---------------------------------------------------------------- labels (bent world -> straight frame)
    def label(self, P):
        S = self.bend.unbend(np.asarray(P, float))
        x, y, z = S[..., 0], S[..., 1], S[..., 2]
        th, rho = np.arctan2(y, x), np.hypot(x, y)
        out = np.zeros(S.shape[:-1], np.int16)
        zt = self.mer.zmax
        stalk = (z >= zt - 1e-3) & (z <= zt + self.stalk_len) & (np.abs(x) < self.stalk_w / 2) & \
            (np.abs(y) < self.stalk_w / 2)
        out[stalk] = TI["stalk"]
        rin = (z >= self.mer.zmin) & (z <= zt)
        ro = np.where(rin, self.r_outer(th, z), 0)
        body = rin & (rho <= ro)
        rp = self.r_pulp(z)
        depth = ro - rho
        lab = np.full(S.shape[:-1], TI["peel_parenchyma"], np.int16)
        lab[depth < self.t_chl + self.t_epi] = TI["chlorenchyma"]
        lab[depth < self.t_epi] = TI["epidermis"]
        lab[(rho > rp) & (rho <= rp + self.t_aer)] = TI["aerenchyma"]
        pulp = rho <= rp
        lab[pulp] = TI["pulp"]
        ax = pulp & (rho < self.axis_r)
        lab[ax] = TI["central_axis"]
        # phloem strings: thin bundles on the pulp surface
        if self.p["phloem_strings"] > 0:
            k = int(self.p["phloem_strings"])
            phs = self.seg_phi + np.arange(k) * 2 * np.pi / k + np.pi / k
            dth = np.min(np.abs(((th[..., None] - phs) + np.pi) % (2 * np.pi) - np.pi), -1)
            strg = (np.abs(rho - rp) < 0.35e-3) & (dth * np.maximum(rp, 1e-4) < 0.45e-3) & rin
            lab[strg] = TI["phloem_string"]
        q = (z - self.mer.zmin) / self.L
        lab[q < 0.012] = TI["blossom_tip"]
        out[body] = lab[body]
        C, r = self.ovules()
        from scipy.spatial import cKDTree
        idx = np.nonzero(pulp & body)
        if len(idx[0]):
            d, i = cKDTree(C).query(S[idx])
            hit = d < r[i]
            out[tuple(a[hit] for a in idx)] = TI["ovule"]
        return out

    def tissue_of(self, rn, th, z, P):
        return self.label(P)

    def inside(self, P):
        return self.label(P) > 0

    def bbox(self):
        pts = self.bend.bend(np.array([[sx * self.D * 0.6, sy * self.D * 0.6, z]
                                       for sx in (-1, 1) for sy in (-1, 1)
                                       for z in np.linspace(self.mer.zmin, self.mer.zmax + self.stalk_len, 9)]))
        return pts.min(0) - 1e-3, pts.max(0) + 1e-3

    def ovules(self):
        """Aborted ovules (register 'ovules_total'): two rows per locule (three locules) along the central axis."""
        if self._ovules is None:
            n = max(1, int(self.p["ovules_total"]) // 6)
            zs = np.linspace(self.mer.zmin + 0.15 * self.L, self.mer.zmax - 0.18 * self.L, n)
            rng = np.random.default_rng(self.seed + 13)
            C = []
            for j in range(3):
                for side in (-1, 1):
                    a = self.seg_phi + j * 2 * np.pi / 3 + np.pi / 3 + side * 0.35
                    rr = self.axis_r + 0.8e-3
                    for zz in zs:
                        C.append([rr * np.cos(a), rr * np.sin(a), zz + rng.normal(0, 0.8e-3)])
            self._ovules = (np.array(C), np.full(len(C), self.p["ovule_diameter"] * 5e-4))
        return self._ovules

    def ovule_points(self):
        C, r = self.ovules()
        return self.bend.bend(C), 2 * r

    # ---------------------------------------------------------------- meshes (straight, then bent)
    def _surf(self, rfun, n_z=220, n_th=96, z0=None, z1=None):
        z0 = self.mer.zmin if z0 is None else z0
        z1 = self.mer.zmax if z1 is None else z1
        zz = np.linspace(z0, z1, n_z)
        th = np.linspace(0, 2 * np.pi, n_th, endpoint=False)
        T, Z = np.meshgrid(th, zz)
        R = rfun(T, Z)
        V = np.stack([R * np.cos(T), R * np.sin(T), Z], -1).reshape(-1, 3)
        f = []
        for i in range(n_z - 1):
            for j in range(n_th):
                a, b = i * n_th + j, i * n_th + (j + 1) % n_th
                f += [(a, b, a + n_th), (b, b + n_th, a + n_th)]
        s, e = len(V), len(V) + 1
        V = np.vstack([V, [0, 0, z0], [0, 0, z1]])
        f += [(s, (j + 1) % n_th, j) for j in range(n_th)]
        last = (n_z - 1) * n_th
        f += [(e, last + j, last + (j + 1) % n_th) for j in range(n_th)]
        return ML.tm(V, np.array(f))

    def bent(self, m):
        return self.bend.bend_mesh(m)

    def _zrange(self, rfun, lo, hi, r_min=0.35e-3):
        """z interval inside [lo, hi] where the layer surface rfun has radius > r_min all round (no fold-through)."""
        zz = np.linspace(lo, hi, 3000)
        th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        T, Z = np.meshgrid(th, zz)
        ok = zz[rfun(T, Z).min(1) > r_min]
        return ok[0], ok[-1]

    def peel_surfaces(self):
        """Outer face, the faces under epidermis and chlorenchyma, the aerenchyma's outer face, the pulp face.
        Nested closed surfaces must not share or cross their ends (a cut would not close): each inner surface
        ends inside the one around it."""
        zl, zh = self.mer.zmin + 0.012 * self.L, self.mer.zmax - 0.2e-3
        fns = [lambda T, Z: self.r_outer(T, Z),
               lambda T, Z: self.r_outer(T, Z) - self.t_epi,
               lambda T, Z: self.r_outer(T, Z) - self.t_epi - self.t_chl,
               lambda T, Z: self.r_pulp(Z) + self.t_aer + 0 * T,
               lambda T, Z: self.r_pulp(Z) + 0 * T]
        out, lo, hi = [], zl, zh
        for i, f in enumerate(fns):
            lo, hi = self._zrange(f, lo + (0.4e-3 if i else 0), hi - (0.4e-3 if i else 0))
            out.append(self._surf(f, z0=lo, z1=hi, n_th=120 if i < 3 else 96))
        self._pulp_z = (lo, hi)
        return tuple(out)

    def pulp_segment(self, k, pulp=None, n_arc=36, n_side=8, gap=0.15e-3):
        """Segment k of the pulp: the 120-degree carpel sector between two septa (0.3 mm gap), outside the central
        axis, lofted ring by ring along the axis (short edges, so it bends correctly)."""
        a0 = self.seg_phi + k * 2 * np.pi / 3
        zz = np.linspace(self.mer.zmin, self.mer.zmax, 3000)
        ok = zz[self.r_pulp(zz) > 1.2 * self.axis_r + 0.5e-3]
        if hasattr(self, "_pulp_z"):
            ok = ok[(ok > self._pulp_z[0] + 0.4e-3) & (ok < self._pulp_z[1] - 0.4e-3)]
        rings = []
        for z in np.linspace(ok[0], ok[-1], 200):
            rp = float(self.r_pulp(z))
            da = gap / rp
            arc = np.linspace(a0 + da, a0 + 2 * np.pi / 3 - da, n_arc)
            ri = self.axis_r
            # outer arc, then down the second septum to the axis, back along the axis, out along the first septum
            pts = [np.stack([rp * np.cos(arc), rp * np.sin(arc)], 1)]
            a1 = a0 + 2 * np.pi / 3
            n1 = np.array([-np.sin(a1), np.cos(a1)])
            n0 = np.array([-np.sin(a0), np.cos(a0)])
            for r in np.linspace(rp, ri, n_side)[1:]:
                pts.append([[r * np.cos(a1) - gap * n1[0], r * np.sin(a1) - gap * n1[1]]])
            for a in np.linspace(a1, a0, 6)[1:-1]:
                pts.append([[ri * np.cos(a), ri * np.sin(a)]])
            for r in np.linspace(ri, rp, n_side)[:-1]:
                pts.append([[r * np.cos(a0) + gap * n0[0], r * np.sin(a0) + gap * n0[1]]])
            P = np.vstack(pts)
            rings.append(np.column_stack([P, np.full(len(P), z)]))
        return loft(rings)

    def stalk_mesh(self):
        z0 = self.mer.zmax - 1.0e-3
        w = self.stalk_w / 2
        path = np.linspace([0, 0, z0], [0, 0, z0 + self.stalk_len], 10)
        return ML.tube(path, np.linspace(w * 1.05, w * 0.95, 10), 6)

    def crown_mesh(self):
        z = self.mer.zmax + self.stalk_len
        return ML.ellipsoid([0, 0, z + 0.8e-3], (self.stalk_w * 0.58, self.stalk_w * 0.66, 1.6e-3), n_u=10, n_v=20)

    def tip_mesh(self):
        return ML.ellipsoid([0, 0, self.mer.zmin + 0.8e-3], (1.6e-3, 1.6e-3, 1.8e-3), n_u=10, n_v=20)

    def string_path(self, k):
        n = int(self.p["phloem_strings"])
        a = self.seg_phi + k * 2 * np.pi / n + np.pi / n
        zz = np.linspace(self.mer.zmin + 0.04 * self.L, self.mer.zmax - 0.03 * self.L, 60)
        rr = self.r_pulp(zz) + 0.05e-3
        return np.stack([rr * np.cos(a), rr * np.sin(a), zz], 1)

    def laticifer_points(self):
        m = self._surf(lambda T, Z: self.r_outer(T, Z) - self.t_epi - self.t_chl - 0.3 * self.t_peel, 80, 48)
        pts, _ = ML.points_on(m, 1500, self.rng)
        return self.bend.bend(pts), np.full(len(pts), 0.05e-3)

    def parts(self):
        cache = {}

        def surf(i):
            if "s" not in cache:
                cache["s"] = self.peel_surfaces()
            return cache["s"][i]
        R = {t["name"]: t["density"] for t in TISSUES}
        B = self.bent
        P = [
            Part("peel", "Peel", "food:BananaPeel", None, observability="visible",
                 doc="Exocarp + outer mesocarp; green -> yellow -> brown flecks; ~30-40 % of fruit mass, falling as "
                     "it ripens; peels from the stalk or the tip in 3-5 strips along the ridges."),
            Part("epidermis", "Peel/Epidermis", "food:Epidermis", ("mesh", lambda: B(ML.hollow(surf(0), surf(1)))),
                 "Peel", "visible", "Cuticle + epidermis: the coloured, waxy outer face.", R["epidermis"],
                 interior=False),
            Part("chlorenchyma", "Peel/Chlorenchyma", "food:Chlorenchyma",
                 ("mesh", lambda: B(ML.hollow(surf(1), surf(2)))), "PeelSection", "hidden",
                 "Hypodermal layers with chloroplasts (green) that degreen to yellow (carotenoids unmasked).",
                 R["chlorenchyma"]),
            Part("peel_parenchyma", "Peel/Parenchyma", "food:PeelParenchyma",
                 ("mesh", lambda: B(ML.hollow(surf(2), surf(3)))), "PeelInner", "visible after peeling",
                 "Parenchyma with latex vessels and vascular bundles; the bulk of the peel.", R["peel_parenchyma"]),
            Part("aerenchyma", "Peel/InnerAerenchyma", "food:Aerenchyma",
                 ("mesh", lambda: B(ML.hollow(surf(3), surf(4)))), "PeelInner", "visible after peeling",
                 "White, spongy inner face of the peel touching the pulp.", R["aerenchyma"]),
            Part("laticifers", "Peel/Laticifers", "food:Laticifer", ("points", self.laticifer_points), None, "hidden",
                 "Latex vessels: the sticky sap that browns on cut peel."),
            Part("stalk", "Stalk", "food:Pedicel", ("mesh", lambda: B(self.stalk_mesh())), "Stalk", "visible",
                 "Pedicel ('neck'): snapped to start peeling from the stem end.", R["stalk"], interior=False,
                 attrs={"length_mm": self.p["stalk_length"]}),
            Part("crown_cushion", "CrownCushion", "food:CrownRemnant", ("mesh", lambda: B(self.crown_mesh())),
                 "Crown", "visible", "Cut face where the finger was separated from the crown of the hand.", 700.0,
                 interior=False),
            Part("blossom_tip", "BlossomTip", "food:PerianthRemnant", ("mesh", lambda: B(self.tip_mesh())),
                 "BlossomTip", "visible", "Dry dark tip: remnant of the flower (perianth, style).",
                 R["blossom_tip"], interior=False),
            Part("pulp_group", "Pulp", "food:BananaPulp", None, observability="visible after peeling",
                 doc="Three fused carpels: splits lengthwise into three segments; starch -> sugar on ripening."),
        ]
        for k in range(3):
            P.append(Part(f"segment_{k + 1}", f"Pulp/Segment_{k + 1}", "food:PulpSegment",
                          ("mesh", lambda k=k: B(self.pulp_segment(k))), "Pulp", "visible after peeling",
                          "One carpel's pulp.", R["pulp"]))
        P.append(Part("central_axis", "Pulp/CentralAxis", "food:Placenta",
                      ("mesh", lambda: B(ML.tube(np.linspace([0, 0, self.mer.zmin + 0.08 * self.L],
                                                             [0, 0, self.mer.zmax - 0.08 * self.L], 30),
                                                 self.axis_r, 16))), "Axis", "visible after cutting",
                      "Central placental axis carrying the ovules.", R["central_axis"]))
        P.append(Part("ovules", "Pulp/Ovules", "food:AbortedOvule", ("points", self.ovule_points), None,
                      "visible after cutting", "Tiny dark specks: 200-300 aborted ovules on the placenta (Cavendish "
                      "is seedless, parthenocarpic); guide points, painted on sections."))
        n = int(self.p["phloem_strings"])
        if n:
            P.append(Part("phloem_strings", "Pulp/PhloemStrings", "food:PhloemBundle", None,
                          observability="visible after peeling",
                          doc="'Strings' on the pulp surface: phloem bundles that often stay on the pulp."))
            for k in range(n):
                P.append(Part(f"string_{k + 1}", f"Pulp/PhloemStrings/String_{k + 1}", "food:PhloemBundle",
                              ("mesh", lambda k=k: B(ML.tube(self.string_path(k), 0.3e-3, 6))), "String",
                              "visible after peeling", "", R["phloem_string"]))
        return P

    # ---------------------------------------------------------------- appearance
    def point_uv(self, U, V):
        """Peel texture: u around the axis, v along it (straight frame), bent."""
        zl, zh = self.mer.zmin + 0.012 * self.L, self.mer.zmax
        z = zl + (zh - zl) * np.asarray(V)
        th = 2 * np.pi * np.asarray(U)
        r = self.r_outer(th, z)
        S = np.stack([r * np.cos(th), r * np.sin(th), z], -1)
        return self.bend.bend(S)

    def uv_of(self, P):
        S = self.bend.unbend(np.asarray(P, float))
        zl, zh = self.mer.zmin + 0.012 * self.L, self.mer.zmax
        return np.stack([np.mod(np.arctan2(S[..., 1], S[..., 0]), 2 * np.pi) / (2 * np.pi),
                         np.clip((S[..., 2] - zl) / (zh - zl), 0, 1)], -1)

    def skin_albedo(self, P, seed):
        """Peel by ripeness stage: base colour (register LCh), green tinge at the ends early, darker ridges, sugar
        spots (senescent spotting) from stage 6, bruise-free; fine lengthwise streaks."""
        P = np.asarray(P, float)
        S = self.bend.unbend(P)
        st = float(self.p["ripeness_stage"])
        vn = ValueNoise3(seed + 51)
        col = np.broadcast_to(peel_colour_stage(st), P.shape).copy()
        q = np.clip((S[..., 2] - self.mer.zmin) / self.L, 0, 1)
        ends = np.maximum(1 - smoothstep(0.0, 0.18, q), smoothstep(0.80, 1.0, q))
        green = peel_colour_stage(max(1.0, st - 2.0))
        col = col * (1 - 0.6 * ends[..., None]) + green * 0.6 * ends[..., None]
        th = np.arctan2(S[..., 1], S[..., 0])
        k = self.n_ridge
        a = np.abs(((th - self.phi) % (2 * np.pi / k)) - np.pi / k)
        ridge = np.exp(-(a / 0.06) ** 2) * self.ang / max(self.p["angularity"], 1e-6)
        col = col * (1 - 0.12 * ridge[..., None])
        streak = vn.fbm(np.stack([S[..., 0] * 6, S[..., 1] * 6, S[..., 2] * 0.3], -1), 2e-3, 3)
        col = col * (1 + 0.05 * streak[..., None])
        if st > 5.5:
            rng = np.random.default_rng(seed + 52)
            n = int(self.p["sugar_spot_density"] * (st - 5.5) * 2e4 * self.L * self.D)
            m = self._surf(lambda T, Z: self.r_outer(T, Z), 60, 36)
            C = self.bend.bend(sample_surface(np.asarray(m.vertices), np.asarray(m.faces), max(n, 1), rng))
            R = rng.lognormal(np.log(0.7e-3), 0.5, len(C))
            sp = spots(P, C, R)
            brown = lch_to_srgb(30, 18, 60)
            col = col * (1 - sp[..., None]) + brown * sp[..., None]
        tip = (q < 0.02) | (q > 0.985)
        col[tip] = lch_to_srgb(25, 10, 70)
        return np.clip(col, 0, 1)

    def section_rgb(self, lab, P, seed=0):
        rgb = np.zeros(lab.shape + (3,)) + 0.86
        vn = ValueNoise3(seed + 61)
        for i, t in enumerate(TISSUES):
            rgb[lab == i + 1] = t["colour"]
        st = float(self.p["ripeness_stage"])
        rgb[lab == TI["epidermis"]] = peel_colour_stage(st)
        rgb[lab == TI["chlorenchyma"]] = peel_colour_stage(max(1.0, st - 1.5)) * 1.05
        rgb[lab == TI["peel_parenchyma"]] = np.array([0.90, 0.91, 0.78])   # pale greenish-white peel flesh
        rgb[lab == TI["aerenchyma"]] = np.array([0.95, 0.95, 0.88])
        pulp_c = lch_to_srgb(*self.p["pulp_LCh"])
        m = lab == TI["pulp"]
        rgb[m] = pulp_c * (1 + 0.03 * vn.fbm(P[m], 0.5e-3, 2))[:, None]
        if m.any():                                         # septa: faint radial lines between the three carpels
            S = self.bend.unbend(P[m])
            th = np.arctan2(S[:, 1], S[:, 0])
            rho = np.hypot(S[:, 0], S[:, 1])
            rp = np.maximum(self.r_pulp(S[:, 2]), 1e-4)
            d = np.min(np.abs(((th[:, None] - (self.seg_phi + np.arange(3) * 2 * np.pi / 3)) + np.pi) % (2 * np.pi)
                              - np.pi), 1) * rho
            sep = np.exp(-(d / 0.25e-3) ** 2) * (rho < 0.75 * rp)
            rgb[m] = rgb[m] * (1 - 0.10 * sep[:, None])
            core = np.exp(-(rho / (self.axis_r * 1.6)) ** 2)
            rgb[m] = rgb[m] * (1 - 0.06 * core[:, None])
        rgb[lab == TI["central_axis"]] = pulp_c * 0.93
        return np.clip(rgb, 0, 1)

    # ---------------------------------------------------------------- asset declaration
    def collision_groups(self, V):
        """Convex decomposition along the arc: 6 slabs of the unbent axial coordinate."""
        S = self.bend.unbend(V)
        edges = np.linspace(S[:, 2].min(), S[:, 2].max() + 1e-9, 7)
        return [V[(S[:, 2] >= edges[i] - 2e-3) & (S[:, 2] <= edges[i + 1] + 2e-3)] for i in range(6)]

    def asset_spec(self):
        from .banana_models import banana_beliefs, banana_truth
        p = self.p
        q = 0.5                                              # cut across the middle of the finger
        zc = self.mer.zmin + q * self.L
        o = self.bend.bend(np.array([0.0, 0, zc]))
        tng = self.bend.bend(np.array([0.0, 0, zc + 1e-4])) - o
        tng /= np.linalg.norm(tng)
        e1 = np.array([0, 1.0, 0])
        e2 = np.cross(tng, e1)
        lo, hi = self.bbox()
        ext = float(np.max(hi - lo) * 0.56)
        # longitudinal cut: plane containing the bent axis (x-z plane), normal y
        cen = (lo + hi) / 2
        return {
            "build": {"h": 0.3e-3},
            "textures": [("skin.png", self.point_uv, self.skin_albedo, 2048, 1024)],
            "uv_fns": {"epidermis": self.uv_of},
            "materials": {
                "Peel": dict(diffuse_reflection_color_image="tex:skin.png", specular_reflection_roughness=0.42,
                             specular_reflection_weight=0.4, subsurface_weight=0.05),
                "PeelSection": dict(diffuse_reflection_color=tuple(float(c) for c in peel_colour_stage(
                    max(1.0, p["ripeness_stage"] - 1.5))), specular_reflection_roughness=0.6),
                "PeelInner": dict(diffuse_reflection_color=(0.95, 0.94, 0.86), specular_reflection_roughness=0.75),
                "Stalk": dict(diffuse_reflection_color=tuple(float(c) for c in peel_colour_stage(
                    max(1.0, p["ripeness_stage"] - 2.5)) * 0.8), specular_reflection_roughness=0.6),
                "Crown": dict(diffuse_reflection_color=(0.22, 0.17, 0.09), specular_reflection_roughness=0.85),
                "BlossomTip": dict(diffuse_reflection_color=(0.12, 0.09, 0.06), specular_reflection_roughness=0.8),
                "Pulp": dict(diffuse_reflection_color=tuple(float(c) for c in lch_to_srgb(*p["pulp_LCh"])),
                             specular_reflection_roughness=0.35, subsurface_weight=0.3, subsurface_scale=0.003,
                             subsurface_scattering_color=(0.97, 0.90, 0.65)),
                "Axis": dict(diffuse_reflection_color=(0.90, 0.82, 0.58), specular_reflection_roughness=0.4),
                "Ovule": dict(diffuse_reflection_color=(0.18, 0.13, 0.09), specular_reflection_roughness=0.5),
                "String": dict(diffuse_reflection_color=(0.86, 0.78, 0.52), specular_reflection_roughness=0.5),
            },
            "section_material": {"specular_reflection_roughness": 0.35, "subsurface_weight": 0.2,
                                 "subsurface_scale": 0.002, "subsurface_scattering_color": (0.97, 0.9, 0.65)},
            "root": {"ripeness_stage_class": str(int(round(p["ripeness_stage"]))), "peel_state": "on",
                     "integrity": "whole", "attached_to_hand": "no"},
            "friction": p["friction_peel_rubber"], "restitution": 0.1,
            "collision_fn": self.collision_groups,
            "cuts": [
                {"name": "length", "origin": [cen[0], 0, cen[2]], "e1": [1, 0, 0], "e2": [0, 0, 1], "ext": ext,
                 "states": {"half_a": ("pos", {"integrity": "halved_lengthwise"}),
                            "half_b": ("neg", {"integrity": "halved_lengthwise"})}},
                {"name": "across", "origin": o, "e1": e1, "e2": e2, "ext": float(self.D * 0.75),
                 "res": 1024,
                 "states": {"tip_half": ("neg", {"integrity": "cut_across"}),
                            "stalk_half": ("pos", {"integrity": "cut_across"})}},
            ],
            "beliefs": lambda mp: banana_beliefs(self, mp),
            "truth": lambda: banana_truth(self),
        }
