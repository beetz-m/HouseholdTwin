"""Apple twin (Malus domestica, Gala-like): anatomy after Wikipedia (Apple, Pome), Front. Plant Sci. 2015 (vascular
system, porosity), Konarska 2006 (skin) and USPTO apple plant patents (core, locules, stem) -- docs/ANATOMY.md.

Frame: +z from the calyx ("eye", bottom, resting) to the stem (top), origin at mid-height.
The edible flesh is hypanthium/receptacle tissue (cortex); the core inside the core line is the fused carpels.
Five carpels lie on the SEPAL radii (the dorsal carpellary bundles branch from the sepal bundles), each a leathery
endocarp around an elongated locule with 0-2 seeds pointing towards the stem.

Partonomy (USD paths below /Food/Parts; every prim is an SDT node):
  Skin/{Cuticle (with epicuticular wax), Epidermis, Hypodermis}, Skin/Lenticels (points), Skin GeomSubsets for
  surface regions (StemCavity, CalyxBasin, Shoulder, Cheek, Russet)
  Stem (pedicel)
  Calyx/{Sepal_1..5, CalyxTube, StamenRemnants}
  Cortex (hypanthium flesh)
  VascularSystem/{SepalBundle_1..5, PetalBundle_1..5}   on the core line
  Core/Pith (carpellary tissue inside the core line), Core/Carpel_k/{Endocarp, Locule, Seed_j}
"""

import numpy as np

from . import meshlib as ML
from . import params as PR
from .geometry import Body, Meridian, resample, revolve, smooth_meridian
from .partgraph import Part
from .texture import ValueNoise3, sample_surface, spots

TISSUES = [
    {"name": "cuticle", "density": 1000.0, "colour": (0.72, 0.16, 0.10)},
    {"name": "epidermis", "density": 1000.0, "colour": (0.78, 0.30, 0.18)},
    {"name": "hypodermis", "density": 950.0, "colour": (0.90, 0.75, 0.52)},
    {"name": "cortex", "density": 860.0, "colour": (0.90, 0.84, 0.60)},
    {"name": "vascular_bundle", "density": 950.0, "colour": (0.84, 0.84, 0.58)},
    {"name": "pith", "density": 920.0, "colour": (0.87, 0.85, 0.58)},
    {"name": "endocarp", "density": 1000.0, "colour": (0.94, 0.91, 0.79)},
    {"name": "locule", "density": 1.2, "colour": (0.86, 0.81, 0.64)},
    {"name": "seed", "density": 1100.0, "colour": (0.28, 0.13, 0.05)},
    {"name": "calyx_tube", "density": 700.0, "colour": (0.62, 0.50, 0.32)},
]
TI = {t["name"]: i + 1 for i, t in enumerate(TISSUES)}

# meridian control points in units of R (radius at the widest point); z scaled by the height ratio
CTRL = [(0.00, 0.70), (0.08, 0.72), (0.20, 0.79), (0.34, 0.89), (0.48, 0.97), (0.60, 1.00), (0.76, 0.96),
        (0.90, 0.80), (0.98, 0.52), (1.00, 0.20), (0.97, -0.15), (0.89, -0.48), (0.76, -0.76), (0.62, -0.94),
        (0.50, -1.00), (0.36, -0.97), (0.22, -0.90), (0.10, -0.85), (0.00, -0.84)]
N_MER, N_THETA = 200, 160


class Apple(Body):
    tissues = TISSUES
    kind = "apple"

    def __init__(self, seed=0, overrides=None):
        p = PR.sample("apple", seed)
        p.update(overrides or {})
        rng = np.random.default_rng(p["_rng_state"])
        self.p, self.seed, self.rng = p, seed, rng
        R = p["diameter_D"] * 1e-3 / 2
        hz = p["height_over_D"] * R
        c = np.array(CTRL, float)
        rho, z = smooth_meridian(np.stack([c[:, 0] * R, c[:, 1] * hz], 1), 500)
        self.R, self.hz = R, hz
        self.phi5 = rng.uniform(0, 2 * np.pi / 5)            # azimuth of sepal 1 (= carpel 1, sepal bundle 1)
        self.phi1 = rng.uniform(0, 2 * np.pi)
        self.sun = rng.uniform(0, 2 * np.pi)
        a5, a1 = p["lobe_amplitude"], p["lopsidedness"]

        def mod(th, zz):
            w5 = np.clip((-zz / hz - 0.05) / 0.85, 0, 1) ** 1.5
            return (1 + a5 * np.cos(5 * (th - self.phi5)) * (0.35 + 0.65 * w5)
                    + a1 * np.cos(th - self.phi1) * (zz / hz) + 0.012 * np.cos(2 * (th - self.phi1)))
        super().__init__(Meridian(rho, z), mod)
        # skin layers (konarska2006: 33-73 um; cuticle + 1 epidermal layer + 2-3 hypodermal layers)
        t = p["skin_thickness"] * 1e-6
        self.t_cut, self.t_epi, self.t_hyp = 0.20 * t, 0.30 * t, 0.50 * t
        self.m_c = self.mer.offset(self.t_cut)
        self.m_e = self.mer.offset(self.t_cut + self.t_epi)
        self.m_skin = self.mer.offset(t)
        # core: spindle bounded by the core line, closing towards the stem base and the calyx tube
        self.z_top, self.z_bot = self.mer.z[0], self.mer.z[-1]
        self.r_cl = min(p["core_line_radius"] * 1e-3, 0.45 * R)
        L_core = p["core_length"] * 1e-3
        self.zc = 0.5 * (self.z_top + self.z_bot) + 0.02 * hz
        self.h_up = min(0.55 * L_core, self.z_top - self.zc - 1.5e-3)
        self.h_dn = min(0.45 * L_core, self.zc - self.z_bot - 1.5e-3)
        # carpels on the sepal radii
        self.sepal_theta = self.phi5 + 2 * np.pi * np.arange(5) / 5
        self.loc_theta = self.sepal_theta.copy()
        self.petal_theta = self.sepal_theta + np.pi / 5
        self.loc_h = p["locule_length"] * 1e-3 / 2
        self.loc_ar = p["locule_width"] * 1e-3 / 2
        self.loc_at = 0.55 * self.loc_ar
        self.loc_z = self.zc
        self.loc_r = 1.6e-3 + self.loc_ar
        self.t_endo = PR.APPLE["endocarp_thickness"][1] * 1e-3
        self.r_vb = PR.APPLE["vascular_bundle_radius"][1] * 1e-3
        self.seeds = self._place_seeds()
        # calyx tube with stamen and style remnants, from the calyx basin into the core
        self.tube_h = min(0.35 * (self.zc - self.z_bot), 7e-3)

    # ---------------------------------------------------------------- geometry definitions
    def spindle(self, z):
        """Core-line radius at height z."""
        u = np.where(z >= self.zc, (z - self.zc) / self.h_up, (self.zc - z) / self.h_dn)
        return self.r_cl * np.sqrt(np.clip(1 - u ** 2, 0, 1))

    def locule_centre(self, zz):
        return self.loc_r * (1 - 0.45 * np.clip(zz / self.loc_h, -1, 1) ** 2)

    def locule_e(self, rn, th, z, tk):
        zz = z - self.loc_z
        x = rn * np.cos(th - tk) - self.locule_centre(zz)
        y = rn * np.sin(th - tk)
        at = self.loc_at * (0.45 + 0.55 * np.clip((x + self.loc_ar) / (2 * self.loc_ar), 0, 1))
        return (x / self.loc_ar) ** 2 + (y / at) ** 2 + (zz / self.loc_h) ** 2

    def wall_scale(self):
        return (1 + self.t_endo / self.loc_at) ** 2

    def _place_seeds(self):
        rng = self.rng
        seeds = []
        for k, tk in enumerate(self.loc_theta):
            n = rng.choice([0, 1, 2], p=[0.1, 0.4, 0.5])
            for j in range(n):
                dz = ((0.28 if j == 0 else -0.30) if n == 2 else rng.uniform(-0.1, 0.1)) * self.loc_h
                L = rng.normal(self.p["seed_length"], 0.3) * 1e-3
                L = min(L, 1.15 * self.loc_h)
                seeds.append({"carpel": k, "index": j, "theta": tk + rng.normal(0, 0.02), "z": self.loc_z + dz,
                              "r": self.locule_centre(dz) + 0.15e-3,
                              "len": L, "wid": min(self.p["seed_width"] * 1e-3, 1.7 * self.loc_at),
                              "thk": min(self.p["seed_thickness"] * 1e-3, 1.6 * self.loc_ar * 0.6),
                              "tilt": 0.30 if dz >= 0 else 0.18})
        return seeds

    def seed_frame(self, s):
        er = np.array([np.cos(s["theta"]), np.sin(s["theta"]), 0])
        et = np.array([-np.sin(s["theta"]), np.cos(s["theta"]), 0])
        ez = np.array([0, 0, 1.0])
        ca, sa = np.cos(s["tilt"]), np.sin(s["tilt"])
        ez, er = ca * ez - sa * er, sa * ez + ca * er             # tip leans towards the axis, pointing to the stem
        c = np.array([s["r"] * np.cos(s["theta"]), s["r"] * np.sin(s["theta"]), s["z"]])
        return c, np.stack([et, er, ez], 1)                       # local x = width, y = thickness, z = length

    @staticmethod
    def seed_taper(zn):
        return 1 - 0.55 * np.clip(zn, 0, 1)

    def bundle_path(self, th, n=80):
        """Primary bundle on the core line from the stem base to the calyx tube."""
        z = np.linspace(self.zc + 0.97 * self.h_up, self.zc - 0.97 * self.h_dn, n)
        r = np.maximum(self.spindle(z), 0.6e-3)
        return np.stack([r * np.cos(th), r * np.sin(th), z], 1)

    # ---------------------------------------------------------------- labels (same definitions as the meshes)
    def tissue_of(self, rn, th, z, P):
        out = np.full(len(rn), TI["cortex"], np.int16)
        in_c, in_e, in_s = self.m_c.contains(rn, z), self.m_e.contains(rn, z), self.m_skin.contains(rn, z)
        out[~in_c] = TI["cuticle"]
        out[in_c & ~in_e] = TI["epidermis"]
        out[in_e & ~in_s] = TI["hypodermis"]
        flesh = in_s
        core = flesh & (rn < self.spindle(z))
        out[core] = TI["pith"]
        for tk in self.loc_theta:
            e = self.locule_e(rn, th, z, tk)
            out[flesh & (e < self.wall_scale())] = TI["endocarp"]
            out[flesh & (e < 1)] = TI["locule"]
        for tk in np.concatenate([self.sepal_theta, self.petal_theta]):
            d = np.abs(np.angle(np.exp(1j * (th - tk)))) * rn
            rc = self.spindle(z)
            vb = flesh & (np.hypot(rn - np.maximum(rc, 0.6e-3), d) < self.r_vb) & \
                (z < self.zc + 0.97 * self.h_up) & (z > self.zc - 0.97 * self.h_dn)
            out[vb] = TI["vascular_bundle"]
        for s in self.seeds:
            c, Rm = self.seed_frame(s)
            q = (P - c) @ Rm
            zn = q[:, 2] / (s["len"] / 2)
            tp = self.seed_taper(zn)
            e = (q[:, 0] / (s["wid"] / 2 * tp)) ** 2 + (q[:, 1] / (s["thk"] / 2 * tp)) ** 2 + zn ** 2
            out[(e < 1) & flesh] = TI["seed"]
        tube = flesh & (z < self.z_bot + self.tube_h) & (rn < 1.2e-3 * (1 - 0.55 * (z - self.z_bot) / self.tube_h))
        out[tube] = TI["calyx_tube"]
        return out

    def mass_properties(self, h=0.4e-3, clip=None):
        d = self.p["flesh_density"]
        return super().mass_properties(h, densities={"cortex": d, "pith": d + 60.0}, clip=clip)

    # ---------------------------------------------------------------- meshes
    def surf(self, mer, n_theta=N_THETA, mod=True):
        r, z = resample(mer.rho, mer.z, N_MER)
        r[0] = r[-1] = 0
        return ML.surface_of_revolution(Meridian(r, z), n_theta, modulation=self.mod if mod else None)

    def locule_mesh(self, tk, scale=1.0, n_u=36, n_v=48):
        """Closed surface e = scale of carpel k (scale > 1 = outer endocarp face)."""
        u = np.linspace(0, np.pi, n_u + 1)[1:-1]
        v = np.linspace(0, 2 * np.pi, n_v, endpoint=False)
        U, V = np.meshgrid(u, v, indexing="ij")
        s = np.sqrt(scale)
        zz = self.loc_h * s * np.cos(U)
        x = self.loc_ar * s * np.sin(U) * np.cos(V)
        at = self.loc_at * (0.45 + 0.55 * np.clip((x + self.loc_ar) / (2 * self.loc_ar), 0, 1))
        y = at * s * np.sin(U) * np.sin(V)
        r = self.locule_centre(zz) + x
        ca, sa = np.cos(tk), np.sin(tk)
        pts = np.stack([r * ca - y * sa, r * sa + y * ca, self.loc_z + zz], -1).reshape(-1, 3)
        top = [self.locule_centre(self.loc_h * s) * ca, self.locule_centre(self.loc_h * s) * sa, self.loc_z + self.loc_h * s]
        bot = [self.locule_centre(-self.loc_h * s) * ca, self.locule_centre(-self.loc_h * s) * sa,
               self.loc_z - self.loc_h * s]
        pts = np.vstack([top, pts, bot])
        f = []
        nr = n_u - 1
        for j in range(n_v):
            f.append((0, 1 + j, 1 + (j + 1) % n_v))
        for i in range(nr - 1):
            for j in range(n_v):
                a, b = 1 + i * n_v + j, 1 + i * n_v + (j + 1) % n_v
                f += [(a, a + n_v, b), (b, a + n_v, b + n_v)]
        last, base = len(pts) - 1, 1 + (nr - 1) * n_v
        for j in range(n_v):
            f.append((base + j, last, base + (j + 1) % n_v))
        return ML.tm(pts, np.array(f))

    def seed_mesh(self, s):
        c, Rm = self.seed_frame(s)
        return ML.ellipsoid(c, (s["wid"] / 2, s["thk"] / 2, s["len"] / 2), rot=Rm, n_u=24, n_v=36,
                            taper=self.seed_taper)

    def spindle_mesh(self):
        z = np.linspace(self.zc + self.h_up, self.zc - self.h_dn, 160)
        return ML.surface_of_revolution(Meridian(self.spindle(z), z), 96)

    def stem_mesh(self):
        rng = np.random.default_rng(self.seed + 101)
        L = self.p["stem_length"] * 1e-3
        r0 = self.p["stem_diameter"] * 1e-3 / 2
        bend, az = rng.uniform(0.15, 0.45), rng.uniform(0, 2 * np.pi)
        s = np.linspace(0, 1, 24)
        base = np.array([0, 0, self.z_top - 1.0e-3])
        path = base + np.stack([np.sin(bend * s) * L * s * 0.5 * np.cos(az), np.sin(bend * s) * L * s * 0.5 * np.sin(az),
                                L * s], 1)
        r = r0 * (1 - 0.2 * s) + 0.45e-3 * np.exp(-((1 - s) / 0.06) ** 2)      # flared abscission end
        return ML.tube(path, r, 14)

    def sepal_mesh(self, k):
        th = self.sepal_theta[k]
        c = np.array([2.4e-3 * np.cos(th), 2.4e-3 * np.sin(th), self.z_bot - 0.15e-3])
        ca, sa = np.cos(th), np.sin(th)
        return ML.ellipsoid(c, (2.4e-3, 1.0e-3, 0.4e-3), rot=np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]]),
                            n_u=12, n_v=24)

    def calyx_tube_mesh(self):
        z = np.linspace(self.z_bot + self.tube_h, self.z_bot - 0.2e-3, 30)
        r = 1.2e-3 * (1 - 0.55 * (z - self.z_bot) / self.tube_h)
        r[0] = 0
        return ML.surface_of_revolution(Meridian(np.append(r, 0), np.append(z, z[-1])), 32)

    def stamen_remnants_mesh(self):
        rng = np.random.default_rng(self.seed + 7)
        tubes = []
        for k in range(15):                                    # ~15-20 dried filaments in the eye
            a = 2 * np.pi * k / 15 + rng.normal(0, 0.1)
            r0 = 0.9e-3
            p0 = np.array([r0 * np.cos(a), r0 * np.sin(a), self.z_bot + 0.3e-3])
            p1 = p0 + np.array([0.6e-3 * np.cos(a), 0.6e-3 * np.sin(a), -1.6e-3 + rng.normal(0, 0.2e-3)])
            tubes.append(ML.tube(np.linspace(p0, p1, 6), 0.12e-3, 6))
        return ML.union_meshes(tubes)

    def lenticel_points(self):
        m = self.surf(self.mer, 96)
        n = int(self.p["lenticel_density"] * m.area * 1e4)
        zmin, zmax = self.mer.zmin, self.mer.zmax
        p, nrm = ML.points_on(m, n, self.rng, weight=lambda q: 1.6 - 0.6 * (q[:, 2] - zmin) / (zmax - zmin))
        return p, self.rng.lognormal(np.log(0.3e-3), 0.3, len(p))

    def parts(self):
        S = self.surf
        cache = {}

        def c(name, fn):
            def g():
                if name not in cache:
                    cache[name] = fn()
                return cache[name]
            return g
        outer, mc, me, ms = (c(k, lambda m=m: S(m)) for k, m in
                             (("o", self.mer), ("c", self.m_c), ("e", self.m_e), ("s", self.m_skin)))
        spin = c("spin", self.spindle_mesh)
        endo_out = {k: c(f"eo{k}", lambda k=k: self.locule_mesh(self.loc_theta[k], self.wall_scale())) for k in range(5)}
        endo_in = {k: c(f"ei{k}", lambda k=k: self.locule_mesh(self.loc_theta[k], 1.0)) for k in range(5)}
        seedm = {(s["carpel"], s["index"]): c(f"s{s['carpel']}{s['index']}", lambda s=s: self.seed_mesh(s))
                 for s in self.seeds}
        d = self.p["flesh_density"]
        R = {t["name"]: t["density"] for t in TISSUES}
        P = [
            Part("skin", "Skin", "food:AppleSkin", None, observability="visible",
                 doc="Hypanthial epidermis: cuticle with epicuticular wax, one epidermal layer, 2-3 hypodermal layers; "
                     "33-73 um (konarska2006); tensile strength ~2.6 MPa."),
            Part("cuticle", "Skin/Cuticle", "food:Cuticle", ("mesh", lambda: ML.hollow(outer(), mc())), "Skin",
                 "visible", "Cuticle with epicuticular wax (gloss, water barrier).", R["cuticle"], interior=False),
            Part("epidermis", "Skin/Epidermis", "food:Epidermis", ("mesh", lambda: ML.hollow(mc(), me())), "SkinSection",
                 "hidden", "One cell layer; almost no air space (frontiers2015).", R["epidermis"]),
            Part("hypodermis", "Skin/Hypodermis", "food:Hypodermis", ("mesh", lambda: ML.hollow(me(), ms())),
                 "SkinSection", "hidden", "2-3 cell layers carrying the anthocyanin blush.", R["hypodermis"]),
            Part("lenticels", "Skin/Lenticels", "food:Lenticel", ("points", self.lenticel_points), None, "visible",
                 "Former stomata, 4.3-12.6 per cm2 (ishs_1099_97)."),
            Part("stem", "Stem", "food:Pedicel", ("mesh", self.stem_mesh), "Stem", "visible",
                 "Pedicel 10-28 mm x 2.7-3 mm (uspto); a grasp/orientation feature.", 900.0, interior=False),
            Part("calyx", "Calyx", "food:Calyx", None, observability="visible",
                 doc="The 'eye': persistent sepals with shrivelled stamens and styles (wiki_apple)."),
        ]
        for k in range(5):
            P.append(Part(f"sepal_{k + 1}", f"Calyx/Sepal_{k + 1}", "food:Sepal", ("mesh", lambda k=k: self.sepal_mesh(k)),
                          "Sepal", "visible", "", 600.0, interior=False))
        P += [
            Part("calyx_tube", "Calyx/CalyxTube", "food:CalyxTube", ("mesh", self.calyx_tube_mesh), "CalyxTube",
                 "hidden", "Floral tube from the eye into the core.", R["calyx_tube"]),
            Part("stamen_remnants", "Calyx/StamenRemnants", "food:StamenRemnant", ("mesh", self.stamen_remnants_mesh),
                 "Sepal", "visible", "Dried filaments in the eye.", 600.0, interior=False),
            Part("cortex", "Cortex", "food:Cortex", ("mesh", lambda: ML.hollow(ms(), spin())), "Flesh",
                 "visible after cutting", "Hypanthium flesh; porosity 10.6 % (centre) .. 26.4 % (periphery) "
                 "(frontiers2015); the edible part.", d),
            Part("vascular_system", "VascularSystem", "food:VascularSystem", None, observability="visible after cutting",
                 doc="10 primary bundles (5 sepal + 5 petal) on the core line (frontiers2015)."),
        ]
        for k in range(5):
            P.append(Part(f"sepal_bundle_{k + 1}", f"VascularSystem/SepalBundle_{k + 1}", "food:SepalBundle",
                          ("mesh", lambda k=k: ML.tube(self.bundle_path(self.sepal_theta[k]), self.r_vb, 8)),
                          "Bundle", "visible after cutting", "On the radius of carpel and sepal %d." % (k + 1),
                          R["vascular_bundle"]))
            P.append(Part(f"petal_bundle_{k + 1}", f"VascularSystem/PetalBundle_{k + 1}", "food:PetalBundle",
                          ("mesh", lambda k=k: ML.tube(self.bundle_path(self.petal_theta[k]), self.r_vb, 8)),
                          "Bundle", "visible after cutting", "", R["vascular_bundle"]))
        P += [
            Part("core", "Core", "food:AppleCore", None, observability="visible after cutting",
                 doc="Fused carpels inside the core line, ~25-37 mm wide, ~41 mm long (uspto); the part removed by "
                     "coring (corer bore ~22 mm)."),
            Part("pith", "Core/Pith", "food:CoreTissue",
                 ("mesh", lambda: ML.hollow(spin(), *[endo_out[k]() for k in range(5)])), "Pith",
                 "visible after cutting", "Carpellary tissue inside the core line; denser than the cortex "
                 "(porosity ~14.5 %).", d + 60.0),
        ]
        for k in range(5):
            P.append(Part(f"carpel_{k + 1}", f"Core/Carpel_{k + 1}", "food:Carpel", None,
                          observability="visible after cutting", doc="On the sepal radius %d." % (k + 1)))
            P.append(Part(f"endocarp_{k + 1}", f"Core/Carpel_{k + 1}/Endocarp", "food:Endocarp",
                          ("mesh", lambda k=k: ML.hollow(endo_out[k](), endo_in[k]())), "Endocarp",
                          "visible after cutting", "Leathery/papery locule wall (the 'star' of a transverse cut).",
                          R["endocarp"]))
            inner = [seedm[(s["carpel"], s["index"])] for s in self.seeds if s["carpel"] == k]
            P.append(Part(f"locule_{k + 1}", f"Core/Carpel_{k + 1}/Locule", "food:Locule",
                          ("mesh", lambda k=k, inner=inner: ML.hollow(endo_in[k](), *[f() for f in inner])),
                          "Locule", "visible after cutting", "Air-filled seed chamber.", R["locule"],
                          attrs={"seed_count": float(len(inner))}))
            for s in self.seeds:
                if s["carpel"] == k:
                    P.append(Part(f"seed_{k + 1}_{s['index'] + 1}", f"Core/Carpel_{k + 1}/Seed_{s['index'] + 1}",
                                  "food:AppleSeed", ("mesh", seedm[(k, s["index"])]), "Seed", "visible after cutting",
                                  "Testa dark brown, tip towards the stem (wiki_apple, sciRep2021).", R["seed"]))
        return P

    def skin_regions(self, P):
        """Named surface regions of the skin (GeomSubsets): stem cavity, calyx basin, shoulder, cheek, russet."""
        rn, th, z = self.norm_coords(P)
        zn = z / self.hz
        reg = np.full(len(P), "Cheek", dtype=object)
        reg[zn > 0.45] = "Shoulder"
        reg[(zn > 0.55) & (rn < 0.55 * self.R)] = "StemCavity"
        reg[(zn < -0.70) & (rn < 0.45 * self.R)] = "CalyxBasin"
        reg[(zn < -0.40) & (reg == "Cheek")] = "Base"
        return reg

    # ---------------------------------------------------------------- appearance
    def point_uv(self, U, V):
        t = 1 - V
        k = np.linspace(0, 1, len(self.mer.rho))
        rho, z = np.interp(t, k, self.mer.rho), np.interp(t, k, self.mer.z)
        th = 2 * np.pi * U
        m = self.mod(th, z)
        return np.stack([m * rho * np.cos(th), m * rho * np.sin(th), z], -1)

    def uv_of(self, P):
        from scipy.spatial import cKDTree
        rn, th, z = self.norm_coords(P)
        k = np.linspace(0, 1, len(self.mer.rho))
        _, i = cKDTree(np.stack([self.mer.rho, self.mer.z], 1)).query(np.stack([rn, z], 1))
        return np.stack([np.mod(th, 2 * np.pi) / (2 * np.pi), 1 - k[i]], 1)

    def outer_mesh(self, n_theta=320, theta0=0.0, span=2 * np.pi):
        return revolve(self.mer, n_theta, modulation=self.mod, theta0=theta0, theta_span=span)

    def skin_albedo(self, P, seed):
        rng = np.random.default_rng(seed)
        vn = ValueNoise3(seed)
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        th = np.arctan2(y, x)
        zn = z / self.hz
        ground = np.array([0.86, 0.76, 0.32])          # sRGB; Gala: striped orange-red over yellow (wiki_gala)
        blush_dark = np.array([0.55, 0.07, 0.07])
        blush = np.array([0.82, 0.24, 0.12])
        sun = 0.5 + 0.5 * np.cos(th - self.sun)
        Pa = np.stack([x, y, z * 0.12], -1)
        streak = np.clip(0.5 + 0.9 * vn.fbm(Pa + 11, 1.6e-3, 3), 0, 1)
        cover = np.clip(0.25 + 0.85 * sun + 0.25 * vn.fbm(P, 12e-3, 3) - 0.35 * np.clip(-zn - 0.55, 0, 1), 0, 1)
        cover = np.clip(cover * (0.70 + 0.45 * streak), 0, 1)
        deep = np.clip(sun * cover, 0, 1)
        red = blush * (1 - deep[..., None]) + blush_dark * deep[..., None]
        col = ground * (1 - cover[..., None]) + red * cover[..., None]
        col *= 1 + 0.05 * vn.fbm(P + 5, 1.2e-3, 3)[..., None]
        rus = np.clip((zn - 0.66) / 0.2, 0, 1) * np.clip(1 - np.hypot(x, y) / (0.45 * self.R), 0, 1)
        rus = np.clip(rus * (0.7 + 0.6 * vn.fbm(P, 1.5e-3, 3)), 0, 1)
        col = col * (1 - rus[..., None]) + np.array([0.52, 0.38, 0.20]) * rus[..., None]
        if getattr(self, "_lenticels", None) is None:
            self._lenticels = self.lenticel_points()
        cen, rad = self._lenticels
        sp = spots(P, cen, rad / 2, sharp=1.5)
        col = col * (1 - 0.55 * sp[..., None]) + np.array([0.93, 0.86, 0.55]) * 0.55 * sp[..., None]
        return np.clip(col, 0, 1)

    def section_rgb(self, lab, P, seed=0):
        vn = ValueNoise3(seed + 9)
        rgb = np.zeros(lab.shape + (3,))
        for i, t in enumerate(TISSUES):
            rgb[lab == i + 1] = t["colour"]
        sk = (lab == TI["cuticle"]) | (lab == TI["epidermis"])
        if sk.any():
            rgb[sk] = self.skin_albedo(P[sk], self.seed + 3)
        rgb *= (1 + 0.03 * vn.fbm(P, 0.25e-3, 2))[..., None]          # parenchyma cell granularity
        fl = (lab == TI["cortex"]) | (lab == TI["pith"])
        rho = np.hypot(P[..., 0], P[..., 1])
        g = np.exp(-((rho - self.r_cl) / 2.5e-3) ** 2) * 0.5
        rgb[fl] = rgb[fl] * (1 - 0.05 * g[fl])[:, None] + np.array([0.88, 0.92, 0.62]) * (0.05 * g[fl])[:, None]
        cv = lab == TI["locule"]
        if cv.any():
            rn, th, z = self.norm_coords(P[cv])
            e = np.min([self.locule_e(rn, th, z, tk) for tk in self.loc_theta], axis=0)
            rgb[cv] *= (0.72 + 0.28 * np.clip(e, 0, 1) ** 0.5)[:, None]
        sd = lab == TI["seed"]
        rgb[sd] *= (0.8 + 0.4 * (0.5 + 0.5 * vn.fbm(P[sd], 0.6e-3, 2)))[:, None]
        return np.clip(rgb, 0, 1)
