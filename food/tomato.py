"""Round tomato twin (Solanum lycopersicum, round / truss type): a fleshy berry.

Anatomy (docs/wave2/TOMATO.md for the sources): the SKIN (cuticle, epidermis, a few collenchymatous hypodermal
layers); the PERICARP: the outer wall, the radial walls (SEPTA) dividing the fruit into LOCULES (2-4 in round
tomatoes), and the inner pericarp/central COLUMELLA (the core); the PLACENTA grows from the columella into each
locule and carries the SEEDS, embedded in the LOCULAR GEL (jelly) that fills the locule.  At the stem end the STEM
SCAR with the CALYX (5-6 sepals) and, on truss tomatoes, a piece of PEDICEL with its joint (knuckle); at the blossom
end a small BLOSSOM-END SCAR.

Geometry: an oblate outline with locule-ribbed shoulders; skin and wall are revolved shells; columella, septa,
placenta and gel are regions of the (rho, z) half-plane revolved over angular sectors (food.shapes.revolve_polygon);
the same polygons label points.

Partonomy (USD paths below /Food/Parts):
  Skin, Pericarp/{OuterWall, Septa/Septum_k, Columella}, Locules/Locule_k/{Placenta, Gel, Seeds/Seed_j},
  Calyx/{Sepal_k}, StemScar, Pedicel (truss), BlossomEndScar (attribute/texture)
"""

import numpy as np
from scipy.spatial import cKDTree

from . import meshlib as ML
from . import params as PR
from .geometry import Body, Meridian, resample
from .partgraph import Part
from .shapes import angle_in, region_contains, revolve_polygon, smoothstep
from .texture import ValueNoise3, lch_to_srgb

TISSUES = [
    {"name": "skin", "density": 1050.0, "colour": (0.85, 0.15, 0.08)},
    {"name": "outer_wall", "density": 1020.0, "colour": (0.88, 0.22, 0.12)},
    {"name": "septum", "density": 1020.0, "colour": (0.90, 0.30, 0.18)},
    {"name": "columella", "density": 1010.0, "colour": (0.92, 0.45, 0.30)},
    {"name": "placenta", "density": 1010.0, "colour": (0.93, 0.50, 0.32)},
    {"name": "gel", "density": 1010.0, "colour": (0.85, 0.45, 0.20)},
    {"name": "seed", "density": 1100.0, "colour": (0.90, 0.82, 0.55)},
    {"name": "calyx", "density": 900.0, "colour": (0.20, 0.40, 0.12)},
]
TI = {t["name"]: i + 1 for i, t in enumerate(TISSUES)}
N_TH = 144


def tomato_curve(a, c, n=240, shoulder=0.15):
    """Oblate outline, top pole (stem end, slightly sunken) -> bottom pole (blossom end)."""
    t = np.linspace(0, np.pi, n)
    rho = a * np.sin(t)
    z = c * np.cos(t)
    z = z - shoulder * c * np.exp(-(t / 0.35) ** 2)          # the stem end sits in a shallow dip
    return rho, z


class Tomato(Body):
    tissues = TISSUES
    kind = "tomato"

    def __init__(self, seed=0, overrides=None, stage=None):
        p = PR.sample("tomato", seed)
        p.update(overrides or {})
        if stage is not None:
            p["ripeness_stage"] = float(stage)
        rng = np.random.default_rng(p["_rng_state"])
        self.p, self.seed, self.rng = p, seed, rng
        for t in TISSUES:
            if t["name"] in ("skin", "outer_wall", "septum", "columella", "placenta"):
                t["density"] = p["flesh_density"]
            elif t["name"] == "gel":
                t["density"] = p["gel_density"]
        p["height"] = p["diameter"] * p["height_ratio"]
        self.a, self.c = p["diameter"] * 5e-4, p["height"] * 5e-4
        self.n_loc = int(p["locules"])
        self.t_skin = p["skin_thickness"] * 1e-6
        self.t_wall = p["pericarp_thickness"] * 1e-3
        self.t_sep = p["septum_thickness"] * 1e-3
        self.r_col = p["columella_diameter"] * 5e-4
        self.w_pl = p["placenta_width"] * 1e-3
        self.phi = rng.uniform(0, 2 * np.pi)
        rib = p["rib_depth"]
        rho, z = tomato_curve(self.a, self.c)

        def mod(th, zz):
            sh = smoothstep(-0.2 * self.c, 0.8 * self.c, zz)
            return 1 + rib * sh * np.cos(self.n_loc * (th - self.phi))
        super().__init__(Meridian(rho, z), mod)
        self.sep_angles = self.phi + np.pi / self.n_loc + np.arange(self.n_loc) * 2 * np.pi / self.n_loc + \
            rng.normal(0, 0.08, self.n_loc)
        self._seeds = None

    # ---------------------------------------------------------------- regions
    def curve(self, depth):
        return tomato_curve(self.a - depth, self.c - depth)

    def solid_poly(self, depth):
        r, z = self.curve(depth)
        return np.stack([r, z], 1)

    def ring_poly(self, r_in, depth_out, z_margin=0.0):
        """Between radius r_in (cylinder about the axis) and the outline moved in by depth_out, z-trimmed."""
        ro, zo = self.curve(depth_out)
        keep = (ro > r_in + 0.3e-3) & (zo < zo.max() - z_margin) & (zo > zo.min() + z_margin)
        ro, zo = ro[keep], zo[keep]
        return np.vstack([np.stack([ro, zo], 1), [[r_in, zo[-1]], [r_in, zo[0]]]])

    def col_poly(self):
        r, z = self.curve(self.t_wall)
        zt, zb = z.max(), z.min()
        rc = self.r_col
        return np.array([[0, zt], [rc * 1.15, zt], [rc * 1.1, 0.4 * zt], [rc, 0.0], [rc * 0.8, 0.6 * zb],
                         [rc * 0.5, zb], [0, zb]])

    def placenta_poly(self):
        """Band beside the columella, as tall as the locule's gel region."""
        g = self.ring_poly(self.r_col + self.w_pl, self.t_wall, z_margin=3e-3)
        zt, zb = g[:, 1].max() - 1.0e-3, g[:, 1].min() + 1.0e-3
        r0, r1 = self.r_col, self.r_col + self.w_pl
        return np.array([[r0, zt], [r1, zt], [r1, zb], [r0, zb]])

    def locule_sector(self, k, inset=0.0):
        a0 = self.sep_angles[k] + self.t_sep / 2 / (self.r_col + 6e-3)
        a1 = self.sep_angles[(k + 1) % self.n_loc] + (2 * np.pi if k + 1 == self.n_loc else 0) - \
            self.t_sep / 2 / (self.r_col + 6e-3)
        d = inset / (self.r_col + 6e-3)
        return a0 + d, a1 - d

    def label(self, P):
        P = np.asarray(P, float)
        rn, th, z = self.norm_coords(P)
        out = np.zeros(P.shape[:-1], np.int16)
        ins = region_contains(self.solid_poly(0.0), rn, z)
        out[ins] = TI["outer_wall"]
        out[ins & ~region_contains(self.solid_poly(self.t_skin), rn, z)] = TI["skin"]
        inner = ins & region_contains(self.solid_poly(self.t_wall), rn, z)
        out[inner] = TI["septum"]                            # everything inside the wall that is not a locule
        gel_poly = self.ring_poly(self.r_col + self.w_pl, self.t_wall, z_margin=3e-3)
        pl_poly = self.placenta_poly()
        for k in range(self.n_loc):
            a0, a1 = self.locule_sector(k)
            sec = inner & angle_in(th, a0, a1)
            out[sec & region_contains(pl_poly, rn, z)] = TI["placenta"]
            out[sec & region_contains(gel_poly, rn, z)] = TI["gel"]
        out[inner & region_contains(self.col_poly(), rn, z)] = TI["columella"]
        C, ax = self.seeds()
        sel = np.nonzero(out == TI["gel"])
        if len(C) and len(sel[0]):
            d, i = cKDTree(C).query(P[sel])
            hit = d < ax[i, 0]
            out[tuple(a[hit] for a in sel)] = TI["seed"]
        return out

    def tissue_of(self, rn, th, z, P):
        return self.label(P)

    def inside(self, P):
        return self.label(P) > 0

    def bbox(self):
        r = self.a * (1 + self.p["rib_depth"]) * 1.05
        return np.array([-r, -r, -self.c - 1e-3]), np.array([r, r, self.c + self.p["pedicel_length"] * 1e-3 + 5e-3])

    def seeds(self):
        if self._seeds is None:
            rng = np.random.default_rng(self.seed + 7)
            n_per = int(self.p["seeds_per_fruit"]) // max(self.n_loc, 1)
            C, AX = [], []
            r0 = self.r_col + self.w_pl + 1.2e-3
            for k in range(self.n_loc):
                a0, a1 = self.locule_sector(k, 1.0e-3)
                for _ in range(n_per):
                    a = rng.uniform(a0, a1)
                    zz = rng.uniform(-0.45, 0.45) * self.c
                    rr = r0 + rng.uniform(0, 2.0e-3)
                    C.append([rr * np.cos(a), rr * np.sin(a), zz])
                    AX.append([self.p["seed_length"] * 5e-4, self.p["seed_length"] * 3.6e-4, 0.4e-3])
            self._seeds = (np.array(C).reshape(-1, 3), np.array(AX).reshape(-1, 3))
        return self._seeds

    # ---------------------------------------------------------------- meshes
    def modulate(self, m):
        v = np.asarray(m.vertices).copy()
        k = self.mod(np.arctan2(v[:, 1], v[:, 0]), v[:, 2])
        v[:, :2] *= k[:, None]
        m.vertices = v
        return m

    def surface(self, depth, n=N_TH):
        r, z = self.curve(depth)
        r, z = resample(r, z, 220)
        r[0] = r[-1] = 0
        return ML.surface_of_revolution(Meridian(r, z), n, modulation=self.mod)

    def sepal_paths(self):
        n = int(self.p["sepals"])
        z0 = self.curve(0)[1].max()
        out = []
        for k in range(n):
            a = self.phi + 2 * np.pi * k / n + self.rng.normal(0, 0.12)
            L = self.p["sepal_length"] * 1e-3 * self.rng.uniform(0.8, 1.15)
            s = np.linspace(0, 1, 10)
            r = 2.5e-3 + L * s
            droop = -0.25 * self.c * s ** 2 * (1 - 0.3 * s)
            out.append(np.stack([r * np.cos(a), r * np.sin(a), z0 + 1.5e-3 + droop + 2e-3 * np.sin(np.pi * s)], 1))
        return out

    def pedicel_path(self):
        z0 = self.curve(0)[1].max() + 1.0e-3
        L = self.p["pedicel_length"] * 1e-3
        return np.stack([np.linspace(0, 0.25 * L, 8), np.zeros(8), z0 + np.linspace(0, L, 8)], 1)

    def parts(self):
        cache = {}

        def c(name, fn):
            def g():
                if name not in cache:
                    cache[name] = fn()
                return cache[name]
            return g
        S0, S1, S2 = (c(f"s{d}", lambda d=d: self.surface(d)) for d in (0.0, self.t_skin, self.t_wall))
        R = {t["name"]: t["density"] for t in TISSUES}
        gel_poly = self.ring_poly(self.r_col + self.w_pl, self.t_wall, z_margin=3e-3)
        pl_poly = self.placenta_poly()
        loc_meshes = {}
        for k in range(self.n_loc):
            a0, a1 = self.locule_sector(k)
            loc_meshes[k] = (c(f"pl{k}", lambda a0=a0, a1=a1: self.modulate(revolve_polygon(pl_poly, 40, a0, a1))),
                c(f"g{k}", lambda a0=a0, a1=a1: self.modulate(revolve_polygon(gel_poly, 40, a0, a1))))

        def septa_mesh(k):
            # the septum k: the wedge between locule k-1 and k, from the columella to the wall
            a1 = self.locule_sector((k - 1) % self.n_loc)[1]
            a0 = self.locule_sector(k % self.n_loc)[0]
            if a0 < a1:
                a0 += 2 * np.pi
            return self.modulate(revolve_polygon(self.ring_poly(self.r_col * 1.16, self.t_wall + 0.05e-3), 6,
                                                 a1, a0))
        P = [Part("skin", "Skin", "food:TomatoSkin", ("mesh", lambda: ML.hollow(S0(), S1())), "Skin", "visible",
                  "Cuticle + epidermis + hypodermis: the peel that slips off after blanching.", R["skin"],
                  interior=False, attrs={"thickness_um": self.p["skin_thickness"]}),
             Part("pericarp", "Pericarp", "food:Pericarp", None, observability="visible after cutting",
                  doc="Fruit wall: outer wall, radial walls (septa), inner wall / columella."),
             Part("outer_wall", "Pericarp/OuterWall", "food:OuterPericarp", ("mesh", lambda: ML.hollow(S1(), S2())),
                  "Wall", "visible after cutting", "The firm flesh of a tomato slice.", R["outer_wall"],
                  attrs={"thickness_mm": self.p["pericarp_thickness"]}),
             Part("septa", "Pericarp/Septa", "food:SeptumGroup", None, observability="visible after cutting")]
        for k in range(self.n_loc):
            P.append(Part(f"septum_{k + 1}", f"Pericarp/Septa/Septum_{k + 1}", "food:Septum",
                          ("mesh", lambda k=k: septa_mesh(k)), "Wall", "visible after cutting", "", R["septum"]))
        P.append(Part("columella", "Pericarp/Columella", "food:Columella",
                      ("mesh", lambda: self.modulate(revolve_polygon(self.col_poly(), 48))), "Core",
                      "visible after cutting", "The core (cut out when coring).", R["columella"]))
        P.append(Part("locules", "Locules", "food:LoculeGroup", None, observability="visible after cutting",
                      attrs={"count": self.n_loc}))
        C, AX = self.seeds()
        for k in range(self.n_loc):
            base = f"Locules/Locule_{k + 1}"
            pl, gl = loc_meshes[k]
            P += [Part(f"locule_{k + 1}", base, "food:Locule", None, observability="visible after cutting"),
                  Part(f"placenta_{k + 1}", base + "/Placenta", "food:Placenta", ("mesh", pl), "Placenta",
                       "visible after cutting", "Carries the seeds.", R["placenta"]),
                  Part(f"gel_{k + 1}", base + "/Gel", "food:LocularGel", ("mesh", gl), "Gel",
                       "visible after cutting", "Jelly around the seeds; runs out when cut.", R["gel"])]
            a0, a1 = self.locule_sector(k)
            mine = [j for j, s in enumerate(C) if angle_in(np.arctan2(s[1], s[0]), a0, a1)]
            if mine:
                P.append(Part(f"seeds_{k + 1}", base + "/Seeds", "food:SeedGroup", None,
                              observability="visible after cutting", attrs={"count": len(mine)}))
            for j in mine:
                cc, ax = C[j], AX[j]
                a = np.arctan2(cc[1], cc[0])
                rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]) @ \
                    np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
                P.append(Part(f"seed_{k + 1}_{j + 1}", f"{base}/Seeds/Seed_{j + 1:02d}", "food:TomatoSeed",
                              ("mesh", lambda cc=cc, ax=ax, rot=rot: ML.ellipsoid(cc, (ax[2], ax[1], ax[0]), rot, 8, 16)),
                              "Seed", "visible after cutting", "", R["seed"]))
        P.append(Part("calyx", "Calyx", "food:Calyx", None, observability="visible",
                      doc="Green sepals at the stem end (kept on truss tomatoes)."))
        for k, path in enumerate(self.sepal_paths()):
            P.append(Part(f"sepal_{k + 1}", f"Calyx/Sepal_{k + 1}", "food:Sepal",
                          ("mesh", lambda path=path: ML.tube(path, np.linspace(1.2e-3, 0.25e-3, len(path)), 6)),
                          "Calyx", "visible", "", R["calyx"], interior=False))
        P.append(Part("stem_scar", "StemScar", "food:StemScar",
                      ("mesh", lambda: ML.ellipsoid([0, 0, self.curve(0)[1].max() + 0.3e-3], (3.0e-3, 3.0e-3, 1.0e-3),
                                                    n_u=8, n_v=16)), "Calyx", "visible",
                      "Where the pedicel joins the fruit (cut out with the core).", R["calyx"], interior=False))
        if self.p["pedicel_length"] > 0:
            P.append(Part("pedicel", "Pedicel", "food:Pedicel",
                          ("mesh", lambda: ML.tube(self.pedicel_path(), 1.4e-3, 10)), "Calyx", "visible",
                          "Stalk piece with its joint (truss tomatoes).", R["calyx"], interior=False))
        return P

    # ---------------------------------------------------------------- appearance
    def skin_colour(self):
        s = float(self.p["ripeness_stage"])
        S = np.arange(1, 7, dtype=float)
        return lch_to_srgb(np.interp(s, S, PR.TOMATO["skin_L_by_stage"][1]),
                           np.interp(s, S, PR.TOMATO["skin_C_by_stage"][1]),
                           np.interp(s, S, PR.TOMATO["skin_hue_by_stage"][1]))

    def flesh_colour(self):
        """Pericarp colour: blend green -> red in sRGB by stage (an LCh hue path would pass through brown)."""
        s = float(np.clip(self.p["ripeness_stage"], 1, 6))
        g, w, r = (lch_to_srgb(*self.p[k]) for k in ("flesh_LCh_green", "flesh_LCh_breaker", "flesh_LCh_red"))
        if s <= 2.5:                                        # green -> pale (breaker) -> pink -> red, never brown
            t = (s - 1) / 1.5
            return g * (1 - t) + w * t
        t = ((s - 2.5) / 3.5) ** 0.8
        return w * (1 - t) + r * t

    def point_uv(self, U, V):
        r, z = self.curve(0.0)
        k = np.linspace(0, 1, len(r))
        t = 1 - np.asarray(V)
        rr, zz = np.interp(t, k, r), np.interp(t, k, z)
        th = 2 * np.pi * np.asarray(U)
        m = self.mod(th, zz)
        return np.stack([m * rr * np.cos(th), m * rr * np.sin(th), zz], -1)

    def uv_of(self, P):
        from scipy.spatial import cKDTree as KD
        r, z = self.curve(0.0)
        rn, th, zz = self.norm_coords(np.asarray(P, float))
        k = np.linspace(0, 1, len(r))
        _, i = KD(np.stack([r, z], 1)).query(np.stack([rn, zz], 1))
        return np.stack([np.mod(th, 2 * np.pi) / (2 * np.pi), 1 - k[i]], 1)

    def skin_albedo(self, P, seed):
        P = np.asarray(P, float)
        vn = ValueNoise3(seed + 101)
        base = self.skin_colour()
        col = np.broadcast_to(base, P.shape).copy() * (1 + 0.05 * vn.fbm(P, 5e-3, 3))[..., None]
        zt = self.curve(0)[1].max()
        sh = np.exp(-((zt - P[..., 2]) / (0.3 * self.c)) ** 2)        # shoulder: last to colour
        s = float(self.p["ripeness_stage"])
        green = lch_to_srgb(55, 40, 110)
        g = sh * np.clip((5.5 - s) / 4.0, 0, 1) * 0.7
        col = col * (1 - g[..., None]) + green * g[..., None]
        th = np.arctan2(P[..., 1], P[..., 0])
        star = (0.5 + 0.5 * np.cos(self.n_loc * 2 * (th - self.phi))) * np.exp(-((P[..., 2] + self.c) / 2e-3) ** 2)
        col = col * (1 - 0.25 * star[..., None])
        return np.clip(col, 0, 1)

    def section_rgb(self, lab, P, seed=0):
        rgb = np.zeros(lab.shape + (3,)) + 0.86
        vn = ValueNoise3(seed + 111)
        for i, t in enumerate(TISSUES):
            rgb[lab == i + 1] = t["colour"]
        flesh = self.flesh_colour()
        for n_, f in (("outer_wall", (1.0, 1.0, 1.0)), ("septum", (1.04, 1.06, 1.06)), ("columella", (1.12, 1.22, 1.2)),
                      ("placenta", (1.10, 1.18, 1.15))):
            m = lab == TI[n_]
            rgb[m] = np.clip(flesh * np.array(f), 0, 1) * (1 + 0.04 * vn.fbm(P[m], 0.6e-3, 2))[:, None]
        m = lab == TI["gel"]
        gel = np.clip(flesh * np.array([1.08, 1.0, 0.85]) + np.array([0.02, 0.06, 0.0]), 0, 1)
        rgb[m] = gel * (1 + 0.07 * vn.fbm(P[m], 0.4e-3, 3))[:, None]
        rgb[lab == TI["skin"]] = self.skin_colour()
        return np.clip(rgb, 0, 1)

    # ---------------------------------------------------------------- asset declaration
    def asset_spec(self):
        from .tomato_models import tomato_beliefs, tomato_truth
        p = self.p
        ext = float(self.a * 1.15)
        s = float(p["ripeness_stage"])
        flesh = tuple(float(c) for c in self.flesh_colour())
        skin = tuple(float(c) for c in self.skin_colour())
        a0 = self.locule_sector(0)
        th_c = 0.5 * (a0[0] + a0[1]) + np.pi / 2              # lengthwise plane through two locules
        return {
            "build": {"h": 0.3e-3},
            "textures": [("skin.png", self.point_uv, self.skin_albedo, 2048, 1024)],
            "uv_fns": {"skin": self.uv_of},
            "materials": {
                "Skin": dict(diffuse_reflection_color_image="tex:skin.png", specular_reflection_roughness=0.12,
                             specular_reflection_weight=0.8, coat_weight=0.5, coat_roughness=0.08,
                             subsurface_weight=0.3, subsurface_scale=0.003, subsurface_scattering_color=skin),
                "Wall": dict(diffuse_reflection_color=flesh, specular_reflection_roughness=0.2,
                             specular_reflection_weight=0.7, subsurface_weight=0.4, subsurface_scale=0.003,
                             subsurface_scattering_color=flesh),
                "Core": dict(diffuse_reflection_color=tuple(min(1.0, c * 1.1) for c in flesh),
                             specular_reflection_roughness=0.25),
                "Placenta": dict(diffuse_reflection_color=tuple(min(1.0, c * 1.12) for c in flesh),
                                 specular_reflection_roughness=0.25),
                "Gel": dict(diffuse_reflection_color=tuple(min(1.0, c * k) for c, k in zip(flesh, (1.08, 1.0, 0.85))),
                            specular_reflection_roughness=0.05, specular_reflection_weight=0.9,
                            subsurface_weight=0.6, subsurface_scale=0.004,
                            subsurface_scattering_color=flesh),
                "Seed": dict(diffuse_reflection_color=(0.88, 0.80, 0.52), specular_reflection_roughness=0.4),
                "Calyx": dict(diffuse_reflection_color=(0.16, 0.34, 0.10), specular_reflection_roughness=0.55),
            },
            "section_material": {"specular_reflection_roughness": 0.08, "specular_reflection_weight": 0.85,
                                 "subsurface_weight": 0.35, "subsurface_scale": 0.003,
                                 "subsurface_scattering_color": flesh},
            "root": {"ripeness_class": str(int(round(s))), "cut_state": "whole", "skin_state": "on",
                     "cored": "no"},
            "friction": p["friction_skin_rubber"], "restitution": 0.15,
            "cuts": [
                {"name": "across", "origin": [0, 0, 0.0], "e1": [1, 0, 0], "e2": [0, 1, 0], "ext": ext,
                 "states": {"stem_half": ("pos", {"cut_state": "halved_across"}),
                            "blossom_half": ("neg", {"cut_state": "halved_across"})}},
                {"name": "lengthwise", "origin": [0, 0, 0], "e1": [float(np.cos(th_c)), float(np.sin(th_c)), 0.0],
                 "e2": [0, 0, 1], "ext": ext,
                 "states": {"half_a": ("pos", {"cut_state": "halved_lengthwise"}),
                            "half_b": ("neg", {"cut_state": "halved_lengthwise"})}},
            ],
            "beliefs": lambda mp: tomato_beliefs(self, mp),
            "truth": lambda: tomato_truth(self),
        }
