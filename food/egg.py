"""Hen egg twin: anatomy after the USDA Egg-Grading Manual and the poultry-science literature (docs/ANATOMY.md).

Frame: +z = long axis with the BLUNT end (air cell) up, origin at mid-length; the germinal disc faces +x.
Outer shape: Hügelschäffer model (narushin2021) y(x) = B/2 sqrt((L^2 - 4x^2) / (L^2 + 8wx + 4w^2)), blunt end at x<0.

Partonomy (USD paths below /Food/Parts; every prim is an SDT node -- see parts()):
  Shell/Cuticle, Shell/PalisadeLayer (incl. vertical crystal layer), Shell/MammillaryLayer, Shell/Pores (points)
  ShellMembranes/OuterShellMembrane, ShellMembranes/InnerShellMembrane
  AirCell                         between the membranes at the blunt end
  Albumen/OuterThinAlbumen, Albumen/ThickAlbumen (attached to the membranes at both poles), Albumen/InnerThinAlbumen,
  Albumen/ChalaziferousLayer, Albumen/Chalazae/{BluntEnd, PointedEnd} (twisted cords)
  Yolk/VitellineMembrane, Yolk/YolkBody (concentric yellow/white layers), Yolk/Latebra, Yolk/LatebraNeck,
  Yolk/NucleusOfPander, Yolk/GerminalDisc
Layer volumes are solved so that the albumen layers take 23 / 57 / 17 / 3 % of the white (usda_egg).
"""

import numpy as np

from . import meshlib as ML
from . import params as PR
from .geometry import Body, Meridian, resample, revolve
from .partgraph import Part
from .texture import ValueNoise3, sample_surface, spots

TISSUES = [
    {"name": "cuticle", "density": 1300.0, "colour": (0.62, 0.40, 0.25)},
    {"name": "palisade_layer", "density": 2300.0, "colour": (0.80, 0.62, 0.46)},
    {"name": "mammillary_layer", "density": 2300.0, "colour": (0.86, 0.76, 0.64)},
    {"name": "outer_shell_membrane", "density": 1100.0, "colour": (0.95, 0.94, 0.90)},
    {"name": "inner_shell_membrane", "density": 1100.0, "colour": (0.97, 0.96, 0.93)},
    {"name": "air_cell", "density": 1.2, "colour": (0.62, 0.63, 0.62)},
    {"name": "outer_thin_albumen", "density": 1035.0, "colour": (0.80, 0.83, 0.74)},
    {"name": "thick_albumen", "density": 1035.0, "colour": (0.87, 0.89, 0.80)},
    {"name": "inner_thin_albumen", "density": 1035.0, "colour": (0.82, 0.85, 0.76)},
    {"name": "chalaziferous_layer", "density": 1035.0, "colour": (0.93, 0.93, 0.86)},
    {"name": "chalaza", "density": 1040.0, "colour": (0.96, 0.95, 0.89)},
    {"name": "vitelline_membrane", "density": 1030.0, "colour": (0.99, 0.80, 0.40)},
    {"name": "yolk_body", "density": 1030.0, "colour": (0.98, 0.62, 0.08)},
    {"name": "latebra", "density": 1030.0, "colour": (0.99, 0.80, 0.42)},
    {"name": "nucleus_of_pander", "density": 1030.0, "colour": (0.99, 0.84, 0.52)},
    {"name": "germinal_disc", "density": 1030.0, "colour": (0.99, 0.93, 0.80)},
]
TI = {t["name"]: i + 1 for i, t in enumerate(TISSUES)}
N_MESH_MER, N_MESH_THETA = 200, 128


def hugelschaffer(L, B, w, n=600):
    x = np.linspace(-L / 2, L / 2, 4 * n)
    y = B / 2 * np.sqrt(np.clip((L ** 2 - 4 * x ** 2) / (L ** 2 + 8 * w * x + 4 * w ** 2), 0, None))
    return resample(y, -x, n)          # z = -x: blunt end (x<0) up; ordered top pole -> bottom pole


def yolk_fan_rgb(fan):
    """Approximate sRGB of DSM YolkFan shades 1..16 (ILLUSTRATIVE rendering colours: pale yellow -> deep orange)."""
    t = np.clip((fan - 1) / 15, 0, 1)
    return np.array([0.99, 0.93 - 0.48 * t, 0.45 - 0.42 * t])


class Egg(Body):
    tissues = TISSUES
    kind = "egg"

    def __init__(self, seed=0, overrides=None):
        p = PR.sample("egg", seed)
        p.update(overrides or {})
        rng = np.random.default_rng(p["_rng_state"])
        k = 0.08 if p["storage_temp_C"] < 10 else 0.19
        p["air_cell_height"] = float(np.clip(1.5 + k * p["age_days"] + rng.normal(0, 0.3), 0.8, 10.0))
        p["colour"] = (overrides or {}).get("colour", "brown" if rng.random() < 0.6 else "white")
        p["pigment"] = float(rng.uniform(0.35, 1.0))
        self.p, self.seed, self.rng = p, seed, rng
        L, B, w = p["length_L"] * 1e-3, p["shape_index_B_over_L"] * p["length_L"] * 1e-3, p["asymmetry_w"] * 1e-3
        rho, z = hugelschaffer(L, B, w)
        super().__init__(Meridian(rho, z))
        # ---- shell layers (outside in)
        self.t_s = p["shell_thickness"] * 1e-3
        self.t_cu = p["cuticle_thickness"] * 1e-6
        self.t_mam = PR.EGG["mammillary_thickness"][1] * 1e-6 * p["shell_thickness"] / 0.35
        t_mem = p["membranes_thickness"] * 1e-6
        share = PR.EGG["outer_membrane_share"][1]
        self.t_om, self.t_im = t_mem * share, t_mem * (1 - share)
        self.m_cut = self.mer.offset(self.t_cu)
        self.m_pal = self.mer.offset(self.t_s - self.t_mam)
        self.m_shell = self.mer.offset(self.t_s)
        self.m_om = self.mer.offset(self.t_s + self.t_om)
        # ---- air cell: lens at the blunt end between the outer membrane and the (displaced) inner membrane
        self.h_air = p["air_cell_height"] * 1e-3
        self.z_air = self.m_om.zmax - self.h_air
        self.rho_m = self.m_om.rho.max()
        self.c_lens = 0.9 * self.h_air / self.rho_m ** 2
        self.air_mer, self.m_A = self._air_cell_meridians()
        self.m_alb = self.m_A.offset(self.t_im)          # albumen boundary (inside the inner membrane)
        # ---- yolk: sized by its mass fraction (whole-egg SG 1.087, ro)
        self.V_outer = ML.pappus_volume(self.mer)
        m_tot = self.V_outer * 1087.0
        self.r_yolk = (3 * p["yolk_mass_fraction"] * m_tot / (4 * np.pi * 1030.0)) ** (1 / 3)
        self.c_yolk = np.array([0.0, 0.0, 0.5e-3])
        self.t_vit = 8e-6
        # ---- albumen layers solved from the USDA shares
        V_y = 4 / 3 * np.pi * self.r_yolk ** 3
        V_A = ML.pappus_volume(self.m_alb)
        self.V_alb = V_A - V_y
        self.s_thick = float(np.sqrt(max(1 - 0.23 * self.V_alb / V_A, 0.5)))
        # thick albumen: rho-scaled albumen boundary, pulled 0.25 mm in from the poles (no coincident pole points)
        zc_ = 0.5 * (self.m_alb.zmax + self.m_alb.zmin)
        sz = 1 - 0.25e-3 / (0.5 * (self.m_alb.zmax - self.m_alb.zmin))
        self.m_thick = Meridian(self.m_alb.rho * self.s_thick, zc_ + (self.m_alb.z - zc_) * sz)
        self.r_inner = ((V_y + 0.20 * self.V_alb) * 3 / (4 * np.pi)) ** (1 / 3)
        self.r_chal = ((V_y + 0.03 * self.V_alb) * 3 / (4 * np.pi)) ** (1 / 3)
        # ---- yolk internals: latebra (centre) + neck to the nucleus of Pander under the germinal disc (+x)
        ry = self.r_yolk
        self.r_latebra = 0.20 * ry
        self.d_disc = p["germinal_disc_diameter"] * 1e-3
        self.pander_c = self.c_yolk + np.array([ry - 1.6e-3, 0, 0])
        self.chalaza_paths = self._chalaza_paths()

    # ---------------------------------------------------------------- meridians
    def _air_cell_meridians(self):
        r, z = self.m_om.rho, self.m_om.z
        imax = int(np.argmax(r))
        zl = self.z_air + self.c_lens * r[:imax] ** 2
        i_star = int(np.argmax(z[:imax] <= zl)) or 1
        rs = r[i_star]
        lens_r = np.linspace(rs, 0, 60)
        lens_z = self.z_air + self.c_lens * lens_r ** 2
        air = ML.meridian_region(np.concatenate([r[:i_star], lens_r]), np.concatenate([z[:i_star], lens_z]))
        A = ML.meridian_region(np.concatenate([lens_r[::-1], r[i_star:]]), np.concatenate([lens_z[::-1], z[i_star:]]))
        return air, A

    def _chalaza_paths(self):
        out = {}
        ry = self.r_yolk
        for name, sgn in (("BluntEnd", 1), ("PointedEnd", -1)):
            z0 = self.c_yolk[2] + sgn * (ry - 0.5e-3)
            length = 9e-3 if sgn > 0 else 10e-3
            s = np.linspace(0, 1, 120)
            zz = z0 + sgn * length * s
            strands = []
            for ph in (0.0, np.pi):
                a = 0.65e-3 * (1 - 0.4 * s)                      # strands twist round each other, thinning out
                tw = 2 * np.pi * 3.0 * s + ph
                strands.append(np.stack([a * np.cos(tw), a * np.sin(tw), zz], 1))
            out[name] = strands
        return out

    # ---------------------------------------------------------------- labels (same definitions as the meshes)
    def tissue_of(self, rn, th, z, P):
        out = np.full(len(rn), TI["outer_thin_albumen"], np.int16)
        in_cut, in_pal = self.m_cut.contains(rn, z), self.m_pal.contains(rn, z)
        in_sh, in_om = self.m_shell.contains(rn, z), self.m_om.contains(rn, z)
        out[~in_cut] = TI["cuticle"]
        out[in_cut & ~in_pal] = TI["palisade_layer"]
        out[in_pal & ~in_sh] = TI["mammillary_layer"]
        out[in_sh & ~in_om] = TI["outer_shell_membrane"]
        air = in_om & self.air_mer.contains(rn, z)
        out[air] = TI["air_cell"]
        inA = in_om & ~air
        in_alb = self.m_alb.contains(rn, z)
        out[inA & ~in_alb] = TI["inner_shell_membrane"]
        alb = inA & in_alb
        d = P - self.c_yolk
        r = np.linalg.norm(d, axis=1)
        out[alb & self.m_thick.contains(rn, z)] = TI["thick_albumen"]
        out[alb & (r < self.r_inner)] = TI["inner_thin_albumen"]
        out[alb & (r < self.r_chal)] = TI["chalaziferous_layer"]
        out[alb & (r < self.r_yolk)] = TI["vitelline_membrane"]
        yb = alb & (r < self.r_yolk - self.t_vit)
        out[yb] = TI["yolk_body"]
        out[yb & (r < self.r_latebra)] = TI["latebra"]
        neck = yb & (d[:, 0] > 0) & (d[:, 0] < self.pander_c[0] - self.c_yolk[0]) & (np.hypot(d[:, 1], d[:, 2]) < 0.9e-3)
        out[neck] = TI["latebra"]
        pan = yb & (np.sum(((P - self.pander_c) / np.array([0.9e-3, 1.6e-3, 1.6e-3])) ** 2, axis=1) < 1)
        out[pan] = TI["nucleus_of_pander"]
        disc = alb & (np.sum(((d - [self.r_yolk - 0.12e-3, 0, 0]) /
                              np.array([0.22e-3, self.d_disc / 2, self.d_disc / 2])) ** 2, axis=1) < 1)
        out[disc] = TI["germinal_disc"]
        near = alb & (np.hypot(P[:, 0], P[:, 1]) < 1.6e-3) & (r > self.r_yolk - 1e-3)
        if near.any():
            from scipy.spatial import cKDTree
            pts = np.vstack([s for ss in self.chalaza_paths.values() for s in ss])
            dd, _ = cKDTree(pts).query(P[near])
            sel = np.nonzero(near)[0][dd < 0.36e-3]
            out[sel] = TI["chalaza"]
        return out

    # ---------------------------------------------------------------- parts (USD scene graph + SDT nodes)
    def surf(self, mer, n_theta=N_MESH_THETA):
        r, z = resample(mer.rho, mer.z, N_MESH_MER)
        r[0] = r[-1] = 0
        return ML.surface_of_revolution(Meridian(r, z), n_theta)

    def sphere(self, r, c=None, n=48):
        return ML.ellipsoid(self.c_yolk if c is None else c, (r, r, r), n_u=n, n_v=2 * n)

    def parts(self):
        S = self.surf
        ry = self.r_yolk
        cache = {}

        def s(name, fn):
            def g():
                if name not in cache:
                    cache[name] = fn()
                return cache[name]
            return g
        outer, cut, pal = s("o", lambda: S(self.mer)), s("c", lambda: S(self.m_cut)), s("p", lambda: S(self.m_pal))
        sh, om = s("s", lambda: S(self.m_shell)), s("m", lambda: S(self.m_om))
        A, alb, thick = s("A", lambda: S(self.m_A)), s("alb", lambda: S(self.m_alb)), s("th", lambda: S(self.m_thick))
        sp_in, sp_ch = s("spi", lambda: self.sphere(self.r_inner)), s("spc", lambda: self.sphere(self.r_chal))
        sp_y, sp_v = s("spy", lambda: self.sphere(ry)), s("spv", lambda: self.sphere(ry - self.t_vit))
        sp_l = s("spl", lambda: self.sphere(self.r_latebra, n=24))
        R = {t["name"]: t["density"] for t in TISSUES}

        def chal(name):
            return lambda: ML.union_meshes([ML.tube(p, 0.36e-3, 10) for p in self.chalaza_paths[name]])

        neck_path = np.linspace(self.c_yolk + [self.r_latebra * 0.8, 0, 0], self.pander_c - [0.6e-3, 0, 0], 30)
        return [
            Part("shell", "Shell", "food:Eggshell", None, observability="visible",
                 doc="Calcified shell: ~94 % CaCO3 of dry shell, 8-11 % of egg mass (usda_egg, br)."),
            Part("cuticle", "Shell/Cuticle", "food:EggCuticle", ("mesh", lambda: ML.hollow(outer(), cut())),
                 "Shell", "visible", "Proteinaceous cuticle 5-12 um, plugs the pores; not 'bloom' (usda_egg, ir).",
                 R["cuticle"], interior=False),
            Part("palisade_layer", "Shell/PalisadeLayer", "food:PalisadeLayer",
                 ("mesh", lambda: ML.hollow(cut(), pal())), "ShellSection", "hidden",
                 "Calcite palisade incl. vertical crystal layer, ~300 um; holds ~80 % of the brown pigment (fr, eng).",
                 R["palisade_layer"]),
            Part("mammillary_layer", "Shell/MammillaryLayer", "food:MammillaryLayer",
                 ("mesh", lambda: ML.hollow(pal(), sh())), "ShellSection", "hidden",
                 "Mammillary knobs anchored in the outer membrane, ~100 um (fr).", R["mammillary_layer"]),
            Part("pores", "Shell/Pores", "food:EggshellPore", ("points", self.pore_points), None, "inferred",
                 "10,000-20,000 pores, ~2x denser at the blunt pole (por, fr); gas exchange, bacterial entry."),
            Part("shell_membranes", "ShellMembranes", "food:ShellMembrane", None, observability="hidden",
                 doc="Two fibrous protein membranes, ~70 um together; they hold the contents of a cracked ('check') "
                     "egg (usda_egg)."),
            Part("outer_shell_membrane", "ShellMembranes/OuterShellMembrane", "food:OuterShellMembrane",
                 ("mesh", lambda: ML.hollow(sh(), om())), "Membrane", "hidden", "~3x thicker than the inner one (ubc).",
                 R["outer_shell_membrane"]),
            Part("inner_shell_membrane", "ShellMembranes/InnerShellMembrane", "food:InnerShellMembrane",
                 ("mesh", lambda: ML.hollow(A(), alb())), "Membrane", "hidden",
                 "Separates from the outer membrane at the blunt end to form the air cell.", R["inner_shell_membrane"]),
            Part("air_cell", "AirCell", "food:AirCell", ("mesh", lambda: S(self.air_mer)), "AirCell", "inferred",
                 "Forms as the egg cools after laying; grows by evaporation: USDA AA <= 3.2 mm, A <= 4.8 mm; EU A <= 6, "
                 "extra <= 4 mm (usda_egg, eu589_2008, gr). Observable by candling.", R["air_cell"],
                 attrs={"height_mm": float(self.p["air_cell_height"])}),
            Part("albumen", "Albumen", "food:Albumen", None, observability="visible after opening",
                 doc="Egg white ~58 % of mass, ~90 % water; four layers (usda_egg, wiki egg white)."),
            Part("outer_thin_albumen", "Albumen/OuterThinAlbumen", "food:OuterThinAlbumen",
                 ("mesh", lambda: ML.hollow(alb(), thick())), "AlbumenThin", "hidden", "23 % of the white.",
                 R["outer_thin_albumen"]),
            Part("thick_albumen", "Albumen/ThickAlbumen", "food:ThickAlbumen",
                 ("mesh", lambda: ML.hollow(thick(), sp_in())), "AlbumenThick", "hidden",
                 "57 % of the white; ovomucin gel jacket attached to the membranes at both ends; Haugh units grade "
                 "it (AA >= 72).", R["thick_albumen"]),
            Part("inner_thin_albumen", "Albumen/InnerThinAlbumen", "food:InnerThinAlbumen",
                 ("mesh", lambda: ML.hollow(sp_in(), sp_ch())), "AlbumenThin", "hidden", "17 % of the white.",
                 R["inner_thin_albumen"]),
            Part("chalaziferous_layer", "Albumen/ChalaziferousLayer", "food:ChalaziferousLayer",
                 ("mesh", lambda: ML.hollow(sp_ch(), sp_y())), "Chalaza", "hidden",
                 "3 % of the white: firm layer wrapped round the yolk, continuous with the chalazae.",
                 R["chalaziferous_layer"]),
            Part("chalazae", "Albumen/Chalazae", "food:Chalazae", None, observability="visible after opening",
                 doc="Two twisted cords suspending the yolk towards the two ends (wiki chalaza, unsw)."),
            Part("chalaza_blunt", "Albumen/Chalazae/BluntEnd", "food:Chalaza", ("mesh", chal("BluntEnd")), "Chalaza",
                 "visible after opening", "", R["chalaza"]),
            Part("chalaza_pointed", "Albumen/Chalazae/PointedEnd", "food:Chalaza", ("mesh", chal("PointedEnd")),
                 "Chalaza", "visible after opening", "", R["chalaza"]),
            Part("yolk", "Yolk", "food:Yolk", None, observability="visible after opening",
                 doc="~30 % of egg mass; colour from xanthophylls, graded on the DSM YolkFan (wiki yolk, dsm).",
                 attrs={"yolk_fan": float(self.p["yolk_colour_fan"])}),
            Part("vitelline_membrane", "Yolk/VitellineMembrane", "food:VitellineMembrane",
                 ("mesh", lambda: ML.hollow(sp_y(), sp_v())), "Vitelline", "hidden",
                 "Two-layer colourless membrane; weakens with age as the yolk takes up water (wiki vitelline).",
                 R["vitelline_membrane"]),
            Part("yolk_body", "Yolk/YolkBody", "food:YolkBody", ("mesh", lambda: ML.hollow(sp_v(), sp_l())), "Yolk",
                 "visible after opening", "Concentric yellow and white yolk layers around the latebra (unsw).",
                 R["yolk_body"]),
            Part("latebra", "Yolk/Latebra", "food:Latebra", ("mesh", sp_l), "WhiteYolk", "hidden",
                 "Central flask of white yolk.", R["latebra"]),
            Part("latebra_neck", "Yolk/LatebraNeck", "food:LatebraNeck",
                 ("mesh", lambda: ML.tube(neck_path, 0.9e-3, 12)), "WhiteYolk", "hidden",
                 "Stalk of white yolk from the latebra to the nucleus of Pander.", R["latebra"]),
            Part("nucleus_of_pander", "Yolk/NucleusOfPander", "food:NucleusOfPander",
                 ("mesh", lambda: ML.ellipsoid(self.pander_c, (0.9e-3, 1.6e-3, 1.6e-3), n_u=16, n_v=32)), "WhiteYolk",
                 "hidden", "White-yolk mass under the germinal disc.", R["nucleus_of_pander"]),
            Part("germinal_disc", "Yolk/GerminalDisc", "food:GerminalDisc",
                 ("mesh", lambda: ML.ellipsoid(self.c_yolk + [ry - 0.12e-3, 0, 0],
                                               (0.22e-3, self.d_disc / 2, self.d_disc / 2), n_u=16, n_v=32)),
                 "GerminalDisc", "visible after opening", "Blastodisc 2-3 mm (infertile); it turns to the top of "
                 "the yolk once the egg is opened (ab).", R["germinal_disc"]),
        ]

    def pore_points(self):
        n = int(self.p["pore_count"])
        m = self.surf(self.mer, 96)
        zmin, zmax = self.mer.zmin, self.mer.zmax
        p, nrm = ML.points_on(m, n, self.rng, weight=lambda q: 1 + (q[:, 2] - zmin) / (zmax - zmin))
        return p, np.full(len(p), 15e-6)

    # ---------------------------------------------------------------- mechanics (hidden truth)
    def local_radius(self, polar_angle):
        """Mean principal radius of curvature of the outer shell at the contact's polar angle (0 = blunt pole)."""
        k = np.linspace(0, 1, len(self.mer.rho))
        t = np.clip(polar_angle / np.pi, 0, 1)
        i = int(np.clip(t * (len(k) - 3), 1, len(k) - 3))
        p0 = np.array([self.mer.rho[i - 1], self.mer.z[i - 1]])
        p1 = np.array([self.mer.rho[i], self.mer.z[i]])
        p2 = np.array([self.mer.rho[i + 1], self.mer.z[i + 1]])
        a, b, c = np.linalg.norm(p1 - p0), np.linalg.norm(p2 - p1), np.linalg.norm(p2 - p0)
        u_, v_ = p1 - p0, p2 - p0
        area2 = abs(u_[0] * v_[1] - u_[1] * v_[0])
        R1 = a * b * c / (2 * area2 + 1e-18)
        tng = (p2 - p0) / c
        R2 = p1[0] / max(abs(tng[1]), 1e-6)
        return float(min(2 / (1 / R1 + 1 / R2), 0.2))

    def contact_stiffness(self, polar_angle):
        """Reissner shallow-shell point-load stiffness k = 4 E t^2 / (R sqrt(3 (1 - nu^2)))  [N/m]."""
        E, nu = self.p["shell_E"] * 1e9, PR.EGG["shell_poisson"][1]
        return 4 * E * self.t_s ** 2 / (self.local_radius(polar_angle) * np.sqrt(3 * (1 - nu ** 2)))

    def break_force(self, polar_angle):
        s = np.sin(polar_angle) ** 2
        F = s * self.p["F_break_equatorial"] + (1 - s) * self.p["F_break_axial"]
        return F * (self.p["shell_thickness"] / 0.35) ** 2

    # ---------------------------------------------------------------- appearance
    def point_uv(self, U, V):
        t = 1 - V
        k = np.linspace(0, 1, len(self.mer.rho))
        rho, z = np.interp(t, k, self.mer.rho), np.interp(t, k, self.mer.z)
        th = 2 * np.pi * U
        return np.stack([rho * np.cos(th), rho * np.sin(th), z], -1)

    def uv_of(self, P):
        """Texture coordinates of points near the outer surface (inverse of point_uv)."""
        from scipy.spatial import cKDTree
        k = np.linspace(0, 1, len(self.mer.rho))
        tree = cKDTree(np.stack([self.mer.rho, self.mer.z], 1))
        _, i = tree.query(np.stack([np.hypot(P[:, 0], P[:, 1]), P[:, 2]], 1))
        u = np.mod(np.arctan2(P[:, 1], P[:, 0]), 2 * np.pi) / (2 * np.pi)
        return np.stack([u, 1 - k[i]], 1)

    def shell_base_rgb(self):
        if self.p["colour"] == "brown":
            g = self.p["pigment"]
            return np.array([0.80, 0.60, 0.44]) * (1 - g) + np.array([0.62, 0.38, 0.23]) * g
        return np.array([0.95, 0.93, 0.89])

    def shell_albedo(self, P, seed):
        rng = np.random.default_rng(seed)
        vn = ValueNoise3(seed)
        mott = vn.fbm(P, 6e-3, 4)
        fine = vn.fbm(P + 3.1, 0.5e-3, 3)
        col = self.shell_base_rgb() * (1 + 0.06 * mott[..., None] + 0.025 * fine[..., None])
        if self.p["colour"] == "brown":
            v, f, _, _ = revolve(self.mer, 128)
            cen = sample_surface(v, f, int(900 * self.p["pigment"]), rng)
            keep = rng.random(len(cen)) < 0.35 + 0.65 * np.clip((cen[:, 2] - self.mer.zmin) /
                                                                 (self.mer.zmax - self.mer.zmin), 0, 1)
            cen = cen[keep]
            rad = rng.lognormal(np.log(0.22e-3), 0.5, len(cen))
            sp = spots(P, cen, rad)
            col = col * (1 - 0.45 * sp[..., None]) + np.array([0.36, 0.20, 0.12]) * 0.45 * sp[..., None]
        return np.clip(col, 0, 1)

    def section_rgb(self, lab, P, seed=0):
        vn = ValueNoise3(seed + 7)
        rgb = np.zeros(lab.shape + (3,))
        for i, t in enumerate(TISSUES):
            rgb[lab == i + 1] = t["colour"]
        yc = yolk_fan_rgb(self.p["yolk_colour_fan"])
        y = lab == TI["yolk_body"]
        r = np.linalg.norm(P - self.c_yolk, axis=-1) / self.r_yolk
        rings = 0.5 + 0.5 * np.sin(2 * np.pi * r * 6.0 + 1.5 * np.sin(9 * r))
        rgb[y] = yc * (0.965 + 0.035 * rings[y])[:, None]
        rgb[y] *= (1 + 0.03 * vn.fbm(P[y] + 2, 0.8e-3, 3))[:, None]
        # brown pigment sits mostly in the calcareous shell (eng): brown eggs are brown through the palisade
        if self.p["colour"] == "white":
            for n in ("cuticle", "palisade_layer"):
                rgb[lab == TI[n]] = (0.95, 0.94, 0.91)
        else:
            rgb[lab == TI["cuticle"]] = self.shell_base_rgb() * 0.85
            rgb[lab == TI["palisade_layer"]] = self.shell_base_rgb() * 1.08
        rgb *= (1 + 0.02 * vn.fbm(P, 1.5e-3, 3))[..., None]
        return np.clip(rgb, 0, 1)

    # ---------------------------------------------------------------- discrete grading
    def grade(self, mass_g):
        g = "XL" if mass_g >= 73 else "L" if mass_g >= 63 else "M" if mass_g >= 53 else "S"
        a = self.p["air_cell_height"]
        eu = "A-extra" if a <= 4 else "A" if a <= 6 else "B"
        us = "AA" if a <= 3.2 else "A" if a <= 4.8 else "B"
        return g, eu, us
