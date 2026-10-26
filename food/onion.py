"""Yellow storage onion twin (Allium cepa): a bulb of swollen leaf bases on a compressed stem.

Anatomy (docs/wave2/ONION.md for the sources): papery dry outer scales (tunics) around concentric fleshy storage
scales -- the swollen bases of leaves -- which all spring from the BASAL PLATE (the compressed stem) with the root
crown below it; the scales narrow into the NECK at the top; at the centre the shoot apex with the youngest leaves.
Each fleshy scale is a closed sheath with a thin epidermis film on both faces (the 'onion skin' of microscopy) and
separates easily from its neighbours -- which is why sliced onion falls into rings and dice.

Geometry: every scale is a region of the (rho, z) half-plane revolved about the axis (food.shapes.revolve_polygon):
the bulb outline scaled about a point just below the basal plate gives nested outlines that converge onto the plate,
as the real scales do; the same polygons label points, so meshes, masses and cut faces agree.  A mild azimuthal
modulation (lopsided bulb) is applied to meshes and labels alike.

Partonomy (USD paths below /Food/Parts):
  DryScales/{Tunic_1..n}, FleshyScales/{Scale_01..n}, ShootApex, BasalPlate, Roots (points + tubes), Neck
"""

import numpy as np

from . import meshlib as ML
from . import params as PR
from .geometry import Body, Meridian, smooth_meridian
from .partgraph import Part
from .shapes import region_contains, revolve_polygon, smoothstep
from .texture import ValueNoise3, lch_to_srgb

N_TH = 120


class Onion(Body):
    kind = "onion"

    def __init__(self, seed=0, overrides=None):
        p = PR.sample("onion", seed)
        p.update(overrides or {})
        rng = np.random.default_rng(p["_rng_state"])
        self.p, self.seed, self.rng = p, seed, rng
        p["height"] = p["diameter"] * p["shape_index"]
        D, H = p["diameter"] * 1e-3, p["height"] * 1e-3
        self.D, self.H = D, H
        self.r_plate, self.t_plate = p["plate_diameter"] * 5e-4, p["plate_thickness"] * 1e-3
        self.z_pb = 0.0                                     # plate bottom
        self.z_pt = self.t_plate                            # plate top = where the scales stand
        # outline of the outermost fleshy scale: globe from the neck down to the plate rim (control points)
        R = D / 2
        zc = self.z_pt + 0.45 * H
        a = np.linspace(0, np.pi, 40)
        shp = p["shape_index"]                              # height / diameter
        ctrl = np.stack([R * np.sin(a) ** (0.9 + 0.2 * (shp > 1)), zc + 0.5 * H * np.cos(a)], 1)
        ctrl[:, 1] = np.maximum(ctrl[:, 1], self.z_pt - 0.0)
        top_neck = np.array([[0.0, zc + 0.5 * H + 0.12 * H]])
        rho, z = smooth_meridian(np.vstack([top_neck, ctrl[1:-1], [[self.r_plate * 0.9, self.z_pt - 0.3e-3],
                                                                   [0.0, self.z_pt - 0.3e-3]]]), 500)
        self.outline = (rho, z)
        self.A = np.array([0.0, self.z_pt - 0.8 * self.t_plate])   # scaling anchor inside the plate
        self.phi = rng.uniform(0, 2 * np.pi)
        lop = rng.uniform(0.0, 0.035)

        def mod(th, zz):
            return 1 + lop * np.cos(th - self.phi) * smoothstep(self.z_pt, self.z_pt + H, zz) + \
                0.012 * np.cos(3 * (th - self.phi * 1.7))
        super().__init__(Meridian(rho, z), mod)
        self._layers()

    # ---------------------------------------------------------------- scales as scaled outlines
    def curve(self, s):
        rho, z = self.outline
        q = self.A + s * (np.stack([rho, z], 1) - self.A)
        return q[:, 0], q[:, 1]

    def _clip_above(self, rho, z, zcut):
        """The part of an outline (top pole -> bottom) above zcut, ending exactly on the plane z = zcut."""
        k = np.nonzero(z < zcut)[0]
        if len(k) == 0:
            return rho, z
        j = k[0]
        t = (zcut - z[j - 1]) / (z[j] - z[j - 1])
        r_end = rho[j - 1] + t * (rho[j] - rho[j - 1])
        return np.append(rho[:j], r_end), np.append(z[:j], zcut)

    def _layers(self):
        p = self.p
        R_eq = self.D / 2
        # fleshy scales: thickness tapers from the outer to the inner scales (register), small gap between them
        n = int(p["n_fleshy"])
        t = np.linspace(p["scale_thickness_outer"], p["scale_thickness_inner"], n) * 1e-3
        g = p["scale_gap"] * 1e-3
        s, edges = 1.0, []
        for k in range(n):
            s_in = s - t[k] / R_eq
            edges.append((s, s_in))
            s = s_in - g / R_eq
        self.scale_s = edges
        self.apex_s = s
        # dry tunics outside the outermost fleshy scale
        self.dry_s = []
        s = 1.0 + g / R_eq
        for k in range(int(p["n_dry"])):
            s_out = s + p["dry_thickness"] * 1e-3 / R_eq
            self.dry_s.append((s_out, s))
            s = s_out + 0.05e-3 / R_eq
        self.s_outer = s

    def shell_poly(self, s_out, s_in, zcut):
        ro, zo = self._clip_above(*self.curve(s_out), zcut)
        ri, zi = self._clip_above(*self.curve(s_in), zcut)
        return np.vstack([np.stack([ro, zo], 1), np.stack([ri[::-1], zi[::-1]], 1)])

    def solid_poly(self, s, zcut):
        r, z = self._clip_above(*self.curve(s), zcut)
        return np.vstack([np.stack([r, z], 1), [[0.0, zcut]]])

    def plate_poly(self):
        rp, zt, zb = self.r_plate, self.z_pt, self.z_pb
        a = np.linspace(0, np.pi / 2, 20)
        rim = np.stack([rp - 0.4 * self.t_plate * (1 - np.cos(a)), zb + self.t_plate * 0.5 * (1 - np.sin(a))], 1)
        return np.vstack([[[0.0, zt], [rp, zt]], rim[::-1][:-1], [[0.0, zb]]])

    def regions(self):
        """Ordered (name, polygon): outer layers first; label = first region containing the point."""
        zc_scales = self.z_pt
        zc_dry = self.z_pb + 0.3 * self.t_plate              # dry tunics wrap the plate rim
        R = []
        for k, (so, si) in enumerate(self.dry_s[::-1]):
            R.append((f"tunic_{len(self.dry_s) - k}", self.shell_poly(so, si, zc_dry)))
        for k, (so, si) in enumerate(self.scale_s):
            R.append((f"scale_{k + 1:02d}", self.shell_poly(so, si, zc_scales)))
        R.append(("shoot_apex", self.solid_poly(self.apex_s, zc_scales)))
        R.append(("basal_plate", self.plate_poly()))
        return R

    # ---------------------------------------------------------------- labels
    def label(self, P):
        P = np.asarray(P, float)
        rn, th, z = self.norm_coords(P)
        out = np.zeros(P.shape[:-1], np.int16)
        free = np.ones(P.shape[:-1], bool)
        for i, (name, poly) in enumerate(self.regions()):
            ins = free & region_contains(poly, rn, z)
            out[ins] = i + 1
            free &= ~ins
        return out

    def tissue_of(self, rn, th, z, P):
        return self.label(P)

    def inside(self, P):
        return self.label(P) > 0

    def bbox(self):
        r = self.D / 2 * self.s_outer * 1.06
        return np.array([-r, -r, self.z_pb - 1e-3]), np.array([r, r, self.curve(self.s_outer)[1].max() + 2e-3])

    @property
    def tissues(self):
        cols = self.colours()
        out = []
        for name, _ in self.regions():
            kind = "dry" if name.startswith("tunic") else ("plate" if name == "basal_plate" else "flesh")
            dens = {"dry": self.p["dry_density"], "plate": self.p["plate_density"], "flesh": self.p["flesh_density"]}
            out.append({"name": name, "density": dens[kind], "colour": cols[kind]})
        return out

    def colours(self):
        p = self.p
        return {"dry": lch_to_srgb(*p["skin_LCh"]), "flesh": lch_to_srgb(*p["flesh_LCh"]),
                "plate": lch_to_srgb(p["flesh_LCh"][0] - 8, p["flesh_LCh"][1] + 6, p["flesh_LCh"][2] - 10)}

    # ---------------------------------------------------------------- meshes
    def mesh(self, poly, n=N_TH):
        m = revolve_polygon(poly, n)
        v = np.asarray(m.vertices).copy()
        th = np.arctan2(v[:, 1], v[:, 0])
        k = self.mod(th, v[:, 2])
        v[:, :2] *= k[:, None]
        m.vertices = v
        return m

    def root_paths(self):
        rng = np.random.default_rng(self.seed + 3)
        n = int(self.p["n_roots"])
        out = []
        for _ in range(n):
            a, r = rng.uniform(0, 2 * np.pi), np.sqrt(rng.random()) * self.r_plate * 0.75
            L = self.p["root_length"] * 1e-3 * rng.uniform(0.5, 1.2)
            d = np.array([np.cos(a) * 0.5 * r / self.r_plate, np.sin(a) * 0.5 * r / self.r_plate, -1.0])
            d /= np.linalg.norm(d)
            base = np.array([r * np.cos(a), r * np.sin(a), self.z_pb + 0.3e-3])
            pts = base + np.outer(np.linspace(0, L, 6), d) + np.outer(np.linspace(0, 1, 6) ** 2,
                                                                     rng.normal(0, 1.2e-3, 3))
            out.append(pts)
        return out

    def neck_mesh(self):
        z0 = self.curve(self.s_outer)[1].max() - 1.5e-3
        L = self.p["neck_length"] * 1e-3
        r = self.p["neck_diameter"] * 5e-4
        path = np.linspace([0, 0, z0], [0.4e-3, 0, z0 + L], 8)
        return ML.tube(path, np.linspace(r, 0.8 * r, 8), 14)

    def parts(self):
        P = [Part("dry_scales", "DryScales", "food:DryScaleGroup", None, observability="visible",
                  doc="Papery protective tunics: dried outer leaf bases; brittle, peeled off in the kitchen.")]
        R = dict(self.regions())
        T = {t["name"]: t for t in self.tissues}
        for k in range(len(self.dry_s)):
            n = f"tunic_{k + 1}"
            P.append(Part(n, f"DryScales/Tunic_{k + 1}", "food:DryScale", ("mesh", lambda n=n: self.mesh(R[n])),
                          "DrySkin", "visible" if k == len(self.dry_s) - 1 else "visible after peeling",
                          "", T[n]["density"], interior=(k != len(self.dry_s) - 1)))
        P.append(Part("fleshy_scales", "FleshyScales", "food:FleshyScaleGroup", None,
                      observability="visible after peeling",
                      doc="Swollen leaf bases, concentric closed sheaths from the basal plate; each with an "
                          "epidermis film on both faces; they separate into rings when sliced."))
        for k in range(len(self.scale_s)):
            n = f"scale_{k + 1:02d}"
            P.append(Part(n, f"FleshyScales/Scale_{k + 1:02d}", "food:FleshyScale",
                          ("mesh", lambda n=n: self.mesh(R[n])), "Flesh" if k else "FleshOuter",
                          "visible after cutting", "", T[n]["density"],
                          attrs={"index_from_outside": k + 1,
                                 "thickness_mm": float((self.scale_s[k][0] - self.scale_s[k][1]) * self.D / 2e-3)}))
        P += [
            Part("shoot_apex", "ShootApex", "food:ShootApex", ("mesh", lambda: self.mesh(R["shoot_apex"])), "Apex",
                 "visible after cutting", "Growing point with the youngest leaves (sprouts from here).",
                 T["shoot_apex"]["density"]),
            Part("basal_plate", "BasalPlate", "food:BasalPlate", ("mesh", lambda: self.mesh(R["basal_plate"], 64)),
                 "Plate", "visible", "Compressed stem; all scales and the roots spring from it. Kept on when dicing "
                 "so the layers hold together.", T["basal_plate"]["density"], interior=False),
            Part("roots", "Roots", "food:RootCrown", None, observability="visible",
                 doc="Dried root stubs of the root crown (trimmed off with the root end)."),
        ]
        for k, path in enumerate(self.root_paths()):
            P.append(Part(f"root_{k + 1}", f"Roots/Root_{k + 1:02d}", "food:Root",
                          ("mesh", lambda path=path: ML.tube(path, np.linspace(0.45e-3, 0.25e-3, len(path)), 6)),
                          "Root", "visible", "", 500.0, interior=False))
        P.append(Part("neck", "Neck", "food:Neck", ("mesh", self.neck_mesh), "DrySkin", "visible",
                      "Dried leaf sheaths above the bulb (cut off as the 'top').", self.p["dry_density"],
                      interior=False, attrs={"length_mm": self.p["neck_length"]}))
        return P

    # ---------------------------------------------------------------- appearance
    def point_uv(self, U, V):
        rho, z = self.curve(self.s_outer)
        k = np.linspace(0, 1, len(rho))
        t = 1 - np.asarray(V)
        r, zz = np.interp(t, k, rho), np.interp(t, k, z)
        th = 2 * np.pi * np.asarray(U)
        m = self.mod(th, zz)
        return np.stack([m * r * np.cos(th), m * r * np.sin(th), zz], -1)

    def uv_of(self, P):
        from scipy.spatial import cKDTree
        rho, z = self.curve(self.s_outer)
        rn, th, zz = self.norm_coords(np.asarray(P, float))
        k = np.linspace(0, 1, len(rho))
        _, i = cKDTree(np.stack([rho, z], 1)).query(np.stack([rn, zz], 1))
        return np.stack([np.mod(th, 2 * np.pi) / (2 * np.pi), 1 - k[i]], 1)

    def skin_albedo(self, P, seed):
        """Dry tunic: golden-brown with darker longitudinal veins (meridional streaks), paler patches where flaked."""
        P = np.asarray(P, float)
        vn = ValueNoise3(seed + 61)
        base = lch_to_srgb(*self.p["skin_LCh"])
        th = np.arctan2(P[..., 1], P[..., 0])
        streak = 0.5 + 0.5 * np.cos(th * 110 + 6 * vn.fbm(P, 0.01, 2))
        col = base * (0.86 + 0.18 * streak[..., None])
        col = col * (1 + 0.10 * vn.fbm(P, 4e-3, 3))[..., None]
        flake = np.clip((vn.fbm(P + 3, 0.012, 3) - 0.35) / 0.3, 0, 1)
        col = col * (1 - 0.35 * flake[..., None]) + lch_to_srgb(78, 25, 80) * 0.35 * flake[..., None]
        return np.clip(col, 0, 1)

    def section_rgb(self, lab, P, seed=0):
        rgb = np.zeros(lab.shape + (3,)) + 0.86
        vn = ValueNoise3(seed + 71)
        T = self.tissues
        names = [t["name"] for t in T]
        flesh = lch_to_srgb(*self.p["flesh_LCh"])
        for i, t in enumerate(T):
            m = lab == i + 1
            if t["name"].startswith("scale_"):
                k = int(t["name"].split("_")[1])
                c = flesh * (1.0 - 0.012 * k)               # inner scales a touch more translucent/yellow
                rgb[m] = c * (1 + 0.02 * vn.fbm(P[m], 0.4e-3, 2))[:, None]
            else:
                rgb[m] = t["colour"]
        # epidermis films: a slightly brighter rim at each scale boundary (the shiny film between rings)
        sc = np.isin(lab, [i + 1 for i, n in enumerate(names) if n.startswith("scale_")])
        if sc.any():
            from scipy import ndimage
            edge = sc & (ndimage.maximum_filter(lab, 3) != ndimage.minimum_filter(lab, 3))
            rgb[edge] = np.clip(rgb[edge] * 1.08, 0, 1)
        return np.clip(rgb, 0, 1)

    # ---------------------------------------------------------------- asset declaration
    def asset_spec(self):
        from .onion_models import onion_beliefs, onion_truth
        p = self.p
        flesh = tuple(float(c) for c in lch_to_srgb(*p["flesh_LCh"]))
        lo, hi = self.bbox()
        ext = float(np.max(hi - lo) * 0.56)
        zc = 0.5 * (self.z_pt + self.curve(1.0)[1].max())
        return {
            "build": {"h": 0.3e-3},
            "textures": [("skin.png", self.point_uv, self.skin_albedo, 2048, 1024)],
            "uv_fns": {f"tunic_{len(self.dry_s)}": self.uv_of, "neck": self.uv_of},
            "materials": {
                "DrySkin": dict(diffuse_reflection_color_image="tex:skin.png", specular_reflection_roughness=0.45,
                                specular_reflection_weight=0.45),
                "FleshOuter": dict(diffuse_reflection_color=tuple(min(1.0, c * 1.02) for c in flesh),
                                   specular_reflection_roughness=0.2, specular_reflection_weight=0.7,
                                   subsurface_weight=0.35, subsurface_scale=0.003, subsurface_scattering_color=flesh),
                "Flesh": dict(diffuse_reflection_color=flesh, specular_reflection_roughness=0.2,
                              specular_reflection_weight=0.7, subsurface_weight=0.35, subsurface_scale=0.003,
                              subsurface_scattering_color=flesh),
                "Apex": dict(diffuse_reflection_color=(0.85, 0.88, 0.62), specular_reflection_roughness=0.3),
                "Plate": dict(diffuse_reflection_color=tuple(float(c) for c in self.colours()["plate"]),
                              specular_reflection_roughness=0.7),
                "Root": dict(diffuse_reflection_color=(0.55, 0.45, 0.30), specular_reflection_roughness=0.85),
            },
            "section_material": {"specular_reflection_roughness": 0.18, "specular_reflection_weight": 0.7,
                                 "subsurface_weight": 0.3, "subsurface_scale": 0.002,
                                 "subsurface_scattering_color": flesh},
            "root": {"peel_state": "dry_skin_on", "trim_state": "untrimmed", "cut_state": "whole",
                     "cooked_state": "raw", "sprouting": "no"},
            "friction": p["friction_skin_rubber"], "restitution": 0.2,
            "cuts": [
                {"name": "halved", "origin": [0, 0, 0], "e1": [0, 1, 0], "e2": [0, 0, 1], "ext": ext,
                 "states": {"half_a": ("pos", {"cut_state": "halved_through_root"}),
                            "half_b": ("neg", {"cut_state": "halved_through_root"})}},
                {"name": "across", "origin": [0, 0, zc], "e1": [1, 0, 0], "e2": [0, 1, 0], "ext": ext,
                 "states": {"top_piece": ("pos", {"cut_state": "cut_across"}),
                            "root_piece": ("neg", {"cut_state": "cut_across"})}},
            ],
            "beliefs": lambda mp: onion_beliefs(self, mp),
            "truth": lambda: onion_truth(self),
        }
