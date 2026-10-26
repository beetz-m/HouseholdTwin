"""Cracked and opened egg geometry: shell fragments split along a jagged crack, and the contents as they lie on a
plate or pan (yolk dome on a thick-albumen mound inside a thin-albumen pool).

Quality links (sourced): the yolk index YI = height / diameter of the yolk (fresh ~0.42, falling with age) and the
Haugh unit HU = 100 log10(h - 1.7 w^0.37 + 7.6) with h the thick-albumen height [mm] and w the egg mass [g]
(USDA grades: AA >= 72, A 60-72, B < 60, usda_egg).  HU and YI decline with storage; the decline rates here are
ILLUSTRATIVE (register: open_hu_decline, open_yi_decline).
"""

import numpy as np
import trimesh

from . import meshlib as ML
from .geometry import Meridian


# ------------------------------------------------------------------------------------------------ crack + fragments
def crack_curve(egg, rng, z0=None, amp=1.1e-3, n=48):
    """Jagged crack line round the egg: z_c(theta), piecewise linear through n jittered knots (hair crack facets)."""
    if z0 is None:
        i = int(np.argmax(egg.mer.rho))
        z0 = float(egg.mer.z[i]) - 1.5e-3                  # near the equator, a little towards the pointed end
    th = np.sort(rng.uniform(0, 2 * np.pi, n))
    walk = np.cumsum(rng.normal(0, 0.45, n))
    walk -= np.linspace(0, walk[-1], n)                  # closes on itself
    zz = z0 + amp * walk / (np.abs(walk).max() + 1e-9) + rng.normal(0, 0.25e-3, n)

    def zc(t):
        t = np.mod(t, 2 * np.pi)
        return np.interp(t, np.concatenate([th - 2 * np.pi, th, th + 2 * np.pi]), np.concatenate([zz, zz, zz]))
    return zc, z0


def below_solid(zc, R=0.05, z_bot=-0.08, n=360, ring=(0.004, 0.012, 0.018, 0.024, 0.032, 0.05)):
    """Closed mesh of {r < R, z < z_c(theta)} (the 'lower' side of the crack surface)."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    top = [np.array([0, 0, float(np.mean(zc(th)))])]
    for r in ring:
        top += [np.array([r * np.cos(t), r * np.sin(t), zc(t)]) for t in th]
    top = np.array(top)
    bot = np.array([[R * np.cos(t), R * np.sin(t), z_bot] for t in th] + [[0, 0, z_bot]])
    V = np.vstack([top, bot])
    nb = len(top)
    F = []
    for j in range(n):                                    # centre fan
        F.append((0, 1 + j, 1 + (j + 1) % n))
    for k in range(len(ring) - 1):                        # rings
        a0, b0 = 1 + k * n, 1 + (k + 1) * n
        for j in range(n):
            a, a2, b, b2 = a0 + j, a0 + (j + 1) % n, b0 + j, b0 + (j + 1) % n
            F += [(a, b, a2), (a2, b, b2)]
    o = 1 + (len(ring) - 1) * n                           # side wall from the outer ring down
    for j in range(n):
        a, a2 = o + j, o + (j + 1) % n
        b, b2 = nb + j, nb + (j + 1) % n
        F += [(a, b, a2), (a2, b, b2)]
    c = nb + n
    for j in range(n):                                    # bottom
        F.append((c, nb + (j + 1) % n, nb + j))
    m = trimesh.Trimesh(V, np.array(F), process=True)
    if m.volume < 0:
        m.invert()
    return m


def split(mesh, solid):
    lo = mesh.intersection(solid, engine="manifold")
    hi = mesh.difference(solid, engine="manifold")
    return lo, hi


# ------------------------------------------------------------------------------------------------ opened contents
def haugh_unit(h_mm, w_g):
    return 100 * np.log10(max(h_mm - 1.7 * w_g ** 0.37 + 7.6, 1e-3))


def thick_height_for_hu(hu, w_g):
    return 10 ** (hu / 100) + 1.7 * w_g ** 0.37 - 7.6


OVERRIDE = {}      # {"haugh_unit": .., "yolk_index": ..} from the freshness clock (food/egg_emulator.py)


def quality(egg, mass_g):
    """HU and YI from age and storage temperature (ILLUSTRATIVE decline rates), unless the emulator sets the values
    of its freshness clock (food/egg_lifetime.py, DERIVED from pan2026) in OVERRIDE."""
    if OVERRIDE:
        hu, yi = float(OVERRIDE["haugh_unit"]), float(OVERRIDE["yolk_index"])
        return {"haugh_unit": hu, "yolk_index": yi, "usda_albumen_grade": "AA" if hu >= 72 else "A" if hu >= 60 else "B",
                "thick_height_mm": float(max(thick_height_for_hu(hu, mass_g), 0.8))}
    age = egg.p["age_days"]
    fridge = egg.p["storage_temp_C"] < 10
    hu = 88 - (0.45 if fridge else 1.6) * age
    yi = 0.44 - (0.003 if fridge else 0.008) * age
    return {"haugh_unit": float(hu), "yolk_index": float(np.clip(yi, 0.22, 0.46)),
            "usda_albumen_grade": "AA" if hu >= 72 else "A" if hu >= 60 else "B",
            "thick_height_mm": float(thick_height_for_hu(hu, mass_g))}


def outline_mod(rng, amp=0.08, k=(2, 3, 5, 7)):
    """Irregular puddle outline: rho scale m(theta) = 1 + sum of a few random harmonics (amp = total)."""
    a = rng.normal(0, 1, len(k)) * amp / np.sqrt(len(k)) / np.array(k) ** 0.5
    ph = rng.uniform(0, 2 * np.pi, len(k))
    return lambda th, z: 1 + sum(ai * np.cos(ki * th + pi) for ai, ki, pi in zip(a, k, ph))


def mound(volume=None, height=1e-3, R=None, n=120, z0=0.0, p=4.0, mod=None):
    """Sessile puddle/dome: rho(z) = R (1 - (z/h)^p)^(1/p) (p ~ 2.5 drop-like dome, larger p = flatter top).
    Either the volume or the radius R is given."""
    zs = np.linspace(0, 1, 400)
    fill = np.trapezoid(((1 - zs ** p) ** (1 / p)) ** 2, zs)
    if R is None:
        R = np.sqrt(volume / (np.pi * height * fill))
    z = np.linspace(height, 0, n)
    rho = R * np.clip(1 - (z / height) ** p, 0, 1) ** (1 / p)
    rho = np.concatenate([rho, [0.0]])
    z = np.concatenate([z, [0.0]]) + z0
    return ML.surface_of_revolution(Meridian(rho, z), 160, modulation=mod), R


def opened_contents(egg, info, mass_g, rng, offset=(0.0, 0.0), yolk_broken=False):
    """Meshes (on a flat surface at z = 0) for the released contents, keyed like the whole egg's part paths.
    yolk_broken: the vitelline membrane tore, the yolk runs out as an irregular flat pool over the thick albumen
    (pool height 4.5 mm and outline irregularity ILLUSTRATIVE)."""
    q = quality(egg, mass_g)
    vol = lambda p: info[p]["volume_cm3"] * 1e-6  # noqa: E731
    V_thin = vol("Albumen/OuterThinAlbumen") + vol("Albumen/InnerThinAlbumen")
    V_thick = vol("Albumen/ThickAlbumen") + vol("Albumen/ChalaziferousLayer")
    V_yolk = sum(vol(p) for p in ("Yolk/YolkBody", "Yolk/VitellineMembrane", "Yolk/Latebra", "Yolk/LatebraNeck",
                                  "Yolk/NucleusOfPander"))
    h_thick = q["thick_height_mm"] * 1e-3
    thin_h = 2.2e-3                                        # ILLUSTRATIVE thin-albumen film height
    m_thick, m_thin = outline_mod(rng, 0.10), outline_mod(rng, 0.22)
    thick, R_thick = mound(V_thick, h_thick, p=3.0, mod=m_thick)
    R_thin = np.sqrt(V_thin / (np.pi * thin_h * 0.93) + R_thick ** 2)   # thin albumen rings the thick mound
    pool, _ = mound(height=thin_h, R=R_thin, p=6.0, mod=m_thin)
    thin = pool.difference(thick, engine="manifold")
    yi = q["yolk_index"]
    p_y = 2.5
    zs = np.linspace(0, 1, 400)
    fill = np.trapezoid(((1 - zs ** p_y) ** (1 / p_y)) ** 2, zs)
    base = h_thick * 0.45                                  # the yolk settles into the thick albumen
    # yolk index as measured (USDA / tripod micrometer): height of the yolk TOP above the plate / yolk diameter, so
    # base + dome height = 2 a YI with V = pi a^2 h_dome fill. (Before 2026-10-21 the dome alone was 2 a YI: the
    # rendered yolk then measured ~0.1 higher than its stated index.)
    lo_, hi_ = (base / (2 * yi)) * 1.0001, 0.08
    for _ in range(80):
        a = 0.5 * (lo_ + hi_)
        if np.pi * a * a * (2 * a * yi - base) * fill > V_yolk:
            hi_ = a
        else:
            lo_ = a
    hy = 2 * a * yi - base
    if yolk_broken:                                        # torn membrane: flat irregular yolk pool, same volume
        hy, base = 4.5e-3, h_thick * 0.55
        yolk, a = mound(V_yolk, height=hy, p=5.0, z0=base, mod=outline_mod(rng, 0.32, k=(2, 3, 4, 6, 9)))
    else:
        yolk, _ = mound(height=hy, R=a, p=p_y, z0=base)
    overlap = thick.intersection(yolk, engine="manifold").volume      # albumen displaced by the settled yolk
    thick, R_thick = mound(V_thick + overlap, h_thick, p=3.0, mod=m_thick)
    thick = thick.difference(yolk, engine="manifold")
    c, cz = hy / 2, base + hy / 2
    disc = ML.ellipsoid([0, 0, base + hy - 0.12e-3], (egg.d_disc / 2, egg.d_disc / 2, 0.2e-3), n_u=12, n_v=24)
    chal = []
    for sgn in (1, -1):                                    # chalazae lie on the thick albumen next to the yolk
        s = np.linspace(0, 1, 60)
        x0 = sgn * (a * 0.9)
        path = np.stack([x0 + sgn * 7e-3 * s, 0.4e-3 * np.sin(2 * np.pi * 3 * s), h_thick * (0.85 - 0.25 * s)], 1)
        chal.append(ML.tube(path, 0.22e-3, 8))
    out = {"Albumen/ThinAlbumenPool": thin, "Albumen/ThickAlbumen": thick, "Yolk/YolkBody": yolk,
           "Yolk/GerminalDisc": disc, "Albumen/Chalazae/BluntEnd": chal[0], "Albumen/Chalazae/PointedEnd": chal[1]}
    for m in out.values():
        m.apply_translation([offset[0], offset[1], 0])
    q.update({"R_thin_mm": R_thin * 1e3, "R_thick_mm": R_thick * 1e3, "yolk_diameter_mm": 2 * a * 1e3,
              "yolk_height_mm": (base + hy) * 1e3, "yolk_dome_mm": hy * 1e3})   # height above the plate (measured)
    return out, q
