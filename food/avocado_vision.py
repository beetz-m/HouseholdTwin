"""Avocado twin from a photo: estimate the instance parameters and the robot's beliefs from one RGB image.

What a single uncalibrated photo can and cannot tell (each estimate carries a status label):
  OBSERVED     measured in the image (silhouette aspect ratio, neck taper, skin colour, stem button)
  ESTIMATED    derived from an observation through a model with assumptions (ripeness stage from colour, tubercle
               size from texture; ILLUSTRATIVE calibrations stated with each)
  PRIOR        not observable here, taken from the parameter register (absolute size without a scale reference,
               seed, interior); a measured length (--length-mm) turns the size into OBSERVED
The ripeness posterior uses the twin's own colour chart (avocado.skin_colour: ashton2006 L* 40 -> 27, C* 15 -> 3,
h 125 -> 250 over stages 1..5) with wide noise for unknown white balance and exposure.
"""

import hashlib
import math
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from . import params as PR


@dataclass
class Estimate:
    value: float | str
    sd: float | None
    unit: str
    status: str            # OBSERVED / ESTIMATED / PRIOR
    method: str


@dataclass
class AvocadoObservation:
    image: str
    image_sha1: str
    estimates: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)

    def as_dict(self):
        d = asdict(self)
        d["estimates"] = {k: asdict(v) for k, v in self.estimates.items()}
        return d


# %% segmentation
def fruit_seed(bgr, k=5):
    """Coarse fruit region: the colour cluster (k-means in CIELAB) whose largest blob looks most like a fruit -
    compact and convex (solidity), elongated 1.05..2.5, of a fair size and not touching the image border."""
    h, w = bgr.shape[:2]
    sc = 480.0 / max(h, w)
    small = cv2.resize(bgr, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    X = lab.reshape(-1, 3) * np.array([0.6, 1.0, 1.0], np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    _, lbl, _ = cv2.kmeans(X, k, None, crit, 4, cv2.KMEANS_PP_CENTERS)
    lbl = lbl.reshape(small.shape[:2])
    best = None
    for j in range(k):
        m = cv2.morphologyEx((lbl == j).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n, cc, st, _ = cv2.connectedComponentsWithStats(m)
        for i in range(1, n):
            x, y, ww, hh, area = st[i]
            frac = area / m.size
            if frac < 0.01 or x <= 1 or y <= 1 or x + ww >= m.shape[1] - 1 or y + hh >= m.shape[0] - 1:
                continue
            comp = (cc == i).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            hull = cv2.convexHull(cnts[0])
            solidity = area / max(cv2.contourArea(hull), 1.0)
            if len(cnts[0]) < 5:
                continue
            (_, _), (e1, e2), _ = cv2.fitEllipse(cnts[0])
            aspect = max(e1, e2) / max(min(e1, e2), 1e-6)
            score = solidity ** 6 * min(frac / 0.04, 1.0) * (1.0 if 1.05 < aspect < 2.5 else 0.2)
            if best is None or score > best[0]:
                best = (score, comp)
    if best is None:
        raise ValueError("no fruit-like region found in the image; pass --bbox x,y,w,h")
    return cv2.resize(best[1], (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def segment(bgr, bbox=None):
    """Binary mask of the fruit and its outer contour. GrabCut is seeded with the fruit-like colour region
    (:func:`fruit_seed`, or the rectangle ``bbox`` = x, y, w, h): its eroded core is foreground, everything beyond
    its dilation background; largest component, holes filled."""
    h, w = bgr.shape[:2]
    mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    if bbox is None:
        seed = fruit_seed(bgr).astype(np.uint8)
        r = max(5, int(0.04 * np.sqrt(seed.sum())))
        mask[cv2.dilate(seed, np.ones((2 * r + 1, 2 * r + 1), np.uint8)).astype(bool)] = cv2.GC_PR_BGD
        mask[seed.astype(bool)] = cv2.GC_PR_FGD
        mask[cv2.erode(seed, np.ones((2 * r + 1, 2 * r + 1), np.uint8)).astype(bool)] = cv2.GC_FGD
    else:
        x0, y0, x1, y1 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
        mask[y0:y1, x0:x1] = cv2.GC_PR_FGD
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, None, bg, fg, 6, cv2.GC_INIT_WITH_MASK)
    m = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        raise ValueError("no fruit found in the image")
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab_cc == k).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 1, cv2.FILLED)
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)                                  # the fruit's outline is convex: bright, glossy flanks
    hull_mask = np.zeros_like(filled)                         # lost to the background are filled back in
    cv2.drawContours(hull_mask, [hull], -1, 1, cv2.FILLED)
    gain = float(hull_mask.sum() / max(filled.sum(), 1))
    return hull_mask.astype(bool), hull, gain


# %% shape
def model_profile(neck, n=200):
    """Normalised half-width y(s)/max of the twin's meridian (food/avocado.py), s = 0 blossom end .. 1 stem end."""
    L, D = 1.0, 1.0
    x = np.linspace(-L / 2, L / 2, 2400)
    w = 0.13 * L
    y = D / 2 * np.sqrt(np.clip((L ** 2 - 4 * x ** 2) / (L ** 2 + 8 * w * x + 4 * w ** 2), 0, None))
    s = (x + L / 2) / L
    g = np.clip((s - 0.45) / 0.45, 0, 1)
    y = y * (1 - neck * g ** 2 * (3 - 2 * g))
    si = np.linspace(0, 1, n)
    return si, np.interp(si, s, y) / y.max()


def shape(mask):
    """Main axis, length in pixels, width profile and the stem / blossom ends of the silhouette."""
    ys, xs = np.nonzero(mask)
    P = np.c_[xs, ys].astype(float)
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    ax, perp = Vt[0], Vt[1]
    t, u = (P - c) @ ax, (P - c) @ perp
    edges = np.linspace(t.min(), t.max(), 201)
    idx = np.clip(np.digitize(t, edges) - 1, 0, 199)
    width = np.array([np.ptp(u[idx == i]) if np.any(idx == i) else 0.0 for i in range(200)])
    width = np.convolve(width, np.ones(5) / 5, mode="same")
    q = 200 // 5                                              # compare widths at 20 % from each end
    t_lo, t_hi = float(t.min()), float(t.max())
    if width[q:2 * q].mean() > width[-2 * q:-q].mean():      # the stem end is the narrow end
        s_prof = width                                        # t increasing towards the stem
    else:
        s_prof, ax, t_lo, t_hi = width[::-1], -ax, -t_hi, -t_lo
    return dict(centre=c, axis=ax, perp=perp, length_px=t_hi - t_lo, width_px=s_prof,
                max_width_px=float(width.max()), t_min=t_lo, t_max=t_hi)


def fit_neck(width_px):
    """Neck taper of the twin's meridian that best matches the observed profile (least squares on normalised
    widths, ends excluded); returns (neck, rms)."""
    s = np.linspace(0, 1, len(width_px))
    obs = width_px / width_px.max()
    sel = (s > 0.06) & (s < 0.94)
    best = None
    for neck in np.linspace(0.0, 0.40, 81):
        sm, ym = model_profile(neck, len(width_px))
        r = float(np.sqrt(np.mean((obs[sel] - ym[sel]) ** 2)))
        if best is None or r < best[1]:
            best = (float(neck), r)
    return best


# %% colour -> ripeness
def highlight_gains(bgr, m):
    """White balance from the fruit's own specular highlights (dichromatic reflection: the gloss of the cuticle has
    the colour of the light): per-channel gains that make the brightest unsaturated skin pixels neutral."""
    px = bgr[m].astype(np.float32)
    lum = px.mean(1)
    ok = px.max(1) < 250
    top = px[ok & (lum >= np.percentile(lum[ok], 98.5))]
    if len(top) < 20:
        return np.ones(3, np.float32), 0
    hi = top.mean(0)
    return (hi.mean() / np.maximum(hi, 1.0)).astype(np.float32), len(top)


def skin_lch(bgr, mask, white_balance=True):
    """Median CIELAB L*, C*, h of the skin (mask eroded; specular highlights and deep shadow excluded), after the
    white balance of :func:`highlight_gains` unless disabled."""
    k = max(3, int(0.04 * math.sqrt(mask.sum())))
    m = cv2.erode(mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    gains, n_hi = highlight_gains(bgr, m) if white_balance else (np.ones(3, np.float32), 0)
    img = np.clip(bgr.astype(np.float32) * gains, 0, 255)
    lab = cv2.cvtColor(img / 255.0, cv2.COLOR_BGR2Lab)[m]
    L = lab[:, 0]
    keep = (L > np.percentile(L, 5)) & (L < np.percentile(L, 90))
    Lm, am, bm = np.median(lab[keep], axis=0)
    return float(Lm), float(math.hypot(am, bm)), float(math.degrees(math.atan2(bm, am)) % 360), m, gains, n_hi


def ripeness_posterior(L, C, h, sd_L=9.0, sd_C=4.0):
    """Posterior over the continuous stage 1..5 (flat prior) given skin L*, C*, h.
    L* and C* use the twin's colour chart (avocado.skin_colour: L* 40 -> 27, C* 15 -> 3). Hue enters only as
    'green or not' (Hass turns from green to purple-black: stages 1-2 green, from 3 on not): a green hue
    (90..150 deg) supports the early stages, any other hue the late ones, weighted by chroma because the hue of
    near-black skin is meaningless. Noise widths and the green-band likelihoods are ILLUSTRATIVE."""
    s = np.linspace(1, 5, 401)
    t = (s - 1) / 4
    Lp, Cp = 40 - 13 * t, 15 - 12 * t
    green_share = np.clip((3.0 - s) / 2.0, 0.0, 1.0)
    is_green = 90.0 <= h <= 150.0
    p_hue = green_share * 0.9 + (1 - green_share) * 0.1 if is_green else green_share * 0.1 + (1 - green_share) * 0.9
    w_h = min(1.0, C / 8.0)
    ll = -0.5 * (((L - Lp) / sd_L) ** 2 + ((C - Cp) / sd_C) ** 2) + w_h * np.log(p_hue)
    p = np.exp(ll - ll.max())
    p /= p.sum()
    mean = float((s * p).sum())
    return mean, float(math.sqrt(((s - mean) ** 2 * p).sum())), s, p


# %% texture and stem button
def skin_relief(bgr, mask, d_px):
    """Relief index of the pebbled skin (std of high-passed luminance / mean, over the eroded mask) and the
    dominant feature wavelength in pixels (radial power spectrum of a central patch)."""
    k = max(3, int(0.08 * d_px))
    m = cv2.erode(mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), max(1.0, 0.03 * d_px))
    relief = float(hp[m].std() / max(g[m].mean(), 1.0))
    ys, xs = np.nonzero(m)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    r = int(0.22 * d_px)
    patch = hp[max(cy - r, 0):cy + r, max(cx - r, 0):cx + r]
    patch = patch - patch.mean()
    F = np.abs(np.fft.fftshift(np.fft.fft2(patch * np.outer(np.hanning(patch.shape[0]), np.hanning(patch.shape[1])))))
    yy, xx = np.indices(F.shape)
    rr = np.hypot(yy - F.shape[0] / 2, xx - F.shape[1] / 2).astype(int)
    radial = np.bincount(rr.ravel(), F.ravel()) / np.maximum(np.bincount(rr.ravel()), 1)
    f = np.arange(len(radial)) / patch.shape[0]
    band = (f > 3.0 / patch.shape[0]) & (f < 0.25)
    fpk = f[band][np.argmax(radial[band] * f[band])]
    return relief, float(1.0 / fpk)


def stem_button(bgr, sh, skin_L, skin_C):
    """Stem button at the narrow end: a small region near the axis, lighter and more chromatic (brown) than the
    skin. Returns (present, length_px, centre_xy)."""
    lab = cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    c, ax, perp = sh["centre"], sh["axis"], sh["perp"]
    h, w = bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    P = np.stack([xx - c[0], yy - c[1]], -1).astype(float)
    t, u = P @ ax, P @ perp
    t_tip = sh["t_max"]                                       # the axis points to the stem end
    L_ = sh["length_px"]
    zone = (t > t_tip - 0.12 * L_) & (t < t_tip + 0.04 * L_) & (np.abs(u) < 0.08 * L_)
    Lc, ac, bc = lab[..., 0], lab[..., 1], lab[..., 2]
    cand = zone & (Lc > skin_L + 10) & (np.hypot(ac, bc) > skin_C + 6)
    if cand.sum() < 12:
        return False, 0.0, None
    tt = t[cand]
    ys, xs = np.nonzero(cand)
    return True, float(np.percentile(tt, 98) - np.percentile(tt, 2)), (float(xs.mean()), float(ys.mean()))


# %% the whole estimate
def estimate(path, length_mm=None, bbox=None):
    """Estimate the avocado's instance parameters and beliefs from the image at ``path``."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    s = 1000.0 / max(bgr.shape[:2])
    if s < 1.0:
        bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        bbox = None if bbox is None else [int(v * s) for v in bbox]
    obs = AvocadoObservation(image=path, image_sha1=hashlib.sha1(open(path, "rb").read()).hexdigest()[:12])
    mask, contour, hull_gain = segment(bgr, bbox)
    if hull_gain > 1.08:
        obs.flags.append(f"segmentation filled to its convex hull (+{100 * (hull_gain - 1):.0f} % area): parts of the "
                         "skin were lost to the background (highlights)")
    sh = shape(mask)
    E = obs.estimates
    aspect = sh["length_px"] / sh["max_width_px"]
    E["aspect_L_over_D"] = Estimate(round(aspect, 3), 0.03 * aspect, "1", "OBSERVED",
                                    "silhouette length / maximum width along the principal axis (side view assumed)")
    neck, rms = fit_neck(sh["width_px"])
    E["neck"] = Estimate(round(neck, 3), 0.03, "1", "OBSERVED",
                         f"least-squares fit of the twin's meridian to the silhouette profile (rms {rms:.3f})")
    if rms > 0.06:
        obs.flags.append(f"silhouette fits the meridian model poorly (rms {rms:.3f}): unusual shape or a tilted view")
    reg = PR.AVOCADO
    if length_mm is not None:
        E["length_L"] = Estimate(float(length_mm), 1.0, "mm", "OBSERVED", "measured by the user")
    else:
        mu, sd = reg["length_L"][1][1], reg["length_L"][1][2]
        E["length_L"] = Estimate(mu, sd, "mm", "PRIOR", "register prior (no scale reference in the image); "
                                 "pass --length-mm to measure it")
    Lmm, Lsd = E["length_L"].value, E["length_L"].sd
    E["diameter_D"] = Estimate(round(Lmm / aspect, 1), round(math.hypot(Lsd / aspect, Lmm * 0.03 / aspect), 1), "mm",
                               "ESTIMATED" if length_mm is None else "OBSERVED", "length / observed aspect ratio")
    sL, sC, sh_h, skin_mask, gains, n_hi = skin_lch(bgr, mask)
    E["white_balance_gains_BGR"] = Estimate(", ".join(f"{g:.2f}" for g in gains), None, "-", "OBSERVED",
                                            f"from {n_hi} specular-highlight pixels on the skin (dichromatic model)")
    if n_hi == 0:
        obs.flags.append("no usable specular highlights: colour not white-balanced")
    E["skin_Lab_L"] = Estimate(round(sL, 1), None, "L*", "OBSERVED", "median of the skin (highlights / shadow out)")
    E["skin_chroma"] = Estimate(round(sC, 1), None, "C*", "OBSERVED", "median chroma of the skin")
    E["skin_hue"] = Estimate(round(sh_h, 1), None, "deg", "OBSERVED", "median hue angle of the skin")
    mean, sd, s_grid, post = ripeness_posterior(sL, sC, sh_h)
    E["ripeness_stage"] = Estimate(round(mean, 2), round(sd, 2), "1..5", "ESTIMATED",
                                   "posterior over the Avocados Australia chart: skin L*, C* against the twin's "
                                   "colour model, hue as green / not green (likelihoods ILLUSTRATIVE)")
    obs.flags.append("colour balanced on the fruit's highlights only (no reference card): the ripeness posterior is "
                     "wide by design")
    px_per_mm = sh["length_px"] / Lmm
    relief, lam_px = skin_relief(bgr, mask, sh["max_width_px"])
    E["skin_relief_index"] = Estimate(round(relief, 3), None, "1", "OBSERVED",
                                      "std of high-passed luminance / mean luminance on the skin")
    lo, hi = reg["pebble_scale"][1][1], reg["pebble_scale"][1][2]
    peb = float(np.clip(0.5 * lam_px / px_per_mm, lo, hi))
    E["pebble_scale"] = Estimate(round(peb, 2), round(0.25 * (hi - lo), 2), "mm", "ESTIMATED",
                                 f"half the dominant texture wavelength ({lam_px:.1f} px) at the image scale, clipped "
                                 f"to the register range {lo}-{hi} mm (ILLUSTRATIVE mapping)")
    present, sb_px, sb_xy = stem_button(bgr, sh, sL, sC)
    E["stem_button"] = Estimate("present" if present else "not seen", None, "-", "OBSERVED",
                                "lighter, browner region at the narrow end near the axis")
    sb_lo, sb_hi = reg["stem_button_length"][1][1], reg["stem_button_length"][1][2]
    if present:
        E["stem_button_length"] = Estimate(round(float(np.clip(0.5 * sb_px / px_per_mm, 0.5, 2 * sb_hi)), 1), 1.0, "mm",
                                           "ESTIMATED", "visible length along the axis / 2 (side view of a clipped "
                                           "button, ILLUSTRATIVE)")
    else:
        E["stem_button_length"] = Estimate(0.5, None, "mm", "ESTIMATED", "no button seen: minimal stub")
    upright = abs(sh["axis"][1]) > 0.8 and sh["axis"][1] < 0
    E["pose"] = Estimate("standing on the blossom end" if upright else "lying / tilted", None, "-", "OBSERVED",
                         "direction of the principal axis from the wide to the narrow end in the image")
    E["interior"] = Estimate("not observable", None, "-", "PRIOR",
                             "seed, flesh bands, bruising: register priors until the fruit is cut or squeezed")
    debug = dict(mask=mask, contour=contour, shape=sh, skin_mask=skin_mask, stem_xy=sb_xy, ripeness=(s_grid, post),
                 bgr=bgr, neck_rms=rms)
    return obs, debug


def instance_overrides(obs):
    """Register overrides that make the twin's geometry and state follow the observation."""
    E = obs.estimates
    return {"length_L": float(E["length_L"].value), "diameter_D": float(E["diameter_D"].value),
            "neck": float(E["neck"].value), "pebble_scale": float(E["pebble_scale"].value),
            "stem_button_length": float(E["stem_button_length"].value),
            "ripeness_stage": float(E["ripeness_stage"].value)}


def belief_update(obs, name):
    """SDT belief-layer attributes (root prim) from the observation: means, standard deviations and sources."""
    E = obs.estimates
    src = f"photo {name} ({obs.image_sha1})"
    B = {"sdt:epistemicStatus": "photo observation + register priors",
         "sdt:belief:ripeness_stage_mean": E["ripeness_stage"].value,
         "sdt:belief:ripeness_stage_sd": E["ripeness_stage"].sd,
         "sdt:belief:ripeness_stage_source": f"skin colour in {src}; squeeze to update",
         "sdt:belief:length_mm": E["length_L"].value, "sdt:belief:length_mm_sd": E["length_L"].sd,
         "sdt:belief:length_mm_source": E["length_L"].method,
         "sdt:belief:diameter_mm": E["diameter_D"].value, "sdt:belief:diameter_mm_sd": E["diameter_D"].sd,
         "sdt:belief:aspect_L_over_D": E["aspect_L_over_D"].value,
         "sdt:belief:neck": E["neck"].value,
         "sdt:belief:stem_button": E["stem_button"].value,
         "sdt:belief:skin_relief_index": E["skin_relief_index"].value,
         "sdt:belief:pose": E["pose"].value,
         "sdt:belief:observation_image": obs.image,
         "sdt:belief:mass_g_source": "volume of the photo-shaped twin x register densities (absolute size: "
                                     + E["length_L"].status.lower() + ")"}
    return {"": B}
