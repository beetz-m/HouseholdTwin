"""Egg twin from a photo: what one RGB image can and cannot tell about an egg, as observations for its belief.

What a single uncalibrated photo tells (each estimate carries a status label):
  OBSERVED   silhouette shape index B/L (= max width / length), the asymmetry of the outline (position of the widest
             point -> Hügelschäffer w, food/egg.py), shell colour (white / brown, CIELAB), a dark crack line on the
             shell; with a scale reference (--length-mm, or a known object) also the length
  ESTIMATED  a visible crack (dark thin line inside the outline; top-hat filter, threshold ILLUSTRATIVE, not calibrated)
  PRIOR      freshness (age, air cell, Haugh unit): the shell does not show it -- the register prior stays; absolute
             size without a scale reference
Segmentation (segment_egg): the egg's outline edge -- Canny on lightness, long straight edges (table edge, skirting)
removed, RANSAC ellipse, then the real asymmetric outline per angle (median-filtered, convex). Colour segmentation (the
avocado's k-means/GrabCut, food/avocado_vision.py) failed on eggs: it caught part of a brown egg and leaked into the
wall around white ones. White balance and skin colour reuse food/avocado_vision.py.
Evaluated on 8 twin renders (scripts/eval_egg_photos.py): see docs/EGG_SDT.md (renders only, no real photo yet).

    est = estimate("photo.jpg", length_mm=None)
    obs = observations(est)        # -> food/egg_belief.Observation list (photo_size only with a scale reference)
"""

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import egg_lifetime as EL
from .avocado_vision import highlight_gains, segment  # generic image tools (no avocado model inside)


@dataclass
class Estimate:
    value: object
    status: str
    note: str = ""


@dataclass
class EggPhoto:
    path: str
    estimates: dict = field(default_factory=dict)
    mask_px: int = 0


def _ell_resid(P, e):
    (cx, cy), (MA, ma), ang = e
    a, b = MA / 2, ma / 2
    th = np.deg2rad(ang)
    c, s = np.cos(th), np.sin(th)
    x, y = P[:, 0] - cx, P[:, 1] - cy
    u, v = c * x + s * y, -s * x + c * y
    r = np.sqrt((u / a) ** 2 + (v / b) ** 2)
    return np.abs(r - 1) * (a + b) / 2, np.arctan2(v / b, u / a)


def segment_egg(bgr, seed=0):
    """Egg silhouette from its outline edge: Canny on lightness, long straight edges (table edge, wall skirting)
    removed, RANSAC ellipse on the edge pixels (support within 2 px, coverage of the angle range), then the actual
    (asymmetric) outline = per angle the edge pixel nearest that ellipse (within 8 %) -> filled polygon."""
    rng = np.random.default_rng(seed)
    h, w = bgr.shape[:2]
    L = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0], (5, 5), 0)
    E = cv2.Canny(L, 8, 24)
    lines = cv2.morphologyEx(E, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (81, 1)))
    lines |= cv2.morphologyEx(E, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 81)))
    E = E & ~cv2.dilate(lines, np.ones((5, 5), np.uint8))
    ys, xs = np.nonzero(E)
    P = np.c_[xs, ys].astype(np.float32)
    if len(P) < 50:
        raise ValueError("no egg outline found")
    best = None
    for _ in range(3000):
        S = P[rng.choice(len(P), 6, replace=False)]
        try:
            e = cv2.fitEllipse(S)
        except cv2.error:
            continue
        (cx, cy), (MA, ma), _a = e
        lo, hi = min(MA, ma), max(MA, ma)
        if not (40 < lo and hi < 0.9 * min(w, h) and 0.55 < lo / hi < 0.97 and 0 < cx < w and 0 < cy < h):
            continue
        d, phi = _ell_resid(P, e)
        on = d < 2.0
        cov = len(np.unique(((phi[on] + np.pi) / (2 * np.pi) * 72).astype(int)))   # angular coverage (of 72 bins)
        score = on.sum() * cov / 72
        if best is None or score > best[0]:
            best = (score, e)
    e = best[1]
    for _ in range(3):                                   # refit on the support
        d, _ = _ell_resid(P, e)
        e = cv2.fitEllipse(P[d < 3.0])
    d, phi = _ell_resid(P, e)
    (cx, cy), (MA, ma), _a = e
    near = d < 0.08 * min(MA, ma) / 2
    th = np.deg2rad(_a)
    c_, s_ = np.cos(th), np.sin(th)
    rel = P - np.array([cx, cy], np.float32)
    u, v = c_ * rel[:, 0] + s_ * rel[:, 1], -s_ * rel[:, 0] + c_ * rel[:, 1]
    bins = ((phi + np.pi) / (2 * np.pi) * 360).astype(int) % 360
    q = np.full(360, np.nan)                               # radial scale of the outline vs the ellipse, per angle
    for k in range(360):
        m = near & (bins == k)
        if m.any():
            j = np.argmin(np.where(m, d, np.inf))
            q[k] = np.hypot(u[j] / (MA / 2), v[j] / (ma / 2))
    ok = np.isfinite(q)
    kk = np.arange(360)
    q = np.interp(kk, np.r_[kk[ok] - 360, kk[ok], kk[ok] + 360], np.r_[q[ok], q[ok], q[ok]])   # gaps (an edge
    q = np.array([np.median(np.take(q, range(k - 7, k + 8), mode="wrap")) for k in kk])   # texture / shadow outliers
    ph = (kk + 0.5) / 360 * 2 * np.pi - np.pi                                                 # crossing a line)
    uu, vv = q * MA / 2 * np.cos(ph), q * ma / 2 * np.sin(ph)
    poly = np.c_[cx + c_ * uu - s_ * vv, cy + s_ * uu + c_ * vv].round().astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [cv2.convexHull(poly)], 1)          # an egg outline is convex
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask.astype(bool), e


def outline(mask):
    """Main axis, length in pixels and the width profile along it (blunt end first)."""
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
    if width[:100].sum() < width[100:].sum():                # blunt end = the wider half first
        width, ax = width[::-1], -ax
    return dict(centre=c, axis=ax, length_px=float(t.max() - t.min()), width_px=width,
                max_width_px=float(width.max()))


def fit_asymmetry(width_px, SI):
    """Hügelschäffer w/L that best matches the normalised width profile (least squares, 5 % ends excluded)."""
    obs = width_px / width_px.max()
    s = np.linspace(0, 1, len(obs))
    keep = (s > 0.05) & (s < 0.95)
    best = None
    for wr in np.linspace(0.0, 0.10, 51):
        _, rho, L = EL.meridian(np.array([1000.0]), np.array([SI]), np.array([wr * 1000.0]), n=len(obs))
        prof = rho[0][::-1]                                   # meridian runs pointed -> blunt; observed blunt first
        prof = prof / prof.max()
        e = float(np.sqrt(np.mean((prof[keep] - obs[keep]) ** 2)))
        if best is None or e < best[1]:
            best = (wr, e)
    return best


def shell_colour(bgr, mask):
    k = max(3, int(0.05 * math.sqrt(mask.sum())))
    m = cv2.erode(mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    gains, _ = highlight_gains(bgr, m)
    img = np.clip(bgr.astype(np.float32) * gains, 0, 255)
    lab = cv2.cvtColor(img / 255.0, cv2.COLOR_BGR2Lab)[m]
    L, a, b = np.median(lab, axis=0)
    C, h = math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360
    return float(L), float(C), float(h), ("brown" if C > 14 and L < 80 else "white"), m


def crack_line(bgr, m, thr=0.06):
    """A dark thin line inside the shell (black top-hat on the L channel, elongated components). Threshold and
    minimum length ILLUSTRATIVE (not calibrated on real cracks)."""
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)[..., 0].astype(np.float32) / 255.0
    r = max(3, int(0.015 * math.sqrt(m.sum())))
    bh = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    line = ((bh > thr) & m).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(line)
    longest = 0.0
    for i in range(1, n):
        x, y, w, h, area = st[i]
        elong = max(w, h) / max(min(w, h), 1)
        if elong > 4 and area > 8:
            longest = max(longest, float(max(w, h)))
    return longest / math.sqrt(m.sum())                       # length relative to the egg's size


def estimate(path, length_mm=None, bbox=None):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    if bbox is None:
        mask, _ = segment_egg(bgr)
    else:
        mask, hull, _ = segment(bgr, bbox)
    o = outline(mask)
    SI = o["max_width_px"] / o["length_px"]
    wr, rms = fit_asymmetry(o["width_px"], SI)
    L_, C_, h_, col, m = shell_colour(bgr, mask)
    crack = crack_line(bgr, m)
    est = {"shape_index": Estimate(round(SI, 4), "OBSERVED", "max width / length of the silhouette"),
           "asymmetry_w_over_L": Estimate(round(wr, 4), "ESTIMATED", f"Hügelschäffer fit, rms {rms:.3f}; biased low "
                                          "(-0.027 on the twin renders): not used as an observation"),
           "shell_colour": Estimate(col, "OBSERVED", f"L* {L_:.0f}, C* {C_:.0f}, h {h_:.0f} (white-balanced)"),
           "crack_line": Estimate(bool(crack > 0.25), "ESTIMATED", f"longest dark line {crack:.2f} x sqrt(area); "
                                  "threshold ILLUSTRATIVE"),
           "freshness": Estimate(None, "PRIOR", "age / air cell / Haugh unit are not visible on the shell")}
    if length_mm:
        est["length_mm"] = Estimate(float(length_mm), "OBSERVED", "scale reference")
        est["asymmetry_w_mm"] = Estimate(round(wr * length_mm, 2), "OBSERVED", "")
    else:
        est["length_mm"] = Estimate(None, "PRIOR", "no scale reference in the image")
    return EggPhoto(path, est, int(mask.sum()))


def observations(photo, t_h=0.0):
    """Belief observations from a photo (food/egg_belief.py): the size only with a scale reference."""
    from .egg_belief import Observation
    e = photo.estimates
    out = []
    if e["length_mm"].value:
        out.append(Observation("photo_size", {"length_mm": e["length_mm"].value, "shape_index": e["shape_index"].value},
                               t_h, f"photo {photo.path}"))
    else:
        out.append(Observation("photo_shape", {"shape_index": e["shape_index"].value}, t_h, f"photo {photo.path}"))
    out.append(Observation("photo_crack", {"seen": e["crack_line"].value}, t_h, f"photo {photo.path}"))
    return out
