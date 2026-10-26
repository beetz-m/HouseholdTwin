"""Vision monitor for an egg opened into a bowl / onto a plate: shell shards in the contents, a broken yolk, and the
egg's freshness read off the contents (yolk index, thick-albumen height -> Haugh unit) -- the execution monitor of
CrackEgg / OpenUp and the 'opened' observation of the egg's belief (food/egg_belief.py).

Inputs: RGB, depth (distance to the image plane, m), camera intrinsics and pose (as food/peel_monitor.py), optionally
the egg's mass (the belief's mean) for the Haugh unit. Steps:
  plate plane = the dominant height; contents = everything > 0.4 mm above it (largest blob + what lies inside its hull)
  yolk        = saturated orange-yellow pixels (HSV), largest blob; height = its highest point above the plate,
                diameter = equivalent circle of its footprint (world xy) -> yolk index
  albumen     = contents minus yolk and shards; thick-albumen height = median height in a ring 5-11 mm outside the
                yolk edge (where the Haugh measurement is taken, usda_egg)
  shards      = shell-coloured small blobs on the contents (brown shell: hue 10-40 deg, saturation 0.25-0.65, darker
                than the yolk; white shell: a >= 0.15 mm bump on the contents surface in depth, top-hat over ~10 mm);
                thresholds ILLUSTRATIVE, calibrated on the twin's renders only (scripts/eval_egg_bowl_monitor.py)
  yolk broken = yolk index < 0.2, or a ragged / fragmented yolk (circularity < 0.7 or > 1 large yolk blob)
Verdict: ok | fish out shards (n) | yolk broken (use for scrambled / baking) -- with reasons.
"""

import math
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from . import egg_lifetime as EL
from .peel_monitor import back_project, board_height, camera


@dataclass
class BowlReport:
    shards: int
    shard_area_mm2: float
    yolk_broken: bool
    yolk_index: float
    yolk_height_mm: float
    yolk_diameter_mm: float
    albumen_height_mm: float
    haugh_unit: float
    verdict: str
    reasons: list = field(default_factory=list)
    masks: dict = field(default_factory=dict)

    def as_dict(self):
        d = asdict(self)
        d.pop("masks")
        return d


def _footprint_area(Pw, mask, intr, depth):
    """Horizontal area (m^2) of the masked pixels: footprint of each pixel's ray patch on its surface height."""
    fx = camera(intr)[0]
    a = (depth / fx) ** 2                                   # area at the image plane distance
    M = np.asarray(intr["cam_to_world"], float)
    view = -M[:3, 2]                                       # camera viewing direction (world)
    cos_t = abs(view[2]) / np.linalg.norm(view)
    return float(np.sum(a[mask]) / max(cos_t, 1e-3))


def judge(rgb, depth, intr, mass_g=60.0, shell="auto"):
    Pw = back_project(depth, intr)
    valid = np.isfinite(depth) & (depth > 0) & (depth < 5)
    z0 = board_height(Pw, valid)
    hgt = (Pw[..., 2] - z0) * 1e3                          # mm above the plate
    cont = (valid & (hgt > 0.4)).astype(np.uint8)
    cont = cv2.morphologyEx(cont, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, cc, st, _ = cv2.connectedComponentsWithStats(cont)
    if n <= 1:
        return BowlReport(0, 0.0, False, float("nan"), 0, 0, 0, float("nan"), "not judged", ["no contents found"])
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    main = (cc == k).astype(np.uint8)
    cnts, _ = cv2.findContours(main, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = np.zeros_like(main)
    cv2.drawContours(hull, [cv2.convexHull(max(cnts, key=cv2.contourArea))], -1, 1, cv2.FILLED)
    contents = hull.astype(bool) & valid
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    H, S, V = hsv[..., 0] * 2.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    yolk_c = contents & (H >= 15) & (H <= 55) & (S > 0.62) & (V > 0.45)
    yolk_c = cv2.morphologyEx(yolk_c.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ny, ccy, sty, _ = cv2.connectedComponentsWithStats(yolk_c)
    reasons = []
    if ny <= 1:
        return BowlReport(0, 0.0, True, 0.0, 0, 0, 0, float("nan"), "not judged", ["no yolk found"])
    areas = sty[1:, cv2.CC_STAT_AREA]
    ky = 1 + int(np.argmax(areas))
    yolk = ccy == ky
    big = int(np.sum(areas > 0.15 * areas.max()))
    # the saturated core misses the shaded rim of the dome: grow it into yolk-hued, less saturated pixels connected to
    # it (hysteresis), then measure the diameter from the outline (circle fit in world xy) rather than the pixel area
    weak = valid & (H >= 12) & (H <= 58) & (S > 0.30) & (V > 0.25)
    for _ in range(60):
        g = cv2.dilate(yolk.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & weak
        if g.sum() == yolk.sum():
            break
        yolk = g | yolk
    yolk = cv2.morphologyEx(yolk.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    # yolk geometry in world units
    A_y = _footprint_area(Pw, yolk, intr, depth)
    d_y = 2 * math.sqrt(A_y / math.pi) * 1e3
    bc, _ = cv2.findContours(yolk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if bc and big == 1:                                 # whole yolk: least-squares circle through the outline (xy)
        b_ = max(bc, key=cv2.contourArea)[:, 0, :]
        xy = Pw[b_[:, 1], b_[:, 0], :2]
        ok = np.isfinite(xy).all(1)
        if ok.sum() > 20:
            A_ = np.c_[2 * xy[ok], np.ones(ok.sum())]
            sol, *_ = np.linalg.lstsq(A_, (xy[ok] ** 2).sum(1), rcond=None)
            r_c = math.sqrt(max(sol[2] + sol[0] ** 2 + sol[1] ** 2, 0.0))
            d_y = 2 * r_c * 1e3
    h_y = float(np.percentile(hgt[yolk], 99))
    yi = h_y / max(d_y, 1e-6)
    ycn, _ = cv2.findContours(yolk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(ycn, key=cv2.contourArea)
    circ = 4 * math.pi * cv2.contourArea(c) / max(cv2.arcLength(c, True) ** 2, 1e-6)
    # shards: small opaque pieces outside the yolk. Clear albumen is (nearly) invisible to depth -- in the twin's
    # renders exactly, for real RGB-D sensors largely -- so a shard lying on it shows as a small blob raised above the
    # plate plane; the shell colour (white / brown) is known from the egg's photo (food/egg_vision.py, OBSERVED).
    # Thresholds ILLUSTRATIVE, set on the twin's renders only.
    fx = camera(intr)[0]
    px_mm = float(np.nanmedian(depth[valid])) / fx * 1e3            # mm per pixel at the plate
    ring_out = cv2.dilate(yolk.astype(np.uint8), np.ones((int(6 / px_mm) | 1,) * 2, np.uint8)).astype(bool)
    raised = valid & ~ring_out & (hgt > 0.15) & (hgt < 8.0)
    if shell == "brown":
        col = (H >= 10) & (H <= 40) & (S > 0.25) & (S < 0.65) & (V > 0.25) & (V < 0.85)
    elif shell == "white":
        col = S < 0.15
    else:
        col = np.ones_like(raised)
    cand = cv2.morphologyEx((raised & col).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ns, ccs, sts, _ = cv2.connectedComponentsWithStats(cand)
    shard_mask = np.zeros_like(contents)
    n_sh, a_sh = 0, 0.0
    for i in range(1, ns):
        m = ccs == i
        a = _footprint_area(Pw, m, intr, depth) * 1e6
        if 3.0 <= a <= 80.0:                                # mm^2: a shard, not the plate rim or a reflection
            n_sh += 1
            a_sh += a
            shard_mask |= m
    # albumen height in the Haugh ring (5..11 mm outside the yolk edge)
    ys, xs = np.nonzero(yolk)
    cxy = Pw[ys, xs, :2].mean(0)
    r = np.linalg.norm(Pw[..., :2] - cxy, axis=-1) * 1e3
    ring = contents & ~yolk & ~shard_mask & (r > d_y / 2 + 5) & (r < d_y / 2 + 11)
    h_alb = float(np.median(hgt[ring])) if ring.sum() > 20 else float("nan")
    if yi < 0.2 or circ < 0.7 or big > 1:               # broken yolk: the ring lies on the yolk pool, not on albumen
        h_alb = float("nan")
        reasons.append("albumen height not measured: yolk broken (no Haugh measurement point)")
    elif np.isfinite(h_alb) and h_alb < 0.5:              # the ring sees the plate through clear albumen
        h_alb = float("nan")
        reasons.append("albumen height not observed: clear albumen is transparent to the depth sensor")
    hu = float(EL.haugh_unit(h_alb, mass_g)) if np.isfinite(h_alb) else float("nan")
    broken = yi < 0.2 or circ < 0.7 or big > 1
    if broken:
        reasons.append(f"yolk broken (index {yi:.2f}, circularity {circ:.2f}, {big} yolk blob(s))")
    if n_sh:
        reasons.append(f"{n_sh} shell fragment(s) in the contents ({a_sh:.0f} mm^2)")
    verdict = "yolk broken" if broken else ("fish out shards" if n_sh else "ok")
    return BowlReport(n_sh, a_sh, bool(broken), float(yi), h_y, d_y, h_alb, hu, verdict, reasons,
                      dict(contents=contents, yolk=yolk, shards=shard_mask, ring=ring))


def observation(rep, t_h=0.0):
    """The 'opened' belief observation from a bowl report (the yolk index only when the yolk is whole)."""
    from .egg_belief import Observation
    v = {}
    if np.isfinite(rep.albumen_height_mm):
        v["albumen_height_mm"] = rep.albumen_height_mm
    if not rep.yolk_broken:
        v["yolk_index"] = rep.yolk_index
    return Observation("opened", v, t_h, "bowl monitor (RGB-D)")


def overlay(rgb, rep, alpha=0.5):
    out = rgb.astype(np.float32).copy()
    for k, col in (("ring", (60, 140, 255)), ("yolk", (255, 200, 0)), ("shards", (255, 0, 60))):
        m = rep.masks.get(k)
        if m is not None:
            out[m] = (1 - alpha) * out[m] + alpha * np.array(col, np.float32)
    return out.clip(0, 255).astype(np.uint8)
