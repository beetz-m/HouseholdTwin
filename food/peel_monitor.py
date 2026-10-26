"""Vision monitor for peeling an avocado half: which parts of the visible outer surface are correctly peeled, where
skin remains, where flesh is missing, and whether the peel succeeded - judged against the twin's expectation.

Inputs per view: RGB, depth (distance to the image plane, m), camera intrinsics and pose, and the twin's expected
surface (the ideal peel of this instance, as a depth image from the same camera at the twin's *believed* pose). The
monitor registers the expectation to the observation itself (trimmed ICP, x/y/yaw on the worktop), so pose beliefs
need not be exact. Per pixel of the half:
  peeled_ok       flesh colour and the surface where the twin expects it
  skin_remaining  skin colour / texture (dark, low chroma, pebbled), or standing proud of the expected surface
  flesh_missing   surface deeper than expected (flesh torn out with the skin); in RGB-only mode only the colour cue
                  (inner, paler flesh exposed where the dark-green outer band is expected) is available
  unjudged        steep rim and the board contact band, where depth and registration are unreliable
Measures, thresholds and score weights are ILLUSTRATIVE design values, to be calibrated on real peels.
"""

import math
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from scipy.spatial import cKDTree

UNJUDGED, PEELED_OK, SKIN, MISSING = 0, 1, 2, 3
CLASS_NAMES = {UNJUDGED: "unjudged", PEELED_OK: "peeled_ok", SKIN: "skin_remaining", MISSING: "flesh_missing"}
CLASS_RGB = {UNJUDGED: (90, 90, 90), PEELED_OK: (40, 190, 80), SKIN: (220, 40, 40), MISSING: (255, 160, 0)}


@dataclass
class Criteria:
    """Success criteria (ILLUSTRATIVE, to calibrate)."""
    max_skin_fraction: float = 0.01          # of the judged surface
    max_skin_patch_mm2: float = 50.0
    max_lost_fraction: float = 0.01          # of the half's flesh volume
    max_gouge_depth_mm: float = 3.0
    min_coverage: float = 0.45               # judged / whole outer surface of the half (one view)
    deficit_missing_mm: float = 1.2          # deeper than expected by this -> flesh missing
    proud_skin_mm: float = 0.9               # higher than expected by this (and dark) -> skin
    lateral_registration_mm: float = 0.6     # assumed lateral registration uncertainty after refinement
    deficit_low_mm: float = 0.4              # hysteresis: torn area around a deep seed


@dataclass
class PeelReport:
    judged_area_mm2: float
    coverage: float
    peeled_ok_area_mm2: float
    skin_area_mm2: float
    skin_fraction: float
    largest_skin_patch_mm2: float
    flesh_missing_area_mm2: float
    lost_volume_ml: float
    lost_fraction: float
    deepest_gouge_mm: float
    n_skin_patches: int
    n_gouges: int
    registration: dict
    score: float
    success: bool
    verdict: str
    reasons: list = field(default_factory=list)


# %% geometry from depth
def camera(intr):
    W, H = intr["width"], intr["height"]
    fx = intr["focal_mm"] / intr["h_aperture_mm"] * W
    return fx, fx, W / 2.0, H / 2.0, np.asarray(intr["cam_to_world"], float)


def back_project(depth, intr):
    """World points (H, W, 3) of a depth image (distance to the image plane; USD camera: looks along -z)."""
    fx, fy, cx, cy, M = camera(intr)
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    z = depth.astype(np.float64)
    Pc = np.stack([(u - cx) / fx * z, -(v - cy) / fy * z, -z, np.ones_like(z)], -1)
    return (Pc @ M.T)[..., :3]


def board_height(Pw, valid):
    z = Pw[..., 2][valid]
    hist, edges = np.histogram(z, bins=400)
    k = int(np.argmax(hist))
    return float((edges[k] + edges[k + 1]) / 2)


def object_mask(Pw, valid, z_board, min_height=1.5e-3):
    m = (valid & (Pw[..., 2] > z_board + min_height)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, cc, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.zeros_like(m, bool)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return cc == k


def pixel_areas(Pw, depth, intr):
    """Surface area (m^2) seen by each pixel and its horizontal footprint (m^2)."""
    fx = camera(intr)[0]
    a_img = (depth / fx) ** 2
    du = np.gradient(Pw, axis=1)
    dv = np.gradient(Pw, axis=0)
    n = np.cross(du, dv)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    M = camera(intr)[4]
    eye = M[:3, 3]
    ray = Pw - eye
    ray /= np.maximum(np.linalg.norm(ray, axis=-1, keepdims=True), 1e-12)
    cos_n = np.abs(np.einsum("hwc,hwc->hw", n, ray))
    cos_z = np.abs(ray[..., 2])
    return a_img / np.clip(cos_n, 0.25, 1.0), a_img / np.clip(cos_z, 0.25, 1.0), n


# %% registration of the expectation
def register(obs_pts, exp_pts, iters=40, trim=0.8):
    """Rigid x/y/yaw transform that moves the expected surface onto the observed one (trimmed point-to-point ICP;
    the trimming keeps defects - patches, gouges - from pulling the fit)."""
    R2, t2 = np.eye(2), np.zeros(2)
    tree = cKDTree(obs_pts)
    cur = exp_pts.copy()
    for _ in range(iters):
        d, j = tree.query(cur)
        keep = d <= np.quantile(d, trim)
        A, B = cur[keep][:, :2], obs_pts[j[keep]][:, :2]
        ca, cb = A.mean(0), B.mean(0)
        U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
        Ri = (U @ Vt).T
        if np.linalg.det(Ri) < 0:
            Vt[-1] *= -1
            Ri = (U @ Vt).T
        ti = cb - Ri @ ca
        cur[:, :2] = cur[:, :2] @ Ri.T + ti
        R2, t2 = Ri @ R2, Ri @ t2 + ti
        if np.linalg.norm(ti) < 1e-6 and abs(math.atan2(Ri[1, 0], Ri[0, 0])) < 1e-5:
            break
    return R2, t2, float(np.sqrt(np.mean(d[keep] ** 2)))


def refine_heights(obs_pts, exp_pts, R2, t2, trim=0.7):
    """Refine the x/y/yaw registration by minimising the trimmed mean absolute height residual between the observed
    surface and the expected one (the quantity the monitor judges), starting from the ICP result."""
    from scipy.optimize import minimize
    rs = np.random.default_rng(1)
    q = obs_pts[rs.choice(len(obs_pts), min(4000, len(obs_pts)), replace=False)]
    e = exp_pts.copy()
    e[:, :2] = e[:, :2] @ R2.T + t2
    c = e[:, :2].mean(0)

    def apply(x):
        th = math.radians(x[2])
        Rz = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        ee = e.copy()
        ee[:, :2] = (ee[:, :2] - c) @ Rz.T + c + x[:2] * 1e-3
        return ee, Rz

    def cost(x):
        ee, _ = apply(x)
        r = np.abs(expected_height(ee, q[:, :2]) - q[:, 2])
        r = r[np.isfinite(r)]
        return float(np.mean(np.sort(r)[:int(trim * len(r))])) if len(r) else 1.0
    res = minimize(cost, np.zeros(3), method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-7, maxiter=300))
    _, Rz = apply(res.x)
    R_new = Rz @ R2
    t_new = Rz @ (t2 - c) + c + res.x[:2] * 1e-3
    return R_new, t_new, res.fun


def expected_height(exp_pts_reg, query_xy, cell=0.6e-3):
    """Expected surface height at query positions: max height of the registered expectation per grid cell."""
    lo = np.minimum(exp_pts_reg[:, :2].min(0), query_xy.min(0)) - 2 * cell
    ij = np.floor((exp_pts_reg[:, :2] - lo) / cell).astype(int)
    shape = ij.max(0) + 3
    grid = np.full(shape, np.nan)
    np.fmax.at(grid, (ij[:, 0], ij[:, 1]), exp_pts_reg[:, 2])
    q = np.floor((query_xy - lo) / cell).astype(int)
    q = np.clip(q, 0, shape - 1)
    h = grid[q[:, 0], q[:, 1]]
    hole = np.isnan(h)                                         # fill from the 3x3 neighbourhood
    if hole.any():
        acc = np.full(hole.sum(), np.nan)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                qq = np.clip(q[hole] + [dx, dy], 0, shape - 1)
                acc = np.fmax(acc, grid[qq[:, 0], qq[:, 1]])
        h[hole] = acc
    return h


# %% colour / texture
def colour_classes(rgb, mask, tex_skin=0.6):
    """Skin-like and inner-flesh-like pixels. Skin is pebbled (high local luminance variance; peeled flesh is
    smooth) or much darker than the flesh; inner flesh is paler / yellower than the outer band. The texture
    threshold is calibrated on the twin's renders and has to be recalibrated on real images."""
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    h = np.degrees(np.arctan2(b, a)) % 360
    mean = cv2.blur(L, (5, 5))
    tex = np.sqrt(np.maximum(cv2.blur(L * L, (5, 5)) - mean ** 2, 0))
    core = cv2.erode(mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)   # off the outline
    flesh_L0 = float(np.median(L[core])) if core.any() else 60.0
    raw = core & ((tex > tex_skin) | (L < flesh_L0 - 30))
    skin = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(bool) & core
    flesh_L = np.median(L[mask & ~skin]) if (mask & ~skin).any() else 60.0
    inner = mask & ~skin & (L > flesh_L + 9) & (h > 70) & (h < 115)   # paler / yellower than the outer band
    return skin, inner, dict(L=L, C=C, h=h, tex=tex, flesh_L=float(flesh_L))


# %% the monitor
def judge(rgb, depth, intr, depth_expected, flesh_volume_ml, outer_surface_mm2, criteria=None, rgb_only=False):
    """Classify the visible surface of a peeled half and judge the peel. Returns (class map, report, extras)."""
    cr = criteria or Criteria()
    valid = np.isfinite(depth) & (depth > 0) & (depth < 10)
    Pw = back_project(np.where(valid, depth, 1.0), intr)
    zb = board_height(Pw, valid)
    obj = object_mask(Pw, valid, zb)
    valid_e = np.isfinite(depth_expected) & (depth_expected > 0) & (depth_expected < 10)
    Pe = back_project(np.where(valid_e, depth_expected, 1.0), intr)
    obj_e = object_mask(Pe, valid_e, zb)
    area, area_xy, nrm = pixel_areas(Pw, depth, intr)
    obs_pts = Pw[obj]
    exp_pts = Pe[obj_e]
    rs = np.random.default_rng(0)
    sub_o = obs_pts[rs.choice(len(obs_pts), min(8000, len(obs_pts)), replace=False)]
    sub_e = exp_pts[rs.choice(len(exp_pts), min(8000, len(exp_pts)), replace=False)]
    R2, t2, rms = register(sub_o, sub_e)
    R2, t2, h_res = refine_heights(obs_pts, exp_pts, R2, t2)
    exp_reg = exp_pts.copy()
    exp_reg[:, :2] = exp_reg[:, :2] @ R2.T + t2
    h_exp = expected_height(exp_reg, obs_pts[:, :2])
    deficit = np.full(depth.shape, np.nan)
    deficit[obj] = (h_exp - obs_pts[:, 2]) * 1e3              # mm; > 0: lower than expected (flesh missing)
    # judged region: not the steep rim, not the board-contact band, expectation available
    steep = np.abs(nrm[..., 2]) < 0.45
    low = Pw[..., 2] < zb + 4e-3
    rim = cv2.dilate((obj & ~cv2.erode(obj.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)).astype(np.uint8),
                     np.ones((5, 5), np.uint8)).astype(bool)
    judged_colour = obj & ~low & ~rim                          # skin is recognised by appearance: whole visible half
    judged = obj & ~steep & ~low & ~rim & np.isfinite(deficit) # missing flesh needs reliable depth: flatter part
    judged &= cv2.erode(judged.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    # height uncertainty grows with the slope: a lateral registration error e gives e * tan(slope) of height error
    nz = np.clip(np.abs(nrm[..., 2]), 0.2, 1.0)
    tan_slope = np.sqrt(1 - nz ** 2) / nz
    missing_thr = cr.deficit_missing_mm + 2.0 * cr.lateral_registration_mm * tan_slope
    skin_c, inner_c, feats = colour_classes(rgb, obj)
    cls = np.zeros(depth.shape, np.uint8)
    cls[judged_colour] = PEELED_OK
    if rgb_only:
        missing = judged & inner_c & ~skin_c
        skin = judged_colour & skin_c
    else:
        missing = judged & ~skin_c & (deficit > missing_thr)
        proud = judged & (deficit < -cr.proud_skin_mm) & (feats["tex"] > 0.3)
        skin = judged_colour & (skin_c | proud)
    clean = lambda m, J: cv2.morphologyEx(cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN,  # noqa: E731
                                                           np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE,
                                          np.ones((5, 5), np.uint8)).astype(bool) & J
    skin = clean(skin, judged_colour)
    # skin evidence must form compact, solid blobs (stripes along shading edges and rings round gouge rims do not);
    # where depth is reliable, a skin blob must also stand proud of the expected surface (skin is ~1.7 mm thick)
    n_c, cc_c, st_c, _ = cv2.connectedComponentsWithStats(skin.astype(np.uint8))
    dist = cv2.distanceTransform(skin.astype(np.uint8), cv2.DIST_L2, 3)
    keep = np.zeros(n_c, bool)
    for i in range(1, n_c):
        comp = cc_c == i
        cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.contourArea(cv2.convexHull(max(cnts, key=cv2.contourArea)))
        solid = comp.sum() / max(hull, 1.0)
        thick = dist[comp].max() >= 2.5
        on_depth = comp & judged
        proud_ok = True
        if not rgb_only and on_depth.sum() > 0.3 * comp.sum():
            proud_ok = np.nanmedian(deficit[on_depth]) < -0.3 or np.median(feats["L"][comp]) < feats["flesh_L"] - 30
        keep[i] = solid >= 0.6 and thick and proud_ok
    skin = keep[cc_c] & skin
    # the 5x5 texture window and the closing widen a patch by ~2-3 px: shrink back to its extent
    skin = cv2.erode(skin.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if not rgb_only:                                           # hysteresis: grow the deep seeds over the torn area
        low = judged & ~skin & (deficit > cr.deficit_low_mm)
        n_l, cc_l = cv2.connectedComponents(low.astype(np.uint8))
        seeds = np.unique(cc_l[clean(missing, judged)])
        missing = np.isin(cc_l, seeds[seeds > 0])
    missing = clean(missing, judged) & ~skin
    cls[missing] = MISSING
    cls[skin] = SKIN
    A = area * 1e6                                              # mm^2 per pixel
    comps = lambda m: cv2.connectedComponentsWithStats(m.astype(np.uint8))  # noqa: E731
    n_s, cc_s, _, _ = comps(cls == SKIN)
    patch_areas = [float(A[cc_s == i].sum()) for i in range(1, n_s)]
    patch_areas = [p for p in patch_areas if p >= 4.0]
    n_m, cc_m, _, _ = comps(cls == MISSING)
    gouge_areas = [float(A[cc_m == i].sum()) for i in range(1, n_m)]
    gouges = [i for i, g in zip(range(1, n_m), gouge_areas) if g >= 4.0]
    judged_area = float(A[judged_colour].sum())
    depth_judged_area = float(A[judged].sum())
    skin_area = float(A[cls == SKIN].sum())
    miss_area = float(A[cls == MISSING].sum())
    lost = float(np.nansum(np.where(cls == MISSING, np.clip(deficit, 0, None) * 1e-3 * area_xy, 0.0)) * 1e6)
    depth_n = deficit * np.abs(nrm[..., 2])                     # vertical deficit -> depth along the surface normal
    deepest = float(np.nanmax(np.where(cls == MISSING, depth_n, np.nan))) if miss_area > 0 else 0.0
    coverage = judged_area / max(outer_surface_mm2, 1e-9)
    lost_frac = lost / max(flesh_volume_ml, 1e-9)
    skin_frac = skin_area / max(judged_area, 1e-9)
    reasons = []
    if coverage < cr.min_coverage:
        reasons.append(f"coverage {coverage:.0%} below {cr.min_coverage:.0%}: turn the half and look again")
    if skin_frac > cr.max_skin_fraction:
        reasons.append(f"skin remaining on {skin_frac:.1%} of the judged surface")
    if patch_areas and max(patch_areas) > cr.max_skin_patch_mm2:
        reasons.append(f"skin patch of {max(patch_areas):.0f} mm2")
    if lost_frac > cr.max_lost_fraction:
        reasons.append(f"flesh lost {lost:.2f} mL ({lost_frac:.1%})")
    if deepest > cr.max_gouge_depth_mm:
        reasons.append(f"gouge {deepest:.1f} mm deep")
    # score: 100 minus penalties relative to the limits (ILLUSTRATIVE weights)
    score = 100.0 - 25.0 * skin_frac / cr.max_skin_fraction - 25.0 * lost_frac / cr.max_lost_fraction \
        - 10.0 * deepest / cr.max_gouge_depth_mm - 10.0 * (max(patch_areas) if patch_areas else 0.0) / cr.max_skin_patch_mm2
    score = float(np.clip(score, 0.0, 100.0))
    judgeable = coverage >= cr.min_coverage
    success = judgeable and not reasons
    verdict = ("not judged" if not judgeable else "success" if success else "rework")
    rep = PeelReport(judged_area, coverage, float(A[cls == PEELED_OK].sum()), skin_area, skin_frac,
                     max(patch_areas) if patch_areas else 0.0, miss_area, lost, lost_frac, deepest,
                     len(patch_areas), len(gouges),
                     dict(yaw_deg=math.degrees(math.atan2(R2[1, 0], R2[0, 0])), t_mm=(t2 * 1e3).tolist(),
                          rms_mm=rms * 1e3, height_residual_mm=h_res * 1e3), score, success, verdict, reasons)
    rep.registration["depth_coverage"] = depth_judged_area / max(outer_surface_mm2, 1e-9)
    return cls, rep, dict(deficit=deficit, object=obj, judged=judged_colour, judged_depth=judged, board_z=zb)


def sdt_update(rep):
    """Belief-layer attributes for the peeled half (root prim) from a monitor report."""
    return {"": {"sdt:belief:peel_verdict": rep.verdict, "sdt:belief:peel_score": rep.score,
                 "sdt:belief:peel_skin_remaining_mm2": rep.skin_area_mm2,
                 "sdt:belief:peel_flesh_missing_mm2": rep.flesh_missing_area_mm2,
                 "sdt:belief:peel_lost_volume_ml": rep.lost_volume_ml,
                 "sdt:belief:peel_coverage": rep.coverage,
                 "sdt:belief:peel_reasons": "; ".join(rep.reasons) or "-",
                 "sdt:epistemicStatus": "camera observation (peel monitor) against the twin's expectation"}}


def overlay(rgb, cls, alpha=0.45):
    out = rgb.astype(np.float32).copy()
    for k, col in CLASS_RGB.items():
        if k == UNJUDGED:
            continue
        m = cls == k
        out[m] = (1 - alpha) * out[m] + alpha * np.array(col, np.float32)
    return out.clip(0, 255).astype(np.uint8)


def report_dict(rep):
    return asdict(rep)
