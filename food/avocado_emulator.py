"""Emulator for halving and pitting an avocado: actions in, new USD states + measurements + events out.

State machine (food/schema.py AVOCADO): whole --cut_round_seed--> scored (kerf to the seed) --twist_apart-->
halves (half_a holds the seed; torn flesh where the twist tore) --remove_seed--> half_a_pitted + seed (seed-coat
remnants left in the cavity when the testa adheres to the flesh; spoon gouges). The mechanics are in
food/avocado_process.py; the hidden truth is the instance's register draw (instance.json / truth.usda), the robot
sees only `measured`. Every state is written as a USD file of the part graph (geometry edited by manifold3d
booleans; cut faces and torn flesh are separate, semantically labelled prims), so it can be rendered, simulated and
checked like the original asset.

    emu = AvocadoEmulator("assets/food/avocado_4", "assets/food/avocado_4/process/run0", seed=0)
    out = emu.cut_round_seed(coverage_deg=360); out = emu.twist_apart(); out = emu.remove_seed("spoon")
"""

import json
import math
import os
from dataclasses import asdict, dataclass, field

import manifold3d as m3
import numpy as np
from pxr import Sdf, Usd, UsdGeom

from . import avocado_peel as AP
from . import avocado_process as PRC
from .avocado import Avocado


@dataclass
class EmulatorState:
    integrity: str = "whole"                  # whole / scored / partly_scored / halved / halved_torn
    seed: str = "in"                          # in / out
    pieces: dict = field(default_factory=dict)  # piece name -> USD file
    time_s: float = 0.0
    log: list = field(default_factory=list)


class PreconditionError(Exception):
    pass


class AvocadoEmulator:
    def __init__(self, instance_dir, out_dir, seed=0, stage=None):
        self.src = os.path.abspath(instance_dir)
        self.out = os.path.abspath(out_dir)
        os.makedirs(self.out, exist_ok=True)
        rec = json.load(open(os.path.join(self.src, "instance.json")))
        self.rec = rec
        hp = {k: v for k, v in rec["hidden_params"].items() if not k.startswith("_")}
        self.avo = Avocado(rec.get("seed", 0), overrides=hp)
        self.stage = float(stage if stage is not None else hp["ripeness_stage"])
        az = rec["cuts"]["half_a"]["plane_azimuth_deg"]       # the asset's halves were cut in this plane
        self.plane = PRC.CutPlane(az, 360.0)
        self.rng = np.random.default_rng(seed)
        self.state = EmulatorState(pieces={"avocado": os.path.join(self.src, "whole.usdc")})

    # %% helpers
    def _copy(self, src_usd, name):
        path = os.path.join(self.out, f"{name}.usdc")              # binary: the part graphs are large
        if os.path.exists(path):
            os.remove(path)
        layer = Sdf.Layer.CreateNew(path)
        layer.TransferContent(Sdf.Layer.FindOrOpen(src_usd))
        st = Usd.Stage.Open(layer)
        AP.absolutize_textures(st, self.src)
        return st, layer, path

    def _mark(self, st, **kv):
        root = st.GetPrimAtPath("/Food")
        for k, v in kv.items():
            if isinstance(v, str):
                root.CreateAttribute(f"food:{k}", Sdf.ValueTypeNames.String, custom=True).Set(v)
            else:
                root.CreateAttribute(f"truth:{k}", Sdf.ValueTypeNames.Double, custom=True).Set(float(v))

    def _seed_manifold(self, grow):
        a = self.avo
        e = m3.Manifold.sphere(1.0, 64).scale((a.seed_W / 2 + grow, a.seed_W / 2 + grow, a.seed_L / 2 + grow))
        return e.translate(tuple(map(float, a.seed_c)))

    def _cavity_point(self, psi_deg, lat_deg, side_sign):
        """Point on the seed cavity wall (seed-coat surface) of the half on ``side_sign`` of the cut plane."""
        e1, ez, n = self.plane.basis()
        psi, lat = math.radians(psi_deg), math.radians(abs(lat_deg))
        d = math.cos(lat) * (math.cos(psi) * e1 + math.sin(psi) * ez) + side_sign * math.sin(lat) * n
        g = self.avo.t_testa_o + self.avo.t_testa_i + self.avo.t_endo
        lo, hi = 0.0, 0.06
        for _ in range(40):
            mid = (lo + hi) / 2
            if self.avo.seed_e(self.avo.seed_c + d * mid, g) < 1:
                lo = mid
            else:
                hi = mid
        return self.avo.seed_c + d * lo, d

    def _record(self, out):
        self.state.log.append(dict(out.as_dict(), integrity=self.state.integrity, seed=self.state.seed,
                                   pieces=dict(self.state.pieces), stage=self.stage))
        json.dump(dict(instance=self.src, stage=self.stage, plane=asdict(self.plane), log=self.state.log),
                  open(os.path.join(self.out, "process.json"), "w"), indent=1)
        return out

    # %% actions
    def cut_round_seed(self, coverage_deg=360.0, F_max=40.0):
        if self.state.integrity not in ("whole", "partly_scored"):   # a partial cut may be completed
            raise PreconditionError(f"cut_round_seed needs a whole or partly scored avocado (is "
                                    f"{self.state.integrity})")
        self.plane.coverage_deg = float(coverage_deg)
        out = PRC.cut_round_seed(self.avo, self.plane, self.rng, self.stage, F_max)
        if out.outcome == "no_entry":
            return self._record(out)
        st, layer, path = self._copy(os.path.join(self.src, "whole.usdc"), "scored")
        # kerf: blade-thick slab in the cut plane, over the covered sector, down to the seed coat
        e1, ez, n = self.plane.basis()
        t = 1.6e-3
        R = np.c_[e1, ez, n]
        M = np.eye(4)
        M[:3, :3] = R
        sector = out.truth["coverage_deg"]
        c = self.avo.seed_c - n * float(self.avo.seed_c @ n)
        if sector >= 359.0:
            slab = m3.Manifold.cube((0.3, 0.3, t), True)
        else:                                                   # wedge from the stem direction round by `sector`
            a0 = math.radians(90.0)
            pts = [(0.0, 0.0)] + [(0.15 * math.cos(a0 + a), 0.15 * math.sin(a0 + a))
                                  for a in np.linspace(0, math.radians(sector), 64)]
            slab = m3.Manifold.extrude(m3.CrossSection([pts]), t).translate((0, 0, -t / 2))
        Mc = M.copy()
        Mc[:3, 3] = c
        g = self.avo.t_testa_o + self.avo.t_testa_i + self.avo.t_endo
        kerf = slab.transform(Mc[:3, :]) - self._seed_manifold(g + 0.2e-3)
        res = AP.carve(st, kerf, AP.flesh_paths(st, with_skin=True), label="avocado_cut_surface",
                       sdt_class="food:CutSurface", surface_name="KerfSurface")
        self._mark(st, state="scored", kerf_volume_ml=res["lost_ml"], cut_coverage_deg=sector)
        layer.Save()
        self.state.integrity = out.outcome
        self.state.pieces = {"avocado": path}
        out.truth["kerf_volume_ml"] = res["lost_ml"]
        return self._record(out)

    def twist_apart(self, T_max=1.5):
        if self.state.integrity not in ("scored", "partly_scored"):
            raise PreconditionError(f"twist_apart needs a scored avocado (is {self.state.integrity})")
        out = PRC.twist_apart(self.avo, self.plane, self.rng, self.stage, self.plane.coverage_deg, T_max)
        if not out.ok:
            return self._record(out)
        pieces = {}
        for half, sign in (("half_a", +1), ("half_b", -1)):
            st, layer, path = self._copy(os.path.join(self.src, f"{half}.usdc"), half)
            lost = 0.0
            if half == "half_b" and (out.truth.get("torn_chunks") or out.truth.get("torn_bridge_deg")):
                cutter = None
                for ch in out.truth.get("torn_chunks", []):      # flesh stuck to the seed: holes in the cavity wall
                    p, d = self._cavity_point(ch["psi_deg"], ch["lat_deg"], sign)
                    e = AP.ellipsoid(AP.Defect(p.tolist(), (-d).tolist(), [r * 1e-3 for r in ch["radii_mm"]],
                                               "gouge"))
                    cutter = e if cutter is None else cutter + e
                if out.truth.get("torn_bridge_deg"):            # torn bridge: rough strip along the uncut sector
                    a0 = 90.0 + self.plane.coverage_deg
                    for a in np.linspace(a0, 450.0, max(2, int(out.truth["torn_bridge_deg"] / 15))):
                        p, d = self._cavity_point(a, 0.0, sign)
                        e = AP.ellipsoid(AP.Defect((p + d * 0.008).tolist(), (-d).tolist(), [5e-3, 5e-3, 2e-3],
                                                   "gouge"))
                        cutter = e if cutter is None else cutter + e
                if cutter is not None:
                    lost = AP.carve(st, cutter, AP.flesh_paths(st))["lost_ml"]
            self._mark(st, state=half, integrity=out.outcome, torn_flesh_ml=lost)
            layer.Save()
            pieces[half] = path
            out.truth[f"{half}_torn_flesh_ml"] = lost
        self.state.integrity = out.outcome
        self.state.pieces = pieces
        return self._record(out)

    def remove_seed(self, method="spoon", F_max=25.0, T_max=1.0):
        if self.state.integrity not in ("halved", "halved_torn") or self.state.seed != "in":
            raise PreconditionError("remove_seed needs a halved avocado with the seed in (half_a)")
        out = PRC.remove_seed(self.avo, self.rng, self.stage, method, F_max, T_max)
        if not out.ok:
            return self._record(out)
        src_half = self.state.pieces["half_a"]
        st, layer, path = self._copy(src_half, "half_a_pitted")
        seed_prim = st.GetPrimAtPath("/Food/Parts/Seed")
        frac = out.truth["testa_remnant_fraction"]
        kept_mm2 = 0.0
        _, _, n = self.plane.basis()                              # half_a lies on the +n side of the cut plane
        endo = st.GetPrimAtPath("/Food/Parts/Endocarp/Geom")
        Pe, Fe = AP.mesh_arrays(endo)
        ce = sorted(AP.cut_faces(endo))
        d0 = float((Pe[Fe[ce]].reshape(-1, 3) @ n).mean()) if ce else float(self.avo.seed_c @ n)
        for p in list(Usd.PrimRange(seed_prim)):
            if not p.IsA(UsdGeom.Mesh):
                continue
            if "OuterTesta" in str(p.GetPath()) and frac > 0.05:   # seed coat patches stay in the cavity
                P, F = AP.mesh_arrays(p)
                cen = P[F].mean(1)
                in_cavity = cen @ n - d0 > 0.3e-3                  # only the coat that lined this half's cavity
                if in_cavity.any():
                    # irregular patches: a few torn-off sheets grown round random seeds on the cavity wall until
                    # they cover `frac` of it; ragged edges from a smooth random perturbation of the distance
                    idx = np.nonzero(in_cavity)[0]
                    k = int(self.rng.integers(3, 7))
                    seeds = cen[self.rng.choice(idx, k, replace=False)]
                    dist = np.linalg.norm(cen[idx, None, :] - seeds[None], axis=2).min(1)
                    w = self.rng.normal(size=(3, 3))
                    rough = sum(np.sin(cen[idx] @ w[j] * 900 + self.rng.uniform(0, 6)) for j in range(3))
                    score = dist * (1 + 0.12 * rough)
                    area = AP.tri_area(P, F[idx])
                    order = np.argsort(score)
                    cut_at = order[np.searchsorted(np.cumsum(area[order]) / area.sum(), frac)
                                   if frac < 1 else len(order) - 1]
                    keep = np.zeros(len(F), bool)
                    keep[idx] = score <= score[cut_at]
                    if keep.any():
                        AP.subset_faces(p, keep)
                        kept_mm2 += float(AP.tri_area(P, F[keep]).sum() * 1e6) / 2   # two faces of a thin coat
                        continue
            p.SetActive(False)
        for name in ("Cotyledon_1", "Cotyledon_2", "EmbryonicAxis"):
            q = st.GetPrimAtPath(f"/Food/Parts/Seed/{name}")
            if q:
                q.SetActive(False)
        lost = 0.0
        if out.truth.get("gouges"):
            cutter = None
            for gq in out.truth["gouges"]:
                p, d = self._cavity_point(gq["psi_deg"], gq["lat_deg"], +1)
                e = AP.ellipsoid(AP.Defect(p.tolist(), (-d).tolist(), [r * 1e-3 for r in gq["radii_mm"]], "gouge"))
                cutter = e if cutter is None else cutter + e
            lost = AP.carve(st, cutter, AP.flesh_paths(st))["lost_ml"]
        self._mark(st, state="half_a_pitted", seed="out", testa_left_mm2=kept_mm2, gouged_flesh_ml=lost)
        layer.Save()
        st2, layer2, path2 = self._copy(src_half, "seed")             # the seed as its own piece
        for c in st2.GetPrimAtPath("/Food/Parts").GetChildren():
            if c.GetName() != "Seed":
                c.SetActive(False)
        if frac > 0.95:
            q = st2.GetPrimAtPath("/Food/Parts/Seed/SeedCoat/OuterTesta")
            if q:
                q.SetActive(False)
        self._mark(st2, state="seed")
        layer2.Save()
        out.truth.update(testa_left_mm2=kept_mm2, gouged_flesh_ml=lost)
        self.state.seed = "out"
        self.state.pieces.update(half_a=path, seed=path2)
        return self._record(out)
