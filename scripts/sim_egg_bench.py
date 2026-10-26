"""Egg manipulation bench in Isaac Sim (PhysX 240 Hz): grasp-force sweep with a force-limited parallel gripper, and
drop tests onto kitchen surfaces.  PhysX supplies the kinematics and contact forces; the egg's hidden truth model
(food/dynamics.EggTruth) turns them into shell outcomes; the robot's belief (food/sdt.EggBelief) predicts them
beforehand -- the AICOR H8 admissibility test with numbers.

Experiments (lanes run in parallel in one simulation):
  G  grip sweep: one egg instance cloned into 9 lanes, F_cmd = 0.5 .. 45 N; close on the equator, lift with
     2 m/s^2, hold; measured pad force, slip, shell state; belief P(slip), P(rupture), H8 verdict per lane
  V  instance variation: 8 egg instances at F_cmd = 30 N (same check)
  D  drop: 5 heights x 4 surfaces (granite, oak board, silicone mat, towel), random orientation; first-impact normal
     speed and contact location from PhysX -> crack outcome
Writes test_reports/egg_bench.json and renders/food/egg_bench.png.

Run: OMNI_KIT_ACCEPT_EULA=YES ~/Documents/LabAutomationTwin/env_isaacsim/bin/python scripts/sim_egg_bench.py
"""

import json
import math
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HZ = 240
DT = 1.0 / HZ

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.experimental.prims import RigidPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from omni.physx import get_physx_simulation_interface  # noqa: E402
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

from food.dynamics import SURFACES, EggTruth  # noqa: E402
from food.egg import Egg  # noqa: E402
from food.sdt import EggBelief  # noqa: E402

A = os.path.join(ROOT, "assets", "food")
EGG_SEED = 1
F_SWEEP = [0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 25.0, 35.0, 45.0]
VAR_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
F_VAR = 30.0
HEIGHTS = [0.01, 0.02, 0.04, 0.08, 0.15]
SURF = ["granite", "oak_board", "silicone_mat", "towel"]

omni.usd.get_context().new_stage()
app.update()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
scene = UsdPhysics.Scene.Define(stage, "/World/Physics")
scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr(9.81)
PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim()).CreateTimeStepsPerSecondAttr(HZ)


def phys_mat(path, mu, e):
    m = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
    api.CreateStaticFrictionAttr(mu)
    api.CreateDynamicFrictionAttr(mu * 0.85)
    api.CreateRestitutionAttr(e)
    return m


def static_box(path, size, pos, mat):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    x = UsdGeom.Xformable(c)
    x.AddTranslateOp().Set(Gf.Vec3d(*pos))
    x.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    return c


def body_box(path, size, pos, mass, mat=None, kinematic=False):
    """Rigid body = unscaled Xform (joint frames in metres) with a scaled cube collider child."""
    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp().Set(Gf.Vec3d(*pos))
    c = UsdGeom.Cube.Define(stage, path + "/Shape")
    c.CreateSizeAttr(1.0)
    UsdGeom.Xformable(c).AddScaleOp().Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    if mat is not None:
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
    if kinematic:
        UsdPhysics.RigidBodyAPI(x.GetPrim()).CreateKinematicEnabledAttr(True)
    else:
        UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(mass)
    PhysxSchema.PhysxRigidBodyAPI.Apply(x.GetPrim()).CreateSleepThresholdAttr(0.0)
    return x


counter = phys_mat("/World/Mat/Granite", 0.6, SURFACES["granite"][1])
surf_mats = {s: phys_mat(f"/World/Mat/{s}", 0.6, SURFACES[s][1]) for s in SURF}


def egg_at(path, seed, pos, quat=(1, 0, 0, 0)):
    x = UsdGeom.Xform.Define(stage, path)
    x.GetPrim().GetReferences().AddReference(os.path.join(A, f"egg_{seed}", "whole.usdc"))
    x.AddTranslateOp().Set(Gf.Vec3d(*pos))
    x.AddOrientOp().Set(Gf.Quatf(*quat))
    PhysxSchema.PhysxContactReportAPI.Apply(x.GetPrim()).CreateThresholdAttr().Set(0.0)
    PhysxSchema.PhysxRigidBodyAPI.Apply(x.GetPrim()).CreateSleepThresholdAttr(0.0)
    return x


# ---------------------------------------------------------------- grasp lanes
eggs = {s: Egg(s) for s in set(VAR_SEEDS + [EGG_SEED])}
lanes = [("G", EGG_SEED, F) for F in F_SWEEP] + [("V", s, F_VAR) for s in VAR_SEEDS]
LANE_DY = 0.20
ORIENT_LYING = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)     # long axis (z) -> x
pad_mat = phys_mat("/World/Mat/Pad", 0.6, 0.0)
grasp = []
for i, (exp, seed, F) in enumerate(lanes):
    y0 = i * LANE_DY
    e = eggs[seed]
    B = e.p["shape_index_B_over_L"] * e.p["length_L"] * 1e-3
    static_box(f"/World/G/L{i}/Counter", (0.16, 0.16, 0.02), (0, y0, -0.01), counter)
    egg_at(f"/World/G/L{i}/Egg", seed, (0, y0, B / 2 + 0.0005), ORIENT_LYING)
    zc = B / 2 + 0.0005
    palm = body_box(f"/World/G/L{i}/Palm", (0.02, 0.10, 0.02), (0, y0, zc + 0.045), 0.5, kinematic=True)
    fingers = []
    for k, sy in enumerate((-1, 1)):
        f = body_box(f"/World/G/L{i}/Finger{k}", (0.018, 0.008, 0.04), (0, y0 + sy * 0.040, zc + 0.005), 0.03,
                     pad_mat)
        PhysxSchema.PhysxContactReportAPI.Apply(f.GetPrim()).CreateThresholdAttr().Set(0.0)
        j = UsdPhysics.PrismaticJoint.Define(stage, f"/World/G/L{i}/Joint{k}")
        j.CreateAxisAttr("Y")
        j.CreateBody0Rel().SetTargets([palm.GetPath()])
        j.CreateBody1Rel().SetTargets([f.GetPath()])
        j.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, -0.040))
        j.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        j.CreateLowerLimitAttr(-0.05)
        j.CreateUpperLimitAttr(0.05)
        drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
        drv.CreateTypeAttr("force")
        drv.CreateTargetPositionAttr(sy * 0.040)                 # open
        drv.CreateStiffnessAttr(2.0e5)
        drv.CreateDampingAttr(300.0)
        drv.CreateMaxForceAttr(float(F))
        fingers.append((f, j, sy))
    grasp.append({"exp": exp, "seed": seed, "F": F, "y0": y0, "zc": zc, "palm": palm, "fingers": fingers})

# ---------------------------------------------------------------- drop lanes
drops = []
rng = np.random.default_rng(11)
X0 = 1.0
for si, s in enumerate(SURF):
    for hi, h in enumerate(HEIGHTS):
        x0, y0 = X0 + si * 0.25, hi * 0.20
        static_box(f"/World/D/S{si}_{hi}/Surface", (0.2, 0.18, 0.02), (x0, y0, -0.01), surf_mats[s])
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        egg_at(f"/World/D/S{si}_{hi}/Egg", EGG_SEED, (x0, y0, h + 0.03), tuple(q))
        drops.append({"surface": s, "h": h, "path": f"/World/D/S{si}_{hi}/Egg", "x0": x0, "y0": y0,
                      "surf_path": f"/World/D/S{si}_{hi}/Surface"})

# ---------------------------------------------------------------- run
app_utils.play(commit=True)
SimulationManager.step(steps=2)
egg_g = RigidPrim([f"/World/G/L{i}/Egg" for i in range(len(grasp))])
palms = RigidPrim([str(g["palm"].GetPath()) for g in grasp])
egg_d = RigidPrim([d["path"] for d in drops])
# release drop eggs from exact heights: lowest point of the egg at h above the surface
t = [0.0]
contacts = {}                                  # (a, b) -> max normal force this step window
first_impact = {}
v_hist = {}


def collect():
    heads, data = get_physx_simulation_interface().get_contact_report()
    out = []
    for h in heads:
        a = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
        b = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
        imp = np.zeros(3)
        pos = None
        for k in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
            imp += np.array(data[k].impulse)
            pos = np.array(data[k].position)
        out.append((a, b, imp, pos))
    return out


def egg_root(path):
    for g in ("/World/G/", "/World/D/"):
        if path.startswith(g):
            parts = path.split("/")
            return "/".join(parts[:5])
    return path


def step(n=1, on=None):
    for _ in range(n):
        SimulationManager.step(steps=1)
        t[0] += DT
        if on:
            on(collect())


# drop: record velocity each step, detect first contact with the surface
def drop_watch(cs):
    vel = egg_d.get_velocities()[0].numpy()
    for i, d in enumerate(drops):
        v_hist.setdefault(i, []).append(vel[i].copy())
    for a, b, imp, pos in cs:
        for i, d in enumerate(drops):
            if i in first_impact:
                continue
            if (a.startswith(d["path"]) and b == d["surf_path"]) or (b.startswith(d["path"]) and a == d["surf_path"]):
                first_impact[i] = {"t": t[0], "pos": pos, "imp": imp, "v_before": v_hist[i][-2] if len(v_hist[i]) > 1
                                   else v_hist[i][-1]}


# settle grasp eggs, let drops fall (drop eggs start at rest in the air: gravity on from play)
step(int(0.8 * HZ), on=drop_watch)

_pe = egg_g.get_world_poses()[0].numpy()
for i, g in enumerate(grasp):
    print(f"[diag0] lane {i} egg after settle {np.round(_pe[i] - [0, g['y0'], 0], 4)}")
# ---- grasp: close at gripper speed (Franka Hand class: <= 0.1 m/s per finger); force cap holds at contact
CLOSE_V = 0.05
targets = [UsdPhysics.DriveAPI(j.GetPrim(), "linear").GetTargetPositionAttr() for g in grasp for f, j, sy in
           g["fingers"]]
signs = [sy for g in grasp for f, j, sy in g["fingers"]]
pad_force = {i: [] for i in range(len(grasp))}
pad_contact = {i: None for i in range(len(grasp))}


_dbg = [0]


def grasp_watch(cs):
    _dbg[0] += 1
    if _dbg[0] <= 12:
        for a, b, imp, pos in cs:
            if "/L6/" in a or "/L6/" in b:
                print(f"[dbg] step {_dbg[0]} {a} | {b} |imp|={np.linalg.norm(imp):.4f} pos={np.round(pos, 4) if pos is not None else None}")
    for a, b, imp, pos in cs:
        for i, g in enumerate(grasp):
            eg = f"/World/G/L{i}/Egg"
            fg = f"/World/G/L{i}/Finger"
            if (a.startswith(eg) and b.startswith(fg)) or (b.startswith(eg) and a.startswith(fg)):
                pad_force[i].append(np.linalg.norm(imp) / DT)
                pad_contact[i] = pos


for k in range(int(1.2 * HZ)):
    w = max(0.0, 0.040 - CLOSE_V * k * DT)
    for a_, sy in zip(targets, signs):
        a_.Set(sy * w)
    step(1, on=grasp_watch)
_pe = egg_g.get_world_poses()[0].numpy()
for i, g in enumerate(grasp):
    print(f"[diag] lane {i} F={g['F']} egg after close {np.round(_pe[i] - [0, g['y0'], 0], 4)} contacts={len(pad_force[i])}")
F_meas = {i: float(np.median(pad_force[i][-int(0.3 * HZ):])) if pad_force[i] else 0.0 for i in range(len(grasp))}
# ---- lift with 2 m/s^2 to 0.5 m/s, 0.08 m
p0 = palms.get_world_poses()[0].numpy().copy()
eg0 = egg_g.get_world_poses()[0].numpy().copy()
z, v, a_max, v_max, H = 0.0, 0.0, 2.0, 0.5, 0.08
while z < H:
    dist_stop = v * v / (2 * a_max)
    v = max(0.05, v - a_max * DT) if H - z <= dist_stop else min(v_max, v + a_max * DT)
    z = min(H, z + v * DT)
    p = p0.copy()
    p[:, 2] += z
    palms.set_world_poses(positions=p)
    step(1, on=grasp_watch)
step(int(0.5 * HZ), on=grasp_watch)
eg1 = egg_g.get_world_poses()[0].numpy()
lift = eg1[:, 2] - eg0[:, 2]

# ---------------------------------------------------------------- outcomes
res_g = []
for i, g in enumerate(grasp):
    e = eggs[g["seed"]]
    truth = EggTruth(e, 0.0656)
    Fc = F_meas[i]
    slipped = lift[i] < 0.06
    shell = truth.static_load(t[0], Fc, math.pi / 2, np.random.default_rng(i))
    bel = EggBelief(0.0656, np.random.default_rng(100 + i))
    adm, pr = bel.admissible(g["F"], math.pi / 2, a_max=2.0)
    res_g.append({"exp": g["exp"], "egg": g["seed"], "F_cmd": g["F"], "F_pad_measured": round(Fc, 2),
                  "lift_mm": round(float(lift[i]) * 1e3, 1), "slipped": bool(slipped), "shell": shell,
                  "F_crack_truth": round(e.break_force(math.pi / 2), 1),
                  "belief": {k: round(v, 3) for k, v in pr.items()}, "H8_admissible": bool(adm)})

res_d = []
for i, d in enumerate(drops):
    fi = first_impact.get(i)
    if fi is None:
        res_d.append({"surface": d["surface"], "h_cm": d["h"] * 100, "impact": None})
        continue
    p, q = egg_d.get_world_poses()
    # contact location in the egg frame -> polar angle from the blunt pole (+z of the egg asset)
    qq = q.numpy()[i]
    w, x_, y_, z_ = qq
    R = np.array([[1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - w * z_), 2 * (x_ * z_ + w * y_)],
                  [2 * (x_ * y_ + w * z_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - w * x_)],
                  [2 * (x_ * z_ - w * y_), 2 * (y_ * z_ + w * x_), 1 - 2 * (x_ * x_ + y_ * y_)]])
    loc = R.T @ (fi["pos"] - p.numpy()[i])
    polar = math.acos(np.clip(loc[2] / (np.linalg.norm(loc) + 1e-12), -1, 1))
    v_n = float(-fi["v_before"][2])
    truth = EggTruth(eggs[EGG_SEED], 0.0656)
    F = truth.impact(fi["t"], v_n, polar, d["surface"], np.random.default_rng(i))
    res_d.append({"surface": d["surface"], "h_cm": d["h"] * 100, "v_n": round(v_n, 3),
                  "h_eff_cm": round(v_n ** 2 / (2 * 9.81) * 100, 2), "polar_deg": round(math.degrees(polar), 1),
                  "F_peak": round(float(F), 1), "shell": truth.shell,
                  "h_crit_cm_at_this_polar": round(truth.drop_height_to_crack(polar, d["surface"]) * 100, 2)})

os.makedirs(os.path.join(ROOT, "test_reports"), exist_ok=True)
rep = {"egg": EGG_SEED, "grasp": res_g, "drop": res_d,
       "notes": "PhysX: kinematics + contact forces; outcomes: EggTruth (hidden); predictions: EggBelief (register "
                "priors). Drop release height is the egg's start height above its rest height (approx.)."}
json.dump(rep, open(os.path.join(ROOT, "test_reports", "egg_bench.json"), "w"), indent=1)
for r in res_g:
    print("[grasp]", r)
for r in res_d:
    print("[drop]", r)
app.close()
