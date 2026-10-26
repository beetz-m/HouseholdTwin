"""Banana peeling bench in Isaac Sim: the peel as strips of rigid segments held by breakable joints.

The banana (banana_0 geometry) hangs stem-up, its pulp held kinematically (a hand / gripper holding it).  After the
neck is snapped, a finger grips the top of strip 1 and pulls it outwards and down to the tip.  The peel is split along
its ridges into `ridges` strips (peel tears axially in the corners, khanal23), each cut into N_SEG segments along the
finger.  Joints:
  along a strip     revolute hinges (the strip curls back) that break at 1.5 x the peel fracture force (torn strip)
  strip -> pulp     fixed joints, adhesion 0.45 x the peel fracture force
  strip <-> strip   fixed seam joints along the ridges, 0.30 x the peel fracture force
                    (register peel_bond_ratios, ILLUSTRATIVE ratios on khanal23's 53 -> 2.1 N, SOURCED-VERIFY)
The finger is a dynamic body on a stiff D6 spring to a moving target; force = k x lag.  One lane per ripeness stage
(2, 4, 6).  This bench shows the mechanism and the stage dependence (green peel splits hard, ripe peel zips off); its
force level inherits the ILLUSTRATIVE strip force -- there is no published peeling force (GAP).

Writes test_reports/banana_peel.json and renders/food/banana_peel_*.png.  Run (GPU job alone):
  OMNI_KIT_ACCEPT_EULA=YES ~/Documents/LabAutomationTwin/env_isaacsim/bin/python scripts/sim_banana_peel.py
"""

import json
import math
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HZ = int(os.environ.get("PEEL_HZ", 480))
STAGES = [2.0, 4.0, 6.0]
N_SEG = 12
K_FINGER = 4000.0                                     # N/m finger spring
V_PULL = float(os.environ.get("PEEL_V", 0.02))        # m/s
RENDER = "--no-render" not in sys.argv
RES = (1280, 800)

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": RES[0], "height": RES[1]})

import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.experimental.prims import RigidPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt  # noqa: E402

from food import banana_models as BM  # noqa: E402
from food.banana import Banana, peel_colour_stage  # noqa: E402
from food.shapes import loft  # noqa: E402

omni.usd.get_context().new_stage()
app.update()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
scene = UsdPhysics.Scene.Define(stage, "/World/Physics")
scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr(9.81)
px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
px.CreateTimeStepsPerSecondAttr(HZ)
px.CreateEnableGPUDynamicsAttr(False)

peel_group = UsdPhysics.CollisionGroup.Define(stage, "/World/Groups/Peel")
peel_group.CreateFilteredGroupsRel().AddTarget("/World/Groups/Peel")


def mat(path, rgb, rough=0.5):
    m = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, rgb)))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return m


def mesh_prim(path, m, material=None):
    g = UsdGeom.Mesh.Define(stage, path)
    g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(m.vertices, np.float32)))
    g.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(m.faces), 3, np.int32)))
    g.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(m.faces, np.int32).ravel()))
    g.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    v = np.asarray(m.vertices, np.float32)
    g.CreateExtentAttr([Gf.Vec3f(*map(float, v.min(0))), Gf.Vec3f(*map(float, v.max(0)))])
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(material)
    return g


def quat_x_to(v):
    """Quaternion rotating +x onto unit vector v."""
    v = np.asarray(v, float) / np.linalg.norm(v)
    a = np.cross([1.0, 0, 0], v)
    c = float(np.dot([1.0, 0, 0], v))
    if np.linalg.norm(a) < 1e-9:
        return Gf.Quatf(1, 0, 0, 0) if c > 0 else Gf.Quatf(0, 0, 0, 1)
    q = np.array([1 + c, *a])
    q /= np.linalg.norm(q)
    return Gf.Quatf(*map(float, q))


ban = Banana(0)
ban.peel_surfaces()
n_strip = int(ban.p["ridges"])
zl, zh = ban._pulp_z[0] + 0.5e-3, ban.mer.zmax - 1.0e-3     # peel strips span the pulp length
z_edges = np.linspace(zh, zl, N_SEG + 1)                   # j = 0 at the stem end
th_edges = ban.phi + np.arange(n_strip + 1) * 2 * np.pi / n_strip    # strip boundaries on the ridges


def segment_mesh(k, j, n_arc=10, n_z=6):
    rings = []
    for z in np.linspace(z_edges[j + 1], z_edges[j], n_z):
        th = np.linspace(th_edges[k] + 0.004, th_edges[k + 1] - 0.004, n_arc)
        ro = ban.r_outer(th, np.full_like(th, z))
        ri = np.full_like(th, float(ban.r_pulp(z)) + 0.05e-3)
        P = np.concatenate([np.stack([ro * np.cos(th), ro * np.sin(th)], 1),
                            np.stack([ri[::-1] * np.cos(th[::-1]), ri[::-1] * np.sin(th[::-1])], 1)])
        rings.append(np.column_stack([P, np.full(len(P), z)]))
    return ban.bent(loft(rings))


def bent_point(r, th, z):
    return ban.bend.bend(np.array([r * math.cos(th), r * math.sin(th), z]))


lanes = []
peel_mats = {}
pulp_mat = mat("/World/Looks/Pulp", (0.93, 0.86, 0.66), 0.4)
for li, st in enumerate(STAGES):
    off = np.array([0.0, li * 0.16, 0.30])                # banana hangs with its stem up, 30 cm above the floor
    peel_mats[st] = mat(f"/World/Looks/Peel_{li}", peel_colour_stage(st), 0.45)
    ra, rs, rt = ban.p["peel_bond_ratios"]
    F_adh, F_seam, F_tear = (r * BM.peel_fracture_N(st) for r in (ra, rs, rt))
    base = f"/World/L{li}"
    UsdGeom.Xform.Define(stage, base)
    # pulp (held): visual + kinematic collider slabs
    pulp = UsdGeom.Xform.Define(stage, base + "/Pulp")
    pulp.AddTranslateOp().Set(Gf.Vec3d(*off))
    for k in range(3):
        mesh_prim(f"{base}/Pulp/Seg{k}", ban.bent(ban.pulp_segment(k)), pulp_mat)
    UsdPhysics.RigidBodyAPI.Apply(pulp.GetPrim()).CreateKinematicEnabledAttr(True)
    zz = np.linspace(zl, zh, 7)
    for s_ in range(6):
        c = ban.bend.bend(np.array([0, 0, 0.5 * (zz[s_] + zz[s_ + 1])]))
        rad = float(ban.r_pulp(0.5 * (zz[s_] + zz[s_ + 1]))) * 0.92
        sp = UsdGeom.Capsule.Define(stage, f"{base}/Pulp/Col{s_}")
        sp.CreateRadiusAttr(rad)
        sp.CreateHeightAttr(float(zz[1] - zz[0]))
        sp.CreateAxisAttr("Z")
        sp.AddTranslateOp().Set(Gf.Vec3d(*map(float, c)))
        sp.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(sp.GetPrim())
    segs = {}
    for k in range(n_strip):
        for j in range(N_SEG):
            m = segment_mesh(k, j)
            path = f"{base}/Peel/S{k}_{j:02d}"
            x = UsdGeom.Xform.Define(stage, path)
            c = np.asarray(m.centroid)
            x.AddTranslateOp().Set(Gf.Vec3d(*map(float, c + off)))
            mm = m.copy()
            mm.vertices = np.asarray(m.vertices) - c
            mesh_prim(path + "/Geom", mm, peel_mats[st])
            sph = UsdGeom.Sphere.Define(stage, path + "/Col")
            sph.CreateRadiusAttr(1.0e-3)
            sph.CreatePurposeAttr(UsdGeom.Tokens.guide)
            UsdPhysics.CollisionAPI.Apply(sph.GetPrim())
            peel_group.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget(sph.GetPath())
            UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
            UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(float(abs(m.volume) * ban.p["peel_density"]))
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(x.GetPrim())
            rb.CreateLinearDampingAttr(0.5)
            rb.CreateAngularDampingAttr(float(os.environ.get("PEEL_ADAMP", 2.0)))
            rb.CreateSolverPositionIterationCountAttr(32)
            rb.CreateSolverVelocityIterationCountAttr(4)
            segs[(k, j)] = (path, c)
    jdir = f"{base}/Joints"
    UsdGeom.Scope.Define(stage, jdir)
    nj = 0

    def joint(kind, b0, b1, pw, axis=None, brk=None, tag=""):
        """b0/b1: (path, centroid) or None (pulp); pw: world anchor point (object frame); axis for revolutes."""
        global nj
        J = (UsdPhysics.RevoluteJoint if kind == "rev" else UsdPhysics.FixedJoint).Define(stage, f"{jdir}/J{nj}")
        nj += 1
        for rel, b, setp, setr in ((J.CreateBody0Rel(), b0, J.CreateLocalPos0Attr, J.CreateLocalRot0Attr),
                                   (J.CreateBody1Rel(), b1, J.CreateLocalPos1Attr, J.CreateLocalRot1Attr)):
            if b is None:
                rel.SetTargets([base + "/Pulp"])
                setp().Set(Gf.Vec3f(*map(float, pw)))
            else:
                rel.SetTargets([b[0]])
                setp().Set(Gf.Vec3f(*map(float, pw - b[1])))
            setr().Set(quat_x_to(axis) if axis is not None else Gf.Quatf(1, 0, 0, 0))
        if kind == "rev":
            J.CreateAxisAttr("X")
        if brk is not None and not os.environ.get("PEEL_NOBREAK") and tag in os.environ.get("PEEL_BREAK",
                                                                                             "adh,seam,tear"):
            J.CreateBreakForceAttr(float(brk))             # force only: a point joint standing in for a distributed
                                                            # bond has an artificial moment arm (torque limits cascade)
        return J
    for k in range(n_strip):
        thm = 0.5 * (th_edges[k] + th_edges[k + 1])
        for j in range(N_SEG):
            zc = 0.5 * (z_edges[j] + z_edges[j + 1])
            pa = bent_point(float(ban.r_pulp(zc)) + 0.3e-3, thm, zc)
            joint("fix", None, segs[(k, j)], pa, brk=F_adh, tag="adh")                      # adhesion to the pulp
            if j + 1 < N_SEG:                                                     # hinge to the next segment
                zb = z_edges[j + 1]
                pb = bent_point(float(ban.r_pulp(zb)) + 1.2e-3, thm, zb)
                t1 = bent_point(1.0, thm + 0.01, zb) - bent_point(1.0, thm - 0.01, zb)
                joint("rev", segs[(k, j)], segs[(k, j + 1)], pb, axis=t1, brk=F_tear, tag="tear")
            # seam to the next strip along the ridge (strip n-1 closes the ring)
            kn = (k + 1) % n_strip
            ps = bent_point(float(ban.r_pulp(zc)) + 1.2e-3, th_edges[k + 1], zc)
            if not (j == 0):                                                      # the neck is snapped: top free
                joint("fix", segs[(k, j)], segs[(kn, j)], ps, brk=F_seam, tag="seam")
    # finger: dynamic body on a D6 spring to a moving target, fixed to the top of strip 0
    top = segs[(0, 0)]
    fp = top[1] + off
    finger = UsdGeom.Xform.Define(stage, base + "/Finger")
    finger.AddTranslateOp().Set(Gf.Vec3d(*map(float, fp)))
    fs = UsdGeom.Sphere.Define(stage, base + "/Finger/Geom")
    fs.CreateRadiusAttr(4e-3)
    UsdPhysics.RigidBodyAPI.Apply(finger.GetPrim())
    UsdPhysics.MassAPI.Apply(finger.GetPrim()).CreateMassAttr(0.003)
    frb = PhysxSchema.PhysxRigidBodyAPI.Apply(finger.GetPrim())
    frb.CreateDisableGravityAttr(True)
    frb.CreateSolverPositionIterationCountAttr(32)
    g = UsdPhysics.FixedJoint.Define(stage, base + "/FingerGrip")
    g.CreateBody0Rel().SetTargets([base + "/Finger"])
    g.CreateBody1Rel().SetTargets([top[0]])
    g.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
    g.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    d6 = UsdPhysics.Joint.Define(stage, base + "/FingerSpring")
    d6.CreateBody1Rel().SetTargets([base + "/Finger"])
    d6.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float, fp)))
    drives = {}
    for ax in ("transX", "transY", "transZ"):
        dr = UsdPhysics.DriveAPI.Apply(d6.GetPrim(), ax)
        dr.CreateTypeAttr("force")
        dr.CreateStiffnessAttr(K_FINGER)
        dr.CreateDampingAttr(8.0)
        dr.CreateTargetPositionAttr(0.0)
        drives[ax] = dr
    outward = bent_point(1.0, 0.5 * (th_edges[0] + th_edges[1]), zh) - bent_point(0.0, 0, zh)
    outward[2] = 0
    outward /= np.linalg.norm(outward)
    lanes.append({"stage": st, "F_adh": F_adh, "F_seam": F_seam, "F_tear": F_tear, "off": off, "fp": fp,
                  "drives": drives, "finger": base + "/Finger", "segs": segs, "outward": outward})

floor = UsdGeom.Cube.Define(stage, "/World/Floor")
floor.CreateSizeAttr(1.0)
floor.AddTranslateOp().Set(Gf.Vec3d(0, 0.16, 0.0))
floor.AddScaleOp().Set(Gf.Vec3f(2, 2, 0.01))
UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(mat("/World/Looks/Floor", (0.62, 0.62, 0.6), 0.7))
wall = UsdGeom.Cube.Define(stage, "/World/Wall")
wall.CreateSizeAttr(1.0)
wall.AddTranslateOp().Set(Gf.Vec3d(-0.25, 0.16, 0.5))
wall.AddScaleOp().Set(Gf.Vec3f(0.01, 2, 1))
UsdShade.MaterialBindingAPI.Apply(wall.GetPrim()).Bind(mat("/World/Looks/Wall", (0.85, 0.85, 0.83), 0.6))
light = UsdLux.DomeLight.Define(stage, "/World/Dome")
light.CreateIntensityAttr(350)
key = UsdLux.DistantLight.Define(stage, "/World/Key")
key.CreateIntensityAttr(900)
UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(50, 0, 30))

if RENDER:
    import omni.replicator.core as rep
    from PIL import Image
    cam = UsdGeom.Camera.Define(stage, "/World/Cam")
    cam.CreateFocalLengthAttr(24.0)
    cam.CreateHorizontalApertureAttr(36.0)
    cam.CreateVerticalApertureAttr(36.0 * RES[1] / RES[0])
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.005, 50))     # default near plane is 1 m
    c_eye = np.array([0.42, 0.16, 0.30])
    c_tgt = np.array([0.0, 0.16, 0.27])
    UsdGeom.Xformable(cam).AddTransformOp().Set(
        Gf.Matrix4d().SetLookAt(Gf.Vec3d(*c_eye), Gf.Vec3d(*c_tgt), Gf.Vec3d(0, 0, 1)).GetInverse())
    rp = rep.create.render_product(str(cam.GetPath()), RES)
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(rp)

if os.environ.get("PEEL_DEBUG"):
    stage.GetRootLayer().Export("/tmp/claude-1000/-home-mbeetz-Documents/d65e1594-5674-4c83-9927-e947275a38dd/scratchpad/peel_stage.usda")
if RENDER and os.environ.get("PEEL_DEBUG"):
    for _ in range(10):
        app.update()
    rep.orchestrator.step(rt_subframes=4, delta_time=0.0)
    rep.orchestrator.step(rt_subframes=1, delta_time=0.0)
    Image.fromarray(np.asarray(rgb.get_data())[..., :3].astype(np.uint8)).save(
        os.path.join(ROOT, "renders", "food", "banana_peel_debug.png"))
app_utils.play(commit=True)
if os.environ.get("PEEL_DEBUG"):
    pv = RigidPrim([f"/World/L0/Pulp", lanes[0]["segs"][(0, 0)][0], lanes[0]["segs"][(1, 5)][0], lanes[0]["finger"]])
    for k in range(6):
        print("[dbg]", k, np.round(pv.get_world_poses()[0].numpy(), 4).tolist(), flush=True)
        SimulationManager.step(steps=int(os.environ.get("PEEL_DEBUG")))
    app.close()
    sys.exit(0)
SimulationManager.step(steps=int(0.5 * HZ))
fingers = RigidPrim([l["finger"] for l in lanes])
seg_paths = [[l["segs"][(k, j)][0] for k in range(n_strip) for j in range(N_SEG)] for l in lanes]
seg_views = [RigidPrim(p) for p in seg_paths]
rest = [v.get_world_poses()[0].numpy().copy() for v in seg_views]
lag0 = np.array([l["fp"] for l in lanes]) - fingers.get_world_poses()[0].numpy()   # settled baseline (sag)
OUT = os.path.join(ROOT, "renders", "food")
os.makedirs(OUT, exist_ok=True)
# pull: 15 mm outwards, then down along the finger 25 mm past the tip
L = float(ban.L)
T1, T2 = 0.015 / V_PULL, (L + 0.05) / V_PULL
log = {li: [] for li in range(len(lanes))}
t, frame = 0.0, 0
shots = [1.0, 3.0, 5.5, 8.0, 10.2]
def target(l, t):
    if t < T1:
        return l["outward"] * V_PULL * t
    return l["outward"] * 0.015 + np.array([0, 0, -V_PULL * (t - T1)])


n_sub = int(0.05 * HZ)
while t < min(T1 + T2, float(os.environ.get("PEEL_T", 1e9))):
    for _ in range(n_sub):                               # move the spring targets every physics step
        for l in lanes:
            for ax, v in zip(("transX", "transY", "transZ"), target(l, t)):
                l["drives"][ax].GetTargetPositionAttr().Set(float(v))
        SimulationManager.step(steps=1)
        t += 1.0 / HZ
    fpos = fingers.get_world_poses()[0].numpy()
    for li, l in enumerate(lanes):
        tgt = l["fp"] + target(l, t - 1.0 / HZ)            # the target applied during the last step
        F = K_FINGER * np.linalg.norm(tgt - fpos[li] - lag0[li])
        if os.environ.get("PEEL_DBGF") and li == 2 and int(round(t / 0.05)) % 4 == 0:
            print("[dbgF]", round(t, 2), np.round(tgt * 1e3, 2).tolist(), np.round(fpos[li] * 1e3, 2).tolist(),
                  np.round(lag0[li] * 1e3, 2).tolist(), flush=True)
        P = seg_views[li].get_world_poses()[0].numpy()
        moved = np.linalg.norm(P - rest[li], axis=1) > 3e-3
        log[li].append({"t": round(t, 4), "F_N": float(F), "detached_segments": int(moved.sum()),
                        "strip0_detached": int(moved[:N_SEG].sum()), "others_detached": int(moved[N_SEG:].sum())})
    if RENDER and frame < len(shots) and t >= shots[frame]:
        for _ in range(3):
            app.update()
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0)
        Image.fromarray(np.asarray(rgb.get_data())[..., :3].astype(np.uint8)).save(
            os.path.join(OUT, f"banana_peel_{frame}.png"))
        frame += 1
    if int(round(t / 0.05)) % int(os.environ.get("PEEL_PRINT", 20)) == 0:
        print("[t]", round(t, 2), [(round(log[i][-1]["F_N"], 2), log[i][-1]["strip0_detached"], log[i][-1]["others_detached"])
                                   for i in range(len(lanes))], flush=True)
res = []
for li, l in enumerate(lanes):
    Fs = np.array([r["F_N"] for r in log[li]])
    s0 = log[li][-1]["strip0_detached"]
    res.append({"stage": l["stage"], "F_adh_N": l["F_adh"], "F_seam_N": l["F_seam"], "F_tear_N": l["F_tear"],
                "F_peak_N": float(Fs.max()), "F_median_pulling_N": float(np.median(Fs[Fs > 0.2])) if (Fs > 0.2).any()
                else 0.0, "strip0_segments_detached": s0, "segments_detached_total": log[li][-1]["detached_segments"],
                "others_detached": log[li][-1]["others_detached"],
                "outcome": ("peel slid off as a sleeve" if log[li][-1]["others_detached"] > N_SEG else
                            "strip peeled off" if s0 >= N_SEG - 1 else
                            "strip torn short" if s0 > 0 else "peel did not split"), "log": log[li]})
    print("[result]", l["stage"], res[-1]["outcome"], "peak", round(res[-1]["F_peak_N"], 2), "N median",
          round(res[-1]["F_median_pulling_N"], 2), "detached", s0, "/", N_SEG, "others", res[-1]["others_detached"],
          "| bonds adh/seam/tear", round(l["F_adh"], 2), round(l["F_seam"], 2), round(l["F_tear"], 2), flush=True)
os.makedirs(os.path.join(ROOT, "test_reports"), exist_ok=True)
json.dump({"banana": "banana_0 geometry", "strips": n_strip, "segments_per_strip": N_SEG, "hz": HZ,
           "k_finger_N_per_m": K_FINGER, "v_pull_m_s": V_PULL, "lanes": res},
          open(os.path.join(ROOT, "test_reports", "banana_peel.json"), "w"), indent=1)
app.close()
