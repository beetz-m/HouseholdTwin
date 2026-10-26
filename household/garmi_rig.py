"""GARMI rig for the household twin: the imported GARMI (two FR3 arms with Franka hands, two-stage lift, pan-tilt
head, wheeled base) as one PhysX articulation, base fixed to the floor for manipulation, with

  * damped-least-squares Cartesian IK per arm on the PhysX Jacobian (as the lab's FR3 rig, twin/r01_rig.py), with a
    null-space posture bias,
  * force-limited Franka-hand grasping: finger drives in force mode, the max force = commanded grip force, targets
    ramped at gripper speed (an instantly set stiff target slams the fingers shut -- egg bench lesson),
  * TCP = hand frame + 0.1034 m along its z axis (Franka Hand flange-to-fingertip-centre).

Usage: rig = GarmiRig(stage, "/World/GARMI"); after play: rig.start(); rig.move_to("right", p, q); rig.grip(...)
"""

from __future__ import annotations

import math

import numpy as np

HZ = 240
DT = 1.0 / HZ
TCP_OFF = np.array([0.0, 0.0, 0.1034])


def qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def mat2q(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class GarmiRig:
    ARMS = {"arm_0": "arm_0", "arm_1": "arm_1"}

    def __init__(self, art_path, log=print, on_step=()):
        self.art_path, self.log, self.on_step = art_path, log, list(on_step)
        self.t = 0.0

    # ---------------------------------------------------------------- setup (after play)
    def start(self):
        from isaacsim.core.experimental.prims import Articulation
        self.art = Articulation(self.art_path)
        self.names = list(self.art.dof_names)
        self.links = list(self.art.link_names)
        self.idx = {}
        for a in ("arm_0", "arm_1"):
            self.idx[a] = {"arm": [self.names.index(f"{a}_fr3_joint{i}") for i in range(1, 8)],
                           "fingers": [self.names.index(f"{a}_gripper_fr3_finger_joint1"),
                                       self.names.index(f"{a}_gripper_fr3_finger_joint2")],
                           "hand_link": self.links.index(f"{a}_gripper_fr3_hand")}
        self.lift = [self.names.index(n) for n in ("lift_0_lower_joint", "lift_0_upper_joint")]
        lo, hi = (np.asarray(v.numpy()[0], float) for v in self.art.get_dof_limits())
        self.lim = (lo, hi)
        J = self.art.get_jacobian_matrices().numpy()
        self.floating = J.shape[3] == len(self.names) + 6
        self.log(f"[garmi] dofs={len(self.names)} links={len(self.links)} jacobian {J.shape} "
                 f"floating_layout={self.floating}")
        self.COM = {a: self.art.get_link_coms()[0].numpy()[0, self.idx[a]["hand_link"]].astype(float) for a in self.idx}
        self.posture = {}
        from isaacsim.core.experimental.prims import RigidPrim
        paths = self.art.link_paths[0]
        self.hand_rp = {a: RigidPrim(paths[self.idx[a]["hand_link"]]) for a in self.idx}

    def tcp(self, arm):
        p, q = self.hand_rp[arm].get_world_poses()
        p, q = p.numpy()[0].astype(float), q.numpy()[0].astype(float)
        R = qmat(q)
        return p + R @ TCP_OFF, q, p + R @ self.COM[arm]

    def q(self):
        return self.art.get_dof_positions().numpy()[0].astype(float)

    def targets(self):
        return self.art.get_dof_position_targets().numpy()[0].astype(float)

    def set_targets(self, idx, vals):
        self.art.set_dof_position_targets(np.array([vals], dtype=np.float32), dof_indices=idx)

    def step(self, n=1):
        from isaacsim.core.simulation_manager import SimulationManager
        for _ in range(n):
            SimulationManager.step(steps=1)
            self.t += DT
            for f in self.on_step:
                f(DT)

    # ---------------------------------------------------------------- motion
    def ik_step(self, arm, target_p, target_q, gain=0.15, damping=0.03, max_dq=0.012):
        ii = self.idx[arm]
        p_tcp, q_cur, p_ref = self.tcp(arm)
        J = self.art.get_jacobian_matrices().numpy()[0]
        row = ii["hand_link"] if self.floating else ii["hand_link"] - 1
        cols = [6 + c for c in ii["arm"]] if self.floating else ii["arm"]
        J = J[row][:, cols].astype(float)
        J[:3] -= skew(p_tcp - p_ref) @ J[3:]
        qe = qmul(target_q, qconj(q_cur))
        e = np.concatenate([target_p - p_tcp, 2 * np.sign(qe[0]) * qe[1:]])
        Jinv = J.T @ np.linalg.inv(J @ J.T + damping ** 2 * np.eye(6))
        dq = Jinv @ e
        post = self.posture.get(arm)
        if post is not None:
            bias = post - self.q()[ii["arm"]]
            bias[6] = 0.0
            dq += (np.eye(7) - Jinv @ J) @ (0.02 * bias)
        dq = np.clip(gain * dq, -max_dq, max_dq)
        tgt = self.targets()[ii["arm"]] + dq
        lo, hi = self.lim
        tgt = np.clip(tgt, lo[ii["arm"]] + 0.01, hi[ii["arm"]] - 0.01)
        self.set_targets(ii["arm"], tgt)
        return np.linalg.norm(e[:3]), math.degrees(np.linalg.norm(e[3:]))

    def move_to(self, arm, goal_p, goal_q, speed=0.20, tol_mm=1.0, tol_deg=1.0, timeout=4.0, settle=True):
        p0, q0, _ = self.tcp(arm)
        goal_p = np.asarray(goal_p, float)
        n = max(1, int(np.linalg.norm(goal_p - p0) / speed * HZ))
        for i in range(1, n + 1):
            a = i / n
            q = q0 * (1 - a) + goal_q * a if np.dot(q0, goal_q) >= 0 else q0 * (1 - a) - goal_q * a
            self.ik_step(arm, p0 + (goal_p - p0) * a, q / np.linalg.norm(q))
            self.step()
        if not settle:
            return True, None, None
        for k in range(int(timeout * HZ)):
            ep, er = self.ik_step(arm, goal_p, goal_q)
            self.step()
            if ep * 1e3 < tol_mm and er < tol_deg:
                return True, ep * 1e3, er
        self.log(f"[garmi] {arm} move_to timeout: {ep * 1e3:.1f} mm {er:.1f} deg")
        return False, ep * 1e3, er

    def move_joints(self, idx, goal, seconds=2.0):
        q0 = self.targets()[idx]
        n = max(1, int(seconds * HZ))
        for i in range(1, n + 1):
            self.set_targets(idx, q0 + (np.asarray(goal) - q0) * i / n)
            self.step()

    # ---------------------------------------------------------------- gripper (force-limited Franka hand)
    def set_grip_force(self, arm, F, stage):
        from pxr import UsdPhysics
        for j in ("finger_joint1", "finger_joint2"):
            prim = self._joint_prim(stage, f"{arm}_gripper_fr3_{j}")
            UsdPhysics.DriveAPI(prim, "linear").GetMaxForceAttr().Set(float(F))

    def _joint_prim(self, stage, name):
        for p in stage.Traverse():
            if p.GetName() == name:
                return p
        raise KeyError(name)

    def grip(self, arm, width, speed=0.05):
        """Move both fingers to the opening `width` (m) at gripper speed; the force cap governs at contact."""
        ii = self.idx[arm]["fingers"]
        w0 = self.targets()[ii]
        goal = np.array([width / 2, width / 2])
        n = max(1, int(np.abs(goal - w0).max() / speed * HZ))
        for i in range(1, n + 1):
            self.set_targets(ii, w0 + (goal - w0) * i / n)
            self.step()

    def finger_opening(self, arm):
        return float(self.q()[self.idx[arm]["fingers"]].sum())
