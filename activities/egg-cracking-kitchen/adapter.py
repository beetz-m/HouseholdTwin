"""Cracking eggs in the PR2 kitchen: the egg activity's adapter (activities/egg-cracking/adapter.py: emulator truth,
belief decisions, observations) with PR2 motions around every outcome. Each realisation is
    [PR2 motions to the place of the action] + [step: the emulator outcome -> observation / failure] + [world update]
    + [PR2 motions away]
as coraplex plan nodes (ccram.cram_nodes.step runs the emulator between the motions). The kinematic simulation has no
contact forces: forces, cracks, contents and shell come from the emulator (as in the avocado demo).
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np

from ccram.adapter import Adapter
from ccram.cram_nodes import step

_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "egg-cracking", "adapter.py")
_spec = importlib.util.spec_from_file_location("egg_cracking_adapter", _p)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

import egg_kitchen as K  # noqa: E402

UP = 0.12                 # clearance above the counter for transfers [m] (reach: z <= counter + 0.18)
GRIP_Z = 0.028            # TCP height above an egg's base for a top grasp round its equator
TILT_DEG = 60.0           # how far the cup is tipped over the bowl


class EggKitchenAdapter(base.EggAdapter):
    def __init__(self, demo, world, context):
        base.EggAdapter.__init__(self, eggs={s: dict(instance=i) for s, i in demo.eggs.items()})
        self.world, self.context, self.robot = world, context, context.robot
        self.demo, self.top = demo, demo.counter_top
        self.arm = self.robot.right_arm
        self._ready = False
        self.home = {s: np.array([*K.LAYOUT[s], self.top]) for s in self.emu}

    # -- coraplex execution (the base adapter's callables become step nodes)
    def execute(self, nodes, attach=None):
        Adapter.execute(self, nodes, attach)

    def _p(self, name):
        return self.world.get_body_by_name(name).global_pose.to_np()[:3, 3]

    def _move(self, xyz, R=None):
        import avocado_motions as M
        return M.move(self.world, self.arm, np.asarray(xyz, float), M.TOP_ACROSS if R is None else R)

    def _grip(self, open_):
        import avocado_motions as M
        return M.gripper(self.arm, open_)

    def _attach(self, body):
        import avocado_motions as M
        return step(lambda: M.attach(self.world, self.world.get_body_by_name(body), self.arm))

    def _release(self, body):
        import avocado_motions as M
        return step(lambda: M.release(self.world, self.world.get_body_by_name(body)))

    def _prepare(self):
        """Once: park, torso up, drive to the counter, look at it."""
        if self._ready:
            return []
        from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction
        from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
        from semantic_digital_twin.datastructures.definitions import TorsoState
        from semantic_digital_twin.spatial_types.spatial_types import Pose
        self._ready = True
        c = K.LAYOUT["cup-1"]
        return [ParkArmsAction(self.robot.all_arms), MoveTorsoAction(TorsoState.HIGH),
                NavigateAction(Pose.from_xyz_rpy(*K.BASE_XYZ, reference_frame=self.world.root)),
                LookAtAction(Pose.from_xyz_rpy(c[0], c[1], self.top, reference_frame=self.world.root))]

    def _look_at(self, xyz):
        from coraplex.robot_plans.actions.core.navigation import LookAtAction
        from semantic_digital_twin.spatial_types.spatial_types import Pose
        return LookAtAction(Pose.from_xyz_rpy(*map(float, xyz), reference_frame=self.world.root))

    def _take(self, egg):
        p = self._p(egg)
        return [self._grip(True), self._move(p + [0, 0, UP]), self._move(p + [0, 0, GRIP_Z]), self._grip(False)]

    # -- realisations: motions around the emulator's outcome
    def r_float_testing(self, R):
        egg = R["object"][1]
        g = np.array([*K.LAYOUT["water-glass"], self.top])
        home = self.home[egg]
        return self._prepare() + self._take(egg) + [self._attach(egg), self._move(self._p(egg) + [0, 0, UP]),
                                                    self._move(g + [0, 0, 0.17]), self._move(g + [0, 0, 0.08 + 0.01])] + \
            [step(f) for f in base.EggAdapter.r_float_testing(self, R)] + \
            [self._move(g + [0, 0, 0.17]), self._move(home + [0, 0, UP]), self._move(home + [0, 0, GRIP_Z]),
             self._grip(True), self._release(egg), self._move(home + [0, 0, UP])]

    def r_weighing(self, R):
        egg = R["object"][1]
        return self._prepare() + [self._look_at(self._p(egg))] + [step(f) for f in base.EggAdapter.r_weighing(self, R)]

    def r_picking_up(self, R):
        egg = R["object"][1]
        return self._prepare() + self._take(egg) + [step(f) for f in base.EggAdapter.r_picking_up(self, R)] + \
            [self._attach(egg), self._move(self._p(egg) + [0, 0, UP])]

    def _strike_point(self, tool):
        if tool == "bowl-rim":
            b = K.LAYOUT["batch-bowl"]
            return np.array([b[0] - 0.08, b[1], self.top + 0.07])
        return np.array([*K.LAYOUT["counter"], self.top])

    def r_cracking(self, R):
        p = self._strike_point(R["tool"][1])
        return self._prepare() + [self._move(p + [0, 0, 0.10 + GRIP_Z]), self._move(p + [0, 0, 0.005 + GRIP_Z])] + \
            [step(f) for f in base.EggAdapter.r_cracking(self, R)] + [self._move(p + [0, 0, 0.10 + GRIP_Z])]

    def _show_opened(self, egg, vessel):
        """World update: the egg's shell halves to the shell spot, its contents (a yolk disc) into the vessel."""
        import avocado_motions as M
        inst = self.spec[egg]["instance"]
        b = self.world.get_body_by_name(egg)
        M.release(self.world, b)
        with self.world.modify_world():
            self.world.remove_branch_from_world(b)
        s = K.LAYOUT["shells"]
        for k, dx in (("a", 0.0), ("b", 0.05)):
            K._place(self.world, K._mesh_body(f"{egg}-shell-{k}", K.half_shell_mesh(inst), (0.92, 0.90, 0.85)),
                     K.pose(s[0] + dx, s[1], self.top + 0.0005))
        v = self._p(vessel)
        broken = self.obs[egg]["yolk"] == "broken"
        K._place(self.world, K._mesh_body(f"contents-{egg}", K.disc_mesh("yolk_broken" if broken else "yolk", 0.034 if broken
                                                                       else 0.022, 0.010), (0.98, 0.72, 0.12), False),
                 K.pose(v[0], v[1], v[2] + 0.004))

    def r_opening(self, R):
        egg, dest = R["object"][1], R["destination"][1]
        v = np.array([*K.LAYOUT.get(dest, K.LAYOUT["cup-1"]), self.top])
        outcome = base.EggAdapter.r_opening(self, R)

        def open_and_show():
            try:
                for f in outcome:
                    f()
            finally:
                if self.obs[egg]["where"] == dest:
                    self._show_opened(egg, dest)
        return self._prepare() + [self._move(v + [0, 0, 0.06 + 0.06 + GRIP_Z])] + [step(open_and_show)] + \
            [self._grip(True), self._move(v + [0, 0, 0.17])]

    def r_inspecting(self, R):
        return self._prepare() + [self._look_at(self._p(R["container"][1]))] + \
            [step(f) for f in base.EggAdapter.r_inspecting(self, R)]

    def r_fishing_out(self, R):
        c = self._p(R["container"][1])
        return self._prepare() + [self._grip(False), self._move(c + [0, 0, UP]), self._move(c + [0, 0, 0.035])] + \
            [step(f) for f in base.EggAdapter.r_fishing_out(self, R)] + [self._move(c + [0, 0, UP])]

    def r_emptying(self, R):
        import avocado_motions as M
        cup, bowl = R["object"][1], R["destination"][1]
        c, b = self._p(cup), self._p(bowl)
        egg = self.cups[cup]["egg"]
        from scipy.spatial.transform import Rotation
        tilt = Rotation.from_euler("x", TILT_DEG, degrees=True).as_matrix() @ M.TOP_ACROSS   # reachable (probed):
        #                                                       world-x +45/+60 deg; world-y +45/+60 deg are not

        def tip_and_show():
            for f in base.EggAdapter.r_emptying(self, R):
                f()
            if egg:
                d = self.world.get_body_by_name(f"contents-{egg}")
                M.release(self.world, d)
                T = np.eye(4)
                T[:3, 3] = b + [0, 0, 0.006 + 0.004 * len(self.bowl)]
                from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
                with self.world.modify_world():
                    self.world.remove_branch_from_world(d)
                K._place(self.world, K._mesh_body(f"contents-{egg}", K.disc_mesh("yolk", 0.022, 0.010),
                                                  (0.98, 0.72, 0.12), False), T)
                _ = HomogeneousTransformationMatrix
        moves = [self._grip(True), self._move(c + [0, 0, UP]), self._move(c + [0, 0, 0.045]), self._grip(False),
                 self._attach(cup)]
        if egg:
            moves.append(step(lambda: M.attach(self.world, self.world.get_body_by_name(f"contents-{egg}"), self.arm)))
        moves += [self._move(c + [0, 0, 0.17]), self._move(b + [-0.04, 0, 0.17]),
                  self._move(b + [-0.04, 0, 0.15], tilt), step(tip_and_show),
                  self._move(b + [-0.04, 0, 0.17]), self._move(c + [0, 0, 0.17]), self._move(c + [0, 0, 0.047]),
                  self._grip(True), self._release(cup), self._move(c + [0, 0, UP])]
        return self._prepare() + moves


def make_adapter(demo, world, context):
    """activity.toml [twin] adapter entry point."""
    return EggKitchenAdapter(demo, world, context)
