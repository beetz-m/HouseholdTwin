"""Put things where they belong, in the PR2 kitchen: the put-away adapter (activities/put-away/adapter.py: belonging
verdicts, requests, obligations, the guards) with PR2 motions around every outcome. Each realisation is
    [drive to the stretch] + [PR2 motions] + [step: the kitchen-state outcome] (+ [motions away])
as coraplex plan nodes (ccram.cram_nodes.step). Places are stand-in trays (cram/putaway_kitchen.py).
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np

from ccram.adapter import Adapter
from ccram.cram_nodes import step

_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "put-away", "adapter.py")
_spec = importlib.util.spec_from_file_location("put_away_adapter", _p)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

import putaway_kitchen as K  # noqa: E402

UP = 0.20                       # transfer height above the counter (measured reach: <= counter + 0.20)
SLOTS = [(0.0, 0.0), (0.03, -0.03), (0.03, 0.03), (-0.02, 0.03)]   # where things go on a tray, in order


class PutAwayKitchenAdapter(base.PutAwayAdapter):
    def __init__(self, demo, world, context):
        base.PutAwayAdapter.__init__(self)
        self.world, self.context, self.robot = world, context, context.robot
        self.top = demo.counter_top
        self.arm = self.robot.right_arm
        self._ready, self._at = False, None
        self.slots = {p: 0 for p in K.PLACES}

    def execute(self, nodes, attach=None):                       # back to coraplex (the base adapter calls functions)
        Adapter.execute(self, nodes, attach)

    # -- motions
    def _M(self):
        import avocado_motions as M
        return M

    def _xyz(self, name):
        return self.world.get_body_by_name(name).global_pose.to_np()[:3, 3]

    def _move(self, xyz):
        M = self._M()
        return M.move(self.world, self.arm, np.asarray(xyz, float), M.TOP_ACROSS)

    def _look_at(self, xyz):
        from coraplex.robot_plans.actions.core.navigation import LookAtAction
        from semantic_digital_twin.spatial_types.spatial_types import Pose
        return LookAtAction(Pose.from_xyz_rpy(*map(float, xyz), reference_frame=self.world.root))

    def _goto(self, where):
        """Drive to a stretch of the counter (arms parked); nothing if already there (tracked at plan-build time)."""
        from coraplex.robot_plans.actions.core.navigation import NavigateAction
        from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
        from semantic_digital_twin.datastructures.definitions import TorsoState
        from semantic_digital_twin.spatial_types.spatial_types import Pose
        out = []
        if not self._ready:
            self._ready = True
            out += [ParkArmsAction(self.robot.all_arms), MoveTorsoAction(TorsoState.HIGH)]
        b = K.BASE_OBJECTS if where == "objects" else K.BASE_PLACES
        if self._at != b:
            self._at = b
            out += [ParkArmsAction(self.robot.all_arms),
                    NavigateAction(Pose.from_xyz_rpy(b[0], b[1], 0.0, reference_frame=self.world.root))]
        return out

    def _attach(self, name):
        M = self._M()
        return step(lambda: M.attach(self.world, self.world.get_body_by_name(name), self.arm))

    def _release(self, name):
        M = self._M()
        return step(lambda: M.release(self.world, self.world.get_body_by_name(name)))

    # -- realisations
    def r_surveying(self, R):
        c = np.mean([[x, y] for x, y, *_ in K.OBJECTS.values()], axis=0)
        return self._goto("objects") + [self._look_at([c[0], c[1], self.top])] + \
            [step(f) for f in base.PutAwayAdapter.r_surveying(self, R)]

    def r_putting_away(self, R):
        o, p = R["object"][1], R["destination"][1]
        outcome = [step(f) for f in base.PutAwayAdapter.r_putting_away(self, R)]
        if p not in K.PLACES or self.load(p) >= self.places[p]["capacity"]:   # not a tray, or full: the outcome step
            return self._goto("objects") + [self._look_at(self._xyz(o))] + outcome   # signals / records it
        M = self._M()
        h = K.height(o)
        gz = float(np.clip(0.6 * h, 0.015, 0.09))                          # grasp height above the object's base
        src = self._xyz(o)
        dx, dy = SLOTS[min(self.slots[p], len(SLOTS) - 1)]
        self.slots[p] += 1
        tray = np.array([K.PLACES[p][0] + dx, K.PLACES[p][1] + dy, self.top + K.TRAY[2]])
        return self._goto("objects") + [M.gripper(self.arm, True), self._move(src + [0, 0, UP]),
                                        self._move(src + [0, 0, gz]), M.gripper(self.arm, False), self._attach(o),
                                        self._move(src + [0, 0, UP])] + \
            self._goto("places") + [self._move(tray + [0, 0, UP]), self._move(tray + [0, 0, gz + 0.002]),
                                    M.gripper(self.arm, True), self._release(o), self._move(tray + [0, 0, UP])] + outcome

    def r_asking(self, R):
        o = R["query"][1]
        x = self._xyz(o)
        return self._goto("objects") + [self._look_at(x), self._move(x + [0, 0, UP])] + \
            [step(f) for f in base.PutAwayAdapter.r_asking(self, R)]


def make_adapter(demo, world, context):
    return PutAwayKitchenAdapter(demo, world, context)
