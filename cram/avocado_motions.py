"""PR2 motions for halving and pitting an avocado lying on its side with the cut plane horizontal.

Fruit placement (cram/avocado_world.py FRUIT_CUT_HORIZONTAL): the twin's cut plane is horizontal, half_a (the half
that keeps the seed) below it, the long axis along the world y axis, the stem end at +y. Then

- squeeze: top grasp across the widest section, fingers closed to a width that presses ~4 N into the skin;
- hold: the left gripper pinches the neck from the side (+y), one finger above and one below the cut plane;
- cut round the seed: the knife, held flat (blade in the cut plane), moves its tip once round the seed outline the
  belief predicts (CutRoundSeedAction, a CuttingAction with its own tool path);
- twist: the right gripper takes the upper half from above and turns it about the vertical (= the cut-plane
  normal), lifts it off and puts it down beside, cut face down;
- spoon: the right gripper takes a spoon, slides the bowl down along the seed on the robot's side, under the seed,
  levers it up and lifts the seed out.

Grasped things are attached to the gripper's tool frame between plan segments (and released onto the world root).
"""

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from coraplex.datastructures.enums import CuttingTechnique, ToolPathSegmentKind
from coraplex.robot_plans.actions.composite.tool_based import CuttingAction
from coraplex.robot_plans.actions.composite.tool_paths import ToolPath, ToolPathSegment
from coraplex.robot_plans.actions.core.robot_body import MoveManipulatorAction, SetGripperAction
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose

# gripper orientations (columns: tool x = approach, tool y = finger closing direction, tool z)
TOP_ACROSS = np.column_stack([[0, 0, -1], [1, 0, 0], [0, -1, 0]])     # from above, fingers close along world x
SIDE_FROM_LEFT = np.column_stack([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])  # from +y, fingers close vertically


def rot_z(deg):
    return Rotation.from_euler("z", deg, degrees=True).as_matrix()


def rot_y(deg):
    return Rotation.from_euler("y", deg, degrees=True).as_matrix()


def pose(world, p, R):
    r, pi, y = Rotation.from_matrix(R).as_euler("xyz")
    return Pose.from_xyz_rpy(*map(float, p), roll=r, pitch=pi, yaw=y, reference_frame=world.root)


def move(world, arm, p, R):
    return MoveManipulatorAction(pose(world, p, R), arm.end_effector, True)


def gripper(arm, open_):
    return SetGripperAction(arm.end_effector, GripperState.OPEN if open_ else GripperState.CLOSE)


def attach(world, body, arm):
    with world.modify_world():
        world.move_branch_with_fixed_connection(body, arm.end_effector.tool_frame)


def release(world, body):
    with world.modify_world():
        world.move_branch_with_fixed_connection(body, world.root)


@dataclass
class FruitGeometry:
    """Where the robot believes things are: fruit origin (twin asset origin) in the world, the cut plane height, and
    the seed (centre and semi-axes) from the belief means."""
    origin: np.ndarray
    length_m: float
    diameter_m: float
    seed_length_m: float
    seed_width_m: float

    @property
    def cut_z(self):
        return float(self.origin[2])

    @property
    def seed_centre(self):                     # the twin's seed sits 0.10 L towards the broad end (-y here)
        return self.origin + np.array([0.0, -0.10 * self.length_m, 0.0])

    @property
    def neck(self):
        return self.origin + np.array([0.0, 0.36 * self.length_m, 0.0])


# %% squeeze
FINGER_ANGLE = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.548])
FINGER_GAP = np.array([0.0, 0.018, 0.036, 0.053, 0.070, 0.093])
"""PR2 finger joint angle -> gap between the finger pads (m), read off the kinematic model (tip links 0.0305 m apart
when closed)."""


def fingers_to_gap(arm, gap_m):
    """Set the PR2 finger joints so that the pads stand ``gap_m`` apart (the kinematic simulation has no force
    control: the squeeze is commanded as a width, the force comes from the fruit model)."""
    from coraplex.plans.factories import execute_single
    from coraplex.robot_plans.motions.robot_body import MoveJointsMotion
    th = float(np.interp(gap_m, FINGER_GAP, FINGER_ANGLE))
    side = "l" if "l_gripper" in str(arm.end_effector.tool_frame.name) else "r"
    return execute_single(MoveJointsMotion(names=[f"{side}_gripper_l_finger_joint", f"{side}_gripper_r_finger_joint"],
                                           positions=[th, th]))


def squeeze_steps(world, arm, g, squeeze_mm):
    c = g.origin + np.array([0.0, -0.08 * g.length_m, 0.0])
    return [gripper(arm, True), move(world, arm, c + [0, 0, 0.12], TOP_ACROSS), move(world, arm, c, TOP_ACROSS),
            fingers_to_gap(arm, g.diameter_m), fingers_to_gap(arm, g.diameter_m - squeeze_mm * 1e-3),
            gripper(arm, True), move(world, arm, c + [0, 0, 0.12], TOP_ACROSS)]


# %% hold at the neck
def hold_steps(world, arm, g, dz=0.0):
    p = g.neck + np.array([0.0, 0.0, dz])
    return [gripper(arm, True), move(world, arm, p + [0, 0.12, 0], SIDE_FROM_LEFT), move(world, arm, p, SIDE_FROM_LEFT),
            gripper(arm, False)]


def hold_from_above_steps(world, arm, g):
    """Hold the fruit down at the neck from above, the finger pads 2 cm above the cut plane (clear of the knife
    blade, which lies flat in the plane and reaches past the neck)."""
    p = g.neck + np.array([0.0, 0.0, 0.02])
    return [gripper(arm, True), move(world, arm, p + [0, 0, 0.10], TOP_ACROSS), move(world, arm, p, TOP_ACROSS),
            fingers_to_gap(arm, 0.6 * g.diameter_m)]


def back_off_flat_steps(world, arm, tcp_xyz):
    """Open and draw the gripper back from the knife lying flat, without turning it near the counter."""
    p = np.asarray(tcp_xyz, float)
    return [gripper(arm, True), move(world, arm, p + [0, 0, 0.06], KNIFE_FLAT),
            move(world, arm, p + [0, 0, 0.15], KNIFE_FLAT)]


def let_go_from_above_steps(world, arm, g):
    return [gripper(arm, True), move(world, arm, g.neck + [0, 0, 0.13], TOP_ACROSS)]


def let_go_steps(world, arm, g):
    return [gripper(arm, True), move(world, arm, g.neck + [0, 0.15, 0.03], SIDE_FROM_LEFT)]


# %% cut round the seed
@dataclass(kw_only=True)
class CutRoundSeedAction(CuttingAction):
    """
    Cut through skin and flesh all round the seed: the knife, held flat in the (horizontal) cut plane, enters from the
    robot's side and moves its tip once round the seed outline (belief means + a margin on the seed coat).
    """

    seed_centre: tuple = (0.0, 0.0, 0.0)
    """Seed centre in the world frame (the cut plane passes through it)."""
    seed_semi_axes: tuple = (0.02, 0.016)
    """Semi-axes of the seed outline in the cut plane: along the long axis (y), across (x), in metres."""
    margin: float = 0.002
    """Distance of the knife tip from the seed outline (stays on the seed coat)."""
    technique: CuttingTechnique = CuttingTechnique.HALVING

    def _build_tool_path(self) -> ToolPath:
        a, b = self.seed_semi_axes[0] + self.margin, self.seed_semi_axes[1] + self.margin
        start = np.array([-b - 0.05, 0.0, 0.0])
        entry = np.array([-b, 0.0, 0.0])

        def loop(t):
            th = -math.pi / 2 + 2 * math.pi * t                # from the robot's side, once round
            return np.array([b * math.sin(th), a * math.cos(th), 0.0])

        return ToolPath([
            ToolPathSegment(ToolPathSegmentKind.APPROACH, 1.0, lambda t: start + (entry - start) * t, 0),
            ToolPathSegment(ToolPathSegmentKind.SAW, 4.0, loop, 0),
            ToolPathSegment(ToolPathSegmentKind.RETRACT, 1.0, lambda t: entry + (start - entry) * t, 0)])

    def _path_frame(self) -> HomogeneousTransformationMatrix:
        T = np.eye(4)
        T[:3, 3] = self.seed_centre
        return HomogeneousTransformationMatrix(T, reference_frame=self.world.root)

    align_frame: object = None
    """A body whose y axis is the long axis and whose z axis points to the robot (world -x): the knife (mounted
    with its z against the gripper's approach axis) is then held flat in the cut plane with the gripper pointing
    away from the robot (see cut_alignment_frame)."""

    @property
    def _alignment_target(self):
        return self.align_frame


CUT_ALIGN = np.column_stack([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).astype(float)


def cut_alignment_frame(world, at, name="avocado_cut_frame"):
    """A frame body at ``at`` with the orientation CUT_ALIGN (alignment target of CutRoundSeedAction)."""
    from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
    from semantic_digital_twin.world_description.connections import FixedConnection
    from semantic_digital_twin.world_description.world_entity import Body
    fb = Body(name=PrefixedName(name))
    T = np.eye(4)
    T[:3, :3] = CUT_ALIGN
    T[:3, 3] = at
    with world.modify_world():
        world.add_kinematic_structure_entity(fb)
        world.add_connection(FixedConnection(parent=world.root, child=fb,
                                             parent_T_connection_expression=HomogeneousTransformationMatrix(
                                                 T, reference_frame=world.root)))
    return fb


KNIFE_FLAT = np.column_stack([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).astype(float)
"""Gripper orientation that holds the mounted knife flat, blade along +y, gripper pointing away from the robot."""


def lay_down_tool_steps(world, arm, tcp_xyz):
    p = np.asarray(tcp_xyz, float)
    return [move(world, arm, p + [0, 0, 0.08], KNIFE_FLAT), move(world, arm, p, KNIFE_FLAT)]


# %% twist the upper half off
def twist_steps(world, arm, g, twist_deg=90.0):
    c = g.origin + np.array([0.0, -0.02 * g.length_m, 0.016])
    return [gripper(arm, True), move(world, arm, c + [0, 0, 0.12], TOP_ACROSS), move(world, arm, c, TOP_ACROSS),
            gripper(arm, False)], c


def twist_motion(world, arm, c, twist_deg=90.0):
    R = rot_z(twist_deg) @ TOP_ACROSS                              # turn about the vertical (cut-plane normal)
    return [move(world, arm, c, rot_z(twist_deg / 2) @ TOP_ACROSS), move(world, arm, c, R),
            move(world, arm, c + [0, 0, 0.08], R)]


def put_aside_steps(world, arm, aside_xy, top, grasp_above_cut, twist_deg=90.0):
    """Put the held upper half down beside, cut face down: the tool frame ``grasp_above_cut`` above the counter."""
    R = rot_z(twist_deg) @ TOP_ACROSS
    p = np.array([aside_xy[0], aside_xy[1], top + grasp_above_cut + 0.0005])
    return [move(world, arm, p + [0, 0, 0.08], R), move(world, arm, p, R)]


# %% spoon
TOP_ALONG_Y = np.column_stack([[0, 0, -1], [0, 1, 0], [1, 0, 0]])     # from above, fingers close along world y
SPOON_BOWL = np.array([0.06, 0.0, 0.003])
"""Bowl centre in the spoon mesh frame (coraplex resources spoon.stl: bowl at +x, handle at -x, bowl opening +z)."""
SPOON_HANDLE = np.array([-0.07, 0.0, 0.005])


def spoon_grasp_steps(world, arm, spoon_body):
    T = spoon_body.global_pose.to_np()
    h = T[:3, :3] @ SPOON_HANDLE + T[:3, 3]
    return [gripper(arm, True), move(world, arm, h + [0, 0, 0.12], TOP_ALONG_Y), move(world, arm, h, TOP_ALONG_Y),
            gripper(arm, False)]


def spoon_tool_pose(bowl, tilt_deg, tool_T_spoon):
    """Tool-frame pose that puts the spoon bowl at ``bowl``, the spoon pointing towards the robot (-x) and down by
    ``tilt_deg``, bowl opening up, given how the spoon sits in the gripper (``tool_T_spoon``, 4x4). The hand then
    reaches over the fruit from the far side, as a person scoops a seed out towards themselves."""
    W = np.eye(4)
    W[:3, :3] = rot_z(180.0) @ rot_y(tilt_deg)
    W[:3, 3] = np.asarray(bowl) - W[:3, :3] @ SPOON_BOWL
    T = W @ np.linalg.inv(tool_T_spoon)
    return T[:3, 3], T[:3, :3]


def scoop_waypoints(g, bowl_half_width=0.02):
    """Bowl centre path and tilt: down along the seed on its far side, under the seed towards the robot (then the
    seed is taken up), lever up, lift."""
    s = g.seed_centre
    b = g.seed_width_m / 2
    under = [((s[0] + b + bowl_half_width, s[1], g.cut_z + 0.05), 70.0),
             ((s[0] + b + 0.008, s[1], g.cut_z + 0.005), 65.0),
             ((s[0] + b * 0.6, s[1], g.cut_z - 0.7 * b), 45.0),
             ((s[0] + b * 0.1, s[1], g.cut_z - 0.85 * b), 35.0)]
    out = [((s[0] - b * 0.2, s[1], g.cut_z - 0.4 * b), 25.0),          # lever: the bowl rises under the seed
           ((s[0] - b * 0.2 - 0.03, s[1], g.cut_z + 0.08), 25.0)]      # up and towards the robot
    return under, out


def seed_drop_waypoints(drop_xyz):
    """Carry the seed to ``drop_xyz`` (bowl centre just above the board) and hold it there."""
    p = np.asarray(drop_xyz, float)
    return [((p[0], p[1], p[2] + 0.06), 25.0), ((p[0], p[1], p[2]), 25.0)]


def spoon_moves(world, arm, waypoints, tool_T_spoon):
    return [move(world, arm, *spoon_tool_pose(bowl, tilt, tool_T_spoon)) for bowl, tilt in waypoints]


# %% agent-planned hold, seed-coat peeling, lemon, spoon back
def hold_at_steps(world, arm, g, hold):
    """Hold the fruit from above at the planned point (cram/avocado_agent.plan_hold): pads ``tcp_height`` above the
    cut plane, closed to the believed width there."""
    p = g.origin + np.array([0.0, hold["y"], hold["tcp_height"]])
    if hold.get("kind") == "press":                                  # closed fingers press on the top
        return [fingers_to_gap(arm, 0.0), move(world, arm, p + [0, 0, 0.10], TOP_ACROSS), move(world, arm, p, TOP_ACROSS)]
    return [gripper(arm, True), move(world, arm, p + [0, 0, 0.10], TOP_ACROSS), move(world, arm, p, TOP_ACROSS),
            fingers_to_gap(arm, max(0.0, hold["width"] - 0.002))]


def let_go_at_steps(world, arm, g, hold):
    p = g.origin + np.array([0.0, hold["y"], hold["tcp_height"]])
    return [gripper(arm, True), move(world, arm, p + [0, 0, 0.12], TOP_ACROSS)]


def peel_coat_waypoints(g, n=5):
    """Spoon-tip pass along the far wall of the seed cavity (bowl centre path, tilt), then out."""
    s = g.seed_centre
    a, b = g.seed_length_m / 2, g.seed_width_m / 2
    pts = [((s[0] + 0.7 * b, s[1], g.cut_z + 0.03), 55.0)]
    for th in np.linspace(-0.9, 0.9, n):                      # round the far half of the cavity wall
        pts.append(((s[0] + 0.55 * b * math.cos(th), s[1] + 0.55 * a * math.sin(th), g.cut_z - 0.45 * b), 40.0))
    pts.append(((s[0] - 0.03, s[1], g.cut_z + 0.05), 40.0))   # out up and towards the robot (straight up is blocked)
    return pts


def spoon_back_steps(world, arm, spoon_home, tool_T_spoon):
    """Put the spoon back where it was taken from (open the gripper afterwards)."""
    T = spoon_home @ np.linalg.inv(tool_T_spoon)
    p, Rm = T[:3, 3], T[:3, :3]
    return [move(world, arm, p + [0, 0, 0.08], Rm), move(world, arm, p, Rm)], p, Rm


def lemon_grasp_steps(world, arm, lemon_centre, width):
    p = np.asarray(lemon_centre, float)
    return [gripper(arm, True), move(world, arm, p + [0, 0, 0.12], TOP_ACROSS), move(world, arm, p, TOP_ACROSS),
            fingers_to_gap(arm, width - 0.002)]


def lemon_squeeze_steps(world, arm, over_xyz, width):
    """Hold the lemon 8 cm above the cut face and squeeze it (fingers 6 mm narrower: the juice is not simulated)."""
    p = np.asarray(over_xyz, float) + [0, 0, 0.08]
    return [move(world, arm, p + [0, 0, 0.06], TOP_ACROSS), move(world, arm, p, TOP_ACROSS),
            fingers_to_gap(arm, width - 0.008), fingers_to_gap(arm, width - 0.002),
            move(world, arm, p + [0, 0, 0.06], TOP_ACROSS)]
