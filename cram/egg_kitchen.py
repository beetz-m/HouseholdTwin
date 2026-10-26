"""The egg kitchen: the PR2 in the apartment (coraplex.testing.setup_world) with eggs, a cracking cup, a batch bowl
and a glass of water on the counter -- the CRAM twin of the egg-cracking activity (activities/egg-cracking-kitchen).

The eggs are bodies built from the egg twin's shell geometry (food/egg.py, register instance, meshes cached in
cram/meshes/egg_<i>.stl); cup, bowl and glass are open cylinders (trimesh). Outcomes (cracks, contents, shell) come
from the egg emulator, which the adapter runs between the PR2's motions; the world shows them: the egg disappears
when opened, its contents appear in the cup and move with it into the bowl, the two shell halves lie on the counter.

Live: LIVE_VIZ_PORT=8769 CRAMERA_DATA=~/Documents/HouseholdTwin/cram/cramera_data_egg \
      ~/Documents/ConceptualCRAM/bin/ccram run ~/Documents/HouseholdTwin/activities/egg-cracking-kitchen --workbench
      (under cramera-live: see the activity's README)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from coraplex.datastructures.dataclasses import Context  # noqa: E402
from coraplex.datastructures.enums import VisualizationBackend  # noqa: E402
from coraplex.demonstrations import RobotDemonstration  # noqa: E402
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName  # noqa: E402
from semantic_digital_twin.robots.pr2 import PR2  # noqa: E402
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix  # noqa: E402
from semantic_digital_twin.world import World  # noqa: E402
from semantic_digital_twin.world_description.geometry import Color, Mesh  # noqa: E402
from semantic_digital_twin.world_description.shape_collection import ShapeCollection  # noqa: E402
from semantic_digital_twin.world_description.world_entity import Body  # noqa: E402

from avocado_world import pose, surface_height_below  # noqa: E402

MESHES = os.path.join(HERE, "meshes")
BASE_XYZ = (1.70, 2.2, 0.0)          # where the PR2 stands (the avocado demo's verified reach over the counter)
LAYOUT = {                           # on the counter (x, y), robot at BASE_XYZ looking +x; all inside the right arm's
    "egg-1": (2.28, 2.33), "egg-2": (2.28, 2.27),   # measured top-grasp reach (x 2.25-2.40, y 1.95-2.35, z <= +0.18 m)
    "cup-1": (2.30, 2.12), "batch-bowl": (2.36, 1.98), "water-glass": (2.37, 2.31),
    "counter": (2.36, 2.21),         # the flat strike spot
    "shells": (2.42, 2.14),          # where the shell halves are put
}
SHELL_RGB = {"white": (0.93, 0.91, 0.86), "brown": (0.72, 0.50, 0.33)}


def _mesh_body(name, path, rgb, collision=True):
    shape = Mesh(origin=HomogeneousTransformationMatrix(np.eye(4)), filename=path)
    vis = ShapeCollection([shape])
    vis.dye_shapes(Color(*rgb, 1.0))
    coll = ShapeCollection([Mesh(origin=HomogeneousTransformationMatrix(np.eye(4)), filename=path)]) if collision \
        else ShapeCollection([])
    return Body(name=PrefixedName(name), visual=vis, collision=coll)


def _place(world, body, T):
    w = World()
    with w.modify_world():
        w.add_kinematic_structure_entity(body)
    world.merge_world_at_pose(w, HomogeneousTransformationMatrix(T, reference_frame=world.root))
    return world.get_body_by_name(body.name.name)


def egg_mesh(instance):
    """STL of the egg's shell (food/egg.py surface of revolution), long axis along z, lowest point at z = 0."""
    p = os.path.join(MESHES, f"egg_{instance}.stl")
    if not os.path.exists(p):
        from food.egg import Egg
        e = Egg(instance)
        m = e.surf(e.mer)
        m.apply_translation(-np.array([*m.bounds.mean(0)[:2], m.bounds[0][2]]))
        os.makedirs(MESHES, exist_ok=True)
        m.export(p)
    return p


def vessel_mesh(name, radius, height, wall=0.003):
    """An open cylinder vessel (cup, bowl, glass) with its base at z = 0."""
    p = os.path.join(MESHES, f"{name}.stl")
    if not os.path.exists(p):
        import trimesh
        wall_ = trimesh.creation.annulus(r_min=radius - wall, r_max=radius, height=height, sections=48)
        wall_.apply_translation([0, 0, height / 2])
        base = trimesh.creation.cylinder(radius=radius, height=wall, sections=48)
        base.apply_translation([0, 0, wall / 2])
        v = trimesh.util.concatenate([wall_, base])
        os.makedirs(MESHES, exist_ok=True)
        v.export(p)
    return p


def disc_mesh(name, radius, height):
    p = os.path.join(MESHES, f"{name}.stl")
    if not os.path.exists(p):
        import trimesh
        d = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
        d.apply_translation([0, 0, height / 2])
        os.makedirs(MESHES, exist_ok=True)
        d.export(p)
    return p


def half_shell_mesh(instance):
    p = os.path.join(MESHES, f"egg_{instance}_half.stl")
    if not os.path.exists(p):
        import trimesh
        m = trimesh.load(egg_mesh(instance))
        z_mid = (m.bounds[0][2] + m.bounds[1][2]) / 2
        half = m.submesh([np.nonzero(m.triangles_center[:, 2] < z_mid)[0]], append=True)   # the lower half, open top
        half.apply_translation([0, 0, -half.bounds[0][2]])
        half.export(p)
    return p


@dataclass
class EggKitchenDemonstration(RobotDemonstration):
    """The PR2 at the kitchen counter with the activity's eggs (instances by symbol) and vessels."""

    used_robot: type = PR2
    default_visualization_backend: VisualizationBackend = VisualizationBackend.CRAMERA
    collision_avoidance: bool = False
    event_segmentation: bool = False
    eggs: dict = field(default_factory=lambda: {"egg-1": 3, "egg-2": 5})
    counter_top: float = field(default=0.0, init=False)

    def build_simulated_world(self) -> World:
        from coraplex.testing import setup_world
        world = setup_world()
        self.populate_scene(world)
        self._pr2 = PR2.from_world(world)              # before the viewer attaches, so it knows the robot
        return world

    def is_scene_populated(self, world: World) -> bool:
        return any(b.name.name == "cup-1" for b in world.bodies)

    def populate_scene(self, world: World) -> None:
        top = surface_height_below(world, LAYOUT["cup-1"])
        self.counter_top = top
        for sym, inst in self.eggs.items():
            from food.egg import Egg
            colour = Egg(inst).p["colour"]
            x, y = LAYOUT[sym]
            _place(world, _mesh_body(sym, egg_mesh(inst), SHELL_RGB.get(colour, SHELL_RGB["white"])),
                   pose(x, y, top + 0.0005))
        _place(world, _mesh_body("cup-1", vessel_mesh("cup", 0.040, 0.060), (0.85, 0.85, 0.90)),
               pose(*LAYOUT["cup-1"], top + 0.0005))
        _place(world, _mesh_body("batch-bowl", vessel_mesh("bowl", 0.080, 0.070), (0.55, 0.70, 0.85)),
               pose(*LAYOUT["batch-bowl"], top + 0.0005))
        _place(world, _mesh_body("water-glass", vessel_mesh("glass", 0.040, 0.080), (0.70, 0.85, 0.95)),
               pose(*LAYOUT["water-glass"], top + 0.0005))

    def build_context(self, world: World) -> Context:
        robot = getattr(self, "_pr2", None) or PR2.from_world(world)
        context = Context(world=world, robot=robot, _debug=False, ros_node=None)
        context.evaluate_conditions = False             # as the avocado demo: kinematic, no condition checks
        return context

    def build_plan(self, context: Context):
        raise NotImplementedError("the plan comes from the activity's library (ccram)")
