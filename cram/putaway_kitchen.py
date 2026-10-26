"""The put-away kitchen: the PR2 in the apartment (coraplex.testing.setup_world) after breakfast -- the CRAM twin of
the put-away activity (activities/put-away-kitchen).

The apartment has no fridge, dishwasher or bin, so the belonging places are STAND-INS: labelled trays on the island
counter (place-fridge, place-dishwasher, place-cupboard, place-bin), within the PR2's measured reach from a second
base pose. The breakfast things stand on another stretch of the island counter (the "counter" place). The apartment's
own milk and cereal stood on the trays' stretch and are removed. Objects are primitives (trimesh), cached in
cram/meshes/pa_*.stl. Reach (top grasp, right arm, measured 21 Oct): base (1.70, 2.75) reaches x 2.30-2.38,
y 2.52-2.82; base (1.70, 2.15) reaches x 2.33-2.38, y 1.92-2.28; heights counter + 0.03 .. + 0.20.
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
from semantic_digital_twin.robots.pr2 import PR2  # noqa: E402
from semantic_digital_twin.world import World  # noqa: E402

from avocado_world import pose, surface_height_below  # noqa: E402
from egg_kitchen import MESHES, _mesh_body, _place  # noqa: E402

BASE_OBJECTS = (1.70, 2.75)          # the breakfast stretch of the island counter
BASE_PLACES = (1.70, 2.15)           # the stand-in places
OBJECTS = {   # symbol: (x, y, shape, size, rgb) -- the after-breakfast counter of the put-away activity
    "cup-1": (2.30, 2.52, "cup", (0.035, 0.08), (0.95, 0.95, 0.92)),
    "milk-1": (2.38, 2.52, "box", (0.06, 0.06, 0.16), (0.92, 0.92, 0.98)),
    "plate-1": (2.30, 2.62, "disc", (0.055, 0.012), (0.97, 0.97, 0.95)),
    "knife-1": (2.38, 2.62, "box", (0.02, 0.16, 0.012), (0.75, 0.76, 0.78)),
    "jar-1": (2.30, 2.72, "cyl", (0.035, 0.09), (0.55, 0.35, 0.60)),
    "coffee-1": (2.38, 2.72, "cyl", (0.035, 0.11), (0.35, 0.22, 0.12)),
    "sugar-1": (2.34, 2.82, "cyl", (0.03, 0.08), (0.95, 0.95, 0.95)),
}
PLACES = {    # stand-in trays: place symbol -> (x, y, rgb)
    "fridge": (2.33, 1.92, (0.70, 0.85, 0.98)), "dishwasher": (2.33, 2.04, (0.55, 0.60, 0.65)),
    "cupboard": (2.33, 2.16, (0.80, 0.62, 0.40)), "bin": (2.33, 2.28, (0.25, 0.25, 0.25)),
}
TRAY = (0.11, 0.11, 0.006)


def _mesh(name, make):
    p = os.path.join(MESHES, f"pa_{name}.stl")
    if not os.path.exists(p):
        os.makedirs(MESHES, exist_ok=True)
        m = make()
        m.apply_translation([0, 0, -m.bounds[0][2]])           # base at z = 0
        m.export(p)
    return p


def object_mesh(sym):
    import trimesh
    _, _, shape, size, _ = OBJECTS[sym]
    if shape == "cup":
        def make():
            wall = trimesh.creation.annulus(r_min=size[0] - 0.003, r_max=size[0], height=size[1], sections=40)
            wall.apply_translation([0, 0, size[1] / 2])
            return trimesh.util.concatenate([wall, trimesh.creation.cylinder(radius=size[0], height=0.004, sections=40)])
        return _mesh("cup", make)
    if shape == "box":
        return _mesh(sym, lambda: trimesh.creation.box(extents=size))
    return _mesh(sym, lambda: trimesh.creation.cylinder(radius=size[0], height=size[1], sections=40))


def height(sym):
    shape, size = OBJECTS[sym][2], OBJECTS[sym][3]
    return size[2] if shape == "box" else size[1]


@dataclass
class PutAwayKitchenDemonstration(RobotDemonstration):
    used_robot: type = PR2
    default_visualization_backend: VisualizationBackend = VisualizationBackend.CRAMERA
    collision_avoidance: bool = False
    event_segmentation: bool = False
    counter_top: float = field(default=0.0, init=False)

    def build_simulated_world(self) -> World:
        from coraplex.testing import setup_world
        world = setup_world()
        self.populate_scene(world)
        self._pr2 = PR2.from_world(world)
        return world

    def is_scene_populated(self, world: World) -> bool:
        return any(b.name.name == "place-fridge" for b in world.bodies)

    def populate_scene(self, world: World) -> None:
        for n in ("milk.stl", "breakfast_cereal.stl"):          # the apartment's own breakfast things stand where
            b = next((b for b in world.bodies if b.name.name == n), None)   # the stand-in places go
            if b is not None:
                with world.modify_world():
                    world.remove_branch_from_world(b)
        top = surface_height_below(world, OBJECTS["cup-1"][:2])
        self.counter_top = top
        import trimesh
        for p, (x, y, rgb) in PLACES.items():
            _place(world, _mesh_body(f"place-{p}", _mesh(f"tray_{p}", lambda: trimesh.creation.box(extents=TRAY)), rgb),
                   pose(x, y, top + 0.0005))
        for sym, (x, y, _, _, rgb) in OBJECTS.items():
            _place(world, _mesh_body(sym, object_mesh(sym), rgb), pose(x, y, top + 0.0005))

    def build_context(self, world: World) -> Context:
        robot = getattr(self, "_pr2", None) or PR2.from_world(world)
        context = Context(world=world, robot=robot, _debug=False, ros_node=None)
        context.evaluate_conditions = False
        return context

    def build_plan(self, context: Context):
        raise NotImplementedError("the plan comes from the activity's library (ccram)")
