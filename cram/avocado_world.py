"""The avocado in a CRAM world (semantic_digital_twin): its semantic annotation, carrying the robot's SDT node
(cram/avocado_sdt.py), and its pieces as bodies built from the twin's meshes (cram/export_meshes.py).

Fruit frame: x = the cut-plane normal of the twin (half_a lies on +x), y = the fruit's long axis (stem end +y),
z = x cross y. The fruit lies on its side with z up, so a vertical knife stroke along y halves it lengthwise round
the seed. Every piece is a root body (its skin, carrying the collision shape) with its other parts (flesh bands,
endocarp, seed coat) fixed to it as visual-only bodies.
"""

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Optional

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import Fruit
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Color, Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from avocado_sdt import AvocadoSDT

HERE = os.path.dirname(os.path.abspath(__file__))


SDT_NODES = {}
"""The robots' SDT nodes per avocado piece (root body name). Kept outside the annotation: the world records every
annotation it adds or removes as JSON, and the node (a particle belief) is not meant to be serialised with it."""


@dataclass(eq=False)
class Avocado(Fruit):
    """
    An avocado (or a piece of one). The robot's SDT node of it (belief, state, competence answers) is attached once
    it is detected.
    """

    @property
    def sdt(self) -> Optional[AvocadoSDT]:
        """The robot's SDT node of this avocado, None until the avocado has been detected."""
        return SDT_NODES.get(self.root.name.name)

    @sdt.setter
    def sdt(self, node: Optional[AvocadoSDT]) -> None:
        SDT_NODES[self.root.name.name] = node


@dataclass
class AvocadoMeshes:
    """
    The exported meshes of one avocado instance and the transform from the twin's asset frame to the fruit frame.
    """

    instance: str = "avocado_4"

    def __post_init__(self):
        self.root = os.path.join(HERE, "meshes", self.instance)
        self.manifest = json.load(open(os.path.join(self.root, "manifest.json")))
        n = np.asarray(self.manifest["cut_normal"], float)
        ez = np.array([0.0, 0.0, 1.0])
        R = np.vstack([n, ez, np.cross(n, ez)])          # rows: fruit x, y, z in asset coordinates
        self.fruit_T_asset = np.eye(4)
        self.fruit_T_asset[:3, :3] = R

    def vertices(self, piece):
        """All mesh vertices of a piece in the asset frame (cached)."""
        cache = self.__dict__.setdefault("_verts", {})
        if piece not in cache:
            import trimesh
            cache[piece] = np.vstack([trimesh.load(os.path.join(self.root, p["file"])).vertices
                                      for p in self.manifest["pieces"][piece]["parts"]])
        return cache[piece]

    def frame_of(self, piece):
        """Asset -> piece frame: the fruit frame for avocado pieces, identity for things in their own frame (lemon)."""
        return np.eye(4) if self.manifest["pieces"][piece].get("own_frame") else self.fruit_T_asset

    def bounds(self, piece, T=None):
        """Axis-aligned bounds of a piece's vertices in the fruit frame (or after the further transform T)."""
        M = self.frame_of(piece) if T is None else T @ self.frame_of(piece)
        v = self.vertices(piece) @ M[:3, :3].T + M[:3, 3]
        return v.min(0), v.max(0)


def _body(name, path, rgb, collision, origin):
    shape = Mesh(origin=HomogeneousTransformationMatrix(origin), filename=path)
    visual = ShapeCollection([shape])
    visual.dye_shapes(Color(*rgb, 1.0))
    coll = ShapeCollection([Mesh(origin=HomogeneousTransformationMatrix(origin), filename=path)]) \
        if collision else ShapeCollection([])
    return Body(name=PrefixedName(name), visual=visual, collision=coll)


def spawn_piece(world, meshes, piece, world_T_piece, tag, local=None):
    """
    Build a piece of the avocado (whole, half_a, half_b, half_a_pitted, seed) as bodies and put it at
    ``world_T_piece`` (4x4, piece frame in the world root frame).

    :param local: transform from the fruit frame to the piece frame (e.g. a half turned cut face up); identity if None.
    :return: The piece's root body.
    """
    origin = (np.eye(4) if local is None else local) @ meshes.frame_of(piece)
    parts = meshes.manifest["pieces"][piece]["parts"]
    bodies = [_body(f"{tag}_{p['part']}", os.path.join(meshes.root, p["file"]), p["rgb"], i == 0, origin)
              for i, p in enumerate(parts)]
    piece_world = World()
    with piece_world.modify_world():
        piece_world.add_kinematic_structure_entity(bodies[0])
        for b in bodies[1:]:
            piece_world.add_kinematic_structure_entity(b)
            piece_world.add_connection(FixedConnection(parent=bodies[0], child=b))
    world.merge_world_at_pose(piece_world, HomogeneousTransformationMatrix(world_T_piece, reference_frame=world.root))
    return world.get_body_by_name(f"{tag}_{parts[0]['part']}")


def remove_piece(world, root_body):
    """Take a piece (its root body and the parts fixed to it) out of the world."""
    with world.modify_world():
        world.remove_branch_from_world(root_body)


def surface_height_below(world, xy, below=1.3):
    """Height of the highest horizontal surface under the point xy (top of the apartment body found there)."""
    best = 0.0
    for body in world.bodies:
        if not body.collision.shapes or body.name.name.startswith(("avocado", "r_", "l_", "base", "torso", "head")):
            continue
        bb = body.collision.as_bounding_box_collection_in_frame(world.root).bounding_box()
        if bb.min_x <= xy[0] <= bb.max_x and bb.min_y <= xy[1] <= bb.max_y and bb.max_z < below:
            best = max(best, bb.max_z)
    return best


def pose(x, y, z, yaw=0.0, R=None):
    T = np.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ (np.eye(3) if R is None else R)
    T[:3, 3] = (x, y, z)
    return T


def resting(meshes, piece, x, y, top, yaw=0.0, R=None):
    """world_T_piece that puts the piece's lowest point onto the surface at height ``top``."""
    T = pose(x, y, 0.0, yaw, R)
    lo, hi = meshes.bounds(piece, T)
    T[0, 3] += x - (lo[0] + hi[0]) / 2
    T[1, 3] += y - (lo[1] + hi[1]) / 2
    T[2, 3] = top - lo[2] + 0.0005
    return T


CUT_FACE_UP = {"half_a": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float),    # half_a's cut face looks along
               "half_b": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float)}    # -x, half_b's along +x: -> up
CUT_FACE_UP["half_a_pitted"] = CUT_FACE_UP["half_a"]

FRUIT_CUT_HORIZONTAL = np.column_stack([[0, 0, -1], [0, 1, 0], [1, 0, 0]]).astype(float)
"""World orientation of the fruit frame with the cut plane horizontal: fruit x (cut normal, half_a side) points
down, so half_a (the half that keeps the seed) lies below the cut; the long axis (fruit y) along world y."""


def spawn_board(world, xy, counter_top, size=(0.34, 0.24, 0.04), name="cutting_board"):
    """A wooden cutting board on the counter (box); returns its top height. A thick board lifts the fruit far enough
    above the counter for a knife held flat in the fruit's cut plane (the gripper and forearm stay above the counter)."""
    from semantic_digital_twin.world_description.geometry import Box, Scale
    shape = Box(scale=Scale(*size), color=Color(0.62, 0.45, 0.28, 1.0))
    body = Body(name=PrefixedName(name), visual=ShapeCollection([shape]),
                collision=ShapeCollection([Box(scale=Scale(*size))]))
    board = World()
    with board.modify_world():
        board.add_kinematic_structure_entity(body)
    world.merge_world_at_pose(board, HomogeneousTransformationMatrix(pose(xy[0], xy[1], counter_top + size[2] / 2),
                                                                    reference_frame=world.root))
    return counter_top + size[2]
