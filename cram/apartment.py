"""The apartment the HouseholdTwin's CRAM cases plan in, with the PR2, milk and cereal as CRAM's
coraplex.testing.setup_world places them.

Default since 2026-11-18 (user decision, AssistanceVRB labs): the iai_apartment model of ~/ros_ws. It is the model
the AssistanceVRB knowledge base was measured from and the fetch cases plan in. It has the same 113 bodies at the
same poses as CRAM's test apartment, except side_A_left (0.6 m apart), plus the tall cabinet (cabinet12 and its two
doors).

HOUSEHOLD_APARTMENT=cram-test gives CRAM's own test apartment (coraplex.testing.setup_world) as before.
"""
from __future__ import annotations

import os
from pathlib import Path

IAI_APARTMENT = Path.home() / "ros_ws/install/share/iai_apartment/urdf/apartment.urdf"


def apartment_name() -> str:
    """The apartment chosen by HOUSEHOLD_APARTMENT: 'iai' (default) or 'cram-test'."""
    name = os.environ.get("HOUSEHOLD_APARTMENT", "iai")
    if name not in ("iai", "cram-test"):
        raise SystemExit(f"HOUSEHOLD_APARTMENT={name!r}: use 'iai' or 'cram-test'")
    return name


def apartment_with_pr2():
    """The world: the chosen apartment, the PR2 on an omni drive at (1.5, 2.5), milk and cereal on the counter and the
    milk's semantic annotation, as in coraplex.testing.setup_world (whose recipe this follows for the iai model)."""
    import coraplex.testing as testing
    if apartment_name() == "cram-test":
        return testing.setup_world()
    from semantic_digital_twin.adapters.mesh import STLParser
    from semantic_digital_twin.adapters.urdf import URDFParser
    from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
    from semantic_digital_twin.world_description.connections import OmniDrive
    try:
        from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
    except ImportError:
        from semantic_digital_twin.spatial_types.spatial_types import HomogeneousTransformationMatrix
    resources = Path(testing.__file__).resolve().parent.parent.parent / "resources"
    pr2 = URDFParser.from_file("package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro").create_world()
    world = URDFParser.from_file(str(IAI_APARTMENT)).create_world()
    world.merge_world(STLParser(str(resources / "objects" / "milk.stl")).create_world())
    world.merge_world(STLParser(str(resources / "objects" / "breakfast_cereal.stl")).create_world())
    with world.modify_world():
        drive = OmniDrive.create_with_dofs(parent=world.root, child=pr2.get_body_by_name("base_footprint"), world=world)
        world.merge_world(pr2, drive)
        drive.origin = HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2.5, 0, reference_frame=world.root)
    world.get_body_by_name("milk.stl").parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        2.37, 2, 1.0345, reference_frame=world.root)
    world.get_body_by_name("breakfast_cereal.stl").parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        2.37, 1.8, 1.0487, reference_frame=world.root)
    with world.modify_world():
        world.add_semantic_annotation(Milk(root=world.get_body_by_name("milk.stl")))
    return world
