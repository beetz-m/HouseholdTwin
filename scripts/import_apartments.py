"""Import the EASE (IAI) and GARMI apartments and the GARMI robot from the CRAM viz scene bundles (read in place from
~/ResearchSpace, which stays read-only) into USD with Isaac Sim's URDFImporter.

  assets/apartments/ease/apartment/...     iai_apartment (118 links, doors and drawers as joints)
  assets/apartments/garmi/apartment/...    GARMI (TUM MIRMI) apartment (79 links)
  assets/apartments/garmi_robot/...        GARMI: two FR3 arms, lift, head, mobile base
Run: OMNI_KIT_ACCEPT_EULA=YES ~/Documents/LabAutomationTwin/env_isaacsim/bin/python scripts/import_apartments.py
"""

import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES = os.path.expanduser("~/ResearchSpace/foundation/software/cram/cram_viz/scenes")
JOBS = [("ease/apartment", os.path.join(SCENES, "garmi_ease_apartment", "apartment.urdf"), True, "Convex Decomposition"),
        ("garmi/apartment", os.path.join(SCENES, "garmi_garmiapartment", "scene-bodies.urdf"), True,
         "Convex Decomposition"),
        ("garmi_robot", os.path.join(ROOT, "assets", "apartments", "garmi_robot", "garmi_clean.urdf"), False,
         "Convex Hull")]   # gazebo / ros2_control / transmission blocks stripped (importer cannot store them)
ONLY = [a for a in sys.argv[1:]]
JOBS = [j for j in JOBS if not ONLY or j[0] in ONLY]

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402

for ext in ("isaacsim.asset.importer.urdf", "isaacsim.robot.schema"):
    app_utils.enable_extension(ext)
app.update()
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig  # noqa: E402

ok = True
for name, urdf, fixed, coll in JOBS:
    out_dir = os.path.join(ROOT, "assets", "apartments", name)
    os.makedirs(out_dir, exist_ok=True)
    cfg = URDFImporterConfig(urdf_path=urdf, usd_path=out_dir, merge_fixed_joints=False, fix_base=fixed,
                             collision_from_visuals=False, collision_type=coll, joint_drive_type="force",
                             joint_target_type="position", override_joint_stiffness=400.0 if fixed else 2000.0,
                             override_joint_damping=40.0 if fixed else 150.0,
                             robot_type="Default" if fixed else "Manipulator")
    out = URDFImporter(cfg).import_urdf()
    print(f"[import] {name}: {out}", flush=True)
    ok &= bool(out)
sys.stdout.flush()
os._exit(0 if ok else 1)
