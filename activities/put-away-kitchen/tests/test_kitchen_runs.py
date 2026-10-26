"""Put-away on the CRAM kitchen twin (PR2, kinematic). Needs the CRAM environment:
    source /opt/ros/jazzy/setup.bash; source ~/ros_ws/install/setup.bash
    CORAPLEX_VISUALIZATION=none ~/.virtualenvs/cram-env/bin/python -m pytest -p no:launch_testing -p no:launch_ros \
        -p no:launch_pytest -q tests/test_kitchen_runs.py"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/Documents/ConceptualCRAM"))
pytest.importorskip("coraplex")
os.environ.setdefault("CORAPLEX_VISUALIZATION", "none")
from ccram.activity import load  # noqa: E402
from ccram.session import demonstration  # noqa: E402

ACT = load(HERE)


def test_the_pr2_tidies_the_counter_and_the_world_shows_it():
    d = demonstration(ACT, "tidy-after-breakfast", "leave-the-coffee-out")
    d.run()
    s = d.session
    assert d.failure is None
    w = s.world

    def xy(n):
        return w.get_body_by_name(n).global_pose.to_np()[:2, 3]
    for o, p in (("cup-1", "dishwasher"), ("plate-1", "dishwasher"), ("knife-1", "dishwasher"), ("milk-1", "fridge")):
        assert np.abs(xy(o) - xy(f"place-{p}")).max() < 0.06, (o, p)            # on its (stand-in) place's tray
    for o in ("sugar-1", "coffee-1", "jar-1"):                                    # in place / licensed / asked about
        assert np.abs(xy(o) - np.array(__import__("putaway_kitchen").OBJECTS[o][:2])).max() < 0.005, o
    assert ["obligation-open", "coffee-1", "cupboard", "r-1"] in s.interp.atoms()
