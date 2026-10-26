"""The egg activity on the CRAM kitchen twin (PR2, kinematic). Needs the CRAM environment:
    source /opt/ros/jazzy/setup.bash; source ~/ros_ws/install/setup.bash
    CORAPLEX_VISUALIZATION=none ~/.virtualenvs/cram-env/bin/python -m pytest -p no:launch_testing -p no:launch_ros \
        -p no:launch_pytest -q tests/test_kitchen_runs.py                                            (~3 min)"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/Documents/ConceptualCRAM"))
pytest.importorskip("coraplex")
os.environ.setdefault("CORAPLEX_VISUALIZATION", "none")
from ccram.activity import load  # noqa: E402
from ccram.session import demonstration  # noqa: E402

ACT = load(HERE)


def run(plan, scenario=""):
    d = demonstration(ACT, plan, scenario)
    d.run()
    return d, d.session


def test_two_eggs_into_the_batch_with_the_pr2():
    d, s = run("two-eggs-into-batch")
    assert d.failure is None and s.adapter.bowl == ["egg-1", "egg-2"]
    names = {b.name.name for b in s.world.bodies}
    assert {"egg-1-shell-a", "egg-2-shell-b", "contents-egg-1", "contents-egg-2"} <= names     # the world shows it
    assert "egg-1" not in names and "egg-2" not in names
    bowl_xy = s.world.get_body_by_name("batch-bowl").global_pose.to_np()[:2, 3]
    for egg in ("egg-1", "egg-2"):
        c = s.world.get_body_by_name(f"contents-{egg}").global_pose.to_np()[:2, 3]
        assert abs(c - bowl_xy).max() < 0.02                                                     # tipped into the bowl
    strikes = [n for n in s.interp.episode if n.get("schema") == "cracking"]
    assert all(n["warrant"] == "rational" and n["p_success"] > 0.8 for n in strikes)


def test_the_gate_stops_the_pr2_before_opening_into_the_batch_bowl():
    d, s = run("open-into-the-batch-bowl")
    assert d.failure.kind == "batch-spoiled"
    assert "egg-1" in {b.name.name for b in s.world.bodies}                                    # struck, never opened
