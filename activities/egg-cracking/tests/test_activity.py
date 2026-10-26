"""The egg activity: language checks and runs on the HouseholdTwin egg emulator (seeds fixed: egg-1 = instance 3,
egg-2 = instance 5, emulator seeds 0 / 1). Run: ~/Documents/ConceptualCRAM/.venv/bin/python -m pytest tests -q"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/Documents/ConceptualCRAM"))
from ccram.activity import load  # noqa: E402
from ccram.session import make_run  # noqa: E402

ACT = load(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(plan="two-eggs-into-batch", scenario=""):
    r = make_run(ACT, plan=plan, scenario=scenario)
    r.run()
    ep = r.interp.episode
    return r, [n for n in ep if "schema" in n], [n["note"] for n in ep if "note" in n]


def test_library_lints_clean():
    _, rep = ACT.lint()
    assert not [i for i in rep.items if i[0] in ("ERROR", "WARN")], rep.items


def test_goal_coverage_complete():
    from ccram import goals as G
    cov = G.coverage(ACT.library().forms)
    assert cov["without_telos"] == [] and cov["unrefinable"] == [], cov


def test_two_eggs_reach_the_batch_and_a_broken_yolk_is_accepted_there():
    r, steps, notes = run(scenario="yolk-breaks")
    assert r.failure is None
    assert r.session.adapter.bowl == ["egg-1", "egg-2"]
    assert any(m.startswith("signal yolk-broken: expected here") for m in notes)
    opened = [n for n in steps if n["schema"] == "opening"]
    assert opened[0]["verdict"].startswith("goal achieved (accepted: yolk-broken")
    truth = r.session.adapter.truth()                    # hidden state: nothing of the shell reached the batch
    assert all(t["shards_in_vessel"] == 0 for t in truth.values())


def test_a_strike_predicted_to_fail_is_pre_empted_and_redecided_on_the_belief():
    r, steps, notes = run(scenario="timid-first-strike")
    assert r.failure is None and "handled no-crack (no-crack)" in notes
    strikes = [n for n in steps if n["schema"] == "cracking"]
    assert strikes[0]["verdict"] == "pre-empted: no-crack predicted" and strikes[0]["p_success"] < 0.05   # D4
    assert strikes[1]["bindings"]["speed"]["servant"] == "belief-decision" and strikes[1]["p_success"] > 0.8


def test_grip_force_and_strike_are_belief_decisions_with_p_success():
    r, steps, _ = run(plan="crack-egg-cleanly")
    grip = next(n for n in steps if n["schema"] == "picking-up")
    strike = next(n for n in steps if n["schema"] == "cracking")
    assert grip["bindings"]["force"]["servant"] == "grip-corridor" and grip["bindings"]["force"]["status"] == "anchored"
    assert strike["bindings"]["tool"]["servant"] == "belief-decision" and strike["warrant"] == "rational"
    assert "P(success)" in strike["preact"]


def test_repair_survives_a_belief_that_thinks_shells_are_weak():
    r, steps, notes = run(plan="crack-egg-cleanly", scenario="belief-shells-weak")
    assert r.failure is None
    assert any(m.startswith("cracking: probe first -- weighing") for m in notes)        # VOI-driven probe
    strikes = [n for n in steps if n["schema"] == "cracking"]
    assert len(strikes) >= 2 and float(strikes[-1]["bindings"]["speed"]["symbol"]) > float(strikes[0]["bindings"]["speed"]["symbol"])


def test_a_floating_egg_never_reaches_the_batch():
    r, steps, _ = run(scenario="stale-second-egg")
    assert r.failure is not None and r.failure.kind == "batch-spoiled"
    assert r.session.adapter.bowl == ["egg-1"]
    refused = [n for n in steps if n["verdict"].startswith("refused by the gate")]
    assert refused and refused[0]["schema"] == "emptying"
    assert ["stale", "egg-2"] in r.interp.atoms()       # derived by the rules from the float observation


def test_opening_into_the_batch_bowl_is_refused_by_the_gate():
    r, steps, _ = run(plan="open-into-the-batch-bowl")
    assert r.failure.kind == "batch-spoiled"
    assert steps[-1]["schema"] == "opening" and steps[-1]["verdict"].startswith("refused by the gate")
    assert r.session.adapter.emu["egg-1"].s.shell == "cracked"     # struck, never opened


def test_a_crack_in_the_hand_violates_the_pick_up_goal():
    r, steps, _ = run(scenario="hard-grip")
    assert r.failure.kind == "goal-not-achieved"
    grip = steps[-1]
    assert grip["schema"] == "picking-up" and "(prevent (shell-state egg-1 cracked)" in grip["violations"][0][0]
