"""Put things where they belong (M3): lint, coverage, and the four guard behaviours on the kitchen-state emulator.
Run: ~/Documents/ConceptualCRAM/.venv/bin/python -m pytest tests -q"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/Documents/ConceptualCRAM"))
from ccram.activity import load  # noqa: E402
from ccram.session import make_run  # noqa: E402

ACT = load(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(plan="tidy-after-breakfast", scenarios=()):
    r = make_run(ACT, plan=None)
    r.run()
    s = r.session
    for sc in scenarios:
        ACT.scenarios()[sc](s)
    if plan:
        s.run_plan(plan)
    return s


def at(s, o):
    return s.adapter.truth[o]["at"]


def puts(s):
    return [n["bindings"]["object"]["symbol"] for n in s.interp.episode if n.get("schema") == "putting-away"
            and n["verdict"] == "goal achieved"]


def test_library_lints_clean_and_goals_are_covered():
    _, rep = ACT.lint()
    assert not [i for i in rep.items if i[0] in ("ERROR", "WARN", "META")], rep.items
    from ccram import goals as G
    cov = G.coverage(ACT.library().forms)
    assert cov["without_telos"] == [] and cov["unrefinable"] == []


def test_stray_restored_in_place_left_no_home_asked():
    s = run()
    assert s.failure is None
    assert {at(s, o) for o in ("cup-1", "plate-1", "knife-1")} == {"dishwasher"} and at(s, "milk-1") == "fridge"
    assert at(s, "sugar-1") == "counter" and "sugar-1" not in puts(s)            # no over-tidying
    assert at(s, "jar-1") == "counter" and ("resident", "jar-1") in s.adapter.asked   # no guessing
    put = next(n for n in s.interp.episode if n.get("schema") == "putting-away")
    d = put["bindings"]["destination"]
    assert d["servant"] == "belonging-store" and "adequate" in d["note"]          # admissible != adequate (T3)


def test_resident_request_complied_with_a_debt_that_is_discharged_later():
    s = run(scenarios=("leave-the-coffee-out",))
    assert at(s, "coffee-1") == "counter"
    assert ["obligation-open", "coffee-1", "cupboard", "r-1"] in s.interp.atoms()
    ACT.scenarios()["coffee-request-ends"](s)
    assert ["obligation-due", "coffee-1", "cupboard", "r-1"] in s.interp.atoms()
    s.run_plan("tidy-after-breakfast")
    assert at(s, "coffee-1") == "cupboard"
    assert ["obligation-discharged", "coffee-1", "r-1"] in s.interp.atoms()


def test_guest_request_conflict_abstains_and_asks():
    s = run(scenarios=("guest-asks-to-leave-the-milk-out",))
    assert at(s, "milk-1") == "counter" and any(o == "milk-1" for _, o in s.adapter.asked)


def test_full_place_is_asked_about_not_replaced_by_an_admissible_one():
    s = run(scenarios=("dishwasher-full",))
    assert {at(s, o) for o in ("cup-1", "plate-1", "knife-1")} == {"counter"}     # not moved "somewhere admissible"
    accepted = [n for n in s.interp.episode if n.get("schema") == "putting-away" and "accepted: place-not-admissible"
                in n["verdict"]]
    assert len(accepted) == 3


def test_a_told_answer_becomes_a_norm():
    s = run(scenarios=("jar-answer",))
    s.run_plan("tidy-after-breakfast")
    assert at(s, "jar-1") == "cupboard"
    put = [n for n in s.interp.episode if n.get("schema") == "putting-away" and n["bindings"]["object"]["symbol"] == "jar-1"]
    assert put[-1]["bindings"]["destination"]["status"] == "recorded"


def test_the_gate_refuses_a_dirty_cup_into_the_cupboard():
    s = run()
    s.adapter.truth["cup-1"]["at"] = s.adapter.known["cup-1"]["at"] = "counter"
    import pytest
    from ccram.kernel import KernelFailure
    with pytest.raises(KernelFailure) as e:
        s.interp.execute("(perform (an action (type putting-away) (object (the household-object (tag cup-1))) "
                         "(destination cupboard)))")
    assert e.value.kind == "prohibited-placement" and at(s, "cup-1") == "counter"
