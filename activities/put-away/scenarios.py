"""Scenarios of the put-away activity (ccram run --scenario NAME): requests, people, full places."""


def scenario_leave_the_coffee_out(run):
    """The resident asks: "leave the coffee out" (episodic): comply with a debt -- the coffee is licensed out and a
    restoration obligation to the cupboard is opened (RQ4)."""
    run.adapter.requests.append(("r-1", "resident", "coffee-1", "counter", "episodic"))


def scenario_guest_asks_to_leave_the_milk_out(run):
    """A guest asks to leave the milk out: a guest may not overrule the household's norm -- abstain and ask (RQ3)."""
    run.adapter.requests.append(("r-2", "guest", "milk-1", "counter", "episodic"))
    run.adapter.people.append("guest")


def scenario_dishwasher_full(run):
    """The dishwasher is full: its PL1 admissibility fails; the counter is admissible but not adequate -- ask (T3)."""
    run.adapter.places["dishwasher"]["capacity"] = 0


def scenario_jar_answer(run):
    """The resident will answer the question about the unknown jar: it belongs in the cupboard (a told norm)."""
    run.adapter.answers["jar-1"] = "cupboard"


def scenario_coffee_request_ends(run):
    """The resident is done with the coffee: request r-1's scope ends, the restoration obligation comes due."""
    run.adapter.ended.add("r-1")
