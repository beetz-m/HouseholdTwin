"""Scenarios of the egg activity (ccram run --scenario NAME). run.adapter is the EggAdapter (emulator twin)."""


def scenario_timid_first_strike(run):
    """The first strike is too slow (0.12 m/s): no crack -> the handler retries at the study's speed +15 %."""
    run.adapter.speed_override.append(0.12)


def scenario_stale_second_egg(run):
    """egg-2 has been at room temperature for 12 weeks: it floats (emulator: SG 0.989); its contents must not reach
    the batch (rules: floats -> stale -> unfit cup; gate of emptying)."""
    from food.egg_emulator import EggEmulator
    import os
    a = run.adapter
    a.emu["egg-2"] = EggEmulator(5, os.path.join(a.out, "egg-2"), seed=1, storage="room", age_days=84, write_usd=False)


def scenario_hard_grip(run):
    """A grip far above the corridor (40 N per pad): the shell cracks in the hand (the trajectory verdict catches it)."""
    run.adapter.force_override.append(40.0)


def scenario_belief_shells_weak(run):
    """Corrupted belief: the robot believes shells break at 0.6 x their true force -> it strikes too softly; the
    no-crack observations correct the belief and the retry decides a faster strike (repair survival)."""
    run.adapter.corrupt_belief({"F_break_equatorial": 0.6, "F_break_axial": 0.6})


def scenario_belief_shells_strong(run):
    """Corrupted belief: the robot believes shells break at 1.6 x their true force -> it strikes too hard; a shatter
    is unrecoverable (the dangerous direction of misspecification)."""
    run.adapter.corrupt_belief({"F_break_equatorial": 1.6, "F_break_axial": 1.6})


def scenario_yolk_breaks(run):
    """Disturbance: egg-1's yolk breaks on opening. In add-egg-to-batch the handler accepts it (a batter)."""
    run.adapter.break_yolk.add("egg-1")
