"""Task programs of the avocado microtheory: Task A (assess ripeness by touch, intact) and Task B (halve and pit
within three days), each with the belief-guided policy, fixed baselines and ablations that remove one represented
distinction from the robot's model. Policies read only the plant's observations (Omega) and the robot's belief.

Task A -- the squeeze corridor. A squeeze is informative only above a force (sensor noise 0.1 x 4 N / F relative,
belief measurement model) and bruises above 0.6 x the penetrometer force of the fruit's stage (register
bruise_force_ratio, hia2018). The corridor is [F_inform, F_bruise(1 %)] under the belief; for ripe fruit it is empty.

Task B -- the ripeness band. A clean twist needs stage >= ~3.2 (register clean_twist_stage, p_clean), good eating
needs stage >= ~3.4 (mashable) and the fruit must not be spoiled: the central prediction is a band in time, not a
force. The belief policy plans the day of the cut by receding horizon over the lifetime model, re-sensing (gentle
squeeze) every day; execution repairs a stalled cut and a stuck twist, and lifts seed-coat sheets while they are seen.
"""

import numpy as np

from food import avocado_belief as AB
from food import avocado_process as PRC
from food import params as PR
from food.avocado_lifetime import DECLINE, Q10, mean_of

from .avocado_bws import (READY_STAGE, G_A, G_B, observe_photo, observe_squeeze, p_bruise, p_stage, prior)

F_INFORM = 0.1 * 4.0 / 0.12          # rel. sensor noise <= 12 % (belief measurement model)
PROBE_FORCES = (2.0, 1.5, 1.0)
WINDOW_H = mean_of("ripe_window_counter")
SPOIL_D = mean_of("spoil_days_past_window")
CLEAN_MIN = 0.9


# %% robot-side lifetime prediction over the particles (vectorised; the robot knows the register means only)
def lifetime_predict(b, hr, hours, storage="counter", rate=None):
    """Particles' stage, hours at stage 5, eating quality and spoilage after ``hours`` (robot model)."""
    f = Q10 ** ((( 20.0 if storage == "counter" else 4.0) - 20.0) / 10.0)
    s = b.particles["ripeness_stage"]
    r = b.particles["ripening_rate"] if rate is None else np.full_like(s, rate)
    s2 = np.minimum(5.0, s + r * f * hours / 24.0)
    t_ripe = np.maximum(0.0, (5.0 - s) / np.maximum(r * f, 1e-9)) * 24.0
    hr2 = hr + np.maximum(0.0, hours - t_ripe) * f
    past = np.maximum(0.0, (hr2 - WINDOW_H) / 24.0)
    spoiled = past >= SPOIL_D
    q = np.clip((s2 - 2.5) / 1.5, 0, 1) * np.exp(-DECLINE * past) * (~spoiled)
    return s2, hr2, q, spoiled


def p_clean(stage):
    s0 = PR.AVOCADO["clean_twist_stage"][1]
    return 1.0 / (1.0 + np.exp(-(np.asarray(stage) - s0) / 0.15))


# %% Task A
def task_A(plant, policy, rng, n=1000, ablate=None):
    """Assess ripeness by touch without bruising. Returns (success, decided_ready, squeezes)."""
    b = prior(n)
    observe_photo(b, plant.photo(), rng)
    forces = []
    if policy == "closed loop":
        for F0 in PROBE_FORCES:                                  # a gentle probe the belief says is safe
            if p_bruise(b, F0) <= 0.01:
                observe_squeeze(b, plant.squeeze(F0), F0, rng)
                forces.append(F0)
                break
        if p_bruise(b, F_INFORM) <= 0.01:                        # the corridor after the probe
            observe_squeeze(b, plant.squeeze(F_INFORM), F_INFORM, rng)
            forces.append(F_INFORM)
    elif policy == "open-loop prior plan":
        Fs = [F for F in (F_INFORM, 3.0, 2.5, 2.0, 1.5, 1.0) if p_bruise(b, F) <= 0.01]
        if Fs:
            observe_squeeze(b, plant.squeeze(Fs[0]), Fs[0], rng)
            forces.append(Fs[0])
    elif policy == "firm thumb 10 N":
        observe_squeeze(b, plant.squeeze(10.0), 10.0, rng)
        forces.append(10.0)
    elif policy == "fixed 4 N":
        observe_squeeze(b, plant.squeeze(4.0), 4.0, rng)
        forces.append(4.0)
    elif policy == "bruise not owned by ripeness":                # ablation: one bruise threshold for all fruit
        thr = PR.AVOCADO["bruise_force_ratio"][1] * float(np.median(AB.derived(prior(n).particles)["penetrometer_N"]))
        F = F_INFORM if F_INFORM < thr else 1.5
        observe_squeeze(b, plant.squeeze(F), F, rng)
        forces.append(F)
    elif policy != "photo only":
        raise KeyError(policy)
    ready = p_stage(b, READY_STAGE) > 0.5
    return G_A(plant, ready), ready, forces


# %% Task B
def plan_cut_day(b, hr, days_left, rate=None, band=True):
    """Receding horizon over the lifetime: the wait (0..days_left days) with the highest P(success) under the belief
    (clean twist x edible x not spoiled); ``band=False`` is the ablation 'ripeness as ready / not ready'."""
    ps = []
    for w in range(days_left + 1):
        s2, _, q, sp = lifetime_predict(b, hr, 24.0 * w, rate=rate)
        if band:
            p = float(np.sum(b.weights * p_clean(s2) * (q >= 0.6) * (~sp)))
        else:
            p = float(np.sum(b.weights * (s2 >= READY_STAGE)))
        ps.append(p)
    best_p = max(ps)
    best = next(w for w, p in enumerate(ps) if p >= best_p - 0.02)     # the earliest day close to the best
    return best, ps[best], ps


def execute_halving(plant, b, rng, removal="spoon", repair=True, peel=True, hr=None):
    """Cut round the seed, twist apart, remove the seed, lift seed-coat sheets; repairs on a stalled cut / stuck
    twist. The belief takes every measured outcome (food/avocado_tasks.observations)."""
    from food import avocado_tasks as T
    out = plant.cut()
    for o in T.observations(out):
        b.observe(o, rng)
    if repair and out.outcome == "partly_scored":
        out = plant.cut()
    tw = plant.twist()
    for o in T.observations(tw):
        b.observe(o, rng)
    if repair and not tw.ok:                                     # re-cut the bridge, twist again
        plant.cut()
        tw = plant.twist()
    if not tw.ok:
        return
    rm = plant.remove_seed(removal)
    for o in T.observations(rm):
        b.observe(o, rng)
    passes = 0
    while peel and rm.ok and plant.remnant > 0.05 and passes < 2:
        plant.peel(1)
        passes += 1


def task_B(plant, policy, rng, n=1000, horizon_days=5):
    """Halve and pit within ``horizon_days``. Returns (success, cut_day, label)."""
    b = prior(n)
    observe_photo(b, plant.photo(), rng)
    hr = np.zeros(b.n)
    sense = policy in ("belief band", "knife twist", "no testa side", "no repair", "ready / not ready",
                       "population ripening rate")
    rate = float(np.mean(prior(n).particles["ripening_rate"])) if policy == "population ripening rate" else None
    removal = "knife_twist" if policy == "knife twist" else "spoon"
    day = 0
    predicted = None
    if policy == "cut now":
        pass
    elif policy.startswith("wait "):
        day = int(policy.split()[1])
        plant.wait(24.0 * day)
        b.predict(24.0 * day, rng)
    else:
        while day < horizon_days:
            if sense:                                            # gentle daily squeeze, only where it is safe
                for F in (F_INFORM, 2.0, 1.5, 1.0):
                    if p_bruise(b, F) <= 0.01:
                        observe_squeeze(b, plant.squeeze(F), F, rng)
                        break
            w, pw, _ = plan_cut_day(b, hr, horizon_days - day, rate=rate, band=policy != "ready / not ready")
            if w == 0:
                predicted = pw
                break
            plant.wait(24.0)
            _, hr, _, _ = lifetime_predict(b, hr, 24.0, rate=rate)
            b.predict(24.0, rng)
            day += 1
    execute_halving(plant, b, rng, removal=removal, repair=policy != "no repair", peel=policy != "no testa side")
    from .avocado_bws import failure_label
    task_B.predicted = predicted
    return G_B(plant), day, failure_label(plant)


POLICIES_A = ["closed loop", "open-loop prior plan", "photo only", "fixed 4 N", "firm thumb 10 N"]
ABLATIONS_A = ["bruise not owned by ripeness"]
POLICIES_B = ["belief band", "photo-only band", "cut now", "wait 2 days", "wait 4 days", "knife twist"]
ABLATIONS_B = ["ready / not ready", "no testa side", "no repair", "population ripening rate"]
