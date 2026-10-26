"""
The PR2 stands at the kitchen counter, detects an avocado, creates the avocado's SDT node (the robot's belief about
it) and reasons about what to do with it; then halves and pits it with its own hands while monitoring the execution.

1. Perceive: park, raise the torso, drive to the counter, look, detect (DetectAction, type Avocado). The detection
   creates the SDT node: the register prior over the avocado's hidden properties (4000 particles). The head camera's
   image (an Isaac Sim render of the instance, cram/camera/) is analysed (food/avocado_vision.py) and folded in.
2. Reason: competence questions from the belief (what is it, how ripe, where is the seed), the task models (what can
   I do, what am I doing) and mental simulation (the halving and pitting mechanics run on avocados drawn from the
   belief). Goal: guacamole now, or store the avocado? First: is a squeeze worth it (value of information, bruise
   risk)? If so the PR2 squeezes the fruit (left gripper, force chosen not to bruise) and the belief is updated.
3. Act and monitor, with PR2 motions (cram/avocado_motions.py): the left hand holds the fruit, the knife held flat
   cuts round the seed outline the belief predicts (CutRoundSeedAction, a CuttingAction); the knife is laid down;
   the left hand holds the lower half, the right hand twists the upper half off about the cut-plane normal and puts
   it down; the right hand takes a spoon, slides it under the seed, levers it out and puts it on the board.
   Forces, torques and outcomes come from the emulator's mechanics run on the hidden true avocado (food/
   avocado_process.py; the kinematic simulation has no contact forces). After every step the measured outcome is
   located in the imagined distribution (as expected / unexpected / surprising), the belief is updated and the world
   shows the new pieces.

All records (belief variables, answers, imagined outcomes, monitoring) are offered to the CRAMERA query panel.

Run beside an existing CRAMERA session (own bridge port):
    source ~/ros_ws/install/setup.bash; source ~/.virtualenvs/cram-env/bin/activate
    LIVE_VIZ_PORT=8766 CRAMERA_DATA=~/Documents/HouseholdTwin/cram/cramera_data cramera-live ~/Documents/HouseholdTwin/cram/avocado_pr2_demo.py
    open http://localhost:8711/?live=127.0.0.1:8766
Headless: CORAPLEX_VISUALIZATION=none python ~/Documents/HouseholdTwin/cram/avocado_pr2_demo.py
"""

import json
import logging
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from coraplex.datastructures.dataclasses import Context  # noqa: E402
from coraplex.datastructures.enums import DetectionTechnique  # noqa: E402
from coraplex.execution_environment import kinematically_simulated_robot  # noqa: E402
from coraplex.plans.factories import sequential  # noqa: E402
from coraplex.robot_plans.actions.core.misc import DetectAction  # noqa: E402
from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction  # noqa: E402
from coraplex.robot_plans.actions.core.robot_body import (  # noqa: E402
    MoveTorsoAction, ParkArmsAction, SetGripperAction)
from coraplex.testing import attach_tool, setup_world, start_visualization  # noqa: E402
from experiments.tool_based_actions.simple_demo.demo_world import CUT_MOUNT, parse_object  # noqa: E402
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState  # noqa: E402
from semantic_digital_twin.robots.pr2 import PR2  # noqa: E402
from semantic_digital_twin.semantic_annotations.semantic_annotations import CuttingKnife, Spoon  # noqa: E402
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix  # noqa: E402
from semantic_digital_twin.spatial_types.spatial_types import Pose  # noqa: E402

import avocado_motions as M  # noqa: E402
import avocado_sdt as SDT  # noqa: E402
from avocado_world import (FRUIT_CUT_HORIZONTAL, Avocado, AvocadoMeshes, remove_piece, resting,  # noqa: E402
                           spawn_board, spawn_piece, surface_height_below)
from food import avocado_belief as AB  # noqa: E402
from food import avocado_process as PRC  # noqa: E402
from food.avocado import Avocado as AvocadoModel  # noqa: E402

logger = logging.getLogger("avocado_demo")
INSTANCE = os.path.join(os.path.dirname(HERE), "assets", "food", "avocado_4")
CAMERA_IMAGE = os.path.join(HERE, "camera", "avocado_4_counter.png")
X, Y = 2.4, 2.2                  # the avocado on the cutting board on the kitchen counter
BASE_XYZ = (1.70, 2.2, 0.0)      # where the PR2 stands: far enough back to hold the knife flat close to the fruit
KNIFE_REST = (2.30, 1.80)        # where the knife is laid down (gripper position over the counter)
HALF_ASIDE = (2.31, 2.31)        # where the upper half is put down (front left of the board)
SEED_DROP = (2.30, 2.12)         # where the seed is put on the board (front right)


def say(text):
    print(text, flush=True)
    logger.info(text)


# %% the hidden truth: the instance the avocado was built from, executed by the emulator's mechanics
class TrueAvocado:
    """The real avocado of the simulation: its register draw and the same mechanics and random stream as the
    emulator (food/avocado_emulator.py), so the outcomes match the exported USD states."""

    def __init__(self, instance_dir):
        rec = json.load(open(os.path.join(instance_dir, "instance.json")))
        hp = {k: v for k, v in rec["hidden_params"].items() if not k.startswith("_")}
        self.hp = hp
        self.avo = AvocadoModel(rec.get("seed", 0), overrides=hp)
        self.stage = float(hp["ripeness_stage"])
        self.plane = PRC.CutPlane(rec["cuts"]["half_a"]["plane_azimuth_deg"], 360.0)
        self.rng = np.random.default_rng(2)
        self.sensor = np.random.default_rng(11)          # squeeze sensor noise (not the emulator's stream)
        self.truth = dict(stage=self.stage, seed_length=hp["seed_length"], testa_adheres_to=hp["testa_adheres_to"])

    def squeeze(self, force_N):
        P = {k: np.array([float(self.hp[k])]) for k in AB.RANDOM}
        P["bruised"] = np.array([False])
        P["testa_flesh"] = np.array([self.hp["testa_adheres_to"] == "flesh"])
        D = AB.derived(P)
        k = float(D["squeeze_k_N_per_mm"][0]) * (1 + self.sensor.normal(0, 0.1 * 4.0 / force_N))
        bruised = force_N > AB.PR.AVOCADO["bruise_force_ratio"][1] * float(D["penetrometer_N"][0])
        return k, bruised

    def cut_round_seed(self):
        return PRC.cut_round_seed(self.avo, self.plane, self.rng, self.stage, 40.0)

    def twist_apart(self):
        return PRC.twist_apart(self.avo, self.plane, self.rng, self.stage, self.plane.coverage_deg, 1.5)

    def remove_seed(self, method="spoon"):
        return PRC.remove_seed(self.avo, self.rng, self.stage, method, 25.0, 1.0)


# %% the query panel
class Live:
    """A re-iterable view of a list the node replaces or grows (query domains are read on every query)."""

    def __init__(self, get):
        self.get = get

    def __iter__(self):
        return iter(self.get())


def register_queries(viz, world, node):
    bridge = getattr(getattr(viz, "provider", None), "bridge", None)
    if bridge is None:
        return
    from cramera.knowledge.presets import Preset
    from cramera.knowledge.query_domain import QueryDomain
    from cramera.knowledge.queryable_knowledge import QueryableKnowledge, QueryScope
    domains = [QueryDomain("avocado", Avocado, Live(lambda: world.get_semantic_annotations_by_type(Avocado))),
               QueryDomain("belief_variable", SDT.BeliefVariable, Live(lambda: node.variables)),
               QueryDomain("answer", SDT.CompetenceAnswer, Live(lambda: node.answers)),
               QueryDomain("imagined", SDT.ImaginedOutcome, Live(lambda: node.imagined)),
               QueryDomain("monitoring", SDT.MonitoringRecord, Live(lambda: node.monitoring))]
    knowledge = [QueryableKnowledge(scope=QueryScope.CURRENT_STATE, domains=domains, extra_names={"world": world})]
    presets = [
        Preset("What does the robot believe about the avocado?", "an(entity(belief_variable))"),
        Preset("How ripe is the avocado?",
               "an(entity(belief_variable).where(belief_variable.variable == 'ripeness_stage'))"),
        Preset("Which competence questions has the robot answered?", "an(entity(answer))"),
        Preset("Which answers come from mental simulation?",
               "an(entity(answer).where(answer.method == 'mental simulation'))"),
        Preset("What did the robot imagine for each plan?", "an(entity(imagined))"),
        Preset("How did the execution go?", "an(entity(monitoring))"),
        Preset("Was anything not as expected?", "an(entity(monitoring).where(monitoring.verdict != 'as expected'))"),
        Preset("Which avocado pieces are in the world?", "an(entity(avocado))"),
    ]
    bridge.register_query_source(knowledge, "avocado SDT", presets)


# %% the demo
def perform(viz, steps, context):
    plan = sequential(steps, context=context).plan
    if viz is not None:
        viz.attach_plan(plan)
    with kinematically_simulated_robot:
        return plan.perform()


def ask(node, f, *a):
    r = f(*a)
    say(f"\n  Q: {r.question}\n  A: {r.answer}\n     ({r.method}; {r.evidence})")
    return r


def report(rec):
    say(f"   {rec.step}: {rec.verdict} (expected {rec.expected}); measured {rec.measured}\n      belief "
        f"{rec.belief_before}\n          -> {rec.belief_after}")


def annotate(world, *bodies):
    with world.modify_world():
        world.add_semantic_annotations([Avocado(root=b) for b in bodies])


def main():
    world = setup_world()
    meshes = AvocadoMeshes()
    truth = TrueAvocado(INSTANCE)
    counter = surface_height_below(world, (X, Y))
    top = spawn_board(world, (X, Y + 0.02), counter)
    T_whole = resting(meshes, "whole", X, Y, top, R=FRUIT_CUT_HORIZONTAL)
    whole = spawn_piece(world, meshes, "whole", T_whole, "avocado_whole")
    annotate(world, whole)
    world.merge_world_at_pose(parse_object("spoon.stl"), HomogeneousTransformationMatrix.from_xyz_rpy(
        X - 0.02, Y - 0.26, counter + 0.006, reference_frame=world.root))
    spoon_body = world.get_body_by_name("spoon.stl")
    pr2 = PR2.from_world(world)                      # before the viewer attaches, so it knows the robot
    viz = start_visualization(world)
    knife = CuttingKnife(root=attach_tool(world, pr2.right_arm, parse_object("big-knife.stl"), CUT_MOUNT))
    with world.modify_world():
        world.add_semantic_annotations([knife, Spoon(root=spoon_body)])
    context = Context(world=world, robot=pr2, _debug=False, ros_node=None)
    context.evaluate_conditions = False
    L, R = pr2.left_arm, pr2.right_arm
    say(f"cutting board top at {top:.3f} m; avocado at ({X}, {Y}) with its cut plane horizontal (hidden truth: "
        f"stage {truth.stage:.2f}, seed {truth.truth['seed_length']:.1f} mm, seed coat adheres to the "
        f"{truth.truth['testa_adheres_to']})")

    # 1. perceive
    target = Pose.from_xyz_rpy(X, Y, top + 0.03, reference_frame=world.root)
    perform(viz, [SetGripperAction(R.end_effector, GripperState.CLOSE), ParkArmsAction(pr2.all_arms),
                  MoveTorsoAction(TorsoState.HIGH),
                  NavigateAction(Pose.from_xyz_rpy(*BASE_XYZ, reference_frame=world.root)), LookAtAction(target)],
            context)
    perform(viz, [DetectAction(technique=DetectionTechnique.TYPES, object_sem_annotation=Avocado)], context)
    avocado = world.get_semantic_annotations_by_type(Avocado)[0]
    node = SDT.AvocadoSDT.on_detection("avocado_1")
    avocado.sdt = node
    register_queries(viz, world, node)
    say(f"\ndetected an Avocado at {np.round(avocado.root.global_pose.to_np()[:3, 3], 3)}; SDT node created from the "
        f"register prior: {SDT.summary_line(node.belief)}")
    if os.path.exists(CAMERA_IMAGE):
        obs, _ = SDT.photo_observations(CAMERA_IMAGE, node.belief.time_h)
        node.perceive(obs)
        say(f"head camera image analysed ({os.path.basename(CAMERA_IMAGE)}): {SDT.summary_line(node.belief)}")

    # 2. reason; squeeze if it is worth it
    for f in (node.what_is_it, node.how_ripe, node.where_is_the_seed, node.what_can_be_done):
        ask(node, f)
    force, voi, p_bruise, _ = node.value_of_squeeze()
    a = node.answers[-1]
    say(f"\n  Q: {a.question}\n  A: {a.answer}\n     ({a.method}; {a.evidence})")
    S = node.belief.summary()
    g = M.FruitGeometry(T_whole[:3, 3].copy(), S["length_L"]["mean"] * 1e-3, S["diameter_D"]["mean"] * 1e-3,
                        S["seed_length"]["mean"] * 1e-3, S["seed_width_ratio"]["mean"] * S["seed_length"]["mean"] * 1e-3)
    if force is not None:
        k_mean = node.expected_squeeze(force)[0]
        depth_mm = force / max(k_mean, 0.5)
        node.current_task = "SqueezeTest"
        perform(viz, M.squeeze_steps(world, L, g, depth_mm), context)
        k, bruised = truth.squeeze(force)
        report(node.monitor_squeeze(k, force))
        if bruised:
            say("   (the squeeze bruised the fruit: hidden truth)")
    best_goal, _ = node.decide()
    a = node.answers[-1]
    say(f"\n  Q: {a.question}\n  A: {a.answer}\n     ({a.method}; {a.evidence})")
    t0 = time.time()
    best, wait = node.what_if_halved_now()
    for a in node.answers[-2:]:
        say(f"\n  Q: {a.question}\n  A: {a.answer}\n     ({a.method}; {a.evidence})")
    ask(node, node.how_to_do_it, best)
    say(f"   (mental simulation took {time.time() - t0:.0f} s)")
    say(f"\ndecision: the robot would '{best_goal}'; the task asks to halve and pit it now -> {best.plan}")

    # 3. act and monitor
    S = node.belief.summary()
    g.seed_length_m = S["seed_length"]["mean"] * 1e-3
    g.seed_width_m = S["seed_width_ratio"]["mean"] * g.seed_length_m
    node.current_task = "CutRoundSeed"
    ask(node, node.what_am_i_doing)
    expected = node.expect("CutRoundSeed")
    frame = M.cut_alignment_frame(world, g.seed_centre)
    perform(viz, M.hold_from_above_steps(world, L, g) + [
        M.CutRoundSeedAction(object_to_cut=whole, arm=R, tool=knife, align_frame=frame,
                             seed_centre=tuple(g.seed_centre),
                             seed_semi_axes=(g.seed_length_m / 2, g.seed_width_m / 2), pointer_stride=5)], context)
    out = truth.cut_round_seed()
    report(node.monitor("CutRoundSeed", out, expected))
    remove_piece(world, whole)                                   # scored: two halves, still together
    ha = spawn_piece(world, meshes, "half_a", T_whole, "avocado_half_a")
    hb = spawn_piece(world, meshes, "half_b", T_whole, "avocado_half_b")
    annotate(world, ha, hb)
    rest = (KNIFE_REST[0], KNIFE_REST[1], counter + 0.007)
    perform(viz, M.let_go_from_above_steps(world, L, g) + M.lay_down_tool_steps(world, R, rest), context)
    M.release(world, knife.root)
    perform(viz, M.back_off_flat_steps(world, R, rest), context)

    node.current_task = "TwistApart"
    ask(node, node.what_am_i_doing)
    expected = node.expect("TwistApart")
    grasp, c = M.twist_steps(world, R, g)
    perform(viz, M.hold_steps(world, L, g, dz=-0.012) + grasp, context)
    M.attach(world, hb, R)
    perform(viz, M.twist_motion(world, R, c), context)
    out = truth.twist_apart()
    report(node.monitor("TwistApart", out, expected))
    perform(viz, M.put_aside_steps(world, R, HALF_ASIDE, top, c[2] - g.cut_z), context)
    M.release(world, hb)
    perform(viz, [M.gripper(R, True), M.move(world, R, (HALF_ASIDE[0], HALF_ASIDE[1], top + 0.15), M.rot_z(90) @ M.TOP_ACROSS)] +
            M.let_go_steps(world, L, g), context)

    node.current_task = "RemoveSeed"
    ask(node, node.what_am_i_doing)
    expected = node.expect("RemoveSeed")
    perform(viz, M.spoon_grasp_steps(world, R, spoon_body), context)
    M.attach(world, spoon_body, R)
    tool_T_spoon = np.linalg.inv(R.end_effector.tool_frame.global_pose.to_np()) @ spoon_body.global_pose.to_np()
    under, lift = M.scoop_waypoints(g)
    perform(viz, M.spoon_moves(world, R, under, tool_T_spoon), context)
    out = truth.remove_seed("spoon")
    report(node.monitor("RemoveSeed", out, expected))
    T_half = ha.global_pose.to_np()
    remove_piece(world, ha)
    hp = spawn_piece(world, meshes, "half_a_pitted", T_half, "avocado_half_a_pitted")
    seed = spawn_piece(world, meshes, "seed", T_half, "avocado_seed")
    annotate(world, hp)
    M.attach(world, seed, R)
    drop = (SEED_DROP[0], SEED_DROP[1], top + 0.022)
    perform(viz, M.spoon_moves(world, R, lift + M.seed_drop_waypoints(drop), tool_T_spoon), context)
    p_seed = seed.global_pose.to_np()[:3, 3]
    remove_piece(world, seed)                                    # the seed rolls off the spoon onto the board
    seed = spawn_piece(world, meshes, "seed", resting(meshes, "seed", p_seed[0], p_seed[1], top), "avocado_seed")
    perform(viz, M.spoon_moves(world, R, [((drop[0], drop[1], drop[2] + 0.10), 25.0)], tool_T_spoon), context)

    ask(node, node.what_is_it)
    ask(node, node.what_can_be_done)
    say(f"\nfinal belief: {SDT.summary_line(node.belief)}; hidden truth: stage {truth.stage:.2f}, seed "
        f"{truth.truth['seed_length']:.1f} mm, seed coat adheres to the {truth.truth['testa_adheres_to']}")
    say(f"{len(node.answers)} answers, {len(node.imagined)} imagined plans, {len(node.monitoring)} monitored steps")
    return world, node


if __name__ == "__main__":
    main()
