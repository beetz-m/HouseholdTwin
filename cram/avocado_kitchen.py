"""
CRAMERA interface: the apartment kitchen with the PR2 and an avocado on a cutting board, driven from the viewer.

The scene stays live. The query panel offers the operator's controls as ready-made questions (buttons; typed
requests are matched to them):

- commands: detect the avocado, squeeze it, wait two days, cut round the seed, twist the halves apart, spoon out the
  seed, reset the kitchen. A command is checked against the robot's and the avocado's state (refused with the
  reason) and queued; a worker executes the queue with PR2 motions (cram/avocado_motions.py) while the viewer follows
  the robot. Every step is predicted by mental simulation, monitored against the prediction and folded into the
  avocado's SDT node (cram/avocado_sdt.py).
- questions to the robot: what is it, how ripe, where is the seed, what can I do now, what am I doing (answered at
  once); should I squeeze it first, guacamole now or store it, what happens if I halve it now (mental simulation,
  queued).
- views: the robot's answers, its belief, its predictions, the execution monitoring, the command log, the commands
  with their availability.

The commands sit in the query attribute ``issued`` (``an(entity(cmd_x).where(cmd_x.issued == True))``): the viewer
builds every preset to word it, and only evaluating the query reads the attribute, so listing the presets does not
fire anything.

Run (beside an existing CRAMERA session on 8765):
    source ~/ros_ws/install/setup.bash; source ~/.virtualenvs/cram-env/bin/activate
    LIVE_VIZ_PORT=8766 CRAMERA_DATA=~/Documents/HouseholdTwin/cram/cramera_data cramera-live ~/Documents/HouseholdTwin/cram/avocado_kitchen.py
    open http://localhost:8711/?live=127.0.0.1:8766
Headless test of a command sequence:
    CORAPLEX_VISUALIZATION=none python ~/Documents/HouseholdTwin/cram/avocado_kitchen.py --do detect,how_ripe,squeeze,cut,twist,spoon
"""

import argparse
import logging
import os
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from typing import ClassVar

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
import avocado_agent as AG  # noqa: E402
import avocado_sdt as SDT  # noqa: E402
from food import avocado_process as PRC  # noqa: E402
from food.avocado_lifetime import FruitClock  # noqa: E402
from avocado_pr2_demo import (BASE_XYZ, CAMERA_IMAGE, HALF_ASIDE, INSTANCE, KNIFE_REST, SEED_DROP, X, Y,  # noqa: E402
                              Live, TrueAvocado)
from avocado_world import (FRUIT_CUT_HORIZONTAL, Avocado, AvocadoMeshes, resting, spawn_board,  # noqa: E402
                           spawn_piece, surface_height_below)

logger = logging.getLogger("avocado_kitchen")


def say(text):
    print(text, flush=True)
    logger.info(text)


# %% what the operator sees in the query panel
@dataclass
class Command:
    """A command, question or view the operator can give (one preset each)."""
    name: str
    label: str
    kind: str            # action | question | thought (queued question)
    help: str
    available: str = "-"
    status: str = "ready"
    KITCHEN: ClassVar = None

    @property
    def issued(self):
        """Read only when a query is evaluated (the operator clicked or asked): hands the command to the kitchen."""
        return Command.KITCHEN.request(self)


@dataclass
class CommandRecord:
    """One entry of the command log."""
    name: str
    state: str           # queued | running | done | refused | failed | answered
    detail: str
    clock: str


ACTIONS = [
    ("detect", "Detect the avocado", "park, raise the torso, drive to the counter, look at the board, detect; creates "
     "the avocado's SDT node and analyses the head camera image"),
    ("squeeze", "Squeeze the avocado", "left gripper, the strongest force that will not bruise it (belief); the "
     "stiffness updates the ripeness belief"),
    ("wait", "Wait two days", "the avocado ripens; the belief predicts it at each particle's own rate"),
    ("cut", "Cut round the seed", "hold the neck, knife flat in the cut plane round the believed seed outline, lay "
     "the knife down"),
    ("twist", "Twist the halves apart", "hold the lower half, twist the upper half off about the cut-plane normal, "
     "put it down"),
    ("spoon", "Spoon out the seed", "take the spoon, slide it under the seed, lever it out, put it on the board"),
    ("reset", "Reset the kitchen", "knife back in the hand, spoon back on the counter, a fresh avocado, belief "
     "forgotten"),
    ("mission_now", "Make guacamole now", "the agent: squeeze if worth it, plan (goals to enforce, risks to prevent), "
     "execute with repairs, audit the effects"),
    ("mission_half", "Half now, half for tomorrow's breakfast", "as above; the other half is kept so that it is "
     "still good after 15 h"),
    ("mission_two_days", "Guacamole in two days", "as above; when to cut and where to keep the fruit until then"),
]
QUESTIONS = [
    ("what_is_it", "What is the object?", "belief"),
    ("how_ripe", "How ripe is it?", "belief"),
    ("where_is_the_seed", "Where is the seed?", "belief"),
    ("what_can_i_do", "What can you do now?", "robot state and the task models"),
    ("what_am_i_doing", "What are you doing?", "the current task's designator"),
]
THOUGHTS = [
    ("should_i_squeeze", "Should you squeeze it first?", "value of information (mental simulation, ~20 s)"),
    ("guacamole", "Guacamole now, or store it?", "mental simulation now vs after 48 h (~15 s)"),
    ("what_if", "What happens if you halve it now?", "mental simulation of halving and pitting (~25 s)"),
    ("when_best", "When is it best to eat?", "lifetime analysis over the belief (counter / fridge)"),
]
QUESTIONS += [
    ("risks", "What are the risks?", "the chosen plan's predicted effects that must be prevented"),
    ("why", "Why this plan?", "the agent's decision: goals, bounds, rejected plans"),
]
SAYINGS = {                     # typed requests recognised besides the button labels
    "detect": ["find the avocado", "look at the avocado", "detect it"],
    "squeeze": ["squeeze it", "test the ripeness", "feel the avocado"],
    "wait": ["wait", "let it ripen", "store it"],
    "cut": ["cut it", "halve it", "cut the avocado", "halve the avocado"],
    "twist": ["twist it", "separate the halves", "open the avocado"],
    "spoon": ["remove the seed", "pit it", "take out the stone", "scoop out the seed"],
    "reset": ["start over", "reset"],
    "how_ripe": ["is it ripe", "is it ripe enough"],
    "what_can_i_do": ["what can you do", "what is possible"],
    "mission_now": ["make guacamole", "guacamole now", "prepare the avocado"],
    "mission_half": ["half now half tomorrow", "keep half for tomorrow"],
    "mission_two_days": ["guacamole in two days", "prepare it in two days"],
    "risks": ["what could go wrong", "what are the risks"],
    "why": ["why this plan", "explain your plan"],
    "when_best": ["when should we eat it", "when is it ripe"],
}
LEMON_HOME = (2.47, 1.98)        # the lemon on the counter, right of the board


# %% the kitchen
class AvocadoKitchen:
    def __init__(self):
        self.world = setup_world()
        self.meshes = AvocadoMeshes()
        self.counter = surface_height_below(self.world, (X, Y))
        self.top = spawn_board(self.world, (X, Y + 0.02), self.counter)
        self.world.merge_world_at_pose(parse_object("spoon.stl"), HomogeneousTransformationMatrix.from_xyz_rpy(
            X - 0.02, Y - 0.26, self.counter + 0.006, reference_frame=self.world.root))
        self.spoon = self.world.get_body_by_name("spoon.stl")
        self.spoon_home = self.spoon.global_pose.to_np()
        self.pr2 = PR2.from_world(self.world)
        self.viz = start_visualization(self.world)
        self.knife = CuttingKnife(root=attach_tool(self.world, self.pr2.right_arm, parse_object("big-knife.stl"),
                                                   CUT_MOUNT))
        with self.world.modify_world():
            self.world.add_semantic_annotations([self.knife, Spoon(root=self.spoon)])
        self.context = Context(world=self.world, robot=self.pr2, _debug=False, ros_node=None)
        self.context.evaluate_conditions = False
        self.L, self.R = self.pr2.left_arm, self.pr2.right_arm
        self.align = M.cut_alignment_frame(self.world, np.array([X, Y, self.top + 0.03]))
        Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)            # lemon lying, long axis along y
        self.lemon = spawn_piece(self.world, self.meshes, "lemon", resting(self.meshes, "lemon", *LEMON_HOME,
                                                                            self.counter, R=Rx), "lemon")
        self.lemon_home = self.lemon.global_pose.to_np()
        lo, hi = self.meshes.bounds("lemon", np.c_[np.r_[Rx, [[0, 0, 0]]], [0, 0, 0, 1]])
        self.lemon_width = float(hi[0] - lo[0])
        self.lemon_centre = (lo + hi) / 2 + self.lemon_home[:3, 3]
        self.agent = None
        self.hold = None
        self.queue = queue.Queue()
        self.log = []
        self.busy = None
        self.knife_in_hand = True
        self.spoon_in_hand = False
        self.tool_T_spoon = None
        self.at_counter = False
        self.pieces = {}
        self._last = {}
        self.new_avocado()
        self.commands = {n: Command(n, label, "action", h) for n, label, h in ACTIONS}
        self.commands.update({n: Command(n, label, "question", h) for n, label, h in QUESTIONS})
        self.commands.update({n: Command(n, label, "thought", h) for n, label, h in THOUGHTS})
        Command.KITCHEN = self
        self.refresh()
        self.register()

    # %% scene
    def new_avocado(self):
        self.truth = TrueAvocado(INSTANCE)
        self.truth_clock = FruitClock.of(self.truth.hp)
        self.half_clocks = {}            # tag -> (FruitClock of the cut half, storage treatment)
        self.clock_h = 0.0
        self.remnant = 0.0
        self.bruised = False
        self.node = None
        self.T_whole = resting(self.meshes, "whole", X, Y, self.top, R=FRUIT_CUT_HORIZONTAL)
        self.add_piece("whole", self.T_whole, "avocado_whole")
        self.g = None

    def add_piece(self, piece, T, tag, annotate=True):
        body = spawn_piece(self.world, self.meshes, piece, T, tag)
        if annotate:
            with self.world.modify_world():
                a = Avocado(root=body)
                self.world.add_semantic_annotations([a])
            if self.node is not None:
                a.sdt = self.node
        self.pieces[tag] = body
        return body

    def drop_piece(self, tag):
        body = self.pieces.pop(tag)
        with self.world.modify_world():
            for a in [a for a in self.world.get_semantic_annotations_by_type(Avocado) if a.root is body]:
                self.world.remove_semantic_annotation(a)
            self.world.remove_branch_from_world(body)

    def perform(self, steps):
        plan = sequential(steps, context=self.context).plan
        if self.viz is not None and hasattr(self.viz, "attach_plan"):
            self.viz.attach_plan(plan)
        with kinematically_simulated_robot:
            return plan.perform()

    # %% the query panel
    def register(self):
        bridge = getattr(getattr(self.viz, "provider", None), "bridge", None)
        if bridge is None:
            return
        from cramera.knowledge.presets import Preset
        from cramera.knowledge.query_domain import QueryDomain
        from cramera.knowledge.queryable_knowledge import QueryableKnowledge, QueryScope
        node = (lambda: self.node)
        domains = [QueryDomain(f"cmd_{n}", Command, [c]) for n, c in self.commands.items()]
        domains += [QueryDomain("command", Command, Live(lambda: list(self.commands.values()))),
                    QueryDomain("command_log", CommandRecord, Live(lambda: self.log[::-1])),
                    QueryDomain("avocado", Avocado, Live(lambda: self.world.get_semantic_annotations_by_type(Avocado))),
                    QueryDomain("belief_variable", SDT.BeliefVariable, Live(lambda: node().variables if node() else [])),
                    QueryDomain("answer", SDT.CompetenceAnswer, Live(lambda: node().answers[::-1] if node() else [])),
                    QueryDomain("imagined", SDT.ImaginedOutcome, Live(lambda: node().imagined if node() else [])),
                    QueryDomain("monitoring", SDT.MonitoringRecord, Live(lambda: node().monitoring if node() else [])),
                    QueryDomain("plan_option", AG.OptionResult, Live(lambda: self.agent.results if self.agent else [])),
                    QueryDomain("decision", AG.Decision, Live(lambda: self.agent.decisions if self.agent else [])),
                    QueryDomain("effect_audit", AG.EffectAudit, Live(lambda: self.agent.audits if self.agent else []))]
        knowledge = [QueryableKnowledge(scope=QueryScope.CURRENT_STATE, domains=domains,
                                        extra_names={"world": self.world, "kitchen": self})]

        def code(n):
            return f"an(entity(cmd_{n}).where(cmd_{n}.issued == True))"
        presets = [Preset(c.label, code(n)) for n, c in self.commands.items()]
        presets += [Preset("Show what I can ask or command", "an(entity(command))"),
                    Preset("Show the robot's answers", "an(entity(answer))"),
                    Preset("Show the robot's belief about the avocado", "an(entity(belief_variable))"),
                    Preset("Show what the robot imagined", "an(entity(imagined))"),
                    Preset("Show how the execution went", "an(entity(monitoring))"),
                    Preset("Show the command log", "an(entity(command_log))"),
                    Preset("Show the plans the agent compared", "an(entity(plan_option))"),
                    Preset("Show the agent's decision", "an(entity(decision))"),
                    Preset("Show which goals were achieved and which risks prevented", "an(entity(effect_audit))")]
        unlisted = [Preset(t, code(n)) for n, ts in SAYINGS.items() for t in ts]
        bridge.register_query_source(knowledge, "avocado kitchen", presets, unlisted)

    # %% state and preconditions
    def integrity(self):
        return self.node.integrity if self.node is not None else "whole"

    def refusal(self, name):
        if name == "reset":
            return None
        if name.startswith("mission_"):
            if self.node is not None and self.integrity() != "whole":
                return f"the avocado is already cut ({self.integrity()}): reset for a fresh one"
            return None if self.knife_in_hand else "the knife is not in my hand (reset)"
        if name == "detect":
            return "the avocado is already detected (reset to start over)" if self.node is not None else None
        if self.node is None:
            return "I have not detected an avocado yet: detect it first"
        st, seed = self.integrity(), self.node.seed
        if name == "squeeze":
            return None if st == "whole" else f"it is no longer whole ({st})"
        if name == "wait":
            return None if st == "whole" else f"it is already cut ({st}); cut flesh browns, it does not ripen"
        if name == "cut":
            if st not in ("whole", "partly_scored"):
                return f"it is already cut ({st})"
            return None if self.knife_in_hand else "the knife is not in my hand (reset)"
        if name == "twist":
            return None if st in ("scored", "partly_scored") else (
                "it has to be cut round the seed first" if st == "whole" else f"the halves are already apart ({st})")
        if name == "spoon":
            if st not in ("halved", "halved_torn"):
                return "the halves have to be apart first" if st in ("whole", "scored", "partly_scored") else st
            return None if seed == "in" else "the seed is already out"
        return None

    def refresh(self):
        for n, c in self.commands.items():
            if c.kind == "action":
                r = self.refusal(n)
                c.available = "yes" if r is None else f"no: {r}"
            else:
                c.available = "yes" if (self.node is not None or n == "what_can_i_do") else "no: nothing detected yet"

    def record(self, name, state, detail=""):
        self.log.append(CommandRecord(name, state, detail, time.strftime("%H:%M:%S")))
        if name in self.commands:
            self.commands[name].status = f"{state}: {detail}" if detail else state
        say(f"[{name}] {state}{': ' + detail if detail else ''}")

    # %% requests (from the panel's query evaluation, or the --do list)
    def request(self, cmd):
        now = time.time()
        if now - self._last.get(cmd.name, 0.0) < 1.5:          # the attribute is read more than once per query
            return True
        self._last[cmd.name] = now
        if cmd.kind == "question":
            self.record(cmd.name, "answered", self.answer_now(cmd.name))
            return True
        idle = self.queue.empty() and self.busy is None
        reason = self.refusal(cmd.name) if idle and cmd.kind == "action" else None
        if cmd.kind == "thought" and self.node is None:
            reason = "I have not detected an avocado yet"
        if reason:
            self.record(cmd.name, "refused", reason)
            return True
        self.queue.put(cmd.name)
        self.record(cmd.name, "queued", f"{self.queue.qsize()} in the queue" + (f", busy with {self.busy}"
                                                                                  if self.busy else ""))
        return True

    def answer_now(self, name):
        if name == "what_can_i_do":
            ok = [self.commands[n].label for n, _, _ in ACTIONS if self.refusal(n) is None]
            no = [f"{self.commands[n].label.lower()} ({self.refusal(n)})" for n, _, _ in ACTIONS
                  if self.refusal(n) is not None and n != "reset"]
            return "I can: " + "; ".join(ok) + (". Not now: " + "; ".join(no) if no else "")
        if self.node is None:
            return "I have not detected an avocado yet"
        if name in ("risks", "why"):
            if self.agent is None or not self.agent.decisions:
                return "I have not planned a mission yet"
            d = self.agent.decisions[-1]
            if name == "risks":
                bounds = "; ".join(f"{b.effect} <= {b.p:g}" + (" (safety rule)" if b.hard else "")
                                   for b in self.agent.mission.guards)
                return f"predicted for '{d.chosen}': {d.risks}. Bounds I keep: {bounds}"
            return f"{d.mission}: {d.chosen} -- {d.why}"
        f = {"what_is_it": self.node.what_is_it, "how_ripe": self.node.how_ripe,
             "where_is_the_seed": self.node.where_is_the_seed, "what_am_i_doing": self.node.what_am_i_doing}[name]
        return f().answer

    # %% the worker
    def serve(self, stop_when_idle=False):
        while True:
            if stop_when_idle and self.queue.empty():
                return
            name = self.queue.get()
            reason = self.refusal(name) if self.commands[name].kind == "action" else None
            if reason:
                self.record(name, "refused", reason)
                continue
            self.busy = name
            self.record(name, "running")
            t0 = time.time()
            try:                                                  # an interactive tool reports a failed step and goes on
                detail = getattr(self, f"do_{name}")()
                self.record(name, "done", f"{detail} ({time.time() - t0:.0f} s)")
            except Exception as e:
                logger.error(traceback.format_exc())
                self.record(name, "failed", f"{type(e).__name__}: {str(e)[:200]}")
            self.busy = None
            self.refresh()

    # %% actions
    def geometry(self):
        S = self.node.belief.summary()
        L_ = S["seed_length"]["mean"] * 1e-3
        return M.FruitGeometry(self.T_whole[:3, 3].copy(), S["length_L"]["mean"] * 1e-3,
                               S["diameter_D"]["mean"] * 1e-3, L_, S["seed_width_ratio"]["mean"] * L_)

    def do_detect(self):
        target = Pose.from_xyz_rpy(X, Y, self.top + 0.03, reference_frame=self.world.root)
        steps = [ParkArmsAction(self.pr2.all_arms), MoveTorsoAction(TorsoState.HIGH),
                 NavigateAction(Pose.from_xyz_rpy(*BASE_XYZ, reference_frame=self.world.root)), LookAtAction(target),
                 DetectAction(technique=DetectionTechnique.TYPES, object_sem_annotation=Avocado)]
        self.perform(steps)
        self.at_counter = True
        self.node = SDT.AvocadoSDT.on_detection("avocado_1")
        for a in self.world.get_semantic_annotations_by_type(Avocado):
            a.sdt = self.node
        txt = f"SDT node created ({SDT.summary_line(self.node.belief)})"
        if os.path.exists(CAMERA_IMAGE):
            obs, _ = SDT.photo_observations(CAMERA_IMAGE, self.node.belief.time_h)
            self.node.perceive(obs)
            txt += f"; head camera image: {SDT.summary_line(self.node.belief)}"
        return txt

    def do_squeeze(self):
        force, p = self.node.squeeze_force()
        if force is None:
            return f"not squeezed: every force risks a bruise (P >= {p:.2f})"
        g = self.geometry()
        depth = force / max(self.node.expected_squeeze(force)[0], 0.5)
        self.node.current_task = "SqueezeTest"
        self.perform(M.squeeze_steps(self.world, self.L, g, depth))
        k, bruised = self.truth.squeeze(force)
        self.bruised = self.bruised or bruised
        rec = self.node.monitor_squeeze(k, force)
        return f"{force:.0f} N, {rec.measured}: {rec.verdict}; ripeness {SDT.summary_line(self.node.belief, ('ripeness_stage',))}"

    def do_wait(self):
        return self.advance_time(48.0, "counter")

    def advance_time(self, hours, storage="counter"):
        """Simulated time passes: the whole fruit ripens in ``storage`` (hidden truth and belief), cut halves brown
        and age under their treatments."""
        from food.avocado_lifetime import temperature_factor, STORAGE_T
        before = SDT.summary_line(self.node.belief, ("ripeness_stage",)) if self.node else ""
        if "avocado_whole" in self.pieces or self.integrity() == "whole":
            self.truth_clock = self.truth_clock.advance(hours, storage)
            self.truth.stage = self.truth_clock.stage
            self.truth.avo.p["ripeness_stage"] = self.truth.stage
            if self.node is not None:
                self.node.belief.predict(hours * temperature_factor(STORAGE_T[storage]), self.node.rng)
        for tag, (c, tr) in list(self.half_clocks.items()):
            self.half_clocks[tag] = (c.advance(hours, "fridge" if "fridge" in tr else "counter", tr), tr)
        self.clock_h += hours
        after = SDT.summary_line(self.node.belief, ("ripeness_stage",)) if self.node else ""
        if self.node is not None:
            whole = "avocado_whole" in self.pieces
            self.node.monitoring.append(SDT.MonitoringRecord(f"Time +{hours:.0f} h" + (f" (whole in the {storage})"
                                                                                       if whole else ""),
                                                             "ripening / browning models", f"clock {self.clock_h:.0f} h",
                                                             "as expected", before, after))
        return f"{hours:.0f} h later (clock {self.clock_h:.0f} h): belief {after}"

    def take_knife(self):
        """Pick the knife up again where it lies (a repair needs it)."""
        rest = (KNIFE_REST[0], KNIFE_REST[1], self.counter + 0.007)
        if self.spoon_in_hand:
            self.put_spoon_back()
        self.perform([M.gripper(self.R, True)] + M.lay_down_tool_steps(self.world, self.R, rest) +
                     [M.gripper(self.R, False)])
        M.attach(self.world, self.knife.root, self.R)
        self.knife_in_hand = True
        self.perform([M.move(self.world, self.R, np.array(rest) + [0, 0, 0.15], M.KNIFE_FLAT)])

    def do_cut(self):
        g = self.geometry()
        self.node.current_task = "CutRoundSeed"
        expected = self.node.expect("CutRoundSeed")
        if not self.knife_in_hand:
            self.take_knife()
        if self.hold is None:
            _, self.hold = AG.plan_hold(g)
        if self.hold is None:
            raise RuntimeError("no hold keeps the gripper 10 mm clear of the blade: not cutting")
        tag = "avocado_whole" if "avocado_whole" in self.pieces else "avocado_half_a"
        target = self.pieces[tag]
        self.perform(M.hold_at_steps(self.world, self.L, g, self.hold) + [
            M.CutRoundSeedAction(object_to_cut=target, arm=self.R, tool=self.knife, align_frame=self.align,
                                 seed_centre=tuple(g.seed_centre),
                                 seed_semi_axes=(g.seed_length_m / 2, g.seed_width_m / 2), pointer_stride=5)])
        tcp = self.L.end_effector.tool_frame.global_pose.to_np()[:3, 3]
        self.clearance_mm = 1e3 * (tcp[2] - g.cut_z - AG.PAD_HALF_HEIGHT - AG.BLADE_HALF_THICKNESS)
        out = self.truth.cut_round_seed()
        rec = self.node.monitor("CutRoundSeed", out, expected)
        if "avocado_whole" in self.pieces:
            self.drop_piece("avocado_whole")
            self.add_piece("half_a", self.T_whole, "avocado_half_a")
            self.add_piece("half_b_clean", self.T_whole, "avocado_half_b")
            for tag in ("avocado_half_a", "avocado_half_b"):                    # the cut faces start to brown
                c = self.truth_clock
                self.half_clocks[tag] = (FruitClock(c.stage, c.rate_per_day, c.hours_ripe, c.ripe_window_h,
                                                    c.spoil_days, cut=True), "none")
        rest = (KNIFE_REST[0], KNIFE_REST[1], self.counter + 0.007)
        self.perform(M.let_go_at_steps(self.world, self.L, g, self.hold) + M.lay_down_tool_steps(self.world, self.R, rest))
        M.release(self.world, self.knife.root)
        self.knife_in_hand = False
        self.perform(M.back_off_flat_steps(self.world, self.R, rest))
        return f"{rec.verdict}: {rec.measured}; {self.hold['name']}, blade clearance {self.clearance_mm:.1f} mm"

    def do_twist(self):
        g = self.geometry()
        self.node.current_task = "TwistApart"
        expected = self.node.expect("TwistApart")
        grasp, c = M.twist_steps(self.world, self.R, g)
        self.perform(M.hold_steps(self.world, self.L, g, dz=-0.012) + grasp)
        out = self.truth.twist_apart()
        rec = self.node.monitor("TwistApart", out, expected)
        if not out.ok:
            self.perform([M.gripper(self.R, True), M.move(self.world, self.R, c + [0, 0, 0.12], M.TOP_ACROSS)] +
                         M.let_go_steps(self.world, self.L, g))
            return f"{rec.verdict}: {rec.measured} (the halves stay together)"
        if out.outcome == "halved_torn":                            # the upper half shows the torn cavity wall
            T = self.pieces["avocado_half_b"].global_pose.to_np()
            self.drop_piece("avocado_half_b")
            self.add_piece("half_b", T, "avocado_half_b")
        hb = self.pieces["avocado_half_b"]
        M.attach(self.world, hb, self.R)
        self.perform(M.twist_motion(self.world, self.R, c))
        self.perform(M.put_aside_steps(self.world, self.R, HALF_ASIDE, self.top, c[2] - g.cut_z))
        M.release(self.world, hb)
        self.perform([M.gripper(self.R, True), M.move(self.world, self.R, (HALF_ASIDE[0], HALF_ASIDE[1], self.top + 0.15),
                                                      M.rot_z(90) @ M.TOP_ACROSS)] + M.let_go_steps(self.world, self.L, g))
        return f"{rec.verdict}: {rec.measured}"

    def do_spoon(self):
        g = self.geometry()
        self.node.current_task = "RemoveSeed"
        expected = self.node.expect("RemoveSeed")
        if not self.spoon_in_hand:
            self.perform(M.spoon_grasp_steps(self.world, self.R, self.spoon))
            M.attach(self.world, self.spoon, self.R)
            self.spoon_in_hand = True
            self.tool_T_spoon = (np.linalg.inv(self.R.end_effector.tool_frame.global_pose.to_np()) @
                                 self.spoon.global_pose.to_np())
        under, lift = M.scoop_waypoints(g)
        self.perform(M.spoon_moves(self.world, self.R, under, self.tool_T_spoon))
        out = self.truth.remove_seed("spoon")
        rec = self.node.monitor("RemoveSeed", out, expected)
        self.remnant = out.truth.get("testa_remnant_fraction", 0.0) if out.ok else self.remnant
        self.gouged = bool(out.truth.get("gouges")) if out.ok else False
        if not out.ok:
            self.perform(M.spoon_moves(self.world, self.R, [lift[-1]], self.tool_T_spoon))
            return f"{rec.verdict}: {rec.measured} (the seed stays in)"
        T = self.pieces["avocado_half_a"].global_pose.to_np()
        self.drop_piece("avocado_half_a")
        self.add_piece("half_a_pitted", T, "avocado_half_a_pitted")
        self.half_clocks["avocado_half_a_pitted"] = self.half_clocks.pop("avocado_half_a")
        seed = self.add_piece("seed", T, "avocado_seed", annotate=False)
        M.attach(self.world, seed, self.R)
        drop = (SEED_DROP[0], SEED_DROP[1], self.top + 0.022)
        self.perform(M.spoon_moves(self.world, self.R, lift + M.seed_drop_waypoints(drop), self.tool_T_spoon))
        p = seed.global_pose.to_np()[:3, 3]
        self.drop_piece("avocado_seed")                              # the seed rolls off the spoon onto the board
        self.add_piece("seed", resting(self.meshes, "seed", p[0], p[1], self.top), "avocado_seed", annotate=False)
        self.perform(M.spoon_moves(self.world, self.R, [((drop[0], drop[1], drop[2] + 0.10), 25.0)], self.tool_T_spoon))
        return f"{rec.verdict}: {rec.measured}"

    def do_reset(self):
        steps = []
        if self.spoon_in_hand:                                       # put the spoon back where it was
            self.put_spoon_back()
        if not self.knife_in_hand:                                   # take the knife up again where it lies
            rest = (KNIFE_REST[0], KNIFE_REST[1], self.counter + 0.007)
            self.perform(steps + [M.gripper(self.R, True)] + M.lay_down_tool_steps(self.world, self.R, rest) +
                         [M.gripper(self.R, False)])
            M.attach(self.world, self.knife.root, self.R)
            self.knife_in_hand = True
            steps = [M.move(self.world, self.R, np.array(rest) + [0, 0, 0.15], M.KNIFE_FLAT)]
        self.perform(steps + [ParkArmsAction(self.pr2.all_arms)])
        for tag in list(self.pieces):
            self.drop_piece(tag)
        self.new_avocado()
        self.agent, self.hold = None, None
        return "knife in the hand, spoon on the counter, a fresh avocado on the board"

    def put_spoon_back(self):
        steps, p, Rm = M.spoon_back_steps(self.world, self.R, self.spoon_home, self.tool_T_spoon)
        self.perform(steps)
        M.release(self.world, self.spoon)
        self.spoon_in_hand = False
        self.perform([M.gripper(self.R, True), M.move(self.world, self.R, p + [0, 0, 0.12], Rm)])

    # %% corrective and preventive actions
    def do_peel_coat(self):
        """Lift the seed-coat sheets left in the cavity with the spoon tip (enforces 'cavity clean')."""
        g = self.geometry()
        if not self.spoon_in_hand:
            self.perform(M.spoon_grasp_steps(self.world, self.R, self.spoon))
            M.attach(self.world, self.spoon, self.R)
            self.spoon_in_hand = True
            self.tool_T_spoon = (np.linalg.inv(self.R.end_effector.tool_frame.global_pose.to_np()) @
                                 self.spoon.global_pose.to_np())
        self.perform(M.spoon_moves(self.world, self.R, M.peel_coat_waypoints(g), self.tool_T_spoon))
        out = PRC.peel_seed_coat(None, self.truth.rng, self.remnant, passes=1)
        self.remnant = out.truth["testa_remnant_fraction"]
        self.gouged = self.gouged or bool(out.truth["gouges"])
        seen = out.measured["testa_visible_in_cavity"]
        if not seen:                                                 # the cavity looks clean now
            T = self.pieces["avocado_half_a_pitted"].global_pose.to_np()
            clk = self.half_clocks.pop("avocado_half_a_pitted")
            self.drop_piece("avocado_half_a_pitted")
            self.add_piece("half_a_pitted_clean", T, "avocado_half_a_pitted")
            self.half_clocks["avocado_half_a_pitted"] = clk
            self.node.seed_coat = "removed"
        self.node.monitoring.append(SDT.MonitoringRecord("PeelSeedCoat", "P(clean after one pass) = 0.85",
                                                         "cavity clean" if not seen else "sheets still visible",
                                                         "as expected" if not seen else "unexpected", "", ""))
        return "cavity clean" if not seen else "seed-coat sheets still visible"

    def do_lemon(self, half_tag="avocado_half_a_pitted"):
        """Squeeze the lemon over a cut face (prevents browning; the juice itself is not simulated)."""
        if self.spoon_in_hand:
            self.put_spoon_back()
        self.perform(M.lemon_grasp_steps(self.world, self.R, self.lemon_centre, self.lemon_width))
        M.attach(self.world, self.lemon, self.R)
        over = self.pieces[half_tag].global_pose.to_np()[:3, 3].copy()
        over[2] = self.top + 0.04
        self.perform(M.lemon_squeeze_steps(self.world, self.R, over, self.lemon_width))
        self.perform([M.move(self.world, self.R, self.lemon_centre + [0, 0, 0.12], M.TOP_ACROSS),
                      M.move(self.world, self.R, self.lemon_centre, M.TOP_ACROSS)])
        M.release(self.world, self.lemon)
        self.perform([M.gripper(self.R, True), M.move(self.world, self.R, self.lemon_centre + [0, 0, 0.15],
                                                      M.TOP_ACROSS)])
        return "lemon juice on the cut face"

    def keep_half(self, tag, treatment):
        """Treat and store a half for later. Lemon: PR2 motion; wrap, vacuum and fridge: emulated (no motion)."""
        if "lemon" in treatment:
            self.do_lemon(tag)
        c, _ = self.half_clocks[tag]
        self.half_clocks[tag] = (c, treatment)
        emulated = [t for t in ("wrap", "vacuum", "fridge") if t in treatment]
        self.node.monitoring.append(SDT.MonitoringRecord(f"Keep {tag.replace('avocado_', '')}", treatment,
                                                         treatment + (f" ({', '.join(emulated)} emulated: no PR2 "
                                                                      f"motion yet)" if emulated else ""),
                                                         "as expected", "", ""))

    def serve_halves(self, tags):
        """Serve halves (emulated: no PR2 motion); returns the hidden truth of their eating quality and look."""
        out = []
        for tag in tags:
            c, tr = self.half_clocks[tag]
            out.append(dict(tag=tag, quality=c.eating_quality, dE=c.dE, browned=c.browned, spoiled=c.spoiled,
                            treatment=tr))
            self.node.monitoring.append(SDT.MonitoringRecord(
                f"Serve {tag.replace('avocado_', '')} at {self.clock_h:.0f} h", "edible, not brown",
                f"eating quality {c.eating_quality:.2f}, dE {c.dE:.1f} (simulator truth)",
                "as expected" if c.eating_quality >= 0.6 and not c.browned else "unexpected", "", ""))
        return out

    # %% the agent's missions
    def do_mission_now(self):
        return self.run_mission("guacamole_now")

    def do_mission_half(self):
        return self.run_mission("half_now_half_tomorrow")

    def do_mission_two_days(self):
        return self.run_mission("guacamole_in_two_days")

    def watch(self, after, t_start):
        """Goal watch after a step: goals now below their bound are reported with the reason and what is done."""
        st = dict(integrity=self.integrity(), seed=self.node.seed, remnant=self.remnant)
        for goal, p, bound, why in self.agent.reassess(self.clock_h - t_start, st):
            key = goal
            if key in self._watched:
                continue
            self._watched.add(key)
            action = {"cavity_clean": "enforcing it: peel the sheets off with the spoon tip",
                      "edible_at_service": "nothing restores ripeness after the cut: serving as asked, flagged",
                      "flesh_intact": "cannot be repaired: carry on, quality reduced"}.get(goal, "flagged")
            self.node.monitoring.append(SDT.MonitoringRecord(f"Goal watch after {after}", f"{goal} P >= {bound:g}",
                                                             f"P = {p:.2f}: {why}", "goal at risk", "", action))
            self.node._answer(f"Is the goal '{goal}' still safe?", f"no (P = {p:.2f} < {bound:g}): {why}; {action}",
                              "goal watch: the open goals re-predicted from the updated belief", "goal watch")

    def run_mission(self, name):
        m = AG.mission(name)
        t_start = self.clock_h
        self._watched = set()
        if self.node is None:
            self.do_detect()
        self.agent = AG.AvocadoAgent(self.node, m)
        # 1 information before deciding: squeeze if it is worth it and will not bruise
        force, voi, p_bruise, _ = self.node.value_of_squeeze()
        if force is not None:
            self.do_squeeze()
            self.watch("SqueezeTest", t_start)
        # 2 geometry: where the holding hand may be while the blade sweeps the cut plane
        g = self.geometry()
        cands, self.hold = AG.plan_hold(g)
        self.node._answer("Where may my holding hand be while the blade cuts?",
                          "; ".join(f"{c['name']}: pads {1e3 * c['tcp_height']:.0f} mm above the cut, clearance "
                                    f"{c['clearance_mm']:.0f} mm, " + ("grips" if c["grip"] else "cannot grip (fruit "
                                                                         "too low there)") for c in cands)
                          + f" -> {self.hold['name'] if self.hold else 'no safe hold'}",
                          "fruit profile from the belief means; blade held flat (+-6 mm); 10 mm clearance rule",
                          "geometry")
        # 3 decide: goals to enforce, risks to prevent, over the fruit's lifetime
        self.agent.hold_choice = dict(clear=self.hold is not None)
        best, dec = self.agent.decide()
        self.node._answer("Which plan, and why?", f"{dec.chosen}: {dec.why}", f"risks {dec.risks}; lifetime "
                          f"{dec.lifetime}", "mental simulation")
        if self.hold is None:
            return "stopped: no hold keeps the gripper clear of the blade"
        opt = best.option
        # 4 lifetime: keep it whole until it should be cut
        if opt.prepare_at_h > 0:
            self.node.monitoring.append(SDT.MonitoringRecord("Store whole", opt.keep_whole_in,
                                                             f"kept in the {opt.keep_whole_in} (emulated: no PR2 "
                                                             "motion yet)", "as expected", "", ""))
            self.advance_time(opt.prepare_at_h, opt.keep_whole_in)
            best, dec = self.agent.decide()                       # re-decide from the updated belief before cutting
            self.node._answer("Does the plan still hold before I cut?", f"{dec.chosen}: {dec.why}",
                              "re-deliberation after the storage time", "mental simulation")
            opt = best.option
        # 5 cut, enforcing 'scored'
        self.do_cut()
        if self.integrity() == "partly_scored":
            self.do_cut()
        self.watch("CutRoundSeed", t_start)
        # 6 twist, enforcing 'halves apart' (re-cut a stuck fruit once)
        self.do_twist()
        if self.integrity() in ("scored", "partly_scored"):
            self.do_cut()
            self.do_twist()
        apart = self.integrity() in ("halved", "halved_torn")
        self.watch("TwistApart", t_start)
        # 7 seed out (the knife twist is excluded by the safety rule)
        if apart:
            self.do_spoon()
        out = self.node.seed == "out"
        self.watch("RemoveSeed", t_start)
        # 8 cavity clean: peel the seed-coat sheets while they are seen (at most two passes)
        passes = 0
        while out and opt.peel_coat and self.remnant > 0.05 and passes < 2:
            self.do_peel_coat()
            passes += 1
        if self.spoon_in_hand:
            self.put_spoon_back()
        # 9 servings; a half kept for later is treated at once
        served = []
        halves = [t for t in ("avocado_half_b", "avocado_half_a_pitted") if t in self.half_clocks]
        later = [h for p, h in m.servings if h > m.first_service_h]
        for k, (portion, at_h) in enumerate(m.servings):
            wait = t_start + at_h - self.clock_h
            if wait > 0:
                self.advance_time(wait, "counter")
            if portion == "whole":
                served += self.serve_halves(halves)
            else:
                tag = halves[0] if k == 0 else halves[-1]
                served += self.serve_halves([tag])
                if k == 0 and later:
                    self.keep_half(halves[-1], opt.kept_treatment)
        # 10 audit: were the goals achieved and the risks prevented?
        twist_torn = self.integrity() == "halved_torn"
        observed = dict(halves_apart=apart, seed_out=out, cavity_clean=out and self.remnant <= 0.05,
                        flesh_intact=apart and not twist_torn and not getattr(self, "gouged", False),
                        edible_at_service=all(s["quality"] >= 0.6 for s in served),
                        fresh_at_service=not any(s["browned"] for s in served), bruise=self.bruised,
                        torn=twist_torn, gouge=getattr(self, "gouged", False), testa_left=out and self.remnant > 0.05,
                        hazard=False, blade_near_hand=getattr(self, "clearance_mm", 99) < 1e3 * AG.CLEARANCE_MIN - 0.5,
                        spoiled=any(s["spoiled"] for s in served), browned=any(s["browned"] for s in served))
        audits = self.agent.audit(observed)
        bad = [a for a in audits if a.status in ("missed", "occurred")]
        return (f"{m.label}: {dec.chosen}. " + (f"Not as intended: {', '.join(a.effect for a in bad)}" if bad else
                                                "every goal achieved, every risk prevented"))

    # %% queued questions
    def do_should_i_squeeze(self):
        self.node.value_of_squeeze()
        return self.node.answers[-1].answer

    def do_guacamole(self):
        self.node.decide()
        return self.node.answers[-1].answer

    def do_when_best(self):
        ag = AG.AvocadoAgent(self.node, AG.mission("guacamole_now"))
        txt = ag.lifetime_summary()
        self.node._answer("When is it best to eat?", txt, "lifetime model (ripening at the storage temperature, ripe "
                          "window, decline, spoilage) over the belief", "lifetime analysis")
        return txt

    def do_what_if(self):
        best, _ = self.node.what_if_halved_now()
        self.node.how_to_do_it(best)
        return self.node.answers[-3].answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--do", default="", help="comma-separated commands to run headless, then exit")
    args, _ = ap.parse_known_args()
    k = AvocadoKitchen()
    say("avocado kitchen ready: " + ", ".join(c.label for c in k.commands.values()))
    if args.do:
        for n in args.do.split(","):
            k._last.clear()
            k.request(k.commands[n])
            k.serve(stop_when_idle=True)
        for c in k.commands.values():
            say(f"   {c.label:36s} available: {c.available[:80]:80s} last: {c.status[:120]}")
        if k.agent is not None:
            for d in k.agent.decisions:
                say(f"DECISION {d.chosen} | {d.why}")
            for a in k.agent.audits:
                say(f"AUDIT {a.kind:8s} {a.effect:18s} predicted {a.predicted:10s} observed {a.observed:4s} -> {a.status}")
        if k.node is not None:
            for m in k.node.monitoring:
                say(f"STEP {m.step:28s} {m.verdict:12s} {m.measured[:90]}")
        return k
    worker = threading.Thread(target=k.serve, daemon=True)
    worker.start()
    return k


if __name__ == "__main__":
    main()
