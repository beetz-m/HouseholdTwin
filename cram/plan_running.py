"""Running a list of coraplex actions as one plan on the kinematically simulated PR2.

CRAM main from November 2026 has no plan tree, no ``sequential`` factory and no shared ``Context``: a plan is a cramph
``Sequence`` of actions, compiled with an EndMotion into one statechart and run by a plan executor, and a statechart
context serves a single executor. Every call therefore builds a fresh context and executor over the same world.
Collision avoidance stays off, as under the former ``kinematically_simulated_robot``.
"""
from __future__ import annotations

from coraplex.plans.context_extensions import RobotAccess, StatementGrounding
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from cramph.composites import Sequence
from cramph.context import StatechartContext
from cramph.statechart import Statechart


def perform_steps(world, robot, steps, collision_avoidance=False, extensions=()):
    """Run ``steps`` (coraplex actions or other cramph nodes) in order until all succeeded; a failed step raises.

    :param world: The world the robot acts in.
    :param robot: The robot performing the steps.
    :param steps: The plan's steps, in order.
    :param collision_avoidance: Whether the robot avoids its surroundings and itself.
    :param extensions: Executor extensions, e.g. a visualization.
    :return: The plan node that ran.
    """
    context = StatechartContext.with_extensions(world, [RobotAccess(robot), StatementGrounding()])
    executor = KinematicallySimulatedPlanExecutor(context, collision_avoidance=collision_avoidance,
                                                  extensions=list(extensions))
    plan = Sequence(nodes=list(steps))
    statechart = Statechart(context=executor.context)
    statechart.add_node(plan)
    executor.compile(statechart)
    executor.execute()
    executor.close()
    return plan


def perform_between_world_edits(perform, nodes):
    """Run ``nodes`` with ``perform`` (a function running a list of plan nodes as one plan), but call each step
    (ccram.cram_nodes.step, a cramph FunctionCall) between plans instead of inside one.

    The twins' steps change the kinematic structure of the world: they attach a grasped body to the gripper with a
    fixed connection, remove an opened egg, add its contents and shell halves. Inside one statechart Giskard's
    controller still commands the old degrees of freedom until the statechart recompiles, which fails ("Commands
    length 129 does not match number of free variables 122"). Between plans, as with the former ExecutionBoundaryNode,
    every plan is compiled against the world as the steps before it left it.

    :param perform: Runs a list of plan nodes as one plan.
    :param nodes: The realisation: plan nodes and steps, in order.
    """
    from cramph.threaded_nodes import FunctionCall
    motions = []
    for node in nodes:
        if isinstance(node, FunctionCall):
            if motions:
                perform(motions)
                motions = []
            node.function()
        else:
            motions.append(node)
    if motions:
        perform(motions)
