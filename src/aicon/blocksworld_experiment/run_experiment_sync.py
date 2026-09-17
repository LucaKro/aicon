"""
Synchronous experiment runner for the blocks world domain.
This module provides functionality to run blocks world experiments in a synchronous manner,
without ROS middleware, for testing and development purposes.
"""

import random
from typing import Callable

import numpy as np
import torch

from aicon.blocksworld_experiment.actions import ActionSelection
from aicon.blocksworld_experiment.experiment_specifications import get_building_functions_basic_blocks_world
from aicon.middleware.python_sequential import build_components, run_component_sequence


class Run:
    """Parameters for one run. Subclass and override what differs."""
    setup = "3towers"       # "3towers" or "0towers"
    goal = "StackAonBonC"   # StackAonB, StackAonBonC, SmartStackAonBonC, UnstackAonB
    required_signs = True   # False clears BelowLikelihood's required_signs_dict
    action_selection = ActionSelection.ORIGINAL   # as shipped; see ActionSelection
    steps = 200
    seed = 111


class Default(Run):
    """The original experiment. Solves at step 59 in 8 moves."""


class NoSigns(Default):
    """Without the signs, action selection as shipped: after some moves the loop in
    BlockPuttingAction can no longer terminate. The exhaustion check reports this once
    per step instead of hanging."""
    required_signs = False


class NoSignsFixedLoop(NoSigns):
    """Same, with the loop fixed so it always terminates. It still never solves: the
    failure without the signs is not just the loop, no gradient supports the subgoal."""
    action_selection = ActionSelection.POSITIVE_ONLY


EXPERIMENTS = [Default, NoSigns, NoSignsFixedLoop]


def start_all_components(functions_constructor: Callable):
    """
    Initialize all components for the blocks world experiment.

    Args:
        functions_constructor: Function that returns component builders, connection builders, and frame rates

    Returns:
        Dictionary of initialized components
    """
    component_builders, connection_builders, frame_rates = functions_constructor()
    return build_components(component_builders)


def run_experiment(functions_constructor: Callable, cfg=Default):
    """
    Run a blocks world experiment synchronously.

    Args:
        functions_constructor: Function that returns component builders, connection builders, and frame rates
        cfg: Run subclass holding the parameters for this run
    """
    components = start_all_components(functions_constructor=functions_constructor)
    components["BlockPuttingAction"].action_selection = cfg.action_selection
    if not cfg.required_signs:
        for component in components.values():
            for connection in component.connections.values():
                connection.required_signs_dict = {}

    def determine_goals_fulfilled(cs):
        """
        Check if all active goals in the components are fulfilled.

        Args:
            cs: Dictionary of components

        Returns:
            True if all goals are fulfilled, False otherwise
        """
        for _, comp in cs.items():
            for _, goal in comp.goals.items():
                if goal.is_active:
                    return False
        return True

    # Run the experiment for up to cfg.steps timesteps or until goals are fulfilled
    solved_at = None
    for i in range(cfg.steps):
        if determine_goals_fulfilled(components):
            solved_at = i
            break
        components = run_component_sequence(components, torch.tensor(i * 0.01))
        action = components["BlockPuttingAction"].internal_action
        if action is not None:
            components["BlocksBelowSensor"].take_action(action)

    moves = components["BlocksBelowSensor"].taken_valid_actions
    print(f"SUMMARY {cfg.__name__}: solved_at={solved_at} moves={moves}\n")
    return solved_at


def run_blocksworld(cfg=Default):
    torch.random.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    print(f"=== {cfg.__name__}: setup={cfg.setup} goal={cfg.goal} "
          f"required_signs={cfg.required_signs} ===")

    setup_func = lambda: get_building_functions_basic_blocks_world(init_setup=cfg.setup, goal=cfg.goal)
    return run_experiment(functions_constructor=setup_func, cfg=cfg)


if __name__ == "__main__":
    # Set up deterministic behavior for reproducibility
    torch.set_default_dtype(torch.float64)
    torch._dynamo.config.capture_func_transforms = True
    torch.use_deterministic_algorithms(True)
    torch.set_printoptions(profile="full", precision=20)
    torch.autograd.set_detect_anomaly(True)

    # one class, or e.g. EXPERIMENTS to run them all in sequence
    for case in [Default]:
        run_blocksworld(case)
