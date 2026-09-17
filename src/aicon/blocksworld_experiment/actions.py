import random
from enum import Enum
from typing import Dict, Callable, Union

import torch

from aicon.base_classes.components import ActionComponent
from aicon.base_classes.connections import ActiveInterconnection
from aicon.base_classes.util import collect_derivatives
from aicon.blocksworld_experiment.global_params import NUM_BLOCKS


class ActionSelection(Enum):
    """How a candidate action is picked from the gradients."""

    #: As originally shipped. Rejected candidates are retired by writing 0.0, which
    #: makes them indistinguishable from candidates that never had a gradient, so
    #: once the positive ones run out this can keep reselecting zeros forever.
    #: The selection itself is unchanged, but the loop now proves when it can no
    #: longer succeed (every selectable candidate examined and rejected) and raises
    #: instead of spinning. ActionComponent._attempt_update catches that, so it is
    #: reported once per step rather than ending the run; proving exhaustion means
    #: sampling every candidate, which costs roughly a minute over 200 steps.
    ORIGINAL = "original"

    #: Zero-gradient actions may still be taken at random, but each candidate is
    #: retired at most once, so the search always terminates.
    RANDOM_WALK = "random_walk"

    #: Only actions whose gradient actually reduces the cost are eligible.
    POSITIVE_ONLY = "positive_only"


class BlockPuttingAction(ActionComponent):

    def __init__(self, name: str, connections: Dict[str, ActiveInterconnection], goals : Union[None,Dict[str, Callable]] = None,
                 dtype:torch.dtype=torch.get_default_dtype(), device:torch.device=torch.get_default_device(),
                 mockbuild : bool = False, send_action_func :  Union[None,Dict[str, Callable]] = None):
        super().__init__(name, connections, goals=goals, dtype=dtype, device=device, mockbuild=mockbuild)
        if mockbuild:
            return

        self.internal_action = None
        self.action_selection = ActionSelection.POSITIVE_ONLY
        self.send_actions = False
        self.wait_for_action_counter = 0
        self.tried_actions = dict()
        self._send_action_func = send_action_func

    def _start(self):
        self.send_actions = (self._send_action_func is not None)

    def _stop(self):
        self.send_actions = False

    def send_action_values(self) -> None:
        if self.internal_action is not None and self.send_actions:
            self._send_action_func(self.internal_action)

    def determine_new_actions(self):
        try:
            _, relevant_backward_derivatives = collect_derivatives(self.connections)
            my_derivatives = relevant_backward_derivatives["action_blocks"]
        except KeyError:
            print("No Derivatives for Actions yet!")
            return
        if self.wait_for_action_counter > 5:
            oldest_stamp_for_each_grad = torch.cat(
                [torch.min(torch.cat(stamps)).unsqueeze(0) for stamps in my_derivatives.timestamps])
            age_of_newest_grad = torch.tensor(self.timestamp, dtype=self.dtype, device=self.device) - torch.max(oldest_stamp_for_each_grad)
            if age_of_newest_grad > 0.2:
                return
            gradient_steepness = my_derivatives.derivatives_tensor
            with self.connections["BelowLikelihoodDummySensing"].lock:
                current_state = self.connections["BelowLikelihoodDummySensing"].connected_quantities["below_state_sensed"]
                already_tried_mask = torch.zeros((2, NUM_BLOCKS, NUM_BLOCKS), dtype=torch.bool, device=self.device)
                for k, v in self.tried_actions.items():
                    if torch.equal(k, current_state):
                        already_tried_mask = v
                clear = 1 - torch.clip(torch.norm(current_state, dim=0), 0, 1)
            gradient_steepness[:, :, already_tried_mask] = 0.0
            choosen_gradient_idx = None
            found_action = False
            as_shipped = self.action_selection is ActionSelection.ORIGINAL
            retired = torch.zeros_like(gradient_steepness, dtype=torch.bool)
            minus_inf = torch.full_like(gradient_steepness, float("-inf"))
            while True:
                if as_shipped:
                    if torch.sum(torch.abs(gradient_steepness)) <= 0:
                        break
                    remaining = gradient_steepness
                else:
                    remaining = torch.where(retired, minus_inf, gradient_steepness)
                steepest = torch.max(remaining)
                # a negative gradient means the action increases the cost, and a zero
                # gradient means it does not advance the goal at all
                if not as_shipped and (steepest < 0 or (steepest == 0
                        and self.action_selection is ActionSelection.POSITIVE_ONLY)):
                    break
                possible_steepest_gradient_idxs = (remaining == steepest).nonzero()
                n_idxs = possible_steepest_gradient_idxs.shape[0]
                choosen_gradient_idx = possible_steepest_gradient_idxs[random.randint(0, n_idxs-1)]
                action_idx = choosen_gradient_idx[2:5]
                if action_idx[0] == 0:
                    if clear[action_idx[1]] and clear[action_idx[2]] and current_state[action_idx[1], action_idx[2]] == 0:
                        found_action = True
                        break
                elif action_idx[0] == 1:
                        if clear[action_idx[1]] and current_state[action_idx[1], action_idx[2]] == 1:
                            found_action = True
                            break
                retired[tuple(choosen_gradient_idx)] = True
                if as_shipped:
                    gradient_steepness[tuple(choosen_gradient_idx)] = 0.0
                    # Legality does not depend on the gradients, and this loop only
                    # ever changes them by zeroing, so once every entry that can still
                    # be selected has been examined and rejected, no later iteration
                    # can succeed either: it would spin forever.
                    selectable = gradient_steepness == torch.max(gradient_steepness)
                    if bool(torch.all(retired[selectable])):
                        raise RuntimeError(
                            f"action selection cannot terminate: all {int(selectable.sum())} "
                            "currently selectable candidates were examined and rejected, "
                            "but retiring a candidate by zeroing it leaves it selectable")
            if not found_action:
                print("No gradients available!")
                self.internal_action = None
                return
            self.internal_action = torch.zeros((2, NUM_BLOCKS, NUM_BLOCKS), dtype=self.dtype, device=self.device)
            self.internal_action[tuple(action_idx)] = 1
            self.tried_actions[current_state] = torch.logical_or(already_tried_mask, self.internal_action)
            print(self.internal_action)
            self.wait_for_action_counter = 0
        else:
            self.wait_for_action_counter += 1
            self.internal_action = None

    def initial_definitions(self):
        self.quantities["action_blocks"] = torch.zeros((2, NUM_BLOCKS,NUM_BLOCKS), dtype=self.dtype, device=self.device)

    def initialize_quantities(self) -> bool:
        self.quantities["action_blocks"] = torch.ones((2, NUM_BLOCKS, NUM_BLOCKS), dtype=self.dtype, device=self.device) * 1.0
        return True