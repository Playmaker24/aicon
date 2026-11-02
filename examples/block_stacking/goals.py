"""
Goals for the Lift Example

Authors: Adrian Pfisterer
License: MIT
"""

__author__ = "Adrian Pfisterer"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Union

import torch

from aicon_base.components import Goal


class CubeLiftedGoal(Goal):
    """The estimated cube center should be at least `target_height` [m] (world z). Above it the cost is zero, so the
    system holds the cube there, and the cost returns if the cube drops."""

    def __init__(
        self,
        is_active: bool,
        target_height: float,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        self.target_height = target_height
        super().__init__(name="LiftCube", is_active=is_active, dtype=dtype, device=device, mockbuild=mockbuild)

    def define_goal_cost_function(self):
        target_height = torch.tensor(self.target_height, dtype=self.dtype, device=self.device)

        def goal_func(position_cube):
            cost = 10.0 * torch.relu(target_height - position_cube[2:3])
            return cost, cost

        return goal_func




class CubeStackedGoal(Goal):
    """The estimated center of cube A should reach a target 3D position."""

    def __init__(
        self,
        is_active: bool,
        target_position,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        self.target_position = target_position
        super().__init__(
            name="StackCube",
            is_active=is_active,
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_goal_cost_function(self):
        def goal_func(position_cube):
            target = self.target_position() if callable(self.target_position) else self.target_position
            target = torch.as_tensor(target, dtype=self.dtype, device=self.device).reshape(3)
            cost = 10.0 * torch.linalg.vector_norm(position_cube - target)
            return cost.reshape(1), cost.reshape(1)

        return goal_func


class StackLikelihoodGoal(Goal):
    """Cube A rests on cube B and has been let go: the stack likelihood should be 1. Once it is close to 1 the goal is
    fulfilled and switches itself off, so the robot stops."""

    def __init__(
        self,
        is_active: bool,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name="StackCube",
            is_active=is_active,
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
            fulfilled_value=1.0,  # cost 10 * (1 - likelihood) <= 1, i.e. likelihood >= 0.9
        )

    def define_goal_cost_function(self):
        def goal_func(likelihood_stacked_cube):
            cost = 10.0 * (1.0 - likelihood_stacked_cube)
            return cost, cost

        return goal_func