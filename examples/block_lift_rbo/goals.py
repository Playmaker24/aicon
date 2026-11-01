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