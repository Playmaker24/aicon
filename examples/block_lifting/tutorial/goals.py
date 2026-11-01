"""
Goals for the Simulated Block Lifting Example

Authors: Adrian Pfisterer, Patrick Lowin, Vito Mengers
License: MIT
"""

__author__ = "Adrian Pfisterer, Patrick Lowin, Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Union

import torch

from aicon_base.components import Goal


class CubeLiftGoal(Goal):
    """Bring the estimated height (z coordinate) of the cube to `target_height` [m]."""

    def __init__(
        self,
        is_active: bool,
        target_height: float,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        self.target_height = target_height
        super().__init__(
            name="ReduceCubeHeightDifference", is_active=is_active, dtype=dtype, device=device, mockbuild=mockbuild
        )

    def define_goal_cost_function(self):
        target_height = torch.tensor(self.target_height, dtype=self.dtype, device=self.device)

        def goal_func(position_cube):
            cost = 10.0 * torch.abs(position_cube[2:3] - target_height)
            return cost, cost

        return goal_func


class CubeGraspGoal(Goal):
    """Bring the grasp likelihood of the cube to 1, i.e. hold the cube."""

    def __init__(
        self,
        is_active: bool,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name="IncreaseGraspLikelihood", is_active=is_active, dtype=dtype, device=device, mockbuild=mockbuild
        )

    def define_goal_cost_function(self):
        def goal_func(likelihood_grasped_cube):
            cost = 10.0 * torch.abs(likelihood_grasped_cube - 1.0)
            return cost, cost

        return goal_func