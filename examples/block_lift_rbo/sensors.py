"""
Simulated Sensors for the Lift Example

The cube sensor reads the cube's ground-truth position from the simulation. The robot's own measurements come with
its actions (see actions.py).

Authors: Adrian Pfisterer
License: MIT
"""

__author__ = "Adrian Pfisterer"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Callable, Dict, Tuple, Union

import torch

from aicon_base.components import SensorComponent

__all__ = ["CubePoseSensor"]


class SimulationSensor(SensorComponent):
    """A sensor that reads from the simulation environment `env`."""

    def __init__(
        self,
        name: str,
        connections: Dict[str, Callable],
        env,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        self.env = env
        super().__init__(name, connections, dtype=dtype, device=device, mockbuild=mockbuild)

    def measurement(self, values) -> torch.Tensor:
        return torch.as_tensor(values, dtype=self.dtype, device=self.device).reshape(-1).clone()


class CubePoseSensor(SimulationSensor):
    """Position of the cube's center."""

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"cube_pos_meas": (3,)}

    def obtain_measurements(self) -> bool:
        self.quantities["cube_pos_meas"] = self.measurement(self.env.env.get_cube_pos())
        return True
