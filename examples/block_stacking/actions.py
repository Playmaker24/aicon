"""
Robot Interface for the Lift Example

The arm and the gripper are each commanded and measured at the same time, so each is one sensor-action component:
the arm commands the end-effector velocity and measures the end-effector position, the gripper commands its
activation and measures the distance between its fingers. The simulation loop reads the action quantities and passes
them to the environment, so the components send nothing themselves.

Authors: Adrian Pfisterer
License: MIT
"""

__author__ = "Adrian Pfisterer"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Callable, Dict, Tuple, Union

import torch

from .sensor_action_components import SensorActionComponent

__all__ = ["GripperProprioAct", "RobotProprioAct"]


class SimulatedSensorAction(SensorActionComponent):
    """A sensor-action component that reads its measurements from the simulation environment `env`, and whose action
    values the simulation loop reads from `self.quantities`."""

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

    def _start(self):
        pass

    def _stop(self):
        pass

    def send_action_values(self) -> None:
        pass


class RobotProprioAct(SimulatedSensorAction):
    """The arm: commands the translational end-effector velocity (the input of robosuite's OSC_POSITION controller)
    and measures the end-effector position.

    The velocity is a step against the combined goal gradient, i.e. the steepest path with the other paths projected
    into its nullspace. The gradient's magnitude passes through many estimators and says little about distances, so
    only its direction is used. The step length adapts as in Rprop: it grows while consecutive gradients agree and
    halves when the gradient turns, which it does when the end-effector passes or overshoots the point the gradient
    leads to."""

    cls_action_names_and_shapes: Dict[str, Tuple[int]] = {"action_velo_ee": (3,)}
    cls_measurement_names_and_shapes: Dict[str, Tuple[int]] = {"ee_pos_meas": (3,)}
    step_min = 0.002
    step_max = 0.25
    step_increase = 1.2
    step_decrease = 0.5
    # cosine between consecutive gradient directions below which the gradient counts as turning (about 25 degrees)
    turn_cosine = 0.9

    def initialize_actions(self) -> bool:
        self.quantities["action_velo_ee"] = torch.zeros(3, dtype=self.dtype, device=self.device)
        self.step = 0.1
        self.last_direction = torch.zeros(3, dtype=self.dtype, device=self.device)
        return True

    def obtain_measurements(self) -> bool:
        self.quantities["ee_pos_meas"] = self.measurement(self.env.env.get_ee_pos())
        return True

    def determine_new_actions(self):
        grad = torch.nan_to_num(
            self.get_steepest_plus_projected("action_velo_ee", time_threshold=self.timestamp - 0.5)[:3]
        )
        if torch.norm(grad) == 0:
            self.quantities["action_velo_ee"] = torch.zeros(3, dtype=self.dtype, device=self.device)
            self.last_direction = torch.zeros(3, dtype=self.dtype, device=self.device)
            self.used_gradients = ([], [])
            return
        direction = -grad / torch.norm(grad)
        if torch.norm(self.last_direction) > 0:
            if torch.dot(direction, self.last_direction) > self.turn_cosine:
                self.step = min(self.step * self.step_increase, self.step_max)
            else:
                self.step = max(self.step * self.step_decrease, self.step_min)
        self.last_direction = direction
        self.quantities["action_velo_ee"] = direction * self.step


class GripperProprioAct(SimulatedSensorAction):
    """The gripper: commands its activation, 0.01 (open) or 0.99 (closed), switched by the sign of the steepest goal
    gradient, and measures the distance between the fingers (finger joint encoders) and its rate of change."""

    cls_action_names_and_shapes: Dict[str, Tuple[int]] = {"gripper_activation": (1,)}
    cls_measurement_names_and_shapes: Dict[str, Tuple[int]] = {
        "gripper_aperture_meas": (1,),
        "gripper_aperture_rate_meas": (1,),
    }

    def initialize_actions(self) -> bool:
        self.quantities["gripper_activation"] = torch.ones(1, dtype=self.dtype, device=self.device) * 0.01
        return True

    def obtain_measurements(self) -> bool:
        self.quantities["gripper_aperture_meas"] = self.measurement([self.env.env.get_gripper_aperture()])
        self.quantities["gripper_aperture_rate_meas"] = self.measurement([self.env.env.get_gripper_aperture_rate()])
        return True

    def determine_new_actions(self):
        steepest_grad, _, _ = self.get_steepest_gradient("gripper_activation")
        if steepest_grad < -1e-15 and self.quantities["gripper_activation"] != 0.99:
            print("Closing the gripper")
            self.quantities["gripper_activation"] = torch.ones(1, dtype=self.dtype, device=self.device) * 0.99
        elif steepest_grad > 1e-15 and self.quantities["gripper_activation"] != 0.01:
            print("Opening the gripper")
            self.quantities["gripper_activation"] = torch.ones(1, dtype=self.dtype, device=self.device) * 0.01
