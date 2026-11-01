# -*- coding: utf-8 -*-
"""
Sensor-Action Components in AICON

A part of the robot that is commanded and measured at the same time (e.g. a gripper with finger encoders, or an arm
with its joint encoders) as a single component: its action quantities are set by the goal gradients, as in any
action component, and its measurement quantities are read from the robot, as in any sensor component. Both reach the
estimators through the component's active interconnections, e.g. a single kinematic model with which an estimator
predicts the state from the command and corrects it with the measurement.

Authors: Adrian Pfisterer
License: MIT
"""

__author__ = "Adrian Pfisterer"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import traceback
from abc import abstractmethod
from typing import Dict, List, Tuple, Union

import torch

from aicon_base.components.action_components import ActionComponent


class SensorActionComponent(ActionComponent):
    """An action component that also measures. Its quantities are the action quantities
    (`cls_action_names_and_shapes`) and the measurement quantities (`cls_measurement_names_and_shapes`). Implement
    `initialize_actions`, `obtain_measurements` (as in a sensor component), `determine_new_actions`,
    `send_action_values` and `_start` / `_stop`. In each update the component first measures, then decides on and
    sends its actions; the gradients that reach the measurement quantities are not used."""

    cls_action_names_and_shapes: Dict[str, Tuple[int]] = dict()
    cls_measurement_names_and_shapes: Dict[str, Tuple[int]] = dict()

    def initial_definitions(self, quantity_names_and_shapes: Union[Dict[str, Tuple[int]], None] = None) -> None:
        if quantity_names_and_shapes is None:
            quantity_names_and_shapes = {**self.cls_action_names_and_shapes, **self.cls_measurement_names_and_shapes}
        super().initial_definitions(quantity_names_and_shapes)

    def action_quantities(self) -> List[str]:
        return [k for k in self.quantities.keys() if k not in self.cls_measurement_names_and_shapes]

    @abstractmethod
    def initialize_actions(self) -> bool:
        """Set the initial values of the action quantities; return False to try again on the next update."""
        return False

    @abstractmethod
    def obtain_measurements(self) -> bool:
        """Write the measurement quantities; return whether a measurement was obtained."""
        return False

    def initialize_quantities(self) -> bool:
        return self.initialize_actions() and self.obtain_measurements()

    def _attempt_update(self, new_time: torch.tensor) -> None:
        try:
            with self.lock:
                self.obtain_measurements()
                self.determine_new_actions()
                if self.active:
                    self.send_action_values()
        except Exception:
            print("Could not update " + str(self.name) + ":")
            traceback.print_exc()
