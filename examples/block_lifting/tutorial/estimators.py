"""
Estimators for the Block Lifting Example

Authors: Adrian Pfisterer, Patrick Lowin, Vito Mengers
License: MIT
"""

__author__ = "Adrian Pfisterer, Patrick Lowin, Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Dict, Tuple

import torch

from aicon_base.components import EstimationComponent
from aicon_base.inference.ekf import (
    update_ekf,
    predict_ekf_other_quantity,
    predict_ekf,
    update_switching_ekf,
)
from aicon_base.inference.util import gradient_preserving_clipping


class EEPoseEstimator(EstimationComponent):
    """End-effector position: predicted with the commanded velocity, corrected by the measured position."""

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"pose_ee": (3,), "uncertainty_ee": (3, 3)}

    def initialize_quantities(self):
        c_proprio = self.connections["DirectMeasurement"]
        with c_proprio.lock:
            if not c_proprio.connected_quantities_initialized["ee_pos_meas"]:
                return False
            pose_measured_ee = c_proprio.connected_quantities["ee_pos_meas"]
        self.quantities["pose_ee"] = pose_measured_ee
        self.quantities["uncertainty_ee"] = torch.eye(3, dtype=self.dtype, device=self.device) * 0.001
        return True

    def define_estimation_function_f(self):
        c_proprio = self.connections["DirectMeasurement"].c_func
        c_action = self.connections["ForwardKinematics"].c_func

        def f_func(pose_ee, uncertainty_ee, ee_pos_meas, action_velo_ee):
            eye = torch.eye(3, dtype=self.dtype, device=self.device)
            mu_pred, Sigma_pred = predict_ekf_other_quantity(
                c_action,
                pose_ee,
                uncertainty_ee,
                action_velo_ee,
                torch.zeros(3, 3, dtype=self.dtype, device=self.device),
                eye * 0.01,
            )
            mu_new, Sigma_new = update_ekf(c_proprio, mu_pred, Sigma_pred, ee_pos_meas, eye * 0.01, eye * 0.01)
            Sigma_new = Sigma_new.detach()
            # The measurement is exact, so the value is set to it; the gradient still flows through the prediction,
            # i.e. towards the commanded velocity.
            mu_new = mu_new + c_proprio(mu_new, ee_pos_meas).detach()
            return (mu_new, Sigma_new), (mu_new, Sigma_new)

        return f_func, ["pose_ee", "uncertainty_ee"], ["pose_ee", "uncertainty_ee"]


class CubePositionEstimator(EstimationComponent):
    """Position of the cube: from the direct measurement, and from the end-effector while grasped."""

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"position_cube": (3,), "uncertainty_cube": (3, 3)}

    def initialize_quantities(self):
        c_meas = self.connections["CubeDirectMeasurement"]
        with c_meas.lock:
            if not c_meas.connected_quantities_initialized["cube_pos_meas"]:
                return False
            cube_pos_meas = c_meas.connected_quantities["cube_pos_meas"]
        self.quantities["position_cube"] = cube_pos_meas
        self.quantities["uncertainty_cube"] = torch.eye(3, dtype=self.dtype, device=self.device) * 0.001
        return True

    def define_estimation_function_f(self):
        c_grasped = self.connections["GraspedCubeKinematics"].c_func
        c_direct_meas = self.connections["CubeDirectMeasurement"].c_func

        def forward_cube(mu):
            return mu  # a cube does not move by itself

        def f_func(
            position_cube, uncertainty_cube, pose_ee, uncertainty_ee, cube_pos_meas, likelihood_grasped_cube, dt
        ):
            eye = torch.eye(3, dtype=self.dtype, device=self.device)
            # a grasped cube may move with the end-effector
            forward_noise = likelihood_grasped_cube * 0.15 * dt + 0.005 * dt
            #print("LIKEIHOOD GRASPED CUBE:", likelihood_grasped_cube)
            mu_new, Sigma_new = predict_ekf(forward_cube, position_cube, uncertainty_cube, eye * forward_noise)
            mu_new, Sigma_new = update_switching_ekf(
                c_grasped,
                mu_new,
                Sigma_new,
                pose_ee,
                uncertainty_ee,
                eye * 0.03,
                likelihood_grasped_cube,
                outlier_rejection_threshold=0.05,
                prevent_uncertainty_state_grad=False,
            )
            mu_new, Sigma_new = update_ekf(
                c_direct_meas,
                mu_new,
                Sigma_new,
                cube_pos_meas,
                eye * 0.03,
                eye * 0.03,
                prevent_uncertainty_state_grad=False,
            )
            return (mu_new, Sigma_new), (mu_new, Sigma_new)

        return f_func, ["position_cube", "uncertainty_cube"], ["position_cube", "uncertainty_cube"]


class GraspedEstimator(EstimationComponent):
    """Likelihood that the cube is grasped."""

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"likelihood_grasped_cube": (1,)}

    def initialize_quantities(self):
        self.quantities["likelihood_grasped_cube"] = torch.ones(1, dtype=self.dtype, device=self.device) * 0.01
        return True

    def define_estimation_function_f(self):
        c_func = self.connections["GraspedLikelihood"].c_func

        def f_func(likelihood_grasped_cube, pose_ee, position_cube, ee_force_mag_meas, gripper_activation):
            innovation = c_func(
                likelihood_grasped_cube, pose_ee, position_cube, ee_force_mag_meas, gripper_activation
            )
            new_likelihood = gradient_preserving_clipping(likelihood_grasped_cube + innovation, 1e-10, 0.9999999)
            return (new_likelihood,), (new_likelihood,)

        return f_func, ["likelihood_grasped_cube"], ["likelihood_grasped_cube"]