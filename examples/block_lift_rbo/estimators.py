"""
Estimators for the Lift Example

Authors: Adrian Pfisterer
License: MIT
"""

__author__ = "Adrian Pfisterer"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Dict, Tuple

import torch

from aicon_base.components import EstimationComponent
from aicon_base.inference.ekf import predict_ekf, predict_ekf_other_quantity, update_ekf
from aicon_base.inference.util import gradient_preserving_clipping

__all__ = ["CubePositionEstimator", "EECubeDistEstimator", "EEPoseEstimator", "GraspedEstimator"]


class EEPoseEstimator(EstimationComponent):
    """End-effector position: predicted with the commanded velocity, corrected by the measured position (both through
    RobotKinematicsConnection)."""

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"pose_ee": (3,), "uncertainty_ee": (3, 3)}

    def initialize_quantities(self):
        c_kinematics = self.connections["RobotKinematics"]
        with c_kinematics.lock:
            if not c_kinematics.connected_quantities_initialized["ee_pos_meas"]:
                return False
            ee_pos_meas = c_kinematics.connected_quantities["ee_pos_meas"]
        self.quantities["pose_ee"] = ee_pos_meas
        self.quantities["uncertainty_ee"] = torch.eye(3, dtype=self.dtype, device=self.device) * 0.001
        return True

    def define_estimation_function_f(self):
        c_kinematics = self.connections["RobotKinematics"].c_func
        no_velocity = torch.zeros(3, dtype=self.dtype, device=self.device)

        def moved(pose_ee, action_velo_ee):
            # where the command moves the end-effector: the measurement for which the residual vanishes
            return -c_kinematics(pose_ee, action_velo_ee, torch.zeros_like(pose_ee))

        def measured(pose_ee, ee_pos_meas):
            return c_kinematics(pose_ee, no_velocity, ee_pos_meas)

        def f_func(pose_ee, uncertainty_ee, action_velo_ee, ee_pos_meas):
            eye = torch.eye(3, dtype=self.dtype, device=self.device)
            mu_pred, Sigma_pred = predict_ekf_other_quantity(
                moved, pose_ee, uncertainty_ee, action_velo_ee, torch.zeros_like(eye), eye * 0.01
            )
            mu_new, Sigma_new = update_ekf(measured, mu_pred, Sigma_pred, ee_pos_meas, eye * 0.01, eye * 0.01)
            Sigma_new = Sigma_new.detach()
            # The measurement is exact, so the value is set to it; the gradient still flows through the prediction,
            # i.e. towards the commanded velocity.
            mu_new = mu_new + measured(mu_new, ee_pos_meas).detach()
            return (mu_new, Sigma_new), (mu_new, Sigma_new)

        return f_func, ["pose_ee", "uncertainty_ee"], ["pose_ee", "uncertainty_ee"]


class CubePositionEstimator(EstimationComponent):
    """Position of the cube's center: from the direct measurement, and from the end-effector while grasped."""

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"position_cube": (3,), "uncertainty_cube": (3, 3)}

    def initialize_quantities(self):
        c_meas = self.connections["CubeDirectMeasurement"]
        with c_meas.lock:
            if not c_meas.connected_quantities_initialized["cube_pos_meas"]:
                return False
            cube_pos_meas = c_meas.connected_quantities["cube_pos_meas"]
        self.quantities["position_cube"] = cube_pos_meas
        self.quantities["uncertainty_cube"] = torch.eye(3, dtype=self.dtype, device=self.device) * 0.0001
        return True

    def define_estimation_function_f(self):
        c_grasped = self.connections["GraspedCubeKinematics"].c_func
        c_direct_meas = self.connections["CubeDirectMeasurement"].c_func

        def forward_cube(mu):
            return mu  # cubes do not move by themselves

        def f_func(
            position_cube, uncertainty_cube, pose_ee, uncertainty_ee, cube_pos_meas, likelihood_grasped_cube, dt
        ):
            eye = torch.eye(3, dtype=self.dtype, device=self.device)
            # a grasped cube moves with the end-effector
            forward_noise = likelihood_grasped_cube * 0.3 * dt + 0.001 * dt
            mu_new, Sigma_new = predict_ekf(forward_cube, position_cube, uncertainty_cube, eye * forward_noise)
            # The end-effector position as a measurement of the cube: it holds only while the cube is grasped, so its
            # uncertainty (end-effector and additive noise) grows with 1 / likelihood. Its derivatives are what lets
            # the goal reach the end-effector (while grasped) and the grasp (while not).
            mu_new, Sigma_new = update_ekf(
                c_grasped,
                mu_new,
                Sigma_new,
                pose_ee,
                uncertainty_ee / likelihood_grasped_cube,
                eye * 0.0001 / likelihood_grasped_cube,
                prevent_uncertainty_state_grad=False,
            )
            mu_new, Sigma_new = update_ekf(
                c_direct_meas,
                mu_new,
                Sigma_new,
                cube_pos_meas,
                eye * 0.0001,
                eye * 0.0001,
                prevent_uncertainty_state_grad=False,
            )
            return (mu_new, Sigma_new), (mu_new, Sigma_new)

        return f_func, ["position_cube", "uncertainty_cube"], ["position_cube", "uncertainty_cube"]


class EECubeDistEstimator(EstimationComponent):
    """Distance between the end-effector and the cube's center (see EECubeDistanceConnection)."""

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"distance_ee_cube": (1,)}

    def initialize_quantities(self):
        self.quantities["distance_ee_cube"] = torch.ones(1, dtype=self.dtype, device=self.device)
        return True

    def define_estimation_function_f(self):
        c_func = self.connections["EECubeDistance"].c_func

        def f_func(distance_ee_cube, pose_ee, position_cube):
            new_distance = distance_ee_cube + c_func(distance_ee_cube, pose_ee, position_cube)
            return (new_distance,), (new_distance,)

        return f_func, ["distance_ee_cube"], ["distance_ee_cube"]


class GraspedEstimator(EstimationComponent):
    """Likelihood that the cube is grasped (see GraspCubeConnection)."""

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"likelihood_grasped_cube": (1,)}

    def initialize_quantities(self):
        self.quantities["likelihood_grasped_cube"] = torch.ones(1, dtype=self.dtype, device=self.device) * 0.01
        return True

    def define_estimation_function_f(self):
        c_func = self.connections["GraspedLikelihood"].c_func

        def f_func(
            likelihood_grasped_cube,
            distance_ee_cube,
            gripper_activation,
            gripper_aperture_meas,
            gripper_aperture_rate_meas,
        ):
            innovation = c_func(
                likelihood_grasped_cube,
                distance_ee_cube,
                gripper_activation,
                gripper_aperture_meas,
                gripper_aperture_rate_meas,
            )
            new_likelihood = gradient_preserving_clipping(likelihood_grasped_cube + innovation, 1e-10, 0.9999999)
            return (new_likelihood,), (new_likelihood,)

        return f_func, ["likelihood_grasped_cube"], ["likelihood_grasped_cube"]
