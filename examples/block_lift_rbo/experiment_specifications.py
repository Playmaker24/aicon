"""
Lift: Graph Specification

Four estimators (end-effector, cube, their distance, grasp likelihood) between the cube's position sensor and the
robot: the arm and the gripper are each one sensor-action component, commanded and measured at once. The only goal
is on the estimated height of the cube; reaching the cube, closing the gripper around it and lifting it emerge from
the gradients of that goal.

Authors: Adrian Pfisterer
License: MIT
"""

__author__ = "Adrian Pfisterer"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import torch

from .actions import GripperProprioAct, RobotProprioAct
from .connections import build_connections
from .estimators import CubePositionEstimator, EECubeDistEstimator, EEPoseEstimator, GraspedEstimator
from .goals import CubeLiftedGoal
from .sensors import CubePoseSensor

# height of the table top in robosuite's Lift task [m]
TABLE_HEIGHT = 0.8
# the goal: the cube's center at least this far above the table top [m] (it rests at about 0.02)
LIFT_HEIGHT = 0.2


def get_building_functions(env, lift_height=LIFT_HEIGHT):
    """Builders for the lift graph, with the sensors reading from `env` (see env.make_env)."""
    connection_builders = build_connections()
    cpu, double = torch.device("cpu"), torch.double

    def connections(*names):
        return {k: connection_builders[k] for k in names}

    # the sequential middleware updates the components in this order
    component_builders = {
        "GripperProprioAct": lambda mockbuild: GripperProprioAct(
            "GripperProprioAct", connections("GraspedLikelihood"), env, device=cpu, dtype=double, mockbuild=mockbuild
        ),
        "RobotProprioAct": lambda mockbuild: RobotProprioAct(
            "RobotProprioAct", connections("RobotKinematics"), env, device=cpu, dtype=double, mockbuild=mockbuild
        ),
        "CubePosSensor": lambda mockbuild: CubePoseSensor(
            "CubePosSensor", connections("CubeDirectMeasurement"), env, device=cpu, dtype=double, mockbuild=mockbuild
        ),
        "EEPoseEstimator": lambda mockbuild: EEPoseEstimator(
            "EEPoseEstimator",
            connections("RobotKinematics", "GraspedCubeKinematics", "EECubeDistance"),
            device=cpu,
            dtype=double,
            mockbuild=mockbuild,
        ),
        "EECubeDistEstimator": lambda mockbuild: EECubeDistEstimator(
            "EECubeDistEstimator",
            connections("EECubeDistance", "GraspedLikelihood"),
            device=cpu,
            dtype=double,
            mockbuild=mockbuild,
        ),
        "GraspLikelihoodEstimator": lambda mockbuild: GraspedEstimator(
            "GraspLikelihoodEstimator",
            connections("GraspedCubeKinematics", "GraspedLikelihood"),
            device=cpu,
            dtype=double,
            mockbuild=mockbuild,
        ),
        "CubePosEstimator": lambda mockbuild: CubePositionEstimator(
            "CubePosEstimator",
            connections("CubeDirectMeasurement", "GraspedCubeKinematics", "EECubeDistance"),
            goals={
                "LiftCube": lambda d, t, mockbuild=False: CubeLiftedGoal(
                    is_active=True, target_height=TABLE_HEIGHT + lift_height, dtype=t, device=d, mockbuild=mockbuild
                )
            },
            device=cpu,
            dtype=double,
            mockbuild=mockbuild,
        ),
    }
    frame_rates = {k: 10 for k in component_builders}
    return component_builders, connection_builders, frame_rates
