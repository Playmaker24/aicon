"""
Block Lifting: Graph Specification

Three estimators (end-effector, cube, grasp likelihood) between three simulated sensors and two actions
(end-effector velocity, gripper). The only goal is on the cube's height; reaching the cube, grasping it and lifting
it emerge from the gradients of that goal.

Authors: Adrian Pfisterer, Patrick Lowin, Vito Mengers
License: MIT
"""

__author__ = "Adrian Pfisterer, Patrick Lowin, Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import torch

from .actions import GripperAction, VeloEEAction
from .connections import build_connections
from .estimators import CubePositionEstimator, EEPoseEstimator, GraspedEstimator
from .goals import CubeGraspGoal, CubeLiftGoal
from .sensors import CubePoseSensor, EEForceSensor, EEPoseSensor

# initial arm joint positions (Panda, 7 joints); None keeps robosuite's default start pose
INITIAL_QPOS = None
# target height (world z) of the cube's centre [m]: the table top is at 0.8 m and the cube counts as lifted above
# 0.84 m (env.SUCCESS_HEIGHT), so the target lies well above and is not reached on the way
TARGET_HEIGHT = 1.0


def get_building_functions(env):
    """Builders for the tutorial graph, with the sensors reading from `env` (see env.make_env)."""
    connection_builders = build_connections()
    cpu, double = torch.device("cpu"), torch.double

    def connections(*names):
        return {k: connection_builders[k] for k in names}

    # the sequential middleware updates the components in this order
    component_builders = {
        "GripperAction": lambda mockbuild: GripperAction(
            "GripperAction", connections("GraspedLikelihood"), device=cpu, dtype=double, mockbuild=mockbuild
        ),
        "EEForceSensor": lambda mockbuild: EEForceSensor(
            "EEForceSensor", connections("GraspedLikelihood"), env, device=cpu, dtype=double, mockbuild=mockbuild
        ),
        "EEPosSensor": lambda mockbuild: EEPoseSensor(
            "EEPosSensor", connections("DirectMeasurement"), env, device=cpu, dtype=double, mockbuild=mockbuild
        ),
        "CubePosSensor": lambda mockbuild: CubePoseSensor(
            "CubePosSensor",
            connections("CubeDirectMeasurement"),
            env,
            device=cpu,
            dtype=double,
            mockbuild=mockbuild,
        ),
        "EEVelocities": lambda mockbuild: VeloEEAction(
            "EEVelocities", connections("ForwardKinematics"), device=cpu, dtype=double, mockbuild=mockbuild
        ),
        "EEPoseEstimator": lambda mockbuild: EEPoseEstimator(
            "EEPoseEstimator",
            connections("ForwardKinematics", "DirectMeasurement", "GraspedCubeKinematics", "GraspedLikelihood"),
            device=cpu,
            dtype=double,
            mockbuild=mockbuild,
        ),
        "CubePosEstimator": lambda mockbuild: CubePositionEstimator(
            "CubePosEstimator",
            connections("CubeDirectMeasurement", "GraspedCubeKinematics"),
            goals={
                "ReduceCubeHeightDifference": lambda d, t, mockbuild=False: CubeLiftGoal(
                    is_active=True, target_height=TARGET_HEIGHT, dtype=t, device=d, mockbuild=mockbuild
                )
            },
            device=cpu,
            dtype=double,
            mockbuild=mockbuild,
        ),
        "GraspLikelihoodEstimator": lambda mockbuild: GraspedEstimator(
            "GraspLikelihoodEstimator",
            connections("GraspedCubeKinematics", "GraspedLikelihood"),
            goals={
                "IncreaseGraspLikelihood": lambda d, t, mockbuild=False: CubeGraspGoal(
                    is_active=True, dtype=t, device=d, mockbuild=mockbuild
                )
            },
            device=cpu,
            dtype=double,
            mockbuild=mockbuild,
        ),
    }
    frame_rates = {k: 10 for k in component_builders}
    return component_builders, connection_builders, frame_rates