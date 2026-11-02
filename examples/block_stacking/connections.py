"""
Active Interconnections for the Lift Example

Authors: Adrian Pfisterer
License: MIT
"""

__author__ = "Adrian Pfisterer"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Union

import torch

from aicon_base.connections import ActiveInterconnection

# Finger distance [m] above which the gripper is open (about 0.078 when fully open) and below which it is closed on
# nothing (about 0.002); closed on the cube it is about 0.05.
OPEN_APERTURE = 0.07
EMPTY_APERTURE = 0.02
# Below this rate of change of the finger distance [m/s] the fingers stand still (they close at about 0.15 m/s).
STILL_APERTURE_RATE = 0.02
# Length scales [m] of the grasp model: from where the hand can reach the cube, and how close to the cube's center
# the gripper has to be for the cube to lie between the fingers.
REACH_SCALE = 0.15
BETWEEN_FINGERS_SCALE = 0.005
# Largest distance [m] between the gripper's center and the cube's center at which the fingers can hold the cube
# (about half its diagonal, a cube that tipped over is held off-center).
HOLDING_DISTANCE = 0.03
# Fraction of the gap between the grasp evidence and the likelihood closed per tick.
GRASP_EVIDENCE_RATE = 0.3
# Stacking: cube A's center is this far above cube B's center when A rests on B (half sizes 0.02 + 0.025) [m], and A
# is carried this much higher than that while it is not yet above B.
STACK_HEIGHT = 0.045
STACK_CLEARANCE = 0.06
# Length scales [m] of the stack model: the wide ones draw A to the place above B, the narrow one tells when A is above
# B so that it may come down, and the last one how exactly A has to rest on B for the release.
STACK_REACH_HORIZONTAL_SCALE = 0.15
STACK_REACH_VERTICAL_SCALE = 0.15
STACK_ALIGN_SCALE = 0.01
STACK_PLACE_HEIGHT_SCALE = 0.01
# A counts as placed (evidence) within these distances [m] of the stack position
PLACED_HORIZONTAL_TOLERANCE = 0.02
PLACED_VERTICAL_TOLERANCE = 0.012
# Fraction of the gap between the placement evidence and the likelihood closed per tick.
STACK_EVIDENCE_RATE = 0.3


def build_connections():
    return {
        "GraspedCubeKinematics": lambda device, dtype, mockbuild: EECubeGraspedConnection(
            "GraspedCubeKinematics", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "GraspedLikelihood": lambda device, dtype, mockbuild: GraspCubeConnection(
            "GraspedLikelihood", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "EECubeDistance": lambda device, dtype, mockbuild: EECubeDistanceConnection(
            "EECubeDistance", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "RobotKinematics": lambda device, dtype, mockbuild: RobotKinematicsConnection(
            "RobotKinematics", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "CubeDirectMeasurement": lambda device, dtype, mockbuild: CubeDirectMeasurementConnection(
            "CubeDirectMeasurement", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "StackLikelihood": lambda device, dtype, mockbuild: StackCubeConnection(
            "StackLikelihood", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "CubeBDirectMeasurement": lambda device, dtype, mockbuild: CubeBDirectMeasurementConnection(
            "CubeBDirectMeasurement", device=device, dtype=dtype, mockbuild=mockbuild
        ),
    }


class RobotKinematicsConnection(ActiveInterconnection):
    """The robot's own motion: the commanded velocity moves the end-effector (one tick of the action's time step),
    and the measured end-effector position is where it is. As a residual: the measured position minus the estimated
    one moved by the command (no command, zero velocity, compares the measurement with the estimate)."""

    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"pose_ee": (3,), "uncertainty_ee": (3, 3), "action_velo_ee": (3,), "ee_pos_meas": (3,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(pose_ee, action_velo_ee, ee_pos_meas):
            return ee_pos_meas - (pose_ee + action_velo_ee)

        return connection_func


class EECubeGraspedConnection(ActiveInterconnection):
    """A grasped cube moves with the end-effector: their positions coincide."""

    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {
                "pose_ee": (3,),
                "uncertainty_ee": (3, 3),
                "position_cube": (3,),
                "uncertainty_cube": (3, 3),
                "likelihood_grasped_cube": (1,),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
            # Moving the cube requires a grasp, whichever direction it is to be moved in.
            required_signs_dict={"position_cube": {"likelihood_grasped_cube": 1}},
        )

    def define_implicit_connection_function(self):
        def connection_func(position_cube, pose_ee):
            return position_cube - pose_ee

        return connection_func


class EECubeDistanceConnection(ActiveInterconnection):
    """The distance between the end-effector and the cube's center is the length of their difference."""

    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"distance_ee_cube": (1,), "pose_ee": (3,), "position_cube": (3,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(distance_ee_cube, pose_ee, position_cube):
            # the small offset keeps the gradient finite when both coincide
            distance = torch.sqrt(torch.sum(torch.square(position_cube - pose_ee)) + 1e-12).reshape(1)
            return distance - distance_ee_cube

        return connection_func


class GraspCubeConnection(ActiveInterconnection):
    """How the grasp likelihood depends on the distance between hand and cube, and on the gripper.

    The likelihood's value accumulates the evidence of the finger encoders: the gripper was commanded to close, and
    its fingers stopped while still apart, i.e. something is between them, and the hand is close to the cube.

    Its gradient is a model of how the actions would change the grasp:
    - moving the hand towards the cube makes a grasp more likely (`REACH_SCALE`),
    - closing the gripper helps if the cube is between the fingers, and harms otherwise (a new grasp needs an open
      gripper). While closed, the encoders tell: the fingers are held apart by the cube or closed on nothing. While
      fully open, the distance tells: the cube is within about `BETWEEN_FINGERS_SCALE` of the gripper's center. An
      opening gripper has to open fully before it can grasp again.

    The conditions are 0/1 tensors, so "and" is a product and "not" is one minus the condition.
    """

    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {
                "likelihood_grasped_cube": (1,),
                "distance_ee_cube": (1,),
                "gripper_activation": (1,),
                "gripper_aperture_meas": (1,),
                "gripper_aperture_rate_meas": (1,),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(
            likelihood_grasped_cube,
            distance_ee_cube,
            gripper_activation,
            gripper_aperture_meas,
            gripper_aperture_rate_meas,
        ):
            # smooth in the distance, these carry the gradient: the wide one draws the hand to the cube, the narrow one
            # centers it between the fingers
            squared_dist = torch.square(distance_ee_cube)
            reachable = torch.exp(-squared_dist / (2 * REACH_SCALE**2))
            between_fingers = torch.exp(-squared_dist / (2 * BETWEEN_FINGERS_SCALE**2))

            # 0/1 conditions without gradient: they only switch the terms below on and off
            dtype = likelihood_grasped_cube.dtype
            # the gripper was told to close (only a closing gripper can hold anything)
            commanded_closed = (gripper_activation > 0.5).to(dtype)
            # the fingers have not moved yet, or have opened again
            fully_open = (gripper_aperture_meas > OPEN_APERTURE).to(dtype)
            # the fingers did not close all the way, i.e. they are not closed on nothing
            fingers_apart = 1 - (gripper_aperture_meas < EMPTY_APERTURE).to(dtype)
            # the fingers have stopped, i.e. they are not still closing or opening
            fingers_still = (torch.abs(gripper_aperture_rate_meas) < STILL_APERTURE_RATE).to(dtype)
            # the hand is at the cube, so what the fingers hold is the cube
            near_cube = (distance_ee_cube < HOLDING_DISTANCE).to(dtype)

            # value: the evidence of the finger encoders, the fingers closed and stopped on the cube; the likelihood
            # moves towards it by GRASP_EVIDENCE_RATE per tick. A product of 0/1 conditions is their "and": a single
            # failed check makes it 0.
            holding = commanded_closed * fingers_still * near_cube * fingers_apart * (1 - fully_open)
            innovation_from_evidence = GRASP_EVIDENCE_RATE * (holding - likelihood_grasped_cube)

            # gradient: what moving the hand and closing the gripper would do to the grasp. The cube is between the
            # fingers if the encoders say so (closed) or the gripper is centered on it (fully open); closing helps
            # above 0.5 and harms below, so the gripper closes, holds or opens.
            cube_between = commanded_closed * fingers_apart + (1 - commanded_closed) * fully_open * between_fingers
            effect_of_actions = reachable + gripper_activation * (cube_between - 0.5)
            # effect_of_actions adds its gradient but not its value
            return innovation_from_evidence + effect_of_actions - effect_of_actions.detach()

        return connection_func


class CubeDirectMeasurementConnection(ActiveInterconnection):
    """The measured cube position equals the estimated one."""

    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"position_cube": (3,), "uncertainty_cube": (3, 3), "cube_pos_meas": (3,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(position_cube, cube_pos_meas):
            return cube_pos_meas - position_cube

        return connection_func


class CubeBDirectMeasurementConnection(ActiveInterconnection):
    """The measured position of cube B equals the estimated one."""

    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"position_cube_b": (3,), "uncertainty_cube_b": (3, 3), "cube_b_pos_meas": (3,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(position_cube_b, cube_b_pos_meas):
            return cube_b_pos_meas - position_cube_b

        return connection_func


class StackCubeConnection(ActiveInterconnection):
    """How the stack likelihood (cube A rests on cube B and is let go) depends on the positions of the cubes and on the
    gripper. It is built like `GraspCubeConnection`: the value is evidence, the gradient is a model of what the
    actions would change.

    The value moves towards the evidence that A was placed: A is above B within PLACED_*_TOLERANCE, the gripper was
    told to open and its fingers are fully open.

    The gradient:
    - A is drawn to the place above B, but at a height of STACK_HEIGHT + STACK_CLEARANCE above B's center, i.e. clear of
      B's top. Only when A is above B (the narrow `aligned` term) the height it is drawn to drops to STACK_HEIGHT, so
      it comes down onto B and does not hit it from the side. The height `target` is not differentiated: `aligned`
      switches it, it is not something the actions would change.
    - While A is held, opening the gripper helps if A is placed (above B and at the stack height) and harms
      otherwise, so the gripper stays closed while A is carried and opens once A is in place. While A is not held,
      the stack has no opinion on the gripper.
    """

    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {
                "likelihood_stacked_cube": (1,),
                "position_cube": (3,),
                "position_cube_b": (3,),
                "likelihood_grasped_cube": (1,),
                "gripper_activation": (1,),
                "gripper_aperture_meas": (1,),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(
            likelihood_stacked_cube,
            position_cube,
            position_cube_b,
            likelihood_grasped_cube,
            gripper_activation,
            gripper_aperture_meas,
        ):
            difference = position_cube - position_cube_b
            horizontal_sq = torch.sum(torch.square(difference[:2])).reshape(1)
            height = difference[2:3]

            # smooth terms, these carry the gradient
            aligned = torch.exp(-horizontal_sq / (2 * STACK_ALIGN_SCALE**2))
            # A is carried high until it is above B, then it comes down; no gradient through this switch
            target_height = STACK_HEIGHT + STACK_CLEARANCE * (1 - aligned.detach())
            approach = torch.exp(
                -horizontal_sq / (2 * STACK_REACH_HORIZONTAL_SCALE**2)
                - torch.square(height - target_height) / (2 * STACK_REACH_VERTICAL_SCALE**2)
            )
            at_place = aligned * torch.exp(-torch.square(height - STACK_HEIGHT) / (2 * STACK_PLACE_HEIGHT_SCALE**2))

            # 0/1 conditions without gradient: the evidence that A was placed and let go
            dtype = likelihood_stacked_cube.dtype
            above_b = (horizontal_sq < PLACED_HORIZONTAL_TOLERANCE**2).to(dtype)
            at_stack_height = (torch.abs(height - STACK_HEIGHT) < PLACED_VERTICAL_TOLERANCE).to(dtype)
            told_to_open = (gripper_activation < 0.5).to(dtype)
            fingers_open = (gripper_aperture_meas > OPEN_APERTURE).to(dtype)
            placed = above_b * at_stack_height * told_to_open * fingers_open
            innovation_from_evidence = STACK_EVIDENCE_RATE * (placed - likelihood_stacked_cube)

            # gradient: drawing A to the place above B, and opening the gripper once it is there. The gripper only
            # matters for the stack while A is held (the grasp likelihood, not differentiated here): before that,
            # whether to close it is up to the grasp.
            held = likelihood_grasped_cube.detach()
            effect_of_actions = approach + held * (1 - gripper_activation) * (at_place - 0.5)
            # effect_of_actions adds its gradient but not its value
            return innovation_from_evidence + effect_of_actions - effect_of_actions.detach()

        return connection_func