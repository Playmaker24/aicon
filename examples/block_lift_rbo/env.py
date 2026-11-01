"""
Simulated Lift Environment (robosuite)

robosuite's `Lift` task: a Panda arm with a parallel gripper in front of a table with a small cube (edge 4-4.4 cm,
random position and yaw). The cube counts as lifted once its center is LIFT_SUCCESS_HEIGHT above the table top (the
success criterion of robosuite). Besides the robosuite interface, the environment provides the ground-truth
quantities the simulated sensors read (end-effector and cube position, opening and opening rate of the fingers).

Written for robosuite 1.5.

Authors: Adrian Pfisterer
License: MIT
"""

__author__ = "Adrian Pfisterer"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import numpy as np
from robosuite.controllers import load_part_controller_config
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.environments.manipulation.lift import Lift
from robosuite.wrappers import VisualizationWrapper

# robosuite counts the cube as lifted once its center is this far above the table top [m]
LIFT_SUCCESS_HEIGHT = 0.04

# a shifted cube keeps its center this far inside the edges of the table top [m]
CUBE_MARGIN = 0.03

# Indicator drawn when rendering (estimated cube position).
CUBE_INDICATOR = "estimated_cube_position"

# how far from below the hand a cube pulled out of the gripper is put on the table [m]
PULL_OFFSET = 0.08


class LiftEnv(Lift):
    """robosuite's Lift task with access to the ground truth for the simulated sensors."""

    def table_height(self) -> float:
        return float(self.model.mujoco_arena.table_offset[2])

    def get_cube_pos(self) -> np.ndarray:
        return np.array(self.sim.data.body_xpos[self.cube_body_id]).copy()

    def cube_height_above_table(self) -> float:
        return float(self.get_cube_pos()[2] - self.table_height())

    def get_ee_pos(self) -> np.ndarray:
        return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]).copy()

    def get_gripper_aperture(self) -> float:
        """Distance between the two fingers [m] (about 0.08 when open, 0 when closed on nothing)."""
        qpos = self.sim.data.qpos[self.robots[0]._ref_gripper_joint_pos_indexes["right"]]
        return float(qpos[0] - qpos[1])

    def get_gripper_aperture_rate(self) -> float:
        """Rate of change of the finger distance [m/s]."""
        qvel = self.sim.data.qvel[self.robots[0]._ref_gripper_joint_vel_indexes["right"]]
        return float(qvel[0] - qvel[1])

    def get_sim_time(self) -> float:
        return self.sim.data.time

    def is_grasping_cube(self) -> bool:
        """Ground truth of the grasp (fingers in contact with the cube), only used for the evaluation."""
        return bool(self._check_grasp(gripper=self.robots[0].gripper["right"], object_geoms=self.cube))

    def cube_on_table(self) -> bool:
        """Whether the cube lies on the table: in contact with the table top and not grasped."""
        return bool(self.check_contact(self.cube, "table_collision")) and not self.is_grasping_cube()

    def pull_cube(self):
        """Take the cube out of the gripper: it is put upright on the table top, PULL_OFFSET from below the hand (in
        +x, towards the front view camera) so that it is clear of the fingers, which close on nothing."""
        joint = self.cube.joints[0]
        qpos = np.array(self.sim.data.get_joint_qpos(joint))  # [x, y, z, quaternion (w, x, y, z)]
        w, x, y, z = qpos[3:]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        qpos[:2] = self.get_ee_pos()[:2] + [PULL_OFFSET, 0.0]
        qpos[2] = self.table_height() - self.cube.bottom_offset[2]
        qpos[3:] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        self.sim.data.set_joint_qpos(joint, qpos)
        self.sim.data.set_joint_qvel(joint, np.zeros(6))
        self.sim.forward()
        self.shift_cube(0.0, 0.0)  # keeps it on the table top

    def shift_cube(self, dx: float, dy: float):
        """Move the cube by (dx, dy) in the table plane (its height and orientation stay), keeping it on the table
        top; it comes to rest there."""
        joint = self.cube.joints[0]
        qpos = np.array(self.sim.data.get_joint_qpos(joint))  # [x, y, z, quaternion]
        limit = np.array(self.table_full_size[:2]) / 2 - CUBE_MARGIN
        table_xy = np.array(self.table_offset[:2])
        qpos[:2] = np.clip(qpos[:2] + [dx, dy], table_xy - limit, table_xy + limit)
        self.sim.data.set_joint_qpos(joint, qpos)
        self.sim.data.set_joint_qvel(joint, np.zeros(6))
        self.sim.forward()


def make_env(render=False, seed=None):
    """The Lift task at 30 Hz with the "OSC_POSITION" controller ([vx, vy, vz, gripper]), wrapped for visualization,
    after a reset. `seed` sets the cube's random placement (robosuite 1.5 draws it from the environment's own random
    generator, not numpy's global one)."""
    env = LiftEnv(
        robots="Panda",
        controller_configs=refactor_composite_controller_config(
            load_part_controller_config(default_controller="OSC_POSITION"), "Panda", ["right"]
        ),
        has_renderer=render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera="frontview",
        control_freq=30,
        horizon=1000,
        ignore_done=True,
        seed=seed,
        # MuJoCo's own viewer; robosuite's other renderer draws offscreen, which needs EGL on Linux
        renderer="mjviewer",
        # the full physics of robosuite <= 1.4, with which the example was tuned
        lite_physics=False,
    )
    env = VisualizationWrapper(
        env, indicator_configs=[{"name": CUBE_INDICATOR, "type": "sphere", "size": [0.008], "rgba": [0, 1, 0, 0.6]}]
    )
    env.reset()
    return env


def show_estimates(env, cube_position):
    """Move the render indicator to the estimated cube position."""
    env.set_indicator_pos(CUBE_INDICATOR, np.asarray(cube_position)[:3])
    env.env.sim.forward()


def show(env):
    """Draw the current state in the MuJoCo viewer, opening it if it is not open yet (robosuite opens it only in a
    step, and its render() draws nothing with the "mjviewer" renderer)."""
    env.env.viewer.update()
