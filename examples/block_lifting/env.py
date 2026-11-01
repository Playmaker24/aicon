"""
Simulated Block Lifting Environment (robosuite)

A Panda arm with a parallel gripper in front of a table with a cube: robosuite's `Lift` task. The cube is placed
randomly within +-3 cm of the table centre on every reset and counts as lifted once its centre is more than
SUCCESS_HEIGHT above the table top. Besides the robosuite interface, the environment provides the ground-truth
quantities the simulated sensors read (cube position, end-effector position, force at the end-effector).

Written for robosuite 1.5.

Authors: Adrian Pfisterer, Patrick Lowin, James Reynaldo, Vito Mengers
License: MIT
"""

__author__ = "Adrian Pfisterer, Patrick Lowin, James Reynaldo, Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import numpy as np
from robosuite.controllers import load_part_controller_config
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.environments.manipulation.lift import Lift
from robosuite.wrappers import VisualizationWrapper

# The cube counts as lifted when its centre is more than this far above the table top [m] (as in robosuite's Lift).
SUCCESS_HEIGHT = 0.04

# Indicator drawn when rendering (estimated cube position).
CUBE_INDICATOR = "estimated_cube_position"


class BlockLiftEnv(Lift):
    """Lift the cube. `initial_qpos` optionally sets the arm's initial joint positions (default: robosuite's)."""

    def __init__(
        self,
        robots,
        controller_configs=None,
        has_renderer=False,
        render_camera="agentview",
        control_freq=30,
        horizon=1000,
        ignore_done=True,
        initial_qpos=None,
        seed=None,
    ):
        self.custom_initial_qpos = initial_qpos
        super().__init__(
            robots=robots,
            controller_configs=controller_configs,
            use_camera_obs=False,
            has_renderer=has_renderer,
            has_offscreen_renderer=False,
            render_camera=render_camera,
            render_gpu_device_id=-1,
            control_freq=control_freq,
            lite_physics=False,  # the full physics of robosuite <= 1.4, with which the examples were tuned
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=True,
            # MuJoCo's own viewer; robosuite's other renderer draws offscreen, which needs EGL on Linux
            renderer="mjviewer",
            seed=seed,
        )

    def _reset_internal(self):
        super()._reset_internal()
        if self.custom_initial_qpos is not None:
            self.robots[0].set_robot_joint_positions(self.custom_initial_qpos)
            self.robots[0].part_controllers["right"].update_initial_joints(np.array(self.custom_initial_qpos))

    def _check_success(self):
        return self.cube_height_above_table() > SUCCESS_HEIGHT

    def step(self, action):
        obs, reward, _, info = super().step(action)
        return obs, reward, self._check_success(), info

    # --- ground truth for the simulated sensors ---

    def cube_height_above_table(self) -> float:
        """Height of the cube's centre above the table top [m]: about 0.02 when it rests on the table."""
        return float(self.sim.data.body_xpos[self.cube_body_id][2] - self.model.mujoco_arena.table_offset[2])

    def get_cube_pos(self) -> np.ndarray:
        return np.array(self.sim.data.body_xpos[self.cube_body_id]).copy()

    def get_sim_time(self) -> float:
        return self.sim.data.time

    def get_ee_pos(self) -> np.ndarray:
        return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]).copy()

    def get_ee_force_magnitude(self) -> float:
        return float(np.linalg.norm(self.robots[0].ee_force["right"]))


def make_env(initial_qpos=None, controller="OSC_POSITION", render=False, seed=None):
    """The environment at 30 Hz, wrapped for visualization, after a reset. `controller` is a robosuite default
    controller: "OSC_POSITION" takes [vx, vy, vz, gripper], "OSC_POSE" additionally three rotation velocities.
    `seed` fixes the random cube size and position (and the arm's start-pose noise)."""
    env = BlockLiftEnv(
        robots="Panda",
        controller_configs=refactor_composite_controller_config(
            load_part_controller_config(default_controller=controller), "Panda", ["right"]
        ),
        has_renderer=render,
        initial_qpos=None if initial_qpos is None else np.asarray(initial_qpos),
        seed=seed,
    )
    env = VisualizationWrapper(
        env,
        indicator_configs=[{"name": CUBE_INDICATOR, "type": "sphere", "size": [0.01], "rgba": [0, 1, 0, 0.95]}],
    )
    env.reset()
    return env


def show_estimates(env, cube_position):
    """Move the render indicator to the estimated cube position."""
    env.set_indicator_pos(CUBE_INDICATOR, np.asarray(cube_position)[:3])
    env.env.sim.forward()