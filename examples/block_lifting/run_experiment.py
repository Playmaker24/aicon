"""
Simulated Block Lifting with AICON

    python -m examples.block_lifting.run_experiment

The robot has to find the cube, grasp it and lift it; the only goal is on the estimated height of the cube.
`--render` opens the MuJoCo viewer (green: estimated cube position; on macOS, the script restarts itself under
MuJoCo's `mjpython`, which the viewer needs there). A plot of the run is written to outputs/block_lifting/.

Authors: Adrian Pfisterer, Patrick Lowin, James Reynaldo, Vito Mengers
License: MIT
"""

__author__ = "Adrian Pfisterer, Patrick Lowin, James Reynaldo, Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import argparse
import os
import random
import sys

import numpy as np
import torch

from aicon_base.experiment_util.mjpython import relaunch_under_mjpython
from aicon_base.middleware.python_sequential import build_components, run_component_sequence
from .env import make_env, show_estimates, SUCCESS_HEIGHT
from .tutorial import experiment_specifications as spec


def run(
    seconds=30.0,
    seed=0,
    render=False,
    stop_on_success=False,
    verbose=True,
):
    """Run one episode. Returns a dict of per-tick records and the step at which the cube was first lifted."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env = make_env(spec.INITIAL_QPOS, controller="OSC_POSITION", render=render, seed=seed)
    builders, _, _ = spec.get_building_functions(env)
    components = build_components(builders)
    velocity_action, gripper_action = components["EEVelocities"], components["GripperAction"]
    table_height = env.env.model.mujoco_arena.table_offset[2]

    record = {
        k: []
        for k in (
            "t",
            "height_true",
            "height_estimated",
            "gripper",
            "grasp_likelihood",
            "ee",
            "cube_true",
            "cube_estimated",
        )
    }
    first_lifted = None
    t = 0.0
    for step in range(int(round(seconds / env.control_timestep))):
        run_component_sequence(components, torch.tensor(t, dtype=torch.float64))
        velocity = velocity_action.quantities["action_velo_ee"].cpu().numpy()
        gripper = float(gripper_action.quantities["gripper_activation"].squeeze())
        action = np.concatenate([velocity, [2 * gripper - 1]])
        _, _, done, _ = env.step(action)

        cube = components["CubePosEstimator"].quantities["position_cube"].detach().cpu().numpy().copy()
        record["t"].append(t)
        record["height_true"].append(env.env.cube_height_above_table())
        record["height_estimated"].append(float(cube[2] - table_height))
        record["gripper"].append(gripper)
        record["grasp_likelihood"].append(
            float(components["GraspLikelihoodEstimator"].quantities["likelihood_grasped_cube"].reshape(-1)[0])
        )
        record["ee"].append(env.env.get_ee_pos())
        record["cube_true"].append(env.env.get_cube_pos())
        record["cube_estimated"].append(cube)
        if render:
            show_estimates(env, cube)
            env.render()
        if done and first_lifted is None:
            first_lifted = step
            if verbose:
                print(f"Cube lifted after {t:.1f} s")
            if stop_on_success:
                break
        t += env.control_timestep
    env.close()
    return record, first_lifted


def plot_record(record, title, path):
    import matplotlib.pyplot as plt

    t = np.array(record["t"])
    fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    axes[0].plot(t, record["height_true"], label="true")
    axes[0].plot(t, record["height_estimated"], "--", label="estimated")
    axes[0].axhline(SUCCESS_HEIGHT, color="gray", lw=0.8, label="lifted")
    axes[0].set_ylabel("cube height above table [m]")
    axes[0].legend(loc="upper left", fontsize=8)
    error = np.linalg.norm(np.array(record["cube_estimated"]) - np.array(record["cube_true"]), axis=1)
    axes[1].plot(t, error)
    axes[1].set_ylabel("cube position error [m]")
    axes[2].plot(t, record["grasp_likelihood"], label="grasp likelihood")
    axes[2].plot(t, record["gripper"], label="gripper closed")
    axes[2].set_ylabel("likelihood")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="upper left", fontsize=8)
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=30.0, help="simulated time")
    parser.add_argument("--seed", type=int, default=0, help="fixes the cube's random size and start position")
    parser.add_argument("--render", action="store_true", help="show the MuJoCo viewer")
    parser.add_argument("--output-dir", default=os.path.join("outputs", "block_lifting"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.render and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        parser.error("--render needs a display (DISPLAY is not set)")
    if args.render:
        relaunch_under_mjpython()

    torch.set_default_dtype(torch.float64)
    record, first_lifted = run(seconds=args.seconds, seed=args.seed, render=args.render)
    name = "block_lifting"
    if first_lifted is None:
        print(f"{name}: cube not lifted within {args.seconds:.0f} s (height {record['height_true'][-1]:.3f} m)")
    else:
        print(f"{name}: cube lifted after {record['t'][first_lifted]:.1f} s")
    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        os.makedirs(args.output_dir, exist_ok=True)
        path = os.path.join(args.output_dir, name + ".png")
        plot_record(record, name, path)
        print("Plot written to", path)


if __name__ == "__main__":
    main()