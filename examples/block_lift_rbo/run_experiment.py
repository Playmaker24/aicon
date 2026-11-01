"""
Simulated Cube Lifting with AICON

    python -m block_lift.run_experiment [--render] [--graph]

The robot has to reach the cube, grasp it and lift it; the only goal is on the estimated height of the cube.
`--render` opens the MuJoCo viewer (green: estimated cube position; on macOS, the script restarts itself under
MuJoCo's `mjpython`, which the viewer needs there). `--graph` shows the graph in the browser (http://localhost:8765)
with the gradient path each action follows; it starts paused (with the viewer already open).

The run goes on until Ctrl+C unless --seconds is given. Keys in the terminal: space pauses and continues the run, D
pulls the cube out of the gripper, the arrow keys shift the cube while it lies on the table (also while paused), R
resets the scene and the graph to the beginning (also while paused). The pull is a disturbance: the cube is put on the table next to the hand, and the gripper, still commanded closed, closes
on nothing; AICON notices the lost grasp through the finger sensors, opens the gripper and grasps the cube again. A
shift moves the cube by SHIFT_STEP in the table plane, as seen
in the viewer's default front view: up/down away from/towards the viewer, left/right to the sides; AICON follows the
cube with its position sensor. A plot of the run is written to outputs/block_lift/ at the end.

Authors: Adrian Pfisterer
License: MIT
"""

__author__ = "Adrian Pfisterer"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import argparse
import functools
import os
import queue
import random
import sys
import threading
import time

import numpy as np
import torch

from aicon_base.experiment_util.mjpython import relaunch_under_mjpython
from aicon_base.middleware.python_sequential import build_components, run_component_sequence
from . import experiment_specifications as spec
from .connections import STILL_APERTURE_RATE
from .env import make_env, show, show_estimates, LIFT_SUCCESS_HEIGHT
from .keyboard import KeyboardControls

# In the graph view the gripper's gradient path is shown from when its command changes (close or open) while the
# fingers carry the change out (a close takes about 0.25 s, an open about 0.6 s), and this long after [s]: enough to
# bridge the tick in which the command changed but the fingers have not started yet, and single slow ticks while
# they move.
GRIPPER_PATH_HOLD = 0.15

# how far one press of an arrow key shifts the cube [m]
SHIFT_STEP = 0.01

# direction of a shift in the table plane (x, y) per arrow key, as seen from the front view camera (at +x, looking
# towards the robot)
SHIFT_DIRECTIONS = {"up": (-1.0, 0.0), "down": (1.0, 0.0), "left": (0.0, -1.0), "right": (0.0, 1.0)}


def run(
    seconds=None,
    seed=0,
    lift_height=spec.LIFT_HEIGHT,
    render=False,
    graph=False,
    keyboard=False,
    pull_at=(),
    shift_at=(),
    stop_on_success=False,
    verbose=True,
):
    """Run one episode of `seconds` simulated time, or until Ctrl+C if None. Returns a dict of per-tick records and the
    step at which the cube was first lifted.
    With `graph`, the graph and the followed gradient paths are shown in the browser and the run is slowed down to
    real time. With `keyboard`, space in the terminal pauses and continues the run (with `graph`, it starts paused),
    D pulls the cube out of the gripper, the arrow keys shift it while it lies on the table and R resets the run to
    the beginning. `pull_at`: simulated
    times [s] at which the cube is pulled out of the gripper as well; `shift_at`: (time [s], dx, dy) at which the cube
    is shifted by (dx, dy) [m] as well, if it lies on the table."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env = make_env(render=render, seed=seed)
    builders, connection_builders, _ = spec.get_building_functions(env, lift_height=lift_height)
    components = build_components(builders)
    viz = None
    if graph:
        from .visualization.web_graph import WebGraph

        viz = WebGraph(builders, connection_builders)
    initial_state = env.env.sim.get_state()  # restored on a reset, which keeps the viewer open (env.reset() closes it)
    pull_requested = threading.Event()
    reset_requested = threading.Event()
    shifts_requested = queue.SimpleQueue()  # (dx, dy), filled by the keyboard thread
    keys = None
    if keyboard:
        callbacks = {
            "d": ("pull the cube out of the gripper", pull_requested.set),
            "r": ("reset the scene to the beginning", reset_requested.set),
        }
        for key, (dx, dy) in SHIFT_DIRECTIONS.items():
            callbacks[key] = (None, functools.partial(shifts_requested.put, (SHIFT_STEP * dx, SHIFT_STEP * dy)))
        keys = KeyboardControls(paused=graph, callbacks=callbacks, hints=["arrow keys: shift the cube on the table"])
    pull_times = sorted(pull_at)
    shift_times = sorted(shift_at)
    shift_refused = False

    def apply_shifts():
        nonlocal shift_refused
        while not shifts_requested.empty():
            dx, dy = shifts_requested.get()
            if env.env.cube_on_table():
                env.env.shift_cube(dx, dy)
                shift_refused = False
            elif not shift_refused:  # once per attempt, holding an arrow key repeats it
                shift_refused = True
                if verbose:
                    print("The cube can only be shifted while it lies on the table")

    def apply_pull():
        if env.env.cube_on_table():
            if verbose:
                print("The cube can only be pulled out of the gripper while it is lifted")
            return False
        env.env.pull_cube()
        return True

    def reset():
        """Back to the beginning: the scene as after the environment's reset, a newly built graph, time 0 and an empty
        record (so the plot at the end shows the run since the last reset)."""
        nonlocal components, arm, gripper_component, step, t, start, first_lift, pull_times, shift_times
        nonlocal gripper_path_until, last_gripper
        reset_requested.clear()
        pull_requested.clear()
        while not shifts_requested.empty():
            shifts_requested.get()
        env.env.sim.set_state(initial_state)
        env.env.sim.forward()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        components = build_components(builders)
        arm, gripper_component = components["RobotProprioAct"], components["GripperProprioAct"]
        for values in record.values():
            values.clear()
        step, t, start, first_lift = 0, 0.0, time.monotonic(), None
        pull_times, shift_times = sorted(pull_at), sorted(shift_at)
        gripper_path_until, last_gripper = -1.0, None

    def while_paused():
        if reset_requested.is_set():
            reset()
        apply_shifts()
        if render:
            show(env)

    gripper_path_until = -1.0
    last_gripper = None
    start = time.monotonic()
    arm, gripper_component = components["RobotProprioAct"], components["GripperProprioAct"]

    record = {
        k: []
        for k in (
            "t",
            "cube_height_true",
            "cube_height_estimated",
            "distance_ee_cube",
            "gripper",
            "aperture",
            "grasp_likelihood",
            "grasp_true",
            "pulled",
            "cube_xy_true",
            "cube_on_table",
        )
    }
    first_lift = None
    step, t = 0, 0.0
    if render:
        show(env)  # the viewer opens with the first frame, also when the run starts paused
    total_steps = None if seconds is None else int(round(seconds / env.control_timestep))
    try:
        while total_steps is None or step < total_steps:
            if keys is not None and keys.paused:
                keys.wait(while_paused=while_paused)
                start = time.monotonic() - t  # real time goes on from the current simulated time
            if reset_requested.is_set():
                reset()
            pulled = False
            if pull_requested.is_set() or (pull_times and t >= pull_times[0]):
                pull_requested.clear()
                pull_times = [p for p in pull_times if p > t]
                pulled = apply_pull()
            while shift_times and t >= shift_times[0][0]:
                shifts_requested.put(shift_times.pop(0)[1:])
            apply_shifts()
            run_component_sequence(components, torch.tensor(t, dtype=torch.float64))
            velocity = arm.quantities["action_velo_ee"].cpu().numpy()
            gripper = float(gripper_component.quantities["gripper_activation"].squeeze())
            env.step(np.concatenate([velocity, [2 * gripper - 1]]))

            cube = components["CubePosEstimator"].quantities["position_cube"].detach().cpu().numpy().copy()
            record["t"].append(t)
            record["cube_height_true"].append(env.env.cube_height_above_table())
            record["cube_height_estimated"].append(float(cube[2]) - env.env.table_height())
            record["distance_ee_cube"].append(float(np.linalg.norm(env.env.get_ee_pos() - env.env.get_cube_pos())))
            record["gripper"].append(gripper)
            record["aperture"].append(env.env.get_gripper_aperture())
            record["grasp_likelihood"].append(
                float(components["GraspLikelihoodEstimator"].quantities["likelihood_grasped_cube"].reshape(-1)[0])
            )
            record["grasp_true"].append(float(env.env.is_grasping_cube()))
            record["pulled"].append(pulled)
            record["cube_xy_true"].append(env.env.get_cube_pos()[:2])
            record["cube_on_table"].append(env.env.cube_on_table())
            if render:
                show_estimates(env, cube)
                show(env)
            if viz is not None:
                # the gripper's path only while it changes, not while it holds
                rate = float(gripper_component.quantities["gripper_aperture_rate_meas"].squeeze())
                carrying_out = (rate < -STILL_APERTURE_RATE) if gripper > 0.5 else (rate > STILL_APERTURE_RATE)
                if (last_gripper is not None and gripper != last_gripper) or (t < gripper_path_until and carrying_out):
                    gripper_path_until = t + GRIPPER_PATH_HOLD
                last_gripper = gripper
                viz.update(components, hidden=() if t < gripper_path_until else ("GripperProprioAct",))
                time.sleep(max(0.0, start + t - time.monotonic()))
            if env.env._check_success() and first_lift is None:
                first_lift = step
                if verbose:
                    print(f"Cube lifted after {t:.1f} s")
                if stop_on_success:
                    break
            step += 1
            t += env.control_timestep
    except KeyboardInterrupt:  # Ctrl+C ends the run, which is then evaluated as usual
        if verbose:
            print("Stopped")
        # the interrupted tick may be recorded only in part
        ticks = min(len(values) for values in record.values())
        record = {k: values[:ticks] for k, values in record.items()}
    env.close()
    if keys is not None:
        keys.close()
    if viz is not None:
        viz.close()
    return record, first_lift


def plot_record(record, title, path, lift_height):
    import matplotlib.pyplot as plt

    t = np.array(record["t"])
    fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    axes[0].plot(t, record["cube_height_true"], label="true")
    axes[0].plot(t, record["cube_height_estimated"], "--", label="estimated")
    axes[0].axhline(lift_height, color="gray", lw=0.8, ls=":", label="goal")
    axes[0].axhline(LIFT_SUCCESS_HEIGHT, color="gray", lw=0.8, label="lifted")
    axes[0].set_ylabel("cube above table [m]")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].plot(t, record["distance_ee_cube"], label="end-effector to cube")
    axes[1].plot(t, record["aperture"], label="finger distance")
    axes[1].set_ylabel("[m]")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[2].plot(t, record["grasp_likelihood"], label="grasp likelihood")
    axes[2].plot(t, record["gripper"], label="gripper closed")
    axes[2].plot(t, record["grasp_true"], ":", label="grasped (ground truth)")
    axes[2].set_ylabel("likelihood")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="upper left", fontsize=8)
    for ax in axes:
        for pulled_at in t[np.asarray(record["pulled"], dtype=bool)]:
            ax.axvline(pulled_at, color="gray", lw=0.8, ls="--")
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=None, help="simulated time (default: until Ctrl+C)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the cube's random placement")
    parser.add_argument(
        "--lift-height", type=float, default=spec.LIFT_HEIGHT, help="goal height of the cube above the table [m]"
    )
    parser.add_argument("--render", action="store_true", help="show the MuJoCo viewer")
    parser.add_argument("--graph", action="store_true", help="show the graph and its gradient paths in the browser")
    parser.add_argument("--output-dir", default=os.path.join("outputs", "block_lift"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.render and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        parser.error("--render needs a display (DISPLAY is not set)")
    if args.render:
        relaunch_under_mjpython()

    torch.set_default_dtype(torch.float64)
    record, first_lift = run(
        seconds=args.seconds,
        seed=args.seed,
        lift_height=args.lift_height,
        render=args.render,
        graph=args.graph,
        keyboard=True,
    )
    if not record["t"]:
        return
    if first_lift is None:
        print(f"Cube not lifted within {record['t'][-1]:.0f} s (height {record['cube_height_true'][-1]:.3f} m)")
    else:
        print(
            f"Cube lifted after {record['t'][first_lift]:.1f} s, "
            f"{record['cube_height_true'][-1]:.3f} m above the table at the end"
        )
    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        os.makedirs(args.output_dir, exist_ok=True)
        path = os.path.join(args.output_dir, f"lift_seed{args.seed}.png")
        plot_record(record, f"lift (seed {args.seed})", path, args.lift_height)
        print("Plot written to", path)


if __name__ == "__main__":
    main()
