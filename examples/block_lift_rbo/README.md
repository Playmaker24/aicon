# Block Lift

A simulated Panda arm (robosuite's `Lift` task) reaches a cube, grasps it and lifts it. The only goal is on the estimated height of the cube; reaching, grasping and lifting emerge from the gradients of that goal.

## Run

After the setup in the [main README](../README.md), from the repository root:

```
uv run python -m block_lift.run_experiment                    # headless, until Ctrl+C
uv run python -m block_lift.run_experiment --render           # MuJoCo viewer (green: estimated cube position)
uv run python -m block_lift.run_experiment --graph            # graph view at http://localhost:8765, starts paused
uv run python -m block_lift.run_experiment --render --graph   # both
```

Keys in the terminal:

- Space: pause and continue
- D: pull the cube out of the gripper
- Arrow keys: shift the cube on the table
- R: reset

A plot of the run is written to `outputs/block_lift/` at the end. See `python -m block_lift.run_experiment --help` for all options.

## Files

- `experiment_specifications.py`: the graph (which components and connections are built)
- `sensors.py`, `actions.py`, `estimators.py`, `connections.py`, `goals.py`: the AICON components
- `sensor_action_components.py`: base class for parts of the robot that are commanded and measured at once (arm, gripper)
- `env.py`: the robosuite environment with access to the ground truth for the simulated sensors
- `keyboard.py`: keyboard controls in the terminal
- `visualization/`: the browser graph view, highlighting the gradient path each action follows
