# Isaac Lab → GR00T → Arena E2E

The customer-ready Franka workflow is packaged under
[`franka_groot_e2e/`](franka_groot_e2e/README.md).

It contains:

- synthetic-data generation, analysis, and LeRobot conversion scripts;
- the verified 8-GPU GR00T N1.7 SFT launcher;
- the verified 8-GPU IsaacLab Arena parallel evaluator;
- installation and execution instructions from a clean Docker workspace;
- real generation videos, trajectory and failure-analysis plots;
- final-checkpoint attention maps;
- Arena result tables plus one successful and failed video for every task.

Start with the [complete E2E guide](franka_groot_e2e/README.md).


## Franka pull-lift-hang task

The Newton MJWarp scaffold for two Franka arms pulling, lifting, and hanging a panel is under
[`franka_pull_lift_hang/`](franka_pull_lift_hang/README.md). It includes
a dedicated Isaac Lab 3.0 Docker launcher, web/native visualization and task data-collection scripts.
