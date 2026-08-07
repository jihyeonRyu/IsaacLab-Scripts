# Franka Pull Lift Hang task (액자 걸기)

Isaac Lab 3.0 / Newton MJWarp scene scaffold containing two fixed-base Franka
Pandas, a dynamic panel with a rear steel-tube perimeter/hanger, two narrow
tabletop support rails, and a round horizontal hanger rod made from three static box
colliders. The beam extrusion axis is `Y`; from the frontal task view it
appears as a thin horizontal rail.

## Start and validate

```bash
bash franka_pull_lift_hang/docker_start.sh
bash franka_pull_lift_hang/run_smoke.sh
```

The dedicated `franka-dual-arm-hand-hang` container uses
`nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1` and bind-mounts this repository at
`/workspace/IsaacLab-Scripts`.

## View the task in a browser (WebRTC)

Run the dual-Franka scene with Isaac Sim WebRTC:

```bash
ISAACSIM_HOST=SERVER_IP \
  bash franka_pull_lift_hang/run_webrtc.sh
```

Open `http://SERVER_IP:8002`. TCP port `49102` and UDP port `47992` must also be reachable. Only one WebRTC client is supported at a time. Stop it with:

```bash
bash franka_pull_lift_hang/stop_webrtc.sh
```

For local-only viewing, set `ISAACSIM_HOST=127.0.0.1`. WebRTC has no built-in authentication or encryption, so expose these ports only on a trusted network.

## Lightweight Newton debug view (Viser)

```bash
bash franka_pull_lift_hang/run_web.sh
```

For the canonical Hang AutoOps viewer, always use the dedicated launcher:

```bash
bash franka_pull_lift_hang/run_hang_viser.sh
```

This launcher is the single source of truth for Hang visualization. It stops
only stale instances of this scene, starts exactly one process, clears
`DISPLAY` for headless Isaac Sim startup, keeps Viser on port `8080`, disables
RTX task cameras for real-time debug speed, and launches `--auto-ops-task hang`. Do not use an
ad-hoc `docker exec` command for this workflow; differing flags can produce a
blank or stale viewer after a container restart.

Enable the camera preview explicitly when it is needed:

```bash
ENABLE_CAMERA_PREVIEW=1 bash franka_pull_lift_hang/run_hang_viser.sh
```

`--camera-preview` and `--disable-task-cameras` are mutually exclusive. The
launcher selects exactly one mode and never passes both flags.

Open `http://localhost:8080` for the free scene view and
`http://localhost:8081` for the three-camera view (left wrist, right wrist and
hanger front). Replace `localhost` with `SERVER_IP` for a remote machine. For
SSH, forward both ports:

```bash
ssh -L 8080:localhost:8080 -L 8081:localhost:8081 USER@SERVER
```

The physics backend is Newton MJWarp; Viser reads its live scene state. The launch scripts disable CUDA graph for faster startup during geometry iteration. Stop
with `Ctrl-C`.

## Native Newton OpenGL viewer

On a host with an authorized X display:

```bash
bash franka_pull_lift_hang/run_native_newton.sh
```

The browser path is preferred on remote/headless systems.

## Scene modes

```bash
PANEL_STATE=hung bash franka_pull_lift_hang/run_web.sh
PANEL_STATE=staging bash franka_pull_lift_hang/run_web.sh
```

`staging` is the default. The panel lies horizontally across two parallel
rails, 124 mm above the tabletop. Its rear steel frame faces upward. The Franka
bases, rack, panel and hanger are placed inside the arms' working envelope.
`hung` starts the same panel vertically with its deep rear top rail seated on the
two passive S-hook catch bars.

## Auto Ops episode

### Controller modules

AutoOps control is split into three physical task modules. The `long` mode is
only an orchestrator; it does not contain a second copy of any trajectory:

- `PullTaskModule`: handle approach, physical pull, gradual release, extraction,
  and return to the shared ready pose.
- `LiftTaskModule`: object-relative side approach, bimanual contact, lift, and
  closed-chain rotation.
- `HangTaskModule`: nail-relative transport, ring insertion, release, and
  lateral hand clearance.

Module ownership, execution boundaries, and contracts live in the `task_modules/` package. Phase handlers
live in `auto_ops_controller.py` as `_step_phase_0` through `_step_phase_10` and
are dispatched only by their owning module. The previously ambiguous phase-3
boundary is explicit: stages 0-3 belong to Pull; stage 4 onward belongs to Lift.

The Pull→Lift handoff validates measured panel displacement, lateral position,
and an opened pulling gripper. The Lift→Hang handoff validates the measured
panel-to-hand transforms and a physical bimanual enclosure. A failed contract
ends the episode instead of letting the next module compensate for a broken
previous task.

Run the fast ownership/contract regression tests before simulation:

```bash
PYTHONPATH=franka_pull_lift_hang \
  python3 -m unittest franka_pull_lift_hang/test_auto_ops_modules.py -v
```

Use the canonical long viewer launcher for the complete sequence:

```bash
bash franka_pull_lift_hang/run_long_viser.sh
```

```bash
docker exec franka-dual-arm-hand-hang bash -lc "cd /workspace/isaaclab && \
  ./isaaclab.sh -p /workspace/IsaacLab-Scripts/franka_pull_lift_hang/dual_franka_picture_hanging_scene.py \
    --device cuda:0 --viz none --disable-cuda-graph --auto-ops \
    --record-dir .../dataset/episode_000000 --episode-id 0"
```

`--auto-ops` alone is a no-write motion test; `--record-dir` turns on recording.
`--motion-speed-scale` defaults to 4.0, which keeps the grasps and the hang
reliable; raise it only for viewer review. The scripted phases are:

1. Left gripper rolls to the front-edge pose and closes on the board plus the
   rear top rail.
2. It pulls the panel 300 mm toward the robots, staying on the rack rails.
3. It releases, retreats, and both arms return to the joint-space ready pose.
4. Both grippers descend outside the side guards and close on the side edges.
5. Both arms lift the panel clear of the rack, then rotate it to vertical.
6. The panel is carried above the hooks and lowered until the rear rail seats.
7. Both grippers release and retreat; success needs the panel still hanging.

Phases 5-8 drive one desired *panel* pose and derive both wrist targets from the
grasp transform captured when the jaws closed, so the two hands always move as
one rigid body and the wrists roll with the panel.

## Physics notes

See [PHYSICS_TUNING.md](PHYSICS_TUNING.md) for the detailed collision,
penetration, solver, contact, friction, actuator, and bimanual-alignment tuning
guide developed from this task.s physical failure cases.

Two engine behaviours in `isaac-lab:3.0.0-beta2-post1` shape this scene:

- Newton's `RigidObject` reads and writes root quaternions as `(x, y, z, w)`
  while `Articulation` poses and `isaaclab.utils.math` are `(w, x, y, z)`.
  `rigid_object_quat()` converts at that boundary and
  `validate_panel_rest_pose()` fails loudly if the order ever changes back.
- Newton's MuJoCo contact table is a construction-time snapshot, so contact
  flags are logged for diagnostics only. Grasps are verified from jaw width and
  panel motion instead.
