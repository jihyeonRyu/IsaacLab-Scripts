# Newton physics and contact tuning

This guide records failure modes observed while developing Franka Pull Lift Hang with Isaac Lab 3.0 and Newton MJWarp. Treat these values as a known-good baseline and tune against measured contacts and trajectories.

## Diagnose penetration in the right order

Do not start by increasing friction or gripper force. Neither can create a missing collision, and both can make penetration more violent.

1. Verify collision geometry for both objects in the generated Newton model.
2. Verify collider size, pose, `contype`, and `conaffinity`.
3. Ensure commanded displacement per physics substep is smaller than the thinnest collider.
4. Tune contact margin and constraint stiffness (`solimp`/`solref`).
5. Increase solver iterations or substeps if contacts still arrive late.
6. Tune friction, actuator force, and damping only after non-penetration works.

## Collision geometry

Visual and collision meshes are independent. A finger can visibly touch the panel while its convex collision hull ends short of the visible fingertip. In this task the stock Panda hull generated finger-to-finger contacts but no distal finger-to-panel contact. The fingers then crossed their limits and jaw width became negative.

The scene adds a physically sized box collision proxy to each fingertip:

```text
size:       (0.018, 0.012, 0.050) m
local pose: (0.000, 0.000, 0.025) m
material:   HighHangGrip
```

Prefer simple box, capsule, or convex shapes for contact-critical parts. Match the real link shape; do not enlarge a proxy merely to make a trajectory succeed.

Run with `--newton-debug` and inspect the generated model, not only the USD. Expected output contains four `NewtonGraspPad` geoms and the board. A pair can collide when each geom's `contype` is accepted by the other's `conaffinity`.

```text
Panel/Board size=[0.270, 0.430, 0.010]
```

MuJoCo box sizes are half-extents, so this means a 20 mm board.

## Time discretization and CCD

The task uses `--substeps 4` and 32 CCD iterations in the responsive viewer baseline. CCD helps fast thin-body contact but cannot compensate for missing collision geometry.

```text
substep displacement = command displacement per control tick / substeps
```

Keep this below the thinnest collider. The gripper ramps total jaw width at `side close ramp up to 0.010 m` per 15 Hz control tick instead of jumping from 79 mm to zero. If penetration remains after geometry validation, increase substeps, reduce command increments, and keep CCD enabled.

## Contact constraint settings

`harden_contacts()` edits the generated MJWarp model after Newton is built:

```text
global solimp:               (0.95, 0.99, 0.002)
PanelRack solimp:           (0.995, 0.999, 0.0005)
finger solimp:              (0.995, 0.999, 0.0005)
panel solimp:               (0.98, 0.995, 0.002)
PanelRack margin/solref:    0.012 m, (0.015, 2.0), priority 2
finger margin/solref:       0.004 m, (0.008, 2.0), priority 3
panel margin/solref:        0.002 m, (0.020, 1.5), priority 0
contact gap:                 0.0 m
solver iterations:           80
line-search iterations:      30
CCD iterations:              32
```

Include rails, side guards, rear stop, and top stop with the panel and fingers. Hardening only the grasped object still lets it sink through a soft support and tilt.

- `margin` activates contact before geometric overlap.
- `gap=0` avoids delayed contact.
- Larger leading `solimp` values make impedance firmer.
- A smaller `solimp` width sharpens the transition but can ring.
- `solref` controls contact time constant and damping and must match the timestep.

Change one group at a time and log generated values. Editing USD after Newton model construction does not update the active solver.

## Joint limits and actuators

```text
finger jnt_solimp: (0.99, 0.999, 0.001)
finger jnt_solref: (0.02, 1.0)
finger jnt_margin: 0.001 m
hand stiffness:    3500 N/m
hand damping:      650 N s/m
effort limit:      70 N per finger (140 N total)
velocity limit:    0.12 m/s
```

Apply maximum grip only after collision is stable. More force against penetrable contact increases overlap. After closure, retain measured finger positions and constant inward effort; alternating open/close targets causes chatter and twists a bimanual grasp.

## Friction

```text
rails/guards: static 0.45, dynamic 0.30
finger/panel/hanger: static 1.20, dynamic 1.00
```

Friction matters only after contact exists. Near-zero rail friction lets incidental contact move the panel; extreme grasp friction can hide insufficient normal force.

## Robust bimanual alignment

Compute all targets in the live panel frame. After closure, freeze both panel-to-hand transforms and derive both wrists from one desired panel pose. Independent world-space wrist paths make the arms fight.

Before lifting:

- restore horizontal orientation;
- move Y to the rack centreline;
- move X until the rear edge clears the cover;
- preserve both panel-to-hand transforms.

Cover front is `x=0.39 m`, panel half-length is `0.27 m`, and clearance is `0.01 m`, so panel centre must be at or below `x=0.11 m`. If already farther out, never push it backward.

## Rotation-time centering and slip recovery

A frozen measured grasp transform is necessary, but it is not sufficient once a pad creeps. A stale world-frame trajectory keeps moving away from the slipped panel and turns a small error into a dropped object. During lift and rotation:

- read the live panel pose every control tick;
- mirror the two measured wrist offsets in panel-local coordinates so local X and Z match and local Y is equal and opposite;
- reconstruct both wrist targets from one bounded corrected panel pose;
- move the shared centre toward rack `y=0`, never correct each arm independently;
- bound common translation correction to 12 mm per control tick;
- bound angular correction to 0.045 rad per control tick;
- advance shared trajectory progress only while both arms remain within 10 mm and 0.06 rad tracking error.

This feedback is not a kinematic object shortcut: the panel remains fully dynamic. Only robot targets follow the measured object, and the grippers must carry the panel through physical contact. If the panel slips, both hands first follow its live centre and then apply the same limited recentering correction.

## Contact verification caveat

In the tested build, `solver.mj_data.contact` is not a reliable live MJWarp contact buffer. Use it diagnostically, not as the sole runtime gate.

- No board: two 6 mm pads close to about 12 mm.
- 20 mm board: jaws stall around 33-43 mm including proxy thickness and margin.
- Negative or near-zero width indicates penetration or broken limits.

Follow width verification with a short, low-speed lift probe proving that the panel moves with both hands.

## Why a grasp can accelerate into instability

A panel that lifts and then shoots out during rotation is not automatically a friction failure:

- Negative jaw width means collision or joint-limit enforcement failed.
- Width that stalls and collapses only after panel height separates from hand height means the panel escaped first; closing jaws are the consequence.
- Rising tracking error on both arms means the rigid-body command is too fast or outside the shared reachable set.
- Rising rotation error with good position tracking indicates angular trajectory or grasp-transform error.
- Panel tilt before lift indicates support collision problems.

The key lesson is grasp-transform capture. Saving ideal IK targets after contact caused immediate shear because the physical hands were not exactly at those targets. Save measured poses:

```text
T_panel_hand_left  = inverse(T_world_panel) * T_world_hand_left_measured
T_panel_hand_right = inverse(T_world_panel) * T_world_hand_right_measured
```

Then derive both targets from one desired panel pose. The two measured transforms describe one actual rigid grasp; independently idealized transforms introduce inconsistency.

## Damping and force-hold tuning

Damping dissipates velocity-dependent energy; it does not repair geometry or a discontinuous target. Tune in this order:

1. Establish non-penetrating contact with a slow position ramp.
2. Capture measured width and measured panel-to-hand transforms.
3. Keep a fixed position target near contact width.
4. Add inward feed-forward effort within the actuator limit.
5. Increase contact damping if bounce remains.
6. Slow angular motion if rotation error grows.

```text
per-finger inward effort: 17.5 N after verified contact (35 N total per gripper)
hand stiffness:           3500 N/m
hand damping:             650 N s/m
targeted finger solref:   (0.008, 2.0)
horizontal lift:          at least 25 control ticks
panel rotation:           at least 30 control ticks
```

For positive-format MuJoCo `solref=(timeconst, dampratio)`, decreasing time constant makes response faster/stiffer; increasing damping ratio dissipates more energy. Keep the time constant compatible with the substep. If chatter appears, undo the last stiffness change before adding more force.

Once contact is established, latch one mode and target. Use exactly one OPEN-to-CLOSING transition, one CLOSING-to-FORCE_HOLD transition, and one explicit release.

## Verification gates before full lift

1. Close: jaw widths stall in the board-contact band and never become negative.
2. Hold: maintain widths and panel pose.
3. Align: centre Y, clear the cover in X, and restore horizontal orientation.
4. Probe: raise a few millimetres and prove panel and hands move together.
5. Lift: reach rotation-safe height while widths remain stable.
6. Rotate slowly while monitoring errors, width, and panel height.
7. Run full transport and hang.

Abort if either jaw leaves its physical band or panel-to-hand relative pose changes beyond tolerance.

```text
stable width + panel follows hands       = valid grasp
width collapses + panel height separates = panel escaped first
hand Z rises + panel Z falls             = lost grasp
good position + growing rotation error   = angular/IK problem
panel tilts while on rack                 = support collision problem
```

## Failure checklist

When a gripper penetrates:

- Stop; never loosen the success criterion.
- Inspect generated geom size, pose, and collision bits with `--newton-debug`.
- Check for negative jaw width and excessive displacement per substep.
- Correct the fingertip proxy.
- Harden panel, fingers, and supports together.
- Re-run a slow close and small lift probe.

When one hand is shallow:

- Recompute from the latest panel pose.
- Leave symmetric overlap for lateral drift.
- Check whether a rail or guard allowed tilt.
- Align through one shared panel pose, not independent wrists.
