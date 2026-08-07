# Auto Ops dataset contract for dual-Franka picture hanging

Generated samples belong under `dataset/`, which is ignored by Git. One process owns one GPU and writes complete episode directories; processes must never append to the same episode.

## Episode layout

```text
dataset/
  episode_000123/
    episode.json
    frames.jsonl
    left_wrist/{rgb,instance_segmentation}/000000.{png,npy}
    right_wrist/{rgb,instance_segmentation}/000000.{png,npy}
    hanger_front/{rgb,instance_segmentation}/000000.{png,npy}
    */instance_id_labels.json
```

`episode.json` stores the seed, randomized wallpaper/panel/light values, success/failure, FPS and GPU/worker ID. `frames.jsonl` stores aligned state, action and task annotations. RGB is PNG; the uncolorized instance ID image is NPY so IDs are not destroyed by an 8-bit color palette.

## Task annotations

The episode objective is picture hanging. Every captured frame has a `major_task` (`pick` or `hang`) and a finer `subtask`. Phase transitions are written by the Auto Ops state machine from measured predicates — end-effector pose error, jaw width and panel motion — rather than elapsed time. Where Pink leaves a bounded IK residual, a stage advances once tracking stops improving, so a transition means "as close as this arm gets", not "the tolerance was met". Newton's MuJoCo contact table is a construction-time snapshot in this Isaac Lab build, so `contact_flags` in `episode.json` events are diagnostic only and never gate a transition.

| index | major task | subtask |
|---:|---|---|
| 0 | pick | `left_approach_front_edge` |
| 1 | pick | `left_grasp_front_edge` |
| 2 | pick | `left_pull_panel_forward` |
| 3 | pick | `bimanual_approach_side_edges` |
| 4 | pick | `bimanual_grasp_side_edges` |
| 5 | pick | `bimanual_lift_panel` |
| 6 | hang | `transport_panel_to_hanger` |
| 7 | hang | `align_panel_with_hooks` |
| 8 | hang | `lower_panel_onto_hooks` |
| 9 | hang | `release_panel` |
| 10 | hang | `bimanual_retreat` |

Each record also stores `phase_index`, `phase_progress`, English/Korean instructions, and terminal success/failure fields. For GR00T/LeRobot conversion, map the English subtask instruction to `annotation.human.action.task_description`; keep `major_task` and `subtask` as additional analysis columns.

## Bimanual state and action

Use a fixed left-then-right order everywhere.

- Observation state: 20D absolute EEF state: `[left_xyz(3), left_rot6d(6), left_gripper_width(1), right_xyz(3), right_rot6d(6), right_gripper_width(1)]`.
- Action: 14D: `[left_dxyz(3), left_drotvec(3), left_gripper_abs(1), right_dxyz(3), right_drotvec(3), right_gripper_abs(1)]`.

EEF deltas make the two arms share a meaningful scale and match the existing single-Franka GR00T pipeline. Gripper commands remain absolute widths because accumulating gripper deltas causes drift. Store the structured left/right fields in raw JSONL for debugging and the flat 20D/14D arrays for training. All values at frame `t` must represent observation `s_t` and command `a_t` executed to reach `s_(t+1)`.

## Parallel capture

Rendering and physics remain on the process GPU. GPU-to-CPU copies happen at the capture boundary; PNG encoding, NPY writing and JSONL writing run through a bounded background queue. Run one Isaac process per GPU and assign disjoint episode IDs, for example worker `g` receives episode IDs `g, g + num_gpus, ...`. A bounded queue provides backpressure instead of dropping frames, preserving exact state/action/image alignment.

Appearance (room, panel and lights) is sampled once before each episode and remains unchanged until that episode terminates. The next episode gets a new seed. Passing `--scene-seed` reproduces the exact appearance.
