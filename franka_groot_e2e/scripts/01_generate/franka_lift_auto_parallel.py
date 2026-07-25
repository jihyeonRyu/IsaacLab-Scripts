#!/usr/bin/env python3
"""
Vectorized Isaac Lab Franka blue-cube pick-place dataset generator.

Multiple independent scenes, controllers, cameras, and recorders run in one
Kit process and advance through one batched ``env.step(actions)`` call.

Action convention:

- Task: Isaac-Lift-Cube-Franka-IK-Rel-v0
- Action shape: 7
- Action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
- Controller: autonomous Cartesian IK relative-action state machine

Run inside the Isaac Lab container:

    cd /workspace/isaaclab

    ./isaaclab.sh -p /workspace/scripts/franka_lift_auto_parallel.py \
      --num_envs 4 --auto_generate_episodes 100 --enable_cameras
"""

from __future__ import annotations

import argparse
import importlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import json
import math
import os
import re
from queue import Queue
import signal
import subprocess
import sys
from threading import Thread
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from isaaclab.app import AppLauncher


AUTO_TASK = "Isaac-Lift-Cube-Franka-IK-Rel-v0"
FIXED_CAMERA_EYE = (1.3, -1.3, 1.0)
FIXED_CAMERA_TARGET = (0.35, 0.0, 0.25)
EXTERNAL_CAMERA_FOCAL_LENGTH = 28.0
CAMERA_HORIZONTAL_APERTURE = 20.955
WRIST_CAMERA_EYE_EE = (0.10, 0.0, -0.08)
WRIST_CAMERA_TARGET_EE = (0.0, 0.0, 0.12)
WRIST_CAMERA_FOCAL_LENGTH = 10.0
PHYSICS_HZ = 120
CONTROL_HZ = 60
# Keep each backdrop just inside the 4 m vector-environment cell so adjacent
# environments cannot see or overlap each other's wall/floor geometry.
VECTOR_ENV_SPACING = 4.0
BACKDROP_WALL_SIZE = (3.9, 0.02, 4.0)
BACKDROP_SIDE_WALL_SIZE = (0.02, 3.9, 4.0)
BACKDROP_FLOOR_SIZE = (3.9, 3.9, 0.02)
BACKDROP_BACK_POS = (0.0, 1.94, 0.95)
BACKDROP_FRONT_POS = (0.0, -1.94, 0.95)
BACKDROP_LEFT_POS = (-1.94, 0.0, 0.95)
BACKDROP_RIGHT_POS = (1.94, 0.0, 0.95)
BACKDROP_FLOOR_POS = (0.0, 0.0, -1.059)
BACKDROP_CEILING_POS = (0.0, 0.0, 2.96)

TABLETOP_FRANKA_JOINT_POS = {
    "panda_joint1": 0.50,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    "panda_finger_joint.*": 0.0400,
}


def add_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automated Franka pick-place dataset generator.")
    parser.add_argument("--task", default=AUTO_TASK, help="Task name. Must be 7D IK-Rel compatible.")
    parser.add_argument(
        "--asset_version_override", default=None,
        help="Override the Assets/Isaac/<version> URL segment for robot/table USD compatibility.",
    )
    parser.add_argument("--num_envs", type=int, default=4, help="Vectorized environments in one Kit process.")
    parser.add_argument(
        "--parallel_workers", type=int, default=None,
        help="Compatibility alias for --num_envs; no extra Kit processes are launched.",
    )
    parser.add_argument(
        "--multi_gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Automatically use every visible GPU by launching one Isaac Lab process per GPU. "
            "Enabled by default when more than one GPU is visible; use --no-multi_gpu to disable."
        ),
    )
    parser.add_argument(
        "--gpu_ids",
        nargs="+",
        default=None,
        metavar="GPU",
        help=(
            "Optional GPU indices or UUIDs to use. By default CUDA_VISIBLE_DEVICES, "
            "NVIDIA_VISIBLE_DEVICES, or nvidia-smi is used to discover all visible GPUs."
        ),
    )
    parser.add_argument(
        "--multi_gpu_child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--physics_hz", type=int, default=PHYSICS_HZ,
        help="PhysX integration frequency. Must be an integer multiple of --control_hz.",
    )
    parser.add_argument(
        "--control_hz", type=int, default=CONTROL_HZ,
        help="Environment/action frequency. The requested sensor FPS must divide this exactly.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--output_dir", default="/workspace/output/franka_lift_clean")
    parser.add_argument(
        "--scenario",
        choices=("lift_cube", "blue_tray"),
        default="blue_tray",
        help="Scene layout. lift_cube keeps the built-in one-cube task; blue_tray adds random cubes and a tray.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for scenario generation.")
    parser.add_argument("--episode_index", type=int, default=0, help="Seed offset for generating another random scene.")
    parser.add_argument(
        "--validate_layouts_only",
        type=int,
        default=0,
        help="Generate and validate this many fixed-tray vector layouts, then exit before simulation.",
    )
    parser.add_argument("--min_blue_cubes", type=int, default=1)
    parser.add_argument("--max_blue_cubes", type=int, default=3)
    parser.add_argument("--min_red_cubes", type=int, default=2)
    parser.add_argument("--max_red_cubes", type=int, default=2)
    parser.add_argument("--cube_size", type=float, default=0.05, help="Legacy fixed size used when domain randomization is disabled.")
    parser.add_argument("--cube_size_range", type=float, nargs=2, default=(0.05, 0.065), metavar=("MIN", "MAX"), help="Per-axis cuboid size range in meters.")
    parser.add_argument("--min_spawn_spacing", type=float, default=0.04)
    # Safe SeattleLabTable top envelope. Sampling subtracts each object's half
    # extents, so the complete cube/tray remains inside it.
    parser.add_argument("--workspace_x_min", type=float, default=0.33)
    parser.add_argument("--workspace_x_max", type=float, default=0.70)
    parser.add_argument("--workspace_y_min", type=float, default=-0.34)
    parser.add_argument("--workspace_y_max", type=float, default=0.34)
    parser.add_argument(
        "--workspace_radius_max", type=float, default=0.68,
        help="Maximum XY distance from the robot base for loose objects and tray centers.",
    )
    parser.add_argument(
        "--target_workspace_bins",
        type=int,
        nargs=2,
        default=(4, 6),
        metavar=("X_BINS", "Y_BINS"),
        help="Stratified X/Y grid used to spread blue target spawns over the reachable workspace.",
    )
    parser.add_argument(
        "--stratified_target_positions",
        action="store_true",
        default=True,
        help="Cycle blue target spawns through workspace cells instead of biasing the first target away from the tray.",
    )
    parser.add_argument(
        "--no_stratified_target_positions",
        action="store_false",
        dest="stratified_target_positions",
        help="Use legacy distance-spread sampling for blue targets.",
    )
    parser.add_argument("--cube_z", type=float, default=0.026)
    parser.add_argument("--tray_z", type=float, default=0.013)
    parser.add_argument(
        "--settle_steps", type=int, default=72,
        help="Control steps before automation/recording starts (72 steps = 1.2 s at 60 Hz).",
    )
    parser.add_argument("--tray_size", type=float, nargs=2, default=(0.22, 0.18), metavar=("X", "Y"))
    parser.add_argument("--record", action="store_true", help="Write action/state JSONL logs.")
    parser.add_argument("--record_on_start", action="store_true", help="Start recording immediately instead of waiting for O.")
    parser.add_argument(
        "--record_sensors",
        action="store_true",
        help="Attach one camera and save RGB, depth, semantic segmentation, and instance segmentation.",
    )
    parser.add_argument("--record_rgb", action="store_true", help="Legacy alias for --record_sensors.")
    parser.add_argument(
        "--sensor_modalities",
        choices=("rgb", "rgb_depth", "all", "full"),
        default="all",
        help="'rgb', 'rgb_depth', or 'all'/'full'. full saves RGB, depth, semantic seg, and instance seg frames.",
    )
    parser.add_argument(
        "--capture_every_n",
        type=int,
        default=1,
        help="Save one sensor frame every N capture ticks. Use 2 or 3 to reduce rendering load.",
    )
    parser.add_argument(
        "--log_every_n",
        type=int,
        default=0,
        help="Write action/state logs every N sim steps. 0 matches the sensor capture cadence when sensors are enabled.",
    )
    parser.add_argument(
        "--segmentation_stats",
        action="store_true",
        help="Write expensive per-frame segmentation unique-count stats for debugging labels.",
    )
    parser.add_argument("--save_video", action="store_true", help="Also write rgb.mp4 during capture. Slower than PNG only.")
    parser.add_argument("--async_writes", action="store_true", default=True, help="Write images/arrays in background threads.")
    parser.add_argument("--sync_writes", action="store_false", dest="async_writes")
    parser.add_argument("--max_pending_writes", type=int, default=128, help="Backpressure limit for async file writes.")
    parser.add_argument("--video_queue_size", type=int, default=256, help="Buffered frame count for async RGB video writers.")
    parser.add_argument("--realtime", action="store_true", default=True, help="Throttle loop to sim step time.")
    parser.add_argument("--no_realtime", action="store_false", dest="realtime")
    parser.add_argument("--gripper_open_command", type=float, default=1.0, help="7D controller action value for open gripper.")
    parser.add_argument("--gripper_close_command", type=float, default=-1.0, help="7D controller action value for closed gripper.")
    parser.add_argument("--enable_wrist_camera", action="store_true", help="Attach and record a gripper-mounted camera.")
    parser.add_argument(
        "--wrist_camera_pos",
        type=float,
        nargs=3,
        default=WRIST_CAMERA_EYE_EE,
        metavar=("X", "Y", "Z"),
        help="Camera eye offset in the panda_hand local frame; +Z points toward the grasp target.",
    )
    parser.add_argument(
        "--wrist_camera_rot",
        type=float,
        nargs=4,
        default=(0.0, 0.0, 0.0, 1.0),
        metavar=("QX", "QY", "QZ", "QW"),
        help="Deprecated compatibility option; look-at targeting now controls wrist orientation.",
    )
    parser.add_argument(
        "--wrist_camera_target", type=float, nargs=3, default=WRIST_CAMERA_TARGET_EE,
        metavar=("X", "Y", "Z"),
        help="Look-at target offset in the panda_hand local frame.",
    )
    parser.add_argument(
        "--wrist_focal_length", type=float, default=WRIST_CAMERA_FOCAL_LENGTH,
        help="Wrist camera focal length in mm; 10 mm keeps approach and grasp context in view.",
    )
    parser.add_argument(
        "--wrist_follow_look_at", action=argparse.BooleanOptionalAction, default=False,
        help="Deprecated compatibility flag; vector wrist following is always enabled.",
    )
    parser.add_argument(
        "--wrist_follow_eye_offset", type=float, nargs=3, default=(0.0, -0.14, 0.16),
        metavar=("X", "Y", "Z"), help="Deprecated compatibility option.",
    )
    parser.add_argument(
        "--wrist_follow_target_offset", type=float, nargs=3, default=(0.0, 0.0, -0.06),
        metavar=("X", "Y", "Z"), help="Deprecated compatibility option.",
    )
    parser.add_argument(
        "--show_debug_visuals",
        action="store_true",
        help="Keep Isaac Lab debug/helper visuals such as EE axes and command markers visible.",
    )
    parser.add_argument(
        "--preview_view",
        choices=("free", "record_camera", "wrist_camera"),
        default="free",
        help="WebRTC viewport camera. free uses preview_eye/target; record/wrist look through sensor camera prims.",
    )
    parser.add_argument("--preview_eye", type=float, nargs=3, default=FIXED_CAMERA_EYE)
    parser.add_argument("--preview_target", type=float, nargs=3, default=FIXED_CAMERA_TARGET)
    parser.add_argument("--auto_generate_episodes", type=int, default=1, help="Generate N complete episodes, then exit. Enables full recording and video.")
    # Preserve the Cartesian/yaw speeds validated at the previous 50 Hz rate.
    parser.add_argument("--auto_pick_step", type=float, default=0.0375)
    parser.add_argument("--auto_pick_descend_step", type=float, default=0.015)
    parser.add_argument("--auto_pick_tolerance", type=float, default=0.012)
    parser.add_argument(
        "--auto_pick_recenter_tolerance",
        type=float,
        default=0.006,
        help="Tighter Cartesian tolerance used to re-center above a cube after yaw alignment.",
    )
    parser.add_argument("--auto_pick_approach_height", type=float, default=0.10)
    parser.add_argument("--auto_pick_lift_height", type=float, default=0.16)
    parser.add_argument(
        "--auto_pick_min_transport_lift",
        type=float,
        default=0.10,
        help="Advance from lift once the cube itself has risen this far, even if Cartesian x/y tracking is imperfect.",
    )
    parser.add_argument(
        "--auto_place_drop_height",
        type=float,
        default=0.025,
        help="Release the carried cube this many meters above its final tray slot to keep the fingers clear of tray walls.",
    )
    parser.add_argument("--auto_pick_yaw_step", type=float, default=0.0666666667)
    parser.add_argument("--auto_pick_yaw_tolerance", type=float, default=0.035)
    parser.add_argument("--auto_pick_yaw_offset", type=float, default=0.0)
    parser.add_argument("--auto_pick_hold_steps", type=int, default=22, help="22 steps is about 0.367 s at 60 Hz.")
    parser.add_argument("--auto_pick_state_timeout", type=int, default=1080, help="1080 steps is 18 s at 60 Hz.")
    parser.add_argument(
        "--randomize_start_pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Move the EEF to a deterministic random robot-front tabletop position before recording. "
            "The pre-roll translates only, preserving the validated floor-facing tool orientation."
        ),
    )
    parser.add_argument("--start_ee_x_range", type=float, nargs=2, default=(0.36, 0.70), metavar=("MIN", "MAX"))
    parser.add_argument("--start_ee_y_range", type=float, nargs=2, default=(-0.34, 0.34), metavar=("MIN", "MAX"))
    parser.add_argument("--start_ee_z_range", type=float, nargs=2, default=(0.25, 0.55), metavar=("MIN", "MAX"))
    parser.add_argument(
        "--start_ee_radius_min",
        type=float,
        default=0.40,
        help="Minimum XY radius that keeps randomized starts away from the robot-base singular region.",
    )
    parser.add_argument(
        "--start_ee_radius_max",
        type=float,
        default=0.72,
        help="Maximum XY radius for the higher, collision-free randomized start EEF pose.",
    )
    parser.add_argument("--start_pose_step", type=float, default=0.03)
    parser.add_argument("--start_pose_tolerance", type=float, default=0.012)
    parser.add_argument("--start_pose_timeout_steps", type=int, default=480)
    parser.add_argument("--start_pose_hold_steps", type=int, default=12)
    parser.add_argument(
        "--start_pose_max_tilt_deg",
        type=float,
        default=60.0,
        help="Maximum allowed angle between the tool +Z axis and world down during start-pose pre-roll.",
    )
    parser.add_argument(
        "--recovery_waypoint_prob",
        type=float,
        default=0.10,
        help="Probability per blue-cube target of visiting a nearby random waypoint before direct approach.",
    )
    parser.add_argument(
        "--recovery_waypoint_radius_range",
        type=float,
        nargs=2,
        default=(0.04, 0.08),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--recovery_waypoint_height_range",
        type=float,
        nargs=2,
        default=(0.12, 0.18),
        metavar=("MIN", "MAX"),
        help="Recovery waypoint height above the target cube center in meters.",
    )
    parser.add_argument(
        "--partial_progress_2_cube_prob",
        type=float,
        default=0.25,
        help="Probability that a 2-cube episode starts with one blue cube already placed.",
    )
    parser.add_argument(
        "--partial_progress_3_cube_prob",
        type=float,
        default=0.30,
        help="Probability that a 3-cube episode starts with one or two blue cubes already placed.",
    )
    parser.add_argument(
        "--partial_progress_3_cube_two_preplaced_prob",
        type=float,
        default=0.40,
        help="Conditional probability of preplacing two cubes in a partial 3-cube episode.",
    )
    parser.add_argument(
        "--partial_progress_start_xy_radius_range",
        type=float,
        nargs=2,
        default=(0.0, 0.05),
        metavar=("MIN", "MAX"),
        help="EEF XY offset radius around the most recently preplaced tray slot.",
    )
    parser.add_argument(
        "--partial_progress_start_clearance_range",
        type=float,
        nargs=2,
        default=(0.12, 0.20),
        metavar=("MIN", "MAX"),
        help="EEF clearance above the most recently preplaced cube top.",
    )
    parser.add_argument(
        "--solver_recovery_max_attempts",
        type=int,
        default=3,
        help="Maximum safe-raise/recenter retries per cube after IK motion stalls.",
    )
    parser.add_argument(
        "--solver_recovery_stall_steps",
        type=int,
        default=240,
        help="Trigger solver recovery after this many control steps without meaningful distance progress.",
    )
    parser.add_argument(
        "--solver_recovery_progress_epsilon",
        type=float,
        default=0.002,
        help="Target-distance improvement in meters that resets the IK stall counter.",
    )
    parser.add_argument(
        "--solver_recovery_yaw_progress_epsilon",
        type=float,
        default=0.01,
        help="Yaw-error improvement in radians that resets the align-yaw stall counter.",
    )
    parser.add_argument(
        "--solver_recovery_raise_clearance",
        type=float,
        default=0.10,
        help="Minimum vertical rise before moving to the neutral solver-recovery pose.",
    )
    parser.add_argument(
        "--solver_recovery_center",
        type=float,
        nargs=3,
        default=(0.45, 0.0, 0.38),
        metavar=("X", "Y", "Z"),
        help="High, robot-front EEF pose used to escape a poor differential-IK joint configuration.",
    )
    parser.add_argument(
        "--solver_recovery_reorient_step",
        type=float,
        default=0.12,
        help="Maximum raw axis-angle step used to restore the validated tool orientation at the recovery center.",
    )
    parser.add_argument(
        "--solver_recovery_reorient_tolerance",
        type=float,
        default=0.035,
        help="Angular tolerance in radians for completing solver-recovery tool reorientation.",
    )
    parser.add_argument(
        "--solver_recovery_reorient_max_steps",
        type=int,
        default=480,
        help="Maximum control steps spent on optional tool reorientation before a safe partial recovery.",
    )
    parser.add_argument("--grasp_min_width", type=float, default=0.012)
    parser.add_argument("--grasp_min_lift", type=float, default=0.025)
    parser.add_argument("--domain_randomization", action="store_true", default=True)
    parser.add_argument("--no_domain_randomization", action="store_false", dest="domain_randomization")
    parser.add_argument("--cube_mass_range", type=float, nargs=2, default=(0.035, 0.075), metavar=("MIN", "MAX"))
    parser.add_argument("--friction_range", type=float, nargs=2, default=(0.45, 1.10), metavar=("MIN", "MAX"))
    parser.add_argument("--restitution_range", type=float, nargs=2, default=(0.0, 0.12), metavar=("MIN", "MAX"))
    parser.add_argument(
        "--min_scene_lights", type=int, default=3,
        help="Total lights per episode. Defaults guarantee dome, distant key, and sphere fill lights.",
    )
    parser.add_argument("--max_scene_lights", type=int, default=5)
    parser.add_argument("--dome_light_intensity_range", type=float, nargs=2, default=(300.0, 1600.0), metavar=("MIN", "MAX"))
    parser.add_argument("--distant_light_intensity_range", type=float, nargs=2, default=(2000.0, 7000.0), metavar=("MIN", "MAX"))
    parser.add_argument("--sphere_light_intensity_range", type=float, nargs=2, default=(8000.0, 65000.0), metavar=("MIN", "MAX"))
    parser.add_argument("--sphere_light_position_min", type=float, nargs=3, default=(0.10, -0.90, 0.80), metavar=("X", "Y", "Z"))
    parser.add_argument("--sphere_light_position_max", type=float, nargs=3, default=(1.10, 0.90, 1.80), metavar=("X", "Y", "Z"))
    parser.add_argument("--sphere_light_scale_range", type=float, nargs=2, default=(0.05, 0.20), metavar=("MIN", "MAX"))
    parser.add_argument("--camera_position_jitter", type=float, nargs=3, default=(0.10, 0.10, 0.06), metavar=("X", "Y", "Z"))
    parser.add_argument("--camera_target_jitter", type=float, nargs=3, default=(0.04, 0.04, 0.03), metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--randomize_camera",
        action="store_true",
        help="Opt in to camera eye/target randomization. The dataset camera is fixed by default.",
    )
    parser.add_argument(
        "--enable_fabric",
        action="store_true",
        default=True,
        help="Use Fabric scene transforms for correct articulated robot rendering (default: enabled).",
    )
    parser.add_argument(
        "--disable_fabric",
        action="store_false",
        dest="enable_fabric",
        help="Disable Fabric only for renderer troubleshooting.",
    )
    parser.add_argument("--rgb_noise_std", type=float, default=3.0)
    parser.add_argument("--rgb_brightness_range", type=float, nargs=2, default=(0.95, 1.20), metavar=("MIN", "MAX"))
    parser.add_argument("--depth_noise_std", type=float, default=0.002)
    parser.add_argument("--depth_dropout_prob", type=float, default=0.002)
    parser.add_argument(
        "--external_depth_vis_range", type=float, nargs=2, default=(1.3, 2.8),
        metavar=("NEAR", "FAR"), help="Fixed metric range used only for external depth PNG/MP4.",
    )
    parser.add_argument(
        "--wrist_depth_vis_range", type=float, nargs=2, default=(0.04, 1.0),
        metavar=("NEAR", "FAR"), help="Fixed metric range used only for wrist depth PNG/MP4.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = add_args()
args_cli = parser.parse_args()


def discover_visible_gpus() -> list[str]:
    """Return GPU identifiers in the order child processes should use them."""
    if args_cli.gpu_ids:
        return [str(gpu_id).strip() for gpu_id in args_cli.gpu_ids if str(gpu_id).strip()]

    for variable_name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        value = os.environ.get(variable_name, "").strip()
        if not value or value.lower() in {"all", "void", "none"} or value == "-1":
            continue
        visible = [token.strip() for token in value.split(",") if token.strip()]
        if visible:
            return visible

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def replace_cli_option(argv: list[str], option: str, value: str) -> list[str]:
    """Replace a scalar CLI option while preserving every unrelated user argument."""
    rewritten: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == option:
            index += 2
            continue
        if argument.startswith(f"{option}="):
            index += 1
            continue
        rewritten.append(argument)
        index += 1
    rewritten.extend((option, value))
    return rewritten


def run_multi_gpu_parent(gpu_ids: list[str]) -> int:
    """Split episodes across GPUs and wait for all Isaac Lab child processes."""
    total_episodes = int(args_cli.auto_generate_episodes)
    first_episode = int(args_cli.episode_index)
    worker_count = min(len(gpu_ids), total_episodes)
    gpu_ids = gpu_ids[:worker_count]
    base_count, remainder = divmod(total_episodes, worker_count)
    assignments: list[dict[str, int | str]] = []
    processes: list[subprocess.Popen[Any]] = []
    next_episode = first_episode

    for rank, gpu_id in enumerate(gpu_ids):
        episode_count = base_count + (1 if rank < remainder else 0)
        child_argv = replace_cli_option(
            list(sys.argv[1:]), "--auto_generate_episodes", str(episode_count)
        )
        child_argv = replace_cli_option(child_argv, "--episode_index", str(next_episode))
        child_argv = replace_cli_option(child_argv, "--device", "cuda:0")
        child_argv.append("--multi_gpu_child")
        child_env = os.environ.copy()
        child_env.update(
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": gpu_id,
                "FRANKA_MULTI_GPU_RANK": str(rank),
                "FRANKA_MULTI_GPU_WORLD_SIZE": str(worker_count),
                "FRANKA_MULTI_GPU_ID": gpu_id,
                "PYTHONUNBUFFERED": "1",
            }
        )
        assignment: dict[str, int | str] = {
            "episode_index_start": next_episode,
            "episode_number_start": next_episode + 1,
            "episode_count": episode_count,
            "episode_number_end": next_episode + episode_count,
        }
        assignments.append(assignment)
        print(
            f"[MULTI-GPU] rank={rank}/{worker_count} gpu={gpu_id} "
            f"episodes={assignment['episode_number_start']}..{assignment['episode_number_end']}",
            flush=True,
        )
        processes.append(
            subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), *child_argv],
                env=child_env,
                start_new_session=True,
            )
        )
        next_episode += episode_count

    def stop_children(signum: int, _frame: Any) -> None:
        print(f"[MULTI-GPU] signal={signum}; stopping workers", flush=True)
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    signal.signal(signal.SIGINT, stop_children)
    signal.signal(signal.SIGTERM, stop_children)
    exit_codes = [process.wait() for process in processes]

    output_root = Path(args_cli.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    workers: list[dict[str, Any]] = []
    for rank, (gpu_id, assignment, exit_code) in enumerate(
        zip(gpu_ids, assignments, exit_codes)
    ):
        summary_path = output_root / f"vectorized_summary_gpu_{rank:02d}.json"
        try:
            worker_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            worker_summary = {"summary_error": str(exc)}
        workers.append(
            {
                "rank": rank,
                "gpu_id": gpu_id,
                "exit_code": exit_code,
                "assignment": assignment,
                "summary_path": str(summary_path),
                "summary": worker_summary,
            }
        )
    reported = sum(int(worker["summary"].get("reported_episodes", 0)) for worker in workers)
    successful = sum(int(worker["summary"].get("successful_episodes", 0)) for worker in workers)
    aggregate = {
        "execution_mode": "multi_gpu_processes_with_vectorized_envs",
        "gpu_count": worker_count,
        "gpu_ids": gpu_ids,
        "num_envs_per_gpu": int(args_cli.parallel_workers or args_cli.num_envs),
        "requested_episodes": total_episodes,
        "reported_episodes": reported,
        "successful_episodes": successful,
        "failed_episodes": reported - successful,
        "all_workers_exited_cleanly": all(code == 0 for code in exit_codes),
        "workers": workers,
    }
    summary_path = output_root / "multi_gpu_summary.json"
    summary_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"[MULTI-GPU] aggregate summary={summary_path}", flush=True)
    if all(code == 0 for code in exit_codes):
        print("[MULTI-GPU] all workers completed successfully", flush=True)
        return 0
    print(f"[MULTI-GPU] worker exit codes={exit_codes}", flush=True)
    return next((code for code in exit_codes if code != 0), 1)


visible_gpus = discover_visible_gpus()
if (
    args_cli.multi_gpu
    and not args_cli.multi_gpu_child
    and args_cli.auto_generate_episodes > 0
    and len(visible_gpus) > 1
):
    raise SystemExit(run_multi_gpu_parent(visible_gpus))
if args_cli.parallel_workers is not None:
    args_cli.num_envs = int(args_cli.parallel_workers)
if args_cli.num_envs < 1:
    parser.error("--num_envs must be >= 1.")
if args_cli.fps <= 0:
    parser.error("--fps must be > 0.")
if args_cli.physics_hz <= 0 or args_cli.control_hz <= 0:
    parser.error("--physics_hz and --control_hz must be > 0.")
if args_cli.physics_hz % args_cli.control_hz != 0:
    parser.error("--physics_hz must be an integer multiple of --control_hz.")
if args_cli.control_hz % args_cli.fps != 0:
    parser.error("--fps must divide --control_hz exactly; use 120/60/15 for exact 15 FPS capture.")
if args_cli.capture_every_n < 1:
    parser.error("--capture_every_n must be >= 1.")
if args_cli.settle_steps < 0:
    parser.error("--settle_steps must be >= 0.")
if args_cli.log_every_n < 0:
    parser.error("--log_every_n must be >= 0.")
if args_cli.video_queue_size < 1:
    parser.error("--video_queue_size must be >= 1.")
if args_cli.wrist_focal_length <= 0:
    parser.error("--wrist_focal_length must be > 0.")
if args_cli.min_blue_cubes < 1 or args_cli.max_blue_cubes < args_cli.min_blue_cubes:
    parser.error("blue cube range must satisfy 1 <= min_blue_cubes <= max_blue_cubes.")
if args_cli.min_red_cubes < 0 or args_cli.max_red_cubes < args_cli.min_red_cubes:
    parser.error("red cube range must satisfy 0 <= min_red_cubes <= max_red_cubes.")
if args_cli.workspace_x_max <= args_cli.workspace_x_min or args_cli.workspace_y_max <= args_cli.workspace_y_min:
    parser.error("workspace max bounds must be greater than min bounds.")
if args_cli.workspace_radius_max <= 0:
    parser.error("--workspace_radius_max must be > 0.")
if any(int(value) < 1 for value in args_cli.target_workspace_bins):
    parser.error("--target_workspace_bins values must both be >= 1.")
if math.prod(int(value) for value in args_cli.target_workspace_bins) < args_cli.max_blue_cubes:
    parser.error("--target_workspace_bins must contain at least max_blue_cubes cells.")
if (
    args_cli.start_ee_radius_min <= 0
    or args_cli.start_ee_radius_max < args_cli.start_ee_radius_min
):
    parser.error("start EEF radii must satisfy 0 < MIN <= MAX.")
if args_cli.cube_size <= 0:
    parser.error("--cube_size must be > 0.")
if args_cli.cube_size_range[0] <= 0 or args_cli.cube_size_range[1] < args_cli.cube_size_range[0]:
    parser.error("--cube_size_range must satisfy 0 < MIN <= MAX.")
if args_cli.min_scene_lights < 1 or args_cli.max_scene_lights < args_cli.min_scene_lights:
    parser.error("scene light count must satisfy 1 <= MIN <= MAX.")
for range_name in ("dome_light_intensity_range", "distant_light_intensity_range", "sphere_light_intensity_range", "sphere_light_scale_range"):
    low, high = map(float, getattr(args_cli, range_name))
    if low <= 0.0 or high < low:
        parser.error(f"--{range_name} must satisfy 0 < MIN <= MAX.")
if any(float(hi) < float(lo) for lo, hi in zip(args_cli.sphere_light_position_min, args_cli.sphere_light_position_max)):
    parser.error("--sphere_light_position_max must be >= min on every axis.")
if args_cli.tray_size[0] <= 0 or args_cli.tray_size[1] <= 0:
    parser.error("--tray_size values must be > 0.")
if args_cli.auto_generate_episodes < 0:
    parser.error("--auto_generate_episodes must be >= 0.")
if args_cli.validate_layouts_only < 0:
    parser.error("--validate_layouts_only must be >= 0.")
if (
    args_cli.auto_pick_step <= 0
    or args_cli.auto_pick_descend_step <= 0
    or args_cli.auto_pick_tolerance <= 0
    or args_cli.auto_pick_recenter_tolerance <= 0
):
    parser.error("automatic pick motion step/tolerance values must be > 0.")
if float(args_cli.auto_pick_recenter_tolerance) > float(args_cli.auto_pick_tolerance):
    parser.error("--auto_pick_recenter_tolerance must be <= --auto_pick_tolerance.")
if args_cli.auto_pick_hold_steps < 1 or args_cli.auto_pick_state_timeout < 1:
    parser.error("automatic pick hold/timeout step counts must be >= 1.")
for range_name in ("start_ee_x_range", "start_ee_y_range", "start_ee_z_range"):
    low, high = map(float, getattr(args_cli, range_name))
    if high < low:
        parser.error(f"--{range_name} must satisfy MIN <= MAX.")
if float(args_cli.start_ee_x_range[0]) <= 0.0:
    parser.error("--start_ee_x_range must stay in front of the robot base (MIN > 0).")
if float(args_cli.start_ee_z_range[0]) < 0.25:
    parser.error("--start_ee_z_range MIN must be >= 0.25 m for tabletop clearance.")
if (
    float(args_cli.start_ee_x_range[0]) < float(args_cli.workspace_x_min)
    or float(args_cli.start_ee_x_range[1]) > float(args_cli.workspace_x_max)
    or float(args_cli.start_ee_y_range[0]) < float(args_cli.workspace_y_min)
    or float(args_cli.start_ee_y_range[1]) > float(args_cli.workspace_y_max)
):
    parser.error("start EEF XY ranges must remain inside the configured robot-front workspace.")
if args_cli.start_pose_step <= 0 or args_cli.start_pose_tolerance <= 0:
    parser.error("start pose step/tolerance values must be > 0.")
if args_cli.start_pose_timeout_steps < 1 or args_cli.start_pose_hold_steps < 0:
    parser.error("start pose timeout must be >= 1 and hold steps must be >= 0.")
if not 0.0 < float(args_cli.start_pose_max_tilt_deg) <= 75.0:
    parser.error("--start_pose_max_tilt_deg must be in (0, 75].")
if not 0.0 <= float(args_cli.recovery_waypoint_prob) <= 1.0:
    parser.error("--recovery_waypoint_prob must be in [0, 1].")
recovery_radius_min, recovery_radius_max = map(float, args_cli.recovery_waypoint_radius_range)
if recovery_radius_min <= 0.0 or recovery_radius_max < recovery_radius_min:
    parser.error("--recovery_waypoint_radius_range must satisfy 0 < MIN <= MAX.")
recovery_height_min, recovery_height_max = map(float, args_cli.recovery_waypoint_height_range)
if recovery_height_min < float(args_cli.auto_pick_approach_height) or recovery_height_max < recovery_height_min:
    parser.error("recovery waypoint heights must satisfy approach_height <= MIN <= MAX.")
for probability_name in (
    "partial_progress_2_cube_prob",
    "partial_progress_3_cube_prob",
    "partial_progress_3_cube_two_preplaced_prob",
):
    probability = float(getattr(args_cli, probability_name))
    if not 0.0 <= probability <= 1.0:
        parser.error(f"--{probability_name} must be in [0, 1].")
partial_xy_min, partial_xy_max = map(
    float, args_cli.partial_progress_start_xy_radius_range
)
if partial_xy_min < 0.0 or partial_xy_max < partial_xy_min:
    parser.error(
        "--partial_progress_start_xy_radius_range must satisfy 0 <= MIN <= MAX."
    )
partial_clearance_min, partial_clearance_max = map(
    float, args_cli.partial_progress_start_clearance_range
)
if (
    partial_clearance_min < float(args_cli.auto_pick_approach_height)
    or partial_clearance_max < partial_clearance_min
):
    parser.error(
        "--partial_progress_start_clearance_range must satisfy "
        "approach_height <= MIN <= MAX."
    )
if args_cli.solver_recovery_max_attempts < 0:
    parser.error("--solver_recovery_max_attempts must be >= 0.")
if args_cli.solver_recovery_stall_steps < 1:
    parser.error("--solver_recovery_stall_steps must be >= 1.")
if args_cli.solver_recovery_progress_epsilon <= 0.0:
    parser.error("--solver_recovery_progress_epsilon must be > 0.")
if args_cli.solver_recovery_yaw_progress_epsilon <= 0.0:
    parser.error("--solver_recovery_yaw_progress_epsilon must be > 0.")
if args_cli.solver_recovery_raise_clearance <= 0.0:
    parser.error("--solver_recovery_raise_clearance must be > 0.")
if args_cli.solver_recovery_reorient_step <= 0.0:
    parser.error("--solver_recovery_reorient_step must be > 0.")
if not 0.0 < args_cli.solver_recovery_reorient_tolerance <= args_cli.solver_recovery_reorient_step:
    parser.error(
        "--solver_recovery_reorient_tolerance must be > 0 and <= "
        "--solver_recovery_reorient_step."
    )
if args_cli.solver_recovery_reorient_max_steps < 1:
    parser.error("--solver_recovery_reorient_max_steps must be >= 1.")
solver_center_x, solver_center_y, solver_center_z = map(float, args_cli.solver_recovery_center)
if not (
    float(args_cli.workspace_x_min) <= solver_center_x <= float(args_cli.workspace_x_max)
    and float(args_cli.workspace_y_min) <= solver_center_y <= float(args_cli.workspace_y_max)
    and math.hypot(solver_center_x, solver_center_y) <= float(args_cli.workspace_radius_max)
):
    parser.error("solver recovery center XY must remain inside the reachable robot-front workspace.")
if solver_center_z < 0.30:
    parser.error("solver recovery center Z must be >= 0.30 m for obstacle clearance.")
if not 0.0 <= args_cli.depth_dropout_prob < 1.0:
    parser.error("--depth_dropout_prob must be in [0, 1).")
for range_name in ("external_depth_vis_range", "wrist_depth_vis_range"):
    near, far = map(float, getattr(args_cli, range_name))
    if near < 0.0 or far <= near:
        parser.error(f"--{range_name} must satisfy 0 <= NEAR < FAR.")
if args_cli.preview_view == "wrist_camera":
    args_cli.enable_wrist_camera = True

if args_cli.record_rgb:
    args_cli.record_sensors = True
if args_cli.record_sensors:
    args_cli.record = True
if args_cli.auto_generate_episodes:
    args_cli.scenario = "blue_tray"
    args_cli.enable_wrist_camera = True
    args_cli.record_sensors = True
    args_cli.record = True
    args_cli.record_on_start = True
    args_cli.save_video = True
    args_cli.realtime = False


args_cli.enable_cameras = True
if args_cli.enable_fabric:
    streaming_args = (
        "--/rtx/hydra/progressiveSceneLoad=false "
        "--/rtx/hydra/geometrySyncLoads=true "
        "--/rtx-transient/hydra/geometrystreaming/syncLoad=true"
    )
    args_cli.kit_args = f"{getattr(args_cli, 'kit_args', '')} {streaming_args}".strip()
if not args_cli.enable_fabric:
    rtx_fabric_off = "--/rtx/hydra/readTransformsFromFabricInRenderDelegate=false"
    args_cli.kit_args = f"{getattr(args_cli, 'kit_args', '')} {rtx_fabric_off}".strip()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Imports that require Kit/SimulationApp to be running.
import carb
import gymnasium as gym
import imageio.v2 as imageio
import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401  # registers task names
import omni.kit.app
import omni.kit.viewport.utility as viewport_utils
import omni.usd
import omni.replicator.core as rep
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
import isaaclab.envs.mdp as event_mdp
from isaaclab.sensors import CameraCfg
from isaaclab.utils.math import compute_pose_error, quat_apply
from isaaclab_tasks.utils import parse_env_cfg
from pxr import Gf, Sdf, Semantics, UsdGeom, UsdLux

if not args_cli.enable_fabric:
    carb.settings.get_settings().set_bool("/rtx/hydra/readTransformsFromFabricInRenderDelegate", False)
else:
    rtx_settings = carb.settings.get_settings()
    rtx_settings.set_bool("/rtx/hydra/progressiveSceneLoad", False)
    rtx_settings.set_bool("/rtx/hydra/geometrySyncLoads", True)
    rtx_settings.set_bool("/rtx-transient/hydra/geometrystreaming/syncLoad", True)


def tensor_to_numpy(value: Any) -> np.ndarray:
    # Isaac Lab 3.0 may expose CUDA tensors from a separately loaded torch
    # module, so capability checks are more reliable than isinstance here.
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def env_array(value: Any, env_id: int = 0) -> np.ndarray:
    """Extract one environment from a batched camera output."""
    arr = tensor_to_numpy(value)
    if arr.ndim >= 4:
        arr = arr[env_id]
    return arr


def write_jsonl(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def rgb_to_array(rgb: Any, env_id: int = 0) -> np.ndarray:
    array = env_array(rgb, env_id)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return np.ascontiguousarray(array.copy())


def depth_to_arrays(depth_value: Any, vis_range: tuple[float, float], env_id: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return metric float depth plus a stable, near-is-bright 8-bit visualization."""
    depth = env_array(depth_value, env_id).squeeze().astype(np.float32)
    depth = np.ascontiguousarray(depth.copy())

    near, far = map(float, vis_range)
    valid = np.isfinite(depth)
    if valid.any():
        depth_vis = 1.0 - np.clip((depth - near) / (far - near), 0.0, 1.0)
        depth_vis[~valid] = 0.0
    else:
        depth_vis = np.zeros_like(depth, dtype=np.float32)
    return depth, np.ascontiguousarray((depth_vis * 255).astype(np.uint8))


def segmentation_to_image(seg_value: Any, env_id: int = 0) -> np.ndarray:
    """
    Return a uint8 RGB segmentation preview.

    Isaac Lab may return already-colorized RGB/RGBA output. If it returns integer
    IDs instead, make a deterministic pseudo-color image so the saved PNG is
    inspectable.
    """
    seg = env_array(seg_value, env_id)
    if seg.ndim == 3 and seg.shape[-1] in (3, 4):
        img = seg[..., :3]
        if img.dtype != np.uint8:
            if np.nanmax(img) <= 1.0:
                img = img * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        labels = seg.squeeze().astype(np.int32)
        img = np.stack(
            [(labels * 53) % 255, (labels * 97) % 255, (labels * 193) % 255],
            axis=-1,
        ).astype(np.uint8)
    return np.ascontiguousarray(img.copy())


def segmentation_stats(seg_value: Any, env_id: int = 0) -> dict[str, Any]:
    """Small debug summary for checking all-zero or all-one segmentation."""
    seg = env_array(seg_value, env_id)
    if seg.ndim == 3 and seg.shape[-1] in (3, 4):
        unique_count = np.unique(seg.reshape(-1, seg.shape[-1]), axis=0).shape[0]
    else:
        unique_count = np.unique(seg).shape[0]
    return {
        "shape": list(seg.shape),
        "dtype": str(seg.dtype),
        "min": float(np.nanmin(seg)),
        "max": float(np.nanmax(seg)),
        "unique_count_sample": int(min(unique_count, 999999)),
    }


def camera_data_types() -> list[str]:
    """Sensor modalities requested from the record camera."""
    if not args_cli.record_sensors:
        return ["rgb"]
    if args_cli.sensor_modalities == "rgb":
        return ["rgb"]
    if args_cli.sensor_modalities == "rgb_depth":
        return ["rgb", "distance_to_image_plane"]
    return [
        "rgb",
        "distance_to_image_plane",
        "semantic_segmentation",
        "instance_segmentation_fast",
    ]


def needs_semantic_labels() -> bool:
    data_types = camera_data_types()
    return "semantic_segmentation" in data_types or "instance_segmentation_fast" in data_types



def write_rgb_png(rgb: np.ndarray, path: Path) -> None:
    Image.fromarray(rgb).save(path)


def write_depth_files(depth: np.ndarray, depth_png: np.ndarray, npy_path: Path, png_path: Path) -> None:
    np.save(npy_path, depth)
    Image.fromarray(depth_png).save(png_path)


def write_image_png(img: np.ndarray, path: Path) -> None:
    Image.fromarray(img).save(path)


def submit_or_run(executor: ThreadPoolExecutor | None, pending_writes: list[Any], fn, *args) -> None:
    if executor is None:
        fn(*args)
        return
    pending_writes.append(executor.submit(fn, *args))


def drain_completed_writes(pending_writes: list[Any]) -> None:
    still_pending = []
    for future in pending_writes:
        if future.done():
            future.result()
        else:
            still_pending.append(future)
    pending_writes[:] = still_pending


def apply_write_backpressure(pending_writes: list[Any]) -> None:
    while len(pending_writes) > args_cli.max_pending_writes:
        pending_writes.pop(0).result()
    drain_completed_writes(pending_writes)


class AsyncVideoWriter:
    """Single-writer background queue for RGB mp4 encoding."""

    def __init__(self, path: Path, fps: int, max_queue: int):
        self.path = Path(path)
        self.fps = float(fps)
        self.queue: Queue[np.ndarray | None] = Queue(maxsize=max_queue)
        self.error: BaseException | None = None
        self.thread = Thread(target=self._run, name=f"video-writer-{self.path.name}", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        writer = None
        try:
            writer = imageio.get_writer(self.path, fps=self.fps)
            while True:
                frame = self.queue.get()
                if frame is None:
                    break
                writer.append_data(frame)
        except BaseException as exc:
            self.error = exc
        finally:
            if writer is not None:
                writer.close()

    def append_data(self, frame: np.ndarray) -> None:
        if self.error is not None:
            raise RuntimeError(f"async video writer failed for {self.path}: {self.error}") from self.error
        self.queue.put(np.ascontiguousarray(frame.copy()))

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join()
        if self.error is not None:
            raise RuntimeError(f"async video writer failed for {self.path}: {self.error}") from self.error


def output_sensor_fps() -> float:
    return float(args_cli.fps) / float(args_cli.capture_every_n)


def make_video_writer(path: Path):
    fps = output_sensor_fps()
    if args_cli.async_writes:
        return AsyncVideoWriter(path, fps, args_cli.video_queue_size)
    return imageio.get_writer(path, fps=fps)


Color = tuple[float, float, float]
Vec3 = tuple[float, float, float]


@dataclass
class ScenarioObjectSpec:
    name: str
    role: str
    asset_name: str
    prim_path: str
    pos: Vec3
    size: Vec3
    color: Color
    builtin: bool = False
    yaw: float = 0.0
    mass: float = 0.05
    static_friction: float = 0.8
    dynamic_friction: float = 0.7
    restitution: float = 0.0
    recovery_waypoint: Vec3 | None = None
    position_stratum: int | None = None
    preplaced: bool = False
    initial_placement_slot: int | None = None


@dataclass
class ScenarioLightSpec:
    light_type: str
    intensity: float
    color: Color
    position: Vec3 | None = None
    rotation: Vec3 | None = None
    scale: float | None = None


@dataclass
class ScenarioSpec:
    episode_index: int
    seed: int
    mode: str
    blue_cubes: list[ScenarioObjectSpec]
    red_cubes: list[ScenarioObjectSpec]
    tray_pos: Vec3
    tray_size: Vec3
    tray_color: Color
    table_color: Color
    light_intensity: float
    light_color: Color
    lights: list[ScenarioLightSpec]
    rgb_brightness_gain: float
    placement_positions: list[Vec3]
    camera_eye: Vec3
    camera_target: Vec3
    background_color: Color
    progress_stage: int = 0
    num_blue_total: int = 0
    num_preplaced: int = 0
    num_remaining: int = 0
    preplaced_blue_cube_names: list[str] = field(default_factory=list)
    remaining_blue_cube_names: list[str] = field(default_factory=list)
    start_pose_mode: str = "random_workspace"
    start_ee_target: Vec3 | None = None
    start_ee_actual: Vec3 | None = None
    start_ee_tilt_deg: float | None = None
    start_ee_reached: bool | None = None


def yaw_quat_xyzw(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


def cube_is_inside_tray(cube_pos_env: torch.Tensor, scenario: ScenarioSpec, margin: float = 0.015) -> bool:
    """Return True when a cube center is already inside the tray footprint."""
    tray_pos = scenario.tray_pos
    tray_size = scenario.tray_size
    half_x = max(0.0, float(tray_size[0]) * 0.5 - margin)
    half_y = max(0.0, float(tray_size[1]) * 0.5 - margin)
    dx = abs(float(cube_pos_env[0].item()) - float(tray_pos[0]))
    dy = abs(float(cube_pos_env[1].item()) - float(tray_pos[1]))
    return dx <= half_x and dy <= half_y


def jitter_color(rng: np.random.Generator, base: Color, amount: float = 0.08) -> Color:
    color = np.asarray(base, dtype=np.float32) + rng.uniform(-amount, amount, size=3)
    return tuple(np.clip(color, 0.0, 1.0).round(4).tolist())  # type: ignore[return-value]


def sample_uniform_rgb(rng: np.random.Generator) -> Color:
    """Sample every background RGB channel independently over the full range."""
    return tuple(rng.uniform(0.0, 1.0, size=3).round(4).tolist())  # type: ignore[return-value]


def sample_cuboid_size(rng: np.random.Generator) -> Vec3:
    """Sample XYZ dimensions while keeping the object inside Franka finger reach."""
    if not args_cli.domain_randomization:
        return (float(args_cli.cube_size),) * 3
    low, high = map(float, args_cli.cube_size_range)
    return tuple(rng.uniform(low, high, size=3).round(4).tolist())  # type: ignore[return-value]


def cuboid_center_z(size_z: float) -> float:
    """Keep every randomized cuboid bottom on the original tabletop plane."""
    table_surface_z = float(args_cli.cube_z) - 0.5 * float(args_cli.cube_size)
    return table_surface_z + 0.5 * float(size_z)


def sample_start_ee_target(rng: np.random.Generator) -> Vec3 | None:
    """Sample a reachable robot-front EEF position without changing tool orientation."""
    if not args_cli.randomize_start_pose:
        return None
    bounds = (args_cli.start_ee_x_range, args_cli.start_ee_y_range, args_cli.start_ee_z_range)
    for _ in range(256):
        target = tuple(float(rng.uniform(*axis_bounds)) for axis_bounds in bounds)
        radius = math.hypot(target[0], target[1])
        if (
            float(args_cli.start_ee_radius_min)
            <= radius
            <= float(args_cli.start_ee_radius_max)
        ):
            return tuple(round(value, 6) for value in target)  # type: ignore[return-value]
    raise RuntimeError("failed to sample a reachable randomized start EEF target")


def sample_recovery_waypoint(
    rng: np.random.Generator, cube: ScenarioObjectSpec
) -> Vec3 | None:
    """Sample a safe nearby waypoint for a deliberate off-target recovery approach."""
    if rng.random() >= float(args_cli.recovery_waypoint_prob):
        return None
    margin = 0.02
    for _ in range(256):
        angle = float(rng.uniform(-math.pi, math.pi))
        radius = float(rng.uniform(*args_cli.recovery_waypoint_radius_range))
        xy = (cube.pos[0] + radius * math.cos(angle), cube.pos[1] + radius * math.sin(angle))
        if not float(args_cli.workspace_x_min) + margin <= xy[0] <= float(args_cli.workspace_x_max) - margin:
            continue
        if not float(args_cli.workspace_y_min) + margin <= xy[1] <= float(args_cli.workspace_y_max) - margin:
            continue
        if not within_workspace_reach(xy):
            continue
        height = float(rng.uniform(*args_cli.recovery_waypoint_height_range))
        return (round(xy[0], 6), round(xy[1], 6), round(cube.pos[2] + height, 6))


def sample_partial_progress_count(
    rng: np.random.Generator, blue_cube_count: int
) -> int:
    """Sample how many blue cubes are already complete at episode start."""
    if blue_cube_count == 2:
        return int(rng.random() < float(args_cli.partial_progress_2_cube_prob))
    if blue_cube_count == 3 and rng.random() < float(
        args_cli.partial_progress_3_cube_prob
    ):
        return 2 if rng.random() < float(
            args_cli.partial_progress_3_cube_two_preplaced_prob
        ) else 1
    return 0


def preplace_blue_cubes(
    rng: np.random.Generator,
    blue_cubes: list[ScenarioObjectSpec],
    placement_positions: list[Vec3],
    tray_pos: Vec3,
    tray_size: Vec3,
    num_preplaced: int,
) -> list[ScenarioObjectSpec]:
    """Place a deterministic random subset into the first completed tray slots."""
    if not 0 <= num_preplaced < len(blue_cubes):
        raise RuntimeError(
            f"invalid partial-progress count {num_preplaced} for {len(blue_cubes)} cubes"
        )
    if num_preplaced == 0:
        return []
    selected_indices = [
        int(index)
        for index in rng.choice(len(blue_cubes), size=num_preplaced, replace=False)
    ]
    tray_top = float(tray_pos[2]) + 0.5 * float(tray_size[2])
    preplaced: list[ScenarioObjectSpec] = []
    for slot_index, cube_index in enumerate(selected_indices):
        cube = blue_cubes[cube_index]
        slot = placement_positions[slot_index]
        cube.pos = (
            float(slot[0]),
            float(slot[1]),
            tray_top + 0.5 * float(cube.size[2]) + 0.003,
        )
        cube.preplaced = True
        cube.initial_placement_slot = slot_index
        cube.position_stratum = None
        cube.recovery_waypoint = None
        preplaced.append(cube)
    return preplaced


def sample_partial_progress_start_ee_target(
    rng: np.random.Generator, last_preplaced_cube: ScenarioObjectSpec
) -> Vec3 | None:
    """Sample a safe post-release retreat pose above the last completed cube."""
    if not args_cli.randomize_start_pose:
        return None
    margin = 0.01
    cube_top = float(last_preplaced_cube.pos[2]) + 0.5 * float(
        last_preplaced_cube.size[2]
    )
    for _ in range(256):
        angle = float(rng.uniform(-math.pi, math.pi))
        radius = float(rng.uniform(*args_cli.partial_progress_start_xy_radius_range))
        x = float(last_preplaced_cube.pos[0]) + radius * math.cos(angle)
        y = float(last_preplaced_cube.pos[1]) + radius * math.sin(angle)
        if not (
            float(args_cli.workspace_x_min) + margin
            <= x
            <= float(args_cli.workspace_x_max) - margin
            and float(args_cli.workspace_y_min) + margin
            <= y
            <= float(args_cli.workspace_y_max) - margin
            and within_workspace_reach((x, y))
        ):
            continue
        clearance = float(
            rng.uniform(*args_cli.partial_progress_start_clearance_range)
        )
        return (round(x, 6), round(y, 6), round(cube_top + clearance, 6))
    raise RuntimeError("failed to sample a reachable partial-progress retreat pose")


def sample_cube_physics(rng: np.random.Generator) -> tuple[float, float, float, float]:
    if not args_cli.domain_randomization:
        return 0.05, 0.8, 0.7, 0.0
    mass = float(rng.uniform(*args_cli.cube_mass_range))
    friction_min, friction_max = map(float, args_cli.friction_range)
    static_friction = float(rng.uniform(friction_min, friction_max))
    dynamic_friction = float(rng.uniform(friction_min, static_friction))
    restitution = float(rng.uniform(*args_cli.restitution_range))
    return mass, static_friction, dynamic_friction, restitution


def sample_log_uniform(rng: np.random.Generator, bounds: tuple[float, float] | list[float]) -> float:
    """Sample light intensity evenly in log space so dim and bright scales both occur."""
    low, high = map(float, bounds)
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def generate_episode_lights(
    rng: np.random.Generator, dome_intensity: float, light_color: Color, background_color: Color
) -> list[ScenarioLightSpec]:
    """Sample a fixed light rig for one episode; no per-frame Replicator trigger is used."""
    # Keep illumination neutral and bright. The visibly randomized background is
    # rendered by a separate backdrop instead of abusing the Dome light color.
    dome_color = tuple(0.78 + 0.22 * float(channel) for channel in background_color)
    if not args_cli.domain_randomization:
        return [
            ScenarioLightSpec("dome", dome_intensity, dome_color),
            ScenarioLightSpec("distant", 3000.0, light_color, rotation=(35.0, 0.0, 0.0)),
            ScenarioLightSpec(
                "sphere", 24000.0, light_color, position=(0.55, -0.35, 1.35), scale=0.12
            ),
        ]

    light_count = int(rng.integers(args_cli.min_scene_lights, args_cli.max_scene_lights + 1))
    lights = [ScenarioLightSpec("dome", dome_intensity, dome_color)]
    position_min = np.asarray(args_cli.sphere_light_position_min, dtype=np.float32)
    position_max = np.asarray(args_cli.sphere_light_position_max, dtype=np.float32)
    light_palette: list[Color] = [(1.0, 0.92, 0.80), (0.82, 0.90, 1.0), (1.0, 0.98, 0.92)]
    for index in range(light_count - 1):
        # A distant key has no inverse-square falloff, so dark randomized
        # backgrounds never make the task unreadable. The second light is a sphere fill.
        light_type = "distant" if index == 0 else (
            "sphere" if index == 1 or rng.random() < 0.65 else "distant")
        color = jitter_color(rng, light_palette[int(rng.integers(0, len(light_palette)))], 0.08)
        if light_type == "sphere":
            lights.append(
                ScenarioLightSpec(
                    light_type="sphere",
                    intensity=sample_log_uniform(rng, args_cli.sphere_light_intensity_range),
                    color=color,
                    position=tuple(rng.uniform(position_min, position_max).round(4).tolist()),
                    scale=float(rng.uniform(*args_cli.sphere_light_scale_range)),
                )
            )
        else:
            lights.append(
                ScenarioLightSpec(
                    light_type="distant",
                    intensity=sample_log_uniform(rng, args_cli.distant_light_intensity_range),
                    color=color,
                    rotation=(
                        float(rng.uniform(20.0, 70.0)),
                        float(rng.uniform(0.0, 360.0)),
                        float(rng.uniform(-15.0, 15.0)),
                    ),
                )
            )
    return lights


def within_workspace_reach(xy: tuple[float, float]) -> bool:
    """Reject rectangular workspace corners outside the reliable Franka reach envelope."""
    return math.hypot(float(xy[0]), float(xy[1])) <= float(args_cli.workspace_radius_max)


def sample_xy_in_bounds(
    rng: np.random.Generator,
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    half_extents: tuple[float, float],
) -> tuple[float, float]:
    x = float(rng.uniform(x_bounds[0] + half_extents[0], x_bounds[1] - half_extents[0]))
    y = float(rng.uniform(y_bounds[0] + half_extents[1], y_bounds[1] - half_extents[1]))
    return x, y


def overlaps_2d(
    xy: tuple[float, float],
    half_extents: tuple[float, float],
    occupied: list[tuple[float, float, float, float, str]],
    margin: float,
) -> bool:
    x, y = xy
    hx, hy = half_extents
    for ox, oy, ohx, ohy, _label in occupied:
        if abs(x - ox) < hx + ohx + margin and abs(y - oy) < hy + ohy + margin:
            return True
    return False


def validate_scenario_spawn_clearance(
    scenario: ScenarioSpec, margin: float | None = None
) -> None:
    """Validate tray-clear loose objects and explicitly marked preplaced targets."""
    clearance = 0.0 if margin is None else float(margin)
    tray_half_x = 0.5 * float(scenario.tray_size[0])
    tray_half_y = 0.5 * float(scenario.tray_size[1])
    tray_top = float(scenario.tray_pos[2]) + 0.5 * float(scenario.tray_size[2])
    preplaced = sorted(
        (cube for cube in scenario.blue_cubes if cube.preplaced),
        key=lambda cube: int(cube.initial_placement_slot or 0),
    )
    preplaced_names = [cube.name for cube in preplaced]
    expected_names = list(scenario.preplaced_blue_cube_names or [])
    remaining_names = [cube.name for cube in scenario.blue_cubes if not cube.preplaced]
    expected_remaining = list(scenario.remaining_blue_cube_names or [])
    placement_slots = [cube.initial_placement_slot for cube in preplaced]
    if (
        scenario.progress_stage != len(preplaced)
        or scenario.num_preplaced != len(preplaced)
        or scenario.num_blue_total != len(scenario.blue_cubes)
        or scenario.num_remaining != len(scenario.blue_cubes) - len(preplaced)
        or preplaced_names != expected_names
        or remaining_names != expected_remaining
        or placement_slots != list(range(len(preplaced)))
        or scenario.num_preplaced >= len(scenario.blue_cubes)
    ):
        raise RuntimeError(
            "invalid partial-progress metadata: "
            f"episode={scenario.episode_index + 1} stage={scenario.progress_stage} "
            f"preplaced={preplaced_names} expected={expected_names}"
        )
    for cube in [*scenario.blue_cubes, *scenario.red_cubes]:
        # A cuboid may have arbitrary yaw. Its half diagonal is a conservative
        # axis-aligned footprint radius for every orientation.
        cube_radius = 0.5 * math.hypot(float(cube.size[0]), float(cube.size[1]))
        dx = abs(float(cube.pos[0]) - float(scenario.tray_pos[0]))
        dy = abs(float(cube.pos[1]) - float(scenario.tray_pos[1]))
        if cube.preplaced:
            if cube.role != "target_blue_cube" or cube.initial_placement_slot is None:
                raise RuntimeError(
                    f"invalid preplaced object metadata: {cube.name}"
                )
            wall_clearance = 0.003
            minimum_z = tray_top + 0.5 * float(cube.size[2])
            if (
                dx + cube_radius + wall_clearance > tray_half_x
                or dy + cube_radius + wall_clearance > tray_half_y
                or float(cube.pos[2]) + 1.0e-6 < minimum_z
            ):
                raise RuntimeError(
                    "failed to validate preplaced cube: "
                    f"episode={scenario.episode_index + 1} object={cube.name} "
                    f"dx={dx:.4f} dy={dy:.4f} radius={cube_radius:.4f} "
                    f"z={cube.pos[2]:.4f} tray={scenario.tray_pos}"
                )
            continue
        if cube.initial_placement_slot is not None:
            raise RuntimeError(f"non-preplaced cube has a slot: {cube.name}")
        if (
            dx < tray_half_x + cube_radius + clearance
            and dy < tray_half_y + cube_radius + clearance
        ):
            raise RuntimeError(
                "failed to sample tray-clear layout: "
                f"episode={scenario.episode_index + 1} object={cube.name} "
                f"dx={dx:.4f} dy={dy:.4f} radius={cube_radius:.4f} "
                f"tray={scenario.tray_pos}"
            )


def sample_non_overlapping_xy(
    rng: np.random.Generator,
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    half_extents: tuple[float, float],
    occupied: list[tuple[float, float, float, float, str]],
    margin: float,
    label: str,
    max_tries: int = 500,
) -> tuple[float, float]:
    for _ in range(max_tries):
        xy = sample_xy_in_bounds(rng, x_bounds=x_bounds, y_bounds=y_bounds, half_extents=half_extents)
        if not within_workspace_reach(xy):
            continue
        if not overlaps_2d(xy, half_extents, occupied, margin):
            occupied.append((xy[0], xy[1], half_extents[0], half_extents[1], label))
            return xy
    raise RuntimeError(f"failed to sample non-overlapping placement for {label}; reduce object count or enlarge workspace")


def sample_stratified_target_xy(
    rng: np.random.Generator,
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    half_extents: tuple[float, float],
    occupied: list[tuple[float, float, float, float, str]],
    margin: float,
    label: str,
    cell_order: list[int],
    samples_per_cell: int = 64,
) -> tuple[tuple[float, float], int]:
    """Sample a tray-clear blue target from a deterministic workspace-cell order."""
    x_bins, y_bins = (int(value) for value in args_cli.target_workspace_bins)
    usable_x_min = x_bounds[0] + half_extents[0]
    usable_x_max = x_bounds[1] - half_extents[0]
    usable_y_min = y_bounds[0] + half_extents[1]
    usable_y_max = y_bounds[1] - half_extents[1]
    if usable_x_max <= usable_x_min or usable_y_max <= usable_y_min:
        raise RuntimeError(f"workspace is too small for {label}")

    for cell_id in cell_order:
        x_index, y_index = divmod(int(cell_id), y_bins)
        cell_x_min = usable_x_min + (usable_x_max - usable_x_min) * x_index / x_bins
        cell_x_max = usable_x_min + (usable_x_max - usable_x_min) * (x_index + 1) / x_bins
        cell_y_min = usable_y_min + (usable_y_max - usable_y_min) * y_index / y_bins
        cell_y_max = usable_y_min + (usable_y_max - usable_y_min) * (y_index + 1) / y_bins
        for _ in range(samples_per_cell):
            xy = (
                float(rng.uniform(cell_x_min, cell_x_max)),
                float(rng.uniform(cell_y_min, cell_y_max)),
            )
            if not within_workspace_reach(xy):
                continue
            # The tray is already in occupied, so a target can never spawn in
            # or touching it; the configured spawn margin is enforced as well.
            if overlaps_2d(xy, half_extents, occupied, margin):
                continue
            occupied.append((xy[0], xy[1], half_extents[0], half_extents[1], label))
            return xy, int(cell_id)

    raise RuntimeError(
        f"failed to sample stratified tray-clear placement for {label}; "
        "reduce object count or enlarge workspace"
    )


def target_workspace_cell_order(
    blue_cube_count: int,
    scenario_ordinal: int,
    target_index: int,
) -> list[int]:
    """Return a Y-balanced, seed-stable order of preferred workspace cells."""
    x_bins, y_bins = (int(value) for value in args_cli.target_workspace_bins)
    permutation_rng = np.random.default_rng(
        int(args_cli.seed) * 1_000_003 + int(blue_cube_count) * 97_409 + x_bins * y_bins
    )
    x_permutation = [int(value) for value in permutation_rng.permutation(x_bins)]
    y_permutation = [int(value) for value in permutation_rng.permutation(y_bins)]

    # Cycle every scenario of the same cube count through the Y bands. Search
    # all X cells in that band before changing Y, so tray/reach rejection can
    # never create a systematic hole in one lateral band.
    y_stride = max(1, y_bins // max(1, blue_cube_count))
    x_stride = max(1, x_bins // max(1, blue_cube_count))
    logical_y_start = (
        int(scenario_ordinal) + int(target_index) * y_stride
    ) % y_bins
    logical_x_start = (
        int(scenario_ordinal) // y_bins + int(target_index) * x_stride
    ) % x_bins
    cell_order: list[int] = []
    for y_offset in range(y_bins):
        y_index = y_permutation[(logical_y_start + y_offset) % y_bins]
        for x_offset in range(x_bins):
            x_index = x_permutation[(logical_x_start + x_offset) % x_bins]
            cell_order.append(x_index * y_bins + y_index)
    return cell_order


_BLUE_SCENARIO_ORDINALS: dict[
    tuple[int, int, int], list[tuple[int, int]]
] = {}
_BLUE_SCENARIO_TOTALS: dict[tuple[int, int, int], dict[int, int]] = {}


def blue_scenario_count_and_ordinal(episode_index: int) -> tuple[int, int]:
    """Return blue count and its zero-based ordinal across global episode indices."""
    key = (
        int(args_cli.seed),
        int(args_cli.min_blue_cubes),
        int(args_cli.max_blue_cubes),
    )
    records = _BLUE_SCENARIO_ORDINALS.setdefault(key, [])
    totals = _BLUE_SCENARIO_TOTALS.setdefault(key, {})
    while len(records) <= int(episode_index):
        index = len(records)
        count_rng = np.random.default_rng(int(args_cli.seed) + index)
        count = int(
            count_rng.integers(args_cli.min_blue_cubes, args_cli.max_blue_cubes + 1)
        )
        ordinal = totals.get(count, 0)
        records.append((count, ordinal))
        totals[count] = ordinal + 1
    return records[int(episode_index)]


def sample_spread_non_overlapping_xy(
    rng: np.random.Generator,
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    half_extents: tuple[float, float],
    occupied: list[tuple[float, float, float, float, str]],
    spread_refs: list[tuple[float, float]],
    margin: float,
    label: str,
    candidate_count: int = 512,
) -> tuple[float, float]:
    """Sample a valid robot-front tabletop point while spreading loose cubes."""
    candidates: list[tuple[float, float, float]] = []
    for _ in range(candidate_count):
        xy = sample_xy_in_bounds(rng, x_bounds=x_bounds, y_bounds=y_bounds, half_extents=half_extents)
        if not within_workspace_reach(xy):
            continue
        if overlaps_2d(xy, half_extents, occupied, margin):
            continue
        if spread_refs:
            spread_score = min(math.hypot(xy[0] - px, xy[1] - py) for px, py in spread_refs)
        else:
            spread_score = min(math.hypot(xy[0] - ox, xy[1] - oy) for ox, oy, *_ in occupied)
        candidates.append((spread_score, xy[0], xy[1]))

    if not candidates:
        raise RuntimeError(f"failed to sample spread placement for {label}; enlarge the robot-front workspace")

    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    # Keep objects well spread without always selecting only the farthest edge
    # candidates, which made the observed cube distribution much narrower than
    # the configured workspace.
    top_count = max(1, min(len(candidates), max(16, math.ceil(len(candidates) * 0.35))))
    _, x, y = candidates[int(rng.integers(0, top_count))]

    occupied.append((x, y, half_extents[0], half_extents[1], label))
    spread_refs.append((x, y))
    return x, y


def split_workspace_for_tray_and_cubes(
    rng: np.random.Generator,
    *,
    y_bounds: tuple[float, float],
    tray_half_y: float,
    cube_half_y: float,
    margin: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Put the tray and loose cubes on opposite Y-side regions.

    This is more stable than sampling everything from one shared area because the
    tray is much larger than a cube. The returned bounds are still fed into
    sample_xy_in_bounds(), which applies the object half extents again.
    """
    y_min, y_max = y_bounds
    y_mid = 0.5 * (y_min + y_max)
    # Alternate the tray side deterministically per episode so loose cubes and
    # pick/place motions cover both lateral halves of the table.
    tray_positive_side = bool(rng.integers(0, 2))
    if tray_positive_side:
        tray_y_bounds = (y_mid + margin, y_max)
        cube_y_bounds = (y_min, y_mid - margin)
    else:
        tray_y_bounds = (y_min, y_mid - margin)
        cube_y_bounds = (y_mid + margin, y_max)

    tray_fits = tray_y_bounds[1] - tray_y_bounds[0] > 2.0 * tray_half_y
    cube_fits = cube_y_bounds[1] - cube_y_bounds[0] > 2.0 * cube_half_y
    if tray_fits and cube_fits:
        return tray_y_bounds, cube_y_bounds

    print("[WARN] workspace too small for split tray/cube regions; falling back to shared Y bounds")
    return y_bounds, y_bounds


def _generate_blue_tray_scenario_once(
    episode_index: int,
    layout_attempt: int = 0,
    fixed_tray_pos: Vec3 | None = None,
) -> ScenarioSpec:
    """Generate one deterministic scenario candidate."""
    base_seed = int(args_cli.seed + episode_index)
    seed = int(base_seed + layout_attempt * 1_000_003)
    rng = np.random.default_rng(seed)

    count_rng = np.random.default_rng(base_seed)
    sampled_blue_count = int(
        count_rng.integers(args_cli.min_blue_cubes, args_cli.max_blue_cubes + 1)
    )
    blue_count, blue_scenario_ordinal = blue_scenario_count_and_ordinal(episode_index)
    if blue_count != sampled_blue_count:
        raise RuntimeError(
            f"blue scenario ordinal mismatch: sampled={sampled_blue_count} cached={blue_count}"
        )
    red_count = int(
        count_rng.integers(args_cli.min_red_cubes, args_cli.max_red_cubes + 1)
    )
    cube_size = float(args_cli.cube_size)
    max_cube_size = (
        float(args_cli.cube_size_range[1]) if args_cli.domain_randomization else cube_size
    )
    cube_half_radius = 0.5 * math.sqrt(2.0) * max_cube_size
    cube_half = (cube_half_radius, cube_half_radius)
    tray_x, tray_y = map(float, args_cli.tray_size)
    tray_half = (tray_x * 0.5, tray_y * 0.5)
    x_bounds = (float(args_cli.workspace_x_min), float(args_cli.workspace_x_max))
    y_bounds = (float(args_cli.workspace_y_min), float(args_cli.workspace_y_max))
    # Keep placement slots away from the reach boundary; cubes still use x_bounds.
    tray_x_bounds = (max(x_bounds[0], 0.40), min(x_bounds[1], 0.62))
    tray_y_bounds, cube_y_bounds = split_workspace_for_tray_and_cubes(
        rng,
        y_bounds=y_bounds,
        tray_half_y=tray_half[1],
        cube_half_y=cube_half[1],
        margin=float(args_cli.min_spawn_spacing),
    )
    # Cubes use the entire safe robot-front tabletop. Occupancy checks keep
    # them clear of the tray, and spread sampling prevents tight clusters.
    cube_y_bounds = y_bounds

    occupied: list[tuple[float, float, float, float, str]] = []
    cube_spread_refs: list[tuple[float, float]] = []
    if fixed_tray_pos is None:
        tray_xy = sample_non_overlapping_xy(
            rng,
            x_bounds=tray_x_bounds,
            y_bounds=tray_y_bounds,
            half_extents=tray_half,
            occupied=occupied,
            margin=float(args_cli.min_spawn_spacing),
            label="tray",
        )
        tray_z = float(args_cli.tray_z)
    else:
        tray_xy = (float(fixed_tray_pos[0]), float(fixed_tray_pos[1]))
        tray_z = float(fixed_tray_pos[2])
        if not (
            x_bounds[0] + tray_half[0] <= tray_xy[0] <= x_bounds[1] - tray_half[0]
            and y_bounds[0] + tray_half[1] <= tray_xy[1] <= y_bounds[1] - tray_half[1]
            and within_workspace_reach(tray_xy)
        ):
            raise RuntimeError(
                f"failed to sample: fixed tray pose {fixed_tray_pos} is outside the valid workspace"
            )
        occupied.append(
            (tray_xy[0], tray_xy[1], tray_half[0], tray_half[1], "tray")
        )

    tray_palette: list[Color] = [(0.1, 0.55, 0.22), (0.95, 0.78, 0.18), (0.28, 0.28, 0.3)]
    table_palette: list[Color] = [(0.55, 0.47, 0.37), (0.72, 0.72, 0.68), (0.38, 0.42, 0.44)]
    if args_cli.domain_randomization:
        blue_color = jitter_color(rng, (0.03, 0.16, 0.95), 0.04)
        red_color = jitter_color(rng, (0.95, 0.04, 0.03), 0.04)
        tray_color = jitter_color(rng, tray_palette[int(rng.integers(0, len(tray_palette)))], 0.05)
        table_color = jitter_color(rng, table_palette[int(rng.integers(0, len(table_palette)))], 0.05)
        light_color = jitter_color(rng, (1.0, 0.96, 0.9), 0.05)
        light_intensity = sample_log_uniform(rng, args_cli.dome_light_intensity_range)
        # Exposure is an episode-level camera property. Sampling it per frame
        # produces artificial brightness flicker that looks like unstable lighting.
        rgb_brightness_gain = float(rng.uniform(*args_cli.rgb_brightness_range))
    else:
        blue_color = (0.03, 0.16, 0.95)
        red_color = (0.95, 0.04, 0.03)
        tray_color = tray_palette[0]
        table_color = table_palette[0]
        light_color = (1.0, 0.96, 0.9)
        light_intensity = 700.0
        rgb_brightness_gain = 1.0

    blue_cubes: list[ScenarioObjectSpec] = []
    red_cubes: list[ScenarioObjectSpec] = []
    for idx in range(blue_count):
        size = sample_cuboid_size(rng)
        rotated_half_extent = 0.5 * math.hypot(size[0], size[1])
        mass, static_friction, dynamic_friction, restitution = sample_cube_physics(rng)
        position_stratum: int | None = None
        if args_cli.stratified_target_positions:
            xy, position_stratum = sample_stratified_target_xy(
                rng,
                x_bounds=x_bounds,
                y_bounds=cube_y_bounds,
                half_extents=(rotated_half_extent, rotated_half_extent),
                occupied=occupied,
                margin=float(args_cli.min_spawn_spacing),
                label=f"blue_cube_{idx}",
                cell_order=target_workspace_cell_order(
                    blue_count, blue_scenario_ordinal, idx
                ),
            )
            cube_spread_refs.append(xy)
        else:
            xy = sample_spread_non_overlapping_xy(
                rng,
                x_bounds=x_bounds,
                y_bounds=cube_y_bounds,
                half_extents=(rotated_half_extent, rotated_half_extent),
                occupied=occupied,
                spread_refs=cube_spread_refs,
                margin=float(args_cli.min_spawn_spacing),
                label=f"blue_cube_{idx}",
            )
        builtin = idx == 0
        blue_cubes.append(
            ScenarioObjectSpec(
                name=f"blue_cube_{idx}",
                role="target_blue_cube",
                asset_name="object" if builtin else f"blue_cube_{idx}",
                prim_path="/World/envs/env_0/Object" if builtin else f"/World/envs/env_0/BlueCube_{idx}",
                pos=(xy[0], xy[1], cuboid_center_z(size[2])),
                size=size,
                color=blue_color,
                builtin=builtin,
                yaw=float(rng.uniform(-math.pi, math.pi)) if args_cli.domain_randomization else 0.0,
                mass=mass,
                static_friction=static_friction,
                dynamic_friction=dynamic_friction,
                restitution=restitution,
                position_stratum=position_stratum,
            )
        )

    for idx in range(red_count):
        size = sample_cuboid_size(rng)
        rotated_half_extent = 0.5 * math.hypot(size[0], size[1])
        mass, static_friction, dynamic_friction, restitution = sample_cube_physics(rng)
        xy = sample_spread_non_overlapping_xy(
            rng,
            x_bounds=x_bounds,
            y_bounds=cube_y_bounds,
            half_extents=(rotated_half_extent, rotated_half_extent),
            occupied=occupied,
            spread_refs=cube_spread_refs,
            margin=float(args_cli.min_spawn_spacing),
            label=f"red_cube_{idx}",
        )
        red_cubes.append(
            ScenarioObjectSpec(
                name=f"red_cube_{idx}",
                role="distractor_red_cube",
                asset_name=f"red_cube_{idx}",
                prim_path=f"/World/envs/env_0/RedCube_{idx}",
                pos=(xy[0], xy[1], cuboid_center_z(size[2])),
                size=size,
                color=red_color,
                yaw=float(rng.uniform(-math.pi, math.pi)) if args_cli.domain_randomization else 0.0,
                mass=mass,
                static_friction=static_friction,
                dynamic_friction=dynamic_friction,
                restitution=restitution,
            )
        )

    # Row-major tray slots: +Y is top and -X is left. Up to three targets use
    # upper-left, upper-right, then lower-left.
    placement_positions: list[Vec3] = []
    cols = min(2, blue_count)
    rows = max(1, int(math.ceil(blue_count / cols)))
    # Keep cube centers away from the tray walls. The former 1 cm wall
    # clearance was enough for a cube, but not for Franka's fingers; opening
    # the gripper could pinch/eject the right-hand cube over the rim.
    finger_clearance = 0.05
    max_blue_footprint = max(max(cube.size[0], cube.size[1]) for cube in blue_cubes)
    max_blue_height = max(cube.size[2] for cube in blue_cubes)
    usable_x = max(0.0, tray_x - max_blue_footprint - finger_clearance)
    usable_y = max(0.0, tray_y - max_blue_footprint - finger_clearance)
    # The controller substitutes the selected cuboid exact half-height.
    slot_z = float(args_cli.tray_z) + 0.025 * 0.5 + max_blue_height * 0.5 + 0.003
    for idx in range(blue_count):
        col = idx % cols
        row = idx // cols
        ox = 0.0 if cols == 1 else -usable_x * 0.5 + usable_x * (col / (cols - 1))
        oy = usable_y * 0.5 if rows == 1 else usable_y * 0.5 - usable_y * (row / (rows - 1))
        placement_positions.append((tray_xy[0] + ox, tray_xy[1] + oy, slot_z))

    camera_eye = FIXED_CAMERA_EYE
    camera_target = FIXED_CAMERA_TARGET
    if args_cli.domain_randomization and args_cli.randomize_camera:
        eye_jitter = rng.uniform(-np.asarray(args_cli.camera_position_jitter), args_cli.camera_position_jitter)
        target_jitter = rng.uniform(-np.asarray(args_cli.camera_target_jitter), args_cli.camera_target_jitter)
        camera_eye = tuple((np.asarray(FIXED_CAMERA_EYE) + eye_jitter).tolist())
        camera_target = tuple((np.asarray(FIXED_CAMERA_TARGET) + target_jitter).tolist())
    if args_cli.domain_randomization:
        # Use the complete RGB cube instead of a small muted palette. The dome
        # remains near-neutral, so backdrop diversity does not darken the task.
        background_color = sample_uniform_rgb(rng)
    else:
        background_color = (0.10, 0.12, 0.15)
    lights = generate_episode_lights(rng, light_intensity, light_color, background_color)
    # Progress-stage sampling depends only on the public episode seed, not on a
    # rare layout retry, so the standard/partial mix remains reproducible.
    progress_rng = np.random.default_rng(base_seed ^ 0x3C6EF372)
    num_preplaced = sample_partial_progress_count(progress_rng, blue_count)
    tray_pos = (tray_xy[0], tray_xy[1], tray_z)
    tray_size = (tray_x, tray_y, 0.025)
    preplaced_cubes = preplace_blue_cubes(
        progress_rng,
        blue_cubes,
        placement_positions,
        tray_pos,
        tray_size,
        num_preplaced,
    )
    preplaced_names = [cube.name for cube in preplaced_cubes]
    remaining_names = [cube.name for cube in blue_cubes if not cube.preplaced]

    # Keep motion augmentation independent from layout/appearance RNG so toggling
    # it does not silently change the sampled scene for the same episode seed.
    augmentation_rng = np.random.default_rng(seed ^ 0x5A17A11)
    if preplaced_cubes:
        start_ee_target = sample_partial_progress_start_ee_target(
            augmentation_rng, preplaced_cubes[-1]
        )
        start_pose_mode = (
            "post_placement_retreat" if start_ee_target is not None else "default"
        )
    else:
        start_ee_target = sample_start_ee_target(augmentation_rng)
        start_pose_mode = "random_workspace" if start_ee_target is not None else "default"
    for cube in blue_cubes:
        if not cube.preplaced:
            cube.recovery_waypoint = sample_recovery_waypoint(augmentation_rng, cube)

    return ScenarioSpec(
        episode_index=episode_index,
        seed=seed,
        mode="blue_tray",
        blue_cubes=blue_cubes,
        red_cubes=red_cubes,
        tray_pos=tray_pos,
        tray_size=tray_size,
        tray_color=tray_color,
        table_color=table_color,
        light_intensity=light_intensity,
        light_color=light_color,
        lights=lights,
        rgb_brightness_gain=rgb_brightness_gain,
        placement_positions=placement_positions,
        camera_eye=camera_eye,
        camera_target=camera_target,
        background_color=background_color,
        progress_stage=num_preplaced,
        num_blue_total=blue_count,
        num_preplaced=num_preplaced,
        num_remaining=blue_count - num_preplaced,
        preplaced_blue_cube_names=preplaced_names,
        remaining_blue_cube_names=remaining_names,
        start_pose_mode=start_pose_mode,
        start_ee_target=start_ee_target,
    )


def generate_blue_tray_scenario(
    episode_index: int, fixed_tray_pos: Vec3 | None = None
) -> ScenarioSpec:
    """Generate a scenario, retrying rare greedy placement dead ends."""
    max_layout_attempts = 32
    last_error: RuntimeError | None = None
    for layout_attempt in range(max_layout_attempts):
        try:
            scenario = _generate_blue_tray_scenario_once(
                episode_index, layout_attempt, fixed_tray_pos,
            )
            validate_scenario_spawn_clearance(scenario)
            if layout_attempt > 0:
                print(
                    f"[SCENARIO] episode={episode_index + 1} placement recovered "
                    f"on attempt {layout_attempt + 1}; seed={scenario.seed}"
                )
            return scenario
        except RuntimeError as exc:
            if not str(exc).startswith("failed to sample"):
                raise
            last_error = exc

    raise RuntimeError(
        f"failed to generate episode {episode_index + 1} after "
        f"{max_layout_attempts} placement attempts; last error: {last_error}"
    ) from last_error


def make_scenario() -> ScenarioSpec | None:
    if args_cli.scenario != "blue_tray":
        return None
    return generate_blue_tray_scenario(args_cli.episode_index)


def maybe_disable_resets(env_cfg) -> None:
    """Avoid automatic task reset during one autonomous rollout."""
    terminations = getattr(env_cfg, "terminations", None)
    if terminations is None:
        return
    for name, value in list(vars(terminations).items()):
        if name.startswith("_") or value is None:
            continue
        try:
            setattr(terminations, name, None)
        except Exception:
            pass


def disable_debug_visual_cfg(env_cfg) -> None:
    """Disable task/helper visualizers before they create rendered marker prims."""
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "object_pose"):
        env_cfg.commands.object_pose.debug_vis = False
    if hasattr(env_cfg.scene, "ee_frame"):
        env_cfg.scene.ee_frame.debug_vis = False


def hide_debug_visuals() -> None:
    """Hide already-created Isaac Lab helper visuals so they do not appear in RGB captures."""
    stage = omni.usd.get_context().get_stage()
    for prim_path in ("/Visuals", "/World/DebugPolicy"):
        prim = stage.GetPrimAtPath(prim_path)
        if prim and prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
            print(f"[INFO] hidden debug visual prim: {prim_path}")


def needs_external_camera() -> bool:
    return args_cli.record_sensors or args_cli.preview_view == "record_camera"


def needs_wrist_camera() -> bool:
    return args_cli.enable_wrist_camera or args_cli.preview_view == "wrist_camera"


def attach_record_cameras(env_cfg) -> None:
    """Attach optional external and wrist cameras before gym.make()."""
    if needs_external_camera():
        env_cfg.scene.record_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/RecordCamera",
            update_period=1.0 / output_sensor_fps(),
            width=args_cli.width,
            height=args_cli.height,
            data_types=camera_data_types(),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=EXTERNAL_CAMERA_FOCAL_LENGTH,
                focus_distance=400.0,
                horizontal_aperture=CAMERA_HORIZONTAL_APERTURE,
                clipping_range=(0.05, 20.0),
            ),
            colorize_semantic_segmentation=True,
            colorize_instance_segmentation=True,
        )

    if needs_wrist_camera():
        env_cfg.scene.wrist_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/WristCamera",
            update_period=1.0 / output_sensor_fps(),
            width=args_cli.width,
            height=args_cli.height,
            data_types=camera_data_types(),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=float(args_cli.wrist_focal_length),
                focus_distance=0.30,
                horizontal_aperture=CAMERA_HORIZONTAL_APERTURE,
                clipping_range=(0.02, 5.0),
            ),
            colorize_semantic_segmentation=True,
            colorize_instance_segmentation=True,
        )


def iter_prim_tree(prim):
    """Yield a USD prim and all descendants."""
    yield prim
    for child in prim.GetChildren():
        yield from iter_prim_tree(child)


def find_visual_subtrees(root_path: str) -> list[str]:
    """
    Return visual-only subtree roots under a USD prim.

    Semantic labels should be attached to rendered visual meshes. Do not label
    collision or articulation subtrees because that can invalidate PhysX tensor
    views in Isaac Lab.
    """
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return []

    visual_paths: list[str] = []
    for prim in iter_prim_tree(root):
        name = prim.GetName().lower()
        path = str(prim.GetPath())
        if name in ("visuals", "visual"):
            visual_paths.append(path)
    return visual_paths


def apply_semantic_label_to_subtree(prim_path: str, label: str) -> int:
    """Attach one semantic class label to a visual subtree."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return 0

    count = 0
    for target_prim in iter_prim_tree(prim):
        sem = Semantics.SemanticsAPI.Apply(target_prim, "Semantics")
        sem.CreateSemanticTypeAttr().Set("class")
        sem.CreateSemanticDataAttr().Set(label)
        count += 1
    return count


def apply_display_color_to_subtree(root_path: str, color: Color) -> int:
    """Apply USD displayColor to simple visual geometry under a prim."""
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return 0

    count = 0
    display_color = [Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))]
    for prim in iter_prim_tree(root):
        if prim.IsA(UsdGeom.Gprim):
            UsdGeom.Gprim(prim).CreateDisplayColorAttr().Set(display_color)
            count += 1
    return count


def add_smoke_test_semantics(env) -> dict[str, Any]:
    """Label the built-in Franka lift scene for semantic segmentation."""
    _ = env
    semantic_targets = {
        "/World/envs/env_0/Robot": "robot",
        "/World/envs/env_0/Object": "cube",
        "/World/envs/env_0/Table": "table",
    }

    labels: dict[str, Any] = {}
    missing: dict[str, str] = {}
    for prim_path, label in semantic_targets.items():
        visual_paths = find_visual_subtrees(prim_path)
        if not visual_paths and label != "robot":
            visual_paths = [prim_path]

        count = 0
        applied_paths: list[str] = []
        for visual_path in visual_paths:
            applied_count = apply_semantic_label_to_subtree(visual_path, label)
            if applied_count:
                applied_paths.append(visual_path)
                count += applied_count

        if count:
            labels[label] = {"root_path": prim_path, "visual_paths": applied_paths, "prim_count": count}
        else:
            missing[label] = prim_path

    print(f"[INFO] semantic labels applied: {labels}")
    if missing:
        print(f"[WARN] semantic target prims not found: {missing}")
    return labels


def add_blue_tray_semantics(env, scenario: ScenarioSpec) -> dict[str, Any]:
    """Apply semantic labels for randomized blue-tray scenes."""
    _ = env
    semantic_targets: dict[str, str] = {
        "/World/envs/env_0/Robot": "robot",
        "/World/envs/env_0/Table": "table",
        "/World/envs/env_0/Tray": "tray",
    }
    for cube in scenario.blue_cubes:
        semantic_targets[cube.prim_path] = "blue_cube"
    for cube in scenario.red_cubes:
        semantic_targets[cube.prim_path] = "red_cube"

    labels: dict[str, Any] = {}
    missing: dict[str, str] = {}
    for prim_path, label in semantic_targets.items():
        visual_paths = find_visual_subtrees(prim_path)
        if not visual_paths and label != "robot":
            visual_paths = [prim_path]

        count = 0
        applied_paths: list[str] = []
        for visual_path in visual_paths:
            applied_count = apply_semantic_label_to_subtree(visual_path, label)
            if applied_count:
                applied_paths.append(visual_path)
                count += applied_count

        if count:
            labels.setdefault(label, []).append(
                {"root_path": prim_path, "visual_paths": applied_paths, "prim_count": count}
            )
        else:
            missing[label] = prim_path

    print(f"[INFO] scenario semantic labels applied: {labels}")
    if missing:
        print(f"[WARN] scenario semantic target prims not found: {missing}")
    return labels


def env_local_pose_from_world(
    env, pos_w: torch.Tensor, quat_w: torch.Tensor, env_id: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert world pose to env-local pose for consistent state logging."""
    return pos_w - env.scene.env_origins[env_id], quat_w


def read_object_pose_env(env, env_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    obj = env.scene["object"]
    return env_local_pose_from_world(env, obj.data.root_pos_w[env_id], obj.data.root_quat_w[env_id], env_id)


def read_ee_pose_env(env, env_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    ee_frame = env.scene["ee_frame"]
    return env_local_pose_from_world(
        env, ee_frame.data.target_pos_w[env_id, 0], ee_frame.data.target_quat_w[env_id, 0], env_id
    )


def read_rigid_object_pose_env(env, asset_name: str, env_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    obj = env.scene[asset_name]
    return env_local_pose_from_world(
        env, obj.data.root_pos_w[env_id], obj.data.root_quat_w[env_id], env_id
    )


def read_scenario_objects_for_log(env, scenario: ScenarioSpec | None, env_id: int = 0) -> dict[str, Any]:
    """Return current randomized object poses for action/state logs."""
    if scenario is None:
        return {}

    objects: dict[str, Any] = {}
    for cube in [*scenario.blue_cubes, *scenario.red_cubes]:
        try:
            pos, quat = read_rigid_object_pose_env(env, cube.asset_name, env_id)
            objects[cube.name] = {
                "role": cube.role,
                "asset_name": cube.asset_name,
                "prim_path": cube.prim_path,
                "pos_env": tensor_to_numpy(pos).tolist(),
                "quat_xyzw": tensor_to_numpy(quat).tolist(),
                "color": list(cube.color),
                "size": list(cube.size),
            }
        except Exception as exc:
            objects[cube.name] = {
                "role": cube.role,
                "asset_name": cube.asset_name,
                "prim_path": cube.prim_path,
                "error": str(exc),
            }

    return {
        "scenario_objects": objects,
        "tray": {
            "pos_env": list(scenario.tray_pos),
            "size": list(scenario.tray_size),
            "color": list(scenario.tray_color),
        },
    }


def cuboid_rigid_object_cfg(spec: ScenarioObjectSpec) -> RigidObjectCfg:
    """Create one Isaac Lab rigid cuboid config for an extra randomized cube."""
    semantic_class = "blue_cube" if spec.role == "target_blue_cube" else "red_cube"
    return RigidObjectCfg(
        prim_path=spec.prim_path.replace("/World/envs/env_0", "{ENV_REGEX_NS}"),
        spawn=sim_utils.CuboidCfg(
            size=spec.size,
            semantic_tags=[("class", semantic_class)],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=spec.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=spec.static_friction,
                dynamic_friction=spec.dynamic_friction,
                restitution=spec.restitution,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=spec.color, roughness=0.55),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=spec.pos, rot=yaw_quat_xyzw(spec.yaw)),
    )



def apply_blue_tray_env_cfg(env_cfg, scenario: ScenarioSpec) -> None:
    """Apply blue-tray object layout to the Isaac Lab env config before gym.make()."""
    # Use Isaac Labs official Franka tabletop working pose instead of a straight arm.
    env_cfg.scene.robot.init_state.joint_pos = TABLETOP_FRANKA_JOINT_POS.copy()
    # Disable the task default 3000-intensity dome so scenario lights fully control exposure.
    default_light = getattr(env_cfg.scene, "light", None)
    default_light_spawn = getattr(default_light, "spawn", None)
    if default_light_spawn is not None and hasattr(default_light_spawn, "intensity"):
        default_light_spawn.intensity = 0.0
    built_in_blue = scenario.blue_cubes[0]
    # Replicator annotators discover semantic tags reliably when the tags are
    # present on spawn configs, before gym.make() creates the camera graph.
    for asset_name, semantic_class in (("robot", "robot"), ("table", "table")):
        asset_cfg = getattr(env_cfg.scene, asset_name, None)
        if asset_cfg is not None and getattr(asset_cfg, "spawn", None) is not None:
            asset_cfg.spawn.semantic_tags = [("class", semantic_class)]
    # Replace the task built-in instanced DexCube with the same procedural cuboid
    # used by all other objects. This keeps geometry and physics equal to metadata.
    env_cfg.scene.object = cuboid_rigid_object_cfg(built_in_blue)

    for cube in scenario.blue_cubes[1:]:
        setattr(env_cfg.scene, cube.asset_name, cuboid_rigid_object_cfg(cube))
    for cube in scenario.red_cubes:
        setattr(env_cfg.scene, cube.asset_name, cuboid_rigid_object_cfg(cube))


def spawn_tray_and_apply_domain_randomization(scenario: ScenarioSpec) -> None:
    """Spawn static tray and apply simple table/light/color randomization after env.reset()."""
    stage = omni.usd.get_context().get_stage()

    tray_cfg = sim_utils.CuboidCfg(
        size=scenario.tray_size,
        semantic_tags=[("class", "tray")],
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=scenario.tray_color, roughness=0.65
        ),
    )
    tray_cfg.func("/World/envs/env_0/Tray", tray_cfg, translation=scenario.tray_pos)
    # A visual-only backdrop makes background appearance independent from
    # illumination. It sits behind the complete workspace and all local lights.
    backdrop_cfg = sim_utils.CuboidCfg(
        size=BACKDROP_WALL_SIZE,
        semantic_tags=[("class", "background")],
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=scenario.background_color, roughness=0.9
        ),
    )
    backdrop_cfg.func("/World/envs/env_0/Background", backdrop_cfg, translation=(0.35, 1.15, 0.8))

    try:
        rep.set_global_seed(scenario.seed)
        for index, light_spec in enumerate(scenario.lights):
            kwargs: dict[str, Any] = {
                "light_type": light_spec.light_type,
                "color": light_spec.color,
                "intensity": light_spec.intensity,
                "name": f"ScenarioLight_{index}_{light_spec.light_type}",
            }
            if light_spec.position is not None:
                kwargs["position"] = light_spec.position
            if light_spec.rotation is not None:
                kwargs["rotation"] = light_spec.rotation
            if light_spec.scale is not None:
                kwargs["scale"] = light_spec.scale
            rep.create.light(**kwargs)
        light_types = [light.light_type for light in scenario.lights]
        print(f"[INFO] fixed episode light rig applied: count={len(light_types)}, types={light_types}")
    except Exception as exc:
        light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/ScenarioDomeLight"))
        light.CreateIntensityAttr().Set(float(scenario.light_intensity))
        light.CreateColorAttr().Set(Gf.Vec3f(*[float(v) for v in scenario.light_color]))
        print(f"[WARN] Replicator randomization unavailable; USD fallback used: {exc}")

    table_colored = apply_display_color_to_subtree("/World/envs/env_0/Table/Visuals", scenario.table_color)
    builtin_colored = apply_display_color_to_subtree("/World/envs/env_0/Object", scenario.blue_cubes[0].color)
    print(
        "[INFO] blue_tray visual randomization applied: "
        f"table_colored_prims={table_colored}, builtin_object_colored_prims={builtin_colored}"
    )


def set_active_viewport_camera(camera_prim_path: str) -> bool:
    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath(camera_prim_path)
    if not camera_prim or not camera_prim.IsValid():
        print(f"[WARN] requested viewport camera prim does not exist: {camera_prim_path}")
        return False

    viewport = None
    for _ in range(60):
        viewport = viewport_utils.get_active_viewport()
        if viewport is not None:
            break
        simulation_app.update()
        time.sleep(1.0 / 60.0)
    if viewport is None:
        print("[WARN] no active viewport found; cannot set viewport camera")
        return False

    camera_path = Sdf.Path(camera_prim_path)
    try:
        viewport.camera_path = camera_path
    except Exception:
        viewport.set_active_camera(camera_path)
    print(f"[INFO] active viewport camera set to: {camera_path}")
    return True


def set_preview_camera(env, scenario: ScenarioSpec | None = None) -> None:
    """Use Isaac Lab.s SimulationContext viewport helper for WebRTC preview."""
    if args_cli.headless and int(getattr(args_cli, "livestream", 0)) == 0:
        return
    if args_cli.preview_view == "record_camera":
        set_active_viewport_camera("/World/envs/env_0/RecordCamera")
        return
    if args_cli.preview_view == "wrist_camera":
        set_active_viewport_camera("/World/envs/env_0/WristCamera")
        return
    try:
        eye = scenario.camera_eye if scenario is not None else args_cli.preview_eye
        target = scenario.camera_target if scenario is not None else args_cli.preview_target
        env.sim.set_camera_view(list(eye), list(target))
        print(f"[INFO] preview camera eye={args_cli.preview_eye}, target={args_cli.preview_target}")
    except Exception as exc:
        print(f"[WARN] failed to set preview camera: {exc}")


def update_wrist_follow_camera(env, *, force_recompute: bool = False) -> None:
    """Follow every EE with a physical local eye/target offset in one batch."""
    if not needs_wrist_camera():
        return
    ee_frame = env.scene["ee_frame"]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0]
    count = int(ee_pos_w.shape[0])
    eye_local = torch.tensor(
        args_cli.wrist_camera_pos, device=env.device, dtype=torch.float32
    ).unsqueeze(0).expand(count, -1)
    target_local = torch.tensor(
        args_cli.wrist_camera_target, device=env.device, dtype=torch.float32
    ).unsqueeze(0).expand(count, -1)
    eyes = ee_pos_w + quat_apply(ee_quat_w, eye_local)
    targets = ee_pos_w + quat_apply(ee_quat_w, target_local)
    camera = env.scene["wrist_camera"]
    camera.set_world_poses_from_view(eyes, targets)
    if force_recompute:
        camera.update(0.0, force_recompute=True)


def set_record_camera(env, scenario: ScenarioSpec | None = None) -> None:
    """Point the optional sensor camera at the same tabletop workspace."""
    if not needs_external_camera():
        return
    camera = env.scene["record_camera"]
    eye = scenario.camera_eye if scenario is not None else args_cli.preview_eye
    target = scenario.camera_target if scenario is not None else args_cli.preview_target
    camera.set_world_poses_from_view(
        torch.tensor([list(eye)], device=env.device),
        torch.tensor([list(target)], device=env.device),
    )




def configure_event_randomization(env_cfg, scenario: ScenarioSpec | None) -> None:
    """Use Isaac Lab EventManager for reproducible startup physics and reset state."""
    if scenario is None or not args_cli.domain_randomization or getattr(env_cfg, "events", None) is None:
        return
    events = env_cfg.events
    # The stock lift task resets the built-in cube independently. Replace that
    # term so every recorded pose remains tied to scenario.json.
    if hasattr(events, "reset_object_position"):
        events.reset_object_position = None

    for cube in [*scenario.blue_cubes, *scenario.red_cubes]:
        safe_name = cube.asset_name.replace("-", "_")
        asset_cfg = SceneEntityCfg(cube.asset_name)
        setattr(
            events,
            f"startup_mass_{safe_name}",
            EventTerm(
                func=event_mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": asset_cfg,
                    "mass_distribution_params": (cube.mass, cube.mass),
                    "operation": "abs",
                    "distribution": "uniform",
                    "recompute_inertia": True,
                },
            ),
        )
        setattr(
            events,
            f"startup_material_{safe_name}",
            EventTerm(
                func=event_mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": asset_cfg,
                    "static_friction_range": (cube.static_friction, cube.static_friction),
                    "dynamic_friction_range": (cube.dynamic_friction, cube.dynamic_friction),
                    "restitution_range": (cube.restitution, cube.restitution),
                    "num_buckets": 1,
                },
            ),
        )
        setattr(
            events,
            f"reset_pose_{safe_name}",
            EventTerm(
                func=event_mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "asset_cfg": asset_cfg,
                    "pose_range": {
                        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                        "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
                    },
                    "velocity_range": {
                        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                        "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
                    },
                },
            ),
        )
    print("[INFO] Isaac Lab EventManager randomization configured for cube mass/material/reset state")

def level_scenario_cubes(env, scenario: ScenarioSpec) -> None:
    """Place every cube flat on the table after the initial physics settling."""
    env_origin = env.scene.env_origins[0]
    cubes = [*scenario.blue_cubes, *scenario.red_cubes]
    for cube in cubes:
        obj = env.scene[cube.asset_name]
        root_pose = torch.zeros((1, 7), device=env.device, dtype=torch.float32)
        root_pose[0, :3] = obj.data.root_pos_w[0]
        root_pose[0, 2] = env_origin[2] + float(cube.pos[2])
        # Runtime rigid-object API uses quaternion xyzw. Preserve randomized yaw only.
        root_pose[0, 3] = 0.0
        root_pose[0, 4] = 0.0
        root_pose[0, 5] = math.sin(0.5 * float(cube.yaw))
        root_pose[0, 6] = math.cos(0.5 * float(cube.yaw))
        obj.write_root_pose_to_sim_index(root_pose=root_pose)
        obj.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((1, 6), device=env.device, dtype=torch.float32)
        )
    env.sim.forward()
    print(f"[INFO] leveled {len(cubes)} cubes on the table (roll=pitch=0)")


def apply_asset_version_override(env_cfg) -> None:
    """Optionally rewrite remote Isaac asset URLs without changing task configuration code."""
    version = args_cli.asset_version_override
    if not version:
        return
    for asset_name in ("robot", "table"):
        asset_cfg = getattr(env_cfg.scene, asset_name, None)
        spawn_cfg = getattr(asset_cfg, "spawn", None)
        usd_path = getattr(spawn_cfg, "usd_path", None)
        if not isinstance(usd_path, str):
            continue
        rewritten = re.sub(r"/Assets/Isaac/[^/]+/", f"/Assets/Isaac/{version}/", usd_path, count=1)
        if rewritten != usd_path:
            spawn_cfg.usd_path = rewritten
            print(f"[INFO] asset URL override ({asset_name}): {rewritten}")


def make_env():
    scenario = make_scenario()
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=bool(args_cli.enable_fabric),
    )
    # 120 Hz physics / decimation 2 = 60 Hz control; 60 / 4 = 15 Hz sensors.
    env_cfg.sim.dt = 1.0 / float(args_cli.physics_hz)
    env_cfg.decimation = int(args_cli.physics_hz // args_cli.control_hz)
    env_cfg.scene.num_envs = 1
    apply_asset_version_override(env_cfg)
    maybe_disable_resets(env_cfg)
    if not args_cli.show_debug_visuals:
        disable_debug_visual_cfg(env_cfg)
    if scenario is not None:
        apply_blue_tray_env_cfg(env_cfg, scenario)
        configure_event_randomization(env_cfg, scenario)
    if needs_external_camera() or needs_wrist_camera():
        attach_record_cameras(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    if scenario is not None:
        spawn_tray_and_apply_domain_randomization(scenario)
    if args_cli.record_sensors and needs_semantic_labels():
        if scenario is not None:
            print("[INFO] using pre-spawn semantic tags; skipping runtime edits on instanced robot visuals")
        else:
            add_smoke_test_semantics(env)
        env.sim.render()
    if not args_cli.show_debug_visuals:
        hide_debug_visuals()
    set_record_camera(env, scenario)
    set_preview_camera(env, scenario)
    return env, scenario


def setup_output(recording_index: int, episode_index: int | None = None):
    output_root = Path(args_cli.output_dir)
    episode_number = int(args_cli.episode_index if episode_index is None else episode_index) + 1
    run_dir = output_root / f"episode_{episode_number:06d}"
    if run_dir.exists():
        run_dir = output_root / f"episode_{episode_number:06d}_run_{int(time.time())}"
    log_dir = run_dir / "logs"
    sensor_dirs = {
        "external": {
            "rgb": run_dir / "frames" / "external" / "rgb",
            "depth": run_dir / "frames" / "external" / "depth",
            "semantic_seg": run_dir / "frames" / "external" / "semantic_seg",
            "instance_seg": run_dir / "frames" / "external" / "instance_seg",
        },
        "wrist": {
            "rgb": run_dir / "frames" / "wrist" / "rgb",
            "depth": run_dir / "frames" / "wrist" / "depth",
            "semantic_seg": run_dir / "frames" / "wrist" / "semantic_seg",
            "instance_seg": run_dir / "frames" / "wrist" / "instance_seg",
        },
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    if args_cli.record_sensors:
        for camera_name, camera_dirs in sensor_dirs.items():
            if camera_name == "wrist" and not needs_wrist_camera():
                continue
            for path in camera_dirs.values():
                path.mkdir(parents=True, exist_ok=True)
    return run_dir, log_dir, sensor_dirs


def camera_pose_for_frame(camera_sensor, env_id: int = 0) -> dict[str, Any]:
    return {
        "pos_w": tensor_to_numpy(camera_sensor.data.pos_w[env_id]).tolist(),
        "quat_w_ros": tensor_to_numpy(camera_sensor.data.quat_w_ros[env_id]).tolist(),
    }


def episode_rgb_brightness_gain(scenario: ScenarioSpec | None) -> float:
    """Return one deterministic exposure gain for the entire episode."""
    if not args_cli.domain_randomization:
        return 1.0
    if scenario is not None:
        return float(scenario.rgb_brightness_gain)
    rng = np.random.default_rng(args_cli.seed * 3000017 + args_cli.episode_index * 3011)
    return float(rng.uniform(*args_cli.rgb_brightness_range))


def write_camera_frame(
    camera,
    camera_name: str,
    scenario: ScenarioSpec | None,
    env_id: int,
    frame_idx: int,
    camera_dirs: dict[str, Path],
    video_writers: dict[str, Any],
    write_executor: ThreadPoolExecutor | None,
    pending_writes: list[Any],
) -> dict[str, Any]:
    outputs = camera.data.output
    frame: dict[str, Any] = {}
    episode_index = scenario.episode_index if scenario is not None else args_cli.episode_index

    if "rgb" in outputs:
        rgb = rgb_to_array(outputs["rgb"], env_id)
        if args_cli.domain_randomization:
            camera_seed_offset = 0 if camera_name == "external" else 104729
            rng = np.random.default_rng(
                args_cli.seed * 1000003 + episode_index * 1009 + frame_idx + camera_seed_offset
            )
            gain = episode_rgb_brightness_gain(scenario)
            rgb = np.clip(
                rgb.astype(np.float32) * gain + rng.normal(0.0, args_cli.rgb_noise_std, rgb.shape), 0, 255
            ).astype(np.uint8)
        rgb_path = camera_dirs["rgb"] / f"{frame_idx:06d}.png"
        submit_or_run(write_executor, pending_writes, write_rgb_png, rgb, rgb_path)
        if video_writers.get("rgb") is not None:
            video_writers["rgb"].append_data(rgb)
        frame["rgb"] = str(rgb_path)

    depth_key = "depth" if "depth" in outputs else "distance_to_image_plane"
    if depth_key in outputs:
        vis_range = (
            tuple(args_cli.external_depth_vis_range)
            if camera_name == "external"
            else tuple(args_cli.wrist_depth_vis_range)
        )
        depth, depth_png = depth_to_arrays(outputs[depth_key], vis_range, env_id)
        if args_cli.domain_randomization:
            camera_seed_offset = 0 if camera_name == "external" else 130363
            rng = np.random.default_rng(
                args_cli.seed * 2000003 + episode_index * 2011 + frame_idx + camera_seed_offset
            )
            valid = np.isfinite(depth)
            depth[valid] += rng.normal(0.0, args_cli.depth_noise_std, int(valid.sum())).astype(np.float32)
            depth[rng.random(depth.shape) < args_cli.depth_dropout_prob] = np.nan
            _, depth_png = depth_to_arrays(depth, vis_range)
        depth_npy_path = camera_dirs["depth"] / f"{frame_idx:06d}.npy"
        depth_png_path = camera_dirs["depth"] / f"{frame_idx:06d}.png"
        submit_or_run(write_executor, pending_writes, write_depth_files, depth, depth_png, depth_npy_path, depth_png_path)
        frame["depth_npy"] = str(depth_npy_path)
        frame["depth_png"] = str(depth_png_path)
        if video_writers.get("depth") is not None:
            video_writers["depth"].append_data(np.repeat(depth_png[..., None], 3, axis=-1))

    if "semantic_segmentation" in outputs:
        sem_path = camera_dirs["semantic_seg"] / f"{frame_idx:06d}.png"
        sem_img = segmentation_to_image(outputs["semantic_segmentation"], env_id)
        submit_or_run(write_executor, pending_writes, write_image_png, sem_img, sem_path)
        frame["semantic_seg"] = str(sem_path)
        if video_writers.get("semantic_seg") is not None:
            video_writers["semantic_seg"].append_data(sem_img)
        if args_cli.segmentation_stats:
            frame["semantic_segmentation_stats"] = segmentation_stats(outputs["semantic_segmentation"], env_id)

    if "instance_segmentation_fast" in outputs:
        inst_path = camera_dirs["instance_seg"] / f"{frame_idx:06d}.png"
        inst_img = segmentation_to_image(outputs["instance_segmentation_fast"], env_id)
        submit_or_run(write_executor, pending_writes, write_image_png, inst_img, inst_path)
        frame["instance_seg"] = str(inst_path)
        if video_writers.get("instance_seg") is not None:
            video_writers["instance_seg"].append_data(inst_img)
        if args_cli.segmentation_stats:
            frame["instance_segmentation_stats"] = segmentation_stats(outputs["instance_segmentation_fast"], env_id)

    try:
        frame["pose"] = camera_pose_for_frame(camera, env_id)
    except Exception as exc:
        frame["pose_error"] = str(exc)
    return frame


def write_record_frame(
    env,
    scenario: ScenarioSpec | None,
    env_id: int,
    frame_idx: int,
    sensor_dirs: dict[str, dict[str, Path]],
    video_writers: dict[str, dict[str, Any]],
    write_executor: ThreadPoolExecutor | None,
    pending_writes: list[Any],
) -> dict[str, Any] | None:
    if not args_cli.record_sensors:
        return None

    frame: dict[str, Any] = {}
    frame["external"] = write_camera_frame(
        env.scene["record_camera"],
        "external",
        scenario,
        env_id,
        frame_idx,
        sensor_dirs["external"],
        video_writers.get("external", {}),
        write_executor,
        pending_writes,
    )
    if needs_wrist_camera():
        frame["wrist"] = write_camera_frame(
            env.scene["wrist_camera"],
            "wrist",
            scenario,
            env_id,
            frame_idx,
            sensor_dirs["wrist"],
            video_writers.get("wrist", {}),
            write_executor,
            pending_writes,
        )
    return frame


class EpisodeRecorder:
    """Owns output folders and file handles for one or more recording episodes."""

    def __init__(self, scenario: ScenarioSpec | None, env_id: int = 0):
        self.scenario = scenario
        self.env_id = int(env_id)
        self.recording_index = 0
        self.enabled = False
        self.run_dir: Path | None = None
        self.log_dir: Path | None = None
        self.sensor_dirs: dict[str, dict[str, Path]] | None = None
        self.action_f = None
        self.state_f = None
        self.frame_f = None
        self.event_f = None
        self.video_writers: dict[str, dict[str, Any]] = {}
        self.write_executor: ThreadPoolExecutor | None = None
        self.pending_writes: list[Any] = []
        self.frame_idx = 0

    @property
    def configured(self) -> bool:
        return bool(args_cli.record or args_cli.record_sensors)

    def start_new_episode(self) -> None:
        if not self.configured:
            print("[WARN] recording is not configured. Start with --record or --record_sensors to use O/N recording hotkeys.")
            return

        self.close_current(mark_disabled=False)
        self.run_dir, self.log_dir, self.sensor_dirs = setup_output(self.recording_index, self.scenario.episode_index if self.scenario is not None else None)
        metadata = {
            "recording_index": self.recording_index,
            "scenario": asdict(self.scenario) if self.scenario is not None else {"mode": "lift_cube"},
            "timing": {
                "physics_hz": int(args_cli.physics_hz),
                "physics_dt": 1.0 / float(args_cli.physics_hz),
                "control_hz": int(args_cli.control_hz),
                "control_dt": 1.0 / float(args_cli.control_hz),
                "sensor_fps": output_sensor_fps(),
                "capture_every_control_steps": int(
                    (args_cli.control_hz // args_cli.fps) * args_cli.capture_every_n
                ),
            },
            "args": {
                key: value
                for key, value in vars(args_cli).items()
                if isinstance(value, (str, int, float, bool, list, tuple, type(None)))
            },
        }
        (self.log_dir / "scenario.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.action_f = (self.log_dir / "actions.jsonl").open("w") if args_cli.record else None
        self.state_f = (self.log_dir / "states.jsonl").open("w") if args_cli.record else None
        self.frame_f = (self.log_dir / "frames.jsonl").open("w") if args_cli.record_sensors else None
        self.event_f = (self.log_dir / "automation_events.jsonl").open("w") if args_cli.record else None
        self.video_writers = {}
        if args_cli.record_sensors and args_cli.save_video:
            cameras = ["external"] + (["wrist"] if needs_wrist_camera() else [])
            modalities = ["rgb", "depth", "semantic_seg", "instance_seg"]
            self.video_writers = {
                camera_name: {
                    modality: make_video_writer(self.run_dir / f"{camera_name}_{modality}.mp4")
                    for modality in modalities
                }
                for camera_name in cameras
            }
        self.write_executor = ThreadPoolExecutor(max_workers=4) if args_cli.record_sensors and args_cli.async_writes else None
        self.pending_writes = []
        self.frame_idx = 0
        self.enabled = True
        print(f"[INFO] recording started: {self.run_dir}")
        self.recording_index += 1

    def close_current(self, *, mark_disabled: bool = True) -> None:
        for handle in (self.action_f, self.state_f, self.frame_f, self.event_f):
            if handle:
                handle.close()
        for camera_writers in self.video_writers.values():
            for writer in camera_writers.values():
                writer.close()
        if self.write_executor:
            for future in self.pending_writes:
                future.result()
            self.write_executor.shutdown(wait=True)

        self.action_f = None
        self.state_f = None
        self.frame_f = None
        self.event_f = None
        self.video_writers = {}
        self.write_executor = None
        self.pending_writes = []
        if mark_disabled:
            if self.enabled and self.run_dir is not None:
                print(f"[INFO] recording stopped: {self.run_dir}")
            self.enabled = False

    def toggle(self) -> None:
        if self.enabled:
            self.close_current(mark_disabled=True)
        else:
            self.start_new_episode()

    def write_action(self, sim_step: int, sim_time: float, action: torch.Tensor, automation: dict[str, Any] | None = None) -> None:
        if self.enabled and self.action_f is not None:
            write_jsonl(self.action_f, {"sim_step": sim_step, "sim_time": sim_time,
                                        "action": tensor_to_numpy(action[0]).tolist(),
                                        "automation": automation})

    def write_state(self, env, sim_step: int, sim_time: float, automation: dict[str, Any] | None = None) -> None:
        if not self.enabled or self.state_f is None:
            return
        ee_pos, ee_quat = read_ee_pose_env(env, self.env_id)
        gripper_width, finger_joint_pos = read_gripper_width(env, self.env_id)
        obj_pos, obj_quat = read_object_pose_env(env, self.env_id)
        state_payload = {
            "sim_step": sim_step,
            "sim_time": sim_time,
            "ee_pos_env": tensor_to_numpy(ee_pos).tolist(),
            "ee_quat_xyzw": tensor_to_numpy(ee_quat).tolist(),
            "gripper_width": gripper_width,
            "finger_joint_pos": finger_joint_pos,
            "automation": automation,
            "object_pos_env": tensor_to_numpy(obj_pos).tolist(),
            "object_quat_xyzw": tensor_to_numpy(obj_quat).tolist(),
        }
        state_payload.update(read_scenario_objects_for_log(env, self.scenario, self.env_id))
        write_jsonl(self.state_f, state_payload)

    def write_result(self, result: dict[str, Any]) -> None:
        if self.log_dir is not None:
            (self.log_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    def write_automation_event(self, sim_step: int, sim_time: float, event: dict[str, Any]) -> None:
        if self.enabled and self.event_f is not None:
            write_jsonl(self.event_f, {"sim_step": sim_step, "sim_time": sim_time, **event})

    def write_frame(self, env, sim_step: int, sim_time: float) -> None:
        if not self.enabled or not args_cli.record_sensors or self.sensor_dirs is None:
            return
        frame_record = write_record_frame(
            env,
            self.scenario,
            self.env_id,
            self.frame_idx,
            self.sensor_dirs,
            self.video_writers,
            self.write_executor,
            self.pending_writes,
        )
        if frame_record is not None and self.frame_f is not None:
            frame_record.update({"frame_idx": self.frame_idx, "sim_step": sim_step, "sim_time": sim_time})
            write_jsonl(self.frame_f, frame_record)
        self.frame_idx += 1
        apply_write_backpressure(self.pending_writes)


def quat_yaw_xyzw(quat: torch.Tensor) -> float:
    x, y, z, w = [float(v.item()) for v in quat]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def projected_local_y_yaw_xyzw(quat: torch.Tensor) -> float:
    """Yaw of the gripper local Y axis projected onto the world XY plane."""
    x, y, z, w = [float(v.item()) for v in quat]
    axis_x = 2.0 * (x * y - w * z)
    axis_y = 1.0 - 2.0 * (x * x + z * z)
    return math.atan2(axis_y, axis_x)


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def square_cube_yaw_error(target_yaw: float, gripper_axis_yaw: float) -> float:
    """Shortest yaw error across the four equivalent orientations of a square cube."""
    quarter_turn = 0.5 * math.pi
    return (target_yaw - gripper_axis_yaw + 0.25 * math.pi) % quarter_turn - 0.25 * math.pi


def cuboid_grasp_yaw_error(cuboid_yaw: float, gripper_axis_yaw: float, size: Vec3) -> float:
    """Align the finger-closing axis with a cuboid shorter horizontal dimension."""
    if abs(float(size[0]) - float(size[1])) < 0.003:
        return square_cube_yaw_error(cuboid_yaw, gripper_axis_yaw)
    target_axis_yaw = cuboid_yaw if size[0] <= size[1] else cuboid_yaw + 0.5 * math.pi
    # A gripper axis is bidirectional, so orientations separated by pi are equivalent.
    return (target_axis_yaw - gripper_axis_yaw + 0.5 * math.pi) % math.pi - 0.5 * math.pi


def read_gripper_width(env, env_id: int = 0) -> tuple[float, list[float]]:
    robot = env.scene["robot"]
    joint_ids = robot.find_joints("panda_finger_joint.*")[0]
    values = robot.data.joint_pos[env_id, joint_ids]
    joints = tensor_to_numpy(values).reshape(-1).astype(float).tolist()
    return float(sum(joints)), joints


def tool_down_tilt_deg_xyzw(quat: torch.Tensor) -> float:
    """Return the tool +Z tilt from world down for an xyzw quaternion."""
    x, y, _, _ = [float(value.item()) for value in quat]
    local_z_world_z = 1.0 - 2.0 * (x * x + y * y)
    down_alignment = max(-1.0, min(1.0, -local_z_world_z))
    return math.degrees(math.acos(down_alignment))


def move_to_randomized_start_poses(
    env,
    scenarios: list[ScenarioSpec],
    active_count: int,
) -> None:
    """Translate active EEFs to randomized starts before any frame is recorded."""
    selected = [
        (env_id, scenario)
        for env_id, scenario in enumerate(scenarios[:active_count])
        if scenario.start_ee_target is not None
    ]
    if not selected:
        return
    if env.action_space.shape[-1] != 7:
        raise RuntimeError(
            f"start-pose pre-roll expects 7D actions, got {env.action_space.shape}"
        )

    target_by_env = {
        env_id: torch.tensor(
            scenario.start_ee_target, device=env.device, dtype=torch.float32
        )
        for env_id, scenario in selected
    }
    max_tilt = float(args_cli.start_pose_max_tilt_deg)
    for env_id, _ in selected:
        _, ee_quat = read_ee_pose_env(env, env_id)
        initial_tilt = tool_down_tilt_deg_xyzw(ee_quat)
        if initial_tilt > max_tilt:
            raise RuntimeError(
                f"env {env_id} initial tool tilt {initial_tilt:.2f} deg exceeds "
                f"the floor-facing limit {max_tilt:.2f} deg"
            )

    open_action = torch.zeros(
        (env.num_envs, 7), device=env.device, dtype=torch.float32
    )
    open_action[:, 6] = float(args_cli.gripper_open_command)
    reached: set[int] = set()
    for _ in range(int(args_cli.start_pose_timeout_steps)):
        action = open_action.clone()
        for env_id, _ in selected:
            ee_pos, ee_quat = read_ee_pose_env(env, env_id)
            tilt = tool_down_tilt_deg_xyzw(ee_quat)
            if tilt > max_tilt:
                raise RuntimeError(
                    f"env {env_id} tool tilt {tilt:.2f} deg exceeded the "
                    f"floor-facing limit {max_tilt:.2f} deg during start pre-roll"
                )
            delta = target_by_env[env_id] - ee_pos
            distance = float(torch.linalg.norm(delta).item())
            if distance <= float(args_cli.start_pose_tolerance):
                reached.add(env_id)
                continue
            reached.discard(env_id)
            max_step = float(args_cli.start_pose_step)
            step = delta if distance <= max_step else delta / distance * max_step
            action[env_id, 0:3] = step
        if len(reached) == len(selected):
            break
        env.step(action)

    for _ in range(int(args_cli.start_pose_hold_steps)):
        env.step(open_action)

    for env_id, scenario in selected:
        ee_pos, ee_quat = read_ee_pose_env(env, env_id)
        final_tilt = tool_down_tilt_deg_xyzw(ee_quat)
        distance = float(torch.linalg.norm(target_by_env[env_id] - ee_pos).item())
        scenario.start_ee_actual = tuple(
            round(float(value), 6) for value in tensor_to_numpy(ee_pos).tolist()
        )  # type: ignore[assignment]
        scenario.start_ee_tilt_deg = round(final_tilt, 4)
        scenario.start_ee_reached = distance <= float(args_cli.start_pose_tolerance)
        status = "reached" if scenario.start_ee_reached else "timeout"
        print(
            f"[START-AUG] env={env_id} {status} target={scenario.start_ee_target} "
            f"actual={scenario.start_ee_actual} error={distance:.4f}m "
            f"tool_down_tilt={final_tilt:.2f}deg"
        )
        if final_tilt > max_tilt:
            raise RuntimeError(
                f"env {env_id} final tool tilt {final_tilt:.2f} deg exceeds "
                f"the floor-facing limit {max_tilt:.2f} deg"
            )


class AutoPickPlaceController:
    """Yaw-aligning pick/place FSM with two-signal grasp verification."""

    SOLVER_RECOVERABLE_STATES = frozenset(
        {
            "recovery_waypoint",
            "approach",
            "align_yaw",
            "recenter_after_yaw",
            "descend",
            "move_above_slot",
            "place_descend",
        }
    )
    SOLVER_HOLDING_STATES = frozenset({"move_above_slot", "place_descend"})

    def __init__(self, scenario: ScenarioSpec | None, env_id: int = 0):
        self.scenario = scenario
        self.env_id = int(env_id)
        self.active = self.completed = self.failed = False
        self.state, self.state_steps = "idle", 0
        self.asset_name: str | None = None
        self.label: str | None = None
        self.current_spec: ScenarioObjectSpec | None = None
        self.processed: set[str] = set()
        self.slot_index = 0
        self.grasp_start_z = 0.0
        self.grasp_offset: torch.Tensor | None = None
        self.fixed_target: torch.Tensor | None = None
        self.retry_count = 0
        self.yaw_response_sign = 1.0
        self.yaw_probe_start: float | None = None
        self.recovery_waypoint: torch.Tensor | None = None
        self.recovery_completed = False
        self.solver_recovery_attempts = 0
        self.solver_recovery_holding = False
        self.solver_recovery_return_state: str | None = None
        self.solver_recovery_raise_target: torch.Tensor | None = None
        self.solver_recovery_center_target: torch.Tensor | None = None
        self.solver_recovery_reference_quat: torch.Tensor | None = None
        self.best_position_distance = math.inf
        self.steps_without_motion_progress = 0
        self.events: list[dict[str, Any]] = []

    def status_payload(self) -> dict[str, Any]:
        num_preplaced = self.scenario.num_preplaced if self.scenario is not None else 0
        return {"active": self.active, "completed": self.completed, "failed": self.failed,
                "state": self.state, "target": self.label, "slot_index": self.slot_index,
                "retry_count": self.retry_count, "env_id": self.env_id,
                "progress_stage": (
                    self.scenario.progress_stage if self.scenario is not None else 0
                ),
                "initial_preplaced_count": num_preplaced,
                "initial_remaining_count": (
                    self.scenario.num_remaining if self.scenario is not None else 0
                ),
                "newly_placed_count": max(0, len(self.processed) - num_preplaced),
                "recovery_augmented": self.recovery_waypoint is not None,
                "recovery_completed": self.recovery_completed,
                "solver_recovery_attempts": self.solver_recovery_attempts,
                "solver_recovery_holding": self.solver_recovery_holding}

    def drain_events(self) -> list[dict[str, Any]]:
        events, self.events = self.events, []
        return events

    def _event(self, event: str, **data: Any) -> None:
        payload = {"event": event, **self.status_payload(), **data}
        self.events.append(payload)
        print(f"[AUTO] {event}: {payload}")

    def _transition(self, state: str, **data: Any) -> None:
        self.state, self.state_steps = state, 0
        self.best_position_distance = math.inf
        self.steps_without_motion_progress = 0
        if state == "align_yaw":
            self.yaw_response_sign = 1.0
            self.yaw_probe_start = None
        self._event("state_transition", new_state=state, **data)

    def cancel(self, reason: str = "cancelled") -> None:
        if self.active:
            self._event("cancelled", reason=reason)
        self.active, self.state = False, "idle"

    def start(self, env) -> bool:
        if self.scenario is None or not self.scenario.blue_cubes:
            print("[WARN] full auto pick requires --scenario blue_tray")
            return False
        self.active, self.completed, self.failed = True, False, False
        self.processed = set(self.scenario.preplaced_blue_cube_names or [])
        self.slot_index = int(self.scenario.num_preplaced)
        self.retry_count = 0
        _, reference_quat = read_ee_pose_env(env, self.env_id)
        self.solver_recovery_reference_quat = reference_quat.detach().clone()
        reference_tilt = tool_down_tilt_deg_xyzw(reference_quat)
        if reference_tilt > float(args_cli.start_pose_max_tilt_deg):
            raise RuntimeError(
                f"env {self.env_id} solver-recovery reference tool tilt "
                f"{reference_tilt:.2f} deg exceeds the floor-facing limit "
                f"{float(args_cli.start_pose_max_tilt_deg):.2f} deg"
            )
        self._event(
            "sequence_started",
            placement_positions=self.scenario.placement_positions,
            progress_stage=self.scenario.progress_stage,
            preplaced_blue_cube_names=self.scenario.preplaced_blue_cube_names,
            remaining_blue_cube_names=self.scenario.remaining_blue_cube_names,
            start_pose_mode=self.scenario.start_pose_mode,
            start_ee_target=self.scenario.start_ee_target,
            start_ee_actual=self.scenario.start_ee_actual,
            solver_recovery_reference_quat=tensor_to_numpy(reference_quat).tolist(),
            solver_recovery_reference_tilt_deg=reference_tilt,
        )
        return self._select_next(env)

    def toggle(self, env) -> None:
        self.cancel("B key") if self.active else self.start(env)

    def _select_next(self, env) -> bool:
        assert self.scenario is not None
        candidates = []
        for cube in self.scenario.blue_cubes:
            if cube.name in self.processed:
                continue
            pos, _ = read_rigid_object_pose_env(env, cube.asset_name, self.env_id)
            if not cube_is_inside_tray(pos, self.scenario):
                candidates.append((cube, pos))
        if not candidates:
            self.active, self.completed, self.state = False, True, "complete"
            self._event(
                "sequence_completed",
                placed_count=len(self.processed),
                initially_preplaced_count=self.scenario.num_preplaced,
                newly_placed_count=len(self.processed) - self.scenario.num_preplaced,
            )
            return False
        ee_pos, _ = read_ee_pose_env(env, self.env_id)
        cube, pos = min(candidates, key=lambda item: float(torch.linalg.norm(item[1] - ee_pos).item()))
        self.asset_name, self.label, self.current_spec = cube.asset_name, cube.name, cube
        self.grasp_start_z = float(pos[2].item())
        self.grasp_offset = self.fixed_target = None
        self.solver_recovery_attempts = 0
        self.solver_recovery_holding = False
        self.solver_recovery_return_state = None
        self.solver_recovery_raise_target = None
        self.solver_recovery_center_target = None
        self.recovery_waypoint = (
            torch.tensor(cube.recovery_waypoint, device=env.device, dtype=torch.float32)
            if cube.recovery_waypoint is not None
            else None
        )
        self.recovery_completed = False
        self._transition(
            "open", target_pos=tensor_to_numpy(pos).tolist(),
            recovery_waypoint=cube.recovery_waypoint,
        )
        return True

    def _track_motion_progress(self, metric: float, epsilon: float) -> None:
        if metric + epsilon < self.best_position_distance:
            self.best_position_distance = metric
            self.steps_without_motion_progress = 0
        else:
            self.steps_without_motion_progress += 1

    def _position_action(
        self,
        action,
        ee_pos,
        target,
        max_step: float,
        *,
        track_progress: bool = True,
        tolerance: float | None = None,
        xy_tolerance: float | None = None,
    ) -> bool:
        delta = target - ee_pos
        distance = float(torch.linalg.norm(delta).item())
        if track_progress:
            self._track_motion_progress(
                distance, float(args_cli.solver_recovery_progress_epsilon)
            )
        action[:, 0:6] = 0.0
        position_tolerance = (
            float(args_cli.auto_pick_tolerance)
            if tolerance is None
            else float(tolerance)
        )
        xy_distance = float(torch.linalg.norm(delta[:2]).item())
        if distance <= position_tolerance and (
            xy_tolerance is None or xy_distance <= float(xy_tolerance)
        ):
            return True
        step = delta if distance <= max_step else delta / distance * max_step
        action[:, 0:3] = step.unsqueeze(0)
        return False

    def _orientation_action(
        self,
        action: torch.Tensor,
        ee_pos: torch.Tensor,
        ee_quat: torch.Tensor,
        target_quat: torch.Tensor,
        hold_position: torch.Tensor,
    ) -> tuple[bool, float]:
        """Rotate along the shortest quaternion arc while holding a fixed EEF position."""
        # q and -q describe the same orientation. Canonicalize the target into
        # the current quaternion hemisphere before converting the error to an
        # axis-angle command; otherwise compute_pose_error may request the long
        # (~2*pi) arc after the sign boundary is crossed.
        if float(torch.dot(ee_quat, target_quat).item()) < 0.0:
            target_quat = -target_quat
        _, axis_angle_error = compute_pose_error(
            ee_pos.unsqueeze(0),
            ee_quat.unsqueeze(0),
            ee_pos.unsqueeze(0),
            target_quat.unsqueeze(0),
            rot_error_type="axis_angle",
        )
        error = axis_angle_error[0]
        angular_error = float(torch.linalg.norm(error).item())
        self._track_motion_progress(
            angular_error, float(args_cli.solver_recovery_yaw_progress_epsilon)
        )
        action[:, 0:6] = 0.0
        position_error = hold_position - ee_pos
        position_distance = float(torch.linalg.norm(position_error).item())
        position_step = float(args_cli.auto_pick_step)
        if position_distance > position_step:
            position_error = position_error / position_distance * position_step
        action[:, 0:3] = position_error.unsqueeze(0)
        if angular_error <= float(args_cli.solver_recovery_reorient_tolerance):
            return True, angular_error
        max_step = float(args_cli.solver_recovery_reorient_step)
        step = error if angular_error <= max_step else error / angular_error * max_step
        action[:, 3:6] = step.unsqueeze(0)
        return False, angular_error

    def _placement_slot(self, env) -> torch.Tensor:
        """Return the current slot with the selected cuboid resting on the tray floor."""
        assert self.scenario is not None and self.current_spec is not None
        slot = torch.tensor(self.scenario.placement_positions[self.slot_index], device=env.device)
        slot[2] = (
            float(self.scenario.tray_pos[2])
            + 0.5 * float(self.scenario.tray_size[2])
            + 0.5 * float(self.current_spec.size[2])
            + 0.003
        )
        return slot

    def _start_solver_recovery(
        self,
        env,
        action: torch.Tensor,
        ee_pos: torch.Tensor,
        trigger: str,
    ) -> bool:
        """Escape a stalled differential-IK solution through high neutral waypoints."""
        stalled_state = self.state
        max_attempts = int(args_cli.solver_recovery_max_attempts)
        if (
            stalled_state not in self.SOLVER_RECOVERABLE_STATES
            or self.solver_recovery_attempts >= max_attempts
        ):
            return False

        self.solver_recovery_attempts += 1
        self.solver_recovery_holding = stalled_state in self.SOLVER_HOLDING_STATES
        self.solver_recovery_return_state = (
            "move_above_slot" if self.solver_recovery_holding else "open"
        )
        if stalled_state == "recovery_waypoint":
            # The deliberate augmentation is single-shot; do not repeat it after
            # an actual IK recovery.
            self.recovery_completed = True
        if not self.solver_recovery_holding:
            self.grasp_offset = None
            self.fixed_target = None

        center = None
        if not self.solver_recovery_holding:
            center = torch.tensor(
                args_cli.solver_recovery_center,
                device=env.device,
                dtype=torch.float32,
            )
        raise_target = ee_pos.clone()
        raise_target[2] = max(
            float(ee_pos[2].item()) + float(args_cli.solver_recovery_raise_clearance),
            float(args_cli.solver_recovery_center[2]),
        )
        self.solver_recovery_raise_target = raise_target
        self.solver_recovery_center_target = center
        action[:, 0:6] = 0.0
        action[:, 6] = (
            args_cli.gripper_close_command
            if self.solver_recovery_holding
            else args_cli.gripper_open_command
        )
        self._event(
            "solver_recovery_started",
            trigger=trigger,
            stalled_state=stalled_state,
            attempt=self.solver_recovery_attempts,
            raise_target=tensor_to_numpy(raise_target).tolist(),
            center_target=(
                tensor_to_numpy(center).tolist() if center is not None else None
            ),
            recovery_mode=(
                "raise_only"
                if self.solver_recovery_holding
                else "raise_recenter_reorient"
            ),
            return_state=self.solver_recovery_return_state,
        )
        self._transition("solver_recovery_raise", stalled_state=stalled_state)
        return True

    def apply(self, env, action: torch.Tensor) -> torch.Tensor:
        if not self.active or self.asset_name is None:
            return action
        state_at_start = self.state
        self.state_steps += 1
        action[:, :] = 0.0
        action[:, 6] = args_cli.gripper_open_command
        ee_pos, ee_quat = read_ee_pose_env(env, self.env_id)
        if self.state_steps > int(args_cli.auto_pick_state_timeout):
            failed_state = self.state
            if self._start_solver_recovery(
                env, action, ee_pos, trigger="state_timeout"
            ):
                return action
            reason = "state_timeout"
            if (
                failed_state in self.SOLVER_RECOVERABLE_STATES
                and int(args_cli.solver_recovery_max_attempts) > 0
                and self.solver_recovery_attempts
                >= int(args_cli.solver_recovery_max_attempts)
            ):
                reason = "solver_recovery_exhausted"
            self.failed, self.active, self.state = True, False, "failed"
            self._event(
                "sequence_failed", reason=reason, failed_state=failed_state
            )
            return action

        cube_pos, cube_quat = read_rigid_object_pose_env(env, self.asset_name, self.env_id)
        hold = int(args_cli.auto_pick_hold_steps)

        if self.state == "solver_recovery_raise":
            assert self.solver_recovery_raise_target is not None
            reached = self._position_action(
                action, ee_pos, self.solver_recovery_raise_target,
                float(args_cli.auto_pick_step),
            )
            action[:, 6] = (
                args_cli.gripper_close_command
                if self.solver_recovery_holding
                else args_cli.gripper_open_command
            )
            if reached:
                if self.solver_recovery_holding:
                    return_state = self.solver_recovery_return_state or "move_above_slot"
                    self._event(
                        "solver_recovery_completed",
                        return_state=return_state,
                        attempt=self.solver_recovery_attempts,
                        recovery_mode="raise_only",
                    )
                    self.solver_recovery_holding = False
                    self._transition(return_state)
                else:
                    self._transition("solver_recovery_recenter")
        elif self.state == "solver_recovery_recenter":
            assert self.solver_recovery_center_target is not None
            reached = self._position_action(
                action, ee_pos, self.solver_recovery_center_target,
                float(args_cli.auto_pick_step),
            )
            action[:, 6] = (
                args_cli.gripper_close_command
                if self.solver_recovery_holding
                else args_cli.gripper_open_command
            )
            if reached:
                self._transition("solver_recovery_reorient")
        elif self.state == "solver_recovery_reorient":
            assert self.solver_recovery_reference_quat is not None
            assert self.solver_recovery_center_target is not None
            reached, angular_error = self._orientation_action(
                action,
                ee_pos,
                ee_quat,
                self.solver_recovery_reference_quat,
                self.solver_recovery_center_target,
            )
            action[:, 6] = args_cli.gripper_open_command
            if reached:
                return_state = self.solver_recovery_return_state or "open"
                final_tilt = tool_down_tilt_deg_xyzw(ee_quat)
                self._event(
                    "solver_recovery_reoriented",
                    angular_error=angular_error,
                    tool_down_tilt_deg=final_tilt,
                    reference_quat=tensor_to_numpy(
                        self.solver_recovery_reference_quat
                    ).tolist(),
                )
                self._event(
                    "solver_recovery_completed", return_state=return_state,
                    attempt=self.solver_recovery_attempts,
                    recovery_mode="raise_recenter_reorient",
                )
                self.solver_recovery_holding = False
                self._transition(return_state)
            elif (
                self.steps_without_motion_progress
                >= int(args_cli.solver_recovery_stall_steps)
                or self.state_steps
                >= int(args_cli.solver_recovery_reorient_max_steps)
            ):
                final_tilt = tool_down_tilt_deg_xyzw(ee_quat)
                if final_tilt <= float(args_cli.start_pose_max_tilt_deg):
                    return_state = self.solver_recovery_return_state or "open"
                    self._event(
                        "solver_recovery_reorient_partial",
                        angular_error=angular_error,
                        tool_down_tilt_deg=final_tilt,
                        reason="orientation_stall_with_safe_floor_facing_tool",
                    )
                    self._event(
                        "solver_recovery_completed",
                        return_state=return_state,
                        attempt=self.solver_recovery_attempts,
                        recovery_mode="raise_recenter_reorient_partial",
                    )
                    self.solver_recovery_holding = False
                    self._transition(return_state)
        elif self.state == "open":
            if self.state_steps >= hold:
                next_state = (
                    "recovery_waypoint"
                    if self.recovery_waypoint is not None and not self.recovery_completed
                    else "approach"
                )
                self._transition(next_state)
        elif self.state == "recovery_waypoint":
            assert self.recovery_waypoint is not None
            if self._position_action(
                action, ee_pos, self.recovery_waypoint, float(args_cli.auto_pick_step)
            ):
                self.recovery_completed = True
                self._event(
                    "recovery_waypoint_reached",
                    waypoint=tensor_to_numpy(self.recovery_waypoint).tolist(),
                )
                self._transition("approach")
        elif self.state == "align_yaw":
            # Keep the tool centered above the cube while rotating. Pure
            # rotation commands can otherwise translate the offset EE frame.
            approach_target = cube_pos.clone()
            approach_target[2] += float(args_cli.auto_pick_approach_height)
            self._position_action(
                action, ee_pos, approach_target, float(args_cli.auto_pick_step),
                track_progress=False,
            )
            target_yaw = quat_yaw_xyzw(cube_quat) + float(args_cli.auto_pick_yaw_offset)
            gripper_yaw = projected_local_y_yaw_xyzw(ee_quat)
            assert self.current_spec is not None
            error = cuboid_grasp_yaw_error(target_yaw, gripper_yaw, self.current_spec.size)
            self._track_motion_progress(
                abs(error), float(args_cli.solver_recovery_yaw_progress_epsilon)
            )
            # Probe one fixed positive command long enough to overcome physics
            # response lag, then lock the observed IK yaw polarity.
            probe_steps = int(round(0.4 * float(args_cli.control_hz)))
            if self.yaw_probe_start is None:
                self.yaw_probe_start = gripper_yaw
            if abs(error) > float(args_cli.auto_pick_yaw_tolerance) and self.state_steps <= probe_steps:
                action[:, 5] = min(0.04, float(args_cli.auto_pick_yaw_step))
                return action
            if self.state_steps == probe_steps + 1:
                measured_delta = wrap_angle(gripper_yaw - self.yaw_probe_start)
                if abs(measured_delta) > 1.0e-4:
                    self.yaw_response_sign = 1.0 if measured_delta > 0.0 else -1.0
            yaw_cmd = max(-float(args_cli.auto_pick_yaw_step), min(float(args_cli.auto_pick_yaw_step), error))
            yaw_cmd *= self.yaw_response_sign
            action[:, 5] = yaw_cmd
            if abs(error) <= float(args_cli.auto_pick_yaw_tolerance):
                xy_residual = approach_target[:2] - ee_pos[:2]
                xy_error = float(torch.linalg.norm(xy_residual).item())
                self._event(
                    "yaw_aligned",
                    yaw_error=error,
                    post_yaw_xy_error=xy_error,
                    post_yaw_x_error=abs(float(xy_residual[0].item())),
                    post_yaw_y_error=abs(float(xy_residual[1].item())),
                )
                self._transition("recenter_after_yaw")
        elif self.state == "approach":
            target = cube_pos.clone()
            target[2] += float(args_cli.auto_pick_approach_height)
            if self._position_action(action, ee_pos, target, float(args_cli.auto_pick_step)):
                # Align directly above the cube; this remains reachable after
                # retreating from a previous tray placement.
                self._transition("align_yaw")
        elif self.state == "recenter_after_yaw":
            # Yaw motion can translate the offset EE frame by several
            # millimeters. Hold the completed orientation and explicitly
            # re-center both X and Y above the live cube pose before descending.
            target = cube_pos.clone()
            target[2] += float(args_cli.auto_pick_approach_height)
            if self._position_action(
                action,
                ee_pos,
                target,
                float(args_cli.auto_pick_step),
                tolerance=float(args_cli.auto_pick_recenter_tolerance),
                xy_tolerance=float(args_cli.auto_pick_recenter_tolerance),
            ):
                residual = target - ee_pos
                self._event(
                    "post_yaw_recentered",
                    xy_error=float(torch.linalg.norm(residual[:2]).item()),
                    x_error=abs(float(residual[0].item())),
                    y_error=abs(float(residual[1].item())),
                    z_error=abs(float(residual[2].item())),
                    tolerance=float(args_cli.auto_pick_recenter_tolerance),
                )
                self._transition("descend")
        elif self.state == "descend":
            if self._position_action(
                action,
                ee_pos,
                cube_pos,
                float(args_cli.auto_pick_descend_step),
                xy_tolerance=float(args_cli.auto_pick_recenter_tolerance),
            ):
                self.grasp_start_z = float(cube_pos[2].item())
                residual = cube_pos - ee_pos
                self._event(
                    "pre_grasp_centered",
                    xy_error=float(torch.linalg.norm(residual[:2]).item()),
                    x_error=abs(float(residual[0].item())),
                    y_error=abs(float(residual[1].item())),
                    z_error=abs(float(residual[2].item())),
                    xy_tolerance=float(args_cli.auto_pick_recenter_tolerance),
                )
                self._transition("close")
        elif self.state == "close":
            self._position_action(
                action,
                ee_pos,
                cube_pos,
                float(args_cli.auto_pick_descend_step),
                track_progress=False,
                xy_tolerance=float(args_cli.auto_pick_recenter_tolerance),
            )
            action[:, 6] = args_cli.gripper_close_command
            if self.state_steps >= hold:
                self.fixed_target = ee_pos.clone()
                self.fixed_target[2] += float(args_cli.auto_pick_lift_height)
                self._transition("lift")
        elif self.state == "lift":
            action[:, 6] = args_cli.gripper_close_command
            position_ready = self._position_action(
                action, ee_pos, self.fixed_target, float(args_cli.auto_pick_step)
            )
            cube_lift = float(cube_pos[2].item()) - self.grasp_start_z
            if position_ready or cube_lift >= float(args_cli.auto_pick_min_transport_lift):
                self._transition("verify_grasp")
        elif self.state == "verify_grasp":
            action[:, 6] = args_cli.gripper_close_command
            width, joints = read_gripper_width(env, self.env_id)
            lifted = float(cube_pos[2].item()) - self.grasp_start_z
            if width >= float(args_cli.grasp_min_width) and lifted >= float(args_cli.grasp_min_lift):
                self.grasp_offset = ee_pos - cube_pos
                self._event("grasp_verified", gripper_width=width, finger_joint_pos=joints, cube_lift=lifted)
                self._transition("move_above_slot")
            elif self.state_steps >= hold:
                self._event("grasp_failed", gripper_width=width, finger_joint_pos=joints, cube_lift=lifted)
                if self.retry_count < 1:
                    self.retry_count += 1
                    self._transition("open")
                else:
                    self.failed, self.active, self.state = True, False, "failed"
                    self._event("sequence_failed", reason="grasp_verification")
        elif self.state == "move_above_slot":
            action[:, 6] = args_cli.gripper_close_command
            slot = self._placement_slot(env)
            target = slot + self.grasp_offset
            target[2] += float(args_cli.auto_pick_approach_height)
            if self._position_action(action, ee_pos, target, float(args_cli.auto_pick_step)):
                self._transition("place_descend")
        elif self.state == "place_descend":
            action[:, 6] = args_cli.gripper_close_command
            slot = self._placement_slot(env)
            release_target = slot + self.grasp_offset
            release_target[2] += float(args_cli.auto_place_drop_height)
            if self._position_action(action, ee_pos, release_target, float(args_cli.auto_pick_descend_step)):
                self._transition("release")
        elif self.state == "release":
            if self.state_steps >= hold:
                self.fixed_target = ee_pos.clone()
                self.fixed_target[2] += float(args_cli.auto_pick_approach_height)
                self._transition("retreat")
        elif self.state == "retreat":
            if self._position_action(action, ee_pos, self.fixed_target, float(args_cli.auto_pick_step)):
                slot = self._placement_slot(env)
                placement_error = float(torch.linalg.norm(cube_pos - slot).item())
                if placement_error > 0.05:
                    self.failed, self.active, self.state = True, False, "failed"
                    self._event("sequence_failed", reason="placement_verification", placement_error=placement_error)
                    return action
                placed = self.label
                self.processed.add(placed or "")
                self.slot_index += 1
                self.retry_count = 0
                self._event("cube_placed", placed_label=placed)
                self._select_next(env)
        if (
            self.active
            and self.state == state_at_start
            and state_at_start in self.SOLVER_RECOVERABLE_STATES
            and int(args_cli.solver_recovery_max_attempts) > 0
            and self.steps_without_motion_progress
            >= int(args_cli.solver_recovery_stall_steps)
        ):
            if self._start_solver_recovery(
                env, action, ee_pos, trigger="position_stall"
            ):
                return action
            failed_state = self.state
            self._event(
                "solver_recovery_exhausted",
                stalled_state=failed_state,
                attempts=self.solver_recovery_attempts,
            )
            self.failed, self.active, self.state = True, False, "failed"
            self._event(
                "sequence_failed",
                reason="solver_recovery_exhausted",
                failed_state=failed_state,
            )
        return action

def run_automatic_episode(env, scenario: ScenarioSpec) -> None:
    """Run one autonomous pick-place episode and record all configured outputs."""
    if env.action_space.shape[-1] != 7:
        raise RuntimeError(f"automatic controller expects 7D action, got {env.action_space.shape}")

    recorder = EpisodeRecorder(scenario)
    auto_pick = AutoPickPlaceController(scenario)
    open_action = torch.zeros((env.num_envs, 7), device=env.device, dtype=torch.float32)
    open_action[:, 6] = float(args_cli.gripper_open_command)

    if args_cli.settle_steps:
        print(f"[INFO] settling scene for {args_cli.settle_steps} control steps")
        for _ in range(int(args_cli.settle_steps)):
            env.step(open_action)

    level_scenario_cubes(env, scenario)
    for _ in range(int(round(0.24 * float(args_cli.control_hz)))):
        env.step(open_action)
    move_to_randomized_start_poses(env, [scenario], 1)
    env.sim.render()

    recorder.start_new_episode()
    auto_pick.start(env)

    step_dt = float(getattr(env, "step_dt", env.cfg.sim.dt * env.cfg.decimation))
    actual_control_hz = 1.0 / step_dt
    expected_control_hz = float(args_cli.control_hz)
    if not math.isclose(actual_control_hz, expected_control_hz, rel_tol=0.0, abs_tol=1.0e-6):
        raise RuntimeError(
            f"environment control rate mismatch: expected {expected_control_hz:g} Hz, "
            f"got {actual_control_hz:.9g} Hz"
        )
    capture_ratio = actual_control_hz / float(args_cli.fps)
    capture_steps = int(round(capture_ratio))
    if not math.isclose(capture_ratio, float(capture_steps), rel_tol=0.0, abs_tol=1.0e-6):
        raise RuntimeError(
            f"sensor FPS {args_cli.fps} does not divide control rate {actual_control_hz:.9g} Hz exactly"
        )
    record_every = capture_steps * args_cli.capture_every_n
    log_every = args_cli.log_every_n if args_cli.log_every_n > 0 else (record_every if args_cli.record_sensors else 1)
    output_fps = actual_control_hz / float(record_every)
    print(
        f"[INFO] timing: physics={args_cli.physics_hz} Hz control={actual_control_hz:g} Hz "
        f"sensor={output_fps:g} Hz; capture every {record_every} control steps; "
        f"log every {log_every} control steps"
    )

    sim_step = 0
    try:
        while simulation_app.is_running():
            action = open_action.clone()
            action = auto_pick.apply(env, action)
            timestamp = sim_step * step_dt

            for auto_event in auto_pick.drain_events():
                recorder.write_automation_event(sim_step, timestamp, auto_event)
            if sim_step % log_every == 0:
                recorder.write_action(sim_step, timestamp, action, auto_pick.status_payload())
                recorder.write_state(env, sim_step, timestamp, auto_pick.status_payload())
            if sim_step % record_every == 0:
                update_wrist_follow_camera(env, force_recompute=True)
                recorder.write_frame(env, sim_step, timestamp)

            # The observation/frame at time t is paired with the action that
            # advances the environment from t to t + control_dt.
            env.step(action)
            env.sim.render()
            sim_step += 1
            if auto_pick.completed or auto_pick.failed:
                break
    finally:
        recorder.write_result(auto_pick.status_payload())
        recorder.close_current(mark_disabled=True)

    if auto_pick.failed:
        print(f"[ERROR] automatic episode failed: {auto_pick.status_payload()}")
    else:
        print(f"[DONE] automatic episode completed: {auto_pick.status_payload()}")



def make_vector_template_scenario(first_episode: int) -> ScenarioSpec:
    """Create every possible asset slot once; inactive slots are hidden per environment."""
    saved_counts = (
        args_cli.min_blue_cubes,
        args_cli.max_blue_cubes,
        args_cli.min_red_cubes,
        args_cli.max_red_cubes,
    )
    saved_partial_probs = (
        args_cli.partial_progress_2_cube_prob,
        args_cli.partial_progress_3_cube_prob,
    )
    try:
        args_cli.min_blue_cubes = args_cli.max_blue_cubes
        args_cli.min_red_cubes = args_cli.max_red_cubes
        # The clone template describes asset capacity, not a training episode.
        # Keep every template object loose and non-overlapping before PhysX starts.
        args_cli.partial_progress_2_cube_prob = 0.0
        args_cli.partial_progress_3_cube_prob = 0.0
        template = generate_blue_tray_scenario(first_episode)
    finally:
        (
            args_cli.min_blue_cubes,
            args_cli.max_blue_cubes,
            args_cli.min_red_cubes,
            args_cli.max_red_cubes,
        ) = saved_counts
        (
            args_cli.partial_progress_2_cube_prob,
            args_cli.partial_progress_3_cube_prob,
        ) = saved_partial_probs

    # A common base mesh is required for scene cloning. Isaac Lab's USD event
    # scales each clone independently before PhysX starts.
    base_size = float(args_cli.cube_size)
    for cube in [*template.blue_cubes, *template.red_cubes]:
        cube.size = (base_size, base_size, base_size)
        cube.pos = (cube.pos[0], cube.pos[1], cuboid_center_z(base_size))
    return template


def configure_vector_scale_events(env_cfg, template: ScenarioSpec) -> None:
    """Randomize XYZ scale independently for each clone before simulation starts."""
    if not args_cli.domain_randomization or getattr(env_cfg, "events", None) is None:
        return
    env_cfg.scene.replicate_physics = False
    base_size = float(args_cli.cube_size)
    scale_low = float(args_cli.cube_size_range[0]) / base_size
    scale_high = float(args_cli.cube_size_range[1]) / base_size
    for cube in [*template.blue_cubes, *template.red_cubes]:
        setattr(
            env_cfg.events,
            f"vector_scale_{cube.asset_name}",
            EventTerm(
                func=event_mdp.randomize_rigid_body_scale,
                mode="usd",
                params={
                    "asset_cfg": SceneEntityCfg(cube.asset_name),
                    "scale_range": {
                        "x": (scale_low, scale_high),
                        "y": (scale_low, scale_high),
                        "z": (scale_low, scale_high),
                    },
                },
            ),
        )


def make_vector_env(template: ScenarioSpec):
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=bool(args_cli.enable_fabric),
    )
    env_cfg.sim.dt = 1.0 / float(args_cli.physics_hz)
    env_cfg.decimation = int(args_cli.physics_hz // args_cli.control_hz)
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.scene.env_spacing = max(VECTOR_ENV_SPACING, float(getattr(env_cfg.scene, "env_spacing", 0.0)))
    apply_asset_version_override(env_cfg)
    maybe_disable_resets(env_cfg)
    if not args_cli.show_debug_visuals:
        disable_debug_visual_cfg(env_cfg)
    apply_blue_tray_env_cfg(env_cfg, template)
    configure_vector_scale_events(env_cfg, template)
    if needs_external_camera() or needs_wrist_camera():
        attach_record_cameras(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    if not args_cli.show_debug_visuals:
        hide_debug_visuals()
    set_preview_camera(env, template)
    return env


def read_vector_asset_sizes(env, template: ScenarioSpec) -> list[dict[str, Vec3]]:
    """Read the actual per-env USD scales sampled before PhysX initialization."""
    stage = omni.usd.get_context().get_stage()
    base_size = float(args_cli.cube_size)
    sizes: list[dict[str, Vec3]] = [dict() for _ in range(env.num_envs)]
    for env_id in range(env.num_envs):
        for cube in [*template.blue_cubes, *template.red_cubes]:
            prim_path = cube.prim_path.replace("/World/envs/env_0", f"/World/envs/env_{env_id}")
            prim = stage.GetPrimAtPath(prim_path)
            scale = (1.0, 1.0, 1.0)
            if prim and prim.IsValid():
                for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                        value = op.Get()
                        scale = (float(value[0]), float(value[1]), float(value[2]))
                        break
            sizes[env_id][cube.asset_name] = (
                base_size * scale[0],
                base_size * scale[1],
                base_size * scale[2],
            )
    return sizes


def adapt_vector_scenario(
    scenario: ScenarioSpec,
    template: ScenarioSpec,
    env_id: int,
    asset_sizes: dict[str, Vec3],
    fixed_tray_pos: Vec3 | None,
) -> ScenarioSpec:
    """Make scenario metadata match the cloned assets and fixed per-slot tray."""
    template_objects = {
        cube.asset_name: cube for cube in [*template.blue_cubes, *template.red_cubes]
    }
    partial_start_clearance: float | None = None
    if scenario.preplaced_blue_cube_names and scenario.start_ee_target is not None:
        last_name = scenario.preplaced_blue_cube_names[-1]
        reference = next(cube for cube in scenario.blue_cubes if cube.name == last_name)
        partial_start_clearance = float(scenario.start_ee_target[2]) - (
            float(reference.pos[2]) + 0.5 * float(reference.size[2])
        )
    for cube in [*scenario.blue_cubes, *scenario.red_cubes]:
        physical = template_objects[cube.asset_name]
        cube.size = asset_sizes[cube.asset_name]
        cube.mass = physical.mass
        cube.static_friction = physical.static_friction
        cube.dynamic_friction = physical.dynamic_friction
        cube.restitution = physical.restitution
        if cube.preplaced:
            cube.pos = (
                cube.pos[0],
                cube.pos[1],
                float(scenario.tray_pos[2])
                + 0.5 * float(scenario.tray_size[2])
                + 0.5 * float(cube.size[2])
                + 0.003,
            )
        else:
            cube.pos = (cube.pos[0], cube.pos[1], cuboid_center_z(cube.size[2]))
        cube.prim_path = cube.prim_path.replace(
            "/World/envs/env_0", f"/World/envs/env_{env_id}"
        )

    if partial_start_clearance is not None and scenario.start_ee_target is not None:
        last_name = (scenario.preplaced_blue_cube_names or [""])[-1]
        reference = next(cube for cube in scenario.blue_cubes if cube.name == last_name)
        scenario.start_ee_target = (
            float(scenario.start_ee_target[0]),
            float(scenario.start_ee_target[1]),
            float(reference.pos[2])
            + 0.5 * float(reference.size[2])
            + partial_start_clearance,
        )

    if fixed_tray_pos is not None:
        mismatch = max(
            abs(float(actual) - float(expected))
            for actual, expected in zip(scenario.tray_pos, fixed_tray_pos)
        )
        if mismatch > 1.0e-6:
            raise RuntimeError(
                f"vector scenario tray mismatch: sampled={scenario.tray_pos} fixed={fixed_tray_pos}"
            )
        scenario.tray_pos = tuple(float(value) for value in fixed_tray_pos)  # type: ignore[assignment]

    # Vector environments share one neutral Dome. Local Sphere lights supply
    # independent per-env direction, color, position, and intensity.
    rng = np.random.default_rng(scenario.seed * 7919 + 17)
    light_palette: list[Color] = [
        (1.0, 0.92, 0.80),
        (0.82, 0.90, 1.0),
        (1.0, 0.98, 0.92),
    ]
    local_lights = [
        ScenarioLightSpec("dome", 900.0, (0.86, 0.86, 0.86))
    ]
    pos_min = np.asarray(args_cli.sphere_light_position_min, dtype=np.float32)
    pos_max = np.asarray(args_cli.sphere_light_position_max, dtype=np.float32)
    for _ in range(3):
        local_lights.append(
            ScenarioLightSpec(
                light_type="sphere",
                intensity=sample_log_uniform(rng, args_cli.sphere_light_intensity_range),
                color=jitter_color(
                    rng, light_palette[int(rng.integers(0, len(light_palette)))], 0.08
                ),
                position=tuple(rng.uniform(pos_min, pos_max).round(4).tolist()),
                scale=float(rng.uniform(*args_cli.sphere_light_scale_range)),
            )
        )
    scenario.lights = local_lights
    scenario.light_intensity = 900.0
    scenario.light_color = (0.86, 0.86, 0.86)
    validate_scenario_spawn_clearance(scenario)
    return scenario



def spawn_vector_static_scene(env, scenarios: list[ScenarioSpec]) -> None:
    """Spawn tray, seamless wall/floor backdrop, and local lights per env."""
    dome_cfg = sim_utils.DomeLightCfg(
        intensity=900.0,
        color=(0.86, 0.86, 0.86),
        visible_in_primary_ray=False,
    )
    dome_cfg.func("/World/VectorDomeLight", dome_cfg)

    for env_id, scenario in enumerate(scenarios):
        tray_cfg = sim_utils.CuboidCfg(
            size=scenario.tray_size,
            semantic_tags=[("class", "tray")],
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=scenario.tray_color, roughness=0.65
            ),
        )
        tray_cfg.func(
            f"/World/envs/env_{env_id}/Tray",
            tray_cfg,
            translation=tuple(float(v) for v in scenario.tray_pos),
        )

        backdrop_cfg = sim_utils.CuboidCfg(
            size=BACKDROP_WALL_SIZE,
            semantic_tags=[("class", "background")],
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=scenario.background_color, roughness=0.9
            ),
        )
        backdrop_cfg.func(
            f"/World/envs/env_{env_id}/Background",
            backdrop_cfg,
            translation=BACKDROP_BACK_POS,
        )

        # Close the lower camera frustum so the black world is never exposed.
        backdrop_floor_cfg = sim_utils.CuboidCfg(
            size=BACKDROP_FLOOR_SIZE,
            semantic_tags=[("class", "background")],
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=scenario.background_color, roughness=0.9
            ),
        )
        backdrop_floor_cfg.func(
            f"/World/envs/env_{env_id}/BackgroundFloor",
            backdrop_floor_cfg,
            translation=BACKDROP_FLOOR_POS,
        )

        # Seal the remaining faces so adjacent vector environments never enter
        # this camera frustum. These are visual-only cards (no collision props).
        room_faces = [
            ("BackgroundFront", BACKDROP_WALL_SIZE, BACKDROP_FRONT_POS),
            ("BackgroundLeft", BACKDROP_SIDE_WALL_SIZE, BACKDROP_LEFT_POS),
            ("BackgroundRight", BACKDROP_SIDE_WALL_SIZE, BACKDROP_RIGHT_POS),
            ("BackgroundCeiling", BACKDROP_FLOOR_SIZE, BACKDROP_CEILING_POS),
        ]
        for face_name, face_size, face_pos in room_faces:
            face_cfg = sim_utils.CuboidCfg(
                size=face_size,
                semantic_tags=[("class", "background")],
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=scenario.background_color, roughness=0.9
                ),
            )
            face_cfg.func(
                f"/World/envs/env_{env_id}/{face_name}",
                face_cfg,
                translation=face_pos,
            )

        for light_id, light in enumerate(scenario.lights[1:4]):
            light_cfg = sim_utils.SphereLightCfg(
                intensity=float(light.intensity),
                color=light.color,
                radius=float(light.scale or 0.1),
                normalize=False,
            )
            light_world = tuple(float(v) for v in light.position)
            light_cfg.func(
                f"/World/envs/env_{env_id}/VectorLight_{light_id}",
                light_cfg,
                translation=light_world,
            )


def update_vector_appearance(env, scenarios: list[ScenarioSpec]) -> None:
    """Update per-env colors and local lights between synchronized batches."""
    stage = omni.usd.get_context().get_stage()
    for env_id, scenario in enumerate(scenarios):
        apply_display_color_to_subtree(
            f"/World/envs/env_{env_id}/Table/Visuals", scenario.table_color
        )
        apply_display_color_to_subtree(
            f"/World/envs/env_{env_id}/Tray", scenario.tray_color
        )
        apply_display_color_to_subtree(
            f"/World/envs/env_{env_id}/Background", scenario.background_color
        )
        apply_display_color_to_subtree(
            f"/World/envs/env_{env_id}/BackgroundFloor", scenario.background_color
        )
        for face_name in (
            "BackgroundFront", "BackgroundLeft",
            "BackgroundRight", "BackgroundCeiling",
        ):
            apply_display_color_to_subtree(
                f"/World/envs/env_{env_id}/{face_name}", scenario.background_color
            )
        for cube in [*scenario.blue_cubes, *scenario.red_cubes]:
            apply_display_color_to_subtree(cube.prim_path, cube.color)

        for light_id, light_spec in enumerate(scenario.lights[1:4]):
            prim = stage.GetPrimAtPath(
                f"/World/envs/env_{env_id}/VectorLight_{light_id}"
            )
            if not prim or not prim.IsValid():
                continue
            light = UsdLux.SphereLight(prim)
            light.GetIntensityAttr().Set(float(light_spec.intensity))
            light.GetColorAttr().Set(Gf.Vec3f(*[float(v) for v in light_spec.color]))
            world_pos = np.asarray(light_spec.position)
            xform = UsdGeom.Xformable(prim)
            translate_ops = [
                op for op in xform.GetOrderedXformOps()
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
            ]
            if translate_ops:
                translate_ops[0].Set(Gf.Vec3d(*[float(v) for v in world_pos]))


def set_vector_camera_poses(env, scenarios: list[ScenarioSpec]) -> None:
    if not needs_external_camera():
        return
    origins = env.scene.env_origins
    eyes = torch.tensor(
        [scenario.camera_eye for scenario in scenarios],
        device=env.device,
        dtype=torch.float32,
    ) + origins
    targets = torch.tensor(
        [scenario.camera_target for scenario in scenarios],
        device=env.device,
        dtype=torch.float32,
    ) + origins
    env.scene["record_camera"].set_world_poses_from_view(eyes, targets)


def write_vector_object_poses(
    env, scenarios: list[ScenarioSpec], template: ScenarioSpec
) -> None:
    """Write every active object pose as one batched tensor per asset."""
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    template_objects = [*template.blue_cubes, *template.red_cubes]
    for template_cube in template_objects:
        obj = env.scene[template_cube.asset_name]
        root_pose = torch.zeros((env.num_envs, 7), device=env.device, dtype=torch.float32)
        root_velocity = torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.float32)
        for env_id, scenario in enumerate(scenarios):
            active = {
                cube.asset_name: cube
                for cube in [*scenario.blue_cubes, *scenario.red_cubes]
            }.get(template_cube.asset_name)
            if active is None:
                local_pos = (0.0, 0.0, -1.5)
                yaw = 0.0
            else:
                local_pos = active.pos
                yaw = float(active.yaw)
            root_pose[env_id, :3] = env.scene.env_origins[env_id] + torch.tensor(
                local_pos, device=env.device, dtype=torch.float32
            )
            root_pose[env_id, 3] = 0.0
            root_pose[env_id, 4] = 0.0
            root_pose[env_id, 5] = math.sin(0.5 * yaw)
            root_pose[env_id, 6] = math.cos(0.5 * yaw)
        obj.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        obj.write_root_velocity_to_sim_index(
            root_velocity=root_velocity, env_ids=env_ids
        )
    env.sim.forward()


def run_vector_batch(
    env,
    scenarios: list[ScenarioSpec],
    template: ScenarioSpec,
    active_count: int,
) -> list[dict[str, Any]]:
    """Run one synchronized vector batch with independent env FSMs and recorders."""
    all_env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.reset(env_ids=all_env_ids)
    write_vector_object_poses(env, scenarios, template)
    update_vector_appearance(env, scenarios)
    set_vector_camera_poses(env, scenarios)

    open_action = torch.zeros(
        (env.num_envs, 7), device=env.device, dtype=torch.float32
    )
    open_action[:, 6] = float(args_cli.gripper_open_command)
    for _ in range(int(args_cli.settle_steps)):
        env.step(open_action)
    # Restore exact upright poses after settling and let contacts stabilize.
    write_vector_object_poses(env, scenarios, template)
    for _ in range(int(round(0.24 * float(args_cli.control_hz)))):
        env.step(open_action)
    move_to_randomized_start_poses(env, scenarios, active_count)
    env.sim.render()

    controllers = [
        AutoPickPlaceController(scenarios[env_id], env_id)
        for env_id in range(active_count)
    ]
    recorders = [
        EpisodeRecorder(scenarios[env_id], env_id)
        for env_id in range(active_count)
    ]
    for recorder, controller in zip(recorders, controllers):
        recorder.start_new_episode()
        controller.start(env)

    step_dt = float(getattr(env, "step_dt", env.cfg.sim.dt * env.cfg.decimation))
    actual_control_hz = 1.0 / step_dt
    capture_steps = int(round(actual_control_hz / float(args_cli.fps)))
    record_every = capture_steps * int(args_cli.capture_every_n)
    log_every = (
        int(args_cli.log_every_n)
        if args_cli.log_every_n > 0
        else (record_every if args_cli.record_sensors else 1)
    )
    print(
        f"[VECTOR] batch envs={active_count}/{env.num_envs}, "
        f"control={actual_control_hz:g}Hz, sensor={actual_control_hz / record_every:g}Hz"
    )

    finished: set[int] = set()
    results: list[dict[str, Any] | None] = [None] * active_count
    sim_step = 0
    try:
        while simulation_app.is_running() and len(finished) < active_count:
            action = open_action.clone()
            timestamp = sim_step * step_dt
            if sim_step % record_every == 0:
                update_wrist_follow_camera(env, force_recompute=True)
            for env_id, (controller, recorder) in enumerate(zip(controllers, recorders)):
                if env_id in finished:
                    continue
                controller.apply(env, action[env_id : env_id + 1])
                for event in controller.drain_events():
                    recorder.write_automation_event(sim_step, timestamp, event)
                if sim_step % log_every == 0:
                    recorder.write_action(
                        sim_step,
                        timestamp,
                        action[env_id : env_id + 1],
                        controller.status_payload(),
                    )
                    recorder.write_state(
                        env, sim_step, timestamp, controller.status_payload()
                    )
                if sim_step % record_every == 0:
                    recorder.write_frame(env, sim_step, timestamp)

            env.step(action)
            env.sim.render()
            sim_step += 1

            for env_id, (controller, recorder) in enumerate(zip(controllers, recorders)):
                if env_id in finished:
                    continue
                if controller.completed or controller.failed:
                    payload = controller.status_payload()
                    recorder.write_result(payload)
                    recorder.close_current(mark_disabled=True)
                    results[env_id] = payload
                    finished.add(env_id)
                    status = "DONE" if controller.completed else "FAILED"
                    print(
                        f"[VECTOR] {status} env={env_id} "
                        f"episode={scenarios[env_id].episode_index + 1}"
                    )
    finally:
        for env_id, (controller, recorder) in enumerate(zip(controllers, recorders)):
            if env_id not in finished:
                payload = controller.status_payload()
                payload["interrupted"] = True
                recorder.write_result(payload)
                recorder.close_current(mark_disabled=True)
                results[env_id] = payload

    return [result or {"failed": True, "state": "missing_result"} for result in results]


def run_vectorized_dataset() -> None:
    total_episodes = (
        int(args_cli.auto_generate_episodes)
        if args_cli.auto_generate_episodes
        else int(args_cli.num_envs)
    )
    first_episode = int(args_cli.episode_index)
    template = make_vector_template_scenario(first_episode)
    env = make_vector_env(template)
    asset_sizes = read_vector_asset_sizes(env, template)

    initial_scenarios: list[ScenarioSpec] = []
    for env_id in range(env.num_envs):
        scenario = generate_blue_tray_scenario(first_episode + env_id)
        initial_scenarios.append(
            adapt_vector_scenario(
                scenario, template, env_id, asset_sizes[env_id], None
            )
        )
    fixed_tray_positions = [scenario.tray_pos for scenario in initial_scenarios]
    spawn_vector_static_scene(env, initial_scenarios)
    env.sim.forward()
    env.sim.render()

    all_results: list[dict[str, Any]] = []
    try:
        for batch_start in range(0, total_episodes, env.num_envs):
            active_count = min(env.num_envs, total_episodes - batch_start)
            scenarios: list[ScenarioSpec] = []
            for env_id in range(env.num_envs):
                episode_index = first_episode + batch_start + env_id
                scenario = generate_blue_tray_scenario(
                    episode_index, fixed_tray_positions[env_id]
                )
                scenarios.append(
                    adapt_vector_scenario(
                        scenario,
                        template,
                        env_id,
                        asset_sizes[env_id],
                        fixed_tray_positions[env_id],
                    )
                )
            print(
                f"[VECTOR] starting batch {batch_start // env.num_envs + 1}; "
                f"episodes {first_episode + batch_start + 1}.."
                f"{first_episode + batch_start + active_count}"
            )
            batch_results = run_vector_batch(
                env, scenarios, template, active_count
            )
            for scenario, result in zip(scenarios[:active_count], batch_results):
                all_results.append(
                    {
                        "episode_index": scenario.episode_index,
                        "episode_number": scenario.episode_index + 1,
                        **result,
                    }
                )
            if not simulation_app.is_running():
                break
    finally:
        env.close()

    successes = sum(
        bool(result.get("completed")) and not bool(result.get("failed"))
        for result in all_results
    )
    summary = {
        "execution_mode": (
            "multi_gpu_worker_vectorized_envs"
            if os.environ.get("FRANKA_MULTI_GPU_RANK") is not None
            else "single_process_vectorized_envs"
        ),
        "multi_gpu_rank": (
            int(os.environ["FRANKA_MULTI_GPU_RANK"])
            if os.environ.get("FRANKA_MULTI_GPU_RANK") is not None
            else None
        ),
        "gpu_id": os.environ.get("FRANKA_MULTI_GPU_ID"),
        "visualizer": args_cli.visualizer,
        "num_envs": int(args_cli.num_envs),
        "requested_episodes": total_episodes,
        "reported_episodes": len(all_results),
        "successful_episodes": successes,
        "failed_episodes": len(all_results) - successes,
        "episodes": all_results,
    }
    rank = os.environ.get("FRANKA_MULTI_GPU_RANK")
    summary_name = (
        f"vectorized_summary_gpu_{int(rank):02d}.json"
        if rank is not None
        else "vectorized_summary.json"
    )
    summary_path = Path(args_cli.output_dir) / summary_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[VECTOR] finished success={successes}, "
        f"failed={len(all_results) - successes}; summary={summary_path}"
    )


def run_layout_validation_only() -> None:
    """Stress fixed-tray vector sampling without recording any episodes."""
    count = int(args_cli.validate_layouts_only)
    env_count = max(1, int(args_cli.num_envs))
    first_episode = int(args_cli.episode_index)
    initial = [
        generate_blue_tray_scenario(first_episode + env_id)
        for env_id in range(env_count)
    ]
    fixed_trays = [scenario.tray_pos for scenario in initial]
    blue_counts: dict[int, int] = {}
    progress_stage_counts: dict[str, int] = {}
    blue_positions: dict[int, list[tuple[float, float]]] = {}
    first_blue_positions: dict[int, list[tuple[float, float]]] = {}
    blue_strata: dict[int, list[int]] = {}
    first_blue_strata: dict[int, list[int]] = {}
    for offset in range(count):
        scenario = generate_blue_tray_scenario(
            first_episode + offset, fixed_trays[offset % env_count]
        )
        validate_scenario_spawn_clearance(scenario, margin=float(args_cli.min_spawn_spacing))
        blue_count = len(scenario.blue_cubes)
        blue_counts[blue_count] = blue_counts.get(blue_count, 0) + 1
        stage_key = f"{blue_count}_cube_{scenario.num_preplaced}_preplaced"
        progress_stage_counts[stage_key] = progress_stage_counts.get(stage_key, 0) + 1
        loose_cubes = [cube for cube in scenario.blue_cubes if not cube.preplaced]
        if not loose_cubes:
            raise RuntimeError(f"partial-progress scenario has no remaining target: {stage_key}")
        blue_positions.setdefault(blue_count, []).extend(
            (float(cube.pos[0]), float(cube.pos[1])) for cube in loose_cubes
        )
        first_blue_positions.setdefault(blue_count, []).append(
            (float(loose_cubes[0].pos[0]), float(loose_cubes[0].pos[1]))
        )
        blue_strata.setdefault(blue_count, []).extend(
            int(cube.position_stratum) for cube in loose_cubes
            if cube.position_stratum is not None
        )
        if loose_cubes[0].position_stratum is not None:
            first_blue_strata.setdefault(blue_count, []).append(
                int(loose_cubes[0].position_stratum)
            )

    x_bins, y_bins = (int(value) for value in args_cli.target_workspace_bins)
    x_edges = np.linspace(float(args_cli.workspace_x_min), float(args_cli.workspace_x_max), x_bins + 1)
    y_edges = np.linspace(float(args_cli.workspace_y_min), float(args_cli.workspace_y_max), y_bins + 1)

    def coverage_row(
        positions: list[tuple[float, float]], strata: list[int]
    ) -> dict[str, Any]:
        points = np.asarray(positions, dtype=np.float64)
        counts_2d, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=(x_edges, y_edges))
        flat_counts = counts_2d.astype(np.int64).reshape(-1)
        mean_count = float(flat_counts.mean())
        stratum_counts = np.bincount(strata, minlength=x_bins * y_bins).astype(np.int64)
        stratum_mean = float(stratum_counts.mean())
        return {
            "samples": int(len(points)),
            "occupied_cells": int(np.count_nonzero(flat_counts)),
            "total_cells": int(len(flat_counts)),
            "cell_count_cv": float(flat_counts.std() / mean_count) if mean_count else 0.0,
            "x_min": float(points[:, 0].min()),
            "x_max": float(points[:, 0].max()),
            "y_min": float(points[:, 1].min()),
            "y_max": float(points[:, 1].max()),
            "cell_counts": counts_2d.astype(np.int64).tolist(),
            "position_stratum_counts": stratum_counts.tolist(),
            "position_stratum_cv": (
                float(stratum_counts.std() / stratum_mean) if stratum_mean else 0.0
            ),
        }

    coverage = {
        blue_count: {
            "all_targets": coverage_row(
                blue_positions[blue_count], blue_strata.get(blue_count, [])
            ),
            "first_target": coverage_row(
                first_blue_positions[blue_count], first_blue_strata.get(blue_count, [])
            ),
        }
        for blue_count in sorted(blue_positions)
    }
    print(
        "[LAYOUT-VALIDATION] "
        + json.dumps(
            {
                "validated_layouts": count,
                "vector_env_slots": env_count,
                "fixed_tray_positions": fixed_trays,
                "blue_cube_counts": blue_counts,
                "progress_stage_counts": progress_stage_counts,
                "target_workspace_bins": [x_bins, y_bins],
                "blue_target_coverage": coverage,
                "tray_overlap_count": 0,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    if args_cli.validate_layouts_only > 0:
        run_layout_validation_only()
    else:
        run_vectorized_dataset()


if __name__ == "__main__":
    main()
    simulation_app.close()
