#!/usr/bin/env python3
"""Dual-Franka picture-hanging scene with a clothes-rack-style receiver rod."""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import random
import traceback
from pathlib import Path

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dual-Franka picture-hanging task with a round receiver rod.")
parser.add_argument("--panel-state", choices=("staging", "hung"), default="staging")
parser.add_argument("--max-steps", type=int, default=0, help="Zero keeps the viewer running.")
parser.add_argument("--physics-dt", type=float, default=1.0 / 120.0)
parser.add_argument("--camera-preview", action="store_true", help="Serve a three-camera browser preview.")
parser.add_argument("--camera-preview-port", type=int, default=8081)
parser.add_argument("--camera-preview-fps", type=float, default=10.0)
parser.add_argument(
    "--disable-task-cameras",
    action="store_true",
    help="Disable the three RTX task cameras for a smoother Viser-only motion review.",
)
parser.add_argument(
    "--validation-side-camera",
    action="store_true",
    help="Add a temporary hanger side camera for geometry validation; off for normal three-view episodes.",
)

parser.add_argument("--substeps", type=int, default=2)
parser.add_argument(
    "--scene-seed",
    type=int,
    default=None,
    help="Appearance seed; omitted means a new sample when an episode process starts.",
)
parser.add_argument("--record-dir", type=Path, default=None, help="Async RGB output directory.")
parser.add_argument("--capture-every", type=int, default=8, help="Capture every N physics steps (8 = 15 FPS at 120 Hz).")
parser.add_argument("--writer-queue-size", type=int, default=128)
parser.add_argument("--record-instance-segmentation", action="store_true", help="Also save raw instance ID maps.")
parser.add_argument("--auto-ops", action="store_true", help="Run one scripted pick-to-hang episode.")
parser.add_argument(
    "--auto-ops-task",
    choices=("long", "pull", "lift", "hang"),
    default="long",
    help="Auto Ops scope: full chain or one standalone pull/lift/hang task.",
)
parser.add_argument(
    "--hold-after-auto-ops",
    action="store_true",
    help="Keep the viewer open at the final pose after Auto Ops finishes.",
)
parser.add_argument("--motion-speed-scale", type=float, default=6.0, help="Auto Ops trajectory speed multiplier. Most phases are convergence-gated rather than duration-gated, so the arm rate limits in auto_ops_controller matter more than this.")
parser.add_argument("--episode-id", type=int, default=0)
parser.add_argument("--disable-cuda-graph", action="store_true", help="Start faster at lower runtime performance.")
parser.add_argument("--newton-debug", action="store_true", help="Log Newton solver convergence diagnostics.")
parser.add_argument(
    "--save-newton-mjcf",
    type=Path,
    default=None,
    help="Write Newton's generated MJCF for backend/contact-setting verification.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Recording is explicit: --auto-ops alone is a no-write motion test.
if args_cli.auto_ops and args_cli.max_steps == 0 and not args_cli.hold_after_auto_ops:
    args_cli.max_steps = 24000
args_cli.enable_cameras = not args_cli.disable_task_cameras

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402
try:  # Semantics is loaded under the newer USD namespace without camera extensions.
    from pxr import Semantics  # noqa: E402
except ImportError:  # pragma: no cover - depends on the Isaac Sim extension set
    from pxr import UsdSemantics as Semantics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg  # noqa: E402
from isaaclab_assets import FRANKA_PANDA_CFG  # noqa: E402
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg  # noqa: E402
from isaaclab_newton.sim.schemas import NewtonMaterialPropertiesCfg  # noqa: E402
from auto_ops_controller import (
    GRIPPER_MAX_WIDTH, PANEL_HALF_LENGTH, PANEL_PULL_DISTANCE,
    RACK_COVER_FRONT_X, RACK_ROTATION_CLEARANCE_X, DualFrankaAutoOps,
)  # noqa: E402
from async_sensor_writer import AsyncSensorWriter  # noqa: E402
from camera_preview_server import CameraPreviewServer  # noqa: E402

PANEL_ROOT = "/World/Panel"
TASK_CAMERA_EYE = (-2.2, 2.4, 1.8)
TASK_CAMERA_TARGET = (0.55, 0.0, 0.85)

ROBOT_BASE_X = -0.40
ROBOT_BASE_Y = 0.60
TABLE_SIZE = (1.55, 1.36, 0.42)
TABLE_POSITION = (0.10, 0.0, 0.21)
# Staged at the back of the rack: the panel's rear edge is flush with the rail rear
# at x = 0.86, so it spans [0.32, 0.86] and the pull is a real transport.
# The bare board now rests straight on the rails; the ring hangs between them at y = 0.
PANEL_STAGING_POSITION = (0.42, 0.0, 0.550)
# The rails must still carry the panel after the forward pull, so they reach
# further toward the robots than the panel ever travels.
# The rails run from x = 0.38 to the rack rear at 0.86, so the staged panel's near
# 60 mm overhangs them.  The panel rests on the rack through the underside of its
# frame, which sits exactly at rail-top height, so a rail running beneath the near
# edge leaves the vertical pincer nowhere to put its lower jaw - it can only scrape
# the rim and shove the panel away, which is exactly what runs 933 and 934 did.  The
# overhang puts the 35 mm front rim in free space with the fingertips stopping 26 mm
# short of the rail front.
RACK_RAIL_POSITION_X = 0.305
RACK_RAIL_LENGTH = 1.15
HANGER_ROD_X = 0.28
HANGER_ROD_Z = 1.25
HANGER_ROD_LENGTH = 1.58
HANGER_POST_Y = 0.76
HANGER_POST_THICKNESS = 0.055
HANGER_POST_HEIGHT = 1.20

# Passive S-hook receiver, in the hook body frame anchored at
# (HANGER_ROD_X, y, HANGER_ROD_Z).  The last two profile points form the
# horizontal catch bar the panel rear rail rests on.
S_HOOK_YAW_DEGREES = 30.0
S_HOOK_BAR_RADIUS = 0.012
S_HOOK_SHELF_LOCAL_Z = -0.205
S_HOOK_SHELF_LOCAL_X = (-0.030, -0.088)

PANEL_BOARD_THICKNESS = 0.020
# Panel rear hanging rail, panel-local, measured in the hung (vertical) pose.  All
# four frame members share a depth so the panel sits flat on the rails now that the
# frame is underneath, and the hanging rail keeps full seating area on the hook.
PANEL_RAIL_DEPTH = 0.035
PANEL_RAIL_BOTTOM_Z = 0.180
# Staging rolls the panel 180 deg about Y so the steel frame hangs *below* the board
# and the smooth face is up.  The front frame member then doubles as the lip the left
# hand hooks behind.  The hung pose stays +90 deg about Y, which makes staging->hung a
# -90 deg roll and still lands the hanging rail rear-and-up.
PANEL_STAGING_ORIENTATION = (1.0, 0.0, 0.0, 0.0)
# How far the frame hangs below the board.  This sets the height of the cavity under
# the panel, and therefore whether the hand can get inside it and catch the front rim
# from behind.  At 35 mm it could not: the wrist stands about 100 mm tall in the
# vertical-pincer pose, so the jaws could only press the rim's faces and the pull rode
# on friction.  At 130 mm the hand goes in, and the pull becomes a push on steel.
PANEL_FRAME_HEIGHT = 0.130
# Bent-steel ring at the centre of the board's rear face.  Its standoff is the gap a
# fingertip slides into for the pull, and its bore is what drops over the nail.
# On the board's upper face, near its top edge, where a picture's hanger goes - not
# on the underside, and not at the centre.  Its plane stays parallel to the floor
# while the panel lies on the rack, so a fingertip enters the standoff gap from
# above and the stroke bears on the ring's inner face instead of on friction.  On
# the nail the panel has to hang *below* the ring; a ring on the centreline puts the
# nail at the centre of mass and the board just swings around it.
RING_CENTRE_Z = 0.024
RING_RADIUS = 0.105
RING_BAR_RADIUS = 0.014
# A picture hanger lies almost flat on the back of the frame - just enough clearance
# for the nail's shank.  35 mm was tried so a fingertip could enter the gap and haul
# on the ring, and it read as a lid handle standing off the board rather than a hanger.
RING_STANDOFF = 0.008
HANDLE_LOCAL_X = -0.285
HANDLE_LOCAL_Z = 0.024
RING_SEGMENTS = 10
PANEL_FRAME_DROP = 0.5 * PANEL_BOARD_THICKNESS + PANEL_FRAME_HEIGHT

_S_HOOK_YAW_SCALE = math.cos(math.radians(S_HOOK_YAW_DEGREES))
# Two nails driven straight into the rack rod, pointing at the robots.  The panel's
# top rail drops over them exactly like a picture frame over a pair of wall nails.
NAIL_RADIUS = 0.009
NAIL_LENGTH = 0.090
NAIL_Y = 0.15
NAIL_HEAD_X = HANGER_ROD_X
HANGER_SHELF_FRONT_X = HANGER_ROD_X - NAIL_LENGTH
# The rail rests on top of the nail shaft.
HANGER_SHELF_TOP_Z = HANGER_ROD_Z + NAIL_RADIUS
# Seat the rear rail on the catch bar, keeping the board face clear of the bar's
# front end.  Everything downstream (auto ops targets, success test) uses this.
PANEL_HUNG_POSITION = (
    round(HANGER_ROD_X - HANDLE_LOCAL_Z, 4),
    0.0,
    round(HANGER_ROD_Z - (abs(HANDLE_LOCAL_X) + RING_RADIUS), 4),
)
# Only has to lift the rear rail clear of the catch bar.  A taller entry would
# push the panel top into the rod now that the hook is shorter.
PANEL_HANG_ENTRY_Z = round(PANEL_HUNG_POSITION[2] + 0.050, 4)
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 256
CAMERA_UPDATE_PERIOD = 1.0 / 15.0
# Room views.  HANGER_* is the eye-level front view; TOP_* looks down on the table
# from 2.3 m, standing 350 mm back of the table centre and aimed 200 mm in front of
# it - a 16 degree tilt off vertical.  Both rotations follow the same convention:
# the columns of the matrix are the camera's right, down and forward axes in world.
HANGER_CAMERA_EYE = (-1.35, 0.0, 0.95)
HANGER_CAMERA_ROT_ROS = (-0.5, 0.5, -0.5, 0.5)
TOP_CAMERA_EYE = (-0.25, 0.0, 2.30)
# Authored (x, y, z, w).  ``CameraCfg.OffsetCfg.rot`` is xyzw in this build, the same
# order ``RigidObject`` root poses use, while ``isaaclab.utils.math`` stays wxyz.  The
# front view above cannot show this - (-0.5, 0.5, -0.5, 0.5) is the same rotation read
# either way, up to sign - so it validated nothing, and three wxyz-authored overhead
# rotations in a row rendered the ceiling.  As a rotation matrix the columns are the
# camera's right, down and forward axes in world; forward here is (0.281, 0, -0.960).
TOP_CAMERA_ROT_ROS = (-0.699948, 0.699948, -0.100362, 0.100363)
HANGER_SIDE_CAMERA_EYE = (1.15, -1.65, 1.02)
HANGER_SIDE_CAMERA_ROT_ROS = (0.7071068, -0.7071068, 0.0, 0.0)
# Wrist camera: beside the palm, 20 mm ahead of it, tilted 25 deg in at the TCP.
# Mounted on the flat face of ``panda_hand``, clear of its shell.  The measured
# envelope (printed by _report_hand_envelope) is hand z in [-0.026, 0.066] and
# fingers z in [0.0585, 0.1124] with |x| <= 0.0264, so the old (0.050, 0, 0.020)
# mount sat *inside* the shell and rendered nothing but the hand.  At z = 0.075 the
# camera is past the shell entirely and 48 mm clear of the fingers in x.
# 75 mm off the flat face, centred between the jaws, 20 mm forward.  At the
# front grasp this puts the lens at world (0.262, 0.075, 0.592) - 58 mm ahead of the
# panel edge and just above the board - where (0.075, 0, 0.075) sat 3 mm off the
# rim's steel face and rendered black.
WRIST_CAMERA_POS = (0.075, 0.0, 0.020)
# Optical axis (-0.6318, 0, 0.7750) in the hand frame, aimed exactly at the jaw
# centre in the fingertip plane so the pincer sits in the middle of the frame rather
# than off in a corner.  Image down stays on the hand's -Y, which both work poses map
# to world -Z, so the view is upright through the whole episode.
WRIST_CAMERA_ROT_ROS = (0.0, -0.335330, 0.0, 0.942075)
LOW_SLIDE_MATERIAL_PATH = "/World/Materials/LowSlide"
HIGH_HANG_MATERIAL_PATH = "/World/Materials/HighHangGrip"
GRASP_PAD_NAME = "NewtonGraspPad"
# Convex collision proxy matching the physical Panda fingertip body.
GRASP_PAD_SIZE = (0.030, 0.008, 0.050)
GRASP_PAD_LOCAL_POSITION = (0.0, 0.0, 0.025)
# The panel skid still slides under the pull (needs ~4.7 N against a ~24 N grip),
# but no longer squirts across the rack from an incidental finger touch the way a
# near-frictionless 0.08 did.
# Steel frame on smooth steel rails.  0.10 was tried and is closer to PTFE than to
# any rack anyone builds; at a realistic grip force the pull does not need it.
# The first 0.18/0.12 setting let the 0.8 kg panel travel more than 20 cm
# while the gripper was still almost fully open.  These values resist the
# initial contact impulse but remain low enough for a Panda to pull the panel.
LOW_SLIDE_STATIC_FRICTION = 0.45
LOW_SLIDE_DYNAMIC_FRICTION = 0.30
# Rubber-faced pads on a painted board.  This was briefly run at 2.5, which is not a
# coefficient any real pair of surfaces has; it was compensating for a grip force
# modelled ten times too weak.  Fix the grip instead.
HIGH_HANG_STATIC_FRICTION = 1.20
HIGH_HANG_DYNAMIC_FRICTION = 1.00

ROOM_X_MIN = -2.90
ROOM_X_MAX = 1.90
ROOM_Y_MIN = -2.00
ROOM_Y_MAX = 2.00
ROOM_HEIGHT = 2.70


def rigid_object_quat(quaternion_wxyz):
    """Reorder a (w, x, y, z) quaternion for the Newton ``RigidObject`` root-pose API.

    Isaac Lab 3.0.0-beta2's Newton backend reads and writes ``RigidObject`` root
    quaternions as (x, y, z, w), while ``Articulation`` poses and every helper in
    ``isaaclab.utils.math`` stay (w, x, y, z).  Passing a wxyz identity here means
    ``(1, 0, 0, 0)`` is applied as a 180 degree roll about X, which silently flips
    the panel so its rear frame ends up *under* the board.  Author every quaternion
    wxyz and convert only at this boundary; ``validate_panel_rest_pose`` fails loudly
    if a future release changes the order back.
    """
    w, x, y, z = quaternion_wxyz
    return (x, y, z, w)


def validate_panel_rest_pose(panel):
    """Fail fast if the panel does not settle where the authored geometry says."""
    position = panel.data.root_pos_w.torch[0]
    height = float(position[2].item())
    expected = PANEL_STAGING_POSITION[2]
    if abs(height - expected) > 0.015:
        raise RuntimeError(
            f"Panel settled at z={height:.4f} but its geometry rests at z={expected:.4f}. "
            "The Newton RigidObject root-quaternion order likely changed; revisit rigid_object_quat()."
        )
    print(f"[INFO] panel_rest_pose_verified z={height:.4f} expected={expected:.4f}", flush=True)


def spawn_static_box(path, size, position, color, collision=True, physics_material_path=None):
    cfg = sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg() if collision else None,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.65),
    )
    cfg.func(path, cfg, translation=position)
    if physics_material_path is not None:
        sim_utils.bind_physics_material(f"{path}/geometry/mesh", physics_material_path)


def spawn_static_cylinder(path, radius, height, position, color, physics_material_path=None):
    cfg = sim_utils.CylinderCfg(
        radius=radius,
        height=height,
        axis="Y",
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.45),
    )
    cfg.func(path, cfg, translation=position)
    if physics_material_path is not None:
        sim_utils.bind_physics_material(f"{path}/geometry/mesh", physics_material_path)


def spawn_task_physics_materials():
    """Create Newton-compatible USD materials for sliding and hanging contacts."""
    low_slide = RigidBodyMaterialBaseCfg(
        static_friction=LOW_SLIDE_STATIC_FRICTION,
        dynamic_friction=LOW_SLIDE_DYNAMIC_FRICTION,
        restitution=0.0,
    )
    low_slide.func(LOW_SLIDE_MATERIAL_PATH, low_slide)
    high_hang = NewtonMaterialPropertiesCfg(
        static_friction=HIGH_HANG_STATIC_FRICTION,
        dynamic_friction=HIGH_HANG_DYNAMIC_FRICTION,
        restitution=0.0,
        torsional_friction=0.02,
        rolling_friction=0.002,
    )
    high_hang.func(HIGH_HANG_MATERIAL_PATH, high_hang)



def configure_newton_grasp_contacts():
    """De-instance the arms so their finger geometry can be inspected and materialled.

    Nothing is added to the gripper.  Thin high-friction pads were bolted to the
    fingers for a long stretch of this work on the belief that the stock colliders
    produced no contact; they do.  Each finger's mesh carries a convexHull collider
    with collisionEnabled true, and the recorded contact table shows finger-on-finger
    pairs.  What actually made the jaws pass through the panel was the close command
    covering the full 80 mm stroke inside one control tick, and a contact soft enough
    to be penetrated - both fixed elsewhere.

    The Franka arrives as USD instance proxies, which cannot be authored into, so the
    two arms are de-instanced here.  That step occasionally corrupts the Newton build
    on startup; an episode that dies before ``panel_rest_pose_verified`` should simply
    be relaunched.
    """
    stage = sim_utils.get_current_stage()
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
        if str(prim.GetPath()).startswith("/World/Franka") and prim.IsInstanceable():
            prim.SetInstanceable(False)

    finger_links = [
        stage.GetPrimAtPath(f"/World/Franka{side}/panda_{finger}finger")
        for side in ("Left", "Right")
        for finger in ("left", "right")
    ]
    missing = [index for index, prim in enumerate(finger_links) if not (prim and prim.IsValid())]
    if missing:
        raise RuntimeError(f"Panda finger links missing at indices {missing}")

    # Opposing fingers of one Panda must close through their own contact margins.
    # Filter only same-hand links; finger-to-panel contacts remain enabled.
    filtered_pairs = []
    for side in ("Left", "Right"):
        left_finger = stage.GetPrimAtPath(f"/World/Franka{side}/panda_leftfinger")
        right_finger = stage.GetPrimAtPath(f"/World/Franka{side}/panda_rightfinger")
        pair_api = UsdPhysics.FilteredPairsAPI.Apply(left_finger)
        pair_api.CreateFilteredPairsRel().AddTarget(right_finger.GetPath())
        filtered_pairs.append((str(left_finger.GetPath()), str(right_finger.GetPath())))

    print(
        f"[INFO] franka_fingers_verified count={len(finger_links)} "
        f"stock_colliders=enabled "
        f"self_collision_filtered={filtered_pairs} "
    )
    _report_hand_envelope(stage)
    _report_finger_colliders(stage)


def _report_finger_colliders(stage):
    """Dump what the stock finger geometry actually declares to the physics engine.

    The pads exist only because these prims never produce a contact against the panel.
    If the cause is a missing or disabled collision schema it can be repaired in place,
    and then nothing has to be bolted onto the gripper at all.
    """
    for label in ("panda_leftfinger", "panda_rightfinger", "panda_hand"):
        root = stage.GetPrimAtPath(f"/World/FrankaLeft/{label}")
        if not root.IsValid():
            continue
        for prim in Usd.PrimRange(root):
            if prim.GetName().startswith(GRASP_PAD_NAME):
                continue
            has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
            if not (has_collision or prim.IsA(UsdGeom.Gprim)):
                continue
            enabled = None
            if has_collision:
                attribute = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
                enabled = attribute.Get() if attribute else None
            approximation = None
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                attribute = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr()
                approximation = attribute.Get() if attribute else None
            print(
                f"[INFO] finger_collider {prim.GetPath()} type={prim.GetTypeName()} "
                f"collisionAPI={has_collision} enabled={enabled} approximation={approximation}",
                flush=True,
            )


def _report_hand_envelope(stage):
    """Print the hand and finger extents in the ``panda_hand`` frame.

    The wrist camera mounts on ``panda_hand``, so where it has to sit to see past
    the shell is a question about these numbers, not a guess.  Printed once per
    episode so a camera move can be checked against measured geometry.
    """
    root = "/World/FrankaLeft"
    hand = stage.GetPrimAtPath(f"{root}/panda_hand")
    if not hand.IsValid():
        return
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    for label in ("panda_hand", "panda_leftfinger", "panda_rightfinger"):
        prim = stage.GetPrimAtPath(f"{root}/{label}")
        if not prim.IsValid():
            continue
        # ComputeLocalBound is already expressed in the prim's own frame, so no corner
        # transforms are needed.  Transforming only the min and max of a world-aligned
        # box, as this did before, collapses axes and reported a 268 mm wide hand.
        box = cache.ComputeLocalBound(prim).ComputeAlignedRange()
        low, high = box.GetMin(), box.GetMax()
        span = [(round(low[i], 4), round(high[i], 4)) for i in range(3)]
        print(f"[INFO] hand_envelope {label} x={span[0]} y={span[1]} z={span[2]}", flush=True)


def apply_semantic_label(prim_path, label):
    """Assign one semantic instance label to a complete scene subtree."""
    if not hasattr(Semantics, "SemanticsAPI"):
        return
    stage = sim_utils.get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Cannot label missing prim: {prim_path}")
    semantics = Semantics.SemanticsAPI.Apply(prim, "Semantics")
    semantics.CreateSemanticTypeAttr().Set("class")
    semantics.CreateSemanticDataAttr().Set(label)


def sample_episode_appearance():
    """Sample once at episode-process startup and keep the values fixed."""
    seed = args_cli.scene_seed
    if seed is None:
        seed = random.SystemRandom().randrange(0, 2**31)
    rng = random.Random(seed)

    def sample_color(saturation_range, value_range):
        return colorsys.hsv_to_rgb(
            rng.random(),
            rng.uniform(*saturation_range),
            rng.uniform(*value_range),
        )

    light_palette = ((1.0, 0.90, 0.78), (1.0, 0.98, 0.92), (0.82, 0.90, 1.0))
    lights = []
    for _ in range(rng.randint(2, 5)):
        base_color = rng.choice(light_palette)
        lights.append(
            {
                "position": (
                    rng.uniform(-1.20, 1.20),
                    rng.uniform(-1.45, 1.45),
                    rng.uniform(1.85, 2.45),
                ),
                "intensity": rng.uniform(8000.0, 28000.0),
                "radius": rng.uniform(0.12, 0.28),
                "color": tuple(
                    min(1.0, max(0.65, channel + rng.uniform(-0.04, 0.04)))
                    for channel in base_color
                ),
            }
        )
    return {
        "seed": seed,
        "wallpaper_color": sample_color((0.20, 0.75), (0.38, 0.82)),
        "panel_color": sample_color((0.45, 0.92), (0.45, 0.92)),
        "lights": lights,
        # Non-random safety fill prevents unusably dark observations.
        "ambient_intensity": 750.0,
    }



def spawn_room(appearance):
    """Enclose the task in a fixed room with episode-random wallpaper."""
    x_center = 0.5 * (ROOM_X_MIN + ROOM_X_MAX)
    x_size = ROOM_X_MAX - ROOM_X_MIN
    y_center = 0.5 * (ROOM_Y_MIN + ROOM_Y_MAX)
    y_size = ROOM_Y_MAX - ROOM_Y_MIN
    wall = appearance["wallpaper_color"]
    ceiling = tuple(min(1.0, channel * 1.18 + 0.06) for channel in wall)
    floor = tuple(max(0.10, channel * 0.42) for channel in wall)

    spawn_static_box("/World/Room/Floor", (x_size, y_size, 0.05), (x_center, y_center, -0.025), floor)
    for name, size, position in (
        ("FrontWall", (0.06, y_size, ROOM_HEIGHT), (ROOM_X_MIN, y_center, 0.5 * ROOM_HEIGHT)),
        ("BackWall", (0.06, y_size, ROOM_HEIGHT), (ROOM_X_MAX, y_center, 0.5 * ROOM_HEIGHT)),
        ("LeftWall", (x_size, 0.06, ROOM_HEIGHT), (x_center, ROOM_Y_MIN, 0.5 * ROOM_HEIGHT)),
        ("RightWall", (x_size, 0.06, ROOM_HEIGHT), (x_center, ROOM_Y_MAX, 0.5 * ROOM_HEIGHT)),
    ):
        spawn_static_box(f"/World/Room/{name}", size, position, wall, collision=False)


def spawn_episode_lights(appearance):
    """Create fixed-for-episode local lights plus a minimum ambient fill."""
    dome_cfg = sim_utils.DomeLightCfg(
        intensity=appearance["ambient_intensity"],
        color=(0.86, 0.88, 0.92),
        visible_in_primary_ray=False,
    )
    dome_cfg.func("/World/EpisodeLights/AmbientSafetyFill", dome_cfg)
    for index, light in enumerate(appearance["lights"]):
        light_cfg = sim_utils.SphereLightCfg(
            intensity=light["intensity"],
            color=light["color"],
            radius=light["radius"],
            normalize=False,
        )
        light_cfg.func(f"/World/EpisodeLights/Local{index}", light_cfg, translation=light["position"])


def set_task_camera(sim):
    """Set the camera for both Kit and standalone visualizers."""
    sim.set_camera_view(eye=TASK_CAMERA_EYE, target=TASK_CAMERA_TARGET)

    if "kit" in (args_cli.visualizer or []):
        from omni.kit.viewport.utility import get_active_viewport

        stage = sim_utils.get_current_stage()
        camera_path = "/OmniverseKit_Persp"
        if not stage.GetPrimAtPath(camera_path).IsValid():
            UsdGeom.Camera.Define(stage, camera_path)
        viewport = get_active_viewport()
        if viewport is not None:
            viewport.set_active_camera(camera_path)




def add_compound_cube(stage, path, size, local_position, color, physics_material_path=None, rotation_y_degrees=0.0, collision=True):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*local_position))
    if rotation_y_degrees:
        cube.AddRotateYOp().Set(rotation_y_degrees)
    cube.AddScaleOp().Set(Gf.Vec3d(*size))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        if physics_material_path is not None:
            sim_utils.bind_physics_material(path, physics_material_path, stage=stage)


def add_compound_cylinder_between(
    stage, path, start, end, radius, color, physics_material_path=None, collision=True
):
    """Create one round rod segment between two points in a rigid-body local frame."""
    start_v = Gf.Vec3d(*start)
    end_v = Gf.Vec3d(*end)
    delta = end_v - start_v
    length = delta.GetLength()
    if length <= 1.0e-6:
        raise ValueError(f"Degenerate hook segment: {path}")
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateAxisAttr("Z")
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(length)
    cylinder.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    transform = Gf.Matrix4d(1.0)
    transform.SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), delta / length))
    transform.SetTranslate(0.5 * (start_v + end_v))
    cylinder.AddTransformOp().Set(transform)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
        if physics_material_path is not None:
            sim_utils.bind_physics_material(path, physics_material_path, stage=stage)


def add_horizontal_panel_cube(stage, path, size, local_position, color, physics_material_path=None):
    """Bake a -90 degree Y rotation into an axis-aligned compound cube."""
    horizontal_size = (size[2], size[1], size[0])
    horizontal_position = (-local_position[2], local_position[1], local_position[0])
    add_compound_cube(stage, path, horizontal_size, horizontal_position, color, physics_material_path)


def spawn_panel(panel_state, panel_color):
    stage = sim_utils.get_current_stage()
    if panel_state == "hung":
        root_position = PANEL_HUNG_POSITION
        # Geometry is authored horizontal, so +90 deg about Y makes it vertical.
        root_orientation = (0.7071068, 0.0, 0.7071068, 0.0)
    else:
        # The panel lies face-down across two narrow rack rails. Its rear frame
        # faces upward and the 75 mm clearance below the board admits a hand.
        root_position = PANEL_STAGING_POSITION
        root_orientation = PANEL_STAGING_ORIENTATION
    root = UsdGeom.Xform.Define(stage, PANEL_ROOT)
    root.AddTranslateOp().Set(Gf.Vec3d(*root_position))
    root_prim = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    # 0.5 kg for the whole panel including the steel edge frame.  The pull then only
    # has to overcome ~6 N of rail friction and the bimanual lift ~5 N across four
    # pads, well clear of where this contact solver starts letting the pads creep.
    # 300 g including the steel frame.  The pull has to overcome the panel sliding on
    # its rails, and that resistance scales directly with mass.
    mass_api.CreateMassAttr().Set(0.8)
    # Keep Newton rigid-body coordinates at the panel geometric center.
    # Auto-COM for this asymmetric compound frame shifts root_pos_w by ~76 mm.
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    # X thickness, Y width and Z height. Positive local X is the rear face.
    # Every grasped surface carries the high-friction material: the fingers
    # clamp the board plus a rear frame member in both the front and side grasps.
    add_horizontal_panel_cube(
        stage, f"{PANEL_ROOT}/Board", (PANEL_BOARD_THICKNESS, 0.86, 0.54), (0.0, 0.0, 0.0), panel_color, HIGH_HANG_MATERIAL_PATH
    )

    # A thin steel-tube frame follows the rear perimeter. It stays hidden from
    # the frontal task camera, leaving the visible panel face flat.  The top
    # member is the deeper load-bearing hanging rail.
    steel = (0.26, 0.28, 0.31)
    _face = 0.5 * PANEL_BOARD_THICKNESS
    # One bent-steel ring at the centre of the board's rear face, and nothing else.
    # Every edge member is gone.  The ring does both jobs: lying flat on the rack its
    # plane is horizontal and a finger slides into the gap beneath it, so the pull is
    # carried by the ring's inner face rather than by friction on a clamped edge; stood
    # vertical for the hang, that same plane turns upright and drops over the nail.
    # A clamped edge could never do the first - see PANEL_FRAME_HEIGHT's history.
    ring_points = []
    for index in range(RING_SEGMENTS + 1):
        angle = -0.5 * math.pi + math.pi * index / RING_SEGMENTS
        ring_points.append(
            (
                HANDLE_LOCAL_X - RING_RADIUS * math.cos(angle),
                -RING_RADIUS * math.sin(angle),
                RING_CENTRE_Z,
            )
        )
    for index, (start_point, end_point) in enumerate(zip(ring_points[:-1], ring_points[1:])):
        add_compound_cylinder_between(
            stage,
            f"{PANEL_ROOT}/Ring/Arc{index}",
            start_point,
            end_point,
            radius=RING_BAR_RADIUS,
            color=steel,
            physics_material_path=HIGH_HANG_MATERIAL_PATH,
        )
    for index, sign in enumerate((-1.0, 1.0)):
        add_compound_cylinder_between(
            stage,
            f"{PANEL_ROOT}/Ring/Stem{index}",
             (HANDLE_LOCAL_X - RING_RADIUS * math.cos(angle), sign * RING_RADIUS, 0.010),
            (HANDLE_LOCAL_X - RING_RADIUS * math.cos(angle), sign * RING_RADIUS, RING_CENTRE_Z),
            radius=RING_BAR_RADIUS,
            color=steel,
            physics_material_path=HIGH_HANG_MATERIAL_PATH,
        )
    for name, size, position in ():
        add_horizontal_panel_cube(stage, f"{PANEL_ROOT}/RearFrame/{name}", size, position, steel, HIGH_HANG_MATERIAL_PATH)

    # The continuous Top member is the load-bearing hanging rail. In the hung
    # pose it settles directly onto the round hanger rod.

    sim_utils.update_stage()
    return RigidObject(
        cfg=RigidObjectCfg(
            prim_path=PANEL_ROOT,
            spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=root_position, rot=rigid_object_quat(root_orientation)
            ),
        )
    )


def validate_layout():
    """Reject layouts that violate reach and hanger-clearance requirements."""
    table_half_width = 0.5 * TABLE_SIZE[1]
    hanger_inner_half_width = HANGER_POST_Y - 0.5 * HANGER_POST_THICKNESS
    side_clearance = hanger_inner_half_width - table_half_width
    if side_clearance < 0.05:
        raise ValueError(f"Hanger posts need at least 5 cm table clearance; got {side_clearance:.3f} m")

    table_x_min = TABLE_POSITION[0] - 0.5 * TABLE_SIZE[0]
    table_x_max = TABLE_POSITION[0] + 0.5 * TABLE_SIZE[0]
    if not table_x_min < HANGER_ROD_X < table_x_max:
        raise ValueError("The table must extend through the hanger frame in X.")

    panel_near_edge_x = PANEL_STAGING_POSITION[0] - 0.27
    initial_reach = panel_near_edge_x - ROBOT_BASE_X
    if not 0.20 <= initial_reach <= 0.75:
        raise ValueError(f"Panel near-edge reach should be 0.20-0.75 m; got {initial_reach:.3f} m")

    # The rails must still carry the panel once the left arm has pulled it.
    rail_front_x = RACK_RAIL_POSITION_X - 0.5 * RACK_RAIL_LENGTH
    pulled_panel_x = PANEL_STAGING_POSITION[0] - PANEL_PULL_DISTANCE
    if pulled_panel_x - rail_front_x < 0.05:
        raise ValueError(
            f"The pulled panel centre of mass leaves the rack rails: {pulled_panel_x:.3f} vs rail front {rail_front_x:.3f}"
        )

    # The rotated nail is vertical in Z.  The handle centreline must be
    # coincident with the nail axis in the hung pose; the loop then drops over
    # the shaft instead of relying on a fictitious horizontal shelf.
    handle_axis_x = PANEL_HUNG_POSITION[0] + HANDLE_LOCAL_Z
    if abs(handle_axis_x - HANGER_ROD_X) > 0.02:
        raise ValueError(
            f"Hung handle axis {handle_axis_x:.4f} is not aligned with vertical nail "
            f"axis {HANGER_ROD_X:.4f}"
        )

    # The front grasp reaches into the cavity under the board rather than clamping the
    # edge, so what has to fit is the wrist, not the stack.  The cavity must clear the
    # hand's ~100 mm span in the vertical-pincer pose with room for the jaws to open.
    if PANEL_FRAME_HEIGHT < 0.115:
        raise ValueError(
            f"The under-panel cavity is {PANEL_FRAME_HEIGHT:.3f} m; the wrist needs at least 0.115 m"
        )


def harden_contacts():
    """Raise the MuJoCo contact impedance so the jaws cannot sink into the panel.

    Steel does not yield to a gripper, but the default ``solimp`` of (0.9, 0.95, 0.001)
    lets the solver trade penetration for constraint force: closing on the 43 mm
    board+rim stack settled the jaws 13 mm inside it, and once they are inside, the
    stroke presses the panel down instead of pulling it along.  Every earlier attempt
    here worked around that - bounding the close command, bolting a catch onto the
    pad, lifting the panel off its rails - when the fix is simply to make the contact
    behave like the rigid contact it represents.

    Only the impedance is touched.  ``solref``'s time constant has to stay above twice
    the solver step or the contact rings, and Isaac Lab exposes neither field, so this
    reaches into the MuJoCo Warp model after the solver is built.
    """
    try:
        from isaaclab_newton.physics.newton_manager import NewtonManager

        model = NewtonManager._solver.mjw_model
    except (ImportError, AttributeError) as error:
        print(f"[WARN] harden_contacts skipped: {error}", flush=True)
        return
    try:
        import warp as wp

        solimp = model.geom_solimp.numpy()
        # Keep the global scene moderately firm, then harden only the thin panel and
        # Panda finger contacts so the board cannot pass through a closing jaw.
        solimp[..., 0] = 0.95
        solimp[..., 1] = 0.99
        solimp[..., 2] = 0.002
        import mujoco
        solver = NewtonManager._solver
        geom_solimp = solimp[0] if solimp.ndim == 3 else solimp
        support_geom_ids = []
        finger_geom_ids = []
        panel_geom_ids = []
        for geom_index in range(int(solver.mj_model.ngeom)):
            geom_name = mujoco.mj_id2name(
                solver.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_index
            ) or ""
            if "PanelRack" in geom_name:
                geom_solimp[geom_index, :3] = (0.995, 0.999, 0.0005)
                support_geom_ids.append(geom_index)
            elif "finger" in geom_name:
                # Finger parameters win for finger-object pairs; rack-panel support stays softer.
                geom_solimp[geom_index, :3] = (0.995, 0.999, 0.0005)
                finger_geom_ids.append(geom_index)
            elif "/World/Panel/" in geom_name:
                geom_solimp[geom_index, :3] = (0.98, 0.995, 0.002)
                panel_geom_ids.append(geom_index)
        hardened_geom_ids = support_geom_ids + finger_geom_ids + panel_geom_ids

        # MJWarp does not preserve USD FilteredPairsAPI for these instance-derived
        # finger links. Use MuJoCo collision masks in the live Newton model:
        # panel/scene=bit 1, left finger=bit 2, right finger=bit 4. Both fingers
        # collide with bit-1 objects, while opposing fingers cannot collide.
        collision_masks_applied = False
        if all(hasattr(model, field) for field in ("geom_contype", "geom_conaffinity")):
            geom_contype = model.geom_contype.numpy()
            geom_conaffinity = model.geom_conaffinity.numpy()
            contype_view = geom_contype[0] if geom_contype.ndim == 2 else geom_contype
            conaffinity_view = geom_conaffinity[0] if geom_conaffinity.ndim == 2 else geom_conaffinity
            for geom_index in finger_geom_ids:
                geom_name = mujoco.mj_id2name(
                    solver.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_index
                ) or ""
                if "panda_leftfinger" in geom_name:
                    contype_view[geom_index] = 2
                    conaffinity_view[geom_index] = 1
                elif "panda_rightfinger" in geom_name:
                    contype_view[geom_index] = 4
                    conaffinity_view[geom_index] = 1
            model.geom_contype = wp.array(
                geom_contype, dtype=model.geom_contype.dtype, device=model.geom_contype.device
            )
            model.geom_conaffinity = wp.array(
                geom_conaffinity, dtype=model.geom_conaffinity.dtype, device=model.geom_conaffinity.device
            )
            collision_masks_applied = True
        model.geom_solimp = wp.array(solimp, dtype=model.geom_solimp.dtype, device=model.geom_solimp.device)
        if all(hasattr(model, field) for field in ("geom_margin", "geom_gap", "geom_solref", "geom_priority")):
            geom_margin = model.geom_margin.numpy()
            geom_gap = model.geom_gap.numpy()
            geom_solref = model.geom_solref.numpy()
            geom_priority = model.geom_priority.numpy()
            margin_view = geom_margin[0] if geom_margin.ndim == 2 else geom_margin
            gap_view = geom_gap[0] if geom_gap.ndim == 2 else geom_gap
            solref_view = geom_solref[0] if geom_solref.ndim == 3 else geom_solref
            priority_view = geom_priority[0] if geom_priority.ndim == 2 else geom_priority
            for geom_index in support_geom_ids:
                margin_view[geom_index] = 0.012
                gap_view[geom_index] = 0.0
                solref_view[geom_index, :] = (0.010, 2.5)
                priority_view[geom_index] = 2
            for geom_index in finger_geom_ids:
                margin_view[geom_index] = 0.004
                gap_view[geom_index] = 0.0
                # 8 ms is above 2x the 2.08 ms substep but materially firmer than 20 ms.
                solref_view[geom_index, :] = (0.008, 2.0)
                priority_view[geom_index] = 3
            for geom_index in panel_geom_ids:
                margin_view[geom_index] = 0.002
                gap_view[geom_index] = 0.0
                solref_view[geom_index, :] = (0.02, 1.5)
                priority_view[geom_index] = 0
            model.geom_margin = wp.array(geom_margin, dtype=model.geom_margin.dtype, device=model.geom_margin.device)
            model.geom_gap = wp.array(geom_gap, dtype=model.geom_gap.dtype, device=model.geom_gap.device)
            model.geom_solref = wp.array(geom_solref, dtype=model.geom_solref.dtype, device=model.geom_solref.device)
            model.geom_priority = wp.array(geom_priority, dtype=model.geom_priority.dtype, device=model.geom_priority.device)

        # Newton was allowing Panda finger joints to cross their 0 m lower limit.
        # Strengthen the actual MuJoCo joint-limit constraint instead of limiting the
        # commanded grip width; the actuator may still command a fully closed hand.
        hardened_joint_ids = []
        if all(hasattr(model, field) for field in ("jnt_solimp", "jnt_solref", "jnt_margin")):
            jnt_solimp = model.jnt_solimp.numpy()
            jnt_solref = model.jnt_solref.numpy()
            jnt_margin = model.jnt_margin.numpy()
            joint_solimp = jnt_solimp[0] if jnt_solimp.ndim == 3 else jnt_solimp
            joint_solref = jnt_solref[0] if jnt_solref.ndim == 3 else jnt_solref
            joint_margin = jnt_margin[0] if jnt_margin.ndim == 2 else jnt_margin
            for joint_index in range(int(solver.mj_model.njnt)):
                joint_name = mujoco.mj_id2name(
                    solver.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_index
                ) or ""
                if "panda_finger_joint" in joint_name:
                    joint_solimp[joint_index, 0] = 0.99
                    joint_solimp[joint_index, 1] = 0.999
                    joint_solimp[joint_index, 2] = 0.001
                    joint_solref[joint_index, 0] = 0.008
                    joint_solref[joint_index, 1] = 2.0
                    joint_margin[joint_index] = 0.001
                    hardened_joint_ids.append(joint_index)
            model.jnt_solimp = wp.array(jnt_solimp, dtype=model.jnt_solimp.dtype, device=model.jnt_solimp.device)
            model.jnt_solref = wp.array(jnt_solref, dtype=model.jnt_solref.dtype, device=model.jnt_solref.device)
            model.jnt_margin = wp.array(jnt_margin, dtype=model.jnt_margin.dtype, device=model.jnt_margin.device)
        print(
            f"[INFO] thin_grasp_constraints_hardened geoms={hardened_geom_ids} "
            f"finger_joints={hardened_joint_ids} collision_masks={collision_masks_applied}", flush=True
        )
    except Exception as error:  # noqa: BLE001 - the field is version-specific
        print(f"[WARN] harden_contacts could not write geom_solimp: {error}", flush=True)
        return
    print(
        f"[INFO] contacts_hardened shape={solimp.shape} elements={solimp.size} "
        f"solimp=(0.95, 0.99, 0.002) sample={solimp.reshape(-1, solimp.shape[-1])[0].tolist()}",
        flush=True,
    )


def report_room_camera_poses(cameras):
    """Print what the room cameras actually ended up looking at.

    ``set_world_poses_from_view`` was tried first and never took - the readback kept
    returning the spawn offset - so both room views are authored as static offsets and
    verified here instead of trusted.
    """
    for name in ("hanger_front", "hanger_top"):
        camera = cameras.get(name)
        if camera is None:
            continue
        position = camera.data.pos_w[0].tolist()
        quaternion = camera.data.quat_w_ros[0].tolist()
        print(
            f"[INFO] room_camera {name} pos={[round(v, 3) for v in position]} "
            f"quat_ros={[round(v, 4) for v in quaternion]}",
            flush=True,
        )


def spawn_task_cameras():
    """Create two hand-mounted RGB sensors and one hanger-front sensor."""
    camera_data_types = ["rgb", "instance_segmentation_fast"] if args_cli.record_instance_segmentation else ["rgb"]
    wrist_pinhole_cfg = sim_utils.PinholeCameraCfg(
        # Wider than the external camera so the fingers and nearby grasp
        # workspace remain visible even when the wrist is close to the panel.
        # 12.0 gave an 82 degree field that filled half the frame with the hand;
        # 16.0 is 66 degrees, still wide enough to hold the jaws and the edge.
        focal_length=16.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.03, 10.0),
    )
    front_pinhole_cfg = sim_utils.PinholeCameraCfg(
        focal_length=14.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.03, 10.0),
    )
    # Mounted on the side of the hand just ahead of the palm, tilted in toward the
    # TCP.  The old (0.13, 0, -0.15) mount sat 150 mm *behind* the palm, so the hand
    # and forearm filled most of the frame instead of the grasp.
    wrist_offset = CameraCfg.OffsetCfg(
        pos=WRIST_CAMERA_POS,
        rot=WRIST_CAMERA_ROT_ROS,
        convention="ros",
    )

    cameras = {}
    for name, robot_path in (
        ("left_wrist", "/World/FrankaLeft"),
        ("right_wrist", "/World/FrankaRight"),
    ):
        cameras[name] = Camera(
            cfg=CameraCfg(
                prim_path=f"{robot_path}/panda_hand/{name}_camera",
                update_period=CAMERA_UPDATE_PERIOD,
                height=CAMERA_HEIGHT,
                width=CAMERA_WIDTH,
                data_types=camera_data_types,
                colorize_instance_segmentation=False,
                spawn=wrist_pinhole_cfg,
                offset=wrist_offset,
            )
        )

    # Two room views: the original eye-level front view, and an overhead one aimed by
    # aim_overhead_camera() once the sim is up.
    # Its own cfg instance.  Handing the same PinholeCameraCfg object to two Camera
    # configs left this one rendering from somewhere other than its authored pose -
    # the readback reported (-0.25, 0, 2.30) looking down while the frames showed a
    # ceiling corner.
    top_pinhole_cfg = sim_utils.PinholeCameraCfg(
        focal_length=14.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.03, 10.0),
    )
    cameras["hanger_top"] = Camera(
        cfg=CameraCfg(
            prim_path="/World/HangerTopCamera",
            update_period=CAMERA_UPDATE_PERIOD,
            height=CAMERA_HEIGHT,
            width=CAMERA_WIDTH,
            data_types=camera_data_types,
            colorize_instance_segmentation=False,
            spawn=top_pinhole_cfg,
            offset=CameraCfg.OffsetCfg(
                pos=TOP_CAMERA_EYE,
                rot=TOP_CAMERA_ROT_ROS,
                convention="ros",
            ),
        )
    )
    cameras["hanger_front"] = Camera(
        cfg=CameraCfg(
            prim_path="/World/HangerFrontCamera",
            update_period=CAMERA_UPDATE_PERIOD,
            height=CAMERA_HEIGHT,
            width=CAMERA_WIDTH,
            data_types=camera_data_types,
            colorize_instance_segmentation=False,
            spawn=front_pinhole_cfg,
            offset=CameraCfg.OffsetCfg(
                pos=HANGER_CAMERA_EYE,
                rot=HANGER_CAMERA_ROT_ROS,
                convention="ros",
            ),
        )
    )
    if args_cli.validation_side_camera:
        cameras["hanger_side"] = Camera(
            cfg=CameraCfg(
                prim_path="/World/HangerSideValidationCamera",
                update_period=CAMERA_UPDATE_PERIOD,
                height=CAMERA_HEIGHT,
                width=CAMERA_WIDTH,
                data_types=["rgb"],
                spawn=front_pinhole_cfg,
                offset=CameraCfg.OffsetCfg(
                    pos=HANGER_SIDE_CAMERA_EYE,
                    rot=HANGER_SIDE_CAMERA_ROT_ROS,
                    convention="ros",
                ),
            )
        )
    return cameras


def spawn_hanging_nails(stage):
    """Two long nails driven horizontally where the hooks used to hang.

    A picture goes on nails, and this reads as one: the panel's top rail drops over
    them and rests.  It replaces a pair of articulated S-hooks that swung on the rack
    rod - they never looked like hooks, and a receiver that can move away from the
    board while the board is landing on it is the worst possible contact for this.
    These are static geometry, so the only moving body in the hang is the panel.

    The catch height and reach are unchanged, so validate_layout's clearances between
    the nail, the hung board face and the rear rail still hold.
    """
    steel = (0.72, 0.74, 0.78)
    UsdGeom.Xform.Define(stage, "/World/HangerStand/Nails")
    nail_z = HANGER_SHELF_TOP_Z - NAIL_RADIUS
    for index, y_pos in enumerate((0.0,)):
        nail_path = f"/World/HangerStand/Nails/Nail{index}"
        nail = UsdGeom.Cylinder.Define(stage, nail_path)
        nail.CreateAxisAttr("Z")
        nail.CreateRadiusAttr(NAIL_RADIUS)
        nail.CreateHeightAttr(NAIL_LENGTH)
        nail.AddTranslateOp().Set(Gf.Vec3d(NAIL_HEAD_X - 0.5 * NAIL_LENGTH, y_pos, HANGER_ROD_Z))
        # Explicitly rotate the cylinder's Z axis into the horizontal X axis.
        nail.AddRotateYOp().Set(90.0)
        nail.CreateDisplayColorAttr([Gf.Vec3f(*steel)])
        UsdPhysics.CollisionAPI.Apply(nail.GetPrim())
        sim_utils.bind_physics_material(nail_path, HIGH_HANG_MATERIAL_PATH, stage=stage)


def resolve_hanger_target_from_stage():
    """Resolve the hang target from the authored nail prim, not scene constants."""
    stage = sim_utils.get_current_stage()
    nail_prim = stage.GetPrimAtPath("/World/HangerStand/Nails/Nail0")
    if not nail_prim or not nail_prim.IsValid():
        raise RuntimeError("Hanger nail prim is missing; cannot resolve a physical hang target")
    cache = UsdGeom.XformCache()
    nail_world = cache.GetLocalToWorldTransform(nail_prim).ExtractTranslation()
    # The nail's local +Z axis is rotated into world +X.  Use its actual tip
    # position as the anchor; only panel geometry offsets remain below.
    nail_tip_x = float(nail_world[0]) + 0.5 * NAIL_LENGTH
    hang_position = (
        nail_tip_x - HANDLE_LOCAL_Z,
        float(nail_world[1]),
        float(nail_world[2]) - (abs(HANDLE_LOCAL_X) + RING_RADIUS),
    )
    print(
        f"[INFO] hanger_target_from_nail nail=({float(nail_world[0]):.4f},{float(nail_world[1]):.4f},{float(nail_world[2]):.4f}) "
        f"target=({hang_position[0]:.4f},{hang_position[1]:.4f},{hang_position[2]:.4f})",
        flush=True,
    )
    return hang_position


def design_scene(panel_state):
    validate_layout()
    appearance = sample_episode_appearance()
    spawn_task_physics_materials()
    spawn_room(appearance)
    spawn_episode_lights(appearance)

    left_cfg = FRANKA_PANDA_CFG.copy()
    left_cfg.prim_path = "/World/FrankaLeft"
    left_cfg.spawn.usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    left_cfg.init_state.pos = (ROBOT_BASE_X, ROBOT_BASE_Y, 0.42)
    high_ready_joint_pos = {
        "panda_joint1": 0.0,
        "panda_joint2": -0.785,
        "panda_joint3": 0.0,
        "panda_joint4": -2.356,
        "panda_joint5": 0.0,
        "panda_joint6": 1.571,
        "panda_joint7": 0.785,
        # Start fully open: the front grasp straddles the 57 mm board+rail stack.
        "panda_finger_joint.*": 0.0395,
    }
    left_cfg.init_state.joint_pos = high_ready_joint_pos.copy()
    right_cfg = FRANKA_PANDA_CFG.copy()
    right_cfg.prim_path = "/World/FrankaRight"
    right_cfg.spawn.usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    right_cfg.init_state.pos = (ROBOT_BASE_X, -ROBOT_BASE_Y, 0.42)
    right_cfg.init_state.joint_pos = high_ready_joint_pos.copy()
    for robot_cfg in (left_cfg, right_cfg):
        robot_cfg.spawn.rigid_props.disable_gravity = True

        for actuator_name in ("panda_shoulder", "panda_forearm"):
            robot_cfg.actuators[actuator_name].stiffness = 400.0
            robot_cfg.actuators[actuator_name].damping = 80.0
        robot_cfg.actuators["panda_hand"].velocity_limit_sim = 0.12
        robot_cfg.actuators["panda_hand"].effort_limit_sim = 70.0
        # Position control at 1500 N/m slams ~59 N into the panel the instant the
        # jaws are commanded closed, which pops it off the rails.  ~16 N still gives
        # each pad about 24 N of friction against a 1.6 kg panel.
        # 2000 N/m drives the finger joints unstable here - they overshoot past the
        # 0.04 travel limit and even go negative, since the limits are not enforced.
        # 800 N/m holds ~19 N per finger, and the grip strength that actually mattered
        # for the pull came from the solver's impratio, not from this gain.
        # Stock FRANKA_PANDA_CFG gains.  A real Panda hand closes with up to 70 N and
        # a teleoperator simply grips hard and pulls; at 800 N/m the same command was
        # worth under 5 N, and the shortfall was being papered over with a 2.5 friction
        # coefficient.  These gains chattered before only because the contact was soft
        # enough to be penetrated - see harden_contacts().
        # Critically damp the finger position loop.  The previous 2000/100
        # setting visibly chattered against the handle: Newton contact pushed
        # the measured width past the target and the actuator sprang back.
        robot_cfg.actuators["panda_hand"].stiffness = 3500.0
        robot_cfg.actuators["panda_hand"].damping = 650.0
    left_robot = Articulation(cfg=left_cfg)
    right_robot = Articulation(cfg=right_cfg)

    spawn_static_box(
        "/World/Table",
        TABLE_SIZE,
        TABLE_POSITION,
        (0.24, 0.26, 0.29),
    )

    # Parallel "11" rack rails: the board bridges them, leaving a clear gap
    # between the tabletop and its underside for the first underhand pull.
    # A 0.56 m centre-to-centre gap keeps the 0.86 m panel supported while
    # reducing lateral play during the pull; do not narrow this further because
    # the resulting outer overhang makes the panel easier to tip.
    for name, y_pos in (("LeftRail", -0.28), ("RightRail", 0.28)):
        spawn_static_box(
            f"/World/PanelRack/{name}",
            # Narrow 20 mm bearing strip as requested; the rail centres remain
            # under the frame members so the panel is still supported.
            (RACK_RAIL_LENGTH, 0.020, 0.120),
            (RACK_RAIL_POSITION_X, y_pos, 0.480),
            (0.52, 0.54, 0.58),
            physics_material_path=LOW_SLIDE_MATERIAL_PATH,
        )


    # Open-front side guards prevent an immediate lateral grasp. The panel
    # must first be pulled toward the robots until its front corners clear
    # these walls; there is deliberately no roof over the panel.  Their inner
    # faces stay 35 mm clear of the panel edge so the vertical lift can pass.
    for name, y_pos in (("Left", -0.5600), ("Right", 0.5600)):
        spawn_static_box(
            f"/World/PanelRack/{name}SideGuard",
            # Start behind the side-grasp region: the wrists sit at |y| = 0.50, inside
            # the guard band, so a guard reaching to x = 0.27 puts the arms straight
            # into it once the panel is only pulled ~30 mm forward.
            # Spans x in [0.45, 0.89].  Matching the panel's own 540 mm length was
            # tried, but then the side grasp can never clear the guard: it sits 60 mm
            # ahead of the panel centre, and pulling the panel far enough forward to
            # get it past x = 0.32 walks the centre of mass off the rack rails.  What
            # the guard has to block is the *staged* side grasp at x = 0.53, and this
            # does, while leaving the pulled grasp at 0.40 free.
            # Tall enough to stand over the panel - that is the whole point, the wrists
            # cannot reach the side edges past them - and thickened inward so the inner
            # faces sit 15 mm off the board rather than 35 mm.  Lowering them below the
            # rails was tried and defeats their purpose entirely.
            (0.38, 0.040, 0.20),
            (0.54, y_pos, 0.52),
            (0.36, 0.38, 0.42),
            # The panel must slide along the guard face while it is pulled out;
            # use the same low-friction Newton material as the rails.
            physics_material_path=LOW_SLIDE_MATERIAL_PATH,
        )

    # Rear back-stop: the panel starts flush against this wall.  It blocks only
    # backward motion (+X); the robot-facing side remains open for the handle pull.
    panel_rear_x = PANEL_STAGING_POSITION[0] + 0.5 * 0.54
    spawn_static_box(
        "/World/PanelRack/RearBackStop",
        (0.040, 0.92, 0.20),
        (panel_rear_x + 0.020, 0.0, 0.52),
        (0.36, 0.38, 0.42),
        # Low friction prevents the back-stop from braking the initial pull;
        # it still remains a hard geometric stop in +X.
        physics_material_path=LOW_SLIDE_MATERIAL_PATH,
    )

    # Short anti-pop cover at the rear end only.  It catches an upward impulse
    # while the panel is still staged on the rack.  It sits over the panel
    # rather than behind the rear wall, and is high enough not to block lift.
    spawn_static_box(
        "/World/PanelRack/RearTopStop",
        # Side guards span y=[-0.58, 0.58]; this width meets both outer faces.
        (0.30, 1.16, 0.025),
        # Keep it over the rear tip but above the grasp envelope; the panel is
        # free to translate/lift below this clearance once it leaves the rack.
        # Side guards top out at z=0.620; seat the 25 mm cover directly on them.
        # The rear wall front face is x=0.69; with a 0.30 m cover this
        # centre places the cover's rear edge directly against that wall.
        (0.54, 0.0, 0.6325),
        (0.30, 0.32, 0.35),
        physics_material_path=LOW_SLIDE_MATERIAL_PATH,
    )

    # Clothes-rack-style round rod running horizontally along Y.
    rod_x, rod_z, rod_length = HANGER_ROD_X, HANGER_ROD_Z, HANGER_ROD_LENGTH
    spawn_static_cylinder(
        "/World/HangerStand/Rod",
        radius=0.03,
        height=rod_length,
        position=(rod_x, 0.0, rod_z),
        color=(0.30, 0.32, 0.36),
        physics_material_path=HIGH_HANG_MATERIAL_PATH,
    )

    # Passive S-hooks yield backward under a hard push but retain a careful placement.
    spawn_hanging_nails(sim_utils.get_current_stage())

    # Standalone hanger: two posts and feet support the round horizontal rod.
    for name, y_pos in (("Left", -HANGER_POST_Y), ("Right", HANGER_POST_Y)):
        spawn_static_box(
            f"/World/HangerStand/{name}Post",
            (HANGER_POST_THICKNESS, HANGER_POST_THICKNESS, HANGER_POST_HEIGHT),
            (rod_x, y_pos, 0.5 * HANGER_POST_HEIGHT),
            (0.22, 0.24, 0.27),
        )
        spawn_static_box(
            f"/World/HangerStand/{name}Foot",
            (0.60, 0.14, 0.06),
            (rod_x, y_pos, 0.03),
            (0.22, 0.24, 0.27),
        )
    panel = spawn_panel(panel_state, appearance["panel_color"])
    configure_newton_grasp_contacts()
    for prim_path, semantic_label in (
        ("/World/Room", "room_background"),
        ("/World/Table", "table"),
        ("/World/PanelRack", "panel_rack"),
        ("/World/HangerStand", "hanger"),
        (PANEL_ROOT, "panel"),
        ("/World/FrankaLeft", "left_robot"),
        ("/World/FrankaRight", "right_robot"),
    ):
        apply_semantic_label(prim_path, semantic_label)
    cameras = {} if args_cli.disable_task_cameras else spawn_task_cameras()
    print(
        "[INFO] episode_appearance "
        f"seed={appearance['seed']} local_lights={len(appearance['lights'])} "
        f"wallpaper_rgb={tuple(round(c, 3) for c in appearance['wallpaper_color'])} "
        f"panel_rgb={tuple(round(c, 3) for c in appearance['panel_color'])}"
    )
    for index, light in enumerate(appearance["lights"]):
        print(
            f"[INFO] episode_light[{index}] position={tuple(round(v, 3) for v in light['position'])} "
            f"intensity={light['intensity']:.1f} radius={light['radius']:.3f}"
        )
    return left_robot, right_robot, panel, cameras, appearance


def reset_entities(left_robot, right_robot, panel):
    for robot in (left_robot, right_robot):
        robot.write_root_pose_to_sim_index(root_pose=robot.data.default_root_pose.torch.clone())
        robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.torch.clone())
        robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
        robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
        robot.reset()
    panel.write_root_pose_to_sim_index(root_pose=panel.data.default_root_pose.torch.clone())
    panel.write_root_velocity_to_sim_index(root_velocity=panel.data.default_root_vel.torch.clone())
    panel.reset()


def set_standalone_panel_start(panel, task_mode):
    """Place the panel at the object-relative start pose for a subtask."""
    if task_mode == "long" or task_mode == "pull":
        return
    pose = panel.data.default_root_pose.torch.clone()
    if task_mode == "lift":
        # Pull task has already cleared the guards; keep the panel horizontal.
        pulled_x = PANEL_STAGING_POSITION[0] - PANEL_PULL_DISTANCE
        rotation_clear_x = (
            RACK_COVER_FRONT_X - PANEL_HALF_LENGTH - RACK_ROTATION_CLEARANCE_X
        )
        pose[0, 0] = min(pulled_x, rotation_clear_x)
        pose[0, 1] = PANEL_STAGING_POSITION[1]
        pose[0, 2] = PANEL_STAGING_POSITION[2]
        pose[0, 3:7] = panel.data.default_root_pose.torch[0, 3:7]
    elif task_mode == "hang":
        # Hang starts with the panel vertical at the hanger-entry height.  The
        # controller's bimanual hold phase then owns the transport/release.
        pose[0, :3] = torch.tensor(
            (PANEL_HUNG_POSITION[0], PANEL_HUNG_POSITION[1], PANEL_HANG_ENTRY_Z),
            device=pose.device,
            dtype=pose.dtype,
        )
        pose[0, 3:7] = torch.tensor((0.7071068, 0.0, 0.7071068, 0.0), device=pose.device, dtype=pose.dtype)
    panel.write_root_pose_to_sim_index(root_pose=pose)
    panel.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(panel.data.default_root_vel.torch))
    panel.reset()


def main():
    if args_cli.physics_dt <= 0.0 or args_cli.substeps < 1 or args_cli.max_steps < 0 or args_cli.motion_speed_scale <= 0.0:
        raise ValueError("physics-dt/substeps/max-steps values are invalid")

    if args_cli.camera_preview and not (1 <= args_cli.camera_preview_port <= 65535 and args_cli.camera_preview_fps > 0):
        raise ValueError("camera-preview-port/fps values are invalid")
    if args_cli.disable_task_cameras and (args_cli.camera_preview or args_cli.record_dir is not None):
        raise ValueError("--disable-task-cameras cannot be combined with camera preview or recording")
    print(
        f"[INFO] cli_auto_ops={args_cli.auto_ops} auto_ops_task={args_cli.auto_ops_task} "
        f"cli_record_dir={args_cli.record_dir!r}",
        flush=True,
    )
    if args_cli.capture_every < 1 or args_cli.writer_queue_size < 1:
        raise ValueError("capture-every/writer-queue-size values must be positive")

    physics_cfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=1000,
            nconmax=512,
            cone="elliptic",
            # Frictional-to-normal constraint impedance.  Grasping wants the tangential
            # constraint stiffer than the normal one, but 50 was chasing a slip that
            # came from too little grip force, not from the solver.
            impratio=10.0,
            iterations=80,
            ls_iterations=30,
            ccd_iterations=32,
            tolerance=1.0e-8,
            update_data_interval=1,
            use_mujoco_contacts=True,
            save_to_mjcf=str(args_cli.save_newton_mjcf) if args_cli.save_newton_mjcf else None,
        ),
        num_substeps=args_cli.substeps,
        debug_mode=args_cli.newton_debug,
        use_cuda_graph=not args_cli.disable_cuda_graph,
    )
    sim = SimulationContext(
        sim_utils.SimulationCfg(
            dt=args_cli.physics_dt,
            render_interval=16 if args_cli.disable_task_cameras else 2,
            device=args_cli.device,
            physics=physics_cfg,
        )
    )
    left_robot, right_robot, panel, cameras, appearance = design_scene(args_cli.panel_state)
    sim.reset()
    harden_contacts()
    set_standalone_panel_start(panel, args_cli.auto_ops_task if args_cli.auto_ops else "long")
    report_room_camera_poses(cameras)
    if args_cli.newton_debug:
        from isaaclab_newton.physics.newton_manager import NewtonManager

        margins = NewtonManager._model.shape_margin.numpy()
        nonzero_margins = margins[margins > 0.0]
        unique_margins = sorted({round(float(value), 7) for value in nonzero_margins})
        mj_opt = NewtonManager._solver.mjw_model.opt
        print(
            "[INFO] newton_backend_verified "
            f"collision_pipeline={type(NewtonManager._collision_pipeline).__name__} "
            f"external_contacts={NewtonManager._needs_collision_pipeline} "
            f"nonzero_shape_margins={len(nonzero_margins)} unique_margins={unique_margins} "
            f"solver={mj_opt.solver} integrator={mj_opt.integrator} cone={mj_opt.cone} "
            f"iterations={mj_opt.iterations} ls_iterations={mj_opt.ls_iterations} "
            f"ccd_iterations={mj_opt.ccd_iterations} impratio_invsqrt={mj_opt.impratio_invsqrt}",
            flush=True,
        )
        # The authored USD and the solved collision geometry must agree; a silent
        # mismatch here shows up as a panel that rests at the wrong height.
        import mujoco

        solver = NewtonManager._solver
        for geom_index in range(int(solver.mj_model.ngeom)):
            geom_name = mujoco.mj_id2name(solver.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_index) or ""
            if not any(token in geom_name for token in ("Panel", "HangerStand", "Table", "finger")):
                continue
            print(
                f"[INFO] collider name={geom_name} type={int(solver.mj_model.geom_type[geom_index])} "
                f"size={solver.mj_model.geom_size[geom_index]} pos={solver.mj_model.geom_pos[geom_index]} "
                f"body={int(solver.mj_model.geom_bodyid[geom_index])} "
                f"margin={float(solver.mj_model.geom_margin[geom_index]):.5f} "
                f"gap={float(solver.mj_model.geom_gap[geom_index]):.5f} "
                f"solref={solver.mj_model.geom_solref[geom_index]} "
                f"priority={int(solver.mj_model.geom_priority[geom_index])} "
                f"contype={int(solver.mj_model.geom_contype[geom_index])} "
                f"conaff={int(solver.mj_model.geom_conaffinity[geom_index])} "
                f"condim={int(solver.mj_model.geom_condim[geom_index])} "
                f"friction={solver.mj_model.geom_friction[geom_index]}",
                flush=True,
            )
        print(f"[INFO] mj_opt gravity={mj_opt.gravity} disableflags={int(mj_opt.disableflags)}", flush=True)
    reset_entities(left_robot, right_robot, panel)
    if args_cli.validation_side_camera:
        cameras["hanger_side"].set_world_poses_from_view(
            eyes=[HANGER_SIDE_CAMERA_EYE],
            targets=[(0.62, 0.0, 1.00)],
        )
    set_task_camera(sim)
    print(f"[INFO] rgb_cameras={','.join(cameras)} resolution={CAMERA_WIDTH}x{CAMERA_HEIGHT}")

    print("[INFO] Dual-Franka picture-hanging task scene is ready.")
    print(f"[INFO] physics=newton_mjwarp dt={args_cli.physics_dt:.6f} substeps={args_cli.substeps}")
    print("[INFO] kinematics_solver=pink qp_solver=daqp gravity_compensation=idealized_link_gravity_disabled")
    print(f"[INFO] panel_state={args_cli.panel_state}")

    step = 0
    captured_frame = 0
    sim_dt = sim.get_physics_dt()
    if args_cli.auto_ops:
        settle_steps = 1 if args_cli.auto_ops_task in ("lift", "hang") else 60
        print(f"[INFO] auto_ops_settle_steps={settle_steps}", flush=True)
        for _ in range(settle_steps):
            for robot in (left_robot, right_robot):
                robot.set_joint_position_target_index(target=robot.data.default_joint_pos.torch)
                robot.write_data_to_sim()
            panel.write_data_to_sim()
            sim.step(render=False)
            left_robot.update(sim_dt)
            right_robot.update(sim_dt)
            panel.update(sim_dt)
        panel.write_root_pose_to_sim_index(root_pose=panel.data.default_root_pose.torch.clone())
        panel.write_root_velocity_to_sim_index(root_velocity=panel.data.default_root_vel.torch.clone())
        panel.update(sim_dt)
        validate_panel_rest_pose(panel)
        # Restore standalone lift/hang starts after the common settle pass;
        # otherwise the settle reset would silently put the panel back at staging.
        set_standalone_panel_start(panel, args_cli.auto_ops_task)
    auto_ops = None
    preview = None
    writer = None
    if args_cli.record_dir is not None:
        writer = AsyncSensorWriter(args_cli.record_dir, cameras, args_cli.writer_queue_size)
        print(f"[INFO] async_sensor_recording={args_cli.record_dir} capture_every={args_cli.capture_every}")
    if args_cli.auto_ops:
        try:
            resolved_hang_position = resolve_hanger_target_from_stage()
            auto_ops = DualFrankaAutoOps(
                left_robot,
                right_robot,
                panel,
                control_decimation=args_cli.capture_every,
                physics_dt=sim_dt,
                motion_speed_scale=args_cli.motion_speed_scale,
                hang_position=resolved_hang_position,
                hang_entry_z=resolved_hang_position[2] + 0.050,
                task_mode=args_cli.auto_ops_task,
            )
        except BaseException as exc:
            print(f"[AUTO_OPS_ERROR] controller_init={exc!r}", flush=True)
            traceback.print_exc()
            raise
        print(f"[INFO] auto_ops=enabled action_hz={1.0 / (sim_dt * args_cli.capture_every):g}")
    preview_stride = max(1, round(1.0 / (args_cli.camera_preview_fps * sim_dt)))
    if args_cli.camera_preview:
        preview = CameraPreviewServer(args_cli.camera_preview_port)
        preview.start()
        print(f"[INFO] camera_preview=http://0.0.0.0:{args_cli.camera_preview_port} fps={args_cli.camera_preview_fps:g}")

    while args_cli.max_steps == 0 or step < args_cli.max_steps:
        if auto_ops is not None:
            auto_ops.apply(step)
        else:
            for robot in (left_robot, right_robot):
                robot.set_joint_position_target_index(target=robot.data.default_joint_pos.torch)
                robot.write_data_to_sim()
        panel.write_data_to_sim()
        sim.step()
        for camera in cameras.values():
            camera.update(sim_dt)
        if writer is not None and step % args_cli.capture_every == 0:
            sample_metadata = auto_ops.sample_metadata() if auto_ops is not None else None
            writer.submit(captured_frame, step, cameras, sample_metadata)
            captured_frame += 1
        if preview is not None and step % preview_stride == 0:
            preview.update(cameras)
        if step == 0:
            for name, camera in cameras.items():
                rgb = camera.data.output.get("rgb")
                instance_map = camera.data.output.get("instance_segmentation_fast")
                rgb_shape = tuple(rgb.shape) if rgb is not None else None
                instance_shape = tuple(instance_map.shape) if instance_map is not None else None
                print(f"[INFO] camera={name} rgb_shape={rgb_shape} instance_shape={instance_shape}")
        left_robot.update(sim_dt)
        right_robot.update(sim_dt)
        panel.update(sim_dt)
        if args_cli.newton_debug and step % 40 == 0:
            # Quaternion is reported xyzw by the Newton RigidObject backend.
            print(
                f"[DEBUG] step={step} panel_pose={[round(v, 5) for v in panel.data.root_pose_w.torch[0].tolist()]}",
                flush=True,
            )
        step += 1

        if auto_ops is not None and auto_ops.done and not args_cli.hold_after_auto_ops:
            break
    if preview is not None:
        preview.stop()
    if writer is not None:
        writer.close()
    panel_position = panel.data.root_pos_w.torch[0].detach().cpu().tolist()
    print(f"[INFO] completed_steps={step} panel_position={panel_position}")
    if auto_ops is not None and args_cli.record_dir is not None:
        episode_metadata = {
            "episode_id": args_cli.episode_id,
            "episode_task": "picture_hanging",
            "auto_ops_task": args_cli.auto_ops_task,
            "success": auto_ops.success,
            "failure_reason": auto_ops.failure_reason,
            "physics_hz": 1.0 / sim_dt,
            "physics_engine": "newton_mjwarp",
            "kinematics_solver": "pink",
            "qp_solver": "daqp",
            "gravity_compensation": False,
            "robot_action_hz": 1.0 / (sim_dt * args_cli.capture_every),
            "camera_fps": 1.0 / CAMERA_UPDATE_PERIOD,
            "image_size": [CAMERA_WIDTH, CAMERA_HEIGHT],
            "captured_frames": captured_frame,
            "scene_appearance": appearance,
            "events": auto_ops.events,
            "state_layout": "left_eef_10d,right_eef_10d",
            "action_layout": "left_delta_pose_gripper_7d,right_delta_pose_gripper_7d",
        }
        episode_path = Path(args_cli.record_dir) / "episode.json"
        episode_path.write_text(json.dumps(episode_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
