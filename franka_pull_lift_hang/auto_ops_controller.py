"""Scripted dual-Franka Auto Ops controller for one picture-hanging episode."""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import pinocchio as pin
import torch
from isaaclab.controllers.pink_ik import (
    DampingTaskCfg,
    FrameTaskCfg,
    PinkIKController,
    PinkIKControllerCfg,
    NullSpacePostureTaskCfg,
)
from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    matrix_from_quat,
    quat_from_angle_axis,
    quat_mul,
    subtract_frame_transforms,
)

from task_schema import pack_bimanual_action, pack_bimanual_state, phase_annotation
from auto_ops_modules import LongTaskOrchestrator, ModuleContractError

# The Franka palm ends where the finger links begin, so the finger-origin plane is
# also the palm face.  The pads live 5..54 mm beyond it; every grasp therefore parks
# the finger origins *outside* the part and lets the pads reach in.  Aiming them past
# the edge instead drives the palm into the panel and shoves it across the rack.
PALM_CLEARANCE = 0.012
# Franka Hand travel is 0..0.04 per finger; commands below are the jaw gap.
GRIPPER_MAX_WIDTH = 0.0795
GRIPPER_OPEN_WIDTH = 0.0790
POLICY_GRIP_FORCE_PER_FINGER = 35.0
# Do not step all four closed-chain finger actuators from zero to their peak
# effort on the same Newton substep. The final policy force is unchanged; this
# short actuator ramp only removes the non-physical impulse at contact latch.
GRIP_FORCE_LATCH_PER_FINGER = 8.0
GRIP_FORCE_RAMP_PER_CONTROL_TICK = 6.0
# Jaw gap removed per 15 Hz control tick while closing (~0.0225 m/s).
GRIPPER_CLOSE_RATE = 0.0015

# Arm rate limits.  Most of an episode is spent waiting for Pink's solution to be
# tracked, not travelling, so these dominate the wall-clock far more than
# --motion-speed-scale does.
IK_STEP_LIMIT = 0.18
IK_TRACKING_GAIN = 3.0
IK_TRACKING_LIMIT = 0.30

# Panel geometry mirrored from dual_franka_picture_hanging_scene.spawn_panel.
# Offsets are relative to the panel root, in the staging (face-up) pose.
PANEL_HALF_LENGTH = 0.27
PANEL_HALF_WIDTH = 0.43
# Rack centreline and rear-cover front edge, derived from scene geometry.
RACK_CENTER_Y = 0.0
RACK_COVER_FRONT_X = 0.39
RACK_CLEARANCE_X = 0.01
# Extra front clearance for the swept panel corner during horizontal-to-vertical rotation.
RACK_ROTATION_CLEARANCE_X = 0.08
PANEL_HALF_THICKNESS = 0.015
FINGER_TIP_LENGTH = 0.054
# The top-down bite is centered halfway between the panel front edge
# (-0.270) and the handle arc apex (-0.340).  This puts the two open fingers
# inside the curved radius before closing, rather than aiming at the empty loop
# centre or the extreme apex.
HANDLE_ARC_APEX_X = -0.340
HANDLE_RING_CENTER_LOCAL_X = -0.285
HANDLE_LOCAL_X = HANDLE_RING_CENTER_LOCAL_X
HANDLE_RING_RADIUS = 0.105
# Aim at the centreline of the handle's frontmost bar.  Adding the bar radius
# moved the gripper 18 mm toward the panel and made it enter the loop instead
# of centring on the handle.
HANDLE_GRASP_LOCAL_X = HANDLE_RING_CENTER_LOCAL_X - HANDLE_RING_RADIUS
HANDLE_LOCAL_Z = 0.024
# The final finger-origin midpoint is object-relative:
# panel_z + panel_half_thickness + fingertip_length.  Since the target is built
# from handle_z first, subtract HANDLE_LOCAL_Z here to avoid counting the
# handle height twice.  Total local target z is therefore 4 + 54 = 58 mm.
HANDLE_INSERT_Z_OFFSET = PANEL_HALF_THICKNESS + FINGER_TIP_LENGTH - HANDLE_LOCAL_Z
# Open-jaw preload toward the robot before closing; this seats the finger behind
# the handle bar instead of closing against the panel edge and bouncing +X.
HANDLE_PRELOAD_X = 0.0
# Maximum Z descent per 15 Hz command.  This is below the 8 mm board thickness,
# so Newton generates a contact before a finger can tunnel through the board/ring.
HANDLE_DESCENT_STEP_Z = 0.030
# High clearance waypoint directly above the handle: avoids sweeping the panel
# while Pink changes the wrist orientation.
HANDLE_ABOVE_MARGIN_Z = 0.300
HANDLE_EDGE_INSET_X = 0.0
HANGER_NAIL_X = 0.55
HANGER_NAIL_Z = 1.25
HANGER_NAIL_LENGTH = 0.090
HANGER_NAIL_HEAD_THICKNESS = 0.024
HANGER_NAIL_HEAD_RADIUS = 0.040
# The vertical ring opening must clear the complete nail head during the +X
# insertion: inner radius (105-14 mm) minus the 40 mm head and 5 mm margin.
HANGER_HEAD_PASSAGE_RADIUS = HANDLE_RING_RADIUS - 0.014 - HANGER_NAIL_HEAD_RADIUS - 0.005
# After the ring clears the head, move its plane forward onto the shaft before
# releasing. The depth is derived from the physical head size, so moving or
# resizing the nail does not leave a world-coordinate waypoint behind.
HANGER_SHAFT_SEAT_DEPTH = 0.5 * HANGER_NAIL_HEAD_RADIUS
FRONT_RAIL_HEIGHT = 0.130
SIDE_RAIL_HEIGHT = 0.130
SIDE_RAIL_CENTER_Y = 0.4025

# Side fingers descend just below the lower face for a positive bite.  The
# panel centre is z=0; 10 mm remains far above the table/rails while allowing
# the pad to seat instead of catching only the upper edge.
SIDE_GRASP_Z = 0.0
SIDE_DESCENT_STEP_Z = 0.010
# The front edge is hooked, not pinched.  With the jaws closed and the jaw axis
# horizontal the hand's thin dimension is vertical, so it fits in the 124 mm gap
# under the board; it slides in past the lip, rises behind it and pulls.
# Vertical pincer clamping board+front rim (43 mm) at the front edge.  Open jaws
# clear that stack by 18 mm a side, and clamping 43 mm rather than the bare 8 mm
# board raises the grip from ~2 N to ~17 N per finger - roughly 40 N of pad friction
# against the ~6 N the 0.5 kg panel needs to slide.
# 20 mm of palm clearance, not 5: Pink leaves a ~12 mm residual on this cross-body
# reach, so a 5 mm gap means the palm rams the board edge and shoves the panel back
# before the jaws ever close.  The pads still cover 34 mm of board.
# Positive: the fingers finish *inside* the cavity under the board, past the rear
# face of the 12 mm front rim, instead of straddling the edge from outside.
FRONT_GRASP_DEPTH = 0.030
# Mid-height of the 130 mm cavity, not the board+rim mid-plane: the jaws no longer
# straddle the edge, they sit behind the rim and bear on its rear face when pulling.
FRONT_GRASP_Z = -(PANEL_HALF_THICKNESS + 0.5 * FRONT_RAIL_HEIGHT)
# Held open inside the cavity.  There is nothing 130 mm tall for the jaws to clamp,
# and nothing needs clamping - the rim's rear face is what carries the stroke.
FRONT_GRASP_WIDTH = 0.040
# Command a fully closed Panda hand.  The physical 28 mm handle collider, not
# an artificial jaw-gap margin, stops the fingers and carries the pull load.
FRONT_HANDLE_HOLD_WIDTH = 0.0
# Close quickly, then start pulling while the fingers continue squeezing.
FRONT_HANDLE_CLOSE_RATE = 0.003
FRONT_HANDLE_PULL_START_WIDTH = 0.060
FRONT_HANDLE_MIN_CLOSE_TICKS = 2
# The pull holds the bite measured at the end of phase 1 rather than a fixed width:
# the rim+board stack the jaws actually catch varies by a few mm run to run.
# The pull is carried by friction alone: the jaws close along Z and the stroke is
# along X, and a flat finger has no surface that can bear on the rim from behind.
# So grip force is the whole budget, and it has to beat the panel sliding on its
# rails - 0.5 kg at the rail friction coefficient.
#
# Squeeze hard, the way a cube grasp does.  Some penetration into the stack is how
# a soft contact model generates force at all, and is not by itself a lost grasp:
# run 934's push-away came from the rack rail blocking the lower jaw, not from the
# overclose.  Run 935 proved the other extreme - a 2 mm command gave 1.6 N against
# a 0.98 N resistance and the panel never moved.  The floor only exists so a lost
# contact cannot collapse the jaws to zero and leave them closed on air (run 933).
FRONT_STACK_HEIGHT = 2.0 * PANEL_HALF_THICKNESS + FRONT_RAIL_HEIGHT
# Overshoot below the measured bite: about 24 N at 2000 N/m, within a real Panda
# hand's 70 N and enough to hold the panel against 1 N of rail drag.  40 N with hard
# contacts threw the panel across the room on first touch; 5 N did nothing and was
# being papered over with a 2.5 friction coefficient.
FRONT_GRIP_SQUEEZE = 0.002
FRONT_GRIP_FLOOR_MINIMUM = FRONT_HANDLE_HOLD_WIDTH
# Jaw gap at which the lower pad's step drops clear of the front rim, so the arm can
# retreat without towing the panel.  The step stands 13 mm above the lower jaw's
# centre and the rim's underside sits 39 mm below the panel's mid-plane, while the
# grasp centres 17.5 mm below it, so the jaws must open past
# 2 * (0.039 - 0.0175 + 0.013) = 69 mm.  Run 942 advanced at 60 mm and the retreat
# dragged the panel 130 mm past its target, tipping it off the rack rails.
FRONT_RELEASE_WIDTH = 0.070
PULL_RELEASE_CLEARANCE_Z = 0.05
# Higher clearance used only after the finger has exited the ring and is
# retreating in X. It keeps the wrist/link sweep above the panel during the
# subsequent return to the shared default joint pose.
PULL_RETURN_CLEARANCE_Z = 0.12
FRONT_PREGRASP_STANDOFF = 0.10
FRONT_DESCENT_STANDOFF = 0.24
# Finger origins sit just outside the side edge, pads reaching in over the side rail.
SIDE_GUARD_INNER_Y = 0.540
# The target is the finger-origin midpoint, not the wrist.  Put that midpoint
# only half a fingertip length outside the panel edge: using the full 54 mm
# tip length left the measured left pad ~15 mm short of the panel face.
SIDE_PAD_REACH = 0.045
SIDE_PAD_FACE_OVERLAP = 0.008
SIDE_GRASP_FINGER_Y = PANEL_HALF_WIDTH + 0.007
SIDE_STANDOFF_LEFT_FINGER_Y = SIDE_GRASP_FINGER_Y + 0.05
SIDE_STANDOFF_RIGHT_FINGER_Y = SIDE_GRASP_FINGER_Y + 0.12
# Grasp slightly forward of the panel centre so the wrists stay clear of the
# side guards; the resulting lift moment is under 1 N.m across both grippers.
SIDE_GRASP_FORWARD = 0.0
# Opposed longitudinal grasp points, one fifth of the panel length in from each end.
# Under the +90 deg Y rotation, +local-X becomes the lower point and -local-X the upper.
SIDE_GRASP_LONGITUDINAL_OFFSET = 0.30 * PANEL_HALF_LENGTH

# The front pinch creeps along the panel under tangential load - the contact solver
# lets the pads slide well before the friction limit, delivering roughly half the
# commanded stroke and varying run to run.  Command a long stroke so even a poor run
# clears PANEL_PULL_MINIMUM; the rack still carries the panel out to x = 0.08.
# Commanded stroke, not the distance the panel travels.  A vertical clamp on a
# horizontal edge always gives up some travel to slip - 130 mm of command moved the
# panel 84 mm - so the stroke is set long enough that the panel clears the side guards
# anyway, which is what a person pulling this would do without thinking about it.
# 160 mm is the ceiling the rack allows: validate_layout() refuses anything that would
# leave the panel's centre of mass within 50 mm of the rail front if it did not slip.
PANEL_PULL_DISTANCE = 0.340
# Height the front edge is raised before the stroke starts, enough to clear the rails.
PANEL_PULL_LIFT = 0.012
PANEL_PULL_MINIMUM = 0.325
PULL_LATERAL_FEEDBACK_GAIN = 0.60
PULL_LATERAL_CORRECTION_LIMIT = 0.10
PULL_CENTER_TOLERANCE = 0.060
PULL_LATERAL_VELOCITY_GAIN = 0.18
PULL_RELEASE_MAX_LATERAL_SPEED = 0.08
PULL_RELEASE_MAX_YAW_RATE = 0.35
# How far the hand may steer sideways while cancelling panel yaw.
PANEL_PULL_YAW_LIMIT = 0.0
PULL_RAIL_CENTER_Y = 0.0
# Clear the rack rails (top 0.54) and the side guards (top 0.62) before rotating.
PANEL_LIFT_Z = 0.92  # legacy fallback; runtime targets are object-relative
ROTATION_CLEARANCE = 0.14
POST_ROTATION_LIFT_MARGIN = 0.06
HANG_NAIL_CLEARANCE = 0.02


class _ArmIK:
    def __init__(self, robot, physics_dt):
        self.robot = robot
        self.arm_joint_ids = robot.find_joints(["panda_joint.*"])[0]
        self.finger_joint_ids = robot.find_joints(["panda_finger_joint.*"])[0]
        self.hand_frame_idx = robot.find_bodies("panda_hand")[0][0]
        self.left_finger_frame_idx = robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_frame_idx = robot.find_bodies("panda_rightfinger")[0][0]
        self.physics_dt = float(physics_dt)
        joint_names = list(robot.joint_names)
        arm_joint_names = [joint_names[index] for index in self.arm_joint_ids]
        pink_cfg = PinkIKControllerCfg(
            urdf_path="/isaac-sim/exts/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf",
            mesh_path="/isaac-sim/exts/isaacsim.asset.importer.urdf/data/urdf/robots",
            articulation_name="robot",
            base_link_name="panda_link0",
            joint_names=arm_joint_names,
            all_joint_names=joint_names,
            show_ik_warnings=True,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTaskCfg(
                    frame="panda_hand",
                    position_cost=50.0,
                    orientation_cost=12.0,
                    lm_damping=0.5,
                    gain=1.0,
                ),
                NullSpacePostureTaskCfg(
                    cost=0.5,
                    gain=0.7,
                    lm_damping=0.05,
                    controlled_joints=arm_joint_names,
                ),
                DampingTaskCfg(cost=0.05),
            ],
        )
        self.controller = PinkIKController(
            pink_cfg, robot.cfg, robot.device, controlled_joint_indices=list(self.arm_joint_ids)
        )
        self.frame_task = self.controller._variable_input_tasks[0]
        self.pink_base_from_sim_world = None
        self.sim_hand_from_pink_hand = None
        self.pink_hand_from_sim_hand = None
        self.target_pose_w = None
        self.joint_override = None
        self.last_joint_target = None
        self.ik_joint_state = None
        self._finger_offset_b = None
        self.finger_bias = None
        self._bias_quat = None
        self.gripper_width = GRIPPER_OPEN_WIDTH
        self.gripper_mode = "OPEN"
        self.gripper_close_latched = False
        self.gripper_force_hold = False
        self.gripper_force_hold_position = None
        self.gripper_force_hold_effort = 0.0

    def pose_w(self):
        return self.robot.data.body_pose_w.torch[:, self.hand_frame_idx]

    def finger_centers_w(self):
        positions = self.robot.data.body_pos_w.torch[0]
        return torch.stack((positions[self.left_finger_frame_idx], positions[self.right_finger_frame_idx]))

    def finger_offset_b(self):
        """Finger-origin midpoint relative to panda_hand, in the hand frame.

        Measured once from the asset rather than taken from the URDF: the USD
        Franka does not place the finger link origins where panda_arm_hand.urdf
        does, and guessing the offset lands the jaws centimetres off the edge.
        """
        if self._finger_offset_b is None:
            pose = self.pose_w()[0]
            rotation = matrix_from_quat(pose[3:7].unsqueeze(0))[0]
            self._finger_offset_b = rotation.transpose(0, 1) @ (self.finger_centers_w().mean(dim=0) - pose[:3])
        return self._finger_offset_b

    def approach_axis(self, quaternion_w=None):
        """World direction from panda_hand toward the finger-origin midpoint."""
        quaternion = self.pose_w()[0, 3:7] if quaternion_w is None else quaternion_w
        offset = self.finger_offset_b()
        return (matrix_from_quat(quaternion.unsqueeze(0))[0] @ offset) / torch.linalg.norm(offset)

    def set_target(self, position_w, quaternion_w=None):
        current = self.pose_w()[0]
        position = torch.as_tensor(position_w, device=self.robot.device, dtype=current.dtype)
        quaternion = current[3:7] if quaternion_w is None else torch.as_tensor(
            quaternion_w, device=self.robot.device, dtype=current.dtype
        )
        self.target_pose_w = torch.cat((position, quaternion)).unsqueeze(0)

    def set_finger_target(self, midpoint_w, quaternion_w):
        """Command the panda_hand pose that puts the finger-origin midpoint at ``midpoint_w``.

        The offset is taken from the *commanded* orientation rather than the measured
        one, so the target cannot drift with the tracking error it is meant to remove.
        """
        current = self.pose_w()[0]
        quaternion = torch.as_tensor(quaternion_w, device=self.robot.device, dtype=current.dtype)
        midpoint = torch.as_tensor(midpoint_w, device=self.robot.device, dtype=current.dtype)
        rotation = matrix_from_quat(quaternion.unsqueeze(0))[0]
        if self.finger_bias is None or self._bias_quat is None or float(
            torch.linalg.norm(quaternion - self._bias_quat).item()
        ) > 1.0e-4:
            self.finger_bias = torch.zeros(3, device=self.robot.device, dtype=current.dtype)
            self._bias_quat = quaternion.clone()
        # Close the loop on the jaws, not just the wrist.  Pink leaves a systematic
        # hand-pose residual, and 15 mm of it is enough for an open jaw to clip the
        # panel edge instead of straddling it.  Integrate only once the jaws are
        # nearly there, so a long traverse cannot wind the bias up.
        residual = midpoint - self.finger_centers_w().mean(dim=0)
        if float(torch.linalg.norm(residual).item()) < 0.04:
            self.finger_bias = torch.clamp(self.finger_bias + 0.15 * residual, -0.005, 0.005)
        self.set_target(midpoint + self.finger_bias - rotation @ self.finger_offset_b(), quaternion)

    def finger_midpoint_error(self, midpoint_w):
        """Distance from the measured finger-origin midpoint to a desired one."""
        midpoint = torch.as_tensor(midpoint_w, device=self.robot.device, dtype=torch.float32)
        return float(torch.linalg.norm(midpoint - self.finger_centers_w().mean(dim=0)).item())

    def reset_finger_bias(self):
        self.finger_bias = None
        self._bias_quat = None

    def set_gripper(self, width):
        target = float(max(0.0, min(GRIPPER_MAX_WIDTH, width)))
        # CLOSING/HOLD are monotonic.  An unrelated phase call cannot reopen
        # the jaws; release_gripper() is the sole OPEN transition.
        if self.gripper_close_latched and target > self.gripper_width:
            return
        self.gripper_width = target

    def latch_gripper_close(self):
        self.gripper_close_latched = True
        self.gripper_mode = "CLOSING"

    def hold_gripper_force(
        self,
        total_force=70.0,
        preload_per_finger=0.008,
        preserve_measured_contact=False,
        position_only=False,
    ):
        """Latch the fully closed Panda position command after verified contact.

        ``panda_hand`` is already an implicit position actuator with a 70 N effort
        limit. Adding a second explicit effort target to that actuator overconstrains
        Newton's four-finger bimanual closed chain. Keep the target fully closed and
        let the configured effort limit plus rigid contact determine the real force.
        """
        self.gripper_close_latched = True
        if position_only:
            self.gripper_force_hold = False
            self.gripper_force_hold_position = None
            self.gripper_force_hold_effort = 0.0
            self.gripper_mode = "MAX_FORCE_POSITION_CLOSE"
            self.gripper_width = 0.0
            return

        # A single-arm pull is not a closed chain and remains stable with direct
        # force control. Preserve its measured contact width so it does not press
        # the horizontal panel into the rail while translating it.
        current = self.robot.data.joint_pos.torch[:, self.finger_joint_ids].detach().clone()
        contact_hold = torch.clamp(current - float(preload_per_finger), min=0.0, max=0.040)
        self.gripper_force_hold = True
        self.gripper_force_hold_position = contact_hold
        self.gripper_force_hold_effort = POLICY_GRIP_FORCE_PER_FINGER
        self.gripper_mode = "FIXED_FORCE_CLOSE"
        self.gripper_width = float(contact_hold.sum().item())

    def hold_gripper(self, width=None):
        self.gripper_close_latched = True
        self.gripper_mode = "HOLD"
        if width is not None:
            self.set_gripper(width)

    def ramp_gripper_force_hold(self):
        """Compatibility no-op: the implicit Panda actuator owns grip force."""
        return

    def release_gripper(self):
        self.gripper_force_hold = False
        self.gripper_force_hold_position = None
        self.gripper_force_hold_effort = 0.0
        self.gripper_close_latched = False
        self.gripper_mode = "OPEN"
        self.gripper_width = GRIPPER_OPEN_WIDTH

    def open_step(self, rate=0.008):
        """Open from the measured jaw width without kicking a trapped object."""
        if self.gripper_mode != "OPENING":
            measured = max(0.0, min(GRIPPER_MAX_WIDTH, self.measured_gripper_width()))
            self.gripper_force_hold = False
            self.gripper_force_hold_position = None
            self.gripper_force_hold_effort = 0.0
            self.gripper_close_latched = False
            self.gripper_width = measured
            self.gripper_mode = "OPENING"
        self.gripper_width = min(GRIPPER_OPEN_WIDTH, self.gripper_width + float(rate))
        if self.gripper_width >= GRIPPER_OPEN_WIDTH:
            self.gripper_mode = "OPEN"

    def close_step(self, rate=GRIPPER_CLOSE_RATE, floor=0.0):
        """Ramp the jaws shut instead of commanding zero outright.

        A single ``set_gripper(0.0)`` moves the jaws the whole 80 mm stroke inside
        one control tick - about 7 mm per physics step - and they tunnel straight
        through the 12 mm board before a contact is ever generated, then settle
        *inside* the panel.  ``velocity_limit_sim`` does not bound this.  Ramping at
        ~0.09 m/s keeps every step well under the thinnest feature being grasped.
        """
        if self.gripper_force_hold:
            return
        if not self.gripper_close_latched:
            self.latch_gripper_close()
        self.set_gripper(max(floor, self.gripper_width - rate))

    def set_task_costs(self, position_cost, orientation_cost):
        self.frame_task.set_position_cost(float(position_cost))
        self.frame_task.set_orientation_cost(float(orientation_cost))

    def measured_gripper_width(self):
        return float(self.robot.data.joint_pos.torch[0, self.finger_joint_ids].sum().item())

    def set_joint_override(self, target):
        self.joint_override = torch.as_tensor(target, device=self.robot.device).reshape(1, -1).clone()

    def clear_joint_override(self):
        self.joint_override = None

    def joint_tracking_error(self):
        if self.joint_override is None:
            return math.inf
        current = self.robot.data.joint_pos.torch[:, self.arm_joint_ids]
        return float(torch.linalg.norm(self.joint_override - current).item())

    def tracking_error(self):
        if self.target_pose_w is None:
            return math.inf
        return float(torch.linalg.norm(self.target_pose_w[0, :3] - self.pose_w()[0, :3]).item())

    def tracking_pose_error(self):
        if self.target_pose_w is None:
            return math.inf, math.inf
        pose = self.pose_w()
        position_error, rotation_error = compute_pose_error(
            pose[:, :3], pose[:, 3:7], self.target_pose_w[:, :3], self.target_pose_w[:, 3:7]
        )
        return float(torch.linalg.norm(position_error[0]).item()), float(torch.linalg.norm(rotation_error[0]).item())

    def apply(self, solve_ik=True):
        if self.target_pose_w is None:
            self.target_pose_w = self.pose_w().clone()
        if not solve_ik and self.last_joint_target is not None:
            joint_target = self.last_joint_target
        elif self.joint_override is not None:
            joint_target = self.joint_override
        else:
            measured_joint_pos = self.robot.data.joint_pos.torch[0].detach().cpu().numpy().astype(np.float64)
            if self.ik_joint_state is None:
                self.ik_joint_state = measured_joint_pos.copy()
            else:
                uncontrolled = np.ones(len(measured_joint_pos), dtype=bool)
                uncontrolled[np.asarray(self.arm_joint_ids, dtype=np.int64)] = False
                self.ik_joint_state[uncontrolled] = measured_joint_pos[uncontrolled]
            base_pose = self.robot.data.root_pose_w.torch[0]
            world_from_base = pin.SE3(
                matrix_from_quat(base_pose[3:7].unsqueeze(0))[0].detach().cpu().numpy().astype(np.float64),
                base_pose[:3].detach().cpu().numpy().astype(np.float64),
            )
            # Calibrate the fixed URDF-panda_hand to USD-panda_hand frame
            # transform at the measured joint configuration.
            if self.pink_hand_from_sim_hand is None:
                pink_q = self.ik_joint_state[self.controller.isaac_lab_to_pink_ordering]
                self.controller.pink_configuration.update(pink_q)
                base_from_pink_hand = self.controller.pink_configuration.get_transform_frame_to_world("panda_hand")
                sim_hand_pose = self.pose_w()[0]
                world_from_sim_hand = pin.SE3(
                    matrix_from_quat(sim_hand_pose[3:7].unsqueeze(0))[0].detach().cpu().numpy().astype(np.float64),
                    sim_hand_pose[:3].detach().cpu().numpy().astype(np.float64),
                )
                base_from_sim_hand = world_from_base.inverse() * world_from_sim_hand
                self.pink_hand_from_sim_hand = base_from_pink_hand.inverse() * base_from_sim_hand

            target_sim = pin.SE3(
                matrix_from_quat(self.target_pose_w[:, 3:7])[0].detach().cpu().numpy().astype(np.float64),
                self.target_pose_w[0, :3].detach().cpu().numpy().astype(np.float64),
            )
            # Convert the desired USD hand frame into Pink hand coordinates.
            base_from_sim_target = world_from_base.inverse() * target_sim
            base_from_pink_target = base_from_sim_target * self.pink_hand_from_sim_hand.inverse()
            self.frame_task.set_target(base_from_pink_target)
            solved_target = self.controller.compute(self.ik_joint_state, self.physics_dt).detach().cpu().numpy()
            measured_arm = measured_joint_pos[np.asarray(self.arm_joint_ids, dtype=np.int64)]
            bounded_target = np.clip(solved_target, measured_arm - IK_STEP_LIMIT, measured_arm + IK_STEP_LIMIT)
            self.ik_joint_state[np.asarray(self.arm_joint_ids, dtype=np.int64)] = bounded_target
            joint_target = torch.tensor(bounded_target, device=self.robot.device, dtype=torch.float32).unsqueeze(0)
        measured_arm_tensor = self.robot.data.joint_pos.torch[:, self.arm_joint_ids]
        tracking_delta = torch.clamp(
            IK_TRACKING_GAIN * (joint_target - measured_arm_tensor), -IK_TRACKING_LIMIT, IK_TRACKING_LIMIT
        )
        joint_target = measured_arm_tensor + tracking_delta
        self.robot.set_joint_position_target_index(target=joint_target, joint_ids=self.arm_joint_ids)
        self.last_joint_target = joint_target.detach().clone()
        if self.gripper_force_hold:
            finger_target = self.gripper_force_hold_position
            finger_effort = torch.full(
                (1, len(self.finger_joint_ids)),
                -self.gripper_force_hold_effort,
                device=self.robot.device,
                dtype=joint_target.dtype,
            )
        else:
            finger_target = torch.full(
                (1, len(self.finger_joint_ids)),
                0.5 * self.gripper_width,
                device=self.robot.device,
                dtype=joint_target.dtype,
            )
            finger_effort = torch.zeros_like(finger_target)
        self.robot.set_joint_position_target_index(target=finger_target, joint_ids=self.finger_joint_ids)
        self.robot.set_joint_effort_target_index(target=finger_effort, joint_ids=self.finger_joint_ids)


class DualFrankaAutoOps:
    """Event-gated pick→hang phase controller sampled at 15 Hz."""

    def __init__(
        self,
        left_robot,
        right_robot,
        panel,
        control_decimation=8,
        physics_dt=1.0 / 120.0,
        motion_speed_scale=4.0,
        hang_position=(0.452, 0.0, 0.820),
        hang_entry_z=0.90,
        task_mode="long",
    ):
        self.left = _ArmIK(left_robot, physics_dt)
        self.right = _ArmIK(right_robot, physics_dt)
        self.panel = panel
        self.control_decimation = int(control_decimation)
        self.motion_speed_scale = float(motion_speed_scale)
        self.hang_position = tuple(float(value) for value in hang_position)
        self.hang_entry_z = float(hang_entry_z)
        self.task_mode = str(task_mode)
        self.orchestrator = LongTaskOrchestrator(self.task_mode)
        self.active_module = self.orchestrator.current
        self.stop_after_phase = None if self.task_mode == "long" else self.active_module.end_phase
        self.phase_index = self.active_module.start_phase
        self.pull_handoff_minimum = PANEL_PULL_MINIMUM
        self.pull_handoff_lateral_tolerance = PULL_CENTER_TOLERANCE
        self.phase_ticks = 0
        self.total_control_ticks = 0
        self.done = False
        self.success = False
        self.failure_reason = None
        self.events = []
        self.initialized = False
        self.left_hold_pose = None
        self.right_hold_pose = None
        self.left_hold_joint_target = None
        self.right_hold_joint_target = None
        self.front_grasp_quat = None
        self.left_side_grasp_quat = None
        self.left_panel_side_sign = None
        self.right_side_grasp_quat = None
        self.initial_panel_pose = None
        self.initial_panel_pos = None
        self.pregrasp_handle_world = None
        self.pull_start_pose = None
        self.pull_phase_start_panel_x = None
        self.pull_command_distance = PANEL_PULL_DISTANCE
        self.pull_success_stable_ticks = 0
        self.front_close_start_panel_x = None
        self.side_close_tick = None
        self.side_alignment_stable_ticks = 0
        self.rotation_alignment_stable_ticks = 0
        self.rotation_center_x = None
        self.rotation_center_y = RACK_CENTER_Y
        self.side_grasp_y_offset = SIDE_GRASP_FINGER_Y
        self.side_preclose_stable_ticks = 0
        self.side_final_approach_progress = 0.0
        self.side_mid_left = None
        self.side_mid_right = None
        self.side_mid_left_quat = None
        self.side_mid_right_quat = None
        self.side_mid_stable_ticks = 0
        self.side_grasp_attempts = 0
        self.side_reanchor_attempts = 0
        self.retreat_start_pose = None
        self.retreat_start_finger_midpoint = None
        self.front_close_tick = None
        self.front_alignment_stable_ticks = 0
        self.front_contact_stable_ticks = 0
        self.front_grasp_attempts = 0
        self.front_grasp_reference = None
        self.front_grip_floor = FRONT_GRIP_FLOOR_MINIMUM
        self.side_grasp_reference = None
        self.release_hand_pose = {}
        # Constant panel->hand transforms captured when the bimanual grasp closes.
        self.grasp_offsets = None
        self.grasp_finger_locals = None
        self.lift_probe_start_z = None
        self.grasp_panel_pose = None
        self.rotation_safe_z = None
        self.lift_target_z = None
        self.vertical_support_pose = None
        self.waypoint_start = None
        self.panel_motion_progress = 0.0
        self.above_handle_waypoint_start = None
        self.panel_height_history = deque(maxlen=6)
        self.stage = 0
        self.stage_tick = 0
        self.stage_best_error = math.inf
        self.stage_best_tick = 0
        self.contact_names_logged = False
        self.hang_grasp_ready = self.task_mode != "hang"
        self.hang_bootstrap_stage = 0
        self.hang_bootstrap_stable_ticks = 0
        self.hang_bootstrap_close_tick = None
        self.last_action = np.zeros(14, dtype=np.float32)

    # ------------------------------------------------------------------ utils

    def _duration(self, nominal_ticks):
        return max(1, int(round(float(nominal_ticks) / self.motion_speed_scale)))

    def _panel_pos(self):
        return self.panel.data.root_pos_w.torch[0]

    def _panel_pose(self):
        """Panel pose in the controller's legacy internal wxyz convention."""
        pose = self.panel.data.root_pose_w.torch[0]
        return torch.cat((pose[:3], pose[[6, 3, 4, 5]]))

    def _panel_rotation(self, panel_pose):
        """Rotation matrix for a legacy-wxyz panel pose.

        ``matrix_from_quat`` consumes native xyzw in this Newton build. Keep
        controller poses in their established wxyz convention, and perform the
        conversion at this single matrix boundary.
        """
        quat_xyzw = panel_pose[[4, 5, 6, 3]]
        return matrix_from_quat(quat_xyzw.reshape(1, 4))[0]

    def _ring_center_world(self, panel_pose=None):
        """Actual handle-ring center from the live panel transform."""
        if panel_pose is None:
            panel_pose = self._panel_pose()
        rotation = self._panel_rotation(panel_pose)
        local = torch.tensor((HANDLE_LOCAL_X, 0.0, HANDLE_LOCAL_Z), device=panel_pose.device, dtype=panel_pose.dtype)
        return panel_pose[:3] + rotation @ local

    def _ring_target_world(self, above=False):
        """Nail-centred ring target derived from the authored hanger target."""
        target = torch.tensor((
            self.hang_position[0] + HANDLE_LOCAL_Z,
            self.hang_position[1],
            self.hang_position[2] + abs(HANDLE_LOCAL_X) + HANDLE_RING_RADIUS,
        ), device=self.left.robot.device, dtype=torch.float32)
        if above:
            target[2] += HANG_NAIL_CLEARANCE
        return target

    def _hang_panel_target(self):
        """Panel root target whose upper handle-ring centre intersects the nail."""
        return torch.tensor((
            self.hang_position[0],
            self.hang_position[1],
            self.hang_position[2] + HANDLE_RING_RADIUS,
        ), device=self.left.robot.device, dtype=torch.float32)

    def _event(self, name, **values):
        values.setdefault("task_module", self.active_module.name)
        self.events.append({"control_tick": self.total_control_ticks, "phase_index": self.phase_index, "event": name, **values})
        if name in {
            "auto_ops_started",
            "phase_transition",
            "stage_transition",
            "left_front_gripper_close_commanded",
            "front_pincer_closed_around_edge",
            "front_physical_contact_verified",
            "object_relative_side_waypoints",
            "bimanual_physical_contact_verified",
            "episode_finished",
        }:
            print(f"[AUTO_OPS] tick={self.total_control_ticks} phase={self.phase_index} event={name} values={values}", flush=True)

    def _transition(self, phase_index, reason):
        self._event("phase_transition", next_phase=phase_index, reason=reason)
        self.phase_index = phase_index
        self.phase_ticks = 0
        self._begin_stage(0)
        if not self.done and self.stop_after_phase is not None and phase_index > self.stop_after_phase:
            self.success = True
            self.failure_reason = None
            self.done = True
            self._event(
                "episode_finished",
                success=True,
                task_mode=self.task_mode,
                completed_phase=self.stop_after_phase,
            )

    def _begin_stage(self, stage, reason=None):
        if reason is not None:
            self._event("stage_transition", next_stage=stage, reason=reason)
        self.stage = int(stage)
        self.stage_tick = self.phase_ticks
        self.stage_best_error = math.inf
        self.stage_best_tick = self.phase_ticks
        self.panel_motion_progress = 0.0
        if self.phase_index == 0 and self.stage != 0:
            self.above_handle_waypoint_start = None
        try:
            handoff = self.orchestrator.sync(self)
        except ModuleContractError as error:
            self._fail("module_handoff_contract_failed", contract_error=str(error))
            return
        self.active_module = self.orchestrator.current
        if self.orchestrator.completed and not self.done:
            self.success = True
            self.failure_reason = None
            self.done = True
            self._event(
                "episode_finished",
                success=True,
                task_mode=self.task_mode,
                completed_module=self.active_module.name,
            )
            return
        if handoff is not None:
            self._event(
                "task_module_handoff",
                source_module=handoff.source,
                target_module=handoff.target,
                panel_pose=list(handoff.panel_pose),
                left_gripper_width=handoff.left_gripper_width,
                right_gripper_width=handoff.right_gripper_width,
            )

    def _stage_elapsed(self):
        return self.phase_ticks - self.stage_tick

    def _fail(self, reason, **values):
        left_position_error, left_rotation_error = self.left.tracking_pose_error()
        right_position_error, right_rotation_error = self.right.tracking_pose_error()
        values.setdefault("stage", self.stage)
        values.setdefault("left_position_error", left_position_error)
        values.setdefault("left_rotation_error", left_rotation_error)
        values.setdefault("right_position_error", right_position_error)
        values.setdefault("right_rotation_error", right_rotation_error)
        values.setdefault("left_gripper_width", self.left.measured_gripper_width())
        values.setdefault("right_gripper_width", self.right.measured_gripper_width())
        values.setdefault("panel_pose", self._panel_pose().detach().cpu().tolist())
        values.setdefault("left_finger_centers", self.left.finger_centers_w().detach().cpu().tolist())
        values.setdefault("right_finger_centers", self.right.finger_centers_w().detach().cpu().tolist())
        values.setdefault("contact_flags", self._newton_contact_flags())
        self.done = True
        self.success = False
        self.failure_reason = reason
        self._event("episode_finished", success=False, failure_reason=reason, **values)

    def _pose_ready(self, arm, position_tolerance, rotation_tolerance, patience=None):
        """Gate on position *and* rotation separately.

        A single blended scalar lets a 20 degree wrist error hide behind a good
        position, and the tilted open jaws then shove the panel on the way in.
        """
        position_error, rotation_error = arm.tracking_pose_error()
        if position_error <= position_tolerance and rotation_error <= rotation_tolerance:
            return True
        return self._stage_ready(position_error + 0.10 * rotation_error, -1.0, patience)

    def _stage_ready(self, error, tolerance, patience=None):
        """Advance when the target is reached, or once IK tracking stops improving.

        Pink leaves a bounded residual on cross-body reaches, so a pure tolerance
        gate deadlocks the script.  Stalling for ``patience`` ticks is treated as
        "this is as close as the arm gets" and the phase moves on.
        """
        if error <= tolerance:
            return True
        if error < self.stage_best_error - 0.0015:
            self.stage_best_error = error
            self.stage_best_tick = self.phase_ticks
            return False
        patience = self._duration(50) if patience is None else patience
        return self.phase_ticks - self.stage_best_tick >= max(int(patience), 12)

    def _newton_contact_flags(self):
        """Return real pad-panel and panel-hanger contacts from Newton/MuJoCo."""
        flags = {"left_pad_panel": False, "right_pad_panel": False, "panel_hanger": False}
        try:
            import mujoco
            from isaaclab_newton.physics.newton_manager import NewtonManager

            solver = NewtonManager._solver
            observed = set()
            for contact_index in range(int(solver.mj_data.ncon)):
                contact = solver.mj_data.contact[contact_index]
                geom1 = mujoco.mj_id2name(solver.mj_model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or ""
                geom2 = mujoco.mj_id2name(solver.mj_model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or ""
                pair = geom1 + "|" + geom2
                observed.add(pair)
                is_panel = "Panel" in pair
                if is_panel and ("leftfinger" in pair or "rightfinger" in pair):
                    if "FrankaLeft" in pair:
                        flags["left_pad_panel"] = True
                    elif "FrankaRight" in pair:
                        flags["right_pad_panel"] = True
                    else:
                        # Newton may flatten the prim path; fall back to both.
                        flags["left_pad_panel"] = True
                        flags["right_pad_panel"] = True
                if is_panel and ("HangerStand" in pair or "SHook" in pair or "RoundSegment" in pair):
                    flags["panel_hanger"] = True
            if not self.contact_names_logged and observed:
                self.contact_names_logged = True
                print(f"[AUTO_OPS_CONTACT_NAMES] {sorted(observed)[:24]}", flush=True)
        except Exception:
            pass
        return flags

    def _grasp_width_valid(self, arm, low=0.028, high=0.052):
        return low <= arm.measured_gripper_width() <= high

    def _bimanual_grasp_is_valid(self):
        """Validate physical enclosure in the live panel frame.

        MJWarp contact names are not the live contact buffer in this build, so use
        measured jaw stall plus both finger midpoints straddling the panel faces.
        """
        panel_pose = self._panel_pose()
        rotation_inv = self._panel_rotation(panel_pose).transpose(0, 1)
        left_local = rotation_inv @ (self.left.finger_centers_w().mean(dim=0) - panel_pose[:3])
        right_local = rotation_inv @ (self.right.finger_centers_w().mean(dim=0) - panel_pose[:3])
        widths_valid = self._grasp_width_valid(self.left, 0.025, 0.048) and self._grasp_width_valid(self.right, 0.025, 0.048)
        faces_valid = (
            left_local[1] * right_local[1] < 0.0
            and abs(float(left_local[1].item())) <= PANEL_HALF_WIDTH + SIDE_PAD_REACH
            and abs(float(right_local[1].item())) <= PANEL_HALF_WIDTH + SIDE_PAD_REACH
            and abs(float(left_local[0].item())) <= PANEL_HALF_LENGTH
            and abs(float(right_local[0].item())) <= PANEL_HALF_LENGTH
            and abs(float(left_local[2].item())) <= 0.04
            and abs(float(right_local[2].item())) <= 0.04
        )
        return widths_valid and faces_valid

    def _bimanual_panel_symmetry_error(self):
        """Return finger-midpoint asymmetry measured in the live panel frame."""
        panel_pose = self._panel_pose()
        inverse_rotation = self._panel_rotation(panel_pose).transpose(0, 1)
        left_local = inverse_rotation @ (self.left.finger_centers_w().mean(dim=0) - panel_pose[:3])
        right_local = inverse_rotation @ (self.right.finger_centers_w().mean(dim=0) - panel_pose[:3])
        return max(
            abs(float(left_local[0].item() + right_local[0].item())),
            abs(float(left_local[2].item() - right_local[2].item())),
            abs(float(left_local[1].item() + right_local[1].item())),
        )

    # ------------------------------------------------------- rigid grasp replay

    def _capture_grasp_offsets(
        self, update_clearance=True, diagonal_realign=False, preserve_measured=False
    ):
        """Freeze the panel->hand transforms so both wrists move as one rigid body."""
        panel_pose = self._panel_pose()
        panel_pos = panel_pose[:3].unsqueeze(0)
        panel_quat = panel_pose[3:7].unsqueeze(0)
        panel_inverse_rotation = self._panel_rotation(panel_pose).transpose(0, 1)
        left_finger_local = panel_inverse_rotation @ (
            self.left.finger_centers_w().mean(dim=0) - panel_pose[:3]
        )
        right_finger_local = panel_inverse_rotation @ (
            self.right.finger_centers_w().mean(dim=0) - panel_pose[:3]
        )
        self.grasp_finger_locals = {
            "left": left_finger_local.clone(),
            "right": right_finger_local.clone(),
        }
        measured_side_distance = 0.5 * (
            abs(float(left_finger_local[1].item())) + abs(float(right_finger_local[1].item()))
        )
        # Keep the finger origins as close as the physical palm clearance permits.
        # This maximizes pad overlap and is re-applied from the live panel pose every lift tick.
        self.side_grasp_y_offset = max(
            PANEL_HALF_WIDTH + PALM_CLEARANCE, measured_side_distance - 0.012
        )
        offsets = {}
        for name, arm in (("left", self.left), ("right", self.right)):
            # Capture the measured physical contact pose.  Reusing the ideal IK
            # target introduces a pose jump at lift start and shears the board out.
            hand = arm.pose_w()[0]
            offset_pos, offset_quat = subtract_frame_transforms(
                panel_pos, panel_quat, hand[:3].unsqueeze(0), hand[3:7].unsqueeze(0)
            )
            offsets[name] = (offset_pos, offset_quat)

        # Remove grasp-time skew without inventing world-frame targets.  Preserve
        # the measured reach, but mirror both wrist positions about the live panel
        # centre in panel-local coordinates: equal x/z and opposite equal-magnitude y.
        left_pos, left_quat = offsets["left"]
        right_pos, right_quat = offsets["right"]
        if not preserve_measured:
            if diagonal_realign:
                half_x = 0.5 * (torch.abs(left_pos[:, 0:1]) + torch.abs(right_pos[:, 0:1]))
                left_x = torch.sign(left_pos[:, 0:1]) * half_x
                right_x = -torch.sign(left_pos[:, 0:1]) * half_x
            else:
                common_x = 0.5 * (left_pos[:, 0:1] + right_pos[:, 0:1])
                left_x = common_x
                right_x = common_x
            common_z = 0.5 * (left_pos[:, 2:3] + right_pos[:, 2:3])
            half_span = 0.5 * (torch.abs(left_pos[:, 1:2]) + torch.abs(right_pos[:, 1:2]))
            left_sign = torch.sign(left_pos[:, 1:2])
            right_sign = -left_sign
            offsets["left"] = (torch.cat((left_x, left_sign * half_span, common_z), dim=1), left_quat)
            offsets["right"] = (torch.cat((right_x, right_sign * half_span, common_z), dim=1), right_quat)
        self.grasp_offsets = offsets
        self.grasp_panel_pose = panel_pose.clone()
        if update_clearance:
            self.rotation_safe_z = float(panel_pose[2].item()) + PANEL_HALF_LENGTH + ROTATION_CLEARANCE
            self.lift_target_z = max(
                self.rotation_safe_z, HANGER_NAIL_Z - (abs(HANDLE_LOCAL_X) + HANDLE_RING_RADIUS)
            )

    def _grasp_slip_correction(self, name, arm, limit=0.003):
        """Right-hand-only local-X re-seat during the 45--60 degree rotation window."""
        zero = torch.zeros(3, device=self.left.robot.device)
        if self.grasp_finger_locals is None or name != "right":
            return zero
        live_pose = self._panel_pose()
        live_rotation = self._panel_rotation(live_pose)
        vertical_alignment = torch.clamp(torch.abs(live_rotation[2, 0]), 0.0, 1.0)
        angle_deg = torch.rad2deg(torch.asin(vertical_alignment))
        # Smoothly enter over 45--48 deg, hold through 57 deg, and leave by 60 deg.
        ramp_in = torch.clamp((angle_deg - 45.0) / 3.0, 0.0, 1.0)
        ramp_out = torch.clamp((60.0 - angle_deg) / 3.0, 0.0, 1.0)
        correction_scale = torch.minimum(ramp_in, ramp_out)
        if float(correction_scale.item()) <= 0.0:
            return zero
        current_mid_local = live_rotation.transpose(0, 1) @ (
            arm.finger_centers_w().mean(dim=0) - live_pose[:3]
        )
        local_error = self.grasp_finger_locals[name] - current_mid_local
        # Re-seat only along the panel long axis. Lateral/normal corrections
        # overconstrain the closed bimanual chain and previously produced NaNs.
        local_correction = torch.stack((local_error[0], local_error[1] * 0.0, local_error[2] * 0.0))
        local_correction = torch.clamp(local_correction, min=-limit, max=limit)
        return (live_rotation @ local_correction) * correction_scale

    def _drive_panel_pose(self, position, quaternion, rotate_grasp=False):
        """Keep both hands centred on the live panel while nudging it toward the goal."""
        desired_pos = torch.as_tensor(position, device=self.left.robot.device, dtype=torch.float32).reshape(1, 3)
        desired_quat = torch.as_tensor(quaternion, device=self.left.robot.device, dtype=torch.float32).reshape(1, 4)
        live_pose = self._panel_pose()
        live_pos = live_pose[:3].reshape(1, 3)
        live_quat = live_pose[3:7].reshape(1, 4)
        live_rotation = self._panel_rotation(live_pose)

        # Command the planned object pose itself. Anchoring this target to the live
        # panel created a deadlock: when grasp contact failed to move the panel, both
        # wrists were forever commanded only 4 mm above it and never performed the
        # lift. The trajectory is already rate-limited by shared progress, and the
        # captured panel-to-hand transforms keep both wrists in one closed chain.
        panel_pos = desired_pos
        panel_quat = desired_quat
        # Replay the single measured panel-to-hand closed chain.  The diagonal
        # local-X offsets are preserved; no stage is allowed to recenter them.
        for name, arm in (("left", self.left), ("right", self.right)):
            offset_pos, offset_quat = self.grasp_offsets[name]
            hand_pos, hand_quat = combine_frame_transforms(
                panel_pos, panel_quat, offset_pos, offset_quat
            )
            arm.set_task_costs(100.0, 80.0)
            arm.set_target(hand_pos[0], hand_quat[0])
            arm.close_step()


    def _drive_arm_from_panel(self, name, position, quaternion):
        """Drive one supporting hand from a desired panel pose."""
        arm = self.left if name == "left" else self.right
        panel_pos = torch.as_tensor(
            position, device=self.left.robot.device, dtype=torch.float32
        ).reshape(1, 3)
        panel_quat = torch.as_tensor(
            quaternion, device=self.left.robot.device, dtype=torch.float32
        ).reshape(1, 4)
        offset_pos, offset_quat = self.grasp_offsets[name]
        hand_pos, hand_quat = combine_frame_transforms(
            panel_pos, panel_quat, offset_pos, offset_quat
        )
        arm.set_target(hand_pos[0], hand_quat[0])
        arm.close_step()

    def _vertical_regrasp_targets(self, reference, y_offset):
        """Return symmetric mid-edge targets expressed in the live vertical panel."""
        rotation = self._panel_rotation(reference)
        device, dtype = reference.device, reference.dtype
        left_sign = self.left_panel_side_sign
        right_sign = -left_sign
        # With the verified +90-deg-Y Newton pose, local +X maps to the lower
        # world-Z grasp and local -X to the upper grasp.
        local_left = torch.tensor(
            (SIDE_GRASP_LONGITUDINAL_OFFSET, left_sign * y_offset, 0.0), device=device, dtype=dtype
        )
        local_right = torch.tensor(
            (-SIDE_GRASP_LONGITUDINAL_OFFSET, right_sign * y_offset, 0.0), device=device, dtype=dtype
        )
        left_target = reference[:3] + rotation @ local_left
        right_target = reference[:3] + rotation @ local_right
        # Resolve active/passive quaternion convention from the resulting world
        # geometry, not a hard-coded local-axis sign. Hang inherits the verified
        # lift ordering: left hand lower, right hand upper.
        if float(left_target[2].item()) >= float(right_target[2].item()):
            local_left[0] = -local_left[0]
            local_right[0] = -local_right[0]
            left_target = reference[:3] + rotation @ local_left
            right_target = reference[:3] + rotation @ local_right
        left_base = (
            self.left_side_grasp_quat if left_sign > 0.0
            else self.right_side_grasp_quat
        )
        right_base = (
            self.left_side_grasp_quat if right_sign > 0.0
            else self.right_side_grasp_quat
        )
        left_quat = quat_mul(
            reference[3:7].unsqueeze(0), left_base.unsqueeze(0)
        )[0]
        right_quat = quat_mul(
            reference[3:7].unsqueeze(0), right_base.unsqueeze(0)
        )[0]
        return left_target, right_target, left_quat, right_quat


    def _synchronized_bimanual_progress(
        self, duration, position_tolerance=0.010, rotation_tolerance=0.06
    ):
        """Advance a shared panel trajectory only when both arms track it.

        This is the command-level closed-chain constraint above the two joint
        actuators: a faster arm cannot advance the object trajectory while its
        partner is lagging.
        """
        errors = (
            self.left.tracking_pose_error(),
            self.right.tracking_pose_error(),
        )
        worst_position = max(error[0] for error in errors)
        worst_rotation = max(error[1] for error in errors)
        position_quality = position_tolerance / max(worst_position, position_tolerance)
        rotation_quality = rotation_tolerance / max(worst_rotation, rotation_tolerance)
        # A rigid grasp may not outrun either arm.  Freeze object progress while
        # either wrist is outside the closed-chain tracking envelope; both Pink
        # solvers continue converging to the same frozen object pose.
        if worst_position > 0.025 or worst_rotation > 0.12:
            progress_scale = 0.0
        else:
            progress_scale = min(1.0, position_quality, rotation_quality)
        self.panel_motion_progress = min(
            1.0, self.panel_motion_progress + progress_scale / max(float(duration), 1.0)
        )
        return self.panel_motion_progress

    def _waypoint(self, target_position, target_quaternion, nominal_ticks, floor_ticks):
        """Interpolate the panel from the stage start pose to a goal; return the progress."""
        if self.waypoint_start is None:
            self.waypoint_start = (
                torch.as_tensor(self.grasp_panel_pose[:3], device=self.left.robot.device).clone(),
                torch.as_tensor(self.grasp_panel_pose[3:7], device=self.left.robot.device).clone(),
            )
        duration = max(self._duration(nominal_ticks), floor_ticks)
        fraction = min(1.0, self._stage_elapsed() / duration)
        smooth = 0.5 - 0.5 * math.cos(math.pi * fraction)
        start_pos, start_quat = self.waypoint_start
        goal_pos = torch.as_tensor(target_position, device=start_pos.device, dtype=start_pos.dtype)
        position = (1.0 - smooth) * start_pos + smooth * goal_pos
        if target_quaternion is None:
            quaternion = start_quat
        else:
            quaternion = target_quaternion
        self._drive_panel_pose(position, quaternion)
        return fraction

    def _begin_waypoint(self, stage, reason, position, quaternion):
        self._begin_stage(stage, reason)
        self.waypoint_start = (position.clone(), quaternion.clone())

    def _vertical_quat(self, fraction=1.0):
        """Panel orientation after rotating ``fraction`` of +90 deg about world Y."""
        device = self.left.robot.device
        # Staging is horizontal, so standing the panel up is a
        # -90 deg roll from there; the frame ends on the hanger side (+X) and the
        # smooth face looks back at the robot and the front camera.
        rotation = quat_from_angle_axis(
            torch.tensor([0.5 * math.pi * float(fraction)], device=device),
            torch.tensor([[0.0, 1.0, 0.0]], device=device),
        )
        return quat_mul(rotation, self.grasp_panel_pose[3:7].unsqueeze(0))[0]

    def _hang_transport_quat(self):
        """Vertical orientation used after lift or by standalone hang."""
        if self.task_mode == "hang":
            return self.grasp_panel_pose[3:7]
        return self._vertical_quat(1.0)

    def start_hang_episode_from_current_hold(self):
        """Continue a physically completed lift as standalone hang tick 0."""
        if not self.success or self.grasp_offsets is None:
            raise RuntimeError("Hang continuation requires a successful physical lift hold")
        # The panel->hand transforms captured before lift rotation are rigid-body
        # transforms and therefore remain valid after rotation.  Re-measuring
        # each hand separately here bakes tiny tracking/contact errors into a new
        # overconstrained closed chain and stalls Newton at the first hang move.
        # Keep the verified offsets and update only the object's vertical pose.
        self.grasp_panel_pose = self._panel_pose().clone()
        self.task_mode = "hang"
        self.orchestrator = LongTaskOrchestrator("hang")
        self.active_module = self.orchestrator.current
        self.stop_after_phase = self.active_module.end_phase
        self.phase_index = self.active_module.start_phase
        self.phase_ticks = 0
        self.total_control_ticks = 0
        # Preserve the final verified lift commands for a few physics steps.
        # Re-solving both wrists in a newly captured object frame on the very
        # first handoff step overconstrains Newton's already-closed chain.
        self.stage = -1
        self.stage_tick = 0
        self.done = False
        self.success = False
        self.failure_reason = None
        self.events.clear()
        self.waypoint_start = None
        self.panel_motion_progress = 0.0
        self.panel_height_history.clear()
        self.hang_grasp_ready = True
        self._event(
            "auto_ops_started_from_completed_lift_hold",
            left_gripper_width=self.left.measured_gripper_width(),
            right_gripper_width=self.right.measured_gripper_width(),
            panel_pose=self._panel_pose().detach().cpu().tolist(),
        )

    # --------------------------------------------------------------- lifecycle

    def initialize(self):
        left_pose = self.left.pose_w()[0].clone()
        right_pose = self.right.pose_w()[0].clone()
        self.left_hold_pose = left_pose.clone()
        self.right_hold_pose = right_pose.clone()
        self.left_hold_joint_target = self.left.robot.data.default_joint_pos.torch[:, self.left.arm_joint_ids].clone()
        self.right_hold_joint_target = self.right.robot.data.default_joint_pos.torch[:, self.right.arm_joint_ids].clone()
        self.initial_panel_pose = self._panel_pose().clone()
        self.initial_panel_pos = self.initial_panel_pose[:3].clone()
        initial_rotation = self._panel_rotation(self.initial_panel_pose)
        initial_inv_rotation = initial_rotation.transpose(0, 1)
        initial_left_y = float((initial_inv_rotation @ (self.left.finger_centers_w().mean(dim=0) - self.initial_panel_pos))[1].item())
        initial_right_y = float((initial_inv_rotation @ (self.right.finger_centers_w().mean(dim=0) - self.initial_panel_pos))[1].item())
        self.left_panel_side_sign = 1.0 if initial_left_y >= initial_right_y else -1.0
        # Resolve the authored handle from the measured Newton panel root pose.
        # These exact world coordinates drive every grasp waypoint.
        self.handle_world = torch.tensor((
            float(self.initial_panel_pos[0].item()) + HANDLE_GRASP_LOCAL_X,
            float(self.initial_panel_pos[1].item()),
            float(self.initial_panel_pos[2].item()) + HANDLE_LOCAL_Z,
        ), device=left_pose.device, dtype=left_pose.dtype)
        self.handle_above_world = self.handle_world + torch.tensor(
            (0.0, 0.0, HANDLE_ABOVE_MARGIN_Z), device=left_pose.device, dtype=left_pose.dtype)
        self.handle_entry_world = self.handle_world + torch.tensor(
            (0.0, 0.0, HANDLE_INSERT_Z_OFFSET), device=left_pose.device, dtype=left_pose.dtype)
        self.pull_end_world_x = float(self.handle_entry_world[0].item() + HANDLE_PRELOAD_X - PANEL_PULL_DISTANCE)

        # Explicit work-frame orientations (wxyz).  In all three poses the
        # jaw-opening axis is world +Z.  The tool approach axis points +X at
        # the front edge, -Y from the left side, and +Y from the right side.
        quaternion = lambda values: torch.tensor(values, device=left_pose.device, dtype=left_pose.dtype)
        # Approach +X with the jaw axis along world Z: a vertical pincer that closes
        # around the front edge, lower jaw in the pocket behind the rim.
        base_front_quat = quaternion((0.5, 0.5, 0.5, 0.5))
        self.front_grasp_quat = base_front_quat
        # First approach with the gripper mouth facing the floor.  The yaw is
        # applied only after reaching the point above the loop; selecting its sign
        # from the actual USD finger axes keeps approach=-Z while putting the jaw
        # opening along X (one finger enters the circular handle).
        self.top_down_quat = quaternion((0.0, 1.0, 0.0, 0.0))
        measured_rotation = matrix_from_quat(left_pose[3:7].unsqueeze(0))[0]
        local_jaw = measured_rotation.transpose(0, 1) @ (
            self.left.finger_centers_w()[0] - self.left.finger_centers_w()[1]
        )
        local_jaw = local_jaw / torch.linalg.norm(local_jaw)
        yaw_axis = torch.tensor([[0.0, 0.0, 1.0]], device=left_pose.device, dtype=left_pose.dtype)
        yaw_half = torch.tensor([0.5 * math.pi], device=left_pose.device, dtype=left_pose.dtype)
        yaw_plus = quat_from_angle_axis(yaw_half, yaw_axis)[0]
        yaw_minus = quat_from_angle_axis(-yaw_half, yaw_axis)[0]
        candidates = (
            quat_mul(yaw_plus, self.top_down_quat),
            quat_mul(self.top_down_quat, yaw_plus),
            quat_mul(yaw_minus, self.top_down_quat),
            quat_mul(self.top_down_quat, yaw_minus),
        )
        approach_ref = self.left.finger_offset_b()
        best_score = float("inf")
        self.top_grasp_quat = None
        for candidate in candidates:
            rotation = matrix_from_quat(candidate.unsqueeze(0))[0]
            approach = rotation @ approach_ref
            jaw = rotation @ local_jaw
            score = float(
                torch.linalg.norm(approach - torch.tensor([0.0, 0.0, -torch.linalg.norm(approach_ref)], device=left_pose.device, dtype=left_pose.dtype)).item()
                + abs(float(jaw[1].item()))
                + abs(float(jaw[2].item()))
                + abs(abs(float(jaw[0].item())) - float(torch.linalg.norm(approach_ref).item()))
            )
            if score < best_score:
                best_score = score
                self.top_grasp_quat = candidate
        # Choose side orientations from the measured USD finger frame.  The
        # old hard-coded quaternions looked like X-rolls but in this asset frame
        # produced approach=-Z (down into the table).
        side_axis = torch.tensor([[1.0, 0.0, 0.0]], device=left_pose.device, dtype=left_pose.dtype)
        side_half = torch.tensor([0.5 * math.pi], device=left_pose.device, dtype=left_pose.dtype)
        side_candidates = []
        for sign in (1.0, -1.0):
            side_rot = quat_from_angle_axis(sign * side_half, side_axis)[0]
            side_candidates.extend((
                quat_mul(side_rot, self.top_down_quat),
                quat_mul(self.top_down_quat, side_rot),
            ))

        def choose_side_quat(expected_y):
            target = torch.tensor([0.0, expected_y * torch.linalg.norm(approach_ref), 0.0], device=left_pose.device, dtype=left_pose.dtype)
            best = None
            best_score = float("inf")
            for candidate in side_candidates:
                rotation = matrix_from_quat(candidate.unsqueeze(0))[0]
                approach = rotation @ approach_ref
                jaw = rotation @ local_jaw
                score = float(torch.linalg.norm(approach - target).item()) + abs(float(jaw[0].item())) + abs(float(jaw[1].item())) + abs(abs(float(jaw[2].item())) - float(torch.linalg.norm(local_jaw).item()))
                if score < best_score:
                    best_score, best = score, candidate
            return best

        self.left_side_grasp_quat = choose_side_quat(-1.0)
        self.right_side_grasp_quat = choose_side_quat(1.0)

        # Side grasp invariant: the finger approach axis must point from each
        # robot toward the panel centre, never down toward the table.
        for label, side_quat, expected_y in (
            ("left", self.left_side_grasp_quat, -1.0),
            ("right", self.right_side_grasp_quat, 1.0),
        ):
            side_rotation = matrix_from_quat(side_quat.unsqueeze(0))[0]
            side_approach = side_rotation @ self.left.finger_offset_b()
            if abs(float(side_approach[2].item())) > 0.02 or float(side_approach[1].item()) * expected_y < 0.8 * float(torch.linalg.norm(approach_ref).item()):
                raise RuntimeError(
                    f"Invalid side grasp orientation for {label}: "
                    f"approach={[round(float(v),4) for v in side_approach.tolist()]}"
                )
            print(
                f"[AUTO_OPS_SIDE_ORIENTATION] {label} approach="
                f"{[round(float(v),4) for v in side_approach.tolist()]} "
                "(toward panel, jaw vertical)", flush=True
            )

        # Verify the frame convention against the actual USD finger links.  The
        # printed vectors are the commanded finger approach direction and the jaw
        # separation direction in world coordinates; this catches a 90-degree
        # sideways gripper before the episode starts.
        top_rotation = matrix_from_quat(self.top_grasp_quat.unsqueeze(0))[0]
        top_approach = top_rotation @ self.left.finger_offset_b()
        top_jaw = top_rotation @ local_jaw
        print(
            "[AUTO_OPS_ORIENTATION] top_down "
            f"approach={[round(float(v),4) for v in top_approach.tolist()]} "
            f"jaw={[round(float(v),4) for v in top_jaw.tolist()]}",
            flush=True,
        )

        self.left.set_target(left_pose[:3], left_pose[3:7])
        self.right.set_target(right_pose[:3], right_pose[3:7])
        # Pink's posture task supplies the redundant-joint seed.
        self.left.set_gripper(GRIPPER_OPEN_WIDTH)
        self.right.set_gripper(GRIPPER_OPEN_WIDTH)
        # Standalone Lift begins with the panel already pulled out and both arms
        # already in the default pose. Enter Lift's first exclusively-owned
        # bimanual stage; phase-3 stages 0..3 belong only to Pull release.
        if self.task_mode in {"lift", "lift_hang"}:
            self.stage = 4
            self.stage_tick = 0
        self.initialized = True
        for name, arm in (("left", self.left), ("right", self.right)):
            print(
                f"[AUTO_OPS] {name}_finger_offset_b={[round(v, 5) for v in arm.finger_offset_b().tolist()]} "
                f"norm={float(torch.linalg.norm(arm.finger_offset_b()).item()):.5f}",
                flush=True,
            )
        self._event(
            "auto_ops_started",
            initial_panel_position=self.initial_panel_pos.detach().cpu().tolist(),
            handle_world=self.handle_world.detach().cpu().tolist(),
            handle_above_world=self.handle_above_world.detach().cpu().tolist(),
            handle_entry_world=self.handle_entry_world.detach().cpu().tolist(),
            pull_end_world_x=self.pull_end_world_x,
            hang_position=list(self.hang_position),
            hang_entry_z=self.hang_entry_z,
        )

    def _panel_side_targets(self, reference, y_offset, z_offset):
        """Return symmetric side finger targets in the panel-local frame.

        ``y_offset`` is mirrored about the panel centre; the panel pose rotates
        the pair into world coordinates.  The side gripper quaternions receive
        the same panel yaw so their approach axes remain normal to the panel
        edges after translation or yaw randomization.
        """
        rotation = self._panel_rotation(reference)
        device = reference.device
        dtype = reference.dtype
        # Keep arm-to-side assignment fixed for the episode; recomputing it while moving can swap both targets.
        left_sign = self.left_panel_side_sign
        right_sign = -left_sign
        local_left = torch.tensor((SIDE_GRASP_LONGITUDINAL_OFFSET, left_sign * y_offset, z_offset), device=device, dtype=dtype)
        local_right = torch.tensor((-SIDE_GRASP_LONGITUDINAL_OFFSET, right_sign * y_offset, z_offset), device=device, dtype=dtype)
        left_target = reference[:3] + rotation @ local_left
        right_target = reference[:3] + rotation @ local_right
        edge = rotation @ torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        edge_xy = edge[:2] / torch.clamp(torch.linalg.norm(edge[:2]), min=1.0e-6)
        yaw_delta = torch.atan2(edge_xy[1], edge_xy[0]) - 0.5 * math.pi
        yaw_quat = quat_from_angle_axis(
            yaw_delta.reshape(1),
            torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype),
        )[0]
        left_base_quat = self.left_side_grasp_quat if left_sign > 0.0 else self.right_side_grasp_quat
        right_base_quat = self.left_side_grasp_quat if right_sign > 0.0 else self.right_side_grasp_quat
        left_quat = quat_mul(yaw_quat.unsqueeze(0), left_base_quat.unsqueeze(0))[0]
        right_quat = quat_mul(yaw_quat.unsqueeze(0), right_base_quat.unsqueeze(0))[0]
        return left_target, right_target, left_quat, right_quat

    def _panel_side_standoff_targets(self, reference, z_offset):
        """Return reachable, collision-clear open-hand waits for each arm."""
        left_target, _, left_quat, _ = self._panel_side_targets(
            reference, SIDE_STANDOFF_LEFT_FINGER_Y, z_offset
        )
        _, right_target, _, right_quat = self._panel_side_targets(
            reference, SIDE_STANDOFF_RIGHT_FINGER_Y, z_offset
        )
        return left_target, right_target, left_quat, right_quat

    def _standalone_hang_grasp_tick(self):
        """Build the physical bimanual start state before hang episode tick 0."""
        reference = self.initial_panel_pose
        # Keep the dynamic board at its authored hang-entry pose only while the
        # two arms establish the initial grasp. Once verified, Newton owns it.
        raw_pose = torch.cat(
            (reference[:3], reference[4:7], reference[3:4])
        ).reshape(1, 7)
        self.panel.write_root_pose_to_sim_index(root_pose=raw_pose)
        self.panel.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros_like(self.panel.data.default_root_vel.torch)
        )

        left_final, right_final, left_q, right_q = self._vertical_regrasp_targets(
            reference, SIDE_GRASP_FINGER_Y
        )
        left_standoff, right_standoff, _, _ = self._vertical_regrasp_targets(
            reference, SIDE_GRASP_FINGER_Y + 0.06
        )
        if self.hang_bootstrap_stage == 0:
            self.left.release_gripper()
            self.right.release_gripper()
            self.left.set_task_costs(80.0, 20.0)
            self.right.set_task_costs(80.0, 20.0)
            self.left.set_finger_target(left_standoff, left_q)
            self.right.set_finger_target(right_standoff, right_q)
            error = max(
                self.left.finger_midpoint_error(left_standoff),
                self.right.finger_midpoint_error(right_standoff),
            )
            self.hang_bootstrap_stable_ticks = (
                self.hang_bootstrap_stable_ticks + 1 if error <= 0.012 else 0
            )
            if self.hang_bootstrap_stable_ticks >= 2:
                self.hang_bootstrap_stage = 1
                self.hang_bootstrap_stable_ticks = 0
                self._event("hang_initial_standoff_reached")
        elif self.hang_bootstrap_stage == 1:
            self.left.set_finger_target(left_final, left_q)
            self.right.set_finger_target(right_final, right_q)
            error = max(
                self.left.finger_midpoint_error(left_final),
                self.right.finger_midpoint_error(right_final),
            )
            self.hang_bootstrap_stable_ticks = (
                self.hang_bootstrap_stable_ticks + 1 if error <= 0.009 else 0
            )
            if self.hang_bootstrap_stable_ticks >= 2:
                self.hang_bootstrap_stage = 2
                self.hang_bootstrap_stable_ticks = 0
                self.left.latch_gripper_close()
                self.right.latch_gripper_close()
                self.hang_bootstrap_close_tick = self.total_control_ticks
                self._event("hang_initial_gripper_close_commanded")
        else:
            self.left.set_finger_target(left_final, left_q)
            self.right.set_finger_target(right_final, right_q)
            self.left.close_step(rate=0.010)
            self.right.close_step(rate=0.010)
            target_error = max(
                self.left.finger_midpoint_error(left_final),
                self.right.finger_midpoint_error(right_final),
            )
            enclosed = (
                self._grasp_width_valid(self.left, 0.020, 0.052)
                and self._grasp_width_valid(self.right, 0.020, 0.052)
                and target_error <= 0.015
            )
            if enclosed:
                self.left.hold_gripper_force(
                    POLICY_GRIP_FORCE_PER_FINGER,
                    preload_per_finger=0.001,
                    preserve_measured_contact=True,
                    position_only=True,
                )
                self.right.hold_gripper_force(
                    POLICY_GRIP_FORCE_PER_FINGER,
                    preload_per_finger=0.001,
                    preserve_measured_contact=True,
                    position_only=True,
                )
                self._capture_grasp_offsets(preserve_measured=True)
                self.hang_grasp_ready = True
                self.events.clear()
                self.total_control_ticks = 0
                self.phase_ticks = 0
                self.stage = 0
                self.stage_tick = 0
                self.waypoint_start = None
                self.panel_motion_progress = 0.0
                self.panel_height_history.clear()
                self._event(
                    "auto_ops_started_from_bimanual_hold",
                    left_gripper_width=self.left.measured_gripper_width(),
                    right_gripper_width=self.right.measured_gripper_width(),
                    left_grasp_height=float(left_final[2].item()),
                    right_grasp_height=float(right_final[2].item()),
                )
            elif (
                self.hang_bootstrap_close_tick is not None
                and self.total_control_ticks - self.hang_bootstrap_close_tick >= 30
            ):
                self._fail("hang_initial_bimanual_grasp_not_established")
        if self.total_control_ticks % 30 == 0:
            print(
                "[HANG_BOOTSTRAP] "
                f"stage={self.hang_bootstrap_stage} "
                f"left_error={self.left.finger_midpoint_error(left_standoff if self.hang_bootstrap_stage == 0 else left_final):.4f} "
                f"right_error={self.right.finger_midpoint_error(right_standoff if self.hang_bootstrap_stage == 0 else right_final):.4f} "
                f"left_target={[round(float(v), 4) for v in (left_standoff if self.hang_bootstrap_stage == 0 else left_final).tolist()]} "
                f"right_target={[round(float(v), 4) for v in (right_standoff if self.hang_bootstrap_stage == 0 else right_final).tolist()]}",
                flush=True,
            )

    def _step_phase_0(self, panel_pose, px, py, pz, device):
        # Resolve the handle from the *current* panel pose every tick.  This
        # keeps pre-grasp valid if the panel settles, is randomized, or is
        # displaced by an earlier contact; no stale world coordinate is used.
        panel_rotation = self._panel_rotation(panel_pose)
        handle_local = torch.tensor((HANDLE_GRASP_LOCAL_X, 0.0, HANDLE_LOCAL_Z), device=device, dtype=panel_pose.dtype)
        self.handle_world = panel_pose[:3] + panel_rotation @ handle_local
        self.handle_above_world = self.handle_world + torch.tensor((0.0, 0.0, HANDLE_ABOVE_MARGIN_Z), device=device, dtype=panel_pose.dtype)
        self.handle_entry_world = self.handle_world + torch.tensor((0.0, 0.0, HANDLE_INSERT_Z_OFFSET), device=device, dtype=panel_pose.dtype)
        # Top-down cube-pick of the curved handle radius.  Move above the
        # midpoint between the panel edge and arc radius, descend in Z, close,
        # then pull in -X while holding the same height and orientation.
        # The gripper stays
        # fully open while it moves above the handle, descends vertically,
        # and only then transitions to the close phase.
        self.right.set_target(self.right_hold_pose[:3], self.right_hold_pose[3:7])
        self.right.set_joint_override(self.right_hold_joint_target)
        self.left.set_gripper(GRIPPER_OPEN_WIDTH)
        self.right.set_gripper(GRIPPER_OPEN_WIDTH)
        handle_x, handle_y, handle_z = (float(value.item()) for value in self.handle_world)
        above_z = float(self.handle_above_world[2].item())
        if self.stage == 0:
            # Move from the default ready pose directly to the high-margin
            # point above the handle.  Keep the default wrist orientation for
            # this traverse so no low-height rotation can hit the panel.
            self.left.set_task_costs(95.0, 30.0)
            above = torch.tensor((handle_x, handle_y, above_z), device=device, dtype=panel_pose.dtype)
            if self.above_handle_waypoint_start is None:
                self.above_handle_waypoint_start = self.left.finger_centers_w().mean(dim=0).clone()
            approach_duration = max(self._duration(32), 8)
            approach_fraction = min(1.0, self._stage_elapsed() / approach_duration)
            approach_midpoint = self.above_handle_waypoint_start + approach_fraction * (above - self.above_handle_waypoint_start)
            self.left.set_finger_target(approach_midpoint, self.left_hold_pose[3:7])
            error = self.left.finger_midpoint_error(above)
            if error <= 0.015:
                self._begin_stage(1, "default_pose_above_handle_reached")
            elif self._stage_elapsed() >= max(self._duration(360), 180):
                self._fail("default_pose_above_handle_not_reached", midpoint_error=error)

        elif self.stage == 1:
            self.left.set_task_costs(80.0, 55.0)
            above = (handle_x, handle_y, above_z)
            self.left.set_finger_target(above, self.top_grasp_quat)
            error = self.left.finger_midpoint_error(above)
            _, rotation_error = self.left.tracking_pose_error()
            if error <= 0.015 and rotation_error <= 0.12:
                self.pregrasp_handle_world = self.handle_world.clone()
                self._begin_stage(2, "handle_yaw_aligned_above")

        elif self.stage == 2:
            handle_x, handle_y, handle_z = (float(value.item()) for value in self.handle_world)
            self.left.set_task_costs(85.0, 55.0)
            goal_z = handle_z + HANDLE_INSERT_Z_OFFSET
            current_mid_z = float(self.left.finger_centers_w().mean(dim=0)[2].item())
            entry_z = max(goal_z, current_mid_z - HANDLE_DESCENT_STEP_Z)
            entry = (handle_x, handle_y, entry_z)
            self.left.set_finger_target(entry, self.top_grasp_quat)
            error = self.left.finger_midpoint_error((handle_x, handle_y, goal_z))
            if error <= 0.025:
                self.front_grasp_reference = panel_pose.clone()
                self._transition(1, "top_handle_pregrasp_reached")

        elif self.stage == 3:
            # Return both arms to the same known joint-space readiness pose.
            # This removes the twisted front-pull IK seed before side IK.
            self.left.set_joint_override(self.left_hold_joint_target)
            self.right.set_joint_override(self.right_hold_joint_target)
            self.left.set_gripper(GRIPPER_OPEN_WIDTH)
            self.right.set_gripper(GRIPPER_OPEN_WIDTH)
            joint_error = max(self.left.joint_tracking_error(), self.right.joint_tracking_error())
            if self._stage_ready(joint_error, 0.15, patience=self._duration(80)):
                self.left.clear_joint_override()
                self.right.clear_joint_override()
                self.side_grasp_reference = panel_pose.clone()
                self._event("side_grasp_reference_frozen", panel_pose=panel_pose.detach().cpu().tolist())
                self._transition(4, "pull_release_retreat_complete") if self.task_mode == "pull" else self._begin_stage(4, "bimanual_default_joint_readiness_reached")

        elif self.stage == 4:
            # Stage both hands outside the side guards at clearance height,
            # already rolled for a vertical jaw opening.
            self.left.clear_joint_override()
            self.right.clear_joint_override()
            self.left.set_task_costs(60.0, 20.0)
            self.right.set_task_costs(60.0, 20.0)
            left_high, right_high, left_side_quat, right_side_quat = self._panel_side_standoff_targets(reference, SIDE_GRASP_Z)
            self.left.set_finger_target(left_high, left_side_quat)
            self.right.set_finger_target(right_high, right_side_quat)
            if self._pose_ready(self.left, 0.040, 0.12) and self._pose_ready(self.right, 0.040, 0.12):
                self._begin_stage(5, "bimanual_high_side_staging_reached")

        elif self.stage == 5:
            # Descend outside the side guards in bounded object-relative steps.
            # A one-shot drop to the panel plane makes Pink choose a low elbow
            # path that visually dives below the table.
            panel_z = float(reference[2].item())
            left_mid_z = float(self.left.finger_centers_w().mean(dim=0)[2].item())
            right_mid_z = float(self.right.finger_centers_w().mean(dim=0)[2].item())
            next_z_offset = max(SIDE_GRASP_Z, min(
                left_mid_z - panel_z - SIDE_DESCENT_STEP_Z,
                right_mid_z - panel_z - SIDE_DESCENT_STEP_Z,
            ))
            left_standoff, right_standoff, left_side_quat, right_side_quat = self._panel_side_standoff_targets(reference, next_z_offset)
            self.left.set_finger_target(left_standoff, left_side_quat)
            self.right.set_finger_target(right_standoff, right_side_quat)
            # Settle both jaw centres out here, level with the panel and 150 mm
            # clear of its edges, so the inward run cannot clip and shove it.
            jaw_error = max(
                self.left.finger_midpoint_error(left_standoff),
                self.right.finger_midpoint_error(right_standoff),
            )
            if jaw_error <= 0.008 or self._stage_ready(jaw_error, -1.0):
                # Re-reference here, not back in stage 2: the panel drifts while the
                # arms travel out and down, and a stale y left the jaws closing 34 mm
                # outboard of the edge.  Both hands are clear of it at this point.
                self.side_grasp_reference = panel_pose.clone()
                self._event("side_grasp_reference_refreshed", panel_pose=panel_pose.detach().cpu().tolist())
                self._begin_stage(6, "bimanual_side_descent_reached")

        elif self.stage == 6:
            # Final approach is lateral only: no roll or descent near the panel.
            self.left.set_task_costs(90.0, 20.0)
            self.right.set_task_costs(90.0, 20.0)
            # Symmetric bimanual line: both midpoint targets share exactly
            # the same x/z; only y is mirrored about the panel centre.
            left_grasp, right_grasp, left_side_quat, right_side_quat = self._panel_side_targets(reference, SIDE_GRASP_FINGER_Y, SIDE_GRASP_Z)
            self.left.set_finger_target(left_grasp, left_side_quat)
            self.right.set_finger_target(right_grasp, right_side_quat)
            jaw_error = max(
                self.left.finger_midpoint_error(left_grasp),
                self.right.finger_midpoint_error(right_grasp),
            )
            if jaw_error <= 0.008 or self._stage_ready(jaw_error, -1.0):
                self._transition(4, "both_side_edges_centered")

    def _step_phase_1(self, panel_pose, px, py, pz, device):
        # Freeze the object frame captured at the end of phase 0.  Following
        # the live panel pose here creates a chase loop: an open finger moves
        # the panel, the target follows it, and the panel slides again.
        reference = self.front_grasp_reference if self.front_grasp_reference is not None else panel_pose
        grasp_midpoint = (
            float(reference[0].item()) + HANDLE_GRASP_LOCAL_X + HANDLE_PRELOAD_X + HANDLE_EDGE_INSET_X,
            float(reference[1].item()),
            float(reference[2].item()) + HANDLE_LOCAL_Z + HANDLE_INSERT_Z_OFFSET,
        )
        self.left.set_task_costs(80.0, 40.0)
        self.left.set_finger_target(grasp_midpoint, self.top_grasp_quat)
        if self.front_close_tick is None:
            self.left.set_gripper(GRIPPER_OPEN_WIDTH)
            measured_midpoint = self.left.finger_centers_w().mean(dim=0)
            grasp_target = torch.as_tensor(grasp_midpoint, device=device, dtype=measured_midpoint.dtype)
            grasp_delta = grasp_target - measured_midpoint
            jaw_error = float(torch.linalg.norm(grasp_delta).item())
            z_error = abs(float(grasp_delta[2].item()))
            _, rotation_error = self.left.tracking_pose_error()
            # Pink can retain roughly 15-20 mm of Cartesian residual near
            # this top-down joint-limit configuration.  The enlarged ring
            # still encloses the open jaws at that residual; require three
            # stable ticks and let the following Newton contact/width gate
            # determine whether a real grasp was established.
            aligned = jaw_error <= 0.020 and z_error <= 0.015 and rotation_error <= 0.12
            self.front_alignment_stable_ticks = self.front_alignment_stable_ticks + 1 if aligned else 0
            if self.front_alignment_stable_ticks >= 3:
                self.front_grasp_reference = panel_pose.clone()
                self.front_close_start_panel_x = float(panel_pose[0].item())
                self.front_close_tick = self.phase_ticks
                self.front_contact_stable_ticks = 0
                self.left.latch_gripper_close()
                self._event(
                    "left_front_gripper_close_commanded",
                    grasp_midpoint=list(grasp_midpoint),
                    measured_finger_midpoint=measured_midpoint.detach().cpu().tolist(),
                    alignment_error=jaw_error,
                    alignment_z_error=z_error,
                )
            elif self._stage_elapsed() >= max(self._duration(90), 45):
                self._fail(
                    "front_handle_alignment_not_reached",
                    alignment_error=jaw_error,
                    alignment_z_error=z_error,
                    rotation_error=rotation_error,
                )
        else:
            self.left.close_step(rate=FRONT_HANDLE_CLOSE_RATE, floor=FRONT_HANDLE_HOLD_WIDTH)
            close_elapsed = self.phase_ticks - self.front_close_tick
            width = self.left.measured_gripper_width()
            grasp_slip = self.front_close_start_panel_x - px
            # Positive displacement is already-progressing pull motion
            # (-X), not a slip. Only fail when closing pushes the panel
            # backward toward +X; desired motion is deducted below from
            # the remaining commanded pull distance.
            if grasp_slip < -0.050:
                self._fail(
                    "panel_pushed_backward_during_gripper_close",
                    grasp_slip=grasp_slip,
                    left_gripper_width=width,
                )
                return
            if grasp_slip > 0.080:
                self.front_grasp_attempts += 1
                if self.front_grasp_attempts <= 2:
                    self.left.release_gripper()
                    self.front_close_tick = None
                    self.front_alignment_stable_ticks = 0
                    self.front_contact_stable_ticks = 0
                    self.front_grasp_reference = panel_pose.clone()
                    self._event(
                        "front_handle_regrasp_after_close_motion",
                        attempt=self.front_grasp_attempts,
                        close_panel_displacement=grasp_slip,
                        panel_pose=panel_pose.detach().cpu().tolist(),
                    )
                    return
                self._fail(
                    "front_handle_unstable_during_close",
                    attempts=self.front_grasp_attempts,
                    close_panel_displacement=grasp_slip,
                )
                return
            handle_physically_enclosed = 0.020 <= width <= FRONT_HANDLE_PULL_START_WIDTH
            self.front_contact_stable_ticks = (
                self.front_contact_stable_ticks + 1 if handle_physically_enclosed else 0
            )
            if (
                close_elapsed >= FRONT_HANDLE_MIN_CLOSE_TICKS
                and self.front_contact_stable_ticks >= 3
            ):
                self.front_grip_floor = FRONT_HANDLE_HOLD_WIDTH
                self.left.hold_gripper_force(
                    preload_per_finger=0.001, preserve_measured_contact=True
                )
                self.pull_start_pose = self.left.pose_w()[0].clone()
                self.pull_phase_start_panel_x = px
                pre_pull_displacement = float(self.initial_panel_pos[0].item()) - px
                self.pull_command_distance = max(
                    0.0,
                    PANEL_PULL_DISTANCE - max(0.0, pre_pull_displacement),
                )
                self._event(
                    "front_pincer_closed_around_edge",
                    left_gripper_width=width,
                    grip_floor=self.front_grip_floor,
                    pre_pull_displacement=pre_pull_displacement,
                    pull_command_distance=self.pull_command_distance,
                )
                self._transition(2, "front_pincer_closed_around_edge")
            elif close_elapsed >= FRONT_HANDLE_MIN_CLOSE_TICKS and width < 0.020:
                self.front_grasp_attempts += 1
                if self.front_grasp_attempts <= 2:
                    self.left.release_gripper()
                    self.front_close_tick = None
                    self.front_alignment_stable_ticks = 0
                    self.front_contact_stable_ticks = 0
                    # Re-anchor to the object that actually moved during
                    # contact; the next target remains panel-relative.
                    self.front_grasp_reference = panel_pose.clone()
                    self._event(
                        "front_handle_regrasp_requested",
                        attempt=self.front_grasp_attempts,
                        left_gripper_width=width,
                        panel_pose=panel_pose.detach().cpu().tolist(),
                    )
                else:
                    self._fail(
                        "front_handle_not_enclosed_before_pull",
                        attempts=self.front_grasp_attempts,
                        left_gripper_width=width,
                    )
            elif close_elapsed >= 50:
                self._fail("handle_contact_not_established", left_gripper_width=width)

    def _step_phase_2(self, panel_pose, px, py, pz, device):
        panel_velocity = self.panel.data.root_com_vel_w.torch[0]
        panel_vy = float(panel_velocity[1].item())
        panel_yaw_rate = float(panel_velocity[5].item())
        pull_duration = max(self._duration(20), 15)
        fraction = min(1.0, self.phase_ticks / pull_duration)
        smooth = fraction * fraction * (3.0 - 2.0 * fraction)
        target = self.pull_start_pose[:3].clone()
        target[0] -= self.pull_command_distance * smooth
        # Pull parallel to rack X while bringing the grasped ring onto the
        # rail centreline. Completion requires the physical panel to remain
        # centred for three control ticks, not merely cross the threshold.
        lateral_feedback = max(
            -PULL_LATERAL_CORRECTION_LIMIT,
            min(
                PULL_LATERAL_CORRECTION_LIMIT,
                PULL_LATERAL_FEEDBACK_GAIN * py
                + PULL_LATERAL_VELOCITY_GAIN * panel_vy,
            ),
        )
        target[1] = (
            self.pull_start_pose[1] * (1.0 - smooth)
            - smooth * lateral_feedback
        )
        target[2] = self.pull_start_pose[2]
        self.left.set_target(target, self.top_grasp_quat)
        self.left.close_step(floor=self.front_grip_floor)
        total_pulled_distance = float(self.initial_panel_pos[0].item()) - px
        phase_pull_distance = float(self.pull_phase_start_panel_x) - px
        pull_width = self.left.measured_gripper_width()
        early_grip_lost = (
            self.phase_ticks <= pull_duration + 5
            and pull_width < 0.020
            and phase_pull_distance < 0.050
            and total_pulled_distance < PANEL_PULL_MINIMUM
        )
        if early_grip_lost:
            self.front_grasp_attempts += 1
            if self.front_grasp_attempts <= 2:
                self.left.release_gripper()
                self.front_close_tick = None
                self.front_alignment_stable_ticks = 0
                self.front_contact_stable_ticks = 0
                self.pull_success_stable_ticks = 0
                self.front_grasp_reference = panel_pose.clone()
                self._event(
                    "front_grasp_lost_during_pull_recovery",
                    attempt=self.front_grasp_attempts,
                    left_gripper_width=pull_width,
                    phase_pull_distance=phase_pull_distance,
                )
                self._transition(1, "front_grasp_lost_reapproach_live_handle")
                return
            self._fail(
                "front_grasp_repeatedly_lost_during_pull",
                attempts=self.front_grasp_attempts,
                left_gripper_width=pull_width,
            )
            return
        required_phase_pull = (
            0.0
            if self.pull_command_distance <= 0.020
            else max(
                0.10,
                min(PANEL_PULL_MINIMUM - 0.050, self.pull_command_distance - 0.020),
            )
        )
        pull_physically_ready = (
            total_pulled_distance >= PANEL_PULL_MINIMUM
            and phase_pull_distance >= required_phase_pull
            and abs(py) <= PULL_CENTER_TOLERANCE
            and abs(panel_vy) <= PULL_RELEASE_MAX_LATERAL_SPEED
            and abs(panel_yaw_rate) <= PULL_RELEASE_MAX_YAW_RATE
        )
        self.pull_success_stable_ticks = (
            self.pull_success_stable_ticks + 1 if pull_physically_ready else 0
        )
        # At 15 Hz, three extra closed-grasp ticks add enough travel to yaw
        # the already-clear panel into a side guard. The condition is based
        # on measured panel displacement and lateral position, so release on
        # the first physically valid sample instead of overshooting it.
        if self.pull_success_stable_ticks >= 1:
            self._event(
                "physical_panel_pull_verified",
                pulled_distance=total_pulled_distance,
                phase_pull_distance=phase_pull_distance,
                panel_lateral_offset=py,
                panel_lateral_speed=panel_vy,
                panel_yaw_rate=panel_yaw_rate,
            )
            self.retreat_start_pose = self.left.pose_w()[0].clone()
            self.retreat_start_finger_midpoint = self.left.finger_centers_w().mean(dim=0).clone()
            self._transition(3, "panel_pulled_clear_of_side_guards")
            return
        if self.phase_ticks >= pull_duration + max(self._duration(180), 90):
            self._fail(
                "physical_panel_pull_not_detected",
                pulled_distance=total_pulled_distance,
                phase_pull_distance=phase_pull_distance,
                panel_lateral_offset=py,
                panel_lateral_speed=panel_vy,
                panel_yaw_rate=panel_yaw_rate,
            )

    def _step_phase_3(self, panel_pose, px, py, pz, device):
        reference = self.initial_panel_pose if self.side_grasp_reference is None else self.side_grasp_reference
        # Side face targets are panel-local points transformed by the live pose.
        # This makes their world Z follow tilt/roll instead of using a fixed center Z.
        if self.stage >= 4:
            reference = panel_pose.clone()
        if self.stage == 0:
            self.left.open_step(rate=0.008)
            self.right.set_gripper(GRIPPER_OPEN_WIDTH)
            self.left.set_target(self.retreat_start_pose[:3], self.retreat_start_pose[3:7])
            # Do not wait for full jaw width while a finger is still inside
            # the ring: the ring itself limits the measured opening. Begin
            # the vertical extraction immediately after the open command
            # has reached Newton for two control ticks.
            if self._stage_elapsed() >= 2:
                self._begin_stage(1, "front_grasp_open_command_applied")
        elif self.stage == 1:
            # Open, lift vertically, then retreat: never drag the panel along X.
            self.left.set_task_costs(80.0, 6.0)
            self.left.open_step(rate=0.008)
            lift = self.retreat_start_finger_midpoint.clone()
            lift[2] = float(reference[2].item()) + HANDLE_LOCAL_Z + PULL_RETURN_CLEARANCE_Z
            self.left.set_finger_target(lift, self.retreat_start_pose[3:7])
            lift_error = self.left.finger_midpoint_error(lift)
            opened_width = self.left.measured_gripper_width()
            if lift_error <= 0.025 and opened_width >= 0.070 and self._stage_elapsed() >= 3:
                self._begin_stage(2, "left_lifted_clear_of_panel")
            elif self._stage_elapsed() >= max(self._duration(45), 20):
                self._fail(
                    "left_did_not_lift_clear_after_pull",
                    lift_error=lift_error,
                    opened_width=opened_width,
                )
        elif self.stage == 2:
            self.left.set_task_costs(80.0, 6.0)
            self.left.open_step(rate=0.008)
            retreat = self.retreat_start_finger_midpoint.clone()
            retreat[0] -= 0.22
            retreat[2] = float(reference[2].item()) + HANDLE_LOCAL_Z + PULL_RETURN_CLEARANCE_Z
            self.left.set_finger_target(retreat, self.retreat_start_pose[3:7])
            retreat_error = self.left.finger_midpoint_error(retreat)
            opened_width = self.left.measured_gripper_width()
            if (
                retreat_error <= 0.035
                and opened_width >= 0.070
                and self._stage_elapsed() >= 3
            ):
                self._begin_stage(3, "left_backed_off_from_panel_edge")
            elif self._stage_elapsed() >= max(self._duration(60), 25):
                self._fail(
                    "left_did_not_retreat_after_pull",
                    retreat_error=retreat_error,
                    opened_width=opened_width,
                )
        elif self.stage == 3:
            self.left.set_joint_override(self.left_hold_joint_target)
            self.right.set_joint_override(self.right_hold_joint_target)
            self.left.set_gripper(GRIPPER_OPEN_WIDTH)
            self.right.set_gripper(GRIPPER_OPEN_WIDTH)
            joint_error = max(self.left.joint_tracking_error(), self.right.joint_tracking_error())
            if self._stage_ready(joint_error, 0.15, patience=self._duration(80)):
                self.left.clear_joint_override()
                self.right.clear_joint_override()
                self.side_grasp_reference = panel_pose.clone()
                left_final, right_final, left_q, right_q = self._panel_side_targets(
                    self.side_grasp_reference, SIDE_GRASP_FINGER_Y, SIDE_GRASP_Z
                )
                # Midpoints are the lateral waiting points beside the same
                # object-relative ±1/5 longitudinal grasp locations.  X and Z
                # exactly match the final grasp; only panel-local Y is outside.
                self.side_mid_left, self.side_mid_right, _, _ = self._panel_side_standoff_targets(
                    self.side_grasp_reference, SIDE_GRASP_Z
                )
                self.side_mid_left_quat = left_q.clone()
                self.side_mid_right_quat = right_q.clone()
                self.side_mid_stable_ticks = 0
                self.side_final_approach_progress = 0.0
                self._event(
                    "object_relative_side_waypoints",
                    left_mid=self.side_mid_left.detach().cpu().tolist(),
                    right_mid=self.side_mid_right.detach().cpu().tolist(),
                    left_final=left_final.detach().cpu().tolist(),
                    right_final=right_final.detach().cpu().tolist(),
                )
                self._begin_stage(4, "object_relative_bimanual_midpoints_computed")
        elif self.stage == 4:
            self.left.clear_joint_override(); self.right.clear_joint_override()
            self.left.set_task_costs(80.0, 1.0); self.right.set_task_costs(80.0, 1.0)
            # The released panel can settle or yaw on the rails. Keep both
            # open-hand waiting points attached to the live object frame;
            # freeze that frame only after both arms have arrived.
            self.side_mid_left, self.side_mid_right, live_left_q, live_right_q = self._panel_side_standoff_targets(
                panel_pose, SIDE_GRASP_Z
            )
            self.side_mid_left_quat = live_left_q.clone()
            self.side_mid_right_quat = live_right_q.clone()
            self.left.set_finger_target(self.side_mid_left, self.side_mid_left_quat)
            self.right.set_finger_target(self.side_mid_right, self.side_mid_right_quat)
            hand_position_error = max(
                self.left.tracking_pose_error()[0],
                self.right.tracking_pose_error()[0],
            )
            if hand_position_error <= 0.020:
                self.side_grasp_reference = panel_pose.clone()
                self.side_mid_left, self.side_mid_right, live_left_q, live_right_q = self._panel_side_standoff_targets(
                    self.side_grasp_reference, SIDE_GRASP_Z
                )
                self.side_mid_left_quat = live_left_q.clone()
                self.side_mid_right_quat = live_right_q.clone()
                self.side_mid_stable_ticks = 1
                self._begin_stage(5, "both_arms_reached_object_relative_midpoints")
            elif self._stage_elapsed() >= max(self._duration(360), 180):
                self._fail(
                    "bimanual_side_midpoints_not_reached",
                    hand_position_error=hand_position_error,
                )
        elif self.stage == 5:
            self.left.set_task_costs(80.0, 6.0)
            self.right.set_task_costs(80.0, 6.0)
            self.side_mid_left, self.side_mid_right, live_left_q, live_right_q = (
                self._panel_side_standoff_targets(panel_pose, SIDE_GRASP_Z)
            )
            self.side_mid_left_quat = live_left_q.clone()
            self.side_mid_right_quat = live_right_q.clone()
            self.left.set_finger_target(self.side_mid_left, self.side_mid_left_quat)
            self.right.set_finger_target(self.side_mid_right, self.side_mid_right_quat)
            mid_error = max(
                self.left.finger_midpoint_error(self.side_mid_left),
                self.right.finger_midpoint_error(self.side_mid_right),
            )
            # Stage 4 already established the precise 12 mm arrival. Once
            # both arms hold beside the panel, Newton/IK settling can leave
            # a small static residual. The interpolated final approach
            # below still enforces the strict 10/8 mm pre-close gates.
            midpoint_hold_ready = mid_error <= 0.025
            self.side_mid_stable_ticks = (
                self.side_mid_stable_ticks + 1 if midpoint_hold_ready else 0
            )
            if self.side_mid_stable_ticks >= 3:
                self.side_grasp_reference = panel_pose.clone()
                _, _, live_left_q, live_right_q = self._panel_side_standoff_targets(
                    self.side_grasp_reference, SIDE_GRASP_Z
                )
                # Start the final interpolation from the measured fingertip
                # locations that just satisfied the midpoint hold. Rebuilding
                # ideal standoffs here made the right arm step 30 mm outward
                # before approaching the panel and deadlocked Pink at 15%.
                self.side_mid_left = self.left.finger_centers_w().mean(dim=0).clone()
                self.side_mid_right = self.right.finger_centers_w().mean(dim=0).clone()
                self.side_mid_left_quat = live_left_q.clone()
                self.side_mid_right_quat = live_right_q.clone()
                self._begin_stage(6, "both_arms_waited_at_object_relative_midpoints")
            elif self._stage_elapsed() >= max(self._duration(240), 120):
                self._fail("bimanual_side_midpoint_hold_not_stable", midpoint_error=mid_error)
        elif self.stage == 6:
            reference_shift = float(
                torch.linalg.norm(panel_pose[:2] - self.side_grasp_reference[:2]).item()
            )
            if reference_shift > 0.060 and self.side_reanchor_attempts < 2:
                self.side_reanchor_attempts += 1
                self.side_grasp_reference = panel_pose.clone()
                self.side_mid_left = self.left.finger_centers_w().mean(dim=0).clone()
                self.side_mid_right = self.right.finger_centers_w().mean(dim=0).clone()
                self.side_final_approach_progress = 0.0
                self.side_preclose_stable_ticks = 0
                self._event(
                    "side_approach_reanchored_to_live_panel",
                    attempt=self.side_reanchor_attempts,
                    reference_shift=reference_shift,
                    panel_pose=panel_pose.detach().cpu().tolist(),
                )
                return
            # Stage 4 reached collision-clear standoffs around a settled
            # panel. Freeze that object frame for the short pre-close path;
            # following first contact creates a chase loop that pushes the
            # released panel away from the slower arm.
            _, _, left_q, right_q = self._panel_side_standoff_targets(
                self.side_grasp_reference, SIDE_GRASP_Z
            )
            live_mid_left = self.side_mid_left
            live_mid_right = self.side_mid_right
            left_final, right_final, _, _ = self._panel_side_targets(
                self.side_grasp_reference, SIDE_GRASP_FINGER_Y, SIDE_GRASP_Z
            )
            progress = self.side_final_approach_progress
            if progress >= 1.0:
                self.left.set_task_costs(100.0, 10.0)
                self.right.set_task_costs(100.0, 10.0)
            else:
                self.left.set_task_costs(80.0, 6.0)
                self.right.set_task_costs(80.0, 6.0)
            left_target = (1.0 - progress) * live_mid_left + progress * left_final
            right_target = (1.0 - progress) * live_mid_right + progress * right_final
            self.left.set_finger_target(left_target, left_q)
            self.right.set_finger_target(right_target, right_q)
            waypoint_error = max(
                self.left.finger_midpoint_error(left_target),
                self.right.finger_midpoint_error(right_target),
            )
            if waypoint_error <= 0.018 and progress < 1.0:
                self.side_final_approach_progress = min(1.0, progress + 0.05)
                self.side_preclose_stable_ticks = 0
            elif progress >= 1.0:
                final_rotation_error = max(
                    self.left.tracking_pose_error()[1],
                    self.right.tracking_pose_error()[1],
                )
                final_pose_ready = (
                    waypoint_error <= 0.018 and final_rotation_error <= 0.50
                )
                self.side_preclose_stable_ticks = (
                    self.side_preclose_stable_ticks + 1 if final_pose_ready else 0
                )
            else:
                self.side_preclose_stable_ticks = 0
            if self.side_final_approach_progress >= 1.0 and self.side_preclose_stable_ticks >= 2:
                self.side_grasp_reference = panel_pose.clone()
                self.side_preclose_stable_ticks = 0
                self._begin_stage(7, "simultaneous_side_approach_completed")
            elif self._stage_elapsed() >= max(self._duration(360), 180):
                self._fail(
                    "bimanual_live_side_approach_not_reached",
                    waypoint_error=waypoint_error,
                    rotation_error=max(
                        self.left.tracking_pose_error()[1],
                        self.right.tracking_pose_error()[1],
                    ),
                    progress=progress,
                )
        elif self.stage == 7:
            left_final, _, left_q, _ = self._panel_side_targets(
                self.side_grasp_reference, SIDE_GRASP_FINGER_Y, SIDE_GRASP_Z
            )
            self.left.set_finger_target(left_final, left_q)
            left_error = self.left.finger_midpoint_error(left_final)
            self.side_preclose_stable_ticks = (
                self.side_preclose_stable_ticks + 1 if left_error <= 0.018 else 0
            )
            if self.side_preclose_stable_ticks >= 2:
                self.side_preclose_stable_ticks = 0
                self._begin_stage(8, "left_side_micro_alignment_completed")
        else:
            _, right_final, _, right_q = self._panel_side_targets(
                self.side_grasp_reference, SIDE_GRASP_FINGER_Y, SIDE_GRASP_Z
            )
            self.right.set_finger_target(right_final, right_q)
            right_error = self.right.finger_midpoint_error(right_final)
            self.side_preclose_stable_ticks = (
                self.side_preclose_stable_ticks + 1 if right_error <= 0.018 else 0
            )
            if self.side_preclose_stable_ticks >= 2:
                self._transition(4, "sequential_side_alignment_completed")

    def _step_phase_4(self, panel_pose, px, py, pz, device):
        # Keep the pre-close object frame fixed while closing. Chasing the
        # live panel pose during contact pushes a displaced board farther
        # away. Closed-chain alignment happens in phase 5.
        reference = self.side_grasp_reference
        # Compute both grasp points from the same panel-local frame.
        left_midpoint, right_midpoint, left_side_quat, right_side_quat = self._panel_side_targets(reference, SIDE_GRASP_FINGER_Y, SIDE_GRASP_Z)
        self.left.set_finger_target(left_midpoint, left_side_quat)
        self.right.set_finger_target(right_midpoint, right_side_quat)
        if self.side_close_tick is None:
            self.left.latch_gripper_close()
            self.right.latch_gripper_close()
            self.side_close_tick = self.phase_ticks
            self._event("bimanual_gripper_close_commanded")
        else:
            # Keep the target fully closed; contact must stop the joints above zero.
            self.left.close_step(rate=0.010)
            self.right.close_step(rate=0.010)
            close_elapsed = self.phase_ticks - self.side_close_tick
            target_alignment_error = max(
                self.left.finger_midpoint_error(left_midpoint),
                self.right.finger_midpoint_error(right_midpoint),
            )
            measured_side_enclosure = (
                self._grasp_width_valid(self.left, 0.025, 0.052)
                and self._grasp_width_valid(self.right, 0.025, 0.052)
                and target_alignment_error <= 0.015
            )
            grasp_verified = self._bimanual_grasp_is_valid() or measured_side_enclosure
            if close_elapsed >= 1 and grasp_verified:
                # Latch the measured contact width and add the fixed inward
                # effort.  An extra 8 mm position preload here overconstrains
                # Newton (stiff position servo plus force on both closed jaws)
                # and stalls the first alignment step.
                self.left.hold_gripper_force(
                    POLICY_GRIP_FORCE_PER_FINGER,
                    preload_per_finger=0.001,
                    preserve_measured_contact=True,
                    position_only=True,
                )
                self.right.hold_gripper_force(
                    POLICY_GRIP_FORCE_PER_FINGER,
                    preload_per_finger=0.001,
                    preserve_measured_contact=True,
                    position_only=True,
                )
                # Fixed policy-time grip effort: once CLOSE contact is
                # verified, use one constant value through lift and rotation.
                self._capture_grasp_offsets(preserve_measured=True)
                self._event(
                    "bimanual_physical_contact_verified",
                    left_gripper_width=self.left.measured_gripper_width(),
                    right_gripper_width=self.right.measured_gripper_width(),
                )
                self._transition(5, "bimanual_physical_grasp_established")
                self.lift_probe_start_z = float(self._panel_pose()[2].item())
                self.waypoint_start = None
                self._begin_stage(1, "bimanual_grasp_verified_lift_started")
            elif close_elapsed >= max(self._duration(40), 20):
                self._fail("bimanual_finger_contact_not_detected")

    def _step_phase_5(self, panel_pose, px, py, pz, device):
        # Keep the verified physical grasp at the Panda actuator limit for
        # the entire load-bearing lift, transport and rotation.
        self.left.ramp_gripper_force_hold()
        self.right.ramp_gripper_force_hold()
        self.left.gripper_mode = "FIXED_FORCE_CLOSE"
        self.right.gripper_mode = "FIXED_FORCE_CLOSE"
        # First square the physically grasped panel on the rack. Both hand
        # targets come from one desired panel pose, preserving the measured
        # bimanual panel-to-hand transforms.
        if self.stage == 0:
            clear_x = RACK_COVER_FRONT_X - PANEL_HALF_LENGTH - RACK_ROTATION_CLEARANCE_X
            aligned_x = min(float(self.grasp_panel_pose[0].item()), clear_x)
            aligned_position = torch.tensor(
                (aligned_x, RACK_CENTER_Y, float(self.grasp_panel_pose[2].item())),
                device=device,
            )
            horizontal_quaternion = torch.tensor(
                (1.0, 0.0, 0.0, 0.0), device=device
            )
            fraction = self._waypoint(
                aligned_position, horizontal_quaternion, nominal_ticks=4, floor_ticks=3
            )
            live_aligned_pose = self._panel_pose()
            _, live_align_rotation_error = compute_pose_error(
                live_aligned_pose[:3].reshape(1, 3), live_aligned_pose[3:7].reshape(1, 4),
                aligned_position.reshape(1, 3), horizontal_quaternion.reshape(1, 4),
            )
            physically_aligned = (
                abs(float(live_aligned_pose[0].item()) - aligned_x) <= 0.015
                and abs(float(live_aligned_pose[1].item()) - RACK_CENTER_Y) <= 0.012
                and float(torch.linalg.norm(live_align_rotation_error[0]).item()) <= 0.06
            )
            if fraction >= 1.0 and physically_aligned:
                self.grasp_panel_pose = torch.cat(
                    (aligned_position, horizontal_quaternion)
                )
                self.rotation_safe_z = (
                    float(aligned_position[2].item())
                    + PANEL_HALF_LENGTH
                    + ROTATION_CLEARANCE
                )
                self.lift_target_z = max(
                    self.rotation_safe_z, HANGER_NAIL_Z - (abs(HANDLE_LOCAL_X) + HANDLE_RING_RADIUS)
                )
                self.waypoint_start = None
                self.lift_probe_start_z = float(pz)
                self._begin_stage(1, "panel_squared_on_rack_and_clear_of_cover")
            return

        # Raise horizontally until the complete panel length clears the rack.
        start_pos = self.grasp_panel_pose[:3]
        if self.stage == 1:
            duration = max(self._duration(75), 25)
            fraction = self._synchronized_bimanual_progress(duration)
            smooth = 0.5 - 0.5 * math.cos(math.pi * fraction)
            height = (
                (1.0 - smooth) * float(start_pos[2].item())
                + smooth * self.lift_target_z
            )
            position = torch.tensor(
                (
                    float(start_pos[0].item()),
                    RACK_CENTER_Y,
                    height,
                ),
                device=device,
            )
            self._drive_panel_pose(position, self.grasp_panel_pose[3:7])
            if fraction >= 1.0:
                live_lift_pose = self._panel_pose()
                # Height clearance is the only stage-1 gate.  Centre and
                # horizontal attitude are corrected explicitly in stage 2;
                # requiring them here deadlocks an already-safe lifted panel.
                if float(pz) >= self.rotation_safe_z - 0.02:
                    self.rotation_center_x = float(live_lift_pose[0].item())
                    self.rotation_center_y = float(live_lift_pose[1].item())
                    # Never slide the wrists inside an already closed grasp.
                    # Re-capture the two measured panel->hand transforms and
                    # rotate that exact closed chain as one rigid mechanism.
                    self._capture_grasp_offsets(
                        update_clearance=False, preserve_measured=True
                    )
                    self._begin_stage(3, "safe_height_reached_closed_chain_rotation_started")
                elif self._stage_elapsed() >= duration + max(
                    self._duration(240), 60
                ):
                    self._fail(
                        "object_relative_lift_clearance_not_reached",
                        panel_height=pz,
                        required_height=self.rotation_safe_z,
                    )
            return

        # At safe height, re-capture the measured closed chain and let both
        # wrists settle symmetrically before rotation.  Do not translate the
        # panel toward the nail yet.
        if self.stage == 2:
            position = torch.tensor(
                (self.rotation_center_x, self.rotation_center_y, self.lift_target_z),
                device=device,
            )
            self._drive_panel_pose(position, self.grasp_panel_pose[3:7])
            left_pos_error, left_rot_error = self.left.tracking_pose_error()
            right_pos_error, right_rot_error = self.right.tracking_pose_error()
            symmetry_error = self._bimanual_panel_symmetry_error()
            aligned = (
                max(left_pos_error, right_pos_error) <= 0.015
                and max(left_rot_error, right_rot_error) <= 0.06
                and symmetry_error <= 0.015
                and self._bimanual_grasp_is_valid()
            )
            self.rotation_alignment_stable_ticks = (
                self.rotation_alignment_stable_ticks + 1 if aligned else 0
            )
            if self.rotation_alignment_stable_ticks >= 2:
                self._begin_stage(3, "bimanual_grasp_realigned_rotation_started")
            elif self._stage_elapsed() >= max(self._duration(180), 60):
                self._fail(
                    "bimanual_pre_rotation_alignment_not_reached",
                    alignment_position_error=max(left_pos_error, right_pos_error),
                    alignment_rotation_error=max(left_rot_error, right_rot_error),
                    grasp_symmetry_error=symmetry_error,
                )
            return

        # Symmetry is corrected continuously during rotation; it is not a
        # blocking gate because waiting for perfection can lose the grasp.
        symmetry_error = self._bimanual_panel_symmetry_error()

        # Rotation turns gravity into tangential load on the pads.  Once the
        # physical grasp and all clearance gates are verified, hold the real
        # Panda actuator limit continuously; do not pulse open/close targets.
        # Preserve the verified contact preload captured at grasp time. A 70 N
        # per-finger override drove Newton joints below their lower limit and
        # destroyed the contact manifold as rotation began.
        self.left.gripper_force_hold_effort = POLICY_GRIP_FORCE_PER_FINGER
        self.right.gripper_force_hold_effort = POLICY_GRIP_FORCE_PER_FINGER
        self.left.gripper_mode = "FIXED_FORCE_CLOSE"
        self.right.gripper_mode = "FIXED_FORCE_CLOSE"
        # Rotate only after the panel is high, Y-centred and X-aligned to the nail tip.
        duration = max(self._duration(75), 30)
        fraction = self._synchronized_bimanual_progress(duration)
        smooth = 0.5 - 0.5 * math.cos(math.pi * fraction)
        position = torch.tensor(
            (
                self.rotation_center_x,
                self.rotation_center_y,
                self.lift_target_z,
            ),
            device=device,
        )
        self._drive_panel_pose(position, self._vertical_quat((89.0 / 90.0) * smooth), rotate_grasp=True)
        if fraction >= 1.0:
            live_vertical_pose = self._panel_pose()
            live_rotation = matrix_from_quat(live_vertical_pose[3:7].reshape(1, 4))[0]
            # The panel's local +X long axis must point downward, which puts
            # the handle on local -X at the top. This geometric test is
            # invariant to the q/-q quaternion representation.
            vertical_axis_alignment = abs(float(live_rotation[2, 0].item()))
            panel_bottom_z = float(live_vertical_pose[2].item()) - PANEL_HALF_LENGTH
            rack_clearance_z = float(self.initial_panel_pos[2].item()) + 0.030
            grasp_held = (
                # The loaded thin panel can compress the measured total jaw
                # width below 20 mm while remaining physically enclosed.
                # Reject a fully crossed/closed jaw, not a valid 12--20 mm
                # maximum-force grasp.
                self._grasp_width_valid(self.left, 0.010, 0.052)
                and self._grasp_width_valid(self.right, 0.010, 0.052)
            )
            # Rack/floor contact after rotation is recoverable while both
            # grippers still hold the nearly vertical panel.  The following
            # leveling and nail-height stages lift it clear again, so bottom
            # height is diagnostic rather than a phase-blocking condition.
            if vertical_axis_alignment >= 0.90 and grasp_held:
                self._event(
                    "physical_panel_lift_verified",
                    panel_height=float(live_vertical_pose[2].item()),
                    panel_bottom_height=panel_bottom_z,
                    rack_clearance_height=rack_clearance_z,
                    bottom_contact_recoverable=panel_bottom_z < rack_clearance_z,
                    vertical_axis_alignment=vertical_axis_alignment,
                )
                # Cancel the last rotation convergence target before the physics step.
                # Hold the measured EE poses for one tick, then phase 6 sends X transport.
                for arm in (self.left, self.right):
                    measured = arm.pose_w()[0].clone()
                    arm.set_target(measured[:3], measured[3:7])
                    arm.close_step()
                self._transition(6, "panel_lifted_with_grasp_held_rotation_acceptable")
                self.post_rotation_level_position = live_vertical_pose[:3].clone()
                self._begin_stage(-1, "post_rotation_bimanual_leveling_started")
            else:
                self._fail(
                    "physical_panel_lift_not_detected",
                    panel_height=float(live_vertical_pose[2].item()),
                    panel_bottom_height=panel_bottom_z,
                    required_bottom_height=rack_clearance_z,
                    vertical_axis_alignment=vertical_axis_alignment,
                    grasp_held=grasp_held,
                )

    def _step_phase_6(self, panel_pose, px, py, pz, device):
        if self.stage == -1:
            # Preserve the successful physical grasp exactly as measured at
            # the end of rotation. Replaying an ideal object transform here
            # separated both fingers from the compliant panel.
            self.left.close_step()
            self.right.close_step()
            live_level_pose = self._panel_pose()
            quat_xyzw = live_level_pose[[4, 5, 6, 3]]
            live_level_rotation = matrix_from_quat(quat_xyzw.reshape(1, 4))[0]
            edge_level_error = abs(float(live_level_rotation[2, 1].item()))
            if self._stage_elapsed() >= 2:
                self._event(
                    "post_rotation_bimanual_leveling_completed",
                    panel_edge_level_error=edge_level_error,
                    panel_top_edge_world=live_level_rotation[:, 1].detach().cpu().tolist(),
                    panel_long_axis_world=live_level_rotation[:, 0].detach().cpu().tolist(),
                )
                self.waypoint_start = None
                self._begin_stage(0, "leveled_panel_ready_for_nail_height")
            return

        if self.stage == 0:
            # Move only in Z, but gate on the live ring centre rather than
            # elapsed trajectory time.  Compliant grasp slip can otherwise
            # leave the ring far below the nail while the hands finish.
            if self.waypoint_start is None:
                self.waypoint_start = (
                    panel_pose[:3].clone(), self._vertical_quat(89.0 / 90.0)
                )
                self.hang_hand_waypoint_start = {
                    "left": self.left.pose_w()[0].clone(),
                    "right": self.right.pose_w()[0].clone(),
                }
                self.hang_ring_start = self._ring_center_world(panel_pose).clone()
                self.hang_ring_goal = self.hang_ring_start.clone()
                self.hang_ring_goal[2] = self._ring_target_world()[2]
                self.hang_hand_correction = torch.zeros(3, device=device)
            duration = max(self._duration(15), 6)
            fraction = min(1.0, self._stage_elapsed() / duration)
            smooth = fraction * fraction * (3.0 - 2.0 * fraction)
            planned_delta = (self.hang_ring_goal - self.hang_ring_start) * smooth
            ring_error = self._ring_center_world(panel_pose) - self._ring_target_world()
            if fraction >= 1.0:
                self.hang_hand_correction -= torch.clamp(ring_error, -0.010, 0.010)
                self.hang_hand_correction[0:2] = 0.0
                self.hang_hand_correction = torch.clamp(
                    self.hang_hand_correction, -0.030, 0.030
                )
            for name, arm in (("left", self.left), ("right", self.right)):
                start = self.hang_hand_waypoint_start[name]
                arm.set_target(
                    start[:3] + planned_delta + self.hang_hand_correction,
                    start[3:7],
                )
                arm.close_step()
            if fraction >= 1.0 and abs(float(ring_error[2].item())) <= HANGER_HEAD_PASSAGE_RADIUS:
                self.waypoint_start = (
                    panel_pose[:3].clone(), self._vertical_quat(89.0 / 90.0)
                )
                self.hang_hand_waypoint_start = {
                    "left": self.left.pose_w()[0].clone(),
                    "right": self.right.pose_w()[0].clone(),
                }
                self.hang_ring_start = self._ring_center_world(panel_pose).clone()
                self.hang_ring_goal = self._ring_target_world().clone()
                self.hang_hand_correction = torch.zeros(3, device=device)
                self._begin_stage(1, "nail_height_completed_direct_x_transport")
            elif self._stage_elapsed() >= max(self._duration(90), 30):
                self._fail(
                    "ring_nail_height_not_reached",
                    ring_center=self._ring_center_world(panel_pose).detach().cpu().tolist(),
                    ring_target=self._ring_target_world().detach().cpu().tolist(),
                    vertical_error=float(ring_error[2].item()),
                )
            return

        # Then translate along X at the cleared height until the ring is
        # directly above the nail. Y remains nail-centred throughout.
        duration = max(self._duration(36), 12)
        fraction = min(1.0, self._stage_elapsed() / duration)
        smooth = fraction * fraction * (3.0 - 2.0 * fraction)
        planned_delta = (self.hang_ring_goal - self.hang_ring_start) * smooth
        ring_error = self._ring_center_world(panel_pose) - self._ring_target_world()
        if fraction >= 1.0:
            self.hang_hand_correction -= torch.clamp(ring_error, -0.010, 0.010)
            self.hang_hand_correction = torch.clamp(
                self.hang_hand_correction, -0.030, 0.030
            )
        for name, arm in (("left", self.left), ("right", self.right)):
            start = self.hang_hand_waypoint_start[name]
            arm.set_target(
                start[:3] + planned_delta + self.hang_hand_correction,
                start[3:7],
            )
            arm.close_step()
        radial_error = torch.sqrt(ring_error[1] ** 2 + ring_error[2] ** 2)
        ring_aligned = (
            abs(float(ring_error[0].item())) <= 0.070
            and float(radial_error.item()) <= HANGER_HEAD_PASSAGE_RADIUS
        )
        if fraction >= 1.0 and ring_aligned:
            self._transition(7, "panel_ring_center_positioned_at_nail")
        elif self._stage_elapsed() >= max(self._duration(90), 30):
            self._fail(
                "ring_3d_alignment_not_reached",
                ring_center=self._ring_center_world(panel_pose).detach().cpu().tolist(),
                ring_target=self._ring_target_world().detach().cpu().tolist(),
                ring_error=ring_error.detach().cpu().tolist(),
            )

    def _step_phase_7(self, panel_pose, px, py, pz, device):
        # The preceding phase puts the ring in front of the enlarged nail
        # head. Keep both jaws closed and advance the shared grasp along
        # +X until the ring plane is on the shaft before releasing. The
        # hanger rod is on +X; the exposed nail head faces the robots at -X.
        ring_target = self._ring_target_world()
        shaft_seat_target = ring_target.clone()
        shaft_seat_target[0] += HANGER_NAIL_HEAD_THICKNESS + HANGER_SHAFT_SEAT_DEPTH
        if self.stage == 0:
            self.hang_hand_waypoint_start = {
                "left": self.left.pose_w()[0].clone(),
                "right": self.right.pose_w()[0].clone(),
            }
            self.hang_ring_start = self._ring_center_world(panel_pose).clone()
            self.hang_ring_goal = shaft_seat_target.clone()
            self.hang_hand_correction = torch.zeros(3, device=device)
            self._begin_stage(1, "nail_head_cleared_seating_ring_on_shaft")

        duration = max(self._duration(12), 6)
        fraction = min(1.0, self._stage_elapsed() / duration)
        smooth = fraction * fraction * (3.0 - 2.0 * fraction)
        planned_delta = (self.hang_ring_goal - self.hang_ring_start) * smooth
        ring_center = self._ring_center_world(panel_pose)
        ring_error = ring_center - shaft_seat_target
        if fraction >= 1.0:
            self.hang_hand_correction -= torch.clamp(ring_error, -0.006, 0.006)
            self.hang_hand_correction = torch.clamp(
                self.hang_hand_correction, -0.020, 0.020
            )
        for name, arm in (("left", self.left), ("right", self.right)):
            start = self.hang_hand_waypoint_start[name]
            arm.set_target(
                start[:3] + planned_delta + self.hang_hand_correction,
                start[3:7],
            )
            arm.close_step()
        radial_error = torch.sqrt(ring_error[1] ** 2 + ring_error[2] ** 2)
        shaft_back_x = float(ring_target[0].item()) + HANGER_NAIL_HEAD_THICKNESS
        shaft_front_x = shaft_back_x + HANGER_NAIL_LENGTH
        ring_plane_x = float(ring_center[0].item())
        ring_plane_on_shaft = (
            shaft_back_x + 0.005 <= ring_plane_x <= shaft_front_x - 0.005
        )
        settled = (
            fraction >= 1.0
            and ring_plane_on_shaft
            and float(radial_error.item()) <= HANGER_HEAD_PASSAGE_RADIUS
        )
        if settled:
                self._event(
                    "ring_seated_behind_nail_head",
                    panel_height=pz,
                    ring_center=ring_center.detach().cpu().tolist(),
                    ring_target=shaft_seat_target.detach().cpu().tolist(),
                    insertion_error=abs(float(ring_error[0].item())),
                    lateral_error=abs(float(ring_error[1].item())),
                    vertical_error=abs(float(ring_error[2].item())),
                    radial_error=float(radial_error.item()),
                    shaft_interval=[shaft_back_x, shaft_front_x],
                    contact_required=False,
                )
                self.release_panel_height = pz
                self._transition(9, "ring_on_shaft_behind_head_release_allowed")
        elif self._stage_elapsed() >= max(self._duration(45), 18):
            self._fail(
                "panel_alignment_above_hooks_not_reached",
                ring_center=ring_center.detach().cpu().tolist(),
                ring_target=shaft_seat_target.detach().cpu().tolist(),
                ring_error=ring_error.detach().cpu().tolist(),
                shaft_interval=[shaft_back_x, shaft_front_x],
            )

    def _step_phase_8(self, panel_pose, px, py, pz, device):
        if self.stage == 0:
            self.hang_entry_panel_goal = self._hang_panel_target()
            self.hang_seat_panel_goal = self._hang_panel_target()
            self._begin_waypoint(1, "lowering_onto_hooks", self.hang_entry_panel_goal, self._hang_transport_quat())
            self.hang_hand_waypoint_start = {
                "left": self.left.pose_w()[0].clone(),
                "right": self.right.pose_w()[0].clone(),
            }
        duration = max(self._duration(60), 25)
        entry = self.hang_entry_panel_goal
        seat = self.hang_seat_panel_goal
        fraction = min(1.0, self._stage_elapsed() / duration)
        smooth = fraction * fraction * (3.0 - 2.0 * fraction)
        delta = (seat - entry) * smooth
        for name, arm in (("left", self.left), ("right", self.right)):
            start = self.hang_hand_waypoint_start[name]
            arm.set_target(
                start[:3] + delta + self._grasp_slip_correction(name, arm),
                start[3:7],
            )
            arm.close_step()
        if fraction >= 1.0 and self._stage_elapsed() >= duration:
            contacts = self._newton_contact_flags()
            # Release only after the nail is geometrically inside the ring:
            # X is insertion along the nail shaft, while Y and Z constrain
            # the ring bore around the shaft and below its supporting point.
            target_panel = self._hang_panel_target()
            insertion_error = abs(px - float(target_panel[0].item()))
            lateral_error = abs(py - float(target_panel[1].item()))
            seating_error = pz - float(target_panel[2].item())
            ring_fully_over_nail = (
                insertion_error <= 0.050
                and lateral_error <= 0.040
                and -0.080 <= seating_error <= 0.035
            )
            if ring_fully_over_nail:
                self._event(
                    "ring_fully_over_nail",
                    panel_height=pz,
                    insertion_error=insertion_error,
                    lateral_error=lateral_error,
                    seating_error=seating_error,
                    contacts=contacts,
                )
                self._transition(9, "ring_fully_inserted_release_allowed")
            else:
                self._fail(
                    "ring_not_fully_over_nail",
                    insertion_error=insertion_error,
                    lateral_error=lateral_error,
                    seating_error=seating_error,
                )

    def _step_phase_9(self, panel_pose, px, py, pz, device):
        self.left.release_gripper()
        self.right.release_gripper()
        opened = min(self.left.measured_gripper_width(), self.right.measured_gripper_width())
        if self._stage_elapsed() >= max(self._duration(8), 4) and opened >= 0.075:
            for name, arm in (("left", self.left), ("right", self.right)):
                self.release_hand_pose[name] = arm.pose_w()[0].clone()
            self._transition(10, "grippers_released_panel")
        elif self._stage_elapsed() >= max(self._duration(90), 45):
            self._fail("grippers_did_not_open_for_release", opened_width=opened)

    def _step_phase_10(self, panel_pose, px, py, pz, device):
        # After release, clear only along the panel-side Y directions and
        # finish.  An additional X retreat can sweep the open fingers back
        # through the hanging panel and knock it off the nail.
        lateral_clearance = 0.12
        lateral_duration = max(self._duration(12), 6)
        if self.stage == 0:
            fraction = min(1.0, self._stage_elapsed() / lateral_duration)
            smooth = fraction * fraction * (3.0 - 2.0 * fraction)
            for name, arm, sign in (("left", self.left, 1.0), ("right", self.right, -1.0)):
                start = self.release_hand_pose[name]
                target = start[:3].clone()
                target[1] += sign * lateral_clearance * smooth
                arm.set_target(target, start[3:7])
                arm.set_gripper(GRIPPER_OPEN_WIDTH)
            if fraction >= 1.0:
                self._event("hands_cleared_panel_laterally")
                self._begin_stage(1, "released_panel_physics_settle_started")
        else:
            for name, arm, sign in (("left", self.left, 1.0), ("right", self.right, -1.0)):
                start = self.release_hand_pose[name]
                target = start[:3].clone()
                target[1] += sign * lateral_clearance
                arm.set_target(target, start[3:7])
                arm.set_gripper(GRIPPER_OPEN_WIDTH)
            if self._stage_elapsed() >= max(self._duration(24), 12):
                self._finish_episode(panel_pose)

    def high_level_tick(self):
        if self.done:
            return
        if not self.initialized:
            self.initialize()
        self.total_control_ticks += 1
        self.phase_ticks += 1
        panel_pose = self._panel_pose()
        px, py, pz = (float(value.item()) for value in panel_pose[:3])
        self.panel_height_history.append(pz)
        device = self.left.robot.device

        if self.task_mode == "hang" and not self.hang_grasp_ready:
            self._standalone_hang_grasp_tick()
            self._update_last_action()
            return

        try:
            self.active_module.step(self, panel_pose, px, py, pz, device)
        except ModuleContractError as error:
            self._fail("task_module_execution_contract_failed", contract_error=str(error))
            return

        if self.total_control_ticks % 30 == 0:
            left_position_error, left_rotation_error = self.left.tracking_pose_error()
            right_position_error, right_rotation_error = self.right.tracking_pose_error()
            print(
                f"[AUTO_OPS_PROGRESS] tick={self.total_control_ticks} phase={self.phase_index} stage={self.stage} "
                f"left_pos={left_position_error:.4f} left_rot={left_rotation_error:.4f} "
                f"right_pos={right_position_error:.4f} right_rot={right_rotation_error:.4f} "
                f"left_grip={self.left.measured_gripper_width():.4f} right_grip={self.right.measured_gripper_width():.4f} "
                f"left_cmd={self.left.gripper_width:.4f}/{self.left.gripper_mode} "
                f"right_cmd={self.right.gripper_width:.4f}/{self.right.gripper_mode} "
                f"panel=({px:.3f},{py:.3f},{pz:.3f})",
                flush=True,
            )

        phase_timeouts = {0: 320, 1: 520, 2: 170, 3: 560, 4: 180, 5: 280, 6: 200, 7: 120, 8: 180, 9: 90, 10: 120}
        if not self.done and self.phase_ticks > phase_timeouts[self.phase_index]:
            self._fail(f"phase_{self.phase_index}_timeout")
        self._update_last_action()

    def _finish_episode(self, panel_pose):
        """Success needs the panel resting in the hung pose without the grippers."""
        # Once released, the panel may yaw or roll slightly around the nail.
        # Hanging validity depends on its long local X axis remaining vertical,
        # not on matching one arbitrary full quaternion.
        live_rotation = self._panel_rotation(panel_pose)
        vertical_axis_alignment = torch.clamp(torch.abs(live_rotation[2, 0]), 0.0, 1.0)
        orientation_error = float(torch.acos(vertical_axis_alignment).item())
        px, py, pz = (float(value.item()) for value in panel_pose[:3])
        supported = self._newton_contact_flags()["panel_hanger"]
        # Contact-name reporting can be incomplete, but the nail must still lie
        # geometrically inside the released ring.  Height retention alone can
        # incorrectly accept a panel that fell below the hanger.
        ring_center = self._ring_center_world(panel_pose)
        ring_target = self._ring_target_world()
        ring_error = ring_center - ring_target
        insertion_error = abs(float(ring_error[0].item()))
        radial_error = math.sqrt(
            float(ring_error[1].item()) ** 2 + float(ring_error[2].item()) ** 2
        )
        nail_head_front_x = float(ring_target[0].item())
        nail_base_x = nail_head_front_x + HANGER_NAIL_HEAD_THICKNESS
        nail_tip_x = nail_base_x + HANGER_NAIL_LENGTH
        ring_plane_x = float(ring_center[0].item())
        ring_on_nail_shaft = (
            nail_base_x - 0.010 <= ring_plane_x <= nail_tip_x + 0.010
        )
        nail_inside_ring = (
            ring_on_nail_shaft
            and radial_error <= HANDLE_RING_RADIUS + 0.015
        )
        heights = list(self.panel_height_history)
        stationary = len(heights) < 2 or (max(heights) - min(heights)) <= 0.01
        release_height = getattr(self, "release_panel_height", pz)
        retained_height = pz >= release_height - 0.05
        panel_bottom_height = pz - PANEL_HALF_LENGTH
        suspended = panel_bottom_height >= float(self.initial_panel_pos[2].item()) + 0.020
        placed = nail_inside_ring and suspended
        if placed and float(vertical_axis_alignment.item()) >= 0.85:
            self.success = True
            self.failure_reason = None
            self.done = True
            self._event(
                "episode_finished",
                success=True,
                panel_pose=panel_pose.detach().cpu().tolist(),
                orientation_error=orientation_error,
                vertical_axis_alignment=float(vertical_axis_alignment.item()),
                release_height=release_height,
                retained_height=retained_height,
                panel_bottom_height=panel_bottom_height,
                suspended=suspended,
                stationary=stationary,
                ring_center=ring_center.detach().cpu().tolist(),
                ring_target=ring_target.detach().cpu().tolist(),
                ring_insertion_error=insertion_error,
                ring_radial_error=radial_error,
                nail_base_x=nail_base_x,
                nail_tip_x=nail_tip_x,
                ring_on_nail_shaft=ring_on_nail_shaft,
                nail_inside_ring=nail_inside_ring,
                panel_hanger_contact=supported,
                failure_reason=None,
            )
        else:
            self._fail(
                "physical_hang_success_conditions_not_met",
                panel_hanger_contact=supported,
                placed=placed,
                stationary=stationary,
                retained_height=retained_height,
                panel_bottom_height=panel_bottom_height,
                suspended=suspended,
                ring_center=ring_center.detach().cpu().tolist(),
                ring_target=ring_target.detach().cpu().tolist(),
                ring_insertion_error=insertion_error,
                ring_radial_error=radial_error,
                nail_base_x=nail_base_x,
                nail_tip_x=nail_tip_x,
                ring_on_nail_shaft=ring_on_nail_shaft,
                nail_inside_ring=nail_inside_ring,
                orientation_error=orientation_error,
                vertical_axis_alignment=float(vertical_axis_alignment.item()),
            )

    def _update_last_action(self):
        commands = []
        for arm in (self.left, self.right):
            pose = arm.pose_w()
            position_error, rotation_error = compute_pose_error(
                pose[:, :3], pose[:, 3:7], arm.target_pose_w[:, :3], arm.target_pose_w[:, 3:7]
            )
            dp = torch.clamp(position_error[0], -0.04, 0.04).detach().cpu().numpy()
            dr = torch.clamp(rotation_error[0], -0.15, 0.15).detach().cpu().numpy()
            commands.append((dp, dr, arm.gripper_width))
        self.last_action = pack_bimanual_action(*commands[0], *commands[1])

    def _apply_closed_chain_projection(self):
        if self.grasp_offsets is None or self.phase_index < 5:
            return
        """Project the two Pink solutions onto one rigid bimanual grasp QP."""
        if self.left.last_joint_target is None or self.right.last_joint_target is None:
            return
        arms = (self.left, self.right)
        jacobians_world = []
        measured = []
        nominal_delta = []
        for arm in arms:
            configuration = arm.controller.pink_configuration
            jacobian_local = configuration.get_frame_jacobian("panda_hand")
            ordering = arm.controller.pink_to_isaac_lab_controlled_ordering
            jacobian_local = jacobian_local[:, ordering]
            hand_rotation = matrix_from_quat(arm.pose_w()[:, 3:7])[0].detach().cpu().numpy()
            rotate_twist = np.zeros((6, 6), dtype=np.float64)
            rotate_twist[:3, :3] = hand_rotation
            rotate_twist[3:, 3:] = hand_rotation
            jacobians_world.append(rotate_twist @ jacobian_local)
            q = arm.robot.data.joint_pos.torch[0, arm.arm_joint_ids].detach().cpu().numpy().astype(np.float64)
            target = arm.last_joint_target[0].detach().cpu().numpy().astype(np.float64)
            measured.append(q)
            nominal_delta.append(target - q)

        left_j, right_j = jacobians_world
        left_pose = self.left.pose_w()[0]
        right_pose = self.right.pose_w()[0]
        separation = (right_pose[:3] - left_pose[:3]).detach().cpu().numpy().astype(np.float64)
        skew_separation = np.array(
            ((0.0, -separation[2], separation[1]),
             (separation[2], 0.0, -separation[0]),
             (-separation[1], separation[0], 0.0)), dtype=np.float64
        )
        # Pinocchio Motion/Jacobian vectors are [linear; angular]. For a rigid
        # grasp: v_R - v_L - omega_L x (p_R-p_L) = 0 and omega_R-omega_L = 0.
        linear_constraint = np.concatenate(
            (-left_j[:3] + skew_separation @ left_j[3:], right_j[:3]), axis=1
        )
        angular_constraint = np.concatenate((-left_j[3:], right_j[3:]), axis=1)
        equality = np.concatenate((linear_constraint, angular_constraint), axis=0)

        # Translational relative error must be expressed in the same world frame
        # as the rotated Jacobians. subtract_frame_transforms returns left-local
        # translation, which caused the coupled solve to command the opposite way.
        current_separation = right_pose[:3] - left_pose[:3]
        target_separation = self.right.target_pose_w[0, :3] - self.left.target_pose_w[0, :3]
        relative_position_error_world = target_separation - current_separation
        zero = torch.zeros((1, 3), device=left_pose.device, dtype=left_pose.dtype)
        _, current_relative_quat = subtract_frame_transforms(
            zero, left_pose[3:7].reshape(1, 4), zero, right_pose[3:7].reshape(1, 4)
        )
        _, target_relative_quat = subtract_frame_transforms(
            zero, self.left.target_pose_w[:, 3:7], zero, self.right.target_pose_w[:, 3:7]
        )
        _, relative_rotation_error_local = compute_pose_error(
            zero, current_relative_quat, zero, target_relative_quat
        )
        left_rotation_world = matrix_from_quat(left_pose[3:7].reshape(1, 4))[0]
        relative_rotation_error_world = left_rotation_world @ relative_rotation_error_local[0]
        correction_goal = np.concatenate((
            0.55 * relative_position_error_world.detach().cpu().numpy(),
            0.55 * relative_rotation_error_world.detach().cpu().numpy(),
        )).astype(np.float64)
        nominal = np.concatenate(nominal_delta)

        # Equality-constrained least-squares QP:
        # min ||dq-dq_pink||^2, subject to A dq = b.
        residual = correction_goal - equality @ nominal
        regularized = equality @ equality.T + 1.0e-4 * np.eye(6)
        correction = equality.T @ np.linalg.solve(regularized, residual)
        coupled_delta = np.clip(nominal + correction, -0.06, 0.06)

        for index, arm in enumerate(arms):
            start = 7 * index
            target_np = measured[index] + coupled_delta[start:start + 7]
            target = torch.tensor(target_np, device=arm.robot.device, dtype=torch.float32).unsqueeze(0)
            arm.robot.set_joint_position_target_index(target=target, joint_ids=arm.arm_joint_ids)
            arm.last_joint_target = target.detach().clone()
        if self.total_control_ticks % 5 == 0:
            print(
                f"[AUTO_OPS_COUPLED_QP] tick={self.total_control_ticks} "
                f"relative_pos_error={float(torch.linalg.norm(relative_position_error_world).item()):.4f} "
                f"relative_rot_error={float(torch.linalg.norm(relative_rotation_error_world).item()):.4f} "
                f"projection_norm={float(np.linalg.norm(correction)):.4f}", flush=True,
            )

    def apply(self, physics_step):
        finite_state = (
            torch.isfinite(self.panel.data.root_pose_w.torch).all()
            and torch.isfinite(self.left.robot.data.joint_pos.torch).all()
            and torch.isfinite(self.right.robot.data.joint_pos.torch).all()
        )
        if not bool(finite_state.item()):
            if not self.done:
                self._fail("non_finite_newton_state")
            return
        if physics_step % self.control_decimation == 0:
            self.high_level_tick()
        solve_ik = physics_step % 2 == 0
        self.left.apply(solve_ik=solve_ik)
        self.right.apply(solve_ik=solve_ik)
        self.left.robot.write_data_to_sim()
        self.right.robot.write_data_to_sim()

    @staticmethod
    def _rot6d(quaternion_wxyz):
        matrix = matrix_from_quat(quaternion_wxyz.unsqueeze(0))[0]
        return matrix[:2, :].reshape(-1).detach().cpu().numpy()

    def sample_metadata(self):
        left_pose = self.left.pose_w()[0]
        right_pose = self.right.pose_w()[0]
        left_width = float(self.left.robot.data.joint_pos.torch[0, self.left.finger_joint_ids].sum().item())
        right_width = float(self.right.robot.data.joint_pos.torch[0, self.right.finger_joint_ids].sum().item())
        state = pack_bimanual_state(
            left_pose[:3].detach().cpu().numpy(), self._rot6d(left_pose[3:7]), left_width,
            right_pose[:3].detach().cpu().numpy(), self._rot6d(right_pose[3:7]), right_width,
        )
        nominal_ticks = {0: 110, 1: 70, 2: 80, 3: 220, 4: 70, 5: 190, 6: 100, 7: 50, 8: 90, 9: 40, 10: 70}
        progress = min(1.0, self.phase_ticks / max(1, self._duration(nominal_ticks[self.phase_index])))
        return {
            "annotation": phase_annotation(self.phase_index, progress),
            "observation_state": state,
            "action": self.last_action,
        }
