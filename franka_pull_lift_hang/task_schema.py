"""Task taxonomy and bimanual GR00T state/action packing contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TaskPhase:
    phase_index: int
    major_task: str
    subtask: str
    instruction_en: str
    instruction_ko: str


TASK_PHASES = (
    TaskPhase(0, "pick", "left_approach_front_edge", "Approach the panel front edge with the left gripper.", "왼쪽 그리퍼로 판 앞쪽 엣지에 접근한다."),
    TaskPhase(1, "pick", "left_grasp_front_edge", "Grasp the panel front edge with the left gripper.", "왼쪽 그리퍼로 판 앞쪽 엣지를 잡는다."),
    TaskPhase(2, "pick", "left_pull_panel_forward", "Pull the panel forward with the left gripper.", "왼쪽 그리퍼로 판을 앞으로 당긴다."),
    TaskPhase(3, "pick", "bimanual_approach_side_edges", "Approach both exposed side edges with both grippers.", "양쪽 그리퍼로 노출된 판의 양옆에 접근한다."),
    TaskPhase(4, "pick", "bimanual_grasp_side_edges", "Grasp both side edges of the panel.", "양쪽 그리퍼로 판의 양옆을 잡는다."),
    TaskPhase(5, "pick", "bimanual_lift_panel", "Lift and rotate the panel with both arms.", "양팔로 판을 들어 올리고 세운다."),
    TaskPhase(6, "hang", "transport_panel_to_hanger", "Move the lifted panel toward the hanger.", "들어 올린 판을 행거 쪽으로 옮긴다."),
    TaskPhase(7, "hang", "align_panel_with_hooks", "Align the panel top edge above both hanger hooks.", "판의 위쪽 엣지를 양쪽 행거 고리 위에 정렬한다."),
    TaskPhase(8, "hang", "lower_panel_onto_hooks", "Lower the panel until its top edge is supported by both hooks.", "판의 위쪽 엣지가 양쪽 고리에 지지될 때까지 내린다."),
    TaskPhase(9, "hang", "release_panel", "Open both grippers and release the hanging panel.", "양쪽 그리퍼를 열어 걸린 판을 놓는다."),
    TaskPhase(10, "hang", "bimanual_retreat", "Move both grippers away from the hanging panel.", "양쪽 그리퍼를 걸린 판에서 물린다."),
)

PHASE_BY_INDEX = {phase.phase_index: phase for phase in TASK_PHASES}


def phase_annotation(phase_index: int, progress: float = 0.0) -> dict:
    """Return the frame-level task annotation changed only by the Auto Ops state machine."""
    phase = PHASE_BY_INDEX[phase_index]
    value = asdict(phase)
    # Dataset instructions are kept in English for GR00T/AutoOps training.
    value.pop("instruction_ko", None)
    value["phase_progress"] = float(np.clip(progress, 0.0, 1.0))
    return value


def pack_bimanual_state(left_eef_xyz, left_rot6d, left_gripper_width, right_eef_xyz, right_rot6d, right_gripper_width):
    """Pack [left 10D, right 10D] absolute observation state."""
    return np.concatenate(
        (
            np.asarray(left_eef_xyz, dtype=np.float32).reshape(3),
            np.asarray(left_rot6d, dtype=np.float32).reshape(6),
            np.asarray([left_gripper_width], dtype=np.float32),
            np.asarray(right_eef_xyz, dtype=np.float32).reshape(3),
            np.asarray(right_rot6d, dtype=np.float32).reshape(6),
            np.asarray([right_gripper_width], dtype=np.float32),
        )
    )


def pack_bimanual_action(left_delta_xyz, left_delta_rotvec, left_gripper_command, right_delta_xyz, right_delta_rotvec, right_gripper_command):
    """Pack [left 7D, right 7D] action; gripper commands are absolute widths."""
    return np.concatenate(
        (
            np.asarray(left_delta_xyz, dtype=np.float32).reshape(3),
            np.asarray(left_delta_rotvec, dtype=np.float32).reshape(3),
            np.asarray([left_gripper_command], dtype=np.float32),
            np.asarray(right_delta_xyz, dtype=np.float32).reshape(3),
            np.asarray(right_delta_rotvec, dtype=np.float32).reshape(3),
            np.asarray([right_gripper_command], dtype=np.float32),
        )
    )
