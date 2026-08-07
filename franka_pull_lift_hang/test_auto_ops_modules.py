"""Fast regression tests for task ownership and long handoff contracts."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from task_modules import LongTaskOrchestrator, ModuleContractError


class _Scalar(float):
    def item(self):
        return float(self)


class _Pose(list):
    def __init__(self, values):
        super().__init__(_Scalar(value) for value in values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return [float(value) for value in self]


class _Arm:
    def __init__(self, width: float):
        self.width = width

    def measured_gripper_width(self) -> float:
        return self.width


class _Controller:
    def __init__(self):
        self.task_mode = "long"
        self.phase_index = 3
        self.stage = 3
        self.total_control_ticks = 10
        self.initial_panel_pos = _Pose((0.42, 0.0, 0.55))
        self.panel_pose = _Pose((0.05, 0.01, 0.55, 1.0, 0.0, 0.0, 0.0))
        self.pull_handoff_minimum = 0.325
        self.pull_handoff_lateral_tolerance = 0.06
        self.left = _Arm(0.079)
        self.right = _Arm(0.079)
        self.grasp_offsets = None

    def _panel_pose(self):
        return self.panel_pose

    def _bimanual_grasp_is_valid(self):
        return True


class ModuleBoundaryTests(unittest.TestCase):
    def test_phase_three_boundary_is_stage_owned(self):
        orchestrator = LongTaskOrchestrator("long")
        self.assertEqual(orchestrator.module_for_state(3, 3).name, "pull")
        self.assertEqual(orchestrator.module_for_state(3, 4).name, "lift")
        self.assertEqual(orchestrator.module_for_state(5, 1).name, "lift")
        self.assertEqual(orchestrator.module_for_state(5, 3).name, "hang")

    def test_pull_to_lift_handoff_records_physical_state(self):
        controller = _Controller()
        orchestrator = LongTaskOrchestrator("long")
        controller.stage = 4
        handoff = orchestrator.sync(controller)
        self.assertIsNotNone(handoff)
        self.assertEqual(handoff.source, "pull")
        self.assertEqual(handoff.target, "lift")
        self.assertEqual(orchestrator.current.name, "lift")

    def test_pull_handoff_rejects_closed_gripper(self):
        controller = _Controller()
        controller.left.width = 0.02
        controller.stage = 4
        orchestrator = LongTaskOrchestrator("long")
        with self.assertRaises(ModuleContractError):
            orchestrator.sync(controller)

    def test_lift_to_hang_requires_grasp_transform(self):
        controller = _Controller()
        orchestrator = LongTaskOrchestrator("long")
        controller.stage = 4
        orchestrator.sync(controller)
        controller.phase_index = 6
        controller.stage = 0
        with self.assertRaises(ModuleContractError):
            orchestrator.sync(controller)

    def test_standalone_pull_completes_at_lift_boundary(self):
        controller = _Controller()
        controller.stage = 4
        orchestrator = LongTaskOrchestrator("pull")
        self.assertIsNone(orchestrator.sync(controller))
        self.assertTrue(orchestrator.completed)

    def test_lift_hang_handoff_starts_at_rotation(self):
        controller = _Controller()
        controller.phase_index = 5
        controller.stage = 3
        controller.grasp_offsets = {"left": object(), "right": object()}
        orchestrator = LongTaskOrchestrator("lift_hang")
        handoff = orchestrator.sync(controller)
        self.assertEqual(handoff.source, "lift")
        self.assertEqual(handoff.target, "hang")


if __name__ == "__main__":
    unittest.main()
