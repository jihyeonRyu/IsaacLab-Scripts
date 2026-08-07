"""Lift task ownership, execution, and physical contracts."""

from typing import Any

from .base import AutoOpsTaskModule, ModuleContractError


class LiftTaskModule(AutoOpsTaskModule):
    name = "lift"
    start_phase = 3
    end_phase = 5

    def owns(self, phase_index: int, stage: int) -> bool:
        return ((phase_index == 3 and stage >= 4) or phase_index == 4 or (phase_index == 5 and stage < 3))

    def step(self, controller: Any, panel_pose: Any, px: float, py: float, pz: float, device: Any) -> None:
        phase = controller.phase_index
        if phase not in (3, 4, 5):
            raise ModuleContractError(f"lift cannot execute phase={phase}")
        self._run_phase(controller, phase, panel_pose, px, py, pz, device)

    def validate_enter(self, controller: Any) -> None:
        panel_pose = controller._panel_pose()
        displacement = float(controller.initial_panel_pos[0].item() - panel_pose[0].item())
        if controller.task_mode == "long" and displacement < controller.pull_handoff_minimum:
            raise ModuleContractError("Lift entered before Pull produced a clear panel")

    def validate_exit(self, controller: Any) -> None:
        if controller.grasp_offsets is None:
            raise ModuleContractError("Lift handoff requires captured bimanual grasp transforms")
        if not controller._bimanual_grasp_is_valid():
            raise ModuleContractError("Lift handoff requires a physically enclosed bimanual grasp")
