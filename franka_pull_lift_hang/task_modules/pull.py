"""Pull task ownership, execution, and exit contract."""

from typing import Any

from .base import AutoOpsTaskModule, ModuleContractError


class PullTaskModule(AutoOpsTaskModule):
    name = "pull"
    start_phase = 0
    end_phase = 3

    def owns(self, phase_index: int, stage: int) -> bool:
        return phase_index <= 2 or (phase_index == 3 and stage <= 3)

    def step(self, controller: Any, panel_pose: Any, px: float, py: float, pz: float, device: Any) -> None:
        phase = controller.phase_index
        if phase not in (0, 1, 2, 3):
            raise ModuleContractError(f"pull cannot execute phase={phase}")
        self._run_phase(controller, phase, panel_pose, px, py, pz, device)

    def validate_exit(self, controller: Any) -> None:
        panel_pose = controller._panel_pose()
        displacement = float(controller.initial_panel_pos[0].item() - panel_pose[0].item())
        if displacement < controller.pull_handoff_minimum:
            raise ModuleContractError(
                f"Pull handoff displacement {displacement:.4f} < {controller.pull_handoff_minimum:.4f}"
            )
        if abs(float(panel_pose[1].item())) > controller.pull_handoff_lateral_tolerance:
            raise ModuleContractError("Pull handoff panel is outside the rack lateral envelope")
        if controller.left.measured_gripper_width() < 0.060:
            raise ModuleContractError("Pull handoff requires the pulling gripper to be released")
