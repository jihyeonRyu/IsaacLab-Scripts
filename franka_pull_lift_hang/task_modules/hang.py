"""Hang task ownership, execution, and entry contract."""

from typing import Any

from .base import AutoOpsTaskModule, ModuleContractError


class HangTaskModule(AutoOpsTaskModule):
    name = "hang"
    start_phase = 6
    end_phase = 10

    def owns(self, phase_index: int, stage: int) -> bool:
        return (phase_index == 5 and stage >= 3) or phase_index >= 6

    def step(self, controller: Any, panel_pose: Any, px: float, py: float, pz: float, device: Any) -> None:
        self._run_phase(controller, controller.phase_index, panel_pose, px, py, pz, device)

    def validate_enter(self, controller: Any) -> None:
        if controller.grasp_offsets is None:
            raise ModuleContractError("Hang requires Lift's measured panel-to-hand transforms")
