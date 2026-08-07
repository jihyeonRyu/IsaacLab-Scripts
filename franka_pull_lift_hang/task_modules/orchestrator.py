"""Composition-only orchestrator for standalone and long episodes."""

from typing import Any

from .base import AutoOpsTaskModule, ModuleContractError, ModuleHandoff
from .hang import HangTaskModule
from .lift import LiftTaskModule
from .pull import PullTaskModule

MODULES: dict[str, AutoOpsTaskModule] = {
    "pull": PullTaskModule(),
    "lift": LiftTaskModule(),
    "hang": HangTaskModule(),
}


class LongTaskOrchestrator:
    """Compose modules and validate handoffs; robot motion stays in modules."""

    def __init__(self, task_mode: str):
        sequences = {
            "long": ("pull", "lift", "hang"),
            "pull": ("pull",),
            "lift": ("lift",),
            "hang": ("hang",),
            "lift_hang": ("lift", "hang"),
        }
        try:
            self.sequence = tuple(MODULES[name] for name in sequences[task_mode])
        except KeyError as exc:
            raise ValueError(f"Unknown Auto Ops task mode: {task_mode!r}") from exc
        self.current = self.sequence[0]
        self.handoffs: list[ModuleHandoff] = []
        self.completed = False

    def module_for_state(self, phase_index: int, stage: int) -> AutoOpsTaskModule:
        for module in MODULES.values():
            if module.owns(phase_index, stage):
                return module
        return self.current

    def sync(self, controller: Any) -> ModuleHandoff | None:
        desired = self.module_for_state(controller.phase_index, controller.stage)
        if desired is self.current:
            return None
        if desired not in self.sequence:
            if self.current is not self.sequence[-1]:
                raise ModuleContractError(
                    f"Required module {desired.name} is missing from active sequence"
                )
            self.current.validate_exit(controller)
            self.completed = True
            return None
        current_index = self.sequence.index(self.current)
        desired_index = self.sequence.index(desired)
        if desired_index != current_index + 1:
            raise ModuleContractError(
                f"Illegal module transition {self.current.name}->{desired.name}"
            )
        self.current.validate_exit(controller)
        desired.validate_enter(controller)
        panel_pose = controller._panel_pose().detach().cpu().tolist()
        handoff = ModuleHandoff(
            source=self.current.name,
            target=desired.name,
            control_tick=controller.total_control_ticks,
            phase_index=controller.phase_index,
            stage=controller.stage,
            panel_pose=tuple(float(value) for value in panel_pose),
            left_gripper_width=controller.left.measured_gripper_width(),
            right_gripper_width=controller.right.measured_gripper_width(),
        )
        self.current = desired
        self.handoffs.append(handoff)
        return handoff
