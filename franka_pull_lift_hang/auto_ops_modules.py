"""Task-module ownership and handoff contracts for Franka Pull-Lift-Hang."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ModuleContractError(RuntimeError):
    """Raised when a module tries to hand off an invalid physical state."""


@dataclass(frozen=True)
class ModuleHandoff:
    source: str | None
    target: str
    control_tick: int
    phase_index: int
    stage: int
    panel_pose: tuple[float, ...]
    left_gripper_width: float
    right_gripper_width: float


class AutoOpsTaskModule:
    name: str

    def owns(self, phase_index: int, stage: int) -> bool:
        raise NotImplementedError

    def validate_enter(self, controller: Any) -> None:
        return

    def validate_exit(self, controller: Any) -> None:
        return

    def step(self, controller: Any, panel_pose: Any, px: float, py: float, pz: float, device: Any) -> None:
        """Run only a phase owned by this module."""
        if not self.owns(controller.phase_index, controller.stage):
            raise ModuleContractError(
                f"{self.name} cannot execute phase={controller.phase_index} stage={controller.stage}"
            )
        handler = getattr(controller, f"_step_phase_{controller.phase_index}")
        handler(panel_pose, px, py, pz, device)


class PullTaskModule(AutoOpsTaskModule):
    name = "pull"
    start_phase = 0
    end_phase = 3

    def owns(self, phase_index: int, stage: int) -> bool:
        # Phase 3 stages 0..3 are release, vertical extraction, retreat and
        # return-to-ready. Side approach is owned exclusively by Lift.
        return phase_index <= 2 or (phase_index == 3 and stage <= 3)

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


class LiftTaskModule(AutoOpsTaskModule):
    name = "lift"
    start_phase = 3
    end_phase = 5

    def owns(self, phase_index: int, stage: int) -> bool:
        # Lift ends after vertical clearance. Phase-5 stage 3 starts rotation,
        # which belongs to Hang by task definition.
        return (
            (phase_index == 3 and stage >= 4)
            or phase_index == 4
            or (phase_index == 5 and stage < 3)
        )

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


class HangTaskModule(AutoOpsTaskModule):
    name = "hang"
    start_phase = 6
    end_phase = 10

    def owns(self, phase_index: int, stage: int) -> bool:
        return (phase_index == 5 and stage >= 3) or phase_index >= 6

    def validate_enter(self, controller: Any) -> None:
        if controller.grasp_offsets is None:
            raise ModuleContractError("Hang requires Lift's measured panel-to-hand transforms")


MODULES = {
    "pull": PullTaskModule(),
    "lift": LiftTaskModule(),
    "hang": HangTaskModule(),
}


class LongTaskOrchestrator:
    """Own module order and enforce physical handoff contracts."""

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
        # Resolve physical ownership globally, independently of which subset a
        # standalone command requested.
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
