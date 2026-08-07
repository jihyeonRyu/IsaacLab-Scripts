"""Shared task-module protocol and handoff data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ModuleContractError(RuntimeError):
    """Raised when a physical task-module boundary is invalid."""


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
    """Interface implemented by every reusable physical task module."""

    name: str
    start_phase: int
    end_phase: int

    def owns(self, phase_index: int, stage: int) -> bool:
        raise NotImplementedError

    def validate_enter(self, controller: Any) -> None:
        return

    def validate_exit(self, controller: Any) -> None:
        return

    def step(self, controller: Any, panel_pose: Any, px: float, py: float, pz: float, device: Any) -> None:
        raise NotImplementedError

    def _run_phase(self, controller: Any, phase: int, panel_pose: Any, px: float, py: float, pz: float, device: Any) -> None:
        if controller.phase_index != phase or not self.owns(phase, controller.stage):
            raise ModuleContractError(
                f"{self.name} cannot execute phase={controller.phase_index} stage={controller.stage}"
            )
        getattr(controller, f"_step_phase_{phase}")(panel_pose, px, py, pz, device)
