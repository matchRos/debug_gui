from __future__ import annotations

from typing import Iterable, List

from cable_routing.debug_gui.pipeline.base_step import BaseStep

from .action_step import ActionStep
from .base_action import BasePipelineAction


class PipelineOrchestrator:
    """
    Owns the ordered list of pipeline actions.

    Today it materializes them as GUI-runner steps. Later the same action list can
    be exposed via ROS action servers or a richer state machine.
    """

    def __init__(self, actions: Iterable[BasePipelineAction]) -> None:
        self.actions = list(actions)

    def get_action_names(self) -> List[str]:
        return [action.name for action in self.actions]

    def build_steps(self) -> List[BaseStep]:
        return [ActionStep(action) for action in self.actions]

