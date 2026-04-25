from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from cable_routing.debug_gui.orchestration.action_types import ActionFeedback, ActionResult
from cable_routing.debug_gui.pipeline.state import PipelineState


class BasePipelineAction(ABC):
    """
    High-level unit of execution for the routing pipeline.

    The goal is to mirror the lifecycle we later want from ROS actions:
    goal -> active -> feedback -> result.
    """

    name = "unnamed_action"
    description = "No description provided."

    def emit_feedback(self, state: PipelineState, feedback: ActionFeedback) -> None:
        if not hasattr(state, "action_feedback"):
            state.action_feedback = {}
        state.action_feedback[self.name] = feedback
        state.log(
            f"[action:{self.name}] stage={feedback.stage}"
            + (f" msg={feedback.message}" if feedback.message else "")
        )

    @abstractmethod
    def execute(self, state: PipelineState) -> ActionResult:
        raise NotImplementedError

