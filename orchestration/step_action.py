from __future__ import annotations

import time
from typing import Dict

from cable_routing.debug_gui.orchestration.action_types import ActionFeedback, ActionResult, ActionStatus
from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState

from .base_action import BasePipelineAction


class StepBackedAction(BasePipelineAction):
    """
    Adapter that exposes an existing debug step as an action.
    """

    def __init__(self, step: BaseStep) -> None:
        self.step = step
        self.name = step.name
        self.description = step.description

    def execute(self, state: PipelineState) -> ActionResult:
        started = time.time()
        self.emit_feedback(
            state,
            ActionFeedback(stage="started", message=self.description),
        )

        try:
            outputs: Dict[str, object] = self.step.run(state)
            finished = time.time()
            self.emit_feedback(
                state,
                ActionFeedback(stage="finished", message="completed successfully"),
            )
            return ActionResult(
                status=ActionStatus.SUCCEEDED,
                action_name=self.name,
                message="completed successfully",
                outputs=outputs,
                started_at_s=started,
                finished_at_s=finished,
                duration_s=finished - started,
            )
        except Exception as exc:
            finished = time.time()
            self.emit_feedback(
                state,
                ActionFeedback(stage="failed", message=str(exc)),
            )
            return ActionResult(
                status=ActionStatus.FAILED,
                action_name=self.name,
                message=str(exc),
                error_type=type(exc).__name__,
                started_at_s=started,
                finished_at_s=finished,
                duration_s=finished - started,
            )

