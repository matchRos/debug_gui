from typing import Any, Dict, List, Tuple

from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState


class StepRunner:
    """
    Executes the configured steps one-by-one.
    """

    def __init__(self, steps: List[BaseStep]) -> None:
        self.steps = steps
        self.current_idx = 0

    def reset(self) -> None:
        """
        Reset the internal step pointer.
        """
        self.current_idx = 0

    def has_next(self) -> bool:
        """
        Returns True if another step is available.
        """
        return self.current_idx < len(self.steps)

    def get_step_names(self) -> List[str]:
        """
        Returns a list of all configured step names.
        """
        return [step.name for step in self.steps]

    def get_current_step_name(self) -> str:
        """
        Returns the current step name or a terminal label if complete.
        """
        if not self.has_next():
            return "finished"
        return self.steps[self.current_idx].name

    def run_next(self, state: PipelineState) -> Tuple[str, Dict[str, Any]]:
        """
        Execute the next step in sequence.

        Args:
            state: Shared pipeline state.

        Returns:
            Tuple of (step_name, result_dict)
        """
        if not self.has_next():
            raise RuntimeError("No more steps available.")

        step = self.steps[self.current_idx]
        result = step.run(state)

        state.finished_steps.append(step.name)
        self.current_idx += 1
        return step.name, result

    def run_step_by_name(
        self,
        state: PipelineState,
        step_name: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Execute a specific step by name.

        This does not enforce strict sequential execution and is mainly
        intended for debugging.

        Args:
            state: Shared pipeline state.
            step_name: Name of the step to execute.

        Returns:
            Tuple of (step_name, result_dict)
        """
        for idx, step in enumerate(self.steps):
            if step.name == step_name:
                result = step.run(state)

                # Mark step as finished only once
                if step.name not in state.finished_steps:
                    state.finished_steps.append(step.name)

                # Move current pointer forward if needed
                if idx >= self.current_idx:
                    self.current_idx = idx + 1

                return step.name, result

        raise ValueError(f"Unknown step name: {step_name}")

    def set_pointer_to_step_name(self, step_name: str) -> None:
        """
        Move the sequential 'Next step' pointer to the given step without running
        any steps. Use for faster debugging when intermediate steps are not needed.
        """
        for idx, step in enumerate(self.steps):
            if step.name == step_name:
                self.current_idx = idx
                return
        raise ValueError(f"Unknown step name: {step_name}")
