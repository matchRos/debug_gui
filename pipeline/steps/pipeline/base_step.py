from abc import ABC, abstractmethod
from typing import Any, Dict

from cable_routing.debug_gui.pipeline.state import PipelineState


class BaseStep(ABC):
    """
    Abstract base class for one step in the debug pipeline.
    """

    name = "unnamed_step"
    description = "No description provided."

    @abstractmethod
    def run(self, state: PipelineState) -> Dict[str, Any]:
        """
        Execute the step and optionally return structured debug information.

        Args:
            state: Shared pipeline state.

        Returns:
            Dictionary with optional debug metadata/results.
        """
        raise NotImplementedError
