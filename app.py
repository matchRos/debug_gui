import os
import sys

# Prevent OpenCV Qt plugin conflicts with PyQt5
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("QT_PLUGIN_PATH", None)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

from PyQt5.QtWidgets import QApplication

from cable_routing.debug_gui.controllers.gui_controller import GuiController
from cable_routing.debug_gui.main_window import MainWindow
from cable_routing.debug_gui.pipeline.runner import StepRunner
from cable_routing.debug_gui.pipeline.state import PipelineState
from cable_routing.debug_gui.pipeline.steps.init_environment_step import (
    InitEnvironmentStep,
)
from cable_routing.debug_gui.pipeline.steps.prepare_routing_step import (
    PrepareRoutingStep,
)
from cable_routing.debug_gui.pipeline.steps.trace_cable_step import TraceCableStep


def build_runner() -> StepRunner:
    steps = [
        InitEnvironmentStep(),
        PrepareRoutingStep(),
        TraceCableStep(),
    ]
    return StepRunner(steps)


def main() -> None:
    app = QApplication(sys.argv)

    state = PipelineState()
    runner = build_runner()
    controller = GuiController(state=state, runner=runner)
    window = MainWindow(controller=controller)
    controller.set_window(window)

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
