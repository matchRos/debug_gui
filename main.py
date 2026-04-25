import os
import sys

# Prevent OpenCV Qt plugin conflicts with PyQt5
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("QT_PLUGIN_PATH", None)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

from PyQt5.QtWidgets import QApplication

from cable_routing.debug_gui.controllers.gui_controller import GuiController
from cable_routing.debug_gui.main_window import MainWindow
from cable_routing.debug_gui.orchestration.default_pipeline import (
    build_default_orchestrator,
)
from cable_routing.debug_gui.pipeline.runner import StepRunner
from cable_routing.debug_gui.pipeline.state import PipelineState


def build_runner() -> StepRunner:
    orchestrator = build_default_orchestrator()
    return StepRunner(orchestrator.build_steps())


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

# First thing: detect errors (cable too loose, cable not grasped) using perception
# Then create recovery primitives:
# If cable too loose pull it along based on the rollers until it's tight if you can, otherwise just drop it from one hand
# If cable not grasped from one hand but grasped from another, grasp it with the ungrasped hand and then pull it tight
# If cable not grasped from either hand, find all the valid points using perception, check which one of the valid (tight, in path)
# points is the last one in a series of valid points in the path, and then grasp around there (try grasping with both hands after and one before one after)
