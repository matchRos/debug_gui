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
from cable_routing.debug_gui.pipeline.steps.trace_to_world_step import TraceToWorldStep
from cable_routing.debug_gui.pipeline.steps.compute_orientation_step import (
    ComputeOrientationStep,
)
from cable_routing.debug_gui.pipeline.steps.grasp_planning_step import GraspPlanningStep
from cable_routing.debug_gui.pipeline.steps.grasp_pose_step import GraspPoseStep
from cable_routing.debug_gui.pipeline.steps.visualize_grasps_step import (
    VisualizeGraspsStep,
)
from cable_routing.debug_gui.pipeline.steps.pregrasp_pose_step import PreGraspPoseStep
from cable_routing.debug_gui.pipeline.steps.robot_motion_step import RobotMotionStep
from cable_routing.debug_gui.pipeline.steps.home_arms_step import HomeArmsStep
from cable_routing.debug_gui.pipeline.steps.descend_to_grasp_step import (
    DescendToGraspStep,
)
from cable_routing.debug_gui.pipeline.steps.lift_after_grasp_step import (
    LiftAfterGraspStep,
)
from cable_routing.debug_gui.pipeline.steps.handover_fine_orient_step import (
    HandoverFineOrientStep,
)
from cable_routing.debug_gui.pipeline.steps.handover_move_exchange_step import (
    HandoverMoveExchangeStep,
)
from cable_routing.debug_gui.pipeline.steps.present_cable_vertical_step import (
    PresentCableVerticalStep,
)
from cable_routing.debug_gui.pipeline.steps.second_arm_side_approach_step import (
    SecondArmSideApproachStep,
)
from cable_routing.debug_gui.pipeline.unwind_wrists_step import UnwindWristsStep
from cable_routing.debug_gui.pipeline.steps.descend_second_to_grasp_step import (
    DescendSecondToGraspStep,
)
from cable_routing.debug_gui.pipeline.steps.close_first_gripper_step import (
    CloseFirstGripperStep,
)
from cable_routing.debug_gui.pipeline.steps.close_second_gripper_step import (
    CloseSecondGripperStep,
)
from cable_routing.debug_gui.pipeline.steps.plan_first_route_step import (
    PlanFirstRouteStep,
)
from cable_routing.debug_gui.pipeline.steps.execute_first_route_step import (
    ExecuteFirstRouteStep,
)


def build_runner() -> StepRunner:
    steps = [
        HomeArmsStep(),
        InitEnvironmentStep(),
        PrepareRoutingStep(),
        TraceCableStep(),
        TraceToWorldStep(),
        ComputeOrientationStep(),
        GraspPlanningStep(),
        GraspPoseStep(),
        VisualizeGraspsStep(),
        PreGraspPoseStep(),
        RobotMotionStep(),
        UnwindWristsStep(),
        DescendToGraspStep(),
        CloseFirstGripperStep(),
        LiftAfterGraspStep(),
        HandoverFineOrientStep(),
        HandoverMoveExchangeStep(),
        PresentCableVerticalStep(),
        SecondArmSideApproachStep(),
        CloseSecondGripperStep(),
        PlanFirstRouteStep(),
        ExecuteFirstRouteStep(),
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
