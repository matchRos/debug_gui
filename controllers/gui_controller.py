from typing import Any, Dict, List, Optional

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

from cable_routing.debug_gui.pipeline.runner import StepRunner
from cable_routing.debug_gui.pipeline.state import PipelineState


class GuiController:
    """
    Connects GUI actions to the debug pipeline backend.
    """

    def __init__(self, state: PipelineState, runner: StepRunner) -> None:
        self.state = state
        self.runner = runner
        self.window = None

    def set_window(self, window: Any) -> None:
        """
        Attach the main window after construction.
        """
        self.window = window
        self._populate_step_list()
        self._append_log("GUI controller initialized.")
        self._append_log(f"Current step: {self.runner.get_current_step_name()}")

    def _populate_step_list(self) -> None:
        """
        Fill the GUI step list from the runner configuration.
        """
        if self.window is None:
            return

        self.window.step_list.clear()
        for step_name in self.runner.get_step_names():
            self.window.step_list.addItem(step_name)

    def _append_log(self, message: str) -> None:
        """
        Add a log line to both state and GUI.
        """
        self.state.log(message)

        if self.window is not None:
            self.window.log_box.append(message)

    def _update_step_highlight(self) -> None:
        """
        Highlight the currently active step in the list widget.
        """
        if self.window is None:
            return

        current_name = self.runner.get_current_step_name()
        for i in range(self.window.step_list.count()):
            item = self.window.step_list.item(i)
            if item.text() == current_name:
                self.window.step_list.setCurrentItem(item)
                return

    def _numpy_to_pixmap(self, image) -> Optional[QPixmap]:
        """
        Convert a numpy RGB image to a QPixmap.

        Expected image format: H x W x 3, dtype uint8.
        """
        if image is None:
            return None

        if len(image.shape) != 3 or image.shape[2] != 3:
            self._append_log(
                "Image conversion skipped: expected RGB image with shape HxWx3."
            )
            return None

        height, width, channels = image.shape
        bytes_per_line = channels * width

        qimage = QImage(
            image.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        )
        return QPixmap.fromImage(qimage.copy())

    def _refresh_image_view(self) -> None:
        """
        Show the newest available image in the GUI.
        Priority:
        grasp_overlay > trace_overlay > routing_overlay > rgb_image
        """
        if self.window is None:
            return

        image = None
        if self.state.grasp_overlay is not None:
            image = self.state.grasp_overlay
        elif self.state.trace_overlay is not None:
            image = self.state.trace_overlay
        elif self.state.routing_overlay is not None:
            image = self.state.routing_overlay
        elif self.state.rgb_image is not None:
            image = self.state.rgb_image

        pixmap = self._numpy_to_pixmap(image)
        if pixmap is None:
            self.window.image_label.setText("No image available")
            return

        scaled = pixmap.scaled(
            self.window.image_label.width(),
            self.window.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.window.image_label.setPixmap(scaled)

    def _handle_step_result(self, step_name: str, result: Dict[str, Any]) -> None:
        """
        Display step result metadata.
        """
        self._append_log(f"Finished step: {step_name}")

        if result:
            for key, value in result.items():
                self._append_log(f"  {key}: {value}")

        self._append_log(f"Next step: {self.runner.get_current_step_name()}")
        self._update_step_highlight()
        self._refresh_image_view()

    def on_next_step(self) -> None:
        """
        GUI callback: execute the next pipeline step.
        """
        try:
            step_name, result = self.runner.run_next(self.state)
            self._handle_step_result(step_name, result)
        except Exception as exc:
            self._append_log(f"ERROR while running next step: {exc}")

    def on_run_selected(self) -> None:
        """
        GUI callback: execute the currently selected step by name.
        """
        if self.window is None:
            return

        current_item = self.window.step_list.currentItem()
        if current_item is None:
            self._append_log("No step selected.")
            return

        step_name = current_item.text()

        try:
            executed_name, result = self.runner.run_step_by_name(self.state, step_name)
            self._handle_step_result(executed_name, result)
        except Exception as exc:
            self._append_log(f"ERROR while running selected step '{step_name}': {exc}")

    def on_reset(self) -> None:
        """
        GUI callback: reset runner and shared state.
        """
        self.runner.reset()
        self.state.reset_runtime_data()

        if self.window is not None:
            self.window.log_box.clear()
            self.window.image_label.clear()
            self.window.image_label.setText("No image loaded")

        self._append_log("Pipeline reset.")
        self._append_log(f"Current step: {self.runner.get_current_step_name()}")
        self._update_step_highlight()
