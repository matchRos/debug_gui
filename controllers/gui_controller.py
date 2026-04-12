from typing import Any, Dict, Optional

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

from cable_routing.debug_gui.backend.cable_trace_io import CableTraceIO
from cable_routing.debug_gui.backend.tracing_service import TracingService
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
        self.trace_io = CableTraceIO()
        self.tracing_service = TracingService()

    def set_window(self, window: Any) -> None:
        self.window = window
        self._populate_step_list()
        self._append_log("GUI controller initialized.")
        self._append_log(f"Current step: {self.runner.get_current_step_name()}")

    def _populate_step_list(self) -> None:
        if self.window is None:
            return

        self.window.step_list.clear()
        for step_name in self.runner.get_step_names():
            self.window.step_list.addItem(step_name)

    def _append_log(self, message: str) -> None:
        self.state.log(message)

        if self.window is not None:
            self.window.log_box.append(message)

    def _update_step_highlight(self) -> None:
        if self.window is None:
            return

        current_name = self.runner.get_current_step_name()
        for i in range(self.window.step_list.count()):
            item = self.window.step_list.item(i)
            if item.text() == current_name:
                self.window.step_list.setCurrentItem(item)
                return

    def _numpy_to_pixmap(self, image) -> Optional[QPixmap]:
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
        self._append_log(f"Finished step: {step_name}")

        if result:
            for key, value in result.items():
                self._append_log(f"  {key}: {value}")

        self._append_log(f"Next step: {self.runner.get_current_step_name()}")
        self._update_step_highlight()
        self._refresh_image_view()

    def on_next_step(self) -> None:
        try:
            step_name, result = self.runner.run_next(self.state)
            self._handle_step_result(step_name, result)
        except Exception as exc:
            self._append_log(f"ERROR while running next step: {exc}")

    def on_run_selected(self) -> None:
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
        self.runner.reset()
        self.state.reset_runtime_data()

        if self.window is not None:
            self.window.log_box.clear()
            self.window.image_label.clear()
            self.window.image_label.setText("No image loaded")

        self._append_log("Pipeline reset.")
        self._append_log(f"Current step: {self.runner.get_current_step_name()}")
        self._update_step_highlight()

    def on_save_trace(self) -> None:
        if self.window is None:
            return

        if self.state.path_in_pixels is None:
            self._append_log("No cable trace available to save.")
            return

        try:
            filepath = self.window.ask_save_trace_path()
            if not filepath:
                self._append_log("Save cable trace cancelled.")
                return

            self.trace_io.save_csv(filepath, self.state.path_in_pixels)
            self._append_log(f"Saved cable trace to: {filepath}")
        except Exception as exc:
            self._append_log(f"ERROR while saving cable trace: {exc}")

    def on_load_trace(self) -> None:
        if self.window is None:
            return

        try:
            filepath = self.window.ask_load_trace_path()
            if not filepath:
                self._append_log("Load cable trace cancelled.")
                return

            self.state.loaded_trace_path = filepath
            path_in_pixels = self.trace_io.load_csv(filepath)

            if self.state.rgb_image is None:
                if self.state.env is not None and self.state.env.camera is not None:
                    image_rgb, _ = self.tracing_service.acquire_image(
                        camera=self.state.env.camera,
                        fallback_image_path=None,
                    )
                    self.state.rgb_image = image_rgb
                elif self.state.config is not None:
                    image_rgb, _ = self.tracing_service.acquire_image(
                        camera=None,
                        fallback_image_path=self.state.config.debug_image_path,
                    )
                    self.state.rgb_image = image_rgb

            if self.state.rgb_image is None:
                raise RuntimeError("No image available for trace preview.")

            start_points = []
            end_points = None

            if self.state.config is not None and hasattr(
                self.state.config, "trace_start_points"
            ):
                start_points = [tuple(p) for p in self.state.config.trace_start_points]

            if self.state.config is not None and hasattr(
                self.state.config, "trace_end_points"
            ):
                pts = self.state.config.trace_end_points
                if pts is not None:
                    end_points = [tuple(p) for p in pts]

            overlay = self.tracing_service.create_trace_overlay(
                image_rgb=self.state.rgb_image,
                start_points=start_points,
                end_points=end_points,
                path_in_pixels=path_in_pixels,
            )

            self.state.path_in_pixels = path_in_pixels
            self.state.trace_overlay = overlay
            self.state.path_in_world = None
            self.state.cable_orientations = None
            self.state.grasp_overlay = None

            self._append_log(f"Loaded cable trace from: {filepath}")
            self._append_log(f"  num_path_points: {len(path_in_pixels)}")
            self._refresh_image_view()

        except Exception as exc:
            self._append_log(f"ERROR while loading cable trace: {exc}")
