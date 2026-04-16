from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import Qt


class MainWindow(QMainWindow):
    """
    Main window for the cable routing debug GUI.
    """

    def __init__(self, controller) -> None:
        super().__init__()
        self.controller = controller

        self.setWindowTitle("Cable Routing Debug Pipeline")
        self.resize(1400, 800)

        self.step_list = QListWidget()
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("border: 1px solid gray;")

        self.next_button = QPushButton("Next Step")
        self.run_selected_button = QPushButton("Run Selected")
        self.jump_pointer_button = QPushButton("Auto-run to selected")
        self.reset_button = QPushButton("Reset")

        self.save_trace_button = QPushButton("Save Cable Trace")
        self.load_trace_button = QPushButton("Load Cable Trace")
        self.trace_mode_combo = QComboBox()
        self.trace_mode_combo.addItem("Auto from config", "auto_from_config")
        self.trace_mode_combo.addItem(
            "Auto white rings (first clip)", "auto_white_rings_from_clip"
        )
        self.trace_mode_combo.addItem("Auto from clip A", "auto_from_clip_a")
        self.trace_mode_combo.addItem("Manual two clicks", "manual_two_clicks")

        self._build_layout()
        self._connect_signals()

    def _build_layout(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Pipeline Steps"))
        left_layout.addWidget(self.step_list)

        center_layout = QVBoxLayout()
        center_layout.addWidget(QLabel("Logs"))
        center_layout.addWidget(self.log_box)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Current Visualization"))
        right_layout.addWidget(self.image_label, stretch=1)
        right_layout.addWidget(self.next_button)
        right_layout.addWidget(self.run_selected_button)
        right_layout.addWidget(self.jump_pointer_button)
        right_layout.addWidget(QLabel("Trace Start Mode"))
        right_layout.addWidget(self.trace_mode_combo)
        right_layout.addWidget(self.save_trace_button)
        right_layout.addWidget(self.load_trace_button)
        right_layout.addWidget(self.reset_button)

        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addLayout(center_layout, stretch=2)
        main_layout.addLayout(right_layout, stretch=3)

    def _connect_signals(self) -> None:
        self.next_button.clicked.connect(self.controller.on_next_step)
        self.run_selected_button.clicked.connect(self.controller.on_run_selected)
        self.jump_pointer_button.clicked.connect(
            self.controller.on_auto_run_to_selected
        )
        self.reset_button.clicked.connect(self.controller.on_reset)
        self.save_trace_button.clicked.connect(self.controller.on_save_trace)
        self.load_trace_button.clicked.connect(self.controller.on_load_trace)
        self.trace_mode_combo.currentIndexChanged.connect(
            self.controller.on_trace_start_mode_changed
        )

    def ask_save_trace_path(self) -> str:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Cable Trace",
            "cable_trace.csv",
            "CSV Files (*.csv)",
        )
        return path

    def ask_load_trace_path(self) -> str:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Cable Trace",
            "",
            "CSV Files (*.csv)",
        )
        return path
