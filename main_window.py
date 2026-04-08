from PyQt5.QtWidgets import (
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
        self.reset_button = QPushButton("Reset")

        self._build_layout()
        self._connect_signals()

    def _build_layout(self) -> None:
        """
        Create and assign the main layout.
        """
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Left: step list
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Pipeline Steps"))
        left_layout.addWidget(self.step_list)

        # Center: logs
        center_layout = QVBoxLayout()
        center_layout.addWidget(QLabel("Logs"))
        center_layout.addWidget(self.log_box)

        # Right: image + buttons
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Current Visualization"))
        right_layout.addWidget(self.image_label, stretch=1)
        right_layout.addWidget(self.next_button)
        right_layout.addWidget(self.run_selected_button)
        right_layout.addWidget(self.reset_button)

        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addLayout(center_layout, stretch=2)
        main_layout.addLayout(right_layout, stretch=3)

    def _connect_signals(self) -> None:
        """
        Connect button callbacks.
        """
        self.next_button.clicked.connect(self.controller.on_next_step)
        self.run_selected_button.clicked.connect(self.controller.on_run_selected)
        self.reset_button.clicked.connect(self.controller.on_reset)
