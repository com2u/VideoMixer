import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QComboBox,
                             QCheckBox, QPushButton, QFrame, QFileDialog, QInputDialog, QSlider, QMenu, QGroupBox)
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QAction
from gui_thread import VideoThread
from utils import get_available_cameras

class VideoMixerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Mixer (Qt)")
        self.resize(1200, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Sidebar
        sidebar = QFrame()
        sidebar.setFixedWidth(350)
        sidebar.setFrameShape(QFrame.StyledPanel)
        sidebar_layout = QVBoxLayout(sidebar)
        main_layout.addWidget(sidebar)

        # Input Section
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout(input_group)

        # Source 1
        input_layout.addWidget(QLabel("Source 1 - File or Camera Selection"))
        self.s1_entry = QLineEdit("0")
        input_layout.addWidget(self.s1_entry)
        s1_buttons = QHBoxLayout()
        btn1_file = QPushButton("Select Video File")
        btn1_file.clicked.connect(lambda: self.browse(self.s1_entry))
        s1_buttons.addWidget(btn1_file)
        btn1_cam = QPushButton("Select Camera")
        btn1_cam.clicked.connect(lambda: self.show_camera_menu(self.s1_entry, btn1_cam))
        s1_buttons.addWidget(btn1_cam)
        input_layout.addLayout(s1_buttons)

        # Source 2
        input_layout.addWidget(QLabel("Source 2 - File or Camera Selection"))
        self.s2_entry = QLineEdit("test1.mp4")
        input_layout.addWidget(self.s2_entry)
        s2_buttons = QHBoxLayout()
        btn2_file = QPushButton("Select Video File")
        btn2_file.clicked.connect(lambda: self.browse(self.s2_entry))
        s2_buttons.addWidget(btn2_file)
        btn2_cam = QPushButton("Select Camera")
        btn2_cam.clicked.connect(lambda: self.show_camera_menu(self.s2_entry, btn2_cam))
        s2_buttons.addWidget(btn2_cam)
        input_layout.addLayout(s2_buttons)

        sidebar_layout.addWidget(input_group)

        # Mixer Section
        mixer_group = QGroupBox("Mixer")
        mixer_layout = QVBoxLayout(mixer_group)

        # Mixing Mode
        mixer_layout.addWidget(QLabel("Mixing Mode"))
        self.mode_combo = QComboBox()
        modes = ["add", "subtract", "multiply", "minimum", "maximum", "difference",
                 "screen", "overlay", "hard_light", "soft_light", "color_dodge",
                 "color_burn", "linear_burn", "exclusion", "average", "negation",
                 "divide", "power", "gamma", "threshold", "bitwise_and", "bitwise_or",
                 "bitwise_xor", "alpha_composite", "weighted_blend", "crossfade",
                 "luminance_key", "chroma_key", "posterize", "invert", "log", "sigmoid",
                 "opacity", "normalize", "clamp"]
        self.mode_combo.addItems(modes)
        mixer_layout.addWidget(self.mode_combo)

        # Sliders
        mixer_layout.addWidget(QLabel("Alpha / Weight / Opacity (0-100)"))
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(50)
        mixer_layout.addWidget(self.alpha_slider)

        mixer_layout.addWidget(QLabel("Threshold / Clamp (0-255)"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(128)
        mixer_layout.addWidget(self.threshold_slider)

        mixer_layout.addWidget(QLabel("Gamma (0.1 - 5.0)"))
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(1, 50)
        self.gamma_slider.setValue(10)
        mixer_layout.addWidget(self.gamma_slider)

        mixer_layout.addWidget(QLabel("Posterize Levels (1-16)"))
        self.posterize_slider = QSlider(Qt.Horizontal)
        self.posterize_slider.setRange(1, 16)
        self.posterize_slider.setValue(4)
        mixer_layout.addWidget(self.posterize_slider)

        sidebar_layout.addWidget(mixer_group)

        # Output Section
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(output_group)

        # Output Size Mode
        output_layout.addWidget(QLabel("Output Size Mode"))
        self.size_mode_combo = QComboBox()
        self.size_mode_combo.addItems(["Source 1 Size", "Source 2 Size", "Max Width/Height", "Overlapping Area (Min)"])
        output_layout.addWidget(self.size_mode_combo)

        # Padding Color
        output_layout.addWidget(QLabel("Padding Color"))
        self.padding_combo = QComboBox()
        self.padding_combo.addItems(["black", "white"])
        output_layout.addWidget(self.padding_combo)

        # Output Options
        self.virtual_check = QCheckBox("Activate Virtual Device (Streaming)")
        output_layout.addWidget(self.virtual_check)

        self.save_check = QCheckBox("Save to Video File")
        output_layout.addWidget(self.save_check)
        
        output_layout.addWidget(QLabel("Output File"))
        save_row = QHBoxLayout()
        self.save_path_entry = QLineEdit("output.mp4")
        save_row.addWidget(self.save_path_entry)
        btn_save = QPushButton("...")
        btn_save.clicked.connect(self.select_save_path)
        save_row.addWidget(btn_save)
        output_layout.addLayout(save_row)

        sidebar_layout.addWidget(output_group)

        # Start Button
        self.start_btn = QPushButton("START MIXER")
        self.start_btn.setStyleSheet("background-color: green; color: white; font-weight: bold; height: 40px;")
        self.start_btn.clicked.connect(self.toggle_mixer)
        sidebar_layout.addWidget(self.start_btn)
        sidebar_layout.addStretch()

        # Preview
        self.preview_label = QLabel("Live Preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #222; color: white; border: 2px solid #444;")
        self.preview_label.setScaledContents(True)
        main_layout.addWidget(self.preview_label, 1)

        self.thread = None

    def browse(self, entry):
        f, _ = QFileDialog.getOpenFileName(self, "Select Video")
        if f: entry.setText(f)

    def select_save_path(self):
        f, _ = QFileDialog.getSaveFileName(self, "Save Video", "", "Video Files (*.mp4 *.avi)")
        if f: self.save_path_entry.setText(f)

    def show_camera_menu(self, entry, button):
        menu = QMenu(self)
        cameras = get_available_cameras()
        if not cameras:
            menu.addAction("No cameras found")
        else:
            for idx, name in cameras:
                action = QAction(name, self)
                action.triggered.connect(lambda checked=False, i=idx: entry.setText(str(i)))
                menu.addAction(action)
        menu.exec(button.mapToGlobal(button.rect().bottomLeft()))

    @Slot(int, int)
    def set_output_size(self, w, h):
        self.preview_label.setMaximumSize(w, h)

    def toggle_mixer(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
            self.start_btn.setText("START MIXER")
            self.start_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; height: 45px; border-radius: 5px;")
        else:
            self.thread = VideoThread(self)
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.output_size_signal.connect(self.set_output_size)
            self.thread.start()
            self.start_btn.setText("STOP MIXER")
            self.start_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; height: 45px; border-radius: 5px;")

    @Slot(np.ndarray)
    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.preview_label.contentsRect().width(), self.preview_label.contentsRect().height(), Qt.KeepAspectRatio)
        self.preview_label.setPixmap(QPixmap.fromImage(p))