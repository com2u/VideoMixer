import sys
import cv2
import numpy as np
import pyvirtualcam
import threading
import time
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QComboBox,
                             QCheckBox, QPushButton, QFrame, QFileDialog, QInputDialog, QSlider, QMenu, QGroupBox)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QThread
from PySide6.QtGui import QImage, QPixmap, QAction

def get_available_cameras():
    cameras = []
    # Linux specific camera detection
    if os.path.exists('/sys/class/video4linux'):
        for d in sorted(os.listdir('/sys/class/video4linux')):
            if d.startswith('video'):
                try:
                    with open(f'/sys/class/video4linux/{d}/name', 'r') as f:
                        name = f.read().strip()
                    idx = int(d[5:])
                    cameras.append((idx, f"{name} (/{d})"))
                except: pass
    else:
        # Fallback for other OS or if sysfs is not available
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append((i, f"Camera {i}"))
                cap.release()
    return cameras

class VideoThread(QThread):
    change_pixmap_signal = Signal(np.ndarray)
    output_size_signal = Signal(int, int)

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self._run_flag = True

    def stop(self):
        self._run_flag = False

    def _resize_and_pad(self, frame, target_size, padding_color):
        if target_size is None:
            return frame
        tw, th = target_size
        h, w = frame.shape[:2]
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh))
        color = (0, 0, 0) if padding_color == 'black' else (255, 255, 255)
        canvas = np.full((th, tw, 3), color, dtype=np.uint8)
        x_offset = (tw - nw) // 2
        y_offset = (th - nh) // 2
        canvas[y_offset:y_offset+nh, x_offset:x_offset+nw] = resized
        return canvas

    def run(self):
        s1_path = self.parent.s1_entry.text()
        s2_path = self.parent.s2_entry.text()
        
        cap1 = cv2.VideoCapture(int(s1_path) if s1_path.isdigit() else s1_path)
        cap2 = cv2.VideoCapture(int(s2_path) if s2_path.isdigit() else s2_path)

        if not cap1.isOpened() or not cap2.isOpened():
            print("Error: Could not open one or both sources.")
            return

        # Get native sizes
        w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

        size_mode = self.parent.size_mode_combo.currentText()
        if size_mode == "Source 1 Size":
            out_w, out_h = w1, h1
        elif size_mode == "Source 2 Size":
            out_w, out_h = w2, h2
        elif size_mode == "Max Width/Height":
            out_w, out_h = max(w1, w2), max(h1, h2)
        elif size_mode == "Overlapping Area (Min)":
            out_w, out_h = min(w1, w2), min(h1, h2)
        else:
            out_w, out_h = 640, 480

        self.output_size_signal.emit(out_w, out_h)

        use_virtual = self.parent.virtual_check.isChecked()
        save_file = self.parent.save_check.isChecked()
        save_path = self.parent.save_path_entry.text()

        cam = None
        if use_virtual:
            try:
                cam = pyvirtualcam.Camera(width=out_w, height=out_h, fps=30)
            except: pass

        writer = None
        if save_file and save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, 30.0, (out_w, out_h))

        t = 0
        while self._run_flag:
            mode = self.parent.mode_combo.currentText()
            padding = self.parent.padding_combo.currentText()
            alpha_val = self.parent.alpha_slider.value() / 100.0
            threshold_val = self.parent.threshold_slider.value()
            gamma_val = self.parent.gamma_slider.value() / 10.0
            posterize_val = self.parent.posterize_slider.value()

            ret1, f1_raw = cap1.read()
            ret2, f2_raw = cap2.read()

            if not ret1:
                cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret1, f1_raw = cap1.read()
            if not ret2:
                cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret2, f2_raw = cap2.read()

            if not ret1 or not ret2: break

            # Resize both to output size
            m1_f = self._resize_and_pad(f1_raw, (out_w, out_h), padding).astype(np.float32)
            m2_f = self._resize_and_pad(f2_raw, (out_w, out_h), padding).astype(np.float32)

            m1, m2 = m1_f / 255.0, m2_f / 255.0
            res = None

            if mode == 'add': res = m1_f + m2_f
            elif mode == 'subtract': res = m1_f - m2_f
            elif mode == 'multiply': res = (m1_f * m2_f) / 255.0
            elif mode == 'minimum': res = np.minimum(m1_f, m2_f)
            elif mode == 'maximum': res = np.maximum(m1_f, m2_f)
            elif mode == 'difference': res = cv2.absdiff(m1_f, m2_f)
            elif mode == 'screen': res = 255 * (1 - (1 - m1) * (1 - m2))
            elif mode == 'overlay':
                res = 255 * np.where(m1 < 0.5, 2 * m1 * m2, 1 - 2 * (1 - m1) * (1 - m2))
            elif mode == 'hard_light':
                res = 255 * np.where(m2 < 0.5, 2 * m1 * m2, 1 - 2 * (1 - m1) * (1 - m2))
            elif mode == 'soft_light':
                res = 255 * ((1 - 2 * m2) * m1**2 + 2 * m2 * m1)
            elif mode == 'color_dodge':
                res = 255 * np.divide(m1, 1 - m2 + 1e-6)
            elif mode == 'color_burn':
                res = 255 * (1 - np.divide(1 - m1, m2 + 1e-6))
            elif mode == 'linear_burn': res = m1_f + m2_f - 255
            elif mode == 'exclusion': res = m1_f + m2_f - 2 * (m1_f * m2_f) / 255.0
            elif mode == 'average': res = (m1_f + m2_f) / 2.0
            elif mode == 'negation': res = 255 - np.abs(255 - m1_f - m2_f)
            elif mode == 'divide': res = 255 * np.divide(m1, m2 + 1e-6)
            elif mode == 'power': res = 255 * np.power(m1, m2 + 1e-6)
            elif mode == 'gamma': res = 255 * np.power(m1, 1.0 / (gamma_val + 1e-6))
            elif mode == 'threshold':
                gray = cv2.cvtColor(f1_raw, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
                mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                res = self._resize_and_pad(mask_3d, (out_w, out_h), padding).astype(np.float32)
            elif mode == 'bitwise_and': res = cv2.bitwise_and(m1_f.astype(np.uint8), m2_f.astype(np.uint8)).astype(np.float32)
            elif mode == 'bitwise_or': res = cv2.bitwise_or(m1_f.astype(np.uint8), m2_f.astype(np.uint8)).astype(np.float32)
            elif mode == 'bitwise_xor': res = cv2.bitwise_xor(m1_f.astype(np.uint8), m2_f.astype(np.uint8)).astype(np.float32)
            elif mode == 'alpha_composite':
                res = m1_f * alpha_val + m2_f * (1 - alpha_val)
            elif mode == 'weighted_blend':
                res = m1_f * alpha_val + m2_f * (1 - alpha_val)
            elif mode == 'crossfade':
                cf_alpha = (np.sin(t) + 1) / 2.0
                res = m1_f * cf_alpha + m2_f * (1 - cf_alpha)
            elif mode == 'luminance_key':
                gray2 = cv2.cvtColor(f2_raw, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray2, threshold_val, 255, cv2.THRESH_BINARY)
                mask = self._resize_and_pad(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (out_w, out_h), padding)
                mask = mask.astype(np.float32) / 255.0
                res = m2_f * mask + m1_f * (1 - mask)
            elif mode == 'chroma_key':
                hsv2 = cv2.cvtColor(f2_raw, cv2.COLOR_BGR2HSV)
                lower_green = np.array([35, 50, 50])
                upper_green = np.array([85, 255, 255])
                mask = cv2.inRange(hsv2, lower_green, upper_green)
                mask = cv2.bitwise_not(mask)
                mask = self._resize_and_pad(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (out_w, out_h), padding)
                mask = mask.astype(np.float32) / 255.0
                res = m2_f * mask + m1_f * (1 - mask)
            elif mode == 'posterize':
                n = max(1, posterize_val)
                res = np.floor(m1_f / (256.0 / n)) * (256.0 / n)
            elif mode == 'invert': res = 255 - m1_f
            elif mode == 'log': res = 255 * (np.log(1 + m1) / np.log(2))
            elif mode == 'sigmoid':
                res = 255 * (1 / (1 + np.exp(-10 * (m1 - 0.5))))
            elif mode == 'opacity': res = m1_f * alpha_val
            elif mode == 'normalize':
                res = cv2.normalize(m1_f, None, 0, 255, cv2.NORM_MINMAX)
            elif mode == 'clamp':
                res = np.clip(m1_f, threshold_val, 255)
            else: res = m1_f

            mixed = np.clip(res, 0, 255).astype(np.uint8)
            self.change_pixmap_signal.emit(mixed)

            if cam:
                cam.send(cv2.cvtColor(mixed, cv2.COLOR_BGR2RGB))
                cam.sleep_until_next_frame()
            
            if writer:
                writer.write(mixed)
            
            t += 0.1
            time.sleep(0.01)

        cap1.release()
        cap2.release()
        if cam: cam.close()
        if writer: writer.release()

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

        # Input Group
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout(input_group)

        # Source 1
        input_layout.addWidget(QLabel("Source 1"))
        self.s1_entry = QLineEdit("0")
        input_layout.addWidget(self.s1_entry)
        s1_buttons_layout = QHBoxLayout()
        btn1_file = QPushButton("Select Video File")
        btn1_file.clicked.connect(lambda: self.browse(self.s1_entry))
        s1_buttons_layout.addWidget(btn1_file)
        btn1_cam = QPushButton("Select Camera")
        btn1_cam.clicked.connect(lambda: self.show_camera_menu(self.s1_entry, btn1_cam))
        s1_buttons_layout.addWidget(btn1_cam)
        input_layout.addLayout(s1_buttons_layout)

        # Source 2
        input_layout.addWidget(QLabel("Source 2"))
        self.s2_entry = QLineEdit("test1.mp4")
        input_layout.addWidget(self.s2_entry)
        s2_buttons_layout = QHBoxLayout()
        btn2_file = QPushButton("Select Video File")
        btn2_file.clicked.connect(lambda: self.browse(self.s2_entry))
        s2_buttons_layout.addWidget(btn2_file)
        btn2_cam = QPushButton("Select Camera")
        btn2_cam.clicked.connect(lambda: self.show_camera_menu(self.s2_entry, btn2_cam))
        s2_buttons_layout.addWidget(btn2_cam)
        input_layout.addLayout(s2_buttons_layout)

        sidebar_layout.addWidget(input_group)

        # Mix Group
        mix_group = QGroupBox("Mix")
        mix_layout = QVBoxLayout(mix_group)

        mix_layout.addWidget(QLabel("Output Size Mode"))
        self.size_mode_combo = QComboBox()
        self.size_mode_combo.addItems(["Source 1 Size", "Source 2 Size", "Max Width/Height", "Overlapping Area (Min)"])
        mix_layout.addWidget(self.size_mode_combo)

        mix_layout.addWidget(QLabel("Mixing Mode"))
        self.mode_combo = QComboBox()
        modes = ["add", "subtract", "multiply", "minimum", "maximum", "difference",
                 "screen", "overlay", "hard_light", "soft_light", "color_dodge",
                 "color_burn", "linear_burn", "exclusion", "average", "negation",
                 "divide", "power", "gamma", "threshold", "bitwise_and", "bitwise_or",
                 "bitwise_xor", "alpha_composite", "weighted_blend", "crossfade",
                 "luminance_key", "chroma_key", "posterize", "invert", "log", "sigmoid",
                 "opacity", "normalize", "clamp"]
        self.mode_combo.addItems(modes)
        mix_layout.addWidget(self.mode_combo)

        self.padding_combo = QComboBox()
        self.padding_combo.addItems(["black", "white"])
        mix_layout.addWidget(QLabel("Padding Color"))
        mix_layout.addWidget(self.padding_combo)

        sidebar_layout.addWidget(mix_group)

        # Filter Group
        filter_group = QGroupBox("Filter")
        filter_layout = QVBoxLayout(filter_group)

        # Sliders for parameters
        filter_layout.addWidget(QLabel("Alpha / Weight / Opacity (0-100)"))
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(50)
        filter_layout.addWidget(self.alpha_slider)

        filter_layout.addWidget(QLabel("Threshold / Clamp (0-255)"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(128)
        filter_layout.addWidget(self.threshold_slider)

        filter_layout.addWidget(QLabel("Gamma (0.1 - 5.0)"))
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(1, 50)
        self.gamma_slider.setValue(10)
        filter_layout.addWidget(self.gamma_slider)

        filter_layout.addWidget(QLabel("Posterize Levels (1-16)"))
        self.posterize_slider = QSlider(Qt.Horizontal)
        self.posterize_slider.setRange(1, 16)
        self.posterize_slider.setValue(4)
        filter_layout.addWidget(self.posterize_slider)

        sidebar_layout.addWidget(filter_group)

        # Output Group
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(output_group)

        self.virtual_check = QCheckBox("Activate Virtual Device (Streaming)")
        output_layout.addWidget(self.virtual_check)

        self.save_check = QCheckBox("Save to Video File")
        output_layout.addWidget(self.save_check)

        save_row = QHBoxLayout()
        self.save_path_entry = QLineEdit("output.mp4")
        save_row.addWidget(self.save_path_entry)
        btn_save = QPushButton("...")
        btn_save.clicked.connect(self.select_save_path)
        save_row.addWidget(btn_save)
        output_layout.addLayout(save_row)

        self.start_btn = QPushButton("START MIXER")
        self.start_btn.setStyleSheet("background-color: green; color: white; font-weight: bold; height: 40px;")
        self.start_btn.clicked.connect(self.toggle_mixer)
        output_layout.addWidget(self.start_btn)

        sidebar_layout.addWidget(output_group)
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoMixerApp()
    window.show()
    sys.exit(app.exec())