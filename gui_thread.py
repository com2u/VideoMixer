import cv2
import numpy as np
import pyvirtualcam
import threading
import time
from PySide6.QtCore import QThread, Signal
from video_mixer import resize_and_pad, mix_frames

class VideoThread(QThread):
    change_pixmap_signal = Signal(np.ndarray)
    output_size_signal = Signal(int, int)

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self._run_flag = True

    def stop(self):
        self._run_flag = False

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
            m1 = resize_and_pad(f1_raw, (out_w, out_h), padding)
            m2 = resize_and_pad(f2_raw, (out_w, out_h), padding)

            mixed = mix_frames(m1, m2, mode, alpha_val, threshold_val, gamma_val, posterize_val, t)
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