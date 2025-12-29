import sys
from PySide6.QtWidgets import QApplication
from gui_app import VideoMixerApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoMixerApp()
    window.show()
    sys.exit(app.exec())